"""The NVENC encoder: an OpenGL frame in, an H.264, HEVC or AV1 stream out.

On Linux the session is opened with NvEncodeAPI's OpenGL device type, which is
what makes the frame's journey a short one -- the encoder reads the texture the
renderer just wrote, in GPU memory, and nothing crosses the bus but the
compressed result. On Windows the session is opened on a Direct3D 11 device and
reads Direct3D textures that OpenGL draws into through
:mod:`pyopengl_video.nvenc.direct3d`; the encoder makes those textures itself,
so :meth:`NVENCEncoder.new_input` is how an input is obtained on both. A GL
context must be current when the encoder is built, and the same context must be
current for every call afterwards: the session belongs to it.

Registration is separate from encoding because it is the expensive half. A
recorder cycles through a handful of textures so the renderer is never writing
the one the encoder is reading; each is registered once, at the start.
:attr:`NVENCEncoder.input_slots` says how many that is -- an encoder holding
frames back for reordering or lookahead is still reading them, and a texture it
holds cannot be handed to it again until its picture comes out.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from OpenGL.GL import GL_RENDERER, GL_TEXTURE_2D, glGetString

from pyopengl_video import codecs, inputs
from pyopengl_video.encoder import (
    DEFAULT_BITS_PER_PIXEL,
    Encoder,
    EncoderError,
    Packet,
    default_bitrate,
    frame_rate_ratio,
)
from pyopengl_video.nvenc import api

if TYPE_CHECKING:
    from pyopengl_video.nvenc.direct3d import Direct3DInputs
    from pyopengl_video.windows import d3d11

log = logging.getLogger(__name__)

__all__ = ['NVENCEncoder', 'InputHandle', 'DEFAULT_BITS_PER_PIXEL',
           'DIRECTX_PLATFORM', 'RegisteredInput', 'SPARE_OUTPUT_BUFFERS', 'configure_codec', 'current_gl_renderer',
           'open_session', 'session_codecs', 'unsupported_device_detail']

#: Where textures reach the encoder through Direct3D 11 rather than by OpenGL
#: name. NVIDIA supports the OpenGL device type on Linux alone.
DIRECTX_PLATFORM = sys.platform == 'win32'

#: How many compressed-output buffers to keep, over and above the frames the
#: encoder may hold for reordering.
SPARE_OUTPUT_BUFFERS = 4


def current_gl_renderer() -> str | None:
    """What the current OpenGL context draws with, or ``None`` if there is none.

    ``glGetString`` needs a context to answer, and returns nothing without one,
    which is how the two cases are told apart.
    """
    renderer = glGetString(GL_RENDERER)
    if not renderer:
        return None
    if isinstance(renderer, bytes):
        return renderer.decode('utf-8', 'replace')
    return str(renderer)


def unsupported_device_detail() -> str:
    """Why NVENC would not take the OpenGL context it was offered.

    ``NV_ENC_ERR_UNSUPPORTED_DEVICE`` arrives for two different situations, and
    they send a reader to opposite ends of a program. There may be no context
    current on this thread. Or there may be one that is not on an NVIDIA GPU --
    a software renderer, or the integrated part of a hybrid machine -- which
    NVENC cannot read a texture out of however good the other adapter is. The
    renderer the context reports is what separates them.
    """
    renderer = current_gl_renderer()
    if renderer is None:
        return 'an OpenGL context must be current on this thread'
    if 'NVIDIA' in renderer.upper():
        return (f'the current OpenGL context is on {renderer!r}, which is an '
                'NVIDIA renderer, so the encoder refused the device for some '
                'other reason')
    return (f'the current OpenGL context draws with {renderer!r}, and NVENC '
            'reads a texture only from a context on the NVIDIA GPU that holds '
            'the encoder')


def open_session(library: api.NvEncodeAPI,
                 device: ctypes.c_void_p | None = None) -> ctypes.c_void_p:
    """An encoder session on `device`, or on the OpenGL context current here.

    `device` is an ``ID3D11Device`` on Windows, where the session is opened
    through Direct3D; without one the session is opened on the OpenGL context
    current on this thread.

    Raises :class:`~pyopengl_video.nvenc.api.NVENCError` where the driver will
    not take the device, saying why where it can.
    """
    params = api.NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS()
    params.version = library.struct_version(1)
    if device is None:
        params.deviceType = api.NV_ENC_DEVICE_TYPE_OPENGL
        params.device = None                  # the OpenGL device is the current context
    else:
        params.deviceType = api.NV_ENC_DEVICE_TYPE_DIRECTX
        params.device = device
    params.apiVersion = library.api_version
    session = ctypes.c_void_p()
    status = library.nvEncOpenEncodeSessionEx(
        ctypes.byref(params), ctypes.byref(session))
    if status != api.NV_ENC_SUCCESS:
        raise api.NVENCError(
            status, 'nvEncOpenEncodeSessionEx',
            unsupported_device_detail() if device is None else
            'the Direct3D device must be on the NVIDIA GPU that holds the encoder')
    return session


def session_codecs(library: api.NvEncodeAPI,
                   session: ctypes.c_void_p) -> dict[str, tuple[int, int]]:
    """The codecs `session` encodes, each with the largest frame it takes.

    The session lists the codecs of the GPU it was opened on, which differ by
    generation: every NVENC part has H.264, most have HEVC, and AV1 arrived
    with the RTX 40 series. A codec this package has no name for is left out.
    """
    count = ctypes.c_uint32()
    library.check(library.nvEncGetEncodeGUIDCount(session, ctypes.byref(count)),
                  'nvEncGetEncodeGUIDCount', session)
    offered = (api.GUID * max(count.value, 1))()
    library.check(
        library.nvEncGetEncodeGUIDs(session, offered, count.value, ctypes.byref(count)),
        'nvEncGetEncodeGUIDs', session)
    listed = list(offered)[:count.value]
    found = {}
    for name, guid in api.CODEC_GUIDS.items():
        if guid in listed:
            found[name] = (_capability(library, session, guid, api.NV_ENC_CAPS_WIDTH_MAX),
                           _capability(library, session, guid, api.NV_ENC_CAPS_HEIGHT_MAX))
    return found


def _capability(library: api.NvEncodeAPI, session: ctypes.c_void_p, codec: api.GUID,
                capability: int) -> int:
    """One ``NV_ENC_CAPS`` value of `codec` on the session's GPU."""
    query = api.NV_ENC_CAPS_PARAM()
    query.version = library.struct_version(1)
    query.capsToQuery = capability
    value = ctypes.c_int()
    library.check(
        library.nvEncGetEncodeCaps(session, codec, ctypes.byref(query), ctypes.byref(value)),
        'nvEncGetEncodeCaps', session)
    return value.value


def configure_codec(config: api.NV_ENC_CONFIG, codec: str, *, gop: int,
                    frame_rate: tuple[int, int]) -> None:
    """Fill in `codec`'s arm of an encoder configuration.

    Every codec gets a key frame each `gop` pictures, 4:2:0 sampling, its
    headers repeated before every key frame so that a raw stream decodes from
    any of them, and a colour description of limited-range BT.709. The driver
    converts RGB to YUV with a matrix of its own choosing, so the stream has to
    say which, or a player guesses and the picture comes back with the wrong
    saturation.

    The rest of the arm is left as the preset set it.
    """
    numerator, denominator = frame_rate
    if codec == 'h264':
        config.profileGUID = api.NV_ENC_H264_PROFILE_HIGH_GUID
        h264 = config.encodeCodecConfig.h264Config
        h264.idrPeriod = gop
        h264.chromaFormatIDC = 1
        h264.repeatSPSPPS = 1
        # A tick is half a frame in H.264, which counts time in fields.
        _describe_vui(h264.h264VUIParameters, denominator, numerator * 2)
    elif codec == 'hevc':
        hevc = config.encodeCodecConfig.hevcConfig
        hevc.idrPeriod = gop
        hevc.chromaFormatIDC = 1
        hevc.repeatSPSPPS = 1
        _describe_vui(hevc.hevcVUIParameters, denominator, numerator)
    elif codec == 'av1':
        av1 = config.encodeCodecConfig.av1Config
        av1.idrPeriod = gop
        av1.chromaFormatIDC = 1
        av1.repeatSeqHdr = 1
        av1.colorPrimaries = api.NV_ENC_VUI_BT709
        av1.transferCharacteristics = api.NV_ENC_VUI_BT709
        av1.matrixCoefficients = api.NV_ENC_VUI_BT709
        av1.colorRange = 0                        # limited range, 16-235
    else:
        raise EncoderError(
            f'the NVENC backend encodes {", ".join(sorted(api.CODEC_GUIDS))}, '
            f'not {codec!r}')


def _describe_vui(vui: api.NV_ENC_CONFIG_H264_VUI_PARAMETERS, units_in_tick: int,
                  time_scale: int) -> None:
    """State limited-range BT.709 and the frame rate in a stream's VUI."""
    vui.videoSignalTypePresentFlag = 1
    vui.videoFormat = 5                           # unspecified
    vui.videoFullRangeFlag = 0                    # limited range, 16-235
    vui.colourDescriptionPresentFlag = 1
    vui.colourPrimaries = api.NV_ENC_VUI_BT709
    vui.transferCharacteristics = api.NV_ENC_VUI_BT709
    vui.colourMatrix = api.NV_ENC_VUI_BT709
    vui.timingInfoPresentFlag = 1
    vui.numUnitInTicks = units_in_tick
    vui.timeScale = time_scale


class RegisteredInput(inputs.InputHandle):
    """An input the encoder has registered with the driver.

    registration -- the driver's handle for it, None before it is registered
        and after it is given back
    release -- what closing the handle calls first, set by the encoder to
        unregister it. The driver reads the texture when it unregisters, so a
        texture freed before then is read after it is gone; unregistering from
        the handle's own :meth:`close` makes the order the caller closes in
        immaterial.
    """

    registration: ctypes.c_void_p | None = None
    release: Callable[[Any], None] | None = None

    def close(self) -> None:
        """Unregister from the encoder, then give back the texture."""
        if self.release is not None:
            self.release(self)            # refuses an input still being encoded
            self.release = None
        super().close()


@dataclass
class InputHandle(RegisteredInput):
    """A texture the encoder has been told about.

    texture/target -- what was registered
    registration -- the driver's handle for it
    description -- the structure the driver was handed, kept alive alongside the
        registration it describes
    framebuffer/owns_texture -- see
        :class:`pyopengl_video.inputs.InputHandle`; set when the encoder made
        the texture itself, through
        :meth:`~pyopengl_video.encoder.Encoder.new_input`
    """

    texture: int = 0
    target: int = 0
    registration: ctypes.c_void_p | None = None
    description: api.NV_ENC_INPUT_RESOURCE_OPENGL_TEX | None = None
    framebuffer: int = 0
    owns_texture: bool = False


@dataclass
class _Submission:
    """A picture given to the encoder, waiting for its compressed form."""

    output: ctypes.c_void_p
    mapped: ctypes.c_void_p
    handle: Any


class NVENCEncoder(Encoder):
    """H.264, HEVC or AV1 from OpenGL textures, on NVIDIA hardware.

    width/height -- frame size; :func:`session_codecs` gives the largest the
        GPU takes in each codec
    codec -- ``'h264'``, ``'hevc'`` or ``'av1'``, where the GPU has an encoder
        for it; one it lacks is refused with the list of those it has
    fps -- frames per second, as a number or a ``(numerator, denominator)`` pair.
        It sets the stream's declared frame rate and the rate control's idea of
        time; it does not pace anything.
    bitrate -- target bits per second; by default derived from the frame size and
        rate at :data:`DEFAULT_BITS_PER_PIXEL`
    preset -- ``'p1'`` (fastest) through ``'p7'`` (best quality)
    tuning -- ``'high_quality'``, ``'low_latency'``, ``'ultra_low_latency'``,
        ``'lossless'`` or ``'ultra_high_quality'``
    rate_control -- ``'vbr'``, ``'cbr'`` or ``'constqp'``
    gop -- frames between key frames; defaults to two seconds' worth. Every
        group starts with an IDR or an AV1 key frame, which is where a player
        can seek to.
    bframes -- B-pictures between reference pictures. Above zero the encoder
        holds frames back before emitting them and packets come out in decode
        order, which :attr:`reorders_frames` reports.

    adapter -- Windows only: which GPU to encode on. By default it is the one
        the current OpenGL context runs on, which must be NVIDIA's.

    Once open, :attr:`input_slots` says how many textures to cycle through.
    Obtain each with :meth:`new_input`, which works on both platforms; on
    Windows it is the only way, since the textures are Direct3D resources.
    """

    codec = 'h264'
    zero_copy = True
    #: Windows inputs are Direct3D resources that only the encoder can make.
    allocates_inputs = DIRECTX_PLATFORM

    #: How many textures a caller should register and cycle through. Set when
    #: the encoder is initialised, from the configuration the driver settled on.
    input_slots = 1

    def __init__(self, width: int, height: int, *, codec: str = 'h264',
                 fps: float | tuple[int, int] = 60, bitrate: int | None = None,
                 preset: str = 'p4', tuning: str = 'high_quality',
                 rate_control: str = 'vbr', gop: int | None = None,
                 bframes: int = 0, adapter: d3d11.Adapter | None = None):
        if adapter is not None and not DIRECTX_PLATFORM:
            raise EncoderError(
                'an adapter can be named only on Windows, where a session is '
                'opened on a Direct3D device; elsewhere it uses the current '
                'OpenGL context')
        codec = codecs.canonical(codec)
        if codec not in api.CODEC_GUIDS:
            raise EncoderError(
                f'the NVENC backend encodes {", ".join(sorted(api.CODEC_GUIDS))}, '
                f'not {codec!r}')
        self.codec = codec
        self.size = (int(width), int(height))
        self.frame_rate = self._as_ratio(fps)
        self.bframes = int(bframes)
        self.reorders_frames = self.bframes > 0
        rate_num, rate_den = self.frame_rate
        self.gop = int(gop) if gop is not None else max(1, round(2 * rate_num / rate_den))
        self.bitrate = int(bitrate) if bitrate is not None else self._default_bitrate()

        self.api = api.NvEncodeAPI.load()
        self._closed = False
        self._registered: list[Any] = []
        self._pending: deque[_Submission] = deque()
        self.session = ctypes.c_void_p()
        self.direct3d: Direct3DInputs | None = None
        self.buffer_format = api.NV_ENC_BUFFER_FORMAT_ABGR
        try:
            self.session = self._open_session(adapter)
            self._require_codec()
            self._initialise(preset, tuning, rate_control)
            self._free_output = [self._create_bitstream_buffer()
                                 for _ in range(self.bframes + SPARE_OUTPUT_BUFFERS)]
        except Exception:
            self.close()
            raise


    # ---------------------------------------------------------------- setup

    _as_ratio = staticmethod(frame_rate_ratio)

    def _default_bitrate(self) -> int:
        return default_bitrate(self.size, self.frame_rate)

    def _open_session(self, adapter: d3d11.Adapter | None) -> ctypes.c_void_p:
        """Open an encoder session on the current OpenGL context's GPU.

        Through the context itself on Linux, and through a Direct3D 11 device
        on the same GPU on Windows.
        """
        if not DIRECTX_PLATFORM:
            return open_session(self.api)
        from pyopengl_video.nvenc import (  # noqa: PLC0415 - imports the Windows OpenGL bindings
            direct3d,
        )

        self.direct3d = direct3d.Direct3DInputs(adapter)
        self.buffer_format = direct3d.BUFFER_FORMAT
        return open_session(self.api, self.direct3d.pointer)

    def _require_codec(self) -> None:
        """Refuse a codec this GPU's encoder does not have, naming those it has."""
        offered = session_codecs(self.api, self.session)
        if self.codec not in offered:
            raise EncoderError(
                f'the NVIDIA encoder on this GPU does not encode {self.codec}; '
                f'it offers {", ".join(sorted(offered)) or "nothing this package names"}')
        most = offered[self.codec]
        if self.size[0] > most[0] or self.size[1] > most[1]:
            raise EncoderError(
                f'{self.size[0]}x{self.size[1]} is larger than the {most[0]}x{most[1]} '
                f'this GPU encodes in {self.codec}')

    def _preset_config(self, preset: str, tuning: str) -> api.NV_ENC_PRESET_CONFIG:
        """The driver's own configuration for `preset` and `tuning`."""
        try:
            preset_guid = api.PRESET_GUIDS[preset]
        except KeyError:
            raise EncoderError(
                f'unknown preset {preset!r}; known: {sorted(api.PRESET_GUIDS)}') from None
        try:
            tuning_info = api.TUNING_INFO[tuning]
        except KeyError:
            raise EncoderError(
                f'unknown tuning {tuning!r}; known: {sorted(api.TUNING_INFO)}') from None
        config = api.NV_ENC_PRESET_CONFIG()
        config.version = self.api.struct_version(5, big=True)
        config.presetCfg.version = self.api.struct_version(9, big=True)
        self.api.check(
            self.api.nvEncGetEncodePresetConfigEx(
                self.session, api.CODEC_GUIDS[self.codec], preset_guid,
                tuning_info, ctypes.byref(config)),
            'nvEncGetEncodePresetConfigEx', self.session)
        self._preset_guid = preset_guid
        self._tuning_info = tuning_info
        return config

    def _initialise(self, preset: str, tuning: str, rate_control: str) -> None:
        """Configure and start the encoder."""
        preset_config = self._preset_config(preset, tuning)
        config = preset_config.presetCfg
        config.version = self.api.struct_version(9, big=True)
        config.gopLength = self.gop
        config.frameIntervalP = self.bframes + 1

        modes = {'vbr': api.NV_ENC_PARAMS_RC_VBR, 'cbr': api.NV_ENC_PARAMS_RC_CBR,
                 'constqp': api.NV_ENC_PARAMS_RC_CONSTQP}
        try:
            config.rcParams.rateControlMode = modes[rate_control]
        except KeyError:
            raise EncoderError(
                f'unknown rate control {rate_control!r}; known: {sorted(modes)}') from None
        config.rcParams.averageBitRate = self.bitrate
        config.rcParams.maxBitRate = (self.bitrate if rate_control == 'cbr'
                                      else self.bitrate * 3 // 2)

        configure_codec(config, self.codec, gop=self.gop, frame_rate=self.frame_rate)

        params = api.NV_ENC_INITIALIZE_PARAMS()
        params.version = self.api.struct_version(7, big=True)
        params.encodeGUID = api.CODEC_GUIDS[self.codec]
        params.presetGUID = self._preset_guid
        params.encodeWidth = params.maxEncodeWidth = params.darWidth = self.size[0]
        params.encodeHeight = params.maxEncodeHeight = params.darHeight = self.size[1]
        params.frameRateNum, params.frameRateDen = self.frame_rate
        params.enableEncodeAsync = 0              # completion is reported synchronously
        params.enablePTD = 1                      # the encoder decides picture types
        params.tuningInfo = self._tuning_info
        params.bufferFormat = self.buffer_format
        params.encodeConfig = ctypes.pointer(config)
        self.api.check(
            self.api.nvEncInitializeEncoder(self.session, ctypes.byref(params)),
            'nvEncInitializeEncoder', self.session)
        # A frame the encoder is still holding is a frame it may still read, so
        # the caller needs one texture more than the encoder can hold at once.
        # A preset may have turned lookahead on by itself, which is why this is
        # read back from the configuration rather than worked out from bframes.
        held = config.frameIntervalP + (
            config.rcParams.lookaheadDepth if config.rcParams.enableLookahead else 0)
        self.input_slots = held + 1
        # The driver copies the configuration during initialisation; holding it
        # keeps the pointer we passed valid for the length of that call.
        self._config = preset_config

    def _create_bitstream_buffer(self) -> ctypes.c_void_p:
        """One buffer for the driver to write a compressed picture into."""
        buffer = api.NV_ENC_CREATE_BITSTREAM_BUFFER()
        buffer.version = self.api.struct_version(1)
        self.api.check(
            self.api.nvEncCreateBitstreamBuffer(self.session, ctypes.byref(buffer)),
            'nvEncCreateBitstreamBuffer', self.session)
        return ctypes.c_void_p(buffer.bitstreamBuffer)

    # ------------------------------------------------------------- encoding

    def new_input(self) -> Any:
        """A texture to draw the next frame into, with a framebuffer that fills it.

        Ask for :attr:`input_slots` of them at the start of a recording and
        cycle through them, drawing into each inside its ``for_drawing`` scope.
        On Windows the texture is one allocation seen twice, as a Direct3D
        texture the encoder reads and an OpenGL texture the renderer draws into.
        """
        if not DIRECTX_PLATFORM:
            return super().new_input()
        self._require_open()
        assert self.direct3d is not None
        width, height = self.size
        shared = self.direct3d.create(width, height)
        try:
            shared.registration = self._register_resource(
                api.NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX,
                ctypes.cast(shared.resource.pointer, ctypes.c_void_p), 0)
        except Exception:
            shared.close()
            raise
        return self._adopt(shared)

    def register(self, texture: int, target: int | None = None) -> InputHandle:
        """Tell the encoder about `texture`, returning the handle to encode from.

        The texture must be RGBA with 8 bits a channel and the encoder's frame
        size, and it must stay alive as long as the handle does. Windows
        encoders read Direct3D textures and take no others: call
        :meth:`new_input` there.
        """
        if DIRECTX_PLATFORM:
            raise EncoderError(
                'this encoder reads a Direct3D surface, so it cannot take a texture '
                'made elsewhere; call new_input() for one it can read')
        target = GL_TEXTURE_2D if target is None else target
        description = api.NV_ENC_INPUT_RESOURCE_OPENGL_TEX(
            texture=int(texture), target=int(target))
        # For an OpenGL texture the pitch is the width in components, not bytes
        # of a row of some allocation: four for RGBA.
        registration = self._register_resource(
            api.NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX,
            ctypes.cast(ctypes.byref(description), ctypes.c_void_p), self.size[0] * 4)
        handle = InputHandle(
            texture=int(texture), target=int(target),
            registration=registration, description=description)
        return self._adopt(handle)

    def _adopt(self, handle: Any) -> Any:
        """Keep `handle` registered until it is closed or the encoder is."""
        handle.release = self._unregister
        self._registered.append(handle)
        return handle

    def _unregister(self, handle: Any) -> None:
        """Give the driver back its registration of `handle`.

        Called by the handle as it closes, and by the encoder for the handles
        left when it does. Refused for a picture the encoder is still encoding.
        """
        if any(pending.handle is handle for pending in self._pending):
            raise EncoderError(
                f'texture {handle.texture} is still being encoded; it can be closed '
                'once the encoder has returned its picture')
        self._registered = [kept for kept in self._registered if kept is not handle]
        if handle.registration is not None and self.session:
            self.api.nvEncUnregisterResource(self.session, handle.registration)
        handle.registration = None

    def _register_resource(self, kind: int, resource: ctypes.c_void_p,
                           pitch: int) -> ctypes.c_void_p:
        """Register `resource` of resource type `kind`, returning the driver's handle."""
        request = api.NV_ENC_REGISTER_RESOURCE()
        request.version = self.api.struct_version(5)
        request.resourceType = kind
        request.width, request.height = self.size
        request.pitch = pitch
        request.resourceToRegister = resource
        request.bufferFormat = self.buffer_format
        request.bufferUsage = api.NV_ENC_BUFFER_USAGE_INPUT_IMAGE
        self.api.check(
            self.api.nvEncRegisterResource(self.session, ctypes.byref(request)),
            'nvEncRegisterResource', self.session)
        return ctypes.c_void_p(request.registeredResource)

    def encode(self, handle: Any, timestamp: int, duration: int = 0,
               force_idr: bool = False) -> list[Packet]:
        """Submit the picture in `handle`; return whatever the encoder let go of.

        An encoder with B-pictures or lookahead holds frames back, so an empty
        list is an ordinary answer and the packets arrive a few calls later.
        """
        self._require_open()
        duration = self.frame_duration(duration)

        if any(pending.handle is handle for pending in self._pending):
            raise EncoderError(
                f'texture {handle.texture} is still being encoded; cycle through at '
                f'least {self.input_slots} registered textures so the encoder is '
                'never handed one it has not finished with')
        mapped = self._map(handle)
        output = self._acquire_output()
        picture = api.NV_ENC_PIC_PARAMS()
        picture.version = self.api.struct_version(7, big=True)
        picture.inputWidth, picture.inputHeight = self.size
        picture.inputPitch = self.size[0]
        picture.inputBuffer = mapped
        picture.outputBitstream = output
        picture.bufferFmt = self.buffer_format
        picture.pictureStruct = api.NV_ENC_PIC_STRUCT_FRAME
        picture.inputTimeStamp = int(timestamp)
        picture.inputDuration = int(duration)
        if force_idr:
            picture.encodePicFlags = api.NV_ENC_PIC_FLAG_FORCEIDR

        status = self.api.nvEncEncodePicture(self.session, ctypes.byref(picture))
        self._pending.append(_Submission(output=output, mapped=mapped, handle=handle))
        if status == api.NV_ENC_ERR_NEED_MORE_INPUT:
            return []
        self.api.check(status, 'nvEncEncodePicture', self.session)
        return self._drain()

    def flush(self) -> list[Packet]:
        """End the stream and return the pictures the encoder was still holding."""
        self._require_open()
        picture = api.NV_ENC_PIC_PARAMS()
        picture.version = self.api.struct_version(7, big=True)
        picture.encodePicFlags = api.NV_ENC_PIC_FLAG_EOS
        self.api.check(
            self.api.nvEncEncodePicture(self.session, ctypes.byref(picture)),
            'nvEncEncodePicture(EOS)', self.session)
        return self._drain()

    def headers(self) -> bytes:
        """The stream's configuration, as a container stores it up front.

        The parameter sets as an Annex-B fragment for H.264 and HEVC, the
        sequence header OBU for AV1.
        """
        self._require_open()
        buffer = (ctypes.c_uint8 * 1024)()
        written = ctypes.c_uint32()
        payload = api.NV_ENC_SEQUENCE_PARAM_PAYLOAD()
        payload.version = self.api.struct_version(1)
        payload.inBufferSize = ctypes.sizeof(buffer)
        payload.spsppsBuffer = ctypes.cast(buffer, ctypes.c_void_p)
        payload.outSPSPPSPayloadSize = ctypes.pointer(written)
        self.api.check(
            self.api.nvEncGetSequenceParams(self.session, ctypes.byref(payload)),
            'nvEncGetSequenceParams', self.session)
        return bytes(buffer[:written.value])

    def close(self) -> None:
        """Release the session and everything registered against it."""
        if getattr(self, '_closed', True):
            return
        self._closed = True
        for submission in self._pending:
            self._unmap(submission.mapped)
        self._pending.clear()
        for buffer in getattr(self, '_free_output', []):
            self.api.nvEncDestroyBitstreamBuffer(self.session, buffer)
        self._free_output = []
        for handle in list(self._registered):
            handle.close()
        self._registered = []
        if self.session:
            self.api.nvEncDestroyEncoder(self.session)
            self.session = ctypes.c_void_p()
        if self.direct3d is not None:
            self.direct3d.close()
            self.direct3d = None

    # ------------------------------------------------------------ internals

    def _require_open(self) -> None:
        if self._closed:
            raise EncoderError('the encoder is closed')

    def _map(self, handle: Any) -> ctypes.c_void_p:
        """Make a registered texture readable by the encoder for one picture."""
        request = api.NV_ENC_MAP_INPUT_RESOURCE()
        request.version = self.api.struct_version(4)
        request.registeredResource = handle.registration
        self.api.check(
            self.api.nvEncMapInputResource(self.session, ctypes.byref(request)),
            'nvEncMapInputResource', self.session)
        return ctypes.c_void_p(request.mappedResource)

    def _unmap(self, mapped: ctypes.c_void_p) -> None:
        self.api.nvEncUnmapInputResource(self.session, mapped)

    def _acquire_output(self) -> ctypes.c_void_p:
        """A free compressed-output buffer, making another if the pool is empty."""
        if self._free_output:
            return self._free_output.pop()
        return self._create_bitstream_buffer()

    def _drain(self) -> list[Packet]:
        """Read out every picture the encoder has finished, oldest first."""
        packets = []
        while self._pending:
            submission = self._pending.popleft()
            packets.append(self._read(submission.output))
            self._unmap(submission.mapped)
            self._free_output.append(submission.output)
        return packets

    def _read(self, output: ctypes.c_void_p) -> Packet:
        """Copy one compressed picture out of the driver's buffer."""
        lock = api.NV_ENC_LOCK_BITSTREAM()
        lock.version = self.api.struct_version(2, big=True)
        lock.outputBitstream = output
        self.api.check(
            self.api.nvEncLockBitstream(self.session, ctypes.byref(lock)),
            'nvEncLockBitstream', self.session)
        try:
            data = ctypes.string_at(lock.bitstreamBufferPtr, lock.bitstreamSizeInBytes)
            keyframe = lock.pictureType in (api.NV_ENC_PIC_TYPE_IDR, api.NV_ENC_PIC_TYPE_I)
            return Packet(data=data, timestamp=lock.outputTimeStamp,
                          duration=lock.outputDuration, keyframe=keyframe)
        finally:
            self.api.check(
                self.api.nvEncUnlockBitstream(self.session, output),
                'nvEncUnlockBitstream', self.session)

    def __del__(self) -> None:
        # A recording that ended in an exception should still give the session
        # back; the driver limits how many are open at once. A destructor has
        # nowhere to raise to, so whatever went wrong is logged and dropped.
        try:
            self.close()
        except Exception as error:            # noqa: BLE001  # pragma: no cover
            log.debug('closing the encoder from its destructor failed: %r', error)

