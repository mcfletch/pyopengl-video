"""The NVENC encoder: an OpenGL texture in, an H.264 elementary stream out.

The session is opened with NvEncodeAPI's OpenGL device type, which is what makes
the frame's journey a short one -- the encoder reads the texture the renderer
just wrote, in GPU memory, and nothing crosses the bus but the compressed
result. A GL context must be current when the encoder is built, and the same
context must be current for every call afterwards: the session belongs to it.

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
from collections import deque
from dataclasses import dataclass

from OpenGL.GL import GL_TEXTURE_2D

from pyopengl_video.encoder import Encoder, EncoderError, Packet
from pyopengl_video.nvenc import api

log = logging.getLogger(__name__)

#: Bits per pixel per second, used when a caller names no bitrate. 1080p60 lands
#: near 8.7 Mbit/s, which is a reasonable quality for screen-captured 3D.
DEFAULT_BITS_PER_PIXEL = 0.07

#: How many compressed-output buffers to keep, over and above the frames the
#: encoder may hold for reordering.
SPARE_OUTPUT_BUFFERS = 4


@dataclass
class InputHandle:
    """A texture the encoder has been told about.

    texture/target -- what was registered
    resource -- the driver's handle for it
    description -- the structure the driver was handed, kept alive alongside the
        registration it describes
    """

    texture: int
    target: int
    resource: ctypes.c_void_p
    description: api.NV_ENC_INPUT_RESOURCE_OPENGL_TEX


@dataclass
class _Submission:
    """A picture given to the encoder, waiting for its compressed form."""

    output: ctypes.c_void_p
    mapped: ctypes.c_void_p
    handle: InputHandle


class NVENCEncoder(Encoder):
    """H.264 from OpenGL textures, on NVIDIA hardware.

    width/height -- frame size; H.264 reaches 4096 in each direction
    codec -- ``'h264'``
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
        group starts with an IDR, which is where a player can seek to.
    bframes -- B-pictures between reference pictures. Above zero the encoder
        holds frames back before emitting them and packets come out in decode
        order, which :attr:`reorders_frames` reports.

    Once open, :attr:`input_slots` says how many textures to cycle through.
    """

    codec = 'h264'
    zero_copy = True

    #: How many textures a caller should register and cycle through. Set when
    #: the encoder is initialised, from the configuration the driver settled on.
    input_slots = 1

    def __init__(self, width: int, height: int, *, codec: str = 'h264',
                 fps: float | tuple[int, int] = 60, bitrate: int | None = None,
                 preset: str = 'p4', tuning: str = 'high_quality',
                 rate_control: str = 'vbr', gop: int | None = None,
                 bframes: int = 0):
        if codec != 'h264':
            raise EncoderError(f'NVENC backend encodes h264, not {codec!r}')
        self.size = (int(width), int(height))
        self.frame_rate = self._as_ratio(fps)
        self.bframes = int(bframes)
        self.reorders_frames = self.bframes > 0
        rate_num, rate_den = self.frame_rate
        self.gop = int(gop) if gop is not None else max(1, round(2 * rate_num / rate_den))
        self.bitrate = int(bitrate) if bitrate is not None else self._default_bitrate()

        self.api = api.NvEncodeAPI.load()
        self.session = self._open_session()
        self._closed = False
        self._registered: list[InputHandle] = []
        self._pending: deque[_Submission] = deque()
        try:
            self._initialise(preset, tuning, rate_control)
            self._free_output = [self._create_bitstream_buffer()
                                 for _ in range(self.bframes + SPARE_OUTPUT_BUFFERS)]
        except Exception:
            self.close()
            raise


    # ---------------------------------------------------------------- setup

    @staticmethod
    def _as_ratio(fps: float | tuple[int, int]) -> tuple[int, int]:
        """Frame rate as an exact numerator and denominator.

        A float rate is turned into the ratio broadcast uses for it, so 29.97
        records as 30000/1001 rather than as an approximation that drifts.
        """
        if isinstance(fps, tuple):
            return int(fps[0]), int(fps[1])
        if abs(fps - round(fps)) < 1e-6:
            return int(round(fps)), 1
        return int(round(fps * 1001)), 1001

    def _default_bitrate(self) -> int:
        width, height = self.size
        rate_num, rate_den = self.frame_rate
        return int(width * height * (rate_num / rate_den) * DEFAULT_BITS_PER_PIXEL)

    def _open_session(self) -> ctypes.c_void_p:
        """Open an encoder session against the current OpenGL context."""
        params = api.NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS()
        params.version = self.api.struct_version(1)
        params.deviceType = api.NV_ENC_DEVICE_TYPE_OPENGL
        params.device = None                      # the OpenGL device is the current context
        params.apiVersion = self.api.api_version
        session = ctypes.c_void_p()
        status = self.api.nvEncOpenEncodeSessionEx(
            ctypes.byref(params), ctypes.byref(session))
        if status != api.NV_ENC_SUCCESS:
            raise api.NVENCError(
                status, 'nvEncOpenEncodeSessionEx',
                'an OpenGL context must be current on this thread')
        return session

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
                self.session, api.NV_ENC_CODEC_H264_GUID, preset_guid,
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
        config.profileGUID = api.NV_ENC_H264_PROFILE_HIGH_GUID
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

        h264 = config.encodeCodecConfig.h264Config
        h264.idrPeriod = self.gop
        h264.chromaFormatIDC = 1                  # 4:2:0
        # Parameter sets before every key frame, so a raw stream stays decodable
        # from any key frame. The MP4 muxer keeps its own copy and drops these.
        h264.repeatSPSPPS = 1
        self._describe_colour(h264.h264VUIParameters)

        params = api.NV_ENC_INITIALIZE_PARAMS()
        params.version = self.api.struct_version(7, big=True)
        params.encodeGUID = api.NV_ENC_CODEC_H264_GUID
        params.presetGUID = self._preset_guid
        params.encodeWidth = params.maxEncodeWidth = params.darWidth = self.size[0]
        params.encodeHeight = params.maxEncodeHeight = params.darHeight = self.size[1]
        params.frameRateNum, params.frameRateDen = self.frame_rate
        params.enableEncodeAsync = 0              # Linux reports completion synchronously
        params.enablePTD = 1                      # the encoder decides picture types
        params.tuningInfo = self._tuning_info
        params.bufferFormat = api.NV_ENC_BUFFER_FORMAT_ABGR
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

    def _describe_colour(self, vui: api.NV_ENC_CONFIG_H264_VUI_PARAMETERS) -> None:
        """State the stream's colour and timing in its video usability information.

        The hardware converts RGB to YUV with a matrix of its own choosing, so
        the stream has to say which one, or a player is left guessing and the
        picture comes back with the wrong saturation. This declares limited-range
        BT.709 -- primaries, transfer and matrix alike -- and the frame rate.
        """
        rate_num, rate_den = self.frame_rate
        vui.videoSignalTypePresentFlag = 1
        vui.videoFormat = 5                       # unspecified
        vui.videoFullRangeFlag = 0                # limited range, 16-235
        vui.colourDescriptionPresentFlag = 1
        vui.colourPrimaries = 1                   # BT.709
        vui.transferCharacteristics = 1           # BT.709
        vui.colourMatrix = 1                      # BT.709
        vui.timingInfoPresentFlag = 1
        vui.numUnitInTicks = rate_den
        vui.timeScale = rate_num * 2              # ticks are half a frame, per H.264

    def _create_bitstream_buffer(self) -> ctypes.c_void_p:
        """One buffer for the driver to write a compressed picture into."""
        buffer = api.NV_ENC_CREATE_BITSTREAM_BUFFER()
        buffer.version = self.api.struct_version(1)
        self.api.check(
            self.api.nvEncCreateBitstreamBuffer(self.session, ctypes.byref(buffer)),
            'nvEncCreateBitstreamBuffer', self.session)
        return ctypes.c_void_p(buffer.bitstreamBuffer)

    # ------------------------------------------------------------- encoding

    def register(self, texture: int, target: int | None = None) -> InputHandle:
        """Tell the encoder about `texture`, returning the handle to encode from.

        The texture must be RGBA with 8 bits a channel and the encoder's frame
        size, and it must stay alive as long as the handle does.
        """
        target = GL_TEXTURE_2D if target is None else target
        description = api.NV_ENC_INPUT_RESOURCE_OPENGL_TEX(
            texture=int(texture), target=int(target))
        request = api.NV_ENC_REGISTER_RESOURCE()
        request.version = self.api.struct_version(5)
        request.resourceType = api.NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX
        request.width, request.height = self.size
        # For an OpenGL texture the pitch is the width in components, not bytes
        # of a row of some allocation: four for RGBA.
        request.pitch = self.size[0] * 4
        request.resourceToRegister = ctypes.cast(
            ctypes.byref(description), ctypes.c_void_p)
        request.bufferFormat = api.NV_ENC_BUFFER_FORMAT_ABGR
        request.bufferUsage = api.NV_ENC_BUFFER_USAGE_INPUT_IMAGE
        self.api.check(
            self.api.nvEncRegisterResource(self.session, ctypes.byref(request)),
            'nvEncRegisterResource', self.session)
        handle = InputHandle(
            texture=int(texture), target=int(target),
            resource=ctypes.c_void_p(request.registeredResource),
            description=description)
        self._registered.append(handle)
        return handle

    def encode(self, handle: InputHandle, timestamp: int, duration: int = 0,
               force_idr: bool = False) -> list[Packet]:
        """Submit the picture in `handle`; return whatever the encoder let go of.

        An encoder with B-pictures or lookahead holds frames back, so an empty
        list is an ordinary answer and the packets arrive a few calls later.
        """
        self._require_open()
        if duration <= 0:
            rate_num, rate_den = self.frame_rate
            duration = round(self.timescale * rate_den / rate_num)

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
        picture.bufferFmt = api.NV_ENC_BUFFER_FORMAT_ABGR
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
        """The sequence and picture parameter sets, as an Annex-B fragment."""
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
        for handle in self._registered:
            self.api.nvEncUnregisterResource(self.session, handle.resource)
        self._registered = []
        if self.session:
            self.api.nvEncDestroyEncoder(self.session)
            self.session = ctypes.c_void_p()

    # ------------------------------------------------------------ internals

    def _require_open(self) -> None:
        if self._closed:
            raise EncoderError('the encoder is closed')

    def _map(self, handle: InputHandle) -> ctypes.c_void_p:
        """Make a registered texture readable by the encoder for one picture."""
        request = api.NV_ENC_MAP_INPUT_RESOURCE()
        request.version = self.api.struct_version(4)
        request.registeredResource = handle.resource
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

