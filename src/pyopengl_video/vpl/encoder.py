"""The QuickSync encoder: a Direct3D 11 texture in, a compressed stream out.

Intel's encoder reads a Direct3D surface, and OpenGL cannot hand it one -- so
the encoder makes the surfaces itself, through
:mod:`pyopengl_video.windows.interop`, and each is a GL texture and a D3D11
texture at the same time. :meth:`VPLEncoder.new_input` is where one comes from,
and the renderer draws into it inside its ``for_drawing`` scope::

    with handle.for_drawing():
        blit_the_frame_into(handle.framebuffer)
    packets = encoder.encode(handle, timestamp)

**The encoder and the OpenGL context must be on the same GPU**, or the surfaces
would be in memory the renderer never wrote. The adapter is taken from the
current context unless one is named.

The hardware converts RGB to YUV, as NVIDIA's does, so there is no conversion
pass: the surface is handed over as it was drawn.
"""
from __future__ import annotations

import ctypes
import dataclasses
import logging
import time
import warnings
from collections.abc import Callable
from ctypes import byref, c_void_p
from typing import Any

from pyopengl_video import bounds, codecs, obu
from pyopengl_video.encoder import (
    Encoder,
    EncoderError,
    Packet,
    default_bitrate,
    frame_rate_ratio,
)
from pyopengl_video.inputs import InputHandle
from pyopengl_video.teardown import release_all
from pyopengl_video.vpl import api
from pyopengl_video.vpl.allocator import SurfaceAllocator, aligned
from pyopengl_video.windows import d3d11, interop

log = logging.getLogger(__name__)

#: How long to wait for a picture to come out of the hardware, in milliseconds.
#: Reaching this means the device stopped answering, which is a failure rather
#: than a slow frame: encoding one costs well under a millisecond.
SYNC_TIMEOUT_MS = 5000

#: How long to wait before offering a busy device the same frame again. A
#: device still busy after :data:`SYNC_TIMEOUT_MS` has stopped answering.
DEVICE_BUSY_WAIT = 0.001

#: The quantisers constant-quantiser rate control takes, for 8-bit pictures.
QP_RANGE = (0, 51)

#: The quantiser ``rate_control='constqp'`` codes at when none is named.
DEFAULT_QP = 26

#: The largest width or height a 16-bit field holds once rounded up to
#: :data:`ALIGNMENT`. The hardware's own limit is far below it.
LARGEST_SIDE = bounds.FIELD_16 // 16 * 16

#: The PCI vendor whose adapters have a oneVPL encoder.
INTEL_VENDOR_ID = 0x8086

#: Preset names, shared with the NVENC backend, mapped onto oneVPL's target
#: usage: 1 is the best quality it will do and 7 the fastest.
PRESET_TARGET_USAGE = {'p1': 7, 'p2': 6, 'p3': 5, 'p4': 4, 'p5': 3, 'p6': 2, 'p7': 1}

RATE_CONTROL = {
    'vbr': api.MFX_RATECONTROL_VBR,
    'cbr': api.MFX_RATECONTROL_CBR,
    'constqp': api.MFX_RATECONTROL_CQP,
}

#: The Direct3D format the surfaces are made in, and what oneVPL calls it.
#: ``B8G8R8A8_UNORM`` holds BGRA bytes, which is oneVPL's ``RGB4`` -- the RGB
#: format every generation of the encoder accepts.
SURFACE_FORMAT = d3d11.DXGI_FORMAT_B8G8R8A8_UNORM
SURFACE_FOURCC = api.MFX_FOURCC_RGB4

@dataclasses.dataclass(frozen=True)
class CodecSettings:
    """What the runtime is told to make one codec.

    codec_id/profile -- the ``MFX_CODEC_*`` and ``MFX_PROFILE_*`` values
    idr_interval -- ``IdrInterval`` such that every group opens with a picture
        decoding can start at. H.264 counts from zero and HEVC from one.
    signal_info -- the codec's stream carries a colour description the runtime
        writes from ``mfxExtVideoSignalInfo``
    title -- the codec in words, for messages
    """

    codec_id: int
    profile: int
    idr_interval: int
    signal_info: bool
    title: str


#: The codecs this backend asks the runtime for. Which of them a part encodes
#: depends on its generation, and ``MFXVideoENCODE_Init`` refuses the rest.
CODECS = {
    'h264': CodecSettings(api.MFX_CODEC_AVC, api.MFX_PROFILE_AVC_HIGH, 0, True, 'H.264'),
    'hevc': CodecSettings(api.MFX_CODEC_HEVC, api.MFX_PROFILE_HEVC_MAIN, 1, True, 'HEVC'),
    'av1': CodecSettings(api.MFX_CODEC_AV1, api.MFX_PROFILE_AV1_MAIN, 0, True, 'AV1'),
    'vp9': CodecSettings(api.MFX_CODEC_VP9, api.MFX_PROFILE_VP9_0, 0, False, 'VP9'),
}


def signal_info() -> api.mfxExtVideoSignalInfo:
    """A colour description of limited-range BT.709: primaries, transfer and matrix."""
    signal = api.mfxExtVideoSignalInfo()
    signal.Header.BufferId = api.MFX_EXTBUFF_VIDEO_SIGNAL_INFO
    signal.Header.BufferSz = ctypes.sizeof(signal)
    signal.VideoFormat = 5                 # unspecified
    signal.VideoFullRange = 0              # limited range, 16-235
    signal.ColourDescriptionPresent = 1
    signal.ColourPrimaries = 1             # BT.709
    signal.TransferCharacteristics = 1     # BT.709
    signal.MatrixCoefficients = 1          # BT.709
    return signal


def extension_buffers(codec: str) -> list[Any]:
    """The extension buffers `codec`'s configuration carries.

    The colour description, for the codecs whose stream states one. For AV1
    and VP9, the buffer that keeps the runtime from wrapping the stream in IVF
    headers, so that a packet is a temporal unit or a frame and nothing else.
    """
    found: list[Any] = []
    if CODECS[codec].signal_info:
        found.append(signal_info())
    if codec == 'av1':
        bitstream = api.mfxExtAV1BitstreamParam()
        bitstream.Header.BufferId = api.MFX_EXTBUFF_AV1_BITSTREAM_PARAM
        bitstream.Header.BufferSz = ctypes.sizeof(bitstream)
        bitstream.WriteIVFHeaders = api.MFX_CODINGOPTION_OFF
        found.append(bitstream)
    elif codec == 'vp9':
        vp9 = api.mfxExtVP9Param()
        vp9.Header.BufferId = api.MFX_EXTBUFF_VP9_PARAM
        vp9.Header.BufferSz = ctypes.sizeof(vp9)
        vp9.WriteIVFHeaders = api.MFX_CODINGOPTION_OFF
        found.append(vp9)
    return found


class VPLInput(interop.SharedTexture):
    """A shared texture with the surface the runtime is given for it.

    surface -- this input's own ``mfxFrameSurface1``. An encoder that reorders
        keeps the surfaces it has been given until their pictures are coded, and
        a surface must not change while it does, so the inputs cannot share one.
    forget -- called as the input closes, so the allocator stops resolving a
        texture that is gone
    """

    surface: api.mfxFrameSurface1
    forget: Callable[[], None] | None = None

    @property
    def in_use(self) -> bool:
        """Whether the runtime still holds the surface."""
        return bool(self.surface.Data.Locked)

    def close(self) -> None:
        """Stop the allocator resolving this texture, then give it back."""
        if self.in_use:
            raise EncoderError('this input is still held by the encoder; it can be '
                               'closed once its picture has come out')
        if self.forget is not None:
            self.forget()
            self.forget = None
        super().close()


class VPLEncoder(Encoder):
    """H.264, HEVC, AV1 or VP9 from OpenGL, on Intel hardware, through Direct3D 11.

    width/height -- frame size
    codec -- ``'h264'``, ``'hevc'``, ``'av1'`` or ``'vp9'``, where the part has
        an encoder for it; the runtime refuses one it lacks when the encoder
        is built
    fps -- frames per second, a number or an exact ``(numerator, denominator)``
        pair. It sets the stream's declared rate and rate control's idea of
        time; it paces nothing.
    bitrate -- target bits per second; by default derived from the frame size
        and rate
    preset -- ``'p1'`` (fastest) through ``'p7'`` (best quality), the same names
        the NVENC backend takes
    rate_control -- ``'vbr'``, ``'cbr'`` or ``'constqp'``
    qp -- the quantiser ``'constqp'`` codes every picture at, 0 (largest
        stream) to 51; the other modes do not read it
    gop -- frames between key frames, 1 to 65535; defaults to two seconds' worth
    bframes -- B-pictures between reference pictures, 0 to 65534
    adapter -- which GPU to encode on; by default the one the current OpenGL
        context is running on, which is the only one it can read without a
        copy. An adapter that is not Intel's is refused.

    An OpenGL context must be current when the encoder is built and for every
    call afterwards, :meth:`close` included: the surfaces belong to it. An
    encoder that is collected without having been closed gives back its
    session and its Direct3D objects, warns with a ``ResourceWarning``, and
    leaves the OpenGL names to the context.

    A setting that does not fit the field the runtime reads it from raises
    :class:`~pyopengl_video.encoder.EncoderError` before anything is opened.
    Each input is submitted in a surface of its own, and one the runtime has
    not finished reading is refused: cycle through :attr:`input_slots` inputs.
    """

    codec = 'h264'
    zero_copy = True
    #: The surfaces must be Direct3D resources, which only this backend can make.
    allocates_inputs = True
    input_slots = 2

    def __init__(self, width: int, height: int, *, codec: str = 'h264',
                 fps: float | tuple[int, int] = 60, bitrate: int | None = None,
                 preset: str = 'p4', rate_control: str = 'vbr',
                 gop: int | None = None, bframes: int = 0, qp: int = DEFAULT_QP,
                 adapter: d3d11.Adapter | None = None):
        # Set first, so that a constructor that raises leaves an object whose
        # close() and destructor have nothing to do.
        self._closed = True
        api.require_supported_process()
        codec = codecs.canonical(codec)
        if codec not in CODECS:
            raise EncoderError(
                f'the oneVPL backend encodes {", ".join(sorted(CODECS))}, not {codec!r}')
        self.codec = codec
        self._sequence_header = b''
        self.size = bounds.frame_size(width, height, most=(LARGEST_SIDE, LARGEST_SIDE),
                                      error=EncoderError)
        self.frame_rate = bounds.frame_rate(frame_rate_ratio(fps), error=EncoderError)
        self.bframes = bounds.whole(bframes, 'bframes', 0, bounds.FIELD_16 - 1,
                                    EncoderError)
        self.reorders_frames = self.bframes > 0
        rate_num, rate_den = self.frame_rate
        self.gop = bounds.whole(
            gop if gop is not None
            else min(bounds.FIELD_16, max(1, round(2 * rate_num / rate_den))),
            'gop', 1, bounds.FIELD_16, EncoderError)
        self.bitrate = bounds.whole(
            bitrate if bitrate is not None
            else max(1, default_bitrate(self.size, self.frame_rate)),
            'bitrate', 1, bounds.FIELD_16 * bounds.FIELD_16 * 1000, EncoderError)
        self.qp = bounds.whole(qp, 'qp', *QP_RANGE, EncoderError)

        self._closed = False
        self._initialised = False
        #: Set when the runtime may still be writing the bitstream buffer, so
        #: nothing more is submitted into it.
        self._stopped = ''
        self._frames = 0
        self._inputs: list[VPLInput] = []
        self.session = c_void_p()
        self.device: d3d11.Device | None = None
        self.interop: interop.InteropDevice | None = None
        self.allocator: SurfaceAllocator | None = None

        self.api = api.VPL.load()
        try:
            self.adapter = self._choose_adapter(adapter)
            self.device = d3d11.Device(self.adapter)
            self.interop = interop.InteropDevice(self.device)
            self._open_session()
            self.allocator = SurfaceAllocator(self.device)
            self._initialise(preset, rate_control)
        except Exception:
            self.close()
            raise

    # ---------------------------------------------------------------- setup

    @staticmethod
    def _choose_adapter(adapter: d3d11.Adapter | None) -> d3d11.Adapter:
        """Which GPU to encode on, and refuse to guess if it cannot be known."""
        if adapter is not None:
            if adapter.vendor_id != INTEL_VENDOR_ID:
                raise EncoderError(
                    f'{adapter} is not an Intel adapter, and the oneVPL encoder '
                    'runs on those alone')
            return adapter
        found = interop.adapter_for_context()
        if found is None:
            raise EncoderError(
                'cannot tell which GPU this OpenGL context is on, so an encoder '
                'here might not be able to read what the renderer draws; name an '
                'adapter explicitly to go ahead anyway')
        if found.vendor_id != INTEL_VENDOR_ID:
            raise EncoderError(
                f'the OpenGL context renders on {found}, and the oneVPL encoder '
                'runs on Intel adapters: it would read surfaces the renderer '
                'never wrote')
        return found

    def _open_session(self) -> None:
        """Start a oneVPL session on the chosen adapter, over Direct3D 11."""
        version = api.mfxVersion(0, 1)
        session = c_void_p()
        status = self.api.MFXInit(
            api.MFX_IMPL_HARDWARE_ANY | api.MFX_IMPL_VIA_D3D11,
            byref(version), byref(session))
        if status < 0:
            raise api.VPLError(
                status, 'MFXInit',
                'no Intel hardware encoder is reachable through Direct3D 11')
        self.session = session
        self.api_version = (version.Major, version.Minor)
        assert self.device is not None
        self.api.check(
            self.api.MFXVideoCORE_SetHandle(
                self.session, api.MFX_HANDLE_D3D11_DEVICE, self.device.pointer),
            'MFXVideoCORE_SetHandle')

    def _frame_info(self, info: api.mfxFrameInfo) -> None:
        """Describe this recording's pictures into `info`."""
        width, height = self.size
        info.FourCC = SURFACE_FOURCC
        # An RGB surface has no chroma subsampling; the encoder converts it.
        info.ChromaFormat = api.MFX_CHROMAFORMAT_YUV444
        info.PicStruct = api.MFX_PICSTRUCT_PROGRESSIVE
        info.Width, info.Height = aligned(width), aligned(height)
        info.CropX, info.CropY = 0, 0
        info.CropW, info.CropH = width, height
        info.FrameRateExtN, info.FrameRateExtD = self.frame_rate

    def _parameters(self, preset: str, rate_control: str) -> api.mfxVideoParam:
        """The configuration to initialise the encoder with."""
        try:
            target_usage = PRESET_TARGET_USAGE[preset]
        except KeyError:
            raise EncoderError(
                f'unknown preset {preset!r}; known: '
                f'{sorted(PRESET_TARGET_USAGE)}') from None
        try:
            mode = RATE_CONTROL[rate_control]
        except KeyError:
            raise EncoderError(
                f'unknown rate control {rate_control!r}; known: '
                f'{sorted(RATE_CONTROL)}') from None

        params = api.mfxVideoParam()
        # Every picture is synchronised before encode() returns, so the runtime
        # is never asked to keep more than one in flight.
        params.AsyncDepth = 1
        params.IOPattern = api.MFX_IOPATTERN_IN_VIDEO_MEMORY
        mfx = params.mfx
        settings = CODECS[self.codec]
        mfx.CodecId = settings.codec_id
        mfx.CodecProfile = settings.profile
        mfx.TargetUsage = target_usage
        mfx.RateControlMethod = mode
        if mode == api.MFX_RATECONTROL_CQP:
            mfx.set_quantisers(self.qp, self.qp, self.qp)
        else:
            # Rate control counts in kilobits, in a 16-bit field: the
            # multiplier is what lets a recording ask for more than 65 Mbit/s.
            kilobits = max(1, self.bitrate // 1000)
            multiplier = max(1, (kilobits + 65534) // 65535)
            mfx.BRCParamMultiplier = multiplier
            mfx.TargetKbps = kilobits // multiplier
            mfx.MaxKbps = (mfx.TargetKbps if rate_control == 'cbr'
                           else min(65535, mfx.TargetKbps * 3 // 2))
        mfx.GopPicSize = self.gop
        mfx.GopRefDist = self.bframes + 1
        mfx.IdrInterval = settings.idr_interval
        self._frame_info(mfx.FrameInfo)
        self._attach_extensions(params)
        return params

    def _attach_extensions(self, params: api.mfxVideoParam) -> None:
        """Attach the codec's extension buffers (:func:`extension_buffers`).

        The hardware converts RGB to YUV with a matrix of its own choosing, so
        the stream says which one it was: limited-range BT.709, as the other
        backends declare. A recording should not describe itself differently
        depending on whose GPU made it.

        The buffers are kept on the encoder because the runtime reads them
        during initialisation, from the pointers handed over here.
        """
        self._extensions = extension_buffers(self.codec)
        self._ext_buffers = (c_void_p * len(self._extensions))(
            *(ctypes.cast(byref(buffer), c_void_p) for buffer in self._extensions))
        params.NumExtParam = len(self._extensions)
        params.ExtParam = ctypes.cast(self._ext_buffers, ctypes.POINTER(c_void_p))

    def _initialise(self, preset: str, rate_control: str) -> None:
        """Configure the encoder and start it."""
        assert self.allocator is not None
        self.api.check(
            self.api.MFXVideoCORE_SetFrameAllocator(
                self.session, byref(self.allocator.structure)),
            'MFXVideoCORE_SetFrameAllocator')

        params = self._parameters(preset, rate_control)
        request = api.mfxFrameAllocRequest()
        status = self.api.MFXVideoENCODE_QueryIOSurf(
            self.session, byref(params), byref(request))
        if status >= 0 and request.NumFrameSuggested:
            # The runtime's own answer for how many pictures it may hold at
            # once; one more so the renderer always has a free one to draw into.
            self.input_slots = max(2, int(request.NumFrameSuggested))

        status = self.api.MFXVideoENCODE_Init(self.session, byref(params))
        if status < 0:
            width, height = self.size
            raise api.VPLError(status, 'MFXVideoENCODE_Init',
                               f'{width}x{height} {self.codec}')
        if status == api.MFX_WRN_PARTIAL_ACCELERATION:
            log.warning('this part encodes %s only partly in hardware',
                        CODECS[self.codec].title)
        self._initialised = True

        # What the driver settled on, which is not always what was asked for.
        settled = api.mfxVideoParam()
        self.api.check(
            self.api.MFXVideoENCODE_GetVideoParam(self.session, byref(settled)),
            'MFXVideoENCODE_GetVideoParam')
        self.parameters = settled
        self.reorders_frames = settled.mfx.GopRefDist > 1
        self._bitstream = self._create_bitstream(settled)

    def _create_bitstream(self, settled: api.mfxVideoParam) -> api.mfxBitstream:
        """A buffer for the compressed pictures, sized as the driver asked."""
        multiplier = max(1, int(settled.mfx.BRCParamMultiplier))
        size = int(settled.mfx.BufferSizeInKB) * multiplier * 1000
        width, height = self.size
        # A floor, for a driver that reports nothing: one uncompressed picture
        # is more than any coded one will need.
        size = max(size, width * height * 2, 64 * 1024)
        self._bitstream_buffer = (ctypes.c_uint8 * size)()
        bitstream = api.mfxBitstream()
        bitstream.Data = ctypes.cast(self._bitstream_buffer,
                                     ctypes.POINTER(ctypes.c_uint8))
        bitstream.MaxLength = size
        return bitstream

    # ------------------------------------------------------------- encoding

    def new_input(self) -> InputHandle:
        """A surface to draw the next frame into.

        It is one allocation seen twice: an OpenGL texture the renderer blits
        into, and the Direct3D texture the encoder reads. Draw inside its
        ``for_drawing`` scope.
        """
        self._require_open()
        assert self.interop is not None and self.allocator is not None
        width, height = self.size
        shared = self.interop.create_texture(
            width, height, SURFACE_FORMAT, factory=VPLInput)
        assert isinstance(shared, VPLInput)
        shared.surface = api.mfxFrameSurface1()
        self._frame_info(shared.surface.Info)
        identifier = self.allocator.adopt(shared.resource)
        shared.surface.Data.MemId = c_void_p(identifier)
        allocator = self.allocator

        def forget() -> None:
            allocator.forget(identifier)
            self._inputs = [kept for kept in self._inputs if kept is not shared]

        shared.forget = forget
        self._inputs.append(shared)
        return shared

    def register(self, texture: int, target: int | None = None) -> Any:  # noqa: ARG002 - Encoder.register's signature
        """Not available here: the encoder reads Direct3D, not an OpenGL name."""
        raise EncoderError(
            'this encoder reads a Direct3D surface, so it cannot take a texture '
            'made elsewhere; call new_input() for one it can read')

    def encode(self, handle: Any, timestamp: int, duration: int = 0,
               force_idr: bool = False) -> list[Packet]:
        """Submit the picture in `handle`; return whatever came out."""
        self._require_running()
        if not any(kept is handle for kept in self._inputs):
            raise EncoderError('encode() wants an open handle from this encoder\'s '
                               'new_input()')
        if handle.held:
            raise EncoderError(
                'this input is still held for drawing; encode after leaving its '
                'for_drawing scope, which is what hands it to the encoder')
        if handle.in_use:
            raise EncoderError(
                'this input is still being encoded; cycle through at least '
                f'{self.input_slots} inputs so the encoder is never handed one it '
                'has not finished with')
        duration = self.frame_duration(duration)

        surface = handle.surface
        surface.Data.TimeStamp = bounds.whole(
            timestamp, 'the timestamp', 0, api.MFX_TIMESTAMP_UNKNOWN - 1, EncoderError)
        surface.Data.FrameOrder = self._frames
        self._frames += 1
        control = self._control(force_idr)
        packet = self._submit(control, byref(surface), duration)
        return [] if packet is None else [packet]

    def flush(self) -> list[Packet]:
        """End the stream, returning the pictures the encoder was still holding."""
        self._require_running()
        packets: list[Packet] = []
        while True:
            packet = self._submit(None, None, self.frame_duration())
            if packet is None:
                return packets
            packets.append(packet)

    def _control(self, force_idr: bool) -> Any:
        """Per-picture instructions, or None when there are none."""
        if not force_idr:
            return None
        control = api.mfxEncodeCtrl()
        control.FrameType = (api.MFX_FRAMETYPE_I | api.MFX_FRAMETYPE_IDR
                             | api.MFX_FRAMETYPE_REF)
        self._last_control = control          # outlive the call
        return byref(control)

    def _submit(self, control: Any, surface: Any, duration: int) -> Packet | None:
        """Offer one picture -- or the end of the stream -- and read the result.

        Returns None when the encoder took the picture without producing
        anything, which is ordinary while it fills its pipeline, and at the end
        of a flush.
        """
        self._bitstream.DataOffset = 0
        self._bitstream.DataLength = 0
        sync_point = c_void_p()
        deadline = time.monotonic() + SYNC_TIMEOUT_MS / 1000
        while True:
            status = self.api.MFXVideoENCODE_EncodeFrameAsync(
                self.session, control, surface, byref(self._bitstream),
                byref(sync_point))
            if status != api.MFX_WRN_DEVICE_BUSY:
                break
            if time.monotonic() >= deadline:
                raise api.VPLError(
                    status, 'MFXVideoENCODE_EncodeFrameAsync',
                    f'the device was still busy after {SYNC_TIMEOUT_MS} ms')
            time.sleep(DEVICE_BUSY_WAIT)
        if status == api.MFX_ERR_MORE_DATA:
            return None
        self.api.check(status, 'MFXVideoENCODE_EncodeFrameAsync')
        if not sync_point:
            return None
        status = self.api.MFXVideoCORE_SyncOperation(
            self.session, sync_point, SYNC_TIMEOUT_MS)
        if status != api.MFX_ERR_NONE:
            # Anything but success leaves the operation unfinished: the runtime
            # may write the bitstream buffer at any time from here on.
            self._stopped = (f'a picture was not coded within {SYNC_TIMEOUT_MS} ms '
                             f'({api.STATUS_NAMES.get(status, status)})')
            raise api.VPLError(status, 'MFXVideoCORE_SyncOperation', self._stopped)
        return self._packet(duration)

    def _packet(self, duration: int) -> Packet:
        """Copy the compressed picture out of the shared buffer."""
        bitstream = self._bitstream
        offset, length = bounds.span(
            bitstream.DataOffset, bitstream.DataLength,
            ctypes.sizeof(self._bitstream_buffer), 'the coded picture', EncoderError)
        data = bytes(memoryview(self._bitstream_buffer)[offset:offset + length])
        keyframe = bool(bitstream.FrameType
                        & (api.MFX_FRAMETYPE_I | api.MFX_FRAMETYPE_IDR))
        timestamp = int(bitstream.TimeStamp)
        if timestamp == api.MFX_TIMESTAMP_UNKNOWN:      # pragma: no cover - driver quirk
            timestamp = 0
        if keyframe and self.codec == 'av1':
            self._sequence_header = obu.sequence_header(data) or self._sequence_header
        return Packet(data=data, timestamp=timestamp, duration=duration,
                      keyframe=keyframe)

    def headers(self) -> bytes:
        """The stream's configuration, as a container stores it up front.

        The parameter sets as an Annex-B fragment for H.264 and HEVC. For AV1,
        the sequence header OBU of the last key frame coded, which is empty
        before the first. VP9 has none.
        """
        self._require_open()
        if self.codec == 'av1':
            return self._sequence_header
        if self.codec == 'vp9':
            return b''
        vps = (ctypes.c_uint8 * 1024)()
        sps = (ctypes.c_uint8 * 1024)()
        pps = (ctypes.c_uint8 * 1024)()
        option = api.mfxExtCodingOptionSPSPPS()
        option.Header.BufferId = api.MFX_EXTBUFF_CODING_OPTION_SPSPPS
        option.Header.BufferSz = ctypes.sizeof(option)
        option.SPSBuffer = ctypes.cast(sps, ctypes.POINTER(ctypes.c_uint8))
        option.PPSBuffer = ctypes.cast(pps, ctypes.POINTER(ctypes.c_uint8))
        option.SPSBufSize = ctypes.sizeof(sps)
        option.PPSBufSize = ctypes.sizeof(pps)
        wanted: list[Any] = [option]
        video = api.mfxExtCodingOptionVPS()
        if self.codec == 'hevc':
            video.Header.BufferId = api.MFX_EXTBUFF_CODING_OPTION_VPS
            video.Header.BufferSz = ctypes.sizeof(video)
            video.VPSBuffer = ctypes.cast(vps, ctypes.POINTER(ctypes.c_uint8))
            video.VPSBufSize = ctypes.sizeof(vps)
            wanted.append(video)

        buffers = (c_void_p * len(wanted))(
            *(ctypes.cast(byref(buffer), c_void_p) for buffer in wanted))
        params = api.mfxVideoParam()
        params.NumExtParam = len(wanted)
        params.ExtParam = ctypes.cast(buffers, ctypes.POINTER(c_void_p))
        self.api.check(
            self.api.MFXVideoENCODE_GetVideoParam(self.session, byref(params)),
            'MFXVideoENCODE_GetVideoParam(parameter sets)')
        found = bytes(sps[:option.SPSBufSize]) + bytes(pps[:option.PPSBufSize])
        if self.codec == 'hevc':
            found = bytes(vps[:video.VPSBufSize]) + found
        return found

    # ------------------------------------------------------------ internals

    def _require_open(self) -> None:
        if self._closed:
            raise EncoderError('the encoder is closed')

    def _require_running(self) -> None:
        self._require_open()
        if self._stopped:
            raise EncoderError(f'the encoder has stopped: {self._stopped}; close it')

    def close(self) -> None:
        """Release the session, the surfaces and the device. Safe to call twice.

        Every release is attempted whatever the others do, and the first
        failure is raised once all have been: an OpenGL error here, such as no
        context being current, leaves no session open behind it.
        """
        if self._closed:
            return
        self._closed = True
        release_all(self._close_encoding, self._close_session, self._close_inputs,
                    self._close_allocator, self._close_interop, self._close_device)

    def _close_encoding(self) -> None:
        if self._initialised and self.session:
            self._initialised = False
            self.api.MFXVideoENCODE_Close(self.session)

    def _close_inputs(self) -> None:
        """Close every input, after the session: nothing holds a surface then."""
        inputs, self._inputs = self._inputs, []
        for shared in inputs:
            shared.surface.Data.Locked = 0
        release_all(*(shared.close for shared in inputs))

    def _close_session(self) -> None:
        session, self.session = self.session, c_void_p()
        if session:
            self.api.MFXClose(session)

    def _close_allocator(self) -> None:
        allocator, self.allocator = self.allocator, None
        if allocator is not None:
            allocator.close()

    def _close_interop(self) -> None:
        bridge, self.interop = self.interop, None
        if bridge is not None:
            bridge.close()

    def _close_device(self) -> None:
        device, self.device = self.device, None
        if device is not None:
            device.close()

    def __del__(self) -> None:
        # Collection runs on whichever thread, with whichever OpenGL context,
        # the collector is in, so nothing here touches OpenGL: a name deleted
        # in another context is that context's own object. The session goes
        # first, since the runtime holds the addresses of the callbacks and the
        # buffers this object is about to free.
        if getattr(self, '_closed', True):
            return
        self._closed = True
        warnings.warn(f'{type(self).__name__} was collected without close(); its '
                      'OpenGL textures and framebuffers are left to their context',
                      ResourceWarning, stacklevel=2)
        try:
            release_all(self._close_encoding, self._close_session,
                        self._close_allocator, self._close_device)
        except Exception:
            log.debug('releasing a collected encoder failed', exc_info=True)
