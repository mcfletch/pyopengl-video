"""The QuickSync encoder: a Direct3D 11 texture in, an H.264 stream out.

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
import logging
import time
from ctypes import byref, c_void_p
from typing import Any

from pyopengl_video.encoder import (
    Encoder,
    EncoderError,
    Packet,
    default_bitrate,
    frame_rate_ratio,
)
from pyopengl_video.inputs import InputHandle
from pyopengl_video.vpl import api
from pyopengl_video.windows import d3d11, interop

log = logging.getLogger(__name__)

#: How long to wait for a picture to come out of the hardware, in milliseconds.
#: Reaching this means the device stopped answering, which is a failure rather
#: than a slow frame: encoding one costs well under a millisecond.
SYNC_TIMEOUT_MS = 5000

#: How long to wait before offering a busy device the same frame again.
DEVICE_BUSY_WAIT = 0.001

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

#: Rounding the encoder's own surfaces demand. H.264 codes in macroblocks.
ALIGNMENT = 16


def _aligned(value: int, to: int = ALIGNMENT) -> int:
    """`value` rounded up to a multiple of `to`."""
    return (int(value) + to - 1) // to * to


class SurfaceAllocator:
    """The surfaces oneVPL asks for, as Direct3D textures on our own device.

    The runtime allocates its internal frames -- the reconstructed pictures it
    predicts from, and the NV12 surface it converts our RGB into -- by calling
    back into this. It also resolves the surfaces *we* supply: a
    :meth:`get_handle` on one of our textures is how it learns which Direct3D
    resource a picture is in.

    Memory identifiers are the texture pointers themselves, which are unique for
    as long as the texture is alive and are exactly what ``GetHDL`` has to
    return.
    """

    def __init__(self, device: d3d11.Device):
        self.device = device
        #: Textures this allocator made, by the identifier handed to the runtime.
        self.owned: dict[int, d3d11.Texture] = {}
        #: Textures made elsewhere -- the shared ones -- that it must resolve.
        self.external: dict[int, d3d11.Texture] = {}
        self._responses: list[Any] = []
        # ctypes callbacks must outlive the C structure that points at them.
        self._alloc = api.ALLOC_FUNC(self.allocate)
        self._lock = api.LOCK_FUNC(self.lock)
        self._unlock = api.LOCK_FUNC(self.lock)
        self._get_handle = api.GETHDL_FUNC(self.get_handle)
        self._free = api.FREE_FUNC(self.free)
        self.structure = api.mfxFrameAllocator()
        self.structure.Alloc = self._alloc
        self.structure.Lock = self._lock
        self.structure.Unlock = self._unlock
        self.structure.GetHDL = self._get_handle
        self.structure.Free = self._free

    def adopt(self, texture: d3d11.Texture) -> int:
        """Take responsibility for resolving `texture`, returning its identifier."""
        identifier = ctypes.cast(texture.pointer, c_void_p).value or 0
        self.external[identifier] = texture
        return identifier

    def forget(self, identifier: int) -> None:
        """Stop resolving the texture with this identifier."""
        self.external.pop(identifier, None)

    # ------------------------------------------------------- the callbacks

    def allocate(self, _pthis: Any, request: Any, response: Any) -> int:
        """Make the surfaces the runtime asked for."""
        wanted = request.contents
        count = max(int(wanted.NumFrameMin), int(wanted.NumFrameSuggested))
        if not count:
            return api.MFX_ERR_NONE
        fourcc = int(wanted.Info.FourCC)
        format = _DXGI_FOR_FOURCC.get(fourcc)
        if format is None:
            log.debug('oneVPL asked for %d surfaces of %s, which has no D3D11 format',
                      count, api.fourcc_name(fourcc))
            return -3                                  # MFX_ERR_UNSUPPORTED
        identifiers = (c_void_p * count)()
        made = []
        try:
            for slot in range(count):
                texture = self.device.create_texture(
                    _aligned(wanted.Info.Width), _aligned(wanted.Info.Height), format)
                made.append(texture)
                identifier = ctypes.cast(texture.pointer, c_void_p).value or 0
                identifiers[slot] = c_void_p(identifier)
                self.owned[identifier] = texture
        except Exception as error:                     # noqa: BLE001 - report, not raise
            log.debug('allocating %d %s surfaces failed: %r', count,
                      api.fourcc_name(fourcc), error)
            for texture in made:
                texture.close()
            return -4                                  # MFX_ERR_MEMORY_ALLOC
        response.contents.mids = identifiers
        response.contents.NumFrameActual = count
        # The runtime keeps the array; nothing else here holds a reference.
        self._responses.append(identifiers)
        return api.MFX_ERR_NONE

    def lock(self, _pthis: Any, _mid: Any, _data: Any) -> int:
        """Refuse host access: every surface here lives in video memory."""
        return -3                                      # MFX_ERR_UNSUPPORTED

    def get_handle(self, _pthis: Any, mid: Any, handle: Any) -> int:
        """Give the runtime the Direct3D texture an identifier stands for.

        It expects a pair -- the resource and the index of the subresource
        within it -- and every texture here has exactly one, so the second is
        always zero.
        """
        identifier = int(mid) if mid else 0
        texture = self.owned.get(identifier) or self.external.get(identifier)
        if texture is None:
            log.debug('oneVPL asked for an unknown surface: 0x%x', identifier)
            return api.MFX_ERR_NOT_FOUND
        pair = ctypes.cast(handle, ctypes.POINTER(c_void_p))
        pair[0] = ctypes.cast(texture.pointer, c_void_p)
        pair[1] = None
        return api.MFX_ERR_NONE

    def free(self, _pthis: Any, response: Any) -> int:
        """Give back the surfaces from one allocation."""
        answer = response.contents
        for slot in range(int(answer.NumFrameActual)):
            texture = self.owned.pop(answer.mids[slot] or 0, None)
            if texture is not None:
                texture.close()
        return api.MFX_ERR_NONE

    def close(self) -> None:
        """Release every surface this allocator made."""
        for texture in self.owned.values():
            texture.close()
        self.owned.clear()
        self.external.clear()
        self._responses.clear()


_DXGI_FOR_FOURCC = {
    api.MFX_FOURCC_NV12: d3d11.DXGI_FORMAT_NV12,
    api.MFX_FOURCC_RGB4: d3d11.DXGI_FORMAT_B8G8R8A8_UNORM,
    api.MFX_FOURCC_BGR4: d3d11.DXGI_FORMAT_R8G8B8A8_UNORM,
}


class VPLEncoder(Encoder):
    """H.264 from OpenGL, on Intel hardware, through Direct3D 11.

    width/height -- frame size
    codec -- ``'h264'``
    fps -- frames per second, a number or an exact ``(numerator, denominator)``
        pair. It sets the stream's declared rate and rate control's idea of
        time; it paces nothing.
    bitrate -- target bits per second; by default derived from the frame size
        and rate
    preset -- ``'p1'`` (fastest) through ``'p7'`` (best quality), the same names
        the NVENC backend takes
    rate_control -- ``'vbr'``, ``'cbr'`` or ``'constqp'``
    gop -- frames between key frames; defaults to two seconds' worth
    bframes -- B-pictures between reference pictures
    adapter -- which GPU to encode on; by default the one the current OpenGL
        context is running on, which is the only one it can read without a copy

    An OpenGL context must be current when the encoder is built and for every
    call afterwards: the surfaces belong to it.
    """

    codec = 'h264'
    zero_copy = True
    #: The surfaces must be Direct3D resources, which only this backend can make.
    allocates_inputs = True
    input_slots = 2

    def __init__(self, width: int, height: int, *, codec: str = 'h264',
                 fps: float | tuple[int, int] = 60, bitrate: int | None = None,
                 preset: str = 'p4', rate_control: str = 'vbr',
                 gop: int | None = None, bframes: int = 0,
                 adapter: d3d11.Adapter | None = None):
        if codec != 'h264':
            raise EncoderError(f'the oneVPL backend encodes h264, not {codec!r}')
        self.size = (int(width), int(height))
        self.frame_rate = frame_rate_ratio(fps)
        self.bframes = int(bframes)
        self.reorders_frames = self.bframes > 0
        rate_num, rate_den = self.frame_rate
        self.gop = int(gop) if gop is not None else max(1, round(2 * rate_num / rate_den))
        self.bitrate = int(bitrate) if bitrate is not None else default_bitrate(
            self.size, self.frame_rate)

        self._closed = False
        self._initialised = False
        self._frames = 0
        self._inputs: list[interop.SharedTexture] = []
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
            return adapter
        found = interop.adapter_for_context()
        if found is None:
            raise EncoderError(
                'cannot tell which GPU this OpenGL context is on, so an encoder '
                'here might not be able to read what the renderer draws; name an '
                'adapter explicitly to go ahead anyway')
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
        info.Width, info.Height = _aligned(width), _aligned(height)
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
        mfx.CodecId = api.MFX_CODEC_AVC
        mfx.CodecProfile = api.MFX_PROFILE_AVC_HIGH
        mfx.TargetUsage = target_usage
        mfx.RateControlMethod = mode
        # Rate control counts in kilobits, in a 16-bit field: the multiplier is
        # what lets a recording ask for more than 65 Mbit/s.
        kilobits = max(1, self.bitrate // 1000)
        multiplier = max(1, (kilobits + 65534) // 65535)
        mfx.BRCParamMultiplier = multiplier
        mfx.TargetKbps = kilobits // multiplier
        mfx.MaxKbps = (mfx.TargetKbps if rate_control == 'cbr'
                       else min(65535, mfx.TargetKbps * 3 // 2))
        mfx.GopPicSize = self.gop
        mfx.GopRefDist = self.bframes + 1
        mfx.IdrInterval = 0                    # every group starts with an IDR
        self._frame_info(mfx.FrameInfo)
        self._attach_colour(params)
        return params

    def _attach_colour(self, params: api.mfxVideoParam) -> None:
        """State the stream's colour, rather than leaving a player to guess.

        The hardware converts RGB to YUV with a matrix of its own choosing, so
        the stream says which one it was: limited-range BT.709, primaries,
        transfer and matrix alike. That is what the NVENC backend writes into
        its video usability information, and a recording should not describe
        itself differently depending on whose GPU made it.

        The buffers are kept on the encoder because the runtime reads them
        during initialisation, from the pointers handed over here.
        """
        signal = api.mfxExtVideoSignalInfo()
        signal.Header.BufferId = api.MFX_EXTBUFF_VIDEO_SIGNAL_INFO
        signal.Header.BufferSz = ctypes.sizeof(signal)
        signal.VideoFormat = 5                 # unspecified
        signal.VideoFullRange = 0              # limited range, 16-235
        signal.ColourDescriptionPresent = 1
        signal.ColourPrimaries = 1             # BT.709
        signal.TransferCharacteristics = 1     # BT.709
        signal.MatrixCoefficients = 1          # BT.709
        self._colour = signal
        self._ext_buffers = (c_void_p * 1)(ctypes.cast(byref(signal), c_void_p))
        params.NumExtParam = 1
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
                               f'{width}x{height} h264')
        if status == api.MFX_WRN_PARTIAL_ACCELERATION:
            log.warning('this part encodes H.264 only partly in hardware')
        self._initialised = True

        # What the driver settled on, which is not always what was asked for.
        settled = api.mfxVideoParam()
        self.api.check(
            self.api.MFXVideoENCODE_GetVideoParam(self.session, byref(settled)),
            'MFXVideoENCODE_GetVideoParam')
        self.parameters = settled
        self.reorders_frames = settled.mfx.GopRefDist > 1
        self._bitstream = self._create_bitstream(settled)
        self._surface = api.mfxFrameSurface1()
        self._frame_info(self._surface.Info)

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
        shared = self.interop.create_texture(width, height, SURFACE_FORMAT)
        self.allocator.adopt(shared.resource)
        self._inputs.append(shared)
        return shared

    def register(self, texture: int, target: int | None = None) -> Any:
        """Not available here: the encoder reads Direct3D, not an OpenGL name."""
        raise EncoderError(
            'this encoder reads a Direct3D surface, so it cannot take a texture '
            'made elsewhere; call new_input() for one it can read')

    def encode(self, handle: Any, timestamp: int, duration: int = 0,
               force_idr: bool = False) -> list[Packet]:
        """Submit the picture in `handle`; return whatever came out."""
        self._require_open()
        if not isinstance(handle, interop.SharedTexture):
            raise EncoderError('encode() wants a handle from new_input()')
        if duration <= 0:
            rate_num, rate_den = self.frame_rate
            duration = round(self.timescale * rate_den / rate_num)

        surface = self._surface
        surface.Data.MemId = ctypes.cast(handle.resource.pointer, c_void_p)
        surface.Data.TimeStamp = int(timestamp)
        surface.Data.FrameOrder = self._frames
        self._frames += 1
        control = self._control(force_idr)
        packet = self._submit(control, byref(surface), duration)
        return [] if packet is None else [packet]

    def flush(self) -> list[Packet]:
        """End the stream, returning the pictures the encoder was still holding."""
        self._require_open()
        packets: list[Packet] = []
        while True:
            rate_num, rate_den = self.frame_rate
            packet = self._submit(None, None,
                                  round(self.timescale * rate_den / rate_num))
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
        while True:
            status = self.api.MFXVideoENCODE_EncodeFrameAsync(
                self.session, control, surface, byref(self._bitstream),
                byref(sync_point))
            if status == api.MFX_WRN_DEVICE_BUSY:
                time.sleep(DEVICE_BUSY_WAIT)
                continue
            break
        if status == api.MFX_ERR_MORE_DATA:
            return None
        self.api.check(status, 'MFXVideoENCODE_EncodeFrameAsync')
        if not sync_point:
            return None
        self.api.check(
            self.api.MFXVideoCORE_SyncOperation(
                self.session, sync_point, SYNC_TIMEOUT_MS),
            'MFXVideoCORE_SyncOperation')
        return self._packet(duration)

    def _packet(self, duration: int) -> Packet:
        """Copy the compressed picture out of the shared buffer."""
        bitstream = self._bitstream
        data = ctypes.string_at(
            ctypes.addressof(self._bitstream_buffer) + bitstream.DataOffset,
            bitstream.DataLength)
        keyframe = bool(bitstream.FrameType
                        & (api.MFX_FRAMETYPE_I | api.MFX_FRAMETYPE_IDR))
        timestamp = int(bitstream.TimeStamp)
        if timestamp == api.MFX_TIMESTAMP_UNKNOWN:      # pragma: no cover - driver quirk
            timestamp = 0
        return Packet(data=data, timestamp=timestamp, duration=duration,
                      keyframe=keyframe)

    def headers(self) -> bytes:
        """The sequence and picture parameter sets, as an Annex-B fragment."""
        self._require_open()
        sps = (ctypes.c_uint8 * 1024)()
        pps = (ctypes.c_uint8 * 1024)()
        option = api.mfxExtCodingOptionSPSPPS()
        option.Header.BufferId = api.MFX_EXTBUFF_CODING_OPTION_SPSPPS
        option.Header.BufferSz = ctypes.sizeof(option)
        option.SPSBuffer = ctypes.cast(sps, ctypes.POINTER(ctypes.c_uint8))
        option.PPSBuffer = ctypes.cast(pps, ctypes.POINTER(ctypes.c_uint8))
        option.SPSBufSize = ctypes.sizeof(sps)
        option.PPSBufSize = ctypes.sizeof(pps)

        buffers = (c_void_p * 1)(ctypes.cast(byref(option), c_void_p))
        params = api.mfxVideoParam()
        params.NumExtParam = 1
        params.ExtParam = ctypes.cast(buffers, ctypes.POINTER(c_void_p))
        self.api.check(
            self.api.MFXVideoENCODE_GetVideoParam(self.session, byref(params)),
            'MFXVideoENCODE_GetVideoParam(SPS/PPS)')
        return bytes(sps[:option.SPSBufSize]) + bytes(pps[:option.PPSBufSize])

    # ------------------------------------------------------------ internals

    def _require_open(self) -> None:
        if self._closed:
            raise EncoderError('the encoder is closed')

    def close(self) -> None:
        """Release the session, the surfaces and the device. Safe to call twice."""
        if getattr(self, '_closed', True):
            return
        self._closed = True
        if self._initialised and self.session:
            self.api.MFXVideoENCODE_Close(self.session)
            self._initialised = False
        for shared in self._inputs:
            shared.close()
        self._inputs = []
        if self.session:
            self.api.MFXClose(self.session)
            self.session = c_void_p()
        if self.allocator is not None:
            self.allocator.close()
            self.allocator = None
        if self.interop is not None:
            self.interop.close()
            self.interop = None
        if self.device is not None:
            self.device.close()
            self.device = None

    def __del__(self) -> None:
        # A recording that ended in an exception should still give the driver
        # its session back. A destructor has nowhere to raise to, so whatever
        # went wrong is logged and dropped.
        try:
            self.close()
        except Exception as error:            # noqa: BLE001  # pragma: no cover
            log.debug('closing the encoder from its destructor failed: %r', error)
