"""The QuickSync decoder: compressed samples in, OpenGL textures out, on Intel hardware.

oneVPL's decoder parses the stream itself and does the picture management, so
this module holds no control layer. It decodes into Direct3D 11 surfaces the
runtime asks the allocator for (:mod:`pyopengl_video.vpl.allocator`), then runs
the session's video processor over each picture, which converts it to RGB on the
GPU into a surface that is also an OpenGL texture
(:mod:`pyopengl_video.windows.interop`). The frame handed to the caller is that
texture, so a picture never passes through host memory::

    decode -> NV12 or P010 surface -> video processor -> shared RGBA texture

The video processor is told the stream's matrix and range, which it applies
itself; the result is an ``'rgba'`` frame, and
:class:`~pyopengl_video.convert.NV12Converter` copies it into the player's
texture the right way up. Registering the decoder's own NV12 surface with
OpenGL is not an option: this driver registers it and exposes no planes.

**The decoder and the OpenGL context must be on the same GPU**, or the textures
would be in memory the decoder never wrote. The adapter is taken from the
current context unless one is named.

What the decoder decides without a driver is in :func:`vpp_signal` and
:func:`colour_of_header`, which the suite checks on any machine.
"""
from __future__ import annotations

import ctypes
import logging
import time
from ctypes import byref, c_void_p
from typing import Any

from pyopengl_video import codecs
from pyopengl_video.colour import (
    MATRIX_BT470BG,
    MATRIX_BT601,
    MATRIX_BT709,
    Colour,
)
from pyopengl_video.decoder import DecodedFrame, Decoder, DecoderError, Sample
from pyopengl_video.presentation import Outputs, Presentation
from pyopengl_video.vpl import api
from pyopengl_video.vpl.allocator import SurfaceAllocator, aligned
from pyopengl_video.windows import d3d11, interop

log = logging.getLogger(__name__)

__all__ = ['CODEC_IDS', 'DEFAULT_OUTPUT_SLOTS', 'VPLDecoder', 'VPLDecodeError',
           'colour_of_header', 'vpp_signal']

#: How many decoded frames a caller may hold at once, unless it says otherwise.
#: A player converts each frame as it arrives and holds none for long.
DEFAULT_OUTPUT_SLOTS = 4

#: How long to wait for a picture to come out of the hardware, in milliseconds.
#: Reaching this means the device stopped answering, which is a failure rather
#: than a slow frame.
SYNC_TIMEOUT_MS = 5000

#: How long to wait before offering a busy device the same work again.
DEVICE_BUSY_WAIT = 0.001

#: The most bytes held while waiting for parameter sets that never come.
MAX_PENDING = 16 << 20

#: The runtime's codec identifier for each codec name this backend decodes.
CODEC_IDS = {
    'h264': api.MFX_CODEC_AVC,
    'hevc': api.MFX_CODEC_HEVC,
    'vp9': api.MFX_CODEC_VP9,
    'av1': api.MFX_CODEC_AV1,
}

#: The Direct3D format the shared textures are made in, and what oneVPL calls
#: it: ``B8G8R8A8_UNORM`` holds BGRA bytes, which is ``RGB4``.
SURFACE_FORMAT = d3d11.DXGI_FORMAT_B8G8R8A8_UNORM
SURFACE_FOURCC = api.MFX_FOURCC_RGB4

#: The matrices the video processor converts from, by H.273 code point.
VPP_MATRICES = {
    MATRIX_BT709: api.MFX_TRANSFERMATRIX_BT709,
    MATRIX_BT601: api.MFX_TRANSFERMATRIX_BT601,
    MATRIX_BT470BG: api.MFX_TRANSFERMATRIX_BT601,
}


class VPLDecodeError(api.VPLError, DecoderError):
    """oneVPL refused a call made for a decoder. Carries the status code and its name."""


def colour_of_header(signal: api.mfxExtVideoSignalInfo) -> Colour:
    """The colour description a stream's header stated, as a :class:`Colour`.

    A header that states none gives the unspecified matrix, which
    :meth:`~pyopengl_video.colour.Colour.resolved` settles by picture height.
    """
    if not signal.ColourDescriptionPresent:
        return Colour(full_range=bool(signal.VideoFullRange))
    return Colour(matrix=int(signal.MatrixCoefficients),
                  full_range=bool(signal.VideoFullRange),
                  primaries=int(signal.ColourPrimaries),
                  transfer=int(signal.TransferCharacteristics))


def vpp_signal(colour: Colour, height: int) -> api.mfxExtVPPVideoSignalInfo:
    """What the video processor is told about the pictures it converts.

    The input is the stream's matrix and range, with the matrix chosen by
    `height` where the stream states none; the output is RGB over the full
    range. Raises :class:`~pyopengl_video.decoder.DecoderError` for a matrix
    the video processor has no name for.
    """
    colour = colour.resolved(height)
    try:
        matrix = VPP_MATRICES[colour.matrix]
    except KeyError:
        raise DecoderError(
            f'the Intel video processor converts BT.601 and BT.709 pictures; this '
            f'stream states matrix coefficients {colour.matrix}') from None
    signal = api.mfxExtVPPVideoSignalInfo()
    signal.Header.BufferId = api.MFX_EXTBUFF_VPP_VIDEO_SIGNAL_INFO
    signal.Header.BufferSz = ctypes.sizeof(signal)
    signal.InTransferMatrix = matrix
    signal.InNominalRange = (api.MFX_NOMINALRANGE_0_255 if colour.full_range
                             else api.MFX_NOMINALRANGE_16_235)
    signal.OutTransferMatrix = api.MFX_TRANSFERMATRIX_UNKNOWN
    signal.OutNominalRange = api.MFX_NOMINALRANGE_0_255
    return signal


def _check(status: int, call: str, detail: str = '') -> int:
    """Raise :class:`VPLDecodeError` on a failure status; return the status otherwise."""
    if status < 0:
        raise VPLDecodeError(status, call, detail)
    return status


def _attach(params: api.mfxVideoParam, buffers: list[Any]) -> None:
    """Point `params` at extension `buffers`, which must outlive the calls that read them."""
    pointers = (c_void_p * len(buffers))(
        *(ctypes.cast(byref(buffer), c_void_p) for buffer in buffers))
    params.NumExtParam = len(buffers)
    params.ExtParam = ctypes.cast(pointers, ctypes.POINTER(c_void_p)) if buffers else None


class _Output:
    """A shared texture the video processor writes, with the surface that names it."""

    def __init__(self, shared: interop.SharedTexture, surface: api.mfxFrameSurface1) -> None:
        self.shared = shared
        self.surface = surface


class VPLDecoder(Decoder):
    """H.264, HEVC, VP9 or AV1 decoding on Intel hardware, into OpenGL textures.

    codec -- ``'h264'``, ``'hevc'``, ``'vp9'`` or ``'av1'``, where the part has
        a decoder for it; the runtime refuses one it lacks when the first
        picture arrives
    headers -- the stream's configuration as Annex-B (or the configuration OBUs
        for AV1), when the container keeps it apart from the samples
    size -- accepted for the registry; the stream's own size is used
    output_slots -- how many frames a caller may hold unreleased at once
    adapter -- which GPU to decode on; by default the one the current OpenGL
        context is running on, which is the only one it can write without a copy

    Each sample is one access unit. Frames come out in display order as
    ``'rgba'`` :class:`~pyopengl_video.decoder.DecodedFrame` objects, each a
    texture of its own that OpenGL may read until the frame is released.

    An OpenGL context must be current when the decoder is built and for every
    call afterwards: the textures belong to it.
    """

    zero_copy = True

    def __init__(self, codec: str = 'h264', *, headers: bytes = b'',
                 size: tuple[int, int] | None = None,
                 output_slots: int = DEFAULT_OUTPUT_SLOTS,
                 adapter: d3d11.Adapter | None = None, **unknown: object) -> None:
        if unknown:
            raise DecoderError(
                f'the oneVPL decoder has no setting named {", ".join(sorted(unknown))}; '
                'it takes headers, size, output_slots and adapter')
        codec = codecs.canonical(codec)
        if codec not in CODEC_IDS:
            raise DecoderError(
                f'the oneVPL backend decodes {", ".join(sorted(CODEC_IDS))}, not {codec!r}')
        del size
        self.codec = codec
        self.output_slots = int(output_slots)
        self._headers = bytes(headers)
        self._closed = False
        self._primed = False
        self._pending = b''
        self._presentation = Presentation()
        self._lent: dict[DecodedFrame, _Output] = {}
        self._ready: list[DecodedFrame] = []
        self._outputs: Outputs[_Output] | None = None
        self._surfaces: Any = None
        self._decoding = False
        self._processing = False
        self._buffer = (ctypes.c_uint8 * 0)()
        self._bitstream = api.mfxBitstream()
        self._decode_parameters: api.mfxVideoParam | None = None
        self._frame_rate = (30, 1)
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
            _check(
                self.api.MFXVideoCORE_SetFrameAllocator(
                    self.session, byref(self.allocator.structure)),
                'MFXVideoCORE_SetFrameAllocator')
            if self._headers:
                self._read_header(self._headers, early=True)
        except Exception:
            self.close()
            raise

    def __repr__(self) -> str:
        width, height = self.size
        return f'<{type(self).__name__} {width}x{height} {self.codec} on {self.adapter}>'

    # ---------------------------------------------------------------- setup

    @staticmethod
    def _choose_adapter(adapter: d3d11.Adapter | None) -> d3d11.Adapter:
        """Which GPU to decode on, and refuse to guess if it cannot be known."""
        if adapter is not None:
            return adapter
        found = interop.adapter_for_context()
        if found is None:
            raise DecoderError(
                'cannot tell which GPU this OpenGL context is on, so a decoder '
                'here might write textures the renderer cannot read; name an '
                'adapter explicitly to go ahead anyway')
        return found

    def _open_session(self) -> None:
        """Start a oneVPL session on the chosen adapter, over Direct3D 11."""
        version = api.mfxVersion(0, 1)
        session = c_void_p()
        status = self.api.MFXInit(api.MFX_IMPL_HARDWARE_ANY | api.MFX_IMPL_VIA_D3D11,
                                  byref(version), byref(session))
        if status < 0:
            raise VPLDecodeError(
                status, 'MFXInit',
                'no Intel hardware decoder is reachable through Direct3D 11')
        self.session = session
        assert self.device is not None
        _check(
            self.api.MFXVideoCORE_SetHandle(
                self.session, api.MFX_HANDLE_D3D11_DEVICE, self.device.pointer),
            'MFXVideoCORE_SetHandle')

    def _read_header(self, data: bytes, early: bool = False) -> bool:
        """Read the stream's parameters out of `data`; True once the decoder is built.

        False when `data` holds no complete parameter set yet. Called with the
        headers alone at construction, where a failure is not held against the
        stream: the first sample is read the same way and reports it.
        """
        self._load(data)
        params = api.mfxVideoParam()
        params.mfx.CodecId = CODEC_IDS[self.codec]
        params.IOPattern = api.MFX_IOPATTERN_OUT_VIDEO_MEMORY
        signal = api.mfxExtVideoSignalInfo()
        signal.Header.BufferId = api.MFX_EXTBUFF_VIDEO_SIGNAL_INFO
        signal.Header.BufferSz = ctypes.sizeof(signal)
        _attach(params, [signal])
        status = self.api.MFXVideoDECODE_DecodeHeader(
            self.session, byref(self._bitstream), byref(params))
        if status == api.MFX_ERR_MORE_DATA:
            return False
        if status < 0:
            if early:
                log.debug('the headers alone did not describe the stream: %s',
                          api.STATUS_NAMES.get(status, status))
                return False
            raise VPLDecodeError(status, 'MFXVideoDECODE_DecodeHeader',
                                 f'{self.codec}; the stream is not one this part decodes')
        info = params.mfx.FrameInfo
        self.colour = colour_of_header(signal)
        self.size = (int(info.CropW), int(info.CropH))
        vpp_signal(self.colour, self.size[1])          # refuses a matrix it cannot convert
        if early:
            return False
        self._build(params)
        return True

    def _build(self, params: api.mfxVideoParam) -> None:
        """Start the decoder and the video processor for the stream `params` describes."""
        info = params.mfx.FrameInfo
        if info.ChromaFormat != api.MFX_CHROMAFORMAT_YUV420:
            raise DecoderError(
                f'the oneVPL backend decodes 4:2:0 pictures; this stream has '
                f'chroma format {info.ChromaFormat}')
        if info.FourCC not in (api.MFX_FOURCC_NV12, api.MFX_FOURCC_P010):
            raise DecoderError(
                f'the oneVPL backend decodes NV12 and P010 pictures; this stream '
                f'decodes to {api.fourcc_name(info.FourCC)}')
        params.AsyncDepth = 1
        params.IOPattern = api.MFX_IOPATTERN_OUT_VIDEO_MEMORY
        _attach(params, [])
        assert self.allocator is not None
        status = self.api.MFXVideoDECODE_Init(self.session, byref(params))
        if status < 0:
            width, height = self.size
            raise VPLDecodeError(status, 'MFXVideoDECODE_Init',
                                 f'{width}x{height} {self.codec}')
        self._decoding = True
        self._decode_parameters = params
        # The runtime asked the allocator for the surfaces it decodes into while
        # it initialised, and the application hands it those back as the work
        # surfaces: any others it does not recognise.
        pool = list(self.allocator.latest)
        self._surfaces = (api.mfxFrameSurface1 * len(pool))()
        for slot, identifier in enumerate(pool):
            self._surfaces[slot].Info = info
            self._surfaces[slot].Data.MemId = identifier
        self._start_processor(info)
        self._outputs = Outputs(self.output_slots, self._new_output)

    def _start_processor(self, info: api.mfxFrameInfo) -> None:
        """Start the video processor that converts decoded pictures to RGB."""
        width, height = self.size
        params = api.mfxVideoParam()
        params.AsyncDepth = 1
        params.IOPattern = api.MFX_IOPATTERN_IN_VIDEO_MEMORY | api.MFX_IOPATTERN_OUT_VIDEO_MEMORY
        source = params.vpp.In
        source.FourCC = info.FourCC
        source.ChromaFormat = info.ChromaFormat
        source.BitDepthLuma, source.BitDepthChroma = info.BitDepthLuma, info.BitDepthChroma
        source.Shift = info.Shift
        source.PicStruct = api.MFX_PICSTRUCT_PROGRESSIVE
        source.Width, source.Height = info.Width, info.Height
        source.CropX, source.CropY, source.CropW, source.CropH = (
            info.CropX, info.CropY, info.CropW, info.CropH)
        source.FrameRateExtN, source.FrameRateExtD = (
            info.FrameRateExtN or 30, info.FrameRateExtD or 1)
        self._frame_rate = (source.FrameRateExtN, source.FrameRateExtD)
        self._frame_info(params.vpp.Out)
        self._signal = vpp_signal(self.colour, height)
        _attach(params, [self._signal])
        status = self.api.MFXVideoVPP_Init(self.session, byref(params))
        if status < 0:
            raise VPLDecodeError(status, 'MFXVideoVPP_Init', f'{width}x{height} to RGB')
        self._processing = True

    def _frame_info(self, info: api.mfxFrameInfo) -> None:
        """Describe an RGB picture of the stream's size and rate into `info`."""
        width, height = self.size
        info.FourCC = SURFACE_FOURCC
        info.ChromaFormat = api.MFX_CHROMAFORMAT_YUV444
        info.BitDepthLuma = info.BitDepthChroma = 8
        info.PicStruct = api.MFX_PICSTRUCT_PROGRESSIVE
        info.Width, info.Height = aligned(width), aligned(height)
        info.CropX, info.CropY, info.CropW, info.CropH = 0, 0, width, height
        info.FrameRateExtN, info.FrameRateExtD = self._frame_rate

    def _new_output(self) -> _Output:
        """One more texture for the video processor to write."""
        assert self.interop is not None and self.allocator is not None
        width, height = self.size
        shared = self.interop.create_texture(
            aligned(width), aligned(height), d3d11.DXGI_FORMAT_B8G8R8A8_UNORM,
            access=interop.WGL_ACCESS_READ_ONLY_NV)
        surface = api.mfxFrameSurface1()
        self._frame_info(surface.Info)
        surface.Data.MemId = self.allocator.adopt(shared.resource)
        return _Output(shared, surface)

    # ------------------------------------------------------------- decoding

    def _load(self, data: bytes) -> None:
        """Put `data` in the bitstream buffer, for the runtime to read from."""
        if len(data) > len(self._buffer):
            self._buffer = (ctypes.c_uint8 * max(len(data), 2 * len(self._buffer)))()
        ctypes.memmove(self._buffer, data, len(data))
        stream = self._bitstream
        stream.Data = ctypes.cast(self._buffer, ctypes.POINTER(ctypes.c_uint8))
        stream.MaxLength = len(self._buffer)
        stream.DataOffset = 0
        stream.DataLength = len(data)
        stream.DataFlag = api.MFX_BITSTREAM_COMPLETE_FRAME

    def decode(self, sample: Sample) -> list[DecodedFrame]:
        """Decode one access unit; return the frames now ready, in display order."""
        self._check_open()
        data = bytes(sample.data)
        if not self._primed:
            data = self._headers + data
        self._presentation.fed(sample)
        if not self._decoding:
            self._pending += data
            if len(self._pending) > MAX_PENDING:
                raise DecoderError('the stream has no parameter sets in its first '
                                   f'{MAX_PENDING >> 20} MiB')
            if not self._read_header(self._pending):
                return []
            data, self._pending = self._pending, b''
        self._primed = True
        self._load(data)
        self._bitstream.TimeStamp = sample.timestamp
        return self._run(feeding=True)

    def flush(self) -> list[DecodedFrame]:
        """Return every frame still held back, at the end of a stream."""
        self._check_open()
        if not self._decoding:
            return []
        frames = self._run(feeding=False)
        self.reset()
        return frames

    def reset(self) -> None:
        """Drop every picture held back. Frames already lent stay valid."""
        self._check_open()
        self._pending = b''
        self._presentation.restart()
        self._primed = False
        if self._decoding:
            assert self._decode_parameters is not None
            _check(
                self.api.MFXVideoDECODE_Reset(self.session, byref(self._decode_parameters)),
                'MFXVideoDECODE_Reset')

    def _work_surface(self) -> Any:
        """A decode surface nothing is using."""
        for surface in self._surfaces:
            if not surface.Data.Locked:
                return surface
        raise DecoderError('every decode surface is in use; the stream asks for '
                           'more reference pictures than the decoder allocated')

    def _run(self, feeding: bool) -> list[DecodedFrame]:
        """Decode what is in the bitstream buffer; return the frames that came out."""
        stream = byref(self._bitstream) if feeding else None
        try:
            while True:
                picture = ctypes.POINTER(api.mfxFrameSurface1)()
                sync = c_void_p()
                status = self.api.MFXVideoDECODE_DecodeFrameAsync(
                    self.session, stream, byref(self._work_surface()), byref(picture),
                    byref(sync))
                if status == api.MFX_WRN_DEVICE_BUSY:
                    time.sleep(DEVICE_BUSY_WAIT)
                    continue
                if status == api.MFX_ERR_MORE_DATA:
                    break
                if status == api.MFX_ERR_INCOMPATIBLE_VIDEO_PARAM:
                    raise VPLDecodeError(
                        status, 'MFXVideoDECODE_DecodeFrameAsync',
                        'the stream changed its picture size, which this decoder '
                        'does not follow; open a decoder for the new stream')
                if status == api.MFX_ERR_MORE_SURFACE:
                    continue
                _check(status, 'MFXVideoDECODE_DecodeFrameAsync')
                if sync and picture:
                    self._emit(picture.contents, sync)
        except BaseException:
            for frame in self._ready:
                frame.release()
            self._ready = []
            raise
        ready, self._ready = self._ready, []
        return ready

    def _emit(self, picture: api.mfxFrameSurface1, sync: c_void_p) -> None:
        """Convert a decoded picture to RGB and queue it as a frame."""
        timestamp = int(picture.Data.TimeStamp)
        duration = self._presentation.shown(timestamp)
        _check(self.api.MFXVideoCORE_SyncOperation(self.session, sync, SYNC_TIMEOUT_MS),
                       'MFXVideoCORE_SyncOperation(decode)')
        if duration is None:
            return
        assert self._outputs is not None
        output = self._outputs.take()
        if output is None:
            held = list(self._lent)
            raise DecoderError(
                f'every one of the {self._outputs.limit} output textures is in use '
                f'by frames still held ({held}); release frames before decoding '
                'more, or open the decoder with more output_slots')
        try:
            self._convert(picture, output)
            output.shared.acquire()
        except BaseException:
            self._outputs.give(output)
            raise
        width, height = self.size
        frame = DecodedFrame(
            timestamp=timestamp, duration=duration, planes=(output.shared.texture,),
            layout='rgba', texture_size=output.shared.size, visible=(0, 0, width, height),
            colour=self.colour, on_release=self._released)
        self._lent[frame] = output
        self._ready.append(frame)

    def _convert(self, picture: api.mfxFrameSurface1, output: _Output) -> None:
        """Run the video processor from a decoded picture into `output`'s texture."""
        sync = c_void_p()
        while True:
            status = self.api.MFXVideoVPP_RunFrameVPPAsync(
                self.session, byref(picture), byref(output.surface), None, byref(sync))
            if status == api.MFX_WRN_DEVICE_BUSY:
                time.sleep(DEVICE_BUSY_WAIT)
                continue
            break
        _check(status, 'MFXVideoVPP_RunFrameVPPAsync')
        if sync:
            _check(
                self.api.MFXVideoCORE_SyncOperation(self.session, sync, SYNC_TIMEOUT_MS),
                'MFXVideoCORE_SyncOperation(video processor)')

    # ------------------------------------------------------------ lifetimes

    def _released(self, frame: DecodedFrame) -> None:
        output = self._lent.pop(frame)
        if output.shared.held:
            output.shared.relinquish()
        if self._outputs is not None:
            self._outputs.give(output)

    def _check_open(self) -> None:
        if self._closed:
            raise DecoderError('this decoder has been closed')

    def close(self) -> None:
        """Release the session, the textures and the device, frames lent included."""
        if getattr(self, '_closed', True):
            return
        self._closed = True
        for frame in list(self._lent):
            frame.release()
        if self._processing:
            self.api.MFXVideoVPP_Close(self.session)
            self._processing = False
        if self._decoding:
            self.api.MFXVideoDECODE_Close(self.session)
            self._decoding = False
        if self.session:
            self.api.MFXClose(self.session)
            self.session = c_void_p()
        if self.interop is not None:
            self.interop.close()
            self.interop = None
        if self.allocator is not None:
            self.allocator.close()
            self.allocator = None
        if self.device is not None:
            self.device.close()
            self.device = None
        self._surfaces = None
