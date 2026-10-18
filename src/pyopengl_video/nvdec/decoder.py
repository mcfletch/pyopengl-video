"""The NVDEC decoder: H.264 samples in, NV12 OpenGL textures out.

NVIDIA's parser reads the bitstream and does the picture management, so this
module holds no H.264 control layer. The parser calls back three times over:
with each sequence, where the hardware decoder and the output textures are
made; with each picture to decode, which is handed to the hardware as it
arrives; and with each picture to show, in display order.

A decoded picture is in CUDA device memory, and the parser reuses that
picture as soon as the display callback returns, so the picture is copied
out inside the callback: into a pair of OpenGL textures, luma as ``GL_R8`` and
chroma as ``GL_RG8``, which the conversion pass
(:mod:`pyopengl_video.convert`) turns into RGBA. Two transfers do the copy:

- :class:`Interop` registers the textures with CUDA and copies device memory
  to device memory. It needs the OpenGL context to render on the GPU that
  decodes.
- :class:`Upload` copies the picture to host memory and uploads it, for a
  context on another vendor's GPU.

What the decoder decides without a driver is in :class:`Sequence`,
:class:`Presentation`, :class:`Outputs` and :func:`plane_copies`, which the
suite checks on any machine.
"""
from __future__ import annotations

import ctypes
import dataclasses
import functools
import logging
import threading
from collections.abc import Callable
from ctypes import c_ubyte, c_uint, c_ulonglong, c_void_p
from typing import Any, Protocol

from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_LINEAR,
    GL_PIXEL_UNPACK_BUFFER,
    GL_PIXEL_UNPACK_BUFFER_BINDING,
    GL_R8,
    GL_RED,
    GL_RG,
    GL_RG8,
    GL_TEXTURE_2D,
    GL_TEXTURE_BINDING_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNPACK_ALIGNMENT,
    GL_UNPACK_ROW_LENGTH,
    GL_UNPACK_SKIP_PIXELS,
    GL_UNPACK_SKIP_ROWS,
    GL_UNSIGNED_BYTE,
    glBindBuffer,
    glBindTexture,
    glDeleteTextures,
    glGenTextures,
    glGetIntegerv,
    glPixelStorei,
    glTexParameteri,
    glTexStorage2D,
    glTexSubImage2D,
)

from pyopengl_video.colour import Colour
from pyopengl_video.decoder import DecodedFrame, Decoder, DecoderError, Sample
from pyopengl_video.mp4 import split_annexb
from pyopengl_video.nvdec import api
from pyopengl_video.presentation import Outputs, Presentation
from pyopengl_video.vaapi.h264 import StreamError
from pyopengl_video.vaapi.h264_parse import (
    NAL_SPS,
    SequenceParameterSet,
    colour_of,
    parse_nal,
    parse_sps,
)

log = logging.getLogger(__name__)

__all__ = ['DEFAULT_OUTPUT_SLOTS', 'DEINTERLACE_MODES', 'Interop', 'NVDECDecoder',
           'Outputs', 'Presentation', 'Sequence', 'Upload', 'plane_copies']

#: How many decoded frames a caller may hold at once, unless it says otherwise.
#: A player converts each frame as it arrives and holds none for long.
DEFAULT_OUTPUT_SLOTS = 4

#: How an interlaced picture becomes one frame, by the name a caller gives:
#: ``'weave'`` interleaves the two fields as they were coded, ``'bob'`` scales
#: one field to the frame's height, and ``'adaptive'`` chooses between them
#: for each part of the picture by its motion. A progressive stream is not
#: deinterlaced, whichever is named.
DEINTERLACE_MODES = {
    'weave': api.cudaVideoDeinterlaceMode_Weave,
    'bob': api.cudaVideoDeinterlaceMode_Bob,
    'adaptive': api.cudaVideoDeinterlaceMode_Adaptive,
}

#: Pictures the parser holds past the point their order is known. A picture
#: shown the moment it is submitted has to be waited for; held for two more
#: samples, it has been decoded by the time it is copied out, which is what
#: keeps the copy from stalling the thread that draws.
DISPLAY_DELAY = 2

#: The bit of ``CUVIDDECODECAPS.nOutputFormatMask`` that stands for NV12.
NV12_OUTPUT = 1 << api.cudaVideoSurfaceFormat_NV12


# ------------------------------------------------- decisions with no driver


@dataclasses.dataclass(frozen=True)
class Sequence:
    """What a decoder and its textures are built for: one coded video sequence.

    codec -- the ``cudaVideoCodec`` value
    coded_size -- the size pictures are decoded at, whole macroblocks
    visible -- the picture within it, as ``(x, y, width, height)``
    surfaces -- how many decode surfaces the stream needs
    progressive -- every picture is a frame, none a pair of fields
    chroma_format -- the ``cudaVideoChromaFormat`` value
    bit_depth -- bits a sample, luma then chroma
    colour -- the colour description the parser read; it does not report
        where chroma samples sit
    """

    codec: int
    coded_size: tuple[int, int]
    visible: tuple[int, int, int, int]
    surfaces: int
    progressive: bool
    chroma_format: int
    bit_depth: tuple[int, int]
    colour: Colour

    @classmethod
    def of(cls, format: api.CUVIDEOFORMAT) -> Sequence:
        """The sequence the parser describes in `format`."""
        area, signal = format.display_area, format.video_signal_description
        return cls(
            codec=int(format.codec),
            coded_size=(int(format.coded_width), int(format.coded_height)),
            visible=(int(area.left), int(area.top),
                     int(area.right - area.left), int(area.bottom - area.top)),
            surfaces=int(format.min_num_decode_surfaces),
            progressive=bool(format.progressive_sequence),
            chroma_format=int(format.chroma_format),
            bit_depth=(8 + int(format.bit_depth_luma_minus8),
                       8 + int(format.bit_depth_chroma_minus8)),
            colour=Colour(matrix=int(signal.matrix_coefficients),
                          full_range=bool(signal.video_full_range_flag),
                          primaries=int(signal.color_primaries),
                          transfer=int(signal.transfer_characteristics)))

    @property
    def size(self) -> tuple[int, int]:
        """The visible picture's size, as (width, height)."""
        return self.visible[2], self.visible[3]

    def refusal(self, capabilities: api.CUVIDDECODECAPS, device: str) -> str:
        """Why `device` cannot decode this sequence into NV12, or '' where it can.

        capabilities -- what ``cuvidGetDecoderCaps`` answered for this
            sequence's codec, chroma format and bit depth
        """
        width, height = self.coded_size
        chroma = api.CHROMA_FORMAT_NAMES.get(self.chroma_format,
                                             f'chroma format {self.chroma_format}')
        if self.chroma_format != api.cudaVideoChromaFormat_420 or self.bit_depth != (8, 8):
            return (f'the stream is {chroma} at {max(self.bit_depth)} bits, and this '
                    'decoder takes 8-bit 4:2:0')
        if not capabilities.bIsSupported:
            return f'{device} has no hardware decoder for this stream ({chroma}, 8 bits)'
        if not capabilities.nOutputFormatMask & NV12_OUTPUT:
            return f'{device} does not decode this stream to NV12'
        most = (int(capabilities.nMaxWidth), int(capabilities.nMaxHeight))
        least = (int(capabilities.nMinWidth), int(capabilities.nMinHeight))
        if width > most[0] or height > most[1]:
            return (f'a {width}x{height} picture is larger than the '
                    f'{most[0]}x{most[1]} {device} decodes')
        if width < least[0] or height < least[1]:
            return (f'a {width}x{height} picture is smaller than the '
                    f'{least[0]}x{least[1]} {device} decodes')
        macroblocks = (width // 16) * (height // 16)
        if macroblocks > capabilities.nMaxMBCount:
            return (f'a {width}x{height} picture is {macroblocks} macroblocks, and '
                    f'{device} decodes up to {int(capabilities.nMaxMBCount)}')
        return ''


def plane_copies(pointer: int, pitch: int,
                 coded_size: tuple[int, int]) -> tuple[api.CUDA_MEMCPY2D, api.CUDA_MEMCPY2D]:
    """The source halves of the two copies that take a mapped NV12 picture out.

    pointer/pitch -- where ``cuvidMapVideoFrame`` put the picture in device
        memory, and the bytes from one row to the next
    coded_size -- the size the decoder was created with

    Luma is `height` rows of `width` bytes. Chroma follows it `height` rows
    down: half as many rows, each `width` bytes of interleaved Cb and Cr.
    The destination of each copy is the transfer's to fill in.
    """
    width, height = coded_size
    planes = []
    for offset, rows in ((0, height), (pitch * height, height // 2)):
        copy = api.CUDA_MEMCPY2D()
        copy.srcMemoryType = api.CU_MEMORYTYPE_DEVICE
        copy.srcDevice = pointer + offset
        copy.srcPitch = pitch
        copy.WidthInBytes = width
        copy.Height = rows
        planes.append(copy)
    return planes[0], planes[1]


# ------------------------------------------------------------ the transfers


@dataclasses.dataclass(eq=False)
class TexturePair:
    """One picture's worth of output: a luma and a chroma texture.

    resources -- the two textures as CUDA registered them, for :class:`Interop`
    """

    luma: int
    chroma: int
    resources: Any = None

    @property
    def planes(self) -> tuple[int, int]:
        return self.luma, self.chroma


class Transfer(Protocol):
    """How a mapped picture gets from CUDA device memory into a texture pair."""

    zero_copy: bool

    def adopt(self, pair: TexturePair) -> None: ...

    def copy(self, pair: TexturePair, pointer: int, pitch: int) -> None: ...

    def forget(self, pair: TexturePair) -> None: ...


def _plane_texture(internal_format: int, width: int, height: int) -> int:
    """An immutable plane texture, leaving the 2D binding as it was."""
    texture = int(glGenTextures(1))
    previous = int(glGetIntegerv(GL_TEXTURE_BINDING_2D))
    glBindTexture(GL_TEXTURE_2D, texture)
    try:
        glTexStorage2D(GL_TEXTURE_2D, 1, internal_format, width, height)
        for parameter, value in (
            (GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
            (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
        ):
            glTexParameteri(GL_TEXTURE_2D, parameter, value)
    finally:
        glBindTexture(GL_TEXTURE_2D, previous)
    return texture


class Interop:
    """Copies a picture into textures CUDA has registered, device to device.

    The textures and the decoder have to be on one GPU, which is what
    ``cuGLGetDevices`` answers for the current OpenGL context.
    """

    zero_copy = True

    def __init__(self, cuda: api.Cuda, coded_size: tuple[int, int]) -> None:
        self.cuda = cuda
        self.coded_size = coded_size

    def adopt(self, pair: TexturePair) -> None:
        """Register both textures with CUDA, as images it only ever overwrites."""
        cuda = self.cuda
        # Held by the pair from the start, so one that registers and one that
        # does not are both given back by forget().
        resources = pair.resources = (c_void_p * 2)()
        for index, texture in enumerate(pair.planes):
            resource = c_void_p()
            cuda.check(
                cuda.cuGraphicsGLRegisterImage(
                    ctypes.byref(resource), texture, GL_TEXTURE_2D,
                    api.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD),
                'cuGraphicsGLRegisterImage', f'texture {texture}')
            resources[index] = resource.value

    def copy(self, pair: TexturePair, pointer: int, pitch: int) -> None:
        """Copy the mapped picture at `pointer` into `pair`'s textures."""
        cuda = self.cuda
        cuda.check(cuda.cuGraphicsMapResources(2, pair.resources, None),
                   'cuGraphicsMapResources')
        try:
            for index, plane in enumerate(plane_copies(pointer, pitch, self.coded_size)):
                # The array a resource maps to may differ from one mapping to the next.
                array = c_void_p()
                cuda.check(cuda.cuGraphicsSubResourceGetMappedArray(
                    ctypes.byref(array), pair.resources[index], 0, 0),
                    'cuGraphicsSubResourceGetMappedArray')
                plane.dstMemoryType = api.CU_MEMORYTYPE_ARRAY
                plane.dstArray = array
                cuda.check(cuda.cuMemcpy2D(ctypes.byref(plane)), 'cuMemcpy2D',
                           'device memory to a texture')
        finally:
            cuda.cuGraphicsUnmapResources(2, pair.resources, None)

    def forget(self, pair: TexturePair) -> None:
        """Unregister the textures, before they are deleted."""
        resources, pair.resources = pair.resources, None
        for resource in resources or ():
            if resource:
                self.cuda.cuGraphicsUnregisterResource(resource)


class Upload:
    """Copies a picture to host memory and uploads it to the textures.

    For an OpenGL context that renders on another GPU than the one decoding:
    each picture crosses the bus twice, about 3 MB at 1080p.
    """

    zero_copy = False

    def __init__(self, cuda: api.Cuda, coded_size: tuple[int, int]) -> None:
        self.cuda = cuda
        self.coded_size = coded_size
        width, height = coded_size
        self._luma = (c_ubyte * (width * height))()
        self._chroma = (c_ubyte * (width * (height // 2)))()

    def adopt(self, pair: TexturePair) -> None:
        """Nothing to register: the textures are OpenGL's alone."""

    def copy(self, pair: TexturePair, pointer: int, pitch: int) -> None:
        """Copy the mapped picture at `pointer` out, then into `pair`'s textures."""
        cuda = self.cuda
        width, height = self.coded_size
        buffers = (self._luma, self._chroma)
        for plane, buffer in zip(plane_copies(pointer, pitch, self.coded_size), buffers,
                                 strict=True):
            plane.dstMemoryType = api.CU_MEMORYTYPE_HOST
            plane.dstHost = ctypes.addressof(buffer)
            plane.dstPitch = width
            cuda.check(cuda.cuMemcpy2D(ctypes.byref(plane)), 'cuMemcpy2D',
                       'device memory to host memory')
        uploads = ((pair.luma, GL_RED, (width, height), self._luma),
                   (pair.chroma, GL_RG, (width // 2, height // 2), self._chroma))
        # A renderer may have left any of these set, and each changes what
        # glTexSubImage2D reads from the buffer.
        packing = {name: int(glGetIntegerv(name))
                   for name in (GL_UNPACK_ALIGNMENT, GL_UNPACK_ROW_LENGTH,
                                GL_UNPACK_SKIP_ROWS, GL_UNPACK_SKIP_PIXELS)}
        unpack_buffer = int(glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING))
        texture = int(glGetIntegerv(GL_TEXTURE_BINDING_2D))
        try:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            for name in packing:
                glPixelStorei(name, 1 if name == GL_UNPACK_ALIGNMENT else 0)
            for name, format, (plane_width, plane_height), buffer in uploads:
                glBindTexture(GL_TEXTURE_2D, name)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, plane_width, plane_height,
                                format, GL_UNSIGNED_BYTE, buffer)
        finally:
            glBindTexture(GL_TEXTURE_2D, texture)
            for name, value in packing.items():
                glPixelStorei(name, value)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, unpack_buffer)

    def forget(self, pair: TexturePair) -> None:
        """Nothing was registered."""


# -------------------------------------------------------------- one sequence


class _Session:
    """The hardware decoder and the output textures of one sequence.

    The decoder is made when the parser reports the sequence. Texture pairs
    are made as they are first needed (:class:`Outputs`), and each is kept for
    the session's life.
    """

    def __init__(self, cuda: api.Cuda, cuvid: api.Cuvid, context: c_void_p,
                 sequence: Sequence, transfer: Transfer, slots: int,
                 deinterlace: int) -> None:
        self.cuda = cuda
        self.cuvid = cuvid
        self.context = context
        self.sequence = sequence
        self.transfer = transfer
        self.decoder = c_void_p()
        # Every picture the parser holds may be shown by one flush, on top of
        # the frames the caller already keeps.
        self.outputs: Outputs[TexturePair] = Outputs(sequence.surfaces + slots, self._pair)
        width, height = sequence.coded_size
        information = api.CUVIDDECODECREATEINFO(
            ulWidth=width, ulHeight=height, ulMaxWidth=width, ulMaxHeight=height,
            ulTargetWidth=width, ulTargetHeight=height,
            ulNumDecodeSurfaces=sequence.surfaces, ulNumOutputSurfaces=1,
            CodecType=sequence.codec, ChromaFormat=sequence.chroma_format,
            bitDepthMinus8=sequence.bit_depth[0] - 8,
            ulCreationFlags=api.cudaVideoCreate_PreferCUVID,
            OutputFormat=api.cudaVideoSurfaceFormat_NV12,
            DeinterlaceMode=(api.cudaVideoDeinterlaceMode_Weave if sequence.progressive
                             else deinterlace))
        # The whole coded picture, unscaled: the visible rectangle is the
        # conversion pass's to apply, as it is for every backend.
        information.display_area.right = width
        information.display_area.bottom = height
        cuda.check(cuvid.cuvidCreateDecoder(ctypes.byref(self.decoder),
                                            ctypes.byref(information)),
                   'cuvidCreateDecoder',
                   f'{width}x{height}, {sequence.surfaces} surfaces')

    def _pair(self) -> TexturePair:
        width, height = self.sequence.coded_size
        pair = TexturePair(_plane_texture(GL_R8, width, height),
                           _plane_texture(GL_RG8, width // 2, height // 2))
        try:
            self.transfer.adopt(pair)
        except BaseException:
            self.transfer.forget(pair)
            glDeleteTextures(list(pair.planes))
            raise
        return pair

    def decode(self, parameters: int) -> None:
        """Hand one picture's parameters, as the parser built them, to the hardware."""
        self.cuda.check(self.cuvid.cuvidDecodePicture(self.decoder, parameters),
                        'cuvidDecodePicture')

    def copy_out(self, picture: api.CUVIDPARSERDISPINFO, pair: TexturePair) -> None:
        """Copy a displayed picture into `pair`, waiting for its decode to finish."""
        cuda, cuvid = self.cuda, self.cuvid
        processing = api.CUVIDPROCPARAMS(
            progressive_frame=picture.progressive_frame,
            top_field_first=picture.top_field_first,
            unpaired_field=int(picture.repeat_first_field < 0))
        pointer, pitch = c_ulonglong(0), c_uint(0)
        cuda.check(
            cuvid.cuvidMapVideoFrame(self.decoder, picture.picture_index,
                                     ctypes.byref(pointer), ctypes.byref(pitch),
                                     ctypes.byref(processing)),
            'cuvidMapVideoFrame', f'picture {picture.picture_index}')
        try:
            self.transfer.copy(pair, pointer.value, pitch.value)
        finally:
            cuvid.cuvidUnmapVideoFrame(self.decoder, pointer)

    def stop(self) -> None:
        """Destroy the hardware decoder. The textures already filled stay valid.

        The driver's call waits for the video engine: 36 ms where another
        decoder is still open and 240 ms where this is the last one, measured
        on an RTX 3060 Ti. It needs the CUDA context and no OpenGL, so it runs
        on a thread of its own and the thread that draws does not wait for it.
        The interpreter joins the thread before it exits.
        """
        decoder, self.decoder = self.decoder, c_void_p()
        if decoder:
            threading.Thread(target=self._destroy, args=(decoder,),
                             name='nvdec-destroy').start()

    def _destroy(self, decoder: c_void_p) -> None:
        with self.cuda.current(self.context):
            self.cuvid.cuvidDestroyDecoder(decoder)

    def close(self) -> None:
        """Destroy the decoder and give back every texture pair."""
        self.stop()
        with self.cuda.current(self.context):
            for pair in self.outputs.made:
                self.transfer.forget(pair)
        textures = [texture for pair in self.outputs.made for texture in pair.planes]
        if textures:
            glDeleteTextures(textures)
        self.outputs.made.clear()


# ------------------------------------------------------------------ decoder


def _callback(method: Callable[..., int]) -> Callable[..., int]:
    """Make a method safe to call from the parser.

    An exception cannot cross the C frames between the parser and
    :meth:`NVDECDecoder._parse`, so it is kept, the parse is failed by
    returning 0, and ``_parse`` raises it once the parser has returned.
    """
    @functools.wraps(method)
    def trapping(self: NVDECDecoder, *arguments: Any) -> int:
        try:
            return method(self, *arguments)
        except BaseException as error:  # noqa: BLE001 - re-raised by _parse, outside the C frames
            self._failure = self._failure or error
            return 0
    return trapping


class NVDECDecoder(Decoder):
    """H.264 decoding on NVIDIA hardware, into OpenGL textures.

    codec -- ``'h264'``
    headers -- the stream's SPS and PPS as Annex-B, when the container keeps
        them apart from the samples
    size -- accepted for the registry; the stream's own size is used
    output_slots -- how many frames a caller may hold unreleased at once
    deinterlace -- how an interlaced stream's fields become frames:
        ``'adaptive'``, ``'bob'`` or ``'weave'`` (:data:`DEINTERLACE_MODES`)
    interop -- True to copy each picture into the textures on the GPU, which
        needs the current OpenGL context to render on the NVIDIA GPU; False
        to decode on the first NVIDIA GPU and copy through host memory

    Each sample is one access unit: a picture and the units before it.
    Frames come out in display order as ``'nv12'``
    :class:`~pyopengl_video.decoder.DecodedFrame` objects, each holding its
    own copy of the picture.
    """

    def __init__(self, codec: str = 'h264', *, headers: bytes = b'',
                 size: tuple[int, int] | None = None,
                 output_slots: int = DEFAULT_OUTPUT_SLOTS,
                 deinterlace: str = 'adaptive', interop: bool = True,
                 **unknown: object) -> None:
        if unknown:
            raise DecoderError(
                f'the NVDEC decoder has no setting named {", ".join(sorted(unknown))}; '
                'it takes headers, size, output_slots, deinterlace and interop')
        if codec not in api.CODECS:
            raise DecoderError(f'the NVDEC backend decodes h264, not {codec!r}')
        if deinterlace not in DEINTERLACE_MODES:
            raise DecoderError(
                f'deinterlace is one of {", ".join(sorted(DEINTERLACE_MODES))}, '
                f'not {deinterlace!r}')
        del size
        self.codec = codec
        self.output_slots = int(output_slots)
        self.zero_copy = bool(interop)
        self._deinterlace = DEINTERLACE_MODES[deinterlace]
        self._headers = bytes(headers)
        self._closed = False
        self._sps: SequenceParameterSet | None = None
        self._presentation = Presentation()
        self._session: _Session | None = None
        self._retired: list[_Session] = []
        self._lent: dict[DecodedFrame, tuple[_Session, TexturePair]] = {}
        self._ready: list[DecodedFrame] = []
        self._failure: BaseException | None = None
        self._parser = c_void_p()
        self._primed = False
        # The parser keeps the addresses of these, so they live as long as it does.
        self._callbacks = (api.SEQUENCE_CALLBACK(self._on_sequence),
                           api.DECODE_CALLBACK(self._on_decode),
                           api.DISPLAY_CALLBACK(self._on_display))
        try:
            self._cuda = api.Cuda.load()
            self._cuvid = api.Cuvid.load()
        except OSError as error:
            raise DecoderError(f"NVIDIA's decoder library is not loadable: {error}") from error
        self._device = self._choose_device(interop)
        self.device = self._cuda.device_name(self._device)
        self._context = self._cuda.context(self._device)
        self._note_parameter_sets(self._headers)

    def __repr__(self) -> str:
        width, height = self.size
        return f'<{type(self).__name__} {width}x{height} {self.codec} on {self.device}>'

    def _choose_device(self, interop: bool) -> int:
        cuda = self._cuda
        if not interop:
            devices = cuda.devices()
            if not devices:
                raise DecoderError('CUDA reports no NVIDIA GPU in this machine')
            return devices[0]
        devices = cuda.gl_devices()
        if not devices:
            raise DecoderError(
                'the current OpenGL context does not render on an NVIDIA GPU, so '
                'CUDA cannot write its textures; open the decoder with '
                'interop=False to copy each picture through host memory')
        return devices[0]

    def _note_parameter_sets(self, data: bytes) -> None:
        """Keep the last sequence parameter set in `data`, for what the parser omits.

        The parser reports a sequence at its first picture, and never where
        chroma samples sit. This package's own reading of the set gives the
        size before the first sample and the whole colour description. Sets
        are looked for in the headers, in key frames and in the first sample
        decoded, which is where a stream carries them.
        """
        for unit in split_annexb(data):
            if unit[0] & 0x1F != NAL_SPS:
                continue
            try:
                self._sps = parse_sps(parse_nal(unit).rbsp)
            except StreamError as error:
                # The hardware may decode a set this package does not read.
                log.debug('a sequence parameter set was not read here: %s', error)
                self._sps = None
                continue
            if self._session is None:
                self.size = self._sps.size
                self.colour = colour_of(self._sps)

    # ------------------------------------------------------------ decoding

    def decode(self, sample: Sample) -> list[DecodedFrame]:
        """Decode one access unit; return the frames now ready, in display order."""
        self._check_open()
        data = bytes(sample.data)
        if sample.keyframe or not self._primed:
            self._note_parameter_sets(data)
        with self._cuda.current(self._context):
            if not self._primed:
                self._primed = True
                if self._headers:
                    self._parse(self._headers, 0)
            self._presentation.fed(sample)
            return self._parse(data, api.CUVID_PKT_TIMESTAMP | api.CUVID_PKT_ENDOFPICTURE,
                               sample.timestamp)

    def flush(self) -> list[DecodedFrame]:
        """Return every frame still held back, at the end of a stream."""
        self._check_open()
        with self._cuda.current(self._context):
            try:
                return self._parse(b'', api.CUVID_PKT_ENDOFSTREAM)
            finally:
                self._restart()

    def reset(self) -> None:
        """Drop every picture held back. Frames already lent stay valid."""
        self._check_open()
        with self._cuda.current(self._context):
            self._restart()

    def _restart(self) -> None:
        """Destroy the parser, so the next sample starts a new one with no references.

        The hardware decoder and the textures are kept: a new parser reports
        the sequence again, and one that matches needs nothing rebuilt.
        """
        if self._parser:
            self._cuvid.cuvidDestroyVideoParser(self._parser)
            self._parser = c_void_p()
        self._presentation.restart()
        self._primed = False

    def _parse(self, data: bytes, flags: int, timestamp: int = 0) -> list[DecodedFrame]:
        """Hand `data` to the parser; return the frames its callbacks produced."""
        cuda, cuvid = self._cuda, self._cuvid
        if not self._parser:
            parameters = api.CUVIDPARSERPARAMS(
                CodecType=api.CODECS[self.codec],
                # The sequence callback answers with the count the stream needs.
                ulMaxNumDecodeSurfaces=1,
                ulMaxDisplayDelay=DISPLAY_DELAY)
            (parameters.pfnSequenceCallback, parameters.pfnDecodePicture,
             parameters.pfnDisplayPicture) = self._callbacks
            cuda.check(cuvid.cuvidCreateVideoParser(ctypes.byref(self._parser),
                                                    ctypes.byref(parameters)),
                       'cuvidCreateVideoParser')
        packet = api.CUVIDSOURCEDATAPACKET(
            flags=flags, payload_size=len(data), timestamp=timestamp,
            payload=ctypes.cast(ctypes.c_char_p(data), c_void_p) if data else None)
        status = cuvid.cuvidParseVideoData(self._parser, ctypes.byref(packet))
        ready, self._ready = self._ready, []
        failure, self._failure = self._failure, None
        try:
            if failure is not None:
                raise failure
            cuda.check(status, 'cuvidParseVideoData', f'{len(data)} bytes')
        except BaseException:
            for frame in ready:
                frame.release()
            raise
        return ready

    # ------------------------------------------------- the parser's callbacks

    @_callback
    def _on_sequence(self, _user: Any, format: Any) -> int:
        """A sequence begins: make its decoder, unless the one in use suits it."""
        sequence = Sequence.of(format.contents)
        session = self._session
        if session is None or session.sequence != sequence:
            self._open(sequence)
        self._describe(sequence)
        # Above 1, the count the parser cycles its pictures through.
        return max(sequence.surfaces, 1)

    def _open(self, sequence: Sequence) -> None:
        cuda, cuvid = self._cuda, self._cuvid
        capabilities = api.CUVIDDECODECAPS(
            eCodecType=sequence.codec, eChromaFormat=sequence.chroma_format,
            nBitDepthMinus8=sequence.bit_depth[0] - 8)
        cuda.check(cuvid.cuvidGetDecoderCaps(ctypes.byref(capabilities)),
                   'cuvidGetDecoderCaps')
        refusal = sequence.refusal(capabilities, self.device)
        if refusal:
            raise DecoderError(refusal)
        previous, self._session = self._session, None
        if previous is not None:
            self._retire(previous)
        log.debug('%s: decoding %dx%d with %d surfaces', self.device,
                  *sequence.coded_size, sequence.surfaces)
        transfer_type = Interop if self.zero_copy else Upload
        self._session = _Session(cuda, cuvid, self._context, sequence,
                                 transfer_type(cuda, sequence.coded_size),
                                 self.output_slots, self._deinterlace)

    def _describe(self, sequence: Sequence) -> None:
        self.size = sequence.size
        described = colour_of(self._sps) if self._sps is not None else sequence.colour
        self.colour = described.resolved(sequence.size[1])

    @_callback
    def _on_decode(self, _user: Any, parameters: int) -> int:
        """A picture is ready to decode: the hardware takes it as the parser built it."""
        session = self._session
        if session is None:
            raise DecoderError('the parser asked for a picture before it described a sequence')
        session.decode(parameters)
        return 1

    @_callback
    def _on_display(self, _user: Any, information: Any) -> int:
        """A picture is due to be shown: copy it out before the parser reuses it."""
        picture = information.contents
        duration = self._presentation.shown(picture.timestamp)
        session = self._session
        if duration is None or session is None:
            return 1
        pair = session.outputs.take()
        if pair is None:
            held = [frame for frame, (owner, _) in self._lent.items() if owner is session]
            raise DecoderError(
                f'every one of the {session.outputs.limit} output textures is in use '
                f'by frames still held ({held}); release frames before decoding '
                'more, or open the decoder with more output_slots')
        try:
            session.copy_out(picture, pair)
        except BaseException:
            session.outputs.give(pair)
            raise
        sequence = session.sequence
        frame = DecodedFrame(
            timestamp=picture.timestamp, duration=duration, planes=pair.planes,
            layout='nv12', texture_size=sequence.coded_size, visible=sequence.visible,
            colour=self.colour, on_release=self._released)
        self._lent[frame] = (session, pair)
        self._ready.append(frame)
        return 1

    # ------------------------------------------------------------ lifetimes

    def _released(self, frame: DecodedFrame) -> None:
        owner, pair = self._lent.pop(frame)
        owner.outputs.give(pair)
        if owner in self._retired and not self._lends_from(owner):
            self._retired.remove(owner)
            owner.close()

    def _lends_from(self, session: _Session) -> bool:
        return any(owner is session for owner, _ in self._lent.values())

    def _retire(self, session: _Session) -> None:
        """Stop a session's decoder; keep its textures while frames of them are held."""
        if self._lends_from(session):
            session.stop()
            self._retired.append(session)
        else:
            session.close()

    def close(self) -> None:
        """Release the decoder, the parser and the textures, frames lent included."""
        if self._closed:
            return
        self._closed = True
        for frame in list(self._lent):
            frame.release()
        with self._cuda.current(self._context):
            self._restart()
        for session in [*self._retired, *([self._session] if self._session else [])]:
            session.close()
        self._retired = []
        self._session = None
        self._callbacks = ()  # type: ignore[assignment]  # dropped to break the cycle through the bound methods

    def _check_open(self) -> None:
        if self._closed:
            raise DecoderError('this decoder has been closed')
