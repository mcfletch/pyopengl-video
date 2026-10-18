"""The NVDEC decoder: compressed samples in, OpenGL textures out.

NVIDIA's parser reads the bitstream and does the picture management for
H.264, HEVC, VP9 and AV1 alike, so this module holds no control layer for any
of them. The parser calls back three times over:
with each sequence, where the hardware decoder and the output textures are
made; with each picture to decode, which is handed to the hardware as it
arrives; and with each picture to show, in display order.

A decoded picture is in CUDA device memory, and the parser reuses that
picture as soon as the display callback returns, so the picture is copied
out inside the callback: into a pair of OpenGL textures, luma and interleaved
chroma, which the conversion pass (:mod:`pyopengl_video.convert`) turns into
RGBA. An 8-bit stream decodes to NV12 (``GL_R8`` and ``GL_RG8`` textures) and
a 10-bit one to P016 (``GL_R16`` and ``GL_RG16``, each sample in the high bits
of its channel), as :data:`SURFACES` lists. Two transfers do the copy:

- :class:`Interop` registers the textures with CUDA and copies device memory
  to device memory. It needs the OpenGL context to render on the GPU that
  decodes.
- :class:`Upload` copies the picture to host memory and uploads it, for a
  context on another vendor's GPU.

The parser is not trusted with what it builds. Each picture it asks to have
decoded is read first (:mod:`pyopengl_video.nvdec.guard`), and one that
states more bitstream than the parser was fed, or predicts from a picture the
hardware never decoded, is refused. An AV1 sample is read by this package's
own parser before NVIDIA's sees it (:mod:`pyopengl_video.nvdec.av1`).

What the decoder decides without a driver is in :class:`Sequence`,
:class:`Presentation`, :class:`Outputs` and :func:`plane_copies`, which the
suite checks on any machine. :mod:`pyopengl_video.nvdec.description` reads the
picture size and the colour description the parser does not report.
"""
from __future__ import annotations

import ctypes
import dataclasses
import functools
import logging
import threading
from collections.abc import Callable
from ctypes import c_ubyte, c_uint, c_ulonglong, c_ushort, c_void_p
from typing import Any, Protocol

from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_LINEAR,
    GL_PIXEL_UNPACK_BUFFER,
    GL_PIXEL_UNPACK_BUFFER_BINDING,
    GL_R8,
    GL_R16,
    GL_RED,
    GL_RG,
    GL_RG8,
    GL_RG16,
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
    GL_UNPACK_SWAP_BYTES,
    GL_UNSIGNED_BYTE,
    GL_UNSIGNED_SHORT,
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

from pyopengl_video import bounds, codecs
from pyopengl_video.colour import Colour
from pyopengl_video.decoder import DecodedFrame, Decoder, DecoderError, Sample
from pyopengl_video.glcontext import Owner
from pyopengl_video.nvdec import api, guard
from pyopengl_video.nvdec.av1 import Gate
from pyopengl_video.nvdec.description import Description, describe
from pyopengl_video.nvdec.vp9 import Frame, Slots
from pyopengl_video.presentation import Outputs, Presentation
from pyopengl_video.teardown import release_all

log = logging.getLogger(__name__)

__all__ = ['DEFAULT_OUTPUT_SLOTS', 'DEINTERLACE_MODES', 'Interop', 'NVDECDecoder',
           'Outputs', 'Presentation', 'SURFACES', 'Sequence', 'SurfaceFormat',
           'Upload', 'plane_copies']

#: How many decoded frames a caller may hold at once, unless it says otherwise.
#: A player converts each frame as it arrives and holds none for long.
DEFAULT_OUTPUT_SLOTS = 4

#: The most frames a caller may ask to hold at once: each is a pair of
#: textures the size of the picture.
MOST_OUTPUT_SLOTS = 64

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

#: The codecs whose pictures are shown as their samples are parsed, with no
#: delay. A VP9 frame shown again is shown by this module and not by the
#: parser (:mod:`pyopengl_video.nvdec.vp9`), and the two keep their order
#: only when the parser holds nothing back.
SHOWN_AS_PARSED = frozenset({codecs.VP9})


@dataclasses.dataclass(frozen=True)
class SurfaceFormat:
    """How pictures of one sample depth are decoded and held in textures.

    depth -- bits a sample
    surface -- the ``cudaVideoSurfaceFormat`` decoded into
    name -- what the headers call that format, for a message
    luma/chroma -- the internal formats of the two textures
    kind -- the type of one channel, as OpenGL names it
    sample_bytes -- bytes a sample, in device memory and in the textures
    layout -- what a :class:`~pyopengl_video.decoder.DecodedFrame` calls it
    """

    depth: int
    surface: int
    name: str
    luma: int
    chroma: int
    kind: int
    sample_bytes: int
    layout: str

    @property
    def output_bit(self) -> int:
        """The bit of ``CUVIDDECODECAPS.nOutputFormatMask`` that stands for it."""
        return 1 << self.surface

    @property
    def host_type(self) -> Any:
        """The ctypes type of one sample in host memory."""
        return c_ubyte if self.sample_bytes == 1 else c_ushort

    def plane_sizes(self, size: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        """The luma and the chroma plane of a surface of `size`, each as (width, height)."""
        width, height = size
        return (width, height), (width // 2, height // 2)


#: The surface formats decoded into, by bits a sample. P016 holds a sample in
#: the high bits of 16, which for 10 bits is the layout called P010.
SURFACES = {
    8: SurfaceFormat(8, api.cudaVideoSurfaceFormat_NV12, 'NV12', GL_R8, GL_RG8,
                     GL_UNSIGNED_BYTE, 1, 'nv12'),
    10: SurfaceFormat(10, api.cudaVideoSurfaceFormat_P016, 'P016', GL_R16, GL_RG16,
                      GL_UNSIGNED_SHORT, 2, 'p010'),
}

#: The chroma formats a two-plane surface holds. A stream with no chroma
#: decodes into one too, its chroma plane at the mid value.
TWO_PLANE_CHROMA = (api.cudaVideoChromaFormat_420, api.cudaVideoChromaFormat_Monochrome)

#: What one sample is handed to the parser as: timed, and a whole picture.
PICTURE = api.CUVID_PKT_TIMESTAMP | api.CUVID_PKT_ENDOFPICTURE


# ------------------------------------------------- decisions with no driver


@dataclasses.dataclass(frozen=True)
class Sequence:
    """What a decoder and its textures are built for: one coded video sequence.

    codec -- the ``cudaVideoCodec`` value
    coded_size -- the size pictures are decoded at: whole blocks in H.264 and
        HEVC, the frame's own size in VP9 and AV1
    visible -- the picture within it, as ``(x, y, width, height)``
    surfaces -- how many decode surfaces the stream needs
    progressive -- every picture is a frame, none a pair of fields
    chroma_format -- the ``cudaVideoChromaFormat`` value
    bit_depth -- bits a sample, luma then chroma
    colour -- the colour description the parser read, with nothing stated
        for a codec in :attr:`UNDESCRIBED`; it does not report where chroma
        samples sit
    """

    codec: int
    coded_size: tuple[int, int]
    visible: tuple[int, int, int, int]
    surfaces: int
    progressive: bool
    chroma_format: int
    bit_depth: tuple[int, int]
    colour: Colour

    #: The codecs the parser reports no colour description for: every field
    #: is zero, and a matrix of zero is the code point for RGB.
    UNDESCRIBED = frozenset({api.cudaVideoCodec_VP9})

    @classmethod
    def of(cls, format: api.CUVIDEOFORMAT) -> Sequence:
        """The sequence the parser describes in `format`."""
        area, signal = format.display_area, format.video_signal_description
        described = Colour() if int(format.codec) in cls.UNDESCRIBED else Colour(
            matrix=int(signal.matrix_coefficients),
            full_range=bool(signal.video_full_range_flag),
            primaries=int(signal.color_primaries),
            transfer=int(signal.transfer_characteristics))
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
            colour=described)

    @property
    def size(self) -> tuple[int, int]:
        """The visible picture's size, as (width, height)."""
        return self.visible[2], self.visible[3]

    @property
    def surface_size(self) -> tuple[int, int]:
        """The size pictures are copied out at: the coded size, made even each way.

        VP9 and AV1 code a picture of any size, and a 4:2:0 surface has a
        chroma sample for every two luma samples. A decoder asked for an odd
        size scales the picture to an even one; asked for the even size above,
        it hands the picture over as decoded, with a column or a row past its
        edge that the visible rectangle leaves out.
        """
        width, height = self.coded_size
        return width + width % 2, height + height % 2

    @property
    def depth(self) -> int:
        """Bits a luma sample."""
        return self.bit_depth[0]

    @property
    def monochrome(self) -> bool:
        return self.chroma_format == api.cudaVideoChromaFormat_Monochrome

    @property
    def surface_format(self) -> SurfaceFormat | None:
        """The surfaces this sequence decodes into, or None where it is not decoded here."""
        if self.chroma_format not in TWO_PLANE_CHROMA:
            return None
        if not self.monochrome and self.bit_depth[1] != self.depth:
            return None
        return SURFACES.get(self.depth)

    def refusal(self, capabilities: api.CUVIDDECODECAPS, device: str) -> str:
        """Why `device` cannot decode this sequence into textures, or '' where it can.

        capabilities -- what ``cuvidGetDecoderCaps`` answered for this
            sequence's codec, chroma format and bit depth
        """
        width, height = self.coded_size
        chroma = api.CHROMA_FORMAT_NAMES.get(self.chroma_format,
                                             f'chroma format {self.chroma_format}')
        title = api.CODEC_TITLES.get(self.codec, f'codec {self.codec}')
        stream = f'{title}, {chroma} at {max(self.bit_depth)} bits'
        surface = self.surface_format
        if surface is None:
            return (f'the stream is {stream}, and this decoder takes 4:2:0 or '
                    f'monochrome at {" or ".join(map(str, SURFACES))} bits')
        if not capabilities.bIsSupported:
            return f'{device} has no hardware decoder for this stream ({stream})'
        if not capabilities.nOutputFormatMask & surface.output_bit:
            return f'{device} does not decode this stream ({stream}) to {surface.name}'
        most = (int(capabilities.nMaxWidth), int(capabilities.nMaxHeight))
        least = (int(capabilities.nMinWidth), int(capabilities.nMinHeight))
        if width > most[0] or height > most[1]:
            return (f'a {width}x{height} picture is larger than the '
                    f'{most[0]}x{most[1]} {device} decodes in {title}')
        if width < least[0] or height < least[1]:
            return (f'a {width}x{height} picture is smaller than the '
                    f'{least[0]}x{least[1]} {device} decodes in {title}')
        macroblocks = -(-width // 16) * -(-height // 16)
        if macroblocks > capabilities.nMaxMBCount:
            return (f'a {width}x{height} picture is {macroblocks} macroblocks, and '
                    f'{device} decodes up to {int(capabilities.nMaxMBCount)}')
        if self.surfaces < 1:
            return 'the parser asks for a decoder with no decode surfaces'
        left, top, visible_width, visible_height = self.visible
        surface_width, surface_height = self.surface_size
        if not (left >= 0 and top >= 0 and visible_width >= 1 and visible_height >= 1
                and left + visible_width <= surface_width
                and top + visible_height <= surface_height):
            return (f'the parser states a visible picture of {visible_width}x'
                    f'{visible_height} at ({left}, {top}), which is not inside the '
                    f'{surface_width}x{surface_height} it is decoded at')
        return ''


def plane_copies(pointer: int, pitch: int, size: tuple[int, int],
                 surface: SurfaceFormat = SURFACES[8],
                 ) -> tuple[api.CUDA_MEMCPY2D, api.CUDA_MEMCPY2D]:
    """The source halves of the two copies that take a mapped picture out.

    pointer/pitch -- where ``cuvidMapVideoFrame`` put the picture in device
        memory, and the bytes from one row to the next
    size -- the size the picture is mapped at (:attr:`Sequence.surface_size`)
    surface -- the format the picture was decoded into

    Luma is `height` rows of `width` samples. Chroma follows it `height` rows
    down: half as many rows, each `width` samples of interleaved Cb and Cr.
    The destination of each copy is the transfer's to fill in.
    """
    width, height = size
    planes = []
    for offset, rows in ((0, height), (pitch * height, height // 2)):
        copy = api.CUDA_MEMCPY2D()
        copy.srcMemoryType = api.CU_MEMORYTYPE_DEVICE
        copy.srcDevice = pointer + offset
        copy.srcPitch = pitch
        copy.WidthInBytes = width * surface.sample_bytes
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
    if not texture:
        raise DecoderError(
            'OpenGL made no texture for a decoded picture: no OpenGL context is '
            'current on this thread')
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

    def __init__(self, cuda: api.Cuda, size: tuple[int, int],
                 surface: SurfaceFormat = SURFACES[8]) -> None:
        self.cuda = cuda
        self.size = size
        self.surface = surface

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
            for index, plane in enumerate(
                    plane_copies(pointer, pitch, self.size, self.surface)):
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
    each picture crosses the bus twice, about 3 MB at 1080p and 8 bits.
    """

    zero_copy = False

    def __init__(self, cuda: api.Cuda, size: tuple[int, int],
                 surface: SurfaceFormat = SURFACES[8]) -> None:
        self.cuda = cuda
        self.size = size
        self.surface = surface
        self._sizes = surface.plane_sizes(size)
        (width, height), (chroma_width, chroma_height) = self._sizes
        self._luma = (surface.host_type * (width * height))()
        self._chroma = (surface.host_type * (2 * chroma_width * chroma_height))()

    def adopt(self, pair: TexturePair) -> None:
        """Nothing to register: the textures are OpenGL's alone."""

    def copy(self, pair: TexturePair, pointer: int, pitch: int) -> None:
        """Copy the mapped picture at `pointer` out, then into `pair`'s textures."""
        cuda = self.cuda
        buffers = (self._luma, self._chroma)
        copies = plane_copies(pointer, pitch, self.size, self.surface)
        for plane, buffer in zip(copies, buffers, strict=True):
            plane.dstMemoryType = api.CU_MEMORYTYPE_HOST
            plane.dstHost = ctypes.addressof(buffer)
            plane.dstPitch = plane.WidthInBytes
            cuda.check(cuda.cuMemcpy2D(ctypes.byref(plane)), 'cuMemcpy2D',
                       'device memory to host memory')
        uploads = zip(pair.planes, (GL_RED, GL_RG), self._sizes, buffers, strict=True)
        # A renderer may have left any of these set, and each changes what
        # glTexSubImage2D reads from the buffer: where a row begins, and the
        # order of the two bytes of a 16-bit sample. The rest of the pixel
        # store applies to bitmaps and to three-dimensional images.
        packing = {name: int(glGetIntegerv(name))
                   for name in (GL_UNPACK_ALIGNMENT, GL_UNPACK_ROW_LENGTH,
                                GL_UNPACK_SKIP_ROWS, GL_UNPACK_SKIP_PIXELS,
                                GL_UNPACK_SWAP_BYTES)}
        unpack_buffer = int(glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING))
        texture = int(glGetIntegerv(GL_TEXTURE_BINDING_2D))
        try:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            for name in packing:
                glPixelStorei(name, 1 if name == GL_UNPACK_ALIGNMENT else 0)
            for name, format, (plane_width, plane_height), buffer in uploads:
                glBindTexture(GL_TEXTURE_2D, name)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, plane_width, plane_height,
                                format, self.surface.kind, buffer)
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

    owner -- the OpenGL context the textures are made in, whose share group
        they belong to
    decoded -- the pictures the hardware decoder has decoded, by index, which
        are the ones a later picture may predict from and the ones that may
        be shown
    """

    #: Held while a hardware decoder is made or destroyed, in any session. A
    #: decoder is destroyed on a thread of its own (:meth:`stop`), so without
    #: it the next one is made while that is under way, and the driver has
    #: faulted inside ``cuvidDestroyDecoder`` with a second destroy and a
    #: ``cuvidCreateDecoder`` running beside it.
    lifecycle = threading.Lock()

    def __init__(self, cuda: api.Cuda, cuvid: api.Cuvid, context: c_void_p,
                 sequence: Sequence, surface: SurfaceFormat, transfer: Transfer,
                 slots: int, deinterlace: int, owner: Owner) -> None:
        self.cuda = cuda
        self.cuvid = cuvid
        self.context = context
        self.sequence = sequence
        self.surface = surface
        self.transfer = transfer
        self.owner = owner
        self.decoded: set[int] = set()
        self.decoder = c_void_p()
        # Every picture the parser holds may be shown by one flush, on top of
        # the frames the caller already keeps.
        self.outputs: Outputs[TexturePair] = Outputs(sequence.surfaces + slots, self._pair)
        width, height = sequence.coded_size
        mapped_width, mapped_height = sequence.surface_size
        information = api.CUVIDDECODECREATEINFO(
            ulWidth=width, ulHeight=height, ulMaxWidth=width, ulMaxHeight=height,
            ulTargetWidth=mapped_width, ulTargetHeight=mapped_height,
            ulNumDecodeSurfaces=sequence.surfaces, ulNumOutputSurfaces=1,
            CodecType=sequence.codec, ChromaFormat=sequence.chroma_format,
            bitDepthMinus8=sequence.depth - 8,
            ulCreationFlags=api.cudaVideoCreate_PreferCUVID,
            OutputFormat=surface.surface,
            DeinterlaceMode=(api.cudaVideoDeinterlaceMode_Weave if sequence.progressive
                             else deinterlace))
        # The whole coded picture, unscaled: the visible rectangle is the
        # conversion pass's to apply, as it is for every backend.
        information.display_area.right = mapped_width
        information.display_area.bottom = mapped_height
        with self.lifecycle:
            status = cuvid.cuvidCreateDecoder(ctypes.byref(self.decoder),
                                              ctypes.byref(information))
        cuda.check(status, 'cuvidCreateDecoder',
                   f'{width}x{height} {surface.name}, {sequence.surfaces} surfaces')

    def _pair(self) -> TexturePair:
        surface = self.surface
        luma, chroma = surface.plane_sizes(self.sequence.surface_size)
        pair = TexturePair(_plane_texture(surface.luma, *luma),
                           _plane_texture(surface.chroma, *chroma))
        try:
            self.transfer.adopt(pair)
        except BaseException:
            self.transfer.forget(pair)
            glDeleteTextures(list(pair.planes))
            raise
        return pair

    def decode(self, picture: api.CUVIDPICPARAMS, fed: int) -> None:
        """Hand one picture's parameters, as the parser built them, to the hardware.

        fed -- the bytes handed to the parser that no picture has yet been
            decoded from

        Raises :class:`DecoderError` for a picture :func:`guard.refusal`
        names a fault in, which the hardware is then not handed.
        """
        codec = self.sequence.codec
        refused = guard.refusal(picture, codec=codec, surfaces=self.sequence.surfaces,
                                decoded=self.decoded, fed=fed)
        # The parser takes the picture as decoded into whatever happens here,
        # so what its index held before is no longer a picture to predict from.
        targets = guard.targets(codec, picture)
        self.decoded.difference_update(targets)
        if refused:
            raise DecoderError(f'a picture NVIDIA\'s parser built is refused: {refused}')
        self.cuda.check(
            self.cuvid.cuvidDecodePicture(self.decoder, ctypes.byref(picture)),
            'cuvidDecodePicture')
        self.decoded.update(targets)

    def forget_pictures(self) -> None:
        """Hold no decoded picture: the parser that named them was replaced."""
        self.decoded.clear()

    def copy_out(self, picture: api.CUVIDPARSERDISPINFO, pair: TexturePair) -> None:
        """Copy a displayed picture into `pair`, waiting for its decode to finish."""
        cuda, cuvid = self.cuda, self.cuvid
        index = int(picture.picture_index)
        if not 0 <= index < self.sequence.surfaces:
            raise DecoderError(
                f'the parser shows picture {index} of a decoder made with '
                f'{self.sequence.surfaces}')
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

        Decoders are destroyed one at a time, and none is made while one is
        being destroyed (:attr:`lifecycle`): a sequence that begins while a
        decoder is still going waits for it.
        """
        decoder, self.decoder = self.decoder, c_void_p()
        self.decoded.clear()
        if decoder:
            threading.Thread(target=self._destroy, args=(decoder,),
                             name='nvdec-destroy').start()

    def _destroy(self, decoder: c_void_p) -> None:
        with self.lifecycle, self.cuda.current(self.context):
            self.cuvid.cuvidDestroyDecoder(decoder)

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has released everything the session held."""
        return not self.decoder and not self.outputs.made

    def close(self) -> None:
        """Destroy the decoder and give back every texture pair.

        The hardware decoder is destroyed whichever OpenGL context is
        current. The textures belong to the share group of the context the
        session was made in: with a context outside it current, or none, they
        are kept, registered with CUDA as they were, and
        :class:`DecoderError` says so once the decoder is gone. Closing again
        with a context of the group current releases them.
        """
        release_all(self.stop, self._release_textures)

    def _release_textures(self) -> None:
        if not self.outputs.made:
            return
        self.owner.require("an NVDEC decoder's plane textures", DecoderError)
        pairs = self.outputs.drain()
        release_all(functools.partial(self._forget, pairs),
                    functools.partial(glDeleteTextures,
                                      [texture for pair in pairs for texture in pair.planes]))

    def _forget(self, pairs: list[TexturePair]) -> None:
        with self.cuda.current(self.context):
            for pair in pairs:
                self.transfer.forget(pair)


# ------------------------------------------------------------------ decoder


def _callback(*, shows: bool = False) -> Callable[[Callable[..., int]], Callable[..., int]]:
    """Make a method safe to call from the parser.

    shows -- the callback shows a picture that has already been decoded, so
        its failure leaves the pictures the stream predicts from as they were

    An exception cannot cross the C frames between the parser and
    :meth:`NVDECDecoder._parse`, so it is kept, the parse is failed by
    returning 0, and ``_parse`` raises it once the parser has returned.
    """
    def wrap(method: Callable[..., int]) -> Callable[..., int]:
        @functools.wraps(method)
        def trapping(self: NVDECDecoder, *arguments: Any) -> int:
            try:
                return method(self, *arguments)
            except BaseException as error:  # noqa: BLE001 - re-raised by _parse, outside the C frames
                if self._failure is None:
                    self._failure, self._failed_showing = error, shows
                return 0
        return trapping
    return wrap


#: What a decode or a flush with no context of the share group current is told.
WRONG_CONTEXT = (
    'neither the OpenGL context this decoder was opened in nor one that '
    'shares objects with it is current on this thread, and its pictures are '
    'copied into textures of that context')


class NVDECDecoder(Decoder):
    """Decoding on NVIDIA hardware, into OpenGL textures.

    codec -- ``'h264'``, ``'hevc'``, ``'vp9'`` or ``'av1'``, where the GPU
        decodes it; :func:`pyopengl_video.decoders` lists them for this machine
    headers -- the stream's configuration, when the container keeps it apart
        from the samples: parameter sets as Annex-B for H.264 and HEVC, the
        configuration OBUs for AV1
    size -- accepted for the registry; the stream's own size is used
    output_slots -- how many frames a caller may hold unreleased at once
    deinterlace -- how an interlaced stream's fields become frames:
        ``'adaptive'``, ``'bob'`` or ``'weave'`` (:data:`DEINTERLACE_MODES`)
    interop -- True to copy each picture into the textures on the GPU, which
        needs the current OpenGL context to render on the NVIDIA GPU; False
        to decode on the first NVIDIA GPU and copy through host memory

    Each sample is one access unit, temporal unit or frame, as its container
    holds it. Frames come out in display order as
    :class:`~pyopengl_video.decoder.DecodedFrame` objects, ``'nv12'`` from an
    8-bit stream and ``'p010'`` from a 10-bit one, each holding its own copy
    of the picture.

    The decoder's textures belong to the share group of the OpenGL context
    current when it is made: :meth:`decode` and :meth:`flush` raise
    :class:`DecoderError` with a context outside that group current, or none.

    An AV1 stream is read by this package's parser before NVIDIA's
    (:class:`pyopengl_video.nvdec.av1.Gate`). A sample that breaks the AV1
    specification raises :class:`DecoderError` and is not decoded, and the
    samples from there to the next key frame are passed over, as are those
    before the first key frame of a stream or after a :meth:`reset`.
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
        name = codecs.canonical(codec)
        if name not in api.CODECS:
            raise DecoderError(
                f'the NVDEC backend decodes {", ".join(sorted(api.CODECS))}, not {codec!r}')
        if deinterlace not in DEINTERLACE_MODES:
            raise DecoderError(
                f'deinterlace is one of {", ".join(sorted(DEINTERLACE_MODES))}, '
                f'not {deinterlace!r}')
        del size
        self.codec = name
        self.output_slots = bounds.whole(output_slots, 'output_slots', 1,
                                         MOST_OUTPUT_SLOTS, DecoderError)
        self.zero_copy = bool(interop)
        self._deinterlace = DEINTERLACE_MODES[deinterlace]
        self._headers = bytes(headers)
        self._closed = False
        #: The context the plane textures are made in; its share group deletes them.
        self._owner = Owner.current()
        #: AV1 alone: what reads each sample before NVIDIA's parser does.
        self._gate = Gate() if name == codecs.AV1 else None
        if self._gate is not None:
            self._headers = self._gate.configure(self._headers)
        self._described: Description | None = None
        self._presentation = Presentation()
        # VP9 alone: the reference slots, the coded frame the parser has been
        # handed, and the sample whose timestamp a picture shown now takes.
        self._slots = Slots() if name == codecs.VP9 else None
        self._coding: Frame | None = None
        self._showing: Sample | None = None
        self._session: _Session | None = None
        self._retired: list[_Session] = []
        self._lent: dict[DecodedFrame, tuple[_Session, TexturePair]] = {}
        self._ready: list[DecodedFrame] = []
        self._failure: BaseException | None = None
        #: Whether that failure was in showing a picture and not in decoding one.
        self._failed_showing = False
        #: The bytes handed to the parser that no picture has been decoded
        #: from, which is the most a picture's bitstream can be.
        self._unconsumed = 0
        #: Whether the parser handed the hardware a picture in the call under way.
        self._pictured = False
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
        self._note_description(self._headers)

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

    def _note_description(self, data: bytes) -> None:
        """Keep what this package reads of the picture in `data`.

        The parser reports a sequence at its first picture, and never where
        chroma samples sit. This package's own reading gives the size before
        the first sample and the whole colour description
        (:mod:`pyopengl_video.nvdec.description`). It is looked for in the
        headers, in key frames and in the first sample decoded, which is
        where a stream carries it.
        """
        try:
            found = describe(self.codec, data)
        except Exception as error:  # noqa: BLE001 - whatever a parser raises of bytes it does not read, the description is not had
            # The hardware may decode a stream of H.264, HEVC or VP9 that
            # this package does not read; an AV1 one never gets this far.
            log.debug('the %s description was not read here: %s',
                      codecs.title(self.codec), error)
            self._described = None
            return
        if found is None:
            return
        self._described = found
        if self._session is None:
            self.size = found.size
            self.colour = found.colour

    # ------------------------------------------------------------ decoding

    def decode(self, sample: Sample) -> list[DecodedFrame]:
        """Decode one access unit; return the frames now ready, in display order."""
        self._check_open()
        self._require_context()
        data = bytes(sample.data)
        if self._gate is not None:
            admitted = self._gate.admit(data)
            if admitted is None:
                log.debug('an AV1 sample at %d is passed over: decoding starts at a '
                          'key frame that is shown', sample.timestamp)
            if not admitted:
                return []
            data = admitted
        if sample.keyframe or not self._primed:
            self._note_description(data)
        with self._cuda.current(self._context):
            if not self._primed:
                self._primed = True
                if self._headers:
                    self._parse(self._headers, 0)
            self._presentation.fed(sample)
            if self._slots is not None:
                return self._decode_vp9(self._slots, sample, data)
            return self._parse(data, PICTURE, sample.timestamp)

    def _decode_vp9(self, slots: Slots, sample: Sample, data: bytes) -> list[DecodedFrame]:
        """Decode one VP9 sample a frame at a time, showing again what it says to.

        The parser is handed each coded frame by itself, so the picture it
        decodes next is that frame's, and a frame shown again is shown between
        the two coded frames it sits between.
        """
        try:
            frames = slots.frames(data)
        except Exception as error:  # noqa: BLE001 - whatever the parser raises of a sample it does not read, the slots are unknown
            # The hardware may decode what this package does not read: the
            # parser takes the sample whole, and the slots are unknown.
            log.debug('a VP9 sample was not read here: %s', error)
            slots.forget()
            frames = [Frame(data)]
        ready: list[DecodedFrame] = []
        self._showing = sample
        try:
            for frame in frames:
                if frame.shows_slot is None:
                    self._coding = frame
                    ready += self._parse(frame.data, PICTURE, sample.timestamp)
                else:
                    ready += self._show_again(slots, frame.shows_slot, sample.timestamp)
        except BaseException:
            for lent in ready:
                lent.release()
            raise
        finally:
            self._showing = self._coding = None
        return ready

    def _show_again(self, slots: Slots, slot: int, timestamp: int) -> list[DecodedFrame]:
        """The picture reference `slot` holds, as the frame shown at `timestamp`."""
        picture = slots.picture(slot)
        if picture is None or self._session is None:
            log.warning('a VP9 frame shows reference slot %d again, and no frame '
                        'decoded here was stored in it; the picture is not shown', slot)
            self._presentation.shown(timestamp)
            return []
        self._show(api.CUVIDPARSERDISPINFO(picture_index=picture, progressive_frame=1),
                   timestamp)
        ready, self._ready = self._ready, []
        return ready

    def flush(self) -> list[DecodedFrame]:
        """Return every frame still held back, at the end of a stream."""
        self._check_open()
        self._require_context()
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
        self._unconsumed = 0
        if self._session is not None:
            # A new parser numbers its pictures from nothing.
            self._session.forget_pictures()
        if self._slots is not None:
            self._slots.forget()
        if self._gate is not None:
            self._gate.restart()
            self._gate.configure(self._headers)

    def _parse(self, data: bytes, flags: int, timestamp: int = 0) -> list[DecodedFrame]:
        """Hand `data` to the parser; return the frames its callbacks produced."""
        cuda, cuvid = self._cuda, self._cuvid
        if not self._parser:
            parameters = api.CUVIDPARSERPARAMS(
                CodecType=api.CODECS[self.codec],
                # The sequence callback answers with the count the stream needs.
                ulMaxNumDecodeSurfaces=1,
                ulMaxDisplayDelay=0 if self.codec in SHOWN_AS_PARSED else DISPLAY_DELAY)
            (parameters.pfnSequenceCallback, parameters.pfnDecodePicture,
             parameters.pfnDisplayPicture) = self._callbacks
            cuda.check(cuvid.cuvidCreateVideoParser(ctypes.byref(self._parser),
                                                    ctypes.byref(parameters)),
                       'cuvidCreateVideoParser')
        packet = api.CUVIDSOURCEDATAPACKET(
            flags=flags, payload_size=len(data), timestamp=timestamp,
            payload=ctypes.cast(ctypes.c_char_p(data), c_void_p) if data else None)
        self._unconsumed += len(data)
        self._pictured = False
        status = cuvid.cuvidParseVideoData(self._parser, ctypes.byref(packet))
        if self._pictured:
            self._unconsumed = 0
        ready, self._ready = self._ready, []
        failure, self._failure = self._failure, None
        showing, self._failed_showing = self._failed_showing, False
        try:
            if failure is not None:
                raise failure
            cuda.check(status, 'cuvidParseVideoData',
                       f'the parser refused {len(data)} bytes of {codecs.title(self.codec)}')
        except BaseException:
            if self._gate is not None and not showing:
                # A picture the frames after this one predict from was not
                # decoded. One that was decoded and could not be shown leaves
                # the references whole, and the stream decodes on.
                self._gate.lose()
            for frame in ready:
                frame.release()
            raise
        return ready

    # ------------------------------------------------- the parser's callbacks

    @_callback()
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
        """Replace the session with one for `sequence`.

        The session before is retired whatever becomes of the new one: the
        pictures of a sequence that is refused are not handed to the decoder
        of the sequence before it. It is retired once the new decoder has
        been made, so that making it does not wait for the old one to go.
        """
        previous, self._session = self._session, None
        if self._slots is not None:
            # The pictures the slots named were the decoder's before.
            self._slots.lose_pictures()
        try:
            self._session = self._make_session(sequence)
        finally:
            if previous is not None:
                self._retire(previous)

    def _make_session(self, sequence: Sequence) -> _Session:
        """A session for `sequence`; :class:`DecoderError` where the GPU does not decode it."""
        cuda, cuvid = self._cuda, self._cuvid
        capabilities = api.CUVIDDECODECAPS(
            eCodecType=sequence.codec, eChromaFormat=sequence.chroma_format,
            nBitDepthMinus8=sequence.depth - 8)
        cuda.check(cuvid.cuvidGetDecoderCaps(ctypes.byref(capabilities)),
                   'cuvidGetDecoderCaps')
        refusal = sequence.refusal(capabilities, self.device)
        surface = sequence.surface_format
        if refusal or surface is None:
            raise DecoderError(refusal or 'the stream decodes to no surface format')
        log.debug('%s: decoding %dx%d with %d surfaces', self.device,
                  *sequence.coded_size, sequence.surfaces)
        transfer_type = Interop if self.zero_copy else Upload
        return _Session(cuda, cuvid, self._context, sequence, surface,
                        transfer_type(cuda, sequence.surface_size, surface),
                        self.output_slots, self._deinterlace, self._owner)

    def _describe(self, sequence: Sequence) -> None:
        self.size = sequence.size
        described = self._described.colour if self._described is not None else sequence.colour
        self.colour = described.resolved(sequence.size[1])

    @_callback()
    def _on_decode(self, _user: Any, parameters: Any) -> int:
        """A picture is ready to decode: the hardware takes it once it has been read here."""
        session = self._session
        if session is None:
            raise DecoderError('the parser asked for a picture before it described a sequence')
        if not parameters:
            raise DecoderError('the parser asked for a picture and gave no parameters')
        picture = parameters.contents
        session.decode(picture, self._unconsumed)
        self._pictured = True
        if self._slots is not None and self._coding is not None:
            self._slots.decoded(self._coding, int(picture.CurrPicIdx))
        return 1

    @_callback(shows=True)
    def _on_display(self, _user: Any, information: Any) -> int:
        """A picture is due to be shown: copy it out before the parser reuses it."""
        picture = information.contents
        showing = self._showing
        self._show(picture, picture.timestamp if showing is None else showing.timestamp)
        return 1

    def _show(self, picture: api.CUVIDPARSERDISPINFO, timestamp: int) -> None:
        """Copy `picture` into a texture pair and lend it as the frame at `timestamp`."""
        duration = self._presentation.shown(timestamp)
        session = self._session
        if duration is None or session is None:
            return
        index = int(picture.picture_index)
        if index not in session.decoded:
            # The parser holds pictures back, and shows the ones it held for
            # the sequence before from whichever decoder is there now.
            log.warning('the parser shows picture %d at %d, which this decoder has '
                        'not decoded; the picture is not shown', index, timestamp)
            return
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
            timestamp=timestamp, duration=duration, planes=pair.planes,
            layout=session.surface.layout, texture_size=sequence.surface_size,
            visible=sequence.visible, colour=self.colour, on_release=self._released)
        self._lent[frame] = (session, pair)
        self._ready.append(frame)

    # ------------------------------------------------------------ lifetimes

    def _released(self, frame: DecodedFrame) -> None:
        held = self._lent.pop(frame, None)
        if held is None:
            return
        owner, pair = held
        owner.outputs.give(pair)
        if owner in self._retired and not self._lends_from(owner):
            self._retire(owner)

    def _lends_from(self, session: _Session) -> bool:
        return any(owner is session for owner, _ in self._lent.values())

    def _retire(self, session: _Session) -> None:
        """Stop a session's decoder; keep its textures while they cannot be deleted.

        That is while frames of them are held, and while no context of
        their share group is current: :meth:`close` deletes them then.
        """
        if session in self._retired:
            self._retired.remove(session)
        if self._lends_from(session) or not self._owner.is_shared():
            session.stop()
        else:
            session.close()
        if not session.closed:
            self._retired.append(session)

    def close(self) -> None:
        """Release the decoder, the parser and the textures, frames lent included.

        Every release runs whatever the others do. The parser and the
        hardware decoders are released whichever OpenGL context is current.
        The plane textures belong to the share group of the context the
        decoder was opened in: with a context outside it current, or none,
        they are kept, :class:`DecoderError` says so, and closing again with
        a context of the group current releases them. Closing a decoder that has
        nothing left to release does nothing.
        """
        self._closed = True
        release_all(self._release_frames, self._close_parser, self._close_sessions)

    def _release_frames(self) -> None:
        lent, self._lent = self._lent, {}
        self._ready = []
        for frame in lent:
            frame.release()

    def _close_parser(self) -> None:
        if self._parser:
            with self._cuda.current(self._context):
                self._restart()
        # Dropped to break the cycle through the bound methods; the parser
        # that held their addresses is gone.
        self._callbacks = ()  # type: ignore[assignment]  # nothing reads them once the parser is destroyed

    def _close_sessions(self) -> None:
        sessions = [*self._retired, *([self._session] if self._session else [])]
        self._session = None
        self._retired = []
        try:
            release_all(*(session.close for session in sessions))
        finally:
            self._retired = [session for session in sessions if not session.closed]

    def _check_open(self) -> None:
        if self._closed:
            raise DecoderError('this decoder has been closed')

    def _require_context(self) -> None:
        if not self._owner.is_shared():
            raise DecoderError(WRONG_CONTEXT)
