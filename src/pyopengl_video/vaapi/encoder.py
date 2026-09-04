"""The VA-API encoder: an OpenGL texture in, an H.264 elementary stream out.

The frame never leaves the GPU. A texture is exported as a DMA-BUF and imported
as a VA surface, the driver's video processing block converts it from RGB to
NV12, and the video engine codes that. What crosses the bus is the compressed
result.

libva leaves the codec decisions to the caller, so this encoder is paired with
:mod:`pyopengl_video.vaapi.h264`, which decides what each picture is and writes
the parameter sets the driver puts in the stream. Everything here is the
plumbing between the two.

**Pictures are pipelined by one.** :meth:`VAAPIEncoder.encode` submits the
picture it is given and returns the packet from the picture before it, so the
GPU has a whole frame to finish the drawing, the colour conversion and the
coding before anything waits on them. An empty list from the first
:meth:`~VAAPIEncoder.encode` is ordinary, and :meth:`VAAPIEncoder.flush` returns
what is still inside. Packets keep their submitted order, so nothing a container
does has to change.
"""
from __future__ import annotations

import ctypes
import dataclasses
import functools
import logging
import os
from collections.abc import Callable
from typing import Any

from pyopengl_video import inputs
from pyopengl_video.encoder import (
    Encoder,
    EncoderError,
    Packet,
    default_bitrate,
    frame_rate_ratio,
)
from pyopengl_video.linux import dmabuf
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.h264 import (
    COLOUR_PRIMARIES_BT709,
    MATRIX_BT709,
    SLICE_TYPE_I,
    SLICE_TYPE_P,
    TRANSFER_BT709,
    GroupOfPictures,
    ParameterSets,
    Picture,
    annexb,
)

log = logging.getLogger(__name__)

__all__ = ['VAAPIEncoder', 'InputHandle', 'RATE_CONTROL_MODES',
           'PIPELINE_DEPTH', 'colour_pipeline', 'as_encoder_error']

#: How many pictures are in flight at once. One means :meth:`encode` returns the
#: previous picture's packet, which is what keeps the GPU from being waited on
#: for work submitted a moment ago.
PIPELINE_DEPTH = 1

#: Rate control a caller may ask for, and the libva mode each names.
RATE_CONTROL_MODES = {
    'cbr': api.VA_RC_CBR,
    'vbr': api.VA_RC_VBR,
    'cqp': api.VA_RC_CQP,
}

#: The packed headers this backend supplies, and so the only ones it asks the
#: driver to expect. Asking for one is a promise to write it every picture that
#: needs it, and the driver writes none of its own once asked.
#:
#: The slice header is in the list because a libva encoder need not write one:
#: Mesa's writes the coded macroblocks and nothing in front of them, so without
#: this every picture reaches the stream with no NAL header on it and no player
#: finds a picture at all. The sequence header carries both parameter sets,
#: which is how a driver taking one header per picture is given both.
PACKED_HEADERS_WRITTEN = (api.VA_ENC_PACKED_HEADER_SEQUENCE
                          | api.VA_ENC_PACKED_HEADER_SLICE)

#: What fraction of the ceiling a variable-bitrate stream aims at.
VBR_TARGET_PERCENTAGE = 70

#: How much video the hypothetical reference decoder's buffer holds, as a
#: multiple of one second's bits.
HRD_BUFFER_SECONDS = 2


def as_encoder_error(method: Callable[..., Any]) -> Callable[..., Any]:
    """Let a method raise only :class:`EncoderError`, keeping the cause.

    Two error types come from underneath this backend: libva's, raised by every
    driver call through :meth:`~pyopengl_video.vaapi.api.VA.check`, and the
    DMA-BUF shim's, raised when a texture will not export. Both say something
    worth reading and neither is an :class:`EncoderError`, so a caller who
    wrapped a recording in one ``except`` clause would have them go straight
    past.

    They are translated here rather than being made encoder errors in
    themselves: :mod:`pyopengl_video.linux.dmabuf` serves more than encoders,
    and libva decodes as well as encodes, so neither is an encoder's error
    until an encoder is what raised it. The original stays on ``__cause__``,
    which is where a traceback shows it.
    """
    @functools.wraps(method)
    def translating(*arguments: Any, **named: Any) -> Any:
        try:
            return method(*arguments, **named)
        except EncoderError:
            raise
        except (api.VAError, dmabuf.DMABufError) as error:
            raise EncoderError(str(error)) from error
    return translating


def colour_pipeline(surface: int) -> api.VAProcPipelineParameterBuffer:
    """The video processing pass that turns a full-range RGB surface into NV12.

    A renderer's colours are full range, and the stream this backend produces
    declares limited-range BT.709 -- so the conversion has to say the same
    thing. The two are set in different places and nothing but a picture coming
    out grey and washed notices when they part, which is why this is one
    function and :func:`tests.test_vaapi_encode` reads it back from a real
    surface.
    """
    pipeline = api.VAProcPipelineParameterBuffer()
    pipeline.surface = surface
    pipeline.surface_color_standard = api.VAProcColorStandardNone
    pipeline.output_color_standard = api.VAProcColorStandardBT709
    pipeline.input_color_properties.color_range = api.VA_SOURCE_RANGE_FULL
    pipeline.output_color_properties.color_range = api.VA_SOURCE_RANGE_REDUCED
    pipeline.output_color_properties.colour_primaries = COLOUR_PRIMARIES_BT709
    pipeline.output_color_properties.transfer_characteristics = TRANSFER_BT709
    pipeline.output_color_properties.matrix_coefficients = MATRIX_BT709
    return pipeline


@dataclasses.dataclass
class InputHandle(inputs.InputHandle):
    """A texture the encoder reads, and the VA surface naming the same memory.

    texture/target -- what was registered
    framebuffer/owns_texture -- see :class:`pyopengl_video.inputs.InputHandle`
    exported -- the DMA-BUF export keeping the descriptors alive
    surface -- the VA surface over that buffer, which the conversion reads
    fence -- an OpenGL fence made when drawing into the texture finished, which
        the conversion waits on rather than the whole pipeline being flushed
    """

    texture: int = 0
    target: int = 0
    framebuffer: int = 0
    owns_texture: bool = False
    exported: dmabuf.ExportedImage | None = None
    surface: int = 0
    fence: Any = None

    def for_drawing(self):
        """Hold the texture for OpenGL, and note when the drawing is done.

        The fence made at the end of this scope is what the encoder waits on
        before the conversion reads the texture. Waiting on a fence rather than
        finishing the whole pipeline leaves the driver free to keep working on
        everything else that was queued.
        """
        return _Drawing(self)

    def close(self) -> None:
        """Give back the framebuffer, and the texture if this handle made it."""
        self._forget_fence()
        super().close()

    def _forget_fence(self) -> None:
        if self.fence is not None:
            from OpenGL.GL import glDeleteSync
            glDeleteSync(self.fence)
            self.fence = None


class _Drawing:
    """The scope :meth:`InputHandle.for_drawing` opens."""

    def __init__(self, handle: InputHandle) -> None:
        self.handle = handle

    def __enter__(self) -> InputHandle:
        self.handle._forget_fence()
        return self.handle

    def __exit__(self, *exception: object) -> None:
        from OpenGL.GL import GL_SYNC_GPU_COMMANDS_COMPLETE, glFenceSync
        self.handle.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)


@dataclasses.dataclass
class _Submission:
    """One picture handed to the encoder and not yet read back."""

    coded_buffer: int
    source: int
    timestamp: int
    duration: int
    keyframe: bool


class VAAPIEncoder(Encoder):
    """H.264 encoding on Intel and AMD hardware, fed from an OpenGL texture.

    width/height -- the picture size, both even
    codec -- ``'h264'``; nothing else is produced here
    fps -- frame rate, as a number or an exact ``(numerator, denominator)``
    bitrate -- bits per second; the default is derived from the frame size and
        rate
    gop -- frames between key frames
    rate_control -- ``'cbr'``, ``'vbr'`` or ``'cqp'``; a mode the driver does
        not offer is refused by name
    qp -- the quantiser, which is what ``'cqp'`` codes at and what the other
        modes start from
    bframes -- must be zero: pictures are coded in display order here
    device -- the DRM render node to open, such as ``/dev/dri/renderD128``;
        the default is the first that has an H.264 encoder
    """

    allocates_inputs = False
    zero_copy = True
    reorders_frames = False

    @as_encoder_error
    def __init__(self, width: int, height: int, codec: str = 'h264', *,
                 fps: float | tuple[int, int] = 30,
                 bitrate: int | None = None, gop: int | None = None,
                 rate_control: str = 'cbr', qp: int = 26, bframes: int = 0,
                 device: str | None = None, **unknown: object) -> None:
        if unknown:
            raise EncoderError(
                f'the VA-API encoder has no setting named '
                f'{", ".join(sorted(unknown))}; it takes fps, bitrate, gop, '
                'rate_control, qp, bframes and device')
        if codec != 'h264':
            raise EncoderError(
                f'the VA-API backend encodes h264, not {codec!r}')
        if bframes:
            raise EncoderError(
                'the VA-API backend codes pictures in display order, so '
                f'bframes must be 0, not {bframes}; a stream with B-frames '
                'needs the reordering control layer this backend does not have')
        if rate_control not in RATE_CONTROL_MODES:
            raise EncoderError(
                f'{rate_control!r} is not a rate control mode here; '
                f'choose one of {sorted(RATE_CONTROL_MODES)}')

        self.codec = codec
        self.size = (int(width), int(height))
        self.frame_rate = frame_rate_ratio(fps)
        self.bitrate = int(bitrate) if bitrate else default_bitrate(
            self.size, self.frame_rate)
        self.gop = int(gop) if gop else max(1, round(
            self.frame_rate[0] / self.frame_rate[1]))
        self.rate_control_name = rate_control
        self.qp = int(qp)

        self.api = api.VA.instance()
        self.display: Any = None
        self.device = device
        self._config = api.VAConfigID(0)
        self._context = api.VAContextID(0)
        self._vpp_config = api.VAConfigID(0)
        self._vpp_context = api.VAContextID(0)
        self._sources: list[int] = []
        self._reconstructions: list[int] = []
        self._coded_buffers: list[int] = []
        self._registered: list[InputHandle] = []
        self._pending: list[_Submission] = []
        self._slot = 0
        self._recon_slot = 0
        self._coded_slot = 0
        self._coded_sets: dict[int, bytes] = {}
        self._closed = False
        # Kept alive for as long as the display is: libva holds the pointer.
        self._silence = api.MESSAGE_CALLBACK(lambda context, message: None)

        try:
            self._open()
        except Exception:
            self.close()
            raise

    def __repr__(self) -> str:
        width, height = self.size
        return (f'<{type(self).__name__} {width}x{height} h264 '
                f'{self.rate_control_name} {self.bitrate} bit/s '
                f'on {self.device}>')

    # ---------------------------------------------------------------- setup

    def _open(self) -> None:
        """Open the device, negotiate what it can do, and build the contexts."""
        self._open_display()
        self._negotiate()
        self._build_parameter_sets()
        self._create_encode_context()
        self._create_vpp_context()
        self.structure = GroupOfPictures(
            gop=self.gop, max_num_ref_frames=self.max_num_ref_frames,
            log2_max_frame_num=self.sets.log2_max_frame_num)

    def _open_display(self) -> None:
        """Open a DRM render node and initialise libva on it."""
        nodes = [self.device] if self.device else api.render_nodes()
        if not nodes:
            raise EncoderError(
                'no DRM render node to open: /dev/dri holds no renderD* device, '
                'so there is no GPU here to encode with')
        failures = []
        for node in nodes:
            try:
                self._initialise(node)
            except (OSError, api.VAError) as error:
                failures.append(f'{node}: {error}')
                continue
            self.device = node
            return
        raise EncoderError('no DRM render node here has an H.264 encoder: '
                           + '; '.join(failures))

    def _initialise(self, node: str) -> None:
        """Open one render node, or raise and leave nothing behind."""
        self._device_fd = os.open(node, os.O_RDWR)
        display = self.api.vaGetDisplayDRM(self._device_fd)
        if not display:
            os.close(self._device_fd)
            self._device_fd = -1
            raise api.VAError(-1, 'vaGetDisplayDRM', node)
        # Before vaInitialize, so the driver's own log lines never reach a
        # caller's stderr: this is a library, and discovery must be quiet.
        self.api.vaSetInfoCallback(display, self._silence, None)
        self.api.vaSetErrorCallback(display, self._silence, None)
        major, minor = ctypes.c_int(), ctypes.c_int()
        status = self.api.vaInitialize(display, ctypes.byref(major),
                                       ctypes.byref(minor))
        if status != api.VA_STATUS_SUCCESS:
            os.close(self._device_fd)
            self._device_fd = -1
            raise api.VAError(status, 'vaInitialize', node)
        self.display = display
        self.version = (major.value, minor.value)
        vendor = self.api.vaQueryVendorString(display)
        self.vendor = vendor.decode('utf-8', 'replace') if vendor else ''
        if not self._has_h264_encoder():
            self._close_display()
            raise api.VAError(-1, 'vaQueryConfigEntrypoints',
                              f'{node} has no H.264 encode entrypoint')

    def _has_h264_encoder(self) -> bool:
        """Does this device offer H.264 slice encoding at all?"""
        count = self.api.vaMaxNumEntrypoints(self.display)
        if count <= 0:
            return False
        entrypoints = (api.VAEntrypoint * count)()
        found = ctypes.c_int(0)
        status = self.api.vaQueryConfigEntrypoints(
            self.display, api.VAProfileH264High, entrypoints,
            ctypes.byref(found))
        if status != api.VA_STATUS_SUCCESS:
            return False
        return api.VAEntrypointEncSlice in list(entrypoints)[:found.value]

    def _attributes(self, *types: int) -> dict[int, int]:
        """What the H.264 encode entrypoint says about each attribute.

        An attribute the driver will not answer comes back as
        ``VA_ATTRIB_NOT_SUPPORTED`` rather than as an error, so it is dropped
        here and the caller sees only what was actually stated.
        """
        attributes = (api.VAConfigAttrib * len(types))()
        for index, kind in enumerate(types):
            attributes[index].type = kind
        self.api.check(
            self.api.vaGetConfigAttributes(
                self.display, api.VAProfileH264High, api.VAEntrypointEncSlice,
                attributes, len(types)),
            'vaGetConfigAttributes', 'H264High/EncSlice')
        return {entry.type: entry.value for entry in attributes
                if entry.value != api.VA_ATTRIB_NOT_SUPPORTED}

    def _negotiate(self) -> None:
        """Settle what this driver will actually do, and refuse what it will not."""
        stated = self._attributes(
            api.VAConfigAttribRTFormat, api.VAConfigAttribRateControl,
            api.VAConfigAttribEncPackedHeaders, api.VAConfigAttribEncMaxRefFrames)

        formats = stated.get(api.VAConfigAttribRTFormat, 0)
        if not formats & api.VA_RT_FORMAT_YUV420:
            raise EncoderError(
                f'{self.device} encodes H.264 but not from 4:2:0 surfaces '
                f'(it offers format bits {formats:#x}), which is the only '
                'sampling this backend produces')

        modes = stated.get(api.VAConfigAttribRateControl, 0)
        wanted = RATE_CONTROL_MODES[self.rate_control_name]
        if not modes & wanted:
            offered = sorted(name for name, bit in RATE_CONTROL_MODES.items()
                             if modes & bit)
            raise EncoderError(
                f'{self.device} does not offer {self.rate_control_name} rate '
                f'control; it offers {offered or ["none this backend knows"]}')
        self.rate_control = wanted

        # Only the headers this backend actually writes are asked for. The
        # attribute states everything the driver *can* take, slice headers
        # included, and requesting one is a promise to supply it: a driver told
        # to expect packed slice headers stops writing its own, and every
        # picture then comes out as slice payload with no NAL header on it.
        self.packed_headers = stated.get(
            api.VAConfigAttribEncPackedHeaders, 0) & PACKED_HEADERS_WRITTEN

        # The low half is the size of reference list zero. AMD parts state one,
        # which is exactly what a stream of I and P pictures needs; asking for
        # more is refused by the driver rather than quietly reduced.
        reference_frames = stated.get(api.VAConfigAttribEncMaxRefFrames, 1)
        self.max_num_ref_frames = max(1, reference_frames & 0xFFFF)
        log.debug('%s: %s, reference frames %d, packed headers %#x',
                  self.device, self.vendor, self.max_num_ref_frames,
                  self.packed_headers)

    def _build_parameter_sets(self) -> None:
        """Build the sequence and picture parameter sets this stream carries."""
        width, height = self.size
        self.sets = ParameterSets(
            width=width, height=height, frame_rate=self.frame_rate,
            max_num_ref_frames=self.max_num_ref_frames, init_qp=self.qp,
            bitrate=self.bitrate, transform_8x8=True)
        self.input_slots = PIPELINE_DEPTH + 1

    def _create_encode_context(self) -> None:
        """Create the encode config, its surfaces and the context over them."""
        attribute = api.VAConfigAttrib(type=api.VAConfigAttribRTFormat,
                                       value=api.VA_RT_FORMAT_YUV420)
        attributes = [attribute,
                      api.VAConfigAttrib(type=api.VAConfigAttribRateControl,
                                         value=self.rate_control)]
        if self.packed_headers:
            attributes.append(api.VAConfigAttrib(
                type=api.VAConfigAttribEncPackedHeaders,
                value=self.packed_headers))
        array = (api.VAConfigAttrib * len(attributes))(*attributes)
        self.api.check(
            self.api.vaCreateConfig(
                self.display, api.VAProfileH264High, api.VAEntrypointEncSlice,
                array, len(attributes), ctypes.byref(self._config)),
            'vaCreateConfig', 'H264High/EncSlice')

        coded_width, coded_height = self.sets.coded_size
        # One source per picture in flight, and enough reconstructions that the
        # picture being coded never lands on one still being referenced.
        self._sources = self._create_surfaces(
            api.VA_RT_FORMAT_YUV420, coded_width, coded_height,
            PIPELINE_DEPTH + 1)
        self._reconstructions = self._create_surfaces(
            api.VA_RT_FORMAT_YUV420, coded_width, coded_height,
            self.max_num_ref_frames + 1 + PIPELINE_DEPTH)

        targets = self._sources + self._reconstructions
        surfaces = (api.VASurfaceID * len(targets))(*targets)
        self.api.check(
            self.api.vaCreateContext(
                self.display, self._config, coded_width, coded_height,
                api.VA_PROGRESSIVE, surfaces, len(targets),
                ctypes.byref(self._context)),
            'vaCreateContext', f'{coded_width}x{coded_height} encode')

        size = self._coded_buffer_size()
        for _ in range(PIPELINE_DEPTH + 1):
            buffer = api.VABufferID(0)
            self.api.check(
                self.api.vaCreateBuffer(
                    self.display, self._context, api.VAEncCodedBufferType,
                    size, 1, None, ctypes.byref(buffer)),
                'vaCreateBuffer', 'coded output')
            self._coded_buffers.append(buffer.value)

    def _coded_buffer_size(self) -> int:
        """How much room one coded picture is given.

        An uncompressed 4:2:0 frame, which no coded picture reaches: an
        intra picture at a very low quantiser is the worst case and stays well
        inside it.
        """
        width, height = self.sets.coded_size
        return max(width * height * 3 // 2, 1 << 16)

    def _create_surfaces(self, format: int, width: int, height: int,
                         count: int) -> list[int]:
        """`count` surfaces the driver allocates for itself."""
        surfaces = (api.VASurfaceID * count)()
        self.api.check(
            self.api.vaCreateSurfaces(self.display, format, width, height,
                                      surfaces, count, None, 0),
            'vaCreateSurfaces', f'{count} x {width}x{height}')
        return [int(surface) for surface in surfaces]

    def _create_vpp_context(self) -> None:
        """Create the video processing context that converts RGB into NV12."""
        self.api.check(
            self.api.vaCreateConfig(self.display, api.VAProfileNone,
                                    api.VAEntrypointVideoProc, None, 0,
                                    ctypes.byref(self._vpp_config)),
            'vaCreateConfig', 'VideoProc')
        coded_width, coded_height = self.sets.coded_size
        array = (api.VASurfaceID * len(self._sources))(*self._sources)
        self.api.check(
            self.api.vaCreateContext(
                self.display, self._vpp_config, coded_width, coded_height,
                api.VA_PROGRESSIVE, array, len(self._sources),
                ctypes.byref(self._vpp_context)),
            'vaCreateContext', 'VideoProc')

    # ------------------------------------------------------------- registering

    @as_encoder_error
    def register(self, texture: int, target: int | None = None) -> InputHandle:
        """Export `texture` as a DMA-BUF and import it as a VA surface.

        The texture keeps its contents and stays an ordinary OpenGL texture;
        what the encoder gets is a second name for the same memory. It must
        outlive the handle, and it must be ``GL_RGBA8`` of exactly the
        encoder's size.
        """
        from OpenGL.GL import GL_TEXTURE_2D

        self._check_open()
        width, height = self.size
        exported = dmabuf.export_texture(texture, width, height)
        try:
            surface = self._import_surface(exported)
        except Exception:
            exported.close()
            raise
        handle = InputHandle(texture=int(texture),
                             target=int(target or GL_TEXTURE_2D),
                             exported=exported, surface=surface)
        self._registered.append(handle)
        return handle

    def _import_surface(self, exported: dmabuf.ExportedImage) -> int:
        """Make a VA surface over an exported buffer, without copying it."""
        if len(exported.planes) != 1:
            raise EncoderError(
                f'the texture exported as {len(exported.planes)} planes; an '
                'encoder input here is a single-plane colour surface')
        if exported.fourcc != api.DRM_FORMAT_ABGR8888:
            raise EncoderError(
                f'the texture exported as {api.fourcc_name(exported.fourcc)}, '
                'not ABGR8888; the encoder reads a GL_RGBA8 texture')
        plane = exported.planes[0]
        descriptor = api.VADRMPRIMESurfaceDescriptor()
        descriptor.fourcc = api.VA_FOURCC_RGBA
        descriptor.width = exported.width
        descriptor.height = exported.height
        descriptor.num_objects = 1
        descriptor.objects[0].fd = plane.fd
        descriptor.objects[0].size = plane.offset + plane.stride * exported.height
        descriptor.objects[0].drm_format_modifier = exported.modifier
        descriptor.num_layers = 1
        descriptor.layers[0].drm_format = exported.fourcc
        descriptor.layers[0].num_planes = 1
        descriptor.layers[0].object_index[0] = 0
        descriptor.layers[0].offset[0] = plane.offset
        descriptor.layers[0].pitch[0] = plane.stride

        attributes = (api.VASurfaceAttrib * 2)()
        attributes[0].type = api.VASurfaceAttribMemoryType
        attributes[0].flags = api.VA_SURFACE_ATTRIB_SETTABLE
        attributes[0].value.type = api.VAGenericValueTypeInteger
        attributes[0].value.value.i = api.VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2
        attributes[1].type = api.VASurfaceAttribExternalBufferDescriptor
        attributes[1].flags = api.VA_SURFACE_ATTRIB_SETTABLE
        attributes[1].value.type = api.VAGenericValueTypePointer
        attributes[1].value.value.p = ctypes.cast(ctypes.byref(descriptor),
                                                  ctypes.c_void_p)
        surface = api.VASurfaceID(0)
        self.api.check(
            self.api.vaCreateSurfaces(
                self.display, api.VA_RT_FORMAT_RGB32, exported.width,
                exported.height, ctypes.byref(surface), 1, attributes, 2),
            'vaCreateSurfaces',
            f'importing a {exported.width}x{exported.height} DMA-BUF')
        return int(surface.value)

    # ---------------------------------------------------------------- encoding

    @as_encoder_error
    def encode(self, handle: InputHandle, timestamp: int, duration: int = 0,
               force_idr: bool = False) -> list[Packet]:
        """Submit the picture in `handle`, and return the one before it.

        The list is empty until the pipeline has filled, which takes one frame;
        after that it holds exactly one packet per call.
        """
        self._check_open()
        if handle not in self._registered:
            raise EncoderError(
                'this handle was not registered with this encoder, or the '
                'encoder has been closed since it was')
        self._wait_for_drawing(handle)
        source = self._sources[self._slot]
        self._slot = (self._slot + 1) % len(self._sources)
        self._convert(handle, source)
        picture = self.structure.next_picture(force_idr=force_idr)
        self._submit(picture, source, timestamp, duration)
        # Submitted first, then read: the picture whose packet comes back has
        # had this call's conversion and coding queued in front of it, so the
        # wait is on work that is a whole frame old.
        return self._drain(keep=PIPELINE_DEPTH)

    def _wait_for_drawing(self, handle: InputHandle) -> None:
        """Wait until OpenGL has finished drawing into the handle's texture.

        A caller that used :meth:`InputHandle.for_drawing` left a fence behind,
        and waiting on that costs nothing once it has been signalled. Without
        one -- a caller that drew its own way -- the whole pipeline is flushed,
        because there is nothing finer to wait on.

        The fence is used up here. It covers the drawing in the scope that made
        it and nothing after, so keeping it would mean a later frame waiting on
        a fence that signalled before that frame was drawn -- which returns at
        once and lets the conversion read a texture still being written. Falling
        back to flushing is slower and right.
        """
        from OpenGL.GL import (
            GL_SYNC_FLUSH_COMMANDS_BIT,
            GL_TIMEOUT_IGNORED,
            glClientWaitSync,
            glFinish,
        )
        if handle.fence is None:
            glFinish()
            return
        glClientWaitSync(handle.fence, GL_SYNC_FLUSH_COMMANDS_BIT,
                         GL_TIMEOUT_IGNORED)
        handle._forget_fence()

    def _convert(self, handle: InputHandle, source: int) -> None:
        """Convert the handle's RGB surface into `source`."""
        buffer = self._buffer(self._vpp_context,
                              api.VAProcPipelineParameterBufferType,
                              colour_pipeline(handle.surface))
        self._picture(self._vpp_context, source, [buffer], 'colour conversion')

    def _submit(self, picture: Picture, source: int, timestamp: int,
                duration: int) -> None:
        """Hand one picture to the encoder, with everything it has to be told."""
        coded = self._coded_buffers[self._coded_slot]
        self._coded_slot = (self._coded_slot + 1) % len(self._coded_buffers)
        reconstruction = self._reconstructions[self._recon_slot]
        self._recon_slot = (self._recon_slot + 1) % len(self._reconstructions)

        buffers: list[int] = []
        if picture.idr:
            buffers.append(self._sequence_buffer())
            buffers.extend(self._rate_control_buffers())
            buffers.extend(self._parameter_set_buffers())
        buffers.append(self._picture_buffer(picture, coded, reconstruction))
        buffers.extend(self._slice_header_buffers(picture))
        buffers.append(self._slice_buffer(picture))

        self._picture(self._context, source, buffers,
                      f'picture {picture.frame_num}')
        self._remember(picture, coded, source, reconstruction, timestamp,
                       duration)

    def _remember(self, picture: Picture, coded: int, source: int,
                  reconstruction: int, timestamp: int, duration: int) -> None:
        """Record a submitted picture, and what it reconstructed into."""
        self._references = [(picture.frame_num, reconstruction, picture.poc)] + [
            entry for entry in getattr(self, '_references', [])
            if entry[0] in picture.references
        ][:self.max_num_ref_frames - 1]
        if picture.idr:
            self._references = [(picture.frame_num, reconstruction, picture.poc)]
        self._pending.append(_Submission(
            coded_buffer=coded, source=source, timestamp=int(timestamp),
            duration=int(duration), keyframe=picture.idr))

    def _picture(self, context: Any, target: int, buffers: list[int],
                 what: str) -> None:
        """Run one begin/render/end sequence, and give the buffers back after.

        ``vaRenderPicture`` queues buffers and ``vaEndPicture`` is what submits
        the picture, so a buffer is held until the sequence closes. Mesa reads
        each one as it is queued and would not notice an earlier release, but
        the interface does not promise that of every driver, and the cost of
        holding them one more call is nothing.
        """
        array = (api.VABufferID * len(buffers))(*buffers)
        try:
            self.api.check(
                self.api.vaBeginPicture(self.display, context, target),
                'vaBeginPicture', what)
            self.api.check(
                self.api.vaRenderPicture(self.display, context, array,
                                         len(buffers)),
                'vaRenderPicture', f'{len(buffers)} buffer(s) for {what}')
            self.api.check(self.api.vaEndPicture(self.display, context),
                           'vaEndPicture', what)
        finally:
            for buffer in buffers:
                self.api.vaDestroyBuffer(self.display, buffer)

    def _buffer(self, context: Any, kind: int, payload: Any) -> int:
        """Create a VA buffer holding `payload`."""
        buffer = api.VABufferID(0)
        self.api.check(
            self.api.vaCreateBuffer(self.display, context, kind,
                                    ctypes.sizeof(payload), 1,
                                    ctypes.byref(payload),
                                    ctypes.byref(buffer)),
            'vaCreateBuffer', type(payload).__name__)
        return int(buffer.value)

    def _sequence_buffer(self) -> int:
        """The sequence parameters, which say what the whole stream is."""
        sets = self.sets
        parameters = api.VAEncSequenceParameterBufferH264()
        parameters.seq_parameter_set_id = 0
        parameters.level_idc = sets.level_idc
        parameters.intra_period = self.gop
        parameters.intra_idr_period = self.gop
        parameters.ip_period = 1
        parameters.bits_per_second = self.bitrate
        parameters.max_num_ref_frames = self.max_num_ref_frames
        parameters.picture_width_in_mbs = sets.width_in_mbs
        parameters.picture_height_in_mbs = sets.height_in_mbs
        bits = parameters.seq_fields.bits
        bits.chroma_format_idc = 1
        bits.frame_mbs_only_flag = 1
        bits.direct_8x8_inference_flag = 1
        bits.log2_max_frame_num_minus4 = sets.log2_max_frame_num - 4
        bits.pic_order_cnt_type = 0
        bits.log2_max_pic_order_cnt_lsb_minus4 = sets.log2_max_poc_lsb - 4

        coded_width, coded_height = sets.coded_size
        right = (coded_width - sets.width) // 2
        bottom = (coded_height - sets.height) // 2
        parameters.frame_cropping_flag = 1 if (right or bottom) else 0
        parameters.frame_crop_right_offset = right
        parameters.frame_crop_bottom_offset = bottom

        parameters.vui_parameters_present_flag = 1
        vui = parameters.vui_fields.bits
        vui.aspect_ratio_info_present_flag = 1
        vui.timing_info_present_flag = 1
        vui.bitstream_restriction_flag = 1
        vui.motion_vectors_over_pic_boundaries_flag = 1
        vui.log2_max_mv_length_horizontal = 15
        vui.log2_max_mv_length_vertical = 15
        vui.fixed_frame_rate_flag = 1
        parameters.aspect_ratio_idc = 1
        parameters.num_units_in_tick = self.frame_rate[1]
        parameters.time_scale = self.frame_rate[0] * 2
        return self._buffer(self._context, api.VAEncSequenceParameterBufferType,
                            parameters)

    def _rate_control_buffers(self) -> list[int]:
        """Bitrate, frame rate and buffering, as miscellaneous parameters."""
        control = api.VAEncMiscParameterRateControl()
        control.bits_per_second = self.bitrate
        control.target_percentage = (
            100 if self.rate_control == api.VA_RC_CBR else VBR_TARGET_PERCENTAGE)
        control.window_size = 1000
        control.initial_qp = self.qp
        control.min_qp = 0
        control.max_qp = 51
        rate = api.VAEncMiscParameterFrameRate()
        # The numerator goes in the low half and the denominator above it; a
        # denominator of zero means the numerator is whole frames a second.
        rate.framerate = (self.frame_rate[0]
                          | (self.frame_rate[1] << 16 if self.frame_rate[1] > 1
                             else 0))
        hrd = api.VAEncMiscParameterHRD()
        hrd.buffer_size = self.bitrate * HRD_BUFFER_SECONDS
        hrd.initial_buffer_fullness = hrd.buffer_size // 2
        return [
            self._buffer(self._context, api.VAEncMiscParameterBufferType,
                         api.misc_parameter(kind, payload))
            for kind, payload in (
                (api.VAEncMiscParameterTypeRateControl, control),
                (api.VAEncMiscParameterTypeFrameRate, rate),
                (api.VAEncMiscParameterTypeHRD, hrd),
            )
        ]

    def _packed_header(self, kind: int, data: bytes, bits: int) -> list[int]:
        """One packed header: the parameters describing it, then its bytes.

        The two go together and in that order; a driver reads the description
        and takes the next data buffer as what it described.
        """
        parameters = api.VAEncPackedHeaderParameterBuffer()
        parameters.type = kind
        parameters.bit_length = bits
        # The bytes already carry the escapes that keep a start code out of
        # them, so the driver must not insert its own.
        parameters.has_emulation_bytes = 1
        payload = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        return [
            self._buffer(self._context,
                         api.VAEncPackedHeaderParameterBufferType, parameters),
            self._buffer(self._context, api.VAEncPackedHeaderDataBufferType,
                         payload),
        ]

    def _parameter_set_buffers(self) -> list[int]:
        """Both parameter sets, as the one sequence header a driver takes."""
        if not self.packed_headers & api.VA_ENC_PACKED_HEADER_SEQUENCE:
            return []
        data, bits = self.sets.packed_parameter_sets()
        return self._packed_header(api.VAEncPackedHeaderSequence, data, bits)

    def _slice_header_buffers(self, picture: Picture) -> list[int]:
        """The NAL header and slice header for one picture."""
        if not self.packed_headers & api.VA_ENC_PACKED_HEADER_SLICE:
            return []
        data, bits = self.sets.packed_slice_header(picture)
        return self._packed_header(api.VAEncPackedHeaderSlice, data, bits)

    def _picture_buffer(self, picture: Picture, coded: int,
                        reconstruction: int) -> int:
        """What the driver is told about the picture it is about to code."""
        parameters = api.VAEncPictureParameterBufferH264()
        parameters.CurrPic.picture_id = reconstruction
        parameters.CurrPic.frame_idx = picture.frame_num
        parameters.CurrPic.flags = 0
        parameters.CurrPic.TopFieldOrderCnt = picture.poc
        parameters.CurrPic.BottomFieldOrderCnt = picture.poc
        for index in range(16):
            parameters.ReferenceFrames[index] = api.VAPictureH264.unused()
        for index, entry in enumerate(self._reference_pictures(picture)):
            parameters.ReferenceFrames[index] = entry
        parameters.coded_buf = coded
        parameters.pic_parameter_set_id = 0
        parameters.seq_parameter_set_id = 0
        parameters.last_picture = 0
        parameters.frame_num = picture.frame_num
        parameters.pic_init_qp = self.qp
        parameters.num_ref_idx_l0_active_minus1 = 0
        parameters.num_ref_idx_l1_active_minus1 = 0
        bits = parameters.pic_fields.bits
        bits.idr_pic_flag = 1 if picture.idr else 0
        bits.reference_pic_flag = 1 if picture.reference else 0
        bits.entropy_coding_mode_flag = 1
        bits.transform_8x8_mode_flag = 1
        bits.deblocking_filter_control_present_flag = 1
        return self._buffer(self._context, api.VAEncPictureParameterBufferType,
                            parameters)

    def _reference_pictures(self, picture: Picture) -> list[api.VAPictureH264]:
        """The decoded picture buffer as the driver is given it."""
        by_frame_num = {frame_num: (surface, poc)
                        for frame_num, surface, poc in getattr(
                            self, '_references', [])}
        entries = []
        for frame_num in picture.references:
            found = by_frame_num.get(frame_num)
            if found is None:
                continue
            surface, poc = found
            entries.append(api.VAPictureH264(
                picture_id=surface, frame_idx=frame_num,
                flags=api.VA_PICTURE_H264_SHORT_TERM_REFERENCE,
                TopFieldOrderCnt=poc, BottomFieldOrderCnt=poc))
        return entries

    def _slice_buffer(self, picture: Picture) -> int:
        """The one slice each picture is coded as, and its reference list."""
        sets = self.sets
        parameters = api.VAEncSliceParameterBufferH264()
        parameters.macroblock_address = 0
        parameters.num_macroblocks = sets.width_in_mbs * sets.height_in_mbs
        parameters.slice_type = SLICE_TYPE_I if picture.intra else SLICE_TYPE_P
        parameters.pic_parameter_set_id = 0
        parameters.idr_pic_id = picture.idr_pic_id
        parameters.pic_order_cnt_lsb = picture.poc % (1 << sets.log2_max_poc_lsb)
        parameters.num_ref_idx_active_override_flag = 1
        parameters.num_ref_idx_l0_active_minus1 = 0
        parameters.num_ref_idx_l1_active_minus1 = 0
        for index in range(32):
            parameters.RefPicList0[index] = api.VAPictureH264.unused()
            parameters.RefPicList1[index] = api.VAPictureH264.unused()
        for index, entry in enumerate(self._reference_pictures(picture)):
            parameters.RefPicList0[index] = entry
        parameters.slice_qp_delta = 0
        parameters.disable_deblocking_filter_idc = 0
        return self._buffer(self._context, api.VAEncSliceParameterBufferType,
                            parameters)

    # ---------------------------------------------------------------- reading

    def _drain(self, keep: int) -> list[Packet]:
        """Read back every submission but the newest `keep`."""
        packets = []
        while len(self._pending) > keep:
            packets.append(self._read(self._pending.pop(0)))
        return packets

    def _read(self, submission: _Submission) -> Packet:
        """Wait for one picture to be coded and take its bytes."""
        self.api.check(
            self.api.vaSyncSurface(self.display, submission.source),
            'vaSyncSurface', 'waiting for a coded picture')
        pointer = ctypes.c_void_p()
        self.api.check(
            self.api.vaMapBuffer(self.display, submission.coded_buffer,
                                 ctypes.byref(pointer)),
            'vaMapBuffer', 'reading a coded picture')
        try:
            data = self._collect(pointer)
        finally:
            self.api.check(
                self.api.vaUnmapBuffer(self.display, submission.coded_buffer),
                'vaUnmapBuffer', 'reading a coded picture')
        if submission.keyframe:
            self._remember_parameter_sets(data)
        return Packet(data=data, timestamp=submission.timestamp,
                      duration=submission.duration,
                      keyframe=submission.keyframe)

    def _collect(self, pointer: ctypes.c_void_p) -> bytes:
        """Follow a coded buffer's segments and join what they hold.

        An encoder may answer in more than one run of bytes, and the chain is
        how it says so.
        """
        segments = []
        address = pointer.value
        while address:
            segment = api.VACodedBufferSegment.from_address(address)
            if segment.size and segment.buf:
                segments.append(ctypes.string_at(segment.buf, segment.size))
            quantiser = segment.status & api.VA_CODED_BUF_STATUS_PICTURE_AVE_QP_MASK
            log.debug('coded segment: %d bytes, average quantiser %d',
                      segment.size, quantiser)
            address = ctypes.cast(segment.next, ctypes.c_void_p).value
        return b''.join(segments)

    def _remember_parameter_sets(self, data: bytes) -> None:
        """Keep the parameter sets a coded picture turned out to carry.

        A driver may amend what it was handed -- a capability the hardware does
        not have is one the picture parameter set must not claim -- so the sets
        in the stream are the ones that describe it, and those are what
        :meth:`headers` reports.
        """
        from pyopengl_video.mp4 import split_annexb

        for unit in split_annexb(data):
            kind = unit[0] & 0x1F
            if kind in (7, 8):
                self._coded_sets[kind] = unit

    @as_encoder_error
    def flush(self) -> list[Packet]:
        """Read back every picture still inside the encoder."""
        self._check_open()
        return self._drain(keep=0)

    def headers(self) -> bytes:
        """The parameter sets, as an Annex-B fragment.

        Before the first picture is coded these are the sets this backend
        built. Afterwards they are the ones the stream carries, which is what a
        container has to describe the stream with.
        """
        if len(self._coded_sets) == 2:
            return annexb(self._coded_sets[7], self._coded_sets[8])
        return self.sets.headers()

    # ----------------------------------------------------------------- closing

    def close(self) -> None:
        """Release everything, in the order that leaves nothing dangling."""
        if self._closed:
            return
        self._closed = True
        if self.display is not None:
            for handle in self._registered:
                self._release_handle(handle)
            for buffer in self._coded_buffers:
                self.api.vaDestroyBuffer(self.display, buffer)
            for context, config in ((self._vpp_context, self._vpp_config),
                                    (self._context, self._config)):
                if context:
                    self.api.vaDestroyContext(self.display, context)
                if config:
                    self.api.vaDestroyConfig(self.display, config)
            self._destroy_surfaces(self._sources + self._reconstructions)
        self._registered = []
        self._coded_buffers = []
        self._sources = []
        self._reconstructions = []
        self._pending = []
        self._close_display()

    def _release_handle(self, handle: InputHandle) -> None:
        """Give back the surface and the buffer behind one registered input."""
        if handle.surface:
            surface = api.VASurfaceID(handle.surface)
            self.api.vaDestroySurfaces(self.display, ctypes.byref(surface), 1)
            handle.surface = 0
        if handle.exported is not None:
            handle.exported.close()
            handle.exported = None

    def _destroy_surfaces(self, surfaces: list[int]) -> None:
        if not surfaces:
            return
        array = (api.VASurfaceID * len(surfaces))(*surfaces)
        self.api.vaDestroySurfaces(self.display, array, len(surfaces))

    def _close_display(self) -> None:
        """Terminate libva and close the render node, once."""
        if self.display is not None:
            self.api.vaTerminate(self.display)
            self.display = None
        fd = getattr(self, '_device_fd', -1)
        if fd >= 0:
            os.close(fd)
            self._device_fd = -1

    def _check_open(self) -> None:
        if self._closed:
            raise EncoderError('this encoder has been closed')
