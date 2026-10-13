"""The VA-API encoder: an OpenGL texture in, a compressed elementary stream out.

The frame never leaves the GPU. A texture is exported as a DMA-BUF and imported
as a VA surface, the driver's video processing block converts it from RGB to
NV12, and the video engine codes that. What crosses the bus is the compressed
result.

libva leaves the codec decisions to the caller, so this encoder is paired with
a control layer (:mod:`pyopengl_video.vaapi.control`), one for each codec in
:data:`pyopengl_video.vaapi.codecs.ENCODERS`, which decides what each picture
is and writes the headers the driver puts in the stream. Everything here is the
plumbing around it, the same for every codec.

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
from collections.abc import Callable
from typing import Any

from OpenGL.GL import (
    GL_SYNC_FLUSH_COMMANDS_BIT,
    GL_SYNC_GPU_COMMANDS_COMPLETE,
    GL_TEXTURE_2D,
    GL_TIMEOUT_IGNORED,
    glClientWaitSync,
    glDeleteSync,
    glFenceSync,
    glFinish,
)

from pyopengl_video import bounds, inputs
from pyopengl_video.encoder import (
    Encoder,
    EncoderError,
    Packet,
    default_bitrate,
    frame_rate_ratio,
)
from pyopengl_video.linux import dmabuf
from pyopengl_video.teardown import release_all
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.codecs import ENCODERS, support_for
from pyopengl_video.vaapi.control import (
    LARGEST_BITRATE,
    LARGEST_SIDE,
    QP_RANGE,
    Buffer,
    CodedPicture,
    EncodeControl,
    EncodeSettings,
)
from pyopengl_video.vaapi.device import Device, abandon, worded
from pyopengl_video.vaapi.h264 import COLOUR_PRIMARIES_BT709, MATRIX_BT709, TRANSFER_BT709

log = logging.getLogger(__name__)

__all__ = ['VAAPIEncoder', 'InputHandle', 'RATE_CONTROL_MODES',
           'MAX_CODED_SEGMENTS', 'MAX_REFERENCES', 'PIPELINE_DEPTH',
           'colour_pipeline', 'as_encoder_error']

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

#: The most reconstructed pictures a control layer may keep to predict from:
#: the largest decoded picture buffer of any codec here, H.264's and HEVC's 16.
MAX_REFERENCES = 16

#: The most segments a coded buffer is followed through. A driver answers
#: with one segment a picture, or one a slice; a chain longer than this is
#: one that does not end.
MAX_CODED_SEGMENTS = 1024

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

    Every other :class:`Exception` a setting or a driver brings about is
    translated the same way, :class:`MemoryError` included.
    :class:`KeyboardInterrupt`, :class:`SystemExit` and :class:`GeneratorExit`
    pass through.
    """
    @functools.wraps(method)
    def translating(*arguments: Any, **named: Any) -> Any:
        try:
            return method(*arguments, **named)
        except EncoderError:
            raise
        except Exception as error:
            raise EncoderError(worded(error)) from error
    return translating


def colour_pipeline(surface: int, size: tuple[int, int] | None = None,
                    ) -> api.VAProcPipelineParameterBuffer:
    """The video processing pass that turns a full-range RGB surface into NV12.

    size -- the picture, as (width, height), each from 1 to 65535, which is
        what the rectangle's 16-bit fields hold. The pass writes that
        rectangle at the top left of its target and nothing else, so a target
        padded out to a codec's block size holds the picture unscaled.
        Without it the picture is scaled to fill the target.

    A renderer's colours are full range, and the stream every control layer
    here produces declares limited-range BT.709 -- so the conversion has to say
    the same thing. The two are set in different places and nothing but a picture coming
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
    if size is not None:
        width, height = bounds.frame_size(
            *size, most=(bounds.FIELD_16, bounds.FIELD_16), error=EncoderError)
        # The pipeline keeps the rectangle alive: ctypes holds what a pointer
        # field was assigned from for as long as the structure lives.
        pipeline.output_region = ctypes.pointer(api.VARectangle(0, 0, width, height))
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

    def for_drawing(self) -> _Drawing:
        """Hold the texture for OpenGL, and note when the drawing is done.

        The fence made at the end of this scope is what the encoder waits on
        before the conversion reads the texture. Waiting on a fence rather than
        finishing the whole pipeline leaves the driver free to keep working on
        everything else that was queued.
        """
        return _Drawing(self)

    def _held(self) -> list[Callable[[], object]]:
        """A release for each OpenGL object the handle still holds, the fence first."""
        fence = [self.forget_fence] if self.fence is not None else []
        return [*fence, *super()._held()]

    def forget_fence(self) -> None:
        """Delete the fence made when drawing last finished, if there is one."""
        if self.fence is not None:
            fence, self.fence = self.fence, None
            glDeleteSync(fence)


class _Drawing:
    """The scope :meth:`InputHandle.for_drawing` opens."""

    def __init__(self, handle: InputHandle) -> None:
        self.handle = handle

    def __enter__(self) -> InputHandle:
        self.handle.forget_fence()
        return self.handle

    def __exit__(self, *exception: object) -> None:
        self.handle.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)


@dataclasses.dataclass
class _Submission:
    """One picture handed to the encoder and not yet read back."""

    picture: CodedPicture
    coded_buffer: int
    source: int
    timestamp: int
    duration: int


class VAAPIEncoder(Encoder):
    """Encoding on Intel and AMD hardware, fed from an OpenGL texture.

    width/height -- the picture size, both even and at most
        :data:`~pyopengl_video.vaapi.control.LARGEST_SIDE`
    codec -- any codec in :data:`pyopengl_video.vaapi.codecs.ENCODERS` that the
        device offers; :func:`pyopengl_video.encoders` lists them for this machine
    fps -- frame rate, as a number or an exact ``(numerator, denominator)``.
        The ratio in lowest terms has to fit 16 bits each way, which is how
        libva carries it: 30000/1001 does, and so does 90000/3003.
    bitrate -- bits per second, up to 2**31 - 1; the default is derived from
        the frame size and rate
    gop -- frames between key frames, up to 2**32 - 1; the default is one
        second of them
    rate_control -- ``'cbr'``, ``'vbr'`` or ``'cqp'``; a mode the driver does
        not offer is refused by name
    qp -- the quantiser, which is what ``'cqp'`` codes at and what the other
        modes start from. It is on H.264's scale of 0 to 51 for every codec,
        so one setting gives a similar picture in each.
    bframes -- must be zero: pictures are coded in display order here
    device -- the DRM render node to open, such as ``/dev/dri/renderD128``;
        the default is the first that has an encoder for the codec

    The picture is coded as 8-bit 4:2:0, limited-range BT.709.

    Every setting is checked before a device is opened, and
    :class:`~pyopengl_video.encoder.EncoderError` names the one that does not
    fit. An :meth:`encode` that fails part-way stops the encoder: the picture
    was not coded, so the ones after it would predict from a picture the
    stream does not hold. :meth:`flush` still returns the packets of the
    pictures before it, and :meth:`close` releases everything.
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
        support = support_for(ENCODERS, codec)
        if support is None:
            raise EncoderError(
                f'the VA-API backend encodes {", ".join(sorted(ENCODERS))}, '
                f'not {codec!r}')
        self._closed = False
        self._stopped = ''
        self._device: Device | None = None
        self._config = api.VAConfigID(0)
        self._context = api.VAContextID(0)
        self._vpp_config = api.VAConfigID(0)
        self._vpp_context = api.VAContextID(0)
        self._sources: list[int] = []
        self._reconstructions: list[int] = []
        self._coded_buffers: list[int] = []
        self._coded_size = 0
        self._registered: list[InputHandle] = []
        self._pending: list[_Submission] = []
        self._slot = 0
        self._recon_slot = 0
        self._coded_slot = 0
        self._conversion_kept = False

        if bounds.whole(bframes, 'bframes', 0, bounds.FIELD_16, EncoderError):
            raise EncoderError(
                'the VA-API backend codes pictures in display order, so '
                f'bframes must be 0, not {bframes}; a stream with B-frames '
                'needs the reordering control layer this backend does not have')
        if rate_control not in RATE_CONTROL_MODES:
            raise EncoderError(
                f'{rate_control!r} is not a rate control mode here; '
                f'choose one of {sorted(RATE_CONTROL_MODES)}')

        self.codec = support.codec
        self._support = support
        self.size = bounds.frame_size(width, height, most=(LARGEST_SIDE, LARGEST_SIDE),
                                      even=True, error=EncoderError)
        self.frame_rate = bounds.frame_rate(frame_rate_ratio(fps), EncoderError)
        self.bitrate = bounds.whole(
            bitrate if bitrate else min(
                LARGEST_BITRATE, max(1, default_bitrate(self.size, self.frame_rate))),
            'bitrate', 1, LARGEST_BITRATE, EncoderError)
        self.gop = bounds.whole(
            gop if gop else min(bounds.FIELD_32, max(1, round(
                self.frame_rate[0] / self.frame_rate[1]))),
            'gop', 1, bounds.FIELD_32, EncoderError)
        self.rate_control_name = rate_control
        self.qp = bounds.whole(qp, 'qp', *QP_RANGE, EncoderError)
        # Made here, before a device is opened, since making it is what holds
        # each setting to the field a control layer writes it into.
        self._settings = EncodeSettings(
            size=self.size, frame_rate=self.frame_rate, bitrate=self.bitrate,
            gop=self.gop, qp=self.qp, rate_control=RATE_CONTROL_MODES[rate_control])

        self.api = api.VA.instance()
        self.device = device

        try:
            self._open()
        except BaseException:
            abandon(self.close)
            raise

    def __repr__(self) -> str:
        width, height = self.size
        return (f'<{type(self).__name__} {width}x{height} {self.codec} '
                f'{self.rate_control_name} {self.bitrate} bit/s '
                f'on {self.device}>')

    # ---------------------------------------------------------------- setup

    def _open(self) -> None:
        """Open the device, negotiate what it can do, and build the contexts."""
        self._open_display()
        self._negotiate()
        self._create_encode_context()
        self._create_vpp_context()

    @property
    def display(self) -> Any:
        """The libva display the session runs on, or None once closed."""
        return None if self._device is None else self._device.display

    def _open_display(self) -> None:
        """Open the first DRM render node with an encoder for the codec."""
        support = self._support
        nodes = [self.device] if self.device else api.render_nodes()
        self._device = Device.first(nodes, support.objection,
                                    f'{support.title} encoding')
        self._control_class: type[EncodeControl] = support.load()
        self.device = self._device.node
        self.vendor = self._device.vendor
        self.version = self._device.version

    def _attributes(self, *types: int) -> dict[int, int]:
        """What the codec's encode entrypoint says about each attribute.

        An attribute the driver will not answer comes back as
        ``VA_ATTRIB_NOT_SUPPORTED`` rather than as an error, so it is dropped
        here and the caller sees only what was actually stated.
        """
        attributes = (api.VAConfigAttrib * len(types))()
        for index, kind in enumerate(types):
            attributes[index].type = kind
        self.api.check(
            self.api.vaGetConfigAttributes(
                self.display, self._control_class.profile, api.VAEntrypointEncSlice,
                attributes, len(types)),
            'vaGetConfigAttributes', f'{self._control_class.title}/EncSlice')
        return {entry.type: entry.value for entry in attributes
                if entry.value != api.VA_ATTRIB_NOT_SUPPORTED}

    def _negotiate(self) -> None:
        """Settle what this driver will actually do, and refuse what it will not."""
        layer = self._control_class
        stated = self._attributes(
            api.VAConfigAttribRTFormat, api.VAConfigAttribRateControl,
            *layer.attributes)

        formats = stated.get(api.VAConfigAttribRTFormat, 0)
        if not formats & api.VA_RT_FORMAT_YUV420:
            raise EncoderError(
                f'{self.device} encodes {layer.title} but not from 4:2:0 surfaces '
                f'(it offers format bits {formats:#x}), which is the only '
                'sampling this backend produces')

        modes = stated.get(api.VAConfigAttribRateControl, 0)
        wanted = RATE_CONTROL_MODES[self.rate_control_name]
        if not modes & wanted:
            offered = sorted(name for name, bit in RATE_CONTROL_MODES.items()
                             if modes & bit)
            raise EncoderError(
                f'{self.device} does not offer {self.rate_control_name} rate '
                f'control for {layer.title}; it offers '
                f'{offered or ["none this backend knows"]}')
        self.rate_control = wanted

        self.control = layer(self._settings, stated)
        bounds.whole(self.control.references, f'the pictures {layer.title} keeps to '
                     'predict from', 1, MAX_REFERENCES, EncoderError)
        bounds.frame_size(*self.control.coded_size, most=(bounds.FIELD_16, bounds.FIELD_16),
                          error=EncoderError)
        self.packed_headers = self.control.packed_headers
        self.input_slots = PIPELINE_DEPTH + 1
        log.debug('%s: %s, %s, references %d, packed headers %#x',
                  self.device, self.vendor, layer.title, self.control.references,
                  self.packed_headers)

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
                self.display, self.control.profile, api.VAEntrypointEncSlice,
                array, len(attributes), ctypes.byref(self._config)),
            'vaCreateConfig', f'{self.control.title}/EncSlice')

        coded_width, coded_height = self.control.coded_size
        # One source per picture in flight, and enough reconstructions that the
        # picture being coded never lands on one still being referenced.
        self._sources = self._create_surfaces(
            api.VA_RT_FORMAT_YUV420, coded_width, coded_height,
            PIPELINE_DEPTH + 1)
        self._reconstructions = self._create_surfaces(
            api.VA_RT_FORMAT_YUV420, coded_width, coded_height,
            self.control.references + 1 + PIPELINE_DEPTH)

        targets = self._sources + self._reconstructions
        surfaces = (api.VASurfaceID * len(targets))(*targets)
        self.api.check(
            self.api.vaCreateContext(
                self.display, self._config, coded_width, coded_height,
                api.VA_PROGRESSIVE, surfaces, len(targets),
                ctypes.byref(self._context)),
            'vaCreateContext', f'{coded_width}x{coded_height} encode')

        # Kept: a driver's segment lengths are compared with it when a
        # picture is read back.
        self._coded_size = size = self._coded_buffer_size()
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
        width, height = self.control.coded_size
        return bounds.whole(max(width * height * 3 // 2, 1 << 16),
                            f'the coded buffer of a {width}x{height} picture, in bytes',
                            1, (1 << 31) - 1, EncoderError)

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
        coded_width, coded_height = self.control.coded_size
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
        encoder's size: OpenGL is asked, and a texture of another size or
        format is refused.
        """
        self._check_open()
        width, height = self.size
        exported = dmabuf.export_texture(texture, width, height)
        try:
            surface = self._import_surface(exported)
        except BaseException:
            abandon(exported.close)
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
        # The layout is EGL's answer, and each number goes into a 32-bit field
        # the driver sizes its reads by.
        offset = bounds.whole(plane.offset, "the exported plane's offset", 0,
                              bounds.FIELD_32, EncoderError)
        stride = bounds.whole(plane.stride, "the exported plane's stride",
                              exported.width * 4, bounds.FIELD_32, EncoderError)
        descriptor = api.VADRMPRIMESurfaceDescriptor()
        descriptor.fourcc = api.VA_FOURCC_RGBA
        descriptor.width = exported.width
        descriptor.height = exported.height
        descriptor.num_objects = 1
        descriptor.objects[0].fd = plane.fd
        descriptor.objects[0].size = bounds.whole(
            offset + stride * exported.height, 'the exported buffer, in bytes', 1,
            bounds.FIELD_32, EncoderError)
        descriptor.objects[0].drm_format_modifier = exported.modifier
        descriptor.num_layers = 1
        descriptor.layers[0].drm_format = exported.fourcc
        descriptor.layers[0].num_planes = 1
        descriptor.layers[0].object_index[0] = 0
        descriptor.layers[0].offset[0] = offset
        descriptor.layers[0].pitch[0] = stride

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

        A call that fails once the picture has been handed to the driver
        stops the encoder, and every later call is refused: the control
        layer has counted a picture the stream does not hold. :meth:`flush`
        returns the pictures coded before it.
        """
        self._check_open()
        if self._stopped:
            raise EncoderError(
                f'the encoder has stopped: {self._stopped}; flush it for the '
                'pictures already coded, and close it')
        duration = self.frame_duration(duration)
        if handle not in self._registered:
            raise EncoderError(
                'this handle was not registered with this encoder, or the '
                'encoder has been closed since it was')
        self._wait_for_drawing(handle)
        try:
            source = self._sources[self._slot]
            self._convert(handle, source)
            if not self._conversion_kept:
                self._keep_conversion(handle, source)
            picture = self.control.next_picture(force_key=force_idr)
            self._submit(picture, source, timestamp, duration)
            # Advanced once the picture is in flight: until then the source
            # is not one a later picture has to keep clear of.
            self._slot = (self._slot + 1) % len(self._sources)
            # Submitted first, then read: the picture whose packet comes back
            # has had this call's conversion and coding queued in front of it,
            # so the wait is on work that is a whole frame old.
            return self._drain(keep=PIPELINE_DEPTH)
        except BaseException as error:
            self._stopped = worded(error)
            raise

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
        if handle.fence is None:
            glFinish()
            return
        glClientWaitSync(handle.fence, GL_SYNC_FLUSH_COMMANDS_BIT,
                         GL_TIMEOUT_IGNORED)
        handle.forget_fence()

    def _convert(self, handle: InputHandle, source: int) -> None:
        """Convert the handle's RGB surface into `source`."""
        # The driver follows the pipeline's pointer to its output rectangle
        # when the picture is submitted, so the pipeline is held until then.
        pipeline = colour_pipeline(handle.surface, self.size)
        buffer = self._buffer(self._vpp_context,
                              api.VAProcPipelineParameterBufferType, pipeline)
        self._picture(self._vpp_context, source, [buffer], 'colour conversion')

    def _keep_conversion(self, handle: InputHandle, source: int) -> None:
        """Convert the first picture a second time, before anything is coded.

        Mesa takes a conversion followed at once by its encode as one the
        encoder's own converter can do in its place. It then codes from the
        RGB surface, and from the seventeenth picture it skips the conversion
        asked for here, whose ``vaEndPicture`` fails on a device that also has
        a video processing engine. Two conversions with no encode between
        them tell Mesa the pattern does not hold, and it leaves every
        conversion of the session to this pass. The pictures coded are then
        the ones :func:`colour_pipeline` describes, on every driver.
        """
        self._convert(handle, source)
        self._conversion_kept = True

    def _submit(self, picture: CodedPicture, source: int, timestamp: int,
                duration: int) -> None:
        """Hand one picture to the encoder, with everything it has to be told.

        The coded buffer and the reconstruction are the next of each, and the
        counters that choose them advance once the driver has taken the
        picture, so a refused picture leaves both free.
        """
        coded = self._coded_buffers[self._coded_slot]
        reconstruction = self._reconstructions[self._recon_slot]

        described = self.control.buffers(picture, coded, reconstruction)
        self._picture(self._context, source, self._buffers(self._context, described),
                      picture.name)
        self._coded_slot = (self._coded_slot + 1) % len(self._coded_buffers)
        self._recon_slot = (self._recon_slot + 1) % len(self._reconstructions)
        self._pending.append(_Submission(
            picture=picture, coded_buffer=coded, source=source,
            timestamp=int(timestamp), duration=int(duration)))

    def _picture(self, context: Any, target: int, buffers: list[int],
                 what: str) -> None:
        """Run one begin/render/end sequence, and give the buffers back after."""
        self._opened().picture(context, target, [buffers], what)

    def _buffer(self, context: Any, kind: int, payload: Any) -> int:
        """Create a VA buffer holding `payload`."""
        return self._opened().buffer(context, kind, payload)

    def _opened(self) -> Device:
        """The device, where the encoder still has one."""
        if self._device is None:
            raise EncoderError('this encoder has been closed')
        return self._device

    def _buffers(self, context: Any, described: list[Buffer]) -> list[int]:
        """Create a VA buffer for each of `described`, or none if one fails."""
        made: list[int] = []
        try:
            for kind, payload in described:
                made.append(self._buffer(context, kind, payload))
        except BaseException:
            for buffer in made:
                self.api.vaDestroyBuffer(self.display, buffer)
            raise
        return made

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
        picture = submission.picture
        return Packet(data=self.control.coded(picture, data),
                      timestamp=submission.timestamp,
                      duration=submission.duration, keyframe=picture.keyframe)

    def _collect(self, pointer: ctypes.c_void_p) -> bytes:
        """Follow a coded buffer's segments and join what they hold.

        An encoder may answer in more than one run of bytes, and the chain is
        how it says so. The lengths are the driver's, so each segment and
        their sum are held to the size the coded buffer was created at, and
        the chain to :data:`MAX_CODED_SEGMENTS`.
        :class:`~pyopengl_video.encoder.EncoderError` for a length outside
        those, and for a segment whose status says the picture overflowed the
        buffer or the bitstream is bad: what such a buffer holds is not a
        picture.
        """
        address = pointer.value
        if not address:
            raise EncoderError('vaMapBuffer gave no address for the coded picture')
        segments = []
        total = 0
        for _ in range(MAX_CODED_SEGMENTS):
            segment = api.VACodedBufferSegment.from_address(address)
            status = int(segment.status)
            if status & api.VA_CODED_BUF_STATUS_SLICE_OVERFLOW_MASK:
                raise EncoderError(
                    f'the coded picture overflowed its {self._coded_size} byte '
                    f'buffer (segment status {status:#x})')
            if status & api.VA_CODED_BUF_STATUS_BAD_BITSTREAM:
                raise EncoderError(
                    f'the driver marks the coded picture as a bad bitstream '
                    f'(segment status {status:#x})')
            _, size = bounds.span(total, segment.size, self._coded_size,
                                  'a segment of the coded picture', EncoderError)
            if size:
                if not segment.buf:
                    raise EncoderError(
                        f'a segment of the coded picture states {size} bytes '
                        'at no address')
                segments.append(ctypes.string_at(segment.buf, size))
                total += size
            log.debug('coded segment: %d bytes, average quantiser %d', size,
                      status & api.VA_CODED_BUF_STATUS_PICTURE_AVE_QP_MASK)
            address = ctypes.cast(segment.next, ctypes.c_void_p).value
            if not address:
                return b''.join(segments)
        raise EncoderError(
            f'the coded picture has more than {MAX_CODED_SEGMENTS} segments')

    @as_encoder_error
    def flush(self) -> list[Packet]:
        """Read back every picture still inside the encoder."""
        self._check_open()
        return self._drain(keep=0)

    @as_encoder_error
    def headers(self) -> bytes:
        """The stream's configuration, as a container stores it up front.

        Parameter sets as an Annex-B fragment for H.264 and HEVC, the sequence
        header OBU for AV1. Once a key picture has been coded these are the
        ones the stream carries.
        """
        return self.control.headers()

    # ----------------------------------------------------------------- closing

    @as_encoder_error
    def close(self) -> None:
        """Release everything, in the order that leaves nothing dangling.

        Every release runs whatever the others do, and the display and the
        render node are closed last, always. Each registered input's EGL
        image belongs to the OpenGL context it was registered in: with
        another current, or none, the image is kept,
        :class:`~pyopengl_video.encoder.EncoderError` says so once everything
        of libva's is released, and closing again with that context current
        gives it back. Closing an encoder with nothing left to release does
        nothing.
        """
        self._closed = True
        self._pending = []
        release_all(self._release_inputs, self._destroy_coded_buffers,
                    *(functools.partial(self._destroy, name, call)
                      for name, call in (('_vpp_context', 'vaDestroyContext'),
                                         ('_vpp_config', 'vaDestroyConfig'),
                                         ('_context', 'vaDestroyContext'),
                                         ('_config', 'vaDestroyConfig'))),
                    self._destroy_surfaces, self._close_display)

    def _release_inputs(self) -> None:
        """Give back every registered input; one that could not be is kept."""
        handles, self._registered = self._registered, []
        try:
            release_all(*(functools.partial(self._release_handle, handle)
                          for handle in handles))
        finally:
            self._registered = [handle for handle in handles
                                if handle.surface or handle.exported is not None]

    def _release_handle(self, handle: InputHandle) -> None:
        """Give back the surface and the buffer behind one registered input."""
        release_all(functools.partial(self._destroy_input_surface, handle),
                    functools.partial(self._close_export, handle))

    def _destroy_input_surface(self, handle: InputHandle) -> None:
        surface, handle.surface = handle.surface, 0
        if surface and self.display is not None:
            identifier = api.VASurfaceID(surface)
            self.api.vaDestroySurfaces(self.display, ctypes.byref(identifier), 1)

    @staticmethod
    def _close_export(handle: InputHandle) -> None:
        # Cleared once closed: an export whose image another context holds
        # stays on the handle for the next close.
        if handle.exported is not None:
            handle.exported.close()
            handle.exported = None

    def _destroy_coded_buffers(self) -> None:
        buffers, self._coded_buffers = self._coded_buffers, []
        if self.display is not None:
            release_all(*(functools.partial(self.api.vaDestroyBuffer, self.display, buffer)
                          for buffer in buffers))

    def _destroy(self, name: str, call: str) -> None:
        """Destroy the libva context or configuration held as attribute `name`."""
        held = getattr(self, name)
        setattr(self, name, type(held)(0))
        if held and self.display is not None:
            getattr(self.api, call)(self.display, held)

    def _destroy_surfaces(self) -> None:
        surfaces = self._sources + self._reconstructions
        self._sources, self._reconstructions = [], []
        if self._device is not None:
            self._device.destroy_surfaces(surfaces)

    def _close_display(self) -> None:
        """Terminate libva and close the render node, once."""
        device, self._device = self._device, None
        if device is not None:
            device.close()

    def _check_open(self) -> None:
        if self._closed:
            raise EncoderError('this encoder has been closed')
