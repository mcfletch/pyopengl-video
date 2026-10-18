"""The VA-API decoder: compressed samples in, OpenGL textures out.

The picture never leaves the GPU. Each decode surface is exported once as a
DMA-BUF of two layers, luma and interleaved chroma, and imported into OpenGL as
two textures (:mod:`pyopengl_video.linux.dmabuf`). Decoding writes the surface
and the textures show it; the conversion pass (:mod:`pyopengl_video.convert`)
turns the two planes into RGBA. An 8-bit stream decodes into NV12 surfaces
(``GL_R8`` and ``GL_RG8`` textures) and a 10-bit one into P010 (``GL_R16`` and
``GL_RG16``).

libva is a slice-level interface: the driver reconstructs blocks, and the
caller supplies the parsed syntax and the picture management. That is the
codec's control layer (:mod:`pyopengl_video.vaapi.control`), one for each codec
in :data:`pyopengl_video.vaapi.codecs.DECODERS`. :class:`VAAPIDecoder` is the
device, the surfaces and the textures, the same for every codec.

**The OpenGL context has to be an EGL context**, as it does for the encoder.
"""
from __future__ import annotations

import ctypes
import functools
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from OpenGL.GL import (
    GL_SYNC_FLUSH_COMMANDS_BIT,
    GL_SYNC_GPU_COMMANDS_COMPLETE,
    GL_TIMEOUT_IGNORED,
    glClientWaitSync,
    glDeleteSync,
    glFenceSync,
)

from pyopengl_video import bounds
from pyopengl_video.colour import Colour
from pyopengl_video.decoder import (
    MAX_OUTPUT_SLOTS,
    DecodedFrame,
    Decoder,
    DecoderError,
    Sample,
)
from pyopengl_video.glcontext import Owner
from pyopengl_video.linux import dmabuf
from pyopengl_video.teardown import release_all
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.codecs import DECODERS, support_for
from pyopengl_video.vaapi.control import Buffer, DecodeControl, Format
from pyopengl_video.vaapi.device import Device, abandon, worded

log = logging.getLogger(__name__)

__all__ = ['DEFAULT_OUTPUT_SLOTS', 'LAYOUTS', 'MAX_FORMAT_SURFACES',
           'MAX_OUTPUT_SLOTS', 'SurfaceLayout', 'SurfaceOrder', 'VAAPIDecoder',
           'allocation', 'as_decoder_error']

#: How many decoded frames a caller may hold at once, unless it says otherwise.
#: A player converts each frame as it arrives and holds none for long.
DEFAULT_OUTPUT_SLOTS = 4

#: The most surfaces a control layer may ask for in a
#: :class:`~pyopengl_video.vaapi.control.Format`. The largest any codec here
#: needs is AV1's with film grain: eight reference slots and the frame being
#: decoded, each a reconstruction and the picture shown, which is 18. H.264
#: and HEVC hold at most 16 pictures and the one being decoded, and VP9 nine.
MAX_FORMAT_SURFACES = 18


def as_decoder_error(method: Callable[..., Any]) -> Callable[..., Any]:
    """Let a method raise only :class:`DecoderError`, keeping the cause.

    libva's errors, the DMA-BUF shim's and the parser's each say something
    worth reading, and none is a :class:`DecoderError` in itself, since each
    module serves more than decoding. Every other :class:`Exception` a stream
    or a driver brings about is translated the same way, :class:`MemoryError`
    included, with the original on ``__cause__``. :class:`KeyboardInterrupt`,
    :class:`SystemExit` and :class:`GeneratorExit` are not errors of the
    stream and pass through.
    """
    @functools.wraps(method)
    def translating(*arguments: Any, **named: Any) -> Any:
        try:
            return method(*arguments, **named)
        except DecoderError:
            raise
        except Exception as error:
            raise DecoderError(worded(error)) from error
    return translating


# ------------------------------------------------------------- surfaces


class SurfaceLayout:
    """How a decode surface of one sample depth is allocated and exported.

    depth -- bits a sample
    rt_format -- the libva render target format
    fourcc -- the surface's pixel format
    layers -- the DRM formats its luma and chroma planes export as
    layout -- what a :class:`~pyopengl_video.decoder.DecodedFrame` calls it
    """

    def __init__(self, depth: int, rt_format: int, fourcc: int,
                 layers: tuple[int, int], layout: str) -> None:
        self.depth = depth
        self.rt_format = rt_format
        self.fourcc = fourcc
        self.layers = layers
        self.layout = layout


#: The surface layouts decoded into, by bits a sample.
LAYOUTS = {
    8: SurfaceLayout(8, api.VA_RT_FORMAT_YUV420, api.VA_FOURCC_NV12,
                     (api.DRM_FORMAT_R8, api.DRM_FORMAT_GR88), 'nv12'),
    10: SurfaceLayout(10, api.VA_RT_FORMAT_YUV420_10, api.VA_FOURCC_P010,
                      (api.DRM_FORMAT_R16, api.DRM_FORMAT_GR1616), 'p010'),
}


class SurfaceOrder:
    """Which free surface to decode into next: the one returned longest ago.

    A surface OpenGL has just given back may still be read by a conversion
    the GPU has not run yet, and decoding into it waits on that; the surface
    returned longest ago is the one least likely to make the decoder wait.
    """

    def __init__(self, surfaces: Sequence[int]) -> None:
        self._order = list(surfaces)

    def choose(self, in_use: set[int]) -> int | None:
        """The free surface returned longest ago, or None when all are in use."""
        for surface in self._order:
            if surface not in in_use:
                return surface
        return None

    def returned(self, surface: int) -> None:
        """Note that `surface` has just come back from OpenGL."""
        self._order.remove(surface)
        self._order.append(surface)


def allocation(format: Format, limits: Mapping[int, int], vendor: str) -> tuple[int, int]:
    """The size surfaces are allocated at: the coded size, at least the driver's minimum.

    format -- what the stream decodes into
    limits -- the surface attributes the device states for the configuration,
        by attribute type; one it leaves out is no limit
    vendor -- whose device it is, for a message

    A picture smaller than the minimum decodes into the top left of a larger
    surface, unless its format needs the surface to be the coded size exactly,
    and then it is refused by its size. One larger than the maximum is refused
    by its size.
    """
    width, height = format.coded_size
    most = (limits.get(api.VASurfaceAttribMaxWidth, width),
            limits.get(api.VASurfaceAttribMaxHeight, height))
    if width > most[0] or height > most[1]:
        raise DecoderError(
            f'a {width}x{height} picture is larger than the {most[0]}x{most[1]} '
            f'this decoder ({vendor}) reaches')
    least = (limits.get(api.VASurfaceAttribMinWidth, 0),
             limits.get(api.VASurfaceAttribMinHeight, 0))
    if format.exact_size and (width < least[0] or height < least[1]):
        raise DecoderError(
            f'a {width}x{height} {format.name} picture is smaller than the '
            f'{least[0]}x{least[1]} this decoder ({vendor}) reaches')
    return max(width, least[0]), max(height, least[1])


class _Pool:
    """Decode surfaces and the OpenGL textures over them, for one sequence.

    Every surface is exported and imported once, when the pool is made, so
    decoding never creates a GL object. A fence made when a lent frame comes
    back is waited on before its surface is decoded into again, so the decoder
    never writes a picture OpenGL is still reading.

    The textures, images and fences belong to the share group of the OpenGL
    context current when the pool is made, and :meth:`close` releases them
    with that context current or one that shares objects with it.
    """

    def __init__(self, device: Device, format: Format, count: int) -> None:
        self.device = device
        self.format = format
        self.profile = format.profile
        self.coded_size = coded_size = format.coded_size
        if format.depth not in LAYOUTS:
            raise DecoderError(
                f'{format.name or "the stream"} has {format.depth} bits a sample, '
                f'and this decoder takes {" or ".join(map(str, LAYOUTS))}')
        self.layout = LAYOUTS[format.depth]
        #: What is allocated: the coded size, raised to the driver's minimum.
        self.size = coded_size
        self.count = count
        self.surfaces: list[int] = []
        self.textures: dict[int, tuple[int, int]] = {}
        self._images: list[tuple[int, dmabuf.ImportedImage]] = []
        self._fences: dict[int, Any] = {}
        self._owner = Owner.current()
        self.config = api.VAConfigID(0)
        self.context = api.VAContextID(0)
        try:
            self._create()
        except BaseException:
            abandon(self.close)
            raise

    def _create(self) -> None:
        va, display = self.device.va, self.device.display
        layout = self.layout
        attribute = api.VAConfigAttrib(type=api.VAConfigAttribRTFormat,
                                       value=layout.rt_format)
        self.device.check(
            va.vaCreateConfig(display, self.profile, api.VAEntrypointVLD,
                              ctypes.byref(attribute), 1, ctypes.byref(self.config)),
            'vaCreateConfig', f'{self.format.name} decode, profile {self.profile}')
        self.size = allocation(self.format, self.device.surface_limits(self.config),
                               self.device.vendor)
        width, height = self.size
        format_attribute = api.VASurfaceAttrib()
        format_attribute.type = api.VASurfaceAttribPixelFormat
        format_attribute.flags = api.VA_SURFACE_ATTRIB_SETTABLE
        format_attribute.value.type = api.VAGenericValueTypeInteger
        format_attribute.value.value.i = layout.fourcc
        surfaces = (api.VASurfaceID * self.count)()
        self.device.check(
            va.vaCreateSurfaces(display, layout.rt_format, width, height,
                                surfaces, self.count, ctypes.byref(format_attribute), 1),
            'vaCreateSurfaces',
            f'{self.count} x {width}x{height} {api.fourcc_name(layout.fourcc)}')
        self.surfaces = [int(surface) for surface in surfaces]
        self.device.check(
            va.vaCreateContext(display, self.config, width, height, api.VA_PROGRESSIVE,
                               surfaces, self.count, ctypes.byref(self.context)),
            'vaCreateContext', f'{width}x{height} decode')
        for surface in self.surfaces:
            self.textures[surface] = self._import(surface)
        self.order = SurfaceOrder(self.surfaces)

    def _import(self, surface: int) -> tuple[int, int]:
        """Export `surface` as DMA-BUF layers and give each an OpenGL name."""
        descriptor = api.VADRMPRIMESurfaceDescriptor()
        self.device.check(
            self.device.va.vaExportSurfaceHandle(
                self.device.display, surface, api.VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                api.VA_EXPORT_SURFACE_READ_ONLY | api.VA_EXPORT_SURFACE_SEPARATE_LAYERS,
                ctypes.byref(descriptor)),
            'vaExportSurfaceHandle', f'surface {surface}')
        try:
            formats = [descriptor.layers[index].drm_format
                       for index in range(descriptor.num_layers)]
            if formats != list(self.layout.layers):
                raise DecoderError(
                    f'surface {surface} exported as layers '
                    f'{[api.fourcc_name(value) for value in formats]}, not the '
                    f'{" and ".join(api.fourcc_name(value).strip() for value in self.layout.layers)} '
                    f'planes of {api.fourcc_name(self.layout.fourcc)}')
            width, height = self.size
            planes = []
            for index, size in enumerate(((width, height), (width // 2, height // 2))):
                layer = descriptor.layers[index]
                source = descriptor.objects[layer.object_index[0]]
                texture, image = dmabuf.import_texture(
                    layer.drm_format, *size, source.drm_format_modifier,
                    (dmabuf.Plane(fd=source.fd, offset=layer.offset[0],
                                  stride=layer.pitch[0]),))
                self._images.append((texture, image))
                planes.append(texture)
            return planes[0], planes[1]
        finally:
            for index in range(descriptor.num_objects):
                fd = descriptor.objects[index].fd
                if fd >= 0:
                    os.close(fd)

    def ready(self, surface: int) -> None:
        """Wait until OpenGL has finished reading `surface` from its last lending."""
        fence = self._fences.pop(surface, None)
        if fence is not None:
            glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED)
            glDeleteSync(fence)

    def returned(self, surface: int) -> None:
        """Note the point in OpenGL's commands where a lent surface came back.

        Nothing is noted once the pool's context is destroyed, or with an
        OpenGL context current that does not share objects with the pool's:
        a fence is an object of the share group it is made in.
        """
        if self.context and self._owner.is_shared():
            self.order.returned(surface)
            previous = self._fences.pop(surface, None)
            if previous is not None:
                glDeleteSync(previous)
            self._fences[surface] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has released everything the pool held."""
        return not (self._images or self._fences or self.context or self.surfaces
                    or self.config)

    def close(self) -> None:
        """Give back the textures, the images, the surfaces and the context.

        Every release runs whatever the others do. The libva context,
        surfaces and configuration are released whichever OpenGL context is
        current. With a context outside the pool's share group current, or
        none, the textures, images and fences are kept and
        :class:`DecoderError` says so once the libva objects are gone; closing
        again with a context of the group current releases them.
        """
        self.textures = {}
        release_all(self._release_textures, self._destroy_context,
                    self._destroy_surfaces, self._destroy_config)

    def _release_textures(self) -> None:
        if not (self._fences or self._images):
            return
        self._owner.require("a decoder's surface textures", DecoderError)
        fences, self._fences = self._fences, {}
        images, self._images = self._images, []
        release_all(
            *(functools.partial(glDeleteSync, fence) for fence in fences.values()),
            *(functools.partial(dmabuf.release_imported_texture, texture, image)
              for texture, image in images))

    def _destroy_context(self) -> None:
        context, self.context = self.context, api.VAContextID(0)
        if context and self.device.display is not None:
            self.device.va.vaDestroyContext(self.device.display, context)

    def _destroy_surfaces(self) -> None:
        surfaces, self.surfaces = self.surfaces, []
        self.device.destroy_surfaces(surfaces)

    def _destroy_config(self) -> None:
        config, self.config = self.config, api.VAConfigID(0)
        if config and self.device.display is not None:
            self.device.va.vaDestroyConfig(self.device.display, config)


# --------------------------------------------------------------- decoder


class _Session:
    """The surfaces a control layer decodes into, and the frames lent from them.

    This is the :class:`~pyopengl_video.vaapi.control.DecodeSession` a control
    layer is given. Pictures it outputs become
    :class:`~pyopengl_video.decoder.DecodedFrame` objects, collected until
    :meth:`take`.
    """

    def __init__(self, device: Device, output_slots: int,
                 make_pool: Callable[[Device, Format, int], _Pool] = _Pool) -> None:
        self.device = device
        self.output_slots = output_slots
        self._make_pool = make_pool
        self.control: DecodeControl | None = None
        #: The surfaces pictures are decoded into, or None: before the first
        #: format is activated, and after one the device refused.
        self.pool: _Pool | None = None
        self._refused = False
        self._retired: list[_Pool] = []
        self._lent: dict[DecodedFrame, tuple[_Pool, int]] = {}
        self._ready: list[DecodedFrame] = []

    def take(self) -> list[DecodedFrame]:
        """The frames output since the last call, in display order."""
        ready, self._ready = self._ready, []
        return ready

    def activate(self, format: Format) -> None:
        """Make the surfaces suit `format`, replacing them when they do not.

        The pool before is given up first, so a device never holds the
        surfaces of two sequences at once. When the new pool cannot be made
        the session holds none, and every picture is refused until a format
        is activated that the device takes.
        """
        surfaces = bounds.whole(format.surfaces, 'the number of decode surfaces '
                                f'{format.name or "the stream"} asks for',
                                1, MAX_FORMAT_SURFACES, DecoderError)
        count = surfaces + self.output_slots
        pool = self.pool
        if (pool is not None and count <= pool.count
                and (pool.profile, pool.coded_size, pool.layout.depth)
                == (format.profile, format.coded_size, format.depth)):
            return
        # Cleared before the old pool is released, and assigned once the new
        # one is whole: a submission never finds a destroyed context here.
        self.pool = None
        self._refused = True
        if pool is not None:
            self._retire(pool)
        log.debug('%s: %d surfaces of %dx%d for %s', self.device.node, count,
                  *format.coded_size, format.name)
        self.pool = self._make_pool(self.device, format, count)
        self._refused = False

    def _live(self, surface: int | None = None) -> _Pool:
        """The pool pictures are decoded into; `surface`, if given, is one of its own."""
        pool = self.pool
        if pool is None:
            if self._refused:
                raise DecoderError(
                    'the decode surfaces for the stream could not be made, so '
                    'there is nothing to decode into until a sequence begins '
                    'that this device takes')
            raise DecoderError('a picture arrived before anything said what the stream is')
        if surface is not None and surface not in pool.textures:
            raise DecoderError(
                f'surface {surface} is not one of the {pool.count} decode '
                'surfaces in use; it belongs to a sequence whose surfaces '
                'have been given back')
        return pool

    def acquire(self) -> int:
        pool = self._live()
        held = self.control.held() if self.control is not None else set()
        lent = [frame for frame, (owner, _) in self._lent.items() if owner is pool]
        in_use = held | {self._lent[frame][1] for frame in lent}
        surface = pool.order.choose(in_use)
        if surface is None:
            raise DecoderError(
                f'every one of the {pool.count} decode surfaces is in use: '
                f'{len(held)} by the pictures the decoder keeps and '
                f'{len(lent)} by frames still held ({lent}); release frames before '
                'decoding more, or open the decoder with more output_slots')
        pool.ready(surface)
        return surface

    def submit(self, surface: int, batches: Sequence[Sequence[Buffer]], what: str) -> None:
        pool = self._live(surface)
        made: list[list[int]] = []
        try:
            for batch in batches:
                made.append([])
                for kind, payload in batch:
                    made[-1].append(self.device.buffer(pool.context, kind, payload))
        except BaseException:
            for buffer in (buffer for batch in made for buffer in batch):
                self.device.va.vaDestroyBuffer(self.device.display, buffer)
            raise
        self.device.picture(pool.context, surface, made, what)

    def output(self, surface: int, *, timestamp: int, duration: int,
               visible: tuple[int, int, int, int], colour: Colour) -> None:
        """Wait for the picture in `surface` and lend it as the next frame.

        A surface the driver will not synchronise is not lent: no frame is
        made, the error is raised, and the surface stays the control layer's
        to hold or to decode into again. What it holds is undefined.
        """
        pool = self._live(surface)
        self.device.check(self.device.va.vaSyncSurface(self.device.display, surface),
                          'vaSyncSurface', f'surface {surface}')
        frame = DecodedFrame(
            timestamp=timestamp, duration=duration, planes=pool.textures[surface],
            layout=pool.layout.layout, texture_size=pool.size, visible=visible,
            colour=colour, on_release=self._released)
        self._lent[frame] = (pool, surface)
        self._ready.append(frame)

    def _released(self, frame: DecodedFrame) -> None:
        found = self._lent.pop(frame, None)
        if found is None:
            return
        owner, surface = found
        if not any(lent == (owner, surface) for lent in self._lent.values()):
            owner.returned(surface)
        if owner in self._retired and not any(
                pool is owner for pool, _ in self._lent.values()):
            try:
                owner.close()
            finally:
                if owner.closed:
                    self._retired.remove(owner)

    def _retire(self, pool: _Pool) -> None:
        """Close `pool`, once no frame lent from it is held.

        A pool that will not close whole, its OpenGL context not being
        current, is kept for :meth:`close`.
        """
        if any(owner is pool for owner, _ in self._lent.values()):
            self._retired.append(pool)
            return
        try:
            pool.close()
        finally:
            if not pool.closed:
                self._retired.append(pool)

    def close(self) -> None:
        """Release the surfaces and their textures, frames lent included.

        Every frame is released and every pool closed whatever the others
        do. A pool that could not release everything, because its OpenGL
        context was not current, is kept for the next call to close.
        """
        pools = [*self._retired, *([self.pool] if self.pool else [])]
        lent, self._lent = self._lent, {}
        self.pool = None
        self._retired = []
        self._ready = []
        try:
            release_all(*(frame.release for frame in lent),
                        *(pool.close for pool in pools))
        finally:
            self._retired = [pool for pool in pools if not pool.closed]


class VAAPIDecoder(Decoder):
    """Decoding on Intel and AMD hardware, into OpenGL textures.

    codec -- any codec in :data:`pyopengl_video.vaapi.codecs.DECODERS` that the
        device offers; :func:`pyopengl_video.decoders` lists them for this machine
    headers -- the stream's configuration, when the container keeps it apart
        from the samples: parameter sets as Annex-B for H.264 and HEVC, the
        configuration OBUs for AV1
    size -- accepted for the registry; the stream's own size is used
    device -- the DRM render node to open; the default is the first with a
        decoder for the codec. Afterwards, the node that was opened.
    output_slots -- how many frames a caller may hold unreleased at once

    Each sample is one access unit, temporal unit or frame, as its container
    holds it. Frames come out in display order as
    :class:`~pyopengl_video.decoder.DecodedFrame` objects, ``'nv12'`` from an
    8-bit stream and ``'p010'`` from a 10-bit one.
    """

    zero_copy = True

    @as_decoder_error
    def __init__(self, codec: str = 'h264', *, headers: bytes = b'',
                 size: tuple[int, int] | None = None, device: str | None = None,
                 output_slots: int = DEFAULT_OUTPUT_SLOTS, **unknown: object) -> None:
        if unknown:
            raise DecoderError(
                f'the VA-API decoder has no setting named {", ".join(sorted(unknown))}; '
                'it takes headers, size, device and output_slots')
        support = support_for(DECODERS, codec)
        if support is None:
            raise DecoderError(
                f'the VA-API backend decodes {", ".join(sorted(DECODERS))}, not {codec!r}')
        del size
        self.codec = support.codec
        self.output_slots = bounds.whole(output_slots, 'output_slots', 1,
                                         MAX_OUTPUT_SLOTS, DecoderError)
        self._closed = False
        nodes = [device] if device else api.render_nodes()
        self._device = Device.first(nodes, support.objection,
                                    f'{support.title} decoding')
        self.device = self._device.node
        self.vendor = self._device.vendor
        self._session = _Session(self._device, self.output_slots)
        try:
            self._control: DecodeControl = support.load()(self._session, self.output_slots)
            self._session.control = self._control
            self._control.configure(headers)
            self._describe()
        except BaseException:
            abandon(self.close)
            raise

    def __repr__(self) -> str:
        width, height = self.size
        return f'<{type(self).__name__} {width}x{height} {self.codec} on {self.device}>'

    def _describe(self) -> None:
        """Take the picture size and colour the control layer has read."""
        self.size = self._control.size
        self.colour = self._control.colour

    @as_decoder_error
    def decode(self, sample: Sample) -> list[DecodedFrame]:
        """Decode one sample; return the frames now ready, in display order."""
        self._check_open()
        try:
            self._control.decode(sample)
        finally:
            self._describe()
        return self._session.take()

    @as_decoder_error
    def flush(self) -> list[DecodedFrame]:
        """Decode what is gathered and return every frame still held back."""
        self._check_open()
        self._control.flush()
        return self._session.take()

    @as_decoder_error
    def reset(self) -> None:
        """Drop every picture held back. Frames already lent stay valid."""
        self._check_open()
        self._control.reset()

    @as_decoder_error
    def close(self) -> None:
        """Release the surfaces and their textures, frames lent included.

        The libva objects, the display and the render node are released
        whichever OpenGL context is current. The surface textures belong to
        the share group of the context the decoder was opened in: with a
        context outside it current, or none, they are kept,
        :class:`DecoderError` says so, and closing again with a context of
        the group current releases them.
        Closing a decoder that has nothing left to release does nothing.
        """
        self._closed = True
        release_all(self._close_session, self._close_device)

    def _close_session(self) -> None:
        session = getattr(self, '_session', None)
        if session is not None:
            session.close()

    def _close_device(self) -> None:
        device = getattr(self, '_device', None)
        if device is not None:
            device.close()

    def _check_open(self) -> None:
        if self._closed:
            raise DecoderError('this decoder has been closed')
