"""Exporting an OpenGL texture as a DMA-BUF, and importing one back.

A DMA-BUF is a file descriptor naming a buffer that more than one driver can
read. It is what lets a frame reach a video encoder without being copied: the
texture the renderer drew into is exported, the encoder's driver imports the
same memory, and nothing crosses the bus.

The export is an EGL extension, ``EGL_MESA_image_dma_buf_export``, so the
OpenGL context has to be an EGL one. GLFW makes a GLX context by default on
X11, and asks for EGL when told::

    glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)

:func:`available` reports whether this context can export at all, and says why
not when it cannot, which is the first thing to look at when a backend that
should be here is missing.

The import is the same extension's counterpart, ``EGL_EXT_image_dma_buf_import``:
a buffer another driver allocated, given a name in OpenGL. That is the
direction a backend uses when the encoder's own allocator has to make the
surface -- the layout is then the encoder's to describe rather than something
OpenGL has to be talked into matching.
"""
from __future__ import annotations

import ctypes
import dataclasses
import logging
import os

log = logging.getLogger(__name__)

#: ``EGL_GL_TEXTURE_2D_KHR``: the target of an EGLImage made from a texture.
EGL_GL_TEXTURE_2D_KHR = 0x30B1
#: ``EGL_LINUX_DMA_BUF_EXT``: the target of an EGLImage made from a DMA-BUF.
EGL_LINUX_DMA_BUF_EXT = 0x3270

#: The attributes an import names its planes with. Three planes is as many as
#: any format here has, and the names are laid out per plane in the order the
#: extension defines them.
_IMPORT_ATTRIBUTES = (
    (0x3272, 0x3273, 0x3274, 0x3443, 0x3444),   # plane 0 fd/offset/pitch/mods
    (0x3275, 0x3276, 0x3277, 0x3445, 0x3446),
    (0x3278, 0x3279, 0x327A, 0x3447, 0x3448),
)
EGL_WIDTH = 0x3057
EGL_HEIGHT = 0x3056
EGL_LINUX_DRM_FOURCC_EXT = 0x3271
EGL_NONE = 0x3038

#: Extensions that have to be on the display for either direction to work.
EXPORT_EXTENSIONS = ('EGL_KHR_image_base', 'EGL_MESA_image_dma_buf_export')
IMPORT_EXTENSIONS = ('EGL_KHR_image_base', 'EGL_EXT_image_dma_buf_import')


class DMABufError(RuntimeError):
    """A texture could not be exported, or a buffer could not be imported."""


@dataclasses.dataclass(frozen=True)
class Plane:
    """One plane of an exported buffer.

    fd -- the DMA-BUF descriptor, owned by the :class:`ExportedImage` that
        holds it and closed when that is closed
    offset -- where the plane starts in the buffer
    stride -- bytes between one row of the plane and the next, which is not
        the row's width: a driver pads rows out to a size it likes
    """

    fd: int
    offset: int
    stride: int


class ExportedImage:
    """An OpenGL texture, named as a buffer another driver can read.

    fourcc -- the DRM format code of the buffer, which says how samples are
        laid out; ``GL_RGBA8`` exports as ``DRM_FORMAT_ABGR8888``
    width/height -- the size of the texture that was exported
    modifier -- the DRM format modifier, which describes any tiling or
        compression the layout has. It must be carried across as it is: a
        surface imported as linear when it is tiled reads as noise, and
        nothing reports that.
    planes -- one per plane of the format

    Closing it closes the descriptors and destroys the EGLImage. The exported
    buffer is the texture's own memory, so the texture must outlive this.
    """

    def __init__(self, display: object, image: object, fourcc: int,
                 width: int, height: int, modifier: int,
                 planes: tuple[Plane, ...]) -> None:
        self._display = display
        self._image = image
        self.fourcc = fourcc
        self.width = width
        self.height = height
        self.modifier = modifier
        self.planes = planes

    def __repr__(self) -> str:
        from pyopengl_video.vaapi.api import fourcc_name
        return (f'<{type(self).__name__} {self.width}x{self.height} '
                f'{fourcc_name(self.fourcc)} modifier {self.modifier:#x}, '
                f'{len(self.planes)} plane(s)>')

    def close(self) -> None:
        """Close the descriptors and give the EGLImage back. Safe to call twice."""
        for plane in self.planes:
            if plane.fd >= 0:
                os.close(plane.fd)
        self.planes = ()
        if self._image is not None:
            from OpenGL.EGL.KHR.image_base import eglDestroyImageKHR
            eglDestroyImageKHR(self._display, self._image)
            self._image = None


def _extensions() -> str:
    """The current EGL display's extension string, or '' with no EGL context."""
    from OpenGL import EGL

    display = EGL.eglGetCurrentDisplay()
    if not display:
        return ''
    text = EGL.eglQueryString(display, EGL.EGL_EXTENSIONS)
    if not text:
        return ''
    return text.decode() if isinstance(text, bytes) else str(text)


def unavailable_because() -> str:
    """Why this context cannot export a texture, or '' when it can.

    Written as the reason rather than as a flag because there are three of
    them and they send a reader to different places: no EGL context at all
    usually means the window was asked for without
    ``glfw.CONTEXT_CREATION_API``, while a missing extension is the driver's
    answer.
    """
    from OpenGL import EGL

    if not EGL.eglGetCurrentDisplay():
        return ('this OpenGL context is not an EGL context, so it cannot '
                'export a texture as a DMA-BUF; create the window with '
                'glfw.window_hint(glfw.CONTEXT_CREATION_API, '
                'glfw.EGL_CONTEXT_API)')
    if not EGL.eglGetCurrentContext():
        return 'no OpenGL context is current on this thread'
    extensions = _extensions()
    missing = [name for name in EXPORT_EXTENSIONS if name not in extensions]
    if missing:
        return f'this EGL display does not offer {", ".join(missing)}'
    return ''


def available() -> bool:
    """Can the current OpenGL context export a texture as a DMA-BUF?

    Quiet and cheap, so a backend probe can call it.
    """
    try:
        return not unavailable_because()
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('EGL is not usable here: %s', error)
        return False


def import_available() -> bool:
    """Can the current OpenGL context import a DMA-BUF as a texture?"""
    try:
        extensions = _extensions()
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('EGL is not usable here: %s', error)
        return False
    return all(name in extensions for name in IMPORT_EXTENSIONS)


def export_texture(texture: int, width: int, height: int) -> ExportedImage:
    """Name `texture` as a DMA-BUF another driver can read.

    The texture keeps its contents and stays usable in OpenGL; what comes back
    is a second name for the same memory. It must not be deleted while the
    exported image is open.

    Raises :class:`DMABufError` when the context cannot export, which
    :func:`unavailable_because` explains.
    """
    from OpenGL import EGL
    from OpenGL.EGL.KHR.image_base import EGL_IMAGE_PRESERVED_KHR, eglCreateImageKHR
    from OpenGL.EGL.MESA.image_dma_buf_export import (
        eglExportDMABUFImageMESA,
        eglExportDMABUFImageQueryMESA,
    )

    reason = unavailable_because()
    if reason:
        raise DMABufError(reason)
    display = EGL.eglGetCurrentDisplay()
    context = EGL.eglGetCurrentContext()

    attributes = (EGL.EGLint * 3)(EGL_IMAGE_PRESERVED_KHR, EGL.EGL_TRUE, EGL_NONE)
    image = eglCreateImageKHR(display, context, EGL_GL_TEXTURE_2D_KHR,
                              ctypes.c_void_p(int(texture)), attributes)
    if not image:
        raise DMABufError(
            f'eglCreateImageKHR refused texture {texture}: the texture must be '
            'complete and GL_RGBA8, and it must belong to this context')

    fourcc = EGL.EGLint(0)
    plane_count = EGL.EGLint(0)
    modifiers = (ctypes.c_uint64 * 4)()
    if not eglExportDMABUFImageQueryMESA(
            display, image, ctypes.byref(fourcc), ctypes.byref(plane_count),
            ctypes.cast(modifiers, ctypes.POINTER(ctypes.c_uint64))):
        _destroy(display, image)
        raise DMABufError('eglExportDMABUFImageQueryMESA would not describe the '
                          'texture, so it cannot be exported')

    count = int(plane_count.value)
    fds = (EGL.EGLint * 4)()
    strides = (EGL.EGLint * 4)()
    offsets = (EGL.EGLint * 4)()
    if not eglExportDMABUFImageMESA(display, image, fds, strides, offsets):
        _destroy(display, image)
        raise DMABufError('eglExportDMABUFImageMESA would not export the texture')

    planes = tuple(Plane(fd=int(fds[index]), offset=int(offsets[index]),
                         stride=int(strides[index]))
                   for index in range(count))
    return ExportedImage(display, image, int(fourcc.value), int(width),
                         int(height), int(modifiers[0]), planes)


def _destroy(display: object, image: object) -> None:
    from OpenGL.EGL.KHR.image_base import eglDestroyImageKHR
    eglDestroyImageKHR(display, image)


def import_texture(fourcc: int, width: int, height: int, modifier: int,
                   planes: tuple[Plane, ...]) -> tuple[int, object]:
    """Give a buffer another driver allocated a name in OpenGL.

    Returns the texture and the EGLImage it was made from; the image has to
    stay alive as long as the texture does, and
    :func:`release_imported_texture` gives both back.

    The descriptors in `planes` are not taken over: EGL duplicates what it
    needs, and the caller closes its own.
    """
    from OpenGL import EGL
    from OpenGL.EGL.KHR.image_base import eglCreateImageKHR
    from OpenGL.GL import (
        GL_CLAMP_TO_EDGE,
        GL_LINEAR,
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_WRAP_S,
        GL_TEXTURE_WRAP_T,
        glBindTexture,
        glGenTextures,
        glTexParameteri,
    )
    from OpenGL.GLES2.OES.EGL_image import glEGLImageTargetTexture2DOES

    if not import_available():
        raise DMABufError(
            'this EGL display does not offer EGL_EXT_image_dma_buf_import, so '
            "a buffer the encoder allocated cannot be given an OpenGL name")
    display = EGL.eglGetCurrentDisplay()

    values = [EGL_WIDTH, int(width), EGL_HEIGHT, int(height),
              EGL_LINUX_DRM_FOURCC_EXT, int(fourcc)]
    from pyopengl_video.vaapi.api import DRM_FORMAT_MOD_INVALID

    for index, plane in enumerate(planes):
        fd_name, offset_name, pitch_name, low_name, high_name = _IMPORT_ATTRIBUTES[index]
        values += [fd_name, plane.fd, offset_name, plane.offset,
                   pitch_name, plane.stride]
        # A modifier of "invalid" means the driver would not name the layout,
        # and passing it on would be a claim about the layout rather than the
        # absence of one.
        if modifier != DRM_FORMAT_MOD_INVALID:
            values += [low_name, modifier & 0xFFFFFFFF,
                       high_name, (modifier >> 32) & 0xFFFFFFFF]
    values.append(EGL_NONE)

    attributes = (EGL.EGLint * len(values))(*values)
    image = eglCreateImageKHR(display, EGL.EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT,
                              None, attributes)
    if not image:
        raise DMABufError(
            f'eglCreateImageKHR would not import a {width}x{height} buffer of '
            f'{len(planes)} plane(s) with modifier {modifier:#x}')

    texture = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, texture)
    for parameter, value in (
        (GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
        (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
    ):
        glTexParameteri(GL_TEXTURE_2D, parameter, value)
    glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, image)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture, image


def release_imported_texture(texture: int, image: object) -> None:
    """Give back what :func:`import_texture` returned."""
    from OpenGL import EGL
    from OpenGL.GL import glDeleteTextures

    if texture:
        glDeleteTextures([int(texture)])
    if image is not None:
        _destroy(EGL.eglGetCurrentDisplay(), image)
