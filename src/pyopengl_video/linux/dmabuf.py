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
import functools
import logging
import os
from collections.abc import Callable
from types import ModuleType
from typing import Any, TypeVar

from pyopengl_video.glcontext import Owner
from pyopengl_video.teardown import release_all
from pyopengl_video.vaapi.api import DRM_FORMAT_MOD_INVALID, fourcc_name

T = TypeVar('T')

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

#: ``GL_RGBA8`` and ``GL_RGBA``, the internal formats an exported texture may
#: report. A texture made with the unsized ``GL_RGBA`` reports that, and the
#: format code of its export says how many bits a channel the driver gave it.
GL_RGBA8 = 0x8058
GL_RGBA = 0x1908
RGBA_FORMATS = (GL_RGBA8, GL_RGBA)

#: Extensions that have to be on the display for either direction to work.
EXPORT_EXTENSIONS = ('EGL_KHR_image_base', 'EGL_MESA_image_dma_buf_export')
IMPORT_EXTENSIONS = ('EGL_KHR_image_base', 'EGL_EXT_image_dma_buf_import')


class DMABufError(RuntimeError):
    """A texture could not be exported, or a buffer could not be imported."""


def _require(description: str, call: Callable[..., T], *arguments: Any) -> T:
    """Make an EGL call whose false answer, or whose exception, means refused.

    A driver has two ways of saying no to the same request: answer with a null
    handle or a false, or set an error, which PyOpenGL raises. Which one
    arrives is the driver's business -- the two here disagree about it, and
    about which call in a sequence does the checking. A caller is owed the same
    :class:`DMABufError` either way, naming what was asked for, rather than a
    bare ``EGLError`` from an entry point it never called.
    """
    try:
        answer = call(*arguments)
    except Exception as error:
        raise DMABufError(f'{description}: {error}') from error
    if not answer:
        raise DMABufError(description)
    return answer


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

    owner -- the OpenGL context the texture was exported in; the one current
        when this is made, unless given

    Closing it closes the descriptors and destroys the EGLImage. The exported
    buffer is the texture's own memory, so the texture must outlive this.
    """

    def __init__(self, display: object, image: object, fourcc: int,
                 width: int, height: int, modifier: int,
                 planes: tuple[Plane, ...], owner: Owner | None = None) -> None:
        self._display = display
        self._image = image
        self._owner = Owner.current() if owner is None else owner
        self.fourcc = fourcc
        self.width = width
        self.height = height
        self.modifier = modifier
        self.planes = planes

    def __repr__(self) -> str:
        return (f'<{type(self).__name__} {self.width}x{self.height} '
                f'{fourcc_name(self.fourcc)} modifier {self.modifier:#x}, '
                f'{len(self.planes)} plane(s)>')

    def close(self) -> None:
        """Close the descriptors and give the EGLImage back. Safe to call twice.

        The descriptors are closed whichever OpenGL context is current. The
        image is given back on the display it was made on, with the context
        the texture was exported in current; with another current, or none,
        it is kept, :class:`DMABufError` says so once the descriptors are
        closed, and closing again in the right context gives it back.
        """
        release_all(self._close_descriptors, self._destroy_image)

    def _close_descriptors(self) -> None:
        planes, self.planes = self.planes, ()
        release_all(*(functools.partial(os.close, plane.fd)
                      for plane in planes if plane.fd >= 0))

    def _destroy_image(self) -> None:
        if self._image is None:
            return
        self._owner.require('an exported texture', DMABufError)
        image, self._image = self._image, None
        _destroy(self._display, image)


class ImportedImage:
    """The EGLImage an imported texture shows, and where both were made.

    texture -- the texture's name, 0 once it has been released
    image -- the EGLImage, None once it has been released
    owner -- the OpenGL context the texture was made in, and its display
    """

    __slots__ = ('image', 'owner', 'texture')

    def __init__(self, texture: int, image: object, owner: Owner) -> None:
        self.texture = texture
        self.image = image
        self.owner = owner

    def __repr__(self) -> str:
        return f'<{type(self).__name__} texture {self.texture} of {self.owner!r}>'


#: Why nothing here can run without an EGL library, said once.
NO_EGL_LIBRARY = (
    'PyOpenGL has no EGL library to bind to on this machine, so a texture '
    'cannot be exported as a DMA-BUF; EGL ships with the graphics driver, and '
    'a machine with no driver installed has none to load'
)


def _egl() -> ModuleType | None:
    """PyOpenGL's EGL bindings, or None where the machine has no EGL library.

    A machine with no graphics driver installed has no libEGL, and the EGL
    bindings do not import at all there rather than importing and answering
    no. Every route into EGL below comes through here, so that absence reads
    as "this context cannot export" -- which is one of the answers this module
    exists to give -- rather than as an exception out of an import.
    """
    try:
        from OpenGL import EGL  # noqa: PLC0415 - OpenGL.EGL needs an EGL library
    except Exception as error:  # noqa: BLE001 - however it fails, there is no EGL
        # Any exception, because the failure has more than one shape: PyOpenGL
        # raises ImportError from 4.0 on, and the releases this package also
        # supports reach for an entry point on the library that is not there
        # and raise AttributeError.
        log.debug('the EGL bindings are not usable here: %s', error)
        return None
    return EGL


def _require_egl() -> ModuleType:
    """:func:`_egl`, refusing the work rather than returning nothing."""
    egl = _egl()
    if egl is None:
        raise DMABufError(NO_EGL_LIBRARY)
    return egl


def _extensions() -> str:
    """The current EGL display's extension string, or '' with no EGL context."""
    egl = _egl()
    if egl is None:
        return ''
    display = egl.eglGetCurrentDisplay()
    if not display:
        return ''
    text = egl.eglQueryString(display, egl.EGL_EXTENSIONS)
    if not text:
        return ''
    return text.decode() if isinstance(text, bytes) else str(text)


def unavailable_because() -> str:
    """Why this context cannot export a texture, or '' when it can.

    Written as the reason rather than as a flag because there are four of
    them and they send a reader to different places: no EGL library is the
    machine's, no EGL context usually means the window was asked for without
    ``glfw.CONTEXT_CREATION_API``, and a missing extension is the driver's
    answer.
    """
    egl = _egl()
    if egl is None:
        return NO_EGL_LIBRARY
    if not egl.eglGetCurrentDisplay():
        return ('this OpenGL context is not an EGL context, so it cannot '
                'export a texture as a DMA-BUF; create the window with '
                'glfw.window_hint(glfw.CONTEXT_CREATION_API, '
                'glfw.EGL_CONTEXT_API)')
    if not egl.eglGetCurrentContext():
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


def _require_described(texture: int, width: int, height: int) -> None:
    """Raise :class:`DMABufError` unless `texture` is RGBA and `width` by `height`.

    The importer of the buffer is told the size, and sizes its reads by it,
    so the size is OpenGL's answer and not the caller's statement alone.
    """
    from OpenGL.GL import (  # noqa: PLC0415 - OpenGL.GL stays off the probe path
        GL_TEXTURE_2D,
        GL_TEXTURE_BINDING_2D,
        GL_TEXTURE_HEIGHT,
        GL_TEXTURE_INTERNAL_FORMAT,
        GL_TEXTURE_WIDTH,
        glBindTexture,
        glGetIntegerv,
        glGetTexLevelParameteriv,
    )
    try:
        previous = int(glGetIntegerv(GL_TEXTURE_BINDING_2D))
        glBindTexture(GL_TEXTURE_2D, int(texture))
        try:
            found_width, found_height, found_format = (
                int(glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, parameter))
                for parameter in (GL_TEXTURE_WIDTH, GL_TEXTURE_HEIGHT,
                                  GL_TEXTURE_INTERNAL_FORMAT))
        finally:
            glBindTexture(GL_TEXTURE_2D, previous)
    except Exception as error:
        raise DMABufError(
            f'OpenGL would not say what size texture {texture} is, so it '
            f'cannot be exported: {error}') from error
    if (found_width, found_height) != (int(width), int(height)):
        raise DMABufError(
            f'texture {texture} is {found_width}x{found_height}, and it was to '
            f'be exported as {width}x{height}; the two must be the same')
    if found_format not in RGBA_FORMATS:
        raise DMABufError(
            f'texture {texture} has the internal format {found_format:#x}, and '
            f'a texture is exported from GL_RGBA8 ({GL_RGBA8:#x})')


def export_texture(texture: int, width: int, height: int) -> ExportedImage:
    """Name `texture` as a DMA-BUF another driver can read.

    width/height -- the texture's size, which OpenGL is asked to confirm

    The texture keeps its contents and stays usable in OpenGL; what comes back
    is a second name for the same memory. It must not be deleted while the
    exported image is open.

    Raises :class:`DMABufError` when the context cannot export, which
    :func:`unavailable_because` explains, and when the texture is not
    ``GL_RGBA8`` of the size stated.
    """
    EGL = _require_egl()
    from OpenGL.EGL.KHR.image_base import (  # noqa: PLC0415 - OpenGL.EGL needs an EGL library
        EGL_IMAGE_PRESERVED_KHR,
        eglCreateImageKHR,
    )
    from OpenGL.EGL.MESA.image_dma_buf_export import (  # noqa: PLC0415 - OpenGL.EGL needs an EGL library
        eglExportDMABUFImageMESA,
        eglExportDMABUFImageQueryMESA,
    )

    reason = unavailable_because()
    if reason:
        raise DMABufError(reason)
    owner = Owner.current()
    display = EGL.eglGetCurrentDisplay()
    context = EGL.eglGetCurrentContext()

    attributes = (EGL.EGLint * 3)(EGL_IMAGE_PRESERVED_KHR, EGL.EGL_TRUE, EGL_NONE)
    image = _require(
        f'eglCreateImageKHR refused texture {texture}: the texture must be '
        'complete and GL_RGBA8, and it must belong to this context',
        eglCreateImageKHR, display, context, EGL_GL_TEXTURE_2D_KHR,
        ctypes.c_void_p(int(texture)), attributes)

    fourcc = EGL.EGLint(0)
    plane_count = EGL.EGLint(0)
    modifiers = (ctypes.c_uint64 * 4)()
    fds = (EGL.EGLint * 4)()
    strides = (EGL.EGLint * 4)()
    offsets = (EGL.EGLint * 4)()
    # The image is this function's until both queries have answered.
    try:
        _require_described(texture, width, height)
        _require(
            'eglExportDMABUFImageQueryMESA would not describe the texture, so '
            'it cannot be exported',
            eglExportDMABUFImageQueryMESA, display, image, ctypes.byref(fourcc),
            ctypes.byref(plane_count),
            ctypes.cast(modifiers, ctypes.POINTER(ctypes.c_uint64)))
        count = int(plane_count.value)
        # The arrays below are what the driver writes a descriptor for each
        # plane into, so its count is held to their length before it does.
        if not 0 < count <= len(fds):
            raise DMABufError(
                f'the texture was described as {count} planes, and a format '
                f'here has from 1 to {len(fds)}')
        _require('eglExportDMABUFImageMESA would not export the texture',
                 eglExportDMABUFImageMESA, display, image, fds, strides, offsets)
    except BaseException:
        _destroy(display, image)
        raise

    planes = tuple(Plane(fd=int(fds[index]), offset=int(offsets[index]),
                         stride=int(strides[index]))
                   for index in range(count))
    return ExportedImage(display, image, int(fourcc.value), int(width),
                         int(height), int(modifiers[0]), planes, owner)


def _destroy(display: object, image: object) -> None:
    from OpenGL.EGL.KHR.image_base import (  # noqa: PLC0415 - OpenGL.EGL needs an EGL library
        eglDestroyImageKHR,
    )
    eglDestroyImageKHR(display, image)


def import_texture(fourcc: int, width: int, height: int, modifier: int,
                   planes: tuple[Plane, ...]) -> tuple[int, ImportedImage]:
    """Give a buffer another driver allocated a name in OpenGL.

    Returns the texture and the :class:`ImportedImage` it shows; the image
    has to stay alive as long as the texture does, and
    :func:`release_imported_texture` gives both back. The texture binding in
    force when this is called is put back.

    The descriptors in `planes` are not taken over: EGL duplicates what it
    needs, and the caller closes its own.
    """
    EGL = _require_egl()
    from OpenGL.EGL.KHR.image_base import (  # noqa: PLC0415 - OpenGL.EGL needs an EGL library
        eglCreateImageKHR,
    )
    from OpenGL.GL import (  # noqa: PLC0415 - OpenGL.GL stays off the probe path
        GL_CLAMP_TO_EDGE,
        GL_LINEAR,
        GL_TEXTURE_2D,
        GL_TEXTURE_BINDING_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_WRAP_S,
        GL_TEXTURE_WRAP_T,
        glBindTexture,
        glDeleteTextures,
        glGenTextures,
        glGetIntegerv,
        glTexParameteri,
    )
    from OpenGL.GLES2.OES.EGL_image import (  # noqa: PLC0415 - OpenGL.GL stays off the probe path
        glEGLImageTargetTexture2DOES,
    )

    if not import_available():
        raise DMABufError(
            'this EGL display does not offer EGL_EXT_image_dma_buf_import, so '
            "a buffer the encoder allocated cannot be given an OpenGL name")
    if len(planes) > len(_IMPORT_ATTRIBUTES):
        raise DMABufError(
            f'a buffer of {len(planes)} planes cannot be imported; a format '
            f'here has at most {len(_IMPORT_ATTRIBUTES)}')
    owner = Owner.current()
    display = EGL.eglGetCurrentDisplay()

    values = [EGL_WIDTH, int(width), EGL_HEIGHT, int(height),
              EGL_LINUX_DRM_FOURCC_EXT, int(fourcc)]

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

    layout = (f'modifier {modifier:#x}' if modifier != DRM_FORMAT_MOD_INVALID
              else 'no named layout, which a driver need not accept for a '
                   'buffer that is not linear')
    attributes = (EGL.EGLint * len(values))(*values)
    image = _require(
        f'eglCreateImageKHR would not import a {width}x{height} buffer of '
        f'{len(planes)} plane(s) with {layout}',
        eglCreateImageKHR, display, EGL.EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT,
        None, attributes)

    # Which of the two calls checks the layout is the driver's choice: Mesa
    # answers above, and NVIDIA accepts the image and refuses it here. A caller
    # is owed the same exception either way, and neither the image nor the
    # texture is theirs until both calls have succeeded.
    texture = 0
    try:
        previous = int(glGetIntegerv(GL_TEXTURE_BINDING_2D))
        texture = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, texture)
        try:
            for parameter, value in (
                (GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
                (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
            ):
                glTexParameteri(GL_TEXTURE_2D, parameter, value)
            glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, image)
        finally:
            glBindTexture(GL_TEXTURE_2D, previous)
    except BaseException as error:
        release_all(functools.partial(_destroy, display, image),
                    functools.partial(_delete_texture, glDeleteTextures, texture))
        if not isinstance(error, Exception):
            raise
        raise DMABufError(
            f'this EGL display would not give a {width}x{height} buffer of '
            f'{len(planes)} plane(s) with {layout} an OpenGL name: '
            f'{error}') from error
    return texture, ImportedImage(texture, image, owner)


def _delete_texture(delete: Callable[[list[int]], object], texture: int) -> None:
    if texture:
        delete([texture])


def release_imported_texture(texture: int, image: ImportedImage | None) -> None:
    """Give back what :func:`import_texture` returned. Safe to call twice.

    texture -- the texture; read only where `image` is None, since the image
        records which texture it still has to give back

    The texture is deleted in the context that made it. With another context
    current, or none, nothing is released and :class:`DMABufError` says so:
    the name belongs to something else there.
    """
    from OpenGL.GL import glDeleteTextures  # noqa: PLC0415 - OpenGL.GL stays off the probe path

    if image is None:
        _delete_texture(glDeleteTextures, int(texture))
        return
    if not image.texture and image.image is None:
        return
    image.owner.require('an imported texture', DMABufError)
    release_all(functools.partial(_forget_texture, glDeleteTextures, image),
                functools.partial(_forget_image, image))


def _forget_texture(delete: Callable[[list[int]], object], image: ImportedImage) -> None:
    texture, image.texture = image.texture, 0
    _delete_texture(delete, texture)


def _forget_image(image: ImportedImage) -> None:
    handle, image.image = image.image, None
    if handle is not None:
        _destroy(image.owner.display, handle)
