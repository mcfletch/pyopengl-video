"""One allocation seen twice: a Direct3D 11 texture that OpenGL draws into.

``WGL_NV_DX_interop2`` registers a D3D11 resource with the OpenGL driver and
gives back a GL texture name for it. The renderer blits into that name and the
encoder reads the D3D11 texture; there is no copy between them, because there is
only one allocation. Ownership passes back and forth with
``wglDXLockObjectsNV``, which the driver implements as synchronisation rather
than as a transfer -- :meth:`SharedTexture.for_drawing` is that scope.

The extension is named for NVIDIA and implemented well beyond it: Intel's and
AMD's Windows drivers offer it too, which is what makes this one mechanism serve
every Windows backend. The entry points live in PyOpenGL's
``OpenGL.WGL.NV.DX_interop``; the ``_interop2`` name carries only the flag that
says D3D10 and D3D11 resources are allowed.
"""
from __future__ import annotations

import ctypes
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pyopengl_video.inputs import (
    InputHandle,
    create_framebuffer,
    delete_framebuffer,
)
from pyopengl_video.windows import d3d11
from pyopengl_video.windows.com import InteropError

log = logging.getLogger(__name__)

#: The renderer overwrites the whole frame every time, so OpenGL never needs the
#: previous contents brought back for it.
WGL_ACCESS_READ_ONLY_NV = 0x0000
WGL_ACCESS_READ_WRITE_NV = 0x0001
WGL_ACCESS_WRITE_DISCARD_NV = 0x0002

#: The extension whose presence means D3D11 resources may be registered. The
#: original ``WGL_NV_DX_interop`` covers Direct3D 9 alone.
EXTENSION = 'WGL_NV_DX_interop2'


def wgl_extensions() -> set[str]:
    """The WGL extensions the current context offers.

    Empty when there is no current context, or when the driver does not answer
    -- both of which mean the same thing to a caller: do not use this path.
    """
    try:
        from OpenGL.WGL import wglGetCurrentDC
        from OpenGL.WGL.ARB.extensions_string import wglGetExtensionsStringARB
    except ImportError:                      # pragma: no cover - Windows PyOpenGL has it
        return set()
    try:
        device_context = wglGetCurrentDC()
        if not device_context:
            return set()
        text = wglGetExtensionsStringARB(device_context)
    except Exception as error:               # noqa: BLE001 - any failure means "no"
        log.debug('the WGL extension string is unavailable: %r', error)
        return set()
    if not text:
        return set()
    names = ctypes.cast(text, ctypes.c_char_p).value
    if not names:
        return set()
    return set(names.decode('ascii', 'replace').split())


def available() -> bool:
    """Can the current OpenGL context share Direct3D 11 textures?"""
    return EXTENSION in wgl_extensions()


def handle_array(handle: Any) -> Any:
    """A one-element array holding `handle`, for the calls that take several.

    Locking and unlocking take an array of registered objects, and PyOpenGL
    hands the registration's result back differently depending on which
    dispatch implementation is running: an opaque pointer object under the
    compiled layer, a plain integer under ctypes. Both name the same object, so
    the array is built from the interface's own ``HANDLE`` type and the value is
    reduced to its address, rather than trusting whichever Python type arrived.
    """
    from OpenGL.raw.WGL._types import HANDLE

    if isinstance(handle, int):
        address = handle
    else:
        address = ctypes.cast(handle, ctypes.c_void_p).value or 0
    return (HANDLE * 1)(address)


def context_luid() -> bytes | None:
    """The LUID of the GPU the current OpenGL context is running on.

    None when the driver does not offer ``GL_EXT_memory_object_win32``, which is
    what carries the identifier. Without it a caller cannot tell whether an
    encoder would be reading the renderer's own memory, so the honest answer is
    that it does not know.
    """
    try:
        from OpenGL.GL.EXT.memory_object import glGetUnsignedBytevEXT
        from OpenGL.GL.EXT.memory_object_win32 import GL_DEVICE_LUID_EXT
    except ImportError:                      # pragma: no cover - PyOpenGL binds both
        return None
    buffer = (ctypes.c_ubyte * 8)()
    try:
        glGetUnsignedBytevEXT(GL_DEVICE_LUID_EXT, buffer)
    except Exception as error:               # noqa: BLE001 - an unsupported query
        log.debug('this driver does not report a device LUID: %r', error)
        return None
    return bytes(buffer)


def adapter_for_context() -> d3d11.Adapter | None:
    """The DXGI adapter the current OpenGL context is running on.

    This is what decides whether a backend can offer a zero-copy path: an
    encoder on any other adapter would be reading memory the renderer did not
    write. None when the driver will not say.
    """
    luid = context_luid()
    return None if luid is None else d3d11.adapter_for_luid(luid)


class SharedTexture(InputHandle):
    """A texture that is a Direct3D 11 resource and an OpenGL texture at once.

    :attr:`texture` is the OpenGL name to draw into and :attr:`framebuffer` has
    it attached; :attr:`resource` is the ``ID3D11Texture2D`` an encoder is given.
    Draw inside :meth:`for_drawing`, which is where the two APIs hand the
    allocation to one another.
    """

    def __init__(self, device: InteropDevice, resource: d3d11.Texture,
                 texture: int, handle: Any):
        from OpenGL.GL import GL_TEXTURE_2D
        self.interop = device
        self.resource = resource
        self.texture = int(texture)
        self.target = int(GL_TEXTURE_2D)
        self.handle = handle
        self.framebuffer = create_framebuffer(self.texture, self.target)
        self._locked = False

    @property
    def size(self) -> tuple[int, int]:
        """The frame size this texture holds."""
        return (self.resource.width, self.resource.height)

    @contextmanager
    def for_drawing(self) -> Iterator[SharedTexture]:
        """Hold the texture for OpenGL, giving it back to Direct3D at the end.

        The lock is what orders the renderer's writes against the encoder's
        reads; without it the encoder may read a half-drawn frame. Locking twice
        over is an error the driver reports, so a nested scope is refused here
        with a message that says which call is the extra one.
        """
        if self._locked:
            raise InteropError(0, 'wglDXLockObjectsNV',
                               'this texture is already held for drawing')
        from OpenGL.WGL.NV.DX_interop import wglDXLockObjectsNV, wglDXUnlockObjectsNV
        handles = handle_array(self.handle)
        if not wglDXLockObjectsNV(self.interop.pointer, 1, handles):
            raise InteropError(ctypes.GetLastError(), 'wglDXLockObjectsNV')
        self._locked = True
        try:
            yield self
        finally:
            self._locked = False
            if not wglDXUnlockObjectsNV(self.interop.pointer, 1, handles):
                raise InteropError(ctypes.GetLastError(), 'wglDXUnlockObjectsNV')

    def read(self) -> bytes:
        """The pixels, from the Direct3D side. For checking a frame, not moving one."""
        return self.resource.read()

    def close(self) -> None:
        """Unregister the texture and give both views of it back."""
        if self.texture:
            from OpenGL.GL import glDeleteTextures
            from OpenGL.WGL.NV.DX_interop import wglDXUnregisterObjectNV
            delete_framebuffer(self.framebuffer)
            self.framebuffer = 0
            if self.handle is not None and self.interop.pointer is not None:
                wglDXUnregisterObjectNV(self.interop.pointer, self.handle)
            self.handle = None
            glDeleteTextures([self.texture])
            self.texture = 0
        self.resource.close()

    def __enter__(self) -> SharedTexture:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()


class InteropDevice:
    """The bridge between one OpenGL context and one Direct3D 11 device.

    Both must be on the same GPU -- see
    :func:`adapter_for_context` -- and the OpenGL context must be current on
    this thread whenever a texture is made, drawn into or closed.
    """

    def __init__(self, device: d3d11.Device):
        if not available():
            raise InteropError(
                0, 'wglDXOpenDeviceNV',
                f'{EXTENSION} is not offered by the current OpenGL context')
        from OpenGL.WGL.NV.DX_interop import wglDXOpenDeviceNV
        self.device = device
        self.pointer = wglDXOpenDeviceNV(device.pointer)
        if not self.pointer:
            raise InteropError(
                ctypes.GetLastError(), 'wglDXOpenDeviceNV',
                'the Direct3D device and the OpenGL context are probably on '
                'different GPUs')
        self._textures: list[SharedTexture] = []

    def create_texture(self, width: int, height: int,
                       format: int = d3d11.DXGI_FORMAT_B8G8R8A8_UNORM,
                       access: int = WGL_ACCESS_WRITE_DISCARD_NV) -> SharedTexture:
        """A new texture that OpenGL draws into and Direct3D reads."""
        from OpenGL.GL import GL_TEXTURE_2D, glGenTextures
        from OpenGL.WGL.NV.DX_interop import wglDXRegisterObjectNV
        resource = self.device.create_texture(width, height, format)
        name = int(glGenTextures(1))
        handle = wglDXRegisterObjectNV(self.pointer, resource.pointer, name,
                                       GL_TEXTURE_2D, access)
        if not handle:
            error = ctypes.GetLastError()
            from OpenGL.GL import glDeleteTextures
            glDeleteTextures([name])
            resource.close()
            raise InteropError(error, 'wglDXRegisterObjectNV',
                               f'{width}x{height} DXGI format {format}')
        shared = SharedTexture(self, resource, name, handle)
        self._textures.append(shared)
        return shared

    def close(self) -> None:
        """Close every texture made here, then the bridge. Safe to call twice."""
        for texture in self._textures:
            texture.close()
        self._textures = []
        if self.pointer is not None:
            from OpenGL.WGL.NV.DX_interop import wglDXCloseDeviceNV
            wglDXCloseDeviceNV(self.pointer)
            self.pointer = None

    def __enter__(self) -> InteropDevice:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()
