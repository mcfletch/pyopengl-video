"""The Windows interop shim: getting a rendered frame to a Windows encoder.

Every video encoder on Windows -- NVIDIA's, Intel's and AMD's alike -- takes its
input as a Direct3D 11 texture, and none of them takes an OpenGL texture. What
bridges the two is ``WGL_NV_DX_interop2``, which makes one allocation that is
simultaneously a D3D11 texture and a GL texture: the renderer draws into it as
GL, the encoder reads it as D3D11, and nothing is copied. The extension's name
is historical rather than a vendor restriction -- Intel's and AMD's Windows
drivers offer it too.

Three modules, bottom to top:

:mod:`~pyopengl_video.windows.com`
    the handful of COM calls the rest makes, over ctypes.
:mod:`~pyopengl_video.windows.d3d11`
    which GPUs are present, which one a device is built on, and the textures it
    allocates.
:mod:`~pyopengl_video.windows.interop`
    the shared textures, and which adapter the current OpenGL context is on.

**Zero-copy needs the encoder and the OpenGL context on the same adapter.** On a
machine with switchable graphics they need not be:
:func:`~pyopengl_video.windows.interop.adapter_for_context` answers which one
the renderer is using, so a backend can offer itself only where it would
genuinely be reading the memory the renderer wrote.
"""
from __future__ import annotations

from pyopengl_video.windows.com import InteropError

__all__ = ['InteropError']
