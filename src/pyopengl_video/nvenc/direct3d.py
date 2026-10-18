"""NVENC over Direct3D 11: how an OpenGL frame reaches the encoder on Windows.

NVIDIA supports the encoder's OpenGL device type on Linux alone. On Windows the
session is opened on a Direct3D 11 device and reads ``ID3D11Texture2D``
resources, which OpenGL cannot create -- so the encoder makes the surfaces
itself, through :mod:`pyopengl_video.windows.interop`, and each is a GL texture
and a D3D11 texture at the same time. The renderer blits into the GL name inside
the handle's ``for_drawing`` scope and the encoder reads the D3D11 texture; there
is no copy between them.

The D3D11 device and the OpenGL context must be on the same GPU, and for NVENC
that GPU must be NVIDIA's. On a laptop with switchable graphics a process gets
the integrated part by default; the application's graphics preference in
Windows' display settings chooses the NVIDIA one.
"""
from __future__ import annotations

import ctypes
import logging

from pyopengl_video.backend import context_kind
from pyopengl_video.encoder import EncoderError
from pyopengl_video.nvenc import NO_NVIDIA_GPU, api
from pyopengl_video.nvenc.encoder import RegisteredInput
from pyopengl_video.windows import d3d11, interop

log = logging.getLogger(__name__)

#: The Direct3D format the input surfaces are made in. ``R8G8B8A8_UNORM`` holds
#: the bytes in the order an OpenGL ``GL_RGBA8`` texture does, which NVENC calls
#: ``ABGR``: it is a word-ordered name, and the word is read little-endian.
SURFACE_FORMAT = d3d11.DXGI_FORMAT_R8G8B8A8_UNORM
BUFFER_FORMAT = api.NV_ENC_BUFFER_FORMAT_ABGR

NVIDIA = 'NVIDIA'

#: Why a context that renders elsewhere cannot feed the encoder.
NOT_ON_AN_NVIDIA_GPU = (
    'the current OpenGL context does not render on an NVIDIA GPU, and NVENC '
    'reads only what the GPU that holds the encoder has drawn; on a laptop with '
    'switchable graphics, set this program\'s graphics preference to the NVIDIA '
    'GPU in Windows\' display settings')


class DirectXInput(RegisteredInput, interop.SharedTexture):
    """A shared texture the encoder has registered.

    :attr:`registration` is NVENC's handle for the Direct3D texture, which is
    what mapping and encoding name.
    """


def unavailable_because() -> str:
    """Why the encoder cannot serve the current OpenGL context, or '' where it can.

    With no context current the answer is whether an NVIDIA GPU is installed,
    since discovery often runs before a window exists. With one, the context
    has to share Direct3D textures and run on an NVIDIA GPU.
    """
    if context_kind() is None:
        if any(found.vendor == NVIDIA for found in d3d11.adapters()):
            return ''
        return NO_NVIDIA_GPU
    if not interop.available():
        return f'this OpenGL context does not offer {interop.EXTENSION}'
    adapter = interop.adapter_for_context()
    if adapter is None:
        return 'this OpenGL driver does not say which GPU the context is on'
    if adapter.vendor != NVIDIA:
        return NOT_ON_AN_NVIDIA_GPU
    return ''


def choose_adapter(adapter: d3d11.Adapter | None) -> d3d11.Adapter:
    """The GPU to encode on: `adapter` if one is named, else the context's own.

    Refuses a GPU that is not NVIDIA's, and a context whose GPU cannot be told.
    """
    if adapter is None:
        adapter = interop.adapter_for_context()
        if adapter is None:
            raise EncoderError(
                'cannot tell which GPU this OpenGL context is on, so an encoder '
                'here might not be able to read what the renderer draws; name an '
                'adapter explicitly to go ahead anyway')
    if adapter.vendor != NVIDIA:
        raise EncoderError(f'{adapter} is not an NVIDIA GPU. ' + NOT_ON_AN_NVIDIA_GPU)
    return adapter


def discovery_adapter() -> d3d11.Adapter:
    """The NVIDIA GPU to ask about codecs: the context's, else the first installed."""
    if context_kind() is not None:
        found = interop.adapter_for_context()
        if found is not None and found.vendor == NVIDIA:
            return found
    for installed in d3d11.adapters():
        if installed.vendor == NVIDIA:
            return installed
    raise EncoderError(NO_NVIDIA_GPU)


class Direct3DInputs:
    """The Direct3D 11 device behind an encoder session, and the textures it reads.

    adapter -- which GPU to build on; by default the one the current OpenGL
        context runs on

    An OpenGL context must be current when this is built, and whenever a texture
    is made, drawn into or closed.
    """

    def __init__(self, adapter: d3d11.Adapter | None = None):
        self.adapter = choose_adapter(adapter)
        self.device: d3d11.Device | None = d3d11.Device(self.adapter)
        try:
            self.interop: interop.InteropDevice | None = interop.InteropDevice(self.device)
        except Exception:
            self.device.close()
            raise

    @property
    def pointer(self) -> ctypes.c_void_p:
        """The ``ID3D11Device`` a session is opened on."""
        assert self.device is not None
        return self.device.pointer

    def create(self, width: int, height: int) -> DirectXInput:
        """A new input of this size, drawable by OpenGL and readable by NVENC."""
        assert self.interop is not None
        made = self.interop.create_texture(
            width, height, SURFACE_FORMAT, factory=DirectXInput)
        assert isinstance(made, DirectXInput)
        return made

    def close(self) -> None:
        """Close every texture made here, then the bridge and the device."""
        if self.interop is not None:
            self.interop.close()
            self.interop = None
        if self.device is not None:
            self.device.close()
            self.device = None
