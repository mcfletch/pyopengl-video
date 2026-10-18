"""The surfaces oneVPL allocates through the application, as Direct3D 11 textures.

The runtime asks for its frames -- the reconstructed pictures an encoder predicts
from, the decoded pictures a decoder writes -- by calling back into the five
functions of an ``mfxFrameAllocator``. :class:`SurfaceAllocator` answers them
with textures on the device the session was given, and resolves the textures the
application supplies itself: a ``GetHDL`` on one of those is how the runtime
learns which Direct3D resource a picture is in.
"""
from __future__ import annotations

import ctypes
import logging
from ctypes import c_void_p
from typing import Any

from pyopengl_video.vpl import api
from pyopengl_video.windows import d3d11

log = logging.getLogger(__name__)

#: Rounding the runtime's surfaces demand. H.264 codes in macroblocks.
ALIGNMENT = 16

#: The Direct3D format each pixel format the runtime asks for is made in.
DXGI_FOR_FOURCC = {
    api.MFX_FOURCC_NV12: d3d11.DXGI_FORMAT_NV12,
    api.MFX_FOURCC_P010: d3d11.DXGI_FORMAT_P010,
    api.MFX_FOURCC_RGB4: d3d11.DXGI_FORMAT_B8G8R8A8_UNORM,
    api.MFX_FOURCC_BGR4: d3d11.DXGI_FORMAT_R8G8B8A8_UNORM,
}


def aligned(value: int, to: int = ALIGNMENT) -> int:
    """`value` rounded up to a multiple of `to`."""
    return (int(value) + to - 1) // to * to


class SurfaceAllocator:
    """The surfaces oneVPL asks for, as Direct3D textures on our own device.

    Memory identifiers are the texture pointers themselves, which are unique for
    as long as the texture is alive and are exactly what ``GetHDL`` has to
    return.

    A decoder is given the surfaces it asked for when it was initialised, and
    the application decodes into those: :attr:`latest` names them.
    """

    def __init__(self, device: d3d11.Device):
        self.device = device
        #: Textures this allocator made, by the identifier handed to the runtime.
        self.owned: dict[int, d3d11.Texture] = {}
        #: Textures made elsewhere -- the shared ones -- that it must resolve.
        self.external: dict[int, d3d11.Texture] = {}
        #: The identifiers of the most recent allocation, in the order the
        #: runtime was given them.
        self.latest: list[int] = []
        self._responses: list[Any] = []
        # ctypes callbacks must outlive the C structure that points at them.
        self._alloc = api.ALLOC_FUNC(self.allocate)
        self._lock = api.LOCK_FUNC(self.lock)
        self._unlock = api.LOCK_FUNC(self.lock)
        self._get_handle = api.GETHDL_FUNC(self.get_handle)
        self._free = api.FREE_FUNC(self.free)
        self.structure = api.mfxFrameAllocator()
        self.structure.Alloc = self._alloc
        self.structure.Lock = self._lock
        self.structure.Unlock = self._unlock
        self.structure.GetHDL = self._get_handle
        self.structure.Free = self._free

    def adopt(self, texture: d3d11.Texture) -> int:
        """Take responsibility for resolving `texture`, returning its identifier."""
        identifier = ctypes.cast(texture.pointer, c_void_p).value or 0
        self.external[identifier] = texture
        return identifier

    def forget(self, identifier: int) -> None:
        """Stop resolving the texture with this identifier."""
        self.external.pop(identifier, None)

    # ------------------------------------------------------- the callbacks

    def allocate(self, _pthis: Any, request: Any, response: Any) -> int:
        """Make the surfaces the runtime asked for."""
        wanted = request.contents
        count = max(int(wanted.NumFrameMin), int(wanted.NumFrameSuggested))
        if not count:
            return api.MFX_ERR_NONE
        fourcc = int(wanted.Info.FourCC)
        format = DXGI_FOR_FOURCC.get(fourcc)
        if format is None:
            log.debug('oneVPL asked for %d surfaces of %s, which has no D3D11 format',
                      count, api.fourcc_name(fourcc))
            return -3                                  # MFX_ERR_UNSUPPORTED
        # The video processor reads a decoded picture through a shader
        # resource view as well as the decoder's own binding.
        bind = None
        if wanted.Type & api.MFX_MEMTYPE_FROM_DECODE:
            bind = d3d11.D3D11_BIND_DECODER | d3d11.D3D11_BIND_SHADER_RESOURCE
        identifiers = (c_void_p * count)()
        made = []
        try:
            for slot in range(count):
                texture = self.device.create_texture(
                    aligned(wanted.Info.Width), aligned(wanted.Info.Height), format,
                    bind=bind)
                made.append(texture)
                identifier = ctypes.cast(texture.pointer, c_void_p).value or 0
                identifiers[slot] = c_void_p(identifier)
                self.owned[identifier] = texture
        except Exception as error:                     # noqa: BLE001 - report, not raise
            log.debug('allocating %d %s surfaces failed: %r', count,
                      api.fourcc_name(fourcc), error)
            for texture in made:
                self.owned.pop(ctypes.cast(texture.pointer, c_void_p).value or 0, None)
                texture.close()
            return -4                                  # MFX_ERR_MEMORY_ALLOC
        response.contents.mids = identifiers
        response.contents.NumFrameActual = count
        self.latest = [int(identifiers[slot] or 0) for slot in range(count)]
        # The runtime keeps the array; nothing else here holds a reference.
        self._responses.append(identifiers)
        return api.MFX_ERR_NONE

    def lock(self, _pthis: Any, _mid: Any, _data: Any) -> int:
        """Refuse host access: every surface here lives in video memory."""
        return -3                                      # MFX_ERR_UNSUPPORTED

    def get_handle(self, _pthis: Any, mid: Any, handle: Any) -> int:
        """Give the runtime the Direct3D texture an identifier stands for.

        It expects a pair -- the resource and the index of the subresource
        within it -- and every texture here has exactly one, so the second is
        always zero.
        """
        identifier = int(mid) if mid else 0
        texture = self.owned.get(identifier) or self.external.get(identifier)
        if texture is None:
            log.debug('oneVPL asked for an unknown surface: 0x%x', identifier)
            return api.MFX_ERR_NOT_FOUND
        pair = ctypes.cast(handle, ctypes.POINTER(c_void_p))
        pair[0] = ctypes.cast(texture.pointer, c_void_p)
        pair[1] = None
        return api.MFX_ERR_NONE

    def free(self, _pthis: Any, response: Any) -> int:
        """Give back the surfaces from one allocation."""
        answer = response.contents
        for slot in range(int(answer.NumFrameActual)):
            texture = self.owned.pop(answer.mids[slot] or 0, None)
            if texture is not None:
                texture.close()
        return api.MFX_ERR_NONE

    def close(self) -> None:
        """Release every surface this allocator made."""
        for texture in self.owned.values():
            texture.close()
        self.owned.clear()
        self.external.clear()
        self.latest = []
        self._responses.clear()
