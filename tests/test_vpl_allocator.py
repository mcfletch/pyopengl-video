"""The surfaces oneVPL asks the backend to allocate on its behalf.

The runtime calls these back from inside its own code, where nothing can watch
them, so they are driven directly here: build a request, call the callback, and
look at what it produced. It needs a Direct3D device but no OpenGL context and
no encoder.
"""
import ctypes
import sys

import pytest

from pyopengl_video.vpl import api
from pyopengl_video.vpl.encoder import SurfaceAllocator
from pyopengl_video.windows import d3d11

pytestmark = pytest.mark.skipif(sys.platform != 'win32',
                                reason='Direct3D surfaces are Windows-only')


@pytest.fixture
def device():
    adapters = d3d11.adapters()
    if not adapters:
        pytest.skip('no DXGI adapters here')
    with d3d11.Device(adapters[0], video=False) as opened:
        yield opened


@pytest.fixture
def allocator(device):
    made = SurfaceAllocator(device)
    yield made
    made.close()


def request_for(fourcc, width=320, height=240, count=3):
    """An allocation request of the shape the runtime makes."""
    wanted = api.mfxFrameAllocRequest()
    wanted.Info.FourCC = fourcc
    wanted.Info.Width, wanted.Info.Height = width, height
    wanted.Info.CropW, wanted.Info.CropH = width, height
    wanted.NumFrameMin = count
    wanted.NumFrameSuggested = count
    wanted.Type = api.MFX_MEMTYPE_FROM_ENCODE | api.MFX_MEMTYPE_INTERNAL_FRAME
    return ctypes.pointer(wanted)


def test_it_makes_the_surfaces_that_were_asked_for(allocator):
    response = api.mfxFrameAllocResponse()
    status = allocator.allocate(None, request_for(api.MFX_FOURCC_NV12),
                                ctypes.pointer(response))
    assert status == api.MFX_ERR_NONE
    assert response.NumFrameActual == 3
    assert len(allocator.owned) == 3
    for slot in range(3):
        assert response.mids[slot]


def test_the_surfaces_it_makes_can_be_resolved_back_to_a_texture(allocator):
    response = api.mfxFrameAllocResponse()
    allocator.allocate(None, request_for(api.MFX_FOURCC_RGB4),
                       ctypes.pointer(response))
    pair = (ctypes.c_void_p * 2)()
    status = allocator.get_handle(None, response.mids[0], ctypes.pointer(pair))
    assert status == api.MFX_ERR_NONE
    assert pair[0] == response.mids[0]
    assert pair[1] is None, 'a texture with one subresource reports index zero'


def test_a_surface_made_elsewhere_can_be_adopted_and_resolved(allocator, device):
    """The shared textures are made by the interop layer, not by this."""
    with device.create_texture(64, 64) as texture:
        identifier = allocator.adopt(texture)
        pair = (ctypes.c_void_p * 2)()
        assert allocator.get_handle(None, identifier, ctypes.pointer(pair)) == 0
        assert pair[0] == identifier
        allocator.forget(identifier)
        assert allocator.get_handle(
            None, identifier, ctypes.pointer(pair)) == api.MFX_ERR_NOT_FOUND


def test_an_unknown_surface_is_reported_as_not_found(allocator):
    pair = (ctypes.c_void_p * 2)()
    status = allocator.get_handle(None, 0xDEAD0000, ctypes.pointer(pair))
    assert status == api.MFX_ERR_NOT_FOUND


def test_a_format_with_no_direct3d_equivalent_is_refused(allocator):
    response = api.mfxFrameAllocResponse()
    status = allocator.allocate(None, request_for(api.fourcc('P010')),
                                ctypes.pointer(response))
    assert status == -3                                # MFX_ERR_UNSUPPORTED
    assert not allocator.owned


def test_asking_for_no_surfaces_succeeds_without_making_any(allocator):
    response = api.mfxFrameAllocResponse()
    status = allocator.allocate(None, request_for(api.MFX_FOURCC_NV12, count=0),
                                ctypes.pointer(response))
    assert status == api.MFX_ERR_NONE
    assert not allocator.owned


def test_host_access_is_refused_because_the_surfaces_are_in_video_memory(allocator):
    assert allocator.lock(None, 0, None) == -3         # MFX_ERR_UNSUPPORTED


def test_freeing_a_response_gives_its_surfaces_back(allocator):
    response = api.mfxFrameAllocResponse()
    allocator.allocate(None, request_for(api.MFX_FOURCC_NV12),
                       ctypes.pointer(response))
    assert allocator.free(None, ctypes.pointer(response)) == api.MFX_ERR_NONE
    assert not allocator.owned


def test_the_callbacks_are_reachable_through_the_structure_it_hands_over(allocator):
    """The runtime is given a struct of function pointers, and it must be filled."""
    for name in ('Alloc', 'Lock', 'Unlock', 'GetHDL', 'Free'):
        assert getattr(allocator.structure, name), f'{name} is not set'


def test_the_surfaces_are_rounded_up_to_whole_macroblocks(allocator):
    """H.264 codes in sixteens, and a surface has to be at least that big."""
    response = api.mfxFrameAllocResponse()
    allocator.allocate(None, request_for(api.MFX_FOURCC_RGB4, width=300, height=200),
                       ctypes.pointer(response))
    texture = next(iter(allocator.owned.values()))
    assert (texture.width, texture.height) == (304, 208)
