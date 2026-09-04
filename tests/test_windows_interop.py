"""The Windows interop shim: adapter identity, D3D11 devices, shared textures.

The shared-texture tests are the ones that matter: they prove that a texture the
renderer draws into is the same memory the encoder will read, and they settle
which byte order arrives there -- which is what decides the pixel format the
Windows backends declare.
"""
import sys

import numpy as np
import pytest

from pyopengl_video.windows import d3d11, interop

pytestmark = pytest.mark.skipif(sys.platform != 'win32',
                                reason='the interop shim is Windows-only')


@pytest.fixture
def adapters():
    found = d3d11.adapters()
    if not found:
        pytest.skip('no DXGI adapters here')
    return found


@pytest.fixture
def gl_adapter(gl_context, adapters):
    """The DXGI adapter the current OpenGL context is running on."""
    adapter = interop.adapter_for_context()
    if adapter is None:
        pytest.skip('this GL driver does not report a device LUID')
    return adapter


@pytest.fixture
def device(gl_adapter):
    with d3d11.Device(gl_adapter) as opened:
        yield opened


@pytest.fixture
def shared(gl_context, device):
    if not interop.available():
        pytest.skip('WGL_NV_DX_interop2 is not offered by this driver')
    with interop.InteropDevice(device) as interop_device:
        yield interop_device


# ---------------------------------------------------------------- adapters

def test_adapters_are_listed_with_a_vendor_and_a_luid(adapters):
    for adapter in adapters:
        assert adapter.luid and len(adapter.luid) == 8
        assert adapter.description
        assert adapter.vendor


def test_each_adapter_has_a_distinct_luid(adapters):
    luids = [adapter.luid for adapter in adapters]
    assert len(set(luids)) == len(luids)


@pytest.mark.parametrize('vendor_id, name', [
    (0x10DE, 'NVIDIA'), (0x8086, 'Intel'), (0x1002, 'AMD'), (0x1414, 'Microsoft'),
])
def test_known_vendor_ids_get_names(vendor_id, name):
    assert d3d11.vendor_name(vendor_id) == name


def test_an_unknown_vendor_id_is_reported_as_its_number():
    assert '0x' in d3d11.vendor_name(0xBEEF)


def test_adapter_for_luid_finds_the_one_it_names(adapters):
    wanted = adapters[0]
    assert d3d11.adapter_for_luid(wanted.luid) == wanted
    assert d3d11.adapter_for_luid(b'\xff' * 8) is None


# ------------------------------------------------------------------ device

def test_a_device_opens_on_the_adapter_it_was_given(gl_adapter):
    with d3d11.Device(gl_adapter) as device:
        assert device.adapter == gl_adapter
        assert device.pointer


def test_closing_a_device_twice_is_harmless(gl_adapter):
    device = d3d11.Device(gl_adapter)
    device.close()
    device.close()
    assert not device.pointer


def test_a_texture_reports_the_size_it_was_made_with(device):
    with device.create_texture(64, 32) as texture:
        assert (texture.width, texture.height) == (64, 32)
        assert texture.pointer


# --------------------------------------------------------- shared textures

def test_the_context_runs_on_an_adapter_dxgi_also_lists(gl_adapter, adapters):
    assert gl_adapter in adapters


def test_a_shared_texture_is_a_complete_framebuffer(shared):
    from OpenGL.GL import (
        GL_DRAW_FRAMEBUFFER,
        GL_FRAMEBUFFER_COMPLETE,
        glBindFramebuffer,
        glCheckFramebufferStatus,
    )
    with shared.create_texture(64, 64) as texture:
        with texture.for_drawing():
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, texture.framebuffer)
            assert int(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER)) == GL_FRAMEBUFFER_COMPLETE
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)


@pytest.mark.parametrize('dxgi_format, red_at, blue_at', [
    (d3d11.DXGI_FORMAT_B8G8R8A8_UNORM, 2, 0),
    (d3d11.DXGI_FORMAT_R8G8B8A8_UNORM, 0, 2),
])
def test_a_shared_texture_carries_what_opengl_drew_into_it(
        shared, dxgi_format, red_at, blue_at):
    """The whole premise: OpenGL writes, and Direct3D reads the same pixels.

    It also settles the byte order, which is what an encoder is told the surface
    holds. **The interop is faithful to the format rather than swizzling**: red
    written by OpenGL arrives in the texture's red component, which lands at
    whichever byte that format puts it. So a `B8G8R8A8_UNORM` texture holds BGRA
    bytes and an `R8G8B8A8_UNORM` one holds RGBA, and the backends name the
    matching pixel format for each.

    A colour with three different components is used on purpose: a swapped red
    and blue cannot pass.
    """
    from OpenGL.GL import (
        GL_COLOR_BUFFER_BIT,
        GL_DRAW_FRAMEBUFFER,
        glBindFramebuffer,
        glClear,
        glClearColor,
        glFinish,
    )
    with shared.create_texture(16, 8, dxgi_format) as texture:
        with texture.for_drawing():
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, texture.framebuffer)
            glClearColor(1.0, 0.5, 0.0, 1.0)          # red 255, green 128, blue 0
            glClear(GL_COLOR_BUFFER_BIT)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
            glFinish()
        pixels = np.frombuffer(texture.read(), dtype=np.uint8).reshape(8, 16, 4)

    assert (pixels[..., red_at] == 255).all(), 'the red OpenGL drew is not where it should be'
    assert (pixels[..., blue_at] == 0).all(), 'the blue OpenGL drew is not where it should be'
    assert abs(int(pixels[0, 0, 1]) - 128) <= 2, 'green is always the second byte'
    assert (pixels[..., 3] == 255).all()


def test_a_shared_texture_can_be_drawn_into_more_than_once(shared):
    """A recording locks and unlocks the same texture every time round the ring."""
    from OpenGL.GL import (
        GL_COLOR_BUFFER_BIT,
        GL_DRAW_FRAMEBUFFER,
        glBindFramebuffer,
        glClear,
        glClearColor,
        glFinish,
    )
    with shared.create_texture(8, 8) as texture:
        for level in (0.0, 1.0, 0.25):
            with texture.for_drawing():
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, texture.framebuffer)
                glClearColor(level, level, level, 1.0)
                glClear(GL_COLOR_BUFFER_BIT)
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
                glFinish()
            pixels = np.frombuffer(texture.read(), dtype=np.uint8).reshape(8, 8, 4)
            assert abs(int(pixels[0, 0, 0]) - round(level * 255)) <= 2


def test_closing_a_shared_texture_twice_is_harmless(shared):
    texture = shared.create_texture(8, 8)
    texture.close()
    texture.close()
    assert texture.texture == 0
