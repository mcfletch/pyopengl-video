"""NVENC reached from OpenGL through Direct3D 11, which is how Windows does it.

These need a machine whose OpenGL context runs on an NVIDIA GPU. The cases about
a context on another GPU stand in for one, so they run anywhere on Windows.
"""
import sys

import pytest
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DRAW_FRAMEBUFFER,
    glBindFramebuffer,
    glClear,
    glClearColor,
)

from pyopengl_video import encoders, nvenc
from pyopengl_video.encoder import EncoderError
from pyopengl_video.nvenc import encoder as nvenc_encoder
from pyopengl_video.nvenc.encoder import NVENCEncoder

if sys.platform == 'win32':
    from pyopengl_video.nvenc import direct3d
    from pyopengl_video.windows import d3d11, interop

pytestmark = pytest.mark.skipif(sys.platform != 'win32',
                                reason='Direct3D 11 is how NVENC is reached on Windows')

SIZE = (320, 240)


def adapter(vendor_id, name):
    """A GPU as DXGI describes it, for a context to claim to be running on."""

    return d3d11.Adapter(index=0, description=name, vendor_id=vendor_id,
                         device_id=1, luid=b'\x01' * 8)


@pytest.fixture
def intel_context(monkeypatch):
    """A current context that claims to run on an Intel GPU."""

    monkeypatch.setattr(direct3d, 'context_kind', lambda: 'other')
    monkeypatch.setattr(interop, 'available', lambda: True)
    monkeypatch.setattr(interop, 'adapter_for_context',
                        lambda: adapter(0x8086, 'Intel(R) UHD Graphics 630'))


@pytest.fixture
def encoder(nvenc_available):  # noqa: ARG001 - requested to skip where the context is not on NVIDIA
    built = NVENCEncoder(*SIZE, fps=30, bitrate=2_000_000)
    yield built
    built.close()


def draw(handle, colour):
    """Fill an input with one colour, the way a recorder's blit would."""
    with handle.for_drawing():
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, handle.framebuffer)
        glClearColor(*colour)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)


class TestWhenTheContextIsOnAnotherGPU:
    def test_the_backend_is_unavailable(self, intel_context):  # noqa: ARG002 - requested to install the stand-in
        assert nvenc.probe() is False

    def test_it_says_which_preference_to_change(self, intel_context):  # noqa: ARG002 - requested to install the stand-in
        reason = nvenc.explain()
        assert 'NVIDIA' in reason
        assert 'graphics preference' in reason

    def test_the_encoder_is_refused_by_name(self, intel_context):  # noqa: ARG002 - requested to install the stand-in
        with pytest.raises(EncoderError) as caught:
            NVENCEncoder(*SIZE)
        assert 'Intel' in str(caught.value)
        assert 'not an NVIDIA GPU' in str(caught.value)

    def test_a_context_whose_gpu_is_unknown_is_not_guessed_at(self, monkeypatch):

        monkeypatch.setattr(interop, 'adapter_for_context', lambda: None)
        with pytest.raises(EncoderError, match='cannot tell which GPU'):
            direct3d.choose_adapter(None)

    def test_a_driver_without_the_interop_extension_is_named(self, monkeypatch):

        monkeypatch.setattr(direct3d, 'context_kind', lambda: 'other')
        monkeypatch.setattr(interop, 'available', lambda: False)
        assert interop.EXTENSION in direct3d.unavailable_because()


def test_with_no_context_a_machine_is_judged_by_its_hardware(monkeypatch):

    monkeypatch.setattr(direct3d, 'context_kind', lambda: None)
    monkeypatch.setattr(d3d11, 'adapters', lambda: [adapter(0x8086, 'Intel')])
    assert direct3d.unavailable_because() == nvenc.NO_NVIDIA_GPU
    monkeypatch.setattr(d3d11, 'adapters', lambda: [adapter(0x8086, 'Intel'),
                                                    adapter(0x10DE, 'GeForce')])
    assert direct3d.unavailable_because() == ''


def test_a_machine_with_no_nvidia_gpu_says_nothing_to_act_on(monkeypatch):

    monkeypatch.setattr(direct3d, 'unavailable_because', lambda: nvenc.NO_NVIDIA_GPU)
    assert nvenc.explain() == ''


class TestOnTheNVIDIAGPU:
    def test_the_backend_is_discovered(self, nvenc_available):  # noqa: ARG002 - requested to skip off NVIDIA
        found = {backend.name: backend for backend in encoders()}
        assert 'h264' in found['nvenc'].codecs

    def test_the_encoder_runs_on_the_adapter_the_context_uses(self, encoder):

        assert encoder.direct3d.adapter == interop.adapter_for_context()
        assert encoder.zero_copy and encoder.allocates_inputs

    def test_a_foreign_texture_is_refused_with_the_way_forward(self, encoder):
        with pytest.raises(EncoderError, match='new_input'):
            encoder.register(7)

    def test_an_adapter_may_be_named(self, nvenc_available):  # noqa: ARG002 - requested to skip off NVIDIA

        adapter = interop.adapter_for_context()
        with NVENCEncoder(*SIZE, adapter=adapter) as named:
            assert named.direct3d.adapter == adapter

    def test_an_input_is_a_drawable_texture_of_the_frame_size(self, encoder):
        handle = encoder.new_input()
        assert handle.texture and handle.framebuffer
        assert handle.size == SIZE

    def test_what_is_drawn_is_what_the_encoder_reads(self, encoder):
        """The bytes on the Direct3D side are the colour OpenGL drew, in RGBA order."""
        handle = encoder.new_input()
        draw(handle, (1.0, 0.0, 0.0, 1.0))
        pixels = handle.read()
        assert pixels[:4] == bytes((255, 0, 0, 255))
        assert len(pixels) == SIZE[0] * SIZE[1] * 4

    def test_the_encoder_gives_back_everything_it_made(self, nvenc_available):  # noqa: ARG002 - requested to skip off NVIDIA
        built = NVENCEncoder(*SIZE)
        handles = [built.new_input() for _ in range(built.input_slots)]
        built.close()
        assert built.direct3d is None
        assert all(handle.texture == 0 for handle in handles)


@pytest.mark.usefixtures('nvenc_available')
def test_an_adapter_is_refused_off_windows(monkeypatch):
    monkeypatch.setattr(nvenc_encoder, 'DIRECTX_PLATFORM', False)
    with pytest.raises(EncoderError, match='only on Windows'):
        NVENCEncoder(*SIZE, adapter=adapter(0x10DE, 'GeForce'))
