"""Encoder inputs: the shared rate helpers, and the default way one is made."""
import pytest

from pyopengl_video.encoder import (
    DEFAULT_BITS_PER_PIXEL,
    Encoder,
    default_bitrate,
    frame_rate_ratio,
)
from pyopengl_video.inputs import InputHandle


def test_a_whole_frame_rate_is_that_many_over_one():
    assert frame_rate_ratio(60) == (60, 1)
    assert frame_rate_ratio(24) == (24, 1)
    assert frame_rate_ratio(30.0) == (30, 1)


def test_a_broadcast_rate_becomes_the_exact_ratio_for_it():
    """29.97 is 30000/1001, and an approximation of it drifts over a recording."""
    assert frame_rate_ratio(29.97) == (30000, 1001)
    assert frame_rate_ratio(23.976) == (24000, 1001)


def test_a_rate_given_as_a_ratio_is_kept_as_one():
    assert frame_rate_ratio((30000, 1001)) == (30000, 1001)


def test_the_default_bitrate_follows_the_frame_size_and_rate():
    assert default_bitrate((1920, 1080), (60, 1)) == int(
        1920 * 1080 * 60 * DEFAULT_BITS_PER_PIXEL)
    small = default_bitrate((640, 480), (30, 1))
    large = default_bitrate((1920, 1080), (30, 1))
    assert large > small


class ForeignTextureEncoder(Encoder):
    """An encoder that takes a texture the caller made, as the Linux ones do."""

    size = (32, 16)

    def __init__(self):
        self.registered = []

    def register(self, texture, target=None):
        handle = InputHandle()
        handle.texture = int(texture)
        self.registered.append(handle)
        return handle

    def encode(self, handle, timestamp, duration=0, force_idr=False):
        return []

    def flush(self):
        return []

    def headers(self):
        return b''

    def close(self):
        pass


def test_the_default_new_input_makes_a_texture_and_a_framebuffer(gl_context):
    """A backend that accepts a foreign texture gets this for free."""
    from OpenGL.GL import (
        GL_DRAW_FRAMEBUFFER,
        GL_FRAMEBUFFER_COMPLETE,
        glBindFramebuffer,
        glCheckFramebufferStatus,
    )
    encoder = ForeignTextureEncoder()
    assert encoder.allocates_inputs is False
    handle = encoder.new_input()
    try:
        assert handle.texture and handle.framebuffer
        assert handle.owns_texture, 'the encoder made it, so it should free it'
        assert encoder.registered == [handle]
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, handle.framebuffer)
        assert int(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER)) == GL_FRAMEBUFFER_COMPLETE
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        with handle.for_drawing() as held:
            assert held is handle
    finally:
        handle.close()


def test_closing_an_input_the_encoder_made_gives_the_texture_back(gl_context):
    from OpenGL.GL import glIsTexture
    encoder = ForeignTextureEncoder()
    handle = encoder.new_input()
    texture = handle.texture
    assert glIsTexture(texture)
    handle.close()
    assert not glIsTexture(texture)
    assert handle.framebuffer == 0


def test_building_a_framebuffer_leaves_the_binding_as_it_found_it(gl_context):
    """A recorder builds its ring mid-frame; it must not disturb the renderer."""
    from OpenGL.GL import (
        GL_DRAW_FRAMEBUFFER,
        GL_DRAW_FRAMEBUFFER_BINDING,
        glBindFramebuffer,
        glGenFramebuffers,
        glGetIntegerv,
    )

    from pyopengl_video.inputs import delete_framebuffer
    other = int(glGenFramebuffers(1))
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, other)
    encoder = ForeignTextureEncoder()
    handle = encoder.new_input()
    try:
        assert int(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING)) == other
    finally:
        handle.close()
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        delete_framebuffer(other)


@pytest.mark.parametrize('fps', [60, 30, 29.97, (24000, 1001)])
def test_every_accepted_rate_shape_gives_two_positive_integers(fps):
    numerator, denominator = frame_rate_ratio(fps)
    assert numerator > 0 and denominator > 0
