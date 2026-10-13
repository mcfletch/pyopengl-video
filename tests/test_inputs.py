"""Encoder inputs: the shared rate helpers, and the default way one is made."""
import pytest
from OpenGL.error import GLError
from OpenGL.GL import (
    GL_DRAW_FRAMEBUFFER,
    GL_DRAW_FRAMEBUFFER_BINDING,
    GL_FRAMEBUFFER_COMPLETE,
    GL_PIXEL_UNPACK_BUFFER,
    GL_PIXEL_UNPACK_BUFFER_BINDING,
    GL_STREAM_DRAW,
    GL_TEXTURE_2D,
    GL_TEXTURE_BINDING_2D,
    glBindBuffer,
    glBindFramebuffer,
    glBindTexture,
    glBufferData,
    glCheckFramebufferStatus,
    glDeleteBuffers,
    glDeleteTextures,
    glGenBuffers,
    glGenFramebuffers,
    glGenTextures,
    glGetIntegerv,
    glIsFramebuffer,
    glIsTexture,
)

from pyopengl_video import encoder as encoder_module
from pyopengl_video import inputs
from pyopengl_video.encoder import (
    DEFAULT_BITS_PER_PIXEL,
    Encoder,
    EncoderError,
    default_bitrate,
    frame_rate_ratio,
)
from pyopengl_video.inputs import (
    InputHandle,
    create_framebuffer,
    create_rgba_texture,
    delete_framebuffer,
)

#: The `contexts` fixture: a second OpenGL context beside the suite's.
pytest_plugins = ('tests.other_context',)


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

    def register(self, texture, target=None):  # noqa: ARG002 - the Encoder interface's signature
        handle = InputHandle()
        handle.texture = int(texture)
        self.registered.append(handle)
        return handle

    def encode(self, handle, timestamp, duration=0, force_idr=False):  # noqa: ARG002 - the Encoder interface's signature
        return []

    def flush(self):
        return []

    def headers(self):
        return b''

    def close(self):
        pass


@pytest.mark.usefixtures('gl_context')
def test_the_default_new_input_makes_a_texture_and_a_framebuffer():
    """A backend that accepts a foreign texture gets this for free."""
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


@pytest.mark.usefixtures('gl_context')
def test_closing_an_input_the_encoder_made_gives_the_texture_back():
    encoder = ForeignTextureEncoder()
    handle = encoder.new_input()
    texture = handle.texture
    assert glIsTexture(texture)
    handle.close()
    assert not glIsTexture(texture)
    assert handle.framebuffer == 0


@pytest.mark.usefixtures('gl_context')
def test_building_a_framebuffer_leaves_the_binding_as_it_found_it():
    """A recorder builds its ring mid-frame; it must not disturb the renderer."""
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


@pytest.mark.usefixtures('gl_context')
def test_a_framebuffer_that_cannot_be_built_leaves_the_binding_as_it_found_it():
    """An attachment the driver refuses raises, and the renderer's binding is back."""
    other = int(glGenFramebuffers(1))
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, other)
    try:
        with pytest.raises(GLError):
            create_framebuffer(987654)
        assert int(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING)) == other
    finally:
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        delete_framebuffer(other)


@pytest.mark.parametrize('fps', [60, 30, 29.97, (24000, 1001)])
def test_every_accepted_rate_shape_gives_two_positive_integers(fps):
    numerator, denominator = frame_rate_ratio(fps)
    assert numerator > 0 and denominator > 0


# ------------------------------------------------ the state around a texture


@pytest.mark.usefixtures('gl_context')
def test_making_a_texture_leaves_the_texture_binding_as_it_found_it():
    other = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, other)
    made = 0
    try:
        made = create_rgba_texture(16, 16)
        assert int(glGetIntegerv(GL_TEXTURE_BINDING_2D)) == other
    finally:
        glBindTexture(GL_TEXTURE_2D, 0)
        glDeleteTextures([other, made])


@pytest.mark.usefixtures('gl_context')
def test_a_texture_is_not_filled_from_a_pixel_unpack_buffer_the_renderer_left_bound():
    """With one bound, a null data pointer is an offset into the buffer."""
    buffer = int(glGenBuffers(1))
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer)
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 16, None, GL_STREAM_DRAW)
    made = 0
    try:
        made = create_rgba_texture(64, 64)
        assert int(glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING)) == buffer
    finally:
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        glDeleteBuffers(1, [buffer])
        glDeleteTextures([made])


# ------------------------------------------------- the context of an input


class TestAnotherContextCurrent:
    def handle(self):
        return ForeignTextureEncoder().new_input()

    @pytest.mark.parametrize('current', ['second', None])
    def test_an_input_is_not_deleted_from_another_context(self, contexts, current):
        handle = self.handle()
        texture, framebuffer = handle.texture, handle.framebuffer
        contexts.use(getattr(contexts, current) if current else None)
        with pytest.raises(EncoderError, match='OpenGL context'):
            handle.close()
        contexts.use(contexts.first)
        assert glIsTexture(texture) and glIsFramebuffer(framebuffer)
        handle.close()
        assert not glIsTexture(texture) and not glIsFramebuffer(framebuffer)

    def test_the_other_contexts_texture_of_the_same_name_survives(self, contexts):
        handle = self.handle()
        contexts.use(contexts.second)
        victim = contexts.texture_named(handle.texture)
        with pytest.raises(EncoderError):
            handle.close()
        assert glIsTexture(victim)
        contexts.use(contexts.first)
        handle.close()

    @pytest.mark.usefixtures('gl_context')
    def test_a_closed_input_closes_again_with_no_context_current(self, contexts):
        handle = self.handle()
        handle.close()
        contexts.use(None)
        handle.close()


class TestClosingWhenADeleteFails:
    @pytest.mark.usefixtures('gl_context')
    def test_the_texture_is_deleted_when_the_framebuffer_cannot_be(self, monkeypatch):
        handle = ForeignTextureEncoder().new_input()
        texture = handle.texture
        deleting = inputs.delete_framebuffer

        def refuse(framebuffer):
            deleting(framebuffer)
            raise GLError(err=1282, baseOperation='glDeleteFramebuffers')
        monkeypatch.setattr(inputs, 'delete_framebuffer', refuse)
        with pytest.raises(GLError):
            handle.close()
        assert not glIsTexture(texture)
        assert (handle.framebuffer, handle.texture) == (0, 0)
        handle.close()


@pytest.mark.usefixtures('gl_context')
def test_an_input_that_cannot_be_registered_leaves_no_texture(monkeypatch):
    """The texture is the encoder's own until a handle owns it."""
    made = []
    create = encoder_module.create_rgba_texture

    def creating(width, height):
        made.append(create(width, height))
        return made[-1]

    class Refusing(ForeignTextureEncoder):
        def register(self, texture, target=None):  # noqa: ARG002 - the Encoder interface's signature
            raise EncoderError('the driver would not take the texture')

    monkeypatch.setattr(encoder_module, 'create_rgba_texture', creating)
    with pytest.raises(EncoderError, match='would not take'):
        Refusing().new_input()
    assert made and not glIsTexture(made[0])
