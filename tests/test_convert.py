"""The NV12 conversion pass, on planes uploaded from host memory.

Needs an OpenGL context and no decoder. The planes are put in textures larger
than the picture, as a decoder's surfaces are, with the padding filled with
values that would show if they were read.
"""
import ctypes

import numpy as np
import pytest
from OpenGL.GL import (
    GL_ACTIVE_TEXTURE,
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_CURRENT_PROGRAM,
    GL_DRAW_FRAMEBUFFER_BINDING,
    GL_FRAMEBUFFER_SRGB,
    GL_LINEAR,
    GL_R8,
    GL_RED,
    GL_RG,
    GL_RG8,
    GL_RGBA,
    GL_SCISSOR_TEST,
    GL_SRGB8_ALPHA8,
    GL_TEXTURE0,
    GL_TEXTURE1,
    GL_TEXTURE_2D,
    GL_TEXTURE_BINDING_2D,
    GL_TEXTURE_INTERNAL_FORMAT,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_ARRAY_BINDING,
    GL_VIEWPORT,
    glActiveTexture,
    glBindTexture,
    glDeleteTextures,
    glDisable,
    glEnable,
    glGenTextures,
    glGetIntegerv,
    glGetTexImage,
    glGetTexLevelParameteriv,
    glIsEnabled,
    glPixelStorei,
    glTexImage2D,
    glTexParameteri,
    glViewport,
)

from pyopengl_video.colour import (
    MATRIX_BT601,
    MATRIX_BT709,
    MATRIX_BT2020,
    MATRIX_UNSPECIFIED,
    Colour,
)
from pyopengl_video.convert import NV12Converter, create_output_texture

pytestmark = pytest.mark.gl

#: The texture is a surface's allocation; the picture is a corner of it.
TEXTURE = (48, 40)
VISIBLE = (0, 0, 36, 30)
#: What fills the padding, chosen to be far from anything the tests expect.
PADDING = 255


@pytest.fixture
def converter(gl_context):  # noqa: ARG001 - requested so a GL context is current first
    built = NV12Converter()
    yield built
    built.close()


@pytest.fixture
def textures(gl_context):  # noqa: ARG001 - requested so a GL context is current first
    made = []

    def plane(data, internal, external):
        texture = int(glGenTextures(1))
        made.append(texture)
        glBindTexture(GL_TEXTURE_2D, texture)
        for parameter, value in (
            (GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
            (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
        ):
            glTexParameteri(GL_TEXTURE_2D, parameter, value)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        height, width = data.shape[:2]
        glTexImage2D(GL_TEXTURE_2D, 0, internal, width, height, 0, external,
                     GL_UNSIGNED_BYTE, np.ascontiguousarray(data))
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glBindTexture(GL_TEXTURE_2D, 0)
        return texture

    def nv12(luma, chroma):
        """Textures for planes given in surface order, first row at the top."""
        return plane(luma, GL_R8, GL_RED), plane(chroma, GL_RG8, GL_RG)

    def output(width, height):
        texture = create_output_texture(width, height)
        made.append(texture)
        return texture

    nv12.output = output
    yield nv12
    glDeleteTextures(made)


def read(texture, width, height):
    """An RGBA texture's contents, row 0 at the bottom as OpenGL keeps them."""
    glBindTexture(GL_TEXTURE_2D, texture)
    data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
    glBindTexture(GL_TEXTURE_2D, 0)
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)


def planes(y, u, v, texture=TEXTURE, visible=VISIBLE):
    """Constant planes over the visible picture, padding elsewhere."""
    width, height = texture
    x0, y0, w, h = visible
    luma = np.full((height, width), PADDING, np.uint8)
    luma[y0:y0 + h, x0:x0 + w] = y
    chroma = np.full((height // 2, width // 2, 2), PADDING, np.uint8)
    chroma[y0 // 2:(y0 + h) // 2, x0 // 2:(x0 + w) // 2] = (u, v)
    return luma, chroma


def expected(colour, y, u, v):
    rgb = np.array(colour.yuv_to_rgb()) @ np.array([y / 255, u / 255, v / 255, 1.0])
    return np.clip(np.round(rgb * 255), 0, 255)


COLOURS = [(81, 90, 240), (145, 54, 34), (41, 240, 110), (128, 128, 128),
           (16, 128, 128), (235, 128, 128), (200, 60, 200)]


class TestColour:
    @pytest.mark.parametrize('matrix', [MATRIX_BT601, MATRIX_BT709, MATRIX_BT2020])
    @pytest.mark.parametrize('full_range', [False, True])
    def test_every_matrix_and_range_matches_the_arithmetic(
            self, converter, textures, matrix, full_range):
        colour = Colour(matrix=matrix, full_range=full_range)
        width, height = VISIBLE[2:]
        target = textures.output(width, height)
        for y, u, v in COLOURS:
            luma, chroma = textures(*planes(y, u, v))
            converter.draw(luma, chroma, visible=VISIBLE, colour=colour,
                           target=target, size=(width, height))
            found = read(target, width, height)
            want = expected(colour, y, u, v)
            assert np.abs(found[..., :3].astype(int) - want).max() <= 1, (y, u, v)
            assert (found[..., 3] == 255).all()

    def test_an_unspecified_matrix_is_chosen_by_height(self, converter, textures):
        width, height = VISIBLE[2:]
        target = textures.output(width, height)
        luma, chroma = textures(*planes(81, 90, 240))
        converter.draw(luma, chroma, visible=VISIBLE,
                       colour=Colour(matrix=MATRIX_UNSPECIFIED), target=target,
                       size=(width, height))
        want = expected(Colour(matrix=MATRIX_BT601), 81, 90, 240)
        assert np.abs(read(target, width, height)[0, 0, :3].astype(int) - want).max() <= 1


class TestGeometry:
    def test_the_picture_comes_out_the_right_way_up_for_opengl(self, converter, textures):
        """Surface row 0 is the top; texture row 0 is the bottom."""
        width, height = VISIBLE[2:]
        luma, chroma = planes(0, 128, 128)
        luma[:height, :width] = np.arange(height, dtype=np.uint8)[:, None] * 6 + 16
        target = textures.output(width, height)
        converter.draw(*textures(luma, chroma), visible=VISIBLE,
                       colour=Colour(matrix=MATRIX_BT601, full_range=True),
                       target=target, size=(width, height))
        found = read(target, width, height)[..., 0]
        assert (found[0] == luma[height - 1, 0]).all()
        assert (found[height - 1] == luma[0, 0]).all()

    def test_only_the_visible_rectangle_is_read(self, converter, textures):
        visible = (4, 6, 30, 24)
        luma, chroma = planes(100, 128, 128, visible=visible)
        width, height = visible[2:]
        target = textures.output(width, height)
        colour = Colour(matrix=MATRIX_BT601, full_range=True)
        converter.draw(*textures(luma, chroma), visible=visible, colour=colour,
                       target=target, size=(width, height))
        found = read(target, width, height)[..., :3].astype(int)
        assert np.abs(found - expected(colour, 100, 128, 128)).max() <= 1

    def test_chroma_at_the_edges_does_not_reach_into_the_padding(self, converter, textures):
        """The last visible column sits half way to the first padding chroma."""
        luma, chroma = planes(128, 60, 200)
        width, height = VISIBLE[2:]
        target = textures.output(width, height)
        colour = Colour(matrix=MATRIX_BT709)
        converter.draw(*textures(luma, chroma), visible=VISIBLE, colour=colour,
                       target=target, size=(width, height))
        found = read(target, width, height)[..., :3].astype(int)
        assert np.abs(found - expected(colour, 128, 60, 200)).max() <= 1

    def test_chroma_sits_where_the_stream_says(self, converter, textures):
        """Left-sited chroma: luma column 2i is exactly on chroma sample i."""
        luma, chroma = planes(128, 128, 128)
        columns = VISIBLE[2] // 2
        chroma[:, :columns, 0] = (np.arange(columns) * 8 + 40)[None, :]
        target = textures.output(*VISIBLE[2:])
        colour = Colour(matrix=MATRIX_BT601, full_range=True, chroma_location=0)
        converter.draw(*textures(luma, chroma), visible=VISIBLE, colour=colour,
                       target=target, size=VISIBLE[2:])
        blue = read(target, *VISIBLE[2:])[..., 2].astype(int)
        on_sample = [expected(colour, 128, 40 + 8 * i, 128)[2] for i in (3, 4)]
        between = expected(colour, 128, 40 + 8 * 3.5, 128)[2]
        row = blue[10]
        assert abs(row[6] - on_sample[0]) <= 1
        assert abs(row[8] - on_sample[1]) <= 1
        assert abs(row[7] - between) <= 1


class TestState:
    def test_the_pass_leaves_the_state_it_found(self, converter, textures):
        luma, chroma = textures(*planes(100, 128, 128))
        target = textures.output(*VISIBLE[2:])
        bound = textures.output(4, 4)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, bound)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, bound)
        glEnable(GL_BLEND)
        glEnable(GL_SCISSOR_TEST)
        glEnable(GL_FRAMEBUFFER_SRGB)
        glViewport(1, 2, 3, 4)

        def state():
            found = {name: int(glGetIntegerv(name)) for name in (
                GL_CURRENT_PROGRAM, GL_DRAW_FRAMEBUFFER_BINDING,
                GL_VERTEX_ARRAY_BINDING, GL_ACTIVE_TEXTURE, GL_TEXTURE_BINDING_2D)}
            found['viewport'] = tuple(glGetIntegerv(GL_VIEWPORT))
            found['enabled'] = tuple(bool(glIsEnabled(cap)) for cap in (
                GL_BLEND, GL_SCISSOR_TEST, GL_FRAMEBUFFER_SRGB))
            glActiveTexture(GL_TEXTURE1)
            found['unit1'] = int(glGetIntegerv(GL_TEXTURE_BINDING_2D))
            glActiveTexture(found[GL_ACTIVE_TEXTURE])
            return found

        before = state()
        try:
            converter.draw(luma, chroma, visible=VISIBLE, colour=Colour(),
                           target=target, size=VISIBLE[2:])
            assert state() == before
        finally:
            glDisable(GL_BLEND)
            glDisable(GL_SCISSOR_TEST)
            glDisable(GL_FRAMEBUFFER_SRGB)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, 0)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, 0)

    def test_srgb_encoding_is_off_while_it_draws(self, converter, textures):
        """The values are stored as computed; the sampler linearises them."""
        luma, chroma = textures(*planes(128, 128, 128))
        target = textures.output(*VISIBLE[2:])
        glEnable(GL_FRAMEBUFFER_SRGB)
        try:
            converter.draw(luma, chroma, visible=VISIBLE,
                           colour=Colour(matrix=MATRIX_BT601, full_range=True),
                           target=target, size=VISIBLE[2:])
        finally:
            glDisable(GL_FRAMEBUFFER_SRGB)
        assert abs(int(read(target, *VISIBLE[2:])[0, 0, 0]) - 128) <= 1


class TestOutputTextures:
    def test_an_output_texture_is_srgb(self, textures):
        texture = textures.output(16, 8)
        glBindTexture(GL_TEXTURE_2D, texture)
        internal = ctypes.c_int()
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, internal)
        glBindTexture(GL_TEXTURE_2D, 0)
        assert internal.value == GL_SRGB8_ALPHA8
