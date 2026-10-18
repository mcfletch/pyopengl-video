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
    GL_R16,
    GL_RED,
    GL_RG,
    GL_RG8,
    GL_RG16,
    GL_RGBA,
    GL_RGBA8,
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
    GL_UNSIGNED_SHORT,
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
    glIsProgram,
    glIsVertexArray,
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
from pyopengl_video.decoder import DecodedFrame, DecoderError

#: The `contexts` fixture: two more OpenGL contexts beside the suite's.
pytest_plugins = ('tests.other_context',)

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

    def plane(data, internal, external, kind=GL_UNSIGNED_BYTE):
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
                     kind, np.ascontiguousarray(data))
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glBindTexture(GL_TEXTURE_2D, 0)
        return texture

    def nv12(luma, chroma):
        """Textures for planes given in surface order, first row at the top."""
        return plane(luma, GL_R8, GL_RED), plane(chroma, GL_RG8, GL_RG)

    def p010(luma, chroma):
        """Sixteen-bit textures holding 10-bit planes in their high bits."""
        return (plane((luma.astype(np.uint16) << 6), GL_R16, GL_RED, GL_UNSIGNED_SHORT),
                plane((chroma.astype(np.uint16) << 6), GL_RG16, GL_RG, GL_UNSIGNED_SHORT))

    nv12.p010 = p010

    def rgba(data):
        """A texture holding an RGBA picture given first row at the top."""
        return plane(data, GL_RGBA8, GL_RGBA)

    nv12.rgba = rgba

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


def planes(y, u, v, texture=TEXTURE, visible=VISIBLE, dtype=np.uint8):
    """Constant planes over the visible picture, padding elsewhere."""
    width, height = texture
    x0, y0, w, h = visible
    luma = np.full((height, width), PADDING, dtype)
    luma[y0:y0 + h, x0:x0 + w] = y
    chroma = np.full((height // 2, width // 2, 2), PADDING, dtype)
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

    @pytest.mark.parametrize('full_range', [False, True])
    def test_ten_bit_planes_match_the_arithmetic(self, converter, textures, full_range):
        colour = Colour(matrix=MATRIX_BT709, full_range=full_range)
        width, height = VISIBLE[2:]
        target = textures.output(width, height)
        for y, u, v in ((324, 360, 960), (64, 512, 512), (940, 512, 512), (801, 243, 799)):
            luma, chroma = textures.p010(*planes(y, u, v, dtype=np.uint16))
            converter.draw(luma, chroma, visible=VISIBLE, colour=colour,
                           target=target, size=(width, height), depth=10)
            rgb = np.array(colour.yuv_to_rgb(depth=10)) @ np.array(
                [y / 1023, u / 1023, v / 1023, 1.0])
            want = np.clip(np.round(rgb * 255), 0, 255)
            found = read(target, width, height)
            assert np.abs(found[..., :3].astype(int) - want).max() <= 1, (y, u, v)

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


class TestRGBAFrames:
    """A picture a decoder converted itself is copied: cropped and flipped, nothing more."""

    def picture(self, textures):
        """A 48x40 RGBA picture whose red is the row and green the column."""
        width, height = TEXTURE
        data = np.zeros((height, width, 4), np.uint8)
        data[..., 0] = np.arange(height, dtype=np.uint8)[:, None] * 5
        data[..., 1] = np.arange(width, dtype=np.uint8)[None, :] * 4
        data[..., 3] = 255
        return data, textures.rgba(data)

    def test_the_picture_is_cropped_and_comes_out_the_right_way_up(self, converter, textures):
        data, texture = self.picture(textures)
        visible = (4, 6, 30, 24)
        width, height = visible[2:]
        target = textures.output(width, height)
        converter.copy(texture, visible=visible, target=target)
        found = read(target, width, height)
        x, y = visible[:2]
        assert (found[..., :3] == data[y:y + height, x:x + width, :3][::-1]).all()

    def test_a_frame_of_that_layout_goes_through_convert(self, converter, textures):
        data, texture = self.picture(textures)
        width, height = VISIBLE[2:]
        target = textures.output(width, height)
        frame = DecodedFrame(timestamp=0, duration=1, planes=(texture,), layout='rgba',
                             texture_size=TEXTURE, visible=VISIBLE, colour=Colour(),
                             on_release=lambda _frame: None)
        converter.convert(frame, target)
        assert (read(target, width, height)[..., :3] == data[:height, :width, :3][::-1]).all()

    def test_the_copy_leaves_the_state_it_found(self, converter, textures):
        data, texture = self.picture(textures)
        width, height = VISIBLE[2:]
        target = textures.output(width, height)
        glEnable(GL_BLEND)
        glViewport(1, 2, 3, 4)
        before = (int(glGetIntegerv(GL_CURRENT_PROGRAM)),
                  int(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING)),
                  int(glGetIntegerv(GL_VERTEX_ARRAY_BINDING)),
                  [int(value) for value in glGetIntegerv(GL_VIEWPORT)],
                  int(glGetIntegerv(GL_TEXTURE_BINDING_2D)))
        try:
            converter.copy(texture, visible=VISIBLE, target=target)
            after = (int(glGetIntegerv(GL_CURRENT_PROGRAM)),
                     int(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING)),
                     int(glGetIntegerv(GL_VERTEX_ARRAY_BINDING)),
                     [int(value) for value in glGetIntegerv(GL_VIEWPORT)],
                     int(glGetIntegerv(GL_TEXTURE_BINDING_2D)))
            assert after == before
            assert glIsEnabled(GL_BLEND)
        finally:
            glDisable(GL_BLEND)

    @pytest.mark.parametrize('visible', [
        (0, 0, 49, 30), (0, 0, 36, 41), (20, 0, 36, 30), (-1, 0, 36, 30),
        (0, 0, 0, 30), (0, 0, 36.5, 30),
    ])
    def test_a_rectangle_that_is_not_inside_the_texture_is_refused(
            self, converter, textures, visible):
        _, texture = self.picture(textures)
        with pytest.raises(DecoderError, match='visible'):
            converter.copy(texture, visible=visible, target=textures.output(8, 8),
                           size=(8, 8), texture_size=TEXTURE)

    @pytest.mark.parametrize('size', [(37, 30), (36, 31), (0, 30)])
    def test_a_size_the_rectangle_does_not_cover_is_refused(self, converter, textures, size):
        _, texture = self.picture(textures)
        with pytest.raises(DecoderError, match='size'):
            converter.copy(texture, visible=VISIBLE, target=textures.output(8, 8),
                           size=size, texture_size=TEXTURE)

    def test_a_frame_whose_rectangle_leaves_its_texture_is_refused(self, converter, textures):
        _, texture = self.picture(textures)
        frame = DecodedFrame(timestamp=0, duration=0, planes=(texture,), layout='rgba',
                             texture_size=TEXTURE, visible=(0, 0, 64, 64), colour=Colour(),
                             on_release=lambda _frame: None)
        with pytest.raises(DecoderError, match='visible'):
            converter.convert(frame, textures.output(64, 64))

    def test_the_shader_holds_the_fetch_to_the_texture(self, converter, textures):
        """A rectangle past the texture, with no texture size to refuse it by."""
        data = np.full((16, 16, 4), (200, 100, 50, 255), np.uint8)
        texture = textures.rgba(data)
        target = textures.output(32, 32)
        converter.copy(texture, visible=(0, 0, 32, 32), target=target)
        assert (read(target, 32, 32)[..., :3] == (200, 100, 50)).all()

    def test_the_shader_holds_the_fetch_to_the_visible_rectangle(self, converter, textures):
        """The pass itself, drawn larger than its rectangle: the padding is not read."""
        data = np.full((*TEXTURE[::-1], 4), (PADDING, PADDING, PADDING, 255), np.uint8)
        data[:VISIBLE[3], :VISIBLE[2]] = (200, 100, 50, 255)
        texture = textures.rgba(data)
        target = textures.output(*TEXTURE)
        converter._copy_pass(texture, VISIBLE, target, TEXTURE)  # noqa: SLF001 - the shader, with the refusals in front of it stepped around
        assert (read(target, *TEXTURE)[..., :3] == (200, 100, 50)).all()

    def test_a_frame_that_is_not_one_of_the_layouts_is_refused(self, converter, textures):
        frame = DecodedFrame(timestamp=0, duration=1, planes=(1,), layout='yuy2',
                             texture_size=TEXTURE, visible=VISIBLE, colour=Colour(),
                             on_release=lambda _frame: None)
        with pytest.raises(DecoderError, match='yuy2 frame has no conversion'):
            converter.convert(frame, textures.output(4, 4))


class TestWhatIsFetched:
    """No rectangle or size makes the pass read outside the luma texture."""

    COLOUR = Colour(matrix=MATRIX_BT601, full_range=True)

    @pytest.mark.parametrize('visible', [
        (0, 0, 49, 30), (0, 0, 36, 41), (20, 0, 36, 30), (0, 20, 36, 30),
        (-1, 0, 36, 30), (0, -1, 36, 30), (0, 0, 0, 30), (0, 0, 36, 0),
        (0, 0, -136, -136), (0, 0, 36.5, 30),
    ])
    def test_a_rectangle_that_is_not_inside_the_texture_is_refused(
            self, converter, textures, visible):
        luma, chroma = textures(*planes(100, 128, 128))
        target = textures.output(8, 8)
        with pytest.raises(DecoderError, match='visible'):
            converter.draw(luma, chroma, visible=visible, colour=self.COLOUR,
                           target=target, size=(8, 8), texture_size=TEXTURE)

    @pytest.mark.parametrize('visible', [(0, 0, 0, 30), (0, 0, -136, -136), (-4, 0, 36, 30)])
    def test_an_empty_or_negative_rectangle_is_refused_whatever_the_texture(
            self, converter, textures, visible):
        luma, chroma = textures(*planes(100, 128, 128))
        with pytest.raises(DecoderError, match='visible'):
            converter.draw(luma, chroma, visible=visible, colour=self.COLOUR,
                           target=textures.output(8, 8), size=(8, 8))

    @pytest.mark.parametrize('size', [(37, 30), (36, 31), (640, 480), (0, 30), (36, -1)])
    def test_a_size_the_rectangle_does_not_cover_is_refused(self, converter, textures, size):
        luma, chroma = textures(*planes(100, 128, 128))
        with pytest.raises(DecoderError, match='size'):
            converter.draw(luma, chroma, visible=VISIBLE, colour=self.COLOUR,
                           target=textures.output(8, 8), size=size, texture_size=TEXTURE)

    def test_a_frame_whose_rectangle_leaves_its_texture_is_refused(self, converter, textures):
        luma, chroma = textures(*planes(100, 128, 128))
        frame = DecodedFrame(timestamp=0, duration=0, planes=(luma, chroma), layout='nv12',
                             texture_size=TEXTURE, visible=(0, 0, 64, 64), colour=self.COLOUR,
                             on_release=lambda _frame: None)
        with pytest.raises(DecoderError, match='visible'):
            converter.convert(frame, textures.output(64, 64))

    def test_a_smaller_size_than_the_rectangle_is_drawn(self, converter, textures):
        luma, chroma = textures(*planes(100, 128, 128))
        target = textures.output(8, 8)
        converter.draw(luma, chroma, visible=VISIBLE, colour=self.COLOUR, target=target,
                       size=(8, 8), texture_size=TEXTURE)
        found = read(target, 8, 8)[..., :3].astype(int)
        assert np.abs(found - expected(self.COLOUR, 100, 128, 128)).max() <= 1

    def test_the_shader_holds_luma_to_the_texture(self, converter, textures):
        """A rectangle past the texture, with no texture size to refuse it by."""
        size = (16, 16)
        luma, chroma = textures(*planes(100, 128, 128, texture=size, visible=(0, 0, *size)))
        target = textures.output(32, 32)
        converter.draw(luma, chroma, visible=(0, 0, 32, 32), colour=self.COLOUR,
                       target=target)
        found = read(target, 32, 32)[..., :3].astype(int)
        assert np.abs(found - expected(self.COLOUR, 100, 128, 128)).max() <= 1

    def test_the_shader_holds_luma_to_the_visible_rectangle(self, converter, textures):
        """The pass itself, drawn larger than its rectangle: the padding is not read."""
        luma, chroma = textures(*planes(100, 128, 128))
        target = textures.output(*TEXTURE)
        converter._pass(luma, chroma, VISIBLE, self.COLOUR, target, TEXTURE, 8)  # noqa: SLF001 - the shader, with the refusals in front of it stepped around
        found = read(target, *TEXTURE)[..., :3].astype(int)
        assert np.abs(found - expected(self.COLOUR, 100, 128, 128)).max() <= 1


class TestTheContextItWasBuiltIn:
    def test_nothing_is_deleted_with_another_context_current(self, gl_context, contexts):  # noqa: ARG002 - requested so a GL context is current first
        converter = NV12Converter()
        program = converter._program  # noqa: SLF001 - the name a wrong delete would take from the other context
        contexts.use(contexts.second)
        with pytest.raises(DecoderError, match='another OpenGL context'):
            converter.close()
        contexts.use(None)
        with pytest.raises(DecoderError, match='no OpenGL context'):
            converter.close()
        contexts.use(contexts.first)
        assert glIsProgram(program)
        converter.close()
        assert not glIsProgram(program)
        converter.close()

    def test_a_framebuffer_is_not_forgotten_from_another_context(self, textures, contexts):
        converter = NV12Converter()
        target = textures.output(8, 8)
        luma, chroma = textures(*planes(100, 128, 128))
        converter.draw(luma, chroma, visible=VISIBLE, colour=Colour(), target=target,
                       size=(8, 8))
        contexts.use(contexts.second)
        with pytest.raises(DecoderError, match='another OpenGL context'):
            converter.forget(target)
        contexts.use(contexts.first)
        converter.forget(target)
        converter.close()

    def test_a_context_that_shares_deletes_the_program_and_keeps_the_rest(
            self, textures, contexts):
        converter = NV12Converter()
        target = textures.output(8, 8)
        luma, chroma = textures(*planes(100, 128, 128))
        converter.draw(luma, chroma, visible=VISIBLE, colour=Colour(), target=target,
                       size=(8, 8))
        program = converter._program  # noqa: SLF001 - the name the share group holds
        vertex_array = converter._vertex_array  # noqa: SLF001 - the name its own context holds
        contexts.use(contexts.sharing)
        with pytest.raises(DecoderError, match='framebuffer.*not shared'):
            converter.forget(target)
        with pytest.raises(DecoderError, match='not shared'):
            converter.close()
        assert not glIsProgram(program)
        contexts.use(contexts.first)
        assert glIsVertexArray(vertex_array)
        converter.close()
        assert not glIsVertexArray(vertex_array)
        converter.close()


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
