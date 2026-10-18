"""The conversion pass: a decoded picture into an RGBA texture, the right way up.

Every hardware decoder here produces two planes: full-size luma, and half-size
interleaved Cb and Cr. An 8-bit stream gives NV12, read as ``GL_R8`` and
``GL_RG8`` textures; a 10-bit stream gives P010, read as ``GL_R16`` and
``GL_RG16`` with each sample in the high bits of its channel.
:class:`NV12Converter` samples the two planes of either and draws one triangle
into the output texture's framebuffer, applying in one pass:

- the stream's matrix and range (:meth:`~pyopengl_video.colour.Colour.yuv_to_rgb`)
- the visible rectangle, since a decoder's surface is padded past the picture
- the vertical flip: a surface's first row is the picture's top, and a GL
  texture's first row is its bottom
- the chroma siting, with chroma reads held inside the visible picture

A backend whose hardware converts to RGB itself hands over an ``'rgba'`` frame,
and :meth:`NV12Converter.convert` copies that with the same crop and flip, no
matrix involved.

Luma is read texel by texel, and a texel fetch outside its texture is
undefined, so :meth:`NV12Converter.draw` refuses a visible rectangle that
leaves the texture and an output larger than the rectangle, and the shader
holds every luma coordinate to the rectangle and to the texture besides.

The output is ``GL_SRGB8_ALPHA8``: the values are stored as computed, and a
sampler reading the texture linearises them as it does any 8-bit colour image.
The program is GLSL 3.30, and runs in a core or a compatibility context.
"""
from __future__ import annotations

import ctypes
import functools

from OpenGL.GL import (
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_ATTACHMENT0,
    GL_COMPILE_STATUS,
    GL_CULL_FACE,
    GL_DEPTH_TEST,
    GL_DRAW_FRAMEBUFFER,
    GL_DRAW_FRAMEBUFFER_BINDING,
    GL_FRAGMENT_SHADER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_FRAMEBUFFER_SRGB,
    GL_LINEAR,
    GL_LINK_STATUS,
    GL_RASTERIZER_DISCARD,
    GL_SCISSOR_TEST,
    GL_SRGB8_ALPHA8,
    GL_STENCIL_TEST,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_BINDING_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_TRUE,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glAttachShader,
    glBindFramebuffer,
    glBindSampler,
    glBindTexture,
    glBindVertexArray,
    glCheckFramebufferStatus,
    glColorMask,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteFramebuffers,
    glDeleteProgram,
    glDeleteShader,
    glDeleteVertexArrays,
    glDisable,
    glDrawArrays,
    glFramebufferTexture2D,
    glGenerateMipmap,
    glGenFramebuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetIntegerv,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
    glTexParameteri,
    glTexStorage2D,
    glUniform1i,
    glUniform2f,
    glUniform4f,
    glUniformMatrix4fv,
    glUseProgram,
    glViewport,
)

from pyopengl_video import bounds
from pyopengl_video.colour import Colour
from pyopengl_video.decoder import DecodedFrame, DecoderError
from pyopengl_video.glcontext import Owner
from pyopengl_video.glstate import preserved_state
from pyopengl_video.teardown import release_all

__all__ = ['LAYOUT_DEPTHS', 'NV12Converter', 'checked_rectangle',
           'create_output_texture', 'generate_mipmaps']

#: Bits a sample for each two-plane layout a decoded frame may have.
LAYOUT_DEPTHS = {'nv12': 8, 'p010': 10}

#: Bits a channel of the textures a layout of more than 8 bits is read from.
WIDE_CHANNEL_BITS = 16


def sample_scale(depth: int) -> float:
    """What a texture read is multiplied by to give code value over the largest.

    An 8-bit texture reads as code / 255 already. A wider sample sits in the
    high bits of a 16-bit channel, so the texture reads as
    ``code * 2 ** (16 - depth) / 65535``.
    """
    if depth == 8:
        return 1.0
    shifted = ((1 << depth) - 1) << (WIDE_CHANNEL_BITS - depth)
    return ((1 << WIDE_CHANNEL_BITS) - 1) / shifted

VERTEX_SHADER = '''#version 330 core
// One triangle covering the viewport, from the vertex index alone.
void main() {
    vec2 corner = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
'''

FRAGMENT_SHADER = '''#version 330 core
uniform sampler2D luma;
uniform sampler2D chroma;
uniform vec4 visible;            // x, y, width, height in luma texels, top first
uniform vec2 chroma_offset;      // chroma siting, in chroma texels
uniform mat4 yuv_to_rgb;
out vec4 colour;

void main() {
    // The output's row 0 is the picture's bottom, the surface's row 0 its top.
    vec2 source = vec2(visible.x + gl_FragCoord.x,
                       visible.y + visible.w - gl_FragCoord.y);
    // A fetch outside the texture is undefined, so the texel is held to the
    // visible rectangle and then to the texture, whatever the uniforms say.
    ivec2 first = ivec2(visible.xy);
    ivec2 last = max(first + ivec2(visible.zw) - 1, first);
    ivec2 texel = min(max(ivec2(floor(source)), first), last);
    texel = clamp(texel, ivec2(0), max(textureSize(luma, 0) - 1, ivec2(0)));
    float y = texelFetch(luma, texel, 0).r;
    vec2 low = visible.xy * 0.5 + 0.5;
    vec2 high = (visible.xy + visible.zw) * 0.5 - 0.5;
    vec2 site = clamp(source * 0.5 + chroma_offset, low, high);
    vec2 cbcr = texture(chroma, site / vec2(textureSize(chroma, 0))).rg;
    vec3 rgb = (yuv_to_rgb * vec4(y, cbcr, 1.0)).rgb;
    colour = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
'''

COPY_FRAGMENT_SHADER = '''#version 330 core
uniform sampler2D picture;
uniform vec4 visible;            // x, y, width, height in texels, top first
out vec4 colour;

void main() {
    // The output's row 0 is the picture's bottom, the source's row 0 its top.
    vec2 source = vec2(visible.x + gl_FragCoord.x,
                       visible.y + visible.w - gl_FragCoord.y);
    colour = vec4(texelFetch(picture, ivec2(floor(source)), 0).rgb, 1.0);
}
'''

#: Capabilities a renderer may have left on that would change the pass's
#: result: blending, tests, culling and the sRGB encode on write.
DISABLED_FOR_THE_PASS = (GL_BLEND, GL_DEPTH_TEST, GL_STENCIL_TEST, GL_SCISSOR_TEST,
                         GL_CULL_FACE, GL_FRAMEBUFFER_SRGB, GL_RASTERIZER_DISCARD)


def create_output_texture(width: int, height: int, levels: int = 1) -> int:
    """An immutable ``GL_SRGB8_ALPHA8`` texture for the pass to draw into.

    levels -- mipmap levels to allocate; 1 for none
    """
    texture = int(glGenTextures(1))
    previous = int(glGetIntegerv(GL_TEXTURE_BINDING_2D))
    glBindTexture(GL_TEXTURE_2D, texture)
    try:
        glTexStorage2D(GL_TEXTURE_2D, levels, GL_SRGB8_ALPHA8, int(width), int(height))
        for parameter, value in (
            (GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
            (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
        ):
            glTexParameteri(GL_TEXTURE_2D, parameter, value)
    finally:
        glBindTexture(GL_TEXTURE_2D, previous)
    return texture


def _compile(kind: int, source: str) -> int:
    shader = glCreateShader(kind)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader)
        glDeleteShader(shader)
        raise DecoderError(f'the conversion shader would not compile: {log!r}')
    return int(shader)


def _program(fragment: str = FRAGMENT_SHADER) -> int:
    shaders = [_compile(GL_VERTEX_SHADER, VERTEX_SHADER),
               _compile(GL_FRAGMENT_SHADER, fragment)]
    program = int(glCreateProgram())
    for shader in shaders:
        glAttachShader(program, shader)
    glLinkProgram(program)
    for shader in shaders:
        glDeleteShader(shader)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = glGetProgramInfoLog(program)
        glDeleteProgram(program)
        raise DecoderError(f'the conversion program would not link: {log!r}')
    return program


def checked_rectangle(visible: tuple[int, int, int, int], size: tuple[int, int] | None,
                      texture_size: tuple[int, int] | None,
                      ) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """A visible rectangle and an output size the pass reads only the picture for.

    visible -- ``(x, y, width, height)`` in luma texels
    size -- the output size, or None for the rectangle's own
    texture_size -- the luma texture's size, or None where it is not known

    Returns the rectangle and the size as whole numbers. Raises
    :class:`~pyopengl_video.decoder.DecoderError` for a rectangle that is
    empty, starts before the texture or ends after it, and for a size that is
    empty or larger than the rectangle: the pass reads one luma texel for each
    output texel, so a larger output reads past the picture.
    """
    described = f'the visible rectangle {tuple(visible)}'
    if len(visible) != 4:
        raise DecoderError(f'{described} is not an (x, y, width, height)')
    most = texture_size if texture_size is not None else (bounds.FIELD_16, bounds.FIELD_16)
    x = bounds.whole(visible[0], f'the left of {described}', 0, most[0] - 1, DecoderError)
    y = bounds.whole(visible[1], f'the top of {described}', 0, most[1] - 1, DecoderError)
    width = bounds.whole(visible[2], f'the width of {described}', 1, most[0] - x,
                         DecoderError)
    height = bounds.whole(visible[3], f'the height of {described}', 1, most[1] - y,
                          DecoderError)
    if size is None:
        return (x, y, width, height), (width, height)
    drawn = f'the output size {tuple(size)} of a {width}x{height} picture'
    if len(size) != 2:
        raise DecoderError(f'{drawn} is not a (width, height)')
    return (x, y, width, height), (
        bounds.whole(size[0], f'the width of {drawn}', 1, width, DecoderError),
        bounds.whole(size[1], f'the height of {drawn}', 1, height, DecoderError))


class NV12Converter:
    """Draws decoded frames into RGBA textures: NV12 or P010 planes, or a picture already RGBA.

    Built and used on the thread whose OpenGL context is current. It owns a
    program, an empty vertex array and one framebuffer per output texture it
    has drawn into; :meth:`close` gives them back. The GL state the pass
    changes is restored before :meth:`draw` returns, on every path out.

    The two programs belong to the share group of the context the pass was
    built in, and the vertex array and the framebuffers to that context alone.
    With a context that shares objects with it current, :meth:`close` deletes
    the programs, keeps the rest and raises
    :class:`~pyopengl_video.decoder.DecoderError`, as :meth:`forget` does;
    with a context outside the group current, or none, nothing is deleted.
    Closing again with its own context current gives back what is left.
    """

    def __init__(self) -> None:
        self._owner = Owner.current()
        self._program = _program()
        self._copy_program = _program(COPY_FRAGMENT_SHADER)
        self._copy_uniforms = {name: glGetUniformLocation(self._copy_program, name)
                               for name in ('picture', 'visible')}
        self._vertex_array = int(glGenVertexArrays(1))
        self._framebuffers: dict[int, int] = {}
        self._uniforms = {name: glGetUniformLocation(self._program, name)
                          for name in ('luma', 'chroma', 'visible',
                                       'chroma_offset', 'yuv_to_rgb')}

    def convert(self, frame: DecodedFrame, target: int) -> None:
        """Draw a decoded frame into `target`, which is frame-sized.

        An ``'nv12'`` or ``'p010'`` frame is converted with its colour
        description; an ``'rgba'`` frame is copied.
        """
        if frame.released:
            raise DecoderError('this frame has been released; its surface may '
                               'already hold another picture')
        if frame.layout == 'rgba':
            self.copy(frame.planes[0], visible=frame.visible, target=target)
            return
        if frame.layout not in LAYOUT_DEPTHS:
            raise DecoderError(f'a {frame.layout} frame has no conversion')
        luma, chroma = frame.planes
        self.draw(luma, chroma, visible=frame.visible, colour=frame.colour,
                  target=target, depth=LAYOUT_DEPTHS[frame.layout],
                  texture_size=frame.texture_size)

    def draw(self, luma: int, chroma: int, *, visible: tuple[int, int, int, int],
             colour: Colour, target: int, size: tuple[int, int] | None = None,
             depth: int = 8, texture_size: tuple[int, int] | None = None) -> None:
        """Draw the `visible` part of two planes into `target`.

        luma/chroma -- ``GL_R8`` and ``GL_RG8`` textures, first row at the
            top; ``GL_R16`` and ``GL_RG16`` for a `depth` above 8, the samples
            in the high bits
        visible -- ``(x, y, width, height)`` of the picture in the luma
            texture, in whole texels: not empty, and inside `texture_size`
            where that is given
        colour -- the stream's colour description
        target -- an RGBA texture, usually from :func:`create_output_texture`
        size -- how much of the target is drawn, from its lower left, one
            output texel for each luma texel; the visible size by default and
            no larger. A smaller size draws the lower left of the picture.
        depth -- bits a sample
        texture_size -- the luma texture's ``(width, height)``, which
            `visible` is checked against

        Raises :class:`~pyopengl_video.decoder.DecoderError` for a rectangle
        or a size outside those limits, before anything is drawn.
        """
        visible, size = checked_rectangle(visible, size, texture_size)
        self._pass(luma, chroma, visible, colour, target, size, depth)

    def _pass(self, luma: int, chroma: int, visible: tuple[int, int, int, int],
              colour: Colour, target: int, size: tuple[int, int], depth: int) -> None:
        """Draw, with the rectangle and the size taken as they are given."""
        width, height = size
        framebuffer = self._framebuffer(target)
        colour = colour.resolved(visible[3])
        with preserved_state(DISABLED_FOR_THE_PASS, units=2):
            for capability in DISABLED_FOR_THE_PASS:
                glDisable(capability)
            glColorMask(True, True, True, True)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer)
            glViewport(0, 0, width, height)
            glUseProgram(self._program)
            glBindVertexArray(self._vertex_array)
            for unit, texture in enumerate((luma, chroma)):
                glActiveTexture(GL_TEXTURE0 + unit)
                glBindTexture(GL_TEXTURE_2D, texture)
                glBindSampler(unit, 0)
            uniforms = self._uniforms
            glUniform1i(uniforms['luma'], 0)
            glUniform1i(uniforms['chroma'], 1)
            glUniform4f(uniforms['visible'], *(float(value) for value in visible))
            glUniform2f(uniforms['chroma_offset'], *colour.chroma_offset())
            scale = sample_scale(depth)
            rows = [*((y * scale, cb * scale, cr * scale, offset)
                      for y, cb, cr, offset in colour.yuv_to_rgb(depth)),
                    (0.0, 0.0, 0.0, 1.0)]
            matrix = (ctypes.c_float * 16)(*(value for row in rows for value in row))
            glUniformMatrix4fv(uniforms['yuv_to_rgb'], 1, GL_TRUE, matrix)
            glDrawArrays(GL_TRIANGLES, 0, 3)

    def copy(self, picture: int, *, visible: tuple[int, int, int, int], target: int,
             size: tuple[int, int] | None = None) -> None:
        """Draw the `visible` part of an RGBA texture into `target`, flipped to OpenGL's orientation.

        picture -- an RGBA texture, first row at the top
        visible -- ``(x, y, width, height)`` of the picture in it
        target -- an RGBA texture, usually from :func:`create_output_texture`
        size -- the target's size; the visible size by default
        """
        width, height = size or visible[2:]
        framebuffer = self._framebuffer(target)
        with preserved_state(DISABLED_FOR_THE_PASS, units=1):
            for capability in DISABLED_FOR_THE_PASS:
                glDisable(capability)
            glColorMask(True, True, True, True)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer)
            glViewport(0, 0, width, height)
            glUseProgram(self._copy_program)
            glBindVertexArray(self._vertex_array)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, picture)
            glBindSampler(0, 0)
            glUniform1i(self._copy_uniforms['picture'], 0)
            glUniform4f(self._copy_uniforms['visible'], *(float(value) for value in visible))
            glDrawArrays(GL_TRIANGLES, 0, 3)

    def _framebuffer(self, target: int) -> int:
        """The framebuffer drawing into `target`, made the first time it is named."""
        found = self._framebuffers.get(target)
        if found is not None:
            return found
        framebuffer = int(glGenFramebuffers(1))
        previous = int(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING))
        try:
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer)
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, int(target), 0)
            status = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER)
        finally:
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, previous)
        if status != GL_FRAMEBUFFER_COMPLETE:
            glDeleteFramebuffers(1, [framebuffer])
            raise DecoderError(
                f'texture {target} cannot be drawn into (framebuffer status '
                f'{status:#x}); the pass draws into an RGBA texture')
        self._framebuffers[target] = framebuffer
        return framebuffer

    def forget(self, target: int) -> None:
        """Delete the framebuffer kept for `target`, before the texture is deleted."""
        if target not in self._framebuffers:
            return
        self._owner.require_own("the conversion pass's framebuffer", DecoderError)
        glDeleteFramebuffers(1, [self._framebuffers.pop(target)])

    def close(self) -> None:
        """Give back the programs, the vertex array and the framebuffers.

        Safe to call twice. Every one is given back whatever the others do.
        """
        release_all(*(functools.partial(self.forget, target)
                      for target in list(self._framebuffers)),
                    self._delete_vertex_array, self._delete_program,
                    self._delete_copy_program)

    def _delete_vertex_array(self) -> None:
        if not self._vertex_array:
            return
        self._owner.require_own("the conversion pass's vertex array", DecoderError)
        vertex_array, self._vertex_array = self._vertex_array, 0
        glDeleteVertexArrays(1, [vertex_array])

    def _delete_program(self) -> None:
        if not self._program:
            return
        self._owner.require("the conversion pass's program", DecoderError)
        program, self._program = self._program, 0
        glDeleteProgram(program)

    def _delete_copy_program(self) -> None:
        if not self._copy_program:
            return
        self._owner.require("the conversion pass's copy program", DecoderError)
        program, self._copy_program = self._copy_program, 0
        glDeleteProgram(program)


def generate_mipmaps(texture: int) -> None:
    """Make `texture`'s mipmap levels from its first, leaving the binding as it was."""
    previous = int(glGetIntegerv(GL_TEXTURE_BINDING_2D))
    glBindTexture(GL_TEXTURE_2D, texture)
    try:
        glGenerateMipmap(GL_TEXTURE_2D)
    finally:
        glBindTexture(GL_TEXTURE_2D, previous)
