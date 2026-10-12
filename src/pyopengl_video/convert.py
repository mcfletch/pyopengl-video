"""The conversion pass: NV12 planes into an RGBA texture, the right way up.

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

The output is ``GL_SRGB8_ALPHA8``: the values are stored as computed, and a
sampler reading the texture linearises them as it does any 8-bit colour image.
The program is GLSL 3.30, and runs in a core or a compatibility context.
"""
from __future__ import annotations

import ctypes

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

from pyopengl_video.colour import Colour
from pyopengl_video.decoder import DecodedFrame, DecoderError
from pyopengl_video.glstate import preserved_state

__all__ = ['LAYOUT_DEPTHS', 'NV12Converter', 'create_output_texture',
           'generate_mipmaps']

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
    float y = texelFetch(luma, ivec2(floor(source)), 0).r;
    vec2 low = visible.xy * 0.5 + 0.5;
    vec2 high = (visible.xy + visible.zw) * 0.5 - 0.5;
    vec2 site = clamp(source * 0.5 + chroma_offset, low, high);
    vec2 cbcr = texture(chroma, site / vec2(textureSize(chroma, 0))).rg;
    vec3 rgb = (yuv_to_rgb * vec4(y, cbcr, 1.0)).rgb;
    colour = vec4(clamp(rgb, 0.0, 1.0), 1.0);
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


def _program() -> int:
    shaders = [_compile(GL_VERTEX_SHADER, VERTEX_SHADER),
               _compile(GL_FRAGMENT_SHADER, FRAGMENT_SHADER)]
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


class NV12Converter:
    """Draws NV12 or P010 planes into RGBA textures.

    Built and used on the thread whose OpenGL context is current. It owns a
    program, an empty vertex array and one framebuffer per output texture it
    has drawn into; :meth:`close` gives them back. The GL state the pass
    changes is restored before :meth:`draw` returns, on every path out.
    """

    def __init__(self) -> None:
        self._program = _program()
        self._vertex_array = int(glGenVertexArrays(1))
        self._framebuffers: dict[int, int] = {}
        self._uniforms = {name: glGetUniformLocation(self._program, name)
                          for name in ('luma', 'chroma', 'visible',
                                       'chroma_offset', 'yuv_to_rgb')}

    def convert(self, frame: DecodedFrame, target: int) -> None:
        """Draw a decoded ``'nv12'`` or ``'p010'`` frame into `target`, which is frame-sized."""
        if frame.layout not in LAYOUT_DEPTHS:
            raise DecoderError(f'a {frame.layout} frame needs no conversion')
        if frame.released:
            raise DecoderError('this frame has been released; its surface may '
                               'already hold another picture')
        luma, chroma = frame.planes
        self.draw(luma, chroma, visible=frame.visible, colour=frame.colour,
                  target=target, depth=LAYOUT_DEPTHS[frame.layout])

    def draw(self, luma: int, chroma: int, *, visible: tuple[int, int, int, int],
             colour: Colour, target: int, size: tuple[int, int] | None = None,
             depth: int = 8) -> None:
        """Draw the `visible` part of two planes into `target`.

        luma/chroma -- ``GL_R8`` and ``GL_RG8`` textures, first row at the
            top; ``GL_R16`` and ``GL_RG16`` for a `depth` above 8, the samples
            in the high bits
        visible -- ``(x, y, width, height)`` of the picture in the luma texture
        colour -- the stream's colour description
        target -- an RGBA texture, usually from :func:`create_output_texture`
        size -- the target's size; the visible size by default
        depth -- bits a sample
        """
        width, height = size or visible[2:]
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
        framebuffer = self._framebuffers.pop(target, None)
        if framebuffer is not None:
            glDeleteFramebuffers(1, [framebuffer])

    def close(self) -> None:
        """Give back the program, the vertex array and the framebuffers."""
        for target in list(self._framebuffers):
            self.forget(target)
        if self._vertex_array:
            glDeleteVertexArrays(1, [self._vertex_array])
            self._vertex_array = 0
        if self._program:
            glDeleteProgram(self._program)
            self._program = 0


def generate_mipmaps(texture: int) -> None:
    """Make `texture`'s mipmap levels from its first, leaving the binding as it was."""
    previous = int(glGetIntegerv(GL_TEXTURE_BINDING_2D))
    glBindTexture(GL_TEXTURE_2D, texture)
    try:
        glGenerateMipmap(GL_TEXTURE_2D)
    finally:
        glBindTexture(GL_TEXTURE_2D, previous)
