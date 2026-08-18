"""Record a spinning triangle to an MP4, rendering with plain PyOpenGL.

This is the whole shape of a recording: render, blit the finished frame into a
texture the encoder has been told about, encode, mux. Run it with a path::

    python examples/record_triangle.py triangle.mp4 --frames 120

Two details matter to anyone recording their own renderer, and both are here.

**The frame is upside down unless the blit turns it over.** OpenGL's framebuffer
starts at the bottom left; the encoder reads a texture from its first row and
calls that the top of the picture. Blitting the source's bottom edge to the
destination's top edge -- the flipped ``dstY`` below -- costs nothing and puts
the video the right way up.

**One texture is not enough.** An encoder that reorders frames is still reading
a texture after ``encode`` returns, so the recording cycles through
``encoder.input_slots`` of them.
"""
from __future__ import annotations

import argparse
import math
import sys

import glfw
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COLOR_ATTACHMENT0,
    GL_COLOR_BUFFER_BIT,
    GL_COMPILE_STATUS,
    GL_DRAW_FRAMEBUFFER,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_FRAMEBUFFER,
    GL_LINEAR,
    GL_NEAREST,
    GL_READ_FRAMEBUFFER,
    GL_RGBA,
    GL_RGBA8,
    GL_STATIC_DRAW,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLES,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    glAttachShader,
    glBindBuffer,
    glBindFramebuffer,
    glBindTexture,
    glBindVertexArray,
    glBlitFramebuffer,
    glBufferData,
    glClear,
    glClearColor,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDrawArrays,
    glEnableVertexAttribArray,
    glFramebufferTexture2D,
    glGenBuffers,
    glGenFramebuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
    glTexImage2D,
    glTexParameteri,
    glUniform1f,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)

from pyopengl_video import open_encoder
from pyopengl_video.mp4 import MP4Writer

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 position;
out vec3 tint;
uniform float turn;
void main() {
    float c = cos(turn), s = sin(turn);
    gl_Position = vec4(mat2(c, -s, s, c) * position, 0.0, 1.0);
    tint = vec3(position * 0.5 + 0.5, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 tint;
out vec4 colour;
void main() { colour = vec4(tint, 1.0); }
"""


def compile_program():
    """Build the two-shader program that draws the triangle."""
    program = glCreateProgram()
    for kind, source in ((GL_VERTEX_SHADER, VERTEX_SHADER),
                         (GL_FRAGMENT_SHADER, FRAGMENT_SHADER)):
        shader = glCreateShader(kind)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise SystemExit(glGetShaderInfoLog(shader).decode())
        glAttachShader(program, shader)
    glLinkProgram(program)
    return program


def make_triangle():
    """A vertex array holding one triangle, apex up."""
    array = glGenVertexArrays(1)
    glBindVertexArray(array)
    buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, buffer)
    corners = (0.0, 0.8, -0.8, -0.6, 0.8, -0.6)
    glBufferData(GL_ARRAY_BUFFER, (len(corners) * 4),
                 (__import__('array').array('f', corners)).tobytes(), GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, False, 8, None)
    return array


def make_capture_target(width, height):
    """A texture the encoder can read, and a framebuffer that blits into it."""
    texture = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, None)
    framebuffer = int(glGenFramebuffers(1))
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, texture, 0)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return texture, framebuffer


def capture(framebuffer, width, height):
    """Copy the finished frame into the capture texture, the right way up.

    The destination's Y coordinates run the other way from the source's, which
    is the flip: what OpenGL drew at the bottom of the frame is what the encoder
    must see at the bottom of the picture.
    """
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer)
    glBlitFramebuffer(0, 0, width, height,
                      0, height, width, 0,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('path', help='the .mp4 to write')
    parser.add_argument('--size', default='640x480', metavar='WxH')
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--bframes', type=int, default=0,
                        help='B-pictures between reference pictures')
    options = parser.parse_args(argv)
    width, _, height = options.size.partition('x')
    width, height = int(width), int(height)

    if not glfw.init():
        raise SystemExit('GLFW will not start')
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # Nothing is shown: the frames go to a file, not to a screen.
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(width, height, 'recording', None, None)
    if not window:
        raise SystemExit('no OpenGL context')
    glfw.make_context_current(window)

    program = compile_program()
    triangle = make_triangle()
    turn = glGetUniformLocation(program, 'turn')

    with open_encoder(width, height, fps=options.fps,
                      bframes=options.bframes) as encoder:
        ring = [make_capture_target(width, height) for _ in range(encoder.input_slots)]
        handles = [encoder.register(texture) for texture, _ in ring]
        step = encoder.timescale * 1 // options.fps

        with MP4Writer(options.path, encoder) as movie:
            for index in range(options.frames):
                glViewport(0, 0, width, height)
                glClearColor(0.05, 0.05, 0.08, 1.0)
                glClear(GL_COLOR_BUFFER_BIT)
                glUseProgram(program)
                glUniform1f(turn, index * 2 * math.pi / options.frames)
                glBindVertexArray(triangle)
                glDrawArrays(GL_TRIANGLES, 0, 3)

                slot = index % len(ring)
                capture(ring[slot][1], width, height)
                movie.write(encoder.encode(handles[slot], timestamp=index * step))
            movie.write(encoder.flush())

    glfw.terminate()
    print(f'{options.frames} frames of {width}x{height} written to {options.path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
