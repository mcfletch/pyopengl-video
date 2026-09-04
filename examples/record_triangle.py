"""Record a spinning triangle to an MP4, rendering with plain PyOpenGL.

This is the whole shape of a recording: render, blit the finished frame into a
texture the encoder reads, encode, mux. Run it with a path::

    python examples/record_triangle.py triangle.mp4 --frames 120

Three details matter to anyone recording their own renderer, and all are here.

**The frame is upside down unless the blit turns it over.** OpenGL's framebuffer
starts at the bottom left; the encoder reads a texture from its first row and
calls that the top of the picture. Blitting the source's bottom edge to the
destination's top edge -- the flipped ``dstY`` below -- costs nothing and puts
the video the right way up.

**One texture is not enough.** An encoder that reorders frames is still reading
a texture after ``encode`` returns, so the recording cycles through
``encoder.input_slots`` of them.

**The encoder makes its own.** ``new_input()`` returns a texture it can read and
a framebuffer to blit into, because on Windows the surface is a Direct3D
resource only the backend can allocate. Drawing happens inside
``for_drawing()``, which is where such a surface passes between the two
graphics APIs and does nothing where nothing is shared.
"""
from __future__ import annotations

import argparse
import math
import sys

import glfw
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COLOR_BUFFER_BIT,
    GL_COMPILE_STATUS,
    GL_DRAW_FRAMEBUFFER,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_FRAMEBUFFER,
    GL_NEAREST,
    GL_READ_FRAMEBUFFER,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_VERTEX_SHADER,
    glAttachShader,
    glBindBuffer,
    glBindFramebuffer,
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
    glGenBuffers,
    glGenVertexArrays,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
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
    if sys.platform.startswith('linux'):
        # A Linux encoder is handed the frame as a DMA-BUF exported from the
        # texture, which is an EGL extension; GLFW makes a GLX context by
        # default on X11, and a GLX context cannot export one.
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
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
        # The encoder allocates its own inputs: on some platforms its surface is
        # a resource only the driver can make, and this way the loop is the same
        # everywhere.
        ring = [encoder.new_input() for _ in range(encoder.input_slots)]
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

                handle = ring[index % len(ring)]
                # The scope is where a surface shared with another graphics API
                # changes hands; where nothing is shared it does nothing.
                with handle.for_drawing():
                    capture(handle.framebuffer, width, height)
                movie.write(encoder.encode(handle, timestamp=index * step))
            movie.write(encoder.flush())

    glfw.terminate()
    print(f'{options.frames} frames of {width}x{height} written to {options.path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
