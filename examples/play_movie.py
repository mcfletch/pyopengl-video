"""Play a movie into an OpenGL window, rendering with plain PyOpenGL.

The whole shape of playback: open the movie with no GL, make a player in the
GL context, and each frame ask it for the texture showing the current time.
Run it with an MP4, an IVF file or a raw ``.h264`` or ``.h265`` stream, in any
codec the machine decodes (H.264, HEVC, AV1 or VP9)::

    python examples/play_movie.py movie.mp4
    python examples/play_movie.py movie.mp4 --loop
    python examples/play_movie.py movie.mp4 --screenshot frame.png --at 1.5

Space pauses, the left and right arrows seek five seconds, and Escape quits.
``examples/record_triangle.py`` makes a movie to play.

Three details carry over to any renderer that draws the texture.

The texture's first row is the bottom of the picture, so ordinary texture
coordinates draw it the right way up.

The texture is ``GL_SRGB8_ALPHA8``, which samples as linear colour, as an image
texture does in a lit scene. Drawn straight to the screen, the default
framebuffer encodes it back to sRGB when ``GL_FRAMEBUFFER_SRGB`` is enabled.

``frame_at()`` shows the frame for the time the program passes it, so a paused
clock holds a frame and a fixed-step clock gets every frame in order.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
import zlib
from pathlib import Path

import glfw
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_COMPILE_STATUS,
    GL_FRAGMENT_SHADER,
    GL_FRAMEBUFFER_SRGB,
    GL_RGB,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TRIANGLE_STRIP,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glAttachShader,
    glBindTexture,
    glBindVertexArray,
    glClear,
    glClearColor,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDrawArrays,
    glEnable,
    glFinish,
    glGenVertexArrays,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glReadPixels,
    glShaderSource,
    glUniform1i,
    glUniform2f,
    glUseProgram,
    glViewport,
)

from pyopengl_video import decoders
from pyopengl_video.playback import Movie

VERTEX_SHADER = """
#version 330 core
// A quad from the vertex index, scaled to keep the movie's shape.
uniform vec2 scale;
out vec2 coordinate;
void main() {
    vec2 corner = vec2(gl_VertexID & 1, gl_VertexID >> 1);
    coordinate = corner;
    gl_Position = vec4((corner * 2.0 - 1.0) * scale, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
uniform sampler2D movie;
in vec2 coordinate;
out vec4 colour;
void main() { colour = texture(movie, coordinate); }
"""

SEEK_SECONDS = 5.0


def compile_program():
    """Build the program that draws the movie's texture on a quad."""
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


def fit(movie_size, window_size):
    """How much of the window the movie fills on each axis, keeping its shape."""
    movie_width, movie_height = movie_size
    width, height = window_size
    movie_aspect = movie_width / movie_height
    window_aspect = width / max(height, 1)
    if movie_aspect > window_aspect:
        return 1.0, window_aspect / movie_aspect
    return movie_aspect / window_aspect, 1.0


def write_png(path, width, height, rows):
    """An RGB PNG from bottom-up rows, as glReadPixels returns them."""
    stride = width * 3
    raw = b''.join(b'\x00' + rows[row * stride:(row + 1) * stride]
                   for row in range(height - 1, -1, -1))

    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data
                + struct.pack('>I', zlib.crc32(kind + data) & 0xFFFFFFFF))

    Path(path).write_bytes(  # noqa: OGC121 a screenshot the caller asked for, written once
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


class Clock:
    """Movie time that pauses, seeks and loops."""

    def __init__(self, duration, loop=False, start=0.0):
        self.duration = duration
        self.loop = loop
        self.position = start
        self.paused = False

    def advance(self, seconds):
        if not self.paused:
            self.position += seconds
        if self.loop and self.duration > 0:
            self.position %= self.duration
        self.position = min(max(self.position, 0.0), self.duration)

    def seek(self, seconds):
        self.position = min(max(self.position + seconds, 0.0), self.duration)

    @property
    def finished(self):
        return not self.loop and self.position >= self.duration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('path', help='an .mp4, an .ivf, or a raw .h264 or .h265 stream')
    parser.add_argument('--loop', action='store_true', help='start again at the end')
    parser.add_argument('--size', default=None, metavar='WxH',
                        help="the window's size; the movie's own by default")
    parser.add_argument('--screenshot', metavar='PNG',
                        help='draw one frame into a hidden window, save it and stop')
    parser.add_argument('--at', type=float, default=0.0,
                        help='the movie time, in seconds, to start at or to capture')
    options = parser.parse_args(argv)

    # Reading the index needs no GL, so a game can do it on a loading thread.
    movie = Movie.open(options.path)
    width, height = movie.size
    if options.size:
        width, _, height = options.size.partition('x')
        width, height = int(width), int(height)

    if not glfw.init():
        raise SystemExit('GLFW will not start')
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SRGB_CAPABLE, glfw.TRUE)
    if sys.platform.startswith('linux'):
        # The VA-API decoder hands its surfaces to OpenGL as DMA-BUFs, which
        # an EGL context imports; GLFW makes a GLX context by default on X11.
        # NVDEC plays into either kind.
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
    if options.screenshot:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(width, height, Path(options.path).name, None, None)
    if not window:
        raise SystemExit('no OpenGL context')
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    print(f'{movie}; decoders here: {[backend.name for backend in decoders()]}')
    program = compile_program()
    quad = glGenVertexArrays(1)
    scale = glGetUniformLocation(program, 'scale')
    glUseProgram(program)
    glUniform1i(glGetUniformLocation(program, 'movie'), 0)
    glEnable(GL_FRAMEBUFFER_SRGB)

    clock = Clock(movie.duration, loop=options.loop, start=options.at)

    def keys(_window, key, _scancode, action, _mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            clock.paused = not clock.paused
        elif key == glfw.KEY_RIGHT:
            clock.seek(SEEK_SECONDS)
        elif key == glfw.KEY_LEFT:
            clock.seek(-SEEK_SECONDS)

    glfw.set_key_callback(window, keys)

    with movie, movie.player() as player:
        last = time.perf_counter()
        while not glfw.window_should_close(window):
            now = time.perf_counter()
            clock.advance(now - last)
            last = now
            texture = player.frame_at(clock.position)

            framebuffer_size = glfw.get_framebuffer_size(window)
            glViewport(0, 0, *framebuffer_size)
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            glUseProgram(program)
            glUniform2f(scale, *fit(movie.size, framebuffer_size))
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, texture)
            glBindVertexArray(quad)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

            if options.screenshot:
                glFinish()
                pixels = glReadPixels(0, 0, *framebuffer_size, GL_RGB, GL_UNSIGNED_BYTE)
                write_png(options.screenshot, *framebuffer_size, pixels)
                print(f'frame {player.frame} at {clock.position:.3f}s written to '
                      f'{options.screenshot}')
                break
            glfw.swap_buffers(window)
            glfw.poll_events()
            if clock.finished:
                break

    glfw.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())
