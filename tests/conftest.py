"""Fixtures for the tests that need a live OpenGL context or a real encoder.

The context is a hidden GLFW window: encoding reads a texture and never presents
anything, so there is nothing to show, and an unmapped window keeps a test run
from flashing over whatever the person running it is doing.
"""
import numpy as np
import pytest

from pyopengl_video import nvenc


@pytest.fixture(scope='session')
def gl_context():
    """A current OpenGL context for the whole session, hidden from the display."""
    glfw = pytest.importorskip('glfw')
    if not glfw.init():
        pytest.skip('GLFW will not initialise here')
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(64, 64, 'pyopengl-video tests', None, None)
    if not window:
        glfw.terminate()
        pytest.skip('no OpenGL context available')
    glfw.make_context_current(window)
    yield window
    glfw.terminate()


@pytest.fixture
def nvenc_available(gl_context):
    """Skip unless this machine has an NVIDIA encoder to talk to."""
    if not nvenc.probe():
        # Two different things fail this, and naming only the library sends a
        # reader looking for a file that is very often sitting right there:
        # NVIDIA supports the encoder's OpenGL device type on Linux alone, so
        # every other platform reports unavailable however good the hardware.
        pytest.skip('no NVENC encoder reachable from OpenGL here: either the '
                    'driver library is absent, or this is not Linux, where '
                    "NVENC's OpenGL device type is the only one supported "
                    '(see plans/WINDOWS-SUPPORT.md)')
    return True


@pytest.fixture
def vpl_available(gl_context):
    """Skip unless Intel's encoder is reachable from this OpenGL context."""
    from pyopengl_video import vpl

    if not vpl.probe():
        pytest.skip('no Intel encoder library here (oneVPL is reached through '
                    'Direct3D, so this backend is Windows-only for now)')
    from pyopengl_video.windows import interop

    if not interop.available():
        pytest.skip('this OpenGL driver does not offer WGL_NV_DX_interop2')
    adapter = interop.adapter_for_context()
    if adapter is None or adapter.vendor != 'Intel':
        pytest.skip(f'this OpenGL context is on {adapter}, not an Intel GPU, so '
                    'the Intel encoder could not read what it draws')
    return adapter


def gradient_frame(width, height, phase=0):
    """An RGBA frame with a diagonal gradient that moves with `phase`.

    Something with structure in both directions and motion between frames, so a
    stream made of it exercises intra and inter prediction rather than
    compressing to nothing.
    """
    x = (np.arange(width, dtype=np.int32)[None, :] + phase * 4) % 256
    y = (np.arange(height, dtype=np.int32)[:, None] + phase * 2) % 256
    frame = np.empty((height, width, 4), dtype=np.uint8)
    frame[..., 0] = x
    frame[..., 1] = y
    frame[..., 2] = (x + y) % 256
    frame[..., 3] = 255
    return frame


@pytest.fixture
def upload_texture(gl_context):
    """Return a helper that puts an RGBA array into a new GL texture."""
    from OpenGL.GL import (
        GL_CLAMP_TO_EDGE,
        GL_LINEAR,
        GL_RGBA,
        GL_RGBA8,
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_WRAP_S,
        GL_TEXTURE_WRAP_T,
        GL_UNSIGNED_BYTE,
        glBindTexture,
        glDeleteTextures,
        glGenTextures,
        glTexImage2D,
        glTexParameteri,
        glTexSubImage2D,
    )
    made = []

    def make(frame):
        height, width = frame.shape[:2]
        texture = int(glGenTextures(1))
        made.append(texture)
        glBindTexture(GL_TEXTURE_2D, texture)
        for parameter, value in (
            (GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
            (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
        ):
            glTexParameteri(GL_TEXTURE_2D, parameter, value)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, frame)
        return texture

    def update(texture, frame):
        height, width = frame.shape[:2]
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        GL_RGBA, GL_UNSIGNED_BYTE, frame)

    make.update = update
    yield make
    if made:
        glDeleteTextures(made)
