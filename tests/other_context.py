"""A second OpenGL context, for what must not happen across two of them.

An OpenGL name means something only in the context that made it, and the two
contexts here share nothing, so each numbers its objects from one: a texture
made in one has the name of a texture in the other.
"""
import sys

import pytest
from OpenGL.GL import (
    GL_NEAREST,
    GL_TEXTURE_2D,
    GL_TEXTURE_MIN_FILTER,
    glBindTexture,
    glGenTextures,
    glTexParameteri,
)


class Contexts:
    """The suite's context and another one, with nothing shared between them."""

    def __init__(self, glfw, first, second):
        self._glfw = glfw
        self.first = first
        self.second = second

    def use(self, window):
        """Make `window`'s context current on this thread; None for no context."""
        self._glfw.make_context_current(window)

    def texture_named(self, name):
        """A texture of the current context whose name is `name`."""
        made = []
        while not made or made[-1] < name:
            made.append(int(glGenTextures(1)))
            glBindTexture(GL_TEXTURE_2D, made[-1])
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)
        assert made[-1] == name, f'this context numbers its textures past {name}'
        return name


@pytest.fixture
def contexts(gl_context):
    """The suite's context and a second one; the suite's is current again afterwards."""
    import glfw  # noqa: PLC0415 - the gl_context fixture has skipped where there is none

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    if sys.platform.startswith('linux'):
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
    second = glfw.create_window(64, 64, 'pyopengl-video tests, second context', None, None)
    if not second:
        pytest.skip('no second OpenGL context available')
    try:
        yield Contexts(glfw, gl_context, second)
    finally:
        glfw.make_context_current(gl_context)
        glfw.destroy_window(second)
