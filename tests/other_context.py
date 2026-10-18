"""Two more OpenGL contexts, for what must and must not happen across contexts.

An OpenGL name means something only in the share group that made it. The
`second` context shares nothing with the suite's, so each numbers its objects
from one: a texture made in one has the name of a texture in the other. The
`sharing` context is in the suite's share group: a texture, program or sync
object of either is the same object in the other, and a framebuffer or vertex
array is not.
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
    """The suite's context, one that shares nothing with it, and one that shares its objects."""

    def __init__(self, glfw, first, second, sharing):
        self._glfw = glfw
        self.first = first
        self.second = second
        self.sharing = sharing

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
    """The suite's context and two more; the suite's is current again afterwards."""
    import glfw  # noqa: PLC0415 - the gl_context fixture has skipped where there is none

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    if sys.platform.startswith('linux'):
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
    made = []
    try:
        for title, share in (('second context', None), ('sharing context', gl_context)):
            window = glfw.create_window(64, 64, f'pyopengl-video tests, {title}', None, share)
            if not window:
                pytest.skip(f'no {title} available')
            made.append(window)
        yield Contexts(glfw, gl_context, *made)
    finally:
        glfw.make_context_current(gl_context)
        for window in made:
            glfw.destroy_window(window)
