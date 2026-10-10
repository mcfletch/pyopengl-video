"""Putting back the OpenGL state a pass changes.

A pass in this package draws inside a renderer's frame, so it leaves the
bindings and capabilities as it found them, on every path out of it:

    with preserved_state(capabilities=(GL_BLEND, GL_DEPTH_TEST), units=2):
        glDisable(GL_BLEND)
        ...                                 # bind, draw

This is the state manager the project's OGC151 check is configured to accept,
which is why the restoring calls below are exempt from it.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from OpenGL.GL import (
    GL_ACTIVE_TEXTURE,
    GL_COLOR_WRITEMASK,
    GL_CURRENT_PROGRAM,
    GL_DRAW_FRAMEBUFFER,
    GL_DRAW_FRAMEBUFFER_BINDING,
    GL_SAMPLER_BINDING,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_BINDING_2D,
    GL_VERTEX_ARRAY_BINDING,
    GL_VIEWPORT,
    glActiveTexture,
    glBindFramebuffer,
    glBindSampler,
    glBindTexture,
    glBindVertexArray,
    glColorMask,
    glDisable,
    glEnable,
    glGetBooleanv,
    glGetIntegerv,
    glIsEnabled,
    glUseProgram,
    glViewport,
)

__all__ = ['preserved_state']


@contextmanager
def preserved_state(capabilities: Iterable[int] = (), units: int = 0) -> Iterator[None]:
    """Restore, on leaving, the state a full-screen pass changes.

    Saved: the program, the draw framebuffer, the vertex array, the viewport,
    the colour mask, the active texture unit, whether each of `capabilities`
    is enabled, and the 2D texture and sampler bound on units 0 to `units` - 1.
    """
    program = int(glGetIntegerv(GL_CURRENT_PROGRAM))
    framebuffer = int(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING))
    vertex_array = int(glGetIntegerv(GL_VERTEX_ARRAY_BINDING))
    viewport = [int(value) for value in glGetIntegerv(GL_VIEWPORT)]
    active = int(glGetIntegerv(GL_ACTIVE_TEXTURE))
    colour_mask = [bool(value) for value in glGetBooleanv(GL_COLOR_WRITEMASK)]
    enabled = {capability: bool(glIsEnabled(capability)) for capability in capabilities}
    bound = []
    for unit in range(units):
        glActiveTexture(GL_TEXTURE0 + unit)
        bound.append((int(glGetIntegerv(GL_TEXTURE_BINDING_2D)),
                      int(glGetIntegerv(GL_SAMPLER_BINDING))))
    glActiveTexture(active)
    try:
        yield
    finally:
        for unit, (texture, sampler) in enumerate(bound):
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, texture)
            glBindSampler(unit, sampler)
        glActiveTexture(active)
        glBindVertexArray(vertex_array)
        glUseProgram(program)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer)
        glViewport(*viewport)
        glColorMask(*colour_mask)
        for capability, was_enabled in enabled.items():
            (glEnable if was_enabled else glDisable)(capability)
