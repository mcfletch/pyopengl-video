"""The textures an encoder reads, and the framebuffers that fill them.

An encoder's input is a texture, and a recorder gets the frame into it by
blitting. Which side may create that texture differs by platform -- on Linux an
encoder will take one the caller made, while a Windows encoder reads a Direct3D
resource that only the backend can allocate -- so
:meth:`~pyopengl_video.encoder.Encoder.new_input` is how an input is obtained
everywhere, and :class:`InputHandle` is what comes back.

Drawing into one happens inside :meth:`InputHandle.for_drawing`. On a backend
whose texture is shared with another API, that scope is where it changes hands;
on the rest it does nothing, so a recorder is written once::

    with handle.for_drawing():
        copy_the_frame_into(handle.framebuffer)
    packets = encoder.encode(handle, timestamp)
"""
from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext


class InputHandle:
    """A texture an encoder has been told about, with a framebuffer to fill it.

    texture -- the OpenGL texture name to draw into
    target -- its texture target, ``GL_TEXTURE_2D`` unless a backend says
        otherwise
    framebuffer -- a framebuffer object with :attr:`texture` as its only colour
        attachment, which is what a blit names as its destination
    owns_texture -- whether closing this handle should delete the texture too,
        which is so when the encoder made it and not when a caller handed one in
    """

    texture: int = 0
    target: int = 0
    framebuffer: int = 0
    owns_texture: bool = False

    def for_drawing(self) -> AbstractContextManager[InputHandle]:
        """Hold the texture for OpenGL to draw into.

        The base does nothing: a texture that belongs to OpenGL alone is always
        available to it. Backends sharing a texture with another API take it
        back here and hand it over again at the end of the scope.

        Declared as the context manager a caller uses rather than as a
        generator, so a backend can answer with a scope of its own -- which is
        what the VA-API handle does, to leave a fence behind at the end of it.
        """
        return nullcontext(self)

    def close(self) -> None:
        """Give back the framebuffer, and the texture if this handle made it.

        Safe to call twice. Unregistering the input from the encoder is the
        encoder's own business, done when it closes.
        """
        delete_framebuffer(self.framebuffer)
        self.framebuffer = 0
        if self.owns_texture:
            delete_texture(self.texture)
            self.texture = 0


def create_rgba_texture(width: int, height: int) -> int:
    """A new ``GL_RGBA8`` texture of this size, with no mipmaps.

    Eight bits a channel is what every encoder here reads a colour surface as,
    and the size must be the encoder's own.
    """
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
        glGenTextures,
        glTexImage2D,
        glTexParameteri,
    )
    texture = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, texture)
    for parameter, value in (
        (GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
        (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
    ):
        glTexParameteri(GL_TEXTURE_2D, parameter, value)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, int(width), int(height), 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, None)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture


def create_framebuffer(texture: int, target: int | None = None) -> int:
    """A framebuffer with `texture` as its colour attachment.

    A `target` of None or zero means ``GL_TEXTURE_2D``, so a handle that never
    said which target it wanted still gets the usual one rather than an
    attachment the driver refuses.

    The binding in force when this is called is put back, so building a
    recorder's ring does not disturb whatever the renderer had bound.
    """
    from OpenGL.GL import (
        GL_COLOR_ATTACHMENT0,
        GL_DRAW_FRAMEBUFFER,
        GL_DRAW_FRAMEBUFFER_BINDING,
        GL_TEXTURE_2D,
        glBindFramebuffer,
        glFramebufferTexture2D,
        glGenFramebuffers,
        glGetIntegerv,
    )
    target = target or GL_TEXTURE_2D
    previous = int(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING))
    framebuffer = int(glGenFramebuffers(1))
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer)
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           target, int(texture), 0)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, previous)
    return framebuffer


def delete_framebuffer(framebuffer: int) -> None:
    """Give a framebuffer object back, if there is one."""
    if framebuffer:
        from OpenGL.GL import glDeleteFramebuffers
        glDeleteFramebuffers(1, [int(framebuffer)])


def delete_texture(texture: int) -> None:
    """Give a texture back, if there is one."""
    if texture:
        from OpenGL.GL import glDeleteTextures
        glDeleteTextures([int(texture)])
