"""The OpenGL context an object was made in, so it is released where its name is its own.

An OpenGL name is a small integer that means something only in the share
group that made it: ``glDeleteTextures([3])`` with a context of another group
current deletes that group's texture 3. An object here that owns OpenGL names
takes an :class:`Owner` when it makes them and asks it before deleting them::

    self._owner = Owner.current()
    ...
    self._owner.require('the conversion pass', DecoderError)
    glDeleteProgram(self._program)

Which contexts are asked for depends on the kind of object:

* Textures, programs, shaders, buffers, renderbuffers and sync objects belong
  to the share group. :meth:`Owner.require` passes in the context that made
  them and in every context that shares objects with it.
* Framebuffers, vertex arrays, queries and transform feedbacks belong to one
  context, sharing or not. :meth:`Owner.require_own` passes in that context
  alone.

The context is read from EGL where an EGL context is current, and otherwise
from the platform binding PyOpenGL loaded (GLX or WGL). Where neither names a
context when the object is made, the owner is unknown and nothing is checked.

EGL, GLX and WGL have no query for which contexts share, so the share group
is found by reading: the first :class:`Owner` made in a group leaves a shader
object there whose source is a random token, and a context in which that name
reads back as a shader with that source is a context of the group. The shader
is never compiled, and it lasts as long as the group does.
"""
from __future__ import annotations

import ctypes
import functools
import logging
import uuid
from types import ModuleType
from typing import Any

from pyopengl_video.backend import VideoError

__all__ = ['Owner']

log = logging.getLogger(__name__)


@functools.cache
def _egl() -> ModuleType | None:
    """PyOpenGL's EGL bindings, or None where the machine has no EGL library."""
    try:
        from OpenGL import EGL  # noqa: PLC0415 - OpenGL.EGL needs an EGL library
    except Exception as error:  # noqa: BLE001 - ImportError or AttributeError, by PyOpenGL release
        log.debug('the EGL bindings are not usable here: %s', error)
        return None
    return EGL


def _address(handle: Any) -> int:
    """A context handle as the integer two handles are compared by."""
    if isinstance(handle, int):
        return handle
    return ctypes.cast(handle, ctypes.c_void_p).value or 0


def _platform_context() -> int:
    """The context PyOpenGL's platform binding reports as current, or 0."""
    from OpenGL import platform  # noqa: PLC0415 - the binding PyOpenGL chose at run time

    current: Any = platform.PLATFORM.GetCurrentContext
    try:
        # Only the bindings already loaded are asked; PyOpenGL's Linux
        # platform alone takes the argument.
        found = current(probe=False)
    except TypeError:
        found = current()
    return _address(found) if found else 0


def _current() -> tuple[tuple[str, int] | None, Any]:
    """The context current on this thread as (key, EGL display); key None for none."""
    try:
        egl = _egl()
        if egl is not None:
            context = egl.eglGetCurrentContext()
            if context:
                return ('egl', _address(context)), egl.eglGetCurrentDisplay()
        context = _platform_context()
    except Exception as error:  # noqa: BLE001 - a binding that cannot answer names no context
        log.debug('PyOpenGL could not say which context is current: %s', error)
        return None, None
    return (('platform', context) if context else None), None


#: The mark of the share group each context seen so far is in, by context key.
#: A key outlives its context and may come to name a later one, so an entry is
#: read back in the current context before it is believed.
_marks: dict[tuple[str, int], tuple[int, bytes]] = {}


def _reads(mark: tuple[int, bytes]) -> bool:
    """Whether the current context is in the share group `mark` was left in."""
    from OpenGL.GL import (  # noqa: PLC0415 - OpenGL.GL stays off `import pyopengl_video`
        glGetShaderSource,
        glIsShader,
    )

    shader, token = mark
    try:
        if not glIsShader(shader):
            return False
        source = glGetShaderSource(shader)
    except Exception as error:  # noqa: BLE001 - a context that cannot answer is not shown to share
        log.debug('shader %d could not be read as a share group mark: %s', shader, error)
        return False
    return (source.encode('ascii', 'replace') if isinstance(source, str) else source) == token


def _mark(key: tuple[str, int]) -> tuple[int, bytes] | None:
    """The mark of the current context's share group, left now if it has none.

    key -- the current context's key

    None where the context makes no shader object; the context then stands
    for itself alone.
    """
    known = _marks.get(key)
    if known is not None and _reads(known):
        return known
    found = next((mark for mark in set(_marks.values()) if _reads(mark)), None)
    if found is None:
        from OpenGL.GL import (  # noqa: PLC0415 - OpenGL.GL stays off `import pyopengl_video`
            GL_VERTEX_SHADER,
            glCreateShader,
            glShaderSource,
        )

        token = f'// pyopengl-video share group {uuid.uuid4().hex}'.encode('ascii')
        try:
            shader = int(glCreateShader(GL_VERTEX_SHADER))
            glShaderSource(shader, token)
        except Exception as error:  # noqa: BLE001 - a context without shaders shares with nothing we can show
            log.debug('this context\'s share group could not be marked: %s', error)
            _marks.pop(key, None)
            return None
        found = (shader, token)
    _marks[key] = found
    return found


class Owner:
    """The OpenGL context that was current when an object was made.

    display -- the EGL display of that context, for the EGL calls that
        release what was made on it; None where the context is not EGL's
    """

    __slots__ = ('_key', '_mark', 'display')

    def __init__(self, key: tuple[str, int] | None, display: Any = None,
                 mark: tuple[int, bytes] | None = None) -> None:
        self._key = key
        self._mark = mark
        self.display = display

    def __repr__(self) -> str:
        if self._key is None:
            return f'<{type(self).__name__} unknown>'
        return f'<{type(self).__name__} {self._key[0]} context {self._key[1]:#x}>'

    @classmethod
    def current(cls) -> Owner:
        """The owner for an object being made on this thread."""
        key, display = _current()
        return cls(key, display, None if key is None else _mark(key))

    @property
    def group(self) -> bytes | None:
        """The token of the context's share group; None where it has no mark."""
        return None if self._mark is None else self._mark[1]

    def is_current(self) -> bool:
        """Whether the context is current on this thread; True for an unknown owner."""
        return self._key is None or _current()[0] == self._key

    def is_shared(self) -> bool:
        """Whether the current context is this one or shares objects with it.

        True for an unknown owner. A context whose share group has no mark
        shares with itself alone.
        """
        if self._key is None:
            return True
        current = _current()[0]
        if current == self._key:
            return True
        return current is not None and self._mark is not None and _reads(self._mark)

    def require(self, what: str, error: type[Exception] = VideoError) -> None:
        """Raise `error` unless the current context has the share group's objects.

        what -- what is being released, for the message

        For textures, programs, shaders, buffers, renderbuffers and sync
        objects: it passes in the context and in any that shares with it.
        """
        if self.is_shared():
            return
        raise error(
            f'{what} holds OpenGL objects of the context it was made in, and '
            f'{_found()}; deleting them here would delete whatever has those '
            'names in this context. Make its own context, or one that shares '
            'objects with it, current and close it again')

    def require_own(self, what: str, error: type[Exception] = VideoError) -> None:
        """Raise `error` unless the context itself is current on this thread.

        what -- what is being released, for the message

        For framebuffers, vertex arrays, queries and transform feedbacks,
        which a context keeps to itself: a context that shares with this one
        has other objects under the same names.
        """
        if self.is_current():
            return
        raise error(
            f'{what} belongs to the OpenGL context it was made in and is not '
            f'shared with any other, and {_found()}; deleting it here would '
            'delete whatever has that name in this context. Make its own '
            'context current and close it again')


def _found() -> str:
    """What is current on this thread in place of an owner's context."""
    return ('no OpenGL context is current on this thread' if _current()[0] is None
            else 'another OpenGL context is current on this thread')
