"""The OpenGL context an object was made in, so it is released in the same one.

An OpenGL name is a small integer that means something only in the context
that made it: ``glDeleteTextures([3])`` with another context current deletes
that context's texture 3. An object here that owns OpenGL names takes an
:class:`Owner` when it makes them and asks it before deleting them::

    self._owner = Owner.current()
    ...
    self._owner.require('the conversion pass', DecoderError)
    glDeleteProgram(self._program)

The context is read from EGL where an EGL context is current, and otherwise
from the platform binding PyOpenGL loaded (GLX or WGL). Where neither names a
context when the object is made, the owner is unknown and nothing is checked.

A context that shares objects with the owner is refused like any other: EGL
has no query for which contexts share.
"""
from __future__ import annotations

import ctypes
import functools
import logging
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


class Owner:
    """The OpenGL context that was current when an object was made.

    display -- the EGL display of that context, for the EGL calls that
        release what was made on it; None where the context is not EGL's
    """

    __slots__ = ('_key', 'display')

    def __init__(self, key: tuple[str, int] | None, display: Any = None) -> None:
        self._key = key
        self.display = display

    def __repr__(self) -> str:
        if self._key is None:
            return f'<{type(self).__name__} unknown>'
        return f'<{type(self).__name__} {self._key[0]} context {self._key[1]:#x}>'

    @classmethod
    def current(cls) -> Owner:
        """The owner for an object being made on this thread."""
        return cls(*_current())

    def is_current(self) -> bool:
        """Whether the context is current on this thread; True for an unknown owner."""
        return self._key is None or _current()[0] == self._key

    def require(self, what: str, error: type[Exception] = VideoError) -> None:
        """Raise `error` unless the context is current on this thread.

        what -- what is being released, for the message
        """
        if self.is_current():
            return
        current = _current()[0]
        found = ('no OpenGL context is current on this thread' if current is None
                 else 'another OpenGL context is current on this thread')
        raise error(
            f'{what} holds OpenGL objects of the context it was made in, and '
            f'{found}; deleting them here would delete whatever has those names '
            'in this context. Make its own context current and close it again')
