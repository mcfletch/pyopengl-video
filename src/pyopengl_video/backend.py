"""What encoder and decoder backends have in common: the record, discovery, errors.

A backend is one implementation of an encoder or a decoder: NVENC, VA-API,
oneVPL. :class:`Backend` says what it can do and how to build it; discovery asks
each one whether this machine can run it, and an unavailable one may say why.
:func:`context_kind` is what a probe asks about the OpenGL context it would
serve.
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Generic, TypeVar

log = logging.getLogger(__name__)

T = TypeVar('T')


class VideoError(RuntimeError):
    """Encoding or decoding failed: the base of every error this package raises.

    One ``except VideoError`` covers a program that records and plays both.
    """


@dataclasses.dataclass(frozen=True)
class Backend(Generic[T]):
    """What one encoder or decoder implementation can do, and how to construct it.

    name -- short identifier a caller can pass to :func:`~pyopengl_video.open_encoder`
        or :func:`~pyopengl_video.open_decoder`
    vendor -- whose hardware it drives
    codecs -- the codec names it handles
    max_size -- the largest frame it will take, as (width, height)
    zero_copy -- True when frames move between it and OpenGL without passing
        through host memory
    probe -- returns True when this machine can actually run the backend; it is
        called during discovery, so it must be cheap and must not raise
    factory -- builds the encoder or decoder; :func:`~pyopengl_video.open_encoder`
        and :func:`~pyopengl_video.open_decoder` document the arguments
    explain -- optional: why this backend is unavailable here, in a sentence
        the caller could act on, or '' when there is nothing useful to say.
        The error raised when nothing fits includes it, since the reason is
        very often a fixable one.
    """

    name: str
    vendor: str
    codecs: frozenset[str]
    max_size: tuple[int, int]
    zero_copy: bool
    probe: Callable[[], bool]
    factory: Callable[..., T]
    explain: Callable[[], str] | None = None

    def supports(self, width: int, height: int, codec: str) -> bool:
        """Can this backend handle a `width` x `height` frame of `codec`?"""
        return (codec in self.codecs
                and width <= self.max_size[0] and height <= self.max_size[1])

    def replace(self, **changes: Any) -> Backend[T]:
        """A copy of this record with `changes` applied."""
        return dataclasses.replace(self, **changes)


def available(backends: Iterable[Backend[T]]) -> Iterator[Backend[T]]:
    """Yield the backends whose hardware and driver are present."""
    for backend in backends:
        if backend.probe():
            yield backend


def explanations(backends: Iterable[Backend[Any]]) -> list[str]:
    """Why each unavailable backend declined, for the ones that will say.

    A backend that is available, that offers no explanation, or that has
    nothing to say contributes nothing. An explanation that raises is dropped,
    so the error it would have decorated is still the one the caller gets.
    """
    found = []
    for backend in backends:
        if backend.explain is None:
            continue
        try:
            if backend.probe():
                continue
            reason = backend.explain()
        except Exception:
            log.debug('%s could not say why it is unavailable', backend.name,
                      exc_info=True)
            continue
        if reason:
            found.append(f'{backend.name}: {reason}')
    return found


def with_reasons(message: str, backends: Iterable[Backend[Any]],
                 only: str | None = None) -> str:
    """`message`, followed by what the unavailable backends said, one per line."""
    reasons = explanations(backends)
    if only is not None:
        reasons = [line for line in reasons if line.startswith(f'{only}:')]
    return '\n'.join([message, *(f'  {line}' for line in reasons)])


def context_kind() -> str | None:
    """Which binding made the OpenGL context current on this thread.

    ``'egl'`` for an EGL context, ``'other'`` for one made through GLX or
    WGL, and None where no context is current. A GLX context cannot export or
    import a DMA-BUF, so the VA-API backends cannot serve it.
    """
    try:
        from OpenGL import EGL  # noqa: PLC0415 - OpenGL.EGL needs an EGL library

        if EGL.eglGetCurrentContext():
            return 'egl'
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('EGL is not usable here: %s', error)
    try:
        from OpenGL import platform  # noqa: PLC0415 - the binding PyOpenGL chose at run time

        current: Any = platform.PLATFORM.GetCurrentContext
        try:
            # Ask only the bindings already loaded: a process holding a GLX
            # context has loaded GLX, and loading it to ask costs a driver.
            # Only PyOpenGL's Linux platform takes the argument.
            found = current(probe=False)
        except TypeError:
            found = current()
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('PyOpenGL could not say which context is current: %s', error)
        return None
    return 'other' if found else None
