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
from collections.abc import Callable, Iterable, Iterator, Mapping
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
    codecs -- the codec names it handles. On a record from
        :func:`~pyopengl_video.encoders` or :func:`~pyopengl_video.decoders`
        these are the codecs this machine's hardware has.
    max_size -- the largest frame it will take in any codec, as (width, height)
    sizes -- the largest frame for each codec that has a limit of its own;
        :meth:`largest` reads it
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
    capabilities -- optional: the codecs this machine's hardware offers through
        the backend, each with its largest frame. Called during discovery after
        `probe` has passed, so it must not raise. A backend without one is
        reported with the `codecs` it was registered with.
    """

    name: str
    vendor: str
    codecs: frozenset[str]
    max_size: tuple[int, int]
    zero_copy: bool
    probe: Callable[[], bool]
    factory: Callable[..., T]
    explain: Callable[[], str] | None = None
    capabilities: Callable[[], Mapping[str, tuple[int, int]]] | None = None
    sizes: Mapping[str, tuple[int, int]] = dataclasses.field(
        default_factory=dict, hash=False)

    def largest(self, codec: str) -> tuple[int, int]:
        """The largest frame of `codec` this backend takes, as (width, height)."""
        return self.sizes.get(codec, self.max_size)

    def supports(self, width: int, height: int, codec: str) -> bool:
        """Can this backend handle a `width` x `height` frame of `codec`?"""
        most = self.largest(codec)
        return codec in self.codecs and width <= most[0] and height <= most[1]

    def on_this_machine(self) -> Backend[T] | None:
        """This record narrowed to what the hardware here offers.

        None when the hardware offers none of the backend's codecs. A backend
        with no `capabilities` is returned as it is.
        """
        if self.capabilities is None:
            return self
        try:
            found = {codec: size for codec, size in self.capabilities().items()
                     if codec in self.codecs}
        except Exception:
            log.debug('%s could not list its codecs', self.name, exc_info=True)
            return None
        if not found:
            return None
        return self.replace(
            codecs=frozenset(found), sizes=found,
            max_size=(max(width for width, _ in found.values()),
                      max(height for _, height in found.values())))

    def replace(self, **changes: Any) -> Backend[T]:
        """A copy of this record with `changes` applied."""
        return dataclasses.replace(self, **changes)


def available(backends: Iterable[Backend[T]]) -> Iterator[Backend[T]]:
    """Yield the backends whose hardware and driver are present.

    Each record carries the codecs the hardware here offers, where the backend
    can ask it.
    """
    for backend in backends:
        if not backend.probe():
            continue
        found = backend.on_this_machine()
        if found is not None:
            yield found


def describe(backends: Iterable[Backend[Any]]) -> str:
    """Each backend's name, codecs and largest frame, for an error message."""
    found = [(backend.name, sorted(backend.codecs), backend.max_size)
             for backend in backends]
    return str(found) if found else 'none'


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
