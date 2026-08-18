"""The interface every encoder backend implements, and the registry of backends.

A backend turns an OpenGL colour buffer into compressed video. What differs
between them is how the frame reaches the encoder -- an NVIDIA session takes a
texture name, a VA-API session takes a DMA-BUF exported from one -- so the
interface is written in terms of a *registered input*: the caller hands over a
texture once, gets a handle back, and encodes with that handle every frame.

Encoders may hold frames back. An encoder configured with B-frames or with
lookahead consumes several pictures before it emits any, so :meth:`Encoder.encode`
returns a *list* of packets: empty is ordinary, two is ordinary, and no caller
may assume one frame in means one packet out. :meth:`Encoder.flush` drains
whatever is still inside at the end of a recording.
"""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import Any


class EncoderError(RuntimeError):
    """An encoder rejected a call, or failed part-way through a stream."""


class EncoderUnavailable(EncoderError):
    """No backend on this machine can encode what was asked for."""


@dataclasses.dataclass(frozen=True)
class Packet:
    """One compressed picture, and what a container needs in order to place it.

    data -- the compressed bytes, as an Annex-B elementary stream fragment
    timestamp -- presentation time, in the encoder's timescale
    duration -- how long the picture is shown, in the encoder's timescale
    keyframe -- True when the picture can be decoded without any earlier one
    """

    data: bytes
    timestamp: int
    duration: int
    keyframe: bool

    def __len__(self) -> int:
        return len(self.data)


@dataclasses.dataclass(frozen=True)
class Backend:
    """What one encoder implementation can do, and how to construct it.

    name -- short identifier a caller can pass to :func:`open_encoder`
    vendor -- whose hardware it drives
    codecs -- the codec names it accepts
    max_size -- the largest frame it will encode, as (width, height)
    zero_copy -- True when frames reach the encoder without passing through
        host memory
    probe -- returns True when this machine can actually run the backend; it is
        called during discovery, so it must be cheap and must not raise
    factory -- called as ``factory(width, height, codec=..., **options)``
    """

    name: str
    vendor: str
    codecs: frozenset[str]
    max_size: tuple[int, int]
    zero_copy: bool
    probe: Callable[[], bool]
    factory: Callable[..., Encoder]

    def supports(self, width: int, height: int, codec: str) -> bool:
        """Can this backend encode a `width` x `height` frame as `codec`?"""
        return (codec in self.codecs
                and width <= self.max_size[0] and height <= self.max_size[1])

    def replace(self, **changes: Any) -> Backend:
        """A copy of this record with `changes` applied."""
        return dataclasses.replace(self, **changes)


class Encoder(ABC):
    """Compresses OpenGL colour buffers into a video elementary stream.

    A live encoder owns driver resources and, on most backends, is bound to the
    OpenGL context that was current when it was opened. Use it from that thread,
    and close it when the recording ends -- as a context manager, if that suits::

        with open_encoder(1920, 1080, fps=60) as encoder:
            handle = encoder.register(texture)
            packets = encoder.encode(handle, timestamp)
    """

    #: the codec of the stream being produced, e.g. ``'h264'``
    codec: str = 'h264'
    #: frame size, as (width, height)
    size: tuple[int, int] = (0, 0)
    #: units of a second that timestamps and durations are counted in
    timescale: int = 90000
    #: True when frames reach the encoder without passing through host memory
    zero_copy: bool = False
    #: True when packets come out in decode order rather than display order,
    #: which is what a container needs to know before it writes composition times
    reorders_frames: bool = False

    @abstractmethod
    def register(self, texture: int, target: int | None = None) -> Any:
        """Make `texture` available to the encoder, returning an input handle.

        Registration is expensive and the handle is reusable: register the
        textures a recording will cycle through once, then encode from them
        repeatedly.
        """

    @abstractmethod
    def encode(self, handle: Any, timestamp: int, duration: int = 0,
               force_idr: bool = False) -> list[Packet]:
        """Submit the picture currently in `handle`; return whatever came out.

        timestamp/duration are in :attr:`timescale` units. `force_idr` starts a
        new closed group of pictures, which is what a seek point is made of.
        """

    @abstractmethod
    def flush(self) -> list[Packet]:
        """Finish the stream, returning every picture still held inside."""

    @abstractmethod
    def headers(self) -> bytes:
        """The stream's parameter sets, for containers that store them up front."""

    @abstractmethod
    def close(self) -> None:
        """Release the driver resources. Encoding after this is an error."""

    def __enter__(self) -> Encoder:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()


#: Every backend that exists, in the order :func:`open_encoder` prefers them.
#: Populated at import of :mod:`pyopengl_video`; tests replace it wholesale.
BACKENDS: list[Backend] = []


def available_backends(backends: Iterable[Backend] | None = None) -> Iterator[Backend]:
    """Yield the backends whose hardware and driver are present."""
    for backend in (BACKENDS if backends is None else backends):
        if backend.probe():
            yield backend
