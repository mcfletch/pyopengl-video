"""The interface every encoder backend implements, and the registry of backends.

A backend turns an OpenGL colour buffer into compressed video. What differs
between them is how the frame reaches the encoder -- an NVIDIA session takes a
texture name, a VA-API session takes a DMA-BUF exported from one -- so the
interface is written in terms of a *registered input*: the caller hands over a
texture once, gets a handle back, and encodes with that handle every frame.

Which side creates that texture is not the same everywhere. A Linux encoder
takes one the caller made; a Windows encoder reads a Direct3D resource that only
the backend can allocate. So :meth:`Encoder.new_input` is how an input is
obtained on every backend, and :meth:`Encoder.register` remains for a caller
bringing its own texture to one that accepts a foreign one.

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

from pyopengl_video.inputs import InputHandle

#: Bits per pixel per second, used when a caller names no bitrate. 1080p60 lands
#: near 8.7 Mbit/s, which is a reasonable quality for screen-captured 3D.
DEFAULT_BITS_PER_PIXEL = 0.07


def frame_rate_ratio(fps: float | tuple[int, int]) -> tuple[int, int]:
    """A frame rate as an exact numerator and denominator.

    A float rate becomes the ratio broadcast uses for it, so 29.97 records as
    30000/1001 rather than as an approximation that drifts over a long
    recording.
    """
    if isinstance(fps, tuple):
        return int(fps[0]), int(fps[1])
    if abs(fps - round(fps)) < 1e-6:
        return int(round(fps)), 1
    return int(round(fps * 1001)), 1001


def default_bitrate(size: tuple[int, int], frame_rate: tuple[int, int]) -> int:
    """Bits per second for a frame size and rate, at :data:`DEFAULT_BITS_PER_PIXEL`."""
    width, height = size
    numerator, denominator = frame_rate
    return int(width * height * (numerator / denominator) * DEFAULT_BITS_PER_PIXEL)


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
    #: True when :meth:`register` will not take a texture the caller made, so
    #: :meth:`new_input` is the only way to get an input. Windows backends read a
    #: Direct3D resource, which has to exist before OpenGL can name it.
    allocates_inputs: bool = False

    def new_input(self) -> InputHandle:
        """A texture this encoder reads, with a framebuffer that fills it.

        The way to get an input on any backend. Ask for
        :attr:`input_slots` of them at the start of a recording and cycle
        through them; draw into one inside its
        :meth:`~pyopengl_video.inputs.InputHandle.for_drawing` scope.

        The default makes an ordinary ``GL_RGBA8`` texture and registers it,
        which is what a backend accepting a foreign texture wants. Backends
        whose input must be allocated by the driver override this.
        """
        from pyopengl_video.inputs import create_framebuffer, create_rgba_texture
        texture = create_rgba_texture(*self.size)
        handle = self.register(texture)
        handle.framebuffer = create_framebuffer(texture, handle.target)
        handle.owns_texture = True
        return handle

    @abstractmethod
    def register(self, texture: int, target: int | None = None) -> Any:
        """Make `texture` available to the encoder, returning an input handle.

        Registration is expensive and the handle is reusable: register the
        textures a recording will cycle through once, then encode from them
        repeatedly. A backend whose :attr:`allocates_inputs` is set refuses a
        texture it did not make; use :meth:`new_input` there.
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
