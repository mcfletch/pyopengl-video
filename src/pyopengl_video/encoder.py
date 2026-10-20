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
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

from pyopengl_video.backend import Backend, VideoError, available
from pyopengl_video.backend import explanations as backend_explanations
from pyopengl_video.inputs import (
    InputHandle,
    create_framebuffer,
    create_rgba_texture,
    delete_texture,
)

__all__ = ['BACKENDS', 'Backend', 'Encoder', 'EncoderError', 'EncoderUnavailable',
           'Packet', 'available_backends', 'default_bitrate', 'explanations',
           'frame_rate_ratio']

log = logging.getLogger(__name__)

#: Bits per pixel per second, used when a caller names no bitrate. 1080p60 lands
#: near 31 Mbit/s. Dense foliage at 60 frames a second changes every pixel
#: between frames, and at a quarter of this rate it blocks and smears.
DEFAULT_BITS_PER_PIXEL = 0.25


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


class EncoderError(VideoError):
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


class Encoder(ABC):
    """Compresses OpenGL colour buffers into a video elementary stream.

    A live encoder owns driver resources and, on most backends, is bound to the
    OpenGL context that was current when it was opened. Use it from that thread,
    and close it when the recording ends -- as a context manager, if that suits::

        with open_encoder(1920, 1080, fps=60) as encoder:
            handle = encoder.new_input()
            with handle.for_drawing():
                copy_the_frame_into(handle.framebuffer)
            packets = encoder.encode(handle, timestamp)
    """

    #: the codec of the stream being produced, e.g. ``'h264'``
    codec: str = 'h264'
    #: frame size, as (width, height)
    size: tuple[int, int] = (0, 0)
    #: units of a second that timestamps and durations are counted in
    timescale: int = 90000
    #: how often frames are shown, as an exact (numerator, denominator); set by
    #: the backend as it opens, and what :meth:`frame_duration` measures a
    #: frame against
    frame_rate: tuple[int, int] = (60, 1)
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
        texture = create_rgba_texture(*self.size)
        try:
            handle = self.register(texture)
        except BaseException:
            delete_texture(texture)
            raise
        # From here the handle deletes the texture, when it is closed.
        handle.owns_texture = True
        try:
            handle.framebuffer = create_framebuffer(texture, handle.target)
        except BaseException:
            handle.close()
            raise
        return handle

    @abstractmethod
    def register(self, texture: int, target: int | None = None) -> Any:
        """Make `texture` available to the encoder, returning an input handle.

        Registration is expensive and the handle is reusable: register the
        textures a recording will cycle through once, then encode from them
        repeatedly. A backend whose :attr:`allocates_inputs` is set refuses a
        texture it did not make; use :meth:`new_input` there.
        """

    def frame_duration(self, duration: int = 0) -> int:
        """`duration` in :attr:`timescale` units, or one frame's worth if it is 0.

        Every backend puts an incoming duration through this. A container
        takes its per-sample durations from what the encoder reports, and a
        stream of zeroes gives a movie of zero length whose frame rate is
        undefined -- ``duration=0`` and ``r_frame_rate=1/0`` -- which players
        refuse although every picture decodes.

        Raises :class:`EncoderError` for a frame rate of no frames, which has
        no frame duration.
        """
        if duration > 0:
            return int(duration)
        numerator, denominator = self.frame_rate
        if numerator <= 0 or denominator <= 0:
            raise EncoderError(
                f'a frame rate of {numerator}/{denominator} has no frame duration')
        return round(self.timescale * denominator / numerator)

    @abstractmethod
    def encode(self, handle: Any, timestamp: int, duration: int = 0,
               force_idr: bool = False) -> list[Packet]:
        """Submit the picture currently in `handle`; return whatever came out.

        timestamp/duration are in :attr:`timescale` units, and a `duration` of 0
        means one frame at :attr:`frame_rate` -- put it through
        :meth:`frame_duration` rather than passing it on. `force_idr` starts a
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
BACKENDS: list[Backend[Encoder]] = []


def available_backends(backends: Iterable[Backend[Encoder]] | None = None,
                       ) -> Iterator[Backend[Encoder]]:
    """Yield the encoder backends whose hardware and driver are present."""
    return available(BACKENDS if backends is None else backends)


def explanations(backends: Iterable[Backend[Encoder]] | None = None) -> list[str]:
    """Why each unavailable encoder backend declined, for the ones that will say."""
    return backend_explanations(BACKENDS if backends is None else backends)
