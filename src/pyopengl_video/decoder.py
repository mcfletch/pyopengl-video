"""The interface every decoder backend implements, and the registry of backends.

A backend turns compressed samples into OpenGL textures. Containers
(:class:`~pyopengl_video.mp4.MP4Reader`, :class:`~pyopengl_video.annexb.AnnexBReader`)
hand out :class:`Sample` objects, a decoder takes them in decoding order, and
the pictures come back in display order.

A decoder holds pictures back until it knows their display order, so
:meth:`Decoder.decode` returns a list: empty is ordinary, and so is more than
one. :meth:`Decoder.flush` drains what is left at the end of a stream.

A :class:`DecodedFrame` is lent, not given. Its textures are the decoder's own
surfaces, so the frame stays the decoder's until :meth:`DecodedFrame.release`,
and a decoder whose surfaces are all lent out raises :class:`DecoderError`
naming the frames still held.
"""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from pyopengl_video.backend import Backend, VideoError, available
from pyopengl_video.backend import explanations as backend_explanations
from pyopengl_video.colour import Colour

__all__ = ['BACKENDS', 'DecodedFrame', 'Decoder', 'DecoderError',
           'DecoderUnavailable', 'Sample', 'available_backends', 'explanations']


class DecoderError(VideoError):
    """A decoder rejected a stream or a call, or failed part-way through one."""


class DecoderUnavailable(DecoderError):
    """No backend on this machine can decode what was asked for."""


@dataclasses.dataclass(frozen=True)
class Sample:
    """One compressed sample, as a container stores it.

    data -- for video, an Annex-B fragment: each NAL unit behind a start code;
        for audio, the codec's own frame as the container holds it
    timestamp -- presentation time, in the track's timescale
    duration -- how long the sample is shown or played, in the same units
    keyframe -- decoding can start at this sample
    index -- the sample's position in decoding order
    """

    data: bytes
    timestamp: int
    duration: int
    keyframe: bool
    index: int

    def __len__(self) -> int:
        return len(self.data)


class DecodedFrame:
    """One decoded picture in OpenGL textures, lent by its decoder.

    timestamp/duration -- the sample's, in the decoder's timescale
    planes -- the texture names: luma (``GL_R8``) and chroma (``GL_RG8``, Cb
        in red and Cr in green) for the ``'nv12'`` layout, or one RGBA texture
        for ``'rgba'``
    layout -- ``'nv12'`` or ``'rgba'``
    texture_size -- the luma texture's size, which is the decoder's surface
        and usually larger than the picture
    visible -- the picture within it, as ``(x, y, width, height)``, first row
        at the top: the textures hold the picture as the decoder wrote it
    colour -- how the samples become RGB
    on_release -- what the decoder does when the frame is given back

    :class:`~pyopengl_video.convert.NV12Converter` turns an ``'nv12'`` frame
    into an RGBA texture the right way up for OpenGL.
    """

    def __init__(self, *, timestamp: int, duration: int, planes: tuple[int, ...],
                 layout: str, texture_size: tuple[int, int],
                 visible: tuple[int, int, int, int], colour: Colour,
                 on_release: Callable[[DecodedFrame], None]) -> None:
        self.timestamp = timestamp
        self.duration = duration
        self.planes = planes
        self.layout = layout
        self.texture_size = texture_size
        self.visible = visible
        self.colour = colour
        self._on_release: Callable[[DecodedFrame], None] | None = on_release

    def __repr__(self) -> str:
        width, height = self.size
        state = ' released' if self.released else ''
        return (f'<{type(self).__name__} {self.layout} {width}x{height} '
                f'at {self.timestamp}{state}>')

    @property
    def size(self) -> tuple[int, int]:
        """The picture's visible size, as (width, height)."""
        return self.visible[2], self.visible[3]

    @property
    def released(self) -> bool:
        return self._on_release is None

    def release(self) -> None:
        """Give the frame's surface back to the decoder. Safe to call twice.

        The textures must not be read after this: the decoder writes the next
        picture into them.
        """
        on_release, self._on_release = self._on_release, None
        if on_release is not None:
            on_release(self)

    def __enter__(self) -> DecodedFrame:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.release()


class Decoder(ABC):
    """Decodes a compressed video stream into OpenGL textures.

    A live decoder owns driver resources and is bound to the OpenGL context
    current when it was opened. Use it from that thread, and close it when the
    stream ends::

        with open_decoder('h264', headers=track.parameter_sets()) as decoder:
            for frame in decoder.decode(sample):
                draw(frame)
                frame.release()
    """

    #: the codec of the stream, e.g. ``'h264'``
    codec: str = 'h264'
    #: the visible picture size, as (width, height); (0, 0) until the stream's
    #: parameter sets have been seen
    size: tuple[int, int] = (0, 0)
    #: how the stream's samples become RGB
    colour: Colour = Colour()
    #: True when frames never pass through host memory
    zero_copy: bool = False
    #: how many frames a caller may hold unreleased at once
    output_slots: int = 0

    @abstractmethod
    def decode(self, sample: Any) -> list[DecodedFrame]:
        """Decode one sample; return the frames now ready, in display order."""

    @abstractmethod
    def flush(self) -> list[DecodedFrame]:
        """Return every frame still held back, at the end of a stream."""

    @abstractmethod
    def reset(self) -> None:
        """Drop every picture held back, before decoding from a seek point."""

    @abstractmethod
    def close(self) -> None:
        """Release the driver resources. Decoding after this is an error."""

    def __enter__(self) -> Decoder:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()


#: Every decoder backend, in the order :func:`~pyopengl_video.open_decoder`
#: prefers them. Populated at import of :mod:`pyopengl_video`.
BACKENDS: list[Backend[Decoder]] = []


def available_backends(backends: Iterable[Backend[Decoder]] | None = None,
                       ) -> Iterator[Backend[Decoder]]:
    """Yield the decoder backends whose hardware and driver are present."""
    return available(BACKENDS if backends is None else backends)


def explanations(backends: Iterable[Backend[Decoder]] | None = None) -> list[str]:
    """Why each unavailable decoder backend declined, for the ones that will say."""
    return backend_explanations(BACKENDS if backends is None else backends)
