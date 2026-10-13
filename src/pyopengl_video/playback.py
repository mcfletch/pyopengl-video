"""Playing a movie into an OpenGL texture, with no scenegraph.

:class:`Movie` is the file: its video track, opened and indexed with no GL, so
it can be opened on any thread. :class:`Player` is a movie playing in one
OpenGL context: a decoder, the conversion pass and a few output textures::

    movie = Movie.open('intro.mp4')          # any thread: reads the index
    player = movie.player()                  # GL thread: decoder, converter
    texture = player.frame_at(seconds)       # an RGBA texture, right way up

:meth:`Player.frame_at` shows the last frame at or before the time asked for:
the first frame before the start, the last after the end. Playing forward
decodes each sample once. A time behind the frame shown, or past the next
key frame, resets the decoder and decodes from the key frame before it. The
player decodes ahead into spare output textures, so the next frame is usually
converted before it is asked for.

The caller's clock decides which time is asked for, so a fixed-step capture
gets every frame in order.

The decisions are in :class:`Timeline` and :class:`Player`, which take their
decoder, converter and texture factory as arguments and hold no GL of their
own; :meth:`Movie.player` supplies the real ones.

A container states a picture size, and a stream has one. :attr:`Movie.size`
is what the container states, which nothing holds it to. The player's
textures are the size of the pictures the decoder returns, each one, so a
texture always holds exactly the picture drawn into it; :attr:`Player.size`
is the size of the frame shown.
"""
from __future__ import annotations

import bisect
import collections
import dataclasses
import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from pyopengl_video import codecs, inputs, ivf, open_decoder
from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.container import ContainerError
from pyopengl_video.convert import NV12Converter, create_output_texture, generate_mipmaps
from pyopengl_video.decoder import (
    DecodedFrame,
    Decoder,
    DecoderError,
    DecoderUnavailable,
    Sample,
)
from pyopengl_video.ivf import IVFReader
from pyopengl_video.mp4 import MP4Error, MP4Reader
from pyopengl_video.teardown import release_all

__all__ = ['DEFAULT_AHEAD', 'Movie', 'Player', 'Timeline', 'VideoSource']

log = logging.getLogger(__name__)

#: How many converted frames the player keeps ready beyond the one shown.
DEFAULT_AHEAD = 2

#: The first box types an MP4 file may open with.
MP4_BOXES = (b'ftyp', b'moov', b'mdat', b'free', b'skip', b'wide', b'pnot')

#: Containers and streams this package does not read, by the bytes they open with.
OTHER_CONTAINERS = (
    (b'RIFF', 'an AVI or other RIFF file'),
    (b'\x00\x00\x01\xba', 'an MPEG program stream'),
    (b'\x00\x00\x01\xb3', 'an MPEG-1 or MPEG-2 video stream'),
    (b'\x1a\x45\xdf\xa3', 'a Matroska or WebM file'),
    (b'OggS', 'an Ogg file'),
)


class VideoSource(Protocol):
    """What a container's video track offers playback.

    :class:`~pyopengl_video.mp4.VideoTrack`,
    :class:`~pyopengl_video.annexb.AnnexBReader` and
    :class:`~pyopengl_video.ivf.IVFReader` are each one.

    codec -- the stream's codec, by any name
        :func:`pyopengl_video.codecs.canonical` takes
    size -- the picture size the source states, as (width, height)
    """

    codec: str
    timescale: int

    @property
    def sync_samples(self) -> list[int]: ...

    @property
    def size(self) -> tuple[int, int]: ...

    def __len__(self) -> int: ...

    def sample(self, index: int, parameter_sets: bool = False) -> Sample: ...

    def presentation_time(self, index: int) -> int: ...

    def keyframe_before(self, index: int) -> int: ...

    def parameter_sets(self) -> bytes: ...


class Timeline:
    """Where each frame of a source falls in time, and where to decode it from.

    A frame's *rank* is its place in display order. Times are in seconds from
    the first frame shown, whatever the container's own times start at.

    Raises :class:`~pyopengl_video.container.ContainerError` for a source of
    no samples, or one whose timescale is not above zero.
    """

    def __init__(self, source: VideoSource) -> None:
        self.source = source
        count = len(source)
        if not count:
            raise ContainerError('the video track has no samples')
        if source.timescale <= 0:
            raise ContainerError(f'the video track states a timescale of '
                                 f'{source.timescale}, and a second has at least one unit')
        times = [source.presentation_time(index) for index in range(count)]
        self.order = sorted(range(count), key=times.__getitem__)
        self.rank_of_sample = [0] * count
        for rank, index in enumerate(self.order):
            self.rank_of_sample[index] = rank
        self._rank_of_time = {times[index]: rank for rank, index in enumerate(self.order)}
        origin = times[self.order[0]]
        self.timescale = source.timescale
        self._seconds = [(times[index] - origin) / self.timescale for index in self.order]
        last = source.sample(self.order[-1]).duration
        self.duration = self._seconds[-1] + last / self.timescale

    def __len__(self) -> int:
        return len(self.order)

    def seconds(self, rank: int) -> float:
        """When the frame of `rank` is first shown."""
        return self._seconds[rank]

    def rank_at(self, seconds: float) -> int:
        """The frame shown at `seconds`: the last one at or before it."""
        found = bisect.bisect_right(self._seconds, seconds) - 1
        return min(max(found, 0), len(self.order) - 1)

    def rank_of_timestamp(self, timestamp: int) -> int | None:
        """The rank of the frame whose presentation time is `timestamp`."""
        return self._rank_of_time.get(timestamp)

    def start_for(self, rank: int) -> int:
        """The sample to start decoding at to show the frame of `rank`.

        The sync sample before it in decoding order, unless that key frame is
        shown after it: a leading picture of an open group, which needs the
        group before.
        """
        key = self.source.keyframe_before(self.order[rank])
        while key > 0 and self.rank_of_sample[key] > rank:
            key = self.source.keyframe_before(key - 1)
        return key


class Movie:
    """A movie's video track, indexed for playback. No GL.

    source -- the track: an MP4 :class:`~pyopengl_video.mp4.VideoTrack`, an
        :class:`~pyopengl_video.annexb.AnnexBReader` or an
        :class:`~pyopengl_video.ivf.IVFReader`
    size -- the picture size to state, where the source's own is not the
        one to use

    codec -- the stream's codec: ``'h264'``, ``'hevc'``, ``'av1'`` or ``'vp9'``
    size -- the picture size the container states, as (width, height). An MP4
        or an IVF file may state any size, (0, 0) included, whatever its
        stream holds; a raw stream's is read from its parameter sets. It
        chooses the decoder backend and sizes nothing: :attr:`Player.size`
        is the size of the pictures themselves.
    """

    def __init__(self, source: VideoSource, *, size: tuple[int, int] | None = None,
                 owner: Any = None) -> None:
        self.source = source
        self.codec = codecs.canonical(source.codec)
        self.timeline = Timeline(source)
        self.headers = source.parameter_sets()
        self.size: tuple[int, int] = size or source.size
        self._owner = owner

    @classmethod
    def open(cls, source: str | Path | BinaryIO,
             frame_rate: tuple[int, int] | None = None) -> Movie:
        """Open an MP4 file, an IVF file or a raw H.264 or HEVC stream, by its contents.

        frame_rate -- for a raw stream that states none, frames per second as
            ``(numerator, denominator)``

        Raises :class:`~pyopengl_video.container.ContainerError`, which is a
        :class:`ValueError` and a :class:`~pyopengl_video.backend.VideoError`,
        for a file that is none of them, naming the container where its first
        bytes say which it is. A file that is one of them and breaks its
        syntax raises that reader's error:
        :class:`~pyopengl_video.mp4.MP4Error`,
        :class:`~pyopengl_video.ivf.IVFError` or
        :class:`~pyopengl_video.vaapi.h264.StreamError`. The file is closed
        before any of them is raised.
        """
        if isinstance(source, (str, Path)):
            with Path(source).open('rb') as handle:
                head = handle.read(8)
        else:
            position = source.tell()
            head = source.read(8)
            source.seek(position)
        for opening, named in OTHER_CONTAINERS:
            if head.startswith(opening):
                raise ContainerError(f'{source} is {named}; a Movie reads MP4 and IVF '
                                     'files and raw H.264 and HEVC streams')
        owner: MP4Reader | IVFReader | AnnexBReader
        if head[4:8] in MP4_BOXES:
            owner = MP4Reader(source)
        elif head.startswith(ivf.MAGIC):
            owner = IVFReader(source)
        else:
            owner = AnnexBReader(source, frame_rate=frame_rate)
        try:
            return cls(cls._track(owner, source), owner=owner)
        except BaseException:
            owner.close()
            raise

    @staticmethod
    def _track(reader: MP4Reader | IVFReader | AnnexBReader, source: Any) -> VideoSource:
        """The track of `reader` a movie plays: an MP4's video, or the stream itself."""
        if not isinstance(reader, MP4Reader):
            return reader
        if reader.video is None:
            found = sorted({track.codec for track in reader.tracks
                            if track.handler == 'vide'})
            raise MP4Error(
                f'{source} has no video track in a codec a Movie plays '
                f'({", ".join(sorted(codecs.CODECS))}); its video is '
                f'{", ".join(found) or "absent"}')
        if not len(reader.video):
            raise MP4Error(f'the video track of {source} has no samples')
        return reader.video

    def __repr__(self) -> str:
        width, height = self.size
        return (f'<{type(self).__name__} {width}x{height} {self.codec}, '
                f'{self.frames} frames, {self.duration:.2f}s>')

    @property
    def frames(self) -> int:
        return len(self.timeline)

    @property
    def duration(self) -> float:
        """The movie's length in seconds."""
        return self.timeline.duration

    def player(self, *, backend: str | None = None, ahead: int = DEFAULT_AHEAD,
               mipmaps: bool = False, **options: Any) -> Player:
        """A player for this movie in the current OpenGL context.

        backend -- the decoder backend by name; the best available otherwise
        ahead -- converted frames kept ready beyond the one shown
        mipmaps -- give the output textures mipmaps, made after each conversion
        options -- passed to the decoder

        The size the container states chooses the backend, where it states
        one: a width or a height of 0 is no statement, and any backend that
        decodes the codec is taken. So is one where no backend takes the
        stated size, since the stream may be smaller than its container
        says; a decoder refuses a picture it cannot hold when it meets one.
        """
        stated = self.size if min(self.size) > 0 else None
        try:
            decoder = open_decoder(self.codec, headers=self.headers, size=stated,
                                   backend=backend, **options)
        except DecoderUnavailable:
            if stated is None:
                raise
            decoder = open_decoder(self.codec, headers=self.headers, size=None,
                                   backend=backend, **options)
        try:
            converter = NV12Converter()
        except BaseException:
            decoder.close()
            raise
        return Player(self, decoder, converter, ahead=ahead, mipmaps=mipmaps)

    def close(self) -> None:
        """Close the file, where the movie opened it."""
        owner, self._owner = self._owner, None
        if owner is not None:
            owner.close()

    def __enter__(self) -> Movie:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()


class Converter(Protocol):
    """What the player draws decoded frames into its textures with."""

    def convert(self, frame: DecodedFrame, target: int) -> None: ...

    def forget(self, target: int) -> None:
        """Drop what is kept for `target`, which is about to be deleted."""

    def close(self) -> None: ...


@dataclasses.dataclass
class _Ready:
    """A converted frame: its rank, the texture holding it, and its size."""

    rank: int
    texture: int
    size: tuple[int, int]


class Player:
    """A movie playing in one OpenGL context.

    movie -- what to play
    decoder -- a decoder for its stream; the player owns it and closes it
    converter -- draws decoded frames into output textures; owned likewise
    ahead -- converted frames to keep ready beyond the one shown
    mipmaps -- make mipmaps for each output texture after converting into it
    make_texture/delete_texture -- the output texture factory; RGBA textures
        from :func:`~pyopengl_video.convert.create_output_texture` by default

    A texture :meth:`frame_at` returns is valid until the next call.

    Each output texture is the size of the picture converted into it, which is
    the decoder's and not the size :attr:`Movie.size` states. Where a stream
    changes size, the textures of the old size are deleted as they fall out of
    use and textures of the new size are made, so :attr:`size` may differ
    from one frame to the next.
    """

    def __init__(self, movie: Movie, decoder: Decoder, converter: Converter, *,
                 ahead: int = DEFAULT_AHEAD, mipmaps: bool = False,
                 make_texture: Callable[[int, int], int] | None = None,
                 delete_texture: Callable[[int], None] | None = None) -> None:
        self.movie = movie
        self.timeline = movie.timeline
        self.decoder = decoder
        self.converter = converter
        self.ahead = int(ahead)
        self.mipmaps = mipmaps
        self._make = make_texture or self._output_texture
        self._delete = delete_texture or inputs.delete_texture
        #: every output texture there is, and its size
        self._textures: dict[int, tuple[int, int]] = {}
        self._free: list[int] = []
        self._ready: list[_Ready] = []
        self._shown: _Ready | None = None
        self._next = 0
        self._start = 0
        self._fed = False
        self._exhausted = False
        self._floor = 0
        self._closed = False

    def __repr__(self) -> str:
        shown = 'nothing' if self._shown is None else f'frame {self._shown.rank}'
        return f'<{type(self).__name__} {self.movie!r} showing {shown}>'

    @property
    def frame(self) -> int | None:
        """The rank of the frame shown, or None before the first call."""
        return None if self._shown is None else self._shown.rank

    @property
    def texture(self) -> int:
        """The texture of the frame shown, or 0 before the first call."""
        return 0 if self._shown is None else self._shown.texture

    @property
    def size(self) -> tuple[int, int] | None:
        """The size of the frame shown and of :attr:`texture`, as (width, height).

        None before the first call. It is the decoded picture's own size.
        """
        return None if self._shown is None else self._shown.size

    @property
    def ready_ahead(self) -> int:
        """How many converted frames wait beyond the one shown."""
        return len(self._ready)

    def frame_at(self, seconds: float) -> int:
        """The RGBA texture showing the movie at `seconds`.

        Raises :class:`~pyopengl_video.decoder.DecoderError` after
        :meth:`close`, and for a picture the decoder returns with no pixels.
        """
        if self._closed:
            raise DecoderError('the player is closed')
        target = self.timeline.rank_at(seconds)
        if self._shown is not None and self._shown.rank == target:
            return self._shown.texture
        if self._needs_seek(target):
            self._seek(target)
        while not self._covers(target) and not self._exhausted:
            self._step()
        self._show(target)
        while len(self._ready) < self.ahead and not self._exhausted:
            self._step()
        return self.texture

    # ---------------------------------------------------------- decisions

    def _needs_seek(self, target: int) -> bool:
        """Behind the frame shown, or past a key frame not yet reached."""
        if self._shown is not None and target < self._shown.rank:
            return True
        if self._covers(target):
            return False
        return self.timeline.start_for(target) > self._next

    def _covers(self, target: int) -> bool:
        return any(entry.rank >= target for entry in self._ready)

    def _seek(self, target: int) -> None:
        for entry in self._ready:
            self._free.append(entry.texture)
        self._ready = []
        if self._fed:
            self.decoder.reset()
            self._fed = False
        self._start = self._next = self.timeline.start_for(target)
        self._exhausted = False
        self._floor = 0

    def _show(self, target: int) -> None:
        """Make the latest ready frame at or before `target` the one shown."""
        if not self._ready:
            return
        chosen = 0
        for index, entry in enumerate(self._ready):
            if entry.rank <= target:
                chosen = index
        if self._shown is not None:
            self._free.append(self._shown.texture)
        for entry in self._ready[:chosen]:
            self._free.append(entry.texture)
        self._shown = self._ready[chosen]
        self._ready = self._ready[chosen + 1:]
        self._floor = self._shown.rank + 1

    # ------------------------------------------------------------ decoding

    def _step(self) -> None:
        """Feed the decoder one sample and convert what comes out.

        Every frame the decoder lends is given back, whether it was converted
        or the step failed before reaching it.
        """
        index = self._next
        source = self.movie.source
        frames: collections.deque[DecodedFrame] = collections.deque()
        try:
            frames.extend(self.decoder.decode(
                source.sample(index, parameter_sets=index == self._start)))
            self._fed = True
            self._next += 1
            if self._next >= len(source):
                self._exhausted = True
                frames.extend(self.decoder.flush())
            while frames:
                self._accept(frames.popleft())
        finally:
            release_all(*(frame.release for frame in frames))

    def _accept(self, frame: DecodedFrame) -> None:
        with frame:
            rank = self.timeline.rank_of_timestamp(frame.timestamp)
            if rank is None or rank < self._floor:
                return
            width, height = size = frame.size
            if width <= 0 or height <= 0:
                raise DecoderError(f'the decoder returned a picture of {width}x{height}, '
                                   'which has no pixels')
            texture = self._texture_of(size)
            try:
                self.converter.convert(frame, texture)
            except BaseException:
                self._free.append(texture)
                raise
        if self.mipmaps:
            generate_mipmaps(texture)
        self._ready.append(_Ready(rank, texture, size))
        self._ready.sort(key=lambda entry: entry.rank)

    def _texture_of(self, size: tuple[int, int]) -> int:
        """An output texture of `size` that holds no frame in use.

        A free texture of that size, or a new one. Free textures of any other
        size are deleted first: the stream has changed size, and nothing is
        drawn into a texture that is not the picture's size.
        """
        for texture in reversed(self._free):
            if self._textures[texture] == size:
                self._free.remove(texture)
                return texture
        while self._free:
            self._retire(self._free.pop())
        texture = self._make(*size)
        self._textures[texture] = size
        return texture

    def _retire(self, texture: int) -> None:
        """Delete an output texture, and the converter's framebuffer for it first.

        OpenGL gives a deleted texture's name to a later one, which must not
        find the framebuffer of the texture that had the name before.
        """
        self.converter.forget(texture)
        self._delete(texture)
        del self._textures[texture]

    def _output_texture(self, width: int, height: int) -> int:
        levels = max(width, height).bit_length() if self.mipmaps else 1
        return create_output_texture(width, height, levels)

    def close(self) -> None:
        """Close the converter and the decoder and delete the output textures.

        Each is released whatever the others do; the first failure is raised
        once all have been tried. Safe to call twice.
        """
        if self._closed:
            return
        self._closed = True
        textures, self._textures = list(self._textures), {}
        self._free = []
        self._ready = []
        self._shown = None
        release_all(self.converter.close, self.decoder.close,
                    *(functools.partial(self._delete, texture) for texture in textures))

    def __enter__(self) -> Player:
        return self

    def __exit__(self, kind: type[BaseException] | None, *exception: Any) -> None:
        if kind is None:
            self.close()
            return
        try:
            self.close()
        except Exception as error:  # noqa: BLE001 - the block's own exception is the one to raise
            log.warning('the player was not closed cleanly: %s', error)
