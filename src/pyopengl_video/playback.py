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
"""
from __future__ import annotations

import bisect
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from pyopengl_video import inputs, open_decoder
from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.convert import NV12Converter, create_output_texture, generate_mipmaps
from pyopengl_video.decoder import DecodedFrame, Decoder, Sample
from pyopengl_video.mp4 import MP4Reader, split_annexb
from pyopengl_video.vaapi.h264_parse import NAL_SPS, parse_nal, parse_sps

__all__ = ['DEFAULT_AHEAD', 'Movie', 'Player', 'Timeline', 'VideoSource']

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

    :class:`~pyopengl_video.mp4.VideoTrack` and
    :class:`~pyopengl_video.annexb.AnnexBReader` are both one.
    """

    timescale: int
    sync_samples: list[int]

    def __len__(self) -> int: ...

    def sample(self, index: int, parameter_sets: bool = False) -> Sample: ...

    def presentation_time(self, index: int) -> int: ...

    def keyframe_before(self, index: int) -> int: ...

    def parameter_sets(self) -> bytes: ...


class Timeline:
    """Where each frame of a source falls in time, and where to decode it from.

    A frame's *rank* is its place in display order. Times are in seconds from
    the first frame shown, whatever the container's own times start at.
    """

    def __init__(self, source: VideoSource) -> None:
        self.source = source
        count = len(source)
        if not count:
            raise ValueError('the video track has no samples')
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

    source -- the track: an MP4 :class:`~pyopengl_video.mp4.VideoTrack` or an
        :class:`~pyopengl_video.annexb.AnnexBReader`
    size -- the picture size, when the parameter sets do not say it
    """

    codec = 'h264'

    def __init__(self, source: VideoSource, *, size: tuple[int, int] | None = None,
                 owner: Any = None) -> None:
        self.source = source
        self.timeline = Timeline(source)
        self.headers = source.parameter_sets()
        self.size = size or _size_of(self.headers)
        self._owner = owner

    @classmethod
    def open(cls, source: str | Path | BinaryIO,
             frame_rate: tuple[int, int] | None = None) -> Movie:
        """Open an MP4 file or a raw H.264 stream, by its contents.

        Raises ValueError for a file that is neither, naming the container
        where its first bytes say which it is.

        frame_rate -- for a raw stream that states none, frames per second as
            ``(numerator, denominator)``
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
                raise ValueError(f'{source} is {named}; a Movie reads MP4 files '
                                 'and raw H.264 streams')
        if head[4:8] in MP4_BOXES:
            reader = MP4Reader(source)
            if reader.video is None:
                reader.close()
                raise ValueError(f'{source} has no H.264 video track')
            return cls(reader.video, owner=reader)
        stream = AnnexBReader(source, frame_rate=frame_rate)
        return cls(stream, owner=stream)

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
        """
        decoder = open_decoder(self.codec, headers=self.headers, size=self.size,
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


def _size_of(headers: bytes) -> tuple[int, int]:
    """The visible size the first sequence parameter set in `headers` states."""
    for unit in split_annexb(headers):
        found = parse_nal(unit)
        if found.nal_unit_type == NAL_SPS:
            return parse_sps(found.rbsp).size
    raise ValueError('the movie states no sequence parameter set, so its size is unknown')


class Converter(Protocol):
    """What the player draws decoded frames into its textures with."""

    def convert(self, frame: DecodedFrame, target: int) -> None: ...

    def close(self) -> None: ...


@dataclasses.dataclass
class _Ready:
    """A converted frame: its rank and the texture holding it."""

    rank: int
    texture: int


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
        self._textures: list[int] = []
        self._free: list[int] = []
        self._ready: list[_Ready] = []
        self._shown: _Ready | None = None
        self._next = 0
        self._start = 0
        self._fed = False
        self._exhausted = False
        self._floor = 0

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
    def ready_ahead(self) -> int:
        """How many converted frames wait beyond the one shown."""
        return len(self._ready)

    def frame_at(self, seconds: float) -> int:
        """The RGBA texture showing the movie at `seconds`."""
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
        """Feed the decoder one sample and convert what comes out."""
        index = self._next
        source = self.movie.source
        frames = self.decoder.decode(source.sample(index, parameter_sets=index == self._start))
        self._fed = True
        self._next += 1
        if self._next >= len(source):
            frames += self.decoder.flush()
            self._exhausted = True
        for frame in frames:
            self._accept(frame)

    def _accept(self, frame: DecodedFrame) -> None:
        with frame:
            rank = self.timeline.rank_of_timestamp(frame.timestamp)
            if rank is None or rank < self._floor:
                return
            texture = self._free.pop() if self._free else self._new_texture()
            self.converter.convert(frame, texture)
        if self.mipmaps:
            generate_mipmaps(texture)
        self._ready.append(_Ready(rank, texture))
        self._ready.sort(key=lambda entry: entry.rank)

    def _new_texture(self) -> int:
        texture = self._make(*self.movie.size)
        self._textures.append(texture)
        return texture

    def _output_texture(self, width: int, height: int) -> int:
        levels = max(width, height).bit_length() if self.mipmaps else 1
        return create_output_texture(width, height, levels)

    def close(self) -> None:
        """Close the decoder and the converter and delete the output textures."""
        self.converter.close()
        self.decoder.close()
        for texture in self._textures:
            self._delete(texture)
        self._textures = []
        self._free = []
        self._ready = []
        self._shown = None

    def __enter__(self) -> Player:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()
