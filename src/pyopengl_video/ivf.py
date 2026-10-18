"""Reading and writing IVF: the plain container VP9 and AV1 streams are kept in.

An IVF file is a 32-byte header and then each frame behind twelve bytes giving
its size and presentation time. It holds one video stream and nothing else,
which makes it to VP9 and AV1 what a ``.h264`` file is to H.264: the form the
reference encoders write and a decoder's tests read.

:class:`IVFReader` is a :class:`~pyopengl_video.playback.VideoSource`, so a
:class:`~pyopengl_video.playback.Movie` plays one. :class:`IVFWriter` takes an
encoder's packets::

    with open_encoder(1280, 720, codec='av1') as encoder, \\
            IVFWriter('out.ivf', encoder) as stream:
        ...
        stream.write(encoder.encode(handle, timestamp))

Each AV1 frame is a temporal unit and each VP9 frame a frame or a superframe,
which is what the decoders here take as a sample.
"""
from __future__ import annotations

import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO

from pyopengl_video import codecs, obu
from pyopengl_video.decoder import Sample
from pyopengl_video.encoder import Encoder, Packet
from pyopengl_video.vaapi.bits import BitReader
from pyopengl_video.vaapi.h264 import StreamError

__all__ = ['FOURCCS', 'IVFError', 'IVFReader', 'IVFWriter', 'MAGIC',
           'av1_keyframe', 'vp9_keyframe']

MAGIC = b'DKIF'
HEADER = struct.Struct('<4sHH4sHHIII4x')
FRAME = struct.Struct('<IQ')

#: The four-character code an IVF header names each codec by.
FOURCCS = {codecs.AV1: b'AV01', codecs.VP9: b'VP90'}
_CODECS = {code: codec for codec, code in FOURCCS.items()}

AV1_KEY_FRAME = 0


class IVFError(ValueError):
    """The file is not IVF, or holds a codec or a frame this reader cannot use."""


def vp9_keyframe(frame: bytes) -> bool:
    """Whether a VP9 frame, or the first frame of a superframe, is a key frame."""
    try:
        reader = BitReader(frame[:2])
        if reader.f(2) != 2:
            return False
        profile = reader.f(1) | reader.f(1) << 1
        if profile == 3:
            reader.f(1)
        if reader.flag():                   # show_existing_frame
            return False
        return reader.f(1) == 0             # frame_type: 0 is a key frame
    except StreamError:
        return False


def av1_keyframe(unit_bytes: bytes) -> bool:
    """Whether an AV1 temporal unit opens with a shown key frame.

    The unit is a run of OBUs (:mod:`pyopengl_video.obu`). The frame header's
    opening bits say what the frame is; a stream of reduced still pictures has
    no such bits, and every one of its frames is a key frame.
    """
    reduced = False
    try:
        for unit in obu.split(unit_bytes):
            if unit.kind == obu.SEQUENCE_HEADER and unit.payload:
                reduced = bool(unit.payload[0] & 0x08)  # after profile and still_picture
            elif unit.kind in (obu.FRAME_HEADER, obu.FRAME):
                if reduced:
                    return True
                reader = BitReader(unit.payload[:1])
                if reader.flag():                       # show_existing_frame
                    return False
                return reader.f(2) == AV1_KEY_FRAME and bool(reader.flag())
    except StreamError:
        return False
    return False


_KEYFRAME = {codecs.AV1: av1_keyframe, codecs.VP9: vp9_keyframe}


class IVFReader:
    """The frames of an IVF file, timed and marked where decoding can start.

    source -- a path, or a binary file object, which stays the caller's to close

    codec -- ``'av1'`` or ``'vp9'``
    size -- the picture size the header states, as (width, height)
    timescale -- units of a second the presentation times are counted in
    sync_samples -- the indices of the key frames

    Opening indexes the file. A VP9 stream is seeked through, reading the
    start of each frame; an AV1 stream is read through once, since a temporal
    unit's frame header may follow other OBUs.
    """

    def __init__(self, source: str | Path | BinaryIO) -> None:
        self._owned = isinstance(source, (str, Path))
        self.path: Path | None = Path(source) if isinstance(source, (str, Path)) else None
        self._file: BinaryIO = (self.path.open('rb')  # noqa: OGC121 read as samples are asked for
                                if self.path is not None else source)  # type: ignore[assignment]  # a BinaryIO when not a path
        try:
            self._read_header()
            self._index()
        except Exception:
            self.close()
            raise

    def _read_header(self) -> None:
        self._start = self._file.tell()
        head = self._file.read(HEADER.size)
        if len(head) < HEADER.size or not head.startswith(MAGIC):
            raise IVFError(f'{self.path or "the stream"} is not an IVF file: '
                           f'it does not open with {MAGIC!r}')
        _, version, length, code, width, height, rate, scale, _ = HEADER.unpack(head)
        if version != 0:
            raise IVFError(f'IVF version {version}; this reader takes version 0')
        if code not in _CODECS:
            raise IVFError(
                f'the IVF file holds {code.decode("latin-1")!r}; this reader takes '
                f'{", ".join(value.decode() for value in _CODECS)}')
        self.codec = _CODECS[code]
        self.size = (width, height)
        # A time base of scale/rate seconds. One that is not a whole fraction
        # of a second keeps the rate as the timescale and multiplies each time.
        rate, scale = rate or 30, scale or 1
        self.timescale = rate
        self._tick = scale
        self._first = self._start + length

    def _index(self) -> None:
        offsets: list[int] = []
        sizes: list[int] = []
        times: list[int] = []
        keys: list[int] = []
        is_key = _KEYFRAME[self.codec]
        position = self._first
        end = self._file.seek(0, 2)
        while True:
            self._file.seek(position)
            head = self._file.read(FRAME.size)
            if len(head) < FRAME.size:
                break
            size, time = FRAME.unpack(head)
            if position + FRAME.size + size > end:
                break                                   # a frame cut short by the end of the file
            opening = self._file.read(min(size, 64))
            if len(opening) < min(size, 64):
                break                                   # a frame cut short by the end of the file
            if is_key(opening if self.codec == codecs.VP9
                      else opening + self._file.read(size - len(opening))):
                keys.append(len(offsets))
            offsets.append(position + FRAME.size)
            sizes.append(size)
            times.append(time * self._tick)
            position += FRAME.size + size
        if not offsets:
            raise IVFError(f'{self.path or "the stream"} holds no frames')
        self._offsets, self._sizes, self._times = offsets, sizes, times
        self.sync_samples = keys or [0]
        self._sync = frozenset(self.sync_samples)
        spans = [later - earlier for earlier, later in zip(times, times[1:], strict=False)]
        self._durations = [*spans, spans[-1] if spans else self._tick]

    def __len__(self) -> int:
        return len(self._offsets)

    def __repr__(self) -> str:
        width, height = self.size
        return (f'<{type(self).__name__} {self.path or "stream"} {self.codec} '
                f'{width}x{height}, {len(self)} frames>')

    @property
    def duration(self) -> float:
        """The stream's length in seconds."""
        return (self._times[-1] - self._times[0] + self._durations[-1]) / self.timescale

    def sample(self, index: int, parameter_sets: bool = False) -> Sample:
        """Frame `index` as the decoders take it.

        parameter_sets -- accepted for the interface; an AV1 key frame carries
            its sequence header and a VP9 frame needs none
        """
        del parameter_sets
        self._file.seek(self._offsets[index])
        return Sample(data=self._file.read(self._sizes[index]),
                      timestamp=self._times[index], duration=self._durations[index],
                      keyframe=index in self._sync, index=index)

    def presentation_time(self, index: int) -> int:
        """When frame `index` is shown, in :attr:`timescale` units."""
        return self._times[index]

    def keyframe_before(self, index: int) -> int:
        """The last key frame at or before `index`, or the first frame."""
        found = [key for key in self.sync_samples if key <= index]
        return found[-1] if found else 0

    def parameter_sets(self) -> bytes:
        """Nothing: an IVF file keeps no configuration apart from its frames."""
        return b''

    def close(self) -> None:
        """Close the file, where this reader opened it."""
        if self._owned and not self._file.closed:
            self._file.close()

    def __enter__(self) -> IVFReader:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()


class IVFWriter:
    """Writes an encoder's packets as an IVF file.

    path -- where to write
    encoder -- the encoder the packets come from, which gives the codec, the
        size and the timescale
    codec/width/height/timescale -- the same, for packets with no encoder

    The frame count in the header is written when the file is closed.
    """

    def __init__(self, path: str | Path, encoder: Encoder | None = None, *,
                 codec: str | None = None, width: int | None = None,
                 height: int | None = None, timescale: int | None = None) -> None:
        if encoder is not None:
            codec = encoder.codec
            width, height = encoder.size
            timescale = encoder.timescale
        if codec is None or width is None or height is None or timescale is None:
            raise ValueError('give an encoder, or codec, width, height and timescale')
        codec = codecs.canonical(codec)
        if codec not in FOURCCS:
            raise IVFError(f'IVF holds {", ".join(sorted(FOURCCS))}, not {codec}; '
                           'an MP4Writer takes the other codecs')
        self.path = Path(path)
        self.codec = codec
        self.size = (int(width), int(height))
        self.timescale = int(timescale)
        self.frames = 0
        self._closed = False
        self._file = self.path.open('wb')  # noqa: OGC121 a recording written as it is captured
        self._file.write(self._header())

    def _header(self) -> bytes:
        return HEADER.pack(MAGIC, 0, HEADER.size, FOURCCS[self.codec], *self.size,
                           self.timescale, 1, self.frames)

    def write(self, packets: Packet | Iterable[Packet]) -> None:
        """Add a packet, or everything in an iterable of them."""
        if self._closed:
            raise ValueError('the stream is closed')
        if isinstance(packets, Packet):
            packets = (packets,)
        for packet in packets:
            self._file.write(FRAME.pack(len(packet.data), int(packet.timestamp)))
            self._file.write(packet.data)
            self.frames += 1

    def close(self) -> None:
        """Write the frame count and close the file. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._file.seek(0)
        self._file.write(self._header())
        self._file.close()

    def __enter__(self) -> IVFWriter:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()
