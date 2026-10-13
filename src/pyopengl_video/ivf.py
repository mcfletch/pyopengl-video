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

import bisect
import logging
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO

from pyopengl_video import bounds, codecs, container, obu
from pyopengl_video.decoder import Sample
from pyopengl_video.encoder import Encoder, Packet
from pyopengl_video.vaapi.bits import BitReader
from pyopengl_video.vaapi.h264 import StreamError

__all__ = ['FOURCCS', 'IVFError', 'IVFReader', 'IVFWriter', 'MAGIC',
           'av1_keyframe', 'vp9_keyframe']

log = logging.getLogger(__name__)

MAGIC = b'DKIF'
HEADER = struct.Struct('<4sHH4sHHIII4x')
FRAME = struct.Struct('<IQ')

#: Bytes of a VP9 frame read to index it: its first says what the frame is.
VP9_OPENING = 2

#: The four-character code an IVF header names each codec by.
FOURCCS = {codecs.AV1: b'AV01', codecs.VP9: b'VP90'}
_CODECS = {code: codec for codec, code in FOURCCS.items()}

AV1_KEY_FRAME = 0


class IVFError(container.ContainerError):
    """The file is not IVF, or holds a codec or a frame this reader cannot use.

    An :class:`IVFWriter` raises it for a size, a timescale or a timestamp the
    file has no field wide enough for.
    """


class Cursor(container.Cursor):
    """The fields of an IVF header, which are little-endian; raises :class:`IVFError`."""

    error = IVFError
    byteorder = 'little'


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
    size -- the picture size the header states, as (width, height). A file
        may state anything from 0 to 65535 each way, whatever its stream
        holds; a decoder reports the stream's own.
    timescale -- units of a second the presentation times are counted in
    sync_samples -- the indices of the key frames

    Opening indexes the file. A VP9 stream is seeked through, reading the
    start of each frame; an AV1 stream is read through once, since a temporal
    unit's frame header may follow other OBUs. A frame that states more bytes
    than the file holds ends the stream before it, as the end of the file
    does, so every frame indexed is whole. Samples can be read from several
    threads at once.

    Raises :class:`IVFError` for a file that is not IVF or holds no whole
    frame.
    """

    def __init__(self, source: str | Path | BinaryIO) -> None:
        self._source = container.FileBytes(source, IVFError)
        self.path = self._source.path
        try:
            self._read_header()
            self._index()
        except BaseException:
            self.close()
            raise

    def _read_header(self) -> None:
        start, end = self._source.start, self._source.size
        named = self.path or 'the stream'
        if end - start < HEADER.size:
            raise IVFError(f'{named} is not an IVF file: it is shorter than '
                           'the header of one')
        head = Cursor(self._source.read(start, HEADER.size, 'the IVF header'),
                      'the IVF header')
        if head.take(4) != MAGIC:
            raise IVFError(f'{named} is not an IVF file: '
                           f'it does not open with {MAGIC!r}')
        version, length, code = head.u16(), head.u16(), head.take(4)
        width, height, rate, scale = head.u16(), head.u16(), head.u32(), head.u32()
        if version != 0:
            raise IVFError(f'IVF version {version}; this reader takes version 0')
        if code not in _CODECS:
            raise IVFError(
                f'the IVF file holds {code.decode("latin-1")!r}; this reader takes '
                f'{", ".join(value.decode() for value in _CODECS)}')
        if not HEADER.size <= length <= end - start:
            raise IVFError(f'the IVF header states its own length as {length} '
                           f'bytes: it is {HEADER.size} or more, and the file '
                           f'holds {end - start}')
        self.codec = _CODECS[code]
        self.size = (width, height)
        # A time base of scale/rate seconds. One that is not a whole fraction
        # of a second keeps the rate as the timescale and multiplies each time.
        self.timescale = rate or 30
        self._tick = scale or 1
        self._first = start + length

    def _index(self) -> None:
        offsets: list[int] = []
        sizes: list[int] = []
        times: list[int] = []
        keys: list[int] = []
        is_key = _KEYFRAME[self.codec]
        opening = VP9_OPENING if self.codec == codecs.VP9 else None
        position, end = self._first, self._source.size
        # A frame header or a frame the file ends inside ends the stream.
        while position + FRAME.size <= end:
            head = Cursor(self._source.read(position, FRAME.size), 'a frame header')
            size, time = head.u32(), head.u64()
            position += FRAME.size
            if position + size > end:
                break
            read = size if opening is None else min(size, opening)
            if is_key(self._source.read(position, read, f'frame {len(offsets)}')):
                keys.append(len(offsets))
            offsets.append(position)
            sizes.append(size)
            times.append(time * self._tick)
            position += size
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

        Raises :class:`IVFError` where the file no longer holds the frame.
        """
        del parameter_sets
        if not 0 <= index < len(self):
            raise IndexError(f'frame {index} of a stream of {len(self)}')
        data = self._source.read(self._offsets[index], self._sizes[index], f'frame {index}')
        return Sample(data=data, timestamp=self._times[index],
                      duration=self._durations[index],
                      keyframe=index in self._sync, index=index)

    def presentation_time(self, index: int) -> int:
        """When frame `index` is shown, in :attr:`timescale` units."""
        return self._times[index]

    def keyframe_before(self, index: int) -> int:
        """The last key frame at or before `index`, or the first frame."""
        found = bisect.bisect_right(self.sync_samples, index)
        return self.sync_samples[found - 1] if found else 0

    def parameter_sets(self) -> bytes:
        """Nothing: an IVF file keeps no configuration apart from its frames."""
        return b''

    def close(self) -> None:
        """Close the file, where this reader opened it."""
        self._source.close()

    def __enter__(self) -> IVFReader:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()


class IVFWriter:
    """Writes an encoder's packets as an IVF file.

    path -- where to write
    encoder -- the encoder the packets come from, which gives the codec, the
        size and the timescale
    codec/width/height/timescale -- the same, for packets with no encoder:
        a width and a height from 1 to 65535, and a timescale from 1 to
        2**32 - 1

    Raises :class:`IVFError` for a codec IVF does not hold and for a size or
    a timescale outside those ranges, before the file is created.

    The frame count in the header is written when the file is closed. As a
    context manager the stream is closed when the block ends; where the block
    ends in an exception, that exception goes on and a failure to close is
    logged.
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
        self.size = bounds.frame_size(width, height, error=IVFError,
                                      most=(bounds.FIELD_16, bounds.FIELD_16))
        self.timescale = bounds.whole(timescale, 'the timescale', 1, bounds.FIELD_32,
                                      IVFError)
        self.frames = 0
        self._closed = False
        self._file = self.path.open('wb')  # noqa: OGC121 a recording written as it is captured
        try:
            self._file.write(self._header())
        except BaseException:
            self._file.close()
            raise

    def _header(self) -> bytes:
        return HEADER.pack(MAGIC, 0, HEADER.size, FOURCCS[self.codec], *self.size,
                           self.timescale, 1, self.frames)

    def write(self, packets: Packet | Iterable[Packet]) -> None:
        """Add a packet, or everything in an iterable of them.

        Raises :class:`IVFError` for a packet the file cannot describe, and
        writes nothing of it: a timestamp that is not a whole number from 0
        to 2**64 - 1, or data of 4 GiB or more.
        """
        if self._closed:
            raise ValueError('the stream is closed')
        if isinstance(packets, Packet):
            packets = (packets,)
        for packet in packets:
            timestamp = bounds.whole(packet.timestamp, 'a timestamp', 0, (1 << 64) - 1,
                                     IVFError)
            if len(packet.data) > bounds.FIELD_32 or self.frames >= bounds.FIELD_32:
                raise IVFError(
                    f'a frame of {len(packet.data)} bytes after {self.frames} '
                    'others: the file counts each in 32 bits')
            self._file.write(FRAME.pack(len(packet.data), timestamp))
            self._file.write(packet.data)
            self.frames += 1

    def close(self) -> None:
        """Write the frame count and close the file. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        try:
            self._file.seek(0)
            self._file.write(self._header())
        finally:
            self._file.close()

    def __enter__(self) -> IVFWriter:
        return self

    def __exit__(self, kind: type[BaseException] | None, *exception: Any) -> None:
        if kind is None:
            self.close()
            return
        try:
            self.close()
        except Exception as error:  # noqa: BLE001 - the block's own exception is the one to raise
            log.warning('%s was not closed cleanly: %s', self.path, error)
