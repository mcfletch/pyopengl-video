"""Writing an H.264 elementary stream into an MP4 file.

An encoder hands out Annex-B fragments -- NAL units separated by start codes --
and that is a fine thing to write to a ``.h264`` file and pipe somewhere, but it
is not what anything plays. MP4 stores the same units length-prefixed, keeps the
parameter sets in the sample description rather than in the stream, and carries
the timing in tables beside the data.

:class:`MP4Writer` does that conversion as the packets arrive. The compressed
data goes straight into ``mdat`` as it comes, so a long recording costs no more
memory than a short one; only the sample table is held, and it is written as
``moov`` when the movie closes. That order means the file is not streamable over
a network without a further pass, which is the trade for not buffering the video.

    with MP4Writer('out.mp4', encoder) as movie:
        movie.write(encoder.encode(handle, timestamp))
        ...
        movie.write(encoder.flush())
"""
from __future__ import annotations

import struct
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pyopengl_video.encoder import Encoder, Packet

#: NAL unit types that belong in the sample description rather than in a sample.
NAL_SPS = 7
NAL_PPS = 8
#: An access unit delimiter says where a picture starts, which the sample table
#: already says.
NAL_AUD = 9

#: How many bytes prefix each NAL unit in a sample. Four is what ``avcC``
#: advertises below, and what every player expects.
LENGTH_PREFIX = 4

#: The unity matrix every video track carries, in 16.16 and 2.30 fixed point.
UNITY_MATRIX = (0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)


def split_annexb(stream: bytes) -> Iterator[bytes]:
    """Yield the NAL units of an Annex-B `stream`, without their start codes.

    Start codes are three or four bytes; the pattern cannot occur inside a unit,
    because an encoder inserts an emulation prevention byte to stop it.
    """
    position = 0
    length = len(stream)
    starts = []
    while position < length - 2:
        if stream[position] == 0 and stream[position + 1] == 0:
            if stream[position + 2] == 1:
                starts.append((position, 3))
                position += 3
                continue
            if (position < length - 3 and stream[position + 2] == 0
                    and stream[position + 3] == 1):
                starts.append((position, 4))
                position += 4
                continue
        position += 1
    for index, (start, size) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else length
        unit = stream[start + size:end]
        if unit:
            yield unit


def box(kind: str, *payloads: bytes) -> bytes:
    """One MP4 box: a length, a four-character type, and the payload."""
    payload = b''.join(payloads)
    return struct.pack('>I4s', len(payload) + 8, kind.encode('ascii')) + payload


def full_box(kind: str, version: int, flags: int, *payloads: bytes) -> bytes:
    """A box whose payload opens with a version and three flag bytes."""
    return box(kind, struct.pack('>I', (version << 24) | flags), *payloads)


class MP4Writer:
    """Collects packets from an encoder into a playable MP4 file.

    path -- where to write
    encoder -- take the frame size, timescale and parameter sets from this
        encoder; give the keyword arguments instead when the packets come from
        somewhere else
    width/height -- frame size in pixels
    timescale -- units of a second that packet timestamps are counted in
    parameter_sets -- ``(sps, pps)`` as raw NAL units. When omitted, they are
        taken from the first packets that carry them.
    """

    #: Units of a second the movie header counts in. The track keeps the
    #: encoder's own timescale; this one only has to describe the whole movie.
    movie_timescale = 1000

    def __init__(self, path: str | Path, encoder: Encoder | None = None, *,
                 width: int | None = None, height: int | None = None,
                 timescale: int | None = None,
                 parameter_sets: tuple[bytes, bytes] | None = None):
        if encoder is not None:
            width, height = encoder.size
            timescale = encoder.timescale
            if parameter_sets is None:
                parameter_sets = self._parameter_sets(encoder.headers())
        if width is None or height is None or timescale is None:
            raise ValueError('give an encoder, or width, height and timescale')
        self.path = Path(path)
        self.size = (int(width), int(height))
        self.timescale = int(timescale)
        self.sps, self.pps = parameter_sets or (b'', b'')

        self._sizes: list[int] = []
        self._durations: list[int] = []
        self._timestamps: list[int] = []
        self._sync: list[int] = []
        self._closed = False

        self._file = self.path.open('wb')
        header = box('ftyp', b'isom', struct.pack('>I', 512),
                     b'isom', b'iso2', b'avc1', b'mp41')
        self._file.write(header)
        # A 64-bit `mdat`, so a long recording is not cut short at four
        # gigabytes. The length is patched in once it is known.
        self._mdat_position = self._file.tell()
        self._file.write(struct.pack('>I4sQ', 1, b'mdat', 0))
        self._data_position = self._file.tell()

    # ------------------------------------------------------------- writing

    def write(self, packets: Packet | Iterable[Packet]) -> None:
        """Add a packet, or everything in an iterable of them."""
        if isinstance(packets, Packet):
            packets = (packets,)
        for packet in packets:
            self._write_packet(packet)

    def _write_packet(self, packet: Packet) -> None:
        if self._closed:
            raise ValueError('the movie is closed')
        sample = bytearray()
        for unit in split_annexb(packet.data):
            kind = unit[0] & 0x1f
            if kind in (NAL_SPS, NAL_PPS):
                self._remember_parameter_set(kind, unit)
                continue
            if kind == NAL_AUD:
                continue
            sample += struct.pack('>I', len(unit)) + unit
        if not sample:
            return
        self._file.write(sample)
        if packet.keyframe:
            self._sync.append(len(self._sizes) + 1)      # sample numbers are one-based
        self._sizes.append(len(sample))
        self._durations.append(int(packet.duration))
        self._timestamps.append(int(packet.timestamp))

    def _remember_parameter_set(self, kind: int, unit: bytes) -> None:
        """Keep the first SPS and PPS seen, for the sample description."""
        if kind == NAL_SPS and not self.sps:
            self.sps = unit
        elif kind == NAL_PPS and not self.pps:
            self.pps = unit

    @staticmethod
    def _parameter_sets(headers: bytes) -> tuple[bytes, bytes]:
        """Pick the SPS and PPS out of an encoder's Annex-B headers."""
        found = {unit[0] & 0x1f: unit for unit in split_annexb(headers)}
        return found.get(NAL_SPS, b''), found.get(NAL_PPS, b'')

    # ------------------------------------------------------------- closing

    def close(self) -> None:
        """Finish the file: patch the media length, then write the tables."""
        if self._closed:
            return
        self._closed = True
        try:
            if not self._sizes:
                raise ValueError(f'{self.path} has no samples in it')
            if not self.sps or not self.pps:
                raise ValueError(
                    f'{self.path} has no parameter sets: give parameter_sets, or '
                    'let the encoder repeat them in the stream')
            end = self._file.tell()
            self._file.seek(self._mdat_position + 8)
            self._file.write(struct.pack('>Q', end - self._mdat_position))
            self._file.seek(end)
            self._file.write(self._moov())
        finally:
            self._file.close()

    def __enter__(self) -> MP4Writer:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()

    # -------------------------------------------------------------- tables

    @property
    def duration(self) -> int:
        """The movie's length, in the track's timescale."""
        return sum(self._durations)

    def _composition_offsets(self) -> list[int]:
        """How far each sample's display time is from its decode time.

        Samples are stored in decode order and a packet's timestamp is its
        display time, so the two part company as soon as an encoder reorders
        pictures. Decode time is the running total of the durations.
        """
        offsets = []
        decode_time = 0
        for timestamp, duration in zip(self._timestamps, self._durations, strict=True):
            offsets.append(timestamp - decode_time)
            decode_time += duration
        return offsets

    @staticmethod
    def _runs(values: Iterable[int]) -> list[tuple[int, int]]:
        """Run-length encode `values` as ``(count, value)`` pairs."""
        runs: list[tuple[int, int]] = []
        for value in values:
            if runs and runs[-1][1] == value:
                runs[-1] = (runs[-1][0] + 1, value)
            else:
                runs.append((1, value))
        return runs

    def _moov(self) -> bytes:
        """The movie box: everything about the track except its data."""
        movie_duration = round(self.duration * self.movie_timescale / self.timescale)
        mvhd = full_box(
            'mvhd', 0, 0,
            struct.pack('>IIII', 0, 0, self.movie_timescale, movie_duration),
            struct.pack('>IHH', 0x00010000, 0x0100, 0),        # rate, volume, reserved
            struct.pack('>II', 0, 0),                          # reserved
            struct.pack('>9I', *UNITY_MATRIX),
            bytes(24),                                         # pre-defined
            struct.pack('>I', 2),                              # next track id
        )
        return box('moov', mvhd, self._trak(movie_duration))

    def _trak(self, movie_duration: int) -> bytes:
        width, height = self.size
        tkhd = full_box(
            'tkhd', 0, 0x000007,                               # enabled, in movie/preview
            struct.pack('>IIIII', 0, 0, 1, 0, movie_duration),
            struct.pack('>IIhhhh', 0, 0, 0, 0, 0, 0),          # reserved, layer, volume
            struct.pack('>9I', *UNITY_MATRIX),
            struct.pack('>II', width << 16, height << 16),
        )
        return box('trak', tkhd, self._mdia())

    def _mdia(self) -> bytes:
        mdhd = full_box(
            'mdhd', 0, 0,
            struct.pack('>IIII', 0, 0, self.timescale, self.duration),
            struct.pack('>HH', 0x55c4, 0),                     # language 'und'
        )
        hdlr = full_box(
            'hdlr', 0, 0,
            struct.pack('>I4s', 0, b'vide'), bytes(12), b'VideoHandler\x00',
        )
        return box('mdia', mdhd, hdlr, self._minf())

    def _minf(self) -> bytes:
        vmhd = full_box('vmhd', 0, 1, struct.pack('>HHHH', 0, 0, 0, 0))
        dinf = box('dinf', full_box('dref', 0, 0, struct.pack('>I', 1),
                                    full_box('url ', 0, 1)))
        return box('minf', vmhd, dinf, self._stbl())

    def _stbl(self) -> bytes:
        tables = [self._stsd(), self._stts()]
        offsets = self._composition_offsets()
        if any(offsets):
            tables.append(self._ctts(offsets))
        if self._sync and len(self._sync) < len(self._sizes):
            tables.append(self._stss())
        tables += [self._stsc(), self._stsz(), self._co64()]
        return box('stbl', *tables)

    def _stsd(self) -> bytes:
        width, height = self.size
        avcc = box(
            'avcC',
            bytes([1, self.sps[1], self.sps[2], self.sps[3],
                   0xfc | (LENGTH_PREFIX - 1), 0xe0 | 1]),
            struct.pack('>H', len(self.sps)), self.sps,
            bytes([1]), struct.pack('>H', len(self.pps)), self.pps,
        )
        avc1 = box(
            'avc1',
            bytes(6), struct.pack('>H', 1),                    # reserved, data reference
            bytes(16),                                         # pre-defined, reserved
            struct.pack('>HH', width, height),
            struct.pack('>II', 0x00480000, 0x00480000),        # 72 dpi each way
            struct.pack('>I', 0),                              # reserved
            struct.pack('>H', 1),                              # frames per sample
            bytes(32),                                         # compressor name
            struct.pack('>Hh', 0x0018, -1),                    # depth, pre-defined
            avcc,
        )
        return full_box('stsd', 0, 0, struct.pack('>I', 1), avc1)

    def _stts(self) -> bytes:
        runs = self._runs(self._durations)
        return full_box('stts', 0, 0, struct.pack('>I', len(runs)),
                        *(struct.pack('>II', count, delta) for count, delta in runs))

    def _ctts(self, offsets: list[int]) -> bytes:
        runs = self._runs(offsets)
        # Version 1 offsets are signed, which is what a display order that runs
        # ahead of the decode order needs.
        return full_box('ctts', 1, 0, struct.pack('>I', len(runs)),
                        *(struct.pack('>Ii', count, offset) for count, offset in runs))

    def _stss(self) -> bytes:
        return full_box('stss', 0, 0, struct.pack('>I', len(self._sync)),
                        *(struct.pack('>I', sample) for sample in self._sync))

    def _stsc(self) -> bytes:
        # Every sample sits in one chunk, so the map has a single entry.
        return full_box('stsc', 0, 0, struct.pack('>I', 1),
                        struct.pack('>III', 1, len(self._sizes), 1))

    def _stsz(self) -> bytes:
        return full_box('stsz', 0, 0, struct.pack('>II', 0, len(self._sizes)),
                        *(struct.pack('>I', size) for size in self._sizes))

    def _co64(self) -> bytes:
        return full_box('co64', 0, 0, struct.pack('>I', 1),
                        struct.pack('>Q', self._data_position))
