"""Writing an H.264 elementary stream into an MP4 file, and reading one back.

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

:class:`MP4Reader` is the reverse: it reads ``moov`` and the sample tables
without reading ``mdat``, and hands each video sample out as Annex-B, the form
every decoder here takes::

    with MP4Reader('movie.mp4') as movie:
        first = movie.video.sample(0, parameter_sets=True)

The box structure is ISO/IEC 14496-12 and the ``avcC`` record ISO/IEC 14496-15.
"""
from __future__ import annotations

import dataclasses
import functools
import struct
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

from pyopengl_video.decoder import Sample
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
    because an encoder inserts an emulation prevention byte to stop it. A zero
    byte in front of a three-byte code is the four-byte code's first.
    """
    starts = []
    position = stream.find(b'\x00\x00\x01')
    while position >= 0:
        starts.append(position)
        position = stream.find(b'\x00\x00\x01', position + 3)
    for index, position in enumerate(starts):
        if index + 1 < len(starts):
            end = starts[index + 1]
            if stream[end - 1] == 0 and end - 1 >= position + 3:
                end -= 1
        else:
            end = len(stream)
        unit = stream[position + 3:end]
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
        # What is here now was advertised rather than observed, and stands only
        # until the stream says otherwise.
        self._sps_from_stream = False
        self._pps_from_stream = False

        self._sizes: list[int] = []
        self._durations: list[int] = []
        self._timestamps: list[int] = []
        self._sync: list[int] = []
        self._closed = False

        self._file = self.path.open('wb')  # noqa: OGC121 a recording written as it is captured, the way a camera writes one
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
        """Keep the parameter sets the stream carries, for the sample description.

        The stream is the authority. An encoder states its parameter sets before
        it has coded anything and a driver may amend them on the way out -- a
        capability the hardware does not have is one the picture parameter set
        must not claim -- so what the samples were coded against is what the
        sample description has to hold. A decoder set up from the other ones
        looks for syntax that is not in the data, and reports a corrupt picture
        rather than a mismatch.

        The first of each is kept per recording, since the sample description
        describes every sample and a stream whose parameter sets changed
        part-way would need a second one.
        """
        if kind == NAL_SPS and not self._sps_from_stream:
            self.sps = unit
            self._sps_from_stream = True
        elif kind == NAL_PPS and not self._pps_from_stream:
            self.pps = unit
            self._pps_from_stream = True

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
        return sum(self._sample_durations())

    def _sample_durations(self) -> list[int]:
        """Each sample's duration, in decode order.

        A sample lasts until the next display time: the n-th decode time is
        the n-th timestamp in display order, so decode times stay on the
        caller's timestamps rather than on a sum of rounded frame durations.
        The last sample keeps the duration its packet gave. Timestamps that do
        not increase say nothing about how long a sample lasts, and then the
        packets' own durations are used.
        """
        shown = sorted(self._timestamps)
        gaps = [later - earlier for earlier, later in zip(shown[:-1], shown[1:], strict=True)]
        if any(gap <= 0 for gap in gaps):
            return list(self._durations)
        return gaps + self._durations[-1:]

    def _composition_offsets(self, durations: list[int]) -> list[int]:
        """How far each sample's display time is from its decode time.

        Samples are stored in decode order and a packet's timestamp is its
        display time, so the two part company as soon as an encoder reorders
        pictures. Decode time is the running total of ``durations``.
        """
        offsets = []
        decode_time = 0
        for timestamp, duration in zip(self._timestamps, durations, strict=True):
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
        durations = self._sample_durations()
        tables = [self._stsd(), self._stts(durations)]
        offsets = self._composition_offsets(durations)
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

    def _stts(self, durations: list[int]) -> bytes:
        runs = self._runs(durations)
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


# =================================================================== reading


class MP4Error(ValueError):
    """A file is not an MP4 this reader can read, or breaks the box syntax."""


def iter_boxes(data: bytes, start: int = 0,
               end: int | None = None) -> Iterator[tuple[str, bytes]]:
    """Yield each box in `data[start:end]` as ``(type, payload)``."""
    end = len(data) if end is None else end
    position = start
    while position + 8 <= end:
        size, kind = struct.unpack_from('>I4s', data, position)
        header = 8
        if size == 1:
            if position + 16 > end:
                raise MP4Error(f'a box at {position} is cut short')
            size = struct.unpack_from('>Q', data, position + 8)[0]
            header = 16
        elif size == 0:
            size = end - position
        if size < header or position + size > end:
            raise MP4Error(
                f'a {kind!r} box at {position} claims {size} bytes, which the '
                'space it is in cannot hold')
        yield kind.decode('latin-1'), data[position + header:position + size]
        position += size


def _children(payload: bytes) -> dict[str, bytes]:
    """The boxes inside a container box, by type; the first of each type."""
    found: dict[str, bytes] = {}
    for kind, child in iter_boxes(payload):
        found.setdefault(kind, child)
    return found


def _full_box(payload: bytes) -> tuple[int, int, bytes]:
    """A full box's version, flags and the rest of its payload."""
    word = struct.unpack_from('>I', payload)[0]
    return word >> 24, word & 0xFFFFFF, payload[4:]


def _descriptor(data: bytes, position: int) -> tuple[int, int, int]:
    """An MPEG-4 descriptor's tag, and where its body starts and ends."""
    tag = data[position]
    position += 1
    length = 0
    for _ in range(4):
        byte = data[position]
        position += 1
        length = (length << 7) | (byte & 0x7F)
        if not byte & 0x80:
            break
    return tag, position, position + length


class Track:
    """One track of an MP4 file: its sample table, and how to read a sample.

    track_id -- the ``tkhd`` track identifier
    handler -- ``'vide'`` or ``'soun'``, or another handler type
    codec -- the sample entry's four-character code, such as ``'avc1'``
    timescale -- units of a second the track's times are counted in
    duration -- the track's length in those units
    sync_samples -- the indices, in decoding order, of the samples decoding
        can start at
    """

    def __init__(self, reader: MP4Reader, track_id: int, handler: str,
                 codec: str, timescale: int, duration: int,
                 table: dict[str, bytes], edit_offset: int) -> None:
        self._reader = reader
        self.track_id = track_id
        self.handler = handler
        self.codec = codec
        self.timescale = timescale
        self.duration = duration
        self._edit_offset = edit_offset
        self._sizes = self._read_sizes(table)
        count = len(self._sizes)
        self._offsets = self._read_offsets(table, count)
        self._decode_times, self._durations = self._read_times(table, count)
        self._composition = self._read_composition(table, count)
        self.sync_samples = self._read_sync(table, count)

    def __len__(self) -> int:
        return len(self._sizes)

    def __repr__(self) -> str:
        return (f'<{type(self).__name__} {self.track_id} {self.codec}, '
                f'{len(self)} samples>')

    # -------------------------------------------------------- the tables

    @staticmethod
    def _read_sizes(table: dict[str, bytes]) -> list[int]:
        if 'stsz' in table:
            _, _, body = _full_box(table['stsz'])
            size, count = struct.unpack_from('>II', body)
            if size:
                return [size] * count
            return list(struct.unpack_from(f'>{count}I', body, 8))
        if 'stz2' in table:
            _, _, body = _full_box(table['stz2'])
            field, count = body[3], struct.unpack_from('>I', body, 4)[0]
            if field == 16:
                return list(struct.unpack_from(f'>{count}H', body, 8))
            if field == 8:
                return list(body[8:8 + count])
            nibbles = body[8:8 + (count + 1) // 2]
            return [(nibbles[index // 2] >> (4 if index % 2 == 0 else 0)) & 0xF
                    for index in range(count)]
        raise MP4Error('a track has no sample size table (stsz)')

    def _read_offsets(self, table: dict[str, bytes], count: int) -> list[int]:
        """Each sample's file offset, from its chunk's offset and the sizes before it."""
        if 'stco' in table:
            _, _, body = _full_box(table['stco'])
            chunks = list(struct.unpack_from(f'>{struct.unpack_from(">I", body)[0]}I', body, 4))
        elif 'co64' in table:
            _, _, body = _full_box(table['co64'])
            chunks = list(struct.unpack_from(f'>{struct.unpack_from(">I", body)[0]}Q', body, 4))
        else:
            raise MP4Error('a track has no chunk offset table (stco or co64)')
        if 'stsc' not in table:
            raise MP4Error('a track has no sample-to-chunk table (stsc)')
        _, _, body = _full_box(table['stsc'])
        entries = struct.unpack_from('>I', body)[0]
        runs = [struct.unpack_from('>III', body, 4 + 12 * index)
                for index in range(entries)]
        offsets: list[int] = []
        for run, (first, per_chunk, _description) in enumerate(runs):
            last = runs[run + 1][0] if run + 1 < len(runs) else len(chunks) + 1
            for chunk in range(first, last):
                position = chunks[chunk - 1]
                for _ in range(per_chunk):
                    if len(offsets) == count:
                        return offsets
                    offsets.append(position)
                    position += self._sizes[len(offsets) - 1]
        if len(offsets) != count:
            raise MP4Error(
                f'the chunk tables place {len(offsets)} samples of {count}')
        return offsets

    @staticmethod
    def _read_times(table: dict[str, bytes], count: int) -> tuple[list[int], list[int]]:
        """Each sample's decode time and duration (stts)."""
        if 'stts' not in table:
            raise MP4Error('a track has no decoding time table (stts)')
        _, _, body = _full_box(table['stts'])
        entries = struct.unpack_from('>I', body)[0]
        times, durations = [], []
        time = 0
        for index in range(entries):
            runs, delta = struct.unpack_from('>II', body, 4 + 8 * index)
            for _ in range(runs):
                times.append(time)
                durations.append(delta)
                time += delta
        if len(times) < count:
            raise MP4Error(f'the decoding time table covers {len(times)} samples of {count}')
        return times[:count], durations[:count]

    @staticmethod
    def _read_composition(table: dict[str, bytes], count: int) -> list[int]:
        """Each sample's composition offset (ctts), signed, or zero without one."""
        if 'ctts' not in table:
            return [0] * count
        _, _, body = _full_box(table['ctts'])
        entries = struct.unpack_from('>I', body)[0]
        offsets: list[int] = []
        for index in range(entries):
            # Signed in either version: version 0 files from some muxers carry
            # negative offsets, and no positive one reaches 2**31.
            runs, offset = struct.unpack_from('>Ii', body, 4 + 8 * index)
            offsets.extend([offset] * runs)
        return (offsets + [0] * count)[:count]

    @staticmethod
    def _read_sync(table: dict[str, bytes], count: int) -> list[int]:
        """The indices of the sync samples; every sample is one without ``stss``."""
        if 'stss' not in table:
            return list(range(count))
        _, _, body = _full_box(table['stss'])
        entries = struct.unpack_from('>I', body)[0]
        return [number - 1 for number in struct.unpack_from(f'>{entries}I', body, 4)]

    # ------------------------------------------------------------ samples

    def presentation_time(self, index: int) -> int:
        """When sample `index` is shown, with the edit list applied."""
        return self._decode_times[index] + self._composition[index] - self._edit_offset

    def display_order(self) -> list[int]:
        """Sample indices in the order their presentation times put them."""
        return sorted(range(len(self)), key=self.presentation_time)

    def keyframe_before(self, index: int) -> int:
        """The last sync sample at or before `index`, in decoding order."""
        found = 0
        for sync in self.sync_samples:
            if sync > index:
                break
            found = sync
        return found

    def read(self, index: int) -> bytes:
        """Sample `index`'s bytes, as the file stores them."""
        if not 0 <= index < len(self):
            raise IndexError(f'sample {index} of a track of {len(self)}')
        return self._reader.read_at(self._offsets[index], self._sizes[index])

    def sample(self, index: int) -> Sample:
        """Sample `index`, with its presentation time and key flag."""
        return Sample(data=self.read(index), timestamp=self.presentation_time(index),
                      duration=self._durations[index],
                      keyframe=self._is_sync(index), index=index)

    def _is_sync(self, index: int) -> bool:
        return index in self._sync_set

    @functools.cached_property
    def _sync_set(self) -> frozenset[int]:
        return frozenset(self.sync_samples)


class VideoTrack(Track):
    """An H.264 video track (``avc1`` or ``avc3``).

    width/height -- the picture size the sample entry states
    sps/pps -- the parameter sets ``avcC`` carries, as NAL units
    length_size -- bytes in the length prefix of each NAL unit in a sample
    """

    width = 0
    height = 0
    length_size = LENGTH_PREFIX
    sps: list[bytes]
    pps: list[bytes]

    def describe(self, entry: bytes) -> None:
        """Take the picture size and the decoder configuration from the sample entry."""
        self.width, self.height = struct.unpack_from('>HH', entry, 24)
        self.sps, self.pps = [], []
        config = _children(entry[78:]).get('avcC')
        if config is None:
            return
        self.length_size = (config[4] & 3) + 1
        position = 6
        for _ in range(config[5] & 0x1F):
            length = struct.unpack_from('>H', config, position)[0]
            self.sps.append(config[position + 2:position + 2 + length])
            position += 2 + length
        for _ in range(config[position]):
            length = struct.unpack_from('>H', config, position + 1)[0]
            self.pps.append(config[position + 3:position + 3 + length])
            position += 2 + length

    def parameter_sets(self) -> bytes:
        """The sample entry's SPS and PPS, as an Annex-B fragment."""
        return b''.join(b'\x00\x00\x00\x01' + unit for unit in self.sps + self.pps)

    def sample(self, index: int, parameter_sets: bool = False) -> Sample:
        """Sample `index` as Annex-B, the form every decoder here takes.

        parameter_sets -- put the sample entry's SPS and PPS in front, which a
            decoder needs before the first sample it sees and after a seek
        """
        found = super().sample(index)
        raw = found.data
        units = []
        position = 0
        size = self.length_size
        while position + size <= len(raw):
            length = int.from_bytes(raw[position:position + size], 'big')
            position += size
            units.append(b'\x00\x00\x00\x01' + raw[position:position + length])
            position += length
        prefix = self.parameter_sets() if parameter_sets else b''
        return dataclasses.replace(found, data=prefix + b''.join(units))


class AudioTrack(Track):
    """An audio track, reported for a decoder this package does not have.

    sample_rate/channels -- as the sample entry states them
    object_type -- the MPEG-4 objectTypeIndication from ``esds``, such as 0x40
        for MPEG-4 audio (AAC) or 0x6B for MP3; None without ``esds``
    config -- the codec's configuration: the DecoderSpecificInfo from ``esds``
        (an AudioSpecificConfig for AAC), or the payload of ``dOps`` (Opus) or
        ``dfLa`` (FLAC)
    """

    sample_rate = 0
    channels = 0
    object_type: int | None = None
    config = b''

    def describe(self, entry: bytes) -> None:
        """Read an AudioSampleEntry, QuickTime's versions 1 and 2 included."""
        version = struct.unpack_from('>H', entry, 8)[0]
        self.channels = struct.unpack_from('>H', entry, 16)[0]
        self.sample_rate = struct.unpack_from('>I', entry, 24)[0] >> 16
        children_at = 28
        if version == 1:
            children_at += 16
        elif version == 2:
            self.sample_rate = round(struct.unpack_from('>d', entry, 32)[0])
            self.channels = struct.unpack_from('>I', entry, 40)[0]
            children_at += 36
        children = _children(entry[children_at:])
        if 'esds' in children:
            self._read_esds(children['esds'])
        for kind in ('dOps', 'dfLa'):
            if kind in children:
                self.config = children[kind]

    def _read_esds(self, payload: bytes) -> None:
        """The ES descriptor's decoder configuration (ISO/IEC 14496-1, 7.2.6)."""
        _, _, body = _full_box(payload)
        tag, start, end = _descriptor(body, 0)
        if tag != 0x03:
            return
        flags = body[start + 2]
        position = start + 3
        if flags & 0x80:
            position += 2
        if flags & 0x40:
            position += 1 + body[position]
        if flags & 0x20:
            position += 2
        while position < end:
            tag, inner, after = _descriptor(body, position)
            if tag == 0x04:
                self.object_type = body[inner]
                position = inner + 13
                while position < after:
                    tag, info, info_end = _descriptor(body, position)
                    if tag == 0x05:
                        self.config = bytes(body[info:info_end])
                    position = info_end
            position = after


class MP4Reader:
    """The tracks of an MP4 file, read by index without reading the media.

    source -- a path, or a binary file object open for reading. A file object
        stays the caller's to close.

    Opening reads the ``moov`` box and nothing of ``mdat``; a sample is read
    when it is asked for. The reader needs no GL, so it can be opened on any
    thread. Fragmented MP4 (``moof``) is not read.

    The box structure is ISO/IEC 14496-12 and the ``avcC`` record ISO/IEC
    14496-15.
    """

    def __init__(self, source: str | Path | BinaryIO) -> None:
        if isinstance(source, (str, Path)):
            self.path: Path | None = Path(source)
            self._file: BinaryIO = self.path.open('rb')  # noqa: OGC121 read as samples are asked for, which is the reader's purpose
            self._owns_file = True
        else:
            self.path = None
            self._file = source
            self._owns_file = False
        self.tracks: list[Track] = []
        try:
            self._read_movie()
        except Exception:
            self.close()
            raise

    def __repr__(self) -> str:
        return f'<{type(self).__name__} {self.path or self._file!r} {self.tracks}>'

    @property
    def video(self) -> VideoTrack | None:
        """The first video track, or None."""
        return next((track for track in self.tracks if isinstance(track, VideoTrack)), None)

    @property
    def audio(self) -> list[AudioTrack]:
        """Every audio track, in file order."""
        return [track for track in self.tracks if isinstance(track, AudioTrack)]

    @property
    def duration(self) -> float:
        """The longest track's length, in seconds."""
        return max((track.duration / track.timescale for track in self.tracks
                    if track.timescale), default=0.0)

    def read_at(self, offset: int, size: int) -> bytes:
        """`size` bytes from `offset` in the file."""
        self._file.seek(offset)
        data = self._file.read(size)
        if len(data) != size:
            raise MP4Error(f'the file ends {size - len(data)} bytes into a '
                           f'sample at {offset}')
        return data

    def close(self) -> None:
        """Close the file, if this reader opened it."""
        if self._owns_file:
            self._file.close()

    def __enter__(self) -> MP4Reader:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()

    # ------------------------------------------------------------ the index

    def _top_level(self) -> Iterator[tuple[str, int, int]]:
        """Each top-level box as ``(type, payload offset, payload size)``."""
        self._file.seek(0, 2)
        end = self._file.tell()
        position = 0
        while position + 8 <= end:
            self._file.seek(position)
            header = self._file.read(16)
            size, kind = struct.unpack_from('>I4s', header)
            length = 8
            if size == 1:
                size = struct.unpack_from('>Q', header, 8)[0]
                length = 16
            elif size == 0:
                size = end - position
            if size < length:
                raise MP4Error(f'a box at {position} claims {size} bytes')
            name = kind.decode('latin-1')
            if position == 0 and name not in ('ftyp', 'moov', 'mdat', 'free',
                                              'skip', 'wide', 'pnot'):
                raise MP4Error(
                    f'the file starts with {kind!r}, which is not an MP4 box')
            yield name, position + length, size - length
            position += size

    def _read_movie(self) -> None:
        for kind, offset, size in self._top_level():
            if kind == 'moov':
                movie = self.read_at(offset, size)
                break
            if kind == 'moof':
                raise MP4Error('fragmented MP4 (moof) is not read here')
        else:
            raise MP4Error('the file has no movie box (moov), so it has no index')
        boxes = list(iter_boxes(movie))
        header = dict(boxes).get('mvhd')
        movie_timescale = 1000
        if header is not None:
            version, _, body = _full_box(header)
            movie_timescale = struct.unpack_from('>I', body, 16 if version == 1 else 8)[0]
        for kind, payload in boxes:
            if kind == 'trak':
                track = self._read_track(payload, movie_timescale)
                if track is not None:
                    self.tracks.append(track)

    def _read_track(self, payload: bytes, movie_timescale: int) -> Track | None:
        trak = _children(payload)
        mdia = _children(trak.get('mdia', b''))
        minf = _children(mdia.get('minf', b''))
        stbl = _children(minf.get('stbl', b''))
        if 'tkhd' not in trak or 'mdhd' not in mdia or 'stsd' not in stbl:
            return None
        version, _, body = _full_box(trak['tkhd'])
        track_id = struct.unpack_from('>I', body, 16 if version == 1 else 8)[0]
        version, _, body = _full_box(mdia['mdhd'])
        if version == 1:
            timescale, duration = struct.unpack_from('>IQ', body, 16)
        else:
            timescale, duration = struct.unpack_from('>II', body, 8)
        handler = mdia['hdlr'][8:12].decode('latin-1') if 'hdlr' in mdia else ''
        _, _, entries = _full_box(stbl['stsd'])
        kind, entry = next(iter_boxes(entries, 4))
        edit_offset = self._edit_offset(trak.get('edts'), timescale, movie_timescale)
        cls: type[Track] = Track
        if handler == 'vide' and kind in ('avc1', 'avc3'):
            cls = VideoTrack
        elif handler == 'soun':
            cls = AudioTrack
        track = cls(self, track_id, handler, kind, timescale, duration, stbl,
                    edit_offset)
        if isinstance(track, (VideoTrack, AudioTrack)):
            track.describe(entry)
        return track

    @staticmethod
    def _edit_offset(edts: bytes | None, timescale: int, movie_timescale: int) -> int:
        """The media time shown at time zero, less any empty edit's delay.

        An edit list most often says that presentation starts a few frames into
        the media, which is how a muxer cancels the delay that B-frame
        reordering puts before the first picture.
        """
        if not edts:
            return 0
        elst = _children(edts).get('elst')
        if elst is None:
            return 0
        version, _, body = _full_box(elst)
        count = struct.unpack_from('>I', body)[0]
        layout, width = ('>Qq', 16) if version == 1 else ('>Ii', 12)
        delay = 0
        for index in range(count):
            duration, media_time = struct.unpack_from(layout, body, 4 + width * index)
            if media_time == -1:
                delay += duration * timescale // max(movie_timescale, 1)
                continue
            return media_time - delay
        return -delay
