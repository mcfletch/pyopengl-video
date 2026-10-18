"""Writing an encoder's elementary stream into an MP4 file, and reading one back.

An H.264 or HEVC encoder hands out Annex-B fragments -- NAL units separated by
start codes -- and that is a fine thing to write to a ``.h264`` file and pipe
somewhere, but it is not what anything plays. MP4 stores the same units
length-prefixed, keeps the parameter sets in the sample description rather than
in the stream, and carries the timing in tables beside the data. An AV1 stream
is stored as its OBUs less the temporal delimiters, with the sequence header in
the sample description, and a VP9 stream frame by frame.

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

The box structure is ISO/IEC 14496-12. The ``avcC`` and ``hvcC`` records are
ISO/IEC 14496-15, ``av1C`` the AV1 Codec ISO Media File Format Binding and
``vpcC`` the VP Codec ISO Media File Format Binding.
"""
from __future__ import annotations

import dataclasses
import functools
import struct
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

from pyopengl_video import codecs, obu
from pyopengl_video.decoder import Sample
from pyopengl_video.encoder import Encoder, Packet
from pyopengl_video.vaapi import av1_parse, hevc_parse

#: NAL unit types that belong in the sample description rather than in a sample.
NAL_SPS = 7
NAL_PPS = 8
#: An access unit delimiter says where a picture starts, which the sample table
#: already says.
NAL_AUD = 9

#: How many bytes prefix each NAL unit in a sample. Four is what ``avcC``
#: advertises below, and what every player expects.
LENGTH_PREFIX = 4

START_CODE = b'\x00\x00\x00\x01'

#: Where the NAL unit arrays begin in an ``hvcC`` record, and the byte before
#: them whose low two bits are the length prefix's size less one.
HVCC_ARRAYS = 22
HVCC_LENGTH_SIZE = 21

#: The NAL unit types an ``hvcC`` array may hold that a decoder is configured with.
HEVC_VPS = 32
HEVC_SPS = 33
HEVC_PPS = 34

#: Where the configuration OBUs begin in an ``av1C`` record.
AV1C_OBUS = 4

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


# ------------------------------------------------------ sample descriptions


def visual_sample_entry(code: str, width: int, height: int, *children: bytes) -> bytes:
    """A VisualSampleEntry of four-character `code` (ISO/IEC 14496-12, 12.1.3)."""
    return box(
        code,
        bytes(6), struct.pack('>H', 1),                    # reserved, data reference
        bytes(16),                                         # pre-defined, reserved
        struct.pack('>HH', width, height),
        struct.pack('>II', 0x00480000, 0x00480000),        # 72 dpi each way
        struct.pack('>I', 0),                              # reserved
        struct.pack('>H', 1),                              # frames per sample
        bytes(32),                                         # compressor name
        struct.pack('>Hh', 0x0018, -1),                    # depth, pre-defined
        *children,
    )


class SampleDescription:
    """What an MP4 track says its samples are, for one codec.

    headers -- the stream's configuration as its encoder states it, which
        stands until the stream itself carries one

    The stream is the authority. An encoder states its configuration before it
    has coded anything and a driver may amend it on the way out, since a
    capability the hardware does not have is one a parameter set must not
    claim. What the samples were coded against is what the sample description
    has to hold: a decoder set up from another one looks for syntax that is
    not in the data, and reports a corrupt picture. The first configuration a
    recording carries is kept, since the description covers every sample.
    """

    #: the codec's name
    codec = ''
    #: the sample entry's four-character code, and the brand ``ftyp`` lists
    code = ''

    def __init__(self, headers: bytes = b'') -> None:
        self.configure(headers)

    def configure(self, headers: bytes) -> None:
        """Take the configuration an encoder advertises."""

    def store(self, data: bytes) -> bytes:
        """One packet as the sample MP4 stores, noting any configuration in it."""
        return data

    def missing(self) -> str:
        """What the description still lacks, in words, or '' when it is whole."""
        return ''

    def entry(self, width: int, height: int) -> bytes:
        """The sample entry box."""
        raise NotImplementedError


class _NalDescription(SampleDescription):
    """A codec of NAL units: Annex-B in, length-prefixed out."""

    #: NAL unit types kept in the sample description, in the order it lists them
    sets: tuple[int, ...] = ()
    #: an access unit delimiter says where a picture starts, which the sample
    #: table already says
    delimiter = -1

    def __init__(self, headers: bytes = b'') -> None:
        self.units: dict[int, bytes] = {}
        self._from_stream: set[int] = set()
        super().__init__(headers)

    @staticmethod
    def kind(unit: bytes) -> int:
        """The NAL unit type of `unit`."""
        raise NotImplementedError

    def configure(self, headers: bytes) -> None:
        for unit in split_annexb(headers):
            if self.kind(unit) in self.sets:
                self.units.setdefault(self.kind(unit), unit)

    def store(self, data: bytes) -> bytes:
        sample = bytearray()
        for unit in split_annexb(data):
            kind = self.kind(unit)
            if kind in self.sets:
                if kind not in self._from_stream:
                    self.units[kind] = unit
                    self._from_stream.add(kind)
            elif kind != self.delimiter:
                sample += struct.pack('>I', len(unit)) + unit
        return bytes(sample)

    def missing(self) -> str:
        return '' if all(kind in self.units for kind in self.sets) else 'parameter sets'


class AVCDescription(_NalDescription):
    """H.264 as ``avc1`` with an ``avcC`` record (ISO/IEC 14496-15, 5.3.3)."""

    codec = codecs.H264
    code = 'avc1'
    sets = (NAL_SPS, NAL_PPS)
    delimiter = NAL_AUD

    @staticmethod
    def kind(unit: bytes) -> int:
        return unit[0] & 0x1F

    def entry(self, width: int, height: int) -> bytes:
        sps, pps = self.units[NAL_SPS], self.units[NAL_PPS]
        record = box(
            'avcC',
            bytes([1, sps[1], sps[2], sps[3], 0xfc | (LENGTH_PREFIX - 1), 0xe0 | 1]),
            struct.pack('>H', len(sps)), sps,
            bytes([1]), struct.pack('>H', len(pps)), pps,
        )
        return visual_sample_entry(self.code, width, height, record)


class HEVCDescription(_NalDescription):
    """HEVC as ``hvc1`` with an ``hvcC`` record (ISO/IEC 14496-15, 8.3.3)."""

    codec = codecs.HEVC
    code = 'hvc1'
    sets = (HEVC_VPS, HEVC_SPS, HEVC_PPS)
    delimiter = hevc_parse.NAL_AUD

    @staticmethod
    def kind(unit: bytes) -> int:
        return (unit[0] >> 1) & 0x3F

    def entry(self, width: int, height: int) -> bytes:
        unit = hevc_parse.parse_nal(self.units[HEVC_SPS])
        sps = hevc_parse.parse_sps(unit.rbsp)
        # The general profile, tier and level are the twelve bytes after the
        # set's first, in the order the record wants them.
        general = unit.rbsp[1:13]
        record = box(
            'hvcC',
            bytes([1]), general,
            struct.pack('>H', 0xF000),                     # min_spatial_segmentation_idc
            bytes([0xFC,                                   # parallelismType: unknown
                   0xFC | sps.chroma_format_idc,
                   0xF8 | sps.bit_depth_luma_minus8,
                   0xF8 | sps.bit_depth_chroma_minus8]),
            struct.pack('>H', 0),                          # avgFrameRate: unspecified
            bytes([(sps.sps_max_sub_layers_minus1 + 1) << 3
                   | sps.sps_temporal_id_nesting_flag << 2 | (LENGTH_PREFIX - 1),
                   len(self.sets)]),
            *(bytes([0x80 | kind]) + struct.pack('>HH', 1, len(self.units[kind]))
              + self.units[kind] for kind in self.sets),
        )
        return visual_sample_entry(self.code, width, height, record)


class AV1Description(SampleDescription):
    """AV1 as ``av01`` with an ``av1C`` record (AV1 Codec ISO Media File Format Binding).

    A sample is its temporal unit less the temporal delimiters, and the
    record carries the sequence header.
    """

    codec = codecs.AV1
    code = 'av01'

    def __init__(self, headers: bytes = b'') -> None:
        self.sequence_header = b''
        self._from_stream = False
        super().__init__(headers)

    def configure(self, headers: bytes) -> None:
        self.sequence_header = obu.sequence_header(headers)

    def store(self, data: bytes) -> bytes:
        if not self._from_stream:
            found = obu.sequence_header(data)
            if found:
                self.sequence_header = found
                self._from_stream = True
        return obu.without(data, obu.TEMPORAL_DELIMITER, obu.PADDING)

    def missing(self) -> str:
        return '' if self.sequence_header else 'sequence header'

    def entry(self, width: int, height: int) -> bytes:
        unit = next(obu.split(self.sequence_header))
        sequence = av1_parse.parse_sequence_header(unit.payload)
        point = sequence.operating_points[0]
        colour = sequence.color_config
        record = box(
            'av1C',
            bytes([0x81,                                   # marker, version 1
                   sequence.seq_profile << 5 | point.seq_level_idx,
                   point.seq_tier << 7 | (colour.bit_depth > 8) << 6
                   | (colour.bit_depth == 12) << 5 | colour.mono_chrome << 4
                   | colour.subsampling_x << 3 | colour.subsampling_y << 2
                   | colour.chroma_sample_position,
                   0]),                                    # no initial presentation delay
            self.sequence_header,
        )
        return visual_sample_entry(self.code, width, height, record)


#: VP9 levels by the largest luma picture each allows, as ten times the level
#: (VP9 levels definition); the record states the lowest that holds the picture.
VP9_LEVELS = ((36864, 10), (73728, 11), (122880, 20), (245760, 21), (552960, 30),
              (983040, 31), (2228224, 40), (8912896, 50), (35651584, 60))


class VP9Description(SampleDescription):
    """VP9 as ``vp09`` with a ``vpcC`` record (VP Codec ISO Media File Format Binding).

    The record describes what the encoders here produce: profile 0, 8 bits,
    4:2:0 with co-located chroma, limited-range BT.709.
    """

    codec = codecs.VP9
    code = 'vp09'

    def entry(self, width: int, height: int) -> bytes:
        level = next((level for most, level in VP9_LEVELS if width * height <= most),
                     VP9_LEVELS[-1][1])
        record = full_box(
            'vpcC', 1, 0,
            bytes([0, level,
                   8 << 4 | 1 << 1 | 0,                    # depth, 4:2:0 co-located, limited
                   1, 1, 1]),                              # BT.709 primaries, transfer, matrix
            struct.pack('>H', 0),                          # no initialisation data
        )
        return visual_sample_entry(self.code, width, height, record)


#: The sample description for each codec the writer stores.
DESCRIPTIONS: dict[str, type[SampleDescription]] = {
    description.codec: description
    for description in (AVCDescription, HEVCDescription, AV1Description, VP9Description)
}


class MP4Writer:
    """Collects packets from an encoder into a playable MP4 file.

    path -- where to write
    encoder -- take the codec, frame size, timescale and configuration from
        this encoder; give the keyword arguments instead when the packets come
        from somewhere else
    width/height -- frame size in pixels
    timescale -- units of a second that packet timestamps are counted in
    codec -- ``'h264'``, ``'hevc'``, ``'av1'`` or ``'vp9'``; H.264 when neither
        this nor an encoder says
    parameter_sets -- for H.264, ``(sps, pps)`` as raw NAL units
    headers -- for any codec, the stream's configuration as an encoder's
        :meth:`~pyopengl_video.encoder.Encoder.headers` gives it: parameter
        sets as Annex-B, or a sequence header OBU. When neither this nor
        `parameter_sets` is given, the configuration is taken from the first
        packets that carry one.

    description -- the track's :class:`SampleDescription`, which holds the
        configuration the file will state
    """

    #: Units of a second the movie header counts in. The track keeps the
    #: encoder's own timescale; this one only has to describe the whole movie.
    movie_timescale = 1000

    def __init__(self, path: str | Path, encoder: Encoder | None = None, *,
                 width: int | None = None, height: int | None = None,
                 timescale: int | None = None, codec: str | None = None,
                 parameter_sets: tuple[bytes, bytes] | None = None,
                 headers: bytes | None = None):
        if encoder is not None:
            width, height = encoder.size
            timescale = encoder.timescale
            codec = codec or encoder.codec
            if parameter_sets is None and headers is None:
                headers = encoder.headers()
        if width is None or height is None or timescale is None:
            raise ValueError('give an encoder, or width, height and timescale')
        self.codec = codecs.canonical(codec or codecs.H264)
        if self.codec not in DESCRIPTIONS:
            raise ValueError(f'an MP4Writer stores {", ".join(sorted(DESCRIPTIONS))}, '
                             f'not {codec!r}')
        if parameter_sets is not None:
            headers = b''.join(START_CODE + unit for unit in parameter_sets if unit)
        self.path = Path(path)
        self.size = (int(width), int(height))
        self.timescale = int(timescale)
        # What is here now was advertised rather than observed, and stands only
        # until the stream says otherwise.
        self.description = DESCRIPTIONS[self.codec](headers or b'')

        self._sizes: list[int] = []
        self._durations: list[int] = []
        self._timestamps: list[int] = []
        self._sync: list[int] = []
        self._closed = False

        self._file = self.path.open('wb')  # noqa: OGC121 a recording written as it is captured, the way a camera writes one
        header = box('ftyp', b'isom', struct.pack('>I', 512),
                     b'isom', b'iso2', self.description.code.encode('ascii'), b'mp41')
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
        sample = self.description.store(packet.data)
        if not sample:
            return
        self._file.write(sample)
        if packet.keyframe:
            self._sync.append(len(self._sizes) + 1)      # sample numbers are one-based
        self._sizes.append(len(sample))
        self._durations.append(int(packet.duration))
        self._timestamps.append(int(packet.timestamp))

    # ------------------------------------------------------------- closing

    def close(self) -> None:
        """Finish the file: patch the media length, then write the tables."""
        if self._closed:
            return
        self._closed = True
        try:
            if not self._sizes:
                raise ValueError(f'{self.path} has no samples in it')
            missing = self.description.missing()
            if missing:
                raise ValueError(
                    f'{self.path} has no {missing}: give headers, or let the '
                    'encoder repeat them in the stream')
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
        return full_box('stsd', 0, 0, struct.pack('>I', 1),
                        self.description.entry(*self.size))

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


#: The most samples one track may declare. A table claims its length in a
#: 32-bit field, and a constant-size ``stsz`` carries no entries to check it
#: against; this is 77 hours at 60 frames a second.
MAX_SAMPLES = 1 << 24


def _table(body: bytes, width: int, offset: int = 4) -> int:
    """The entry count at the head of a table, checked against the bytes that hold it.

    A box claims its entry count in a 32-bit field; allocating from the claim
    rather than from the box lets a file of a few hundred bytes ask for gigabytes.
    """
    count = struct.unpack_from('>I', body)[0]
    if offset + count * width > len(body):
        raise MP4Error(f'a table claims {count} entries of {width} bytes in a box of '
                       f'{len(body)} bytes')
    return count


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
                if count > MAX_SAMPLES:
                    raise MP4Error(f'a track declares {count} samples')
                return [size] * count
            count = _table(body[4:], 4, 4)
            return list(struct.unpack_from(f'>{count}I', body, 8))
        if 'stz2' in table:
            _, _, body = _full_box(table['stz2'])
            field, count = body[3], struct.unpack_from('>I', body, 4)[0]
            if count * field > 8 * (len(body) - 8):
                raise MP4Error(f'stz2 claims {count} sizes of {field} bits in a box of '
                               f'{len(body)} bytes')
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
            chunks = list(struct.unpack_from(f'>{_table(body, 4)}I', body, 4))
        elif 'co64' in table:
            _, _, body = _full_box(table['co64'])
            chunks = list(struct.unpack_from(f'>{_table(body, 8)}Q', body, 4))
        else:
            raise MP4Error('a track has no chunk offset table (stco or co64)')
        if 'stsc' not in table:
            raise MP4Error('a track has no sample-to-chunk table (stsc)')
        _, _, body = _full_box(table['stsc'])
        entries = _table(body, 12)
        runs = [struct.unpack_from('>III', body, 4 + 12 * index)
                for index in range(entries)]
        if any(not 1 <= first <= len(chunks) for first, _per_chunk, _description in runs):
            raise MP4Error('the sample-to-chunk table names a chunk the file does not have')
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
        entries = _table(body, 8)
        times, durations = [], []
        time = 0
        for index in range(entries):
            runs, delta = struct.unpack_from('>II', body, 4 + 8 * index)
            for _ in range(min(runs, count - len(times))):
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
        entries = _table(body, 8)
        offsets: list[int] = []
        for index in range(entries):
            # Signed in either version: version 0 files from some muxers carry
            # negative offsets, and no positive one reaches 2**31.
            runs, offset = struct.unpack_from('>Ii', body, 4 + 8 * index)
            offsets.extend([offset] * min(runs, count - len(offsets)))
        return (offsets + [0] * count)[:count]

    @staticmethod
    def _read_sync(table: dict[str, bytes], count: int) -> list[int]:
        """The indices of the sync samples; every sample is one without ``stss``."""
        if 'stss' not in table:
            return list(range(count))
        _, _, body = _full_box(table['stss'])
        entries = _table(body, 4)
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
    """A video track in a codec this package decodes.

    H.264 (``avc1``, ``avc3``), HEVC (``hvc1``, ``hev1``), AV1 (``av01``) or
    VP9 (``vp09``). :func:`pyopengl_video.codecs.canonical` turns
    :attr:`~Track.codec` into the name a decoder is opened by.

    width/height -- the picture size the sample entry states
    vps/sps/pps -- the parameter sets ``avcC`` or ``hvcC`` carries, as NAL
        units; ``vps`` is empty for H.264, and all three for AV1 and VP9
    configuration -- the configuration OBUs ``av1C`` carries; empty otherwise
    length_size -- bytes in the length prefix of each NAL unit in a sample
    """

    width = 0
    height = 0
    length_size = LENGTH_PREFIX
    configuration = b''
    vps: list[bytes]
    sps: list[bytes]
    pps: list[bytes]

    @property
    def size(self) -> tuple[int, int]:
        """The picture size the sample entry states, as (width, height)."""
        return self.width, self.height

    def describe(self, entry: bytes) -> None:
        """Take the picture size and the decoder configuration from the sample entry."""
        self.width, self.height = struct.unpack_from('>HH', entry, 24)
        self.vps, self.sps, self.pps = [], [], []
        self._kind = codecs.of_sample_entry(self.codec)
        children = _children(entry[78:])
        if 'avcC' in children:
            self._read_avcc(children['avcC'])
        elif 'hvcC' in children:
            self._read_hvcc(children['hvcC'])
        elif 'av1C' in children:
            self.configuration = bytes(children['av1C'][AV1C_OBUS:])

    @property
    def _nal_units(self) -> bool:
        """Whether a sample is length-prefixed NAL units, as H.264 and HEVC are."""
        return self._kind in (codecs.H264, codecs.HEVC)

    def _read_avcc(self, config: bytes) -> None:
        """The SPS and PPS of an AVCDecoderConfigurationRecord."""
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

    def _read_hvcc(self, config: bytes) -> None:
        """The VPS, SPS and PPS arrays of an HEVCDecoderConfigurationRecord."""
        if len(config) <= HVCC_ARRAYS:
            raise MP4Error(f'an hvcC record of {len(config)} bytes ends before its arrays')
        self.length_size = (config[HVCC_LENGTH_SIZE] & 3) + 1
        kinds = {HEVC_VPS: self.vps, HEVC_SPS: self.sps, HEVC_PPS: self.pps}
        position = HVCC_ARRAYS + 1
        for _ in range(config[HVCC_ARRAYS]):
            kind = config[position] & 0x3F
            count = struct.unpack_from('>H', config, position + 1)[0]
            position += 3
            for _ in range(count):
                length = struct.unpack_from('>H', config, position)[0]
                unit = config[position + 2:position + 2 + length]
                position += 2 + length
                if kind in kinds:
                    kinds[kind].append(unit)

    def parameter_sets(self) -> bytes:
        """The sample entry's configuration, as a decoder's ``headers``.

        The parameter sets as an Annex-B fragment for H.264 and HEVC, the
        configuration OBUs for AV1, and nothing for VP9.
        """
        if self._nal_units:
            return b''.join(START_CODE + unit for unit in self.vps + self.sps + self.pps)
        return self.configuration

    def sample(self, index: int, parameter_sets: bool = False) -> Sample:
        """Sample `index` in the form every decoder here takes.

        Annex-B for H.264 and HEVC; the temporal unit's OBUs for AV1 and the
        frame or superframe for VP9, each as stored.

        parameter_sets -- put the sample entry's configuration in front, which
            a decoder needs before the first sample it sees and after a seek
        """
        found = super().sample(index)
        raw = found.data
        prefix = self.parameter_sets() if parameter_sets else b''
        if not self._nal_units:
            return dataclasses.replace(found, data=prefix + raw) if prefix else found
        units = []
        position = 0
        size = self.length_size
        while position + size <= len(raw):
            length = int.from_bytes(raw[position:position + size], 'big')
            position += size
            units.append(START_CODE + raw[position:position + length])
            position += length
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

    A video track in H.264, HEVC, AV1 or VP9 is a :class:`VideoTrack`; one in
    another codec is listed in :attr:`tracks` as a plain :class:`Track`.
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
        """The first video track in a codec this package decodes, or None."""
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
        end = self._file.seek(0, 2)
        if offset < 0 or size < 0 or offset + size > end:
            raise MP4Error(f'{size} bytes at {offset} run past the end of a file of '
                           f'{end} bytes')
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
        """Index the movie, turning any malformed box into an :class:`MP4Error`."""
        try:
            self._parse_movie()
        except (struct.error, IndexError, StopIteration, UnicodeError,
                OverflowError, MemoryError) as error:
            raise MP4Error(f'the movie box is malformed: {error!r}') from error

    def _parse_movie(self) -> None:
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
        if handler == 'vide' and codecs.of_sample_entry(kind):
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
