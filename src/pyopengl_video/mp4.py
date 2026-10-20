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

import array
import bisect
import dataclasses
import functools
import itertools
import logging
import math
import struct
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, BinaryIO

from pyopengl_video import bounds, codecs, container, obu
from pyopengl_video.decoder import Sample
from pyopengl_video.encoder import Encoder, Packet
from pyopengl_video.vaapi import av1_parse, hevc_parse

log = logging.getLogger(__name__)

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

#: The largest file offset and the largest time a signed 64-bit field holds.
LARGEST_OFFSET = LARGEST_TIME = (1 << 63) - 1

#: The range of a composition offset: a signed 32-bit field of ``ctts``.
OFFSET_RANGE = (-(1 << 31), (1 << 31) - 1)

#: How far a packet's timestamp may fall behind the latest one written, and how
#: far from zero the first may be, in the track's timescale. A composition
#: offset is the first timestamp plus how far reordering moved the sample, so
#: holding each to this keeps their sum inside :data:`OFFSET_RANGE`.
REORDER_LIMIT = 1 << 30


class MP4Error(container.ContainerError):
    """A file is not an MP4 this reader can read, or breaks the box syntax.

    An :class:`MP4Writer` raises it for a size, a timescale or a timestamp the
    file has no field wide enough for.
    """


class Cursor(container.Cursor):
    """The fields of an MP4 box, which are big-endian; raises :class:`MP4Error`."""

    error = MP4Error


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
    #: how many times the configuration has changed. A writer builds the entry
    #: again after each change, while the packet that made it can be refused.
    revision = 0

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
        """The sample entry box.

        Raises :class:`MP4Error` or
        :class:`~pyopengl_video.vaapi.h264.StreamError` for a configuration
        no entry can be made from.
        """
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
            if self.kind(unit) in self.sets and self.kind(unit) not in self.units:
                self.units[self.kind(unit)] = unit
                self.revision += 1

    def store(self, data: bytes) -> bytes:
        sample = bytearray()
        for unit in split_annexb(data):
            kind = self.kind(unit)
            if kind in self.sets:
                if kind not in self._from_stream:
                    self.units[kind] = unit
                    self._from_stream.add(kind)
                    self.revision += 1
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
        if len(sps) < 4:
            raise MP4Error(f'a sequence parameter set of {len(sps)} bytes has no '
                           'profile and level for the avcC record to state')
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
        self.revision += 1

    def store(self, data: bytes) -> bytes:
        if not self._from_stream:
            found = obu.sequence_header(data)
            if found:
                self.sequence_header = found
                self._from_stream = True
                self.revision += 1
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


class SampleTimes:
    """The timestamps and durations of a track's samples, in decoding order.

    timestamps/durations -- as the packets gave them, one of each a sample

    The sample tables state a duration in 32 bits and the distance from a
    sample's decode time to its display time in 32 signed bits, with decode
    times counted from zero. :meth:`add` refuses a packet whose times those
    fields could not state, so every packet it has taken is one the tables
    can describe.
    """

    def __init__(self) -> None:
        self.timestamps: list[int] = []
        self.durations: list[int] = []
        self._seen: set[int] = set()
        self._latest = 0
        #: a timestamp has repeated, so decode times are sums of `durations`
        self._repeated = False
        self._summed = 0
        #: the least and greatest of each timestamp less the durations before it
        self._summed_offsets: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self.timestamps)

    def add(self, timestamp: object, duration: object) -> None:
        """Take one packet's times, or raise :class:`MP4Error` and take nothing."""
        stamp = bounds.whole(timestamp, 'a timestamp', -REORDER_LIMIT, LARGEST_TIME, MP4Error)
        lasts = bounds.whole(duration, 'a duration', 0, bounds.FIELD_32, MP4Error)
        offset = stamp - self._summed
        low, high = self._summed_offsets or (offset, offset)
        offsets = (min(low, offset), max(high, offset))
        repeated = self._repeated or stamp in self._seen
        if repeated:
            if offsets[0] < OFFSET_RANGE[0] or offsets[1] > OFFSET_RANGE[1]:
                raise MP4Error(
                    f'the timestamp {stamp} repeats, which puts decode times on '
                    'the packets\' durations, and those leave a display time '
                    f'{max(-offsets[0], offsets[1])} ticks from its decode time: '
                    f'more than the {OFFSET_RANGE[1]} a composition offset states')
        elif not self.timestamps:
            if stamp >= REORDER_LIMIT:
                raise MP4Error(
                    f'the first timestamp is {stamp}, and a track\'s decode times '
                    f'start at zero: the first must be under {REORDER_LIMIT} ticks, '
                    'so count timestamps from the start of the recording')
        elif stamp < self._latest - REORDER_LIMIT:
            raise MP4Error(
                f'the timestamp {stamp} is {self._latest - stamp} ticks behind the '
                f'latest, {self._latest}; a picture may be at most {REORDER_LIMIT} behind')
        elif stamp > self._latest + bounds.FIELD_32:
            raise MP4Error(
                f'the timestamp {stamp} is {stamp - self._latest} ticks after the '
                f'one before it; a sample lasts at most {bounds.FIELD_32}')
        self._latest = max(self._latest, stamp) if self.timestamps else stamp
        self.timestamps.append(stamp)
        self.durations.append(lasts)
        self._seen.add(stamp)
        self._repeated = repeated
        self._summed += lasts
        self._summed_offsets = offsets

    def sample_durations(self) -> list[int]:
        """Each sample's duration, in decode order.

        A sample lasts until the next display time: the n-th decode time is
        the n-th timestamp in display order, so decode times stay on the
        caller's timestamps rather than on a sum of rounded frame durations.
        The last sample keeps the duration its packet gave. Timestamps that
        repeat say nothing about how long a sample lasts, and then the
        packets' own durations are used.
        """
        if self._repeated:
            return list(self.durations)
        shown = sorted(self.timestamps)
        return [later - earlier for earlier, later in itertools.pairwise(shown)
                ] + self.durations[-1:]

    def composition_offsets(self, durations: list[int]) -> list[int]:
        """How far each sample's display time is from its decode time.

        Samples are stored in decode order and a packet's timestamp is its
        display time, so the two part company as soon as an encoder reorders
        pictures. Decode time is the running total of ``durations``.
        """
        decode_times = itertools.accumulate(durations, initial=0)
        return [timestamp - decode_time for timestamp, decode_time
                in zip(self.timestamps, decode_times, strict=False)]


def _header_times(duration: int, *between: int) -> tuple[int, bytes]:
    """A header box's version, and its fields from the creation time to the duration.

    between -- the 32-bit fields the box has between its modification time
        and its duration

    Version 1 states the times in 64 bits, and is used for a duration 32 bits
    do not hold.
    """
    if duration > bounds.FIELD_32:
        return 1, struct.pack(f'>QQ{len(between)}IQ', 0, 0, *between, duration)
    return 0, struct.pack(f'>II{len(between)}II', 0, 0, *between, duration)


#: The least media a segment holds before the writer takes a checkpoint, in bytes.
CHECKPOINT_BYTES = 4 << 20

#: A checkpoint is taken when the media since the last one is this many times the
#: size of the movie box it writes, so the retired boxes stay a small share of the
#: file however long the recording runs.
MOVIE_SHARE = 64

#: The most media one segment holds. A segment's ``mdat`` states its length in
#: 32 bits.
LARGEST_SEGMENT = 1 << 30


class MP4Writer:
    """Collects packets from an encoder into a playable MP4 file.

    path -- where to write
    encoder -- take the codec, frame size, timescale and configuration from
        this encoder; give the keyword arguments instead when the packets come
        from somewhere else
    width/height -- frame size in pixels, each from 1 to 65535
    timescale -- units of a second that packet timestamps are counted in,
        from 1 to 2**32 - 1
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
    checkpoint -- the least media, in bytes, to write between checkpoints, or
        None for a file with its tables at the end only

    A checkpoint finishes the ``mdat`` box being written and follows it with a
    complete ``moov`` box, and the next one turns that box into a ``free`` box
    and writes its own. A process that dies mid-recording leaves a file that
    plays up to its last checkpoint, with the samples after it in an ``mdat``
    that runs to the end of the file. The finished file is ``ftyp``, the
    segments and their retired boxes, and one ``moov`` last, which every player
    reads. A recording shorter than `checkpoint` is ``ftyp``, ``mdat`` and
    ``moov``. Checkpoints are spaced so that the retired boxes are a small share
    of the file: at least `checkpoint` bytes apart, and further apart as the
    sample tables grow.

    Raises :class:`MP4Error` for a size or a timescale outside those ranges,
    before the file is created.

    As a context manager the movie is closed when the block ends. Where the
    block ends in an exception, what was written before it is finished as a
    movie and the exception goes on; a failure to finish is logged.
    """

    #: Units of a second the movie header counts in. The track keeps the
    #: encoder's own timescale; this one only has to describe the whole movie.
    movie_timescale = 1000

    def __init__(self, path: str | Path, encoder: Encoder | None = None, *,
                 width: int | None = None, height: int | None = None,
                 timescale: int | None = None, codec: str | None = None,
                 parameter_sets: tuple[bytes, bytes] | None = None,
                 headers: bytes | None = None,
                 checkpoint: int | None = CHECKPOINT_BYTES):
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
        if checkpoint is not None and checkpoint < 1:
            raise ValueError(f'checkpoint is a number of bytes from 1, not {checkpoint!r}; '
                             'None writes the tables at the end only')
        self.path = Path(path)
        # The sample entry states each in 16 bits.
        self.size = bounds.frame_size(width, height, error=MP4Error,
                                      most=(bounds.FIELD_16, bounds.FIELD_16))
        self.timescale = bounds.whole(timescale, 'the timescale', 1, bounds.FIELD_32,
                                      MP4Error)
        # What is here now was advertised rather than observed, and stands only
        # until the stream says otherwise.
        self.description = DESCRIPTIONS[self.codec](headers or b'')
        self._described = -1

        self._sizes: list[int] = []
        self._times = SampleTimes()
        self._sync: list[int] = []
        self._closed = False
        self._checkpoint = checkpoint
        #: ``(offset, samples)`` of each finished segment of media
        self._chunks: list[tuple[int, int]] = []
        #: where the segment being written starts, or None between segments
        self._segment: int | None = None
        self._segment_data = 0
        self._segment_samples = 0
        self._segment_bytes = 0
        #: where the last checkpoint's movie box starts
        self._movie_at: int | None = None

        self._file = self.path.open('wb')  # noqa: OGC121 a recording written as it is captured, the way a camera writes one
        try:
            self._file.write(box(
                'ftyp', b'isom', struct.pack('>I', 512),
                b'isom', b'iso2', self.description.code.encode('ascii'), b'mp41'))
        except BaseException:
            self._file.close()
            raise

    # ------------------------------------------------------------- writing

    def write(self, packets: Packet | Iterable[Packet]) -> None:
        """Add a packet, or everything in an iterable of them.

        Raises :class:`MP4Error` for a packet the file cannot describe, and
        writes nothing of it: the packets before it are still a movie.

        - a timestamp or a duration that is not a whole number, or a duration
          outside 0 to 2**32 - 1;
        - a first timestamp of :data:`REORDER_LIMIT` (2**30) or more, or any
          below ``-REORDER_LIMIT``: a track's decode times start at zero and a
          display time is stated as a 32-bit offset from one;
        - a timestamp more than :data:`REORDER_LIMIT` behind the latest, or
          more than 2**32 - 1 after it;
        - a configuration the sample entry cannot be made from.
        """
        if isinstance(packets, Packet):
            packets = (packets,)
        for packet in packets:
            self._write_packet(packet)

    def _write_packet(self, packet: Packet) -> None:
        if self._closed:
            raise ValueError('the movie is closed')
        sample = self.description.store(packet.data)
        self._check_description()
        if not sample:
            return
        if len(sample) > bounds.FIELD_32 or len(self._sizes) >= bounds.FIELD_32:
            raise MP4Error(
                f'a sample of {len(sample)} bytes after {len(self._sizes)} others: '
                f'the sample size table counts each in 32 bits')
        self._times.add(packet.timestamp, packet.duration)
        if self._segment is None:
            self._open_segment()
        self._file.write(sample)
        if packet.keyframe:
            self._sync.append(len(self._sizes) + 1)      # sample numbers are one-based
        self._sizes.append(len(sample))
        self._segment_samples += 1
        self._segment_bytes += len(sample)
        if self._checkpoint is not None and self._checkpoint_due():
            self._take_checkpoint()

    def _open_segment(self) -> None:
        """Start an ``mdat`` box whose length is filled in when the segment ends."""
        self._segment = self._file.tell()
        if self._checkpoint is None:
            # A 64-bit box, so a long recording is not cut short at four gigabytes.
            self._file.write(struct.pack('>I4sQ', 1, b'mdat', 0))
        else:
            # A length of 0 is a box that runs to the end of the file, which is
            # what the segment is until it is finished.
            self._file.write(struct.pack('>I4s', 0, b'mdat'))
        self._segment_data = self._file.tell()
        self._segment_samples = 0
        self._segment_bytes = 0

    def _close_segment(self) -> None:
        """State the length of the segment being written, and keep it as a chunk."""
        assert self._segment is not None
        end = self._file.tell()
        length = end - self._segment
        if self._checkpoint is None:
            self._file.seek(self._segment + 8)
            self._file.write(struct.pack('>Q', length))
        else:
            if length > bounds.FIELD_32:
                raise MP4Error(f'{length} bytes of samples came with no configuration '
                               'for a decoder, so they could not be indexed')
            self._file.seek(self._segment)
            self._file.write(struct.pack('>I', length))
        self._file.seek(end)
        self._chunks.append((self._segment_data, self._segment_samples))
        self._segment = None

    def _checkpoint_due(self) -> bool:
        """Whether the media since the last checkpoint is worth another movie box."""
        assert self._checkpoint is not None
        if self._segment_bytes >= LARGEST_SEGMENT:
            return True
        movie_estimate = 4 * len(self._sizes) + 1024
        return self._segment_bytes >= max(self._checkpoint, MOVIE_SHARE * movie_estimate)

    def _take_checkpoint(self) -> None:
        """Finish the segment and follow it with the movie as it stands.

        Not taken while the stream has stated no configuration, since a movie
        box cannot be made without one.
        """
        if self.description.missing():
            return
        self._close_segment()
        self._write_movie()

    def _write_movie(self) -> None:
        """Write the movie box, and turn the one before it into a ``free`` box."""
        moov = self._moov()
        position = self._file.tell()
        self._file.write(moov)
        self._file.flush()
        if self._movie_at is not None:
            self._file.seek(self._movie_at + 4)
            self._file.write(b'free')
            self._file.seek(position + len(moov))
            self._file.flush()
        self._movie_at = position

    def _check_description(self) -> None:
        """Build the sample entry whenever the configuration has changed.

        A configuration no entry can be made from is then refused with the
        packet that carried it.
        """
        if self._described != self.description.revision and not self.description.missing():
            self.description.entry(*self.size)
            self._described = self.description.revision

    # ------------------------------------------------------------- closing

    def close(self) -> None:
        """Finish the file: patch the media length, then write the tables.

        Raises :class:`MP4Error` for a movie that cannot be finished. One with
        no samples is removed, since the file holds nothing. One whose samples
        came with no configuration for a decoder is left as it is, the samples
        in an ``mdat`` box with no ``moov`` after it, which no player opens.
        Safe to call twice.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if not self._sizes:
                self._file.close()
                self.path.unlink(missing_ok=True)
                raise MP4Error(f'{self.path} has no samples in it, and was removed')
            missing = self.description.missing()
            if missing:
                raise MP4Error(
                    f'{self.path} has no {missing}, so it has no index: give '
                    'headers, or let the encoder repeat them in the stream')
            if self._segment is not None:
                self._close_segment()
                self._write_movie()
        finally:
            self._file.close()

    def __enter__(self) -> MP4Writer:
        return self

    def __exit__(self, kind: type[BaseException] | None, *exception: Any) -> None:
        if kind is None:
            self.close()
            return
        try:
            self.close()
        except Exception as error:  # noqa: BLE001 - the block's own exception is the one to raise
            log.warning('%s was not finished: %s', self.path, error)

    # -------------------------------------------------------------- tables

    @property
    def duration(self) -> int:
        """The movie's length, in the track's timescale."""
        return sum(self._times.sample_durations())

    @staticmethod
    def _runs(values: Iterable[int]) -> list[tuple[int, int]]:
        """Run-length encode `values` as ``(count, value)`` pairs."""
        return [(len(list(run)), value) for value, run in itertools.groupby(values)]

    def _moov(self) -> bytes:
        """The movie box: everything about the track except its data."""
        movie_duration = round(self.duration * self.movie_timescale / self.timescale)
        version, times = _header_times(movie_duration, self.movie_timescale)
        mvhd = full_box(
            'mvhd', version, 0, times,
            struct.pack('>IHH', 0x00010000, 0x0100, 0),        # rate, volume, reserved
            struct.pack('>II', 0, 0),                          # reserved
            struct.pack('>9I', *UNITY_MATRIX),
            bytes(24),                                         # pre-defined
            struct.pack('>I', 2),                              # next track id
        )
        return box('moov', mvhd, self._trak(movie_duration))

    def _trak(self, movie_duration: int) -> bytes:
        width, height = self.size
        version, times = _header_times(movie_duration, 1, 0)   # track id, reserved
        tkhd = full_box(
            'tkhd', version, 0x000007,                         # enabled, in movie/preview
            times,
            struct.pack('>IIhhhh', 0, 0, 0, 0, 0, 0),          # reserved, layer, volume
            struct.pack('>9I', *UNITY_MATRIX),
            struct.pack('>II', width << 16, height << 16),
        )
        return box('trak', tkhd, self._mdia())

    def _mdia(self) -> bytes:
        version, times = _header_times(self.duration, self.timescale)
        mdhd = full_box(
            'mdhd', version, 0, times,
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
        durations = self._times.sample_durations()
        tables = [self._stsd(), self._stts(durations)]
        offsets = self._times.composition_offsets(durations)
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
        entries: list[tuple[int, int]] = []
        for number, (_, count) in enumerate(self._chunks, 1):
            if not entries or entries[-1][1] != count:
                entries.append((number, count))
        return full_box('stsc', 0, 0, struct.pack('>I', len(entries)),
                        *(struct.pack('>III', first, count, 1) for first, count in entries))

    def _stsz(self) -> bytes:
        return full_box('stsz', 0, 0, struct.pack('>II', 0, len(self._sizes)),
                        *(struct.pack('>I', size) for size in self._sizes))

    def _co64(self) -> bytes:
        return full_box('co64', 0, 0, struct.pack('>I', len(self._chunks)),
                        *(struct.pack('>Q', offset) for offset, _ in self._chunks))


# =================================================================== reading


def iter_boxes(data: bytes, start: int = 0,
               end: int | None = None) -> Iterator[tuple[str, bytes]]:
    """Yield each box in `data[start:end]` as ``(type, payload)``.

    Raises :class:`MP4Error` for a box that states more bytes than the span
    holds, or fewer than its own header.
    """
    cursor = Cursor(data, 'a list of boxes', start, end)
    while cursor.remaining >= 8:
        position = cursor.position
        size, kind = cursor.u32(), cursor.take(4)
        header = 8
        if size == 1:
            if cursor.remaining < 8:
                raise MP4Error(f'a {kind!r} box at {position} is cut short')
            size = cursor.u64()
            header = 16
        elif size == 0:
            size = cursor.end - position
        if size < header or size - header > cursor.remaining:
            raise MP4Error(
                f'a {kind!r} box at {position} claims {size} bytes, which the '
                'space it is in cannot hold')
        yield kind.decode('latin-1'), cursor.take(size - header)


def _children(payload: bytes) -> dict[str, bytes]:
    """The boxes inside a container box, by type; the first of each type."""
    found: dict[str, bytes] = {}
    for kind, child in iter_boxes(payload):
        found.setdefault(kind, child)
    return found


def _full_box(boxes: dict[str, bytes], kind: str) -> tuple[int, Cursor]:
    """Full box `kind` of `boxes`: its version, and a cursor past its flags."""
    cursor = Cursor(boxes[kind], f'the {kind} box')
    return cursor.u32() >> 24, cursor


def _descriptor(cursor: Cursor) -> tuple[int, Cursor]:
    """The next MPEG-4 descriptor in `cursor`: its tag, and a cursor over its body."""
    tag = cursor.u8()
    length = 0
    for _ in range(4):
        byte = cursor.u8()
        length = (length << 7) | (byte & 0x7F)
        if not byte & 0x80:
            break
    return tag, Cursor(cursor.take(length), f'descriptor {tag:#04x} of {cursor.what}')


class SampleTable:
    """A track's sample tables, read whole into arrays of a few bytes a sample.

    table -- the boxes of the track's ``stbl``, by type
    file_size -- the length of the file the samples are in

    sizes -- each sample's size in bytes
    offsets -- where each sample starts in the file
    decode_times/durations -- each sample's decode time and duration
    composition -- how far each sample's display time is from its decode time
    sync -- the indices of the samples decoding can start at, ascending: a
        list, or a range where every sample is one

    Raises :class:`MP4Error` for a table that states more entries than its
    box holds, for a constant sample size that places more bytes than the
    file has, and for chunks or sync samples the track does not have. A run
    that names more samples than the track has is read as far as the track
    goes.
    """

    def __init__(self, table: dict[str, bytes], file_size: int) -> None:
        self.sizes = self._read_sizes(table, file_size)
        count = len(self.sizes)
        self.offsets = self._read_offsets(table, count)
        self.decode_times, self.durations = self._read_times(table, count)
        self.composition = self._read_composition(table, count)
        self.sync = self._read_sync(table, count)

    @staticmethod
    def _read_sizes(table: dict[str, bytes], file_size: int) -> array.array[int]:
        """Each sample's size in bytes (stsz, or the compact stz2)."""
        if 'stsz' in table:
            _, cursor = _full_box(table, 'stsz')
            size, count = cursor.u32(), cursor.u32()
            if not size:
                return cursor.table('I', count)
            if count > file_size // size:
                raise MP4Error(
                    f'the stsz box states {count} samples of {size} bytes each, '
                    f'in a file of {file_size} bytes')
            return array.array('I', [size]) * count
        if 'stz2' in table:
            _, cursor = _full_box(table, 'stz2')
            cursor.skip(3)
            field, count = cursor.u8(), cursor.u32()
            if field == 16:
                return cursor.table('H', count)
            if field == 8:
                return cursor.table('B', count)
            if field == 4:
                packed = cursor.table('B', (count + 1) // 2)
                return array.array('B', (nibble for byte in packed
                                         for nibble in (byte >> 4, byte & 0xF)))[:count]
            raise MP4Error(f'the stz2 box states sizes of {field} bits each, and '
                           'a compact size is 4, 8 or 16')
        raise MP4Error('a track has no sample size table (stsz)')

    def _read_offsets(self, table: dict[str, bytes], count: int) -> array.array[int]:
        """Each sample's file offset, from its chunk's offset and the sizes before it."""
        if 'stco' in table:
            _, cursor = _full_box(table, 'stco')
            chunks = cursor.table('I', cursor.count(4))
        elif 'co64' in table:
            _, cursor = _full_box(table, 'co64')
            chunks = cursor.table('Q', cursor.count(8))
        else:
            raise MP4Error('a track has no chunk offset table (stco or co64)')
        if 'stsc' not in table:
            raise MP4Error('a track has no sample-to-chunk table (stsc)')
        _, cursor = _full_box(table, 'stsc')
        runs = cursor.table('I', 3 * cursor.count(12))
        firsts, per_chunks = runs[0::3], runs[1::3]
        for earlier, first in itertools.pairwise([0, *firsts]):
            if not earlier < first <= len(chunks):
                raise MP4Error(
                    f'the stsc box names chunk {first} after chunk {earlier}, in '
                    f'a track of {len(chunks)} chunks: chunks count from 1 and ascend')
        offsets: array.array[int] = array.array('Q')
        # A run covers the chunks from its first to the next run's first.
        spans = itertools.pairwise([*firsts, len(chunks) + 1])
        for (first, last), per_chunk in zip(spans, per_chunks, strict=True):
            for chunk in range(first, last):
                if len(offsets) == count:
                    return offsets
                sizes = self.sizes[len(offsets):len(offsets) + per_chunk]
                if not sizes:
                    continue
                position = chunks[chunk - 1]
                if position + sum(sizes) > LARGEST_OFFSET:
                    raise MP4Error(f'chunk {chunk} is placed at byte {position}, '
                                   'which no file reaches')
                offsets.extend(itertools.accumulate(sizes[:-1], initial=position))
        if len(offsets) != count:
            raise MP4Error(
                f'the chunk tables place {len(offsets)} samples of {count}')
        return offsets

    @staticmethod
    def _read_times(table: dict[str, bytes],
                    count: int) -> tuple[array.array[int], array.array[int]]:
        """Each sample's decode time and duration (stts)."""
        if 'stts' not in table:
            raise MP4Error('a track has no decoding time table (stts)')
        _, cursor = _full_box(table, 'stts')
        runs = cursor.table('I', 2 * cursor.count(8))
        times: array.array[int] = array.array('q')
        durations: array.array[int] = array.array('I')
        time = 0
        for stated, delta in zip(runs[0::2], runs[1::2], strict=True):
            # A run past the last sample describes nothing, and is not expanded.
            samples = min(stated, count - len(times))
            end = time + samples * delta
            if end > LARGEST_TIME:
                raise MP4Error(f'the stts box puts a sample at time {end}, past '
                               'what 64 bits state')
            times.extend(range(time, end, delta) if delta
                         else itertools.repeat(time, samples))
            durations += array.array('I', [delta]) * samples
            time = end
        if len(times) < count:
            raise MP4Error(f'the decoding time table covers {len(times)} samples of {count}')
        return times, durations

    @staticmethod
    def _read_composition(table: dict[str, bytes], count: int) -> array.array[int]:
        """Each sample's composition offset (ctts), signed, or zero without one."""
        offsets: array.array[int] = array.array('i')
        if 'ctts' in table:
            _, cursor = _full_box(table, 'ctts')
            runs = cursor.table('I', 2 * cursor.count(8))
            for stated, offset in zip(runs[0::2], runs[1::2], strict=True):
                # Signed in either version: version 0 files from some muxers
                # carry negative offsets, and no positive one reaches 2**31.
                signed = offset - (1 << 32) if offset >> 31 else offset
                offsets += array.array('i', [signed]) * min(stated, count - len(offsets))
        offsets += array.array('i', [0]) * (count - len(offsets))
        return offsets

    @staticmethod
    def _read_sync(table: dict[str, bytes], count: int) -> Sequence[int]:
        """The indices of the sync samples, ascending; every sample without ``stss``."""
        if 'stss' not in table:
            return range(count)
        _, cursor = _full_box(table, 'stss')
        numbers = cursor.table('I', cursor.count(4))
        for number in numbers:
            if not 1 <= number <= count:
                raise MP4Error(f'the stss box names sample {number} as a sync '
                               f'sample, in a track of {count}: samples count from 1')
        return sorted({number - 1 for number in numbers})

class Track:
    """One track of an MP4 file: its sample table, and how to read a sample.

    track_id -- the ``tkhd`` track identifier
    handler -- ``'vide'`` or ``'soun'``, or another handler type
    codec -- the sample entry's four-character code, such as ``'avc1'``
    timescale -- units of a second the track's times are counted in
    duration -- the track's length in those units
    sync_samples -- the indices, in decoding order, of the samples decoding
        can start at

    The sample tables are read the first time the track's samples, their
    count or their times are asked for, and cost about 30 bytes a sample from
    then on. A track nobody reads, such as an hour of PCM audio beside the
    video, costs nothing. :class:`MP4Error` for a table that states more than
    its box or the file holds is raised then.
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
        self._boxes = table

    @functools.cached_property
    def _table(self) -> SampleTable:
        return SampleTable(self._boxes, self._reader.size)

    @property
    def sync_samples(self) -> list[int]:
        return list(self._table.sync)

    def __len__(self) -> int:
        return len(self._table.sizes)

    def __repr__(self) -> str:
        return f'<{type(self).__name__} {self.track_id} {self.codec}>'

    # ------------------------------------------------------------ samples

    def presentation_time(self, index: int) -> int:
        """When sample `index` is shown, with the edit list applied."""
        table = self._table
        return table.decode_times[index] + table.composition[index] - self._edit_offset

    def display_order(self) -> list[int]:
        """Sample indices in the order their presentation times put them."""
        return sorted(range(len(self)), key=self.presentation_time)

    def keyframe_before(self, index: int) -> int:
        """The last sync sample at or before `index`, in decoding order.

        The first sample where none is.
        """
        sync = self._table.sync
        found = bisect.bisect_right(sync, index)
        return sync[found - 1] if found else 0

    def read(self, index: int) -> bytes:
        """Sample `index`'s bytes, as the file stores them.

        Raises :class:`MP4Error` for a sample the file does not hold whole.
        """
        if not 0 <= index < len(self):
            raise IndexError(f'sample {index} of a track of {len(self)}')
        return self._reader.read_at(self._table.offsets[index], self._table.sizes[index],
                                    f'sample {index}')

    def sample(self, index: int) -> Sample:
        """Sample `index`, with its presentation time and key flag."""
        return Sample(data=self.read(index), timestamp=self.presentation_time(index),
                      duration=self._table.durations[index],
                      keyframe=self._is_sync(index), index=index)

    def _is_sync(self, index: int) -> bool:
        sync = self._table.sync
        found = bisect.bisect_left(sync, index)
        return found < len(sync) and sync[found] == index


class VideoTrack(Track):
    """A video track in a codec this package decodes.

    H.264 (``avc1``, ``avc3``), HEVC (``hvc1``, ``hev1``), AV1 (``av01``) or
    VP9 (``vp09``). :func:`pyopengl_video.codecs.canonical` turns
    :attr:`~Track.codec` into the name a decoder is opened by.

    width/height -- the picture size the sample entry states, which a file
        may state as anything from 0 to 65535 whatever its stream holds; a
        decoder reports the stream's own
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
        cursor = Cursor(entry, f'the {self.codec} sample entry')
        cursor.skip(24)
        self.width, self.height = cursor.u16(), cursor.u16()
        cursor.skip(50)
        self.vps, self.sps, self.pps = [], [], []
        self._kind = codecs.of_sample_entry(self.codec)
        children = _children(cursor.rest())
        if 'avcC' in children:
            self._read_avcc(Cursor(children['avcC'], 'the avcC record'))
        elif 'hvcC' in children:
            self._read_hvcc(Cursor(children['hvcC'], 'the hvcC record'))
        elif 'av1C' in children:
            record = Cursor(children['av1C'], 'the av1C record')
            record.skip(AV1C_OBUS)
            self.configuration = record.rest()

    @property
    def _nal_units(self) -> bool:
        """Whether a sample is length-prefixed NAL units, as H.264 and HEVC are."""
        return self._kind in (codecs.H264, codecs.HEVC)

    def _read_avcc(self, record: Cursor) -> None:
        """The SPS and PPS of an AVCDecoderConfigurationRecord."""
        record.skip(4)
        self.length_size = (record.u8() & 3) + 1
        for _ in range(record.u8() & 0x1F):
            self.sps.append(record.take(record.u16()))
        for _ in range(record.u8()):
            self.pps.append(record.take(record.u16()))

    def _read_hvcc(self, record: Cursor) -> None:
        """The VPS, SPS and PPS arrays of an HEVCDecoderConfigurationRecord."""
        record.skip(HVCC_LENGTH_SIZE)
        self.length_size = (record.u8() & 3) + 1
        kinds = {HEVC_VPS: self.vps, HEVC_SPS: self.sps, HEVC_PPS: self.pps}
        for _ in range(record.u8()):
            kind = record.u8() & 0x3F
            for _ in range(record.u16()):
                unit = record.take(record.u16())
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

        Raises :class:`MP4Error` for a sample the file does not hold whole,
        and for one whose NAL unit lengths do not add up to it.
        """
        found = super().sample(index)
        prefix = self.parameter_sets() if parameter_sets else b''
        if not self._nal_units:
            return dataclasses.replace(found, data=prefix + found.data) if prefix else found
        stream = bytearray(prefix)
        stored = Cursor(found.data, f'sample {index}')
        while stored.remaining:
            length = int.from_bytes(stored.take(self.length_size), 'big')
            stream += START_CODE
            stream += stored.take(length)
        return dataclasses.replace(found, data=bytes(stream))


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
        cursor = Cursor(entry, f'the {self.codec} sample entry')
        cursor.skip(8)
        version = cursor.u16()
        cursor.skip(6)
        self.channels = cursor.u16()
        cursor.skip(6)
        self.sample_rate = cursor.u32() >> 16
        if version == 1:
            cursor.skip(16)
        elif version == 2:
            cursor.skip(4)
            rate = cursor.f64()
            if not (math.isfinite(rate) and 0 <= rate <= bounds.FIELD_32):
                raise MP4Error(f'the {self.codec} sample entry states a sample '
                               f'rate of {rate}')
            self.sample_rate = round(rate)
            self.channels = cursor.u32()
            cursor.skip(20)
        children = _children(cursor.rest())
        if 'esds' in children:
            self._read_esds(children)
        for kind in ('dOps', 'dfLa'):
            if kind in children:
                self.config = children[kind]

    def _read_esds(self, children: dict[str, bytes]) -> None:
        """The ES descriptor's decoder configuration (ISO/IEC 14496-1, 7.2.6)."""
        _, cursor = _full_box(children, 'esds')
        tag, stream = _descriptor(cursor)
        if tag != 0x03:
            return
        stream.skip(2)                                      # ES_ID
        flags = stream.u8()
        if flags & 0x80:
            stream.skip(2)                                  # dependsOn_ES_ID
        if flags & 0x40:
            stream.skip(stream.u8())                        # URL
        if flags & 0x20:
            stream.skip(2)                                  # OCR_ES_Id
        while stream.remaining:
            tag, decoder = _descriptor(stream)
            if tag != 0x04:
                continue
            self.object_type = decoder.u8()
            decoder.skip(12)                                # stream type, buffer, bit rates
            while decoder.remaining:
                tag, specific = _descriptor(decoder)
                if tag == 0x05:
                    self.config = specific.rest()


class MP4Reader:
    """The tracks of an MP4 file, read by index without reading the media.

    source -- a path, or a binary file object open for reading. A file object
        stays the caller's to close.

    size -- the file's length in bytes when it was opened

    Opening reads the ``moov`` box and nothing of ``mdat``; a sample is read
    when it is asked for. The reader needs no GL, so it can be opened on any
    thread, and samples can be read from several threads at once. Fragmented
    MP4 (``moof``) is not read.

    A video track in H.264, HEVC, AV1 or VP9 is a :class:`VideoTrack`; one in
    another codec is listed in :attr:`tracks` as a plain :class:`Track`.

    Raises :class:`MP4Error` for a file that is not MP4, and for one whose
    boxes state more than the file holds. A track's sample tables are read
    when its samples are first asked for, and raise it then. No count a file
    states is allocated for before the bytes behind it are seen to be there.
    """

    def __init__(self, source: str | Path | BinaryIO) -> None:
        self._source = container.FileBytes(source, MP4Error)
        self.path = self._source.path
        self.size = self._source.size
        self.tracks: list[Track] = []
        try:
            self._read_movie()
        except BaseException:
            self.close()
            raise

    def __repr__(self) -> str:
        return f'<{type(self).__name__} {self._source!r} {self.tracks}>'

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
        return max((track.duration / track.timescale for track in self.tracks),
                   default=0.0)

    def read_at(self, offset: int, size: int, what: str = 'a sample') -> bytes:
        """`size` bytes from `offset` in the file.

        what -- what the bytes are, for the message

        Raises :class:`MP4Error` for a span that is not inside the file.
        """
        return self._source.read(offset, size, what)

    def close(self) -> None:
        """Close the file, if this reader opened it."""
        self._source.close()

    def __enter__(self) -> MP4Reader:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()

    # ------------------------------------------------------------ the index

    def _top_level(self) -> Iterator[tuple[str, int, int]]:
        """Each top-level box as ``(type, payload offset, payload size)``."""
        end = self.size
        position = 0
        while position + 8 <= end:
            header = Cursor(self._source.read(position, min(16, end - position)),
                            f'the box at byte {position}')
            size, kind = header.u32(), header.take(4)
            length = 8
            if size == 1:
                size = header.u64()
                length = 16
            elif size == 0:
                size = end - position
            name = kind.decode('latin-1')
            if position == 0 and name not in ('ftyp', 'moov', 'mdat', 'free',
                                              'skip', 'wide', 'pnot'):
                raise MP4Error(
                    f'the file starts with {kind!r}, which is not an MP4 box')
            if size < length or position + size > end:
                raise MP4Error(
                    f'the {name} box at byte {position} claims {size} bytes, in '
                    f'a file of {end}')
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
                movie = self.read_at(offset, size, 'the moov box')
                break
            if kind == 'moof':
                raise MP4Error('fragmented MP4 (moof) is not read here')
        else:
            raise MP4Error('the file has no movie box (moov), so it has no index')
        boxes = list(iter_boxes(movie))
        movie_timescale = 1000
        if any(kind == 'mvhd' for kind, _ in boxes):
            version, header = _full_box(dict(boxes), 'mvhd')
            header.skip(16 if version == 1 else 8)
            movie_timescale = header.u32()
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
        version, header = _full_box(trak, 'tkhd')
        header.skip(16 if version == 1 else 8)
        track_id = header.u32()
        version, header = _full_box(mdia, 'mdhd')
        header.skip(16 if version == 1 else 8)
        timescale = header.u32()
        duration = header.u64() if version == 1 else header.u32()
        if not timescale:
            raise MP4Error(f'track {track_id} states a timescale of 0')
        handler = ''
        if 'hdlr' in mdia:
            _, header = _full_box(mdia, 'hdlr')
            header.skip(4)
            handler = header.take(4).decode('latin-1')
        _, entries = _full_box(stbl, 'stsd')
        entries.skip(4)                                     # the entry count
        kind, entry = next(iter_boxes(entries.rest()), ('', b''))
        if not kind:
            raise MP4Error(f'the stsd box of track {track_id} holds no sample entry')
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
        boxes = _children(edts)
        if 'elst' not in boxes:
            return 0
        version, cursor = _full_box(boxes, 'elst')
        delay = 0
        for _ in range(cursor.count(20 if version == 1 else 12)):
            if version == 1:
                duration, media_time = cursor.u64(), cursor.i64()
            else:
                duration, media_time = cursor.u32(), cursor.i32()
            cursor.skip(4)                                  # the media rate
            if media_time == -1:
                delay += duration * timescale // max(movie_timescale, 1)
                continue
            return media_time - delay
        return -delay
