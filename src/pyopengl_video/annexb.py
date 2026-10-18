"""Reading a raw H.264 or HEVC elementary stream: the Annex-B file.

A raw stream (``.h264``, ``.h265``, ``.hevc``) has no index and no timestamps.
Opening one scans it once: start codes divide it into NAL units, the slice
headers divide those into access units (one picture each), and the decoded
picture buffer, run over the headers alone, gives the display order. A
picture's timestamp is its place in that order at the stream's frame rate,
which comes from the parameter sets where they state one.

Both codecs frame their NAL units the same way, and the codec is read from the
stream's first sequence parameter set: :attr:`AnnexBReader.codec` says which.

Samples come out as the file's own bytes, already Annex-B, so
:class:`AnnexBReader` and :class:`~pyopengl_video.mp4.MP4Reader` hand a decoder
the same thing.
"""
from __future__ import annotations

import mmap
from pathlib import Path
from typing import Any, BinaryIO

from pyopengl_video import codecs
from pyopengl_video.decoder import Sample
from pyopengl_video.vaapi import h264_dpb, h264_parse, hevc_dpb, hevc_parse
from pyopengl_video.vaapi.h264 import StreamError

__all__ = ['AnnexBReader', 'DEFAULT_FRAME_RATE']

#: Frames per second for a stream that states no rate and a caller that names
#: none: the PAL rate, which is what players assume of a raw stream.
DEFAULT_FRAME_RATE = (25, 1)

#: H.264 NAL unit types that begin an access unit when they follow a picture
#: (7.4.1.2.3): SEI, the parameter sets, the delimiter, and types 14 to 18.
ACCESS_UNIT_STARTERS = frozenset({h264_parse.NAL_SEI, h264_parse.NAL_SPS,
                                  h264_parse.NAL_PPS, 9, 14, 15, 16, 17, 18})

#: HEVC NAL unit types that begin an access unit when they follow a picture
#: (7.4.2.4.4): the delimiter, the parameter sets, prefix SEI, and the
#: reserved and unspecified types 41 to 44 and 48 to 55.
HEVC_ACCESS_UNIT_STARTERS = frozenset({
    hevc_parse.NAL_AUD, hevc_parse.NAL_VPS, hevc_parse.NAL_SPS, hevc_parse.NAL_PPS,
    hevc_parse.NAL_PREFIX_SEI, *range(41, 45), *range(48, 56)})

#: The first byte of an H.264 sequence parameter set: type 7, at each
#: nal_ref_idc a parameter set may have.
H264_SPS_HEADERS = frozenset({0x27, 0x47, 0x67})
#: The first byte of an HEVC sequence parameter set of the base layer: type 33.
HEVC_SPS_HEADER = 0x42

#: The SEI payloadType of a recovery point (D.1.8).
SEI_RECOVERY_POINT = 6

#: Bytes of a slice NAL unit read to parse its header. No header reaches this:
#: the longest, a weighted prediction table for 32 references in each list,
#: is under a kilobyte.
SLICE_HEADER_BYTES = 4096


def nal_spans(data: Any) -> list[tuple[int, int, int]]:
    """Each NAL unit in `data` as ``(start code offset, payload offset, end)``.

    A four-byte start code is the three-byte one behind a zero byte, and the
    zero is counted with the start code. Trailing zero bytes are left out of
    the unit before them.
    """
    starts = []
    position = data.find(b'\x00\x00\x01')
    while position >= 0:
        starts.append(position)
        position = data.find(b'\x00\x00\x01', position + 3)
    spans = []
    for index, position in enumerate(starts):
        start = position - 1 if position and data[position - 1] == 0 else position
        end = starts[index + 1] if index + 1 < len(starts) else len(data)
        while end > position + 3 and data[end - 1] == 0:
            end -= 1
        if end > position + 3:
            spans.append((start, position + 3, end))
    return spans


def codec_of(data: Any, spans: list[tuple[int, int, int]]) -> str:
    """The codec of a raw stream: that of its first sequence parameter set.

    The two codecs' NAL unit headers differ, and neither's sequence parameter
    set reads as the other's: an H.264 one is a reserved type or a higher
    layer as HEVC, and an HEVC one is a data partition as H.264, which no
    stream begins with. A stream with neither is taken as H.264.
    """
    for _, payload, end in spans:
        first = data[payload]
        if first in H264_SPS_HEADERS:
            return codecs.H264
        if first == HEVC_SPS_HEADER and end - payload > 1 and data[payload + 1] >> 3 == 0:
            return codecs.HEVC
    return codecs.H264


def has_recovery_point(rbsp: bytes) -> bool:
    """Whether an SEI RBSP carries a recovery point message."""
    position = 0
    while position < len(rbsp) and rbsp[position] != 0x80:
        kind = size = 0
        while position < len(rbsp) and rbsp[position] == 0xFF:
            kind += 255
            position += 1
        if position >= len(rbsp):
            return False
        kind += rbsp[position]
        position += 1
        while position < len(rbsp) and rbsp[position] == 0xFF:
            size += 255
            position += 1
        if position >= len(rbsp):
            return False
        size += rbsp[position]
        position += 1 + size
        if kind == SEI_RECOVERY_POINT:
            return True
    return False


class _AccessUnit:
    """One picture's span of the file, while the index is built.

    header -- the picture's first slice header, of its codec's kind
    key -- decoding can start at the picture
    """

    def __init__(self, start: int, parameter_sets: tuple[bytes, ...]) -> None:
        self.start = start
        self.end = start
        self.parameter_sets = parameter_sets
        self.header: Any = None
        self.key = False


class _Scan:
    """One codec's reading of a raw stream: its access units, sized and ordered.

    units -- the access units in decoding order
    size -- the first picture's visible size
    frame_rate -- the rate the stream states, or None
    """

    def __init__(self, data: Any, spans: list[tuple[int, int, int]]) -> None:
        self.units: list[_AccessUnit] = []
        #: The parameter sets in force, each as its NAL unit, by kind and id.
        self._sets: dict[tuple[int, int], bytes] = {}
        self._current = _AccessUnit(0, ())
        for start, payload, end in spans:
            self._unit(data, start, payload, end)
            self._current.end = end
        if self._current.header is not None:
            self.units.append(self._current)
        if not self.units:
            raise StreamError('the stream holds no pictures')
        self.size: tuple[int, int] = self.units[0].header.sps.size
        self.frame_rate: tuple[int, int] | None = self._stated_rate()

    def _unit(self, data: Any, start: int, payload: int, end: int) -> None:
        """Take one NAL unit into the index."""
        raise NotImplementedError

    def _stated_rate(self) -> tuple[int, int] | None:
        raise NotImplementedError

    def display_order(self) -> list[int]:
        """Sample indices in the order the pictures are shown."""
        raise NotImplementedError

    def _close(self, start: int) -> None:
        """End the current access unit where the next one starts, and begin that one."""
        self._current.end = start
        self.units.append(self._current)
        self._current = _AccessUnit(start, tuple(self._sets.values()))

    def _picture(self, header: Any, key: bool) -> None:
        """Note the first slice header of the current access unit's picture."""
        self._current.header = header
        self._current.key = key
        self._current.parameter_sets = tuple(self._sets.values())

    def _ordered(self, order: list[int]) -> list[int]:
        """`order`, then the pictures in it nowhere: one that is never shown
        still has a place in the stream."""
        shown = set(order)
        return order + [index for index in range(len(self.units)) if index not in shown]


class _H264Scan(_Scan):
    """A raw H.264 stream.

    A picture is a key where it is IDR, carries a recovery point, or has only
    I slices: decoding can start there. At a start that is not IDR, the
    pictures that follow in decoding order and precede in display order may
    refer to earlier ones, and a player drops them after a seek.
    """

    def __init__(self, data: Any, spans: list[tuple[int, int, int]]) -> None:
        self._sequences: dict[int, h264_parse.SequenceParameterSet] = {}
        self._pictures: dict[int, h264_parse.PictureParameterSet] = {}
        self._last: h264_parse.SliceHeader | None = None
        self._recovery_point = False
        super().__init__(data, spans)

    def _unit(self, data: Any, start: int, payload: int, end: int) -> None:
        kind = data[payload] & 0x1F
        if kind in ACCESS_UNIT_STARTERS and self._current.header is not None:
            self._next(start)
        if kind == h264_parse.NAL_SPS:
            unit = h264_parse.parse_nal(bytes(data[payload:end]))
            sps = h264_parse.parse_sps(unit.rbsp)
            self._sequences[sps.seq_parameter_set_id] = sps
            self._sets[(kind, sps.seq_parameter_set_id)] = unit.data
        elif kind == h264_parse.NAL_PPS:
            unit = h264_parse.parse_nal(bytes(data[payload:end]))
            pps = h264_parse.parse_pps(unit.rbsp, self._sequences)
            self._pictures[pps.pic_parameter_set_id] = pps
            self._sets[(kind, pps.pic_parameter_set_id)] = unit.data
        elif kind == h264_parse.NAL_SEI:
            unit = h264_parse.parse_nal(bytes(data[payload:end]))
            self._recovery_point |= has_recovery_point(unit.rbsp)
        elif kind in (h264_parse.NAL_SLICE, h264_parse.NAL_IDR):
            limit = min(end, payload + SLICE_HEADER_BYTES)
            header = h264_parse.parse_slice_header(
                h264_parse.parse_nal(bytes(data[payload:limit])),
                self._sequences, self._pictures)
            if header.redundant_pic_cnt:
                return
            if self._last is not None and h264_parse.starts_new_picture(self._last, header):
                self._next(start)
            intra = header.slice_type in (h264_parse.SLICE_I, h264_parse.SLICE_SI)
            if self._current.header is None:
                self._picture(header, header.idr or self._recovery_point or intra)
            else:
                # Every slice after the first has to be intra too, unless the
                # picture is a key by its type or its recovery point.
                self._current.key &= header.idr or self._recovery_point or intra
            self._last = header

    def _next(self, start: int) -> None:
        self._close(start)
        self._last = None
        self._recovery_point = False

    def _stated_rate(self) -> tuple[int, int] | None:
        rate: tuple[int, int] | None = self.units[0].header.sps.frame_rate
        return rate

    def display_order(self) -> list[int]:
        buffer = h264_dpb.DecodedPictureBuffer()
        order: list[int] = []
        for index, unit in enumerate(self.units):
            picture = buffer.begin(unit.header)
            order += [frame.surface for frame in buffer.finish(picture, surface=index)]
        order += [frame.surface for frame in buffer.flush()]
        return self._ordered(order)


class _HEVCScan(_Scan):
    """A raw HEVC stream.

    A picture is a key where it is an intra random access point: IDR, CRA or
    BLA. After a seek to a CRA picture, the decoder drops its leading pictures
    that predict from before it.
    """

    def __init__(self, data: Any, spans: list[tuple[int, int, int]]) -> None:
        self._videos: dict[int, hevc_parse.VideoParameterSet] = {}
        self._sequences: dict[int, hevc_parse.SequenceParameterSet] = {}
        self._pictures: dict[int, hevc_parse.PictureParameterSet] = {}
        super().__init__(data, spans)

    def _unit(self, data: Any, start: int, payload: int, end: int) -> None:
        if end - payload < 2:
            return
        kind = (data[payload] >> 1) & 0x3F
        if data[payload] & 1 or data[payload + 1] >> 3:
            return                              # a layer above the base one
        if kind in HEVC_ACCESS_UNIT_STARTERS and self._current.header is not None:
            self._close(start)
        if kind == hevc_parse.NAL_VPS:
            unit = hevc_parse.parse_nal(bytes(data[payload:end]))
            vps = hevc_parse.parse_vps(unit.rbsp)
            self._videos[vps.vps_video_parameter_set_id] = vps
            self._sets[(kind, vps.vps_video_parameter_set_id)] = unit.data
        elif kind == hevc_parse.NAL_SPS:
            unit = hevc_parse.parse_nal(bytes(data[payload:end]))
            sps = hevc_parse.parse_sps(unit.rbsp)
            self._sequences[sps.sps_seq_parameter_set_id] = sps
            self._sets[(kind, sps.sps_seq_parameter_set_id)] = unit.data
        elif kind == hevc_parse.NAL_PPS:
            unit = hevc_parse.parse_nal(bytes(data[payload:end]))
            pps = hevc_parse.parse_pps(unit.rbsp)
            self._pictures[pps.pps_pic_parameter_set_id] = pps
            self._sets[(kind, pps.pps_pic_parameter_set_id)] = unit.data
        elif (kind < hevc_parse.NAL_FIRST_NON_VCL and end - payload > 2
              and data[payload + 2] & 0x80):
            # first_slice_segment_in_pic_flag: the picture's other slice
            # segments say nothing the index needs.
            if self._current.header is not None:
                self._close(start)
            limit = min(end, payload + SLICE_HEADER_BYTES)
            header = hevc_parse.parse_slice_header(
                hevc_parse.parse_nal(bytes(data[payload:limit])),
                self._sequences, self._pictures)
            self._picture(header, header.irap)

    def _stated_rate(self) -> tuple[int, int] | None:
        sps: hevc_parse.SequenceParameterSet = self.units[0].header.sps
        vps = self._videos.get(sps.sps_video_parameter_set_id)
        return sps.frame_rate or (vps.frame_rate if vps is not None else None)

    def display_order(self) -> list[int]:
        buffer = hevc_dpb.DecodedPictureBuffer()
        order: list[int] = []
        for index, unit in enumerate(self.units):
            picture = buffer.begin(unit.header)
            order += [frame.surface for frame in buffer.outputs()]
            if not picture.skipped:
                order += [frame.surface for frame in buffer.finish(picture, surface=index)]
        order += [frame.surface for frame in buffer.flush()]
        return self._ordered(order)


#: The scan for each codec a raw stream may hold.
SCANS: dict[str, type[_Scan]] = {codecs.H264: _H264Scan, codecs.HEVC: _HEVCScan}


class AnnexBReader:
    """The access units of a raw H.264 or HEVC stream, ordered and timed.

    source -- a path, or a binary file object; a file object is read whole
        and stays the caller's to close
    frame_rate -- ``(numerator, denominator)`` frames per second, when the
        stream states none or states the wrong one; otherwise the rate its
        parameter sets state, or :data:`DEFAULT_FRAME_RATE`

    codec -- ``'h264'`` or ``'hevc'``, read from the stream and not from the
        file's name
    sync_samples -- the samples decoding can start at: for H.264 the IDR
        pictures, recovery points and pictures of I slices, and for HEVC the
        intra random access point pictures

    Opening scans the whole stream. A path is mapped rather than read, so a
    long stream costs its index in memory, not its length.
    """

    #: Units of a second timestamps are counted in, as in an MPEG system stream.
    timescale = 90000

    def __init__(self, source: str | Path | BinaryIO,
                 frame_rate: tuple[int, int] | None = None) -> None:
        self._file: BinaryIO | None = None
        self._map: mmap.mmap | None = None
        self._data: Any
        if isinstance(source, (str, Path)):
            self.path: Path | None = Path(source)
            self._file = self.path.open('rb')  # noqa: OGC121 mapped for reading as samples are asked for
            try:
                self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            except ValueError:
                self._map = None
            self._data = self._map if self._map is not None else b''
        else:
            self.path = None
            self._data = source.read()
        try:
            spans = nal_spans(self._data)
            self.codec = codec_of(self._data, spans)
            scan = SCANS[self.codec](self._data, spans)
            self._order = scan.display_order()
        except Exception:
            self.close()
            raise
        self._units = scan.units
        self.frame_rate = frame_rate or scan.frame_rate or DEFAULT_FRAME_RATE
        self.size = scan.size
        self.sync_samples = [index for index, unit in enumerate(self._units) if unit.key]
        numerator, denominator = self.frame_rate
        self._frame_duration = round(self.timescale * denominator / numerator)
        self._timestamps = [0] * len(self._units)
        for rank, index in enumerate(self._order):
            self._timestamps[index] = rank * self._frame_duration

    def __len__(self) -> int:
        return len(self._units)

    def __repr__(self) -> str:
        width, height = self.size
        return (f'<{type(self).__name__} {self.path or "stream"} {self.codec} '
                f'{width}x{height}, {len(self)} pictures>')

    def close(self) -> None:
        """Release the mapping and the file, where this reader opened them."""
        if self._map is not None:
            self._map.close()
            self._map = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> AnnexBReader:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()

    @property
    def duration(self) -> float:
        """The stream's length in seconds, at its frame rate."""
        return len(self) * self._frame_duration / self.timescale

    # ------------------------------------------------------------ samples

    def sample(self, index: int, parameter_sets: bool = False) -> Sample:
        """Access unit `index`, with its display time.

        parameter_sets -- put every parameter set seen up to this sample in
            front, which a decoder starting here needs
        """
        if not 0 <= index < len(self):
            raise IndexError(f'picture {index} of a stream of {len(self)}')
        unit = self._units[index]
        prefix = b''.join(b'\x00\x00\x00\x01' + entry for entry in unit.parameter_sets)
        return Sample(data=(prefix if parameter_sets else b'') + bytes(
                          self._data[unit.start:unit.end]),
                      timestamp=self._timestamps[index],
                      duration=self._frame_duration,
                      keyframe=unit.key, index=index)

    def presentation_time(self, index: int) -> int:
        """When picture `index` is shown, in :attr:`timescale` units."""
        return self._timestamps[index]

    def parameter_sets(self) -> bytes:
        """The parameter sets the first picture decodes with, as Annex-B:
        SPS and PPS for H.264, and VPS, SPS and PPS for HEVC."""
        return b''.join(b'\x00\x00\x00\x01' + entry
                        for entry in self._units[0].parameter_sets)

    def display_order(self) -> list[int]:
        """Sample indices in the order the pictures are shown."""
        return list(self._order)

    def keyframe_before(self, index: int) -> int:
        """The last sync sample at or before `index`, in decoding order."""
        found = 0
        for sync in self.sync_samples:
            if sync > index:
                break
            found = sync
        return found
