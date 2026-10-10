"""Reading a raw H.264 elementary stream: the Annex-B ``.h264`` file.

A raw stream has no index and no timestamps. Opening one scans it once:
start codes divide it into NAL units, the slice headers divide those into
access units (one picture each), and the decoded picture buffer, run over the
headers alone, gives the display order. A picture's timestamp is its place in
that order at the stream's frame rate, which comes from the sequence parameter
set where it states one.

Samples come out as the file's own bytes, already Annex-B, so
:class:`AnnexBReader` and :class:`~pyopengl_video.mp4.MP4Reader` hand a decoder
the same thing.
"""
from __future__ import annotations

import mmap
from pathlib import Path
from typing import Any, BinaryIO

from pyopengl_video.decoder import Sample
from pyopengl_video.vaapi.h264 import StreamError
from pyopengl_video.vaapi.h264_dpb import DecodedPictureBuffer
from pyopengl_video.vaapi.h264_parse import (
    NAL_IDR,
    NAL_PPS,
    NAL_SEI,
    NAL_SLICE,
    NAL_SPS,
    SLICE_I,
    SLICE_SI,
    PictureParameterSet,
    SequenceParameterSet,
    SliceHeader,
    parse_nal,
    parse_pps,
    parse_slice_header,
    parse_sps,
    starts_new_picture,
)

__all__ = ['AnnexBReader', 'DEFAULT_FRAME_RATE']

#: Frames per second for a stream that states no rate and a caller that names
#: none: the PAL rate, which is what players assume of a raw stream.
DEFAULT_FRAME_RATE = (25, 1)

#: NAL unit types that begin an access unit when they follow a picture
#: (7.4.1.2.3): SEI, the parameter sets, the delimiter, and types 14 to 18.
ACCESS_UNIT_STARTERS = frozenset({NAL_SEI, NAL_SPS, NAL_PPS, 9, 14, 15, 16, 17, 18})

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


def has_recovery_point(rbsp: bytes) -> bool:
    """Whether an SEI RBSP carries a recovery point message."""
    position = 0
    while position < len(rbsp) and rbsp[position] != 0x80:
        kind = size = 0
        while rbsp[position] == 0xFF:
            kind += 255
            position += 1
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
    """One picture's span of the file, while the index is built."""

    def __init__(self, start: int, parameter_sets: tuple[bytes, ...]) -> None:
        self.start = start
        self.end = start
        self.parameter_sets = parameter_sets
        self.first: SliceHeader | None = None
        self.last: SliceHeader | None = None
        self.recovery_point = False
        self.intra = True


class AnnexBReader:
    """The access units of a raw H.264 stream, ordered and timed.

    source -- a path, or a binary file object; a file object is read whole
        and stays the caller's to close
    frame_rate -- ``(numerator, denominator)`` frames per second, when the
        stream states none or states the wrong one; otherwise the SPS's rate,
        or :data:`DEFAULT_FRAME_RATE`

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
        self.sequences: dict[int, SequenceParameterSet] = {}
        self.pictures: dict[int, PictureParameterSet] = {}
        self._units: list[_AccessUnit] = []
        try:
            self._index()
            first_sps = self._units[0].first.sps if self._units[0].first else None
            stated = first_sps.frame_rate if first_sps is not None else None
            self.frame_rate = frame_rate or stated or DEFAULT_FRAME_RATE
            self.size = first_sps.size if first_sps is not None else (0, 0)
            self._order = self._display_order()
        except Exception:
            self.close()
            raise
        self.sync_samples = [index for index, unit in enumerate(self._units)
                             if self._is_key(unit)]
        numerator, denominator = self.frame_rate
        self._frame_duration = round(self.timescale * denominator / numerator)
        self._timestamps = [0] * len(self._units)
        for rank, index in enumerate(self._order):
            self._timestamps[index] = rank * self._frame_duration

    def __len__(self) -> int:
        return len(self._units)

    def __repr__(self) -> str:
        width, height = self.size
        return (f'<{type(self).__name__} {self.path or "stream"} {width}x{height}, '
                f'{len(self)} pictures>')

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

        parameter_sets -- put every SPS and PPS seen up to this sample in
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
                      keyframe=self._is_key(unit), index=index)

    def presentation_time(self, index: int) -> int:
        """When picture `index` is shown, in :attr:`timescale` units."""
        return self._timestamps[index]

    def parameter_sets(self) -> bytes:
        """The SPS and PPS the first picture decodes with, as Annex-B."""
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

    @staticmethod
    def _is_key(unit: _AccessUnit) -> bool:
        """IDR, a recovery point, or a picture of I slices: decoding can start here.

        At a non-IDR start, the pictures that follow in decoding order and
        precede in display order may refer to earlier ones, and a player
        drops them after a seek.
        """
        first = unit.first
        return first is not None and (first.idr or unit.recovery_point or unit.intra)

    # -------------------------------------------------------------- index

    def _index(self) -> None:
        """Find the access units, keeping the parameter sets in force at each."""
        raw_sets: dict[tuple[int, int], bytes] = {}
        current = _AccessUnit(0, ())
        for start, payload, end in nal_spans(self._data):
            kind = self._data[payload] & 0x1F
            if kind in ACCESS_UNIT_STARTERS and current.first is not None:
                current = self._close(current, start, tuple(raw_sets.values()))
            if kind == NAL_SPS:
                unit = parse_nal(bytes(self._data[payload:end]))
                sps = parse_sps(unit.rbsp)
                self.sequences[sps.seq_parameter_set_id] = sps
                raw_sets[(NAL_SPS, sps.seq_parameter_set_id)] = unit.data
            elif kind == NAL_PPS:
                unit = parse_nal(bytes(self._data[payload:end]))
                pps = parse_pps(unit.rbsp, self.sequences)
                self.pictures[pps.pic_parameter_set_id] = pps
                raw_sets[(NAL_PPS, pps.pic_parameter_set_id)] = unit.data
            elif kind == NAL_SEI:
                unit = parse_nal(bytes(self._data[payload:end]))
                current.recovery_point |= has_recovery_point(unit.rbsp)
            elif kind in (NAL_SLICE, NAL_IDR):
                limit = min(end, payload + SLICE_HEADER_BYTES)
                header = parse_slice_header(parse_nal(bytes(self._data[payload:limit])),
                                            self.sequences, self.pictures)
                if header.redundant_pic_cnt:
                    continue
                if current.last is not None and starts_new_picture(current.last, header):
                    current = self._close(current, start, tuple(raw_sets.values()))
                if current.first is None:
                    current.first = header
                    current.parameter_sets = tuple(raw_sets.values())
                current.last = header
                current.intra &= header.slice_type in (SLICE_I, SLICE_SI)
            current.end = end
        if current.first is not None:
            self._units.append(current)
        if not self._units:
            raise StreamError('the stream holds no pictures')

    def _close(self, unit: _AccessUnit, start: int,
               parameter_sets: tuple[bytes, ...]) -> _AccessUnit:
        """End `unit` where the next one starts, and begin that one."""
        unit.end = start
        self._units.append(unit)
        return _AccessUnit(start, parameter_sets)

    def _display_order(self) -> list[int]:
        """The order the decoded picture buffer outputs the pictures in."""
        buffer = DecodedPictureBuffer()
        order: list[int] = []
        for index, unit in enumerate(self._units):
            assert unit.first is not None
            picture = buffer.begin(unit.first)
            order += [frame.surface for frame in buffer.finish(picture, surface=index)]
        order += [frame.surface for frame in buffer.flush()]
        # A picture an IDR discarded unseen still has a place in the stream.
        shown = set(order)
        return order + [index for index in range(len(self._units)) if index not in shown]
