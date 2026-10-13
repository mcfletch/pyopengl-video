"""AV1's framing: a temporal unit as a run of open bitstream units.

An AV1 stream is a sequence of OBUs, each a one or two byte header, usually a
size, and a payload (AV1 specification, section 5.3). A *temporal unit* is the
OBUs of one instant: what an MP4 sample, an IVF frame and a WebM block hold,
and what the decoders and encoders here pass as a sample or a packet.

:func:`split` walks one. Containers use it to find the sequence header a
decoder is configured with and to drop the units a container does not store.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator

from pyopengl_video import container
from pyopengl_video.vaapi.bits import leb128, read_leb128
from pyopengl_video.vaapi.h264 import StreamError

__all__ = ['FRAME', 'FRAME_HEADER', 'METADATA', 'PADDING', 'REDUNDANT_FRAME_HEADER',
           'SEQUENCE_HEADER', 'TEMPORAL_DELIMITER', 'TILE_GROUP', 'TILE_LIST', 'Obu',
           'join', 'sequence_header', 'split', 'without']

SEQUENCE_HEADER = 1
TEMPORAL_DELIMITER = 2
FRAME_HEADER = 3
TILE_GROUP = 4
METADATA = 5
FRAME = 6
REDUNDANT_FRAME_HEADER = 7
TILE_LIST = 8
PADDING = 15


class Cursor(container.Cursor):
    """The bytes of a temporal unit read in turn; raises :class:`StreamError`."""

    error = StreamError


@dataclasses.dataclass(frozen=True)
class Obu:
    """One open bitstream unit.

    kind -- obu_type, one of the constants of this module
    payload -- what follows the header and the size
    temporal_id/spatial_id -- the layer the unit belongs to, 0 without an
        extension byte
    has_extension -- the header carried the extension byte
    raw -- the unit as it is in the stream, header and size included
    """

    kind: int
    payload: bytes
    temporal_id: int = 0
    spatial_id: int = 0
    has_extension: bool = False
    raw: bytes = b''

    def encoded(self) -> bytes:
        """The unit with a size field, which is the form a container stores."""
        header = bytes([self.kind << 3 | (0x04 if self.has_extension else 0) | 0x02])
        if self.has_extension:
            header += bytes([self.temporal_id << 5 | self.spatial_id << 3])
        return header + leb128(len(self.payload)) + self.payload


def split(data: bytes) -> Iterator[Obu]:
    """Each OBU of `data`, in order.

    A unit with no size field runs to the end of the data, as the last unit of
    a sample may. Raises :class:`~pyopengl_video.vaapi.h264.StreamError` for a
    unit that is cut short or whose header has the forbidden bit set.
    """
    cursor = Cursor(data, 'a temporal unit')
    while cursor.remaining:
        start = cursor.position
        header = cursor.u8()
        if header & 0x80:
            raise StreamError(f'the OBU at byte {start} has its forbidden bit set')
        has_extension = bool(header & 0x04)
        temporal_id = spatial_id = 0
        if has_extension:
            if not cursor.remaining:
                raise StreamError(f'the OBU at byte {start} ends inside its header')
            layers = cursor.u8()
            temporal_id, spatial_id = layers >> 5, (layers >> 3) & 3
        if header & 0x02:
            size, used = read_leb128(data, cursor.position)
            cursor.skip(used)
        else:
            size = cursor.remaining
        if size > cursor.remaining:
            raise StreamError(
                f'the OBU at byte {start} states {size} bytes and {cursor.remaining} remain')
        payload = bytes(cursor.take(size))
        yield Obu(kind=(header >> 3) & 0xF, payload=payload,
                  temporal_id=temporal_id, spatial_id=spatial_id,
                  has_extension=has_extension, raw=bytes(data[start:cursor.position]))


def join(units: Iterable[Obu]) -> bytes:
    """`units` as one run of bytes, each with a size field."""
    return b''.join(unit.encoded() for unit in units)


def without(data: bytes, *kinds: int) -> bytes:
    """`data` less every OBU of `kinds`, the rest each with a size field."""
    return join(unit for unit in split(data) if unit.kind not in kinds)


def sequence_header(data: bytes) -> bytes:
    """The first sequence header OBU in `data`, with a size field, or b''."""
    for unit in split(data):
        if unit.kind == SEQUENCE_HEADER:
            return unit.encoded()
    return b''
