"""Reading and writing bits most-significant first, for the AV1 and VP9 headers.

Both codecs write their headers as fixed-width fields with no escape bytes and
no stop bit, which is what :class:`BitReader` and :class:`BitWriter` here read
and write. H.264 and HEVC payloads carry emulation prevention and a trailing
stop bit; :mod:`pyopengl_video.vaapi.h264` has the reader and writer for those.
"""
from __future__ import annotations

from pyopengl_video.vaapi.h264 import StreamError

__all__ = ['BitReader', 'BitWriter', 'leb128', 'read_leb128']


#: The largest value a base-128 field may hold (leb128(), 4.10.5 of the AV1
#: Bitstream & Decoding Process Specification).
LEB128_MAX = (1 << 32) - 1

#: The most bytes a base-128 field takes.
LEB128_BYTES = 8


class BitReader:
    """Fixed-width fields read from `data`, most significant bit first.

    position -- the next bit to read, counted from the start of `data`; from 0
        to the length of `data` in bits

    A field is read from the bytes it spans, so reading costs what is read
    and not the length of `data`. Reading past the end, a negative count and a
    position outside `data` each raise
    :class:`~pyopengl_video.vaapi.h264.StreamError`.
    """

    def __init__(self, data: bytes, position: int = 0) -> None:
        self.data = data
        self._bits = len(data) * 8
        self._position = 0
        self.position = position

    @property
    def position(self) -> int:
        return self._position

    @position.setter
    def position(self, position: int) -> None:
        if not 0 <= position <= self._bits:
            raise StreamError(
                f'bit {position} is outside the {self._bits} bits of the header')
        self._position = position

    def f(self, count: int) -> int:
        """An unsigned field of `count` bits."""
        if count < 0:
            raise StreamError(f'a field of {count} bits was asked for')
        start = self._position
        end = start + count
        if end > self._bits:
            raise StreamError(
                f'the header ends at bit {self._bits} and a field runs to bit {end}')
        self._position = end
        last = (end + 7) >> 3
        spanned = int.from_bytes(self.data[start >> 3:last], 'big')
        return (spanned >> (last * 8 - end)) & ((1 << count) - 1)

    def flag(self) -> int:
        """One bit."""
        return self.f(1)

    def su(self, count: int) -> int:
        """A two's complement field of `count` bits, which is at least its sign bit."""
        if count < 1:
            raise StreamError(
                f'a signed field of {count} bits was asked for, which has no sign bit')
        value = self.f(count)
        return value - (1 << count) if value >> (count - 1) else value

    def signed_magnitude(self, count: int) -> int:
        """A magnitude of `count` bits followed by its sign bit, as VP9 writes one."""
        value = self.f(count)
        return -value if self.flag() else value

    def leb128(self) -> int:
        """A little-endian base-128 value, which starts on a byte."""
        if self._position & 7:
            raise StreamError(
                f'a size field is read at bit {self._position}, which is not the '
                'start of a byte')
        value, used = read_leb128(self.data, self._position >> 3)
        self._position += used * 8
        return value

    @property
    def byte_position(self) -> int:
        """The byte holding the next bit to read."""
        return self._position // 8

    def byte_align(self) -> None:
        """Skip to the start of the next byte, unless already at one."""
        self._position = (self._position + 7) & ~7

    @property
    def remaining(self) -> int:
        """How many bits are left."""
        return self._bits - self._position


def read_leb128(data: bytes, offset: int = 0) -> tuple[int, int]:
    """The little-endian base-128 value at `offset`, and how many bytes it took.

    A value takes at most eight bytes and is at most 32 bits (4.10.5); one
    that is longer, larger, or runs past the end of `data` is a
    :class:`~pyopengl_video.vaapi.h264.StreamError`.
    """
    if offset < 0:
        raise StreamError(f'a size field is read at offset {offset}, before the data')
    value = 0
    for index in range(LEB128_BYTES):
        if offset + index >= len(data):
            raise StreamError('a size field runs past the end of the data')
        byte = data[offset + index]
        value |= (byte & 0x7F) << (index * 7)
        if not byte & 0x80:
            if value > LEB128_MAX:
                raise StreamError(
                    f'a size field holds {value}, which is more than the 32 bits '
                    'one may hold')
            return value, index + 1
    raise StreamError('a size field is longer than the eight bytes one may take')


def leb128(value: int, size: int | None = None) -> bytes:
    """`value` as little-endian base-128, in `size` bytes when one is given.

    A fixed size pads with continuation bytes, which lets a length be written
    before what it measures is known.
    """
    if value < 0:
        raise ValueError(f'{value} is negative, and a base-128 value is unsigned')
    groups = []
    while True:
        groups.append(value & 0x7F)
        value >>= 7
        if not value:
            break
    if size is not None:
        if len(groups) > size:
            raise ValueError(f'the value needs {len(groups)} bytes, not {size}')
        groups += [0] * (size - len(groups))
    return bytes(group | 0x80 for group in groups[:-1]) + bytes(groups[-1:])


class BitWriter:
    """Fixed-width fields written most significant bit first."""

    def __init__(self) -> None:
        self._value = 0
        self._bits = 0

    def __len__(self) -> int:
        """How many bits have been written."""
        return self._bits

    def f(self, value: int, count: int) -> None:
        """Write `value` as an unsigned field of `count` bits."""
        if not 0 <= value < (1 << count):
            raise ValueError(f'{value} does not fit a field of {count} bits')
        self._value = (self._value << count) | value
        self._bits += count

    def flag(self, value: object) -> None:
        """Write one bit."""
        self.f(1 if value else 0, 1)

    def su(self, value: int, count: int) -> None:
        """Write `value` as a two's complement field of `count` bits."""
        if count < 1 or not -(1 << (count - 1)) <= value < (1 << (count - 1)):
            raise ValueError(f'{value} does not fit a signed field of {count} bits')
        self.f(value & ((1 << count) - 1), count)

    def byte_align(self) -> None:
        """Pad with zero bits to the start of the next byte."""
        self.f(0, -self._bits % 8)

    def bytes(self) -> bytes:
        """What has been written, zero bits filling the last byte."""
        padding = -self._bits % 8
        return ((self._value << padding).to_bytes((self._bits + padding) // 8, 'big')
                if self._bits else b'')
