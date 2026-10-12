"""Reading and writing bits most-significant first, for the AV1 and VP9 headers.

Both codecs write their headers as fixed-width fields with no escape bytes and
no stop bit, which is what :class:`BitReader` and :class:`BitWriter` here read
and write. H.264 and HEVC payloads carry emulation prevention and a trailing
stop bit; :mod:`pyopengl_video.vaapi.h264` has the reader and writer for those.
"""
from __future__ import annotations

from pyopengl_video.vaapi.h264 import StreamError

__all__ = ['BitReader', 'BitWriter', 'leb128', 'read_leb128']


class BitReader:
    """Fixed-width fields read from `data`, most significant bit first.

    position -- the next bit to read, counted from the start of `data`

    Reading past the end raises :class:`~pyopengl_video.vaapi.h264.StreamError`.
    """

    def __init__(self, data: bytes, position: int = 0) -> None:
        self.data = data
        self.position = position
        self._value = int.from_bytes(data, 'big')
        self._bits = len(data) * 8

    def f(self, count: int) -> int:
        """An unsigned field of `count` bits."""
        end = self.position + count
        if end > self._bits:
            raise StreamError(
                f'the header ends at bit {self._bits} and a field runs to bit {end}')
        self.position = end
        return (self._value >> (self._bits - end)) & ((1 << count) - 1)

    def flag(self) -> int:
        """One bit."""
        return self.f(1)

    def su(self, count: int) -> int:
        """A two's complement field of `count` bits."""
        value = self.f(count)
        return value - (1 << count) if value >> (count - 1) else value

    def signed_magnitude(self, count: int) -> int:
        """A magnitude of `count` bits followed by its sign bit, as VP9 writes one."""
        value = self.f(count)
        return -value if self.flag() else value

    def leb128(self) -> int:
        """A little-endian base-128 value, which must start on a byte."""
        value, used = read_leb128(self.data, self.position // 8)
        self.position += used * 8
        return value

    @property
    def byte_position(self) -> int:
        """The byte holding the next bit to read."""
        return self.position // 8

    def byte_align(self) -> None:
        """Skip to the start of the next byte, unless already at one."""
        self.position = (self.position + 7) & ~7

    @property
    def remaining(self) -> int:
        """How many bits are left."""
        return self._bits - self.position


def read_leb128(data: bytes, offset: int = 0) -> tuple[int, int]:
    """The little-endian base-128 value at `offset`, and how many bytes it took."""
    value = 0
    for index in range(8):
        if offset + index >= len(data):
            raise StreamError('a size field runs past the end of the data')
        byte = data[offset + index]
        value |= (byte & 0x7F) << (index * 7)
        if not byte & 0x80:
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
        self.f(value & ((1 << count) - 1), count)

    def byte_align(self) -> None:
        """Pad with zero bits to the start of the next byte."""
        self.f(0, -self._bits % 8)

    def bytes(self) -> bytes:
        """What has been written, zero bits filling the last byte."""
        padding = -self._bits % 8
        return ((self._value << padding).to_bytes((self._bits + padding) // 8, 'big')
                if self._bits else b'')
