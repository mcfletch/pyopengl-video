"""Holding a value to its range before it reaches a fixed-width field.

ctypes stores an integer that does not fit a field by dropping its high bits,
and raises nothing: 65552 written to a 16-bit width reads back as 16. A length
or an offset a driver reports is likewise only a number until it is compared
with the size of the buffer it describes. The functions here make those
comparisons and name the value in the error.

Each takes the error class to raise, so an encoder refuses with an
:class:`~pyopengl_video.encoder.EncoderError` and a decoder with a
:class:`~pyopengl_video.decoder.DecoderError`.
"""
from __future__ import annotations

from pyopengl_video.backend import VideoError

__all__ = ['FIELD_16', 'FIELD_32', 'frame_rate', 'frame_size', 'span', 'whole']

#: The largest value an unsigned field of 16 and of 32 bits holds.
FIELD_16 = (1 << 16) - 1
FIELD_32 = (1 << 32) - 1


def whole(value: object, what: str, low: int, high: int,
          error: type[VideoError] = VideoError) -> int:
    """`value` as an int, where it is a whole number from `low` to `high`.

    what -- the value's name, for the message

    Raises `error` for anything else: a fraction, a boolean, a string, or a
    number outside the range.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f'{what} is {value!r}, which is not a whole number')
    if value != value or value in (float('inf'), float('-inf')) or value != int(value):
        raise error(f'{what} is {value!r}, which is not a whole number')
    number = int(value)
    if not low <= number <= high:
        raise error(f'{what} is {number}, outside {low} to {high}')
    return number


def frame_size(width: object, height: object, *, most: tuple[int, int],
               even: bool = False,
               error: type[VideoError] = VideoError) -> tuple[int, int]:
    """A picture size as (width, height), each from 1 to its limit in `most`.

    even -- refuse an odd width or height, as a 4:2:0 surface needs neither
    """
    described = f'a {width}x{height} picture'
    try:
        size = (whole(width, 'the width', 1, most[0], error),
                whole(height, 'the height', 1, most[1], error))
    except error as refused:
        raise error(f'{described} cannot be made: {refused}') from None
    if even and (size[0] % 2 or size[1] % 2):
        raise error(f'{described} cannot be made: the width and the height '
                    'must both be even')
    return size


def frame_rate(rate: tuple[object, object],
               error: type[VideoError] = VideoError) -> tuple[int, int]:
    """A frame rate as (numerator, denominator), both from 1 to 2**32 - 1."""
    try:
        return (whole(rate[0], 'the numerator', 1, FIELD_32, error),
                whole(rate[1], 'the denominator', 1, FIELD_32, error))
    except error as refused:
        raise error(f'a frame rate of {rate[0]}/{rate[1]} cannot be coded: '
                    f'{refused}') from None


def span(offset: int, length: int, limit: int, what: str,
         error: type[VideoError] = VideoError) -> tuple[int, int]:
    """`offset` and `length`, where they lie inside a buffer of `limit` bytes.

    what -- what the bytes are, for the message

    Raises `error` when the span starts before the buffer, ends after it, or
    has a negative length.
    """
    offset, length = int(offset), int(length)
    if offset < 0 or length < 0 or offset + length > limit:
        raise error(f'{what} is stated as {length} bytes at offset {offset}, '
                    f'which is not inside the {limit} bytes there are')
    return offset, length
