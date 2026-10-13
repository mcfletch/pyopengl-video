"""What the container readers share: their error, a cursor, and positioned reads.

A media file is written by somebody else, and every length, count and offset
in it is whatever its writer chose. The readers hold each one to the bytes it
describes before using it:

- :class:`Cursor` reads fields from bytes already in memory. A field or a
  length the bytes cannot satisfy raises the container's error before anything
  is read or allocated.
- :class:`FileBytes` reads a span of a file. A span the file does not hold
  raises before anything is allocated for it, and no read moves a position
  another thread is using.

:class:`ContainerError` is the base of :class:`~pyopengl_video.mp4.MP4Error`
and :class:`~pyopengl_video.ivf.IVFError`.
"""
from __future__ import annotations

import array
import os
import struct
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, Literal

from pyopengl_video.backend import VideoError

__all__ = ['ContainerError', 'Cursor', 'FileBytes']

_DOUBLE = {'big': struct.Struct('>d'), 'little': struct.Struct('<d')}

#: Whether the platform reads at an offset in one call. Windows does not.
#: ``os.pread`` where the platform has it, which Windows does not.
_PREAD: Callable[[int, int, int], bytes] | None = getattr(os, 'pread', None)
_HAS_PREAD = _PREAD is not None


class ContainerError(VideoError, ValueError):
    """A file is not a container this package reads, or breaks its syntax.

    Also raised by a writer for a value its container has no field wide
    enough for. It is a :class:`ValueError` as well as a
    :class:`~pyopengl_video.backend.VideoError`.
    """


class Cursor:
    """Fields read in turn from `data`, each one checked against what is left.

    data -- the bytes to read
    what -- what the bytes are, for the message: ``'an stsz box'``
    position -- where the next field starts
    end -- where the readable bytes stop; the end of `data` by default

    A container subclasses this to name its error and its byte order.
    """

    #: what a read past the end raises
    error: type[Exception] = ContainerError
    #: the order multi-byte fields are stored in
    byteorder: Literal['big', 'little'] = 'big'

    def __init__(self, data: bytes, what: str, position: int = 0,
                 end: int | None = None) -> None:
        self.data = data
        self.what = what
        self.end = len(data) if end is None else min(end, len(data))
        self.position = min(position, self.end)

    @property
    def remaining(self) -> int:
        """How many bytes are left to read."""
        return self.end - self.position

    def _advance(self, count: int) -> int:
        """Move past `count` bytes and return where they start."""
        if not 0 <= count <= self.remaining:
            raise self.error(
                f'{self.what} states {count} bytes at byte {self.position}, '
                f'where {self.remaining} remain')
        start = self.position
        self.position = start + count
        return start

    def take(self, count: int) -> bytes:
        """The next `count` bytes, which are all there or the read is refused."""
        start = self._advance(count)
        return self.data[start:start + count]

    def skip(self, count: int) -> None:
        """Pass over `count` bytes."""
        self._advance(count)

    def rest(self) -> bytes:
        """Everything not yet read."""
        return self.take(self.remaining)

    def _whole(self, width: int, signed: bool = False) -> int:
        return int.from_bytes(self.take(width), self.byteorder, signed=signed)

    def u8(self) -> int:
        return self._whole(1)

    def u16(self) -> int:
        return self._whole(2)

    def u32(self) -> int:
        return self._whole(4)

    def u64(self) -> int:
        return self._whole(8)

    def i32(self) -> int:
        return self._whole(4, signed=True)

    def i64(self) -> int:
        return self._whole(8, signed=True)

    def f64(self) -> float:
        """An IEEE double, which may be a NaN or an infinity."""
        value: float = _DOUBLE[self.byteorder].unpack(self.take(8))[0]
        return value

    def count(self, entry_size: int) -> int:
        """A 32-bit count of entries, each `entry_size` bytes, that all follow.

        A count of more entries than the remaining bytes hold is refused, so
        the count can size a loop or an allocation.
        """
        stated = self.u32()
        if stated > self.remaining // entry_size:
            raise self.error(
                f'{self.what} states {stated} entries of {entry_size} bytes, '
                f'where {self.remaining} bytes remain')
        return stated

    def table(self, code: str, count: int) -> array.array[int]:
        """`count` unsigned values as an array of type `code`: B, H, I or Q.

        The values take one, two, four or eight bytes each in the file. The
        bytes are checked to be there before the array is made.
        """
        values: array.array[int] = array.array(code)
        values.frombytes(self.take(count * values.itemsize))
        if sys.byteorder != self.byteorder:
            values.byteswap()
        return values


class FileBytes:
    """Positioned reads from a file that several callers may share.

    source -- a path, which is opened here and closed by :meth:`close`, or a
        binary file object, which stays the caller's to close
    error -- what a read the file cannot satisfy raises

    start -- where the file object stood when it was given; 0 for a path
    size -- the file's length when it was opened

    A file opened by path is read with ``os.pread`` where the platform has it,
    which takes its offset as an argument. A file object, and any file on a
    platform without ``pread``, is read by ``seek`` and ``read`` under a lock.
    Either way two threads reading through one object each get their own bytes.
    """

    def __init__(self, source: str | Path | BinaryIO, error: type[Exception]) -> None:
        self.error = error
        self._lock = threading.Lock()
        self._descriptor: int | None = None
        self.path: Path | None = None
        self.start = 0
        if isinstance(source, (str, Path)):
            self.path = Path(source)
            self._file: BinaryIO | None = self.path.open('rb')  # noqa: OGC121 read as samples are asked for, which is the readers' purpose
            self._owned = True
        else:
            self._file = source
            self._owned = False
        try:
            if self._owned:
                self.size = os.fstat(self._file.fileno()).st_size
                if _HAS_PREAD:
                    self._descriptor = self._file.fileno()
            else:
                self.start = self._file.tell()
                self.size = self._file.seek(0, os.SEEK_END)
                self._file.seek(self.start)
        except BaseException:
            self.close()
            raise

    def __repr__(self) -> str:
        return f'<{type(self).__name__} {self.path or self._file!r}, {self.size} bytes>'

    def read(self, offset: int, count: int, what: str = 'the data') -> bytes:
        """`count` bytes from `offset`, all of them or the read is refused.

        what -- what the bytes are, for the message

        Raises :attr:`error` for a span that is not inside the file, before
        anything is allocated for it, and for a file that has been shortened
        since it was opened.
        """
        if offset < 0 or count < 0 or offset + count > self.size:
            raise self.error(
                f'{what} is stated as {count} bytes at offset {offset}, which '
                f'is not inside the {self.size} bytes of the file')
        if self._file is None:
            raise self.error(f'{what} cannot be read: the file is closed')
        if self._descriptor is not None:
            data = self._pread(self._descriptor, offset, count)
        else:
            with self._lock:
                self._file.seek(offset)
                data = self._file.read(count)
        if len(data) != count:
            raise self.error(
                f'the file ends {count - len(data)} bytes short of {what} at '
                f'offset {offset}: it has been shortened since it was opened')
        return data

    @staticmethod
    def _pread(descriptor: int, offset: int, count: int) -> bytes:
        """`count` bytes at `offset`, or fewer where the file ends first."""
        pieces: list[bytes] = []
        while count:
            if _PREAD is None:
                raise OSError('this platform has no positioned read')
            piece = _PREAD(descriptor, count, offset)
            if not piece:
                break
            pieces.append(piece)
            offset += len(piece)
            count -= len(piece)
        return pieces[0] if len(pieces) == 1 else b''.join(pieces)

    def close(self) -> None:
        """Close the file, where it was opened here. Safe to call twice."""
        found, self._file, self._descriptor = self._file, None, None
        if found is not None and self._owned:
            found.close()

    def __enter__(self) -> FileBytes:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()
