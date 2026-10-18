"""What every decoder does between the hardware and the caller's frames.

The hardware produces pictures in display order with the timestamp of the sample
each came from, and nothing else: how long a picture is shown is a property of
the sample, and the surfaces a picture lives in are a fixed pool that frames are
lent from. :class:`Presentation` and :class:`Outputs` hold those two decisions
for the backends that copy each picture out of the decoder's own surfaces.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Generic, TypeVar

from pyopengl_video.decoder import Sample

__all__ = ['Outputs', 'Presentation']

Pair = TypeVar('Pair')


class Presentation:
    """Which displayed pictures reach the caller, and how long each is shown.

    The parser carries a sample's timestamp to its picture and nothing else,
    so the durations wait here. It also decodes whatever follows the sample it
    is started at: after a seek to a key frame that is not an IDR, that
    includes leading pictures, which are shown before the key frame and refer
    to pictures from before it that were never decoded. A picture shown before
    the first sample fed is one of those, and is dropped.
    """

    def __init__(self) -> None:
        self._durations: dict[int, int] = {}
        self._first: int | None = None

    def fed(self, sample: Sample) -> None:
        """Note a sample handed to the parser."""
        if self._first is None:
            self._first = sample.timestamp
        self._durations[sample.timestamp] = sample.duration

    def shown(self, timestamp: int) -> int | None:
        """The duration of the picture shown at `timestamp`, or None to drop it.

        Pictures are shown in rising order of timestamp, so a sample still
        waiting with an earlier one produced no picture and is forgotten.
        """
        duration = self._durations.pop(timestamp, 0)
        for earlier in [waiting for waiting in self._durations if waiting < timestamp]:
            del self._durations[earlier]
        if self._first is not None and timestamp < self._first:
            return None
        return duration

    def restart(self) -> None:
        """Forget everything fed: the next sample is where decoding starts."""
        self._durations.clear()
        self._first = None


class Outputs(Generic[Pair]):
    """The texture pairs pictures are copied into, lent out and taken back.

    limit -- how many pairs there may be
    make -- creates one more pair

    A pair is made the first time none is free, so a movie played forward
    holds as many as its caller keeps at once, and only a flush that releases
    a whole decoded picture buffer reaches the limit.
    """

    def __init__(self, limit: int, make: Callable[[], Pair]) -> None:
        self.limit = limit
        self._make = make
        self.made: list[Pair] = []
        self._free: deque[Pair] = deque()

    def take(self) -> Pair | None:
        """A pair nothing holds, or None when all `limit` are lent."""
        if self._free:
            return self._free.popleft()
        if len(self.made) >= self.limit:
            return None
        pair = self._make()
        self.made.append(pair)
        return pair

    def give(self, pair: Pair) -> None:
        """Take back a pair that was lent."""
        self._free.append(pair)
