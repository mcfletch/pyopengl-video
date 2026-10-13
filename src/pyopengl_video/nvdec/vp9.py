"""What NVIDIA's parser leaves undone for VP9: showing a frame again.

A VP9 frame may be nothing but ``show_existing_frame``: an instruction to show
the frame a reference slot already holds, which is how an encoder shows a
hidden alternate reference frame once its turn comes. NVIDIA's parser decodes
every coded frame and shows the ones that say so, and calls back with nothing
for a frame shown again.

:class:`Slots` follows the eight reference slots beside the parser. Each coded
frame's header says which slots the frame is stored in
(``refresh_frame_flags``), and the parser's decode callback says which picture
the frame was decoded into, so the picture a slot holds is known when the
stream asks for it to be shown. The decoder
(:mod:`pyopengl_video.nvdec.decoder`) copies that picture out as it does any
other.

The parser also times a VP9 picture by a sample other than its own once a
sample has held more than one frame, so the decoder gives a VP9 picture the
timestamp of the sample being parsed when it is shown.
"""
from __future__ import annotations

import dataclasses

from pyopengl_video.vaapi.vp9_parse import (
    NUM_REF_FRAMES,
    DecodingState,
    parse_uncompressed_header,
    split_superframe,
)

__all__ = ['Frame', 'Slots']


@dataclasses.dataclass(frozen=True)
class Frame:
    """One frame of a sample, as the decoder has to treat it.

    data -- the frame's bytes, which the parser is handed when it is coded
    shows_slot -- the reference slot whose frame is shown again, or None for
        a coded frame
    refreshes -- the reference slots a coded frame is stored in, a bit each
    """

    data: bytes
    shows_slot: int | None = None
    refreshes: int = 0


class Slots:
    """Which decoded picture each of a VP9 stream's reference slots holds."""

    def __init__(self) -> None:
        self._state = DecodingState()
        self._pictures: list[int | None] = [None] * NUM_REF_FRAMES

    def frames(self, data: bytes) -> list[Frame]:
        """The frames of the sample `data`, each read as far as its header.

        Raises :class:`~pyopengl_video.vaapi.h264.StreamError` for a sample
        whose headers are not read here. The slots are then unknown until the
        next key frame fills them all, and :meth:`forget` says so.
        """
        found = []
        for frame in split_superframe(data):
            header = parse_uncompressed_header(frame, self._state)
            if header.show_existing_frame:
                found.append(Frame(frame, shows_slot=header.frame_to_show_map_idx))
            else:
                self._state.refresh(header)
                found.append(Frame(frame, refreshes=header.refresh_frame_flags))
        return found

    def decoded(self, frame: Frame, picture: int) -> None:
        """Note that the coded `frame` was decoded into `picture`."""
        for slot in range(NUM_REF_FRAMES):
            if frame.refreshes >> slot & 1:
                self._pictures[slot] = picture

    def picture(self, slot: int) -> int | None:
        """The picture `slot` holds, or None where no frame read here was stored in it."""
        return self._pictures[slot]

    def lose_pictures(self) -> None:
        """Hold no picture, the frames read still known: the hardware decoder was replaced."""
        self._pictures = [None] * NUM_REF_FRAMES

    def forget(self) -> None:
        """Hold nothing: the pictures are gone, or the frames stored were not read."""
        self._state = DecodingState()
        self._pictures = [None] * NUM_REF_FRAMES
