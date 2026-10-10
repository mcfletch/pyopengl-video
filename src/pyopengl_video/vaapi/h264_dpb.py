"""The decoded picture buffer: picture order, reference marking, lists and output.

A libva driver reconstructs macroblocks, and the caller supplies everything in
clause 8.2 of ITU-T H.264. :class:`DecodedPictureBuffer` does that for a stream
of progressive frames:

- picture order counts of all three types (8.2.1)
- gaps in frame_num, filled with "non-existing" frames (8.2.5.2)
- reference marking by sliding window and by memory management control
  operations (8.2.5.3, 8.2.5.4)
- reference picture lists 0 and 1, initialised and then modified (8.2.4)
- output order: the bumping process of annex C.4.5.3, bounded by the
  stream's reorder depth so a picture leaves as soon as nothing can precede it

It holds no libva and no GL. A frame names the surface it was decoded into as
an opaque value, and a caller's payload (a timestamp, say) travels with it.

One picture is decoded as::

    picture = buffer.begin(first_slice_header)
    for header in slice_headers:
        list0, list1 = buffer.reference_lists(picture, header)
        ...                                 # hand the slice to the driver
    displayed = buffer.finish(picture, surface, payload)

Field pictures are refused with :class:`StreamError`; every list here holds
frames.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

from pyopengl_video.vaapi.h264 import StreamError
from pyopengl_video.vaapi.h264_parse import (
    SLICE_B,
    SLICE_I,
    SLICE_SI,
    MemoryManagementOperation,
    SequenceParameterSet,
    SliceHeader,
)

log = logging.getLogger(__name__)

__all__ = ['DecodedPictureBuffer', 'Frame', 'Picture']

#: The reference markings a frame can carry (8.2.5).
SHORT_TERM = 'short'
LONG_TERM = 'long'


@dataclasses.dataclass(eq=False)
class Frame:
    """One frame in the decoded picture buffer.

    surface -- whatever the caller decoded the frame into; None for a frame
        inferred from a gap in frame_num, which was never decoded
    payload -- whatever the caller attached when the frame was finished
    reference -- :data:`SHORT_TERM`, :data:`LONG_TERM` or None
    needed_for_output -- still waiting to be displayed
    non_existing -- inferred from a gap in frame_num (8.2.5.2)
    """

    surface: Any
    frame_num: int
    top_field_order_cnt: int
    bottom_field_order_cnt: int
    payload: Any = None
    reference: str | None = None
    long_term_frame_idx: int = 0
    needed_for_output: bool = False
    non_existing: bool = False

    @property
    def poc(self) -> int:
        """PicOrderCnt of the frame: the lesser of its two field counts."""
        return min(self.top_field_order_cnt, self.bottom_field_order_cnt)

    @property
    def short_term(self) -> bool:
        return self.reference == SHORT_TERM

    @property
    def long_term(self) -> bool:
        return self.reference == LONG_TERM


@dataclasses.dataclass
class Picture:
    """The picture being decoded, between :meth:`DecodedPictureBuffer.begin`
    and :meth:`~DecodedPictureBuffer.finish`."""

    header: SliceHeader
    top_field_order_cnt: int
    bottom_field_order_cnt: int
    poc_msb: int = 0
    frame_num_offset: int = 0

    @property
    def frame_num(self) -> int:
        return self.header.frame_num

    @property
    def poc(self) -> int:
        return min(self.top_field_order_cnt, self.bottom_field_order_cnt)

    @property
    def idr(self) -> bool:
        return self.header.idr

    @property
    def is_reference(self) -> bool:
        return self.header.nal_ref_idc != 0

    @property
    def has_mmco5(self) -> bool:
        """Whether the picture's marking includes operation 5, which resets
        frame_num, the picture order count and every reference."""
        return any(op.operation == 5 for op in self.header.memory_management)


class DecodedPictureBuffer:
    """Reference marking, reference lists and output order for one stream.

    The buffer's size and the reorder depth come from the sequence parameter
    set of each picture's slices. Frames handed back by :meth:`finish` and
    :meth:`flush` are in display order; a frame may still be in
    :attr:`frames` after it is output, while later pictures refer to it.
    """

    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self._outputs: list[Frame] = []
        self._forget_history()

    def _forget_history(self) -> None:
        """The state picture order and gap detection carry between pictures."""
        self._prev_ref_frame_num = 0
        self._prev_poc_msb = 0
        self._prev_poc_lsb = 0
        self._prev_frame_num = 0
        self._prev_frame_num_offset = 0
        self._max_long_term_frame_idx: int | None = None
        #: Whether a picture has been decoded since the start or a reset, which
        #: is what a gap in frame_num is measured from.
        self._decoded_any = False

    def reset(self) -> None:
        """Drop every frame without output, as before a seek.

        The next picture starts the count afresh: an IDR, or a recovery point
        whose frame_num is not a gap after anything.
        """
        self.frames = []
        self._outputs = []
        self._forget_history()

    # ------------------------------------------------------------ pictures

    def begin(self, header: SliceHeader) -> Picture:
        """Start decoding the picture whose first slice header is `header`."""
        if header.field_pic_flag:
            raise StreamError(
                'a field picture: this decoder takes progressive frames only, '
                'and the stream codes its pictures as separate fields')
        sps = header.sps
        if header.idr:
            self._prev_poc_msb = self._prev_poc_lsb = 0
            self._prev_frame_num_offset = 0
        elif self._decoded_any:
            self._fill_gaps(header)
        offset = self._frame_num_offset(header)
        if sps.pic_order_cnt_type == 0:
            msb, top, bottom = self._order_type_zero(header)
            return Picture(header, top, bottom, poc_msb=msb, frame_num_offset=offset)
        if sps.pic_order_cnt_type == 1:
            top, bottom = self._order_type_one(header, offset)
        else:
            top = bottom = self._order_type_two(header, offset)
        return Picture(header, top, bottom, frame_num_offset=offset)

    def finish(self, picture: Picture, surface: Any, payload: Any = None) -> list[Frame]:
        """Mark and store the decoded picture; return the frames now to display."""
        header = picture.header
        frame = Frame(surface=surface, frame_num=header.frame_num,
                      top_field_order_cnt=picture.top_field_order_cnt,
                      bottom_field_order_cnt=picture.bottom_field_order_cnt,
                      payload=payload, needed_for_output=True)
        if picture.idr or picture.has_mmco5:
            # C.4.4: everything decoded before is output, unless an IDR says
            # its predecessors are not to be shown.
            if picture.idr and header.no_output_of_prior_pics_flag:
                self.frames = []
            else:
                self._bump_all()
        self._mark(picture, frame)
        if picture.has_mmco5:
            # 8.2.1: the picture is the new origin of the picture order count.
            origin = frame.poc
            frame.top_field_order_cnt -= origin
            frame.bottom_field_order_cnt -= origin
        self._store(frame, header.sps)
        self._remember(picture, frame)
        self._decoded_any = True
        outputs, self._outputs = self._outputs, []
        return outputs

    def flush(self) -> list[Frame]:
        """Output every frame still waiting, in display order, and empty the buffer."""
        self._bump_all()
        outputs, self._outputs = self._outputs, []
        return outputs

    def reference_frames(self) -> list[Frame]:
        """The frames marked as references, which the driver is told about."""
        return [frame for frame in self.frames if frame.reference]

    # ----------------------------------------------------- picture order

    def _order_type_zero(self, header: SliceHeader) -> tuple[int, int, int]:
        """PicOrderCntMsb and the two field counts by 8.2.1.1."""
        maximum = header.sps.max_pic_order_cnt_lsb
        lsb, prev_lsb, prev_msb = (header.pic_order_cnt_lsb, self._prev_poc_lsb,
                                   self._prev_poc_msb)
        if lsb < prev_lsb and prev_lsb - lsb >= maximum // 2:
            msb = prev_msb + maximum
        elif lsb > prev_lsb and lsb - prev_lsb > maximum // 2:
            msb = prev_msb - maximum
        else:
            msb = prev_msb
        top = msb + lsb
        return msb, top, top + header.delta_pic_order_cnt_bottom

    def _frame_num_offset(self, header: SliceHeader) -> int:
        """FrameNumOffset of 8.2.1.2 and 8.2.1.3."""
        if header.idr:
            return 0
        if self._prev_frame_num > header.frame_num:
            return self._prev_frame_num_offset + header.sps.max_frame_num
        return self._prev_frame_num_offset

    @staticmethod
    def _order_type_one(header: SliceHeader, offset: int) -> tuple[int, int]:
        """The two field counts by 8.2.1.2."""
        sps = header.sps
        cycle = sps.offset_for_ref_frame
        absolute = offset + header.frame_num if cycle else 0
        if not header.nal_ref_idc and absolute > 0:
            absolute -= 1
        expected = 0
        if absolute > 0:
            count, position = divmod(absolute - 1, len(cycle))
            expected = count * sum(cycle) + sum(cycle[:position + 1])
        if not header.nal_ref_idc:
            expected += sps.offset_for_non_ref_pic
        top = expected + header.delta_pic_order_cnt[0]
        bottom = (top + sps.offset_for_top_to_bottom_field
                  + header.delta_pic_order_cnt[1])
        return top, bottom

    @staticmethod
    def _order_type_two(header: SliceHeader, offset: int) -> int:
        """tempPicOrderCnt by 8.2.1.3, which both fields of a frame share."""
        if header.idr:
            return 0
        count = 2 * (offset + header.frame_num)
        return count if header.nal_ref_idc else count - 1

    def _remember(self, picture: Picture, frame: Frame) -> None:
        """Carry what the next picture's order count and gap check depend on."""
        header = picture.header
        reset = picture.has_mmco5
        if picture.is_reference:
            self._prev_ref_frame_num = 0 if reset else header.frame_num
            if reset:
                self._prev_poc_msb = 0
                self._prev_poc_lsb = frame.top_field_order_cnt
            else:
                self._prev_poc_msb = picture.poc_msb
                self._prev_poc_lsb = header.pic_order_cnt_lsb
        self._prev_frame_num = 0 if reset else header.frame_num
        self._prev_frame_num_offset = 0 if reset else picture.frame_num_offset

    # ------------------------------------------------------------ marking

    def _fill_gaps(self, header: SliceHeader) -> None:
        """Infer the frames a jump in frame_num skipped (8.2.5.2)."""
        sps = header.sps
        maximum = sps.max_frame_num
        expected = (self._prev_ref_frame_num + 1) % maximum
        if header.frame_num in (self._prev_ref_frame_num, expected):
            return
        if not sps.gaps_in_frame_num_value_allowed_flag:
            log.debug('frame_num jumps from %d to %d in a stream that does not '
                      'allow gaps; the missing frames are inferred',
                      self._prev_ref_frame_num, header.frame_num)
        unused = expected
        while unused != header.frame_num:
            offset = self._prev_frame_num_offset + (
                maximum if self._prev_frame_num > unused else 0)
            frame = Frame(surface=None, frame_num=unused, top_field_order_cnt=0,
                          bottom_field_order_cnt=0, non_existing=True)
            self._sliding_window(sps, unused)
            frame.reference = SHORT_TERM
            self._store(frame, sps)
            self._prev_ref_frame_num = self._prev_frame_num = unused
            self._prev_frame_num_offset = offset
            unused = (unused + 1) % maximum

    @staticmethod
    def _frame_num_wrap(frame: Frame, frame_num: int, maximum: int) -> int:
        """FrameNumWrap of a short-term frame, seen from `frame_num` (8.2.4.1)."""
        return frame.frame_num - maximum if frame.frame_num > frame_num else frame.frame_num

    def _short_term(self, pic_num: int, frame_num: int, maximum: int) -> Frame | None:
        for frame in self.frames:
            if (frame.short_term
                    and self._frame_num_wrap(frame, frame_num, maximum) == pic_num):
                return frame
        return None

    def _long_term(self, index: int) -> Frame | None:
        for frame in self.frames:
            if frame.long_term and frame.long_term_frame_idx == index:
                return frame
        return None

    def _sliding_window(self, sps: SequenceParameterSet, frame_num: int) -> None:
        """Unmark the oldest short-term frame when the references are full (8.2.5.3)."""
        maximum = sps.max_frame_num
        while len(self.reference_frames()) >= max(sps.max_num_ref_frames, 1):
            short = [frame for frame in self.frames if frame.short_term]
            if not short:
                return
            oldest = min(short, key=lambda frame: self._frame_num_wrap(
                frame, frame_num, maximum))
            oldest.reference = None

    def _mark(self, picture: Picture, frame: Frame) -> None:
        """Decoded reference picture marking (8.2.5)."""
        header = picture.header
        if picture.idr:
            for stored in self.frames:
                stored.reference = None
            if header.long_term_reference_flag:
                frame.reference, frame.long_term_frame_idx = LONG_TERM, 0
                self._max_long_term_frame_idx = 0
            else:
                frame.reference = SHORT_TERM
                self._max_long_term_frame_idx = None
            return
        if not picture.is_reference:
            return
        if header.adaptive_ref_pic_marking_mode_flag:
            for operation in header.memory_management:
                self._operate(operation, picture, frame)
        else:
            self._sliding_window(header.sps, header.frame_num)
        if not frame.long_term:
            frame.reference = SHORT_TERM

    def _operate(self, operation: MemoryManagementOperation, picture: Picture,
                 frame: Frame) -> None:
        """One memory_management_control_operation (8.2.5.4)."""
        kind = operation.operation
        frame_num = picture.frame_num
        maximum = picture.header.sps.max_frame_num
        if kind in (1, 3):
            pic_num = frame_num - (operation.difference_of_pic_nums_minus1 + 1)
            target = self._short_term(pic_num, frame_num, maximum)
            if target is None:
                log.debug('operation %d names picture %d, which is not a '
                          'short-term reference', kind, pic_num)
                return
            if kind == 1:
                target.reference = None
            else:
                self._take_long_term_index(operation.long_term_frame_idx)
                target.reference = LONG_TERM
                target.long_term_frame_idx = operation.long_term_frame_idx
        elif kind == 2:
            target = self._long_term(operation.long_term_pic_num)
            if target is not None:
                target.reference = None
        elif kind == 4:
            plus1 = operation.max_long_term_frame_idx_plus1
            self._max_long_term_frame_idx = plus1 - 1 if plus1 else None
            for stored in self.frames:
                if stored.long_term and (plus1 == 0
                                         or stored.long_term_frame_idx > plus1 - 1):
                    stored.reference = None
        elif kind == 5:
            for stored in self.frames:
                stored.reference = None
            self._max_long_term_frame_idx = None
        elif kind == 6:
            self._take_long_term_index(operation.long_term_frame_idx)
            frame.reference = LONG_TERM
            frame.long_term_frame_idx = operation.long_term_frame_idx

    def _take_long_term_index(self, index: int) -> None:
        """Free a LongTermFrameIdx for reassignment."""
        holder = self._long_term(index)
        if holder is not None:
            holder.reference = None

    # ------------------------------------------------------------ storage

    def _store(self, frame: Frame, sps: SequenceParameterSet) -> None:
        """Put a frame in the buffer, outputting what has to leave first (C.4.5)."""
        self.frames = [stored for stored in self.frames
                       if stored.reference or stored.needed_for_output]
        size = sps.dpb_frames
        if (frame.needed_for_output and not frame.reference
                and len(self.frames) >= size
                and all(frame.poc < stored.poc for stored in self.frames
                        if stored.needed_for_output)):
            # C.4.5.2: a non-reference picture that would be shown before
            # everything waiting leaves at once, without a buffer of its own.
            frame.needed_for_output = False
            self._outputs.append(frame)
            return
        while len(self.frames) >= size:
            if not self._bump():
                raise StreamError(
                    f'the decoded picture buffer of {size} frames is full of '
                    'reference frames, and the stream marks none of them unused')
        self.frames.append(frame)
        reorder = sps.reorder_frames
        while sum(stored.needed_for_output for stored in self.frames) > reorder:
            self._bump()

    def _bump(self) -> bool:
        """Output the waiting frame with the lowest order count (C.4.5.3)."""
        waiting = [frame for frame in self.frames if frame.needed_for_output]
        if not waiting:
            return False
        first = min(waiting, key=lambda frame: frame.poc)
        first.needed_for_output = False
        self._outputs.append(first)
        if not first.reference:
            self.frames.remove(first)
        return True

    def _bump_all(self) -> None:
        while self._bump():
            pass
        self.frames = []

    # ---------------------------------------------------- reference lists

    def reference_lists(self, picture: Picture, header: SliceHeader,
                        ) -> tuple[list[Frame | None], list[Frame | None]]:
        """RefPicList0 and RefPicList1 for one slice of `picture` (8.2.4).

        Each list has num_ref_idx_lX_active_minus1 + 1 entries, None where the
        stream names more references than the buffer holds. An I slice has
        two empty lists and a P slice an empty list 1.
        """
        if header.slice_type in (SLICE_I, SLICE_SI):
            return [], []
        maximum = header.sps.max_frame_num
        frame_num = header.frame_num

        def pic_num(frame: Frame) -> int:
            return self._frame_num_wrap(frame, frame_num, maximum)

        short = [frame for frame in self.frames if frame.short_term]
        long = sorted((frame for frame in self.frames if frame.long_term),
                      key=lambda frame: frame.long_term_frame_idx)
        if header.slice_type != SLICE_B:
            initial0 = sorted(short, key=pic_num, reverse=True) + long
            return self._modify(initial0, header.ref_pic_list_modification_l0,
                                header.num_ref_idx_l0_active_minus1 + 1, header), []
        current = picture.poc
        existing = [frame for frame in short if not frame.non_existing]
        before = sorted((frame for frame in existing if frame.poc < current),
                        key=lambda frame: frame.poc, reverse=True)
        after = sorted((frame for frame in existing if frame.poc > current),
                       key=lambda frame: frame.poc)
        initial0 = before + after + long
        initial1 = after + before + long
        if len(initial1) > 1 and initial1 == initial0:
            initial1[0], initial1[1] = initial1[1], initial1[0]
        return (
            self._modify(initial0, header.ref_pic_list_modification_l0,
                         header.num_ref_idx_l0_active_minus1 + 1, header),
            self._modify(initial1, header.ref_pic_list_modification_l1,
                         header.num_ref_idx_l1_active_minus1 + 1, header),
        )

    def _modify(self, initial: list[Frame], modifications: tuple[tuple[int, int], ...],
                count: int, header: SliceHeader) -> list[Frame | None]:
        """Apply ref_pic_list_modification to an initial list (8.2.4.3).

        Each operation puts the frame it names at the next index and removes
        that frame from later in the list, which is what comparing PicNumF or
        LongTermPicNumF against the named number does for a list of frames.
        """
        entries: list[Frame | None] = list(initial[:count])
        entries += [None] * (count - len(entries))
        maximum = header.sps.max_frame_num
        current = header.frame_num
        predicted = current
        for index, (idc, value) in enumerate(modifications):
            if idc in (0, 1):
                step = value + 1
                no_wrap = predicted - step if idc == 0 else predicted + step
                if no_wrap < 0:
                    no_wrap += maximum
                elif no_wrap >= maximum:
                    no_wrap -= maximum
                predicted = no_wrap
                wanted = no_wrap - maximum if no_wrap > current else no_wrap
                target = self._short_term(wanted, current, maximum)
                if target is None:
                    raise StreamError(
                        f'a reference list modification names picture {wanted}, '
                        'which is not a short-term reference')
            elif idc == 2:
                target = self._long_term(value)
                if target is None:
                    raise StreamError(
                        f'a reference list modification names long-term picture '
                        f'{value}, which is not a long-term reference')
            else:
                raise StreamError(
                    f'modification_of_pic_nums_idc {idc} belongs to multiview '
                    'coding, which is not decoded here')
            if index >= count:
                raise StreamError(
                    f'{len(modifications)} reference list modifications for a '
                    f'list of {count}')
            kept = [entry for entry in entries[index:] if entry is not target]
            entries = (entries[:index] + [target] + kept + [None] * count)[:count]
        return entries
