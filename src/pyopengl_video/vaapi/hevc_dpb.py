"""The HEVC decoded picture buffer: picture order, reference picture sets and output.

A libva driver reconstructs coding tree blocks, and the caller supplies what
clause 8.3 and annex C of ITU-T H.265 derive. :class:`DecodedPictureBuffer`
does that for one layer:

- picture order counts (8.3.1)
- the reference picture set of each picture, and the marking it implies (8.3.2)
- a stand-in for a reference the buffer does not hold
- which pictures are not decoded: random access skipped leading pictures after
  the picture decoding started at, and whatever precedes that picture (8.1.3)
- reference picture lists 0 and 1, with their modifications (8.3.4)
- output order: the bumping process of C.5.2

It holds no libva and no GL. A frame names the surface it was decoded into as
an opaque value, and a caller's payload (a timestamp, say) travels with it.

One picture is decoded as::

    picture = buffer.begin(first_slice_segment_header)
    displayed = buffer.outputs()            # what has to leave before it
    if not picture.skipped:
        for header in slice_headers:
            list0, list1 = buffer.reference_lists(picture, header)
            ...                             # hand the slice to the driver
        displayed += buffer.finish(picture, surface, payload)
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

from pyopengl_video.vaapi.h264 import StreamError
from pyopengl_video.vaapi.hevc_parse import SLICE_B, SLICE_I, SliceHeader

log = logging.getLogger(__name__)

__all__ = ['DecodedPictureBuffer', 'Frame', 'Picture', 'ReferencePictureSet']

#: The reference markings a frame can carry (8.3.2).
SHORT_TERM = 'short'
LONG_TERM = 'long'


@dataclasses.dataclass(eq=False)
class Frame:
    """One picture in the decoded picture buffer.

    surface -- whatever the caller decoded the picture into
    poc -- PicOrderCntVal
    payload -- whatever the caller attached when the picture was finished
    reference -- :data:`SHORT_TERM`, :data:`LONG_TERM` or None
    needed_for_output -- still waiting to be displayed
    latency -- PicLatencyCount: how many pictures have been decoded since
        that follow this one in output order
    """

    surface: Any
    poc: int
    payload: Any = None
    reference: str | None = None
    needed_for_output: bool = False
    latency: int = 0

    @property
    def long_term(self) -> bool:
        return self.reference == LONG_TERM


@dataclasses.dataclass(frozen=True)
class ReferencePictureSet:
    """The reference picture set of one picture, as frames (8.3.2).

    st_curr_before, st_curr_after -- the short-term pictures the current one
        may predict from, before and after it in output order, nearest first
    lt_curr -- the long-term pictures it may predict from
    foll -- the pictures kept for later pictures to predict from
    missing -- the picture order counts of pictures the current one may
        predict from that the buffer does not hold; each has the nearest
        picture held standing in for it in the lists above
    """

    st_curr_before: tuple[Frame, ...] = ()
    st_curr_after: tuple[Frame, ...] = ()
    lt_curr: tuple[Frame, ...] = ()
    foll: tuple[Frame, ...] = ()
    missing: tuple[int, ...] = ()

    @property
    def frames(self) -> list[Frame]:
        """Every frame of the set once, those predicted from first."""
        found: list[Frame] = []
        for frame in (*self.st_curr_before, *self.st_curr_after, *self.lt_curr, *self.foll):
            if frame not in found:
                found.append(frame)
        return found


@dataclasses.dataclass
class Picture:
    """The picture being decoded, between :meth:`DecodedPictureBuffer.begin`
    and :meth:`~DecodedPictureBuffer.finish`.

    poc -- PicOrderCntVal
    no_rasl_output -- NoRaslOutputFlag: an intra random access point that
        decoding starts or restarts at
    output -- PicOutputFlag: the picture is displayed
    skipped -- the picture is not decoded: it predicts from pictures before
        the point decoding started at
    rps -- its reference picture set
    """

    header: SliceHeader
    poc: int
    no_rasl_output: bool = False
    output: bool = True
    skipped: bool = False
    rps: ReferencePictureSet = ReferencePictureSet()


class DecodedPictureBuffer:
    """Reference marking, reference lists and output order for one stream.

    The buffer's size, the reorder depth and the latency limit come from the
    sequence parameter set of each picture's slices, as the values of its
    highest sub-layer. Frames handed back by :meth:`outputs`, :meth:`finish`
    and :meth:`flush` are in display order; a frame may still be in
    :attr:`frames` after it is output, while later pictures refer to it.
    """

    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self._outputs: list[Frame] = []
        self._forget_history()

    def _forget_history(self) -> None:
        """The state carried between pictures."""
        #: slice_pic_order_cnt_lsb and PicOrderCntMsb of the last picture of
        #: the lowest sub-layer that other pictures may predict from.
        self._prev_lsb = 0
        self._prev_msb = 0
        #: The next picture starts a coded video sequence: nothing has been
        #: decoded since the start, a reset or an end of sequence.
        self._starting = True
        #: The leading pictures of the last random access point predict from
        #: pictures that were never decoded.
        self._skip_rasl = False

    def reset(self) -> None:
        """Drop every frame without output, as before a seek.

        Decoding resumes at the next intra random access point; pictures
        before it are skipped.
        """
        self.frames = []
        self._outputs = []
        self._forget_history()

    def end_of_sequence(self) -> list[Frame]:
        """Output every frame still waiting; the next picture starts a sequence."""
        outputs = self.flush()
        self._starting = True
        return outputs

    # ------------------------------------------------------------ pictures

    def begin(self, header: SliceHeader) -> Picture:
        """Start the picture whose first slice segment header is `header`.

        The reference picture set is applied and the frames that have to
        leave before the picture is stored are output: take them with
        :meth:`outputs` before choosing where to decode it.
        """
        if self._starting and not header.irap:
            # 8.1.3: decoding starts at an intra random access point.
            return Picture(header, poc=header.slice_pic_order_cnt_lsb, output=False,
                           skipped=True)
        if header.rasl and self._skip_rasl:
            return Picture(header, poc=self._order(header, False), output=False,
                           skipped=True)
        no_rasl_output = header.irap and (header.idr or header.bla or self._starting)
        poc = self._order(header, no_rasl_output)
        if header.temporal_id == 0 and not (header.rasl or header.radl
                                            or header.sub_layer_non_reference):
            self._prev_lsb = header.slice_pic_order_cnt_lsb
            self._prev_msb = poc - header.slice_pic_order_cnt_lsb
        if header.irap:
            self._skip_rasl = no_rasl_output
        if no_rasl_output:
            for frame in self.frames:
                frame.reference = None
        rps = self._reference_picture_set(header, poc)
        self._make_room(header, no_rasl_output)
        self._starting = False
        return Picture(header, poc=poc, no_rasl_output=no_rasl_output,
                       output=bool(header.pic_output_flag), rps=rps)

    def outputs(self) -> list[Frame]:
        """The frames output since the last call, in display order."""
        outputs, self._outputs = self._outputs, []
        return outputs

    def finish(self, picture: Picture, surface: Any, payload: Any = None) -> list[Frame]:
        """Store the decoded picture; return the frames now to display (C.5.2.3)."""
        sps = picture.header.sps
        frame = Frame(surface=surface, poc=picture.poc, payload=payload,
                      reference=SHORT_TERM, needed_for_output=picture.output)
        if picture.output:
            for stored in self.frames:
                if stored.needed_for_output and stored.poc > frame.poc:
                    stored.latency += 1
        self.frames.append(frame)
        while self._waiting_too_long(sps.reorder_pics, sps.max_latency_pictures):
            self._bump()
        return self.outputs()

    def flush(self) -> list[Frame]:
        """Output every frame still waiting, in display order, and empty the buffer."""
        while self._bump():
            pass
        self.frames = []
        return self.outputs()

    def reference_frames(self) -> list[Frame]:
        """The frames marked as references, which the driver is told about."""
        return [frame for frame in self.frames if frame.reference]

    # ----------------------------------------------------- picture order

    def _order(self, header: SliceHeader, no_rasl_output: bool) -> int:
        """PicOrderCntVal by 8.3.1."""
        lsb = header.slice_pic_order_cnt_lsb
        if header.irap and no_rasl_output:
            return lsb
        maximum = header.sps.max_pic_order_cnt_lsb
        if lsb < self._prev_lsb and self._prev_lsb - lsb >= maximum // 2:
            msb = self._prev_msb + maximum
        elif lsb > self._prev_lsb and lsb - self._prev_lsb > maximum // 2:
            msb = self._prev_msb - maximum
        else:
            msb = self._prev_msb
        return msb + lsb

    # ------------------------------------------------- reference picture set

    def _reference_picture_set(self, header: SliceHeader, poc: int) -> ReferencePictureSet:
        """Derive the set, mark the frames by it and drop the rest (8.3.2)."""
        if header.idr:
            return ReferencePictureSet()
        maximum = header.sps.max_pic_order_cnt_lsb
        references = [frame for frame in self.frames if frame.reference]
        chosen: list[Frame] = []
        missing: list[int] = []

        def find_long_term(wanted: int, whole: bool) -> Frame | None:
            for frame in references:
                held = frame.poc if whole else frame.poc & (maximum - 1)
                if held == wanted and frame not in chosen:
                    return frame
            return None

        def find_short_term(wanted: int) -> Frame | None:
            for frame in references:
                if frame.poc == wanted and not frame.long_term and frame not in chosen:
                    return frame
            return None

        long_term: list[tuple[Frame | None, int, int]] = []
        for entry in header.long_term:
            wanted = entry.poc_lsb
            if entry.delta_poc_msb_present_flag:
                wanted += poc - entry.delta_poc_msb_cycle * maximum - (poc & (maximum - 1))
            frame = find_long_term(wanted, bool(entry.delta_poc_msb_present_flag))
            if frame is not None:
                chosen.append(frame)
            long_term.append((frame, wanted, entry.used_by_curr_pic))
        rps = header.short_term_ref_pic_set
        short_term: list[list[tuple[Frame | None, int, int]]] = []
        for deltas, used_flags in ((rps.delta_poc_s0, rps.used_by_curr_pic_s0),
                                   (rps.delta_poc_s1, rps.used_by_curr_pic_s1)):
            half = []
            for delta, used in zip(deltas, used_flags, strict=True):
                frame = find_short_term(poc + delta)
                if frame is not None:
                    chosen.append(frame)
                half.append((frame, poc + delta, used))
            short_term.append(half)
        for frame, _, _ in long_term:
            if frame is not None:
                frame.reference = LONG_TERM
        for frame in references:
            if frame not in chosen:
                frame.reference = None

        def current(entries: list[tuple[Frame | None, int, int]]) -> tuple[Frame, ...]:
            found = []
            for frame, wanted, used in entries:
                if not used:
                    continue
                if frame is None:
                    missing.append(wanted)
                    frame = self._stand_in(wanted, poc)
                if frame is not None:
                    found.append(frame)
            return tuple(found)

        foll = tuple(frame for frame, _, used in (*short_term[0], *short_term[1], *long_term)
                     if frame is not None and not used)
        return ReferencePictureSet(
            st_curr_before=current(short_term[0]), st_curr_after=current(short_term[1]),
            lt_curr=current(long_term), foll=foll, missing=tuple(missing))

    def _stand_in(self, wanted: int, poc: int) -> Frame | None:
        """The frame held nearest in output order to a reference that is not.

        A stream cut or damaged in transit names pictures that never arrived.
        The picture is decoded from the nearest one there is, which shows
        something close to it; with nothing held there is nothing to stand in.
        A frame that was only waiting to be shown becomes a reference, so it
        is kept while the picture is decoded from it.
        """
        held = [frame for frame in self.frames if frame.reference] or self.frames
        if not held:
            return None
        nearest = min(held, key=lambda frame: abs(frame.poc - wanted))
        nearest.reference = nearest.reference or SHORT_TERM
        log.warning('picture %d predicts from picture %d, which was not decoded; '
                    'picture %d stands in for it', poc, wanted, nearest.poc)
        return nearest

    # ------------------------------------------------------------- output

    def _make_room(self, header: SliceHeader, no_rasl_output: bool) -> None:
        """Remove and output frames before the current picture is decoded (C.5.2.2)."""
        if no_rasl_output:
            # Decoding restarts here, and nothing held is predicted from again.
            # The frames still waiting are shown first unless the picture says
            # they are not to be.
            while not header.no_output_of_prior_pics_flag and self._bump():
                pass
            self.frames = []
            return
        sps = header.sps
        self.frames = [frame for frame in self.frames
                       if frame.reference or frame.needed_for_output]
        while (self._waiting_too_long(sps.reorder_pics, sps.max_latency_pictures)
               or len(self.frames) >= sps.dpb_size):
            if not self._bump():
                break
        if len(self.frames) > sps.dpb_size:
            # One picture more than the sequence allows still has a surface;
            # a stream that keeps more is not one this sequence describes.
            raise StreamError(
                f'the stream keeps {len(self.frames)} reference pictures in a '
                f'decoded picture buffer of {sps.dpb_size}')

    def _waiting_too_long(self, reorder: int, latency: int | None) -> bool:
        """Whether a frame has to be output now: more wait than may be
        reordered, or one has waited through its latency limit."""
        waiting = [frame for frame in self.frames if frame.needed_for_output]
        if len(waiting) > reorder:
            return True
        return latency is not None and any(frame.latency >= latency for frame in waiting)

    def _bump(self) -> bool:
        """Output the waiting frame with the lowest order count (C.5.2.4)."""
        waiting = [frame for frame in self.frames if frame.needed_for_output]
        if not waiting:
            return False
        first = min(waiting, key=lambda frame: frame.poc)
        first.needed_for_output = False
        self._outputs.append(first)
        if not first.reference:
            self.frames.remove(first)
        return True

    # ---------------------------------------------------- reference lists

    def reference_lists(self, picture: Picture, header: SliceHeader,
                        ) -> tuple[list[Frame], list[Frame]]:
        """RefPicList0 and RefPicList1 for one slice of `picture` (8.3.4).

        Each list has num_ref_idx_lX_active_minus1 + 1 entries. An I slice
        has two empty lists and a P slice an empty list 1.
        """
        if header.slice_type == SLICE_I:
            return [], []
        rps = picture.rps
        before, after, long_term = (list(rps.st_curr_before), list(rps.st_curr_after),
                                    list(rps.lt_curr))
        if not before + after + long_term:
            raise StreamError(
                f'picture {picture.poc} has {header.kind} slices and a reference '
                'picture set with no picture to predict from')
        list0 = self._list(before + after + long_term, header.list_entry_l0,
                           header.num_ref_idx_l0_active_minus1 + 1)
        if header.slice_type != SLICE_B:
            return list0, []
        return list0, self._list(after + before + long_term, header.list_entry_l1,
                                 header.num_ref_idx_l1_active_minus1 + 1)

    @staticmethod
    def _list(candidates: list[Frame], entries: tuple[int, ...], count: int) -> list[Frame]:
        """One list: the candidates repeated to its length, then reordered
        where the slice modifies it (equations 8-8 to 8-11)."""
        length = max(count, len(candidates))
        initial = [candidates[index % len(candidates)] for index in range(length)]
        if not entries:
            return initial[:count]
        if max(entries) >= length:
            raise StreamError(
                f'a reference list modification names entry {max(entries)} of '
                f'{length}')
        return [initial[entry] for entry in entries]
