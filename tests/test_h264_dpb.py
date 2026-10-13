"""The decoded picture buffer: picture order, marking, reference lists, output.

No hardware. A libva driver decodes the macroblocks and nothing else, so the
whole of clause 8.2 is this module's: which pictures are references, the order
the reference lists hold them in, and the order pictures leave for display.
Each case drives the buffer with slice headers and checks it against the
clause it implements.
"""
import pytest

from pyopengl_video.vaapi.h264_dpb import SHORT_TERM, DecodedPictureBuffer, Frame
from pyopengl_video.vaapi.h264_parse import MemoryManagementOperation as Op
from pyopengl_video.vaapi.h264_parse import StreamError
from tests.h264_syntax import header, picture_set, restriction_vui, sequence


def sets(reorder=None, refs=4, buffering=None, **options):
    vui = None
    if reorder is not None:
        vui = restriction_vui(reorder, refs if buffering is None else buffering)
    sps = sequence(max_num_ref_frames=refs, vui=vui, **options)
    return sps, picture_set(sps, refs=(refs - 1, refs - 1))


def decode(dpb, slice_header, name):
    """Decode one picture into a surface called `name`; return what was output."""
    picture = dpb.begin(slice_header)
    return [frame.surface for frame in dpb.finish(picture, surface=name)]


def names(frames):
    return [None if frame is None else frame.surface for frame in frames]


# ------------------------------------------------------- picture order count


class TestPictureOrderCountTypeZero:
    def test_counts_follow_the_lsb(self):
        sps, pps = sets(reorder=0)
        dpb = DecodedPictureBuffer()
        pictures = []
        for index, lsb in enumerate((0, 2, 4)):
            picture = dpb.begin(header(sps, pps, kind='I' if index == 0 else 'P',
                                       idr=index == 0, frame_num=index,
                                       poc_lsb=lsb))
            pictures.append(picture.poc)
            dpb.finish(picture, surface=index)
        assert pictures == [0, 2, 4]

    def test_the_msb_advances_when_the_lsb_wraps(self):
        """MaxPicOrderCntLsb is 64 here (8.2.1.1)."""
        sps, pps = sets(reorder=0)
        dpb = DecodedPictureBuffer()
        counts = []
        for index, lsb in enumerate((0, 30, 60, 4, 34, 0)):
            picture = dpb.begin(header(sps, pps, kind='I' if index == 0 else 'P',
                                       idr=index == 0, frame_num=index % 16,
                                       poc_lsb=lsb))
            counts.append(picture.poc)
            dpb.finish(picture, surface=index)
        assert counts == [0, 30, 60, 68, 98, 128]

    def test_the_msb_comes_from_the_previous_reference_picture(self):
        """A non-reference picture's lsb does not move the reference point."""
        sps, pps = sets(reorder=4)
        dpb = DecodedPictureBuffer()
        decode(dpb, header(sps, pps, kind='I', idr=True, poc_lsb=0), 'I0')
        decode(dpb, header(sps, pps, frame_num=1, poc_lsb=30), 'P1')
        picture = dpb.begin(header(sps, pps, kind='B', frame_num=2, poc_lsb=60,
                                   nal_ref_idc=0))
        assert picture.poc == 60
        dpb.finish(picture, surface='b')
        # Against P1's lsb of 30, 2 is no wrap; against the B picture's 60 it
        # would be one.
        picture = dpb.begin(header(sps, pps, frame_num=2, poc_lsb=2))
        assert picture.poc == 2

    def test_the_bottom_field_delta(self):
        sps, pps = sets(reorder=0)
        dpb = DecodedPictureBuffer()
        picture = dpb.begin(header(sps, pps, kind='I', idr=True, poc_lsb=4,
                                   delta_pic_order_cnt_bottom=-2))
        assert (picture.top_field_order_cnt, picture.bottom_field_order_cnt) == (4, 2)
        assert picture.poc == 2


class TestPictureOrderCountTypeOne:
    """offset_for_ref_frame (2, 4), offset_for_non_ref_pic -1, cycle of two."""

    def test_reference_and_non_reference_pictures(self):
        sps, pps = sets(reorder=4, poc_type=1)
        dpb = DecodedPictureBuffer()
        counts = []
        for frame_num, reference in ((0, True), (1, True), (2, True), (3, False)):
            picture = dpb.begin(header(
                sps, pps, kind='I' if frame_num == 0 else 'P', idr=frame_num == 0,
                frame_num=frame_num, nal_ref_idc=2 if reference else 0))
            counts.append(picture.poc)
            dpb.finish(picture, surface=frame_num)
        # absFrameNum 1 -> 2; 2 -> 2 + 4; 3, non-reference, is 2 -> 6 + 0 - 1.
        assert counts == [0, 2, 6, 5]

    def test_the_slice_delta_is_added(self):
        sps, pps = sets(reorder=4, poc_type=1)
        dpb = DecodedPictureBuffer()
        picture = dpb.begin(header(sps, pps, kind='I', idr=True,
                                   delta_pic_order_cnt=(3, 0)))
        assert picture.poc == 3


class TestPictureOrderCountTypeTwo:
    def test_output_order_is_decoding_order(self):
        sps, pps = sets(reorder=0, poc_type=2)
        dpb = DecodedPictureBuffer()
        counts = []
        for frame_num, reference in ((0, True), (1, True), (2, False), (2, True)):
            picture = dpb.begin(header(
                sps, pps, kind='I' if frame_num == 0 else 'P', idr=frame_num == 0,
                frame_num=frame_num, nal_ref_idc=2 if reference else 0))
            counts.append(picture.poc)
            dpb.finish(picture, surface=len(counts))
        assert counts == [0, 2, 3, 4]

    def test_frame_num_wrapping_carries_into_the_offset(self):
        sps, pps = sets(reorder=0, poc_type=2, refs=1)
        dpb = DecodedPictureBuffer()
        counts = []
        for index in range(18):
            picture = dpb.begin(header(sps, pps, kind='I' if index == 0 else 'P',
                                       idr=index == 0, frame_num=index % 16))
            counts.append(picture.poc)
            dpb.finish(picture, surface=index)
        assert counts == [2 * index for index in range(18)]


# --------------------------------------------------------------------- output


class TestOutputOrder:
    def test_without_reordering_each_picture_leaves_as_it_is_decoded(self):
        sps, pps = sets(reorder=0, refs=1)
        dpb = DecodedPictureBuffer()
        assert decode(dpb, header(sps, pps, kind='I', idr=True), 'I0') == ['I0']
        assert decode(dpb, header(sps, pps, frame_num=1, poc_lsb=2), 'P1') == ['P1']

    def test_b_pictures_leave_in_display_order(self):
        """I0 P6 B2 B4 P12 B8 B10, decoded in that order, shown by count."""
        sps, pps = sets(reorder=2, refs=2, buffering=4)
        dpb = DecodedPictureBuffer()
        out = []
        out += decode(dpb, header(sps, pps, kind='I', idr=True, poc_lsb=0), 0)
        out += decode(dpb, header(sps, pps, frame_num=1, poc_lsb=6), 6)
        out += decode(dpb, header(sps, pps, kind='B', frame_num=2, poc_lsb=2,
                                  nal_ref_idc=0), 2)
        out += decode(dpb, header(sps, pps, kind='B', frame_num=2, poc_lsb=4,
                                  nal_ref_idc=0), 4)
        out += decode(dpb, header(sps, pps, frame_num=2, poc_lsb=12), 12)
        out += decode(dpb, header(sps, pps, kind='B', frame_num=3, poc_lsb=8,
                                  nal_ref_idc=0), 8)
        out += decode(dpb, header(sps, pps, kind='B', frame_num=3, poc_lsb=10,
                                  nal_ref_idc=0), 10)
        out += [frame.surface for frame in dpb.flush()]
        assert out == [0, 2, 4, 6, 8, 10, 12]

    def test_output_is_held_only_as_long_as_the_reorder_depth(self):
        sps, pps = sets(reorder=1, refs=2)
        dpb = DecodedPictureBuffer()
        assert decode(dpb, header(sps, pps, kind='I', idr=True), 0) == []
        assert decode(dpb, header(sps, pps, frame_num=1, poc_lsb=4), 4) == [0]

    def test_a_full_buffer_outputs_to_make_room(self):
        """No VUI: the reorder depth is the buffer's, and fullness bumps."""
        sps, pps = sets(refs=2, level_idc=10, width_mbs=11, height_mbs=9)
        assert sps.dpb_frames == 4 and sps.reorder_frames == 4
        dpb = DecodedPictureBuffer()
        out = decode(dpb, header(sps, pps, kind='I', idr=True), 0)
        for index in range(1, 6):
            out += decode(dpb, header(sps, pps, frame_num=index,
                                      poc_lsb=index * 2), index * 2)
        assert out == [0, 2]
        assert len(dpb.frames) == 4

    def test_an_idr_outputs_everything_before_it(self):
        sps, pps = sets(reorder=4)
        dpb = DecodedPictureBuffer()
        decode(dpb, header(sps, pps, kind='I', idr=True), 'a')
        decode(dpb, header(sps, pps, frame_num=1, poc_lsb=2), 'b')
        out = decode(dpb, header(sps, pps, kind='I', idr=True, idr_pic_id=1), 'c')
        assert out == ['a', 'b']
        assert dpb.flush()[0].surface == 'c'

    def test_no_output_of_prior_pictures_discards_them(self):
        sps, pps = sets(reorder=4)
        dpb = DecodedPictureBuffer()
        decode(dpb, header(sps, pps, kind='I', idr=True), 'a')
        out = decode(dpb, header(sps, pps, kind='I', idr=True,
                                 no_output_of_prior_pics_flag=1), 'c')
        assert out == []
        assert [frame.surface for frame in dpb.flush()] == ['c']

    def test_reset_drops_everything_without_output(self):
        sps, pps = sets(reorder=4)
        dpb = DecodedPictureBuffer()
        decode(dpb, header(sps, pps, kind='I', idr=True), 'a')
        dpb.reset()
        assert dpb.frames == []
        assert dpb.flush() == []

    def test_a_picture_that_is_not_a_reference_leaves_the_buffer_on_output(self):
        sps, pps = sets(reorder=0)
        dpb = DecodedPictureBuffer()
        decode(dpb, header(sps, pps, kind='I', idr=True), 'a')
        decode(dpb, header(sps, pps, kind='B', frame_num=1, poc_lsb=2,
                           nal_ref_idc=0), 'b')
        assert [frame.surface for frame in dpb.frames] == ['a']

    def test_the_payload_travels_with_the_frame(self):
        sps, pps = sets(reorder=0)
        dpb = DecodedPictureBuffer()
        picture = dpb.begin(header(sps, pps, kind='I', idr=True))
        [frame] = dpb.finish(picture, surface='s', payload={'pts': 9})
        assert frame.payload == {'pts': 9}


# ------------------------------------------------------------------ marking


def stream(sps, pps, count, start=0):
    """An IDR, then `count` P pictures, each a reference; returns the buffer."""
    dpb = DecodedPictureBuffer()
    decode(dpb, header(sps, pps, kind='I', idr=True), start)
    for index in range(1, count + 1):
        decode(dpb, header(sps, pps, frame_num=index % 16,
                           poc_lsb=index * 2 % 64), start + index)
    return dpb


class TestSlidingWindow:
    def test_the_oldest_short_term_reference_goes_first(self):
        sps, pps = sets(reorder=0, refs=2)
        dpb = stream(sps, pps, 3)
        assert sorted(frame.surface for frame in dpb.reference_frames()) == [2, 3]

    def test_the_window_counts_by_frame_num_across_a_wrap(self):
        sps, pps = sets(reorder=0, refs=3)
        dpb = stream(sps, pps, 18)
        assert sorted(f.frame_num for f in dpb.reference_frames()) == [0, 1, 2]


class TestAdaptiveMarking:
    def test_operation_one_unmarks_a_short_term_picture(self):
        sps, pps = sets(reorder=0)
        dpb = stream(sps, pps, 2)
        # CurrPicNum 3; 3 - (1 + 1) is picture 1.
        decode(dpb, header(sps, pps, frame_num=3, poc_lsb=6,
                           memory_management=(Op(1, difference_of_pic_nums_minus1=1),)),
               3)
        assert sorted(frame.surface for frame in dpb.reference_frames()) == [0, 2, 3]

    def test_operation_three_makes_a_long_term_reference(self):
        sps, pps = sets(reorder=0)
        dpb = stream(sps, pps, 2)
        decode(dpb, header(sps, pps, frame_num=3, poc_lsb=6,
                           memory_management=(Op(4, max_long_term_frame_idx_plus1=2),
                                              Op(3, difference_of_pic_nums_minus1=2,
                                                 long_term_frame_idx=1))),
               3)
        long_term = [f for f in dpb.reference_frames() if f.long_term]
        assert [(f.surface, f.long_term_frame_idx) for f in long_term] == [(0, 1)]

    def test_operation_six_makes_the_current_picture_long_term(self):
        sps, pps = sets(reorder=0)
        dpb = stream(sps, pps, 1)
        decode(dpb, header(sps, pps, frame_num=2, poc_lsb=4,
                           memory_management=(Op(4, max_long_term_frame_idx_plus1=1),
                                              Op(6, long_term_frame_idx=0))),
               2)
        [long_term] = [f for f in dpb.reference_frames() if f.long_term]
        assert long_term.surface == 2

    def test_operation_two_unmarks_a_long_term_reference(self):
        sps, pps = sets(reorder=0)
        dpb = stream(sps, pps, 1)
        decode(dpb, header(sps, pps, frame_num=2, poc_lsb=4,
                           memory_management=(Op(4, max_long_term_frame_idx_plus1=1),
                                              Op(6, long_term_frame_idx=0))),
               2)
        decode(dpb, header(sps, pps, frame_num=3, poc_lsb=6,
                           memory_management=(Op(2, long_term_pic_num=0),)), 3)
        assert not any(f.long_term for f in dpb.reference_frames())

    def test_operation_four_drops_indices_above_the_new_maximum(self):
        sps, pps = sets(reorder=0)
        dpb = stream(sps, pps, 2)
        decode(dpb, header(sps, pps, frame_num=3, poc_lsb=6,
                           memory_management=(Op(4, max_long_term_frame_idx_plus1=3),
                                              Op(3, difference_of_pic_nums_minus1=0,
                                                 long_term_frame_idx=2),
                                              Op(4, max_long_term_frame_idx_plus1=2))),
               3)
        assert not any(f.long_term for f in dpb.reference_frames())

    def test_a_long_term_index_already_in_use_is_taken_over(self):
        sps, pps = sets(reorder=0)
        dpb = stream(sps, pps, 2)
        decode(dpb, header(sps, pps, frame_num=3, poc_lsb=6,
                           memory_management=(Op(4, max_long_term_frame_idx_plus1=1),
                                              Op(3, difference_of_pic_nums_minus1=0,
                                                 long_term_frame_idx=0),
                                              Op(6, long_term_frame_idx=0))),
               3)
        [long_term] = [f for f in dpb.reference_frames() if f.long_term]
        assert long_term.surface == 3

    def test_operation_five_clears_everything_and_restarts_the_count(self):
        sps, pps = sets(reorder=4)
        dpb = stream(sps, pps, 2)
        out = decode(dpb, header(sps, pps, frame_num=3, poc_lsb=6,
                                 memory_management=(Op(5),)), 3)
        assert out == [0, 1, 2]
        assert [f.surface for f in dpb.reference_frames()] == [3]
        picture = dpb.begin(header(sps, pps, frame_num=1, poc_lsb=2))
        assert picture.poc == 2
        assert dpb.reference_frames()[0].poc == 0

    def test_an_idr_can_be_a_long_term_reference(self):
        sps, pps = sets(reorder=0)
        dpb = DecodedPictureBuffer()
        decode(dpb, header(sps, pps, kind='I', idr=True,
                           long_term_reference_flag=1), 0)
        [frame] = dpb.reference_frames()
        assert frame.long_term and frame.long_term_frame_idx == 0


class TestFrameNumGaps:
    def test_missing_frame_numbers_are_filled_with_non_existing_frames(self):
        sps, pps = sets(reorder=0, refs=4, gaps=1)
        dpb = DecodedPictureBuffer()
        decode(dpb, header(sps, pps, kind='I', idr=True), 0)
        decode(dpb, header(sps, pps, frame_num=3, poc_lsb=6), 3)
        references = dpb.reference_frames()
        assert sorted(f.frame_num for f in references) == [0, 1, 2, 3]
        assert [f.frame_num for f in references if f.non_existing] == [1, 2]

    def test_non_existing_frames_are_never_output(self):
        sps, pps = sets(reorder=0, refs=4, gaps=1)
        dpb = DecodedPictureBuffer()
        out = decode(dpb, header(sps, pps, kind='I', idr=True), 0)
        out += decode(dpb, header(sps, pps, frame_num=3, poc_lsb=6), 3)
        out += [f.surface for f in dpb.flush()]
        assert out == [0, 3]

    def test_a_stream_entered_at_a_recovery_point_has_no_gap_before_it(self):
        """After a reset there is no previous picture to count a gap from."""
        sps, pps = sets(reorder=0, refs=2, gaps=1)
        dpb = DecodedPictureBuffer()
        decode(dpb, header(sps, pps, kind='I', idr=True), 0)
        dpb.reset()
        decode(dpb, header(sps, pps, kind='I', frame_num=9, poc_lsb=40), 9)
        assert [f.frame_num for f in dpb.reference_frames()] == [9]
        decode(dpb, header(sps, pps, frame_num=10, poc_lsb=42), 10)
        assert sorted(f.frame_num for f in dpb.reference_frames()) == [9, 10]

    def test_the_sliding_window_applies_to_them(self):
        sps, pps = sets(reorder=0, refs=2, gaps=1)
        dpb = DecodedPictureBuffer()
        decode(dpb, header(sps, pps, kind='I', idr=True), 0)
        decode(dpb, header(sps, pps, frame_num=5, poc_lsb=10), 5)
        assert sorted(f.frame_num for f in dpb.reference_frames()) == [4, 5]


class Stepwise(DecodedPictureBuffer):
    """8.2.5.2 as it is written: a frame inferred for every frame_num skipped."""

    def _fill_gaps(self, header):
        maximum = header.sps.max_frame_num
        unused = (self._prev_ref_frame_num + 1) % maximum
        if header.frame_num == self._prev_ref_frame_num:
            return
        while unused != header.frame_num:
            offset = self._prev_frame_num_offset + (
                maximum if self._prev_frame_num > unused else 0)
            frame = Frame(surface=None, frame_num=unused, top_field_order_cnt=0,
                          bottom_field_order_cnt=0, non_existing=True)
            self._sliding_window(header.sps, unused)
            frame.reference = SHORT_TERM
            self._store(frame, header.sps)
            self._prev_ref_frame_num = self._prev_frame_num = unused
            self._prev_frame_num_offset = offset
            unused = (unused + 1) % maximum


class Counting(DecodedPictureBuffer):
    """Counts the frames stored, decoded and inferred."""

    stored = 0

    def _store(self, frame, sps):
        self.stored += 1
        super()._store(frame, sps)


class TestLongFrameNumGaps:
    """H6: a gap costs what the sliding window keeps, not its length."""

    def held(self, dpb):
        return [(frame.frame_num, frame.surface, frame.reference, frame.non_existing)
                for frame in dpb.frames]

    @pytest.mark.parametrize('refs', [1, 2, 3, 5])
    @pytest.mark.parametrize('long_term', [0, 1])
    def test_every_gap_ends_as_stepping_through_it_does(self, refs, long_term):
        """MaxFrameNum is 16. From each frame_num to each other one, the
        frames held, what is output and the picture's order count are those
        of the specification's loop; picture order count type 2 states
        FrameNumOffset in the count."""
        sps, pps = sets(refs=refs, gaps=1, poc_type=2)
        for before in range(16):
            for after in range(16):
                found = []
                for dpb in (DecodedPictureBuffer(), Stepwise()):
                    shown = decode(dpb, header(sps, pps, kind='I', idr=True,
                                               long_term_reference_flag=long_term), 100)
                    for frame_num in range(1, before + 1):
                        shown += decode(dpb, header(sps, pps, frame_num=frame_num),
                                        100 + frame_num)
                    picture = dpb.begin(header(sps, pps, frame_num=after))
                    shown += names(dpb.outputs())
                    found.append((picture.poc, picture.frame_num_offset, shown,
                                  self.held(dpb)))
                assert found[0] == found[1], (before, after)

    def test_a_gap_of_a_whole_frame_num_range_stores_the_window_alone(self):
        """MaxFrameNum is 65536, the largest there is, and the gap is all of
        it but two: four frames are inferred, and the picture is stored."""
        sps, pps = sets(refs=4, gaps=1, log2_max_frame_num_minus4=12)
        dpb = Counting()
        decode(dpb, header(sps, pps, kind='I', idr=True), 0)
        decode(dpb, header(sps, pps, frame_num=65535, poc_lsb=2), 1)
        assert dpb.stored == 1 + 4 + 1
        assert [(frame.frame_num, frame.non_existing)
                for frame in dpb.reference_frames()] == [
            (65532, True), (65533, True), (65534, True), (65535, False)]

    def test_frames_from_before_a_long_gap_are_no_longer_references(self):
        sps, pps = sets(refs=2, gaps=1, log2_max_frame_num_minus4=12)
        dpb = DecodedPictureBuffer()
        decode(dpb, header(sps, pps, kind='I', idr=True), 0)
        decode(dpb, header(sps, pps, frame_num=1, poc_lsb=2), 1)
        decode(dpb, header(sps, pps, frame_num=40000, poc_lsb=4), 2)
        assert [frame.frame_num for frame in dpb.reference_frames()] == [39999, 40000]

    def test_what_a_gap_pushes_out_is_there_before_the_picture_is_finished(self):
        """The frames inferred take the buffer's places, so a frame waiting
        to be shown leaves when the picture begins, while its surface is
        still its own."""
        sps, pps = sets(reorder=2, refs=2, gaps=1)
        dpb = DecodedPictureBuffer()
        assert decode(dpb, header(sps, pps, kind='I', idr=True), 0) == []
        assert decode(dpb, header(sps, pps, frame_num=1, poc_lsb=2), 1) == []
        picture = dpb.begin(header(sps, pps, frame_num=5, poc_lsb=10))
        assert names(dpb.outputs()) == [0, 1]
        assert names(dpb.finish(picture, surface=5)) == []


class TestPictureOrderCountRange:
    """8.2.1: every order count fits 32 signed bits, which is what libva
    carries one in."""

    def test_a_bottom_field_count_past_the_range_is_refused(self):
        sps, pps = sets(reorder=0)
        dpb = DecodedPictureBuffer()
        decode(dpb, header(sps, pps, kind='I', idr=True), 0)
        with pytest.raises(StreamError, match='BottomFieldOrderCnt is 2147483649'):
            dpb.begin(header(sps, pps, frame_num=1, poc_lsb=2,
                             delta_pic_order_cnt_bottom=2**31 - 1))

    def test_the_ends_of_the_range_are_counts(self):
        sps, pps = sets(reorder=0)
        dpb = DecodedPictureBuffer()
        picture = dpb.begin(header(sps, pps, kind='I', idr=True,
                                   delta_pic_order_cnt_bottom=2**31 - 1))
        assert picture.bottom_field_order_cnt == 2**31 - 1


# ---------------------------------------------------------- reference lists


class TestInitialListsForP:
    def test_short_term_by_descending_pic_num(self):
        sps, pps = sets(reorder=0, refs=4)
        dpb = stream(sps, pps, 3)
        current = header(sps, pps, frame_num=4, poc_lsb=8,
                         num_ref_idx_l0_active_minus1=3)
        picture = dpb.begin(current)
        l0, l1 = dpb.reference_lists(picture, current)
        assert names(l0) == [3, 2, 1, 0]
        assert l1 == []

    def test_pic_num_wraps_below_zero(self):
        """frame_num 14 and 15 come before 0 and 1 in a stream at frame 2."""
        sps, pps = sets(reorder=0, refs=4)
        dpb = stream(sps, pps, 17)
        current = header(sps, pps, frame_num=2, poc_lsb=36,
                         num_ref_idx_l0_active_minus1=3)
        picture = dpb.begin(current)
        l0, _ = dpb.reference_lists(picture, current)
        assert [f.frame_num for f in l0] == [1, 0, 15, 14]

    def test_long_term_follow_short_term_by_ascending_index(self):
        sps, pps = sets(reorder=0, refs=4)
        dpb = stream(sps, pps, 2)
        decode(dpb, header(sps, pps, frame_num=3, poc_lsb=6,
                           memory_management=(Op(4, max_long_term_frame_idx_plus1=2),
                                              Op(3, difference_of_pic_nums_minus1=2,
                                                 long_term_frame_idx=1),
                                              Op(3, difference_of_pic_nums_minus1=1,
                                                 long_term_frame_idx=0))),
               3)
        current = header(sps, pps, frame_num=4, poc_lsb=8,
                         num_ref_idx_l0_active_minus1=3)
        picture = dpb.begin(current)
        l0, _ = dpb.reference_lists(picture, current)
        assert names(l0) == [3, 2, 1, 0]

    def test_a_list_longer_than_the_references_ends_in_its_first_frame(self):
        """8.2.4.2 has "no reference picture" there; a driver is given a surface."""
        sps, pps = sets(reorder=0, refs=4)
        dpb = stream(sps, pps, 1)
        current = header(sps, pps, frame_num=2, poc_lsb=4,
                         num_ref_idx_l0_active_minus1=2)
        picture = dpb.begin(current)
        l0, _ = dpb.reference_lists(picture, current)
        assert names(l0) == [1, 0, 1]

    def test_a_frame_inferred_from_a_gap_is_not_predicted_from(self):
        """Frame 2 was never decoded: the list's first decoded frame, frame
        1, is in its place, and has a surface."""
        sps, pps = sets(reorder=0, refs=4, gaps=1)
        dpb = stream(sps, pps, 1)
        current = header(sps, pps, frame_num=3, poc_lsb=6,
                         num_ref_idx_l0_active_minus1=2)
        picture = dpb.begin(current)
        l0, _ = dpb.reference_lists(picture, current)
        assert names(l0) == [1, 1, 0]
        assert not any(frame.non_existing for frame in l0)

    def test_a_predicted_slice_with_no_decoded_frame_is_refused(self):
        """H11: a stream that opens on a P slice has nothing to predict
        from, and no list of surfaces to give a driver."""
        sps, pps = sets(reorder=0, refs=4)
        dpb = DecodedPictureBuffer()
        current = header(sps, pps, frame_num=3, poc_lsb=6)
        picture = dpb.begin(current)
        with pytest.raises(StreamError, match='P slice .* no decoded frame to predict'):
            dpb.reference_lists(picture, current)

    def test_a_bidirectional_slice_with_no_decoded_frame_is_refused(self):
        sps, pps = sets(reorder=2, refs=4)
        dpb = DecodedPictureBuffer()
        current = header(sps, pps, kind='B', frame_num=3, poc_lsb=6)
        picture = dpb.begin(current)
        with pytest.raises(StreamError, match='B slice .* no decoded frame to predict'):
            dpb.reference_lists(picture, current)

    def test_a_list_shorter_than_the_references_is_cut(self):
        sps, pps = sets(reorder=0, refs=4)
        dpb = stream(sps, pps, 3)
        current = header(sps, pps, frame_num=4, poc_lsb=8,
                         num_ref_idx_l0_active_minus1=0)
        picture = dpb.begin(current)
        l0, _ = dpb.reference_lists(picture, current)
        assert names(l0) == [3]


def b_stream(sps, pps, counts):
    """An IDR at count 0, then reference P pictures at the counts given."""
    dpb = DecodedPictureBuffer()
    decode(dpb, header(sps, pps, kind='I', idr=True), 0)
    for index, count in enumerate(counts, start=1):
        decode(dpb, header(sps, pps, frame_num=index, poc_lsb=count), count)
    return dpb


class TestInitialListsForB:
    def test_list_zero_looks_back_and_list_one_looks_forward(self):
        sps, pps = sets(reorder=4, refs=4)
        dpb = b_stream(sps, pps, (8, 16))
        current = header(sps, pps, kind='B', frame_num=3, poc_lsb=12, nal_ref_idc=0,
                         num_ref_idx_l0_active_minus1=2,
                         num_ref_idx_l1_active_minus1=2)
        picture = dpb.begin(current)
        l0, l1 = dpb.reference_lists(picture, current)
        assert names(l0) == [8, 0, 16]
        assert names(l1) == [16, 8, 0]

    def test_identical_lists_swap_the_first_two_of_list_one(self):
        sps, pps = sets(reorder=4, refs=4)
        dpb = b_stream(sps, pps, (8, 16))
        current = header(sps, pps, kind='B', frame_num=3, poc_lsb=20, nal_ref_idc=0,
                         num_ref_idx_l0_active_minus1=2,
                         num_ref_idx_l1_active_minus1=2)
        picture = dpb.begin(current)
        l0, l1 = dpb.reference_lists(picture, current)
        assert names(l0) == [16, 8, 0]
        assert names(l1) == [8, 16, 0]

    def test_long_term_references_end_both_lists(self):
        sps, pps = sets(reorder=4, refs=4)
        dpb = b_stream(sps, pps, (8,))
        decode(dpb, header(sps, pps, frame_num=2, poc_lsb=16,
                           memory_management=(Op(4, max_long_term_frame_idx_plus1=1),
                                              Op(3, difference_of_pic_nums_minus1=1,
                                                 long_term_frame_idx=0))),
               16)
        current = header(sps, pps, kind='B', frame_num=3, poc_lsb=12, nal_ref_idc=0,
                         num_ref_idx_l0_active_minus1=2,
                         num_ref_idx_l1_active_minus1=2)
        picture = dpb.begin(current)
        l0, l1 = dpb.reference_lists(picture, current)
        assert names(l0) == [8, 16, 0]
        assert names(l1) == [16, 8, 0]


class TestListModification:
    def test_a_subtraction_moves_an_older_picture_to_the_front(self):
        sps, pps = sets(reorder=0, refs=4)
        dpb = stream(sps, pps, 3)
        # picNumPred 4; 4 - (2 + 1) = 1.
        current = header(sps, pps, frame_num=4, poc_lsb=8,
                         num_ref_idx_l0_active_minus1=2,
                         ref_pic_list_modification_l0=((0, 2),))
        picture = dpb.begin(current)
        l0, _ = dpb.reference_lists(picture, current)
        assert names(l0) == [1, 3, 2]

    def test_operations_chain_from_the_last_prediction(self):
        sps, pps = sets(reorder=0, refs=4)
        dpb = stream(sps, pps, 3)
        # 4 - 3 = 1, then 1 + 1 = 2.
        current = header(sps, pps, frame_num=4, poc_lsb=8,
                         num_ref_idx_l0_active_minus1=2,
                         ref_pic_list_modification_l0=((0, 2), (1, 0)))
        picture = dpb.begin(current)
        l0, _ = dpb.reference_lists(picture, current)
        assert names(l0) == [1, 2, 3]

    def test_a_subtraction_wraps_through_max_pic_num(self):
        sps, pps = sets(reorder=0, refs=4)
        dpb = stream(sps, pps, 17)
        # Current frame_num 2: 2 - 4 wraps to 14, which is PicNum -2.
        current = header(sps, pps, frame_num=2, poc_lsb=36,
                         num_ref_idx_l0_active_minus1=1,
                         ref_pic_list_modification_l0=((0, 3),))
        picture = dpb.begin(current)
        l0, _ = dpb.reference_lists(picture, current)
        assert [f.frame_num for f in l0] == [14, 1]

    def test_a_long_term_picture_is_named_by_its_long_term_pic_num(self):
        sps, pps = sets(reorder=0, refs=4)
        dpb = stream(sps, pps, 2)
        decode(dpb, header(sps, pps, frame_num=3, poc_lsb=6,
                           memory_management=(Op(4, max_long_term_frame_idx_plus1=1),
                                              Op(3, difference_of_pic_nums_minus1=2,
                                                 long_term_frame_idx=0))),
               3)
        current = header(sps, pps, frame_num=4, poc_lsb=8,
                         num_ref_idx_l0_active_minus1=2,
                         ref_pic_list_modification_l0=((2, 0),))
        picture = dpb.begin(current)
        l0, _ = dpb.reference_lists(picture, current)
        assert names(l0) == [0, 3, 2]

    def test_naming_a_picture_that_is_not_a_reference_is_a_stream_error(self):
        sps, pps = sets(reorder=0, refs=4)
        dpb = stream(sps, pps, 1)
        current = header(sps, pps, frame_num=2, poc_lsb=4,
                         num_ref_idx_l0_active_minus1=0,
                         ref_pic_list_modification_l0=((0, 5),))
        picture = dpb.begin(current)
        with pytest.raises(StreamError, match='not a short-term reference'):
            dpb.reference_lists(picture, current)
