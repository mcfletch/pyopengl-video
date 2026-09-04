"""The group-of-pictures and decoded-picture-buffer bookkeeping.

libva codes the picture it is told to code: which pictures are references, what
each predicts from, and the frame numbers and picture order counts that say so
are the caller's to decide.  This is that decision, with no libva in it.
"""
from itertools import pairwise

import pytest

from pyopengl_video.vaapi.h264 import GroupOfPictures, Picture


def run(gop, count, **changes):
    """`count` pictures from a fresh group, as a list."""
    structure = GroupOfPictures(gop=gop, **changes)
    return [structure.next_picture() for _ in range(count)]


class TestTheFirstPicture:
    def test_a_recording_opens_with_an_idr(self):
        first = GroupOfPictures(gop=30).next_picture()
        assert first.idr
        assert first.intra, 'an IDR is coded without prediction'

    def test_it_numbers_itself_zero(self):
        first = GroupOfPictures(gop=30).next_picture()
        assert first.frame_num == 0
        assert first.poc == 0

    def test_it_predicts_from_nothing(self):
        assert GroupOfPictures(gop=30).next_picture().references == ()

    def test_it_is_a_reference_for_what_follows(self):
        assert GroupOfPictures(gop=30).next_picture().reference


class TestPredictedPictures:
    def test_the_rest_of_a_group_predicts_forward(self):
        pictures = run(gop=8, count=8)
        assert [picture.idr for picture in pictures] == [True] + [False] * 7
        assert not any(picture.intra for picture in pictures[1:])

    def test_each_predicts_from_the_one_before(self):
        pictures = run(gop=8, count=4)
        for earlier, later in pairwise(pictures):
            assert later.references == (earlier.frame_num,)

    def test_the_frame_number_counts_reference_pictures(self):
        assert [picture.frame_num for picture in run(gop=8, count=5)] == [
            0, 1, 2, 3, 4]

    def test_the_picture_order_count_advances_by_two_a_frame(self):
        assert [picture.poc for picture in run(gop=8, count=4)] == [0, 2, 4, 6]


class TestGroupBoundaries:
    def test_a_new_group_starts_at_the_stated_length(self):
        assert [picture.idr for picture in run(gop=4, count=9)] == [
            True, False, False, False, True, False, False, False, True]

    def test_the_counters_restart_at_each_group(self):
        pictures = run(gop=4, count=6)
        assert [picture.frame_num for picture in pictures] == [0, 1, 2, 3, 0, 1]
        assert [picture.poc for picture in pictures] == [0, 2, 4, 6, 0, 2]

    def test_a_new_group_predicts_from_nothing(self):
        assert run(gop=4, count=5)[4].references == ()

    def test_consecutive_groups_are_told_apart(self):
        # Two IDRs in a row with the same idr_pic_id would be one picture
        # repeated as far as a decoder is concerned.
        idrs = [picture.idr_pic_id for picture in run(gop=1, count=4)]
        assert all(a != b for a, b in pairwise(idrs))

    def test_every_picture_is_an_idr_at_a_group_of_one(self):
        assert all(picture.idr for picture in run(gop=1, count=5))

    @pytest.mark.parametrize('gop', [0, -1])
    def test_a_group_length_below_one_is_refused(self, gop):
        with pytest.raises(ValueError):
            GroupOfPictures(gop=gop)


class TestForcedKeyFrames:
    def test_a_forced_key_frame_starts_a_group_early(self):
        structure = GroupOfPictures(gop=100)
        structure.next_picture()
        structure.next_picture()
        forced = structure.next_picture(force_idr=True)
        assert forced.idr
        assert forced.frame_num == 0
        assert forced.poc == 0
        assert forced.references == ()

    def test_the_group_is_measured_from_the_forced_frame(self):
        structure = GroupOfPictures(gop=4)
        structure.next_picture()
        structure.next_picture()
        structure.next_picture(force_idr=True)
        assert [structure.next_picture().idr for _ in range(4)] == [
            False, False, False, True]


class TestTheDecodedPictureBuffer:
    def test_it_holds_no_more_than_the_sequence_declares(self):
        structure = GroupOfPictures(gop=30, max_num_ref_frames=2)
        for _ in range(6):
            picture = structure.next_picture()
        assert len(picture.references) <= 2

    def test_the_most_recent_reference_comes_first(self):
        structure = GroupOfPictures(gop=30, max_num_ref_frames=2)
        pictures = [structure.next_picture() for _ in range(4)]
        assert pictures[3].references == (2, 1), (
            'reference list zero is ordered by descending frame number')

    def test_an_idr_empties_it(self):
        structure = GroupOfPictures(gop=3, max_num_ref_frames=2)
        [structure.next_picture() for _ in range(3)]
        assert structure.next_picture().references == ()

    @pytest.mark.parametrize('count', [0, -1])
    def test_a_buffer_that_holds_nothing_is_refused(self, count):
        with pytest.raises(ValueError):
            GroupOfPictures(gop=30, max_num_ref_frames=count)


class TestFrameNumberWrapping:
    def test_the_frame_number_wraps_at_the_width_the_sequence_declares(self):
        # log2_max_frame_num is 8, so frame_num counts to 255 and starts again.
        structure = GroupOfPictures(gop=1000, log2_max_frame_num=8)
        numbers = [structure.next_picture().frame_num for _ in range(258)]
        assert numbers[255] == 255
        assert numbers[256] == 0
        assert numbers[257] == 1

    def test_the_picture_order_count_does_not_wrap_with_it(self):
        # VA-API is handed the full count and derives the field the slice
        # header carries, so this one keeps counting.
        structure = GroupOfPictures(gop=1000, log2_max_frame_num=8)
        pictures = [structure.next_picture() for _ in range(258)]
        assert pictures[257].poc == 257 * 2


class TestPictureRecord:
    def test_a_picture_says_what_it_is(self):
        picture = GroupOfPictures(gop=30).next_picture()
        assert isinstance(picture, Picture)
        assert repr(picture)
