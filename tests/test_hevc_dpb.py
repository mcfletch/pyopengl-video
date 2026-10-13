"""The HEVC decoded picture buffer, driven by parsed headers with no driver.

Each picture's surface is a label the test chooses, so what the buffer puts in
a reference list or outputs can be named.
"""
import dataclasses
import logging
from pathlib import Path

import pytest

from pyopengl_video.mp4 import split_annexb
from pyopengl_video.vaapi.h264 import StreamError
from pyopengl_video.vaapi.hevc_dpb import LONG_TERM, SHORT_TERM, DecodedPictureBuffer
from pyopengl_video.vaapi.hevc_parse import (
    NAL_BLA_W_LP,
    NAL_CRA,
    NAL_IDR_W_RADL,
    NAL_PPS,
    NAL_RADL_N,
    NAL_RASL_N,
    NAL_SPS,
    NAL_TRAIL_N,
    ShortTermRefPicSet,
    parse_nal,
    parse_pps,
    parse_slice_header,
    parse_sps,
)
from tests.hevc_reference import FIXTURES
from tests.hevc_syntax import Stream

DATA = Path(__file__).parent / 'data' / 'decode'


def surfaces(frames):
    return [frame.surface for frame in frames]


class TestPictureOrder:
    def stream(self):
        return Stream(log2_max_poc_lsb_minus4=0)

    def test_an_idr_picture_is_zero(self):
        stream = self.stream()
        stream.idr()
        assert stream.pictures['idr'].poc == 0

    def test_the_count_carries_past_the_lsb_range(self):
        stream = self.stream()
        stream.idr()
        for name, lsb in (('a', 6), ('b', 12), ('c', 2), ('d', 9), ('e', 15), ('f', 5)):
            stream.decode(name, lsb)
        assert [stream.pictures[name].poc for name in 'abcdef'] == [6, 12, 18, 25, 31, 37]

    def test_an_lsb_far_above_the_last_is_before_it(self):
        stream = self.stream()
        stream.idr()
        for name, lsb in (('a', 6), ('b', 12), ('c', 2), ('d', 13)):
            stream.decode(name, lsb)
        assert stream.pictures['c'].poc == 18
        assert stream.pictures['d'].poc == 13

    def test_a_sub_layer_non_reference_picture_is_not_counted_from(self):
        stream = self.stream()
        stream.idr()
        stream.decode('a', 7)
        stream.decode('b', 14, nal_type=NAL_TRAIL_N)
        stream.decode('c', 1)
        assert stream.pictures['b'].poc == 14
        # Counted from 7, where 1 is behind; from 14 it would have wrapped to 17.
        assert stream.pictures['c'].poc == 1

    def test_a_picture_of_a_higher_sub_layer_is_not_counted_from(self):
        stream = Stream(log2_max_poc_lsb_minus4=0, sub_layers_minus1=1)
        stream.idr()
        stream.decode('a', 7)
        stream.decode('b', 14, temporal_id=1)
        stream.decode('c', 1)
        assert stream.pictures['c'].poc == 1

    def test_a_cra_picture_in_mid_stream_keeps_counting(self):
        stream = self.stream()
        stream.idr()
        for name, lsb in (('a', 6), ('b', 12)):
            stream.decode(name, lsb)
        stream.decode('cra', 2, nal_type=NAL_CRA)
        assert stream.pictures['cra'].poc == 18
        assert not stream.pictures['cra'].no_rasl_output

    def test_a_cra_picture_decoding_starts_at_counts_from_its_lsb(self):
        stream = self.stream()
        stream.decode('cra', 9, nal_type=NAL_CRA)
        stream.decode('a', 13, before=(-4,))
        assert stream.pictures['cra'].poc == 9
        assert stream.pictures['cra'].no_rasl_output
        assert stream.pictures['a'].poc == 13


class TestReferencePictureSet:
    def pyramid(self):
        """An IDR at 0, P at 8, B at 4, then b at 2."""
        stream = Stream()
        stream.idr(0)
        stream.decode(8, 8, before=(-8,))
        stream.decode(4, 4, before=(-4,), after=(4,))
        stream.decode(2, 2, before=(-2,), after=(2, 6), nal_type=NAL_TRAIL_N)
        return stream

    def test_the_pictures_before_and_after(self):
        stream = self.pyramid()
        rps = stream.pictures[2].rps
        assert surfaces(rps.st_curr_before) == [0]
        assert surfaces(rps.st_curr_after) == [4, 8]
        assert rps.foll == ()
        assert rps.missing == ()

    def test_before_is_nearest_first(self):
        stream = self.pyramid()
        picture = stream.buffer.begin(stream.header(12, before=(-4, -8, -12)))
        assert surfaces(picture.rps.st_curr_before) == [8, 4, 0]

    def test_pictures_kept_and_not_predicted_from(self):
        stream = self.pyramid()
        picture = stream.buffer.begin(stream.header(6, before=(-2, -6), after=(2,),
                                                    unused=(-6,)))
        assert surfaces(picture.rps.st_curr_before) == [4]
        assert surfaces(picture.rps.st_curr_after) == [8]
        assert surfaces(picture.rps.foll) == [0]
        assert surfaces(picture.rps.frames) == [4, 8, 0]

    def test_a_picture_the_set_leaves_out_is_no_longer_a_reference(self):
        stream = self.pyramid()
        stream.buffer.begin(stream.header(12, before=(-4,)))
        marks = {frame.surface: frame.reference for frame in stream.buffer.frames}
        assert marks[8] == SHORT_TERM
        assert all(marks[name] is None for name in marks if name != 8)
        assert surfaces(stream.buffer.reference_frames()) == [8]

    def test_the_decoded_picture_is_a_short_term_reference(self):
        stream = self.pyramid()
        assert {frame.surface: frame.reference for frame in stream.buffer.frames}[2] == SHORT_TERM

    def test_an_idr_picture_leaves_no_reference(self):
        stream = self.pyramid()
        stream.idr('second')
        assert stream.pictures['second'].rps.frames == []
        assert surfaces(stream.buffer.reference_frames()) == ['second']

    def test_a_reference_not_held_has_the_nearest_stand_in(self, caplog):
        stream = self.pyramid()
        with caplog.at_level(logging.WARNING):
            picture = stream.buffer.begin(stream.header(12, before=(-2, -4)))
        assert picture.rps.missing == (10,)
        assert surfaces(picture.rps.st_curr_before) == [8, 8]
        assert 'picture 10' in caplog.text

    def test_a_stand_in_is_kept_while_the_picture_is_decoded(self):
        """Picture 0 was shown and is in no set: standing in makes it a
        reference again, or its surface would be free to decode into."""
        stream = Stream(ordering=((2, 0, 0),))
        assert stream.idr(0) == [0]
        picture = stream.buffer.begin(stream.header(4, before=(-2,)))
        assert picture.rps.missing == (2,)
        assert surfaces(picture.rps.st_curr_before) == [0]
        assert stream.held() == ['0']
        assert surfaces(stream.buffer.reference_frames()) == [0]

    def test_a_picture_decoding_restarts_at_has_no_stand_in(self):
        """H10: a broken link picture's set names the picture before it, which
        leaves the buffer with every other. A frame on its way out stands in
        for nothing, so nothing the picture names is outside the buffer."""
        stream = Stream()
        stream.idr(0)
        stream.decode(1, 1, before=(-1,))
        picture = stream.buffer.begin(stream.header(
            2, nal_type=NAL_BLA_W_LP, kind='I', no_output=1, before=(-1,)))
        assert picture.rps.missing == (1,)
        assert picture.rps.frames == []
        assert stream.held() == []
        assert stream.buffer.outputs() == []

    def test_what_a_picture_predicts_from_stays_in_the_buffer(self):
        """A stand-in included: every frame of a set is one the buffer holds
        when the picture begins."""
        stream = self.pyramid()
        picture = stream.buffer.begin(stream.header(12, before=(-2, -4)))
        assert picture.rps.missing
        assert all(frame in stream.buffer.frames for frame in picture.rps.frames)

    def test_an_order_count_past_thirty_two_bits_is_refused(self):
        """8.3.1: PicOrderCntVal fits 32 signed bits, as libva's field does."""
        stream = Stream(log2_max_poc_lsb_minus4=12)
        stream.idr(0)
        buffer = stream.buffer
        # As after 32767 wraps of the LSB.
        buffer._prev_msb, buffer._prev_lsb = 2**31 - 65536, 40000  # noqa: SLF001 - the count so far
        stream.decode('last', 65535)
        assert stream.pictures['last'].poc == 2**31 - 1
        with pytest.raises(StreamError, match='PicOrderCntVal is 2147483648'):
            stream.buffer.begin(stream.header(0))

    def test_a_reference_not_held_with_nothing_to_stand_in(self):
        stream = Stream()
        stream.decode('cra', 8, nal_type=NAL_CRA)
        stream.buffer.frames.clear()
        slice_header = stream.header(12, before=(-4,))
        picture = stream.buffer.begin(slice_header)
        assert picture.rps.missing == (8,)
        assert picture.rps.st_curr_before == ()
        with pytest.raises(StreamError, match='no picture to predict from'):
            stream.buffer.reference_lists(picture, slice_header)

    def test_a_kept_picture_not_held_is_nothing(self):
        stream = self.pyramid()
        picture = stream.buffer.begin(stream.header(12, before=(-2, -4), unused=(-2,)))
        assert picture.rps.missing == ()
        assert surfaces(picture.rps.st_curr_before) == [8]


class TestLongTermPictures:
    def stream(self):
        stream = Stream(log2_max_poc_lsb_minus4=0, long_term=())
        stream.idr(0)
        stream.decode(8, 8, before=(-8,))
        stream.decode(16, 0, before=(-8, -16))
        stream.decode(20, 4, before=(-4, -12, -20))
        return stream

    def test_a_picture_named_by_its_lsb(self):
        stream = self.stream()
        picture = stream.buffer.begin(stream.header(
            8, before=(-4,), long_term={'pics': [(8, 1, None)]}))
        assert picture.poc == 24
        assert surfaces(picture.rps.lt_curr) == [8]
        assert surfaces(picture.rps.st_curr_before) == [20]
        marks = {frame.surface: frame.reference for frame in stream.buffer.frames}
        assert marks == {8: LONG_TERM, 16: None, 20: SHORT_TERM}

    def test_a_picture_named_with_its_msb_cycle(self):
        """Pictures 0 and 16 share an LSB; the cycle says which. Each entry's
        cycle is coded as the step from the entry before."""
        stream = self.stream()
        picture = stream.buffer.begin(stream.header(
            8, before=(-4,), long_term={'pics': [(0, 1, 0), (0, 1, 1)]}))
        assert surfaces(picture.rps.lt_curr) == [16, 0]
        assert picture.rps.missing == ()

    def test_a_long_term_picture_kept_for_later(self):
        stream = self.stream()
        picture = stream.buffer.begin(stream.header(
            8, before=(-4,), long_term={'pics': [(8, 0, None)]}))
        assert picture.rps.lt_curr == ()
        assert surfaces(picture.rps.foll) == [8]
        assert surfaces(picture.rps.frames) == [20, 8]

    def test_a_long_term_picture_stays_one(self):
        stream = self.stream()
        stream.decode(24, 8, before=(-4,), long_term={'pics': [(8, 1, None)]})
        assert {frame.surface: frame.reference
                for frame in stream.buffer.reference_frames()} == {
            8: LONG_TERM, 20: SHORT_TERM, 24: SHORT_TERM}
        # Picture 8 is not found as a short-term picture again.
        picture = stream.buffer.begin(stream.header(12, before=(-4, -20)))
        assert picture.poc == 28
        assert picture.rps.missing == (8,)

    def test_long_term_pictures_come_last_in_both_lists(self):
        stream = self.stream()
        slice_header = stream.header(8, before=(-4, -8), kind='B', refs=(2, 2),
                                     long_term={'pics': [(8, 1, None)]})
        picture = stream.buffer.begin(slice_header)
        list0, list1 = stream.buffer.reference_lists(picture, slice_header)
        assert surfaces(list0) == [20, 16, 8]
        assert surfaces(list1) == [20, 16, 8]


class TestReferenceLists:
    def lists(self, pps=None, **options):
        stream = Stream(pps=pps)
        stream.idr(0)
        stream.decode(8, 8, before=(-8,))
        stream.decode(4, 4, before=(-4,), after=(4,))
        stream.decode(6, 6, before=(-2, -6), after=(2,))
        options.setdefault('before', (-1, -5))
        options.setdefault('after', (1, 3))
        slice_header = stream.header(5, **options)
        picture = stream.buffer.begin(slice_header)
        list0, list1 = stream.buffer.reference_lists(picture, slice_header)
        return surfaces(list0), surfaces(list1)

    def test_list_0_is_before_then_after_and_list_1_the_reverse(self):
        assert self.lists(refs=(3, 3)) == ([4, 0, 6, 8], [6, 8, 4, 0])

    def test_a_list_is_cut_to_its_active_length(self):
        assert self.lists(refs=(1, 0)) == ([4, 0], [6])

    def test_the_pps_gives_the_length_a_slice_does_not(self):
        assert self.lists(pps={'refs': (2, 1)}) == ([4, 0, 6], [6, 8])

    def test_a_list_longer_than_the_set_repeats_it(self):
        assert self.lists(before=(-1,), after=(1,), refs=(4, 2)) == (
            [4, 6, 4, 6, 4], [6, 4, 6])

    def test_a_modified_list(self):
        found = self.lists(pps={'lists_modification': 1}, refs=(2, 1),
                           modification=((3, 3, 0), (1, 2)))
        assert found == ([8, 8, 4], [8, 4])

    def test_one_list_modified_and_one_not(self):
        found = self.lists(pps={'lists_modification': 1}, refs=(1, 1),
                           modification=(None, (2, 0)))
        assert found == ([4, 0], [4, 6])

    def test_a_p_slice_has_no_list_1(self):
        assert self.lists(kind='P', refs=(2, 0)) == ([4, 0, 6], [])

    def test_an_i_slice_has_no_lists(self):
        assert self.lists(kind='I') == ([], [])

    def test_a_modification_past_the_pictures_is_refused(self):
        with pytest.raises(StreamError, match='list_entry_l0 is 3, outside 0 to 2'):
            self.lists(pps={'lists_modification': 1}, before=(-1, -5), after=(1,),
                       refs=(1, 0), modification=((3, 0), None))

    def test_a_header_made_with_an_entry_past_the_list_is_refused(self):
        """No parsed header names one; the buffer holds a header to its list
        however it was made."""
        stream = Stream()
        stream.idr(0)
        stream.decode(1, 1, before=(-1,))
        slice_header = dataclasses.replace(stream.header(2, before=(-1, -2)),
                                           list_entry_l0=(3,))
        picture = stream.buffer.begin(slice_header)
        with pytest.raises(StreamError, match='entry 3'):
            stream.buffer.reference_lists(picture, slice_header)

    def test_a_slice_with_nothing_to_predict_from_is_refused(self):
        stream = Stream()
        stream.idr(0)
        slice_header = stream.header(4, kind='P')
        picture = stream.buffer.begin(slice_header)
        with pytest.raises(StreamError, match='no picture to predict from'):
            stream.buffer.reference_lists(picture, slice_header)


class TestSkippedPictures:
    def test_leading_pictures_of_the_first_cra_are_skipped(self):
        stream = Stream()
        assert stream.decode('cra', 8, nal_type=NAL_CRA) == []
        assert stream.decode('rasl', 4, nal_type=NAL_RASL_N, before=(-8,), after=(4,)) == []
        assert stream.pictures['rasl'].skipped
        assert not stream.pictures['rasl'].output
        assert stream.held() == ['cra']

    def test_decodable_leading_pictures_are_decoded(self):
        stream = Stream()
        stream.decode('cra', 8, nal_type=NAL_CRA)
        stream.decode('radl', 6, nal_type=NAL_RADL_N, after=(2,))
        assert not stream.pictures['radl'].skipped
        assert stream.pictures['radl'].poc == 6

    def test_leading_pictures_of_a_later_cra_are_decoded(self):
        stream = Stream()
        stream.idr(0)
        stream.decode(4, 4, before=(-4,))
        stream.decode('cra', 12, nal_type=NAL_CRA, kind='I', before=(-8,), unused=(-8,))
        stream.decode('rasl', 8, nal_type=NAL_RASL_N, before=(-4,), after=(4,))
        picture = stream.pictures['rasl']
        assert not picture.skipped
        assert surfaces(picture.rps.st_curr_before) == [4]
        assert surfaces(picture.rps.st_curr_after) == ['cra']

    def test_leading_pictures_of_a_bla_picture_are_skipped(self):
        stream = Stream()
        stream.idr(0)
        stream.decode('bla', 12, nal_type=NAL_BLA_W_LP)
        stream.decode('rasl', 8, nal_type=NAL_RASL_N, before=(-8,), after=(4,))
        assert stream.pictures['bla'].no_rasl_output
        assert stream.pictures['rasl'].skipped

    def test_pictures_before_the_first_random_access_point_are_skipped(self):
        stream = Stream()
        assert stream.decode('a', 4, before=(-4,)) == []
        assert stream.pictures['a'].skipped
        stream.decode('cra', 8, nal_type=NAL_CRA)
        assert not stream.pictures['cra'].skipped
        assert stream.held() == ['cra']

    def test_a_reset_starts_again_at_the_next_random_access_point(self):
        stream = Stream()
        stream.idr(0)
        stream.decode(8, 8, before=(-8,))
        stream.buffer.reset()
        assert stream.buffer.frames == []
        stream.decode('late', 12, before=(-4,))
        stream.decode('cra', 20, nal_type=NAL_CRA)
        stream.decode('rasl', 16, nal_type=NAL_RASL_N, before=(-8,), after=(4,))
        stream.decode('after', 24, before=(-4,))
        skipped = {name: picture.skipped for name, picture in stream.pictures.items()}
        assert skipped == {0: False, 8: False, 'late': True, 'cra': False, 'rasl': True,
                           'after': False}
        assert stream.pictures['cra'].poc == 20
        assert surfaces(stream.pictures['after'].rps.st_curr_before) == ['cra']

    def test_an_end_of_sequence_shows_what_waits_and_starts_again(self):
        stream = Stream()
        stream.idr(0)
        stream.decode(8, 8, before=(-8,))
        assert surfaces(stream.buffer.end_of_sequence()) == [0, 8]
        stream.decode('cra', 5, nal_type=NAL_CRA)
        stream.decode('rasl', 3, nal_type=NAL_RASL_N, before=(-3,), after=(2,))
        assert stream.pictures['cra'].no_rasl_output
        assert stream.pictures['cra'].poc == 5
        assert stream.pictures['rasl'].skipped


class TestOutput:
    def test_without_reordering_each_picture_is_shown_as_it_is_decoded(self):
        stream = Stream(ordering=((2, 0, 0),))
        assert stream.idr(0) == [0]
        assert stream.decode(1, 1, before=(-1,)) == [1]
        assert stream.decode(2, 2, before=(-1,)) == [2]

    def test_reordered_pictures_wait_for_those_before_them(self):
        stream = Stream(ordering=((4, 2, 0),))
        assert stream.idr(0) == []
        assert stream.decode(8, 8, before=(-8,)) == []
        assert stream.decode(4, 4, before=(-4,), after=(4,)) == [0]
        assert stream.decode(2, 2, before=(-2,), after=(2, 6), nal_type=NAL_TRAIL_N) == [2]
        assert stream.decode(6, 6, before=(-2,), after=(2,), nal_type=NAL_TRAIL_N) == [4]
        assert surfaces(stream.buffer.flush()) == [6, 8]
        assert stream.buffer.frames == []

    def test_a_shown_picture_stays_while_it_is_a_reference(self):
        stream = Stream(ordering=((2, 0, 0),))
        stream.idr(0)
        stream.decode(1, 1, before=(-1,))
        assert stream.held() == ['0', '1']
        stream.decode(2, 2, before=(-1,))
        assert stream.held() == ['1', '2']

    def test_a_picture_marked_not_for_output_is_never_shown(self):
        stream = Stream(ordering=((2, 0, 0),), pps={'output_flag_present': 1})
        stream.idr(0)
        assert stream.decode(1, 1, before=(-1,), pic_output=0) == []
        assert not stream.pictures[1].output
        assert stream.decode(2, 2, before=(-1,)) == [2]
        assert surfaces(stream.pictures[2].rps.st_curr_before) == [1]
        assert surfaces(stream.buffer.flush()) == []

    def test_an_idr_picture_shows_everything_before_it(self):
        stream = Stream(ordering=((4, 2, 0),))
        stream.idr(0)
        stream.decode(8, 8, before=(-8,))
        assert stream.idr('second') == [0, 8]
        assert stream.held() == ['second']

    def test_an_idr_picture_may_discard_what_was_not_shown(self):
        stream = Stream(ordering=((4, 2, 0),))
        stream.idr(0)
        stream.decode(8, 8, before=(-8,))
        assert stream.decode('second', nal_type=NAL_IDR_W_RADL, kind='I', no_output=1) == []
        assert stream.held() == ['second']

    def test_a_full_buffer_outputs_before_the_next_picture(self):
        """Three pictures held where three are allowed, two of them waiting
        and two may wait: the size is what bumps."""
        stream = Stream(ordering=((2, 2, 0),))
        stream.idr(0)
        stream.decode(8, 8, before=(-8,))
        assert stream.decode(4, 4, before=(-4,), after=(4,)) == [0]
        stream.buffer.begin(stream.header(2, before=(-2,), after=(2,)))
        assert surfaces(stream.buffer.outputs()) == [4, 8]
        assert stream.held() == ['0', '4']

    def keeping(self, stream, poc, deltas):
        """A header for picture `poc` whose set keeps the pictures at `deltas`.

        Made by replacing the parsed set, because the parser holds a set to
        the buffer's size and these are larger.
        """
        return dataclasses.replace(
            stream.header(poc, before=deltas[:1]),
            short_term_ref_pic_set=ShortTermRefPicSet(deltas, (1,) * len(deltas)))

    def test_a_buffer_full_of_references_takes_one_picture_more(self):
        """Two pictures allowed and two kept as references: the third has
        nothing to push out, and still has a surface."""
        stream = Stream(ordering=((1, 0, 0),))
        stream.idr(0)
        stream.decode(1, 1, before=(-1,))
        picture = stream.buffer.begin(self.keeping(stream, 2, (-1, -2)))
        assert surfaces(stream.buffer.outputs() + stream.buffer.finish(picture, 2)) == [2]
        assert stream.held() == ['0', '1', '2']

    def test_more_references_than_the_buffer_holds_are_refused(self):
        stream = Stream(ordering=((1, 0, 0),))
        stream.idr(0)
        stream.decode(1, 1, before=(-1,))
        picture = stream.buffer.begin(self.keeping(stream, 2, (-1, -2)))
        stream.buffer.finish(picture, 2)
        with pytest.raises(StreamError, match='keeps 3 reference pictures in a decoded '
                                              'picture buffer of 2'):
            stream.buffer.begin(self.keeping(stream, 3, (-1, -2, -3)))

    def test_a_picture_that_has_waited_its_latency_is_shown(self):
        """Two may wait and none longer than two pictures: decoding 2 shows
        1 by the count, then 2 and 8 because 8 has waited through two."""
        limited = Stream(ordering=((4, 2, 1),))
        free = Stream(ordering=((4, 2, 0),))
        found = {}
        for name, stream in (('limited', limited), ('free', free)):
            assert stream.sps.max_latency_pictures == (2 if name == 'limited' else None)
            stream.idr(0)
            stream.decode(8, 8, before=(-8,))
            found[name] = [stream.decode(1, 1, before=(-1,), after=(7,)),
                           stream.decode(2, 2, before=(-1,), after=(6,))]
        assert found['free'] == [[0], [1]]
        assert found['limited'] == [[0], [1, 2, 8]]

    def test_latency_counts_only_pictures_shown_first(self):
        stream = Stream(ordering=((4, 2, 1),))
        stream.idr(0)
        stream.decode(1, 1, before=(-1,))
        stream.decode(2, 2, before=(-1,))
        assert all(frame.latency == 0 for frame in stream.buffer.frames)


def pictures_of(name, start=0):
    """The first slice segment header of each picture of a fixture, from
    picture `start` on, with the parameter sets read from the whole stream."""
    sequences, pictures = {}, {}
    found = []
    for data in split_annexb((DATA / f'{name}.h265').read_bytes()):
        unit = parse_nal(data)
        if unit.nal_unit_type == NAL_SPS:
            sps = parse_sps(unit.rbsp)
            sequences[sps.sps_seq_parameter_set_id] = sps
        elif unit.nal_unit_type == NAL_PPS:
            pps = parse_pps(unit.rbsp)
            pictures[pps.pps_pic_parameter_set_id] = pps
        elif unit.is_slice:
            slice_header = parse_slice_header(unit, sequences, pictures)
            if slice_header.first_slice_segment_in_pic_flag:
                found.append(slice_header)
    return found[start:]


class TestFixtureStreams:
    """Every libx265 fixture, through the buffer from its headers alone."""

    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_every_picture_is_shown_once_in_order(self, name):
        buffer = DecodedPictureBuffer()
        shown, groups = [], []
        for index, slice_header in enumerate(pictures_of(name)):
            picture = buffer.begin(slice_header)
            assert not picture.skipped
            assert picture.rps.missing == ()
            if picture.no_rasl_output:
                groups.append([])
            for frame in picture.rps.frames:
                assert frame.surface is not None
            list0, list1 = buffer.reference_lists(picture, slice_header)
            assert len(list0) == (slice_header.num_ref_idx_l0_active_minus1 + 1
                                  if slice_header.kind != 'I' else 0)
            assert len(list1) == (slice_header.num_ref_idx_l1_active_minus1 + 1
                                  if slice_header.kind == 'B' else 0)
            shown += buffer.outputs() + buffer.finish(picture, surface=index,
                                                      payload=len(groups))
            assert len(buffer.frames) <= slice_header.sps.dpb_size
        shown += buffer.flush()
        assert sorted(surfaces(shown)) == list(range(len(pictures_of(name))))
        order = [(frame.payload, frame.poc) for frame in shown]
        assert order == sorted(order)

    def test_starting_at_a_cra_picture_skips_its_leading_pictures(self):
        headers = pictures_of('hevc_open_gop')
        start = next(index for index, slice_header in enumerate(headers)
                     if slice_header.nal_unit_type == NAL_CRA)
        buffer = DecodedPictureBuffer()
        skipped, shown = [], []
        for index, slice_header in enumerate(headers[start:], start):
            picture = buffer.begin(slice_header)
            shown += buffer.outputs()
            if picture.skipped:
                skipped.append(index)
                continue
            assert picture.rps.missing == ()
            shown += buffer.finish(picture, surface=index)
        shown += buffer.flush()
        leading = [index for index, slice_header in enumerate(headers[start:start + 4], start)
                   if slice_header.rasl]
        assert skipped == leading
        assert len(skipped) == 3
        assert shown[0].surface == start
        assert len(shown) == len(headers) - start - 3
