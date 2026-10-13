"""The RBSP bit reader the H.264 and HEVC parsers are built on.

The reader is where a stream's values enter: an exponential-Golomb code is as
long as its writer cares to make it, so the code is bounded here, and the two
ranged readers are what hold a syntax element to the range its clause gives.
"""
import pytest

from pyopengl_video.backend import VideoError
from pyopengl_video.vaapi.h264 import (
    MAX_EXP_GOLOMB_ZEROES,
    SE_MOST,
    UE_MOST,
    BitReader,
    BitWriter,
    StreamError,
    ranged,
)


def reader(write):
    """A reader over what `write` writes."""
    writer = BitWriter()
    write(writer)
    return BitReader(writer.rbsp())


class TestStreamError:
    def test_it_is_one_of_the_packages_errors(self):
        """C9: ``except VideoError`` covers what a parser raises."""
        assert issubclass(StreamError, VideoError)
        with pytest.raises(VideoError):
            BitReader(b'').u(1)

    def test_it_is_still_a_value_error(self):
        assert issubclass(StreamError, ValueError)
        with pytest.raises(ValueError, match='runs past the end'):
            BitReader(b'').u(1)


class TestExpGolomb:
    def test_the_largest_code_number_is_read(self):
        """9.1: 31 leading zeroes, then 32 bits, is 2**32 - 2."""
        assert MAX_EXP_GOLOMB_ZEROES == 31
        found = reader(lambda writer: writer.ue(UE_MOST))
        assert found.ue() == UE_MOST == 2**32 - 2
        assert found.position == 63

    def test_a_code_of_thirty_two_leading_zeroes_is_refused(self):
        """H15: no code number has them, whatever follows."""
        data = bytes(4) + b'\x80' + bytes(4)
        with pytest.raises(StreamError, match='more than 31 leading zeroes'):
            BitReader(data).ue()

    def test_a_run_of_zeroes_is_not_read_to_its_end(self):
        found = BitReader(bytes(1000))
        with pytest.raises(StreamError, match='more than 31 leading zeroes'):
            found.ue()
        assert found.position == 32

    def test_the_signed_code_reaches_both_ends_of_its_range(self):
        found = reader(lambda writer: (writer.se(SE_MOST), writer.se(-SE_MOST)))
        assert (found.se(), found.se()) == (2**31 - 1, -(2**31 - 1))


class TestRangedReads:
    @pytest.mark.parametrize('value', [0, 7, 31])
    def test_an_unsigned_element_inside_its_range_is_its_value(self, value):
        assert reader(lambda writer: writer.ue(value)).ue_in(
            0, 31, 'seq_parameter_set_id') == value

    @pytest.mark.parametrize('value', [32, 5464, UE_MOST])
    def test_an_unsigned_element_above_its_range_names_itself(self, value):
        with pytest.raises(StreamError,
                           match=f'seq_parameter_set_id is {value}, outside 0 to 31'):
            reader(lambda writer: writer.ue(value)).ue_in(0, 31, 'seq_parameter_set_id')

    def test_an_unsigned_element_below_its_range_is_refused(self):
        with pytest.raises(StreamError, match='max_dec_frame_buffering is 3, outside 4 to 16'):
            reader(lambda writer: writer.ue(3)).ue_in(4, 16, 'max_dec_frame_buffering')

    @pytest.mark.parametrize('value', [-6, 0, 6])
    def test_a_signed_element_inside_its_range_is_its_value(self, value):
        assert reader(lambda writer: writer.se(value)).se_in(
            -6, 6, 'slice_beta_offset_div2') == value

    @pytest.mark.parametrize('value', [-7, 7, -200])
    def test_a_signed_element_outside_its_range_names_itself(self, value):
        with pytest.raises(StreamError,
                           match=f'slice_beta_offset_div2 is {value}, outside -6 to 6'):
            reader(lambda writer: writer.se(value)).se_in(-6, 6, 'slice_beta_offset_div2')

    def test_the_element_is_consumed_before_it_is_refused(self):
        """The position is past the element, as after any read."""
        found = reader(lambda writer: (writer.ue(40), writer.ue(2)))
        with pytest.raises(StreamError):
            found.ue_in(0, 31, 'seq_parameter_set_id')
        assert found.ue() == 2

    def test_a_derived_value_is_held_to_a_range_the_same_way(self):
        assert ranged(139264, 1, 139264, 'PicSizeInMbs') == 139264
        with pytest.raises(StreamError, match='PicSizeInMbs is 139265, outside 1 to 139264'):
            ranged(139265, 1, 139264, 'PicSizeInMbs')
