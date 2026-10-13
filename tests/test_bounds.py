"""Values checked before they reach a fixed-width field or a native length.

ctypes stores an out-of-range integer by dropping its high bits and raises
nothing, so a value is held to its range here, where the refusal can say which
value it was. No hardware.
"""
import pytest

from pyopengl_video import bounds
from pyopengl_video.backend import VideoError
from pyopengl_video.encoder import EncoderError


class TestWhole:
    def test_a_value_in_range_comes_back_as_an_int(self):
        assert bounds.whole(7.0, 'gop', 1, 65535) == 7
        assert isinstance(bounds.whole(7.0, 'gop', 1, 65535), int)

    def test_an_integer_of_another_type_is_taken(self):
        """A timestamp out of a NumPy array is an integer and not an int."""
        np = pytest.importorskip('numpy')
        assert bounds.whole(np.int64(7), 'gop', 1, 65535) == 7
        assert type(bounds.whole(np.uint16(7), 'gop', 1, 65535)) is int
        with pytest.raises(VideoError):
            bounds.whole(np.float64(7.5), 'gop', 1, 65535)

    @pytest.mark.parametrize('value', [0, -1, 65536, 2 ** 32 + 16])
    def test_a_value_outside_it_is_refused_by_name(self, value):
        with pytest.raises(VideoError) as caught:
            bounds.whole(value, 'gop', 1, 65535)
        assert 'gop' in str(caught.value) and str(value) in str(caught.value)
        assert '1' in str(caught.value) and '65535' in str(caught.value)

    @pytest.mark.parametrize('value', [1.5, 'wide', None, float('nan'), float('inf')])
    def test_what_is_not_a_whole_number_is_refused(self, value):
        with pytest.raises(VideoError) as caught:
            bounds.whole(value, 'gop', 1, 65535)
        assert 'gop' in str(caught.value)

    def test_a_boolean_is_not_a_count(self):
        with pytest.raises(VideoError):
            bounds.whole(True, 'gop', 1, 65535)

    def test_the_error_is_the_callers_own_kind(self):
        with pytest.raises(EncoderError):
            bounds.whole(0, 'gop', 1, 65535, error=EncoderError)


class TestFrameSize:
    def test_a_size_in_range_comes_back(self):
        assert bounds.frame_size(1920, 1080, most=(8192, 8192)) == (1920, 1080)

    @pytest.mark.parametrize('size', [(0, 240), (320, 0), (-320, 240), (8193, 240),
                                      (320, 2 ** 32 + 240)])
    def test_a_size_outside_it_is_refused(self, size):
        with pytest.raises(VideoError) as caught:
            bounds.frame_size(*size, most=(8192, 8192))
        assert 'x'.join(map(str, size)) in str(caught.value)

    def test_an_odd_size_is_refused_where_the_surface_needs_an_even_one(self):
        assert bounds.frame_size(321, 241, most=(8192, 8192)) == (321, 241)
        with pytest.raises(VideoError) as caught:
            bounds.frame_size(321, 240, most=(8192, 8192), even=True)
        assert 'even' in str(caught.value)


class TestFrameRate:
    def test_a_rate_of_whole_positive_terms_comes_back(self):
        assert bounds.frame_rate((30000, 1001)) == (30000, 1001)

    @pytest.mark.parametrize('rate', [(0, 1), (30, 0), (-30, 1), (30, -1),
                                      (2 ** 32, 1)])
    def test_a_rate_of_no_frames_is_refused(self, rate):
        with pytest.raises(VideoError) as caught:
            bounds.frame_rate(rate)
        assert 'frame rate' in str(caught.value)


class TestSpan:
    def test_a_span_inside_the_buffer_comes_back(self):
        assert bounds.span(4, 60, 64, 'the packet') == (4, 60)
        assert bounds.span(64, 0, 64, 'the packet') == (64, 0)

    @pytest.mark.parametrize('offset, length', [(0, 65), (4, 61), (65, 0), (-1, 4),
                                               (0, -1), (2 ** 40, 1)])
    def test_a_span_that_leaves_it_is_refused(self, offset, length):
        with pytest.raises(VideoError) as caught:
            bounds.span(offset, length, 64, 'the packet')
        assert 'the packet' in str(caught.value) and '64' in str(caught.value)
