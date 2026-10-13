"""What the NVDEC backends read of a stream for themselves, with no driver.

The picture size before the first sample, and the colour description
NVIDIA's parser leaves out, for each codec: from the bytes a container keeps
apart from its samples and from the samples themselves.
"""
from pathlib import Path

import pytest

from pyopengl_video import obu
from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.colour import MATRIX_BT601, MATRIX_BT709, MATRIX_BT2020, Colour
from pyopengl_video.ivf import IVFReader
from pyopengl_video.mp4 import MP4Reader
from pyopengl_video.nvdec.description import READERS, Description, describe
from pyopengl_video.vaapi.h264 import StreamError
from tests.vp9_syntax import frame as vp9_frame
from tests.vp9_syntax import show_existing, superframe

DATA = Path(__file__).parent / 'data' / 'decode'


def annexb(name):
    return AnnexBReader(DATA / name)


def first_sample(name):
    with IVFReader(DATA / name) as stream:
        return stream.sample(0).data


def test_there_is_a_reader_for_every_codec_the_backend_decodes():
    assert set(READERS) == {'h264', 'hevc', 'vp9', 'av1'}


class TestParameterSets:
    def test_an_h264_streams_size_is_its_visible_picture(self):
        with annexb('high_bpyramid.h264') as stream:
            found = describe('h264', stream.parameter_sets())
        assert found == Description((120, 68), Colour(matrix=MATRIX_BT601))

    def test_an_hevc_streams_colour_includes_where_chroma_sits(self):
        with annexb('hevc_large_main10.h265') as stream:
            found = describe('hevc', stream.parameter_sets())
        assert found.size == (178, 150)
        assert found.colour == Colour(matrix=MATRIX_BT2020, primaries=9, transfer=16,
                                      chroma_location=2)

    def test_the_sets_are_found_in_a_key_frames_sample(self):
        with annexb('hevc_large_bpyramid.h265') as stream:
            found = describe('hevc', stream.sample(0, parameter_sets=True).data)
        assert found.size == (180, 148)

    def test_a_sample_with_no_set_describes_nothing(self):
        with annexb('hevc_large_bpyramid.h265') as stream:
            assert describe('hevc', stream.sample(3).data) is None
        with annexb('high_bpyramid.h264') as stream:
            assert describe('h264', stream.sample(3).data) is None

    def test_an_mp4s_parameter_sets_are_read_as_a_streams_are(self):
        with MP4Reader(DATA / 'hevc_large_bpyramid.mp4') as movie:
            found = describe('hevc', movie.video.parameter_sets())
        assert found.size == (180, 148)

    @pytest.mark.parametrize('codec, unit', [
        ('h264', b'\x00\x00\x00\x01\x67\xff'), ('hevc', b'\x00\x00\x00\x01\x42\x01\xff')])
    def test_a_set_cut_short_is_a_stream_error(self, codec, unit):
        with pytest.raises(StreamError):
            describe(codec, unit)


class TestAV1:
    def test_the_size_and_colour_come_from_the_sequence_header(self):
        found = describe('av1', first_sample('av1_large_altref.ivf'))
        assert found == Description((150, 146), Colour(matrix=MATRIX_BT601))

    def test_an_mp4s_configuration_is_read_as_a_temporal_units_is(self):
        with MP4Reader(DATA / 'av1_large_altref.mp4') as movie:
            found = describe('av1', movie.video.parameter_sets())
        assert found.size == (150, 146)

    def test_a_stream_with_no_chroma_is_full_range_as_it_states(self):
        assert describe('av1', first_sample('av1_large_monochrome.ivf')).colour.full_range

    def test_a_temporal_unit_with_no_sequence_header_describes_nothing(self):
        sample = obu.without(first_sample('av1_large_altref.ivf'), obu.SEQUENCE_HEADER)
        assert describe('av1', sample) is None

    def test_bytes_that_are_no_obu_are_a_stream_error(self):
        with pytest.raises(StreamError):
            describe('av1', b'\xff' * 12)


class TestVP9:
    def test_a_key_frame_states_its_size_and_colour(self):
        found = describe('vp9', vp9_frame(size=(1280, 720), color_space=2, color_range=1))
        assert found == Description((1280, 720), Colour(
            matrix=MATRIX_BT709, full_range=True, primaries=1, transfer=1, chroma_location=2))

    def test_a_fixtures_first_sample_is_read(self):
        assert describe('vp9', first_sample('vp9_large_odd.ivf')).size == (151, 145)

    def test_a_key_frame_is_found_inside_a_superframe(self):
        sample = superframe(vp9_frame(size=(320, 240), show_frame=0), show_existing(0))
        assert describe('vp9', sample).size == (320, 240)

    def test_a_frame_that_is_no_key_frame_describes_nothing(self):
        assert describe('vp9', vp9_frame(frame_type=1)) is None
        assert describe('vp9', show_existing(3)) is None

    def test_bytes_that_are_no_frame_are_a_stream_error(self):
        with pytest.raises(StreamError):
            describe('vp9', b'not a frame')


@pytest.mark.parametrize('codec', sorted(READERS))
def test_no_bytes_describe_nothing(codec):
    assert describe(codec, b'') is None
