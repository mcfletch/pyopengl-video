"""oneVPL's codecs: what the runtime is told for each. No runtime is loaded."""
import ctypes

import pytest

from pyopengl_video import vpl
from pyopengl_video.vpl import api
from pyopengl_video.vpl import encoder as vpl_encoder
from pyopengl_video.vpl.encoder import CODECS, extension_buffers


def by_id(buffers):
    return {api.fourcc_name(buffer.Header.BufferId): buffer for buffer in buffers}


def test_the_record_lists_the_codecs_the_encoder_has_settings_for():
    assert vpl.BACKEND.codecs == frozenset(CODECS) == {'h264', 'hevc', 'av1', 'vp9'}
    assert vpl.BACKEND.largest('h264') == (4096, 4096)
    assert vpl.BACKEND.largest('av1') == (8192, 8192)


@pytest.mark.parametrize(('codec', 'code'), [
    ('h264', 'AVC '), ('hevc', 'HEVC'), ('av1', 'AV1 '), ('vp9', 'VP9 ')])
def test_each_codec_is_asked_for_by_its_four_character_code(codec, code):
    assert api.fourcc_name(CODECS[codec].codec_id) == code


def test_every_group_opens_where_decoding_can_start():
    """IdrInterval counts from zero for H.264 and from one for HEVC."""
    assert CODECS['h264'].idr_interval == 0
    assert CODECS['hevc'].idr_interval == 1


@pytest.mark.parametrize('codec', ['h264', 'hevc', 'av1'])
def test_a_stream_that_states_its_colour_declares_limited_range_bt709(codec):
    signal = by_id(extension_buffers(codec))['VSIN']
    assert signal.Header.BufferSz == ctypes.sizeof(signal)
    assert (signal.VideoFullRange, signal.ColourDescriptionPresent) == (0, 1)
    assert (signal.ColourPrimaries, signal.TransferCharacteristics,
            signal.MatrixCoefficients) == (1, 1, 1)


@pytest.mark.parametrize('codec', ['h264', 'hevc'])
def test_the_older_codecs_carry_the_colour_description_and_nothing_else(codec):
    assert list(by_id(extension_buffers(codec))) == ['VSIN']


@pytest.mark.parametrize(('codec', 'buffer'), [('av1', 'A1BS'), ('vp9', '9PAR')])
def test_av1_and_vp9_are_told_not_to_wrap_the_stream_in_ivf(codec, buffer):
    found = by_id(extension_buffers(codec))[buffer]
    assert found.WriteIVFHeaders == api.MFX_CODINGOPTION_OFF
    assert found.Header.BufferSz == ctypes.sizeof(found)


def test_vp9_carries_no_colour_description_buffer():
    assert list(by_id(extension_buffers('vp9'))) == ['9PAR']


def test_the_codec_table_is_the_encoders():
    assert vpl_encoder.VPLEncoder.__init__.__kwdefaults__['codec'] in CODECS
