"""The codec names, and the spellings that reach them."""
import pytest

from pyopengl_video import codecs


@pytest.mark.parametrize(('spelling', 'name'), [
    ('h264', 'h264'), ('H264', 'h264'), ('avc', 'h264'), ('avc1', 'h264'), ('avc3', 'h264'),
    ('hevc', 'hevc'), ('h265', 'hevc'), ('H.265', 'hevc'), ('hvc1', 'hevc'), ('hev1', 'hevc'),
    ('av1', 'av1'), ('av01', 'av1'), ('AV1', 'av1'),
    ('vp9', 'vp9'), ('vp09', 'vp9'),
])
def test_every_spelling_reaches_the_one_name(spelling, name):
    assert codecs.canonical(spelling) == name


def test_a_name_that_is_no_codec_here_is_kept_lower_cased():
    assert codecs.canonical('ProRes') == 'prores'


def test_a_sample_entry_names_its_codec_and_others_name_none():
    assert codecs.of_sample_entry('hvc1') == 'hevc'
    assert codecs.of_sample_entry('mp4v') is None
    assert codecs.of_sample_entry('hevc') is None


def test_a_title_is_the_specifications_spelling():
    assert codecs.title('h264') == 'H.264'
    assert codecs.title('h265') == 'HEVC'
    assert codecs.title('prores') == 'prores'


def test_every_codec_is_listed_under_its_own_name():
    assert set(codecs.CODECS) == {'h264', 'hevc', 'av1', 'vp9'}
    assert all(codec.name == name for name, codec in codecs.CODECS.items())
