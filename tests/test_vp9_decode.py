"""VP9 decoding on real hardware, into real OpenGL textures.

VP9 decoding is exact, so every picture of the libvpx fixtures must come out
of the GPU with the bytes ffmpeg's software decoder produced, read back from
the textures the decoder hands out: NV12 for the 8-bit streams and P010 for
the 10-bit one.
"""
import json
from pathlib import Path

import pytest

from pyopengl_video import DecoderError, open_decoder
from pyopengl_video.decoder import Sample
from pyopengl_video.ivf import IVFReader
from pyopengl_video.vaapi.decoder import VAAPIDecoder
from tests.conftest import decode_all, md5s
from tests.vp9_syntax import INTER, frame, show_existing

DATA = Path(__file__).parent / 'data' / 'decode'
FIXTURES = json.loads((DATA / 'vp9.json').read_text())['streams']


def samples(name, start=0):
    with IVFReader(DATA / f'{name}.ivf') as stream:
        return [stream.sample(index) for index in range(start, len(stream))]


@pytest.fixture
def decoder(vaapi_decodes):
    vaapi_decodes('vp9')
    made = VAAPIDecoder('vp9')
    yield made
    made.close()


class TestConformance:
    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_every_picture_matches_the_reference_decode(self, decoder, name):
        assert decode_all(decoder, samples(name)) == FIXTURES[name]['md5']

    def test_the_fixtures_cover_both_bit_depths(self):
        assert {entry['depth'] for entry in FIXTURES.values()} == {8, 10}

    def test_an_8_bit_stream_decodes_into_nv12_and_a_10_bit_one_into_p010(self, decoder):
        for name, layout in (('vp9_lossless', 'nv12'), ('vp9_profile2', 'p010')):
            frames = decoder.decode(samples(name)[0])
            assert [found.layout for found in frames] == [layout]
            assert frames[0].visible == (0, 0, *FIXTURES[name]['size'])
            for found in frames:
                found.release()

    def test_the_size_and_colour_come_from_the_first_key_frame(self, decoder):
        assert decoder.size == (0, 0)
        for found in decoder.decode(samples('vp9_altref')[0]):
            found.release()
        assert decoder.size == (122, 70)

    def test_it_opens_by_name_with_no_headers(self, vaapi_decodes):
        vaapi_decodes('vp9')
        with open_decoder('vp9', backend='vaapi') as made:
            assert decode_all(made, samples('vp9_resilient')) == FIXTURES['vp9_resilient']['md5']

    def test_frames_carry_their_samples_timestamps(self, decoder):
        shown = []
        expected = []
        for sample in samples('vp9_show_existing'):
            expected.append(sample.timestamp)
            for found in decoder.decode(sample):
                shown.append(found.timestamp)
                found.release()
        assert shown == expected

    def test_one_decoder_takes_streams_of_different_sizes_and_depths(self, decoder):
        for name in ('vp9_lossless', 'vp9_profile2', 'vp9_segmented'):
            assert decode_all(decoder, samples(name)) == FIXTURES[name]['md5']


class TestSeeking:
    @pytest.mark.parametrize('name', ['vp9_altref', 'vp9_show_existing', 'vp9_profile2'])
    def test_decoding_resumes_from_a_later_key_frame_after_a_reset(self, decoder, name):
        with IVFReader(DATA / f'{name}.ivf') as stream:
            key = stream.sync_samples[1]
        decode_all(decoder, samples(name)[:key // 2])
        decoder.reset()
        assert decode_all(decoder, samples(name, key)) == FIXTURES[name]['md5'][key:]

    def test_frames_between_a_reset_and_the_next_key_frame_are_dropped(self, decoder):
        name = 'vp9_altref'
        with IVFReader(DATA / f'{name}.ivf') as stream:
            key = stream.sync_samples[1]
        decode_all(decoder, samples(name)[:3])
        decoder.reset()
        assert decode_all(decoder, samples(name, key - 4)) == FIXTURES[name]['md5'][key:]


class TestSeveralMovies:
    def test_several_decoders_in_one_context_each_decode_their_own(self, vaapi_decodes):
        vaapi_decodes('vp9')
        names = sorted(FIXTURES)
        streams = {name: samples(name) for name in names}
        decoders = {name: VAAPIDecoder('vp9') for name in names}
        found = {name: [] for name in names}
        try:
            for index in range(max(map(len, streams.values()))):
                for name in names:
                    if index < len(streams[name]):
                        found[name] += md5s(decoders[name].decode(streams[name][index]))
            for name in names:
                found[name] += md5s(decoders[name].flush())
        finally:
            for made in decoders.values():
                made.close()
        assert found == {name: FIXTURES[name]['md5'] for name in names}


class TestSurfaces:
    def test_a_frame_shown_twice_is_two_frames_over_one_surface(self, decoder):
        """A repeated picture is lent again while the first lending is still held."""
        name = 'vp9_lossless'
        first, = decoder.decode(samples(name)[0])
        again, = decoder.decode(sample_of(show_existing(0)))
        assert again is not first
        assert again.planes == first.planes
        assert md5s([first, again]) == [FIXTURES[name]['md5'][0]] * 2

    def test_released_frames_free_their_surfaces(self, decoder):
        for _ in range(3):
            decoder.reset()
            decode_all(decoder, samples('vp9_show_existing'))


def sample_of(data):
    return Sample(data, 0, 1, True, 0)


class TestRefusals:
    def test_twelve_bits_are_refused_by_name(self, decoder):
        with pytest.raises(DecoderError, match='12 bits'):
            decoder.decode(sample_of(frame(profile=2, depth=12)))

    def test_444_is_refused_by_name(self, decoder):
        with pytest.raises(DecoderError, match='profile 1.*4:4:4'):
            decoder.decode(sample_of(frame(profile=1, subsampling=(0, 0))))

    def test_a_change_of_size_without_a_key_frame_is_refused_by_name(self, decoder):
        name = 'vp9_lossless'
        md5s(decoder.decode(samples(name)[0]))
        with pytest.raises(DecoderError, match='reference scaling'):
            decoder.decode(sample_of(frame(frame_type=INTER, found_ref=None, size=(80, 48))))

    def test_decoding_goes_on_from_the_next_key_frame_after_a_refusal(self, decoder):
        name = 'vp9_lossless'
        md5s(decoder.decode(samples(name)[0]))
        with pytest.raises(DecoderError):
            decoder.decode(sample_of(frame(frame_type=INTER, found_ref=None, size=(80, 48))))
        with IVFReader(DATA / f'{name}.ivf') as stream:
            key = stream.sync_samples[1]
        assert decode_all(decoder, samples(name, 1)) == FIXTURES[name]['md5'][key:]

    def test_a_frame_that_is_not_vp9_is_refused(self, decoder):
        with pytest.raises(DecoderError, match='frame marker'):
            decoder.decode(sample_of(b'\x00\x00\x00\x01\x67'))

    def test_a_frame_cut_short_is_refused(self, decoder):
        with pytest.raises(DecoderError, match='header ends'):
            decoder.decode(sample_of(samples('vp9_altref')[0].data[:9]))
