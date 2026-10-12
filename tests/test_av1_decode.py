"""AV1 on the VA-API decoder, on real hardware, into real OpenGL textures.

AV1 decoding is exact, film grain included, so every picture of the fixtures
must come out of the GPU with the bytes ffmpeg's software decoder (dav1d)
produced, read back from the textures the decoder hands out.
"""
import json
from pathlib import Path

import pytest

from pyopengl_video import DecoderError, decoders, obu, open_decoder
from pyopengl_video.decoder import Sample
from pyopengl_video.ivf import IVFReader
from pyopengl_video.vaapi.decoder import VAAPIDecoder
from tests.av1_syntax import Sequence, frame, unit
from tests.av1_syntax import obu as write_obu
from tests.conftest import decode_all, md5s

DATA = Path(__file__).parent / 'data' / 'decode'
FIXTURES = json.loads((DATA / 'av1.json').read_text())['streams']


def samples_of(name, first=0):
    with IVFReader(DATA / f'{name}.ivf') as stream:
        return [stream.sample(index) for index in range(first, len(stream))]


@pytest.fixture
def decoder(vaapi_decodes):
    vaapi_decodes('av1')
    made = VAAPIDecoder('av1')
    yield made
    made.close()


class TestConformance:
    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_every_picture_matches_the_reference_decode(self, decoder, name):
        assert decode_all(decoder, samples_of(name)) == FIXTURES[name]['md5']

    @pytest.mark.parametrize(('name', 'layout'), [
        ('av1_altref', 'nv12'), ('av1_grain10', 'p010'), ('av1_switchable10', 'p010')])
    def test_frames_are_laid_out_by_the_streams_depth(self, decoder, name, layout):
        frames = decoder.decode(samples_of(name)[0])
        assert [found.layout for found in frames] == [layout]
        assert frames[0].visible == (0, 0, *FIXTURES[name]['size'])
        assert decoder.size == tuple(FIXTURES[name]['size'])
        for found in frames:
            found.release()

    def test_frames_carry_their_temporal_units_timestamps(self, decoder):
        samples = samples_of('av1_altref')
        shown = []
        for sample in samples:
            for found in decoder.decode(sample):
                shown.append((found.timestamp, found.duration))
                found.release()
        assert shown == [(sample.timestamp, sample.duration) for sample in samples]
        assert decoder.flush() == []

    def test_a_stream_decodes_through_the_registry_from_its_configuration(self, vaapi_decodes):
        """As from an MP4: the sequence header apart, the samples without one."""
        vaapi_decodes('av1')
        samples = samples_of('av1_tiles')
        headers = obu.sequence_header(samples[0].data)
        stripped = [Sample(obu.without(sample.data, obu.SEQUENCE_HEADER,
                                       obu.TEMPORAL_DELIMITER),
                           sample.timestamp, sample.duration, sample.keyframe, sample.index)
                    for sample in samples]
        with open_decoder('av1', headers=headers, backend='vaapi') as made:
            assert made.size == (160, 90)
            assert decode_all(made, stripped) == FIXTURES['av1_tiles']['md5']

    def test_a_grain_stream_decodes_twice_over_with_the_same_pictures(self, decoder):
        """Each surface is used as a reference and as a grain target in turn."""
        for _ in range(2):
            decoder.reset()
            assert decode_all(decoder, samples_of('av1_grain')) == FIXTURES['av1_grain']['md5']


class TestSeveralMovies:
    def test_decoders_in_one_context_each_decode_their_own(self, vaapi_decodes):
        """One session a movie, their temporal units interleaved."""
        vaapi_decodes('av1')
        names = ['av1_altref', 'av1_grain', 'av1_grain10', 'av1_tiles', 'av1_intra_only']
        streams = [samples_of(name) for name in names]
        made = [VAAPIDecoder('av1') for _ in names]
        found = {name: [] for name in names}
        try:
            for index in range(max(len(stream) for stream in streams)):
                for name, stream, one in zip(names, streams, made, strict=True):
                    if index < len(stream):
                        found[name] += md5s(one.decode(stream[index]))
            for name, one in zip(names, made, strict=True):
                found[name] += md5s(one.flush())
        finally:
            for one in made:
                one.close()
        assert found == {name: FIXTURES[name]['md5'] for name in names}


class TestSeeking:
    @pytest.mark.parametrize('name', ['av1_altref', 'av1_grain', 'av1_tiles'])
    def test_decoding_resumes_from_a_later_key_frame_after_a_reset(self, decoder, name):
        with IVFReader(DATA / f'{name}.ivf') as stream:
            key = stream.sync_samples[1]
        decode_all(decoder, samples_of(name)[:7])
        decoder.reset()
        assert decode_all(decoder, samples_of(name, key)) == FIXTURES[name]['md5'][key:]

    def test_frames_before_a_key_frame_are_passed_over(self, decoder):
        """Restarted two temporal units early, nothing is shown until the key frame."""
        name = 'av1_altref'
        with IVFReader(DATA / f'{name}.ivf') as stream:
            key = stream.sync_samples[1]
        decode_all(decoder, samples_of(name)[:3])
        decoder.reset()
        assert decode_all(decoder, samples_of(name, key - 2)) == FIXTURES[name]['md5'][key:]

    def test_a_stream_with_intra_only_frames_restarts_at_its_key_frame(self, decoder):
        name = 'av1_intra_only'
        decode_all(decoder, samples_of(name)[:20])
        decoder.reset()
        assert decode_all(decoder, samples_of(name)) == FIXTURES[name]['md5']


class TestSurfaces:
    def test_a_decoder_that_has_lent_every_surface_says_which_are_held(self, decoder):
        held = []
        with pytest.raises(DecoderError, match='held'):
            for sample in samples_of('av1_altref'):
                held += decoder.decode(sample)
        assert len(held) >= decoder.output_slots
        for found in held:
            found.release()

    def test_released_frames_free_their_surfaces(self, decoder):
        for _ in range(3):
            decoder.reset()
            assert decode_all(decoder, samples_of('av1_grain10')) == (
                FIXTURES['av1_grain10']['md5'])

    def test_one_decoder_takes_a_stream_of_another_size_and_depth(self, decoder):
        for name in ('av1_tiles', 'av1_grain10', 'av1_monochrome', 'av1_altref'):
            assert decode_all(decoder, samples_of(name)) == FIXTURES[name]['md5']
            decoder.reset()


class TestRefusals:
    """What is not decoded is refused in words, and the decoder stays usable."""

    def still_decodes(self, decoder):
        decoder.reset()
        assert decode_all(decoder, samples_of('av1_still')) == FIXTURES['av1_still']['md5']

    def test_high_profile(self, decoder):
        high = Sequence(profile=1)
        with pytest.raises(DecoderError, match='High profile'):
            decoder.decode(Sample(unit(high.obu(), frame(high)), 0, 1, True, 0))
        self.still_decodes(decoder)

    def test_a_scalable_stream(self, decoder):
        layered = Sequence(operating_points=(0x103,))
        key = samples_of('av1_still')[0]
        [whole] = [found for found in obu.split(key.data) if found.kind == obu.FRAME]
        data = unit(layered.obu(), write_obu(obu.FRAME, whole.payload, temporal_id=1))
        with pytest.raises(DecoderError, match='scalable'):
            decoder.decode(Sample(data, 0, 1, True, 0))
        self.still_decodes(decoder)

    def test_large_scale_tile(self, decoder):
        data = unit(Sequence().obu(), write_obu(obu.TILE_LIST, bytes(8)))
        with pytest.raises(DecoderError, match='large scale tile'):
            decoder.decode(Sample(data, 0, 1, True, 0))
        self.still_decodes(decoder)

    def test_a_sample_that_is_not_av1(self, decoder):
        with pytest.raises(DecoderError):
            decoder.decode(Sample(b'\xff\x00\x00\x01', 0, 1, True, 0))
        with pytest.raises(DecoderError, match='sequence header'):
            decoder.decode(Sample(unit(frame()), 0, 1, True, 0))
        self.still_decodes(decoder)

    def test_a_temporal_unit_cut_short(self, decoder):
        key = samples_of('av1_altref')[0]
        with pytest.raises(DecoderError):
            decoder.decode(Sample(key.data[:40], 0, 1, True, 0))
        self.still_decodes(decoder)


class TestRegistry:
    def test_the_backend_lists_av1(self, vaapi_decodes):
        vaapi_decodes('av1')
        [found] = [backend for backend in decoders() if backend.name == 'vaapi']
        assert 'av1' in found.codecs
