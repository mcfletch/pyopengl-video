"""HEVC, VP9 and AV1 on the NVDEC decoder, on real hardware, into real textures.

Decoding these codecs is exact, film grain included, so every picture of the
reference encoders' streams must come out of the GPU with the bytes ffmpeg's
software decoder produced, read back from the textures the decoder hands out:
NV12 from an 8-bit stream and P010 from a 10-bit one, through both ways a
picture reaches its textures.

The streams are the ``large`` ones of each codec's table. An NVIDIA decoder
takes no picture under 144 samples each way in HEVC, or 128 in VP9 and AV1,
and the rest of the fixtures are smaller than that; what the decoder says of
one of those is tested here too.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from OpenGL.GL import GL_RGBA, GL_TEXTURE_2D, GL_UNSIGNED_BYTE, glBindTexture, glGetTexImage

from pyopengl_video import DecoderError, decoders, nvdec, obu, open_decoder
from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.decoder import Sample
from pyopengl_video.ivf import IVFReader
from pyopengl_video.mp4 import MP4Reader
from pyopengl_video.nvdec.decoder import NVDECDecoder
from pyopengl_video.playback import Movie
from pyopengl_video.vaapi.hevc_parse import NAL_CRA
from pyopengl_video.vaapi.vp9_parse import split_superframe
from tests.conftest import decode_all, md5s

DATA = Path(__file__).parent / 'data' / 'decode'
CODECS = ('hevc', 'vp9', 'av1')
TABLES = {codec: json.loads((DATA / f'{codec}.json').read_text()) for codec in CODECS}

#: name -> (codec, the stream's entry), for every stream large enough to decode.
LARGE = {name: (codec, entry) for codec in CODECS
         for name, entry in TABLES[codec]['large'].items()}
#: One of the ordinary fixtures is large enough too, and is the only one with tiles.
LARGE['vp9_tiles'] = ('vp9', TABLES['vp9']['streams']['vp9_tiles'])

#: One stream a codec that is under the smallest picture the hardware takes.
SMALL = {'hevc': 'hevc_bpyramid', 'vp9': 'vp9_altref', 'av1': 'av1_altref'}

pytestmark = pytest.mark.usefixtures('nvdec_available')


def samples_of(name, start=0, stop=None):
    """The stream's samples, the first with its parameter sets where it has them."""
    codec, entry = LARGE.get(name) or next(
        (codec, TABLES[codec]['streams'][name]) for codec in CODECS
        if name in TABLES[codec]['streams'])
    path = DATA / entry['file']
    if codec == 'hevc':
        with AnnexBReader(path) as stream:
            stop = len(stream) if stop is None else stop
            return [stream.sample(index, parameter_sets=index == start)
                    for index in range(start, stop)]
    with IVFReader(path) as stream:
        stop = len(stream) if stop is None else stop
        return [stream.sample(index) for index in range(start, stop)]


def expected(name):
    return LARGE[name][1]['md5']


@pytest.fixture
def needs():
    """A function that skips the test unless this GPU decodes the codec named."""
    def need(codec):
        nvdec.forget_probe()
        if codec not in nvdec.decoder_codecs():
            pytest.skip(f'this NVIDIA GPU has no {codec} decoder')
    return need


@pytest.fixture(params=[True, False], ids=['on-the-gpu', 'through-host-memory'])
def decoder_for(request, needs):
    """A function making a decoder of a codec, for each way a picture reaches its textures."""
    made = []

    def make(codec, **options):
        needs(codec)
        made.append(NVDECDecoder(codec, interop=request.param, **options))
        return made[-1]
    yield make
    for decoder in made:
        decoder.close()


class TestConformance:
    @pytest.mark.parametrize('name', sorted(LARGE))
    def test_every_picture_matches_the_reference_decode(self, decoder_for, name):
        decoder = decoder_for(LARGE[name][0])
        assert decode_all(decoder, samples_of(name)) == expected(name)

    def test_the_streams_cover_every_codec_at_both_depths(self):
        assert {(codec, entry['depth']) for codec, entry in LARGE.values()} == {
            (codec, depth) for codec in CODECS for depth in (8, 10)}

    @pytest.mark.parametrize('name', sorted(LARGE))
    def test_frames_are_laid_out_by_the_streams_depth(self, decoder_for, name):
        codec, entry = LARGE[name]
        decoder = decoder_for(codec)
        frames = [frame for sample in samples_of(name, stop=4)
                  for frame in decoder.decode(sample)] + decoder.flush()
        try:
            assert {frame.layout for frame in frames} == {
                'nv12' if entry['depth'] == 8 else 'p010'}
            assert {frame.size for frame in frames} == {tuple(entry['size'])}
            assert decoder.size == tuple(entry['size'])
        finally:
            for frame in frames:
                frame.release()

    @pytest.mark.parametrize('name', ['vp9_large_odd', 'av1_large_odd'])
    def test_a_picture_of_an_odd_size_is_held_in_even_textures(self, decoder_for, name):
        """A 4:2:0 surface has a chroma sample for every two of luma."""
        decoder = decoder_for(LARGE[name][0])
        frames = decoder.decode(samples_of(name)[0]) + decoder.flush()
        try:
            assert [frame.visible for frame in frames] == [(0, 0, 151, 145)]
            assert [frame.texture_size for frame in frames] == [(152, 146)]
        finally:
            for frame in frames:
                frame.release()

    def test_an_hevc_picture_is_decoded_at_whole_blocks(self, decoder_for):
        decoder = decoder_for('hevc')
        frames = decoder.decode(samples_of('hevc_large_bpyramid')[0]) + decoder.flush()
        try:
            assert [frame.visible for frame in frames] == [(0, 0, 180, 148)]
            assert [frame.texture_size for frame in frames] == [(192, 160)]
        finally:
            for frame in frames:
                frame.release()


class TestColour:
    def test_an_hevc_streams_colour_includes_where_chroma_sits(self, decoder_for):
        """The parser reports the matrix and not the siting; the stream states both."""
        decoder = decoder_for('hevc')
        frames = decoder.decode(samples_of('hevc_large_main10')[0]) + decoder.flush()
        try:
            colour = frames[0].colour
            assert (colour.primaries, colour.transfer, colour.matrix) == (9, 16, 9)
            assert colour.chroma_location == 2
            assert not colour.full_range
        finally:
            for frame in frames:
                frame.release()

    def test_a_vp9_stream_that_states_no_matrix_is_not_taken_for_rgb(self, decoder_for):
        """The parser reports matrix 0 for every VP9 stream, which is RGB's."""
        decoder = decoder_for('vp9')
        md5s(decoder.decode(samples_of('vp9_large_altref')[0]) + decoder.flush())
        assert decoder.colour.matrix == 6            # BT.601 below 720 rows
        assert decoder.colour.chroma_location == 2   # colocated, as VP9 sites it

    def test_an_av1_stream_with_no_chroma_is_full_range(self, decoder_for):
        decoder = decoder_for('av1')
        md5s(decoder.decode(samples_of('av1_large_monochrome')[0]) + decoder.flush())
        assert decoder.colour.full_range

    def test_the_size_and_colour_are_known_from_the_headers(self, needs):
        needs('hevc')
        with AnnexBReader(DATA / 'hevc_large_main10.h265') as stream:
            headers = stream.parameter_sets()
        with open_decoder('h265', headers=headers, backend='nvdec') as made:
            assert made.codec == 'hevc'
            assert made.size == (178, 150)
            assert made.colour.matrix == 9


class TestTimestamps:
    @pytest.mark.parametrize('name', ['hevc_large_bpyramid', 'vp9_large_altref',
                                      'vp9_large_show_existing', 'av1_large_altref'])
    def test_each_picture_carries_its_own_samples_time(self, decoder_for, name):
        """In display order: every sample of these streams shows one picture."""
        decoder = decoder_for(LARGE[name][0])
        samples = samples_of(name)
        shown = []
        for sample in samples:
            for frame in decoder.decode(sample):
                shown.append((frame.timestamp, frame.duration))
                frame.release()
        for frame in decoder.flush():
            shown.append((frame.timestamp, frame.duration))
            frame.release()
        assert shown == sorted((sample.timestamp, sample.duration) for sample in samples)

    def test_a_vp9_picture_is_shown_as_its_sample_is_decoded(self, decoder_for):
        """Nothing is held back, so a frame shown again keeps its place."""
        decoder = decoder_for('vp9')
        for sample in samples_of('vp9_large_show_existing'):
            frames = decoder.decode(sample)
            assert [frame.timestamp for frame in frames] == [sample.timestamp]
            md5s(frames)
        assert decoder.flush() == []


class TestVP9FramesShownAgain:
    NAME = 'vp9_large_show_existing'

    def test_the_stream_shows_frames_again(self):
        again = [sample for sample in samples_of(self.NAME)
                 if any(frame[0] & 0x08 for frame in split_superframe(sample.data))]
        assert len(again) >= 8

    def test_the_slots_are_followed_again_after_a_seek(self, decoder_for):
        decoder = decoder_for('vp9')
        samples = samples_of(self.NAME)
        md5s([frame for sample in samples[:9] for frame in decoder.decode(sample)])
        decoder.reset()
        key = next(index for index, sample in enumerate(samples) if index and sample.keyframe)
        assert decode_all(decoder, samples[key:]) == expected(self.NAME)[key:]

    def test_a_frame_shown_from_a_slot_nothing_was_read_into_is_left_out(
            self, decoder_for, caplog):
        """Decoding from a sample that is no key frame: the slots are unknown."""
        decoder = decoder_for('vp9')
        samples = samples_of(self.NAME)
        first = next(index for index, sample in enumerate(samples)
                     if len(sample.data) == 1)
        with caplog.at_level('WARNING', logger='pyopengl_video.nvdec.decoder'):
            assert decoder.decode(samples[first]) == []
        assert 'shows reference slot' in caplog.text


class TestContainers:
    @pytest.mark.parametrize('name', ['hevc_large_bpyramid', 'vp9_large_altref',
                                      'av1_large_altref'])
    def test_an_mp4_decodes_from_its_sample_entry(self, needs, name):
        codec = LARGE[name][0]
        needs(codec)
        with MP4Reader(DATA / f'{name}.mp4') as movie:
            track = movie.video
            with open_decoder(track.codec, headers=track.parameter_sets(),
                              backend='nvdec') as made:
                assert made.codec == codec
                if codec != 'vp9':               # a vp09 entry holds no headers
                    assert made.size == (track.width, track.height)
                samples = [track.sample(index) for index in range(len(track))]
                assert decode_all(made, samples) == expected(name)

    def test_av1_decodes_with_its_sequence_header_apart_from_the_samples(self, needs):
        needs('av1')
        samples = samples_of('av1_large_grain')
        headers = obu.sequence_header(samples[0].data)
        stripped = [Sample(obu.without(sample.data, obu.SEQUENCE_HEADER,
                                       obu.TEMPORAL_DELIMITER),
                           sample.timestamp, sample.duration, sample.keyframe, sample.index)
                    for sample in samples]
        with open_decoder('av1', headers=headers, backend='nvdec') as made:
            assert made.size == (146, 144)
            assert decode_all(made, stripped) == expected('av1_large_grain')


class TestSeeking:
    @pytest.mark.parametrize('name', ['hevc_large_bpyramid', 'hevc_large_main10',
                                      'vp9_large_altref', 'vp9_large_profile2',
                                      'av1_large_altref', 'av1_large_grain10'])
    def test_decoding_resumes_from_a_key_frame_after_a_reset(self, decoder_for, name):
        decoder = decoder_for(LARGE[name][0])
        samples = samples_of(name)
        md5s([frame for sample in samples[:7] for frame in decoder.decode(sample)])
        decoder.reset()
        key = next(index for index, sample in enumerate(samples) if index and sample.keyframe)
        times = sorted(sample.timestamp for sample in samples)
        resumed = samples_of(name, start=key)
        assert decode_all(decoder, resumed) == expected(name)[
            times.index(resumed[0].timestamp):]

    def test_hevc_starts_at_a_key_frame_that_is_no_idr(self, decoder_for):
        """The pictures before it in display order refer to ones never decoded."""
        name = 'hevc_large_open_gop'
        decoder = decoder_for('hevc')
        with AnnexBReader(DATA / f'{name}.h265') as stream:
            key = stream.sync_samples[1]
            order = stream.display_order()
        resumed = samples_of(name, start=key)
        kinds = {(unit[0] >> 1) & 0x3F for unit in _units(resumed[0].data)}
        assert NAL_CRA in kinds
        assert decode_all(decoder, resumed) == expected(name)[order.index(key):]

    @pytest.mark.parametrize('name', ['hevc_large_open_gop', 'vp9_large_show_existing',
                                      'av1_large_grain'])
    def test_a_stream_decodes_again_after_a_flush(self, decoder_for, name):
        decoder = decoder_for(LARGE[name][0])
        for _ in range(2):
            assert decode_all(decoder, samples_of(name)) == expected(name)

    def test_one_decoder_takes_a_stream_of_another_size_and_depth(self, decoder_for):
        decoder = decoder_for('av1')
        for name in ('av1_large_altref', 'av1_large_grain10', 'av1_large_superres'):
            assert decode_all(decoder, samples_of(name)) == expected(name)
        assert decoder.size == tuple(LARGE['av1_large_superres'][1]['size'])


def _units(data):
    """The NAL units of an Annex-B sample, each without its start code."""
    return [unit for unit in data.split(b'\x00\x00\x01') if unit.strip(b'\x00')]


class TestSeveralMovies:
    def test_decoders_of_every_codec_in_one_context_each_decode_their_own(self, needs):
        """One session a movie, their samples interleaved."""
        names = ['hevc_large_main10', 'vp9_large_show_existing', 'av1_large_grain',
                 'hevc_large_open_gop', 'av1_large_altref']
        for name in names:
            needs(LARGE[name][0])
        streams = {name: samples_of(name) for name in names}
        made = {name: NVDECDecoder(LARGE[name][0]) for name in names}
        found = {name: [] for name in names}
        try:
            for index in range(max(len(samples) for samples in streams.values())):
                for name in names:
                    if index < len(streams[name]):
                        found[name] += md5s(made[name].decode(streams[name][index]))
            for name in names:
                found[name] += md5s(made[name].flush())
        finally:
            for decoder in made.values():
                decoder.close()
        assert found == {name: expected(name) for name in names}


class TestRefusedStreams:
    @pytest.mark.parametrize('codec', CODECS)
    def test_a_picture_under_the_smallest_is_refused_by_both_sizes(self, decoder_for, codec):
        decoder = decoder_for(codec)
        with pytest.raises(DecoderError, match=r'picture is smaller than the \d+x\d+ .* decodes in'):
            decode_all(decoder, samples_of(SMALL[codec]))

    @pytest.mark.parametrize('codec', CODECS)
    def test_a_decoder_that_refused_a_stream_decodes_the_next(self, decoder_for, codec):
        decoder = decoder_for(codec)
        with pytest.raises(DecoderError, match='smaller'):
            decode_all(decoder, samples_of(SMALL[codec]))
        decoder.reset()
        name = next(name for name in sorted(LARGE) if LARGE[name][0] == codec)
        assert decode_all(decoder, samples_of(name)) == expected(name)

    @pytest.mark.parametrize('codec', ['hevc', 'vp9'])
    def test_bytes_that_are_no_stream_produce_no_picture(self, decoder_for, codec):
        decoder = decoder_for(codec)
        for junk in (b'', b'\x00\x00\x00\x01', b'not a stream at all' * 50):
            assert decoder.decode(Sample(junk, 0, 0, True, 0)) == []
        assert decoder.flush() == []

    def test_bytes_that_are_no_av1_are_refused_and_a_stream_decodes_after(self, decoder_for):
        """NVIDIA's AV1 parser fails a sample it cannot split into OBUs."""
        decoder = decoder_for('av1')
        assert decoder.decode(Sample(b'', 0, 0, True, 0)) == []
        with pytest.raises(DecoderError, match='the parser refused 950 bytes of AV1'):
            decoder.decode(Sample(b'not a stream at all' * 50, 0, 0, True, 0))
        decoder.reset()
        name = 'av1_large_altref'
        assert decode_all(decoder, samples_of(name)) == expected(name)


class TestRegistry:
    def test_both_backends_report_the_codecs_the_gpu_decodes(self):
        nvdec.forget_probe()
        offered = nvdec.decoder_codecs()
        assert 'h264' in offered
        found = {backend.name: backend for backend in decoders()}
        for name in ('nvdec', 'nvdec-copy'):
            assert found[name].codecs == set(offered)
            for codec, size in offered.items():
                assert found[name].largest(codec) == size

    def test_the_gpu_is_asked_and_not_assumed(self):
        """A GPU that answers states a limit for each codec it lists."""
        nvdec.forget_probe()
        assert nvdec.decoder_codecs() is not nvdec.ASSUMED_SIZES
        assert all(width >= 4096 and height >= 4096
                   for width, height in nvdec.decoder_codecs().values())

    @pytest.mark.parametrize('spelling, codec', [('h265', 'hevc'), ('hvc1', 'hevc'),
                                                 ('av01', 'av1'), ('vp09', 'vp9')])
    def test_a_codec_opens_by_any_of_its_names(self, needs, spelling, codec):
        needs(codec)
        with open_decoder(spelling, backend='nvdec') as made:
            assert isinstance(made, NVDECDecoder)
            assert made.codec == codec
        with NVDECDecoder(spelling) as made:
            assert made.codec == codec


def picture(texture, size):
    glBindTexture(GL_TEXTURE_2D, texture)
    data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
    glBindTexture(GL_TEXTURE_2D, 0)
    return np.frombuffer(data, np.uint8).reshape(size[1], size[0], 4).copy()


#: file -> the stream's name, for each container and depth a player may open.
MOVIES = {
    'hevc_large_bpyramid.mp4': 'hevc_large_bpyramid',
    'hevc_large_main10.h265': 'hevc_large_main10',
    'vp9_large_altref.mp4': 'vp9_large_altref',
    'vp9_large_show_existing.ivf': 'vp9_large_show_existing',
    'vp9_large_profile2.ivf': 'vp9_large_profile2',
    'vp9_large_odd.ivf': 'vp9_large_odd',
    'av1_large_altref.mp4': 'av1_large_altref',
    'av1_large_grain10.ivf': 'av1_large_grain10',
}


class TestPlaying:
    @pytest.mark.parametrize('backend', ['nvdec', 'nvdec-copy'])
    @pytest.mark.parametrize('path', sorted(MOVIES))
    def test_every_frame_plays_and_a_seek_shows_the_same_picture(self, needs, path, backend):
        codec, entry = LARGE[MOVIES[path]]
        needs(codec)
        with Movie.open(DATA / path) as movie:
            assert movie.size == tuple(entry['size'])
            playing = movie.player(backend=backend)
            try:
                forward = [picture(playing.frame_at(index / 30 + 0.001), movie.size)
                           for index in range(movie.frames)]
                assert playing.frame == movie.frames - 1
                for index in (movie.frames // 2, 1, movie.frames - 2, 0):
                    again = picture(playing.frame_at(index / 30 + 0.001), movie.size)
                    assert playing.frame == index
                    assert np.array_equal(again, forward[index]), index
            finally:
                playing.close()
        # The source moves every frame, so a player stuck on one would show it.
        assert all(not np.array_equal(earlier, later)
                   for earlier, later in zip(forward, forward[1:], strict=False))
        # The last frame, since a fixture may fade in from black.
        assert (forward[-1][..., 3] == 255).all()
        assert forward[-1][..., :3].std() > 10
