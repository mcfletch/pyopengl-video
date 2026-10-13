"""Playback: which frame a time shows, when to seek, and decoding ahead.

The player's decisions are tested with a fake decoder, converter and texture
factory, so no GL is needed: a "texture" here is a number, and the fake
converter records which picture went into which. One case at the end plays a
real movie on the hardware decoder.
"""
import gc
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from OpenGL.GL import GL_RGBA, GL_TEXTURE_2D, GL_UNSIGNED_BYTE, glBindTexture, glGetTexImage

from pyopengl_video import playback
from pyopengl_video.backend import VideoError
from pyopengl_video.colour import Colour
from pyopengl_video.container import ContainerError
from pyopengl_video.decoder import (
    DecodedFrame,
    Decoder,
    DecoderError,
    DecoderUnavailable,
    Sample,
)
from pyopengl_video.mp4 import MP4Error, MP4Writer, full_box
from pyopengl_video.playback import Movie, Player, Timeline
from pyopengl_video.vaapi.encoder import VAAPIEncoder
from tests.conftest import blit, gradient_frame
from tests.descriptors import open_descriptors
from tests.mp4_build import movie as build_movie
from tests.mp4_build import visual_entry

DATA = Path(__file__).parent / 'data' / 'decode'
FIXTURES = json.loads((DATA / 'h264.json').read_text())['streams']
TICK = 3000                                     # 30 fps at 90 kHz


class FakeSource:
    """A video track: decode order, display order and key frames as given.

    shown -- for each sample in decoding order, its display rank
    keys -- the sync samples
    """

    timescale = 90000
    codec = 'h264'
    size = (64, 48)

    def __init__(self, shown, keys, offset=0):
        self.shown = list(shown)
        self.sync_samples = list(keys)
        self.offset = offset

    def __len__(self):
        return len(self.shown)

    def presentation_time(self, index):
        return self.shown[index] * TICK - self.offset

    def display_order(self):
        return sorted(range(len(self)), key=self.presentation_time)

    def keyframe_before(self, index):
        return max(key for key in self.sync_samples if key <= index)

    def sample(self, index, parameter_sets=False):
        return Sample(data=b'P' if parameter_sets else b'', timestamp=self.presentation_time(index),
                      duration=TICK, keyframe=index in self.sync_samples, index=index)

    def parameter_sets(self):
        return b''


def ibbp(groups, gop=6):
    """Decode-order display ranks of closed groups: I P B B P B B ..."""
    shown = []
    for group in range(groups):
        base = group * gop
        shown.append(base)
        position = 1
        while position < gop:
            last = min(position + 2, gop - 1)
            shown.append(base + last)
            shown.extend(range(base + position, base + last))
            position = last + 1
    return shown


class FakeDecoder(Decoder):
    """Releases pictures in display order, holding back `depth` of them.

    size_of -- the size of the picture a sample decodes to, by its index;
        every picture is 16x16 without one
    """

    def __init__(self, depth=2, size_of=None):
        self.depth = depth
        self.size_of = size_of or (lambda _index: (16, 16))
        self.held = []
        self.decoded = []
        self.resets = 0
        self.closed = 0
        self.lent = set()

    def frame(self, sample):
        width, height = self.size_of(sample.index)
        frame = DecodedFrame(timestamp=sample.timestamp, duration=sample.duration,
                             planes=(), layout='nv12', texture_size=(64, 64),
                             visible=(0, 0, width, height), colour=Colour(),
                             on_release=self.lent.discard)
        self.lent.add(frame)
        return frame

    def decode(self, sample):
        self.decoded.append(sample.index)
        self.held.append(sample)
        self.held.sort(key=lambda found: found.timestamp)
        ready = []
        while len(self.held) > self.depth:
            ready.append(self.frame(self.held.pop(0)))
        return ready

    def flush(self):
        ready = [self.frame(sample) for sample in self.held]
        self.held = []
        return ready

    def reset(self):
        self.held = []
        self.resets += 1

    def close(self):
        self.closed += 1


class Textures:
    """The texture factory: a texture is a number, and its size is remembered.

    events -- every texture made, forgotten by the converter and deleted, in
        the order it happened
    """

    def __init__(self):
        self.made = 0
        self.sizes = {}
        self.deleted = []
        self.events = []

    def make(self, width, height):
        self.made += 1
        self.sizes[self.made] = (width, height)
        self.events.append(('make', self.made))
        return self.made

    def delete(self, texture):
        self.deleted.append(texture)
        self.events.append(('delete', texture))

    @property
    def live(self):
        return {texture: size for texture, size in self.sizes.items()
                if texture not in self.deleted}


class FakeConverter:
    """Records which picture went into which texture, and of what size."""

    def __init__(self, textures=None):
        self.contents = {}
        self.drawn = []
        self.textures = textures
        self.closed = 0

    def convert(self, frame, target):
        self.contents[target] = frame.timestamp
        self.drawn.append((target, frame.size))

    def forget(self, target):
        if self.textures is not None:
            self.textures.events.append(('forget', target))

    def close(self):
        self.closed += 1


def player(source, depth=2, ahead=2, size=(16, 16), size_of=None):
    textures = Textures()
    decoder, converter = FakeDecoder(depth, size_of), FakeConverter(textures)
    built = Player(Movie(source, size=size), decoder, converter, ahead=ahead,
                   make_texture=textures.make, delete_texture=textures.delete)
    return built, decoder, converter, textures


def shown_rank(built, converter, seconds):
    texture = built.frame_at(seconds)
    return converter.contents[texture] // TICK


# ------------------------------------------------------------------ timeline


class TestTimeline:
    def test_a_time_shows_the_last_frame_at_or_before_it(self):
        timeline = Timeline(FakeSource(ibbp(2), [0, 6]))
        assert timeline.rank_at(0.0) == 0
        assert timeline.rank_at(1 / 30) == 1
        assert timeline.rank_at(1.5 / 30) == 1
        assert timeline.rank_at(11 / 30) == 11

    def test_before_the_start_is_the_first_frame_and_after_the_end_the_last(self):
        timeline = Timeline(FakeSource(ibbp(2), [0, 6]))
        assert timeline.rank_at(-5.0) == 0
        assert timeline.rank_at(100.0) == 11

    def test_the_duration_runs_to_the_end_of_the_last_frame(self):
        timeline = Timeline(FakeSource(ibbp(2), [0, 6]))
        assert timeline.duration == pytest.approx(12 / 30)

    def test_times_start_at_the_first_frame(self):
        """A track whose edit list starts it late still starts at zero."""
        timeline = Timeline(FakeSource(ibbp(1), [0], offset=-2 * TICK))
        assert timeline.seconds(0) == 0.0

    def test_decoding_for_a_frame_starts_at_the_key_frame_before_it(self):
        source = FakeSource(ibbp(2), [0, 6])
        timeline = Timeline(source)
        assert timeline.start_for(4) == 0
        assert timeline.start_for(7) == 6

    def test_a_leading_picture_starts_at_the_key_frame_before_its_own(self):
        """Open GOP: rank 5 is decoded after key 6 and shown before it."""
        shown = [0, 3, 1, 2, 6, 4, 5, 9, 7, 8]
        timeline = Timeline(FakeSource(shown, [0, 4]))
        assert timeline.start_for(5) == 0
        assert timeline.start_for(6) == 4
        assert timeline.start_for(7) == 4


# -------------------------------------------------------------------- player


class TestPlayingForward:
    def test_every_frame_is_shown_in_turn(self):
        built, _, converter, _ = player(FakeSource(ibbp(3), [0, 6, 12]))
        assert [shown_rank(built, converter, rank / 30) for rank in range(18)] == list(range(18))

    def test_each_sample_is_decoded_once(self):
        built, decoder, converter, _ = player(FakeSource(ibbp(3), [0, 6, 12]))
        for rank in range(18):
            shown_rank(built, converter, rank / 30)
        assert sorted(decoder.decoded) == list(range(18))
        assert decoder.resets == 0

    def test_asking_twice_decodes_nothing_more(self):
        built, decoder, converter, _ = player(FakeSource(ibbp(2), [0, 6]))
        shown_rank(built, converter, 3 / 30)
        count = len(decoder.decoded)
        assert shown_rank(built, converter, 3 / 30) == 3
        assert len(decoder.decoded) == count

    def test_it_decodes_ahead_of_what_is_shown(self):
        built, _, converter, _ = player(FakeSource(ibbp(3), [0, 6, 12]), ahead=3)
        shown_rank(built, converter, 0.0)
        assert built.ready_ahead >= 3

    def test_every_decoded_frame_is_released(self):
        built, decoder, converter, _ = player(FakeSource(ibbp(3), [0, 6, 12]))
        for rank in range(18):
            shown_rank(built, converter, rank / 30)
        assert decoder.lent == set()

    def test_a_few_textures_are_reused(self):
        built, _, converter, textures = player(FakeSource(ibbp(5), [0, 6, 12, 18, 24]))
        for rank in range(30):
            shown_rank(built, converter, rank / 30)
        assert textures.made <= 8

    def test_past_the_end_the_last_frame_stays(self):
        built, _, converter, _ = player(FakeSource(ibbp(2), [0, 6]))
        assert shown_rank(built, converter, 100.0) == 11
        assert shown_rank(built, converter, 200.0) == 11

    def test_a_slow_clock_shows_frames_more_than_once(self):
        built, _, converter, _ = player(FakeSource(ibbp(2), [0, 6]))
        shown = [shown_rank(built, converter, step / 60) for step in range(8)]
        assert shown == [0, 0, 1, 1, 2, 2, 3, 3]

    def test_a_fast_clock_skips_frames_without_seeking_inside_a_group(self):
        built, decoder, converter, _ = player(FakeSource(ibbp(3), [0, 6, 12]))
        shown = [shown_rank(built, converter, step * 2 / 30) for step in range(6)]
        assert shown == [0, 2, 4, 6, 8, 10]
        assert decoder.resets == 0


class TestSeeking:
    def test_going_back_restarts_from_the_key_frame(self):
        built, decoder, converter, _ = player(FakeSource(ibbp(3), [0, 6, 12]))
        shown_rank(built, converter, 14 / 30)
        decoder.decoded.clear()
        assert shown_rank(built, converter, 8 / 30) == 8
        assert decoder.resets == 1
        assert decoder.decoded[0] == 6

    def test_a_seek_hands_the_decoder_the_parameter_sets(self):
        source = FakeSource(ibbp(3), [0, 6, 12])
        built, decoder, converter, _ = player(source)
        sent = []
        decode = decoder.decode
        decoder.decode = lambda sample: (sent.append(sample), decode(sample))[1]
        shown_rank(built, converter, 14 / 30)
        assert sent[0].data == b'P'
        sent.clear()
        shown_rank(built, converter, 2 / 30)
        assert sent[0].data == b'P'

    def test_a_jump_past_the_next_key_frame_skips_the_frames_between(self):
        built, decoder, converter, _ = player(FakeSource(ibbp(5), [0, 6, 12, 18, 24]))
        shown_rank(built, converter, 0.0)
        assert shown_rank(built, converter, 25 / 30) == 25
        assert decoder.resets == 1
        assert min(decoder.decoded[decoder.decoded.index(24):]) == 24
        assert len(decoder.decoded) < 18

    def test_the_first_frame_after_a_seek_backwards(self):
        built, _, converter, _ = player(FakeSource(ibbp(3), [0, 6, 12]))
        shown_rank(built, converter, 17 / 30)
        assert shown_rank(built, converter, 0.0) == 0


class TestClosing:
    def test_close_deletes_the_textures_and_the_decoder(self):
        built, decoder, converter, textures = player(FakeSource(ibbp(2), [0, 6]))
        closed = []
        decoder.close = lambda: closed.append(True)
        shown_rank(built, converter, 5 / 30)
        built.close()
        assert sorted(textures.deleted) == list(range(1, textures.made + 1))
        assert closed == [True]

    def test_a_release_that_fails_does_not_strand_the_others(self):
        """C13: the converter, the decoder and every texture are each released."""
        built, decoder, converter, textures = player(FakeSource(ibbp(2), [0, 6]))
        shown_rank(built, converter, 5 / 30)

        def broken():
            raise RuntimeError('the converter would not close')

        converter.close = broken
        with pytest.raises(RuntimeError, match='would not close'):
            built.close()
        assert decoder.closed == 1
        assert textures.live == {}

    def test_closing_twice_releases_once(self):
        built, decoder, converter, textures = player(FakeSource(ibbp(2), [0, 6]))
        shown_rank(built, converter, 5 / 30)
        built.close()
        built.close()
        assert (decoder.closed, converter.closed) == (1, 1)
        assert len(textures.deleted) == textures.made

    def test_a_closed_player_shows_nothing(self):
        built, _, converter, _ = player(FakeSource(ibbp(2), [0, 6]))
        shown_rank(built, converter, 0.0)
        built.close()
        assert built.texture == 0
        with pytest.raises(DecoderError, match='closed'):
            built.frame_at(1 / 30)

    def test_leaving_the_block_by_an_exception_raises_that_exception(self, caplog):
        built, _, converter, _ = player(FakeSource(ibbp(2), [0, 6]))

        def broken():
            raise RuntimeError('the converter would not close')

        converter.close = broken
        with pytest.raises(ValueError, match='the caller'):
            with built:
                raise ValueError('the caller failed')
        assert 'would not close' in caplog.text


class TestFramesLentByTheDecoder:
    """C13: a frame the player was handed goes back, whatever happens next."""

    def test_a_conversion_that_fails_gives_back_every_frame(self):
        built, decoder, converter, _ = player(FakeSource(ibbp(2), [0, 6]), depth=0)
        decode = decoder.decode
        decoder.decode = lambda sample: decode(sample) + [decoder.frame(sample),
                                                          decoder.frame(sample)]

        def broken(*_):
            raise RuntimeError('the conversion pass failed')

        converter.convert = broken
        with pytest.raises(RuntimeError, match='conversion pass'):
            built.frame_at(0.0)
        assert decoder.lent == set()

    def test_a_flush_that_fails_gives_back_what_decode_returned(self):
        built, decoder, _, _ = player(FakeSource([0], [0]), depth=0)

        def broken():
            raise RuntimeError('the flush failed')

        decoder.flush = broken
        with pytest.raises(RuntimeError, match='flush failed'):
            built.frame_at(0.0)
        assert decoder.lent == set()

    def test_a_texture_whose_conversion_failed_is_used_again(self):
        built, _, converter, textures = player(FakeSource(ibbp(2), [0, 6]))
        convert, calls = converter.convert, []

        def once(frame, target):
            calls.append(target)
            if len(calls) == 1:
                raise RuntimeError('the conversion pass failed')
            convert(frame, target)

        converter.convert = once
        with pytest.raises(RuntimeError):
            built.frame_at(0.0)
        built.frame_at(0.0)
        assert calls[1] == calls[0]
        assert textures.made <= 4


class TestTheSizeIsTheStreams:
    """D1 and D2: a texture is the size of the picture drawn into it."""

    def test_textures_are_the_decoded_pictures_size_whatever_the_container_states(self):
        for stated in [(640, 480), (8, 8), (0, 0), (65535, 65535)]:
            built, _, converter, textures = player(FakeSource(ibbp(2), [0, 6]), size=stated)
            assert built.movie.size == stated
            for rank in range(12):
                shown_rank(built, converter, rank / 30)
            assert set(textures.sizes.values()) == {(16, 16)}
            assert built.size == (16, 16)

    def test_the_size_is_none_before_a_frame_is_shown(self):
        built, _, _, _ = player(FakeSource(ibbp(2), [0, 6]))
        assert built.size is None

    def test_a_stream_that_changes_size_gets_textures_of_the_new_size(self):
        """The second group of pictures is 32x24."""
        sizes = {True: (16, 16), False: (32, 24)}
        built, _, converter, textures = player(
            FakeSource(ibbp(3), [0, 6, 12]), size_of=lambda index: sizes[index < 6])
        shown = []
        for rank in range(18):
            shown_rank(built, converter, rank / 30)
            shown.append(built.size)
        assert shown == [(16, 16)] * 6 + [(32, 24)] * 12
        # Nothing is drawn into a texture of another size.
        for target, size in converter.drawn:
            assert textures.sizes[target] == size
        # The textures of the first size are gone, each forgotten by the
        # converter before it was deleted.
        assert set(textures.live.values()) == {(32, 24)}
        for texture in textures.deleted:
            assert (textures.events.index(('forget', texture))
                    < textures.events.index(('delete', texture)))

    def test_a_seek_back_across_a_size_change_shows_the_old_size_again(self):
        sizes = {True: (16, 16), False: (32, 24)}
        built, _, converter, textures = player(
            FakeSource(ibbp(3), [0, 6, 12]), size_of=lambda index: sizes[index < 6])
        shown_rank(built, converter, 14 / 30)
        assert shown_rank(built, converter, 2 / 30) == 2
        assert built.size == (16, 16)
        assert textures.sizes[built.texture] == (16, 16)
        for target, size in converter.drawn:
            assert textures.sizes[target] == size

    def test_a_few_textures_serve_a_stream_whose_size_keeps_changing(self):
        built, _, converter, textures = player(
            FakeSource(list(range(40)), list(range(40))),
            size_of=lambda index: (16 + 2 * (index % 3), 16))
        for rank in range(40):
            shown_rank(built, converter, rank / 30)
        assert len(textures.live) <= 8

    @pytest.mark.parametrize('size', [(0, 0), (0, 16), (16, -2)])
    def test_a_picture_with_no_pixels_is_refused_before_anything_is_made(self, size):
        built, decoder, converter, textures = player(
            FakeSource(ibbp(2), [0, 6]), size_of=lambda _index: size)
        with pytest.raises(DecoderError, match='no pixels'):
            built.frame_at(0.0)
        assert textures.made == 0
        assert converter.drawn == []
        assert decoder.lent == set()


class TestOpeningTheDecoder:
    """D10: a size the container does not state is not given as one."""

    @pytest.fixture
    def opened(self, monkeypatch):
        asked = []

        def open_decoder(_codec, **options):
            asked.append(options)
            return FakeDecoder()

        monkeypatch.setattr(playback, 'open_decoder', open_decoder)
        monkeypatch.setattr(playback, 'NV12Converter', FakeConverter)
        return asked

    @pytest.mark.parametrize('stated', [(0, 0), (0, 48), (64, 0)])
    def test_a_stated_size_of_nought_is_no_size(self, opened, stated):
        Movie(FakeSource(ibbp(1), [0]), size=stated).player().close()
        assert opened[0]['size'] is None

    def test_a_stated_size_chooses_the_backend(self, opened):
        Movie(FakeSource(ibbp(1), [0])).player().close()
        assert opened[0]['size'] == (64, 48)

    def test_a_stated_size_no_backend_takes_leaves_the_stream_to_say(self, monkeypatch):
        """A thumbnail in a container stating 65535x65535 still has a decoder."""
        asked = []

        def open_decoder(_codec, **options):
            asked.append(options['size'])
            if options['size'] is not None:
                raise DecoderUnavailable('no decoder here can read 65535x65535 h264')
            return FakeDecoder()

        monkeypatch.setattr(playback, 'open_decoder', open_decoder)
        monkeypatch.setattr(playback, 'NV12Converter', FakeConverter)
        Movie(FakeSource(ibbp(1), [0]), size=(65535, 65535)).player().close()
        assert asked == [(65535, 65535), None]

    def test_no_decoder_at_all_is_still_unavailable(self, monkeypatch):
        def open_decoder(_codec, **options):
            raise DecoderUnavailable(f'no decoder here, for {options["size"]}')

        monkeypatch.setattr(playback, 'open_decoder', open_decoder)
        with pytest.raises(DecoderUnavailable, match='for None'):
            Movie(FakeSource(ibbp(1), [0])).player()


# --------------------------------------------------------------------- movies


class TestMovie:
    @pytest.mark.parametrize('path', ['high_bpyramid.mp4', 'high_bpyramid.h264'])
    def test_a_movie_opens_without_gl(self, path):
        with Movie.open(DATA / path) as movie:
            assert movie.size == tuple(FIXTURES['high_bpyramid']['size'])
            assert movie.frames == len(FIXTURES['high_bpyramid']['md5'])
            assert movie.duration == pytest.approx(movie.frames / 30, rel=0.01)
            assert movie.codec == 'h264'
            assert movie.headers.startswith(b'\x00\x00\x00\x01\x67')

    @pytest.mark.parametrize(('head', 'named'), [
        (b'RIFF\x00\x10\x00\x00AVI LIST', 'AVI'),
        (b'\x00\x00\x01\xba\x44\x00\x04\x00', 'MPEG program stream'),
        (b'\x00\x00\x01\xb3\x14\x00\xf0\x13', 'MPEG-1 or MPEG-2 video'),
        (b'\x1a\x45\xdf\xa3\x9f\x42\x86\x81', 'Matroska or WebM'),
        (b'OggS\x00\x02\x00\x00', 'Ogg'),
    ])
    def test_another_container_is_refused_by_its_name(self, tmp_path, head, named):
        movie = tmp_path / 'movie'
        movie.write_bytes(head + bytes(64))
        with pytest.raises(ValueError, match=named):
            Movie.open(movie)
        with movie.open('rb') as handle, pytest.raises(ValueError, match=named):
            Movie.open(handle)

    def test_a_file_that_is_no_movie_is_refused(self, tmp_path):
        bad = tmp_path / 'notes.txt'
        bad.write_text('nothing to see')
        with pytest.raises(ValueError):
            Movie.open(bad)

    def test_a_refusal_is_a_video_error(self, tmp_path):
        riff = tmp_path / 'movie.avi'
        riff.write_bytes(b'RIFF' + bytes(64))
        with pytest.raises(VideoError):
            Movie.open(riff)
        with pytest.raises(ContainerError, match='AVI'):
            Movie.open(riff)

    def test_an_mp4_with_no_video_it_plays_is_an_mp4_error(self, tmp_path):
        path = tmp_path / 'old.mp4'
        path.write_bytes(build_movie(visual_entry('mp4v', 64, 48), [bytes(8)]))
        with pytest.raises(MP4Error, match='mp4v'):
            Movie.open(path)

    def test_a_track_of_no_samples_is_refused_and_its_file_closed(self, tmp_path):
        """C14: a movie that cannot be made does not keep its file."""
        path = tmp_path / 'empty.mp4'
        path.write_bytes(build_movie(
            visual_entry('vp09', 64, 48), [], tables={
                'stsc': full_box('stsc', 0, 0, struct.pack('>I', 0)),
                'stco': full_box('stco', 0, 0, struct.pack('>I', 0)),
                'stts': full_box('stts', 0, 0, struct.pack('>I', 0))}))
        gc.collect()
        before = open_descriptors()
        with pytest.raises(MP4Error, match='no samples'):
            Movie.open(path)
        assert open_descriptors() == before

    def test_a_source_with_no_timescale_is_refused(self):
        source = FakeSource(ibbp(1), [0])
        source.timescale = 0
        with pytest.raises(ContainerError, match='timescale'):
            Timeline(source)

    def test_a_source_with_no_samples_is_refused(self):
        with pytest.raises(ContainerError, match='no samples'):
            Timeline(FakeSource([], []))

    def test_the_size_is_what_the_container_states(self, tmp_path):
        """The MP4 of a 64x64 VP9 stream, stating 640x480."""
        path = tmp_path / 'stated.mp4'
        path.write_bytes(build_movie(visual_entry('vp09', 640, 480), [b'\x82' + bytes(15)]))
        with Movie.open(path) as movie:
            assert movie.size == (640, 480)


@pytest.mark.usefixtures('vaapi_available', 'vaapi_decoder_available')
class TestPlayingOnTheGPU:
    def test_a_recording_plays_back_frame_by_frame(self, tmp_path, upload_texture):
        size = (320, 240)
        frames = [gradient_frame(*size, phase=index * 3) for index in range(20)]
        path = tmp_path / 'recorded.mp4'
        encoder = VAAPIEncoder(*size, fps=30, bitrate=8_000_000, gop=8)
        handle = encoder.new_input()
        source = upload_texture(frames[0])
        try:
            with MP4Writer(path, encoder) as writer:
                for index, frame in enumerate(frames):
                    upload_texture.update(source, frame)
                    blit(handle, source, size)
                    writer.write(encoder.encode(handle, index * TICK, TICK))
                writer.write(encoder.flush())
        finally:
            handle.close()
            encoder.close()

        with Movie.open(path) as movie:
            playing = movie.player()
            try:
                for index in [*range(20), 3, 15, 0]:
                    texture = playing.frame_at(index / 30)
                    glBindTexture(GL_TEXTURE_2D, texture)
                    data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
                    glBindTexture(GL_TEXTURE_2D, 0)
                    picture = np.frombuffer(data, np.uint8).reshape(size[1], size[0], 4)
                    difference = np.abs(picture[::-1, :, :3].astype(int)
                                        - frames[index][..., :3].astype(int)).mean()
                    assert difference < 4.0, (index, difference)
            finally:
                playing.close()
