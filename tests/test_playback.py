"""Playback: which frame a time shows, when to seek, and decoding ahead.

The player's decisions are tested with a fake decoder, converter and texture
factory, so no GL is needed: a "texture" here is a number, and the fake
converter records which picture went into which. One case at the end plays a
real movie on the hardware decoder.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from OpenGL.GL import GL_RGBA, GL_TEXTURE_2D, GL_UNSIGNED_BYTE, glBindTexture, glGetTexImage

from pyopengl_video.colour import Colour
from pyopengl_video.decoder import DecodedFrame, Decoder, Sample
from pyopengl_video.mp4 import MP4Writer
from pyopengl_video.playback import Movie, Player, Timeline
from pyopengl_video.vaapi.encoder import VAAPIEncoder
from tests.conftest import blit, gradient_frame

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
    """Releases pictures in display order, holding back `depth` of them."""

    def __init__(self, depth=2):
        self.depth = depth
        self.held = []
        self.decoded = []
        self.resets = 0
        self.lent = set()

    def frame(self, sample):
        frame = DecodedFrame(timestamp=sample.timestamp, duration=sample.duration,
                             planes=(), layout='nv12', texture_size=(16, 16),
                             visible=(0, 0, 16, 16), colour=Colour(),
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
        pass


class FakeConverter:
    def __init__(self):
        self.contents = {}

    def convert(self, frame, target):
        self.contents[target] = frame.timestamp

    def close(self):
        pass


class Textures:
    def __init__(self):
        self.made = 0
        self.deleted = []

    def make(self, width, height):  # noqa: ARG002 - the factory's signature
        self.made += 1
        return self.made

    def delete(self, texture):
        self.deleted.append(texture)


def player(source, depth=2, ahead=2):
    decoder, converter, textures = FakeDecoder(depth), FakeConverter(), Textures()
    built = Player(Movie(source, size=(16, 16)), decoder, converter, ahead=ahead,
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
