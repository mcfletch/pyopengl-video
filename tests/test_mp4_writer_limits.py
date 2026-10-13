"""What an MP4Writer refuses, and when: before the file is the worse for it.

No GPU. A value the file has no field for is refused where it is given, in the
constructor or in :meth:`MP4Writer.write`, so every packet a movie has accepted
is one :meth:`MP4Writer.close` can describe.
"""
import gc
import logging

import pytest

from pyopengl_video.backend import VideoError
from pyopengl_video.encoder import Packet
from pyopengl_video.mp4 import (
    REORDER_LIMIT,
    AVCDescription,
    MP4Error,
    MP4Reader,
    MP4Writer,
    SampleTimes,
)
from tests.descriptors import open_descriptors

TIMESCALE = 90000
FRAME = 3000
#: A VP9 key frame's first byte, and enough after it to be a sample.
KEY = b'\x82' + bytes(31)


def writer(tmp_path, **settings):
    settings = {'width': 64, 'height': 64, 'timescale': TIMESCALE, 'codec': 'vp9', **settings}
    return MP4Writer(tmp_path / 'out.mp4', **settings)


def packet(timestamp, duration=FRAME):
    return Packet(data=KEY, timestamp=timestamp, duration=duration, keyframe=True)


def recorded(tmp_path, timestamps, duration=FRAME, **settings):
    """A movie of one packet at each of `timestamps`, read back."""
    with writer(tmp_path, **settings) as movie:
        movie.write(packet(timestamp, duration) for timestamp in timestamps)
    return MP4Reader(tmp_path / 'out.mp4')


class TestTheConstructor:
    @pytest.mark.parametrize('settings', [
        {'width': 0}, {'height': 0}, {'width': -4}, {'width': 70000}, {'height': 65536},
        {'width': 64.5}, {'timescale': 0}, {'timescale': -1}, {'timescale': 2 ** 32},
        {'timescale': 29.97}])
    def test_a_value_the_file_cannot_state_is_refused(self, tmp_path, settings):
        with pytest.raises(MP4Error):
            writer(tmp_path, **settings)

    def test_the_refusal_is_a_video_error_and_a_value_error(self, tmp_path):
        with pytest.raises(VideoError):
            writer(tmp_path, width=70000)
        with pytest.raises(ValueError, match='70000'):
            writer(tmp_path, width=70000)

    def test_nothing_is_created_or_left_open(self, tmp_path):
        gc.collect()
        before = open_descriptors()
        for settings in ({'width': 70000}, {'timescale': 0}, {'codec': 'prores'}):
            with pytest.raises(ValueError):
                writer(tmp_path, **settings)
        assert open_descriptors() == before
        assert not (tmp_path / 'out.mp4').exists()

    def test_the_largest_sizes_are_taken(self, tmp_path):
        with writer(tmp_path, width=65535, height=65535, timescale=2 ** 32 - 1) as movie:
            movie.write(packet(0))
        with MP4Reader(tmp_path / 'out.mp4') as reader:
            assert reader.video.size == (65535, 65535)
            assert reader.video.timescale == 2 ** 32 - 1


class TestTimestamps:
    def test_a_first_timestamp_the_offsets_cannot_reach_is_refused_at_write(self, tmp_path):
        with writer(tmp_path) as movie:
            movie.write(packet(0))
        with pytest.raises(MP4Error, match='first timestamp'), writer(tmp_path) as movie:
            movie.write(packet(2 ** 31))

    def test_the_packets_before_a_refusal_are_a_whole_movie(self, tmp_path):
        """What a monotonic clock gives: a recording that starts 6.6 hours in."""
        movie = writer(tmp_path)
        movie.write([packet(0), packet(FRAME)])
        with pytest.raises(MP4Error, match='4294967295'):
            movie.write(packet(FRAME + 2 ** 32 + 10))
        movie.close()
        with MP4Reader(tmp_path / 'out.mp4') as reader:
            assert len(reader.video) == 2

    def test_a_timestamp_far_behind_the_latest_is_refused(self, tmp_path):
        with writer(tmp_path) as movie:
            movie.write([packet(0), packet(REORDER_LIMIT + FRAME)])
            with pytest.raises(MP4Error, match='behind'):
                movie.write(packet(FRAME - 1))

    @pytest.mark.parametrize('timestamp', [-REORDER_LIMIT - 1, 2 ** 63, 1.5, None])
    def test_a_timestamp_that_is_no_time_is_refused(self, tmp_path, timestamp):
        with writer(tmp_path) as movie:
            movie.write(packet(0))
            with pytest.raises(MP4Error, match='timestamp'):
                movie.write(packet(timestamp))

    @pytest.mark.parametrize('duration', [-1, 2 ** 32, 0.5])
    def test_a_duration_the_table_cannot_state_is_refused(self, tmp_path, duration):
        with writer(tmp_path) as movie:
            movie.write(packet(0))
            with pytest.raises(MP4Error, match='duration'):
                movie.write(packet(FRAME, duration))

    def test_a_negative_first_timestamp_is_kept(self, tmp_path):
        with recorded(tmp_path, [-FRAME, 0, FRAME]) as reader:
            assert [reader.video.presentation_time(index) for index in range(3)] == [
                -FRAME, 0, FRAME]

    def test_the_latest_first_timestamp_is_kept(self, tmp_path):
        first = REORDER_LIMIT - 1
        with recorded(tmp_path, [first, first + FRAME]) as reader:
            assert reader.video.presentation_time(0) == first

    def test_a_repeated_timestamp_whose_offset_does_not_fit_is_refused(self, tmp_path):
        """Repeats put decode times on the packets' durations, here all nought."""
        with writer(tmp_path) as movie:
            movie.write([packet(0, 0), packet(REORDER_LIMIT, 0), packet(2 ** 31 + 5, 0)])
            with pytest.raises(MP4Error, match='repeats'):
                movie.write(packet(0, 0))
        with MP4Reader(tmp_path / 'out.mp4') as reader:
            assert len(reader.video) == 3

    def test_every_accepted_run_of_timestamps_closes(self, tmp_path):
        """Any packets write() takes, close() can describe."""
        import random  # noqa: PLC0415 - this case only

        chooser = random.Random(20261013)
        steps = [0, 1, FRAME, -FRAME, 2 ** 20, -(2 ** 29), 2 ** 31, 2 ** 32 - 1, 2 ** 32]
        for _ in range(60):
            path = tmp_path / 'fuzz.mp4'
            movie = MP4Writer(path, width=64, height=64, timescale=TIMESCALE, codec='vp9')
            time, taken = chooser.choice([0, -5, 2 ** 29, 2 ** 30 - 1]), 0
            for _ in range(chooser.randrange(1, 12)):
                try:
                    movie.write(packet(time, chooser.choice([0, FRAME, 2 ** 32 - 1])))
                    taken += 1
                except MP4Error:
                    pass
                time += chooser.choice(steps)
            if not taken:
                with pytest.raises(MP4Error):
                    movie.close()
                continue
            movie.close()
            with MP4Reader(path) as reader:
                assert len(reader.video) == taken


class TestSampleTimes:
    """The rule, with no file."""

    def test_durations_are_the_gaps_between_display_times(self):
        times = SampleTimes()
        for timestamp in (0, 2 * FRAME, FRAME):
            times.add(timestamp, 777)
        assert times.sample_durations() == [FRAME, FRAME, 777]
        assert times.composition_offsets(times.sample_durations()) == [0, FRAME, -FRAME]

    def test_repeated_timestamps_leave_the_packets_durations(self):
        times = SampleTimes()
        for _ in range(3):
            times.add(0, 5)
        assert times.sample_durations() == [5, 5, 5]

    def test_a_refused_packet_changes_nothing(self):
        times = SampleTimes()
        times.add(0, FRAME)
        with pytest.raises(MP4Error):
            times.add(2 ** 40, FRAME)
        assert times.timestamps == [0]
        times.add(FRAME, FRAME)
        assert times.sample_durations() == [FRAME, FRAME]


class TestLongMovies:
    def test_a_movie_longer_than_32_bits_of_ticks_has_64_bit_headers(self, tmp_path):
        """At 500 ticks a second the movie header's milliseconds pass 32 bits too."""
        step = 2 ** 32 - 1
        with recorded(tmp_path, [0, step, 2 * step], duration=step, timescale=500) as reader:
            assert reader.video.duration == 3 * step
            assert reader.video.presentation_time(2) == 2 * step
            assert reader.duration == pytest.approx(3 * step / 500)
        data = (tmp_path / 'out.mp4').read_bytes()
        for name in (b'mvhd', b'tkhd', b'mdhd'):
            assert data[data.index(name) + 4] == 1, name

    def test_each_header_is_as_wide_as_its_own_duration_needs(self, tmp_path):
        """At 90 kHz the track passes 32 bits of ticks and the movie's milliseconds do not."""
        step = 2 ** 32 - 1
        recorded(tmp_path, [0, step, 2 * step], duration=step).close()
        data = (tmp_path / 'out.mp4').read_bytes()
        versions = {name: data[data.index(name) + 4] for name in (b'mvhd', b'tkhd', b'mdhd')}
        assert versions == {b'mvhd': 0, b'tkhd': 0, b'mdhd': 1}

    def test_a_short_movie_keeps_32_bit_headers(self, tmp_path):
        recorded(tmp_path, [0, FRAME]).close()
        data = (tmp_path / 'out.mp4').read_bytes()
        for name in (b'mvhd', b'tkhd', b'mdhd'):
            assert data[data.index(name) + 4] == 0, name


class TestTheConfiguration:
    def test_a_parameter_set_no_entry_can_be_made_from_is_refused_at_write(self, tmp_path):
        movie = MP4Writer(tmp_path / 'out.mp4', width=64, height=64, timescale=TIMESCALE)
        start = b'\x00\x00\x00\x01'
        with pytest.raises(MP4Error, match='sequence parameter set'):
            movie.write(Packet(data=start + b'\x67\x64' + start + b'\x68\xeb' + start + b'\x65\x00',
                               timestamp=0, duration=FRAME, keyframe=True))
        with pytest.raises(MP4Error):
            movie.close()

    def test_a_description_counts_its_changes(self):
        description = AVCDescription()
        before = description.revision
        description.store(b'\x00\x00\x01\x67\x64\x00\x28\x00\x00\x01\x68\xeb')
        assert description.revision == before + 2
        description.store(b'\x00\x00\x01\x67\x64\x00\x28\x00\x00\x01\x68\xeb')
        assert description.revision == before + 2


class TestLeavingTheBlock:
    def test_an_exception_in_the_block_is_the_one_raised(self, tmp_path):
        with pytest.raises(RuntimeError, match='the encoder failed'):
            with writer(tmp_path):
                raise RuntimeError('the encoder failed')

    def test_a_movie_that_was_never_begun_is_removed(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError):
            with writer(tmp_path):
                raise RuntimeError('the encoder failed')
        assert not (tmp_path / 'out.mp4').exists()
        assert 'no samples' in caplog.text

    def test_what_was_recorded_before_the_exception_is_a_whole_movie(self, tmp_path):
        with pytest.raises(RuntimeError):
            with writer(tmp_path) as movie:
                movie.write([packet(0), packet(FRAME)])
                raise RuntimeError('the encoder failed')
        with MP4Reader(tmp_path / 'out.mp4') as reader:
            assert len(reader.video) == 2

    def test_closing_an_empty_movie_removes_the_file_and_says_why(self, tmp_path):
        movie = writer(tmp_path)
        with pytest.raises(MP4Error, match='no samples'):
            movie.close()
        assert not (tmp_path / 'out.mp4').exists()
        movie.close()
