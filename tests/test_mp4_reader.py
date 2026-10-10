"""The MP4 reader: the writer's own files read back, and one ffmpeg wrote.

No GPU. Every sample :class:`MP4Writer` stores must come back with the bytes,
timestamp and key flag it went in with; a file from another muxer must be read
through its edit list and its second track.
"""
import io
import json
import struct
from pathlib import Path

import pytest

from pyopengl_video.decoder import Sample
from pyopengl_video.encoder import Packet
from pyopengl_video.mp4 import MP4Error, MP4Reader, MP4Writer, split_annexb

DATA = Path(__file__).parent / 'data' / 'decode'
FIXTURES = json.loads((DATA / 'fixtures.json').read_text())['streams']

TIMESCALE = 90000
FRAME = 3000
SPS = bytes([0x67, 0x64, 0x00, 0x28, 0xac, 0xd9, 0x40, 0x50])
PPS = bytes([0x68, 0xeb, 0xe3, 0xcb, 0x22, 0xc0])


def annexb(*units):
    return b''.join(b'\x00\x00\x00\x01' + unit for unit in units)


def slice_unit(index, keyframe):
    return bytes([0x65 if keyframe else 0x41]) + bytes([index % 251]) * (20 + index)


def packets(count=30, gop=15, order=None):
    """Packets as an encoder hands them out; `order` reorders the timestamps."""
    made = []
    for index in range(count):
        keyframe = index % gop == 0
        shown = index if order is None else order[index]
        units = ([SPS, PPS] if keyframe else []) + [slice_unit(index, keyframe)]
        made.append(Packet(data=annexb(*units), timestamp=shown * FRAME,
                           duration=FRAME, keyframe=keyframe))
    return made


@pytest.fixture
def movie(tmp_path):
    def write(made=None):
        target = tmp_path / 'out.mp4'
        with MP4Writer(target, width=640, height=480, timescale=TIMESCALE,
                       parameter_sets=(SPS, PPS)) as writer:
            writer.write(packets() if made is None else made)
        return target
    return write


class TestTheWritersOwnFiles:
    def test_one_video_track_and_no_audio(self, movie):
        with MP4Reader(movie()) as reader:
            assert len(reader.tracks) == 1
            assert reader.video is reader.tracks[0]
            assert reader.audio == []

    def test_the_track_describes_the_stream(self, movie):
        with MP4Reader(movie()) as reader:
            video = reader.video
            assert video.codec == 'avc1'
            assert (video.width, video.height) == (640, 480)
            assert video.timescale == TIMESCALE
            assert video.sps == [SPS]
            assert video.pps == [PPS]
            assert len(video) == 30

    def test_every_sample_comes_back_as_it_was_written(self, movie):
        written = packets()
        with MP4Reader(movie(written)) as reader:
            for index, packet in enumerate(written):
                sample = reader.video.sample(index)
                assert isinstance(sample, Sample)
                units = [unit for unit in split_annexb(packet.data)
                         if unit[0] & 0x1F not in (7, 8)]
                assert sample.data == annexb(*units)
                assert sample.timestamp == packet.timestamp
                assert sample.duration == packet.duration
                assert sample.keyframe == packet.keyframe
                assert sample.index == index

    def test_the_raw_sample_is_length_prefixed(self, movie):
        with MP4Reader(movie()) as reader:
            raw = reader.video.read(1)
            assert struct.unpack_from('>I', raw)[0] == len(raw) - 4

    def test_the_parameter_sets_can_lead_a_sample(self, movie):
        with MP4Reader(movie()) as reader:
            sample = reader.video.sample(0, parameter_sets=True)
            assert sample.data.startswith(annexb(SPS, PPS))
            assert reader.video.parameter_sets() == annexb(SPS, PPS)

    def test_key_frames_are_the_sync_samples(self, movie):
        with MP4Reader(movie()) as reader:
            assert reader.video.sync_samples == [0, 15]
            assert reader.video.keyframe_before(20) == 15
            assert reader.video.keyframe_before(14) == 0

    def test_reordered_pictures_keep_their_display_times(self, movie):
        order = [0, 2, 1, 3, 5, 4] + list(range(6, 30))
        with MP4Reader(movie(packets(order=order))) as reader:
            shown = [reader.video.sample(index).timestamp for index in range(6)]
            assert shown == [value * FRAME for value in order[:6]]
            assert reader.video.display_order()[:6] == [0, 2, 1, 3, 5, 4]

    def test_the_duration_is_the_tracks_length(self, movie):
        with MP4Reader(movie()) as reader:
            assert reader.video.duration == 30 * FRAME
            assert reader.duration == pytest.approx(1.0)

    def test_an_index_past_the_end_is_refused(self, movie):
        with MP4Reader(movie()) as reader, pytest.raises(IndexError):
            reader.video.sample(30)


class TestReadingWithoutTheMedia:
    def test_opening_reads_the_index_and_not_the_media(self, movie):
        path = movie(packets(count=30) + [Packet(
            data=annexb(slice_unit(0, False) * 4000), timestamp=30 * FRAME,
            duration=FRAME, keyframe=False)])
        data = path.read_bytes()

        class Counting(io.BytesIO):
            consumed = 0

            def read(self, size=-1):
                chunk = super().read(size)
                Counting.consumed += len(chunk)
                return chunk

        with MP4Reader(Counting(data)):
            pass
        assert Counting.consumed < len(data) // 4

    def test_a_file_object_is_read_like_a_path(self, movie):
        path = movie()
        with path.open('rb') as handle:
            reader = MP4Reader(handle)
            assert reader.video.sample(3).timestamp == 3 * FRAME
            reader.close()
            assert not handle.closed, 'the caller opened it, the caller closes it'

    def test_a_file_that_is_not_mp4_is_refused(self, tmp_path):
        bad = tmp_path / 'bad.mp4'
        bad.write_bytes(b'not a movie at all' * 4)
        with pytest.raises(MP4Error):
            MP4Reader(bad)

    def test_a_file_with_no_movie_box_is_refused(self, tmp_path):
        bad = tmp_path / 'bad.mp4'
        bad.write_bytes(struct.pack('>I4s', 16, b'ftyp') + b'isom' + bytes(4))
        with pytest.raises(MP4Error, match='moov'):
            MP4Reader(bad)


class TestAFileFromAnotherMuxer:
    """``high_bpyramid.mp4`` from ffmpeg: B-frames, an edit list, AAC beside it."""

    @pytest.fixture
    def reader(self):
        with MP4Reader(DATA / 'high_bpyramid.mp4') as reader:
            yield reader

    def test_both_tracks_are_found(self, reader):
        assert reader.video is not None
        assert [track.handler for track in reader.tracks] == ['vide', 'soun']

    def test_the_picture_size(self, reader):
        assert (reader.video.width, reader.video.height) == tuple(
            FIXTURES['high_bpyramid']['size'])

    def test_every_picture_is_a_sample(self, reader):
        assert len(reader.video) == len(FIXTURES['high_bpyramid']['md5'])

    def test_the_edit_list_puts_the_first_picture_at_zero(self, reader):
        video = reader.video
        shown = sorted(video.sample(index).timestamp for index in range(len(video)))
        assert shown[0] == 0
        step = shown[1] - shown[0]
        assert shown == [step * index for index in range(len(video))]

    def test_display_order_differs_from_decoding_order(self, reader):
        order = reader.video.display_order()
        assert sorted(order) == list(range(len(reader.video)))
        assert order != sorted(order)

    def test_the_samples_are_annex_b_slices(self, reader):
        sample = reader.video.sample(0, parameter_sets=True)
        kinds = [unit[0] & 0x1F for unit in split_annexb(sample.data)]
        assert kinds[:2] == [7, 8]
        assert 5 in kinds

    def test_the_audio_track_is_reported(self, reader):
        [audio] = reader.audio
        assert audio.codec == 'mp4a'
        assert audio.object_type == 0x40           # MPEG-4 audio, AAC here
        assert audio.sample_rate == 8000
        assert audio.channels == 1
        assert len(audio.config) >= 2             # AudioSpecificConfig
        assert len(audio) > 0
        assert audio.read(0)

    def test_audio_samples_carry_their_timestamps(self, reader):
        [audio] = reader.audio
        times = [audio.sample(index).timestamp for index in range(len(audio))]
        assert times == sorted(times)
