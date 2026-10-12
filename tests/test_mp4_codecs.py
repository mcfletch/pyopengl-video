"""The MP4 reader on HEVC, AV1 and VP9 sample entries, from files built box by box."""
import io
import struct

import pytest

from pyopengl_video import codecs
from pyopengl_video.mp4 import MP4Reader, box, full_box
from tests.mp4_build import movie, visual_entry

VPS = b'\x40\x01\x0c\x01\xff\xff'
SPS = b'\x42\x01\x01\x01\x60'
PPS = b'\x44\x01\xc1\x72'


def hvcc(*arrays, length_size=4):
    """An ``hvcC`` box: the 22 bytes before the arrays, then each array of NAL units."""
    head = bytes([1]) + bytes(20) + bytes([0xFC | (length_size - 1)]) + bytes([len(arrays)])
    body = b''
    for kind, units in arrays:
        body += bytes([0x80 | kind]) + struct.pack('>H', len(units))
        body += b''.join(struct.pack('>H', len(unit)) + unit for unit in units)
    return box('hvcC', head, body)


def lengths(*units, size=4):
    return b''.join(len(unit).to_bytes(size, 'big') + unit for unit in units)


class TestHEVC:
    SLICE = b'\x26\x01\xaf\x08'
    TRAIL = b'\x02\x01\xd0\x10'

    def reader(self, code='hvc1', length_size=4):
        entry = visual_entry(code, 320, 180,
                             hvcc((32, [VPS]), (33, [SPS]), (34, [PPS]),
                                  length_size=length_size))
        samples = [lengths(self.SLICE, size=length_size),
                   lengths(self.TRAIL, self.TRAIL, size=length_size)]
        return MP4Reader(io.BytesIO(movie(entry, samples, sync=[0])))

    @pytest.mark.parametrize('code', ['hvc1', 'hev1'])
    def test_the_track_is_a_video_track_of_hevc(self, code):
        with self.reader(code) as reader:
            video = reader.video
            assert video is not None
            assert video.codec == code
            assert codecs.canonical(video.codec) == 'hevc'
            assert (video.width, video.height) == (320, 180)

    def test_the_parameter_sets_come_from_the_configuration(self):
        with self.reader() as reader:
            video = reader.video
            assert (video.vps, video.sps, video.pps) == ([VPS], [SPS], [PPS])
            assert video.parameter_sets() == b''.join(
                b'\x00\x00\x00\x01' + unit for unit in (VPS, SPS, PPS))

    @pytest.mark.parametrize('length_size', [4, 2])
    def test_samples_come_back_as_annex_b(self, length_size):
        with self.reader(length_size=length_size) as reader:
            video = reader.video
            assert video.sample(0).data == b'\x00\x00\x00\x01' + self.SLICE
            assert video.sample(1).data == (b'\x00\x00\x00\x01' + self.TRAIL) * 2
            assert video.sample(0, parameter_sets=True).data == (
                video.parameter_sets() + b'\x00\x00\x00\x01' + self.SLICE)
            assert video.sync_samples == [0]


class TestAV1:
    SEQUENCE = b'\x0a\x03\x00\x00\x00'
    FRAME = b'\x32\x04\x10\x00\x00\x00'

    def reader(self):
        av1c = box('av1C', bytes([0x81, 0x00, 0x0c, 0x00]), self.SEQUENCE)
        entry = visual_entry('av01', 160, 90, av1c)
        return MP4Reader(io.BytesIO(movie(entry, [self.SEQUENCE + self.FRAME, self.FRAME],
                                          sync=[0])))

    def test_the_track_is_a_video_track_of_av1(self):
        with self.reader() as reader:
            video = reader.video
            assert codecs.canonical(video.codec) == 'av1'
            assert (video.width, video.height) == (160, 90)

    def test_the_configuration_obus_are_the_headers(self):
        with self.reader() as reader:
            assert reader.video.parameter_sets() == self.SEQUENCE

    def test_a_sample_is_its_obus_as_stored(self):
        with self.reader() as reader:
            video = reader.video
            assert video.sample(1).data == self.FRAME
            assert video.sample(0).data == self.SEQUENCE + self.FRAME

    def test_asking_for_the_headers_puts_the_configuration_in_front(self):
        with self.reader() as reader:
            assert reader.video.sample(1, parameter_sets=True).data == (
                self.SEQUENCE + self.FRAME)


class TestVP9:
    FRAMES = (b'\x82\x49\x83\x42' + bytes(8), b'\x86' + bytes(5))

    def reader(self):
        vpcc = full_box('vpcC', 1, 0, bytes([2, 10, 0xA2, 9, 16, 9]), struct.pack('>H', 0))
        entry = visual_entry('vp09', 96, 54, vpcc)
        return MP4Reader(io.BytesIO(movie(entry, list(self.FRAMES), sync=[0])))

    def test_the_track_is_a_video_track_of_vp9(self):
        with self.reader() as reader:
            video = reader.video
            assert codecs.canonical(video.codec) == 'vp9'
            assert (video.width, video.height) == (96, 54)

    def test_it_has_no_headers_apart_from_its_frames(self):
        with self.reader() as reader:
            assert reader.video.parameter_sets() == b''

    def test_a_sample_is_its_frame_as_stored(self):
        with self.reader() as reader:
            video = reader.video
            assert [video.sample(index, parameter_sets=True).data
                    for index in range(2)] == list(self.FRAMES)


class TestOtherCodecs:
    def test_a_video_track_in_a_codec_not_read_is_listed_and_is_not_the_video(self):
        entry = visual_entry('mp4v', 64, 48)
        with MP4Reader(io.BytesIO(movie(entry, [b'\x00' * 8]))) as reader:
            assert reader.video is None
            assert [track.codec for track in reader.tracks] == ['mp4v']
