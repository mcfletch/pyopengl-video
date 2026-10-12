"""The IVF reader and writer, on files built here from packets."""
import io
import struct

import pytest

from pyopengl_video.encoder import Packet
from pyopengl_video.ivf import IVFError, IVFReader, IVFWriter, av1_keyframe, vp9_keyframe
from pyopengl_video.vaapi.bits import leb128

#: A VP9 frame's first byte: the frame marker, profile 0, not an existing
#: frame, then frame_type, show_frame and error_resilient_mode.
VP9_KEY = b'\x82' + bytes(15)
VP9_INTER = b'\x86' + bytes(9)
VP9_SHOW_EXISTING = b'\x88'


def obu(kind, payload):
    """One OBU with a size field."""
    return bytes([kind << 3 | 0x02]) + leb128(len(payload)) + payload


#: A sequence header whose first byte is profile 0, not a still picture.
AV1_SEQUENCE = obu(1, bytes(10))
#: A frame's first byte: show_existing_frame, frame_type (2 bits), show_frame.
AV1_KEY = obu(2, b'') + AV1_SEQUENCE + obu(6, bytes([0b0001_0000]) + bytes(20))
AV1_INTER = obu(2, b'') + obu(6, bytes([0b0011_0000]) + bytes(8))
AV1_HIDDEN_KEY = obu(2, b'') + AV1_SEQUENCE + obu(6, bytes([0b0000_0000]) + bytes(8))


def written(tmp_path, codec, frames, timescale=30):
    path = tmp_path / 'stream.ivf'
    with IVFWriter(path, codec=codec, width=320, height=240, timescale=timescale) as stream:
        stream.write(Packet(data=data, timestamp=index, duration=1, keyframe=False)
                     for index, data in enumerate(frames))
    return path


class TestKeyFrames:
    def test_a_vp9_key_frame_is_found_by_its_frame_type(self):
        assert vp9_keyframe(VP9_KEY)
        assert not vp9_keyframe(VP9_INTER)

    def test_a_vp9_frame_showing_an_existing_one_is_not_a_key_frame(self):
        assert not vp9_keyframe(VP9_SHOW_EXISTING)

    def test_bytes_that_are_not_a_vp9_frame_are_not_a_key_frame(self):
        assert not vp9_keyframe(b'')
        assert not vp9_keyframe(b'\x00\x00')

    def test_an_av1_key_frame_is_a_shown_key_frame(self):
        assert av1_keyframe(AV1_KEY)
        assert not av1_keyframe(AV1_INTER)
        assert not av1_keyframe(AV1_HIDDEN_KEY)

    def test_every_frame_of_reduced_still_pictures_is_a_key_frame(self):
        reduced = obu(1, bytes([0b0001_1000]) + bytes(4)) + obu(6, bytes(8))
        assert av1_keyframe(reduced)

    def test_an_av1_unit_cut_short_is_not_a_key_frame(self):
        assert not av1_keyframe(AV1_KEY[:5])
        assert not av1_keyframe(b'')


class TestReading:
    def test_frames_come_back_as_they_went_in(self, tmp_path):
        frames = [VP9_KEY, VP9_INTER, VP9_INTER, VP9_KEY, VP9_INTER]
        with IVFReader(written(tmp_path, 'vp9', frames)) as stream:
            assert stream.codec == 'vp9'
            assert stream.size == (320, 240)
            assert len(stream) == 5
            assert [stream.sample(index).data for index in range(5)] == frames

    def test_key_frames_are_the_sync_samples(self, tmp_path):
        frames = [VP9_KEY, VP9_INTER, VP9_INTER, VP9_KEY, VP9_INTER]
        with IVFReader(written(tmp_path, 'vp9', frames)) as stream:
            assert stream.sync_samples == [0, 3]
            assert [stream.keyframe_before(index) for index in range(5)] == [0, 0, 0, 3, 3]
            assert [stream.sample(index).keyframe for index in range(5)] == [
                True, False, False, True, False]

    def test_av1_key_frames_are_found_past_the_opening_bytes(self, tmp_path):
        padded = obu(2, b'') + obu(15, bytes(100)) + AV1_KEY
        with IVFReader(written(tmp_path, 'av1', [padded, AV1_INTER, AV1_KEY])) as stream:
            assert stream.codec == 'av1'
            assert stream.sync_samples == [0, 2]

    def test_times_are_in_the_headers_time_base(self, tmp_path):
        with IVFReader(written(tmp_path, 'vp9', [VP9_KEY, VP9_INTER, VP9_INTER],
                               timescale=25)) as stream:
            assert stream.timescale == 25
            assert [stream.presentation_time(index) for index in range(3)] == [0, 1, 2]
            assert [stream.sample(index).duration for index in range(3)] == [1, 1, 1]
            assert stream.duration == pytest.approx(3 / 25)

    def test_a_time_base_of_several_units_scales_each_time(self):
        head = struct.pack('<4sHH4sHHIII4x', b'DKIF', 0, 32, b'VP90', 64, 48, 30000, 1001, 2)
        body = b''.join(struct.pack('<IQ', len(VP9_KEY), index) + VP9_KEY for index in range(2))
        stream = IVFReader(io.BytesIO(head + body))
        assert stream.timescale == 30000
        assert [stream.presentation_time(index) for index in range(2)] == [0, 1001]

    def test_a_file_object_stays_open_for_its_owner(self, tmp_path):
        with written(tmp_path, 'vp9', [VP9_KEY]).open('rb') as handle:
            IVFReader(handle).close()
            assert not handle.closed

    def test_a_frame_cut_short_ends_the_stream_before_it(self, tmp_path):
        path = written(tmp_path, 'vp9', [VP9_KEY, VP9_INTER])
        path.write_bytes(path.read_bytes()[:-4])
        with IVFReader(path) as stream:
            assert len(stream) == 1

    def test_it_has_no_parameter_sets(self, tmp_path):
        with IVFReader(written(tmp_path, 'vp9', [VP9_KEY])) as stream:
            assert stream.parameter_sets() == b''


class TestRefusals:
    def test_a_file_that_is_not_ivf_is_refused(self):
        with pytest.raises(IVFError, match='not an IVF file'):
            IVFReader(io.BytesIO(b'\x00\x00\x00\x01' + bytes(60)))

    def test_a_codec_it_does_not_hold_is_named(self):
        head = struct.pack('<4sHH4sHHIII4x', b'DKIF', 0, 32, b'VP80', 64, 48, 30, 1, 0)
        with pytest.raises(IVFError, match='VP80'):
            IVFReader(io.BytesIO(head))

    def test_a_file_with_no_frames_is_refused(self):
        head = struct.pack('<4sHH4sHHIII4x', b'DKIF', 0, 32, b'VP90', 64, 48, 30, 1, 0)
        with pytest.raises(IVFError, match='no frames'):
            IVFReader(io.BytesIO(head))

    def test_the_writer_refuses_a_codec_ivf_does_not_hold(self, tmp_path):
        with pytest.raises(IVFError, match='h264'):
            IVFWriter(tmp_path / 'x.ivf', codec='h264', width=64, height=48, timescale=30)
        assert not (tmp_path / 'x.ivf').exists()

    def test_the_writer_needs_to_know_what_it_writes(self, tmp_path):
        with pytest.raises(ValueError, match='encoder'):
            IVFWriter(tmp_path / 'x.ivf', codec='av1')


class TestWriting:
    def test_the_header_counts_the_frames_written(self, tmp_path):
        path = written(tmp_path, 'av1', [AV1_KEY, AV1_INTER, AV1_INTER])
        assert struct.unpack_from('<I', path.read_bytes(), 24)[0] == 3

    def test_writing_after_close_is_refused(self, tmp_path):
        stream = IVFWriter(tmp_path / 'x.ivf', codec='av1', width=64, height=48, timescale=30)
        stream.close()
        stream.close()
        with pytest.raises(ValueError, match='closed'):
            stream.write(Packet(data=AV1_KEY, timestamp=0, duration=1, keyframe=True))
