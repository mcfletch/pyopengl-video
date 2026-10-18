"""Raw H.264 files: access units found, ordered for display, and timed.

No GPU. The display order comes from the decoded picture buffer run over the
parsed headers, so on ``high_bpyramid`` it is checked against the composition
times ffmpeg wrote into the MP4 of the same bitstream.
"""
import io
import json
from pathlib import Path

import pytest

from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.mp4 import MP4Reader, split_annexb
from pyopengl_video.vaapi.h264 import BitWriter, StreamError, annexb, nal_unit
from tests.h264_syntax import sps_bytes, write_pps_head

DATA = Path(__file__).parent / 'data' / 'decode'
FIXTURES = json.loads((DATA / 'h264.json').read_text())['streams']


@pytest.mark.parametrize('name', sorted(FIXTURES))
def test_every_picture_is_one_access_unit(name):
    with AnnexBReader(DATA / f'{name}.h264') as reader:
        assert len(reader) == len(FIXTURES[name]['md5'])


@pytest.mark.parametrize('name', sorted(FIXTURES))
def test_the_size_is_the_cropped_one(name):
    with AnnexBReader(DATA / f'{name}.h264') as reader:
        assert reader.size == tuple(FIXTURES[name]['size'])


def test_display_order_matches_the_muxers_composition_times():
    with AnnexBReader(DATA / 'high_bpyramid.h264') as raw, \
            MP4Reader(DATA / 'high_bpyramid.mp4') as movie:
        assert raw.display_order() == movie.video.display_order()
        assert raw.sync_samples == movie.video.sync_samples


def test_samples_are_the_bytes_of_the_access_unit():
    data = (DATA / 'high_bpyramid.h264').read_bytes()
    with AnnexBReader(DATA / 'high_bpyramid.h264') as reader:
        joined = b''.join(reader.sample(index).data for index in range(len(reader)))
    assert joined == data


def test_a_picture_of_two_slices_is_one_sample():
    with AnnexBReader(DATA / 'high_bpyramid.h264') as reader:
        kinds = [unit[0] & 0x1F for unit in split_annexb(reader.sample(1).data)]
    assert kinds.count(1) + kinds.count(5) == 2


def test_timestamps_count_frames_in_display_order():
    with AnnexBReader(DATA / 'high_bpyramid.h264', frame_rate=(30, 1)) as reader:
        step = reader.timescale // 30
        shown = sorted(reader.sample(index).timestamp for index in range(len(reader)))
        assert shown == [step * index for index in range(len(reader))]
        first = reader.display_order()[0]
        assert reader.sample(first).timestamp == 0


def test_the_frame_rate_is_the_one_the_stream_states():
    with AnnexBReader(DATA / 'baseline_cavlc.h264') as reader:
        assert reader.frame_rate == (30, 1)


def untimed_stream(pictures=3):
    """An SPS with no VUI, a PPS, and IDR pictures of one slice each."""
    sps = nal_unit(sps_bytes(), nal_unit_type=7)
    writer = BitWriter()
    write_pps_head(writer)
    pps = nal_unit(writer.rbsp(), nal_unit_type=8)
    units = [sps, pps]
    for index in range(pictures):
        writer = BitWriter()
        writer.ue(0)
        writer.ue(7)
        writer.ue(0)
        writer.u(0, 4)                          # frame_num
        writer.ue(index % 2)                    # idr_pic_id
        writer.u(0, 6)                          # pic_order_cnt_lsb
        writer.flag(0)
        writer.flag(0)
        writer.se(0)
        writer.ue(1)
        units.append(nal_unit(writer.rbsp() + b'\x88' * 8, nal_unit_type=5))
    return annexb(*units)


def test_the_frame_rate_falls_back_when_the_stream_states_none():
    with AnnexBReader(io.BytesIO(untimed_stream())) as reader:
        assert reader.frame_rate == (25, 1)
        assert len(reader) == 3
        assert reader.sync_samples == [0, 1, 2]


def test_a_caller_may_state_the_frame_rate():
    with AnnexBReader(io.BytesIO(untimed_stream()), frame_rate=(30000, 1001)) as reader:
        assert reader.frame_rate == (30000, 1001)
        assert reader.sample(1).timestamp == 3003


def test_key_frames_are_the_idr_pictures():
    with AnnexBReader(DATA / 'baseline_cavlc.h264') as reader:
        assert reader.sync_samples == [0, 8, 16]
        assert reader.keyframe_before(10) == 8


def test_an_open_gop_stream_marks_its_recovery_points():
    """Non-IDR I pictures with a recovery point are where decoding can start."""
    with AnnexBReader(DATA / 'open_gop.h264') as reader:
        assert reader.sync_samples[0] == 0
        assert len(reader.sync_samples) > 1
        for index in reader.sync_samples:
            assert reader.sample(index).keyframe


def test_parameter_sets_can_lead_a_sample():
    with AnnexBReader(DATA / 'high_bpyramid.h264') as reader:
        sample = reader.sample(20, parameter_sets=True)
        kinds = [unit[0] & 0x1F for unit in split_annexb(sample.data)]
        assert kinds[:2] == [7, 8]


def test_a_file_object_is_read_too():
    data = (DATA / 'baseline_cavlc.h264').read_bytes()
    with AnnexBReader(io.BytesIO(data)) as reader:
        assert len(reader) == len(FIXTURES['baseline_cavlc']['md5'])


def test_a_file_with_no_pictures_is_refused():
    with pytest.raises(StreamError, match='no pictures'):
        AnnexBReader(io.BytesIO(b'\x00\x00\x00\x01\x09\xf0'))


def test_an_sei_made_only_of_ff_bytes_is_not_a_recovery_point():
    """The payload-type and size prefixes are runs of 0xFF; a run cannot walk off the end."""
    from pyopengl_video.annexb import has_recovery_point  # noqa: PLC0415 - used by this case alone

    assert has_recovery_point(b'\xff' * 8) is False
    assert has_recovery_point(b'\x06\xff\xff') is False
