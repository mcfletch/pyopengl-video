"""Raw H.264 files: access units found, ordered for display, and timed.

No GPU. The display order comes from the decoded picture buffer run over the
parsed headers, so on ``high_bpyramid`` it is checked against the composition
times ffmpeg wrote into the MP4 of the same bitstream.
"""
import gc
import io
import itertools
import json
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest

from pyopengl_video.annexb import AnnexBReader, has_recovery_point
from pyopengl_video.mp4 import MP4Reader, split_annexb
from pyopengl_video.vaapi.h264 import BitWriter, StreamError, annexb, nal_unit
from tests.descriptors import open_descriptors
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
    assert has_recovery_point(b'\xff' * 8) is False
    assert has_recovery_point(b'\x06\xff\xff') is False


# ------------------------------------------------- a stream that is not one


def picture_parameter_set(identifier):
    """A PPS of `identifier`, naming SPS 0."""
    writer = BitWriter()
    writer.ue(identifier)
    writer.ue(0)                                # seq_parameter_set_id
    writer.flag(1)
    writer.flag(0)
    writer.ue(0)
    writer.ue(0)
    writer.ue(0)
    writer.flag(0)
    writer.u(0, 2)
    writer.se(-3)
    writer.se(0)
    writer.se(2)
    writer.flag(1)
    writer.flag(0)
    writer.flag(0)
    return nal_unit(writer.rbsp(), nal_unit_type=8)


def held_by_the_index(sets, pictures=1500):
    """Bytes the index of a stream of `sets` PPS and `pictures` pictures keeps."""
    stream = untimed_stream(pictures)
    first_picture = stream.index(b'\x00\x00\x00\x01\x65')
    more = annexb(*(picture_parameter_set(identifier) for identifier in range(1, sets)))
    data = stream[:first_picture] + more + stream[first_picture:]
    gc.collect()
    tracemalloc.start()
    try:
        reader = AnnexBReader(io.BytesIO(data))
        assert len(reader) == pictures
        kept = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    return kept - len(data), reader


def test_the_parameter_sets_are_held_once_and_not_once_a_picture():
    """C2: 200 parameter sets before 1500 pictures cost 200 sets, not 300,000."""
    few, _ = held_by_the_index(1)
    many, reader = held_by_the_index(200)
    assert many - few < 200_000
    kinds = [unit[0] & 0x1F for unit in split_annexb(reader.sample(1499, True).data)]
    assert kinds.count(8) == 200


def test_a_parameter_set_sent_again_changed_is_the_one_a_later_sample_leads_with():
    stream = untimed_stream(2)
    second_picture = stream.rindex(b'\x00\x00\x00\x01\x65')
    changed = nal_unit(sps_bytes(level_idc=31), nal_unit_type=7)
    data = stream[:second_picture] + annexb(changed) + stream[second_picture:]
    with AnnexBReader(io.BytesIO(data)) as reader:
        first = list(split_annexb(reader.sample(0, parameter_sets=True).data))
        second = list(split_annexb(reader.sample(1, parameter_sets=True).data))
    assert first[0] != changed
    assert second[0] == changed


CHILD = """
import sys
from pathlib import Path
from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.vaapi.h264 import StreamError
reader = AnnexBReader(sys.argv[1])
count = len(reader)
whole = reader.sample(0).data
Path(sys.argv[1]).write_bytes(whole)        # another process rewrites the file
assert reader.sample(0).data == whole
try:
    reader.sample(count - 1)
except StreamError as error:
    print('refused:', error)
"""


def test_a_file_shortened_after_opening_is_a_refusal_and_not_a_fault(tmp_path):
    """C3: the reader holds no mapping of the file, so there is none to fault."""
    path = tmp_path / 'stream.h265'
    path.write_bytes((DATA / 'hevc_slices_wpp.h265').read_bytes())
    child = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', CHILD, str(path)],
                           capture_output=True, text=True, timeout=60, check=False)
    assert child.returncode == 0, child.stderr[-2000:]
    assert child.stdout.startswith('refused:')


def test_a_path_is_indexed_in_pieces_and_not_whole(tmp_path):
    """A long stream costs its index in memory, not its length."""
    data = (DATA / 'high_bpyramid.h264').read_bytes()
    padded = tmp_path / 'padded.h264'
    padded.write_bytes(data + b'\x00\x00\x01\x0c' + b'\xff' * (8 << 20))   # filler data
    gc.collect()
    tracemalloc.start()
    try:
        with AnnexBReader(padded) as reader:
            peak = tracemalloc.get_traced_memory()[1]
            assert len(reader) == len(FIXTURES['high_bpyramid']['md5'])
            assert reader.sample(3).data == AnnexBReader(io.BytesIO(data)).sample(3).data
    finally:
        tracemalloc.stop()
    assert peak < 4 << 20


def test_a_path_and_a_file_object_give_the_same_samples():
    for name in ('high_bpyramid.h264', 'open_gop.h264', 'hevc_bpyramid.h265'):
        data = (DATA / name).read_bytes()
        with AnnexBReader(DATA / name) as by_path, AnnexBReader(io.BytesIO(data)) as whole:
            assert len(by_path) == len(whole)
            assert by_path.display_order() == whole.display_order()
            for index in range(len(whole)):
                assert by_path.sample(index, True) == whole.sample(index, True)


def test_a_stream_that_cannot_be_indexed_leaves_no_file_open(tmp_path):
    """C14."""
    path = tmp_path / 'empty.h264'
    path.write_bytes(b'\x00\x00\x00\x01\x09\xf0')
    empty = tmp_path / 'nothing.h264'
    empty.write_bytes(b'')
    gc.collect()
    before = open_descriptors()
    for refused in (path, empty):
        with pytest.raises(StreamError, match='no pictures'):
            AnnexBReader(refused)
    assert open_descriptors() == before


def test_an_sei_message_cut_short_is_not_a_recovery_point():
    """C9: every byte string of three bytes is answered, none with an IndexError."""
    values = (0, 1, 5, 6, 0x7F, 0x80, 0xFF)
    for length in range(4):
        for message in itertools.product(values, repeat=length):
            assert has_recovery_point(bytes(message)) in (True, False)
    assert has_recovery_point(b'\x06\x01\x00\x80')
    assert not has_recovery_point(b'\xff\xff')
    assert not has_recovery_point(b'\x06\xff')


@pytest.mark.parametrize('rate', [(0, 1), (30, 0), (-30, 1), (2.5, 1)])
def test_a_frame_rate_that_is_no_rate_is_refused(rate):
    with pytest.raises(ValueError, match='frame rate'):
        AnnexBReader(io.BytesIO(untimed_stream()), frame_rate=rate)


@pytest.mark.parametrize('index', [-1, 3])
def test_a_picture_index_outside_the_stream_is_an_index_error(index):
    with AnnexBReader(io.BytesIO(untimed_stream())) as reader, pytest.raises(IndexError):
        reader.sample(index)
