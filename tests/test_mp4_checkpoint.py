"""A recording that is cut off is still a movie, up to the last checkpoint.

The sample tables of an MP4 are written when the file is finished, so a program
that dies mid-recording leaves samples no player can find. The writer states the
tables as it goes, at checkpoints, each a complete ``moov`` box that the next one
retires by renaming it ``free``.
"""
import struct

import pytest

from pyopengl_video.encoder import Packet
from pyopengl_video.mp4 import MP4Reader, MP4Writer
from tests.test_mp4 import FRAME, PPS, SPS, TIMESCALE, annexb, frames

#: The least media between checkpoints; the tables' own share of the file sets the rest.
SEGMENT = 1


def fat_frames(count, size=4000, gop=15):
    """Packets of about `size` bytes, a key frame every `gop`, the key frames carrying the sets."""
    made = []
    for index in range(count):
        keyframe = index % gop == 0
        unit = bytes([0x65 if keyframe else 0x41]) + bytes([index % 251]) * size
        units = [SPS, PPS, unit] if keyframe else [unit]
        made.append(Packet(data=annexb(*units), timestamp=index * FRAME,
                           duration=FRAME, keyframe=keyframe))
    return made


def top_level(data):
    """Each top-level box as ``(type, offset, size)``; a size of 0 runs to the end."""
    found = []
    position = 0
    while position + 8 <= len(data):
        size, kind = struct.unpack_from('>I4s', data, position)
        if size == 1:
            size = struct.unpack_from('>Q', data, position + 8)[0]
        if size == 0:
            size = len(data) - position
        found.append((kind.decode(), position, size))
        position += size
    return found


def writer_for(path, **options):
    return MP4Writer(path, width=640, height=480, timescale=TIMESCALE,
                     parameter_sets=(SPS, PPS), **options)


def samples_of(path):
    with MP4Reader(path) as movie:
        track = movie.video
        return [track.read(index) for index in range(len(track))]


@pytest.fixture
def packets():
    return fat_frames(60)


def test_a_finished_recording_has_every_sample_and_one_movie_box(tmp_path, packets):
    target = tmp_path / 'done.mp4'
    with writer_for(target, checkpoint=SEGMENT) as movie:
        movie.write(packets)
    kinds = [kind for kind, _, _ in top_level(target.read_bytes())]
    assert kinds[0] == 'ftyp'
    assert kinds.count('moov') == 1 and kinds[-1] == 'moov'
    assert kinds.count('mdat') > 1                   # one segment for each checkpoint
    assert set(kinds) == {'ftyp', 'mdat', 'moov', 'free'}
    assert len(samples_of(target)) == len(packets)


def test_a_finished_recording_reads_back_the_samples_that_were_written(tmp_path, packets):
    plain = tmp_path / 'plain.mp4'
    with writer_for(plain, checkpoint=None) as movie:
        movie.write(packets)
    checked = tmp_path / 'checked.mp4'
    with writer_for(checked, checkpoint=SEGMENT) as movie:
        movie.write(packets)
    assert samples_of(checked) == samples_of(plain)
    with MP4Reader(checked) as first, MP4Reader(plain) as second:
        assert first.duration == second.duration
        assert first.video.sync_samples == second.video.sync_samples


def test_a_recording_that_stops_without_closing_is_a_movie_up_to_its_last_checkpoint(
        tmp_path, packets):
    """What is on disk when the process dies is what a player is given."""
    target = tmp_path / 'cut.mp4'
    movie = writer_for(target, checkpoint=SEGMENT)
    try:
        movie.write(packets)
        on_disk = target.read_bytes()            # the file as the dead process left it
    finally:
        movie.close()
    cut = tmp_path / 'left.mp4'
    cut.write_bytes(on_disk)
    kept = samples_of(cut)
    assert 30 <= len(kept) < len(packets)        # the last checkpoint, not the last packet
    assert kept == samples_of(target)[:len(kept)]


def test_the_samples_after_the_last_checkpoint_are_a_box_that_runs_to_the_end(
        tmp_path, packets):
    target = tmp_path / 'open.mp4'
    movie = writer_for(target, checkpoint=SEGMENT)
    try:
        movie.write(packets[:59])                # checkpoints fall after packets 18, 37 and 58
        movie._file.flush()  # noqa: SLF001 the process is taken to die after its last write
        on_disk = target.read_bytes()
    finally:
        movie.close()
    kind, offset, _ = top_level(on_disk)[-1]
    assert kind == 'mdat'
    assert struct.unpack_from('>I', on_disk, offset)[0] == 0     # to the end of the file


def test_a_recording_without_checkpoints_is_ftyp_mdat_moov(tmp_path, packets):
    target = tmp_path / 'plain.mp4'
    with writer_for(target, checkpoint=None) as movie:
        movie.write(packets)
    assert [kind for kind, _, _ in top_level(target.read_bytes())] == ['ftyp', 'mdat', 'moov']


def test_a_short_recording_is_the_plain_layout_without_asking(tmp_path):
    """Under the default segment size nothing is checkpointed, so small files are unchanged."""
    target = tmp_path / 'short.mp4'
    with writer_for(target) as movie:
        movie.write(frames(30))
    assert [kind for kind, _, _ in top_level(target.read_bytes())] == ['ftyp', 'mdat', 'moov']


def test_closing_straight_after_a_checkpoint_does_not_write_an_empty_segment(tmp_path):
    target = tmp_path / 'edge.mp4'
    movie = writer_for(target, checkpoint=SEGMENT)
    movie.write(fat_frames(17))                    # ends on a checkpoint
    movie.close()
    kinds = [kind for kind, _, _ in top_level(target.read_bytes())]
    assert kinds.count('moov') == 1 and kinds[-1] == 'moov'
    assert len(samples_of(target)) == 17


def test_a_checkpoint_waits_for_the_stream_to_state_its_configuration(tmp_path):
    """Packets that carry no parameter sets cannot be described, so nothing is written yet."""
    target = tmp_path / 'late.mp4'
    with MP4Writer(target, width=640, height=480, timescale=TIMESCALE,
                   checkpoint=SEGMENT) as movie:
        movie.write(fat_frames(40))                # the key frames carry the sets
    assert len(samples_of(target)) == 40


def test_the_checkpoint_size_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match='checkpoint'):
        writer_for(tmp_path / 'x.mp4', checkpoint=-1)


def test_durations_survive_the_checkpoints(tmp_path, packets):
    target = tmp_path / 'timed.mp4'
    with writer_for(target, checkpoint=SEGMENT) as movie:
        movie.write(packets)
    with MP4Reader(target) as movie:
        assert movie.duration == pytest.approx(len(packets) * FRAME / TIMESCALE)
