"""Encoder and muxer together: textures in, a playable movie out."""
import struct

import pytest

from pyopengl_video import open_encoder
from pyopengl_video.mp4 import MP4Writer
from tests.conftest import gradient_frame
from tests.test_mp4 import boxes, path_to

SIZE = (320, 240)
FRAMES = 24


@pytest.fixture
def movie(tmp_path, nvenc_available, upload_texture):
    """Record a short clip of a moving gradient and return the file's bytes."""
    target = tmp_path / 'clip.mp4'
    texture = upload_texture(gradient_frame(*SIZE))
    with open_encoder(*SIZE, fps=24, bitrate=2_000_000, gop=12) as encoder:
        handle = encoder.register(texture)
        with MP4Writer(target, encoder) as recording:
            for index in range(FRAMES):
                upload_texture.update(texture, gradient_frame(*SIZE, phase=index))
                recording.write(encoder.encode(
                    handle, timestamp=index * encoder.timescale // 24))
            recording.write(encoder.flush())
    return target.read_bytes()


def test_the_recording_is_a_complete_mp4(movie):
    top = boxes(movie)
    assert list(top) == ['ftyp', 'mdat', 'moov']
    assert len(top['mdat'][1]) > 0


def test_the_recording_holds_every_frame(movie):
    stsz = path_to(movie, 'moov', 'trak', 'mdia', 'minf', 'stbl', 'stsz')
    assert struct.unpack_from('>I', stsz, 8)[0] == FRAMES


def test_the_recording_declares_the_encoder_size_and_timescale(movie):
    tkhd = path_to(movie, 'moov', 'trak', 'tkhd')
    assert struct.unpack_from('>II', tkhd, 76) == (SIZE[0] << 16, SIZE[1] << 16)
    mdhd = path_to(movie, 'moov', 'trak', 'mdia', 'mdhd')
    assert struct.unpack_from('>I', mdhd, 12)[0] == 90000


def test_the_parameter_sets_came_from_the_encoder(movie):
    avcc = path_to(movie, 'moov', 'trak', 'mdia', 'minf', 'stbl', 'stsd', 'avc1', 'avcC')
    assert avcc[0] == 1                          # configuration version
    assert avcc[1] in (0x42, 0x4d, 0x64)         # baseline, main or high profile
    assert avcc[4] & 0x03 == 3                   # four-byte length prefixes


def test_the_key_frames_are_where_the_gop_puts_them(movie):
    stss = path_to(movie, 'moov', 'trak', 'mdia', 'minf', 'stbl', 'stss')
    count = struct.unpack_from('>I', stss, 4)[0]
    assert list(struct.unpack_from(f'>{count}I', stss, 8)) == [1, 13]


def test_a_reordered_recording_records_composition_times(tmp_path, nvenc_available,
                                                         upload_texture):
    """With B-pictures the file needs composition offsets, and gets them."""
    target = tmp_path / 'bframes.mp4'
    with open_encoder(*SIZE, fps=24, bitrate=2_000_000, gop=24, bframes=2) as encoder:
        ring = [encoder.register(upload_texture(gradient_frame(*SIZE)))
                for _ in range(encoder.input_slots)]
        with MP4Writer(target, encoder) as recording:
            for index in range(FRAMES):
                handle = ring[index % len(ring)]
                upload_texture.update(handle.texture, gradient_frame(*SIZE, phase=index))
                recording.write(encoder.encode(
                    handle, timestamp=index * encoder.timescale // 24))
            recording.write(encoder.flush())

    stbl = path_to(target.read_bytes(), 'moov', 'trak', 'mdia', 'minf', 'stbl')
    assert 'ctts' in boxes(stbl)
    stsz = path_to(target.read_bytes(), 'moov', 'trak', 'mdia', 'minf', 'stbl', 'stsz')
    assert struct.unpack_from('>I', stsz, 8)[0] == FRAMES
