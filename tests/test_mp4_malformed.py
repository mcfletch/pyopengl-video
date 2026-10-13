"""An MP4 from outside is untrusted: a damaged or hostile file is an MP4Error.

No GPU. Each case is a few hundred bytes whose header fields claim far more
than the file holds; reading one must refuse it by name, quickly, without
allocating what it claims.
"""
import io
import struct

import pytest

from pyopengl_video.mp4 import MP4Error, MP4Reader, box, full_box
from tests.mp4_build import movie, visual_entry

ENTRY = visual_entry('avc1', 64, 48)
SAMPLES = [bytes([index]) * 10 for index in range(4)]
HUGE = 0xFFFFFFFF


def read(**tables):
    return MP4Reader(io.BytesIO(movie(ENTRY, SAMPLES, tables=tables)))


def test_the_crafted_files_are_otherwise_readable():
    with read() as reader:
        assert len(reader.tracks[0]) == len(SAMPLES)


@pytest.mark.parametrize('tables', [
    {'stsz': full_box('stsz', 0, 0, struct.pack('>II', 4, HUGE))},
    {'stsz': full_box('stsz', 0, 0, struct.pack('>II', 0, HUGE))},
    {'stco': full_box('stco', 0, 0, struct.pack('>I', HUGE))},
    {'stsc': full_box('stsc', 0, 0, struct.pack('>I', HUGE))},
    {'stts': full_box('stts', 0, 0, struct.pack('>I', HUGE))},
    {'ctts': full_box('ctts', 0, 0, struct.pack('>I', HUGE))},
    {'stss': full_box('stss', 0, 0, struct.pack('>I', HUGE))},
    {'stsz': full_box('stsz', 0, 0, b'')},
    {'stts': full_box('stts', 0, 0, b'')},
    {'stsd': full_box('stsd', 0, 0, struct.pack('>I', 1))},
    {'stsc': full_box('stsc', 0, 0, struct.pack('>IIIIIII', 2, 1, 4, 1, 0, 1, 1))},
], ids=['stsz-constant-count', 'stsz-count', 'stco-count', 'stsc-count', 'stts-count',
        'ctts-count', 'stss-count', 'stsz-empty', 'stts-empty', 'stsd-no-entries',
        'stsc-chunk-zero'])
def test_a_table_that_claims_more_than_it_holds_is_refused(tables):
    """Refused when the track's tables are first read, which is before any sample is."""
    with pytest.raises(MP4Error), read(**tables) as reader:
        len(reader.tracks[0])


def test_a_run_count_is_not_expanded_past_the_samples_there_are():
    """One ``stts`` run of 2**32-1 samples costs the track's length, not the claim."""
    with read(stts=full_box('stts', 0, 0, struct.pack('>III', 1, HUGE, 1))) as reader:
        assert len(reader.tracks[0]) == len(SAMPLES)
    offsets = struct.pack('>II', HUGE, 5)
    with read(ctts=full_box('ctts', 0, 0, struct.pack('>I', 1), offsets)) as reader:
        assert len(reader.tracks[0]) == len(SAMPLES)


def test_a_box_claiming_more_than_the_file_is_refused_before_it_is_read():
    data = bytearray(movie(ENTRY, SAMPLES))
    moov = bytes(data).index(b'moov') - 4
    struct.pack_into('>I', data, moov, HUGE)
    with pytest.raises(MP4Error):
        MP4Reader(io.BytesIO(bytes(data)))


def test_a_header_cut_short_by_the_end_of_the_file_is_refused():
    data = box('ftyp', b'isom') + struct.pack('>I4s', 1, b'free')
    with pytest.raises(MP4Error):
        MP4Reader(io.BytesIO(data))


def test_a_sample_beyond_the_end_of_the_file_is_refused_without_reading():
    with read(stsz=full_box('stsz', 0, 0, struct.pack('>II', 0xFFFFFFF0, 1))) as reader:
        with pytest.raises(MP4Error, match=r'a file of \d+ bytes'):
            reader.tracks[0].read(0)


def test_a_short_movie_header_is_refused():
    data = movie(ENTRY, SAMPLES)
    start = data.index(b'mvhd') - 4
    size = struct.unpack_from('>I', data, start)[0]
    shortened = data[:start] + box('mvhd', b'\x00' * 6) + data[start + size:]
    # the moov box that holds it now claims the wrong length
    moov = shortened.index(b'moov') - 4
    old = struct.unpack_from('>I', shortened, moov)[0]
    shortened = bytearray(shortened)
    struct.pack_into('>I', shortened, moov, old - (size - 14))
    with pytest.raises(MP4Error):
        MP4Reader(io.BytesIO(bytes(shortened)))
