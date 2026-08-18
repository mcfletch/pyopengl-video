"""The MP4 muxer, on synthetic packets: no GPU and no encoder involved."""
import struct

import pytest

from pyopengl_video.encoder import Packet
from pyopengl_video.mp4 import MP4Writer, split_annexb

TIMESCALE = 90000
FRAME = 3000                       # 30 fps in a 90 kHz timescale
SPS = bytes([0x67, 0x64, 0x00, 0x28, 0xac, 0xd9, 0x40, 0x50])
PPS = bytes([0x68, 0xeb, 0xe3, 0xcb, 0x22, 0xc0])


def annexb(*units):
    """Join NAL units into an Annex-B fragment, four-byte start codes."""
    return b''.join(b'\x00\x00\x00\x01' + unit for unit in units)


def slice_unit(size=64, keyframe=False):
    """A NAL unit that looks like a coded slice, with `size` bytes of payload."""
    return bytes([0x65 if keyframe else 0x41]) + bytes(range(size % 251)) * 1


def frames(count, gop=15):
    """`count` packets, a key frame every `gop`, as an encoder would emit them."""
    made = []
    for index in range(count):
        keyframe = index % gop == 0
        units = [SPS, PPS, slice_unit(40 + index, keyframe)] if keyframe else [
            slice_unit(20 + index)]
        made.append(Packet(data=annexb(*units), timestamp=index * FRAME,
                           duration=FRAME, keyframe=keyframe))
    return made


def boxes(data, start=0, end=None):
    """Parse a box list into ``{type: (offset, payload)}``, outermost only."""
    end = len(data) if end is None else end
    found = {}
    position = start
    while position + 8 <= end:
        size, kind = struct.unpack_from('>I4s', data, position)
        header = 8
        if size == 1:
            size = struct.unpack_from('>Q', data, position + 8)[0]
            header = 16
        if size == 0:
            size = end - position
        found[kind.decode()] = (position, data[position + header:position + size])
        position += size
    return found


def path_to(data, *names):
    """Walk a path of box names, returning the innermost payload."""
    payload = data
    for index, name in enumerate(names):
        found = boxes(payload)
        assert name in found, f'{name!r} missing at {names[:index]}, have {sorted(found)}'
        payload = found[name][1]
        if name == 'stsd':                      # an entry list, not a plain box
            payload = payload[8:]
        elif name == 'avc1':
            payload = payload[78:]
    return payload


@pytest.fixture
def written(tmp_path):
    """Write a 30-frame movie and hand back its bytes."""
    def write(packets=None, **options):
        target = tmp_path / 'out.mp4'
        with MP4Writer(target, width=640, height=480, timescale=TIMESCALE,
                       parameter_sets=(SPS, PPS), **options) as movie:
            movie.write(frames(30) if packets is None else packets)
        return target.read_bytes()
    return write


def test_the_file_is_a_box_tree_with_the_expected_parts(written):
    top = boxes(written())
    assert list(top) == ['ftyp', 'mdat', 'moov']
    assert top['ftyp'][1][:4] == b'isom'


def test_every_packet_becomes_a_sample(written):
    stsz = path_to(written(), 'moov', 'trak', 'mdia', 'minf', 'stbl', 'stsz')
    _version_flags, sample_size, count = struct.unpack_from('>III', stsz, 0)
    assert count == 30
    assert sample_size == 0                     # sizes vary, so they are listed


def test_sample_sizes_count_the_length_prefixed_units(written):
    data = written()
    stsz = path_to(data, 'moov', 'trak', 'mdia', 'minf', 'stbl', 'stsz')
    sizes = list(struct.unpack_from('>30I', stsz, 12))
    mdat = boxes(data)['mdat'][1]
    assert sum(sizes) == len(mdat)
    # the first sample must start with a four-byte length, not a start code
    assert struct.unpack_from('>I', mdat, 0)[0] + 4 <= sizes[0]


def test_parameter_sets_are_in_the_sample_entry_and_not_in_the_samples(written):
    data = written()
    avcc = path_to(data, 'moov', 'trak', 'mdia', 'minf', 'stbl', 'stsd', 'avc1', 'avcC')
    assert SPS in avcc and PPS in avcc
    assert avcc[1:4] == SPS[1:4]                # profile, compatibility, level
    assert b'\x00\x00\x00\x01' not in boxes(data)['mdat'][1]
    assert SPS not in boxes(data)['mdat'][1]


def test_key_frames_are_listed_as_sync_samples(written):
    stss = path_to(written(), 'moov', 'trak', 'mdia', 'minf', 'stbl', 'stss')
    count = struct.unpack_from('>I', stss, 4)[0]
    samples = list(struct.unpack_from(f'>{count}I', stss, 8))
    assert samples == [1, 16]                   # one-based, gop of 15 over 30 frames


def test_timing_is_recorded_as_one_run_of_equal_durations(written):
    stts = path_to(written(), 'moov', 'trak', 'mdia', 'minf', 'stbl', 'stts')
    entry_count = struct.unpack_from('>I', stts, 4)[0]
    assert entry_count == 1
    count, delta = struct.unpack_from('>II', stts, 8)
    assert (count, delta) == (30, FRAME)


def test_the_track_declares_the_frame_size(written):
    tkhd = path_to(written(), 'moov', 'trak', 'tkhd')
    width, height = struct.unpack_from('>II', tkhd, 76)
    assert (width >> 16, height >> 16) == (640, 480)


def test_display_order_is_recorded_when_packets_are_reordered(written):
    """Packets in decode order need composition offsets, and only then."""
    in_order = path_to(written(), 'moov', 'trak', 'mdia', 'minf', 'stbl')
    assert 'ctts' not in boxes(in_order)

    reordered = frames(3)
    # a decode order of I, P, B: the last packet is displayed second
    reordered = [
        Packet(reordered[0].data, timestamp=0, duration=FRAME, keyframe=True),
        Packet(reordered[1].data, timestamp=2 * FRAME, duration=FRAME, keyframe=False),
        Packet(reordered[2].data, timestamp=1 * FRAME, duration=FRAME, keyframe=False),
    ]
    stbl = path_to(written(reordered), 'moov', 'trak', 'mdia', 'minf', 'stbl')
    ctts = boxes(stbl)['ctts'][1]
    count = struct.unpack_from('>I', ctts, 4)[0]
    entries = [struct.unpack_from('>Ii', ctts, 8 + index * 8) for index in range(count)]
    assert [offset for _run, offset in entries] == [0, FRAME, -FRAME]


def test_a_movie_with_no_packets_is_refused(tmp_path):
    with pytest.raises(ValueError, match='no samples'):
        with MP4Writer(tmp_path / 'empty.mp4', width=64, height=64,
                       timescale=TIMESCALE, parameter_sets=(SPS, PPS)):
            pass


def test_split_annexb_finds_units_behind_either_start_code():
    stream = b'\x00\x00\x01' + b'aa' + b'\x00\x00\x00\x01' + b'bbb'
    assert list(split_annexb(stream)) == [b'aa', b'bbb']


def test_split_annexb_keeps_a_start_code_pattern_inside_a_unit():
    """Emulation prevention means 00 00 01 cannot occur inside a coded unit."""
    stream = b'\x00\x00\x00\x01' + bytes([0x41, 0x00, 0x00, 0x03, 0x01, 0x99])
    assert list(split_annexb(stream)) == [bytes([0x41, 0x00, 0x00, 0x03, 0x01, 0x99])]
