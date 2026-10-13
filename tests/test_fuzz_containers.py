"""The fuzz gate for the containers: mutated MP4, IVF and Annex-B files.

No GPU. A few thousand inputs are made from the fixtures in
``tests/data/decode/`` with a fixed seed, by overwriting fields with the
values that break a reader (0, the largest, the sign bit), by cutting the
file short and by damaging several bytes at once. Of every input four things
hold:

- nothing but :class:`MP4Error`, :class:`IVFError` or :class:`StreamError`
  escapes opening it, as a reader and as a :class:`Movie`, and reading every
  sample;
- it takes under a second;
- the memory it takes at its peak is under fifty times the input's size and
  a megabyte;
- every sample returned is as long as the index states.
"""
import io
import random
import struct
import time
import tracemalloc
from pathlib import Path

import pytest

from pyopengl_video.annexb import AnnexBReader, nal_spans
from pyopengl_video.ivf import IVFError, IVFReader
from pyopengl_video.mp4 import MP4Error, MP4Reader, VideoTrack
from pyopengl_video.playback import Movie
from pyopengl_video.vaapi.h264 import StreamError

DATA = Path(__file__).parent / 'data' / 'decode'
SEED = 20261013

#: What a reader may raise for a file it will not read.
REFUSALS = (MP4Error, IVFError, StreamError)
SECONDS = 1.0
#: The peak memory an input may cost: this many times its size, and this much.
TIMES, AND = 50, 1 << 20

FIELDS_32 = (0, 1, 2, 7, 8, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x00FFFFFF, 0xFFFF)


def stated_sizes(source):
    """The size the index of `source` states for each of its samples."""
    if isinstance(source, AnnexBReader):
        return [unit.end - unit.start for unit in source._units]  # noqa: SLF001 - the gate holds each sample to the index's own size for it
    if isinstance(source, IVFReader):
        return list(source._sizes)  # noqa: SLF001 - the gate holds each sample to the index's own size for it
    return list(source._table.sizes)  # noqa: SLF001 - the gate holds each sample to the index's own size for it


def play(data):
    """Open `data` as a movie and walk its timeline, as a player would."""
    with Movie.open(io.BytesIO(data)) as movie:
        repr(movie)
        timeline = movie.timeline
        for rank in {0, len(timeline) // 2, len(timeline) - 1}:
            movie.source.sample(timeline.start_for(rank), parameter_sets=True)


def read_mp4(data):
    with MP4Reader(io.BytesIO(data)) as reader:
        repr(reader)
        assert reader.duration >= 0
        for track in reader.tracks:
            track.display_order()
            for index, size in enumerate(stated_sizes(track)):
                assert len(track.read(index)) == size
                if isinstance(track, VideoTrack):
                    track.sample(index, parameter_sets=True)
                track.keyframe_before(index)
    play(data)


def read_ivf(data):
    with IVFReader(io.BytesIO(data)) as reader:
        repr(reader)
        assert reader.duration == reader.duration
        for index, size in enumerate(stated_sizes(reader)):
            assert len(reader.sample(index).data) == size
            reader.keyframe_before(index)
    play(data)


def read_annexb(data):
    with AnnexBReader(io.BytesIO(data)) as reader:
        repr(reader)
        reader.parameter_sets()
        reader.display_order()
        for index, size in enumerate(stated_sizes(reader)):
            assert len(reader.sample(index).data) == size
            reader.sample(index, parameter_sets=True)
            reader.keyframe_before(index)
    play(data)


def overwritten(base, position, packed):
    return base[:position] + packed + base[position + len(packed):]


def damaged(base, positions, chooser, order):
    """One mutation of `base` at one of `positions`, and what it was."""
    position = chooser.choice(positions)
    kind = chooser.randrange(6)
    if kind == 0:
        value = chooser.choice(FIELDS_32)
        return f'u32@{position}={value:#x}', overwritten(
            base, position, struct.pack(order + 'I', value))
    if kind == 1:
        value = chooser.choice((0, 0xFF, 0x80, base[position] ^ 1, (base[position] + 1) & 0xFF))
        return f'u8@{position}={value:#x}', overwritten(base, position, bytes([value]))
    if kind == 2:
        value = chooser.choice((0, 0xFFFF, 0x8000, 1))
        return f'u16@{position}={value:#x}', overwritten(
            base, position, struct.pack(order + 'H', value))
    if kind == 3:
        return f'cut@{position}', base[:position]
    if kind == 4:
        bit = chooser.randrange(8)
        return f'bit@{position}.{bit}', overwritten(
            base, position, bytes([base[position] ^ (1 << bit)]))
    data = bytearray(base)
    places = [chooser.choice(positions) for _ in range(chooser.randrange(2, 6))]
    for place in places:
        data[place] = chooser.randrange(256)
    return f'bytes@{places}', bytes(data)


def movie_box(data):
    """Where the ``moov`` box of an MP4 starts and ends."""
    position = 0
    while position + 8 <= len(data):
        size, kind = struct.unpack_from('>I4s', data, position)
        if size == 1:
            size = struct.unpack_from('>Q', data, position + 8)[0]
        if kind == b'moov':
            return position, position + size
        position += size
    raise AssertionError('the fixture has no moov box')


def mp4_positions(base):
    start, end = movie_box(base)
    return [*range(48), *range(start, end - 4)]


def ivf_positions(base):
    """The header, every frame header, and the opening of the first frames."""
    positions = [*range(32)]
    position = 32
    while position + 12 <= len(base):
        positions += range(position, position + 12)
        position += 12 + struct.unpack_from('<I', base, position)[0]
    return positions + [*range(44, min(len(base) - 4, 300))]


def annexb_positions(base):
    """The first bytes of every NAL unit, where its headers are."""
    return sorted({position for _, payload, end in nal_spans(base)
                   for position in range(payload, min(end - 4, payload + 48))})


#: fixture -> (how it is read, where it is damaged, byte order, how many inputs)
CASES = {
    'high_bpyramid.mp4': (read_mp4, mp4_positions, '>', 350),
    'hevc_bpyramid.mp4': (read_mp4, mp4_positions, '>', 350),
    'av1_altref.mp4': (read_mp4, mp4_positions, '>', 350),
    'vp9_altref.mp4': (read_mp4, mp4_positions, '>', 350),
    'av1_still.ivf': (read_ivf, ivf_positions, '<', 500),
    'av1_monochrome.ivf': (read_ivf, ivf_positions, '<', 500),
    'vp9_lossless.ivf': (read_ivf, ivf_positions, '<', 500),
    'high_cqm.h264': (read_annexb, annexb_positions, '>', 150),
    'baseline_cavlc.h264': (read_annexb, annexb_positions, '>', 150),
    'open_gop.h264': (read_annexb, annexb_positions, '>', 150),
    'hevc_scaling_default.h265': (read_annexb, annexb_positions, '>', 150),
}


def broken_rule(read, data):
    """Which of the gate's rules `data` breaks, in words, or '' for none."""
    tracemalloc.reset_peak()
    before = tracemalloc.get_traced_memory()[0]
    started = time.perf_counter()
    try:
        read(data)
    except REFUSALS:
        pass
    except Exception as error:  # noqa: BLE001 - any other exception is what the gate reports
        return f'{type(error).__name__}: {str(error)[:80]}'
    took = time.perf_counter() - started
    peak = tracemalloc.get_traced_memory()[1] - before
    if took > SECONDS:
        return f'took {took:.2f} s'
    if peak > TIMES * len(data) + AND:
        return f'took {peak} bytes at its peak, for an input of {len(data)}'
    return ''


@pytest.mark.parametrize('name', sorted(CASES))
def test_no_mutation_of_a_fixture_breaks_a_reader(name):
    read, positions_of, order, count = CASES[name]
    base = (DATA / name).read_bytes()
    chooser = random.Random(f'{SEED} {name}')
    positions = positions_of(base)
    broken = {}
    tracemalloc.start()
    try:
        assert broken_rule(read, base) == '', 'the fixture itself'
        for _ in range(count):
            label, data = damaged(base, positions, chooser, order)
            rule = broken_rule(read, data)
            if rule:
                broken.setdefault(rule.split(':')[0], (label, rule))
    finally:
        tracemalloc.stop()
    assert not broken, '\n'.join(f'{label}: {rule}' for label, rule in broken.values())
