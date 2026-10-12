"""The MP4 writer on HEVC, AV1 and VP9, against the files ``ffmpeg`` wrote of the same streams.

Each fixture exists as an elementary stream and as ``ffmpeg``'s MP4 of it. The
writer is given the elementary stream's samples as packets, and what it stores
must be what ``ffmpeg`` stored.
"""
from pathlib import Path

import pytest

from pyopengl_video import codecs, obu
from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.encoder import Packet
from pyopengl_video.ivf import IVFReader
from pyopengl_video.mp4 import MP4Reader, MP4Writer
from tests.conftest import decode_errors

DATA = Path(__file__).parent / 'data' / 'decode'

#: codec -> (the elementary stream, its reader, ffmpeg's MP4 of it)
STREAMS = {
    'hevc': ('hevc_bpyramid.h265', AnnexBReader, 'hevc_bpyramid.mp4'),
    'av1': ('av1_altref.ivf', IVFReader, 'av1_altref.mp4'),
    'vp9': ('vp9_altref.ivf', IVFReader, 'vp9_altref.mp4'),
}


def configuration(path, record):
    """The payload of the configuration box `record` in the file's sample entry."""
    data = Path(path).read_bytes()
    position = data.index(record.encode('ascii')) - 4
    size = int.from_bytes(data[position:position + 4], 'big')
    return data[position + 8:position + size]


@pytest.fixture(params=sorted(STREAMS))
def written(request, tmp_path):
    """(codec, our MP4 of the stream, ffmpeg's MP4 of it)."""
    codec = request.param
    stream, reader, reference = STREAMS[codec]
    path = tmp_path / f'{codec}.mp4'
    with reader(DATA / stream) as source:
        with MP4Writer(path, codec=codec, width=source.size[0], height=source.size[1],
                       timescale=source.timescale) as movie:
            for index in range(len(source)):
                sample = source.sample(index, parameter_sets=index == 0)
                movie.write(Packet(data=sample.data, timestamp=sample.timestamp,
                                   duration=sample.duration, keyframe=sample.keyframe))
    return codec, path, DATA / reference


def test_the_track_is_read_back_as_the_codec_written(written):
    codec, path, _ = written
    with MP4Reader(path) as reader:
        assert codecs.canonical(reader.video.codec) == codec


def test_every_sample_is_what_ffmpeg_stored_for_the_same_stream(written):
    _, path, reference = written
    with MP4Reader(path) as ours, MP4Reader(reference) as theirs:
        assert len(ours.video) == len(theirs.video)
        for index in range(len(ours.video)):
            assert ours.video.read(index) == theirs.video.read(index), index


def test_the_key_frames_are_the_sync_samples(written):
    _, path, reference = written
    with MP4Reader(path) as ours, MP4Reader(reference) as theirs:
        assert ours.video.sync_samples == theirs.video.sync_samples


def test_the_configuration_decodes_the_stream(written):
    """The headers the reader hands a decoder are the stream's own."""
    _, path, reference = written
    with MP4Reader(path) as ours, MP4Reader(reference) as theirs:
        assert ours.video.parameter_sets() == theirs.video.parameter_sets()


def test_the_file_decodes_without_complaint(written):
    _, path, _ = written
    errors = decode_errors(path)
    if errors is None:
        pytest.skip('ffmpeg is not installed to ask')
    assert errors == []


class TestTheRecords:
    def test_an_hvcc_states_the_profile_level_and_format_of_the_stream(self, tmp_path):
        ours = self.record(tmp_path, 'hevc', 'hvcC')
        theirs = configuration(DATA / 'hevc_bpyramid.mp4', 'hvcC')
        assert ours[:13] == theirs[:13]                 # version, profile, tier, level
        assert ours[16:19] == theirs[16:19]             # chroma format and bit depths
        assert ours[21] & 0x3F == theirs[21] & 0x3F     # layers, nesting, length size

    def test_a_ten_bit_stream_says_so(self, tmp_path):
        path = tmp_path / 'ten.mp4'
        with AnnexBReader(DATA / 'hevc_main10.h265') as source:
            with MP4Writer(path, codec='hevc', width=source.size[0], height=source.size[1],
                           timescale=source.timescale) as movie:
                sample = source.sample(0, parameter_sets=True)
                movie.write(Packet(data=sample.data, timestamp=0, duration=3000, keyframe=True))
        record = configuration(path, 'hvcC')
        assert (record[17] & 7, record[18] & 7) == (2, 2)

    def test_an_av1c_is_the_one_ffmpeg_writes(self, tmp_path):
        ours = self.record(tmp_path, 'av1', 'av1C')
        theirs = configuration(DATA / 'av1_altref.mp4', 'av1C')
        assert ours[:4] == theirs[:4]
        assert ours[4:] == theirs[4:4 + len(ours) - 4]

    def test_a_vpcc_states_profile_zero_at_eight_bits(self, tmp_path):
        record = self.record(tmp_path, 'vp9', 'vpcC')[4:]
        assert record[0] == 0                           # profile
        assert record[1] == 10                          # level 1: the picture is small
        assert record[2] >> 4 == 8

    @staticmethod
    def record(tmp_path, codec, name):
        stream, reader, _ = STREAMS[codec]
        path = tmp_path / f'{codec}.mp4'
        with reader(DATA / stream) as source:
            with MP4Writer(path, codec=codec, width=source.size[0], height=source.size[1],
                           timescale=source.timescale) as movie:
                sample = source.sample(0, parameter_sets=True)
                movie.write(Packet(data=sample.data, timestamp=0, duration=1, keyframe=True))
        return configuration(path, name)


class TestRefusals:
    def test_a_codec_mp4_does_not_store_here_is_named(self, tmp_path):
        with pytest.raises(ValueError, match='ProRes'):
            MP4Writer(tmp_path / 'x.mp4', codec='ProRes', width=64, height=48, timescale=30)

    def test_an_av1_recording_with_no_sequence_header_says_what_is_missing(self, tmp_path):
        writer = MP4Writer(tmp_path / 'x.mp4', codec='av1', width=64, height=48, timescale=30)
        writer.write(Packet(data=b'\x32\x02\x30\x00', timestamp=0, duration=1, keyframe=False))
        with pytest.raises(ValueError, match='sequence header'):
            writer.close()

    def test_headers_given_up_front_stand_until_the_stream_carries_its_own(self, tmp_path):
        with IVFReader(DATA / 'av1_altref.ivf') as source:
            first = source.sample(0).data
        path = tmp_path / 'x.mp4'
        with MP4Writer(path, codec='av1', width=120, height=68, timescale=30,
                       headers=obu.sequence_header(first)) as writer:
            writer.write(Packet(data=obu.without(first, obu.SEQUENCE_HEADER), timestamp=0,
                                duration=1, keyframe=True))
        with MP4Reader(path) as reader:
            assert reader.video.parameter_sets() == obu.sequence_header(first)

