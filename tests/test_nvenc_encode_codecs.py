"""NVENC past H.264 on real hardware: HEVC and AV1, where the GPU encodes them.

The driver writes the headers and manages the references for every codec, so
what is held here is what a recording depends on: the stream's own headers,
key frames where they were asked for, timestamps as they went in, B-pictures
reordered, and a file an outside decoder reads without complaint. Each case
runs for the codecs the GPU behind the context offers, and a codec it lacks is
refused by name.
"""
import pytest

from pyopengl_video import EncoderError, EncoderUnavailable, encoders, nvenc, obu, open_encoder
from pyopengl_video.ivf import IVFReader, IVFWriter
from pyopengl_video.mp4 import MP4Reader, MP4Writer, split_annexb
from pyopengl_video.nvenc import api
from pyopengl_video.nvenc.encoder import NVENCEncoder
from pyopengl_video.vaapi.hevc_parse import NAL_PPS, NAL_SPS, NAL_VPS
from tests.conftest import blit, decode_errors, gradient_frame

SIZE = (640, 480)
FRAMES = 30
TICK = 3000                                  # 30 fps in a 90 kHz timescale

#: The codecs this module is about, and the MP4 sample entry each is stored under.
SAMPLE_ENTRIES = {'hevc': 'hvc1', 'av1': 'av01'}

pytestmark = pytest.mark.usefixtures('nvenc_available')


def need(name):
    """Skip the test unless the GPU behind the context encodes the codec named."""
    if name not in nvenc.session_codecs():
        pytest.skip(f'this NVIDIA GPU has no {name} encoder')
    return name


@pytest.fixture(params=sorted(SAMPLE_ENTRIES))
def codec(request, nvenc_available):  # noqa: ARG001 - requested so the session is asked with a context current
    """Each codec past H.264, skipped where this GPU has no encoder for it."""
    return need(request.param)


@pytest.fixture
def hevc(nvenc_available):  # noqa: ARG001 - requested so the session is asked with a context current
    return need('hevc')


@pytest.fixture
def av1(nvenc_available):  # noqa: ARG001 - requested so the session is asked with a context current
    return need('av1')


@pytest.fixture
def make_encoder():
    """A function building encoders that are closed when the test ends."""
    made = []

    def make(*size, **options):
        made.append(NVENCEncoder(*(size or SIZE), fps=30, **options))
        return made[-1]
    yield make
    for encoder in made:
        encoder.close()


def encode_sequence(encoder, upload, frames=FRAMES):
    """Push `frames` moving pictures through the encoder from a ring of textures."""
    ring = [encoder.register(upload(gradient_frame(*encoder.size)))
            for _ in range(encoder.input_slots)]
    packets = []
    for index in range(frames):
        handle = ring[index % len(ring)]
        upload.update(handle.texture, gradient_frame(*encoder.size, phase=index))
        packets += encoder.encode(handle, timestamp=index * TICK, duration=TICK)
    return packets + encoder.flush()


def hevc_kinds(stream):
    """The nal_unit_type of every unit in an Annex-B `stream`."""
    return [(unit[0] >> 1) & 0x3F for unit in split_annexb(stream)]


class TestTheStream:
    def test_the_encoder_reports_its_codec(self, codec, make_encoder):
        encoder = make_encoder(codec=codec, bitrate=4_000_000, gop=15)
        assert encoder.codec == codec
        assert encoder.size == SIZE
        assert encoder.zero_copy is True

    def test_hevc_headers_are_the_three_parameter_sets(self, hevc, make_encoder):
        headers = make_encoder(codec=hevc, gop=15).headers()
        assert headers.startswith(b'\x00\x00\x00\x01')
        assert hevc_kinds(headers) == [NAL_VPS, NAL_SPS, NAL_PPS]

    def test_av1_headers_are_a_sequence_header(self, av1, make_encoder):
        headers = make_encoder(codec=av1, gop=15).headers()
        assert [unit.kind for unit in obu.split(headers)] == [obu.SEQUENCE_HEADER]

    def test_every_frame_comes_back_as_a_packet(self, codec, make_encoder, upload_texture):
        encoder = make_encoder(codec=codec, bitrate=4_000_000, gop=15)
        packets = encode_sequence(encoder, upload_texture)
        assert len(packets) == FRAMES
        assert all(len(packet) > 0 for packet in packets)

    def test_key_frames_come_at_the_gop_and_timestamps_as_they_went_in(
            self, codec, make_encoder, upload_texture):
        encoder = make_encoder(codec=codec, bitrate=4_000_000, gop=15)
        packets = encode_sequence(encoder, upload_texture)
        assert [index for index, packet in enumerate(packets) if packet.keyframe] == [0, 15]
        assert [packet.timestamp for packet in packets] == [
            index * TICK for index in range(FRAMES)]
        assert {packet.duration for packet in packets} == {TICK}

    def test_an_hevc_key_frame_carries_its_parameter_sets(
            self, hevc, make_encoder, upload_texture):
        """A raw stream then decodes from any key frame."""
        encoder = make_encoder(codec=hevc, bitrate=4_000_000, gop=15)
        packets = encode_sequence(encoder, upload_texture)
        for packet in packets:
            carried = {NAL_VPS, NAL_SPS, NAL_PPS} <= set(hevc_kinds(packet.data))
            assert carried is packet.keyframe

    def test_hevc_b_frames_are_held_back_and_come_out_reordered(
            self, hevc, make_encoder, upload_texture):
        encoder = make_encoder(codec=hevc, bitrate=4_000_000, gop=30, bframes=2)
        assert encoder.reorders_frames is True
        packets = encode_sequence(encoder, upload_texture, frames=12)
        timestamps = [packet.timestamp for packet in packets]
        assert sorted(timestamps) == [index * TICK for index in range(12)]
        assert timestamps != sorted(timestamps), 'decode order matched display order'

    def test_a_moving_sequence_costs_more_than_a_still_one(
            self, codec, make_encoder, upload_texture):
        """What comes out varies with what is in the texture."""
        sizes = {}
        for name in ('still', 'moving'):
            encoder = make_encoder(codec=codec, rate_control='constqp', gop=FRAMES * 2)
            handle = encoder.register(upload_texture(gradient_frame(*SIZE)))
            packets = []
            for index in range(FRAMES):
                if name == 'moving':
                    upload_texture.update(handle.texture, gradient_frame(*SIZE, phase=index))
                packets += encoder.encode(handle, timestamp=index * TICK)
            packets += encoder.flush()
            sizes[name] = sum(len(packet) for packet in packets[1:])
        assert sizes['moving'] > sizes['still'] * 5, sizes


class TestRecordings:
    """The recording as a file, read back here and by a decoder outside this package."""

    def record(self, encoder, upload, path):
        """Write FRAMES moving pictures to an MP4 at `path`."""
        ring = [encoder.new_input() for _ in range(encoder.input_slots)]
        source = upload(gradient_frame(*SIZE))
        try:
            with MP4Writer(path, encoder) as movie:
                for index in range(FRAMES):
                    handle = ring[index % len(ring)]
                    upload.update(source, gradient_frame(*SIZE, phase=index))
                    blit(handle, source, SIZE)
                    movie.write(encoder.encode(handle, timestamp=index * TICK, duration=TICK))
                movie.write(encoder.flush())
        finally:
            for handle in ring:
                handle.close()

    def check(self, path, encoder):
        with MP4Reader(path) as movie:
            track = movie.video
            assert track.codec == SAMPLE_ENTRIES[encoder.codec]
            assert (track.width, track.height) == SIZE
            assert len(track) == FRAMES
            assert track.parameter_sets() == encoder.headers()
            assert sorted(track.presentation_time(index) for index in range(FRAMES)) == [
                index * TICK for index in range(FRAMES)]
        complaints = decode_errors(path)
        if complaints is not None:               # ffmpeg is here to ask
            assert complaints == []

    def test_an_mp4_is_stored_under_the_codecs_sample_entry_and_decodes(
            self, codec, make_encoder, upload_texture, tmp_path):
        encoder = make_encoder(codec=codec, bitrate=4_000_000, gop=15)
        self.record(encoder, upload_texture, tmp_path / 'recorded.mp4')
        self.check(tmp_path / 'recorded.mp4', encoder)

    def test_a_reordered_hevc_recording_decodes(self, hevc, make_encoder, upload_texture,
                                                tmp_path):
        """With B-pictures the container holds each sample's composition offset."""
        encoder = make_encoder(codec=hevc, bitrate=4_000_000, gop=15, bframes=2)
        self.record(encoder, upload_texture, tmp_path / 'reordered.mp4')
        self.check(tmp_path / 'reordered.mp4', encoder)

    def test_av1_is_written_as_ivf(self, av1, make_encoder, upload_texture, tmp_path):
        encoder = make_encoder(codec=av1, bitrate=4_000_000, gop=15)
        path = tmp_path / 'recorded.ivf'
        with IVFWriter(path, encoder) as movie:
            movie.write(encode_sequence(encoder, upload_texture))
        with IVFReader(path) as stream:
            assert stream.codec == 'av1'
            assert len(stream) == FRAMES
        complaints = decode_errors(path)
        if complaints is not None:               # ffmpeg is here to ask
            assert complaints == []


class TestWhatTheGPUOffers:
    def test_discovery_lists_the_codecs_the_session_does(self):
        offered = nvenc.session_codecs()
        found = {backend.name: backend for backend in encoders()}['nvenc']
        assert found.codecs == set(offered)
        assert 'h264' in offered
        for name, size in offered.items():
            assert found.largest(name) == size

    def test_a_codec_the_gpu_lacks_is_refused_with_those_it_has(self):
        offered = nvenc.session_codecs()
        lacking = sorted(set(api.CODEC_GUIDS) - set(offered))
        if not lacking:
            pytest.skip('this NVIDIA GPU encodes every codec the backend drives')
        for name in lacking:
            with pytest.raises(EncoderError,
                               match=f'does not encode {name}; it offers {", ".join(sorted(offered))}'):
                NVENCEncoder(*SIZE, codec=name)
            with pytest.raises(EncoderUnavailable, match=f'640x480 {name}'):
                open_encoder(*SIZE, codec=name, backend='nvenc')

    def test_open_encoder_reaches_it_by_any_of_the_codecs_names(self, codec):
        spelling = {'hevc': 'h265', 'av1': 'av01'}[codec]
        with open_encoder(*SIZE, codec=spelling, backend='nvenc', fps=30) as encoder:
            assert isinstance(encoder, NVENCEncoder)
            assert encoder.codec == codec

    def test_hevc_takes_a_frame_wider_than_h264_does(self, hevc, make_encoder, upload_texture):
        """H.264 stops at 4096 samples a side on NVENC, and HEVC goes on to 8192."""
        size = (5120, 2880)
        assert not {backend.name: backend for backend in encoders()}['nvenc'].supports(
            *size, 'h264')
        with pytest.raises(EncoderUnavailable):
            open_encoder(*size, codec='h264', backend='nvenc')
        encoder = make_encoder(*size, codec=hevc, bitrate=40_000_000, gop=30)
        assert encoder.size == size
        packets = encode_sequence(encoder, upload_texture, frames=3)
        assert len(packets) == 3 and packets[0].keyframe
