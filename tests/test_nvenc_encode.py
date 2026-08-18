"""The NVENC encoder against real hardware, on synthetic frames."""
import numpy as np
import pytest

from pyopengl_video import EncoderError, open_encoder
from pyopengl_video.nvenc.encoder import NVENCEncoder
from tests.conftest import gradient_frame

SIZE = (640, 480)
FRAMES = 30


@pytest.fixture
def encoder(nvenc_available):
    """A 640x480 encoder with no reordering, so packets come out as frames go in."""
    encoder = NVENCEncoder(*SIZE, fps=30, bitrate=4_000_000, gop=15)
    yield encoder
    encoder.close()


def encode_sequence(encoder, texture, upload, frames=FRAMES, still=False):
    """Push `frames` through the encoder, returning every packet produced.

    still -- leave the texture alone between frames, so the sequence is a held
        picture rather than a moving one
    """
    handle = encoder.register(texture)
    packets = []
    for index in range(frames):
        if not still:
            upload.update(texture, gradient_frame(*SIZE, phase=index))
        packets.extend(encoder.encode(handle, timestamp=index * 3000))
    packets.extend(encoder.flush())
    return packets


def test_the_session_reports_what_it_negotiated(encoder):
    assert encoder.size == SIZE
    assert encoder.zero_copy is True
    assert encoder.reorders_frames is False
    assert encoder.api.version <= encoder.api.driver_version


def test_headers_are_a_sequence_and_picture_parameter_set(encoder):
    headers = encoder.headers()
    assert headers.startswith(b'\x00\x00\x00\x01')
    kinds = {byte & 0x1f for byte in nal_unit_headers(headers)}
    assert kinds == {7, 8}                       # SPS and PPS, and nothing else


def test_every_frame_comes_back_as_a_packet(encoder, upload_texture):
    texture = upload_texture(gradient_frame(*SIZE))
    packets = encode_sequence(encoder, texture, upload_texture)
    assert len(packets) == FRAMES
    assert all(len(packet) > 0 for packet in packets)


def test_the_stream_opens_on_a_key_frame_and_repeats_them_at_the_gop(encoder, upload_texture):
    texture = upload_texture(gradient_frame(*SIZE))
    packets = encode_sequence(encoder, texture, upload_texture)
    keyframes = [index for index, packet in enumerate(packets) if packet.keyframe]
    assert keyframes[0] == 0
    assert keyframes == [0, 15]                  # gop=15 over 30 frames


def test_timestamps_come_back_as_they_went_in(encoder, upload_texture):
    texture = upload_texture(gradient_frame(*SIZE))
    packets = encode_sequence(encoder, texture, upload_texture)
    assert [packet.timestamp for packet in packets] == [i * 3000 for i in range(FRAMES)]
    assert all(packet.duration == 3000 for packet in packets)


def test_a_closed_encoder_refuses_to_encode(encoder, upload_texture):
    texture = upload_texture(gradient_frame(*SIZE))
    handle = encoder.register(texture)
    encoder.close()
    with pytest.raises(EncoderError):
        encoder.encode(handle, timestamp=0)


def test_open_encoder_finds_the_backend(nvenc_available):
    with open_encoder(*SIZE, fps=30) as encoder:
        assert isinstance(encoder, NVENCEncoder)


def test_unknown_settings_are_refused_by_name(nvenc_available):
    with pytest.raises(EncoderError, match='preset'):
        NVENCEncoder(*SIZE, preset='fastest')
    with pytest.raises(EncoderError, match='rate control'):
        NVENCEncoder(*SIZE, rate_control='magic')


# The encoder is handed a texture name, and a registration that quietly went
# nowhere would still produce a well-formed stream -- of something else. These
# check that what comes out varies with what is in the texture, which is the
# part no amount of structural checking can show.

def test_a_detailed_frame_costs_more_than_a_flat_one(nvenc_available, upload_texture):
    """A key frame of noise is far larger than a key frame of one colour."""
    sizes = {}
    for name, frame in (('flat', flat_frame(*SIZE)), ('noise', noise_frame(*SIZE))):
        encoder = NVENCEncoder(*SIZE, fps=30, bitrate=20_000_000, gop=30)
        texture = upload_texture(frame)
        packets = encoder.encode(encoder.register(texture), timestamp=0)
        packets.extend(encoder.flush())
        sizes[name] = sum(len(packet) for packet in packets)
        encoder.close()
    assert sizes['noise'] > sizes['flat'] * 10, sizes


def test_a_still_sequence_costs_less_than_a_moving_one(nvenc_available, upload_texture):
    """Frames that do not change compress to almost nothing after the first.

    At a constant quantiser the size of a picture is what it cost to describe;
    under a bitrate target the encoder spends its budget either way, which
    measures the rate control rather than the content.
    """
    sizes = {}
    for name, still in (('still', True), ('moving', False)):
        encoder = NVENCEncoder(*SIZE, fps=30, rate_control='constqp', gop=FRAMES * 2)
        texture = upload_texture(gradient_frame(*SIZE))
        packets = encode_sequence(encoder, texture, upload_texture, still=still)
        # after the opening key frame, what did the rest of the sequence cost?
        sizes[name] = sum(len(packet) for packet in packets[1:])
        encoder.close()
    assert sizes['moving'] > sizes['still'] * 5, sizes


def flat_frame(width, height):
    """One mid-grey covering the whole frame."""
    frame = np.empty((height, width, 4), dtype=np.uint8)
    frame[...] = (128, 128, 128, 255)
    return frame


def noise_frame(width, height, seed=1234):
    """Random pixels, which no encoder can predict its way out of."""
    frame = np.empty((height, width, 4), dtype=np.uint8)
    frame[..., :3] = np.random.default_rng(seed).integers(
        0, 256, size=(height, width, 3), dtype=np.uint8)
    frame[..., 3] = 255
    return frame


def nal_unit_headers(stream):
    """The first byte of every NAL unit in an Annex-B `stream`."""
    headers = []
    position = 0
    while True:
        start = stream.find(b'\x00\x00\x01', position)
        if start < 0:
            return headers
        headers.append(stream[start + 3])
        position = start + 3


# B-pictures are predicted from frames on both sides of them, so the encoder
# cannot emit one until it has the frame after it. Packets then arrive in decode
# order, several at a time, carrying display timestamps that run out of order.

def test_b_frames_are_held_back_and_come_out_reordered(nvenc_available, upload_texture):
    encoder = NVENCEncoder(*SIZE, fps=30, bitrate=4_000_000, gop=30, bframes=2)
    try:
        assert encoder.reorders_frames is True
        # a texture the encoder is still holding cannot be handed back to it,
        # so a reordering encoder is fed from a ring
        ring = [encoder.register(upload_texture(gradient_frame(*SIZE)))
                for _ in range(encoder.input_slots)]
        counts, packets = [], []
        for index in range(12):
            handle = ring[index % len(ring)]
            upload_texture.update(handle.texture, gradient_frame(*SIZE, phase=index))
            produced = encoder.encode(handle, timestamp=index * 3000)
            counts.append(len(produced))
            packets.extend(produced)
        packets.extend(encoder.flush())
    finally:
        encoder.close()

    assert 0 in counts, 'no frame was held back, so none was coded as a B-picture'
    assert max(counts) > 1, 'held frames were never released together'
    timestamps = [packet.timestamp for packet in packets]
    assert sorted(timestamps) == [index * 3000 for index in range(12)]
    assert timestamps != sorted(timestamps), 'decode order matched display order'


def test_reusing_a_texture_the_encoder_still_holds_says_so(nvenc_available, upload_texture):
    """The driver's own answer to this is a bare MAP_FAILED."""
    encoder = NVENCEncoder(*SIZE, fps=30, bframes=2, gop=30)
    try:
        handle = encoder.register(upload_texture(gradient_frame(*SIZE)))
        with pytest.raises(EncoderError, match='cycle through'):
            for index in range(4):
                encoder.encode(handle, timestamp=index * 3000)
    finally:
        encoder.close()
