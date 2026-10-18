"""The NVENC encoder against real hardware, on synthetic frames."""
import numpy as np
import pytest
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DRAW_FRAMEBUFFER,
    glBindFramebuffer,
    glClear,
    glClearColor,
)

from pyopengl_video import EncoderError, open_encoder
from pyopengl_video.mp4 import MP4Writer, split_annexb
from pyopengl_video.nvenc import encoder as nvenc_encoder
from pyopengl_video.nvenc.encoder import NVENCEncoder
from tests.conftest import decode_errors, gradient_frame, put_frame

SIZE = (640, 480)
FRAMES = 30
TICK = 3000                                  # 30 fps in a 90 kHz timescale


@pytest.fixture
def encoder(nvenc_available):  # noqa: ARG001 - requested to skip where there is no NVENC
    """A 640x480 encoder with no reordering, so packets come out as frames go in."""
    encoder = NVENCEncoder(*SIZE, fps=30, bitrate=4_000_000, gop=15)
    yield encoder
    encoder.close()


@pytest.fixture
def handles(encoder):
    """The encoder's own inputs, as many as it says a recording needs."""
    made = [encoder.new_input() for _ in range(encoder.input_slots)]
    yield made
    for handle in made:
        handle.close()


def paint(handle, colour):
    """Fill a handle's texture with one colour, inside its drawing scope."""
    with handle.for_drawing():
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, handle.framebuffer)
        glClearColor(*colour)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)


def encode_sequence(encoder, frames=FRAMES, still=False):
    """Push `frames` through the encoder, returning every packet produced.

    still -- leave the pictures alone between frames, so the sequence is a held
        picture rather than a moving one
    """
    ring = [encoder.new_input() for _ in range(encoder.input_slots)]
    for handle in ring:
        put_frame(handle, gradient_frame(*SIZE))
    packets = []
    for index in range(frames):
        handle = ring[index % len(ring)]
        if not still:
            put_frame(handle, gradient_frame(*SIZE, phase=index))
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


def test_every_frame_comes_back_as_a_packet(encoder):
    packets = encode_sequence(encoder)
    assert len(packets) == FRAMES
    assert all(len(packet) > 0 for packet in packets)


def test_the_stream_opens_on_a_key_frame_and_repeats_them_at_the_gop(encoder):
    packets = encode_sequence(encoder)
    keyframes = [index for index, packet in enumerate(packets) if packet.keyframe]
    assert keyframes[0] == 0
    assert keyframes == [0, 15]                  # gop=15 over 30 frames


def test_timestamps_come_back_as_they_went_in(encoder):
    packets = encode_sequence(encoder)
    assert [packet.timestamp for packet in packets] == [i * 3000 for i in range(FRAMES)]
    assert all(packet.duration == 3000 for packet in packets)


def test_a_closed_encoder_refuses_to_encode(encoder):
    handle = encoder.new_input()
    encoder.close()
    with pytest.raises(EncoderError):
        encoder.encode(handle, timestamp=0)


@pytest.mark.usefixtures('nvenc_available')
def test_open_encoder_finds_the_backend():
    with open_encoder(*SIZE, fps=30) as encoder:
        assert isinstance(encoder, NVENCEncoder)


@pytest.mark.usefixtures('nvenc_available')
def test_unknown_settings_are_refused_by_name():
    with pytest.raises(EncoderError, match='preset'):
        NVENCEncoder(*SIZE, preset='fastest')
    with pytest.raises(EncoderError, match='rate control'):
        NVENCEncoder(*SIZE, rate_control='magic')


# The encoder is handed a texture name, and a registration that quietly went
# nowhere would still produce a well-formed stream -- of something else. These
# check that what comes out varies with what is in the texture, which is the
# part no amount of structural checking can show.

@pytest.mark.usefixtures('nvenc_available')
def test_a_detailed_frame_costs_more_than_a_flat_one():
    """A key frame of noise is far larger than a key frame of one colour."""
    sizes = {}
    for name, frame in (('flat', flat_frame(*SIZE)), ('noise', noise_frame(*SIZE))):
        encoder = NVENCEncoder(*SIZE, fps=30, bitrate=20_000_000, gop=30)
        handle = encoder.new_input()
        put_frame(handle, frame)
        packets = encoder.encode(handle, timestamp=0)
        packets.extend(encoder.flush())
        sizes[name] = sum(len(packet) for packet in packets)
        encoder.close()
    assert sizes['noise'] > sizes['flat'] * 10, sizes


@pytest.mark.usefixtures('nvenc_available')
def test_a_still_sequence_costs_less_than_a_moving_one():
    """Frames that do not change compress to almost nothing after the first.

    At a constant quantiser the size of a picture is what it cost to describe;
    under a bitrate target the encoder spends its budget either way, which
    measures the rate control rather than the content.
    """
    sizes = {}
    for name, still in (('still', True), ('moving', False)):
        encoder = NVENCEncoder(*SIZE, fps=30, rate_control='constqp', gop=FRAMES * 2)
        packets = encode_sequence(encoder, still=still)
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

@pytest.mark.usefixtures('nvenc_available')
def test_b_frames_are_held_back_and_come_out_reordered():
    encoder = NVENCEncoder(*SIZE, fps=30, bitrate=4_000_000, gop=30, bframes=2)
    try:
        assert encoder.reorders_frames is True
        # a texture the encoder is still holding cannot be handed back to it,
        # so a reordering encoder is fed from a ring
        ring = [encoder.new_input() for _ in range(encoder.input_slots)]
        counts, packets = [], []
        for index in range(12):
            handle = ring[index % len(ring)]
            put_frame(handle, gradient_frame(*SIZE, phase=index))
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


@pytest.mark.usefixtures('nvenc_available')
def test_reusing_a_texture_the_encoder_still_holds_says_so():
    """The driver's own answer to this is a bare MAP_FAILED."""
    encoder = NVENCEncoder(*SIZE, fps=30, bframes=2, gop=30)
    try:
        handle = encoder.new_input()
        put_frame(handle, gradient_frame(*SIZE))
        with pytest.raises(EncoderError, match='cycle through'):
            for index in range(4):
                encoder.encode(handle, timestamp=index * 3000)
    finally:
        encoder.close()


class TestWhenOpenGLFailsDuringClose:
    """The session is a limited driver resource, and closing must give it back.

    An encoder closed with no context current -- from a destructor after the
    window is gone -- cannot delete its OpenGL objects.
    """

    def test_the_session_is_released_anyway(self, nvenc_available, monkeypatch):  # noqa: ARG002 - requested to skip where there is no NVENC
        import OpenGL.GL  # noqa: PLC0415 - the module whose entry point is replaced

        built = NVENCEncoder(*SIZE, fps=30)
        built.new_input()
        destroyed = []
        destroy = built.api.nvEncDestroyEncoder
        monkeypatch.setattr(built.api, 'nvEncDestroyEncoder',
                            lambda session: destroyed.append(session) or destroy(session))

        def no_context(*_arguments):
            raise RuntimeError('no context is current')

        monkeypatch.setattr(OpenGL.GL, 'glDeleteFramebuffers', no_context)
        with pytest.raises(RuntimeError, match='no context'):
            built.close()
        assert len(destroyed) == 1
        assert built.session.value is None
        assert built.direct3d is None or built.direct3d.device is None
        monkeypatch.undo()
        built.close()                               # already closed; nothing to do


class TestWhatEncodeAccepts:
    def test_an_input_from_another_encoder_is_refused(self, encoder):
        with NVENCEncoder(*SIZE, fps=30) as other:
            foreign = other.new_input()
            with pytest.raises(EncoderError, match='not registered with this encoder'):
                encoder.encode(foreign, timestamp=0)

    def test_a_closed_input_is_refused_by_name(self, encoder):
        handle = encoder.new_input()
        handle.close()
        with pytest.raises(EncoderError, match='not registered with this encoder'):
            encoder.encode(handle, timestamp=0)

    def test_a_failed_submission_leaves_nothing_waiting(self, encoder, monkeypatch):
        """The input can be closed afterwards, and the next picture encodes."""
        handle = encoder.new_input()
        put_frame(handle, gradient_frame(*SIZE))
        real = encoder.api.nvEncEncodePicture
        monkeypatch.setattr(encoder.api, 'nvEncEncodePicture', lambda *_arguments: 20)
        with pytest.raises(EncoderError, match='nvEncEncodePicture'):
            encoder.encode(handle, timestamp=0)
        monkeypatch.setattr(encoder.api, 'nvEncEncodePicture', real)
        assert len(encoder.encode(handle, timestamp=0)) == 1
        handle.close()


class TestClosingAnInput:
    """The driver reads a texture as it unregisters it, so it must still exist.

    A caller may close its inputs before the encoder or after it. Unregistering
    a texture already freed reads freed memory, which faults only when the
    allocator has reused it.
    """

    def test_a_closed_input_is_unregistered_while_its_texture_exists(
            self, encoder, monkeypatch):
        handle = encoder.new_input()
        texture_when_unregistered = []
        unregister = encoder.api.nvEncUnregisterResource

        def spy(session, registration):
            texture_when_unregistered.append(handle.texture)
            return unregister(session, registration)

        monkeypatch.setattr(encoder.api, 'nvEncUnregisterResource', spy)
        handle.close()
        assert len(texture_when_unregistered) == 1
        assert texture_when_unregistered[0] != 0
        assert handle.texture == 0

    def test_an_input_is_unregistered_once_whichever_closes_first(self, encoder, monkeypatch):
        handle = encoder.new_input()
        calls = []
        unregister = encoder.api.nvEncUnregisterResource
        monkeypatch.setattr(
            encoder.api, 'nvEncUnregisterResource',
            lambda session, registration: calls.append(registration) or unregister(
                session, registration))
        handle.close()
        encoder.close()
        handle.close()
        assert len(calls) == 1

    def test_closing_the_encoder_closes_the_inputs_it_still_holds(self, encoder):
        handle = encoder.new_input()
        encoder.close()
        assert handle.texture == 0
        assert handle.registration is None

    @pytest.mark.usefixtures('nvenc_available')
    def test_an_input_the_encoder_is_still_encoding_cannot_be_closed(self):
        encoder = NVENCEncoder(*SIZE, fps=30, bframes=2, gop=30)
        try:
            ring = [encoder.new_input() for _ in range(encoder.input_slots)]
            for handle in ring:
                put_frame(handle, gradient_frame(*SIZE))
            held = None
            for index, handle in enumerate(ring):
                if not encoder.encode(handle, timestamp=index * TICK):
                    held = handle
                    break
            assert held is not None, 'no picture was held back by the reordering'
            with pytest.raises(EncoderError, match='still being encoded'):
                held.close()
            encoder.flush()
            held.close()                        # the picture came out, so it can
        finally:
            encoder.close()


class TestMuxing:
    """The recording as a file, checked by something outside this package.

    A stream can be well formed, correctly timed, every box where it belongs,
    and still describe itself in a way that sets a decoder up wrongly. Nothing
    inside the encoder or the muxer notices that; decoding the result is what
    does.
    """

    def test_a_muxed_recording_decodes_without_complaint(self, encoder, handles,
                                                         tmp_path):
        """The sample description has to match what the samples hold.

        An encoder states its parameter sets before it codes anything and the
        driver may amend them; a container that kept the first answer describes
        the stream with syntax the pictures do not carry, and a decoder then
        reads a corrupt picture out of a perfectly good one.
        """
        path = tmp_path / 'conformance.mp4'
        with MP4Writer(path, encoder) as movie:
            for index in range(FRAMES):
                handle = handles[index % len(handles)]
                paint(handle, ((index % 10) / 10.0, 0.2, 0.8, 1.0))
                movie.write(encoder.encode(handle, timestamp=index * TICK,
                                           duration=TICK))
            movie.write(encoder.flush())
        complaints = decode_errors(path)
        if complaints is None:
            pytest.skip('no ffmpeg here to decode what was written')
        assert complaints == []

    def test_the_sample_description_holds_the_sets_the_stream_carries(
            self, encoder, handles, tmp_path):
        path = tmp_path / 'sets.mp4'
        with MP4Writer(path, encoder) as movie:
            for index in range(4):
                handle = handles[index % len(handles)]
                paint(handle, (0.3, 0.5, 0.7, 1.0))
                movie.write(encoder.encode(handle, timestamp=index * TICK,
                                           duration=TICK))
            movie.write(encoder.flush())
        stream = {unit[0] & 0x1F: unit
                  for unit in split_annexb(encoder.headers())}
        data = path.read_bytes()
        assert stream[7] in data, 'the sequence parameter set the stream carries'
        assert stream[8] in data, 'the picture parameter set the stream carries'

    @pytest.mark.usefixtures('nvenc_available')
    def test_a_reordered_recording_decodes_without_complaint(self, tmp_path):
        """Composition offsets are what a decoder reads timing back out of.

        With B-pictures the packets arrive in decode order carrying display
        timestamps, and the difference goes into the container. A file whose
        offsets are wrong still plays; the pictures come out in the wrong order.
        """
        encoder = NVENCEncoder(*SIZE, fps=30, bitrate=4_000_000, gop=15, bframes=2)
        made = []
        try:
            made = [encoder.new_input() for _ in range(encoder.input_slots)]
            path = tmp_path / 'reordered.mp4'
            with MP4Writer(path, encoder) as movie:
                for index in range(FRAMES):
                    handle = made[index % len(made)]
                    paint(handle, ((index % 10) / 10.0, 0.2, 0.8, 1.0))
                    movie.write(encoder.encode(handle, timestamp=index * TICK,
                                               duration=TICK))
                movie.write(encoder.flush())
        finally:
            for handle in made:
                handle.close()
            encoder.close()
        complaints = decode_errors(path)
        if complaints is None:
            pytest.skip('no ffmpeg here to decode what was written')
        assert complaints == []


class TestWhyTheDeviceWasRefused:
    """The explanation carried on ``NV_ENC_ERR_UNSUPPORTED_DEVICE``.

    Needs no encoder: what is under test is what the message says about the
    OpenGL context the session was opened against.
    """

    def test_no_context_at_all_says_so(self, monkeypatch):
        monkeypatch.setattr(nvenc_encoder, 'current_gl_renderer', lambda: None)
        detail = nvenc_encoder.unsupported_device_detail()
        assert 'context must be current' in detail

    def test_a_context_on_another_gpu_names_what_it_found(self, monkeypatch):
        """The same status arrives for a context on the wrong adapter.

        A software renderer, or the integrated part on a hybrid machine, is a
        current context that NVENC will not encode from -- and a reader sent
        looking for a missing context looks in the wrong place.
        """
        monkeypatch.setattr(nvenc_encoder, 'current_gl_renderer',
                            lambda: 'llvmpipe (LLVM 20.1.2, 256 bits)')
        detail = nvenc_encoder.unsupported_device_detail()
        assert 'llvmpipe' in detail, 'names the renderer it found'
        assert 'NVIDIA' in detail, 'says what it needed instead'
        assert 'context must be current' not in detail, (
            'a context was current, so saying it was not sends the reader wrong')

    def test_a_context_on_an_nvidia_gpu_does_not_guess(self, monkeypatch):
        """Neither cause fits, so the message does not claim either."""
        monkeypatch.setattr(nvenc_encoder, 'current_gl_renderer',
                            lambda: 'NVIDIA GeForce RTX 3060 Ti/PCIe/SSE2')
        detail = nvenc_encoder.unsupported_device_detail()
        assert 'RTX 3060 Ti' in detail
        assert 'context must be current' not in detail
