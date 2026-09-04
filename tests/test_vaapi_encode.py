"""The VA-API encoder against real hardware, from a real OpenGL texture.

The whole path is exercised: a texture is drawn into, exported as a DMA-BUF,
imported as a VA surface, converted to NV12 by the driver and coded. The
conversion is checked by reading the surface the encoder is about to code, which
is what says the picture reached it the right way up and the right colour --
a stream that went nowhere is still a well-formed stream of something else.
"""
import ctypes
from itertools import pairwise

import numpy as np
import pytest

from pyopengl_video import EncoderError, encoders, open_encoder
from pyopengl_video.mp4 import MP4Writer, split_annexb
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.encoder import PIPELINE_DEPTH, VAAPIEncoder, colour_pipeline
from tests.conftest import gradient_frame

SIZE = (320, 240)
FRAMES = 20
GOP = 10
TICK = 3000                                  # 30 fps in a 90 kHz timescale


@pytest.fixture
def encoder(vaapi_available):
    """A 320x240 encoder, key frames every ten pictures."""
    made = VAAPIEncoder(*SIZE, fps=30, bitrate=2_000_000, gop=GOP)
    yield made
    made.close()


@pytest.fixture
def handles(encoder):
    """The encoder's own inputs, as many as it says a recording needs."""
    made = [encoder.new_input() for _ in range(encoder.input_slots)]
    yield made
    for handle in made:
        handle.close()


def paint(handle, colour):
    """Fill a handle's texture with one colour, inside its drawing scope."""
    from OpenGL.GL import (
        GL_COLOR_BUFFER_BIT,
        GL_DRAW_FRAMEBUFFER,
        glBindFramebuffer,
        glClear,
        glClearColor,
    )
    with handle.for_drawing():
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, handle.framebuffer)
        glClearColor(*colour)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)


def blit(handle, texture):
    """Copy a texture into a handle's own, inside its drawing scope."""
    from OpenGL.GL import (
        GL_COLOR_ATTACHMENT0,
        GL_COLOR_BUFFER_BIT,
        GL_DRAW_FRAMEBUFFER,
        GL_NEAREST,
        GL_READ_FRAMEBUFFER,
        GL_TEXTURE_2D,
        glBindFramebuffer,
        glBlitFramebuffer,
        glDeleteFramebuffers,
        glFramebufferTexture2D,
        glGenFramebuffers,
    )
    width, height = SIZE
    source = int(glGenFramebuffers(1))
    with handle.for_drawing():
        glBindFramebuffer(GL_READ_FRAMEBUFFER, source)
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, int(texture), 0)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, handle.framebuffer)
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
    glDeleteFramebuffers(1, [source])


def encode_sequence(encoder, handles, frames=FRAMES, paint_frame=None):
    """Push `frames` pictures through, returning every packet produced."""
    packets = []
    for index in range(frames):
        handle = handles[index % len(handles)]
        if paint_frame is None:
            paint(handle, ((index % 10) / 10.0, 0.2, 0.8, 1.0))
        else:
            paint_frame(handle, index)
        packets.extend(encoder.encode(handle, timestamp=index * TICK,
                                      duration=TICK))
    packets.extend(encoder.flush())
    return packets


def nal_kinds(packets):
    """The NAL unit types in a run of packets, in order."""
    return [unit[0] & 0x1F
            for packet in packets for unit in split_annexb(packet.data)]


# ------------------------------------------------------------- what it opened


class TestWhatTheSessionNegotiated:
    def test_it_reports_the_shape_of_the_stream(self, encoder):
        assert encoder.size == SIZE
        assert encoder.codec == 'h264'
        assert encoder.zero_copy is True
        assert encoder.reorders_frames is False
        assert encoder.timescale == 90000

    def test_it_names_the_device_it_opened(self, encoder):
        assert encoder.device.startswith('/dev/dri/renderD')
        assert encoder.vendor, 'the driver names itself'

    def test_it_asks_for_no_more_references_than_the_driver_offers(self, encoder):
        assert encoder.max_num_ref_frames >= 1
        assert encoder.sets.max_num_ref_frames == encoder.max_num_ref_frames

    def test_it_asks_only_for_the_packed_headers_it_writes(self, encoder):
        from pyopengl_video.vaapi.encoder import PACKED_HEADERS_WRITTEN

        assert encoder.packed_headers & ~PACKED_HEADERS_WRITTEN == 0

    def test_a_recording_needs_more_than_one_input(self, encoder):
        assert encoder.input_slots == PIPELINE_DEPTH + 1

    def test_open_encoder_finds_the_backend(self, vaapi_available):
        assert 'vaapi' in [backend.name for backend in encoders()]
        with open_encoder(*SIZE, fps=30, backend='vaapi') as found:
            assert isinstance(found, VAAPIEncoder)


# ------------------------------------------------------------------ the stream


class TestTheStream:
    def test_every_frame_comes_back_as_a_packet(self, encoder, handles):
        packets = encode_sequence(encoder, handles)
        assert len(packets) == FRAMES
        assert all(len(packet) > 0 for packet in packets)

    def test_it_opens_on_a_key_frame_and_repeats_them_at_the_gop(
            self, encoder, handles):
        packets = encode_sequence(encoder, handles)
        assert [index for index, packet in enumerate(packets)
                if packet.keyframe] == [0, 10]

    def test_a_key_frame_carries_the_parameter_sets(self, encoder, handles):
        packets = encode_sequence(encoder, handles, frames=3)
        assert nal_kinds(packets[:1]) == [7, 8, 5], 'SPS, PPS, then an IDR slice'

    def test_a_predicted_picture_is_a_slice_and_nothing_else(
            self, encoder, handles):
        packets = encode_sequence(encoder, handles, frames=3)
        assert nal_kinds(packets[1:]) == [1, 1]

    def test_timestamps_come_back_as_they_went_in(self, encoder, handles):
        packets = encode_sequence(encoder, handles)
        assert [packet.timestamp for packet in packets] == [
            index * TICK for index in range(FRAMES)]
        assert all(packet.duration == TICK for packet in packets)

    def test_a_forced_key_frame_starts_a_new_group(self, encoder, handles):
        packets = []
        paint(handles[0], (0.5, 0.5, 0.5, 1.0))
        packets.extend(encoder.encode(handles[0], timestamp=0))
        paint(handles[1], (0.5, 0.5, 0.5, 1.0))
        packets.extend(encoder.encode(handles[1], timestamp=TICK, force_idr=True))
        packets.extend(encoder.flush())
        assert [packet.keyframe for packet in packets] == [True, True]
        assert nal_kinds(packets[1:]) == [7, 8, 5], (
            'a forced key frame carries the parameter sets again')

    def test_the_headers_are_the_sets_the_stream_carries(self, encoder, handles):
        encode_sequence(encoder, handles, frames=3)
        units = list(split_annexb(encoder.headers()))
        assert [unit[0] & 0x1F for unit in units] == [7, 8]

    def test_the_headers_are_available_before_anything_is_coded(self, encoder):
        units = list(split_annexb(encoder.headers()))
        assert [unit[0] & 0x1F for unit in units] == [7, 8]

    def test_every_packet_holds_the_picture_it_says_it_does(self, encoder,
                                                            handles):
        """Ties a packet's bytes to its metadata, over a whole recording.

        Each picture is coded into one of a rotating set of output buffers, and
        a rotation that hands the same buffer to two pictures at once produces
        packets that are well formed, correctly timed, and the wrong picture.
        Reading the slice type back out of every packet is what notices.
        """
        packets = encode_sequence(encoder, handles)
        for index, packet in enumerate(packets):
            kinds = [unit[0] & 0x1F for unit in split_annexb(packet.data)]
            expected = [7, 8, 5] if packet.keyframe else [1]
            assert kinds == expected, f'packet {index} holds {kinds}'

    def test_the_output_buffers_are_used_in_turn(self, encoder, handles):
        """A picture must never be coded into a buffer still holding another.

        The rotation is what keeps that from happening. One that advances with
        the number of pictures in flight rather than with the pictures settles
        on a single buffer and hands it to two at once.
        """
        used = []
        for index in range(FRAMES):
            handle = handles[index % len(handles)]
            paint(handle, ((index % 10) / 10.0, 0.2, 0.8, 1.0))
            encoder.encode(handle, timestamp=index * TICK)
            used.append(encoder._pending[-1].coded_buffer)
        encoder.flush()
        assert set(used) == set(encoder._coded_buffers), 'all of them are used'
        assert all(earlier != later for earlier, later in pairwise(used)), (
            'consecutive pictures went to the same buffer')

    def test_no_start_code_survives_inside_a_unit(self, encoder, handles):
        for packet in encode_sequence(encoder, handles, frames=5):
            for unit in split_annexb(packet.data):
                assert b'\x00\x00\x01' not in unit


class TestPipelining:
    """One picture is held back, so the GPU is never waited on immediately."""

    def test_the_first_picture_produces_nothing_yet(self, encoder, handles):
        paint(handles[0], (0.2, 0.4, 0.6, 1.0))
        assert encoder.encode(handles[0], timestamp=0) == []

    def test_the_next_picture_produces_the_first(self, encoder, handles):
        paint(handles[0], (0.2, 0.4, 0.6, 1.0))
        encoder.encode(handles[0], timestamp=0)
        paint(handles[1], (0.2, 0.4, 0.6, 1.0))
        packets = encoder.encode(handles[1], timestamp=TICK)
        assert [packet.timestamp for packet in packets] == [0]

    def test_flush_gives_back_what_is_still_held(self, encoder, handles):
        paint(handles[0], (0.2, 0.4, 0.6, 1.0))
        encoder.encode(handles[0], timestamp=0)
        assert [packet.timestamp for packet in encoder.flush()] == [0]

    def test_flushing_twice_gives_nothing_the_second_time(self, encoder, handles):
        paint(handles[0], (0.2, 0.4, 0.6, 1.0))
        encoder.encode(handles[0], timestamp=0)
        encoder.flush()
        assert encoder.flush() == []


class TestContentSensitivity:
    """A registration that went nowhere still produces a well-formed stream."""

    def test_noise_costs_far_more_than_a_flat_colour(self, vaapi_available,
                                                     upload_texture):
        def sizes(frames):
            encoder = VAAPIEncoder(*SIZE, fps=30, rate_control='cqp', qp=26,
                                   gop=100)
            handles = [encoder.new_input()
                       for _ in range(encoder.input_slots)]
            try:
                packets = []
                for index, frame in enumerate(frames):
                    handle = handles[index % len(handles)]
                    blit(handle, upload_texture(frame))
                    packets.extend(encoder.encode(handle, index * TICK))
                packets.extend(encoder.flush())
                return sum(len(packet) for packet in packets)
            finally:
                for handle in handles:
                    handle.close()
                encoder.close()

        width, height = SIZE
        flat = [np.full((height, width, 4), 128, np.uint8) for _ in range(6)]
        for frame in flat:
            frame[..., 3] = 255
        generator = np.random.default_rng(20260904)
        noise = [generator.integers(0, 256, (height, width, 4), dtype=np.uint8)
                 for _ in range(6)]
        for frame in noise:
            frame[..., 3] = 255
        assert sizes(noise) > 20 * sizes(flat)

    def test_a_moving_picture_costs_more_than_a_held_one(self, vaapi_available,
                                                         upload_texture):
        width, height = SIZE

        def sizes(moving):
            encoder = VAAPIEncoder(*SIZE, fps=30, rate_control='cqp', qp=26,
                                   gop=100)
            handles = [encoder.new_input()
                       for _ in range(encoder.input_slots)]
            try:
                packets = []
                for index in range(8):
                    handle = handles[index % len(handles)]
                    phase = index if moving else 0
                    blit(handle, upload_texture(
                        gradient_frame(width, height, phase=phase)))
                    packets.extend(encoder.encode(handle, index * TICK))
                packets.extend(encoder.flush())
                # The first picture is intra either way, so compare the rest.
                return sum(len(packet) for packet in packets[1:])
            finally:
                for handle in handles:
                    handle.close()
                encoder.close()

        assert sizes(moving=True) > 2 * sizes(moving=False)


# ------------------------------------------------------- colour and geometry


def read_nv12(encoder, surface):
    """The luma plane of a VA surface, as an array of the picture's size.

    ``vaDeriveImage`` addresses the surface's own memory, so this is what the
    encoder is about to code rather than a decode of what it produced.
    """
    width, height = encoder.size
    image = api.VAImage()
    encoder.api.check(
        encoder.api.vaDeriveImage(encoder.display, surface,
                                  ctypes.byref(image)), 'vaDeriveImage')
    try:
        pointer = ctypes.c_void_p()
        encoder.api.check(
            encoder.api.vaMapBuffer(encoder.display, image.buf,
                                    ctypes.byref(pointer)), 'vaMapBuffer')
        try:
            raw = ctypes.string_at(pointer.value, image.data_size)
            luma = np.frombuffer(raw, np.uint8,
                                 count=image.pitches[0] * height,
                                 offset=image.offsets[0])
            return luma.reshape(height, image.pitches[0])[:, :width].copy()
        finally:
            encoder.api.vaUnmapBuffer(encoder.display, image.buf)
    finally:
        encoder.api.vaDestroyImage(encoder.display, image.image_id)


def coded_luma(encoder, handle):
    """Encode one picture and read back the luma the encoder was handed."""
    surface = encoder._sources[encoder._slot]
    encoder.encode(handle, timestamp=0)
    encoder.flush()
    return read_nv12(encoder, surface)


class TestColour:
    """The conversion and the stream's colour description have to agree.

    They are set in different places -- one asks the driver, the other is
    written into the sequence parameter set -- and nothing but a washed-out
    picture reports it when they part.
    """

    @pytest.mark.parametrize('colour,luma', [
        ((0.0, 0.0, 0.0, 1.0), 16),        # limited range floor
        ((1.0, 1.0, 1.0, 1.0), 235),       # limited range ceiling
        ((1.0, 0.0, 0.0, 1.0), 63),        # BT.709 luma of full red
        ((0.0, 1.0, 0.0, 1.0), 173),
        ((0.0, 0.0, 1.0, 1.0), 32),
    ])
    def test_the_encoder_is_handed_limited_range_bt709(self, encoder, handles,
                                                       colour, luma):
        paint(handles[0], colour)
        found = coded_luma(encoder, handles[0])
        assert abs(int(found.mean()) - luma) <= 2

    def test_the_pipeline_asks_for_what_the_stream_declares(self):
        from pyopengl_video.vaapi import h264

        pipeline = colour_pipeline(surface=7)
        assert pipeline.surface == 7
        assert pipeline.output_color_standard == api.VAProcColorStandardBT709
        assert pipeline.input_color_properties.color_range == api.VA_SOURCE_RANGE_FULL
        assert pipeline.output_color_properties.color_range == (
            api.VA_SOURCE_RANGE_REDUCED)
        assert pipeline.output_color_properties.colour_primaries == (
            h264.COLOUR_PRIMARIES_BT709)
        assert pipeline.output_color_properties.matrix_coefficients == (
            h264.MATRIX_BT709)


class TestOrientation:
    def test_the_first_row_of_the_texture_is_the_top_of_the_picture(
            self, encoder, handles, upload_texture):
        width, height = SIZE
        frame = np.zeros((height, width, 4), np.uint8)
        frame[..., 3] = 255
        frame[:height // 2] = (255, 255, 255, 255)     # white across the top
        blit(handles[0], upload_texture(frame))
        luma = coded_luma(encoder, handles[0])
        assert luma[height // 4].mean() > 200, 'the top of the picture is white'
        assert luma[3 * height // 4].mean() < 40, 'the bottom is black'

    def test_the_left_of_the_texture_is_the_left_of_the_picture(
            self, encoder, handles, upload_texture):
        width, height = SIZE
        frame = np.zeros((height, width, 4), np.uint8)
        frame[..., 3] = 255
        frame[:, :width // 2] = (255, 255, 255, 255)
        blit(handles[0], upload_texture(frame))
        luma = coded_luma(encoder, handles[0])
        assert luma[:, width // 4].mean() > 200
        assert luma[:, 3 * width // 4].mean() < 40


# --------------------------------------------------------------- registration


class TestRegistering:
    def test_a_caller_may_bring_its_own_texture(self, encoder, upload_texture):
        width, height = SIZE
        handle = encoder.register(upload_texture(gradient_frame(width, height)))
        assert handle.surface, 'the texture was imported as a VA surface'
        assert handle.exported is not None
        assert not handle.owns_texture

    def test_an_encoder_made_input_owns_its_texture(self, encoder):
        handle = encoder.new_input()
        try:
            assert handle.owns_texture
            assert handle.texture and handle.framebuffer and handle.surface
        finally:
            handle.close()

    def test_a_texture_of_the_wrong_size_is_refused(self, encoder, upload_texture):
        with pytest.raises((EncoderError, Exception)):
            encoder.register(upload_texture(gradient_frame(64, 64)))

    def test_a_handle_from_another_encoder_is_refused(self, encoder, handles,
                                                      vaapi_available):
        other = VAAPIEncoder(*SIZE, fps=30)
        try:
            stranger = other.new_input()
            with pytest.raises(EncoderError, match='not registered'):
                encoder.encode(stranger, timestamp=0)
            stranger.close()
        finally:
            other.close()

    def test_drawing_leaves_a_fence_for_the_encoder_to_wait_on(self, encoder,
                                                               handles):
        assert handles[0].fence is None
        paint(handles[0], (0.1, 0.2, 0.3, 1.0))
        assert handles[0].fence is not None


# -------------------------------------------------------------- settings


class TestSettings:
    def test_unknown_settings_are_refused_by_name(self, vaapi_available):
        with pytest.raises(EncoderError, match='preset'):
            VAAPIEncoder(*SIZE, preset='fastest')

    def test_an_unknown_rate_control_mode_is_refused(self, vaapi_available):
        with pytest.raises(EncoderError, match='rate control'):
            VAAPIEncoder(*SIZE, rate_control='magic')

    def test_another_codec_is_refused(self, vaapi_available):
        with pytest.raises(EncoderError, match='h264'):
            VAAPIEncoder(*SIZE, codec='hevc')

    def test_b_frames_are_refused_because_nothing_reorders(self, vaapi_available):
        with pytest.raises(EncoderError, match='display order'):
            VAAPIEncoder(*SIZE, bframes=2)

    @pytest.mark.parametrize('mode', ['cqp', 'cbr', 'vbr'])
    def test_each_rate_control_mode_the_driver_offers_records(
            self, vaapi_available, mode):
        made = VAAPIEncoder(*SIZE, fps=30, bitrate=2_000_000, gop=5,
                            rate_control=mode)
        handles = [made.new_input() for _ in range(made.input_slots)]
        try:
            packets = encode_sequence(made, handles, frames=6)
            assert len(packets) == 6
            assert all(len(packet) > 0 for packet in packets)
        finally:
            for handle in handles:
                handle.close()
            made.close()

    def test_a_frame_rate_that_is_not_whole_keeps_its_exact_ratio(
            self, vaapi_available):
        made = VAAPIEncoder(*SIZE, fps=29.97)
        try:
            assert made.frame_rate == (30000, 1001)
        finally:
            made.close()

    def test_the_default_bitrate_follows_the_frame_size_and_rate(
            self, vaapi_available):
        made = VAAPIEncoder(*SIZE, fps=30)
        try:
            assert made.bitrate == int(320 * 240 * 30 * 0.07)
        finally:
            made.close()

    def test_a_named_device_is_the_one_opened(self, vaapi_available):
        nodes = api.render_nodes()
        made = VAAPIEncoder(*SIZE, fps=30, device=nodes[0])
        try:
            assert made.device == nodes[0]
        finally:
            made.close()

    def test_a_device_that_is_not_there_is_refused_by_name(self, vaapi_available):
        with pytest.raises(EncoderError, match='renderD99'):
            VAAPIEncoder(*SIZE, device='/dev/dri/renderD99')


class TestClosing:
    def test_a_closed_encoder_refuses_to_encode(self, encoder, handles):
        handle = handles[0]
        encoder.close()
        with pytest.raises(EncoderError, match='closed'):
            encoder.encode(handle, timestamp=0)

    def test_closing_twice_is_harmless(self, encoder):
        encoder.close()
        encoder.close()

    def test_it_works_as_a_context_manager(self, vaapi_available):
        with VAAPIEncoder(*SIZE, fps=30) as made:
            assert made.size == SIZE
        with pytest.raises(EncoderError):
            made.flush()


def decode_errors(path):
    """What an outside decoder complains about, or None where there is none.

    ``ffmpeg`` is not a dependency of this package and is not needed to run the
    suite. Where it happens to be installed it is worth asking, because a
    stream can be well formed, correctly timed, and still describe itself in a
    way that sets a decoder up wrongly -- which nothing inside this package can
    notice.
    """
    import shutil
    import subprocess

    if shutil.which('ffmpeg') is None:
        return None
    found = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
        capture_output=True, text=True)
    return [line for line in found.stderr.splitlines() if line.strip()]


class TestMuxing:
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

    def test_a_recording_becomes_a_playable_mp4(self, encoder, handles, tmp_path):
        path = tmp_path / 'recording.mp4'
        with MP4Writer(path, encoder) as movie:
            for index in range(FRAMES):
                handle = handles[index % len(handles)]
                paint(handle, ((index % 10) / 10.0, 0.2, 0.8, 1.0))
                movie.write(encoder.encode(handle, timestamp=index * TICK,
                                           duration=TICK))
            movie.write(encoder.flush())
        data = path.read_bytes()
        assert data[4:8] == b'ftyp'
        assert b'avcC' in data
        assert b'moov' in data
        assert len(data) > 1000


class TestTheDrawingFence:
    """What the encoder waits on before the conversion reads a texture.

    A fence covers the drawing inside the scope that made it and nothing else.
    One left lying around after it has been waited on is worse than none at
    all: it is already signalled, so waiting on it a second time returns at
    once and the conversion reads a texture the renderer is still writing.
    """

    def test_drawing_leaves_a_fence(self, encoder, handles):
        assert handles[0].fence is None
        paint(handles[0], (0.1, 0.2, 0.3, 1.0))
        assert handles[0].fence is not None

    def test_the_encode_that_waits_on_it_uses_it_up(self, encoder, handles):
        paint(handles[0], (0.1, 0.2, 0.3, 1.0))
        encoder.encode(handles[0], timestamp=0)
        assert handles[0].fence is None, (
            'a fence that has been waited on must not cover a later frame')

    def test_a_second_encode_without_redrawing_falls_back_to_flushing(
            self, encoder, handles):
        """Drawing outside the scope is allowed; it costs the whole pipeline."""
        paint(handles[0], (0.1, 0.2, 0.3, 1.0))
        encoder.encode(handles[0], timestamp=0)
        packets = encoder.encode(handles[0], timestamp=TICK)
        assert len(packets) == 1, 'the frame still encodes'

    def test_re_entering_the_scope_replaces_the_fence(self, encoder, handles):
        paint(handles[0], (0.1, 0.2, 0.3, 1.0))
        first = handles[0].fence
        paint(handles[0], (0.4, 0.5, 0.6, 1.0))
        assert handles[0].fence is not first
