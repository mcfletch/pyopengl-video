"""AV1 encoding on real hardware, from a real OpenGL texture and back again.

A texture is drawn into, coded by the driver's AV1 encoder, and the packets
are read three ways: by this package's own parser, which says whether the
stream is the one the control layer decided on; by this machine's AV1 decoder,
which says whether the picture that went in is the one that comes out; and by
``ffmpeg`` where there is one, which is a decoder nobody here wrote.
"""
import ctypes

import numpy as np
import pytest
from OpenGL.GL import (
    GL_RGBA,
    GL_TEXTURE_2D,
    GL_UNSIGNED_BYTE,
    glBindTexture,
    glDeleteTextures,
    glGetTexImage,
)

from pyopengl_video import EncoderError, codecs, obu, open_decoder, open_encoder
from pyopengl_video.convert import NV12Converter, create_output_texture
from pyopengl_video.decoder import Sample
from pyopengl_video.ivf import IVFReader, IVFWriter, av1_keyframe
from pyopengl_video.mp4 import MP4Reader, MP4Writer
from pyopengl_video.vaapi import api, av1
from pyopengl_video.vaapi.av1_encode import PACKED_HEADERS_WRITTEN
from pyopengl_video.vaapi.av1_parse import (
    ALL_FRAMES,
    INTER_FRAME,
    KEY_FRAME,
    FrameState,
    parse_frame_header,
    parse_sequence_header,
)
from pyopengl_video.vaapi.device import Device
from pyopengl_video.vaapi.encoder import RATE_CONTROL_MODES, VAAPIEncoder
from tests.conftest import blit, decode_errors, gradient_frame

#: A size the encoders this was written against code as it stands. Both need
#: at least 128 samples each way, and Mesa's pads a width that is not a
#: multiple of 64 or a height that is not a multiple of 16.
SIZE = (320, 240)
#: A size no encoder's blocks divide.
UNEVEN = (200, 130)
FRAMES = 20
GOP = 10
TICK = 3000                                  # 30 fps in a 90 kHz timescale

_FRAME_UNITS = (obu.FRAME, obu.FRAME_HEADER)


@pytest.fixture
def encoder(vaapi_encodes):
    """A 320x240 AV1 encoder, key frames every ten pictures."""
    vaapi_encodes('av1')
    made = VAAPIEncoder(*SIZE, 'av1', fps=30, bitrate=2_000_000, gop=GOP)
    yield made
    made.close()


def record(frames, upload_texture, size=SIZE, forced=(), encoder=None, **settings):
    """Code `frames`, RGBA arrays first row at the top, and return the packets.

    forced -- the indices to ask for a key frame at
    encoder -- an encoder to use and leave open; one is made from `settings`
        and closed otherwise
    """
    made = encoder or VAAPIEncoder(*size, 'av1', **{'fps': 30, 'gop': GOP, **settings})
    handles = [made.new_input() for _ in range(made.input_slots)]
    source = upload_texture(frames[0])
    packets = []
    try:
        for index, frame in enumerate(frames):
            handle = handles[index % len(handles)]
            upload_texture.update(source, frame)
            blit(handle, source, size)
            packets += made.encode(handle, timestamp=index * TICK, duration=TICK,
                                   force_idr=index in forced)
        packets += made.flush()
    finally:
        for handle in handles:
            handle.close()
        if encoder is None:
            made.close()
    return packets


def gradients(count, size=SIZE):
    return [gradient_frame(*size, phase=index) for index in range(count)]


def solid(colour, size=SIZE):
    width, height = size
    frame = np.empty((height, width, 4), np.uint8)
    frame[...] = (*colour, 255)
    return frame


def samples_of(packets):
    return [Sample(packet.data, packet.timestamp, packet.duration, packet.keyframe, index)
            for index, packet in enumerate(packets)]


def decoded(packets):
    """Each packet's picture as RGB, first row at the top, by timestamp.

    The pictures are the size the stream codes, which is the encoder's size
    or that size padded out to the driver's blocks.
    """
    pictures = {}
    converter = NV12Converter()
    targets = {}

    def take(frame):
        width, height = frame.size
        if frame.size not in targets:
            targets[frame.size] = create_output_texture(width, height)
        target = targets[frame.size]
        converter.convert(frame, target)
        frame.release()
        glBindTexture(GL_TEXTURE_2D, target)
        data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
        glBindTexture(GL_TEXTURE_2D, 0)
        picture = np.frombuffer(data, np.uint8).reshape(height, width, 4)
        # The converter's output is the right way up for OpenGL: row 0 is the bottom.
        pictures[frame.timestamp] = picture[::-1, :, :3].astype(int)

    try:
        with open_decoder('av1', backend='vaapi') as decoder:
            for sample in samples_of(packets):
                for frame in decoder.decode(sample):
                    take(frame)
            for frame in decoder.flush():
                take(frame)
    finally:
        converter.close()
        if targets:
            glDeleteTextures(list(targets.values()))
    return pictures


def unit_kinds(packet):
    return [unit.kind for unit in obu.split(packet.data)]


def frame_headers(packets):
    """The sequence header and each packet's frame header, parsed in order."""
    sequence = None
    state = FrameState()
    found = []
    for packet in packets:
        for unit in obu.split(packet.data):
            if unit.kind == obu.SEQUENCE_HEADER:
                sequence = parse_sequence_header(unit.payload)
            elif unit.kind in _FRAME_UNITS:
                header = parse_frame_header(unit.payload, sequence, state)
                state.update(header, sequence)
                found.append(header)
    return sequence, found


# ------------------------------------------------------------- what it opened


class TestWhatTheSessionNegotiated:
    def test_it_reports_the_shape_of_the_stream(self, encoder):
        assert encoder.size == SIZE
        assert encoder.codec == 'av1'
        assert encoder.zero_copy is True
        assert encoder.reorders_frames is False

    def test_it_asks_only_for_the_packed_headers_it_writes(self, encoder):
        assert encoder.packed_headers & ~PACKED_HEADERS_WRITTEN == 0

    def test_open_encoder_finds_the_backend(self, vaapi_encodes):
        vaapi_encodes('av1')
        with open_encoder(*SIZE, codec='av1', fps=30, backend='vaapi') as found:
            assert isinstance(found, VAAPIEncoder)
            assert found.codec == 'av1'

    def test_the_headers_are_a_sequence_header_before_anything_is_coded(self, encoder):
        [unit] = obu.split(encoder.headers())
        assert unit.kind == obu.SEQUENCE_HEADER
        assert parse_sequence_header(unit.payload).size == SIZE


# ------------------------------------------------------------------ the stream


class TestTheStream:
    @pytest.fixture
    def packets(self, encoder, upload_texture):
        return record(gradients(FRAMES), upload_texture, encoder=encoder)

    def test_every_frame_comes_back_as_a_packet(self, packets):
        assert len(packets) == FRAMES
        assert all(len(packet) > 0 for packet in packets)

    def test_it_opens_on_a_key_frame_and_repeats_them_at_the_gop(self, packets):
        assert [index for index, packet in enumerate(packets)
                if packet.keyframe] == [0, 10]

    def test_timestamps_come_back_as_they_went_in(self, packets):
        assert [packet.timestamp for packet in packets] == [
            index * TICK for index in range(FRAMES)]
        assert all(packet.duration == TICK for packet in packets)

    def test_every_packet_is_one_temporal_unit(self, packets):
        for index, packet in enumerate(packets):
            kinds = unit_kinds(packet)
            assert kinds[0] == obu.TEMPORAL_DELIMITER, f'packet {index} holds {kinds}'
            assert kinds.count(obu.TEMPORAL_DELIMITER) == 1
            assert sum(kind in _FRAME_UNITS for kind in kinds) == 1

    def test_a_key_frame_carries_the_sequence_header_and_no_other_frame_does(self, packets):
        for packet in packets:
            kinds = unit_kinds(packet)
            if packet.keyframe:
                assert kinds[:2] == [obu.TEMPORAL_DELIMITER, obu.SEQUENCE_HEADER]
            assert kinds.count(obu.SEQUENCE_HEADER) == int(packet.keyframe)

    def test_every_obu_has_a_size_field(self, packets):
        for packet in packets:
            for unit in obu.split(packet.data):
                assert unit.raw[0] & 0x02
                assert not unit.has_extension

    def test_every_packet_holds_the_frame_it_says_it_does(self, packets):
        """Ties a packet's bytes to its metadata, over a whole recording.

        A rotation that hands one output buffer to two pictures at once
        produces packets that are well formed, correctly timed, and the wrong
        picture. The frame type and the order hint in each packet say which
        picture it is.
        """
        _, headers = frame_headers(packets)
        assert len(headers) == FRAMES
        for index, (packet, header) in enumerate(zip(packets, headers, strict=True)):
            assert av1_keyframe(packet.data) == packet.keyframe
            assert header.frame_type == (KEY_FRAME if packet.keyframe else INTER_FRAME)
            assert header.order_hint == index % GOP, f'packet {index}'

    def test_the_frames_are_the_structure_the_layer_decided(self, packets):
        _, headers = frame_headers(packets)
        for header in headers:
            assert header.show_frame
            assert not header.show_existing_frame
            assert header.refresh_frame_flags == ALL_FRAMES
            assert (header.render_width, header.render_height) == SIZE
            if header.frame_type == INTER_FRAME:
                assert header.primary_ref_frame == 0
                assert not header.error_resilient_mode
                assert not header.reference_select

    def test_the_sequence_is_eight_bit_420_limited_range_bt709(self, packets):
        sequence, _ = frame_headers(packets)
        assert sequence.seq_profile == 0
        color = sequence.color_config
        assert color.bit_depth == 8
        assert (color.subsampling_x, color.subsampling_y) == (1, 1)
        assert (color.color_primaries, color.transfer_characteristics,
                color.matrix_coefficients) == (1, 1, 1)
        assert color.color_range == 0
        assert sequence.timing_info.time_scale == 30
        assert sequence.timing_info.num_units_in_display_tick == 1

    def test_the_headers_are_the_sequence_header_the_stream_carries(self, encoder, packets):
        assert encoder.headers() == obu.sequence_header(packets[0].data)
        assert encoder.headers() == obu.sequence_header(packets[GOP].data)

    def test_a_forced_key_frame_starts_a_new_group(self, encoder, upload_texture):
        packets = record(gradients(8), upload_texture, encoder=encoder, forced={3})
        assert [packet.keyframe for packet in packets] == [
            True, False, False, True, False, False, False, False]
        assert obu.SEQUENCE_HEADER in unit_kinds(packets[3])
        _, headers = frame_headers(packets)
        assert [header.order_hint for header in headers] == [0, 1, 2, 0, 1, 2, 3, 4]


# ---------------------------------------------------------------- settings


def offered_rate_controls():
    """The rate control modes the first render node's AV1 encoder states."""
    device = Device(api.render_nodes()[0])
    try:
        attribute = api.VAConfigAttrib(type=api.VAConfigAttribRateControl)
        device.check(
            device.va.vaGetConfigAttributes(
                device.display, api.VAProfileAV1Profile0, api.VAEntrypointEncSlice,
                ctypes.byref(attribute), 1),
            'vaGetConfigAttributes')
    finally:
        device.close()
    return {name for name, bit in RATE_CONTROL_MODES.items() if attribute.value & bit}


class TestSettings:
    @pytest.mark.parametrize('mode', sorted(RATE_CONTROL_MODES))
    def test_each_rate_control_mode_the_driver_offers_records(
            self, vaapi_encodes, upload_texture, mode):
        vaapi_encodes('av1')
        if mode not in offered_rate_controls():
            pytest.skip(f'this driver has no {mode} rate control for AV1')
        packets = record(gradients(12), upload_texture, bitrate=2_000_000, gop=5,
                         rate_control=mode)
        assert len(packets) == 12
        assert [packet.keyframe for packet in packets] == [
            index % 5 == 0 for index in range(12)]
        assert len(decoded(packets)) == 12

    def test_a_constant_quantiser_is_the_one_asked_for(self, vaapi_encodes, upload_texture):
        vaapi_encodes('av1')
        packets = record(gradients(4), upload_texture, rate_control='cqp', qp=30)
        _, headers = frame_headers(packets)
        assert [header.quantization.base_q_idx for header in headers] == [
            av1.base_q_idx(30)] * 4

    def test_a_coarser_quantiser_costs_fewer_bytes(self, vaapi_encodes, upload_texture):
        vaapi_encodes('av1')
        generator = np.random.default_rng(20261012)
        width, height = SIZE
        frames = [generator.integers(0, 256, (height, width, 4), dtype=np.uint8)
                  for _ in range(4)]

        def size(qp):
            return sum(len(packet) for packet in record(
                frames, upload_texture, rate_control='cqp', qp=qp))

        assert size(40) < size(20)

    def test_a_picture_smaller_than_the_encoder_takes_is_an_encoder_error(
            self, vaapi_encodes):
        vaapi_encodes('av1')
        with pytest.raises(EncoderError):
            VAAPIEncoder(16, 16, 'av1')

    def test_b_frames_are_refused_because_nothing_reorders(self, vaapi_encodes):
        vaapi_encodes('av1')
        with pytest.raises(EncoderError, match='bframes'):
            VAAPIEncoder(*SIZE, 'av1', bframes=2)


class TestContentSensitivity:
    """A registration that went nowhere still produces a well-formed stream."""

    def test_noise_costs_far_more_than_a_flat_colour(self, vaapi_encodes, upload_texture):
        vaapi_encodes('av1')
        width, height = SIZE
        generator = np.random.default_rng(20260904)
        noise = [generator.integers(0, 256, (height, width, 4), dtype=np.uint8)
                 for _ in range(6)]
        flat = [solid((128, 128, 128)) for _ in range(6)]

        def size(frames):
            return sum(len(packet) for packet in record(
                frames, upload_texture, rate_control='cqp', qp=26, gop=100))

        assert size(noise) > 20 * size(flat)

    def test_a_moving_picture_costs_more_than_a_held_one(self, vaapi_encodes, upload_texture):
        vaapi_encodes('av1')

        def size(frames):
            packets = record(frames, upload_texture, rate_control='cqp', qp=26, gop=100)
            # The first picture is intra either way, so compare the rest.
            return sum(len(packet) for packet in packets[1:])

        assert size(gradients(8)) > 2 * size(gradients(1) * 8)


# ------------------------------------------------------------ the round trip


@pytest.fixture
def round_trip(vaapi_encodes, vaapi_decodes, upload_texture):
    """Code frames and decode them on this machine, returning the pictures in order."""
    vaapi_encodes('av1')
    vaapi_decodes('av1')

    def run(frames, size=SIZE, **settings):
        packets = record(frames, upload_texture, size=size,
                         **{'bitrate': 8_000_000, **settings})
        pictures = decoded(packets)
        assert sorted(pictures) == [index * TICK for index in range(len(frames))]
        return [pictures[index * TICK] for index in range(len(frames))]
    return run


class TestRoundTrip:
    def test_what_was_encoded_comes_back(self, round_trip):
        frames = gradients(12)
        pictures = round_trip(frames)
        errors = [float(np.abs(picture - frame[..., :3]).mean())
                  for picture, frame in zip(pictures, frames, strict=True)]
        assert all(picture.shape == (SIZE[1], SIZE[0], 3) for picture in pictures)
        assert max(errors) < 4.0, errors

    def test_inter_frames_predict_from_the_frame_before(self, round_trip):
        """A reference the driver was not told of decodes as another picture."""
        frames = gradients(25)
        pictures = round_trip(frames, gop=25)
        errors = [float(np.abs(picture - frame[..., :3]).mean())
                  for picture, frame in zip(pictures, frames, strict=True)]
        assert max(errors) < 4.0, errors

    @pytest.mark.parametrize('size', [(256, 144), UNEVEN])
    def test_a_recording_runs_past_the_sixteen_pictures_mesa_converts_for(
            self, round_trip, size):
        """Mesa skips the conversion it was asked for from the seventeenth picture.

        It does so where the texture's size is the coded size and the driver
        left the texture uncompressed, which these sizes are on the device
        this was found on. The encoder keeps the conversion for itself, so
        the later pictures are still the ones drawn.
        """
        width, height = size
        frames = gradients(24, size)
        pictures = round_trip(frames, size=size, gop=12)
        errors = [float(np.abs(picture[:height, :width] - frame[..., :3]).mean())
                  for picture, frame in zip(pictures, frames, strict=True)]
        assert max(errors) < 4.0, errors

    @pytest.mark.parametrize('mode', ['cqp', 'vbr'])
    def test_it_comes_back_under_every_rate_control(self, round_trip, mode):
        frames = gradients(8)
        pictures = round_trip(frames, rate_control=mode, qp=22)
        errors = [float(np.abs(picture - frame[..., :3]).mean())
                  for picture, frame in zip(pictures, frames, strict=True)]
        assert max(errors) < 4.0, errors


class TestColour:
    """The conversion and the stream's colour description have to agree.

    One asks the driver for limited-range BT.709 and the other is written into
    the sequence header. A decoder that follows the header gives back the
    colour drawn only where the two agree.
    """

    @pytest.mark.parametrize('size', [SIZE, UNEVEN])
    @pytest.mark.parametrize('colour', [
        (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (128, 128, 128), (200, 120, 40),
    ])
    def test_a_colour_comes_back_as_it_went_in(self, round_trip, colour, size):
        width, height = size
        [picture, _] = round_trip([solid(colour, size)] * 2, size=size,
                                  rate_control='cqp', qp=10)
        found = picture[:height, :width].reshape(-1, 3).mean(axis=0)
        assert np.abs(found - colour).max() <= 4, found


class TestOrientation:
    def test_the_first_row_of_the_texture_is_the_top_of_the_picture(self, round_trip):
        width, height = SIZE
        frame = solid((0, 0, 0))
        frame[:height // 2, :, :3] = 255        # white across the top
        [picture, _] = round_trip([frame] * 2)
        assert picture[height // 4].mean() > 200, 'the top of the picture is white'
        assert picture[3 * height // 4].mean() < 40, 'the bottom is black'

    def test_the_left_of_the_texture_is_the_left_of_the_picture(self, round_trip):
        width, height = SIZE
        frame = solid((0, 0, 0))
        frame[:, :width // 2, :3] = 255
        [picture, _] = round_trip([frame] * 2)
        assert picture[:, width // 4].mean() > 200
        assert picture[:, 3 * width // 4].mean() < 40


class TestASizeThatIsNotWholeBlocks:
    """AV1 has no cropping, so a driver that codes whole blocks codes a larger frame.

    The frame then holds the picture in its top left corner and the render
    size states the picture's own size.
    """

    @pytest.fixture
    def packets(self, vaapi_encodes, upload_texture):
        vaapi_encodes('av1')
        return record(gradients(6, UNEVEN), upload_texture, size=UNEVEN,
                      bitrate=8_000_000)

    def test_the_render_size_is_the_picture(self, packets):
        _, headers = frame_headers(packets)
        assert len(headers) == 6
        for header in headers:
            assert (header.render_width, header.render_height) == UNEVEN
            assert header.frame_width >= UNEVEN[0]
            assert header.frame_height >= UNEVEN[1]

    def test_the_picture_is_the_top_left_of_what_decodes(self, packets, vaapi_decodes):
        vaapi_decodes('av1')
        width, height = UNEVEN
        pictures = decoded(packets)
        assert len(pictures) == 6
        for index, frame in enumerate(gradients(6, UNEVEN)):
            picture = pictures[index * TICK][:height, :width]
            assert picture.shape == (height, width, 3)
            assert float(np.abs(picture - frame[..., :3]).mean()) < 4.0


# ----------------------------------------------------------------- containers


class TestIVF:
    @pytest.fixture
    def written(self, encoder, upload_texture, tmp_path):
        path = tmp_path / 'recording.ivf'
        packets = record(gradients(FRAMES), upload_texture, encoder=encoder)
        with IVFWriter(path, encoder) as stream:
            stream.write(packets)
        return path, packets

    def test_it_reads_back_as_it_was_written(self, written):
        path, packets = written
        with IVFReader(path) as stream:
            assert stream.codec == 'av1'
            assert len(stream) == FRAMES
            assert stream.size == SIZE
            samples = [stream.sample(index) for index in range(len(stream))]
        assert [sample.data for sample in samples] == [packet.data for packet in packets]
        assert [sample.timestamp for sample in samples] == [
            packet.timestamp for packet in packets]
        assert [sample.keyframe for sample in samples] == [
            packet.keyframe for packet in packets]

    def test_it_decodes_here(self, written, vaapi_decodes):
        vaapi_decodes('av1')
        path, _ = written
        with IVFReader(path) as stream:
            samples = [stream.sample(index) for index in range(len(stream))]
        shown = []
        with open_decoder('av1', backend='vaapi') as decoder:
            for sample in samples:
                for frame in decoder.decode(sample):
                    shown.append((frame.timestamp, frame.size))
                    frame.release()
            for frame in decoder.flush():
                shown.append((frame.timestamp, frame.size))
                frame.release()
        assert shown == [(index * TICK, SIZE) for index in range(FRAMES)]

    def test_an_outside_decoder_has_no_complaint(self, written):
        path, _ = written
        complaints = decode_errors(path)
        if complaints is None:
            pytest.skip('no ffmpeg here to decode what was written')
        assert complaints == []

    @pytest.mark.parametrize('mode', ['cqp', 'vbr'])
    def test_nor_under_the_other_rate_controls(self, vaapi_encodes, upload_texture,
                                               tmp_path, mode):
        vaapi_encodes('av1')
        path = tmp_path / f'{mode}.ivf'
        packets = record(gradients(FRAMES, UNEVEN), upload_texture, size=UNEVEN,
                         rate_control=mode, bitrate=1_000_000)
        with IVFWriter(path, codec='av1', width=UNEVEN[0], height=UNEVEN[1],
                       timescale=90000) as stream:
            stream.write(packets)
        complaints = decode_errors(path)
        if complaints is None:
            pytest.skip('no ffmpeg here to decode what was written')
        assert complaints == []


class TestMP4:
    @pytest.fixture
    def written(self, encoder, upload_texture, tmp_path):
        path = tmp_path / 'recording.mp4'
        packets = record(gradients(FRAMES), upload_texture, encoder=encoder)
        with MP4Writer(path, encoder) as movie:
            movie.write(packets)
        return path, packets

    def test_the_sample_description_holds_the_sequence_header_of_the_stream(
            self, encoder, written):
        path, _ = written
        with MP4Reader(path) as movie:
            track = movie.video
            assert codecs.canonical(track.codec) == 'av1'
            assert len(track) == FRAMES
            assert track.parameter_sets() == encoder.headers()

    def test_it_decodes_here_from_its_sample_entry(self, written, vaapi_decodes):
        vaapi_decodes('av1')
        path, _ = written
        with MP4Reader(path) as movie:
            track = movie.video
            samples = [track.sample(index) for index in range(len(track))]
            headers = track.parameter_sets()
        with open_decoder('av1', headers=headers, backend='vaapi') as decoder:
            shown = 0
            for sample in samples:
                for frame in decoder.decode(sample):
                    shown += 1
                    frame.release()
            for frame in decoder.flush():
                shown += 1
                frame.release()
        assert shown == FRAMES

    def test_an_outside_decoder_has_no_complaint(self, written):
        path, _ = written
        complaints = decode_errors(path)
        if complaints is None:
            pytest.skip('no ffmpeg here to decode what was written')
        assert complaints == []
