"""HEVC encoding on real hardware, from a real OpenGL texture and back again.

A texture is drawn into, coded by the driver's HEVC encoder, and the packets
are read three ways: by this package's own parser, which says whether the
stream is the one the control layer decided on; by this machine's HEVC
decoder, which says whether the picture that went in is the one that comes
out; and by ``ffmpeg`` where there is one, which is a decoder nobody here
wrote.
"""
import ctypes
import hashlib
import shutil
import subprocess

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

from pyopengl_video import EncoderError, codecs, encoders, open_decoder, open_encoder
from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.convert import NV12Converter, create_output_texture
from pyopengl_video.decoder import Sample
from pyopengl_video.mp4 import MP4Reader, MP4Writer, split_annexb
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.device import Device
from pyopengl_video.vaapi.encoder import RATE_CONTROL_MODES, VAAPIEncoder, colour_pipeline
from pyopengl_video.vaapi.hevc_encode import PACKED_HEADERS_WRITTEN, HEVCEncoding
from pyopengl_video.vaapi.hevc_parse import (
    NAL_FILLER,
    NAL_IDR_N_LP,
    NAL_PPS,
    NAL_SPS,
    NAL_TRAIL_R,
    NAL_VPS,
    PROFILE_MAIN,
    parse_nal,
    parse_pps,
    parse_slice_header,
    parse_sps,
    parse_vps,
)
from tests.conftest import blit, decode_errors, frame_bytes, gradient_frame

SIZE = (320, 240)
#: A size whose width is not whole coding tree blocks and whose height is not
#: whole coding blocks.
UNEVEN = (200, 130)
FRAMES = 20
GOP = 10
TICK = 3000                                  # 30 fps in a 90 kHz timescale

KEY_PICTURE = [NAL_VPS, NAL_SPS, NAL_PPS, NAL_IDR_N_LP]


@pytest.fixture
def encoder(vaapi_encodes):
    """A 320x240 HEVC encoder, key pictures every ten."""
    vaapi_encodes('hevc')
    made = VAAPIEncoder(*SIZE, 'hevc', fps=30, bitrate=2_000_000, gop=GOP)
    yield made
    made.close()


def record(frames, upload_texture, size=SIZE, forced=(), encoder=None, codec='hevc',
           **settings):
    """Code `frames`, RGBA arrays first row at the top, and return the packets.

    forced -- the indices to ask for a key picture at
    encoder -- an encoder to use and leave open; one is made from `settings`
        and closed otherwise
    """
    made = encoder or VAAPIEncoder(*size, codec, **{'fps': 30, 'gop': GOP, **settings})
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


def decoded(samples, headers=b'', codec='hevc'):
    """Each sample's picture as RGB, first row at the top, by timestamp."""
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
        with open_decoder(codec, headers=headers, backend='vaapi') as decoder:
            for sample in samples:
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
    return [parse_nal(unit).nal_unit_type for unit in split_annexb(packet.data)]


def parsed(packets):
    """The stream's sequence and picture parameter sets and every slice header, in order."""
    sequences, pictures, headers = {}, {}, []
    for packet in packets:
        for data in split_annexb(packet.data):
            unit = parse_nal(data)
            if unit.nal_unit_type == NAL_SPS:
                sps = parse_sps(unit.rbsp)
                sequences[sps.sps_seq_parameter_set_id] = sps
            elif unit.nal_unit_type == NAL_PPS:
                pps = parse_pps(unit.rbsp)
                pictures[pps.pps_pic_parameter_set_id] = pps
            elif unit.is_slice:
                headers.append(parse_slice_header(unit, sequences, pictures))
    return sequences[0], pictures[0], headers


def errors(pictures, frames):
    """The mean absolute difference of each decoded picture from the frame drawn."""
    return [float(np.abs(pictures[index * TICK] - frame[..., :3]).mean())
            for index, frame in enumerate(frames)]


# ------------------------------------------------------------- what it opened


class TestWhatTheSessionNegotiated:
    def test_it_reports_the_shape_of_the_stream(self, encoder):
        assert encoder.size == SIZE
        assert encoder.codec == 'hevc'
        assert encoder.reorders_frames is False
        assert isinstance(encoder.control, HEVCEncoding)

    def test_it_asks_only_for_the_packed_headers_it_writes(self, encoder):
        assert encoder.packed_headers & ~PACKED_HEADERS_WRITTEN == 0

    @pytest.mark.parametrize('name', ['hevc', 'h265', 'H.265'])
    def test_open_encoder_finds_the_backend_by_any_of_the_codecs_names(
            self, vaapi_encodes, name):
        vaapi_encodes('hevc')
        assert 'hevc' in [codec for backend in encoders() if backend.name == 'vaapi'
                          for codec in backend.codecs]
        with open_encoder(*SIZE, codec=name, fps=30, backend='vaapi') as found:
            assert isinstance(found, VAAPIEncoder)
            assert found.codec == 'hevc'

    def test_the_headers_are_three_parameter_sets_before_anything_is_coded(self, encoder):
        assert [parse_nal(unit).nal_unit_type
                for unit in split_annexb(encoder.headers())] == [NAL_VPS, NAL_SPS, NAL_PPS]


# ------------------------------------------------------------------ the stream


class TestTheStream:
    @pytest.fixture
    def packets(self, encoder, upload_texture):
        return record(gradients(FRAMES), upload_texture, encoder=encoder)

    def test_every_frame_comes_back_as_a_packet(self, packets):
        assert len(packets) == FRAMES
        assert all(len(packet) > 0 for packet in packets)

    def test_it_opens_on_a_key_picture_and_repeats_them_at_the_gop(self, packets):
        assert [index for index, packet in enumerate(packets) if packet.keyframe] == [0, GOP]

    def test_timestamps_come_back_as_they_went_in(self, packets):
        assert [packet.timestamp for packet in packets] == [
            index * TICK for index in range(FRAMES)]
        assert all(packet.duration == TICK for packet in packets)

    def test_a_key_picture_carries_the_three_parameter_sets_then_an_idr_slice(self, packets):
        assert unit_kinds(packets[0]) == KEY_PICTURE
        assert unit_kinds(packets[GOP]) == KEY_PICTURE

    def test_every_packet_holds_the_picture_it_says_it_does(self, packets):
        for index, packet in enumerate(packets):
            expected = KEY_PICTURE if packet.keyframe else [NAL_TRAIL_R]
            assert unit_kinds(packet) == expected, f'packet {index}'

    def test_every_packet_is_annex_b(self, packets):
        for packet in packets:
            assert packet.data.startswith(b'\x00\x00\x00\x01')

    def test_no_start_code_survives_inside_a_unit(self, packets):
        for packet in packets:
            for unit in split_annexb(packet.data):
                assert b'\x00\x00\x01' not in unit
                assert b'\x00\x00\x00' not in unit

    def test_the_pictures_are_the_structure_the_layer_decided(self, packets):
        _, _, headers = parsed(packets)
        assert len(headers) == FRAMES
        assert [header.kind for header in headers] == (['I'] + ['P'] * (GOP - 1)) * 2
        assert [header.slice_pic_order_cnt_lsb for header in headers] == list(range(GOP)) * 2
        for header in headers:
            assert header.first_slice_segment_in_pic_flag == 1, 'one slice a picture'
            assert header.short_term_ref_pic_set.delta_poc_s0 == (
                () if header.idr else (-1,))

    def test_the_sequence_is_main_profile_eight_bit_420(self, packets):
        sps, _, _ = parsed(packets)
        assert sps.profile_tier_level.general_profile_idc == PROFILE_MAIN
        assert sps.profile_tier_level.general_tier_flag == 0
        assert sps.chroma_format_idc == 1
        assert sps.bit_depth == 8
        assert sps.size == SIZE

    def test_the_sequence_is_limited_range_bt709_at_the_frame_rate(self, packets):
        sps, _, _ = parsed(packets)
        assert sps.vui.video_full_range_flag == 0
        assert (sps.vui.colour_primaries, sps.vui.transfer_characteristics,
                sps.vui.matrix_coeffs) == (1, 1, 1)
        assert sps.frame_rate == (30, 1)

    def test_the_sequence_states_the_level_and_the_buffer_the_layer_chose(
            self, encoder, packets):
        sps, _, _ = parsed(packets)
        assert sps.profile_tier_level.general_level_idc == encoder.control.sets.level_idc
        assert sps.dpb_size == encoder.control.references + 1
        assert sps.reorder_pics == 0
        [video] = [parse_vps(parse_nal(unit).rbsp) for unit in split_annexb(packets[0].data)
                   if parse_nal(unit).nal_unit_type == NAL_VPS]
        assert video.profile_tier_level == sps.profile_tier_level

    def test_the_tools_the_stream_declares_are_the_ones_the_layer_asked_for(
            self, encoder, packets):
        """The driver writes the sets, from what it was told and what it codes with."""
        sps, pps, _ = parsed(packets)
        tools = encoder.control.sets.tools
        assert sps.amp_enabled_flag == tools.amp
        assert sps.sample_adaptive_offset_enabled_flag == tools.sample_adaptive_offset
        assert sps.strong_intra_smoothing_enabled_flag == tools.strong_intra_smoothing
        assert pps.cu_qp_delta_enabled_flag == tools.cu_qp_delta
        assert pps.sign_data_hiding_enabled_flag == tools.sign_data_hiding
        assert pps.transform_skip_enabled_flag == tools.transform_skip
        assert sps.sps_temporal_mvp_enabled_flag == 0
        assert sps.scaling_list_enabled_flag == 0
        assert pps.tiles_enabled_flag == 0

    def test_the_coding_tree_block_is_the_one_the_driver_stated(self, encoder, packets):
        sps, _, _ = parsed(packets)
        blocks = encoder.control.sets.blocks
        assert sps.ctb_log2_size == blocks.coding_tree
        assert sps.log2_min_luma_coding_block_size_minus3 + 3 == blocks.min_coding

    def test_the_headers_are_the_sets_the_stream_carries(self, encoder, packets):
        sets = [unit for unit in split_annexb(packets[GOP].data)
                if parse_nal(unit).nal_unit_type in (NAL_VPS, NAL_SPS, NAL_PPS)]
        assert list(split_annexb(encoder.headers())) == sets

    def test_a_forced_key_picture_starts_a_new_group(self, encoder, upload_texture):
        packets = record(gradients(8), upload_texture, encoder=encoder, forced={3})
        assert [packet.keyframe for packet in packets] == [
            True, False, False, True, False, False, False, False]
        assert unit_kinds(packets[3]) == KEY_PICTURE, 'it carries the parameter sets again'
        _, _, headers = parsed(packets)
        assert [header.slice_pic_order_cnt_lsb for header in headers] == [
            0, 1, 2, 0, 1, 2, 3, 4]


# ---------------------------------------------------------------- settings


def offered_rate_controls():
    """The rate control modes the first render node's HEVC encoder states."""
    device = Device(api.render_nodes()[0])
    try:
        attribute = api.VAConfigAttrib(type=api.VAConfigAttribRateControl)
        device.check(
            device.va.vaGetConfigAttributes(
                device.display, api.VAProfileHEVCMain, api.VAEntrypointEncSlice,
                ctypes.byref(attribute), 1),
            'vaGetConfigAttributes')
    finally:
        device.close()
    return {name for name, bit in RATE_CONTROL_MODES.items() if attribute.value & bit}


class TestSettings:
    @pytest.mark.parametrize('mode', sorted(RATE_CONTROL_MODES))
    def test_each_rate_control_mode_the_driver_offers_records(
            self, vaapi_encodes, vaapi_decodes, upload_texture, mode):
        vaapi_encodes('hevc')
        vaapi_decodes('hevc')
        if mode not in offered_rate_controls():
            pytest.skip(f'this driver has no {mode} rate control for HEVC')
        frames = gradients(12)
        packets = record(frames, upload_texture, bitrate=2_000_000, gop=5,
                         rate_control=mode)
        assert len(packets) == 12
        assert [packet.keyframe for packet in packets] == [
            index % 5 == 0 for index in range(12)]
        assert max(errors(decoded(samples_of(packets)), frames)) < 4.0

    def test_a_constant_bitrate_the_pictures_do_not_need_is_made_up_with_filler(
            self, vaapi_encodes, vaapi_decodes, upload_texture, tmp_path):
        """Filler data units may follow a picture's slice, and decoders pass over them."""
        vaapi_encodes('hevc')
        vaapi_decodes('hevc')
        frames = [solid((90, 140, 200))] * 40
        made = VAAPIEncoder(*SIZE, 'hevc', fps=30, gop=GOP, rate_control='cbr',
                            bitrate=8_000_000)
        path = tmp_path / 'flat.mp4'
        try:
            packets = record(frames, upload_texture, encoder=made)
            with MP4Writer(path, made) as movie:
                movie.write(packets)
        finally:
            made.close()
        for index, packet in enumerate(packets):
            kinds = unit_kinds(packet)
            expected = KEY_PICTURE if packet.keyframe else [NAL_TRAIL_R]
            assert kinds[:len(expected)] == expected, f'packet {index}'
            assert set(kinds[len(expected):]) <= {NAL_FILLER}, f'packet {index}'
        assert max(errors(decoded(samples_of(packets)), frames)) < 4.0
        assert decode_errors(path) in (None, [])

    @pytest.mark.parametrize('qp', [18, 26, 34])
    def test_a_constant_quantiser_is_the_one_asked_for(self, vaapi_encodes,
                                                       upload_texture, qp):
        vaapi_encodes('hevc')
        packets = record(gradients(4), upload_texture, rate_control='cqp', qp=qp)
        _, pps, headers = parsed(packets)
        assert pps.cu_qp_delta_enabled_flag == 0
        assert [26 + pps.init_qp_minus26 + header.slice_qp_delta for header in headers] == [
            qp] * 4

    def test_a_coarser_quantiser_costs_fewer_bytes(self, vaapi_encodes, upload_texture):
        vaapi_encodes('hevc')
        generator = np.random.default_rng(20261012)
        width, height = SIZE
        frames = [generator.integers(0, 256, (height, width, 4), dtype=np.uint8)
                  for _ in range(4)]

        def size(qp):
            return sum(len(packet) for packet in record(
                frames, upload_texture, rate_control='cqp', qp=qp))

        assert size(40) < size(20)

    def test_a_higher_bitrate_is_spent(self, vaapi_encodes, upload_texture):
        vaapi_encodes('hevc')
        generator = np.random.default_rng(20261012)
        width, height = SIZE
        frames = [generator.integers(0, 256, (height, width, 4), dtype=np.uint8)
                  for _ in range(30)]

        def size(bitrate):
            return sum(len(packet) for packet in record(
                frames, upload_texture, rate_control='cbr', bitrate=bitrate, gop=30))

        assert size(4_000_000) > 2 * size(500_000)

    def test_a_picture_smaller_than_the_encoder_takes_is_an_encoder_error(
            self, vaapi_encodes):
        vaapi_encodes('hevc')
        with pytest.raises(EncoderError):
            VAAPIEncoder(16, 16, 'hevc')

    def test_an_odd_size_is_an_encoder_error(self, vaapi_encodes):
        vaapi_encodes('hevc')
        with pytest.raises(EncoderError, match='321x240'):
            VAAPIEncoder(321, 240, 'hevc')

    def test_b_frames_are_refused_because_nothing_reorders(self, vaapi_encodes):
        vaapi_encodes('hevc')
        with pytest.raises(EncoderError, match='bframes'):
            VAAPIEncoder(*SIZE, 'hevc', bframes=2)


class TestContentSensitivity:
    """A registration that went nowhere still produces a well-formed stream."""

    def test_noise_costs_far_more_than_a_flat_colour(self, vaapi_encodes, upload_texture):
        vaapi_encodes('hevc')
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
        vaapi_encodes('hevc')

        def size(frames):
            packets = record(frames, upload_texture, rate_control='cqp', qp=26, gop=100)
            # The first picture is intra either way, so compare the rest.
            return sum(len(packet) for packet in packets[1:])

        assert size(gradients(8)) > 2 * size(gradients(1) * 8)


# ------------------------------------------------------------ the round trip


@pytest.fixture
def round_trip(vaapi_encodes, vaapi_decodes, upload_texture):
    """Code frames and decode them on this machine, returning the pictures in order."""
    vaapi_encodes('hevc')
    vaapi_decodes('hevc')

    def run(frames, size=SIZE, **settings):
        packets = record(frames, upload_texture, size=size,
                         **{'bitrate': 8_000_000, **settings})
        pictures = decoded(samples_of(packets))
        assert sorted(pictures) == [index * TICK for index in range(len(frames))]
        return [pictures[index * TICK] for index in range(len(frames))]
    return run


class TestRoundTrip:
    def test_what_was_encoded_comes_back(self, round_trip):
        frames = gradients(12)
        pictures = round_trip(frames)
        assert all(picture.shape == (SIZE[1], SIZE[0], 3) for picture in pictures)
        found = [float(np.abs(picture - frame[..., :3]).mean())
                 for picture, frame in zip(pictures, frames, strict=True)]
        assert max(found) < 4.0, found

    def test_predicted_pictures_predict_from_the_picture_before(self, round_trip):
        """A reference the driver was not told of decodes as another picture."""
        frames = gradients(25)
        pictures = round_trip(frames, gop=25)
        found = [float(np.abs(picture - frame[..., :3]).mean())
                 for picture, frame in zip(pictures, frames, strict=True)]
        assert max(found) < 4.0, found

    def test_a_group_longer_than_the_order_count_field_decodes(self, round_trip):
        """slice_pic_order_cnt_lsb is eight bits, and picture 256 is zero again."""
        frames = gradients(270)
        pictures = round_trip(frames, gop=300, rate_control='cqp', qp=24)
        found = [float(np.abs(picture - frame[..., :3]).mean())
                 for picture, frame in zip(pictures, frames, strict=True)]
        assert max(found) < 4.0, max(found)

    def test_a_forced_key_picture_decodes_with_what_follows_it(self, vaapi_encodes,
                                                              vaapi_decodes, upload_texture):
        vaapi_encodes('hevc')
        vaapi_decodes('hevc')
        frames = gradients(9)
        packets = record(frames, upload_texture, forced={4}, bitrate=8_000_000)
        assert max(errors(decoded(samples_of(packets)), frames)) < 4.0
        tail = decoded(samples_of(packets)[4:])
        assert sorted(tail) == [index * TICK for index in range(4, 9)], (
            'decoding starts at the forced key picture')


class TestColour:
    """The conversion and the stream's colour description have to agree.

    One asks the driver for limited-range BT.709 and the other is written into
    the sequence parameter set. A decoder that follows the set gives back the
    colour drawn only where the two agree.
    """

    @pytest.mark.parametrize('colour', [
        (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (128, 128, 128), (200, 120, 40),
    ])
    def test_a_colour_comes_back_as_it_went_in(self, round_trip, colour):
        [picture, _] = round_trip([solid(colour)] * 2, rate_control='cqp', qp=10)
        found = picture.reshape(-1, 3).mean(axis=0)
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


class TestTheConversionIntoALargerSurface:
    """The surface coded from is the coded size, and the picture its top left."""

    def test_the_pass_writes_the_picture_rectangle(self):
        pipeline = colour_pipeline(7, UNEVEN)
        region = pipeline.output_region.contents
        assert (region.x, region.y, region.width, region.height) == (0, 0, *UNEVEN)

    def test_without_a_size_it_fills_its_target(self):
        assert not colour_pipeline(7).output_region

    def test_an_h264_picture_off_the_macroblock_grid_keeps_its_edges(
            self, vaapi_encodes, vaapi_decodes, upload_texture):
        """The conversion is the same pass for every codec."""
        vaapi_encodes('h264')
        vaapi_decodes('h264')
        frames = gradients(4, UNEVEN)
        packets = record(frames, upload_texture, size=UNEVEN, codec='h264',
                         rate_control='cqp', qp=18)
        pictures = decoded(samples_of(packets), codec='h264')
        assert max(errors(pictures, frames)) < 4.0
        for index, frame in enumerate(frames):
            picture = pictures[index * TICK]
            assert picture.shape == (UNEVEN[1], UNEVEN[0], 3)
            assert float(np.abs(picture[-8:] - frame[-8:, :, :3]).mean()) < 5.0


class TestASizeThatIsNotWholeBlocks:
    """The picture is coded larger and the conformance window crops it back."""

    @pytest.fixture
    def packets(self, vaapi_encodes, upload_texture):
        vaapi_encodes('hevc')
        return record(gradients(6, UNEVEN), upload_texture, size=UNEVEN,
                      bitrate=8_000_000)

    def test_the_stream_states_the_coded_size_the_layer_chose(self, vaapi_encodes):
        """A driver writing its own sequence parameter set must round as the layer does."""
        vaapi_encodes('hevc')
        made = VAAPIEncoder(*UNEVEN, 'hevc', fps=30)
        try:
            assert made.control.coded_size[0] >= UNEVEN[0]
            assert made.control.coded_size[1] >= UNEVEN[1]
            assert made.control.coded_size != UNEVEN
            stated = made.control.coded_size
        finally:
            made.close()
        made = VAAPIEncoder(*UNEVEN, 'hevc', fps=30)
        handle = made.new_input()
        try:
            made.encode(handle, timestamp=0)
            [packet] = made.flush()
        finally:
            handle.close()
            made.close()
        sps, _, _ = parsed([packet])
        assert sps.coded_size == stated

    def test_the_window_crops_to_the_picture(self, packets):
        sps, _, _ = parsed(packets)
        assert sps.visible == (0, 0, *UNEVEN)
        assert sps.size == UNEVEN

    def test_it_decodes_at_its_true_size_to_what_was_drawn(self, packets, vaapi_decodes):
        vaapi_decodes('hevc')
        width, height = UNEVEN
        frames = gradients(6, UNEVEN)
        pictures = decoded(samples_of(packets))
        assert len(pictures) == 6
        for picture in pictures.values():
            assert picture.shape == (height, width, 3)
        assert max(errors(pictures, frames)) < 4.0

    def test_the_last_rows_and_columns_are_the_pictures_own(self, packets, vaapi_decodes):
        """A picture scaled to fill the coded size loses its edges to the window."""
        vaapi_decodes('hevc')
        frames = gradients(6, UNEVEN)
        pictures = decoded(samples_of(packets))
        for index, frame in enumerate(frames):
            picture = pictures[index * TICK]
            assert float(np.abs(picture[-8:] - frame[-8:, :, :3]).mean()) < 5.0
            assert float(np.abs(picture[:, -8:] - frame[:, -8:, :3]).mean()) < 5.0

    @pytest.mark.parametrize('size', [(1920, 1080), (162, 130)])
    def test_other_sizes_off_the_grid_come_back_whole(self, round_trip, size):
        frames = gradients(3, size)
        pictures = round_trip(frames, size=size, rate_control='cqp', qp=22)
        assert all(picture.shape == (size[1], size[0], 3) for picture in pictures)
        found = [float(np.abs(picture - frame[..., :3]).mean())
                 for picture, frame in zip(pictures, frames, strict=True)]
        assert max(found) < 4.0, found


# ----------------------------------------------------------------- containers


class TestARawStream:
    """The packets joined end to end are a raw ``.h265`` file."""

    @pytest.fixture
    def written(self, encoder, upload_texture, tmp_path):
        path = tmp_path / 'recording.h265'
        packets = record(gradients(FRAMES), upload_texture, encoder=encoder)
        path.write_bytes(b''.join(packet.data for packet in packets))
        return path, packets

    def test_it_reads_back_as_the_pictures_written(self, written):
        path, packets = written
        with AnnexBReader(path) as stream:
            assert stream.codec == 'hevc'
            assert stream.size == SIZE
            assert stream.frame_rate == (30, 1)
            assert len(stream) == FRAMES
            assert stream.sync_samples == [0, GOP]
            assert stream.display_order() == list(range(FRAMES))
            assert [stream.sample(index).data for index in range(FRAMES)] == [
                packet.data for packet in packets]

    def test_an_outside_decoder_has_no_complaint(self, written):
        path, _ = written
        complaints = decode_errors(path)
        if complaints is None:
            pytest.skip('no ffmpeg here to decode what was written')
        assert complaints == []

    @pytest.mark.parametrize('mode,size', [('cqp', UNEVEN), ('cbr', SIZE), ('vbr', UNEVEN)])
    def test_an_outside_decoder_reconstructs_the_pictures_this_one_does(
            self, vaapi_encodes, vaapi_decodes, upload_texture, tmp_path, mode, size):
        """Two decoders agreeing on every sample is what a conforming stream gives."""
        vaapi_encodes('hevc')
        vaapi_decodes('hevc')
        if shutil.which('ffmpeg') is None:
            pytest.skip('no ffmpeg here to decode what was written')
        path = tmp_path / f'{mode}.h265'
        packets = record(gradients(30, size), upload_texture, size=size,
                         rate_control=mode, bitrate=1_000_000)
        path.write_bytes(b''.join(packet.data for packet in packets))
        found = subprocess.run(
            ['ffmpeg', '-v', 'error', '-i', str(path), '-pix_fmt', 'nv12',
             '-f', 'rawvideo', '-'], capture_output=True)
        assert found.stderr == b''
        picture = size[0] * size[1] * 3 // 2
        theirs = [hashlib.md5(found.stdout[start:start + picture]).hexdigest()
                  for start in range(0, len(found.stdout), picture)]
        ours = []
        with open_decoder('hevc', backend='vaapi') as decoder:
            for sample in samples_of(packets):
                for frame in decoder.decode(sample):
                    ours.append(hashlib.md5(frame_bytes(frame)).hexdigest())
                    frame.release()
            for frame in decoder.flush():
                ours.append(hashlib.md5(frame_bytes(frame)).hexdigest())
                frame.release()
        assert len(ours) == 30
        assert ours == theirs


class TestMP4:
    @pytest.fixture
    def written(self, encoder, upload_texture, tmp_path):
        path = tmp_path / 'recording.mp4'
        frames = gradients(FRAMES)
        packets = record(frames, upload_texture, encoder=encoder)
        with MP4Writer(path, encoder) as movie:
            movie.write(packets)
        return path, packets, frames

    def test_it_is_an_hvc1_track_of_the_pictures_written(self, written):
        path, _, _ = written
        with MP4Reader(path) as movie:
            track = movie.video
            assert codecs.canonical(track.codec) == 'hevc'
            assert track.size == SIZE
            assert len(track) == FRAMES
            assert [track.sample(index).keyframe for index in range(FRAMES)] == [
                index % GOP == 0 for index in range(FRAMES)]
            assert [track.sample(index).timestamp for index in range(FRAMES)] == [
                index * TICK for index in range(FRAMES)]
        assert b'hvc1' in path.read_bytes()

    def test_the_sample_description_holds_the_sets_the_stream_carries(self, encoder, written):
        path, packets, _ = written
        with MP4Reader(path) as movie:
            stored = movie.video.parameter_sets()
        assert stored == encoder.headers()
        assert list(split_annexb(stored)) == list(split_annexb(packets[0].data))[:3]

    def test_it_decodes_here_from_its_sample_entry_to_what_was_drawn(
            self, written, vaapi_decodes):
        vaapi_decodes('hevc')
        path, _, frames = written
        with MP4Reader(path) as movie:
            track = movie.video
            samples = [track.sample(index) for index in range(len(track))]
            headers = track.parameter_sets()
        pictures = decoded(samples, headers=headers)
        assert sorted(pictures) == [index * TICK for index in range(FRAMES)]
        assert max(errors(pictures, frames)) < 4.0

    def test_an_outside_decoder_has_no_complaint(self, written):
        path, _, _ = written
        complaints = decode_errors(path)
        if complaints is None:
            pytest.skip('no ffmpeg here to decode what was written')
        assert complaints == []

    @pytest.mark.parametrize('mode', ['cqp', 'vbr'])
    def test_nor_at_a_size_off_the_grid_under_the_other_rate_controls(
            self, vaapi_encodes, upload_texture, tmp_path, mode):
        vaapi_encodes('hevc')
        path = tmp_path / f'{mode}.mp4'
        made = VAAPIEncoder(*UNEVEN, 'hevc', fps=30, gop=GOP, rate_control=mode,
                            bitrate=1_000_000)
        try:
            packets = record(gradients(FRAMES, UNEVEN), upload_texture, size=UNEVEN,
                             encoder=made)
            with MP4Writer(path, made) as movie:
                movie.write(packets)
        finally:
            made.close()
        with MP4Reader(path) as movie:
            assert movie.video.size == UNEVEN
        complaints = decode_errors(path)
        if complaints is None:
            pytest.skip('no ffmpeg here to decode what was written')
        assert complaints == []
