"""AV1 encoding's control layer, with no driver.

The layer is given the configuration attributes a driver answers as a plain
dictionary, and what it returns is read back: the picture structure, the
parameter buffers of a key and an inter picture, the surfaces each predicts
from, and the temporal unit made of what a driver wrote.
"""
import ctypes
import weakref

import pytest

from pyopengl_video import EncoderError, obu
from pyopengl_video.vaapi import api, av1, av1_api
from pyopengl_video.vaapi.av1_encode import PACKED_HEADERS_WRITTEN, AV1Encoding
from pyopengl_video.vaapi.av1_parse import (
    ALL_FRAMES,
    INTER_FRAME,
    KEY_FRAME,
    PRIMARY_REF_NONE,
    TX_MODE_LARGEST,
    TX_MODE_SELECT,
    FrameState,
    parse_frame_header,
    parse_sequence_header,
)
from pyopengl_video.vaapi.control import HRD_BUFFER_SECONDS, EncodeSettings

SIZE = (160, 90)
BITRATE = 1_000_000

#: What Mesa's radeonsi answers for AV1 encoding: sequence, picture, misc and
#: raw packed headers, one reference, and a two-byte obu_size with
#: TX_MODE_SELECT alone.
RADEONSI = {
    api.VAConfigAttribEncPackedHeaders: 0x1B,
    api.VAConfigAttribEncMaxRefFrames: 1,
    av1_api.VAConfigAttribEncAV1Ext2: 0xFC7,
}

SEQUENCE = api.VAEncSequenceParameterBufferType
PICTURE = api.VAEncPictureParameterBufferType
TILE_GROUP = api.VAEncSliceParameterBufferType
MISC = api.VAEncMiscParameterBufferType
PACKED = api.VAEncPackedHeaderParameterBufferType
PACKED_DATA = api.VAEncPackedHeaderDataBufferType


def layer(rate_control=api.VA_RC_CBR, gop=4, qp=26, stated=None, size=SIZE):
    settings = EncodeSettings(size=size, frame_rate=(30, 1), bitrate=BITRATE,
                              gop=gop, qp=qp, rate_control=rate_control)
    return AV1Encoding(settings, RADEONSI if stated is None else stated)


def kinds(buffers):
    return [kind for kind, _ in buffers]


def one(buffers, kind):
    """The only buffer of `kind`."""
    found = [payload for found_kind, payload in buffers if found_kind == kind]
    assert len(found) == 1
    return found[0]


def packed(buffers, header):
    """The bytes of the packed header of type `header`."""
    for index, (kind, payload) in enumerate(buffers):
        if kind == PACKED and payload.type == header:
            assert buffers[index + 1][0] == PACKED_DATA
            data = bytes(buffers[index + 1][1])
            assert payload.bit_length == len(data) * 8
            return data
    raise AssertionError(f'no packed header of type {header}')


#: How many pictures each layer under test has coded, which picks its next surface.
CODED = weakref.WeakKeyDictionary()


def code(control, surfaces, force_key=False):
    """Decide the next picture and build its buffers, as the encoder does.

    The reconstruction surfaces are taken in turn, as the encoder takes them.
    """
    picture = control.next_picture(force_key=force_key)
    count = CODED.get(control, 0)
    CODED[control] = count + 1
    surface = surfaces[count % len(surfaces)]
    return picture, control.buffers(picture, coded=77, reconstruction=surface), surface


class TestWhatItIs:
    def test_the_codec_and_the_profile(self):
        assert AV1Encoding.codec == 'av1'
        assert AV1Encoding.profile == api.VAProfileAV1Profile0
        assert AV1Encoding.title == 'AV1 Main'

    def test_rate_control_is_told_the_whole_quantiser_scale(self):
        assert AV1Encoding.max_qp == 255

    def test_it_asks_about_the_packed_headers_and_the_obu_size(self):
        assert api.VAConfigAttribEncPackedHeaders in AV1Encoding.attributes
        assert av1_api.VAConfigAttribEncAV1Ext2 in AV1Encoding.attributes

    def test_one_reconstructed_picture_is_kept(self):
        assert layer().references == 1

    def test_the_surfaces_are_the_picture_size(self):
        assert layer().coded_size == SIZE

    def test_an_odd_size_is_an_encoder_error(self):
        with pytest.raises(EncoderError, match='161x90'):
            layer(size=(161, 90))


class TestPackedHeaders:
    def test_only_the_two_it_writes_are_asked_for(self):
        assert PACKED_HEADERS_WRITTEN == (api.VA_ENC_PACKED_HEADER_SEQUENCE
                                          | api.VA_ENC_PACKED_HEADER_PICTURE)
        assert layer().packed_headers == PACKED_HEADERS_WRITTEN

    def test_one_the_driver_does_not_take_is_not_asked_for(self):
        stated = {api.VAConfigAttribEncPackedHeaders: api.VA_ENC_PACKED_HEADER_SEQUENCE}
        assert layer(stated=stated).packed_headers == api.VA_ENC_PACKED_HEADER_SEQUENCE

    def test_a_driver_that_states_none_is_sent_none(self):
        control = layer(stated={})
        assert control.packed_headers == 0
        _, key, _ = code(control, [5, 6, 7])
        _, inter, _ = code(control, [5, 6, 7])
        assert kinds(key) == [SEQUENCE, MISC, MISC, MISC, PICTURE, TILE_GROUP]
        assert kinds(inter) == [PICTURE, TILE_GROUP]


class TestPictureStructure:
    def test_a_key_frame_every_gop_pictures(self):
        control = layer(gop=4)
        pictures = [control.next_picture() for _ in range(9)]
        assert [picture.keyframe for picture in pictures] == [
            True, False, False, False, True, False, False, False, True]
        assert [picture.detail.number for picture in pictures] == [0, 1, 2, 3, 0, 1, 2, 3, 0]

    def test_a_forced_key_frame_starts_a_new_group(self):
        control = layer(gop=4)
        pictures = [control.next_picture(force_key=index == 2) for index in range(7)]
        assert [picture.keyframe for picture in pictures] == [
            True, False, True, False, False, False, True]

    def test_a_gop_of_one_is_all_key_frames(self):
        control = layer(gop=1)
        assert all(control.next_picture().keyframe for _ in range(4))

    def test_each_picture_is_named_for_an_error(self):
        control = layer(gop=4)
        assert [control.next_picture().name for _ in range(2)] == [
            'key frame 0', 'inter frame 1']


class TestBuffers:
    def test_a_key_picture_describes_the_sequence_first(self):
        _, buffers, _ = code(layer(), [5, 6, 7])
        assert kinds(buffers) == [SEQUENCE, MISC, MISC, MISC, PACKED, PACKED_DATA,
                                  PICTURE, PACKED, PACKED_DATA, TILE_GROUP]

    def test_an_inter_picture_is_a_picture_its_header_and_its_tiles(self):
        control = layer()
        code(control, [5, 6, 7])
        _, buffers, _ = code(control, [5, 6, 7])
        assert kinds(buffers) == [PICTURE, PACKED, PACKED_DATA, TILE_GROUP]

    def test_the_packed_sequence_header_is_the_one_the_layer_reports(self):
        control = layer()
        _, buffers, _ = code(control, [5, 6, 7])
        data = packed(buffers, av1_api.VAEncPackedHeaderAV1_SPS)
        assert data == control.headers() == control.stream.sequence_header()

    def test_the_sequence_parameters(self):
        control = layer(gop=4)
        _, buffers, _ = code(control, [5, 6, 7])
        parameters = one(buffers, SEQUENCE)
        assert isinstance(parameters, av1_api.VAEncSequenceParameterBufferAV1)
        assert parameters.seq_profile == 0
        assert (parameters.seq_level_idx, parameters.seq_tier) == (
            control.stream.level, control.stream.tier)
        assert (parameters.intra_period, parameters.ip_period) == (4, 1)
        assert parameters.bits_per_second == BITRATE
        assert parameters.order_hint_bits_minus_1 == av1.ORDER_HINT_BITS - 1
        bits = parameters.seq_fields.bits
        assert (bits.enable_order_hint, bits.enable_cdef) == (1, 1)
        assert (bits.bit_depth_minus8, bits.subsampling_x, bits.subsampling_y) == (0, 1, 1)
        assert not bits.mono_chrome
        assert not bits.still_picture

    def test_rate_control_starts_from_the_converted_quantiser(self):
        _, buffers, _ = code(layer(qp=30), [5, 6, 7])
        misc = [payload for kind, payload in buffers if kind == MISC]
        by_type = {entry.type: entry for entry in misc}
        control = by_type[api.VAEncMiscParameterTypeRateControl].data
        assert control.bits_per_second == BITRATE
        assert control.initial_qp == av1.base_q_idx(30) == 150
        assert control.max_qp == 255
        hrd = by_type[api.VAEncMiscParameterTypeHRD].data
        assert hrd.buffer_size == BITRATE * HRD_BUFFER_SECONDS

    def test_one_tile_group_holds_every_tile(self):
        control = layer(size=(8192, 4352))
        _, buffers, _ = code(control, [5, 6, 7])
        group = one(buffers, TILE_GROUP)
        assert isinstance(group, av1_api.VAEncTileGroupBufferAV1)
        assert (group.tg_start, group.tg_end) == (0, 3)
        picture = one(buffers, PICTURE)
        assert (picture.tile_cols, picture.tile_rows) == (2, 2)
        assert list(picture.width_in_sbs_minus_1[:2]) == [63, 63]
        assert list(picture.height_in_sbs_minus_1[:2]) == [33, 33]
        assert picture.num_tile_groups_minus1 == 0


class TestPictureParameters:
    def test_a_key_picture(self):
        control = layer()
        _, buffers, _ = code(control, [5, 6, 7])
        parameters = one(buffers, PICTURE)
        assert isinstance(parameters, av1_api.VAEncPictureParameterBufferAV1)
        assert (parameters.frame_width_minus_1, parameters.frame_height_minus_1) == (159, 89)
        assert parameters.coded_buf == 77
        assert parameters.reconstructed_frame == 5
        assert list(parameters.reference_frames) == [api.VA_INVALID_SURFACE] * 8
        flags = parameters.picture_flags.bits
        assert flags.frame_type == KEY_FRAME
        assert flags.error_resilient_mode == 1
        assert flags.enable_frame_obu == 1
        assert not flags.disable_cdf_update
        assert not flags.disable_frame_end_update_cdf
        assert parameters.primary_ref_frame == PRIMARY_REF_NONE
        assert parameters.refresh_frame_flags == ALL_FRAMES
        assert parameters.order_hint == 0
        assert parameters.ref_frame_ctrl_l0.value == 0
        assert parameters.ref_frame_ctrl_l1.value == 0

    def test_an_inter_picture_predicts_from_the_picture_before(self):
        control = layer()
        code(control, [5, 6, 7])
        _, buffers, _ = code(control, [5, 6, 7])
        parameters = one(buffers, PICTURE)
        assert parameters.reconstructed_frame == 6
        assert list(parameters.reference_frames) == [5] * 8
        assert list(parameters.ref_frame_idx) == [0] * 7
        flags = parameters.picture_flags.bits
        assert flags.frame_type == INTER_FRAME
        assert flags.error_resilient_mode == 0
        assert parameters.primary_ref_frame == 0
        assert parameters.refresh_frame_flags == ALL_FRAMES
        assert parameters.order_hint == 1
        # LAST_FRAME is searched and nothing else is.
        assert parameters.ref_frame_ctrl_l0.fields.search_idx0 == 1
        assert parameters.ref_frame_ctrl_l0.value == 1
        assert parameters.ref_frame_ctrl_l1.value == 0
        assert parameters.mode_control_flags.bits.reference_mode == 0

    def test_the_quantiser_is_on_the_av1_scale(self):
        control = layer(qp=20)
        _, buffers, _ = code(control, [5, 6, 7])
        parameters = one(buffers, PICTURE)
        assert parameters.base_qindex == 100
        assert (parameters.min_base_qindex, parameters.max_base_qindex) == (1, 255)
        level = av1.loop_filter_level(100)
        assert list(parameters.filter_level) == [level, level]
        assert (parameters.filter_level_u, parameters.filter_level_v) == (level, level)

    def test_the_tile_group_obu_has_a_size_field(self):
        _, buffers, _ = code(layer(), [5, 6, 7])
        assert one(buffers, PICTURE).tile_group_obu_hdr_info.bits.obu_has_size_field == 1

    def test_global_motion_is_the_identity(self):
        _, buffers, _ = code(layer(), [5, 6, 7])
        for model in one(buffers, PICTURE).wm:
            assert model.wmtype == av1_api.VAAV1EncTransformationIdentity
            assert list(model.wmmat[:6]) == [0, 0, 1 << 16, 0, 0, 1 << 16]
            assert not model.invalid

    @pytest.mark.parametrize('ext2, expected', [
        (0xFC7, TX_MODE_SELECT),            # bit 2 of tx_mode_support
        (0x02 << 4, TX_MODE_LARGEST),       # bit 1 alone
        (0x06 << 4, TX_MODE_SELECT),
    ])
    def test_the_transform_mode_is_one_the_driver_takes(self, ext2, expected):
        stated = dict(RADEONSI)
        stated[av1_api.VAConfigAttribEncAV1Ext2] = ext2
        control = layer(stated=stated)
        _, buffers, _ = code(control, [5, 6, 7])
        assert one(buffers, PICTURE).mode_control_flags.bits.tx_mode == expected
        assert control.stream.tx_mode == expected

    def test_a_driver_that_does_not_say_is_given_tx_mode_select(self):
        stated = {api.VAConfigAttribEncPackedHeaders: 0x1B}
        assert layer(stated=stated).stream.tx_mode == TX_MODE_SELECT


class TestThePackedFrameHeader:
    def headers(self, control, count):
        """Each picture's parameters and its packed header, parsed in order."""
        sequence = parse_sequence_header(next(obu.split(control.headers())).payload)
        state = FrameState()
        found = []
        for _ in range(count):
            _, buffers, _ = code(control, [5, 6, 7])
            data = packed(buffers, av1_api.VAEncPackedHeaderAV1_PPS)
            header = parse_frame_header(next(obu.split(data)).payload, sequence, state)
            state.update(header, sequence)
            found.append((one(buffers, PICTURE), header, data))
        return found

    def test_it_says_what_the_picture_parameters_say(self):
        for parameters, header, _ in self.headers(layer(gop=3), 5):
            assert header.frame_type == parameters.picture_flags.bits.frame_type
            assert header.order_hint == parameters.order_hint
            assert header.primary_ref_frame == parameters.primary_ref_frame
            assert header.refresh_frame_flags == parameters.refresh_frame_flags
            assert header.quantization.base_q_idx == parameters.base_qindex
            assert header.loop_filter.level[0] == parameters.filter_level[0]
            assert header.tx_mode == parameters.mode_control_flags.bits.tx_mode
            assert header.error_resilient_mode == parameters.picture_flags.bits.error_resilient_mode
            assert (header.tile_info.tile_cols, header.tile_info.tile_rows) == (
                parameters.tile_cols, parameters.tile_rows)
            assert header.show_frame

    def test_it_opens_a_frame_obu(self):
        for _, _, data in self.headers(layer(), 2):
            assert next(obu.split(data)).kind == obu.FRAME

    def test_the_obu_size_takes_the_bytes_the_driver_states(self):
        # radeonsi's obu_size_bytes_minus1 is 1.
        (_, _, data), = self.headers(layer(), 1)
        assert data[1] & 0x80 and not data[2] & 0x80

    def test_the_obu_size_takes_four_bytes_where_none_is_stated(self):
        stated = {api.VAConfigAttribEncPackedHeaders: 0x1B}
        (_, _, data), = self.headers(layer(stated=stated), 1)
        assert all(byte & 0x80 for byte in data[1:4]) and not data[4] & 0x80

    def test_under_rate_control_the_driver_is_told_where_to_write(self):
        control = layer(rate_control=api.VA_RC_VBR)
        for parameters, header, data in self.headers(control, 2):
            expected = control.stream.frame_header(
                av1.Frame(key=header.frame_type == KEY_FRAME, number=header.order_hint,
                          base_q_idx=130), size_bytes=2)
            assert data == expected.data
            assert parameters.bit_offset_qindex == expected.bit_offset_qindex > 0
            assert parameters.bit_offset_segmentation == expected.bit_offset_segmentation
            assert (parameters.bit_offset_loopfilter_params
                    == expected.bit_offset_loopfilter_params)
            assert parameters.bit_offset_cdef_params == expected.bit_offset_cdef_params
            assert parameters.size_in_bits_cdef_params == 16
            assert parameters.byte_offset_frame_hdr_obu_size == 1
            assert parameters.size_in_bits_frame_hdr_obu == expected.size_in_bits

    def test_at_a_constant_quantiser_only_the_obu_size_is_rewritten(self):
        for parameters, _, _ in self.headers(layer(rate_control=api.VA_RC_CQP), 2):
            assert parameters.bit_offset_qindex == 0
            assert parameters.bit_offset_segmentation == 0
            assert parameters.bit_offset_loopfilter_params == 0
            assert parameters.bit_offset_cdef_params == 0
            assert parameters.size_in_bits_cdef_params == 0
            assert parameters.size_in_bits_frame_hdr_obu == 0
            assert parameters.byte_offset_frame_hdr_obu_size == 1

    def test_no_offset_is_given_for_a_header_that_is_not_sent(self):
        control = layer(stated={})
        _, buffers, _ = code(control, [5, 6, 7])
        parameters = one(buffers, PICTURE)
        assert parameters.bit_offset_qindex == 0
        assert parameters.byte_offset_frame_hdr_obu_size == 0
        assert parameters.size_in_bits_frame_hdr_obu == 0


class TestReferences:
    def test_every_inter_picture_names_the_surface_coded_before_it(self):
        control = layer(gop=4)
        surfaces = [5, 6, 7]
        previous = None
        for _ in range(11):
            picture, buffers, surface = code(control, surfaces)
            parameters = one(buffers, PICTURE)
            assert parameters.reconstructed_frame == surface
            if picture.keyframe:
                assert list(parameters.reference_frames) == [api.VA_INVALID_SURFACE] * 8
            else:
                assert list(parameters.reference_frames) == [previous] * 8
                assert previous != surface
            previous = surface

    def test_a_forced_key_frame_drops_the_reference(self):
        control = layer(gop=10)
        code(control, [5, 6, 7])
        code(control, [5, 6, 7])
        _, buffers, _ = code(control, [5, 6, 7], force_key=True)
        assert list(one(buffers, PICTURE).reference_frames) == [api.VA_INVALID_SURFACE] * 8
        _, buffers, surface = code(control, [5, 6, 7])
        assert len(set(one(buffers, PICTURE).reference_frames)) == 1
        assert one(buffers, PICTURE).reference_frames[0] != surface


# ------------------------------------------------------------ the packet

FRAME_OBU = obu.Obu(obu.FRAME, b'\x10\x00\x55\xAA').encoded()
INTER_OBU = obu.Obu(obu.FRAME, b'\x30\x00\x55').encoded()
DELIMITER = obu.Obu(obu.TEMPORAL_DELIMITER, b'').encoded()


def driver_sequence_header():
    """A sequence header as a driver that codes whole superblocks rewrites it."""
    return av1.Sequence(192, 96, (30, 1), BITRATE).sequence_header()


def unit_kinds(data):
    return [unit.kind for unit in obu.split(data)]


class TestTheTemporalUnit:
    def key(self, control):
        picture = control.next_picture()
        control.buffers(picture, coded=77, reconstruction=5)
        return picture

    def test_a_key_frame_is_a_delimiter_the_sequence_header_and_the_frame(self):
        control = layer()
        data = control.coded(self.key(control), driver_sequence_header() + FRAME_OBU)
        assert data == DELIMITER + driver_sequence_header() + FRAME_OBU

    def test_the_headers_are_the_ones_the_stream_carries(self):
        control = layer()
        assert control.headers() == control.stream.sequence_header()
        control.coded(self.key(control), driver_sequence_header() + FRAME_OBU)
        assert control.headers() == driver_sequence_header()

    def test_a_key_frame_the_driver_gave_no_sequence_header_is_given_the_layers(self):
        control = layer()
        data = control.coded(self.key(control), FRAME_OBU)
        assert data == DELIMITER + control.stream.sequence_header() + FRAME_OBU
        assert control.headers() == control.stream.sequence_header()

    def test_an_inter_frame_is_a_delimiter_and_the_frame(self):
        control = layer()
        control.coded(self.key(control), driver_sequence_header() + FRAME_OBU)
        data = control.coded(control.next_picture(), INTER_OBU)
        assert data == DELIMITER + INTER_OBU

    def test_a_delimiter_the_driver_wrote_is_not_doubled(self):
        control = layer()
        data = control.coded(self.key(control),
                             DELIMITER + driver_sequence_header() + FRAME_OBU)
        assert unit_kinds(data) == [obu.TEMPORAL_DELIMITER, obu.SEQUENCE_HEADER, obu.FRAME]

    def test_a_sequence_header_on_an_inter_frame_is_dropped(self):
        control = layer()
        control.coded(self.key(control), driver_sequence_header() + FRAME_OBU)
        data = control.coded(control.next_picture(), driver_sequence_header() + INTER_OBU)
        assert data == DELIMITER + INTER_OBU

    def test_every_unit_is_given_a_size_field(self):
        control = layer()
        bare = bytes([obu.FRAME << 3]) + b'\x10\x00\x55\xAA'
        data = control.coded(self.key(control), driver_sequence_header() + bare)
        assert data == DELIMITER + driver_sequence_header() + FRAME_OBU

    def test_a_frame_header_and_its_tile_group_are_kept_together(self):
        control = layer()
        header = obu.Obu(obu.FRAME_HEADER, b'\x10\x00\x80').encoded()
        tiles = obu.Obu(obu.TILE_GROUP, b'\x55\xAA').encoded()
        data = control.coded(self.key(control), driver_sequence_header() + header + tiles)
        assert unit_kinds(data) == [obu.TEMPORAL_DELIMITER, obu.SEQUENCE_HEADER,
                                    obu.FRAME_HEADER, obu.TILE_GROUP]

    def test_padding_between_the_units_is_left_out(self):
        control = layer()
        padding = obu.Obu(obu.PADDING, bytes(5)).encoded()
        data = control.coded(self.key(control),
                             driver_sequence_header() + padding + FRAME_OBU)
        assert unit_kinds(data) == [obu.TEMPORAL_DELIMITER, obu.SEQUENCE_HEADER, obu.FRAME]

    def test_output_with_no_frame_in_it_is_refused(self):
        control = layer()
        with pytest.raises(EncoderError, match='key frame 0'):
            control.coded(self.key(control), driver_sequence_header())

    def test_empty_output_is_refused(self):
        control = layer()
        with pytest.raises(EncoderError, match='no frame'):
            control.coded(self.key(control), b'')

    def test_output_that_is_not_obus_is_refused(self):
        control = layer()
        with pytest.raises(EncoderError, match='key frame 0'):
            control.coded(self.key(control), b'\x0A\x7F\x00')


def test_the_buffers_are_ctypes_the_encoder_can_hand_to_libva():
    _, buffers, _ = code(layer(), [5, 6, 7])
    for _, payload in buffers:
        assert ctypes.sizeof(payload) > 0
