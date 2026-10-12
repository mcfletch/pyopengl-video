"""HEVC encoding's control layer, with no driver.

The layer is given the configuration attributes a driver answers as a plain
dictionary, and what it returns is read back: the block sizes and tools it
takes from them, the picture structure, the parameter buffers of a key and a
predicted picture, the surfaces each predicts from, and the parameter sets it
keeps from what a driver wrote.
"""
import pytest

from pyopengl_video import EncoderError
from pyopengl_video.mp4 import split_annexb
from pyopengl_video.vaapi import api, hevc_api
from pyopengl_video.vaapi.control import HRD_BUFFER_SECONDS, EncodeSettings
from pyopengl_video.vaapi.h264 import annexb
from pyopengl_video.vaapi.hevc import Blocks, ParameterSets, Tools
from pyopengl_video.vaapi.hevc_encode import (
    MAX_REFERENCES,
    PACKED_HEADERS_WRITTEN,
    HEVCEncoding,
    blocks_from,
    packed_headers_for,
    tools_from,
)
from pyopengl_video.vaapi.hevc_parse import (
    NAL_IDR_N_LP,
    NAL_PPS,
    NAL_SPS,
    NAL_TRAIL_R,
    NAL_VPS,
    SLICE_B,
    SLICE_I,
    SLICE_P,
    parse_nal,
    parse_pps,
    parse_slice_header,
    parse_sps,
)

SIZE = (200, 130)
BITRATE = 1_000_000

ALL_PACKED = 0x1F
#: What Mesa's radeonsi answers for HEVC encoding: every packed header, one
#: reference, 64-sample coding tree blocks, and asymmetric partitions, sample
#: adaptive offset, strong intra smoothing, constrained intra prediction,
#: transform skip, quantiser deltas and a deblocking switch as optional tools.
FEATURES = 0x10541050
BLOCK_SIZES = 0xCF
RADEONSI = {
    api.VAConfigAttribEncPackedHeaders: ALL_PACKED,
    api.VAConfigAttribEncMaxRefFrames: 1,
    hevc_api.VAConfigAttribEncHEVCFeatures: FEATURES,
    hevc_api.VAConfigAttribEncHEVCBlockSizes: BLOCK_SIZES,
}

SEQUENCE = api.VAEncSequenceParameterBufferType
PICTURE = api.VAEncPictureParameterBufferType
SLICE = api.VAEncSliceParameterBufferType
MISC = api.VAEncMiscParameterBufferType
PACKED = api.VAEncPackedHeaderParameterBufferType
PACKED_DATA = api.VAEncPackedHeaderDataBufferType

KEY = [SEQUENCE, MISC, MISC, MISC, PACKED, PACKED_DATA, PICTURE, PACKED, PACKED_DATA,
       PACKED, PACKED_DATA, SLICE]
PREDICTED = [PICTURE, PACKED, PACKED_DATA, SLICE]


def layer(rate_control=api.VA_RC_CQP, gop=4, qp=26, stated=None, size=SIZE, changes=None):
    """A layer for a driver answering as radeonsi does.

    stated -- every attribute the driver answers, in place of radeonsi's
    changes -- attributes answered differently from radeonsi
    """
    settings = EncodeSettings(size=size, frame_rate=(30, 1), bitrate=BITRATE,
                              gop=gop, qp=qp, rate_control=rate_control)
    return HEVCEncoding(settings, {**RADEONSI, **(changes or {})} if stated is None else stated)


def features(**fields):
    value = hevc_api.VAConfigAttribValEncHEVCFeatures()
    for name, support in fields.items():
        setattr(value.bits, name, support)
    return value.value


def kinds(buffers):
    return [kind for kind, _ in buffers]


def one(buffers, kind):
    """The only buffer of `kind`."""
    found = [payload for found_kind, payload in buffers if found_kind == kind]
    assert len(found) == 1
    return found[0]


def packed(buffers, header):
    """The bytes of the packed header of type `header`, escapes already in them."""
    for index, (kind, payload) in enumerate(buffers):
        if kind == PACKED and payload.type == header:
            assert buffers[index + 1][0] == PACKED_DATA
            data = bytes(buffers[index + 1][1])
            assert payload.bit_length == len(data) * 8
            assert payload.has_emulation_bytes == 1
            return data
    raise AssertionError(f'no packed header of type {header}')


def code(control, count, surfaces=(11, 12, 13), forced=()):
    """Decide and build `count` pictures, as the encoder does.

    The reconstruction surfaces are taken in turn, as the encoder takes them.
    Returns each picture with its buffers and its surface.
    """
    found = []
    for index in range(count):
        picture = control.next_picture(force_key=index in forced)
        surface = surfaces[index % len(surfaces)]
        found.append((picture, control.buffers(picture, coded=77, reconstruction=surface),
                      surface))
    return found


def listed(entries):
    """The pictures of a reference array as (surface, order count, flags)."""
    return [(entry.picture_id, entry.pic_order_cnt, entry.flags) for entry in entries
            if not entry.flags & hevc_api.VA_PICTURE_HEVC_INVALID]


BEFORE = hevc_api.VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE


# ------------------------------------------------------------- what it is


class TestWhatItIs:
    def test_the_codec_and_the_profile(self):
        assert HEVCEncoding.codec == 'hevc'
        assert HEVCEncoding.profile == api.VAProfileHEVCMain
        assert HEVCEncoding.title == 'HEVC Main'

    def test_the_quantiser_is_on_the_scale_the_shell_passes(self):
        assert HEVCEncoding.max_qp == 51

    def test_it_asks_about_the_headers_references_tools_and_blocks(self):
        assert set(HEVCEncoding.attributes) == {
            api.VAConfigAttribEncPackedHeaders, api.VAConfigAttribEncMaxRefFrames,
            hevc_api.VAConfigAttribEncHEVCFeatures,
            hevc_api.VAConfigAttribEncHEVCBlockSizes,
            hevc_api.VAConfigAttribPredictionDirection}

    def test_an_odd_size_is_an_encoder_error(self):
        with pytest.raises(EncoderError, match='201x130'):
            layer(size=(201, 130))

    def test_a_group_of_no_pictures_is_an_encoder_error(self):
        with pytest.raises(EncoderError, match='holds none'):
            layer(gop=0)


class TestTheCodedSize:
    """The width is whole coding tree blocks and the height a multiple of 16."""

    @pytest.mark.parametrize('size,coded', [
        ((200, 130), (256, 144)), ((320, 240), (320, 240)), ((1280, 720), (1280, 720)),
        ((1920, 1080), (1920, 1088)), ((162, 90), (192, 96))])
    def test_with_coding_tree_blocks_of_64(self, size, coded):
        control = layer(size=size)
        assert control.coded_size == coded
        assert control.sets.coded_size == coded

    def test_with_the_coding_tree_blocks_of_a_driver_that_states_none(self):
        control = layer(stated={})
        assert control.sets.blocks == Blocks()
        assert control.coded_size == (224, 144)

    def test_a_smallest_coding_block_over_16_is_what_the_height_is_whole(self):
        sizes = hevc_api.VAConfigAttribValEncHEVCBlockSizes()
        sizes.bits.log2_max_coding_tree_block_size_minus3 = 3
        sizes.bits.log2_min_luma_coding_block_size_minus3 = 2
        sizes.bits.log2_max_luma_transform_block_size_minus2 = 3
        control = layer(changes={hevc_api.VAConfigAttribEncHEVCBlockSizes: sizes.value})
        assert control.coded_size == (256, 160)


class TestPackedHeaders:
    def test_the_three_it_writes_are_asked_for(self):
        assert PACKED_HEADERS_WRITTEN == (api.VA_ENC_PACKED_HEADER_SEQUENCE
                                          | api.VA_ENC_PACKED_HEADER_PICTURE
                                          | api.VA_ENC_PACKED_HEADER_SLICE)
        assert layer().packed_headers == PACKED_HEADERS_WRITTEN

    def test_one_the_driver_does_not_take_is_not_asked_for(self):
        offered = api.VA_ENC_PACKED_HEADER_SEQUENCE | api.VA_ENC_PACKED_HEADER_PICTURE
        assert packed_headers_for(offered | api.VA_ENC_PACKED_HEADER_MISC) == offered

    @pytest.mark.parametrize('offered', [
        api.VA_ENC_PACKED_HEADER_SLICE,
        api.VA_ENC_PACKED_HEADER_SLICE | api.VA_ENC_PACKED_HEADER_SEQUENCE,
        api.VA_ENC_PACKED_HEADER_SLICE | api.VA_ENC_PACKED_HEADER_PICTURE])
    def test_a_slice_header_goes_only_with_both_kinds_of_parameter_set(self, offered):
        assert packed_headers_for(offered) == offered & ~api.VA_ENC_PACKED_HEADER_SLICE

    def test_a_driver_that_states_none_is_sent_none(self):
        control = layer(stated={})
        assert control.packed_headers == 0
        [(_, key, _), (_, predicted, _)] = code(control, 2)
        assert kinds(key) == [SEQUENCE, MISC, MISC, MISC, PICTURE, SLICE]
        assert kinds(predicted) == [PICTURE, SLICE]

    def test_a_driver_taking_only_parameter_sets_is_sent_those(self):
        control = layer(changes={api.VAConfigAttribEncPackedHeaders: 0x3})
        [(_, key, _), (_, predicted, _)] = code(control, 2)
        assert kinds(key) == [SEQUENCE, MISC, MISC, MISC, PACKED, PACKED_DATA, PICTURE,
                              PACKED, PACKED_DATA, SLICE]
        assert kinds(predicted) == [PICTURE, SLICE]


# ---------------------------------------------- what the attributes become


class TestBlockSizes:
    def test_radeonsi_codes_64_sample_trees_of_8_sample_blocks(self):
        assert blocks_from(BLOCK_SIZES) == Blocks(
            coding_tree=6, min_coding=3, max_transform=5, min_transform=2,
            depth_inter=0, depth_intra=0)

    def test_a_driver_that_states_none_gets_the_defaults(self):
        assert blocks_from(None) == Blocks()

    def test_the_largest_tree_and_the_deepest_hierarchy_offered_are_used(self):
        sizes = hevc_api.VAConfigAttribValEncHEVCBlockSizes()
        sizes.bits.log2_max_coding_tree_block_size_minus3 = 2
        sizes.bits.log2_min_coding_tree_block_size_minus3 = 1
        sizes.bits.log2_min_luma_coding_block_size_minus3 = 0
        sizes.bits.log2_max_luma_transform_block_size_minus2 = 3
        sizes.bits.log2_min_luma_transform_block_size_minus2 = 0
        sizes.bits.max_max_transform_hierarchy_depth_inter = 2
        sizes.bits.min_max_transform_hierarchy_depth_inter = 1
        sizes.bits.max_max_transform_hierarchy_depth_intra = 3
        assert blocks_from(sizes.value) == Blocks(
            coding_tree=5, min_coding=3, max_transform=5, min_transform=2,
            depth_inter=2, depth_intra=3)

    def test_sizes_no_stream_has_are_an_encoder_error(self):
        sizes = hevc_api.VAConfigAttribValEncHEVCBlockSizes()
        sizes.bits.log2_max_coding_tree_block_size_minus3 = 0
        with pytest.raises(EncoderError, match='coding tree block of 8'):
            blocks_from(sizes.value)

    def test_the_layer_codes_with_them(self):
        control = layer()
        assert control.sets.blocks == blocks_from(BLOCK_SIZES)
        assert control.sets.ctbs == 4 * 3


class TestTools:
    def test_radeonsi_at_a_constant_quantiser(self):
        assert tools_from(FEATURES, rate_controlled=False) == Tools(
            amp=True, sample_adaptive_offset=True, strong_intra_smoothing=True)

    def test_radeonsi_under_rate_control_changes_the_quantiser_within_a_picture(self):
        assert tools_from(FEATURES, rate_controlled=True) == Tools(
            amp=True, sample_adaptive_offset=True, strong_intra_smoothing=True,
            cu_qp_delta=True)

    def test_a_driver_that_states_none_gets_none(self):
        assert tools_from(None, rate_controlled=True) == Tools()

    def test_a_tool_the_driver_never_uses_is_off(self):
        assert tools_from(features(sao=1), rate_controlled=True) == Tools(
            sample_adaptive_offset=True)

    @pytest.mark.parametrize('name,field', [
        ('amp', 'amp'), ('sao', 'sample_adaptive_offset'),
        ('strong_intra_smoothing', 'strong_intra_smoothing'),
        ('sign_data_hiding', 'sign_data_hiding'),
        ('constrained_intra_pred', 'constrained_intra_pred'),
        ('transform_skip', 'transform_skip'), ('cu_qp_delta', 'cu_qp_delta'),
        ('transquant_bypass', 'transquant_bypass')])
    def test_a_tool_the_driver_always_uses_is_declared(self, name, field):
        required = features(**{name: hevc_api.VA_FEATURE_REQUIRED})
        assert tools_from(required, rate_controlled=False) == Tools(**{field: True})

    @pytest.mark.parametrize('name', [
        'sign_data_hiding', 'constrained_intra_pred', 'transform_skip', 'transquant_bypass'])
    def test_a_tool_it_does_not_prefer_stays_off_where_it_is_optional(self, name):
        optional = features(**{name: hevc_api.VA_FEATURE_SUPPORTED})
        assert tools_from(optional, rate_controlled=True) == Tools()

    @pytest.mark.parametrize('name,words', [
        ('separate_colour_planes', 'separately coded colour planes'),
        ('scaling_lists', 'scaling lists'), ('pcm', 'PCM coding blocks'),
        ('temporal_mvp', 'temporal motion vector prediction'),
        ('dependent_slices', 'dependent slice segments'),
        ('weighted_prediction', 'weighted prediction'),
        ('deblocking_filter_disable', 'a disabled deblocking filter')])
    def test_a_tool_with_no_syntax_here_is_refused_where_it_is_always_used(self, name, words):
        with pytest.raises(EncoderError, match=words):
            tools_from(features(**{name: hevc_api.VA_FEATURE_REQUIRED}),
                       rate_controlled=False)

    @pytest.mark.parametrize('name', ['pcm', 'temporal_mvp', 'weighted_prediction'])
    def test_and_left_off_where_it_is_optional(self, name):
        optional = features(**{name: hevc_api.VA_FEATURE_SUPPORTED})
        assert tools_from(optional, rate_controlled=False) == Tools()

    @pytest.mark.parametrize('mode,delta', [
        (api.VA_RC_CQP, False), (api.VA_RC_CBR, True), (api.VA_RC_VBR, True)])
    def test_the_layer_declares_them_by_its_rate_control(self, mode, delta):
        assert layer(rate_control=mode).sets.tools == Tools(
            amp=True, sample_adaptive_offset=True, strong_intra_smoothing=True,
            cu_qp_delta=delta)


class TestReferences:
    def test_radeonsi_keeps_one(self):
        control = layer()
        assert control.references == 1
        assert control.sets.references == 1
        assert not control.bidirectional

    def test_a_driver_that_states_none_keeps_one(self):
        assert layer(stated={}).references == 1

    def test_as_many_as_reference_list_zero_holds(self):
        assert layer(changes={api.VAConfigAttribEncMaxRefFrames: 3 | 2 << 16}).references == 3

    def test_no_more_than_every_level_buffers(self):
        control = layer(changes={api.VAConfigAttribEncMaxRefFrames: 15})
        assert control.references == MAX_REFERENCES == 5

    def test_a_driver_wanting_both_lists_filled_gets_b_pictures(self):
        control = layer(changes={
            hevc_api.VAConfigAttribPredictionDirection: (
                hevc_api.VA_PREDICTION_DIRECTION_PREVIOUS
                | hevc_api.VA_PREDICTION_DIRECTION_BI_NOT_EMPTY),
            api.VAConfigAttribEncMaxRefFrames: 3 | 2 << 16})
        assert control.bidirectional
        assert control.references == 2, 'both lists hold the same pictures'

    def test_a_driver_predicting_from_earlier_pictures_gets_p_pictures(self):
        control = layer(changes={hevc_api.VAConfigAttribPredictionDirection:
                           hevc_api.VA_PREDICTION_DIRECTION_PREVIOUS})
        assert not control.bidirectional


# -------------------------------------------------------- picture structure


class TestPictureStructure:
    def test_a_key_picture_every_gop_pictures(self):
        control = layer(gop=4)
        pictures = [control.next_picture() for _ in range(9)]
        assert [picture.keyframe for picture in pictures] == [
            True, False, False, False, True, False, False, False, True]

    def test_a_key_picture_is_an_idr(self):
        control = layer(gop=4)
        pictures = [control.next_picture() for _ in range(5)]
        assert [picture.detail.idr for picture in pictures] == [
            True, False, False, False, True]

    def test_a_forced_key_picture_starts_a_new_group(self):
        control = layer(gop=4)
        pictures = [control.next_picture(force_key=index == 2) for index in range(8)]
        assert [picture.keyframe for picture in pictures] == [
            True, False, True, False, False, False, True, False]

    def test_the_order_count_restarts_at_a_key_picture(self):
        control = layer(gop=3)
        assert [control.next_picture().detail.poc for _ in range(7)] == [0, 1, 2, 0, 1, 2, 0]

    def test_a_picture_is_named_by_its_order_count(self):
        control = layer(gop=3)
        assert [control.next_picture().name for _ in range(4)] == [
            'picture 0', 'picture 1', 'picture 2', 'picture 0']


# ---------------------------------------------------------------- the buffers


class TestTheBuffersOfAKeyPicture:
    @pytest.fixture
    def key(self):
        [(_, buffers, _)] = code(layer(rate_control=api.VA_RC_CBR), 1)
        return buffers

    def test_they_are_in_the_order_the_driver_reads_them(self, key):
        """The sequence and the packed sets go ahead of the slice header read against them."""
        assert kinds(key) == KEY

    def test_the_packed_sequence_header_is_the_video_and_sequence_sets(self, key):
        made = layer(rate_control=api.VA_RC_CBR).sets
        data = packed(key, hevc_api.VAEncPackedHeaderHEVC_SPS)
        assert list(split_annexb(data)) == [made.vps(), made.sps()]

    def test_the_packed_picture_header_is_the_picture_set(self, key):
        made = layer(rate_control=api.VA_RC_CBR).sets
        assert packed(key, hevc_api.VAEncPackedHeaderHEVC_PPS) == annexb(made.pps())

    def test_the_packed_slice_header_is_an_idr_slice(self, key):
        made = layer(rate_control=api.VA_RC_CBR).sets
        [unit] = split_annexb(packed(key, hevc_api.VAEncPackedHeaderHEVC_Slice))
        header = parse_slice_header(
            parse_nal(unit), {0: parse_sps(parse_nal(made.sps()).rbsp)},
            {0: parse_pps(parse_nal(made.pps()).rbsp)})
        assert header.nal_unit_type == NAL_IDR_N_LP
        assert header.slice_type == SLICE_I

    def test_the_sequence_parameters_say_what_the_sequence_set_says(self, key):
        control = layer(rate_control=api.VA_RC_CBR)
        sps = parse_sps(parse_nal(control.sets.sps()).rbsp)
        sequence = one(key, SEQUENCE)
        assert sequence.general_profile_idc == sps.profile_tier_level.general_profile_idc == 1
        assert sequence.general_level_idc == sps.profile_tier_level.general_level_idc
        assert sequence.general_tier_flag == 0
        assert (sequence.pic_width_in_luma_samples,
                sequence.pic_height_in_luma_samples) == sps.coded_size == (256, 144)
        bits = sequence.seq_fields.bits
        assert bits.chroma_format_idc == 1
        assert (bits.bit_depth_luma_minus8, bits.bit_depth_chroma_minus8) == (0, 0)
        for name in ('amp_enabled_flag', 'sample_adaptive_offset_enabled_flag',
                     'strong_intra_smoothing_enabled_flag', 'pcm_enabled_flag',
                     'sps_temporal_mvp_enabled_flag', 'scaling_list_enabled_flag'):
            assert getattr(bits, name) == getattr(sps, name), name
        for name in ('log2_min_luma_coding_block_size_minus3',
                     'log2_diff_max_min_luma_coding_block_size',
                     'max_transform_hierarchy_depth_inter',
                     'max_transform_hierarchy_depth_intra'):
            assert getattr(sequence, name) == getattr(sps, name), name
        assert (sequence.log2_min_transform_block_size_minus2
                == sps.log2_min_luma_transform_block_size_minus2)
        assert (sequence.log2_diff_max_min_transform_block_size
                == sps.log2_diff_max_min_luma_transform_block_size)

    def test_the_sequence_parameters_state_the_structure_and_the_rate(self, key):
        sequence = one(key, SEQUENCE)
        assert (sequence.intra_period, sequence.intra_idr_period, sequence.ip_period) == (
            4, 4, 1)
        assert sequence.bits_per_second == BITRATE
        assert sequence.seq_fields.bits.low_delay_seq == 1
        assert sequence.vui_parameters_present_flag == 1
        assert sequence.vui_fields.bits.vui_timing_info_present_flag == 1
        assert (sequence.vui_time_scale, sequence.vui_num_units_in_tick) == (30, 1)
        assert sequence.vui_fields.bits.aspect_ratio_info_present_flag == 1
        assert sequence.aspect_ratio_idc == 1

    def test_rate_control_is_told_the_bitrate_the_rate_and_the_buffer(self, key):
        found = {}
        for kind, payload in key:
            if kind == MISC:
                found[payload.type] = payload
        assert set(found) == {api.VAEncMiscParameterTypeRateControl,
                              api.VAEncMiscParameterTypeFrameRate,
                              api.VAEncMiscParameterTypeHRD}
        control = found[api.VAEncMiscParameterTypeRateControl].data
        assert control.bits_per_second == BITRATE
        assert control.target_percentage == 100
        assert (control.initial_qp, control.min_qp, control.max_qp) == (26, 0, 51)
        assert found[api.VAEncMiscParameterTypeHRD].data.buffer_size == (
            BITRATE * HRD_BUFFER_SECONDS)

    def test_the_picture_parameters_are_an_idr_with_no_references(self, key):
        picture = one(key, PICTURE)
        assert (picture.decoded_curr_pic.picture_id,
                picture.decoded_curr_pic.pic_order_cnt,
                picture.decoded_curr_pic.flags) == (11, 0, 0)
        assert listed(picture.reference_frames) == []
        assert all(entry.picture_id == api.VA_INVALID_SURFACE
                   for entry in picture.reference_frames)
        assert picture.coded_buf == 77
        assert picture.nal_unit_type == NAL_IDR_N_LP
        bits = picture.pic_fields.bits
        assert bits.idr_pic_flag == 1
        assert bits.coding_type == hevc_api.CODING_TYPE_I
        assert bits.reference_pic_flag == 1

    def test_the_picture_parameters_say_what_the_picture_set_says(self):
        control = layer(rate_control=api.VA_RC_CBR, qp=31)
        [(_, buffers, _)] = code(control, 1)
        pps = parse_pps(parse_nal(control.sets.pps()).rbsp)
        picture = one(buffers, PICTURE)
        assert picture.pic_init_qp == 31 == pps.init_qp_minus26 + 26
        assert picture.collocated_ref_pic_index == hevc_api.NO_REFERENCE
        assert picture.slice_pic_parameter_set_id == pps.pps_pic_parameter_set_id
        bits = picture.pic_fields.bits
        assert bits.cu_qp_delta_enabled_flag == pps.cu_qp_delta_enabled_flag == 1
        for name in ('diff_cu_qp_delta_depth', 'pps_cb_qp_offset', 'pps_cr_qp_offset',
                     'log2_parallel_merge_level_minus2',
                     'num_ref_idx_l0_default_active_minus1',
                     'num_ref_idx_l1_default_active_minus1', 'num_tile_columns_minus1',
                     'num_tile_rows_minus1'):
            assert getattr(picture, name) == getattr(pps, name), name
        for name in ('dependent_slice_segments_enabled_flag', 'sign_data_hiding_enabled_flag',
                     'constrained_intra_pred_flag', 'transform_skip_enabled_flag',
                     'weighted_pred_flag', 'weighted_bipred_flag',
                     'transquant_bypass_enabled_flag', 'tiles_enabled_flag',
                     'entropy_coding_sync_enabled_flag',
                     'pps_loop_filter_across_slices_enabled_flag'):
            assert getattr(bits, name) == getattr(pps, name), name

    def test_the_slice_is_the_whole_picture_coded_intra(self, key):
        found = one(key, SLICE)
        assert found.slice_segment_address == 0
        assert found.num_ctu_in_slice == 12
        assert found.slice_type == SLICE_I
        assert found.slice_qp_delta == 0
        assert found.max_num_merge_cand == 5
        assert listed(found.ref_pic_list0) == []
        assert listed(found.ref_pic_list1) == []
        bits = found.slice_fields.bits
        assert bits.last_slice_of_pic_flag == 1
        assert (bits.slice_sao_luma_flag, bits.slice_sao_chroma_flag) == (1, 1)
        assert bits.num_ref_idx_active_override_flag == 0
        assert bits.slice_deblocking_filter_disabled_flag == 0
        assert bits.slice_temporal_mvp_enabled_flag == 0

    def test_a_constant_quantiser_is_the_one_the_picture_starts_from(self):
        control = layer(rate_control=api.VA_RC_CQP, qp=18)
        [(_, buffers, _)] = code(control, 1)
        assert one(buffers, PICTURE).pic_init_qp == 18
        assert one(buffers, SLICE).slice_qp_delta == 0
        assert one(buffers, PICTURE).pic_fields.bits.cu_qp_delta_enabled_flag == 0


class TestTheBuffersOfAPredictedPicture:
    @pytest.fixture
    def coded(self):
        control = layer(gop=4)
        return control, code(control, 6)

    def test_it_is_sent_no_sequence_and_no_parameter_sets(self, coded):
        _, pictures = coded
        assert [kinds(buffers) for _, buffers, _ in pictures] == [
            KEY, PREDICTED, PREDICTED, PREDICTED, KEY, PREDICTED]

    def test_it_is_a_p_picture_reconstructed_into_its_own_surface(self, coded):
        _, pictures = coded
        _, buffers, surface = pictures[2]
        picture = one(buffers, PICTURE)
        assert surface == 13
        assert (picture.decoded_curr_pic.picture_id,
                picture.decoded_curr_pic.pic_order_cnt) == (13, 2)
        assert picture.nal_unit_type == NAL_TRAIL_R
        bits = picture.pic_fields.bits
        assert (bits.idr_pic_flag, bits.coding_type, bits.reference_pic_flag) == (
            0, hevc_api.CODING_TYPE_P, 1)

    def test_it_predicts_from_the_surface_of_the_picture_before(self, coded):
        _, pictures = coded
        found = [listed(one(buffers, PICTURE).reference_frames) for _, buffers, _ in pictures]
        assert found == [[], [(11, 0, BEFORE)], [(12, 1, BEFORE)], [(13, 2, BEFORE)],
                         [], [(12, 0, BEFORE)]]

    def test_its_reference_list_is_that_picture(self, coded):
        _, pictures = coded
        for _, buffers, _ in pictures:
            found = one(buffers, SLICE)
            assert listed(found.ref_pic_list0) == listed(
                one(buffers, PICTURE).reference_frames)
            assert listed(found.ref_pic_list1) == []
            assert found.num_ref_idx_l0_active_minus1 == 0

    def test_its_slice_is_a_p_slice_of_the_whole_picture(self, coded):
        _, pictures = coded
        found = one(pictures[1][1], SLICE)
        assert found.slice_type == SLICE_P
        assert found.num_ctu_in_slice == 12
        assert found.max_num_merge_cand == 5
        assert found.slice_fields.bits.last_slice_of_pic_flag == 1

    def test_its_packed_slice_header_states_its_order_count(self, coded):
        control, pictures = coded
        sequences = {0: parse_sps(parse_nal(control.sets.sps()).rbsp)}
        picture_sets = {0: parse_pps(parse_nal(control.sets.pps()).rbsp)}
        found = []
        for _, buffers, _ in pictures:
            [unit] = split_annexb(packed(buffers, hevc_api.VAEncPackedHeaderHEVC_Slice))
            header = parse_slice_header(parse_nal(unit), sequences, picture_sets)
            found.append((header.kind, header.slice_pic_order_cnt_lsb,
                          header.short_term_ref_pic_set.delta_poc_s0))
        assert found == [('I', 0, ()), ('P', 1, (-1,)), ('P', 2, (-1,)), ('P', 3, (-1,)),
                         ('I', 0, ()), ('P', 1, (-1,))]

    def test_it_holds_one_surface_however_long_the_group(self):
        control = layer(gop=1000)
        assert control.held() == set()
        for index in range(40):
            control.buffers(control.next_picture(), coded=77, reconstruction=11 + index % 3)
            assert control.held() == {11 + index % 3}

    def test_a_forced_key_picture_drops_the_references_before_it(self):
        control = layer(gop=10)
        pictures = code(control, 4, forced=(2,))
        found = [listed(one(buffers, PICTURE).reference_frames) for _, buffers, _ in pictures]
        assert found == [[], [(11, 0, BEFORE)], [], [(13, 0, BEFORE)]]
        assert kinds(pictures[2][1]) == KEY


class TestMoreReferences:
    """A driver offering three references, with five surfaces in rotation."""

    @pytest.fixture
    def coded(self):
        control = layer(gop=6, changes={api.VAConfigAttribEncMaxRefFrames: 3})
        return control, code(control, 8, surfaces=(21, 22, 23, 24, 25))

    def test_the_buffer_fills_from_the_key_picture_nearest_first(self, coded):
        _, pictures = coded
        found = [listed(one(buffers, PICTURE).reference_frames) for _, buffers, _ in pictures]
        assert found == [
            [],
            [(21, 0, BEFORE)],
            [(22, 1, BEFORE), (21, 0, BEFORE)],
            [(23, 2, BEFORE), (22, 1, BEFORE), (21, 0, BEFORE)],
            [(24, 3, BEFORE), (23, 2, BEFORE), (22, 1, BEFORE)],
            [(25, 4, BEFORE), (24, 3, BEFORE), (23, 2, BEFORE)],
            [],
            [(22, 0, BEFORE)],
        ]

    def test_no_picture_is_reconstructed_over_one_it_predicts_from(self, coded):
        _, pictures = coded
        for _, buffers, surface in pictures:
            held = [entry[0] for entry in listed(one(buffers, PICTURE).reference_frames)]
            assert surface not in held

    def test_it_holds_the_surfaces_the_next_picture_may_predict_from(self):
        control = layer(gop=6, changes={api.VAConfigAttribEncMaxRefFrames: 3})
        found = []
        for index in range(8):
            picture = control.next_picture()
            control.buffers(picture, coded=77, reconstruction=21 + index % 5)
            found.append(control.held())
        assert found == [{21}, {21, 22}, {21, 22, 23}, {22, 23, 24}, {23, 24, 25},
                         {24, 25, 21}, {22}, {22, 23}]

    def test_the_slice_states_how_many_it_predicts_from(self, coded):
        _, pictures = coded
        found = [(one(buffers, SLICE).num_ref_idx_l0_active_minus1,
                  one(buffers, SLICE).slice_fields.bits.num_ref_idx_active_override_flag)
                 for _, buffers, _ in pictures]
        assert found == [(0, 0), (0, 1), (1, 1), (2, 0), (2, 0), (2, 0), (0, 0), (0, 1)]

    def test_the_picture_set_default_is_the_full_buffer(self, coded):
        _, pictures = coded
        assert one(pictures[0][1], PICTURE).num_ref_idx_l0_default_active_minus1 == 2

    def test_the_packed_slice_header_names_the_same_pictures(self, coded):
        control, pictures = coded
        sequences = {0: parse_sps(parse_nal(control.sets.sps()).rbsp)}
        picture_sets = {0: parse_pps(parse_nal(control.sets.pps()).rbsp)}
        for picture, buffers, _ in pictures:
            [unit] = split_annexb(packed(buffers, hevc_api.VAEncPackedHeaderHEVC_Slice))
            header = parse_slice_header(parse_nal(unit), sequences, picture_sets)
            stated = [picture.detail.poc + delta
                      for delta in header.short_term_ref_pic_set.delta_poc_s0]
            assert stated == [entry[1] for entry in listed(
                one(buffers, PICTURE).reference_frames)]
            if stated:
                assert header.num_ref_idx_l0_active_minus1 == len(stated) - 1


class TestLowDelayBPictures:
    @pytest.fixture
    def coded(self):
        control = layer(gop=6, changes={
            hevc_api.VAConfigAttribPredictionDirection:
                hevc_api.VA_PREDICTION_DIRECTION_BI_NOT_EMPTY,
            api.VAConfigAttribEncMaxRefFrames: 2 | 2 << 16})
        return control, code(control, 4, surfaces=(31, 32, 33, 34))

    def test_a_predicted_picture_is_a_b_picture(self, coded):
        _, pictures = coded
        assert [one(buffers, PICTURE).pic_fields.bits.coding_type
                for _, buffers, _ in pictures] == [
            hevc_api.CODING_TYPE_I, hevc_api.CODING_TYPE_B, hevc_api.CODING_TYPE_B,
            hevc_api.CODING_TYPE_B]
        assert [one(buffers, SLICE).slice_type for _, buffers, _ in pictures] == [
            SLICE_I, SLICE_B, SLICE_B, SLICE_B]

    def test_its_second_list_repeats_its_first(self, coded):
        _, pictures = coded
        for _, buffers, _ in pictures[1:]:
            found = one(buffers, SLICE)
            assert listed(found.ref_pic_list0)
            assert listed(found.ref_pic_list1) == listed(found.ref_pic_list0)
            assert found.num_ref_idx_l1_active_minus1 == found.num_ref_idx_l0_active_minus1

    def test_the_picture_set_default_covers_both_lists(self, coded):
        _, pictures = coded
        picture = one(pictures[0][1], PICTURE)
        assert (picture.num_ref_idx_l0_default_active_minus1,
                picture.num_ref_idx_l1_default_active_minus1) == (1, 1)

    def test_its_packed_slice_header_is_a_b_slice(self, coded):
        control, pictures = coded
        [unit] = split_annexb(packed(pictures[1][1], hevc_api.VAEncPackedHeaderHEVC_Slice))
        header = parse_slice_header(
            parse_nal(unit), {0: parse_sps(parse_nal(control.sets.sps()).rbsp)},
            {0: parse_pps(parse_nal(control.sets.pps()).rbsp)})
        assert header.slice_type == SLICE_B
        assert (header.num_ref_idx_l0_active_minus1, header.num_ref_idx_l1_active_minus1) == (
            0, 0)


# ------------------------------------------------------- what the driver wrote


def stream_sets(size=(200, 130)):
    """Parameter sets as a driver might write its own: other block sizes."""
    return ParameterSets(*size, (30, 1), alignment=(64, 16), blocks=Blocks(
        coding_tree=6, min_coding=3, max_transform=5, min_transform=2,
        depth_inter=4, depth_intra=4))


class TestTheCodedPicture:
    def test_the_packet_is_what_the_driver_wrote(self):
        control = layer()
        picture = control.next_picture()
        data = annexb(*split_annexb(stream_sets().headers()), b'\x28\x01\xaf\x00')
        assert control.coded(picture, data) == data

    def test_the_headers_are_its_own_sets_before_anything_is_coded(self):
        control = layer()
        assert control.headers() == control.sets.headers()
        assert [parse_nal(unit).nal_unit_type for unit in split_annexb(control.headers())] == [
            NAL_VPS, NAL_SPS, NAL_PPS]

    def test_the_headers_are_the_sets_a_key_picture_carries(self):
        control = layer()
        written = stream_sets()
        assert written.headers() != control.sets.headers()
        control.coded(control.next_picture(),
                      annexb(*split_annexb(written.headers()), b'\x28\x01\xaf\x00'))
        assert control.headers() == written.headers()

    def test_a_predicted_picture_changes_nothing(self):
        control = layer()
        control.next_picture()
        control.coded(control.next_picture(), stream_sets().headers())
        assert control.headers() == control.sets.headers()

    def test_a_key_picture_missing_a_set_leaves_the_layers_own(self):
        control = layer()
        written = stream_sets()
        control.coded(control.next_picture(), annexb(written.sps(), written.pps()))
        assert control.headers() == control.sets.headers()

    def test_a_later_key_picture_replaces_them(self):
        control = layer(gop=1)
        first, second = stream_sets(), stream_sets((190, 120))
        control.coded(control.next_picture(), first.headers())
        control.coded(control.next_picture(), second.headers())
        assert control.headers() == second.headers()
