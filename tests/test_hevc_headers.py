"""The HEVC header writer, read back with the HEVC parser.

No driver here. Every parameter set and slice segment header
:mod:`pyopengl_video.vaapi.hevc` writes is parsed by
:mod:`pyopengl_video.vaapi.hevc_parse`, which is checked against a reference
parser in ``tests/test_hevc_parse.py``, and compared with the values put in.
"""
import itertools

import pytest

from pyopengl_video.mp4 import split_annexb
from pyopengl_video.vaapi.hevc import (
    LEVELS,
    Blocks,
    GroupOfPictures,
    ParameterSets,
    Picture,
    Tools,
    level_for,
    nal_unit,
)
from pyopengl_video.vaapi.hevc_parse import (
    NAL_IDR_N_LP,
    NAL_PPS,
    NAL_SPS,
    NAL_TRAIL_R,
    NAL_VPS,
    PROFILE_MAIN,
    PROFILE_MAIN_10,
    SLICE_B,
    SLICE_I,
    SLICE_P,
    ShortTermRefPicSet,
    parse_nal,
    parse_pps,
    parse_slice_header,
    parse_sps,
    parse_vps,
)

SIZE = (320, 240)
RATE = (30, 1)

IDR = Picture(idr=True, poc=0)


def sets(size=SIZE, rate=RATE, **settings):
    return ParameterSets(*size, rate, **settings)


def vps_of(made):
    return parse_vps(parse_nal(made.vps()).rbsp)


def sps_of(made):
    return parse_sps(parse_nal(made.sps()).rbsp)


def pps_of(made):
    return parse_pps(parse_nal(made.pps()).rbsp)


def header_of(made, picture, qp=None):
    """The slice segment header of `picture`, read against the sets of `made`."""
    return parse_slice_header(parse_nal(made.slice_header(picture, qp)),
                              {0: sps_of(made)}, {0: pps_of(made)})


def predicted(poc, count=1, bidirectional=False):
    return Picture(idr=False, poc=poc, bidirectional=bidirectional,
                   references=tuple(poc - 1 - index for index in range(count)))


# ------------------------------------------------------------------ NAL units


class TestNalUnits:
    @pytest.mark.parametrize('write,kind', [
        (ParameterSets.vps, NAL_VPS), (ParameterSets.sps, NAL_SPS),
        (ParameterSets.pps, NAL_PPS)])
    def test_each_set_is_a_unit_of_its_type_in_the_base_layer(self, write, kind):
        unit = parse_nal(write(sets()))
        assert (unit.nal_unit_type, unit.nuh_layer_id, unit.temporal_id) == (kind, 0, 0)

    def test_a_unit_is_its_header_then_the_escaped_payload(self):
        assert nal_unit(b'\x00\x00\x01\x80', NAL_SPS) == b'\x42\x01\x00\x00\x03\x01\x80'

    def test_the_headers_are_the_three_sets_in_stream_order(self):
        made = sets()
        assert list(split_annexb(made.headers())) == [made.vps(), made.sps(), made.pps()]


# ---------------------------------------------------------- profile and level


class TestProfileTierLevel:
    @pytest.mark.parametrize('read', [vps_of, sps_of])
    def test_it_is_main_profile_main_tier(self, read):
        profile = read(sets()).profile_tier_level
        assert profile.general_profile_space == 0
        assert profile.general_tier_flag == 0
        assert profile.general_profile_idc == PROFILE_MAIN
        assert profile.name == 'Main'

    def test_a_main_stream_also_conforms_to_main_10(self):
        profile = sps_of(sets()).profile_tier_level
        assert profile.compatible(PROFILE_MAIN)
        assert profile.compatible(PROFILE_MAIN_10)
        assert profile.general_profile_compatibility_flags == 0x60000000

    def test_the_source_is_progressive_frames(self):
        profile = sps_of(sets()).profile_tier_level
        assert profile.general_progressive_source_flag == 1
        assert profile.general_interlaced_source_flag == 0
        assert profile.general_frame_only_constraint_flag == 1

    @pytest.mark.parametrize('read', [vps_of, sps_of])
    def test_both_sets_state_the_level_chosen(self, read):
        made = sets((1280, 720))
        assert made.level_idc == 93
        assert read(made).profile_tier_level.general_level_idc == 93


class TestLevelChoice:
    """Tables A.8 and A.9: picture size, sample rate and the Main tier's bitrate."""

    @pytest.mark.parametrize('size,rate,level', [
        ((192, 96), (30, 1), 1.0),
        ((320, 240), (30, 1), 2.0),
        ((640, 368), (30, 1), 2.1),
        ((960, 544), (30, 1), 3.0),
        ((1280, 720), (30, 1), 3.1),
        ((1920, 1088), (30, 1), 4.0),
        ((1920, 1088), (60, 1), 4.1),
        ((3840, 2160), (30, 1), 5.0),
        ((3840, 2160), (60, 1), 5.1),
        ((3840, 2160), (120, 1), 5.2),
        ((7680, 4320), (30, 1), 6.0),
        ((7680, 4320), (60, 1), 6.1),
        ((7680, 4320), (120, 1), 6.2),
    ])
    def test_the_lowest_level_covering_the_size_and_rate(self, size, rate, level):
        assert level_for(*size, rate) == round(level * 30)

    def test_a_sample_rate_at_the_limit_is_covered(self):
        assert 192 * 96 * 30 == 552960
        assert level_for(192, 96, (30, 1)) == 30
        assert level_for(192, 96, (30000, 1000)) == 30
        assert level_for(192, 96, (31, 1)) == 60

    def test_a_fractional_rate_is_compared_exactly(self):
        assert level_for(1920, 1088, (30000, 1001)) == 120
        assert level_for(1920, 1088, (32, 1)) == 120
        assert level_for(1920, 1088, (33, 1)) == 123

    @pytest.mark.parametrize('bitrate,level', [
        (0, 2.0), (1_500_000, 2.0), (1_500_001, 2.1), (4_000_000, 3.0),
        (10_000_000, 3.1), (50_000_000, 5.2), (240_000_000, 6.2)])
    def test_the_bitrate_raises_the_level(self, bitrate, level):
        assert level_for(320, 240, (30, 1), bitrate) == round(level * 30)

    def test_a_long_side_raises_the_level(self):
        """Each side is at most sqrt(8 * MaxLumaPs): 4096 needs level 4's 4222."""
        assert level_for(4096, 64, (30, 1)) == 120
        assert level_for(64, 4096, (30, 1)) == 120

    def test_nothing_covering_the_stream_gives_the_highest_level(self):
        assert level_for(16384, 8192, (120, 1)) == LEVELS[-1][0] == 186
        assert level_for(320, 240, (30, 1), 10**9) == 186

    def test_the_level_is_chosen_from_the_coded_size(self):
        """A picture of 192x90 fits level 1 and its coded 192x128 does not."""
        made = sets((192, 90), alignment=(64, 64))
        assert level_for(192, 90, RATE) == 30
        assert made.coded_size == (192, 128)
        assert made.level_idc == 60

    def test_the_levels_rise(self):
        for lower, higher in itertools.pairwise(LEVELS):
            assert all(low <= high for low, high in zip(lower, higher, strict=True))


# ------------------------------------------------------------------- the VPS


class TestVideoParameterSet:
    def test_it_is_one_layer_and_one_sub_layer(self):
        vps = vps_of(sets())
        assert vps.vps_video_parameter_set_id == 0
        assert vps.vps_max_layers_minus1 == 0
        assert vps.vps_max_sub_layers_minus1 == 0
        assert vps.vps_temporal_id_nesting_flag == 1

    @pytest.mark.parametrize('references', [1, 2, 4])
    def test_the_buffer_holds_the_references_and_the_picture(self, references):
        vps = vps_of(sets(references=references))
        assert vps.vps_max_dec_pic_buffering_minus1 == (references,)
        assert vps.vps_max_num_reorder_pics == (0,)
        assert vps.vps_max_latency_increase_plus1 == (0,)

    @pytest.mark.parametrize('rate', [(30, 1), (30000, 1001), (25, 1), (60, 1)])
    def test_it_states_the_frame_rate(self, rate):
        vps = vps_of(sets(rate=rate))
        assert vps.vps_timing_info_present_flag == 1
        assert (vps.vps_time_scale, vps.vps_num_units_in_tick) == rate
        assert vps.frame_rate == rate


# ------------------------------------------------------------------- the SPS


class TestSequenceParameterSet:
    def test_it_is_eight_bit_420(self):
        sps = sps_of(sets())
        assert sps.sps_video_parameter_set_id == 0
        assert sps.sps_seq_parameter_set_id == 0
        assert sps.sps_max_sub_layers_minus1 == 0
        assert sps.sps_temporal_id_nesting_flag == 1
        assert sps.chroma_format_idc == 1
        assert sps.bit_depth == 8
        assert sps.bit_depth_chroma_minus8 == 0

    def test_it_states_the_size(self):
        sps = sps_of(sets((640, 368)))
        assert sps.coded_size == (640, 368)
        assert sps.size == (640, 368)
        assert sps.conformance_window == (0, 0, 0, 0)

    @pytest.mark.parametrize('references', [1, 3])
    def test_no_picture_is_held_back(self, references):
        sps = sps_of(sets(references=references))
        assert sps.dpb_size == references + 1
        assert sps.reorder_pics == 0
        assert sps.max_latency_pictures is None

    def test_the_order_count_field_is_the_width_the_headers_use(self):
        made = sets()
        assert sps_of(made).max_pic_order_cnt_lsb == 1 << made.log2_max_poc_lsb == 256

    @pytest.mark.parametrize('blocks', [
        Blocks(),
        Blocks(coding_tree=6, min_coding=3, max_transform=5, min_transform=2,
               depth_inter=0, depth_intra=0),
        Blocks(coding_tree=4, min_coding=4, max_transform=4, min_transform=3,
               depth_inter=1, depth_intra=1),
        Blocks(coding_tree=6, min_coding=4, max_transform=5, min_transform=2,
               depth_inter=4, depth_intra=3),
    ])
    def test_it_states_the_block_sizes(self, blocks):
        sps = sps_of(sets((640, 384), blocks=blocks))
        assert sps.ctb_log2_size == blocks.coding_tree
        assert sps.log2_min_luma_coding_block_size_minus3 + 3 == blocks.min_coding
        assert sps.log2_min_luma_transform_block_size_minus2 + 2 == blocks.min_transform
        assert (sps.log2_min_luma_transform_block_size_minus2 + 2
                + sps.log2_diff_max_min_luma_transform_block_size) == blocks.max_transform
        assert sps.max_transform_hierarchy_depth_inter == blocks.depth_inter
        assert sps.max_transform_hierarchy_depth_intra == blocks.depth_intra

    @pytest.mark.parametrize('tool,field', [
        ('amp', 'amp_enabled_flag'),
        ('sample_adaptive_offset', 'sample_adaptive_offset_enabled_flag'),
        ('strong_intra_smoothing', 'strong_intra_smoothing_enabled_flag'),
    ])
    def test_each_sequence_tool_is_declared_when_it_is_on(self, tool, field):
        assert getattr(sps_of(sets()), field) == 0
        on = sps_of(sets(tools=Tools(**{tool: True})))
        assert getattr(on, field) == 1
        assert sum((on.amp_enabled_flag, on.sample_adaptive_offset_enabled_flag,
                    on.strong_intra_smoothing_enabled_flag)) == 1

    def test_the_tools_never_used_are_off(self):
        sps = sps_of(sets(tools=Tools(amp=True, sample_adaptive_offset=True,
                                      strong_intra_smoothing=True)))
        assert sps.scaling_list_enabled_flag == 0
        assert sps.pcm_enabled_flag == 0
        assert sps.long_term_ref_pics_present_flag == 0
        assert sps.sps_temporal_mvp_enabled_flag == 0
        assert sps.sps_range_extension_flag == 0
        assert sps.sps_scc_extension_flag == 0

    def test_one_reference_is_one_set_of_the_picture_before(self):
        assert sps_of(sets()).short_term_ref_pic_sets == (
            ShortTermRefPicSet(delta_poc_s0=(-1,), used_by_curr_pic_s0=(1,)),)

    def test_more_references_are_a_set_for_each_count_of_pictures(self):
        assert sps_of(sets(references=3)).short_term_ref_pic_sets == (
            ShortTermRefPicSet(delta_poc_s0=(-1,), used_by_curr_pic_s0=(1,)),
            ShortTermRefPicSet(delta_poc_s0=(-1, -2), used_by_curr_pic_s0=(1, 1)),
            ShortTermRefPicSet(delta_poc_s0=(-1, -2, -3), used_by_curr_pic_s0=(1, 1, 1)),
        )


class TestVideoUsability:
    def test_the_colour_is_limited_range_bt709(self):
        vui = sps_of(sets()).vui
        assert vui.video_full_range_flag == 0
        assert (vui.colour_primaries, vui.transfer_characteristics, vui.matrix_coeffs) == (
            1, 1, 1)
        assert vui.video_format == 5

    def test_samples_are_square(self):
        vui = sps_of(sets()).vui
        assert vui.aspect_ratio_idc == 1
        assert vui.sar == (1, 1)

    @pytest.mark.parametrize('rate', [(30, 1), (30000, 1001), (24000, 1001), (120, 1)])
    def test_it_states_the_frame_rate(self, rate):
        sps = sps_of(sets(rate=rate))
        assert (sps.vui.vui_time_scale, sps.vui.vui_num_units_in_tick) == rate
        assert sps.frame_rate == rate

    def test_it_is_frames_with_no_buffering_model(self):
        vui = sps_of(sets()).vui
        assert vui.field_seq_flag == 0
        assert vui.frame_field_info_present_flag == 0
        assert vui.default_display_window == (0, 0, 0, 0)
        assert vui.vui_hrd_parameters_present_flag == 0
        assert vui.bitstream_restriction_flag == 0


class TestCropping:
    """A size that is not whole blocks is coded larger and cropped back."""

    def test_the_coded_size_is_whole_smallest_coding_blocks(self):
        made = sets((322, 242))
        assert made.coded_size == (328, 248)
        assert (made.crop_right, made.crop_bottom) == (3, 3)

    @pytest.mark.parametrize('size,alignment,coded', [
        ((200, 130), (64, 16), (256, 144)),
        ((1920, 1080), (64, 16), (1920, 1088)),
        ((1280, 720), (64, 16), (1280, 720)),
        ((162, 90), (64, 16), (192, 96)),
        ((162, 90), (16, 16), (176, 96)),
        ((2, 2), (64, 64), (64, 64)),
    ])
    def test_the_window_gives_back_the_picture(self, size, alignment, coded):
        made = sets(size, alignment=alignment)
        sps = sps_of(made)
        assert made.coded_size == coded
        assert sps.coded_size == coded
        assert sps.visible == (0, 0, *size)
        assert sps.size == size

    def test_the_offsets_are_in_chroma_samples_from_the_right_and_bottom(self):
        sps = sps_of(sets((200, 130), alignment=(64, 16)))
        assert sps.conformance_window == (0, 28, 0, 7)

    def test_the_picture_is_counted_in_coding_tree_blocks(self):
        made = sets((200, 130), alignment=(64, 16), blocks=Blocks(coding_tree=6))
        sps = sps_of(made)
        assert (made.width_in_ctbs, made.height_in_ctbs) == (4, 3)
        assert (sps.width_in_ctbs, sps.height_in_ctbs) == (4, 3)
        assert made.ctbs == 12

    def test_a_last_row_of_part_blocks_is_counted(self):
        made = sets((320, 240), blocks=Blocks(coding_tree=6))
        assert (made.width_in_ctbs, made.height_in_ctbs) == (5, 4)

    @pytest.mark.parametrize('size', [(321, 240), (320, 241)])
    def test_an_odd_dimension_is_refused(self, size):
        with pytest.raises(ValueError, match='odd dimension'):
            sets(size)

    @pytest.mark.parametrize('size', [(0, 240), (320, -2)])
    def test_no_picture_is_refused(self, size):
        with pytest.raises(ValueError, match='not a picture size'):
            sets(size)

    def test_a_coded_size_off_the_coding_blocks_is_refused(self):
        with pytest.raises(ValueError, match='whole coding blocks of 8'):
            sets(alignment=(64, 12))


class TestBlocks:
    @pytest.mark.parametrize('changes,message', [
        ({'coding_tree': 7}, 'coding tree block of 128'),
        ({'coding_tree': 3, 'min_coding': 3}, 'coding tree block of 8'),
        ({'min_coding': 6}, 'smallest coding block of 64'),
        ({'min_transform': 3}, 'transform blocks of 8 to 32'),
        ({'coding_tree': 4, 'min_coding': 3, 'max_transform': 5}, 'transform blocks of 4 to 32'),
        ({'depth_inter': 4}, 'depths of 4 and 2; 3 is the most'),
    ])
    def test_sizes_no_main_profile_stream_has_are_refused(self, changes, message):
        with pytest.raises(ValueError, match=message):
            Blocks(**changes)


# ------------------------------------------------------------------- the PPS


class TestPictureParameterSet:
    def test_it_names_the_sequence(self):
        pps = pps_of(sets())
        assert pps.pps_pic_parameter_set_id == 0
        assert pps.pps_seq_parameter_set_id == 0

    @pytest.mark.parametrize('qp', [0, 20, 26, 51])
    def test_it_starts_from_the_quantiser_asked_for(self, qp):
        assert pps_of(sets(init_qp=qp)).init_qp_minus26 == qp - 26

    @pytest.mark.parametrize('qp', [-1, 52])
    def test_a_quantiser_off_the_scale_is_refused(self, qp):
        with pytest.raises(ValueError, match='quantiser range'):
            sets(init_qp=qp)

    @pytest.mark.parametrize('references', [1, 3])
    def test_the_default_reference_count_is_the_buffers(self, references):
        pps = pps_of(sets(references=references))
        assert pps.num_ref_idx_l0_default_active_minus1 == references - 1
        assert pps.num_ref_idx_l1_default_active_minus1 == 0

    def test_b_pictures_have_the_same_count_in_both_lists(self):
        pps = pps_of(sets(references=3, bidirectional=True))
        assert pps.num_ref_idx_l0_default_active_minus1 == 2
        assert pps.num_ref_idx_l1_default_active_minus1 == 2

    @pytest.mark.parametrize('references', [0, 16])
    def test_a_reference_count_no_buffer_holds_is_refused(self, references):
        with pytest.raises(ValueError, match='reference pictures'):
            sets(references=references)

    @pytest.mark.parametrize('tool,field', [
        ('sign_data_hiding', 'sign_data_hiding_enabled_flag'),
        ('constrained_intra_pred', 'constrained_intra_pred_flag'),
        ('transform_skip', 'transform_skip_enabled_flag'),
        ('cu_qp_delta', 'cu_qp_delta_enabled_flag'),
        ('transquant_bypass', 'transquant_bypass_enabled_flag'),
    ])
    def test_each_picture_tool_is_declared_when_it_is_on(self, tool, field):
        fields = ('sign_data_hiding_enabled_flag', 'constrained_intra_pred_flag',
                  'transform_skip_enabled_flag', 'cu_qp_delta_enabled_flag',
                  'transquant_bypass_enabled_flag')
        off = pps_of(sets())
        on = pps_of(sets(tools=Tools(**{tool: True})))
        assert [getattr(off, name) for name in fields] == [0] * 5
        assert [name for name in fields if getattr(on, name)] == [field]

    def test_the_quantiser_changes_once_a_coding_tree_block(self):
        pps = pps_of(sets(tools=Tools(cu_qp_delta=True)))
        assert pps.diff_cu_qp_delta_depth == 0

    def test_the_syntax_no_slice_header_here_writes_is_off(self):
        pps = pps_of(sets(tools=Tools(cu_qp_delta=True, sign_data_hiding=True)))
        for field in (
                'dependent_slice_segments_enabled_flag', 'output_flag_present_flag',
                'num_extra_slice_header_bits', 'cabac_init_present_flag',
                'pps_slice_chroma_qp_offsets_present_flag', 'weighted_pred_flag',
                'weighted_bipred_flag', 'tiles_enabled_flag',
                'entropy_coding_sync_enabled_flag',
                'pps_loop_filter_across_slices_enabled_flag',
                'deblocking_filter_control_present_flag',
                'pps_deblocking_filter_disabled_flag',
                'pps_scaling_list_data_present_flag', 'lists_modification_present_flag',
                'slice_segment_header_extension_present_flag', 'pps_range_extension_flag',
                'pps_cb_qp_offset', 'pps_cr_qp_offset', 'log2_parallel_merge_level_minus2'):
            assert getattr(pps, field) == 0, field


# ------------------------------------------------------ slice segment headers


class TestTheSliceHeaderOfAnIDRPicture:
    def test_it_is_an_intra_slice_of_an_idr_with_no_leading_pictures(self):
        header = header_of(sets(), IDR)
        assert header.nal_unit_type == NAL_IDR_N_LP
        assert header.temporal_id == 0
        assert header.idr
        assert header.slice_type == SLICE_I

    def test_it_is_the_whole_picture(self):
        header = header_of(sets(), IDR)
        assert header.first_slice_segment_in_pic_flag == 1
        assert header.dependent_slice_segment_flag == 0
        assert header.slice_segment_address == 0
        assert header.slice_pic_parameter_set_id == 0

    def test_it_keeps_what_was_decoded_before_it(self):
        assert header_of(sets(), IDR).no_output_of_prior_pics_flag == 0

    def test_it_predicts_from_nothing(self):
        header = header_of(sets(references=3), IDR)
        assert header.slice_pic_order_cnt_lsb == 0
        assert header.short_term_ref_pic_set == ShortTermRefPicSet()
        assert header.num_pic_total_curr == 0


class TestTheSliceHeaderOfAPredictedPicture:
    def test_it_is_a_p_slice_of_a_trailing_reference_picture(self):
        header = header_of(sets(), predicted(1))
        assert header.nal_unit_type == NAL_TRAIL_R
        assert not header.sub_layer_non_reference
        assert header.slice_type == SLICE_P

    @pytest.mark.parametrize('poc', [1, 2, 9, 255, 256, 257, 1000])
    def test_it_states_its_order_count_modulo_the_field(self, poc):
        assert header_of(sets(), predicted(poc)).slice_pic_order_cnt_lsb == poc % 256

    def test_it_names_the_sequences_set_of_the_picture_before(self):
        header = header_of(sets(), predicted(5))
        assert header.short_term_ref_pic_set_sps_flag == 1
        assert header.st_rps_bits == 0
        assert header.short_term_ref_pic_set.delta_poc_s0 == (-1,)
        assert header.num_ref_idx_active_override_flag == 0
        assert header.num_ref_idx_l0_active_minus1 == 0

    @pytest.mark.parametrize('count', [1, 2, 3])
    def test_it_names_the_set_as_long_as_its_references(self, count):
        header = header_of(sets(references=3), predicted(count, count))
        assert header.short_term_ref_pic_set_idx == count - 1
        assert header.short_term_ref_pic_set.delta_poc_s0 == tuple(
            -1 - index for index in range(count))
        assert header.num_pic_total_curr == count
        assert header.num_ref_idx_l0_active_minus1 == count - 1

    @pytest.mark.parametrize('count,override', [(1, 1), (2, 1), (3, 0)])
    def test_fewer_references_than_the_buffer_override_the_default(self, count, override):
        header = header_of(sets(references=3), predicted(count, count))
        assert header.num_ref_idx_active_override_flag == override

    def test_it_chooses_from_five_merging_candidates(self):
        assert header_of(sets(), predicted(1)).five_minus_max_num_merge_cand == 0

    def test_it_has_no_motion_prediction_from_another_picture(self):
        header = header_of(sets(), predicted(1))
        assert header.slice_temporal_mvp_enabled_flag == 0
        assert header.collocated_ref_idx is None
        assert header.pred_weight_table is None
        assert header.list_entry_l0 == ()


class TestTheSliceHeaderOfALowDelayBPicture:
    def test_it_is_a_b_slice_with_both_lists_as_long(self):
        made = sets(references=3, bidirectional=True)
        header = header_of(made, predicted(2, 2, bidirectional=True))
        assert header.slice_type == SLICE_B
        assert header.num_ref_idx_active_override_flag == 1
        assert header.num_ref_idx_l0_active_minus1 == 1
        assert header.num_ref_idx_l1_active_minus1 == 1
        assert header.mvd_l1_zero_flag == 0

    def test_a_full_buffer_takes_both_counts_from_the_picture_set(self):
        made = sets(references=2, bidirectional=True)
        header = header_of(made, predicted(7, 2, bidirectional=True))
        assert header.num_ref_idx_active_override_flag == 0
        assert header.num_ref_idx_l0_active_minus1 == 1
        assert header.num_ref_idx_l1_active_minus1 == 1


class TestWhatEverySliceHeaderStates:
    PICTURES = (IDR, predicted(1), predicted(3, bidirectional=True))

    @pytest.mark.parametrize('picture', PICTURES)
    @pytest.mark.parametrize('on', [False, True])
    def test_the_slice_filters_as_the_sequence_says(self, picture, on):
        made = sets(tools=Tools(sample_adaptive_offset=on),
                    bidirectional=picture.bidirectional)
        header = header_of(made, picture)
        assert (header.slice_sao_luma_flag, header.slice_sao_chroma_flag) == (on, on)
        assert header.slice_deblocking_filter_disabled_flag == 0

    @pytest.mark.parametrize('picture', PICTURES)
    @pytest.mark.parametrize('init,qp,delta', [(26, None, 0), (26, 30, 4), (40, 22, -18)])
    def test_the_quantiser_is_a_step_from_the_picture_sets(self, picture, init, qp, delta):
        made = sets(init_qp=init, bidirectional=picture.bidirectional)
        assert header_of(made, picture, qp).slice_qp_delta == delta

    @pytest.mark.parametrize('picture', PICTURES)
    @pytest.mark.parametrize('tools', [Tools(), Tools(sample_adaptive_offset=True)])
    def test_the_header_ends_on_a_byte_with_its_alignment(self, picture, tools):
        """The parser raises where the alignment bit is not a one."""
        made = sets(tools=tools, bidirectional=picture.bidirectional)
        unit = parse_nal(made.slice_header(picture))
        header = header_of(made, picture)
        assert header.header_bytes == 2 + len(unit.rbsp)
        assert unit.rbsp[-1] != 0, 'the alignment bit is in the last byte'


# --------------------------------------------------------------- packed forms


class TestPackedHeaders:
    def test_the_sequence_header_is_the_video_set_then_the_sequence_set(self):
        made = sets()
        data, bits = made.packed_sequence()
        assert data.startswith(b'\x00\x00\x00\x01')
        assert list(split_annexb(data)) == [made.vps(), made.sps()]
        assert bits == len(data) * 8

    def test_the_picture_header_is_the_picture_set(self):
        made = sets()
        data, bits = made.packed_picture()
        assert list(split_annexb(data)) == [made.pps()]
        assert bits == len(data) * 8

    @pytest.mark.parametrize('picture', [IDR, predicted(1), predicted(300)])
    def test_the_slice_header_is_one_unit_every_bit_of_it_syntax(self, picture):
        made = sets()
        data, bits = made.packed_slice_header(picture)
        assert data.startswith(b'\x00\x00\x00\x01')
        assert list(split_annexb(data)) == [made.slice_header(picture)]
        assert bits == len(data) * 8

    def test_a_slice_quantiser_reaches_the_packed_header(self):
        made = sets(init_qp=26)
        data, _ = made.packed_slice_header(predicted(1), qp=31)
        [unit] = split_annexb(data)
        assert parse_slice_header(parse_nal(unit), {0: sps_of(made)},
                                  {0: pps_of(made)}).slice_qp_delta == 5


class TestNoStartCodeInsideAUnit:
    """Two zero bytes are never followed by a byte below three within a unit."""

    FORBIDDEN = (b'\x00\x00\x00', b'\x00\x00\x01', b'\x00\x00\x02')

    def units(self):
        for size, rate, qp in itertools.product(
                [(64, 64), (200, 130), (320, 240), (1920, 1080), (4096, 2160)],
                [(1, 1), (30, 1), (30000, 1001), (65536, 1), (1, 65536)],
                [0, 26, 51]):
            made = sets(size, rate, init_qp=qp, references=2, alignment=(64, 16),
                        tools=Tools(sample_adaptive_offset=True, cu_qp_delta=True))
            yield from (made.vps(), made.sps(), made.pps())
            for poc in (0, 1, 2, 255, 256, 511, 512, 1024):
                picture = IDR if not poc else predicted(poc, min(poc, 2))
                yield made.slice_header(picture)
                yield made.slice_header(picture, qp=51 - qp)

    def test_no_unit_holds_one(self):
        for unit in self.units():
            for pattern in self.FORBIDDEN:
                assert pattern not in unit, unit.hex()

    def test_the_timing_fields_are_what_needs_escaping(self):
        """A 32-bit tick count of one is three zero bytes without the escapes."""
        escaped = [unit for unit in self.units() if b'\x00\x00\x03' in unit]
        assert escaped
        made = sets(rate=(30, 1))
        assert b'\x00\x00\x03' in made.vps()
        assert vps_of(made).vps_num_units_in_tick == 1

    def test_a_packed_header_holds_only_its_own_start_codes(self):
        made = sets(rate=(1, 1))
        data, _ = made.packed_sequence()
        assert data.count(b'\x00\x00\x01') == 2
        data, _ = made.packed_slice_header(predicted(256))
        assert data.count(b'\x00\x00\x01') == 1


# ------------------------------------------------------- the group of pictures


class TestGroupOfPictures:
    def run(self, count, **settings):
        structure = GroupOfPictures(**settings)
        return [structure.next_picture() for _ in range(count)]

    def test_a_recording_opens_with_an_idr(self):
        first = GroupOfPictures(gop=30).next_picture()
        assert first == Picture(idr=True, poc=0)
        assert first.intra
        assert first.nal_unit_type == NAL_IDR_N_LP
        assert first.slice_type == SLICE_I

    def test_an_idr_follows_every_gop_pictures(self):
        pictures = self.run(10, gop=4)
        assert [picture.idr for picture in pictures] == [
            True, False, False, False, True, False, False, False, True, False]

    def test_the_order_count_restarts_at_each_idr(self):
        assert [picture.poc for picture in self.run(10, gop=4)] == [
            0, 1, 2, 3, 0, 1, 2, 3, 0, 1]

    def test_each_picture_predicts_from_the_one_before(self):
        pictures = self.run(9, gop=4)
        assert [picture.references for picture in pictures] == [
            (), (0,), (1,), (2,), (), (0,), (1,), (2,), ()]
        assert all(picture.slice_type == SLICE_P for picture in pictures if not picture.idr)
        assert all(picture.nal_unit_type == NAL_TRAIL_R
                   for picture in pictures if not picture.idr)

    def test_more_references_fill_from_the_idr(self):
        pictures = self.run(6, gop=6, references=3)
        assert [picture.references for picture in pictures] == [
            (), (0,), (1, 0), (2, 1, 0), (3, 2, 1), (4, 3, 2)]

    def test_nothing_predicts_across_an_idr(self):
        pictures = self.run(8, gop=4, references=3)
        assert pictures[5].references == (0,)
        assert pictures[6].references == (1, 0)

    def test_a_forced_idr_starts_a_new_group(self):
        structure = GroupOfPictures(gop=4)
        pictures = [structure.next_picture(force_idr=index == 2) for index in range(8)]
        assert [picture.idr for picture in pictures] == [
            True, False, True, False, False, False, True, False]
        assert [picture.poc for picture in pictures] == [0, 1, 0, 1, 2, 3, 0, 1]

    def test_a_gop_of_one_is_all_idr(self):
        assert all(picture.idr for picture in self.run(5, gop=1))

    def test_the_order_count_runs_past_the_width_of_its_field(self):
        pictures = self.run(600, gop=1000)
        assert pictures[-1].poc == 599
        assert pictures[-1].references == (598,)

    def test_predicted_pictures_can_be_b_pictures(self):
        pictures = self.run(3, gop=3, bidirectional=True)
        assert [picture.slice_type for picture in pictures] == [SLICE_I, SLICE_B, SLICE_B]

    @pytest.mark.parametrize('settings,message', [
        ({'gop': 0}, 'holds none'), ({'gop': 4, 'references': 0}, 'nothing to predict from')])
    def test_a_structure_with_no_pictures_or_no_references_is_refused(self, settings, message):
        with pytest.raises(ValueError, match=message):
            GroupOfPictures(**settings)


# ------------------------------------------- what the writers will not write


class TestRefusedArguments:
    """A writer packs what it is given into the width the syntax has for it,
    so an argument the syntax cannot state is refused before anything is
    written, with :class:`ValueError` as for a size."""

    @pytest.mark.parametrize('rate', [(30, 0), (0, 1), (-30, 1), (30, -1), (29.97, 1)])
    def test_a_frame_rate_that_is_no_rate_is_refused(self, rate):
        with pytest.raises(ValueError, match='frame rate'):
            sets(rate=rate)

    @pytest.mark.parametrize('rate', [(2**32, 1), (1, 2**32)])
    def test_a_frame_rate_past_the_timing_fields_is_refused(self, rate):
        with pytest.raises(ValueError, match='32-bit'):
            sets(rate=rate)

    def test_the_fastest_rate_the_fields_hold_is_written(self):
        assert sps_of(sets(rate=(2**32 - 1, 1))).vui.vui_time_scale == 2**32 - 1

    @pytest.mark.parametrize('alignment', [(0, 0), (-8, 8), (8, -8)])
    def test_an_alignment_that_is_no_size_is_refused(self, alignment):
        with pytest.raises(ValueError, match='coded size in multiples'):
            sets(alignment=alignment)

    @pytest.mark.parametrize('size', [(10**7, 64), (64, 16890), (8192, 8192)])
    def test_a_picture_larger_than_any_level_codes_is_refused(self, size):
        with pytest.raises(ValueError, match='larger than any level codes'):
            sets(size=size)

    def test_a_negative_bitrate_is_refused(self):
        with pytest.raises(ValueError, match='bitrate'):
            sets(bitrate=-1)

    @pytest.mark.parametrize('qp', [-300, -1, 52, 200])
    def test_a_slice_quantiser_off_the_scale_is_refused(self, qp):
        with pytest.raises(ValueError, match='quantiser range'):
            sets().slice_header(IDR, qp=qp)

    @pytest.mark.parametrize('qp', [0, 51])
    def test_the_ends_of_the_quantiser_scale_are_written(self, qp):
        assert header_of(sets(init_qp=26), IDR, qp).slice_qp_delta == qp - 26

    def test_a_picture_the_sequences_sets_do_not_describe_is_refused(self):
        made = sets(references=2)
        with pytest.raises(ValueError, match='IDR picture predicting from 1'):
            made.slice_header(Picture(idr=True, poc=0, references=(1,)))
        with pytest.raises(ValueError, match='predicts from 0 pictures'):
            made.slice_header(Picture(idr=False, poc=3))
        with pytest.raises(ValueError, match='predicts from 3 pictures'):
            made.slice_header(predicted(3, count=3))

    @pytest.mark.parametrize(('kind', 'temporal_id', 'named'), [
        (64, 0, 'nal_unit_type 64'), (70, 0, 'nal_unit_type 70'),
        (-1, 0, 'nal_unit_type -1'), (1, 7, 'temporal_id 7'), (1, 9, 'temporal_id 9'),
        (1, -1, 'temporal_id -1')])
    def test_a_nal_header_field_past_its_bits_is_refused(self, kind, temporal_id, named):
        """A type of 70 is not written as type 6, and a temporal_id of 9 does
        not carry into the layer id."""
        with pytest.raises(ValueError, match=named):
            nal_unit(b'\x80', kind, temporal_id)

    def test_the_largest_nal_header_fields_are_written(self):
        unit = parse_nal(nal_unit(b'\x80', 63, 6))
        assert (unit.nal_unit_type, unit.nuh_layer_id, unit.temporal_id) == (63, 0, 6)
