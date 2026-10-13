"""The HEVC parser, against the reference parser and against written syntax.

``tests/hevc_reference.py`` holds what ``ffmpeg -bsf:v trace_headers`` reads
from each libx265 fixture, and every parameter set and slice segment header of
those streams is compared with it. libx265 codes its reference picture sets in
the slice header and emits no tiles, long-term pictures, dependent slice
segments or list modifications, so those are read from syntax written by
``tests/hevc_syntax.py``.
"""
from pathlib import Path

import pytest

from pyopengl_video.mp4 import split_annexb
from pyopengl_video.vaapi.h264 import StreamError
from pyopengl_video.vaapi.hevc_parse import (
    DEFAULT_8X8_INTER,
    DEFAULT_8X8_INTRA,
    DEFAULT_SCALING_LISTS,
    NAL_CRA,
    NAL_IDR_N_LP,
    NAL_IDR_W_RADL,
    NAL_PPS,
    NAL_RASL_N,
    NAL_SPS,
    NAL_TSA_N,
    NAL_VPS,
    SLICE_B,
    SLICE_I,
    SLICE_P,
    LongTermPicture,
    ShortTermRefPicSet,
    parse_nal,
    parse_pps,
    parse_slice_header,
    parse_sps,
    parse_vps,
)
from tests.hevc_reference import FIXTURES, SLICE_COLUMNS
from tests.hevc_syntax import (
    EIGHT,
    FOUR,
    SCALING,
    SETS,
    header,
    nal,
    picture_set,
    pps_bytes,
    sequence,
    slice_rbsp,
    slice_unit,
    sps_bytes,
    vps_bytes,
)

DATA = Path(__file__).parent / 'data' / 'decode'


def parsed(name):
    """Every parameter set and slice segment header of a fixture, in order."""
    found = {'vps': [], 'sps': [], 'pps': [], 'slices': []}
    sequences, pictures = {}, {}
    independent = None
    for data in split_annexb((DATA / f'{name}.h265').read_bytes()):
        unit = parse_nal(data)
        if unit.nal_unit_type == NAL_VPS:
            found['vps'].append(parse_vps(unit.rbsp))
        elif unit.nal_unit_type == NAL_SPS:
            sps = parse_sps(unit.rbsp)
            sequences[sps.sps_seq_parameter_set_id] = sps
            found['sps'].append(sps)
        elif unit.nal_unit_type == NAL_PPS:
            pps = parse_pps(unit.rbsp)
            pictures[pps.pps_pic_parameter_set_id] = pps
            found['pps'].append(pps)
        elif unit.is_slice:
            independent = parse_slice_header(unit, sequences, pictures, independent)
            found['slices'].append(independent)
    return found


@pytest.fixture(scope='module', params=sorted(FIXTURES))
def stream(request):
    return FIXTURES[request.param], parsed(request.param)


class TestAgainstTheReferenceParser:
    def test_the_sequence_parameter_set(self, stream):
        reference, found = stream
        sps = found['sps'][0]
        expected = dict(reference['sps'])
        assert sps.conformance_window == expected.pop('conformance_window')
        assert (sps.sps_max_dec_pic_buffering_minus1[-1], sps.sps_max_num_reorder_pics[-1],
                sps.sps_max_latency_increase_plus1[-1]) == expected.pop('ordering')
        assert (sps.vui.vui_num_units_in_tick, sps.vui.vui_time_scale) == expected.pop('timing')
        full_range, primaries, transfer, matrix = expected.pop('colour')
        assert sps.vui.video_full_range_flag == full_range
        assert (sps.vui.colour_primaries, sps.vui.transfer_characteristics,
                sps.vui.matrix_coeffs) == (primaries or 2, transfer or 2, matrix or 2)
        profile = sps.profile_tier_level
        assert profile.general_profile_idc == expected.pop('general_profile_idc')
        assert profile.general_level_idc == expected.pop('general_level_idc')
        assert {name: getattr(sps, name) for name in expected} == expected

    def test_the_picture_parameter_set(self, stream):
        reference, found = stream
        pps = found['pps'][0]
        expected = reference['pps']
        assert {name: getattr(pps, name) for name in expected} == expected

    def test_the_video_parameter_set_states_no_rate(self, stream):
        reference, found = stream
        assert reference['vps_timing'] == (None, None)
        assert found['vps'][0].frame_rate is None

    def test_every_slice_segment_header(self, stream):
        reference, found = stream
        rows = []
        for slice_header in found['slices']:
            rps = slice_header.short_term_ref_pic_set
            inter = slice_header.slice_type != SLICE_I and slice_header.num_ref_idx_active_override_flag
            rows.append((
                slice_header.nal_unit_type, slice_header.temporal_id,
                slice_header.slice_segment_address, slice_header.slice_type,
                slice_header.slice_pic_order_cnt_lsb, rps.num_negative_pics,
                rps.num_positive_pics, slice_header.st_rps_bits,
                slice_header.num_ref_idx_l0_active_minus1 if inter else -1,
                slice_header.num_ref_idx_l1_active_minus1
                if inter and slice_header.slice_type == SLICE_B else -1,
                slice_header.slice_qp_delta, slice_header.num_entry_point_offsets,
                slice_header.header_bytes))
        assert len(SLICE_COLUMNS) == len(rows[0])
        assert tuple(rows) == reference['slices']


class TestFixtureDetails:
    """Single headers of the fixtures, field by field as the reference reads them."""

    def test_a_bi_predicted_slice_with_explicit_weights(self):
        slice_header = parsed('hevc_bpyramid')['slices'][7]
        assert slice_header.kind == 'B'
        assert slice_header.slice_pic_order_cnt_lsb == 5
        rps = slice_header.short_term_ref_pic_set
        assert rps == ShortTermRefPicSet((-1, -3), (1, 1), (1, 3), (1, 1))
        assert slice_header.num_pic_total_curr == 4
        assert slice_header.slice_temporal_mvp_enabled_flag == 1
        assert (slice_header.slice_sao_luma_flag, slice_header.slice_sao_chroma_flag) == (1, 1)
        assert (slice_header.collocated_from_l0_flag, slice_header.collocated_ref_idx) == (0, 0)
        table = slice_header.pred_weight_table
        assert (table.luma_log2_weight_denom, table.delta_chroma_log2_weight_denom) == (6, 0)
        assert table.delta_luma_weight_l0 == (17, 0)
        assert table.luma_offset_l0 == (-5, 0)
        assert table.delta_chroma_weight_l0 == ((16, 16), (0, 0))
        assert table.delta_luma_weight_l1 == (-10, 0)
        assert table.luma_offset_l1 == (2, 0)
        assert table.delta_chroma_weight_l1 == ((-11, -11), (0, 0))
        assert slice_header.five_minus_max_num_merge_cand == 2
        assert slice_header.slice_loop_filter_across_slices_enabled_flag == 0

    def test_chroma_offsets_are_the_derived_ones(self):
        """Equation 7-56: a weight of 80/64 with no coded offset moves the
        offset by 128 - 128 * 80 / 64."""
        table = parsed('hevc_bpyramid')['slices'][7].pred_weight_table
        assert table.chroma_offset_l0 == ((-32, -32), (0, 0))
        assert table.chroma_offset_l1 == ((22, 22), (0, 0))

    def test_a_slice_with_an_entry_point(self):
        slice_header = parsed('hevc_slices_wpp')['slices'][4]
        assert slice_header.first_slice_segment_in_pic_flag == 0
        assert slice_header.slice_segment_address == 20
        assert slice_header.entry_point_offset_minus1 == (305,)
        assert slice_header.num_ref_idx_active_override_flag == 0
        table = slice_header.pred_weight_table
        assert (table.luma_log2_weight_denom, table.delta_chroma_log2_weight_denom) == (7, -1)

    def test_the_pps_values_stand_where_a_slice_overrides_nothing(self):
        found = parsed('hevc_slices_wpp')
        pps, slice_header = found['pps'][0], found['slices'][4]
        assert (pps.pps_beta_offset_div2, pps.pps_tc_offset_div2) == (-1, 1)
        assert slice_header.deblocking_filter_override_flag == 0
        assert (slice_header.slice_beta_offset_div2, slice_header.slice_tc_offset_div2) == (-1, 1)

    def test_coded_scaling_lists(self):
        """The fixture's lists are 11 + n + 2 * row + 3 * column in raster order."""
        lists = parsed('hevc_scaling_lists')['sps'][0].scaling_lists
        assert lists.lists[0][0] == (16,) * 16
        # Up-right diagonal order: (0, 0), then (row 1, column 0), (0, 1), (2, 0).
        assert lists.lists[0][1][:4] == (13, 15, 16, 17)
        assert lists.lists[0][2] == lists.lists[0][1]
        assert lists.lists[1][0][:3] == (16, 18, 19)
        assert lists.lists[1][0][-1] == 16 + 2 * 7 + 3 * 7
        assert lists.lists[1][5] == lists.lists[1][4]
        assert lists.dc == ((18, 19, 19, 20, 21, 21), (22, 23))
        assert lists.lists[3][1][0] == 11 + 14

    def test_default_scaling_lists_are_coded_as_none(self):
        sps = parsed('hevc_scaling_default')['sps'][0]
        assert sps.scaling_list_enabled_flag == 1
        assert sps.scaling_lists is None

    def test_sub_layers(self):
        found = parsed('hevc_slices_wpp')
        sps = found['sps'][0]
        assert sps.sps_max_sub_layers_minus1 == 1
        assert len(sps.sps_max_dec_pic_buffering_minus1) == 2
        kinds = {slice_header.nal_unit_type: slice_header.temporal_id
                 for slice_header in found['slices']}
        assert kinds[NAL_TSA_N] == 1

    def test_hrd_parameters_are_read_past(self):
        sps = parsed('hevc_scaling_lists')['sps'][0]
        assert sps.vui.vui_hrd_parameters_present_flag == 1
        assert sps.sps_temporal_mvp_enabled_flag == 1

    def test_ten_bits_and_its_colour(self):
        sps = parsed('hevc_main10')['sps'][0]
        assert sps.bit_depth == 10
        assert sps.profile_tier_level.name == 'Main 10'
        assert sps.vui.chroma_sample_loc_type_top_field == 2


class TestDerivedValues:
    def test_the_visible_size_is_the_cropped_one(self):
        sps = parsed('hevc_bpyramid')['sps'][0]
        assert sps.coded_size == (120, 72)
        assert sps.visible == (0, 0, 120, 68)
        assert sps.size == (120, 68)

    def test_the_frame_rate(self):
        assert parsed('hevc_bpyramid')['sps'][0].frame_rate == (30, 1)
        assert sequence(vui={'timing': (1001, 60000)}).frame_rate == (60000, 1001)
        assert sequence().frame_rate is None

    def test_the_vps_rate(self):
        assert parse_vps(vps_bytes(timing=(1, 25))).frame_rate == (25, 1)

    def test_the_coding_tree_grid(self):
        sps = sequence(width=160, height=96, diff_cb=2)
        assert sps.ctb_log2_size == 5
        assert (sps.width_in_ctbs, sps.height_in_ctbs) == (5, 3)

    def test_the_buffer_sizes_are_the_highest_sub_layers(self):
        sps = sequence(sub_layers_minus1=2, ordering=((2, 0, 0), (3, 1, 2), (5, 2, 4)))
        assert sps.dpb_size == 6
        assert sps.reorder_pics == 2
        assert sps.max_latency_pictures == 5

    def test_one_ordering_entry_stands_for_every_sub_layer(self):
        sps = sequence(sub_layers_minus1=2, ordering=((5, 2, 0),))
        assert sps.sps_max_dec_pic_buffering_minus1 == (5, 5, 5)
        assert sps.max_latency_pictures is None

    def test_profile_compatibility(self):
        profile = sequence(profile_idc=2).profile_tier_level
        assert profile.compatible(2)
        assert not profile.compatible(1)

    def test_a_4_2_2_window_is_in_its_own_chroma_units(self):
        sps = sequence(chroma_format_idc=2, conformance_window=(1, 1, 2, 2))
        assert sps.visible == (2, 2, 60, 60)


class TestParameterSets:
    def test_sub_layer_profiles_are_read_past(self):
        sps = sequence(sub_layers_minus1=2, sub_layers=((1, 1), (0, 1)),
                       ordering=((2, 0, 0), (3, 1, 2), (5, 2, 4)), width=160, height=96)
        assert sps.coded_size == (160, 96)

    def test_the_vui(self):
        sps = sequence(vui={'sar': (4, 3), 'colour': (9, 16, 9), 'full_range': 1,
                            'chroma_loc': 2, 'display_window': (1, 2, 3, 4),
                            'timing': (1001, 60000), 'hrd': 1, 'restriction': 1},
                       strong_smoothing=1)
        vui = sps.vui
        assert (vui.aspect_ratio_idc, vui.sar) == (255, (4, 3))
        assert vui.video_full_range_flag == 1
        assert (vui.colour_primaries, vui.transfer_characteristics, vui.matrix_coeffs) == (9, 16, 9)
        assert vui.chroma_sample_loc_type_top_field == 2
        assert vui.default_display_window == (1, 2, 3, 4)
        assert vui.vui_hrd_parameters_present_flag == 1
        assert vui.bitstream_restriction_flag == 1
        assert sps.strong_intra_smoothing_enabled_flag == 1

    def test_pcm(self):
        sps = sequence(pcm=(7, 5, 0, 1, 1), sao=1)
        assert sps.pcm_enabled_flag == 1
        assert sps.pcm_sample_bit_depth_luma_minus1 == 7
        assert sps.pcm_sample_bit_depth_chroma_minus1 == 5
        assert sps.log2_diff_max_min_pcm_luma_coding_block_size == 1
        assert sps.pcm_loop_filter_disabled_flag == 1
        assert sps.sample_adaptive_offset_enabled_flag == 1

    def test_long_term_pictures_of_the_sequence(self):
        sps = sequence(long_term=((3, 1), (40, 0), (77, 1)), temporal_mvp=1)
        assert sps.long_term_ref_pics_present_flag == 1
        assert sps.lt_ref_pic_poc_lsb_sps == (3, 40, 77)
        assert sps.used_by_curr_pic_lt_sps_flag == (1, 0, 1)
        assert sps.sps_temporal_mvp_enabled_flag == 1

    def test_range_extension_flags_are_named(self):
        sps = sequence(chroma_format_idc=2, profile_idc=4,
                       range_extension=(1, 0, 1, 0, 0, 1, 0, 1, 1))
        assert sps.sps_range_extension_flag == 1
        assert sps.range_extension == (
            'transform_skip_rotation_enabled_flag', 'implicit_rdpcm_enabled_flag',
            'intra_smoothing_disabled_flag', 'persistent_rice_adaptation_enabled_flag',
            'cabac_bypass_alignment_enabled_flag')
        assert sps.profile_tier_level.name == 'Format Range Extensions'

    def test_the_other_sequence_extensions_are_reported(self):
        sps = sequence(extension_flags=(0, 0, 1))
        assert sps.sps_scc_extension_flag == 1
        assert sps.sps_range_extension_flag == 0

    def test_a_stated_tile_grid(self):
        sps = sequence(width=160, height=96, diff_cb=2)
        pps = picture_set(tiles={'columns': 3, 'rows': 2, 'widths': (0, 1), 'heights': (1,),
                                 'loop_filter': 0})
        assert pps.tiles_enabled_flag == 1
        assert (pps.num_tile_columns_minus1, pps.num_tile_rows_minus1) == (2, 1)
        assert pps.uniform_spacing_flag == 0
        assert pps.loop_filter_across_tiles_enabled_flag == 0
        assert pps.tile_columns(sps) == (1, 2, 2)
        assert pps.tile_rows(sps) == (2, 1)

    def test_a_uniform_tile_grid(self):
        """Equation 6-3: seven coding tree blocks in three columns are 2, 2, 3."""
        sps = sequence(width=112, height=64)
        pps = picture_set(tiles={'columns': 3, 'rows': 2})
        assert pps.tile_columns(sps) == (2, 2, 3)
        assert pps.tile_rows(sps) == (2, 2)

    def test_no_tiles_is_one_tile(self):
        sps = sequence(width=112, height=64)
        pps = picture_set()
        assert pps.tile_columns(sps) == (7,)
        assert pps.loop_filter_across_tiles_enabled_flag == 1

    def test_tiles_wider_than_the_picture_are_refused(self):
        sps = sequence(width=64, height=64)
        pps = picture_set(tiles={'columns': 3, 'rows': 1, 'widths': (1, 1), 'heights': ()})
        with pytest.raises(StreamError, match='tiles'):
            pps.tile_columns(sps)

    def test_the_picture_set_fields(self):
        pps = picture_set(
            pic_parameter_set_id=3, seq_parameter_set_id=1, dependent_slices=1,
            output_flag_present=1, extra_bits=2, sign_hiding=1, cabac_init_present=1,
            refs=(2, 1), init_qp=-5, cu_qp_delta_depth=2, chroma_offsets=(3, -4),
            slice_chroma_offsets_present=1, weighted_pred=1, weighted_bipred=1, wpp=1,
            loop_filter_across_slices=1, deblocking={'override': 1, 'beta': 2, 'tc': -3},
            lists_modification=1, parallel_merge=2, header_extension=1)
        assert (pps.pps_pic_parameter_set_id, pps.pps_seq_parameter_set_id) == (3, 1)
        assert pps.dependent_slice_segments_enabled_flag == 1
        assert pps.output_flag_present_flag == 1
        assert pps.num_extra_slice_header_bits == 2
        assert pps.num_ref_idx_l0_default_active_minus1 == 2
        assert pps.num_ref_idx_l1_default_active_minus1 == 1
        assert pps.init_qp_minus26 == -5
        assert pps.diff_cu_qp_delta_depth == 2
        assert (pps.pps_cb_qp_offset, pps.pps_cr_qp_offset) == (3, -4)
        assert pps.entropy_coding_sync_enabled_flag == 1
        assert pps.deblocking_filter_override_enabled_flag == 1
        assert (pps.pps_beta_offset_div2, pps.pps_tc_offset_div2) == (2, -3)
        assert pps.lists_modification_present_flag == 1
        assert pps.log2_parallel_merge_level_minus2 == 2
        assert pps.slice_segment_header_extension_present_flag == 1

    def test_the_picture_range_extension(self):
        pps = picture_set(transform_skip=1, range_extension={
            'offset_list': [(1, -1), (2, -2)], 'sao_scale': (1, 2)})
        assert pps.pps_range_extension_flag == 1
        assert pps.log2_max_transform_skip_block_size_minus2 == 1
        assert pps.chroma_qp_offset_list_enabled_flag == 1
        assert (pps.log2_sao_offset_scale_luma, pps.log2_sao_offset_scale_chroma) == (1, 2)


class TestScalingListData:
    @pytest.fixture(params=['sps', 'pps'])
    def lists(self, request):
        if request.param == 'sps':
            return sequence(scaling=SCALING, amp=1).scaling_lists
        return picture_set(scaling=SCALING, parallel_merge=1).scaling_lists

    def test_coded_lists(self, lists):
        assert lists.lists[0][0] == FOUR
        assert lists.lists[0][3] == FOUR[::-1]
        assert lists.lists[1][2] == EIGHT[::-1]
        assert lists.lists[3][0] == EIGHT

    def test_a_delta_of_zero_is_the_default_list(self, lists):
        assert lists.lists[0][1] == (16,) * 16
        assert lists.lists[0][5] == (16,) * 16
        assert lists.lists[1][3] == DEFAULT_8X8_INTER
        assert lists.lists[2][2] == DEFAULT_8X8_INTRA
        assert lists.dc[0][2] == 16

    def test_a_delta_names_an_earlier_list(self, lists):
        assert lists.lists[0][2] == lists.lists[0][1]
        assert lists.lists[0][4] == lists.lists[0][2]
        assert lists.lists[1][5] == EIGHT[::-1]
        assert lists.lists[2][4] == EIGHT
        assert lists.lists[3][1] == EIGHT

    def test_a_copied_list_takes_the_dc_factor_too(self, lists):
        assert lists.dc == ((40, 40, 16, 50, 40, 40), (60, 60))

    def test_what_follows_the_lists_is_still_read(self):
        assert sequence(scaling=SCALING, amp=1).amp_enabled_flag == 1
        assert picture_set(scaling=SCALING, parallel_merge=1).log2_parallel_merge_level_minus2 == 1

    def test_a_list_predicted_from_before_the_first_is_refused(self):
        broken = [[2, 0, 0, 0, 0, 0], *SCALING[1:]]
        with pytest.raises(StreamError, match='scaling_list_pred_matrix_id_delta is 2'):
            sequence(scaling=broken)

    def test_the_defaults(self):
        """Table 7-6, by its corners and the sum of each matrix."""
        assert DEFAULT_8X8_INTRA[:3] == (16, 16, 16)
        assert DEFAULT_8X8_INTRA[-3:] == (88, 88, 115)
        assert DEFAULT_8X8_INTER[-3:] == (71, 71, 91)
        assert (sum(DEFAULT_8X8_INTRA), sum(DEFAULT_8X8_INTER)) == (1939, 1752)
        assert DEFAULT_SCALING_LISTS.lists[3] == (DEFAULT_8X8_INTRA, DEFAULT_8X8_INTER)


class TestReferencePictureSets:
    def test_a_set_coded_outright(self):
        sps = sequence(st_rps=SETS)
        assert sps.short_term_ref_pic_sets[0] == ShortTermRefPicSet(
            (-1, -3, -7), (1, 1, 0), (1, 4), (1, 1))
        assert sps.short_term_ref_pic_sets[0].used == 4
        assert sps.short_term_ref_pic_sets[3] == ShortTermRefPicSet((-2,), (1,))

    def test_a_set_predicted_with_a_negative_shift(self):
        """Every distance moves by -1: the shift itself becomes a picture, 0
        is no picture, and the entry not to be carried over is dropped."""
        sps = sequence(st_rps=SETS)
        assert sps.short_term_ref_pic_sets[1] == ShortTermRefPicSet(
            (-1, -2, -4), (1, 1, 1), (3,), (1,))

    def test_a_set_predicted_with_a_positive_shift(self):
        sps = sequence(st_rps=SETS)
        assert sps.short_term_ref_pic_sets[2] == ShortTermRefPicSet(
            (-2,), (1,), (1, 2), (1, 1))

    def test_what_follows_the_sets_is_still_read(self):
        assert sequence(st_rps=SETS, temporal_mvp=1).sps_temporal_mvp_enabled_flag == 1

    def test_a_slice_names_a_set_of_the_sequence(self):
        sps, pps = sequence(st_rps=SETS), picture_set()
        slice_header = header(sps, pps, kind='P', poc_lsb=9, rps=2)
        assert slice_header.short_term_ref_pic_set_sps_flag == 1
        assert slice_header.short_term_ref_pic_set_idx == 2
        assert slice_header.short_term_ref_pic_set == sps.short_term_ref_pic_sets[2]
        assert slice_header.st_rps_bits == 0

    def test_a_slice_predicts_its_set_from_any_of_the_sequences(self):
        """delta_idx_minus1 of 2 counts back three from the slice's own index."""
        sps, pps = sequence(st_rps=SETS), picture_set()
        slice_header = header(sps, pps, kind='P', poc_lsb=9, rps={
            'delta_idx_minus1': 2, 'delta_rps': 1, 'used': [1, 1, 0, 1, 1],
            'use_delta': [1, 1, 1, 1, 1]})
        assert slice_header.short_term_ref_pic_set == ShortTermRefPicSet(
            (-1, -3), (1, 0), (1, 4), (1, 1))
        # The flag, delta_idx_minus1, the sign, the size, then seven flags.
        assert slice_header.st_rps_bits == 1 + 3 + 1 + 1 + 6
        assert slice_header.num_pic_total_curr == 3

    def test_a_set_predicted_from_before_the_first_is_refused(self):
        sps, pps = sequence(st_rps=SETS[:1]), picture_set()
        with pytest.raises(StreamError, match='delta_idx_minus1 is 3, outside 0 to 0'):
            header(sps, pps, kind='P', rps={
                'delta_idx_minus1': 3, 'delta_rps': 1, 'used': [1], 'use_delta': [1]})

    def test_an_idr_slice_has_no_set(self):
        sps, pps = sequence(st_rps=SETS), picture_set()
        slice_header = header(sps, pps, nal_type=NAL_IDR_W_RADL, kind='I')
        assert slice_header.idr
        assert slice_header.short_term_ref_pic_set == ShortTermRefPicSet()
        assert slice_header.slice_pic_order_cnt_lsb == 0


class TestSliceSegmentHeaders:
    def sets(self, **pps_options):
        sps = sequence(width=160, height=96, diff_cb=2, sao=1, st_rps=SETS,
                       ordering=((10, 2, 0),),
                       long_term=((3, 1), (40, 0), (77, 1)), temporal_mvp=1)
        return sps, picture_set(**pps_options)

    def test_long_term_pictures(self):
        sps, pps = self.sets()
        slice_header = header(sps, pps, kind='P', poc_lsb=4, rps=0, long_term={
            'sps': [(2, None), (0, 3)], 'pics': [(9, 1, 1), (11, 0, 2), (12, 1, None)]})
        assert slice_header.long_term == (
            LongTermPicture(77, 1, 0, 0), LongTermPicture(3, 1, 1, 3),
            LongTermPicture(9, 1, 1, 1), LongTermPicture(11, 0, 1, 3),
            LongTermPicture(12, 1, 0, 3))
        assert slice_header.num_pic_total_curr == 4 + 4

    def test_list_modifications(self):
        sps, pps = self.sets(lists_modification=1, refs=(2, 1))
        slice_header = header(sps, pps, kind='B', poc_lsb=4, rps=0,
                              modification=((2, 0, 1), (3, 1)))
        assert slice_header.list_entry_l0 == (2, 0, 1)
        assert slice_header.list_entry_l1 == (3, 1)
        unmodified = header(sps, pps, kind='B', poc_lsb=4, rps=0)
        assert (unmodified.list_entry_l0, unmodified.list_entry_l1) == ((), ())

    def test_one_reference_leaves_nothing_to_modify(self):
        sps, pps = self.sets(lists_modification=1)
        slice_header = header(sps, pps, kind='P', poc_lsb=4, rps=3, qp_delta=-9)
        assert slice_header.list_entry_l0 == ()
        assert slice_header.slice_qp_delta == -9

    def test_active_reference_counts(self):
        sps, pps = self.sets(refs=(2, 1))
        default = header(sps, pps, kind='B', poc_lsb=4, rps=0)
        assert (default.num_ref_idx_l0_active_minus1,
                default.num_ref_idx_l1_active_minus1) == (2, 1)
        predicted = header(sps, pps, kind='P', poc_lsb=4, rps=0)
        assert (predicted.num_ref_idx_l0_active_minus1,
                predicted.num_ref_idx_l1_active_minus1) == (2, 0)
        overridden = header(sps, pps, kind='B', poc_lsb=4, rps=0, refs=(3, 0))
        assert overridden.num_ref_idx_active_override_flag == 1
        assert (overridden.num_ref_idx_l0_active_minus1,
                overridden.num_ref_idx_l1_active_minus1) == (3, 0)
        intra = header(sps, pps, kind='I', poc_lsb=4, rps=0)
        assert (intra.num_ref_idx_l0_active_minus1, intra.num_ref_idx_l1_active_minus1) == (0, 0)

    def test_the_collocated_picture(self):
        sps, pps = self.sets(refs=(2, 1))
        from_l1 = header(sps, pps, kind='B', poc_lsb=4, rps=0, temporal_mvp=1,
                         collocated=(0, 1), mvd_l1_zero=1)
        assert (from_l1.collocated_from_l0_flag, from_l1.collocated_ref_idx) == (0, 1)
        assert from_l1.mvd_l1_zero_flag == 1
        from_l0 = header(sps, pps, kind='P', poc_lsb=4, rps=0, temporal_mvp=1,
                         collocated=(1, 2))
        assert (from_l0.collocated_from_l0_flag, from_l0.collocated_ref_idx) == (1, 2)
        single = header(sps, pps, kind='P', poc_lsb=4, rps=0, temporal_mvp=1, refs=(0, 0))
        assert single.collocated_ref_idx == 0
        without = header(sps, pps, kind='P', poc_lsb=4, rps=0)
        assert without.collocated_ref_idx is None

    def test_weights(self):
        sps, pps = self.sets(weighted_pred=1)
        slice_header = header(sps, pps, kind='P', poc_lsb=4, rps=0, refs=(3, 0), weights={
            'denom': 5, 'chroma_delta': -1, 'l0': [
                ((3, -4), ((2, 5), (-6, 7))), (None, None), ((1, 1), None),
                (None, ((100, -300), (-100, 300)))]})
        table = slice_header.pred_weight_table
        assert (table.luma_log2_weight_denom, table.delta_chroma_log2_weight_denom) == (5, -1)
        assert table.delta_luma_weight_l0 == (3, 0, 1, 0)
        assert table.luma_offset_l0 == (-4, 0, 1, 0)
        assert table.delta_chroma_weight_l0 == ((2, -6), (0, 0), (0, 0), (100, -100))
        # 128 + 5 - (128 * 18 >> 4) and 128 + 7 - (128 * 10 >> 4), then the clip.
        assert table.chroma_offset_l0 == ((-11, 55), (0, 0), (0, 0), (-128, 127))
        assert table.delta_luma_weight_l1 == ()

    def test_weights_are_coded_only_where_the_pps_says(self):
        sps, pps = self.sets(weighted_pred=1)
        assert header(sps, pps, kind='B', poc_lsb=4, rps=0).pred_weight_table is None
        sps, pps = self.sets(weighted_bipred=1)
        assert header(sps, pps, kind='P', poc_lsb=4, rps=0).pred_weight_table is None
        both = header(sps, pps, kind='B', poc_lsb=4, rps=0, weights={
            'denom': 7, 'l1': [((-5, 6), ((1, 2), (3, 4)))]})
        assert both.pred_weight_table.delta_luma_weight_l1 == (-5,)
        assert both.pred_weight_table.luma_offset_l1 == (6,)

    def test_a_denominator_out_of_range_is_refused(self):
        sps, pps = self.sets(weighted_pred=1)
        unit = slice_unit(sps, pps, kind='P', poc_lsb=4, rps=0,
                          weights={'denom': 7, 'chroma_delta': 1})
        with pytest.raises(StreamError, match='delta_chroma_log2_weight_denom is 1'):
            parse_slice_header(unit, {0: sps}, {0: pps})

    def test_the_fields_a_pps_switches_on(self):
        sps, pps = self.sets(output_flag_present=1, extra_bits=2, cabac_init_present=1,
                             slice_chroma_offsets_present=1, loop_filter_across_slices=1,
                             deblocking={'override': 1, 'beta': 2, 'tc': -3})
        slice_header = header(
            sps, pps, kind='P', poc_lsb=4, rps=0, pic_output=0, sao=(1, 0), cabac_init=1,
            merge=3, qp_delta=-7, chroma_offsets=(1, -2), deblocking={'beta': -1, 'tc': 4},
            loop_filter_across_slices=0)
        assert slice_header.pic_output_flag == 0
        assert (slice_header.slice_sao_luma_flag, slice_header.slice_sao_chroma_flag) == (1, 0)
        assert slice_header.cabac_init_flag == 1
        assert slice_header.five_minus_max_num_merge_cand == 3
        assert slice_header.slice_qp_delta == -7
        assert (slice_header.slice_cb_qp_offset, slice_header.slice_cr_qp_offset) == (1, -2)
        assert slice_header.deblocking_filter_override_flag == 1
        assert slice_header.slice_deblocking_filter_disabled_flag == 0
        assert (slice_header.slice_beta_offset_div2, slice_header.slice_tc_offset_div2) == (-1, 4)
        assert slice_header.slice_loop_filter_across_slices_enabled_flag == 0

    def test_a_slice_that_switches_deblocking_off(self):
        sps, pps = self.sets(loop_filter_across_slices=1,
                             deblocking={'override': 1, 'beta': 2, 'tc': -3})
        slice_header = header(sps, pps, kind='P', poc_lsb=4, rps=0,
                              deblocking={'disabled': 1}, qp_delta=3)
        assert slice_header.slice_deblocking_filter_disabled_flag == 1
        assert (slice_header.slice_beta_offset_div2, slice_header.slice_tc_offset_div2) == (0, 0)
        # With deblocking off and no SAO, the loop filter flag is not coded.
        assert slice_header.slice_loop_filter_across_slices_enabled_flag == 1

    def test_picture_output_defaults_to_shown(self):
        sps, pps = self.sets()
        assert header(sps, pps, kind='P', poc_lsb=4, rps=0).pic_output_flag == 1

    def test_an_irap_slice_carries_no_output_of_prior_pics(self):
        sps, pps = self.sets()
        slice_header = header(sps, pps, nal_type=NAL_CRA, kind='I', poc_lsb=9, rps=3,
                              no_output=1)
        assert slice_header.irap and not slice_header.idr
        assert slice_header.no_output_of_prior_pics_flag == 1
        assert slice_header.slice_pic_order_cnt_lsb == 9

    def test_entry_points_and_the_header_extension(self):
        sps, pps = self.sets(tiles={'columns': 3, 'rows': 2}, header_extension=1)
        slice_header = header(sps, pps, nal_type=NAL_IDR_N_LP, kind='I',
                              entry_points=(5, 300, 2), extension=b'\x01\x02')
        assert slice_header.entry_point_offset_minus1 == (5, 300, 2)
        assert slice_header.num_entry_point_offsets == 3

    def test_a_dependent_slice_segment_continues_its_slice(self):
        sps, pps = self.sets(dependent_slices=1, wpp=1, refs=(1, 0))
        independent = header(sps, pps, kind='P', poc_lsb=4, rps=0, qp_delta=5,
                             temporal_mvp=1, entry_points=(9,))
        dependent = header(sps, pps, independent=independent, first=0, dependent=1,
                           address=7, entry_points=(1, 2))
        assert dependent.dependent_slice_segment_flag == 1
        assert dependent.first_slice_segment_in_pic_flag == 0
        assert dependent.slice_segment_address == 7
        assert dependent.entry_point_offset_minus1 == (1, 2)
        assert dependent.slice_type == SLICE_P
        assert dependent.slice_qp_delta == 5
        assert dependent.slice_pic_order_cnt_lsb == 4
        assert dependent.short_term_ref_pic_set == independent.short_term_ref_pic_set
        # The flag, the PPS, the flag, a four-bit address, two entry points.
        assert dependent.header_bytes == 2 + 3
        assert dependent.header_bytes < independent.header_bytes

    def test_a_dependent_slice_segment_with_no_slice_is_refused(self):
        sps, pps = self.sets(dependent_slices=1)
        unit = slice_unit(sps, pps, first=0, dependent=1, address=7)
        with pytest.raises(StreamError, match='dependent'):
            parse_slice_header(unit, {0: sps}, {0: pps})

    def test_a_second_slice_states_its_address(self):
        sps, pps = self.sets()
        slice_header = header(sps, pps, first=0, address=11, kind='B', poc_lsb=4, rps=0)
        assert slice_header.slice_segment_address == 11
        assert slice_header.dependent_slice_segment_flag == 0
        assert slice_header.slice_type == SLICE_B

    def test_the_range_extension_adds_a_flag(self):
        sps = sequence(chroma_format_idc=2, profile_idc=4, st_rps=SETS[3:])
        pps = picture_set(range_extension={'offset_list': [(1, -1)]}, wpp=1)
        slice_header = header(sps, pps, kind='P', poc_lsb=1, rps=0, cu_chroma_qp_offset=1,
                              entry_points=(3, 4))
        assert slice_header.cu_chroma_qp_offset_enabled_flag == 1
        assert slice_header.entry_point_offset_minus1 == (3, 4)

    def test_the_picture_kinds(self):
        sps, pps = self.sets()
        rasl = header(sps, pps, nal_type=NAL_RASL_N, kind='B', poc_lsb=4, rps=0)
        assert rasl.rasl and rasl.sub_layer_non_reference and not rasl.irap
        trailing = header(sps, pps, kind='P', poc_lsb=4, rps=0, temporal_id=2)
        assert not trailing.sub_layer_non_reference
        assert trailing.temporal_id == 2


class TestWhereTheSliceDataStarts:
    def sets(self):
        return sequence(st_rps=SETS), picture_set(header_extension=1)

    def test_the_offset_counts_the_nal_header(self):
        sps, pps = self.sets()
        unit = slice_unit(sps, pps, kind='P', poc_lsb=4, rps=0, data=b'\xAB\xCD\xEF')
        slice_header = parse_slice_header(unit, {0: sps}, {0: pps})
        assert slice_header.header_escapes == 0
        assert unit.data[slice_header.header_bytes:] == b'\xAB\xCD\xEF'

    def test_escapes_in_the_header_are_counted(self):
        sps, pps = self.sets()
        unit = slice_unit(sps, pps, kind='P', poc_lsb=4, rps=0,
                          extension=bytes(7), data=b'\xAB\xCD\xEF')
        slice_header = parse_slice_header(unit, {0: sps}, {0: pps})
        assert slice_header.header_escapes == unit.data.count(b'\x00\x00\x03') > 0
        start = slice_header.header_bytes + slice_header.header_escapes
        assert unit.data[start:] == b'\xAB\xCD\xEF'

    def test_escapes_in_the_slice_data_are_not(self):
        sps, pps = self.sets()
        unit = slice_unit(sps, pps, kind='P', poc_lsb=4, rps=0, data=bytes(6) + b'\x01')
        slice_header = parse_slice_header(unit, {0: sps}, {0: pps})
        assert b'\x00\x00\x03' in unit.data
        assert slice_header.header_escapes == 0


class TestNalUnits:
    def test_the_header_fields(self):
        unit = parse_nal(nal(NAL_SPS, b'\x12\x34', temporal_id=2))
        assert (unit.nal_unit_type, unit.nuh_layer_id, unit.temporal_id) == (NAL_SPS, 0, 2)
        assert unit.rbsp == b'\x12\x34'
        assert not unit.is_slice

    def test_the_layer_spans_both_bytes(self):
        assert parse_nal(nal(NAL_SPS, b'\x80', layer=37)).nuh_layer_id == 37

    def test_emulation_prevention_is_removed_from_the_rbsp_only(self):
        unit = parse_nal(nal(NAL_SPS, b'\x00\x00\x01\x00\x00\x02'))
        assert unit.rbsp == b'\x00\x00\x01\x00\x00\x02'
        assert unit.data[2:] == b'\x00\x00\x03\x01\x00\x00\x03\x02'

    @pytest.mark.parametrize('data,message', [
        (b'\x42', 'no header'), (b'\xC2\x01\x00', 'forbidden_zero_bit'),
        (b'\x42\x00\x00', 'nuh_temporal_id_plus1')])
    def test_broken_headers_are_refused(self, data, message):
        with pytest.raises(StreamError, match=message):
            parse_nal(data)


class TestBrokenStreams:
    def test_a_slice_naming_an_unknown_pps(self):
        sps, pps = sequence(), picture_set(pic_parameter_set_id=4)
        unit = slice_unit(sps, pps, nal_type=NAL_IDR_W_RADL, kind='I')
        with pytest.raises(StreamError, match='picture parameter set 4'):
            parse_slice_header(unit, {0: sps}, {})

    def test_a_pps_naming_an_unknown_sps(self):
        sps, pps = sequence(), picture_set(seq_parameter_set_id=3)
        unit = slice_unit(sps, pps, nal_type=NAL_IDR_W_RADL, kind='I')
        with pytest.raises(StreamError, match='sequence parameter set 3'):
            parse_slice_header(unit, {0: sps}, {0: pps})

    @pytest.mark.parametrize('flags,name', [
        ((1, 0, 0), 'multilayer'), ((0, 1, 0), '3D'), ((0, 0, 1), 'screen content')])
    def test_slices_of_an_extension_not_read_are_refused_by_name(self, flags, name):
        sps, pps = sequence(), picture_set(extension_flags=flags)
        unit = parse_nal(nal(NAL_IDR_W_RADL, slice_rbsp(sps, picture_set(), nal_type=19,
                                                        kind='I')))
        with pytest.raises(StreamError, match=name):
            parse_slice_header(unit, {0: sps}, {0: pps})

    def test_a_truncated_parameter_set(self):
        with pytest.raises(StreamError, match='past the end'):
            parse_sps(sps_bytes()[:12])
        with pytest.raises(StreamError, match='past the end'):
            parse_pps(pps_bytes()[:1])

    def test_a_slice_naming_a_set_the_sequence_does_not_have(self):
        sps, pps = sequence(st_rps=SETS[:1]), picture_set()
        unit = slice_unit(sequence(st_rps=SETS), pps, kind='P', poc_lsb=1, rps=3)
        with pytest.raises(StreamError):
            parse_slice_header(unit, {0: sps}, {0: pps})

    def test_a_slice_type_that_is_not_one(self):
        sps, pps = sequence(), picture_set()
        unit = parse_nal(nal(NAL_IDR_W_RADL, b'\xA2\x00\x80'))
        with pytest.raises(StreamError, match='slice_type'):
            parse_slice_header(unit, {0: sps}, {0: pps})
