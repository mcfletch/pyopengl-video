"""Every HEVC syntax element is held to its range where it is read.

Each case writes a parameter set or a slice segment header that is in range
but for one element, and the parser names that element in a
:class:`StreamError`. The ranges are those of ITU-T H.265, as Appendix A of
``plans/SECURITY-REVIEW.md`` lists them. A picture parameter set is read
without its sequence, so what the sequence bounds is checked again when a
slice names the two together.
"""
import pytest

from pyopengl_video.vaapi.h264 import StreamError
from pyopengl_video.vaapi.hevc_parse import (
    NAL_BLA_W_LP,
    NAL_CRA,
    NAL_IDR_W_RADL,
    parse_pps,
    parse_sps,
    parse_vps,
)
from tests.hevc_syntax import (
    coded,
    header,
    picture_set,
    pps_bytes,
    sequence,
    sps_bytes,
    vps_bytes,
)

FLAT_4 = (None, (16,) * 16)
FLAT_8 = (None, (16,) * 64)


def scaling(sixteen):
    """Scaling lists that are flat but for the first 16x16 list."""
    return [[FLAT_4, 1, 1, 1, 1, 1], [FLAT_8, 1, 1, 1, 1, 1],
            [sixteen, 1, 1, 1, 1, 1], [(16, (16,) * 64), 1]]


# ------------------------------------------------- sequence parameter sets


@pytest.mark.parametrize(('options', 'message'), [
    ({'seq_parameter_set_id': 16}, 'sps_seq_parameter_set_id is 16, outside 0 to 15'),
    ({'seq_parameter_set_id': 319}, 'sps_seq_parameter_set_id is 319'),
    ({'chroma_format_idc': 4}, 'chroma_format_idc is 4'),
    ({'width': 0}, 'pic_width_in_luma_samples is 0, outside 1 to 16888'),
    ({'height': 0}, 'pic_height_in_luma_samples is 0'),
    ({'width': 16896}, 'pic_width_in_luma_samples is 16896'),
    ({'width': 65600}, 'pic_width_in_luma_samples is 65600'),
    ({'width': 8192, 'height': 8192}, 'PicSizeInSamplesY is 67108864'),
    ({'width': 68}, 'pic_width_in_luma_samples is 68, which is not whole coding blocks'),
    ({'height': 100}, 'pic_height_in_luma_samples is 100, which is not whole'),
    ({'bit_depth': 9}, 'bit_depth_luma_minus8 is 9, outside 0 to 8'),
    ({'bit_depth_chroma': 9}, 'bit_depth_chroma_minus8 is 9'),
    ({'log2_max_poc_lsb_minus4': 13}, 'log2_max_pic_order_cnt_lsb_minus4 is 13'),
    ({'ordering': ((16, 0, 0),)}, 'max_dec_pic_buffering_minus1 is 16, outside 0 to 15'),
    ({'ordering': ((500000, 0, 0),)}, 'max_dec_pic_buffering_minus1 is 500000'),
    ({'ordering': ((2, 3, 0),)}, 'max_num_reorder_pics is 3, outside 0 to 2'),
    ({'min_cb_minus3': 4}, 'log2_min_luma_coding_block_size_minus3 is 4, outside 0 to 3'),
    ({'min_cb_minus3': 300}, 'log2_min_luma_coding_block_size_minus3 is 300'),
    ({'diff_cb': 0}, 'log2_diff_max_min_luma_coding_block_size is 0, outside 1 to 3'),
    ({'diff_cb': 4}, 'log2_diff_max_min_luma_coding_block_size is 4, outside 1 to 3'),
    ({'diff_cb': 260}, 'log2_diff_max_min_luma_coding_block_size is 260'),
    ({'transform': (1, 0, 0, 0)},
     'log2_min_luma_transform_block_size_minus2 is 1, outside 0 to 0'),
    ({'transform': (0, 3, 0, 0)},
     'log2_diff_max_min_luma_transform_block_size is 3, outside 0 to 2'),
    ({'transform': (0, 2, 3, 0)}, 'max_transform_hierarchy_depth_inter is 3, outside 0 to 2'),
    ({'transform': (0, 2, 0, 3)}, 'max_transform_hierarchy_depth_intra is 3, outside 0 to 2'),
    ({'pcm': (8, 7, 0, 0, 0)}, 'pcm_sample_bit_depth_luma_minus1 is 8, outside 0 to 7'),
    ({'pcm': (7, 8, 0, 0, 0)}, 'pcm_sample_bit_depth_chroma_minus1 is 8'),
    ({'pcm': (7, 7, 2, 0, 0)},
     'log2_min_pcm_luma_coding_block_size_minus3 is 2, outside 0 to 1'),
    ({'pcm': (7, 7, 0, 2, 0)},
     'log2_diff_max_min_pcm_luma_coding_block_size is 2, outside 0 to 1'),
    ({'st_rps': [{}] * 65}, 'num_short_term_ref_pic_sets is 65, outside 0 to 64'),
    ({'st_rps': [{'s0': [(0, 1)] * 6}]}, 'num_negative_pics is 6, outside 0 to 5'),
    ({'st_rps': [{'s0': [(0, 1)] * 3, 's1': [(0, 1)] * 3}]},
     'num_positive_pics is 3, outside 0 to 2'),
    ({'st_rps': [{'s0': [(32768, 1)]}]}, 'delta_poc_s0_minus1 is 32768'),
    ({'st_rps': [{'s1': [(32768, 1)]}]}, 'delta_poc_s1_minus1 is 32768'),
    ({'st_rps': [{'s0': [(32767, 1), (0, 1)]}]}, 'DeltaPocS0 is -32769'),
    ({'st_rps': [{'s1': [(32767, 1)]}]}, 'DeltaPocS1 is 32768'),
    ({'st_rps': [{'s0': [(0, 1)]}, {'delta_rps': -32769, 'used': [1, 1],
                                    'use_delta': [1, 1]}]},
     'abs_delta_rps_minus1 is 32768'),
    ({'long_term': [(0, 1)] * 33}, 'num_long_term_ref_pics_sps is 33, outside 0 to 32'),
    ({'scaling': scaling((0, (16,) * 64))}, 'scaling_list_dc_coef_minus8 is -8'),
    ({'scaling': scaling((16, (16, 0) + (16,) * 62))}, 'has a factor of zero'),
    ({'scaling': [[FLAT_4, 2, 1, 1, 1, 1], *scaling((16, (16,) * 64))[1:]]},
     'scaling_list_pred_matrix_id_delta is 2, outside 0 to 1'),
    ({'vui': {'display_window': (0, 0, 0, 32)}},
     'def_disp_win_bottom_offset is 32, outside 0 to 31'),
    ({'vui': {'chroma_loc': 6}}, 'chroma_sample_loc_type_top_field is 6, outside 0 to 5'),
])
def test_a_sequence_parameter_set_element_outside_its_range_is_refused(options, message):
    with pytest.raises(StreamError, match=message):
        parse_sps(sps_bytes(**options))


@pytest.mark.parametrize('options', [
    {'seq_parameter_set_id': 15, 'bit_depth': 8, 'log2_max_poc_lsb_minus4': 12},
    {'ordering': ((15, 15, 0),)},
    {'width': 16888, 'height': 2104},
    {'min_cb_minus3': 3, 'diff_cb': 0, 'transform': (3, 0, 1, 1)},
    {'diff_cb': 3, 'transform': (0, 3, 4, 4)},
    {'pcm': (7, 7, 0, 1, 0)},
    {'ordering': ((5, 2, 0),), 'st_rps': [{'s0': [(0, 1)] * 5}] * 64},
    {'st_rps': [{'s0': [(32767, 1)]}, {'s1': [(32766, 1)]}]},
    {'long_term': [(0, 1)] * 32},
])
def test_the_ends_of_each_range_are_read(options):
    parse_sps(sps_bytes(**options))


def test_a_conformance_window_leaves_a_picture():
    """D4: the offsets of a pair may not meet, so no visible size is below
    one. An offset is two luma samples at 4:2:0."""
    assert sequence(conformance_window=(20, 11, 0, 31)).visible == (40, 0, 2, 2)
    with pytest.raises(StreamError, match='conf_win_right_offset is 12, outside 0 to 11'):
        sequence(conformance_window=(20, 12, 0, 0))
    with pytest.raises(StreamError, match='conf_win_bottom_offset is 40'):
        sequence(conformance_window=(0, 0, 0, 40))


def test_a_predicted_set_is_held_to_the_buffer_as_a_coded_one_is():
    """H16: the set it is predicted from has five pictures, all the buffer
    keeps, and the prediction adds that set's own picture as a sixth."""
    five = {'s0': [(0, 1)] * 5}
    sixth = {'delta_rps': -1, 'used': [1] * 6, 'use_delta': [1] * 6}
    assert sequence(st_rps=[five]).short_term_ref_pic_sets[0].num_delta_pocs == 5
    with pytest.raises(StreamError, match='predicted to 6 pictures, and the decoded '
                                          'picture buffer keeps 5'):
        sequence(st_rps=[five, sixth])


def test_a_predicted_set_reaches_no_further_than_a_coded_one():
    far = {'s0': [(32767, 1)]}
    farther = {'delta_rps': -1, 'used': [1, 0], 'use_delta': [1, 0]}
    with pytest.raises(StreamError, match='DeltaPocS0 is -32769'):
        sequence(st_rps=[far, farther])


def test_more_sub_layers_than_there_are_is_refused():
    damaged = bytearray(sps_bytes())
    damaged[0] |= 0x0E                          # sps_max_sub_layers_minus1 of 7
    with pytest.raises(StreamError, match='sps_max_sub_layers_minus1 is 7'):
        parse_sps(bytes(damaged))
    damaged = bytearray(vps_bytes())
    damaged[1] |= 0x0E                          # vps_max_sub_layers_minus1 of 7
    with pytest.raises(StreamError, match='vps_max_sub_layers_minus1 is 7'):
        parse_vps(bytes(damaged))


def test_a_video_parameter_set_is_held_to_the_same_buffer():
    with pytest.raises(StreamError, match='max_dec_pic_buffering_minus1 is 16'):
        parse_vps(vps_bytes(ordering=((16, 0, 0),)))
    with pytest.raises(StreamError, match='vps_num_layer_sets_minus1 is 1024'):
        parse_vps(vps_bytes(layer_sets=1024))


# -------------------------------------------------- picture parameter sets


@pytest.mark.parametrize(('options', 'message'), [
    ({'pic_parameter_set_id': 64}, 'pps_pic_parameter_set_id is 64, outside 0 to 63'),
    ({'pic_parameter_set_id': 20000}, 'pps_pic_parameter_set_id is 20000'),
    ({'seq_parameter_set_id': 16}, 'pps_seq_parameter_set_id is 16, outside 0 to 15'),
    ({'refs': (15, 0)}, 'num_ref_idx_l0_default_active_minus1 is 15, outside 0 to 14'),
    ({'refs': (0, 15)}, 'num_ref_idx_l1_default_active_minus1 is 15'),
    ({'init_qp': 26}, 'init_qp_minus26 is 26'),
    ({'init_qp': -200}, 'init_qp_minus26 is -200'),
    ({'cu_qp_delta_depth': 4}, 'diff_cu_qp_delta_depth is 4, outside 0 to 3'),
    ({'chroma_offsets': (13, 0)}, 'pps_cb_qp_offset is 13, outside -12 to 12'),
    ({'chroma_offsets': (0, -300)}, 'pps_cr_qp_offset is -300'),
    ({'tiles': {'columns': 21, 'rows': 1}}, 'num_tile_columns_minus1 is 20, outside 0 to 19'),
    ({'tiles': {'columns': 1, 'rows': 23}}, 'num_tile_rows_minus1 is 22, outside 0 to 21'),
    ({'tiles': {'columns': 2, 'rows': 1, 'widths': (1056,), 'heights': ()}},
     'column_width_minus1 is 1056'),
    ({'deblocking': {'beta': 7}}, 'pps_beta_offset_div2 is 7, outside -6 to 6'),
    ({'deblocking': {'tc': -100}}, 'pps_tc_offset_div2 is -100'),
    ({'parallel_merge': 5}, 'log2_parallel_merge_level_minus2 is 5, outside 0 to 4'),
    ({'parallel_merge': 77}, 'log2_parallel_merge_level_minus2 is 77'),
    ({'transform_skip': 1, 'range_extension': {'sao_scale': (7, 0)}},
     'log2_sao_offset_scale_luma is 7'),
    ({'range_extension': {'offset_list': [(13, 0)]}}, 'cb_qp_offset_list is 13'),
    ({'range_extension': {'offset_list': [(0, 0)] * 7}},
     'chroma_qp_offset_list_len_minus1 is 6'),
])
def test_a_picture_parameter_set_element_outside_its_range_is_refused(options, message):
    with pytest.raises(StreamError, match=message):
        parse_pps(pps_bytes(**options))


@pytest.mark.parametrize(('sps_options', 'pps_options', 'message'), [
    ({}, {'init_qp': -27}, 'init_qp_minus26 is -27, outside -26 to 25'),
    ({'bit_depth': 2}, {'init_qp': -39}, 'init_qp_minus26 is -39, outside -38 to 25'),
    ({}, {'cu_qp_delta_depth': 2}, 'diff_cu_qp_delta_depth is 2, outside 0 to 1'),
    ({}, {'parallel_merge': 3}, 'log2_parallel_merge_level_minus2 is 3, outside 0 to 2'),
    ({}, {'tiles': {'columns': 20, 'rows': 22}},
     '20 tiles across a picture of 4 coding tree blocks'),
    ({}, {'tiles': {'columns': 1, 'rows': 5}},
     '5 tiles across a picture of 4 coding tree blocks'),
    ({}, {'tiles': {'columns': 3, 'rows': 1, 'widths': (1, 1), 'heights': ()}},
     'leave none for the last'),
])
def test_a_slice_holds_its_picture_set_to_its_sequence(sps_options, pps_options, message):
    sps, pps = sequence(**sps_options), picture_set(**pps_options)
    with pytest.raises(StreamError, match=message):
        header(sps, pps, nal_type=NAL_IDR_W_RADL, kind='I')


def test_a_uniform_tile_grid_gives_every_tile_a_coding_tree_block():
    """H5: four tiles across four coding tree blocks is the most, and each
    is one block; a fifth would have none, and its width less one is what a
    16-bit field would be given."""
    sps = sequence()
    assert picture_set(tiles={'columns': 4, 'rows': 4}).tile_columns(sps) == (1, 1, 1, 1)
    with pytest.raises(StreamError, match='5 tiles across a picture of 4'):
        picture_set(tiles={'columns': 5, 'rows': 1}).tile_columns(sps)


def test_the_quantiser_reaches_as_far_below_zero_as_the_samples_are_deep():
    sps, pps = sequence(bit_depth=2), picture_set(init_qp=-38)
    assert header(sps, pps, nal_type=NAL_IDR_W_RADL, kind='I',
                  qp_delta=0).pps.init_qp_minus26 == -38


# --------------------------------------------------- slice segment headers


def predicted(*, sps=None, pps=None, **options):
    """The header of a slice predicting from the picture before it, a P slice
    unless `options` say another kind."""
    sps = {'ordering': ((5, 2, 0),), **(sps or {})}
    options.setdefault('rps', coded(before=(-1,)))
    options.setdefault('kind', 'P')
    return header(sequence(**sps), picture_set(**(pps or {})), poc_lsb=4, **options)


@pytest.mark.parametrize(('sps', 'pps', 'options', 'message'), [
    ({}, {}, {'refs': (15, 0)}, 'num_ref_idx_l0_active_minus1 is 15, outside 0 to 14'),
    ({}, {}, {'kind': 'B', 'refs': (0, 15)}, 'num_ref_idx_l1_active_minus1 is 15'),
    ({}, {}, {'merge': 5}, 'five_minus_max_num_merge_cand is 5, outside 0 to 4'),
    ({}, {}, {'merge': 9}, 'five_minus_max_num_merge_cand is 9'),
    ({}, {}, {'qp_delta': 26}, 'slice_qp_delta is 26, outside -26 to 25'),
    ({}, {}, {'qp_delta': -300}, 'slice_qp_delta is -300'),
    ({}, {'init_qp': -20}, {'qp_delta': -7}, 'slice_qp_delta is -7, outside -6 to 45'),
    ({'temporal_mvp': 1}, {}, {'temporal_mvp': 1, 'refs': (1, 0), 'collocated': (1, 2)},
     'collocated_ref_idx is 2, outside 0 to 1'),
    ({'temporal_mvp': 1}, {}, {'temporal_mvp': 1, 'refs': (1, 0), 'collocated': (1, 200)},
     'collocated_ref_idx is 200'),
    ({}, {'slice_chroma_offsets_present': 1}, {'chroma_offsets': (13, 0)},
     'slice_cb_qp_offset is 13, outside -12 to 12'),
    ({}, {'slice_chroma_offsets_present': 1}, {'chroma_offsets': (0, -13)},
     'slice_cr_qp_offset is -13'),
    ({}, {'slice_chroma_offsets_present': 1, 'chroma_offsets': (10, 0)},
     {'chroma_offsets': (5, 0)}, r'pps_cb_qp_offset \+ slice_cb_qp_offset is 15'),
    ({}, {'slice_chroma_offsets_present': 1, 'chroma_offsets': (0, -10)},
     {'chroma_offsets': (0, -5)}, r'pps_cr_qp_offset \+ slice_cr_qp_offset is -15'),
    ({}, {'deblocking': {'override': 1}}, {'deblocking': {'beta': 7}},
     'slice_beta_offset_div2 is 7, outside -6 to 6'),
    ({}, {'deblocking': {'override': 1}}, {'deblocking': {'tc': -7}},
     'slice_tc_offset_div2 is -7'),
    ({}, {'weighted_pred': 1}, {'weights': {'denom': 8}}, 'luma_log2_weight_denom is 8'),
    ({}, {'weighted_pred': 1}, {'weights': {'denom': 3, 'chroma_delta': -4}},
     'delta_chroma_log2_weight_denom is -4, outside -3 to 4'),
    ({}, {'weighted_pred': 1}, {'weights': {'denom': 0, 'l0': [((300, 0), None)]}},
     'delta_luma_weight is 300, outside -128 to 127'),
    ({}, {'weighted_pred': 1}, {'weights': {'denom': 0, 'l0': [((0, -1000), None)]}},
     'luma_offset is -1000, outside -128 to 127'),
    ({}, {'weighted_pred': 1},
     {'weights': {'denom': 0, 'l0': [(None, ((400, 0), (0, 0)))]}},
     'delta_chroma_weight is 400'),
    ({}, {'weighted_pred': 1},
     {'weights': {'denom': 0, 'l0': [(None, ((0, 512), (0, 0)))]}},
     'delta_chroma_offset is 512, outside -512 to 511'),
    ({}, {'wpp': 1}, {'entry_points': (1,) * 4},
     'num_entry_point_offsets is 4, outside 0 to 3'),
    ({}, {'tiles': {'columns': 2, 'rows': 2}}, {'entry_points': (1,) * 4},
     'num_entry_point_offsets is 4, outside 0 to 3'),
    ({}, {'tiles': {'columns': 2, 'rows': 2}, 'wpp': 1}, {'entry_points': (1,) * 8},
     'num_entry_point_offsets is 8, outside 0 to 7'),
    ({}, {'header_extension': 1}, {'extension': bytes(257)},
     'slice_segment_header_extension_length is 257'),
    ({}, {}, {'rps': coded(before=(-1, -2, -3, -4, -5, -6))},
     'num_negative_pics is 6, outside 0 to 5'),
    ({'long_term': ()}, {}, {'rps': coded(before=(-1, -2, -3)),
                             'long_term': {'pics': [(0, 1, None)] * 3}},
     'num_long_term_pics is 3, outside 0 to 2'),
    ({'long_term': ((1, 1), (2, 1), (3, 1))}, {},
     {'rps': coded(before=(-1, -2, -3, -4)), 'long_term': {'sps': [(0, None), (1, None)]}},
     'num_long_term_sps is 2, outside 0 to 1'),
    ({'long_term': ()}, {}, {'long_term': {'pics': [(0, 1, 2**24 + 1)]}},
     'delta_poc_msb_cycle_lt is 16777217'),
    ({}, {'lists_modification': 1},
     {'rps': coded(before=(-1, -2, -3)), 'modification': ((3,), None)},
     'list_entry_l0 is 3, outside 0 to 2'),
])
def test_a_slice_segment_header_element_outside_its_range_is_refused(
        sps, pps, options, message):
    with pytest.raises(StreamError, match=message):
        predicted(sps=sps, pps=pps, **options)


def test_a_slice_segment_address_past_the_picture_is_refused():
    """Six coding tree blocks are addressed in three bits, which reach seven."""
    sps = sequence(width=96, height=64, diff_cb=2)
    assert sps.width_in_ctbs * sps.height_in_ctbs == 6
    assert header(sps, picture_set(), nal_type=NAL_IDR_W_RADL, kind='I', first=0,
                  address=5).slice_segment_address == 5
    with pytest.raises(StreamError, match='slice_segment_address is 7, outside 0 to 5'):
        header(sps, picture_set(), nal_type=NAL_IDR_W_RADL, kind='I', first=0, address=7)


@pytest.mark.parametrize('nal_type', [NAL_IDR_W_RADL, NAL_CRA, NAL_BLA_W_LP])
@pytest.mark.parametrize('kind', ['P', 'B'])
def test_a_predicted_slice_of_an_intra_random_access_point_is_refused(nal_type, kind):
    """7.4.7.1: such a picture has I slices alone, so it never names a
    picture to predict from that the buffer is about to drop."""
    with pytest.raises(StreamError, match=f'a {kind} slice in an intra random access'):
        header(sequence(), picture_set(), nal_type=nal_type, kind=kind,
               rps=coded(before=(-1,)))


def test_a_dependent_slice_segment_is_of_its_slices_parameter_sets():
    sps = sequence()
    pps = picture_set(dependent_slices=1)
    other = sequence(width=128, height=128)
    first = header(sps, pps, nal_type=NAL_IDR_W_RADL, kind='I')
    with pytest.raises(StreamError, match='other parameter sets than the slice it continues'):
        header(other, pps, independent=first, nal_type=NAL_IDR_W_RADL, first=0,
               dependent=1, address=40)
