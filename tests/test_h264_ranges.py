"""Every H.264 syntax element is held to its range where it is read.

Each case writes a parameter set or a slice header that is in range but for
one element, and the parser names that element in a :class:`StreamError`.
The ranges are those of ITU-T H.264, as Appendix A of
``plans/SECURITY-REVIEW.md`` lists them. A value outside its range is never
handed on, because the structures a driver reads have narrower fields than
the syntax and ctypes keeps only the bits that fit.
"""
import pytest

from pyopengl_video.vaapi import h264_parse
from pyopengl_video.vaapi.h264 import BitWriter, StreamError, nal_unit
from pyopengl_video.vaapi.h264_parse import (
    parse_nal,
    parse_pps,
    parse_slice_header,
    parse_sps,
)

#: The largest ue(v) code number, and one that takes 2 GB to act on.
UE_MOST = 2**32 - 2
HUGE = 2**28


def written(syntax, changed):
    """An RBSP of `syntax`: ``(element, kind, value)`` in order, `changed`
    replacing the value of the elements it names."""
    unknown = set(changed) - {name for name, _, _ in syntax}
    assert not unknown, f'{unknown} are not in the syntax'
    writer = BitWriter()
    for name, kind, value in syntax:
        value = changed.get(name, value)
        if kind == 'ue':
            writer.ue(value)
        elif kind == 'se':
            writer.se(value)
        else:
            writer.u(value, int(kind[1:]))
    return writer.rbsp()


def replaced(syntax, name, *entries):
    """`syntax` with the element `name` replaced by `entries`."""
    index = [element for element, _, _ in syntax].index(name)
    return [*syntax[:index], *entries, *syntax[index + 1:]]


# ------------------------------------------------- sequence parameter sets

#: A High profile SPS for a 64x64 picture (7.3.2.1.1).
SPS = [
    ('profile_idc', 'u8', 100), ('constraint_flags', 'u8', 0), ('level_idc', 'u8', 30),
    ('seq_parameter_set_id', 'ue', 0), ('chroma_format_idc', 'ue', 1),
    ('bit_depth_luma_minus8', 'ue', 0), ('bit_depth_chroma_minus8', 'ue', 0),
    ('qpprime_y_zero_transform_bypass_flag', 'u1', 0),
    ('seq_scaling_matrix_present_flag', 'u1', 0),
    ('log2_max_frame_num_minus4', 'ue', 0), ('pic_order_cnt_type', 'ue', 0),
    ('log2_max_pic_order_cnt_lsb_minus4', 'ue', 2), ('max_num_ref_frames', 'ue', 4),
    ('gaps_in_frame_num_value_allowed_flag', 'u1', 0),
    ('pic_width_in_mbs_minus1', 'ue', 3), ('pic_height_in_map_units_minus1', 'ue', 3),
    ('frame_mbs_only_flag', 'u1', 1), ('direct_8x8_inference_flag', 'u1', 1),
    ('frame_cropping_flag', 'u1', 0), ('vui_parameters_present_flag', 'u1', 0),
]

#: With picture order count type 1 and one offset in its cycle.
SPS_ORDER_ONE = replaced(
    replaced(SPS, 'pic_order_cnt_type', ('pic_order_cnt_type', 'ue', 1)),
    'log2_max_pic_order_cnt_lsb_minus4',
    ('delta_pic_order_always_zero_flag', 'u1', 0), ('offset_for_non_ref_pic', 'se', 0),
    ('offset_for_top_to_bottom_field', 'se', 0),
    ('num_ref_frames_in_pic_order_cnt_cycle', 'ue', 1), ('offset_for_ref_frame', 'se', 0))

#: With a cropping window. An offset is two luma samples, so 32 of them cross
#: the picture.
SPS_CROPPED = replaced(
    SPS, 'frame_cropping_flag', ('frame_cropping_flag', 'u1', 1),
    ('frame_crop_left_offset', 'ue', 0), ('frame_crop_right_offset', 'ue', 0),
    ('frame_crop_top_offset', 'ue', 0), ('frame_crop_bottom_offset', 'ue', 0))

#: With a VUI that holds a bitstream restriction and nothing else (E.1.1).
SPS_RESTRICTED = replaced(
    SPS, 'vui_parameters_present_flag', ('vui_parameters_present_flag', 'u1', 1),
    ('aspect_overscan_signal_chroma_timing_hrd_pic_struct', 'u8', 0),
    ('bitstream_restriction_flag', 'u1', 1),
    ('motion_vectors_over_pic_boundaries_flag', 'u1', 1),
    ('max_bytes_per_pic_denom', 'ue', 0), ('max_bits_per_mb_denom', 'ue', 0),
    ('log2_max_mv_length_horizontal', 'ue', 16), ('log2_max_mv_length_vertical', 'ue', 16),
    ('max_num_reorder_frames', 'ue', 2), ('max_dec_frame_buffering', 'ue', 4))

#: With a VUI that states where chroma samples lie.
SPS_CHROMA_LOCATION = replaced(
    SPS, 'vui_parameters_present_flag', ('vui_parameters_present_flag', 'u1', 1),
    ('aspect_overscan_signal', 'u3', 0), ('chroma_loc_info_present_flag', 'u1', 1),
    ('chroma_sample_loc_type_top_field', 'ue', 0),
    ('chroma_sample_loc_type_bottom_field', 'ue', 0),
    ('timing_hrd_pic_struct_restriction', 'u5', 0))


@pytest.mark.parametrize(('syntax', 'element', 'value'), [
    (SPS, 'seq_parameter_set_id', 32),
    (SPS, 'seq_parameter_set_id', UE_MOST),
    (SPS, 'chroma_format_idc', 4),
    (SPS, 'bit_depth_luma_minus8', 7),
    (SPS, 'bit_depth_chroma_minus8', 7),
    (SPS, 'log2_max_frame_num_minus4', 13),
    (SPS, 'log2_max_frame_num_minus4', HUGE),
    (SPS, 'pic_order_cnt_type', 3),
    (SPS, 'log2_max_pic_order_cnt_lsb_minus4', 13),
    (SPS, 'max_num_ref_frames', 17),
    (SPS, 'max_num_ref_frames', 100000),
    (SPS, 'pic_width_in_mbs_minus1', 1055),
    (SPS, 'pic_height_in_map_units_minus1', 1055),
    (SPS_ORDER_ONE, 'num_ref_frames_in_pic_order_cnt_cycle', 256),
    (SPS_ORDER_ONE, 'num_ref_frames_in_pic_order_cnt_cycle', HUGE),
    (SPS_CROPPED, 'frame_crop_left_offset', 32),
    (SPS_CROPPED, 'frame_crop_right_offset', 32),
    (SPS_CROPPED, 'frame_crop_top_offset', 32),
    (SPS_CROPPED, 'frame_crop_bottom_offset', 32),
    (SPS_CROPPED, 'frame_crop_bottom_offset', UE_MOST),
    (SPS_RESTRICTED, 'max_bytes_per_pic_denom', 17),
    (SPS_RESTRICTED, 'max_bits_per_mb_denom', 17),
    (SPS_RESTRICTED, 'log2_max_mv_length_horizontal', 17),
    (SPS_RESTRICTED, 'log2_max_mv_length_vertical', 17),
    (SPS_RESTRICTED, 'max_num_reorder_frames', 5),
    (SPS_RESTRICTED, 'max_dec_frame_buffering', 17),
    (SPS_RESTRICTED, 'max_dec_frame_buffering', 1000000),
    (SPS_RESTRICTED, 'max_dec_frame_buffering', 3),
    (SPS_CHROMA_LOCATION, 'chroma_sample_loc_type_top_field', 6),
    (SPS_CHROMA_LOCATION, 'chroma_sample_loc_type_bottom_field', 6),
])
def test_a_sequence_parameter_set_element_outside_its_range_is_refused(
        syntax, element, value):
    with pytest.raises(StreamError, match=f'{element} is {value}, outside'):
        parse_sps(written(syntax, {element: value}))


@pytest.mark.parametrize(('syntax', 'changed'), [
    (SPS, {'seq_parameter_set_id': 31, 'log2_max_frame_num_minus4': 12,
           'log2_max_pic_order_cnt_lsb_minus4': 12, 'max_num_ref_frames': 16}),
    (SPS, {'bit_depth_luma_minus8': 6, 'bit_depth_chroma_minus8': 6,
           'chroma_format_idc': 0}),
    (SPS, {'pic_width_in_mbs_minus1': 1054, 'pic_height_in_map_units_minus1': 131}),
    (SPS_ORDER_ONE, {'offset_for_non_ref_pic': -(2**31 - 1),
                     'offset_for_top_to_bottom_field': 2**31 - 1}),
    (SPS_CROPPED, {'frame_crop_left_offset': 31, 'frame_crop_top_offset': 15,
                   'frame_crop_bottom_offset': 16}),
    (SPS_RESTRICTED, {'max_num_reorder_frames': 16, 'max_dec_frame_buffering': 16}),
    (SPS_CHROMA_LOCATION, {'chroma_sample_loc_type_top_field': 5}),
])
def test_the_ends_of_each_range_are_read(syntax, changed):
    parse_sps(written(syntax, changed))


def test_a_frame_of_more_macroblocks_than_any_level_has_is_refused():
    """Each side is within Sqrt(MaxFS * 8), and the frame is not within MaxFS."""
    with pytest.raises(StreamError, match='PicSizeInMbs is 1113025'):
        parse_sps(written(SPS, {'pic_width_in_mbs_minus1': 1054,
                                'pic_height_in_map_units_minus1': 1054}))


def test_an_interlaced_frame_is_measured_in_macroblocks_not_map_units():
    with pytest.raises(StreamError, match='FrameHeightInMbs is 1056'):
        parse_sps(written(
            replaced(SPS, 'frame_mbs_only_flag', ('frame_mbs_only_flag', 'u1', 0),
                     ('mb_adaptive_frame_field_flag', 'u1', 0)),
            {'pic_height_in_map_units_minus1': 527}))


def test_a_cropping_window_leaves_a_picture():
    """D4: the offsets of a pair may not meet, so no visible size is below one."""
    sps = parse_sps(written(SPS_CROPPED, {'frame_crop_left_offset': 20,
                                          'frame_crop_right_offset': 11,
                                          'frame_crop_top_offset': 31}))
    assert sps.visible == (40, 62, 2, 2)
    with pytest.raises(StreamError, match='frame_crop_right_offset is 12, outside 0 to 11'):
        parse_sps(written(SPS_CROPPED, {'frame_crop_left_offset': 20,
                                        'frame_crop_right_offset': 12}))


def test_the_decoded_picture_buffer_is_never_more_than_sixteen_frames():
    """H9: no stream states a buffer, or a reorder depth, a session is asked
    for more surfaces than a level allows."""
    sps = parse_sps(written(SPS_RESTRICTED, {'max_num_reorder_frames': 16,
                                             'max_dec_frame_buffering': 16}))
    assert (sps.dpb_frames, sps.reorder_frames) == (16, 16)


def test_a_scaling_list_delta_outside_a_signed_byte_is_refused():
    syntax = replaced(SPS, 'seq_scaling_matrix_present_flag',
                      ('seq_scaling_matrix_present_flag', 'u1', 1),
                      ('seq_scaling_list_present_flag', 'u1', 1), ('delta_scale', 'se', 0))
    with pytest.raises(StreamError, match='delta_scale is 128, outside -128 to 127'):
        parse_sps(written(syntax, {'delta_scale': 128}))


# -------------------------------------------------- picture parameter sets

#: A PPS with CABAC and deblocking control, up to its optional tail (7.3.2.2).
PPS = [
    ('pic_parameter_set_id', 'ue', 0), ('seq_parameter_set_id', 'ue', 0),
    ('entropy_coding_mode_flag', 'u1', 1),
    ('bottom_field_pic_order_in_frame_present_flag', 'u1', 0),
    ('num_slice_groups_minus1', 'ue', 0),
    ('num_ref_idx_l0_default_active_minus1', 'ue', 0),
    ('num_ref_idx_l1_default_active_minus1', 'ue', 0),
    ('weighted_pred_flag', 'u1', 0), ('weighted_bipred_idc', 'u2', 0),
    ('pic_init_qp_minus26', 'se', 0), ('pic_init_qs_minus26', 'se', 0),
    ('chroma_qp_index_offset', 'se', 0),
    ('deblocking_filter_control_present_flag', 'u1', 1),
    ('constrained_intra_pred_flag', 'u1', 0), ('redundant_pic_cnt_present_flag', 'u1', 0),
]

PPS_WITH_TAIL = [*PPS, ('transform_8x8_mode_flag', 'u1', 1),
                 ('pic_scaling_matrix_present_flag', 'u1', 0),
                 ('second_chroma_qp_index_offset', 'se', 0)]

#: Two slice groups mapped as a foreground rectangle and what is left.
PPS_GROUPS = replaced(
    PPS, 'num_slice_groups_minus1', ('num_slice_groups_minus1', 'ue', 1),
    ('slice_group_map_type', 'ue', 2), ('top_left', 'ue', 0), ('bottom_right', 'ue', 5))


def sequences(**changed):
    """The SPS of :data:`SPS`, by its id."""
    sps = parse_sps(written(SPS, changed))
    return {sps.seq_parameter_set_id: sps}


@pytest.mark.parametrize(('syntax', 'element', 'value'), [
    (PPS, 'pic_parameter_set_id', 256),
    (PPS, 'pic_parameter_set_id', 20000),
    (PPS, 'seq_parameter_set_id', 32),
    (PPS, 'num_slice_groups_minus1', 8),
    (PPS, 'num_ref_idx_l0_default_active_minus1', 32),
    (PPS, 'num_ref_idx_l1_default_active_minus1', HUGE),
    (PPS, 'weighted_bipred_idc', 3),
    (PPS, 'pic_init_qp_minus26', -27),
    (PPS, 'pic_init_qp_minus26', 26),
    (PPS, 'pic_init_qs_minus26', -27),
    (PPS, 'pic_init_qs_minus26', 26),
    (PPS, 'chroma_qp_index_offset', -13),
    (PPS, 'chroma_qp_index_offset', 13),
    (PPS_WITH_TAIL, 'second_chroma_qp_index_offset', 13),
    (PPS_GROUPS, 'slice_group_map_type', 7),
    (PPS_GROUPS, 'top_left', 16),
    (PPS_GROUPS, 'bottom_right', 16),
])
def test_a_picture_parameter_set_element_outside_its_range_is_refused(
        syntax, element, value):
    with pytest.raises(StreamError, match=f'{element} is {value}, outside'):
        parse_pps(written(syntax, {element: value}), sequences())


def test_the_quantiser_reaches_as_far_below_zero_as_the_samples_are_deep():
    """pic_init_qp_minus26 is from -(26 + QpBdOffsetY): -38 at ten bits."""
    ten_bits = sequences(bit_depth_luma_minus8=2)
    assert parse_pps(written(PPS, {'pic_init_qp_minus26': -38}),
                     ten_bits).pic_init_qp_minus26 == -38
    with pytest.raises(StreamError, match='pic_init_qp_minus26 is -39, outside -38 to 25'):
        parse_pps(written(PPS, {'pic_init_qp_minus26': -39}), ten_bits)


# ------------------------------------------------------------ slice headers

#: A P slice of a reference picture against :data:`SPS` and :data:`PPS`
#: (7.3.3), and a byte of slice data.
P_SLICE = [
    ('first_mb_in_slice', 'ue', 0), ('slice_type', 'ue', 5),
    ('pic_parameter_set_id', 'ue', 0), ('frame_num', 'u4', 1),
    ('pic_order_cnt_lsb', 'u6', 2), ('num_ref_idx_active_override_flag', 'u1', 0),
    ('ref_pic_list_modification_flag_l0', 'u1', 0),
    ('adaptive_ref_pic_marking_mode_flag', 'u1', 0), ('cabac_init_idc', 'ue', 0),
    ('slice_qp_delta', 'se', 0), ('disable_deblocking_filter_idc', 'ue', 0),
    ('slice_alpha_c0_offset_div2', 'se', 0), ('slice_beta_offset_div2', 'se', 0),
    ('slice_data', 'u8', 0xA5),
]

P_OVERRIDE = replaced(
    P_SLICE, 'num_ref_idx_active_override_flag',
    ('num_ref_idx_active_override_flag', 'u1', 1),
    ('num_ref_idx_l0_active_minus1', 'ue', 0))

P_MODIFIED = replaced(
    P_SLICE, 'ref_pic_list_modification_flag_l0',
    ('ref_pic_list_modification_flag_l0', 'u1', 1),
    ('modification_of_pic_nums_idc', 'ue', 0), ('abs_diff_pic_num_minus1', 'ue', 0),
    ('end_of_modifications', 'ue', 3))

P_LONG_TERM_MODIFIED = replaced(
    P_SLICE, 'ref_pic_list_modification_flag_l0',
    ('ref_pic_list_modification_flag_l0', 'u1', 1),
    ('modification_is_long_term', 'ue', 2), ('long_term_pic_num', 'ue', 0),
    ('end_of_modifications', 'ue', 3))


def marking(operation, *arguments):
    """A P slice whose marking is one memory management control operation."""
    return replaced(
        P_SLICE, 'adaptive_ref_pic_marking_mode_flag',
        ('adaptive_ref_pic_marking_mode_flag', 'u1', 1),
        ('memory_management_control_operation', 'ue', operation),
        *((name, 'ue', 0) for name in arguments), ('end_of_marking', 'ue', 0))


#: An I slice of an IDR picture.
IDR_SLICE = [
    ('first_mb_in_slice', 'ue', 0), ('slice_type', 'ue', 7),
    ('pic_parameter_set_id', 'ue', 0), ('frame_num', 'u4', 0), ('idr_pic_id', 'ue', 0),
    ('pic_order_cnt_lsb', 'u6', 0), ('no_output_of_prior_pics_flag', 'u1', 0),
    ('long_term_reference_flag', 'u1', 0), ('slice_qp_delta', 'se', 0),
    ('disable_deblocking_filter_idc', 'ue', 1), ('slice_data', 'u8', 0xA5),
]

#: A P slice with a weight for its one reference, and the PPS that asks for it.
P_WEIGHTED = replaced(
    P_SLICE, 'adaptive_ref_pic_marking_mode_flag',
    ('luma_log2_weight_denom', 'ue', 0), ('chroma_log2_weight_denom', 'ue', 0),
    ('luma_weight_l0_flag', 'u1', 1), ('luma_weight', 'se', 1), ('luma_offset', 'se', 0),
    ('chroma_weight_l0_flag', 'u1', 1), ('chroma_weight', 'se', 1),
    ('chroma_offset', 'se', 0), ('cr_weight', 'se', 1), ('cr_offset', 'se', 0),
    ('adaptive_ref_pic_marking_mode_flag', 'u1', 0))


def slice_header(syntax, changed=None, *, pps=None, sps=None, idr=False):
    """The header `syntax` writes, parsed against :data:`SPS` and :data:`PPS`
    with the elements `sps` and `pps` change."""
    held = sequences(**(sps or {}))
    picture = parse_pps(written(PPS, pps or {}), held)
    unit = parse_nal(nal_unit(written(syntax, changed or {}), 5 if idr else 1, 2))
    return parse_slice_header(unit, held, {picture.pic_parameter_set_id: picture})


@pytest.mark.parametrize(('syntax', 'element', 'value'), [
    (P_SLICE, 'first_mb_in_slice', 16),
    (P_SLICE, 'first_mb_in_slice', 70000),
    (P_SLICE, 'first_mb_in_slice', UE_MOST),
    (P_SLICE, 'slice_type', 10),
    (P_SLICE, 'pic_parameter_set_id', 256),
    (P_SLICE, 'cabac_init_idc', 3),
    (P_SLICE, 'slice_qp_delta', 26),
    (P_SLICE, 'slice_qp_delta', -27),
    (P_SLICE, 'slice_qp_delta', 200),
    (P_SLICE, 'disable_deblocking_filter_idc', 3),
    (P_SLICE, 'slice_alpha_c0_offset_div2', -7),
    (P_SLICE, 'slice_alpha_c0_offset_div2', -200),
    (P_SLICE, 'slice_beta_offset_div2', 7),
    (P_OVERRIDE, 'num_ref_idx_l0_active_minus1', 16),
    (P_OVERRIDE, 'num_ref_idx_l0_active_minus1', 200),
    (P_OVERRIDE, 'num_ref_idx_l0_active_minus1', HUGE),
    (P_MODIFIED, 'modification_of_pic_nums_idc', 4),
    (P_MODIFIED, 'abs_diff_pic_num_minus1', 16),
    (P_LONG_TERM_MODIFIED, 'long_term_pic_num', 4),
    (P_WEIGHTED, 'luma_log2_weight_denom', 8),
    (P_WEIGHTED, 'luma_log2_weight_denom', HUGE),
    (P_WEIGHTED, 'chroma_log2_weight_denom', 8),
    (P_WEIGHTED, 'luma_weight', 128),
    (P_WEIGHTED, 'luma_offset', -129),
    (P_WEIGHTED, 'chroma_weight', -129),
    (P_WEIGHTED, 'chroma_offset', 128),
    (marking(1, 'difference_of_pic_nums_minus1'), 'difference_of_pic_nums_minus1', 16),
    (marking(2, 'long_term_pic_num'), 'long_term_pic_num', 4),
    (marking(3, 'difference_of_pic_nums_minus1', 'long_term_frame_idx'),
     'long_term_frame_idx', 4),
    (marking(6, 'long_term_frame_idx'), 'long_term_frame_idx', 4),
    (marking(6, 'long_term_frame_idx'), 'long_term_frame_idx', UE_MOST),
    (marking(4, 'max_long_term_frame_idx_plus1'), 'max_long_term_frame_idx_plus1', 5),
    (marking(7), 'memory_management_control_operation', 7),
])
def test_a_slice_header_element_outside_its_range_is_refused(syntax, element, value):
    with pytest.raises(StreamError, match=f'{element} is {value}, outside'):
        slice_header(syntax, {element: value}, pps={'weighted_pred_flag': 1}
                     if syntax is P_WEIGHTED else None)


def test_an_idr_pic_id_past_sixteen_bits_is_refused():
    assert slice_header(IDR_SLICE, {'idr_pic_id': 65535}, idr=True).idr_pic_id == 65535
    with pytest.raises(StreamError, match='idr_pic_id is 65536'):
        slice_header(IDR_SLICE, {'idr_pic_id': 65536}, idr=True)


def test_a_frame_has_sixteen_active_references_at_most():
    """H1: a frame's lists hold 16 entries, where the syntax and libva's
    arrays hold 32; the picture parameter set's default is held to it too."""
    assert slice_header(P_OVERRIDE, {'num_ref_idx_l0_active_minus1': 15}
                        ).num_ref_idx_l0_active_minus1 == 15
    with pytest.raises(StreamError,
                       match='num_ref_idx_l0_default_active_minus1 is 16, outside 0 to 15'):
        slice_header(P_SLICE, pps={'num_ref_idx_l0_default_active_minus1': 16})


def test_a_predicted_slice_of_an_idr_picture_is_refused():
    with pytest.raises(StreamError, match='a P slice in an IDR picture'):
        slice_header(replaced(IDR_SLICE, 'slice_type', ('slice_type', 'ue', 5)), idr=True)


def test_a_predicted_slice_where_no_reference_is_kept_is_refused():
    with pytest.raises(StreamError, match='keeps no reference frame'):
        slice_header(P_SLICE, sps={'max_num_ref_frames': 0})


def test_the_slice_quantiser_is_held_to_the_range_with_the_picture_sets():
    """SliceQPY is 26 + pic_init_qp_minus26 + slice_qp_delta, from 0 to 51."""
    assert slice_header(P_SLICE, {'slice_qp_delta': 31},
                        pps={'pic_init_qp_minus26': -6}).slice_qp_delta == 31
    with pytest.raises(StreamError, match='slice_qp_delta is 32, outside -20 to 31'):
        slice_header(P_SLICE, {'slice_qp_delta': 32}, pps={'pic_init_qp_minus26': -6})


def test_a_picture_set_read_against_a_deeper_sequence_is_held_to_this_one():
    """A sequence parameter set may be sent again, shallower, after the
    picture parameter set was read against it."""
    deep = sequences(bit_depth_luma_minus8=2)
    pps = parse_pps(written(PPS, {'pic_init_qp_minus26': -38}), deep)
    unit = parse_nal(nal_unit(written(P_SLICE, {}), 1, 2))
    with pytest.raises(StreamError, match='pic_init_qp_minus26 is -38, outside -26 to 25'):
        parse_slice_header(unit, sequences(), {0: pps})


def test_more_modifications_than_the_list_has_entries_are_refused():
    """H7: a list of one entry takes one modification, so the operations are
    bounded by the active count and not by the length of the unit."""
    twice = replaced(
        P_MODIFIED, 'end_of_modifications', ('again', 'ue', 0), ('by', 'ue', 0),
        ('end_of_modifications', 'ue', 3))
    with pytest.raises(StreamError, match='more than 1 reference list modifications'):
        slice_header(twice)


def test_more_marking_operations_than_can_act_are_refused():
    operations = [entry for _ in range(67)
                  for entry in (('memory_management_control_operation', 'ue', 1),
                                ('difference_of_pic_nums_minus1', 'ue', 0))]
    syntax = replaced(P_SLICE, 'adaptive_ref_pic_marking_mode_flag',
                      ('adaptive_ref_pic_marking_mode_flag', 'u1', 1), *operations,
                      ('end_of_marking', 'ue', 0))
    with pytest.raises(StreamError, match='more than 66 memory management'):
        slice_header(syntax)


def test_a_header_longer_than_libvas_bit_offset_is_refused(monkeypatch):
    """H7: slice_data_bit_offset is 16 bits wide. With every count in a header
    bounded, none parsed here is that long, so the limit is lowered to one
    this header passes."""
    assert slice_header(P_SLICE).header_bits == 33
    monkeypatch.setattr(h264_parse, 'MAX_HEADER_BITS', 32)
    with pytest.raises(StreamError,
                       match='the length of the slice header in bits is 33, outside 8 to 32'):
        slice_header(P_SLICE)
