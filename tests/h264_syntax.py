"""H.264 syntax written for the tests, in the order of the specification's tables.

Each helper writes one syntax structure of ITU-T H.264 with :class:`BitWriter`,
with keyword arguments for the variations the tests need. The parser's tests
read these back field by field, and the decoded picture buffer's tests parse
them into the headers they drive it with.
"""
from pyopengl_video.vaapi.h264 import BitWriter, nal_unit
from pyopengl_video.vaapi.h264_parse import SliceHeader, parse_nal, parse_pps, parse_sps


def write_scaling_list(writer, values):
    """scaling_list() of 7.3.2.1.1.1, for a list given in zig-zag order."""
    last = 8
    for value in values:
        delta = (value - last + 128) % 256 - 128
        writer.se(delta)
        last = value


def write_default_scaling_list(writer):
    """A list whose first delta makes nextScale 0, asking for the default."""
    writer.se(-8)


def sps_bytes(*, profile_idc=100, poc_type=0, scaling=None, vui=None,
              frame_mbs_only=1, crop=None, width_mbs=4, height_mbs=4,
              max_num_ref_frames=4, gaps=0, constraint_flags=0, level_idc=30,
              chroma_format_idc=1, seq_parameter_set_id=0, bit_depth=0):
    """An SPS in the order of 7.3.2.1.1, with the variations under test."""
    writer = BitWriter()
    writer.u(profile_idc, 8)
    writer.u(constraint_flags, 8)
    writer.u(level_idc, 8)
    writer.ue(seq_parameter_set_id)
    if profile_idc in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135):
        writer.ue(chroma_format_idc)
        if chroma_format_idc == 3:
            writer.flag(0)                      # separate_colour_plane_flag
        writer.ue(bit_depth)
        writer.ue(bit_depth)
        writer.flag(0)                          # qpprime_y_zero_transform_bypass
        writer.flag(scaling is not None)
        if scaling is not None:
            for entry in scaling:
                writer.flag(entry is not None)
                if entry == 'default':
                    write_default_scaling_list(writer)
                elif entry is not None:
                    write_scaling_list(writer, entry)
    writer.ue(0)                                # log2_max_frame_num_minus4
    writer.ue(poc_type)
    if poc_type == 0:
        writer.ue(2)                            # log2_max_pic_order_cnt_lsb_minus4
    elif poc_type == 1:
        writer.flag(0)                          # delta_pic_order_always_zero
        writer.se(-1)                           # offset_for_non_ref_pic
        writer.se(0)                            # offset_for_top_to_bottom_field
        writer.ue(2)                            # num_ref_frames_in_pic_order_cnt_cycle
        writer.se(2)
        writer.se(4)
    writer.ue(max_num_ref_frames)
    writer.flag(gaps)
    writer.ue(width_mbs - 1)
    writer.ue(height_mbs - 1)
    writer.flag(frame_mbs_only)
    if not frame_mbs_only:
        writer.flag(1)                          # mb_adaptive_frame_field_flag
    writer.flag(1)                              # direct_8x8_inference_flag
    writer.flag(crop is not None)
    if crop is not None:
        for value in crop:
            writer.ue(value)
    writer.flag(vui is not None)
    if vui is not None:
        vui(writer)
    return writer.rbsp()


def full_vui(writer):
    """Every optional part of E.1.1, HRD parameters included."""
    writer.flag(1)                              # aspect_ratio_info_present
    writer.u(255, 8)                            # Extended_SAR
    writer.u(4, 16)
    writer.u(3, 16)
    writer.flag(1)                              # overscan_info_present
    writer.flag(1)
    writer.flag(1)                              # video_signal_type_present
    writer.u(5, 3)
    writer.flag(1)                              # video_full_range_flag
    writer.flag(1)                              # colour_description_present
    writer.u(6, 8)
    writer.u(6, 8)
    writer.u(6, 8)
    writer.flag(1)                              # chroma_loc_info_present
    writer.ue(2)
    writer.ue(2)
    writer.flag(1)                              # timing_info_present
    writer.u(1001, 32)
    writer.u(48000, 32)
    writer.flag(0)
    for _ in range(2):                          # NAL and VCL HRD
        writer.flag(1)
        writer.ue(1)                            # cpb_cnt_minus1
        writer.u(4, 4)
        writer.u(5, 4)
        for _ in range(2):
            writer.ue(1000)
            writer.ue(2000)
            writer.flag(0)
        for _ in range(4):
            writer.u(23, 5)
    writer.flag(0)                              # low_delay_hrd_flag
    writer.flag(0)                              # pic_struct_present
    writer.flag(1)                              # bitstream_restriction
    writer.flag(1)
    writer.ue(2)
    writer.ue(1)
    writer.ue(16)
    writer.ue(16)
    writer.ue(2)                                # max_num_reorder_frames
    writer.ue(4)                                # max_dec_frame_buffering


def write_pps_head(writer, *, weighted_pred=0, weighted_bipred=0, refs=(0, 0),
                   bottom_field_pic_order=0, redundant=0, slice_groups=0,
                   entropy=1):
    """The part of 7.3.2.2 every PPS has, up to the optional tail."""
    writer.ue(0)                                # pic_parameter_set_id
    writer.ue(0)                                # seq_parameter_set_id
    writer.flag(entropy)
    writer.flag(bottom_field_pic_order)
    writer.ue(slice_groups)
    if slice_groups:
        writer.ue(0)                            # slice_group_map_type
        for _ in range(slice_groups + 1):
            writer.ue(10)                       # run_length_minus1
    writer.ue(refs[0])
    writer.ue(refs[1])
    writer.flag(weighted_pred)
    writer.u(weighted_bipred, 2)
    writer.se(-3)                               # pic_init_qp_minus26
    writer.se(0)
    writer.se(2)                                # chroma_qp_index_offset
    writer.flag(1)                              # deblocking_filter_control_present
    writer.flag(0)
    writer.flag(redundant)


def slice_unit(write, *, nal_ref_idc=2, nal_unit_type=1):
    writer = BitWriter()
    write(writer)
    return parse_nal(nal_unit(writer.rbsp(), nal_unit_type, nal_ref_idc))


def restriction_vui(reorder, buffering):
    """A VUI holding only bitstream_restriction, stating the reorder depth."""
    def write(writer):
        for _ in range(4):
            writer.flag(0)                      # aspect, overscan, signal, chroma
        writer.flag(0)                          # timing_info_present
        writer.flag(0)                          # nal_hrd
        writer.flag(0)                          # vcl_hrd
        writer.flag(0)                          # pic_struct_present
        writer.flag(1)                          # bitstream_restriction
        writer.flag(1)
        writer.ue(0)
        writer.ue(0)
        writer.ue(16)
        writer.ue(16)
        writer.ue(reorder)
        writer.ue(buffering)
    return write


def sequence(**options):
    """A parsed SPS from :func:`sps_bytes`."""
    return parse_sps(sps_bytes(**options))


def picture_set(sps, **options):
    """A parsed PPS from :func:`write_pps_head`, against `sps`."""
    writer = BitWriter()
    write_pps_head(writer, **options)
    return parse_pps(writer.rbsp(), {sps.seq_parameter_set_id: sps})


def header(sps, pps, *, kind='P', frame_num=0, poc_lsb=0, nal_ref_idc=2,
           idr=False, **fields):
    """A slice header made directly, its other fields at their inferred values."""
    values = dict(
        nal_ref_idc=nal_ref_idc, nal_unit_type=5 if idr else 1,
        first_mb_in_slice=0, slice_type='PBI'.index(kind),
        pic_parameter_set_id=pps.pic_parameter_set_id, frame_num=frame_num,
        field_pic_flag=0, bottom_field_flag=0, idr_pic_id=0,
        pic_order_cnt_lsb=poc_lsb, delta_pic_order_cnt_bottom=0,
        delta_pic_order_cnt=(0, 0), redundant_pic_cnt=0,
        direct_spatial_mv_pred_flag=0, num_ref_idx_active_override_flag=0,
        num_ref_idx_l0_active_minus1=pps.num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_active_minus1=pps.num_ref_idx_l1_default_active_minus1,
        ref_pic_list_modification_l0=(), ref_pic_list_modification_l1=(),
        pred_weight_table=None, no_output_of_prior_pics_flag=0,
        long_term_reference_flag=0, adaptive_ref_pic_marking_mode_flag=0,
        memory_management=(), cabac_init_idc=0, slice_qp_delta=0,
        disable_deblocking_filter_idc=0, slice_alpha_c0_offset_div2=0,
        slice_beta_offset_div2=0, header_bits=40, sps=sps, pps=pps,
    )
    values.update(fields)
    if values['memory_management']:
        values['adaptive_ref_pic_marking_mode_flag'] = 1
    return SliceHeader(**values)
