"""ctypes over libva's HEVC structures and constants.

The layouts are checked against the libva headers by ``tests/test_va_abi.py``,
from the recording ``tools/record_va_abi.py`` makes, as
:mod:`pyopengl_video.vaapi.api` is.
"""
from __future__ import annotations

import ctypes

from pyopengl_video.vaapi.api import (
    PADDING_LOW,
    PADDING_MEDIUM,
    VA_INVALID_SURFACE,
    VABufferID,
    VASurfaceID,
)

# ------------------------------------------------------------------ decoding

VA_PICTURE_HEVC_INVALID = 0x00000001
VA_PICTURE_HEVC_FIELD_PIC = 0x00000002
VA_PICTURE_HEVC_BOTTOM_FIELD = 0x00000004
VA_PICTURE_HEVC_LONG_TERM_REFERENCE = 0x00000008
#: Which subset of the reference picture set a picture the current one
#: predicts from is in. At most one is set on an entry.
VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE = 0x00000010
VA_PICTURE_HEVC_RPS_ST_CURR_AFTER = 0x00000020
VA_PICTURE_HEVC_RPS_LT_CURR = 0x00000040

#: An entry of ``RefPicList`` that names no picture, and ``collocated_ref_idx``
#: in a slice without temporal motion vector prediction.
NO_REFERENCE = 0xFF

#: How many reference frames a picture parameter buffer describes, and how
#: many entries each reference list of a slice has.
REFERENCE_FRAMES = 15


class VAPictureHEVC(ctypes.Structure):
    """One entry of the decoded picture buffer, or the picture being decoded.

    picture_id -- the surface the driver reconstructs into, or reads from
    pic_order_cnt -- the picture's PicOrderCntVal
    flags -- ``VA_PICTURE_HEVC_*`` bits, ``VA_PICTURE_HEVC_INVALID`` for an
        unused slot
    """

    _fields_ = [
        ('picture_id', VASurfaceID),
        ('pic_order_cnt', ctypes.c_int32),
        ('flags', ctypes.c_uint32),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]

    @classmethod
    def unused(cls) -> VAPictureHEVC:
        """A slot holding no picture."""
        return cls(picture_id=VA_INVALID_SURFACE, pic_order_cnt=0,
                   flags=VA_PICTURE_HEVC_INVALID)


class _DecPicFieldsBits(ctypes.Structure):
    _fields_ = [
        ('chroma_format_idc', ctypes.c_uint32, 2),
        ('separate_colour_plane_flag', ctypes.c_uint32, 1),
        ('pcm_enabled_flag', ctypes.c_uint32, 1),
        ('scaling_list_enabled_flag', ctypes.c_uint32, 1),
        ('transform_skip_enabled_flag', ctypes.c_uint32, 1),
        ('amp_enabled_flag', ctypes.c_uint32, 1),
        ('strong_intra_smoothing_enabled_flag', ctypes.c_uint32, 1),
        ('sign_data_hiding_enabled_flag', ctypes.c_uint32, 1),
        ('constrained_intra_pred_flag', ctypes.c_uint32, 1),
        ('cu_qp_delta_enabled_flag', ctypes.c_uint32, 1),
        ('weighted_pred_flag', ctypes.c_uint32, 1),
        ('weighted_bipred_flag', ctypes.c_uint32, 1),
        ('transquant_bypass_enabled_flag', ctypes.c_uint32, 1),
        ('tiles_enabled_flag', ctypes.c_uint32, 1),
        ('entropy_coding_sync_enabled_flag', ctypes.c_uint32, 1),
        ('pps_loop_filter_across_slices_enabled_flag', ctypes.c_uint32, 1),
        ('loop_filter_across_tiles_enabled_flag', ctypes.c_uint32, 1),
        ('pcm_loop_filter_disabled_flag', ctypes.c_uint32, 1),
        ('NoPicReorderingFlag', ctypes.c_uint32, 1),
        ('NoBiPredFlag', ctypes.c_uint32, 1),
        ('ReservedBits', ctypes.c_uint32, 11),
    ]


class _DecPicFields(ctypes.Union):
    _fields_ = [('bits', _DecPicFieldsBits), ('value', ctypes.c_uint32)]


class _DecSliceParsingBits(ctypes.Structure):
    _fields_ = [
        ('lists_modification_present_flag', ctypes.c_uint32, 1),
        ('long_term_ref_pics_present_flag', ctypes.c_uint32, 1),
        ('sps_temporal_mvp_enabled_flag', ctypes.c_uint32, 1),
        ('cabac_init_present_flag', ctypes.c_uint32, 1),
        ('output_flag_present_flag', ctypes.c_uint32, 1),
        ('dependent_slice_segments_enabled_flag', ctypes.c_uint32, 1),
        ('pps_slice_chroma_qp_offsets_present_flag', ctypes.c_uint32, 1),
        ('sample_adaptive_offset_enabled_flag', ctypes.c_uint32, 1),
        ('deblocking_filter_override_enabled_flag', ctypes.c_uint32, 1),
        ('pps_disable_deblocking_filter_flag', ctypes.c_uint32, 1),
        ('slice_segment_header_extension_present_flag', ctypes.c_uint32, 1),
        ('RapPicFlag', ctypes.c_uint32, 1),
        ('IdrPicFlag', ctypes.c_uint32, 1),
        ('IntraPicFlag', ctypes.c_uint32, 1),
        ('ReservedBits', ctypes.c_uint32, 18),
    ]


class _DecSliceParsing(ctypes.Union):
    _fields_ = [('bits', _DecSliceParsingBits), ('value', ctypes.c_uint32)]


class VAPictureParameterBufferHEVC(ctypes.Structure):
    """What a decoder is told about the picture it is about to decode.

    The sequence and picture parameter sets in force, the reference pictures
    in the decoded picture buffer with the subset of the reference picture set
    each is in, and the picture being decoded as ``CurrPic``.

    column_width_minus1, row_height_minus1 -- the tile grid in coding tree
        blocks, filled in for a uniformly spaced grid too
    st_rps_bits -- how many bits the short-term reference picture set coded
        in the slice segment header takes, emulation prevention removed; zero
        where the header names one of the sequence's sets
    """

    _fields_ = [
        ('CurrPic', VAPictureHEVC),
        ('ReferenceFrames', VAPictureHEVC * REFERENCE_FRAMES),
        ('pic_width_in_luma_samples', ctypes.c_uint16),
        ('pic_height_in_luma_samples', ctypes.c_uint16),
        ('pic_fields', _DecPicFields),
        ('sps_max_dec_pic_buffering_minus1', ctypes.c_uint8),
        ('bit_depth_luma_minus8', ctypes.c_uint8),
        ('bit_depth_chroma_minus8', ctypes.c_uint8),
        ('pcm_sample_bit_depth_luma_minus1', ctypes.c_uint8),
        ('pcm_sample_bit_depth_chroma_minus1', ctypes.c_uint8),
        ('log2_min_luma_coding_block_size_minus3', ctypes.c_uint8),
        ('log2_diff_max_min_luma_coding_block_size', ctypes.c_uint8),
        ('log2_min_transform_block_size_minus2', ctypes.c_uint8),
        ('log2_diff_max_min_transform_block_size', ctypes.c_uint8),
        ('log2_min_pcm_luma_coding_block_size_minus3', ctypes.c_uint8),
        ('log2_diff_max_min_pcm_luma_coding_block_size', ctypes.c_uint8),
        ('max_transform_hierarchy_depth_intra', ctypes.c_uint8),
        ('max_transform_hierarchy_depth_inter', ctypes.c_uint8),
        ('init_qp_minus26', ctypes.c_int8),
        ('diff_cu_qp_delta_depth', ctypes.c_uint8),
        ('pps_cb_qp_offset', ctypes.c_int8),
        ('pps_cr_qp_offset', ctypes.c_int8),
        ('log2_parallel_merge_level_minus2', ctypes.c_uint8),
        ('num_tile_columns_minus1', ctypes.c_uint8),
        ('num_tile_rows_minus1', ctypes.c_uint8),
        ('column_width_minus1', ctypes.c_uint16 * 19),
        ('row_height_minus1', ctypes.c_uint16 * 21),
        ('slice_parsing_fields', _DecSliceParsing),
        ('log2_max_pic_order_cnt_lsb_minus4', ctypes.c_uint8),
        ('num_short_term_ref_pic_sets', ctypes.c_uint8),
        ('num_long_term_ref_pic_sps', ctypes.c_uint8),
        ('num_ref_idx_l0_default_active_minus1', ctypes.c_uint8),
        ('num_ref_idx_l1_default_active_minus1', ctypes.c_uint8),
        ('pps_beta_offset_div2', ctypes.c_int8),
        ('pps_tc_offset_div2', ctypes.c_int8),
        ('num_extra_slice_header_bits', ctypes.c_uint8),
        ('st_rps_bits', ctypes.c_uint32),
        ('va_reserved', ctypes.c_uint32 * PADDING_MEDIUM),
    ]


class _LongSliceFlagsBits(ctypes.Structure):
    _fields_ = [
        ('LastSliceOfPic', ctypes.c_uint32, 1),
        ('dependent_slice_segment_flag', ctypes.c_uint32, 1),
        ('slice_type', ctypes.c_uint32, 2),
        ('color_plane_id', ctypes.c_uint32, 2),
        ('slice_sao_luma_flag', ctypes.c_uint32, 1),
        ('slice_sao_chroma_flag', ctypes.c_uint32, 1),
        ('mvd_l1_zero_flag', ctypes.c_uint32, 1),
        ('cabac_init_flag', ctypes.c_uint32, 1),
        ('slice_temporal_mvp_enabled_flag', ctypes.c_uint32, 1),
        ('slice_deblocking_filter_disabled_flag', ctypes.c_uint32, 1),
        ('collocated_from_l0_flag', ctypes.c_uint32, 1),
        ('slice_loop_filter_across_slices_enabled_flag', ctypes.c_uint32, 1),
        ('reserved', ctypes.c_uint32, 18),
    ]


class _LongSliceFlags(ctypes.Union):
    _fields_ = [('value', ctypes.c_uint32), ('fields', _LongSliceFlagsBits)]


class VASliceParameterBufferHEVC(ctypes.Structure):
    """One slice segment's header as a decoder is given it.

    slice_data_byte_offset -- where slice_segment_data() begins, in bytes from
        the start of the NAL unit, its two header bytes included, counted with
        emulation prevention removed; the slice data buffer holds the unit as
        the stream carries it
    RefPicList -- the two reference lists, each entry an index into the
        picture's ``ReferenceFrames`` or :data:`NO_REFERENCE`
    entry_offset_to_subset_array -- how many entry points the picture's
        earlier slice segments have
    slice_data_num_emu_prevn_bytes -- emulation prevention bytes within the
        slice segment header
    """

    _fields_ = [
        ('slice_data_size', ctypes.c_uint32),
        ('slice_data_offset', ctypes.c_uint32),
        ('slice_data_flag', ctypes.c_uint32),
        ('slice_data_byte_offset', ctypes.c_uint32),
        ('slice_segment_address', ctypes.c_uint32),
        ('RefPicList', (ctypes.c_uint8 * REFERENCE_FRAMES) * 2),
        ('LongSliceFlags', _LongSliceFlags),
        ('collocated_ref_idx', ctypes.c_uint8),
        ('num_ref_idx_l0_active_minus1', ctypes.c_uint8),
        ('num_ref_idx_l1_active_minus1', ctypes.c_uint8),
        ('slice_qp_delta', ctypes.c_int8),
        ('slice_cb_qp_offset', ctypes.c_int8),
        ('slice_cr_qp_offset', ctypes.c_int8),
        ('slice_beta_offset_div2', ctypes.c_int8),
        ('slice_tc_offset_div2', ctypes.c_int8),
        ('luma_log2_weight_denom', ctypes.c_uint8),
        ('delta_chroma_log2_weight_denom', ctypes.c_int8),
        ('delta_luma_weight_l0', ctypes.c_int8 * REFERENCE_FRAMES),
        ('luma_offset_l0', ctypes.c_int8 * REFERENCE_FRAMES),
        ('delta_chroma_weight_l0', (ctypes.c_int8 * 2) * REFERENCE_FRAMES),
        ('ChromaOffsetL0', (ctypes.c_int8 * 2) * REFERENCE_FRAMES),
        ('delta_luma_weight_l1', ctypes.c_int8 * REFERENCE_FRAMES),
        ('luma_offset_l1', ctypes.c_int8 * REFERENCE_FRAMES),
        ('delta_chroma_weight_l1', (ctypes.c_int8 * 2) * REFERENCE_FRAMES),
        ('ChromaOffsetL1', (ctypes.c_int8 * 2) * REFERENCE_FRAMES),
        ('five_minus_max_num_merge_cand', ctypes.c_uint8),
        ('num_entry_point_offsets', ctypes.c_uint16),
        ('entry_offset_to_subset_array', ctypes.c_uint16),
        ('slice_data_num_emu_prevn_bytes', ctypes.c_uint16),
        ('va_reserved', ctypes.c_uint32 * (PADDING_LOW - 2)),
    ]


class VAIQMatrixBufferHEVC(ctypes.Structure):
    """The scaling factors in force, each list in raster order.

    ScalingList4x4, ScalingList8x8, ScalingList16x16 -- six lists each:
        intra Y, Cb, Cr, then inter Y, Cb, Cr
    ScalingList32x32 -- the intra and the inter luma list
    ScalingListDC16x16, ScalingListDC32x32 -- the DC factor of each 16x16 and
        32x32 list, whose other factors are spread from its 8x8 grid
    """

    _fields_ = [
        ('ScalingList4x4', (ctypes.c_uint8 * 16) * 6),
        ('ScalingList8x8', (ctypes.c_uint8 * 64) * 6),
        ('ScalingList16x16', (ctypes.c_uint8 * 64) * 6),
        ('ScalingList32x32', (ctypes.c_uint8 * 64) * 2),
        ('ScalingListDC16x16', ctypes.c_uint8 * 6),
        ('ScalingListDC32x32', ctypes.c_uint8 * 2),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


# ------------------------------------------------------------------ encoding
#
# ``va/va_enc_hevc.h``: the sequence parameter buffer an encoder is given at a
# key picture, and the picture and slice parameter buffers it is given for
# every picture.

#: What ``VA_PADDING_HIGH`` expands to, the tail of the picture parameter buffer.
PADDING_HIGH = 16

#: Configuration attributes: the coding tools the driver's HEVC encoder has, as
#: :class:`VAConfigAttribValEncHEVCFeatures`, and the block sizes it codes
#: with, as :class:`VAConfigAttribValEncHEVCBlockSizes`.
VAConfigAttribEncHEVCFeatures = 50
VAConfigAttribEncHEVCBlockSizes = 51

#: A configuration attribute: which way a predicted picture may predict, as
#: ``VA_PREDICTION_DIRECTION_*`` bits.
VAConfigAttribPredictionDirection = 39
VA_PREDICTION_DIRECTION_PREVIOUS = 0x00000001
VA_PREDICTION_DIRECTION_FUTURE = 0x00000002
#: Both reference lists of a predicted picture must hold a picture, so one
#: predicting from earlier pictures alone is coded as a B picture whose second
#: list repeats its first.
VA_PREDICTION_DIRECTION_BI_NOT_EMPTY = 0x00000004

#: What a field of :class:`VAConfigAttribValEncHEVCFeatures` holds: the tool
#: is never used, may be used, or is always used.
VA_FEATURE_NOT_SUPPORTED = 0
VA_FEATURE_SUPPORTED = 1
VA_FEATURE_REQUIRED = 2

#: ``VAEncPackedHeaderTypeHEVC``: the video and sequence parameter sets go in
#: one packed sequence header, the picture parameter set in a packed picture
#: header and the slice segment header in a packed slice header.
VAEncPackedHeaderHEVC_VPS = 1
VAEncPackedHeaderHEVC_SPS = 1
VAEncPackedHeaderHEVC_PPS = 2
VAEncPackedHeaderHEVC_Slice = 3

#: ``coding_type`` of :class:`VAEncPictureParameterBufferHEVC`.
CODING_TYPE_I = 1
CODING_TYPE_P = 2
CODING_TYPE_B = 3


class _EncFeaturesBits(ctypes.Structure):
    _fields_ = [
        ('separate_colour_planes', ctypes.c_uint32, 2),
        ('scaling_lists', ctypes.c_uint32, 2),
        ('amp', ctypes.c_uint32, 2),
        ('sao', ctypes.c_uint32, 2),
        ('pcm', ctypes.c_uint32, 2),
        ('temporal_mvp', ctypes.c_uint32, 2),
        ('strong_intra_smoothing', ctypes.c_uint32, 2),
        ('dependent_slices', ctypes.c_uint32, 2),
        ('sign_data_hiding', ctypes.c_uint32, 2),
        ('constrained_intra_pred', ctypes.c_uint32, 2),
        ('transform_skip', ctypes.c_uint32, 2),
        ('cu_qp_delta', ctypes.c_uint32, 2),
        ('weighted_prediction', ctypes.c_uint32, 2),
        ('transquant_bypass', ctypes.c_uint32, 2),
        ('deblocking_filter_disable', ctypes.c_uint32, 2),
        ('reserved', ctypes.c_uint32, 2),
    ]


class VAConfigAttribValEncHEVCFeatures(ctypes.Union):
    """The value of ``VAConfigAttribEncHEVCFeatures``: a ``VA_FEATURE_*`` for each tool."""

    _fields_ = [('bits', _EncFeaturesBits), ('value', ctypes.c_uint32)]


class _EncBlockSizesBits(ctypes.Structure):
    _fields_ = [
        ('log2_max_coding_tree_block_size_minus3', ctypes.c_uint32, 2),
        ('log2_min_coding_tree_block_size_minus3', ctypes.c_uint32, 2),
        ('log2_min_luma_coding_block_size_minus3', ctypes.c_uint32, 2),
        ('log2_max_luma_transform_block_size_minus2', ctypes.c_uint32, 2),
        ('log2_min_luma_transform_block_size_minus2', ctypes.c_uint32, 2),
        ('max_max_transform_hierarchy_depth_inter', ctypes.c_uint32, 2),
        ('min_max_transform_hierarchy_depth_inter', ctypes.c_uint32, 2),
        ('max_max_transform_hierarchy_depth_intra', ctypes.c_uint32, 2),
        ('min_max_transform_hierarchy_depth_intra', ctypes.c_uint32, 2),
        ('log2_max_pcm_coding_block_size_minus3', ctypes.c_uint32, 2),
        ('log2_min_pcm_coding_block_size_minus3', ctypes.c_uint32, 2),
        ('reserved', ctypes.c_uint32, 10),
    ]


class VAConfigAttribValEncHEVCBlockSizes(ctypes.Union):
    """The value of ``VAConfigAttribEncHEVCBlockSizes``.

    The largest and smallest coding tree block, the smallest coding block, the
    largest and smallest transform block, and the range the transform
    hierarchy depths may be chosen from.
    """

    _fields_ = [('bits', _EncBlockSizesBits), ('value', ctypes.c_uint32)]


class _EncSeqFieldsBits(ctypes.Structure):
    _fields_ = [
        ('chroma_format_idc', ctypes.c_uint32, 2),
        ('separate_colour_plane_flag', ctypes.c_uint32, 1),
        ('bit_depth_luma_minus8', ctypes.c_uint32, 3),
        ('bit_depth_chroma_minus8', ctypes.c_uint32, 3),
        ('scaling_list_enabled_flag', ctypes.c_uint32, 1),
        ('strong_intra_smoothing_enabled_flag', ctypes.c_uint32, 1),
        ('amp_enabled_flag', ctypes.c_uint32, 1),
        ('sample_adaptive_offset_enabled_flag', ctypes.c_uint32, 1),
        ('pcm_enabled_flag', ctypes.c_uint32, 1),
        ('pcm_loop_filter_disabled_flag', ctypes.c_uint32, 1),
        ('sps_temporal_mvp_enabled_flag', ctypes.c_uint32, 1),
        ('low_delay_seq', ctypes.c_uint32, 1),
        ('hierachical_flag', ctypes.c_uint32, 1),
        ('reserved_bits', ctypes.c_uint32, 14),
    ]


class _EncSeqFields(ctypes.Union):
    _fields_ = [('bits', _EncSeqFieldsBits), ('value', ctypes.c_uint32)]


class _EncVuiFieldsBits(ctypes.Structure):
    _fields_ = [
        ('aspect_ratio_info_present_flag', ctypes.c_uint32, 1),
        ('neutral_chroma_indication_flag', ctypes.c_uint32, 1),
        ('field_seq_flag', ctypes.c_uint32, 1),
        ('vui_timing_info_present_flag', ctypes.c_uint32, 1),
        ('bitstream_restriction_flag', ctypes.c_uint32, 1),
        ('tiles_fixed_structure_flag', ctypes.c_uint32, 1),
        ('motion_vectors_over_pic_boundaries_flag', ctypes.c_uint32, 1),
        ('restricted_ref_pic_lists_flag', ctypes.c_uint32, 1),
        ('log2_max_mv_length_horizontal', ctypes.c_uint32, 5),
        ('log2_max_mv_length_vertical', ctypes.c_uint32, 5),
    ]


class _EncVuiFields(ctypes.Union):
    _fields_ = [('bits', _EncVuiFieldsBits), ('value', ctypes.c_uint32)]


class _EncSeqSccBits(ctypes.Structure):
    _fields_ = [
        ('palette_mode_enabled_flag', ctypes.c_uint32, 1),
        ('reserved', ctypes.c_uint32, 31),
    ]


class _EncSeqScc(ctypes.Union):
    _fields_ = [('bits', _EncSeqSccBits), ('value', ctypes.c_uint32)]


class VAEncSequenceParameterBufferHEVC(ctypes.Structure):
    """What an encoder is told about the whole sequence, at a key picture.

    general_level_idc -- thirty times the level number
    intra_period, intra_idr_period -- pictures between intra pictures and
        between IDR pictures
    ip_period -- pictures from one I or P picture to the next: one where no
        picture is held back
    pic_width_in_luma_samples, pic_height_in_luma_samples -- the coded size
    low_delay_seq -- every picture predicts from earlier pictures alone

    The structure has no conformance window: a driver learns one from the
    packed sequence parameter set.
    """

    _fields_ = [
        ('general_profile_idc', ctypes.c_uint8),
        ('general_level_idc', ctypes.c_uint8),
        ('general_tier_flag', ctypes.c_uint8),
        ('intra_period', ctypes.c_uint32),
        ('intra_idr_period', ctypes.c_uint32),
        ('ip_period', ctypes.c_uint32),
        ('bits_per_second', ctypes.c_uint32),
        ('pic_width_in_luma_samples', ctypes.c_uint16),
        ('pic_height_in_luma_samples', ctypes.c_uint16),
        ('seq_fields', _EncSeqFields),
        ('log2_min_luma_coding_block_size_minus3', ctypes.c_uint8),
        ('log2_diff_max_min_luma_coding_block_size', ctypes.c_uint8),
        ('log2_min_transform_block_size_minus2', ctypes.c_uint8),
        ('log2_diff_max_min_transform_block_size', ctypes.c_uint8),
        ('max_transform_hierarchy_depth_inter', ctypes.c_uint8),
        ('max_transform_hierarchy_depth_intra', ctypes.c_uint8),
        ('pcm_sample_bit_depth_luma_minus1', ctypes.c_uint32),
        ('pcm_sample_bit_depth_chroma_minus1', ctypes.c_uint32),
        ('log2_min_pcm_luma_coding_block_size_minus3', ctypes.c_uint32),
        ('log2_max_pcm_luma_coding_block_size_minus3', ctypes.c_uint32),
        ('vui_parameters_present_flag', ctypes.c_uint8),
        ('vui_fields', _EncVuiFields),
        ('aspect_ratio_idc', ctypes.c_uint8),
        ('sar_width', ctypes.c_uint32),
        ('sar_height', ctypes.c_uint32),
        ('vui_num_units_in_tick', ctypes.c_uint32),
        ('vui_time_scale', ctypes.c_uint32),
        ('min_spatial_segmentation_idc', ctypes.c_uint16),
        ('max_bytes_per_pic_denom', ctypes.c_uint8),
        ('max_bits_per_min_cu_denom', ctypes.c_uint8),
        ('scc_fields', _EncSeqScc),
        ('va_reserved', ctypes.c_uint32 * (PADDING_MEDIUM - 1)),
    ]


class _EncPicFieldsBits(ctypes.Structure):
    _fields_ = [
        ('idr_pic_flag', ctypes.c_uint32, 1),
        ('coding_type', ctypes.c_uint32, 3),
        ('reference_pic_flag', ctypes.c_uint32, 1),
        ('dependent_slice_segments_enabled_flag', ctypes.c_uint32, 1),
        ('sign_data_hiding_enabled_flag', ctypes.c_uint32, 1),
        ('constrained_intra_pred_flag', ctypes.c_uint32, 1),
        ('transform_skip_enabled_flag', ctypes.c_uint32, 1),
        ('cu_qp_delta_enabled_flag', ctypes.c_uint32, 1),
        ('weighted_pred_flag', ctypes.c_uint32, 1),
        ('weighted_bipred_flag', ctypes.c_uint32, 1),
        ('transquant_bypass_enabled_flag', ctypes.c_uint32, 1),
        ('tiles_enabled_flag', ctypes.c_uint32, 1),
        ('entropy_coding_sync_enabled_flag', ctypes.c_uint32, 1),
        ('loop_filter_across_tiles_enabled_flag', ctypes.c_uint32, 1),
        ('pps_loop_filter_across_slices_enabled_flag', ctypes.c_uint32, 1),
        ('scaling_list_data_present_flag', ctypes.c_uint32, 1),
        ('screen_content_flag', ctypes.c_uint32, 1),
        ('enable_gpu_weighted_prediction', ctypes.c_uint32, 1),
        ('no_output_of_prior_pics_flag', ctypes.c_uint32, 1),
        ('reserved', ctypes.c_uint32, 11),
    ]


class _EncPicFields(ctypes.Union):
    _fields_ = [('bits', _EncPicFieldsBits), ('value', ctypes.c_uint32)]


class _EncPicSccBits(ctypes.Structure):
    _fields_ = [
        ('pps_curr_pic_ref_enabled_flag', ctypes.c_uint16, 1),
        ('reserved', ctypes.c_uint16, 15),
    ]


class _EncPicScc(ctypes.Union):
    _fields_ = [('bits', _EncPicSccBits), ('value', ctypes.c_uint16)]


class VAEncPictureParameterBufferHEVC(ctypes.Structure):
    """What an encoder is told about the picture it is about to code.

    decoded_curr_pic -- the surface the reconstruction is kept in, and the
        picture's order count
    reference_frames -- the pictures held to predict from, unused slots
        ``VA_PICTURE_HEVC_INVALID``
    coded_buf -- the buffer the coded picture is written into
    collocated_ref_pic_index -- :data:`NO_REFERENCE` without temporal motion
        vector prediction
    pic_init_qp -- the quantiser itself, 26 more than the syntax element
    nal_unit_type -- the type of the picture's slice NAL units
    coding_type -- :data:`CODING_TYPE_I`, :data:`CODING_TYPE_P` or
        :data:`CODING_TYPE_B`
    """

    _fields_ = [
        ('decoded_curr_pic', VAPictureHEVC),
        ('reference_frames', VAPictureHEVC * REFERENCE_FRAMES),
        ('coded_buf', VABufferID),
        ('collocated_ref_pic_index', ctypes.c_uint8),
        ('last_picture', ctypes.c_uint8),
        ('pic_init_qp', ctypes.c_uint8),
        ('diff_cu_qp_delta_depth', ctypes.c_uint8),
        ('pps_cb_qp_offset', ctypes.c_int8),
        ('pps_cr_qp_offset', ctypes.c_int8),
        ('num_tile_columns_minus1', ctypes.c_uint8),
        ('num_tile_rows_minus1', ctypes.c_uint8),
        ('column_width_minus1', ctypes.c_uint8 * 19),
        ('row_height_minus1', ctypes.c_uint8 * 21),
        ('log2_parallel_merge_level_minus2', ctypes.c_uint8),
        ('ctu_max_bitsize_allowed', ctypes.c_uint8),
        ('num_ref_idx_l0_default_active_minus1', ctypes.c_uint8),
        ('num_ref_idx_l1_default_active_minus1', ctypes.c_uint8),
        ('slice_pic_parameter_set_id', ctypes.c_uint8),
        ('nal_unit_type', ctypes.c_uint8),
        ('pic_fields', _EncPicFields),
        ('hierarchical_level_plus1', ctypes.c_uint8),
        ('va_byte_reserved', ctypes.c_uint8),
        ('scc_fields', _EncPicScc),
        ('va_reserved', ctypes.c_uint32 * (PADDING_HIGH - 1)),
    ]


class _EncSliceFieldsBits(ctypes.Structure):
    _fields_ = [
        ('last_slice_of_pic_flag', ctypes.c_uint32, 1),
        ('dependent_slice_segment_flag', ctypes.c_uint32, 1),
        ('colour_plane_id', ctypes.c_uint32, 2),
        ('slice_temporal_mvp_enabled_flag', ctypes.c_uint32, 1),
        ('slice_sao_luma_flag', ctypes.c_uint32, 1),
        ('slice_sao_chroma_flag', ctypes.c_uint32, 1),
        ('num_ref_idx_active_override_flag', ctypes.c_uint32, 1),
        ('mvd_l1_zero_flag', ctypes.c_uint32, 1),
        ('cabac_init_flag', ctypes.c_uint32, 1),
        ('slice_deblocking_filter_disabled_flag', ctypes.c_uint32, 2),
        ('slice_loop_filter_across_slices_enabled_flag', ctypes.c_uint32, 1),
        ('collocated_from_l0_flag', ctypes.c_uint32, 1),
    ]


class _EncSliceFields(ctypes.Union):
    _fields_ = [('bits', _EncSliceFieldsBits), ('value', ctypes.c_uint32)]


class VAEncSliceParameterBufferHEVC(ctypes.Structure):
    """One slice segment as an encoder is told to code it.

    slice_segment_address -- the first coding tree block, in raster order
    num_ctu_in_slice -- how many coding tree blocks the slice holds
    slice_type -- 0 for B, 1 for P, 2 for I, as the syntax numbers them
    ref_pic_list0, ref_pic_list1 -- the reference lists, unused entries
        ``VA_PICTURE_HEVC_INVALID``
    max_num_merge_cand -- MaxNumMergeCand, which is five less the syntax element
    """

    _fields_ = [
        ('slice_segment_address', ctypes.c_uint32),
        ('num_ctu_in_slice', ctypes.c_uint32),
        ('slice_type', ctypes.c_uint8),
        ('slice_pic_parameter_set_id', ctypes.c_uint8),
        ('num_ref_idx_l0_active_minus1', ctypes.c_uint8),
        ('num_ref_idx_l1_active_minus1', ctypes.c_uint8),
        ('ref_pic_list0', VAPictureHEVC * REFERENCE_FRAMES),
        ('ref_pic_list1', VAPictureHEVC * REFERENCE_FRAMES),
        ('luma_log2_weight_denom', ctypes.c_uint8),
        ('delta_chroma_log2_weight_denom', ctypes.c_int8),
        ('delta_luma_weight_l0', ctypes.c_int8 * REFERENCE_FRAMES),
        ('luma_offset_l0', ctypes.c_int8 * REFERENCE_FRAMES),
        ('delta_chroma_weight_l0', (ctypes.c_int8 * 2) * REFERENCE_FRAMES),
        ('chroma_offset_l0', (ctypes.c_int8 * 2) * REFERENCE_FRAMES),
        ('delta_luma_weight_l1', ctypes.c_int8 * REFERENCE_FRAMES),
        ('luma_offset_l1', ctypes.c_int8 * REFERENCE_FRAMES),
        ('delta_chroma_weight_l1', (ctypes.c_int8 * 2) * REFERENCE_FRAMES),
        ('chroma_offset_l1', (ctypes.c_int8 * 2) * REFERENCE_FRAMES),
        ('max_num_merge_cand', ctypes.c_uint8),
        ('slice_qp_delta', ctypes.c_int8),
        ('slice_cb_qp_offset', ctypes.c_int8),
        ('slice_cr_qp_offset', ctypes.c_int8),
        ('slice_beta_offset_div2', ctypes.c_int8),
        ('slice_tc_offset_div2', ctypes.c_int8),
        ('slice_fields', _EncSliceFields),
        ('pred_weight_table_bit_offset', ctypes.c_uint32),
        ('pred_weight_table_bit_length', ctypes.c_uint32),
        ('va_reserved', ctypes.c_uint32 * (PADDING_MEDIUM - 2)),
    ]
