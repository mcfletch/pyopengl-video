"""ctypes over libva's AV1 structures and constants.

The layouts are checked against the libva headers by ``tests/test_va_abi.py``,
from the recording ``tools/record_va_abi.py`` makes, as
:mod:`pyopengl_video.vaapi.api` is.
"""
from __future__ import annotations

import ctypes

from pyopengl_video.vaapi.api import PADDING_LOW, PADDING_MEDIUM, VASurfaceID

# ---------------------------------------------------------------- decoding
#
# ``va/va_dec_av1.h``: the picture parameter buffer a decoder is given once a
# frame, and the slice parameter buffer it is given once a tile.

#: ``VAAV1TransformationType``: the kinds of global motion model.
VAAV1TransformationIdentity = 0
VAAV1TransformationTranslation = 1
VAAV1TransformationRotzoom = 2
VAAV1TransformationAffine = 3
VAAV1TransformationCount = 4


class _SegmentInfoBits(ctypes.Structure):
    _fields_ = [
        ('enabled', ctypes.c_uint32, 1),
        ('update_map', ctypes.c_uint32, 1),
        ('temporal_update', ctypes.c_uint32, 1),
        ('update_data', ctypes.c_uint32, 1),
    ]


class _SegmentInfoFields(ctypes.Union):
    _fields_ = [('bits', _SegmentInfoBits), ('value', ctypes.c_uint32)]


class VASegmentationStructAV1(ctypes.Structure):
    """The segmentation in force for a frame.

    feature_data -- FeatureData[segment][feature], after clipping
    feature_mask -- for each segment, bit `feature` set where it is enabled
    """

    _fields_ = [
        ('segment_info_fields', _SegmentInfoFields),
        ('feature_data', (ctypes.c_int16 * 8) * 8),
        ('feature_mask', ctypes.c_uint8 * 8),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


class _FilmGrainInfoBits(ctypes.Structure):
    _fields_ = [
        ('apply_grain', ctypes.c_uint32, 1),
        ('chroma_scaling_from_luma', ctypes.c_uint32, 1),
        ('grain_scaling_minus_8', ctypes.c_uint32, 2),
        ('ar_coeff_lag', ctypes.c_uint32, 2),
        ('ar_coeff_shift_minus_6', ctypes.c_uint32, 2),
        ('grain_scale_shift', ctypes.c_uint32, 2),
        ('overlap_flag', ctypes.c_uint32, 1),
        ('clip_to_restricted_range', ctypes.c_uint32, 1),
    ]


class _FilmGrainInfoFields(ctypes.Union):
    _fields_ = [('bits', _FilmGrainInfoBits), ('value', ctypes.c_uint32)]


class VAFilmGrainStructAV1(ctypes.Structure):
    """A frame's film grain parameters; all zero where no grain is applied.

    ar_coeffs_y, ar_coeffs_cb, ar_coeffs_cr -- the auto-regression
        coefficients as signed values, 128 taken off what the stream codes
    """

    _fields_ = [
        ('film_grain_info_fields', _FilmGrainInfoFields),
        ('grain_seed', ctypes.c_uint16),
        ('num_y_points', ctypes.c_uint8),
        ('point_y_value', ctypes.c_uint8 * 14),
        ('point_y_scaling', ctypes.c_uint8 * 14),
        ('num_cb_points', ctypes.c_uint8),
        ('point_cb_value', ctypes.c_uint8 * 10),
        ('point_cb_scaling', ctypes.c_uint8 * 10),
        ('num_cr_points', ctypes.c_uint8),
        ('point_cr_value', ctypes.c_uint8 * 10),
        ('point_cr_scaling', ctypes.c_uint8 * 10),
        ('ar_coeffs_y', ctypes.c_int8 * 24),
        ('ar_coeffs_cb', ctypes.c_int8 * 25),
        ('ar_coeffs_cr', ctypes.c_int8 * 25),
        ('cb_mult', ctypes.c_uint8),
        ('cb_luma_mult', ctypes.c_uint8),
        ('cb_offset', ctypes.c_uint16),
        ('cr_mult', ctypes.c_uint8),
        ('cr_luma_mult', ctypes.c_uint8),
        ('cr_offset', ctypes.c_uint16),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


class VAWarpedMotionParamsAV1(ctypes.Structure):
    """One reference frame's global motion model.

    wmtype -- a ``VAAV1Transformation*`` value
    wmmat -- gm_params[ref][0..5]; the last two entries are unused
    invalid -- 1 where the model is not a valid warp (specification 7.11.3.6)
    """

    _fields_ = [
        ('wmtype', ctypes.c_int32),
        ('wmmat', ctypes.c_int32 * 8),
        ('invalid', ctypes.c_uint8),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


class _AV1SeqInfoBits(ctypes.Structure):
    _fields_ = [
        ('still_picture', ctypes.c_uint32, 1),
        ('use_128x128_superblock', ctypes.c_uint32, 1),
        ('enable_filter_intra', ctypes.c_uint32, 1),
        ('enable_intra_edge_filter', ctypes.c_uint32, 1),
        ('enable_interintra_compound', ctypes.c_uint32, 1),
        ('enable_masked_compound', ctypes.c_uint32, 1),
        ('enable_dual_filter', ctypes.c_uint32, 1),
        ('enable_order_hint', ctypes.c_uint32, 1),
        ('enable_jnt_comp', ctypes.c_uint32, 1),
        ('enable_cdef', ctypes.c_uint32, 1),
        ('mono_chrome', ctypes.c_uint32, 1),
        ('color_range', ctypes.c_uint32, 1),
        ('subsampling_x', ctypes.c_uint32, 1),
        ('subsampling_y', ctypes.c_uint32, 1),
        ('chroma_sample_position', ctypes.c_uint32, 1),
        ('film_grain_params_present', ctypes.c_uint32, 1),
    ]


class _AV1SeqInfoFields(ctypes.Union):
    _fields_ = [('fields', _AV1SeqInfoBits), ('value', ctypes.c_uint32)]


class _AV1PicInfoBits(ctypes.Structure):
    _fields_ = [
        ('frame_type', ctypes.c_uint32, 2),
        ('show_frame', ctypes.c_uint32, 1),
        ('showable_frame', ctypes.c_uint32, 1),
        ('error_resilient_mode', ctypes.c_uint32, 1),
        ('disable_cdf_update', ctypes.c_uint32, 1),
        ('allow_screen_content_tools', ctypes.c_uint32, 1),
        ('force_integer_mv', ctypes.c_uint32, 1),
        ('allow_intrabc', ctypes.c_uint32, 1),
        ('use_superres', ctypes.c_uint32, 1),
        ('allow_high_precision_mv', ctypes.c_uint32, 1),
        ('is_motion_mode_switchable', ctypes.c_uint32, 1),
        ('use_ref_frame_mvs', ctypes.c_uint32, 1),
        ('disable_frame_end_update_cdf', ctypes.c_uint32, 1),
        ('uniform_tile_spacing_flag', ctypes.c_uint32, 1),
        ('allow_warped_motion', ctypes.c_uint32, 1),
        ('large_scale_tile', ctypes.c_uint32, 1),
    ]


class _AV1PicInfoFields(ctypes.Union):
    _fields_ = [('bits', _AV1PicInfoBits), ('value', ctypes.c_uint32)]


class _AV1LoopFilterInfoBits(ctypes.Structure):
    _fields_ = [
        ('sharpness_level', ctypes.c_uint8, 3),
        ('mode_ref_delta_enabled', ctypes.c_uint8, 1),
        ('mode_ref_delta_update', ctypes.c_uint8, 1),
    ]


class _AV1LoopFilterInfoFields(ctypes.Union):
    _fields_ = [('bits', _AV1LoopFilterInfoBits), ('value', ctypes.c_uint8)]


class _AV1QMatrixBits(ctypes.Structure):
    _fields_ = [
        ('using_qmatrix', ctypes.c_uint16, 1),
        ('qm_y', ctypes.c_uint16, 4),
        ('qm_u', ctypes.c_uint16, 4),
        ('qm_v', ctypes.c_uint16, 4),
    ]


class _AV1QMatrixFields(ctypes.Union):
    _fields_ = [('bits', _AV1QMatrixBits), ('value', ctypes.c_uint16)]


class _AV1ModeControlBits(ctypes.Structure):
    _fields_ = [
        ('delta_q_present_flag', ctypes.c_uint32, 1),
        ('log2_delta_q_res', ctypes.c_uint32, 2),
        ('delta_lf_present_flag', ctypes.c_uint32, 1),
        ('log2_delta_lf_res', ctypes.c_uint32, 2),
        ('delta_lf_multi', ctypes.c_uint32, 1),
        ('tx_mode', ctypes.c_uint32, 2),
        ('reference_select', ctypes.c_uint32, 1),
        ('reduced_tx_set_used', ctypes.c_uint32, 1),
        ('skip_mode_present', ctypes.c_uint32, 1),
    ]


class _AV1ModeControlFields(ctypes.Union):
    _fields_ = [('bits', _AV1ModeControlBits), ('value', ctypes.c_uint32)]


class _AV1LoopRestorationBits(ctypes.Structure):
    _fields_ = [
        ('yframe_restoration_type', ctypes.c_uint16, 2),
        ('cbframe_restoration_type', ctypes.c_uint16, 2),
        ('crframe_restoration_type', ctypes.c_uint16, 2),
        ('lr_unit_shift', ctypes.c_uint16, 2),
        ('lr_uv_shift', ctypes.c_uint16, 1),
    ]


class _AV1LoopRestorationFields(ctypes.Union):
    _fields_ = [('bits', _AV1LoopRestorationBits), ('value', ctypes.c_uint16)]


class VADecPictureParameterBufferAV1(ctypes.Structure):
    """What a decoder is told about the frame it is about to decode.

    current_frame -- the surface the frame is reconstructed into, which is
        what later frames predict from
    current_display_picture -- the surface the frame is written to with film
        grain added; read only where ``apply_grain`` is set
    ref_frame_map -- the surface in each of the eight reference slots
    ref_frame_idx -- for LAST_FRAME to ALTREF_FRAME, the slot it names
    frame_width_minus1 -- the width after super-resolution, less one
    cdef_y_strengths, cdef_uv_strengths -- the primary strength in bits 2 to
        5 and the secondary strength as coded (0 to 3) in bits 0 and 1
    wm -- the global motion model of LAST_FRAME to ALTREF_FRAME
    """

    _fields_ = [
        ('profile', ctypes.c_uint8),
        ('order_hint_bits_minus_1', ctypes.c_uint8),
        ('bit_depth_idx', ctypes.c_uint8),
        ('matrix_coefficients', ctypes.c_uint8),
        ('seq_info_fields', _AV1SeqInfoFields),
        ('current_frame', VASurfaceID),
        ('current_display_picture', VASurfaceID),
        ('anchor_frames_num', ctypes.c_uint8),
        ('anchor_frames_list', ctypes.POINTER(VASurfaceID)),
        ('frame_width_minus1', ctypes.c_uint16),
        ('frame_height_minus1', ctypes.c_uint16),
        ('output_frame_width_in_tiles_minus_1', ctypes.c_uint16),
        ('output_frame_height_in_tiles_minus_1', ctypes.c_uint16),
        ('ref_frame_map', VASurfaceID * 8),
        ('ref_frame_idx', ctypes.c_uint8 * 7),
        ('primary_ref_frame', ctypes.c_uint8),
        ('order_hint', ctypes.c_uint8),
        ('seg_info', VASegmentationStructAV1),
        ('film_grain_info', VAFilmGrainStructAV1),
        ('tile_cols', ctypes.c_uint8),
        ('tile_rows', ctypes.c_uint8),
        ('width_in_sbs_minus_1', ctypes.c_uint16 * 63),
        ('height_in_sbs_minus_1', ctypes.c_uint16 * 63),
        ('tile_count_minus_1', ctypes.c_uint16),
        ('context_update_tile_id', ctypes.c_uint16),
        ('pic_info_fields', _AV1PicInfoFields),
        ('superres_scale_denominator', ctypes.c_uint8),
        ('interp_filter', ctypes.c_uint8),
        ('filter_level', ctypes.c_uint8 * 2),
        ('filter_level_u', ctypes.c_uint8),
        ('filter_level_v', ctypes.c_uint8),
        ('loop_filter_info_fields', _AV1LoopFilterInfoFields),
        ('ref_deltas', ctypes.c_int8 * 8),
        ('mode_deltas', ctypes.c_int8 * 2),
        ('base_qindex', ctypes.c_uint8),
        ('y_dc_delta_q', ctypes.c_int8),
        ('u_dc_delta_q', ctypes.c_int8),
        ('u_ac_delta_q', ctypes.c_int8),
        ('v_dc_delta_q', ctypes.c_int8),
        ('v_ac_delta_q', ctypes.c_int8),
        ('qmatrix_fields', _AV1QMatrixFields),
        ('mode_control_fields', _AV1ModeControlFields),
        ('cdef_damping_minus_3', ctypes.c_uint8),
        ('cdef_bits', ctypes.c_uint8),
        ('cdef_y_strengths', ctypes.c_uint8 * 8),
        ('cdef_uv_strengths', ctypes.c_uint8 * 8),
        ('loop_restoration_fields', _AV1LoopRestorationFields),
        ('wm', VAWarpedMotionParamsAV1 * 7),
        ('va_reserved', ctypes.c_uint32 * PADDING_MEDIUM),
    ]


class VASliceParameterBufferAV1(ctypes.Structure):
    """One tile's place in the data buffer that follows.

    slice_data_size -- the tile's coded bytes
    slice_data_offset -- where they begin in the data buffer
    tile_row, tile_column -- which tile it is
    """

    _fields_ = [
        ('slice_data_size', ctypes.c_uint32),
        ('slice_data_offset', ctypes.c_uint32),
        ('slice_data_flag', ctypes.c_uint32),
        ('tile_row', ctypes.c_uint16),
        ('tile_column', ctypes.c_uint16),
        ('tg_start', ctypes.c_uint16),
        ('tg_end', ctypes.c_uint16),
        ('anchor_frame_idx', ctypes.c_uint8),
        ('tile_idx_in_tile_list', ctypes.c_uint16),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


# ---------------------------------------------------------------- encoding
#
# ``va/va_enc_av1.h``: the sequence parameter buffer an encoder is given at a
# key frame, the picture parameter buffer it is given once a frame, and the
# tile group buffer that takes the place of a slice parameter buffer.

#: What ``VA_PADDING_HIGH`` expands to, the tail of the encoding buffers.
PADDING_HIGH = 16

#: A configuration attribute: what the driver's AV1 encoder takes of a tile
#: size, an OBU size and a transform mode, as :class:`VAConfigAttribValEncAV1Ext2`.
VAConfigAttribEncAV1Ext2 = 54

#: ``VAEncPackedHeaderTypeAV1``: the sequence header OBU and the frame header.
VAEncPackedHeaderAV1_SPS = 1
VAEncPackedHeaderAV1_PPS = 2

VA_AV1_MAX_SEGMENTS = 8
VA_AV1_SEG_LVL_MAX = 8

#: ``VAEncTransformationTypeAV1``: the kinds of global motion model.
VAAV1EncTransformationIdentity = 0
VAAV1EncTransformationTranslation = 1
VAAV1EncTransformationRotzoom = 2
VAAV1EncTransformationAffine = 3
VAAV1EncTransformationCount = 4


class _EncAV1Ext2Bits(ctypes.Structure):
    _fields_ = [
        ('tile_size_bytes_minus1', ctypes.c_uint32, 2),
        ('obu_size_bytes_minus1', ctypes.c_uint32, 2),
        ('tx_mode_support', ctypes.c_uint32, 3),
        ('max_tile_num_minus1', ctypes.c_uint32, 13),
    ]


class VAConfigAttribValEncAV1Ext2(ctypes.Union):
    """The value of ``VAConfigAttribEncAV1Ext2``.

    obu_size_bytes_minus1 -- the bytes the driver writes an obu_size in, less one
    tx_mode_support -- bit 0 for ONLY_4X4, bit 1 for TX_MODE_LARGEST and bit 2
        for TX_MODE_SELECT
    """

    _fields_ = [('bits', _EncAV1Ext2Bits), ('value', ctypes.c_uint32)]


class _EncAV1SeqBits(ctypes.Structure):
    _fields_ = [
        ('still_picture', ctypes.c_uint32, 1),
        ('use_128x128_superblock', ctypes.c_uint32, 1),
        ('enable_filter_intra', ctypes.c_uint32, 1),
        ('enable_intra_edge_filter', ctypes.c_uint32, 1),
        ('enable_interintra_compound', ctypes.c_uint32, 1),
        ('enable_masked_compound', ctypes.c_uint32, 1),
        ('enable_warped_motion', ctypes.c_uint32, 1),
        ('enable_dual_filter', ctypes.c_uint32, 1),
        ('enable_order_hint', ctypes.c_uint32, 1),
        ('enable_jnt_comp', ctypes.c_uint32, 1),
        ('enable_ref_frame_mvs', ctypes.c_uint32, 1),
        ('enable_superres', ctypes.c_uint32, 1),
        ('enable_cdef', ctypes.c_uint32, 1),
        ('enable_restoration', ctypes.c_uint32, 1),
        ('bit_depth_minus8', ctypes.c_uint32, 3),
        ('subsampling_x', ctypes.c_uint32, 1),
        ('subsampling_y', ctypes.c_uint32, 1),
        ('mono_chrome', ctypes.c_uint32, 1),
    ]


class _EncAV1SeqFields(ctypes.Union):
    _fields_ = [('bits', _EncAV1SeqBits), ('value', ctypes.c_uint32)]


class VAEncSequenceParameterBufferAV1(ctypes.Structure):
    """What an encoder is told about the whole sequence.

    intra_period -- frames from one key frame to the next
    ip_period -- frames from one forward-predicted frame to the next, 1
        where every frame is coded in display order
    bits_per_second -- the bitrate CBR holds, and the ceiling of VBR
    """

    _fields_ = [
        ('seq_profile', ctypes.c_uint8),
        ('seq_level_idx', ctypes.c_uint8),
        ('seq_tier', ctypes.c_uint8),
        ('hierarchical_flag', ctypes.c_uint8),
        ('intra_period', ctypes.c_uint32),
        ('ip_period', ctypes.c_uint32),
        ('bits_per_second', ctypes.c_uint32),
        ('seq_fields', _EncAV1SeqFields),
        ('order_hint_bits_minus_1', ctypes.c_uint8),
        ('va_reserved', ctypes.c_uint32 * PADDING_HIGH),
    ]


class _EncAV1SegFlagsBits(ctypes.Structure):
    _fields_ = [
        ('segmentation_enabled', ctypes.c_uint8, 1),
        ('segmentation_update_map', ctypes.c_uint8, 1),
        ('segmentation_temporal_update', ctypes.c_uint8, 1),
    ]


class _EncAV1SegFlags(ctypes.Union):
    _fields_ = [('bits', _EncAV1SegFlagsBits), ('value', ctypes.c_uint8)]


class VAEncSegParamAV1(ctypes.Structure):
    """A frame's segmentation; all zero where segmentation is off.

    feature_data -- FeatureData[segment][feature], after clipping
    feature_mask -- for each segment, bit `feature` set where it is enabled
    """

    _fields_ = [
        ('seg_flags', _EncAV1SegFlags),
        ('segment_number', ctypes.c_uint8),
        ('feature_data', (ctypes.c_int16 * VA_AV1_SEG_LVL_MAX) * VA_AV1_MAX_SEGMENTS),
        ('feature_mask', ctypes.c_uint8 * VA_AV1_MAX_SEGMENTS),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


class VAEncWarpedMotionParamsAV1(ctypes.Structure):
    """One reference frame's global motion model.

    wmtype -- a ``VAAV1EncTransformation*`` value
    wmmat -- gm_params[ref][0..5]; the last two entries are unused
    """

    _fields_ = [
        ('wmtype', ctypes.c_int32),
        ('wmmat', ctypes.c_int32 * 8),
        ('invalid', ctypes.c_uint8),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


class _RefFrameCtrlBits(ctypes.Structure):
    _fields_ = [
        ('search_idx0', ctypes.c_uint32, 3),
        ('search_idx1', ctypes.c_uint32, 3),
        ('search_idx2', ctypes.c_uint32, 3),
        ('search_idx3', ctypes.c_uint32, 3),
        ('search_idx4', ctypes.c_uint32, 3),
        ('search_idx5', ctypes.c_uint32, 3),
        ('search_idx6', ctypes.c_uint32, 3),
    ]


class VARefFrameCtrlAV1(ctypes.Union):
    """The reference frames to search, in order of preference.

    Each ``search_idx`` is a reference frame name, 1 for LAST_FRAME to 7 for
    ALTREF_FRAME, or 0 for none.
    """

    _fields_ = [('fields', _RefFrameCtrlBits), ('value', ctypes.c_uint32)]


class _EncAV1PictureFlagsBits(ctypes.Structure):
    _fields_ = [
        ('frame_type', ctypes.c_uint32, 2),
        ('error_resilient_mode', ctypes.c_uint32, 1),
        ('disable_cdf_update', ctypes.c_uint32, 1),
        ('use_superres', ctypes.c_uint32, 1),
        ('allow_high_precision_mv', ctypes.c_uint32, 1),
        ('use_ref_frame_mvs', ctypes.c_uint32, 1),
        ('disable_frame_end_update_cdf', ctypes.c_uint32, 1),
        ('reduced_tx_set', ctypes.c_uint32, 1),
        ('enable_frame_obu', ctypes.c_uint32, 1),
        ('long_term_reference', ctypes.c_uint32, 1),
        ('disable_frame_recon', ctypes.c_uint32, 1),
        ('allow_intrabc', ctypes.c_uint32, 1),
        ('palette_mode_enable', ctypes.c_uint32, 1),
    ]


class _EncAV1PictureFlags(ctypes.Union):
    _fields_ = [('bits', _EncAV1PictureFlagsBits), ('value', ctypes.c_uint32)]


class _EncAV1LoopFilterFlagsBits(ctypes.Structure):
    _fields_ = [
        ('sharpness_level', ctypes.c_uint8, 3),
        ('mode_ref_delta_enabled', ctypes.c_uint8, 1),
        ('mode_ref_delta_update', ctypes.c_uint8, 1),
    ]


class _EncAV1LoopFilterFlags(ctypes.Union):
    _fields_ = [('bits', _EncAV1LoopFilterFlagsBits), ('value', ctypes.c_uint8)]


class _EncAV1QMatrixFlagsBits(ctypes.Structure):
    _fields_ = [
        ('using_qmatrix', ctypes.c_uint16, 1),
        ('qm_y', ctypes.c_uint16, 4),
        ('qm_u', ctypes.c_uint16, 4),
        ('qm_v', ctypes.c_uint16, 4),
    ]


class _EncAV1QMatrixFlags(ctypes.Union):
    _fields_ = [('bits', _EncAV1QMatrixFlagsBits), ('value', ctypes.c_uint16)]


class _EncAV1ModeControlFlagsBits(ctypes.Structure):
    _fields_ = [
        ('delta_q_present', ctypes.c_uint32, 1),
        ('delta_q_res', ctypes.c_uint32, 2),
        ('delta_lf_present', ctypes.c_uint32, 1),
        ('delta_lf_res', ctypes.c_uint32, 2),
        ('delta_lf_multi', ctypes.c_uint32, 1),
        ('tx_mode', ctypes.c_uint32, 2),
        ('reference_mode', ctypes.c_uint32, 2),
        ('skip_mode_present', ctypes.c_uint32, 1),
    ]


class _EncAV1ModeControlFlags(ctypes.Union):
    _fields_ = [('bits', _EncAV1ModeControlFlagsBits), ('value', ctypes.c_uint32)]


class _EncAV1LoopRestorationFlagsBits(ctypes.Structure):
    _fields_ = [
        ('yframe_restoration_type', ctypes.c_uint16, 2),
        ('cbframe_restoration_type', ctypes.c_uint16, 2),
        ('crframe_restoration_type', ctypes.c_uint16, 2),
        ('lr_unit_shift', ctypes.c_uint16, 2),
        ('lr_uv_shift', ctypes.c_uint16, 1),
    ]


class _EncAV1LoopRestorationFlags(ctypes.Union):
    _fields_ = [('bits', _EncAV1LoopRestorationFlagsBits), ('value', ctypes.c_uint16)]


class _EncAV1TileGroupObuHeaderBits(ctypes.Structure):
    _fields_ = [
        ('obu_extension_flag', ctypes.c_uint8, 1),
        ('obu_has_size_field', ctypes.c_uint8, 1),
        ('temporal_id', ctypes.c_uint8, 3),
        ('spatial_id', ctypes.c_uint8, 2),
    ]


class _EncAV1TileGroupObuHeader(ctypes.Union):
    _fields_ = [('bits', _EncAV1TileGroupObuHeaderBits), ('value', ctypes.c_uint8)]


class VAEncPictureParameterBufferAV1(ctypes.Structure):
    """What an encoder is told about the frame it is about to code.

    reconstructed_frame -- the surface the decoded frame is kept in
    coded_buf -- the buffer the coded frame is written into
    reference_frames -- the surface in each of the eight reference slots,
        ``VA_INVALID_SURFACE`` for a slot that holds none
    ref_frame_idx -- for LAST_FRAME to ALTREF_FRAME, the slot it names
    ref_frame_ctrl_l0, ref_frame_ctrl_l1 -- the reference frames to search,
        forwards and backwards
    cdef_y_strengths, cdef_uv_strengths -- the primary strength in bits 2 to
        5 and the secondary strength as coded (0 to 3) in bits 0 and 1
    bit_offset_qindex and the offsets after it -- where the fields a driver
        may rewrite lie in the packed frame header, counted from the start of
        its OBU
    """

    _fields_ = [
        ('frame_width_minus_1', ctypes.c_uint16),
        ('frame_height_minus_1', ctypes.c_uint16),
        ('reconstructed_frame', VASurfaceID),
        ('coded_buf', ctypes.c_uint32),
        ('reference_frames', VASurfaceID * 8),
        ('ref_frame_idx', ctypes.c_uint8 * 7),
        ('hierarchical_level_plus1', ctypes.c_uint8),
        ('primary_ref_frame', ctypes.c_uint8),
        ('order_hint', ctypes.c_uint8),
        ('refresh_frame_flags', ctypes.c_uint8),
        ('reserved8bits1', ctypes.c_uint8),
        ('ref_frame_ctrl_l0', VARefFrameCtrlAV1),
        ('ref_frame_ctrl_l1', VARefFrameCtrlAV1),
        ('picture_flags', _EncAV1PictureFlags),
        ('seg_id_block_size', ctypes.c_uint8),
        ('num_tile_groups_minus1', ctypes.c_uint8),
        ('temporal_id', ctypes.c_uint8),
        ('filter_level', ctypes.c_uint8 * 2),
        ('filter_level_u', ctypes.c_uint8),
        ('filter_level_v', ctypes.c_uint8),
        ('loop_filter_flags', _EncAV1LoopFilterFlags),
        ('superres_scale_denominator', ctypes.c_uint8),
        ('interpolation_filter', ctypes.c_uint8),
        ('ref_deltas', ctypes.c_int8 * 8),
        ('mode_deltas', ctypes.c_int8 * 2),
        ('base_qindex', ctypes.c_uint8),
        ('y_dc_delta_q', ctypes.c_int8),
        ('u_dc_delta_q', ctypes.c_int8),
        ('u_ac_delta_q', ctypes.c_int8),
        ('v_dc_delta_q', ctypes.c_int8),
        ('v_ac_delta_q', ctypes.c_int8),
        ('min_base_qindex', ctypes.c_uint8),
        ('max_base_qindex', ctypes.c_uint8),
        ('qmatrix_flags', _EncAV1QMatrixFlags),
        ('reserved16bits1', ctypes.c_uint16),
        ('mode_control_flags', _EncAV1ModeControlFlags),
        ('segments', VAEncSegParamAV1),
        ('tile_cols', ctypes.c_uint8),
        ('tile_rows', ctypes.c_uint8),
        ('reserved16bits2', ctypes.c_uint16),
        ('width_in_sbs_minus_1', ctypes.c_uint16 * 63),
        ('height_in_sbs_minus_1', ctypes.c_uint16 * 63),
        ('context_update_tile_id', ctypes.c_uint16),
        ('cdef_damping_minus_3', ctypes.c_uint8),
        ('cdef_bits', ctypes.c_uint8),
        ('cdef_y_strengths', ctypes.c_uint8 * 8),
        ('cdef_uv_strengths', ctypes.c_uint8 * 8),
        ('loop_restoration_flags', _EncAV1LoopRestorationFlags),
        ('wm', VAEncWarpedMotionParamsAV1 * 7),
        ('bit_offset_qindex', ctypes.c_uint32),
        ('bit_offset_segmentation', ctypes.c_uint32),
        ('bit_offset_loopfilter_params', ctypes.c_uint32),
        ('bit_offset_cdef_params', ctypes.c_uint32),
        ('size_in_bits_cdef_params', ctypes.c_uint32),
        ('byte_offset_frame_hdr_obu_size', ctypes.c_uint32),
        ('size_in_bits_frame_hdr_obu', ctypes.c_uint32),
        ('tile_group_obu_hdr_info', _EncAV1TileGroupObuHeader),
        ('number_skip_frames', ctypes.c_uint8),
        ('reserved16bits3', ctypes.c_uint16),
        ('skip_frames_reduced_size', ctypes.c_int32),
        ('va_reserved', ctypes.c_uint32 * PADDING_HIGH),
    ]


class VAEncTileGroupBufferAV1(ctypes.Structure):
    """One tile group: its first and last tile, counted in raster order."""

    _fields_ = [
        ('tg_start', ctypes.c_uint8),
        ('tg_end', ctypes.c_uint8),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]
