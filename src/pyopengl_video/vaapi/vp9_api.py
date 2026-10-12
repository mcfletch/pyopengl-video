"""ctypes over libva's VP9 structures and constants.

The layouts are checked against the libva headers by ``tests/test_va_abi.py``,
from the recording ``tools/record_va_abi.py`` makes, as
:mod:`pyopengl_video.vaapi.api` is.
"""
from __future__ import annotations

import ctypes

from pyopengl_video.vaapi.api import PADDING_LOW, PADDING_MEDIUM, VASurfaceID

__all__ = ['VADecPictureParameterBufferVP9', 'VASegmentParameterVP9',
           'VASliceParameterBufferVP9']


class _PicFieldsBits(ctypes.Structure):
    _fields_ = [
        ('subsampling_x', ctypes.c_uint32, 1),
        ('subsampling_y', ctypes.c_uint32, 1),
        ('frame_type', ctypes.c_uint32, 1),
        ('show_frame', ctypes.c_uint32, 1),
        ('error_resilient_mode', ctypes.c_uint32, 1),
        ('intra_only', ctypes.c_uint32, 1),
        ('allow_high_precision_mv', ctypes.c_uint32, 1),
        ('mcomp_filter_type', ctypes.c_uint32, 3),
        ('frame_parallel_decoding_mode', ctypes.c_uint32, 1),
        ('reset_frame_context', ctypes.c_uint32, 2),
        ('refresh_frame_context', ctypes.c_uint32, 1),
        ('frame_context_idx', ctypes.c_uint32, 2),
        ('segmentation_enabled', ctypes.c_uint32, 1),
        ('segmentation_temporal_update', ctypes.c_uint32, 1),
        ('segmentation_update_map', ctypes.c_uint32, 1),
        ('last_ref_frame', ctypes.c_uint32, 3),
        ('last_ref_frame_sign_bias', ctypes.c_uint32, 1),
        ('golden_ref_frame', ctypes.c_uint32, 3),
        ('golden_ref_frame_sign_bias', ctypes.c_uint32, 1),
        ('alt_ref_frame', ctypes.c_uint32, 3),
        ('alt_ref_frame_sign_bias', ctypes.c_uint32, 1),
        ('lossless_flag', ctypes.c_uint32, 1),
    ]


class _PicFields(ctypes.Union):
    _fields_ = [('bits', _PicFieldsBits), ('value', ctypes.c_uint32)]


class VADecPictureParameterBufferVP9(ctypes.Structure):
    """What a decoder is told about the frame it is about to decode.

    reference_frames -- the surface in each of the eight reference slots
    frame_header_length_in_bytes -- the uncompressed header's length
    first_partition_size -- the compressed header's length
    """

    _fields_ = [
        ('frame_width', ctypes.c_uint16),
        ('frame_height', ctypes.c_uint16),
        ('reference_frames', VASurfaceID * 8),
        ('pic_fields', _PicFields),
        ('filter_level', ctypes.c_uint8),
        ('sharpness_level', ctypes.c_uint8),
        ('log2_tile_rows', ctypes.c_uint8),
        ('log2_tile_columns', ctypes.c_uint8),
        ('frame_header_length_in_bytes', ctypes.c_uint8),
        ('first_partition_size', ctypes.c_uint16),
        ('mb_segment_tree_probs', ctypes.c_uint8 * 7),
        ('segment_pred_probs', ctypes.c_uint8 * 3),
        ('profile', ctypes.c_uint8),
        ('bit_depth', ctypes.c_uint8),
        ('va_reserved', ctypes.c_uint32 * PADDING_MEDIUM),
    ]


class _SegmentFlagsFields(ctypes.Structure):
    _fields_ = [
        ('segment_reference_enabled', ctypes.c_uint16, 1),
        ('segment_reference', ctypes.c_uint16, 2),
        ('segment_reference_skipped', ctypes.c_uint16, 1),
    ]


class _SegmentFlags(ctypes.Union):
    _fields_ = [('fields', _SegmentFlagsFields), ('value', ctypes.c_uint16)]


class VASegmentParameterVP9(ctypes.Structure):
    """One segment's reference and skip features, filter levels and quantiser scales.

    filter_level -- the loop filter level by reference frame, then by mode
    """

    _fields_ = [
        ('segment_flags', _SegmentFlags),
        ('filter_level', (ctypes.c_uint8 * 2) * 4),
        ('luma_ac_quant_scale', ctypes.c_int16),
        ('luma_dc_quant_scale', ctypes.c_int16),
        ('chroma_ac_quant_scale', ctypes.c_int16),
        ('chroma_dc_quant_scale', ctypes.c_int16),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


class VASliceParameterBufferVP9(ctypes.Structure):
    """Where a frame's data is in the data buffer, and its eight segments.

    A VP9 frame has no slices: one of these goes with each frame, and the
    data buffer holds the frame from its first byte.
    """

    _fields_ = [
        ('slice_data_size', ctypes.c_uint32),
        ('slice_data_offset', ctypes.c_uint32),
        ('slice_data_flag', ctypes.c_uint32),
        ('seg_param', VASegmentParameterVP9 * 8),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]
