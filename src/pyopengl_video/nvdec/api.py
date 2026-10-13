"""ctypes declarations for NVDEC and the CUDA driver API it decodes into.

Two libraries, both part of the NVIDIA display driver:

- ``libnvcuvid.so.1`` (``nvcuvid.dll``) is the decoder. Its parser reads the
  bitstream and calls back with each sequence, each picture to decode and
  each picture to display; its decoder turns a picture into NV12 in CUDA
  device memory.
- ``libcuda.so.1`` (``nvcuda.dll``) is the CUDA driver API: contexts, copies
  between device memory and OpenGL textures, and the registration that lets
  CUDA write a texture. The CUDA toolkit and runtime are not involved.

:class:`Cuda` and :class:`Cuvid` load them and bind the calls this package
makes. Every call returns a ``CUresult``; :meth:`Cuda.check` turns one that is
not success into a :class:`CUDAError` carrying the driver's name for it.

The layouts and enumerations describe the ABI published in
``dynlink_cuviddec.h``, ``dynlink_nvcuvid.h`` and ``dynlink_cuda.h``, as
distributed in nv-codec-headers under the MIT licence -- see NOTICES.md.
``tests/test_nvdec_abi.py`` checks the declarations here against sizes,
offsets and values recorded from those headers.
"""
from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from ctypes import (
    POINTER,
    Structure,
    Union,
    c_byte,
    c_char,
    c_char_p,
    c_int,
    c_longlong,
    c_short,
    c_size_t,
    c_ubyte,
    c_uint,
    c_ulong,
    c_ulonglong,
    c_ushort,
    c_void_p,
)
from typing import Any

from pyopengl_video import codecs
from pyopengl_video.decoder import DecoderError


def library_names() -> tuple[str, str]:
    """The driver's CUDA and decoder libraries on this platform."""
    if sys.platform == 'win32':
        return 'nvcuda.dll', 'nvcuvid.dll'
    return 'libcuda.so.1', 'libnvcuvid.so.1'


CUDA_LIBRARY_NAME, CUVID_LIBRARY_NAME = library_names()

# Both interfaces are __stdcall on Windows and cdecl everywhere else, which
# are one ABI on 64-bit Windows. The branch is written out because ctypes
# offers the Windows spellings on Windows alone.
if sys.platform == 'win32':
    _LOADER = ctypes.WinDLL
    FUNCTYPE = ctypes.WINFUNCTYPE
else:
    _LOADER = ctypes.CDLL
    FUNCTYPE = ctypes.CFUNCTYPE

#: Interface version these declarations describe.
HEADER_VERSION = (13, 1)

#: A mapped picture's address is 64 bits wide in the calls bound here, which
#: is the ABI of a 64-bit process.
SUPPORTED_POINTER_SIZE = ctypes.sizeof(c_void_p) == 8

# --------------------------------------------------------------------------
# enumerations, as plain constants: they cross the ABI as int
# --------------------------------------------------------------------------

CUDA_SUCCESS = 0

CU_MEMORYTYPE_HOST = 1
CU_MEMORYTYPE_DEVICE = 2
CU_MEMORYTYPE_ARRAY = 3

CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2

CU_GL_DEVICE_LIST_ALL = 1

cudaVideoCodec_H264 = 4
cudaVideoCodec_HEVC = 8
cudaVideoCodec_VP9 = 10
cudaVideoCodec_AV1 = 11

cudaVideoSurfaceFormat_NV12 = 0
cudaVideoSurfaceFormat_P016 = 1

cudaVideoChromaFormat_Monochrome = 0
cudaVideoChromaFormat_420 = 1
cudaVideoChromaFormat_422 = 2
cudaVideoChromaFormat_444 = 3

cudaVideoDeinterlaceMode_Weave = 0
cudaVideoDeinterlaceMode_Bob = 1
cudaVideoDeinterlaceMode_Adaptive = 2

cudaVideoCreate_PreferCUVID = 0x04

CUVID_PKT_ENDOFSTREAM = 0x01
CUVID_PKT_TIMESTAMP = 0x02
CUVID_PKT_DISCONTINUITY = 0x04
CUVID_PKT_ENDOFPICTURE = 0x08

#: The codec names this package uses, as the decoder's enumeration.
CODECS = {
    codecs.H264: cudaVideoCodec_H264,
    codecs.HEVC: cudaVideoCodec_HEVC,
    codecs.VP9: cudaVideoCodec_VP9,
    codecs.AV1: cudaVideoCodec_AV1,
}

#: Each of those as its specification names it, for a message.
CODEC_TITLES = {value: codecs.title(name) for name, value in CODECS.items()}

CHROMA_FORMAT_NAMES = {
    cudaVideoChromaFormat_Monochrome: 'monochrome',
    cudaVideoChromaFormat_420: '4:2:0',
    cudaVideoChromaFormat_422: '4:2:2',
    cudaVideoChromaFormat_444: '4:4:4',
}


class CUDAError(DecoderError):
    """A CUDA driver or NVDEC call failed.

    Carries the numeric ``CUresult``, the driver's name for it and the call
    that returned it.
    """

    def __init__(self, status: int, call: str, name: str = '', detail: str = '') -> None:
        self.status = status
        self.call = call
        self.name = name or 'CUDA_ERROR'
        message = f'{call} failed: {self.name} ({status})'
        if detail:
            message = f'{message}: {detail}'
        super().__init__(message)


# --------------------------------------------------------------------------
# structures
# --------------------------------------------------------------------------

class CUDA_MEMCPY2D(Structure):
    """One two-dimensional copy between host memory, device memory and arrays."""

    _fields_ = [
        ('srcXInBytes', c_size_t),
        ('srcY', c_size_t),
        ('srcMemoryType', c_int),
        ('srcHost', c_void_p),
        ('srcDevice', c_ulonglong),
        ('srcArray', c_void_p),
        ('srcPitch', c_size_t),
        ('dstXInBytes', c_size_t),
        ('dstY', c_size_t),
        ('dstMemoryType', c_int),
        ('dstHost', c_void_p),
        ('dstDevice', c_ulonglong),
        ('dstArray', c_void_p),
        ('dstPitch', c_size_t),
        ('WidthInBytes', c_size_t),
        ('Height', c_size_t),
    ]


class _FrameRate(Structure):
    _fields_ = [('numerator', c_uint), ('denominator', c_uint)]


class _Rectangle(Structure):
    _fields_ = [('left', c_int), ('top', c_int), ('right', c_int), ('bottom', c_int)]


class _ShortRectangle(Structure):
    _fields_ = [('left', c_short), ('top', c_short), ('right', c_short),
                ('bottom', c_short)]


class _AspectRatio(Structure):
    _fields_ = [('x', c_int), ('y', c_int)]


class _SignalDescription(Structure):
    """The stream's colour description, as H.273 code points."""

    _fields_ = [
        ('video_format', c_ubyte, 3),
        ('video_full_range_flag', c_ubyte, 1),
        ('reserved_zero_bits', c_ubyte, 4),
        ('color_primaries', c_ubyte),
        ('transfer_characteristics', c_ubyte),
        ('matrix_coefficients', c_ubyte),
    ]


class CUVIDEOFORMAT(Structure):
    """What the parser found in a sequence header."""

    _fields_ = [
        ('codec', c_int),
        ('frame_rate', _FrameRate),
        ('progressive_sequence', c_ubyte),
        ('bit_depth_luma_minus8', c_ubyte),
        ('bit_depth_chroma_minus8', c_ubyte),
        ('min_num_decode_surfaces', c_ubyte),
        ('coded_width', c_uint),
        ('coded_height', c_uint),
        ('display_area', _Rectangle),
        ('chroma_format', c_int),
        ('bitrate', c_uint),
        ('display_aspect_ratio', _AspectRatio),
        ('video_signal_description', _SignalDescription),
        ('seqhdr_data_length', c_uint),
    ]


class CUVIDSOURCEDATAPACKET(Structure):
    """A piece of bitstream handed to the parser."""

    _fields_ = [
        ('flags', c_ulong),
        ('payload_size', c_ulong),
        ('payload', c_void_p),
        ('timestamp', c_longlong),
    ]


class CUVIDPARSERDISPINFO(Structure):
    """A picture ready to be shown, in display order."""

    _fields_ = [
        ('picture_index', c_int),
        ('progressive_frame', c_int),
        ('top_field_first', c_int),
        ('repeat_first_field', c_int),
        ('timestamp', c_longlong),
    ]


class CUVIDH264DPBENTRY(Structure):
    """One frame of an H.264 picture's decoded picture buffer.

    ``PicIdx`` is the picture the frame was decoded into, and is negative for
    an entry that holds none.
    """

    _fields_ = [
        ('PicIdx', c_int),
        ('FrameIdx', c_int),
        ('is_long_term', c_int),
        ('not_existing', c_int),
        ('used_for_reference', c_int),
        ('FieldOrderCnt', c_int * 2),
    ]


class CUVIDH264MVCEXT(Structure):
    _fields_ = [
        ('num_views_minus1', c_int),
        ('view_id', c_int),
        ('inter_view_flag', c_ubyte),
        ('num_inter_view_refs_l0', c_ubyte),
        ('num_inter_view_refs_l1', c_ubyte),
        ('MVCReserved8Bits', c_ubyte),
        ('InterViewRefsL0', c_int * 16),
        ('InterViewRefsL1', c_int * 16),
    ]


class CUVIDH264SVCEXT(Structure):
    _fields_ = [
        ('profile_idc', c_ubyte),
        ('level_idc', c_ubyte),
        ('DQId', c_ubyte),
        ('DQIdMax', c_ubyte),
        ('disable_inter_layer_deblocking_filter_idc', c_ubyte),
        ('ref_layer_chroma_phase_y_plus1', c_ubyte),
        ('inter_layer_slice_alpha_c0_offset_div2', c_byte),
        ('inter_layer_slice_beta_offset_div2', c_byte),
        ('DPBEntryValidFlag', c_ushort),
        ('inter_layer_deblocking_filter_control_present_flag', c_ubyte),
        ('extended_spatial_scalability_idc', c_ubyte),
        ('adaptive_tcoeff_level_prediction_flag', c_ubyte),
        ('slice_header_restriction_flag', c_ubyte),
        ('chroma_phase_x_plus1_flag', c_ubyte),
        ('chroma_phase_y_plus1', c_ubyte),
        ('tcoeff_level_prediction_flag', c_ubyte),
        ('constrained_intra_resampling_flag', c_ubyte),
        ('ref_layer_chroma_phase_x_plus1_flag', c_ubyte),
        ('store_ref_base_pic_flag', c_ubyte),
        ('Reserved8BitsA', c_ubyte),
        ('Reserved8BitsB', c_ubyte),
        ('scaled_ref_layer_left_offset', c_short),
        ('scaled_ref_layer_top_offset', c_short),
        ('scaled_ref_layer_right_offset', c_short),
        ('scaled_ref_layer_bottom_offset', c_short),
        ('Reserved16Bits', c_ushort),
        ('pNextLayer', c_void_p),
        ('bRefBaseLayer', c_int),
    ]


class _H264SliceGroupMap(Union):
    _fields_ = [('slice_group_map_addr', c_ulonglong), ('pMb2SliceGroupMap', c_void_p)]


class _H264Extension(Union):
    _fields_ = [('mvcext', CUVIDH264MVCEXT), ('svcext', CUVIDH264SVCEXT)]


class CUVIDH264PICPARAMS(Structure):
    """An H.264 picture's parameters. ``dpb`` names the pictures it predicts from."""

    _anonymous_ = ('extension',)
    _fields_ = [
        ('log2_max_frame_num_minus4', c_int),
        ('pic_order_cnt_type', c_int),
        ('log2_max_pic_order_cnt_lsb_minus4', c_int),
        ('delta_pic_order_always_zero_flag', c_int),
        ('frame_mbs_only_flag', c_int),
        ('direct_8x8_inference_flag', c_int),
        ('num_ref_frames', c_int),
        ('residual_colour_transform_flag', c_ubyte),
        ('bit_depth_luma_minus8', c_ubyte),
        ('bit_depth_chroma_minus8', c_ubyte),
        ('qpprime_y_zero_transform_bypass_flag', c_ubyte),
        ('entropy_coding_mode_flag', c_int),
        ('pic_order_present_flag', c_int),
        ('num_ref_idx_l0_active_minus1', c_int),
        ('num_ref_idx_l1_active_minus1', c_int),
        ('weighted_pred_flag', c_int),
        ('weighted_bipred_idc', c_int),
        ('pic_init_qp_minus26', c_int),
        ('deblocking_filter_control_present_flag', c_int),
        ('redundant_pic_cnt_present_flag', c_int),
        ('transform_8x8_mode_flag', c_int),
        ('MbaffFrameFlag', c_int),
        ('constrained_intra_pred_flag', c_int),
        ('chroma_qp_index_offset', c_int),
        ('second_chroma_qp_index_offset', c_int),
        ('ref_pic_flag', c_int),
        ('frame_num', c_int),
        ('CurrFieldOrderCnt', c_int * 2),
        ('dpb', CUVIDH264DPBENTRY * 16),
        ('WeightScale4x4', c_ubyte * 16 * 6),
        ('WeightScale8x8', c_ubyte * 64 * 2),
        ('fmo_aso_enable', c_ubyte),
        ('num_slice_groups_minus1', c_ubyte),
        ('slice_group_map_type', c_ubyte),
        ('pic_init_qs_minus26', c_byte),
        ('slice_group_change_rate_minus1', c_uint),
        ('fmo', _H264SliceGroupMap),
        ('mb_adaptive_frame_field_flag', c_uint, 2),
        ('Reserved1', c_uint, 30),
        ('Reserved', c_uint * 11),
        ('extension', _H264Extension),
    ]


class CUVIDHEVCPICPARAMS(Structure):
    """An HEVC picture's parameters.

    ``RefPicIdx`` names the pictures it predicts from, -1 for an entry that
    names none.
    """

    _fields_ = [
        ('pic_width_in_luma_samples', c_int),
        ('pic_height_in_luma_samples', c_int),
        ('log2_min_luma_coding_block_size_minus3', c_ubyte),
        ('log2_diff_max_min_luma_coding_block_size', c_ubyte),
        ('log2_min_transform_block_size_minus2', c_ubyte),
        ('log2_diff_max_min_transform_block_size', c_ubyte),
        ('pcm_enabled_flag', c_ubyte),
        ('log2_min_pcm_luma_coding_block_size_minus3', c_ubyte),
        ('log2_diff_max_min_pcm_luma_coding_block_size', c_ubyte),
        ('pcm_sample_bit_depth_luma_minus1', c_ubyte),
        ('pcm_sample_bit_depth_chroma_minus1', c_ubyte),
        ('pcm_loop_filter_disabled_flag', c_ubyte),
        ('strong_intra_smoothing_enabled_flag', c_ubyte),
        ('max_transform_hierarchy_depth_intra', c_ubyte),
        ('max_transform_hierarchy_depth_inter', c_ubyte),
        ('amp_enabled_flag', c_ubyte),
        ('separate_colour_plane_flag', c_ubyte),
        ('log2_max_pic_order_cnt_lsb_minus4', c_ubyte),
        ('num_short_term_ref_pic_sets', c_ubyte),
        ('long_term_ref_pics_present_flag', c_ubyte),
        ('num_long_term_ref_pics_sps', c_ubyte),
        ('sps_temporal_mvp_enabled_flag', c_ubyte),
        ('sample_adaptive_offset_enabled_flag', c_ubyte),
        ('scaling_list_enable_flag', c_ubyte),
        ('IrapPicFlag', c_ubyte),
        ('IdrPicFlag', c_ubyte),
        ('bit_depth_luma_minus8', c_ubyte),
        ('bit_depth_chroma_minus8', c_ubyte),
        ('log2_max_transform_skip_block_size_minus2', c_ubyte),
        ('log2_sao_offset_scale_luma', c_ubyte),
        ('log2_sao_offset_scale_chroma', c_ubyte),
        ('high_precision_offsets_enabled_flag', c_ubyte),
        ('mv_hevc_enable', c_ubyte),
        ('nuh_layer_id', c_ubyte),
        ('default_ref_layers_active_flag', c_ubyte),
        ('NumDirectRefLayers', c_ubyte),
        ('max_one_active_ref_layer_flag', c_ubyte),
        ('poc_lsb_not_present_flag', c_ubyte),
        ('NumActiveRefLayerPics0', c_ubyte),
        ('NumActiveRefLayerPics1', c_ubyte),
        ('reserved1', c_ubyte * 2),
        ('dependent_slice_segments_enabled_flag', c_ubyte),
        ('slice_segment_header_extension_present_flag', c_ubyte),
        ('sign_data_hiding_enabled_flag', c_ubyte),
        ('cu_qp_delta_enabled_flag', c_ubyte),
        ('diff_cu_qp_delta_depth', c_ubyte),
        ('init_qp_minus26', c_byte),
        ('pps_cb_qp_offset', c_byte),
        ('pps_cr_qp_offset', c_byte),
        ('constrained_intra_pred_flag', c_ubyte),
        ('weighted_pred_flag', c_ubyte),
        ('weighted_bipred_flag', c_ubyte),
        ('transform_skip_enabled_flag', c_ubyte),
        ('transquant_bypass_enabled_flag', c_ubyte),
        ('entropy_coding_sync_enabled_flag', c_ubyte),
        ('log2_parallel_merge_level_minus2', c_ubyte),
        ('num_extra_slice_header_bits', c_ubyte),
        ('loop_filter_across_tiles_enabled_flag', c_ubyte),
        ('loop_filter_across_slices_enabled_flag', c_ubyte),
        ('output_flag_present_flag', c_ubyte),
        ('num_ref_idx_l0_default_active_minus1', c_ubyte),
        ('num_ref_idx_l1_default_active_minus1', c_ubyte),
        ('lists_modification_present_flag', c_ubyte),
        ('cabac_init_present_flag', c_ubyte),
        ('pps_slice_chroma_qp_offsets_present_flag', c_ubyte),
        ('deblocking_filter_override_enabled_flag', c_ubyte),
        ('pps_deblocking_filter_disabled_flag', c_ubyte),
        ('pps_beta_offset_div2', c_byte),
        ('pps_tc_offset_div2', c_byte),
        ('tiles_enabled_flag', c_ubyte),
        ('uniform_spacing_flag', c_ubyte),
        ('num_tile_columns_minus1', c_ubyte),
        ('num_tile_rows_minus1', c_ubyte),
        ('column_width_minus1', c_ushort * 21),
        ('row_height_minus1', c_ushort * 21),
        ('sps_range_extension_flag', c_ubyte),
        ('transform_skip_rotation_enabled_flag', c_ubyte),
        ('transform_skip_context_enabled_flag', c_ubyte),
        ('implicit_rdpcm_enabled_flag', c_ubyte),
        ('explicit_rdpcm_enabled_flag', c_ubyte),
        ('extended_precision_processing_flag', c_ubyte),
        ('intra_smoothing_disabled_flag', c_ubyte),
        ('persistent_rice_adaptation_enabled_flag', c_ubyte),
        ('cabac_bypass_alignment_enabled_flag', c_ubyte),
        ('pps_range_extension_flag', c_ubyte),
        ('cross_component_prediction_enabled_flag', c_ubyte),
        ('chroma_qp_offset_list_enabled_flag', c_ubyte),
        ('diff_cu_chroma_qp_offset_depth', c_ubyte),
        ('chroma_qp_offset_list_len_minus1', c_ubyte),
        ('cb_qp_offset_list', c_byte * 6),
        ('cr_qp_offset_list', c_byte * 6),
        ('reserved2', c_ubyte * 2),
        ('reserved3', c_uint * 8),
        ('NumBitsForShortTermRPSInSlice', c_int),
        ('NumDeltaPocsOfRefRpsIdx', c_int),
        ('NumPocTotalCurr', c_int),
        ('NumPocStCurrBefore', c_int),
        ('NumPocStCurrAfter', c_int),
        ('NumPocLtCurr', c_int),
        ('CurrPicOrderCntVal', c_int),
        ('RefPicIdx', c_int * 16),
        ('PicOrderCntVal', c_int * 16),
        ('IsLongTerm', c_ubyte * 16),
        ('RefPicSetStCurrBefore', c_ubyte * 8),
        ('RefPicSetStCurrAfter', c_ubyte * 8),
        ('RefPicSetLtCurr', c_ubyte * 8),
        ('RefPicSetInterLayer0', c_ubyte * 8),
        ('RefPicSetInterLayer1', c_ubyte * 8),
        ('reserved4', c_uint * 12),
        ('ScalingList4x4', c_ubyte * 16 * 6),
        ('ScalingList8x8', c_ubyte * 64 * 6),
        ('ScalingList16x16', c_ubyte * 64 * 6),
        ('ScalingList32x32', c_ubyte * 64 * 2),
        ('ScalingListDCCoeff16x16', c_ubyte * 6),
        ('ScalingListDCCoeff32x32', c_ubyte * 2),
    ]


class CUVIDVP9PICPARAMS(Structure):
    """A VP9 frame's parameters.

    ``LastRefIdx``, ``GoldenRefIdx`` and ``AltRefIdx`` name the pictures it
    predicts from.
    """

    _fields_ = [
        ('width', c_uint),
        ('height', c_uint),
        ('LastRefIdx', c_ubyte),
        ('GoldenRefIdx', c_ubyte),
        ('AltRefIdx', c_ubyte),
        ('colorSpace', c_ubyte),
        ('profile', c_ushort, 3),
        ('frameContextIdx', c_ushort, 2),
        ('frameType', c_ushort, 1),
        ('showFrame', c_ushort, 1),
        ('errorResilient', c_ushort, 1),
        ('frameParallelDecoding', c_ushort, 1),
        ('subSamplingX', c_ushort, 1),
        ('subSamplingY', c_ushort, 1),
        ('intraOnly', c_ushort, 1),
        ('allow_high_precision_mv', c_ushort, 1),
        ('refreshEntropyProbs', c_ushort, 1),
        ('reserved2Bits', c_ushort, 2),
        ('reserved16Bits', c_ushort),
        ('refFrameSignBias', c_ubyte * 4),
        ('bitDepthMinus8Luma', c_ubyte),
        ('bitDepthMinus8Chroma', c_ubyte),
        ('loopFilterLevel', c_ubyte),
        ('loopFilterSharpness', c_ubyte),
        ('modeRefLfEnabled', c_ubyte),
        ('log2_tile_columns', c_ubyte),
        ('log2_tile_rows', c_ubyte),
        ('segmentEnabled', c_ubyte, 1),
        ('segmentMapUpdate', c_ubyte, 1),
        ('segmentMapTemporalUpdate', c_ubyte, 1),
        ('segmentFeatureMode', c_ubyte, 1),
        ('reserved4Bits', c_ubyte, 4),
        ('segmentFeatureEnable', c_ubyte * 4 * 8),
        ('segmentFeatureData', c_short * 4 * 8),
        ('mb_segment_tree_probs', c_ubyte * 7),
        ('segment_pred_probs', c_ubyte * 3),
        ('reservedSegment16Bits', c_ubyte * 2),
        ('qpYAc', c_int),
        ('qpYDc', c_int),
        ('qpChDc', c_int),
        ('qpChAc', c_int),
        ('activeRefIdx', c_uint * 3),
        ('resetFrameContext', c_uint),
        ('mcomp_filter_type', c_uint),
        ('mbRefLfDelta', c_uint * 4),
        ('mbModeLfDelta', c_uint * 2),
        ('frameTagSize', c_uint),
        ('offsetToDctParts', c_uint),
        ('reserved128Bits', c_uint * 4),
    ]


class _AV1Reference(Structure):
    """One of the seven references of an AV1 inter frame: its size and its picture."""

    _fields_ = [
        ('width', c_uint),
        ('height', c_uint),
        ('index', c_ubyte),
        ('reserved24Bits', c_ubyte * 3),
    ]


class _AV1GlobalMotion(Structure):
    _fields_ = [
        ('invalid', c_ubyte, 1),
        ('wmtype', c_ubyte, 2),
        ('reserved5Bits', c_ubyte, 5),
        ('reserved24Bits', c_char * 3),
        ('wmmat', c_int * 6),
    ]


class CUVIDAV1PICPARAMS(Structure):
    """An AV1 frame's parameters.

    ``ref_frame`` names the seven pictures an inter frame predicts from, and
    ``ref_frame_map`` the picture in each of the eight reference slots.
    ``decodePicIdx`` is the picture the frame is reconstructed into, which is
    the one later frames predict from; it differs from ``CurrPicIdx`` for a
    frame with film grain, whose grain is added into the picture shown.
    """

    _fields_ = [
        ('width', c_uint),
        ('height', c_uint),
        ('frame_offset', c_uint),
        ('decodePicIdx', c_int),
        ('profile', c_uint, 3),
        ('use_128x128_superblock', c_uint, 1),
        ('subsampling_x', c_uint, 1),
        ('subsampling_y', c_uint, 1),
        ('mono_chrome', c_uint, 1),
        ('bit_depth_minus8', c_uint, 4),
        ('enable_filter_intra', c_uint, 1),
        ('enable_intra_edge_filter', c_uint, 1),
        ('enable_interintra_compound', c_uint, 1),
        ('enable_masked_compound', c_uint, 1),
        ('enable_dual_filter', c_uint, 1),
        ('enable_order_hint', c_uint, 1),
        ('order_hint_bits_minus1', c_uint, 3),
        ('enable_jnt_comp', c_uint, 1),
        ('enable_superres', c_uint, 1),
        ('enable_cdef', c_uint, 1),
        ('enable_restoration', c_uint, 1),
        ('enable_fgs', c_uint, 1),
        ('reserved0_7bits', c_uint, 7),
        ('frame_type', c_uint, 2),
        ('show_frame', c_uint, 1),
        ('disable_cdf_update', c_uint, 1),
        ('allow_screen_content_tools', c_uint, 1),
        ('force_integer_mv', c_uint, 1),
        ('coded_denom', c_uint, 3),
        ('allow_intrabc', c_uint, 1),
        ('allow_high_precision_mv', c_uint, 1),
        ('interp_filter', c_uint, 3),
        ('switchable_motion_mode', c_uint, 1),
        ('use_ref_frame_mvs', c_uint, 1),
        ('disable_frame_end_update_cdf', c_uint, 1),
        ('delta_q_present', c_uint, 1),
        ('delta_q_res', c_uint, 2),
        ('using_qmatrix', c_uint, 1),
        ('coded_lossless', c_uint, 1),
        ('use_superres', c_uint, 1),
        ('tx_mode', c_uint, 2),
        ('reference_mode', c_uint, 1),
        ('allow_warped_motion', c_uint, 1),
        ('reduced_tx_set', c_uint, 1),
        ('skip_mode', c_uint, 1),
        ('reserved1_3bits', c_uint, 3),
        ('num_tile_cols', c_uint, 8),
        ('num_tile_rows', c_uint, 8),
        ('context_update_tile_id', c_uint, 16),
        ('tile_widths', c_ushort * 64),
        ('tile_heights', c_ushort * 64),
        ('cdef_damping_minus_3', c_ubyte, 2),
        ('cdef_bits', c_ubyte, 2),
        ('reserved2_4bits', c_ubyte, 4),
        ('cdef_y_strength', c_ubyte * 8),
        ('cdef_uv_strength', c_ubyte * 8),
        ('SkipModeFrame0', c_ubyte, 4),
        ('SkipModeFrame1', c_ubyte, 4),
        ('base_qindex', c_ubyte),
        ('qp_y_dc_delta_q', c_byte),
        ('qp_u_dc_delta_q', c_byte),
        ('qp_v_dc_delta_q', c_byte),
        ('qp_u_ac_delta_q', c_byte),
        ('qp_v_ac_delta_q', c_byte),
        ('qm_y', c_ubyte),
        ('qm_u', c_ubyte),
        ('qm_v', c_ubyte),
        ('segmentation_enabled', c_ubyte, 1),
        ('segmentation_update_map', c_ubyte, 1),
        ('segmentation_update_data', c_ubyte, 1),
        ('segmentation_temporal_update', c_ubyte, 1),
        ('reserved3_4bits', c_ubyte, 4),
        ('segmentation_feature_data', c_short * 8 * 8),
        ('segmentation_feature_mask', c_ubyte * 8),
        ('loop_filter_level', c_ubyte * 2),
        ('loop_filter_level_u', c_ubyte),
        ('loop_filter_level_v', c_ubyte),
        ('loop_filter_sharpness', c_ubyte),
        ('loop_filter_ref_deltas', c_byte * 8),
        ('loop_filter_mode_deltas', c_byte * 2),
        ('loop_filter_delta_enabled', c_ubyte, 1),
        ('loop_filter_delta_update', c_ubyte, 1),
        ('delta_lf_present', c_ubyte, 1),
        ('delta_lf_res', c_ubyte, 2),
        ('delta_lf_multi', c_ubyte, 1),
        ('reserved4_2bits', c_ubyte, 2),
        ('lr_unit_size', c_ubyte * 3),
        ('lr_type', c_ubyte * 3),
        ('primary_ref_frame', c_ubyte),
        ('ref_frame_map', c_ubyte * 8),
        ('temporal_layer_id', c_ubyte, 4),
        ('spatial_layer_id', c_ubyte, 4),
        ('reserved5_32bits', c_ubyte * 4),
        ('ref_frame', _AV1Reference * 7),
        ('global_motion', _AV1GlobalMotion * 7),
        ('apply_grain', c_ushort, 1),
        ('overlap_flag', c_ushort, 1),
        ('scaling_shift_minus8', c_ushort, 2),
        ('chroma_scaling_from_luma', c_ushort, 1),
        ('ar_coeff_lag', c_ushort, 2),
        ('ar_coeff_shift_minus6', c_ushort, 2),
        ('grain_scale_shift', c_ushort, 2),
        ('clip_to_restricted_range', c_ushort, 1),
        ('reserved6_4bits', c_ushort, 4),
        ('num_y_points', c_ubyte),
        ('scaling_points_y', c_ubyte * 2 * 14),
        ('num_cb_points', c_ubyte),
        ('scaling_points_cb', c_ubyte * 2 * 10),
        ('num_cr_points', c_ubyte),
        ('scaling_points_cr', c_ubyte * 2 * 10),
        ('reserved7_8bits', c_ubyte),
        ('random_seed', c_ushort),
        ('ar_coeffs_y', c_short * 24),
        ('ar_coeffs_cb', c_short * 25),
        ('ar_coeffs_cr', c_short * 25),
        ('cb_mult', c_ubyte),
        ('cb_luma_mult', c_ubyte),
        ('cb_offset', c_short),
        ('cr_mult', c_ubyte),
        ('cr_luma_mult', c_ubyte),
        ('cr_offset', c_short),
        ('reserved', c_int * 7),
    ]


class _CodecSpecific(Union):
    """A picture's parameters in its codec's own terms.

    The arms for the codecs this package decodes are declared; ``CodecReserved``
    is the whole of the union, which the headers size it by.
    """

    _fields_ = [
        ('h264', CUVIDH264PICPARAMS),
        ('hevc', CUVIDHEVCPICPARAMS),
        ('vp9', CUVIDVP9PICPARAMS),
        ('av1', CUVIDAV1PICPARAMS),
        ('CodecReserved', c_uint * 1024),
    ]


class CUVIDPICPARAMS(Structure):
    """One picture's parameters, as the parser hands them to the decoder.

    The decoder takes the structure as the parser built it. This package reads
    it first: the picture the frame is decoded into, the bytes of bitstream it
    states, and the pictures it predicts from
    (:func:`pyopengl_video.nvdec.guard.refusal`).
    """

    _fields_ = [
        ('PicWidthInMbs', c_int),
        ('FrameHeightInMbs', c_int),
        ('CurrPicIdx', c_int),
        ('field_pic_flag', c_int),
        ('bottom_field_flag', c_int),
        ('second_field', c_int),
        ('nBitstreamDataLen', c_uint),
        ('pBitstreamData', c_void_p),
        ('nNumSlices', c_uint),
        ('pSliceDataOffsets', c_void_p),
        ('ref_pic_flag', c_int),
        ('intra_pic_flag', c_int),
        ('timestamp', c_longlong),
        ('Reserved', c_uint * 28),
        ('CodecSpecific', _CodecSpecific),
    ]


#: The parser's callbacks. Each returns 0 to fail the parse; the sequence
#: callback returns the number of decode surfaces the parser may cycle through.
SEQUENCE_CALLBACK = FUNCTYPE(c_int, c_void_p, POINTER(CUVIDEOFORMAT))
DECODE_CALLBACK = FUNCTYPE(c_int, c_void_p, POINTER(CUVIDPICPARAMS))
DISPLAY_CALLBACK = FUNCTYPE(c_int, c_void_p, POINTER(CUVIDPARSERDISPINFO))


class CUVIDPARSERPARAMS(Structure):
    _fields_ = [
        ('CodecType', c_int),
        ('ulMaxNumDecodeSurfaces', c_uint),
        ('ulClockRate', c_uint),
        ('ulErrorThreshold', c_uint),
        ('ulMaxDisplayDelay', c_uint),
        ('bAnnexb', c_uint, 1),
        ('bMemoryOptimize', c_uint, 1),
        ('uReserved', c_uint, 30),
        ('uReserved1', c_uint * 4),
        ('pUserData', c_void_p),
        ('pfnSequenceCallback', SEQUENCE_CALLBACK),
        ('pfnDecodePicture', DECODE_CALLBACK),
        ('pfnDisplayPicture', DISPLAY_CALLBACK),
        ('pfnGetOperatingPoint', c_void_p),
        ('pfnGetSEIMsg', c_void_p),
        ('pvReserved2', c_void_p * 5),
        ('pExtVideoInfo', c_void_p),
    ]


class CUVIDDECODECAPS(Structure):
    """What the hardware decodes for one codec, chroma format and bit depth."""

    _fields_ = [
        ('eCodecType', c_int),
        ('eChromaFormat', c_int),
        ('nBitDepthMinus8', c_uint),
        ('reserved1', c_uint * 3),
        ('bIsSupported', c_ubyte),
        ('nNumNVDECs', c_ubyte),
        ('nOutputFormatMask', c_ushort),
        ('nMaxWidth', c_uint),
        ('nMaxHeight', c_uint),
        ('nMaxMBCount', c_uint),
        ('nMinWidth', c_ushort),
        ('nMinHeight', c_ushort),
        ('bIsHistogramSupported', c_ubyte),
        ('nCounterBitDepth', c_ubyte),
        ('nMaxHistogramBins', c_ushort),
        ('bIsDecodeStatsSupported', c_ubyte),
        ('reserved4', c_ubyte * 3),
        ('reserved3', c_uint * 9),
    ]


class CUVIDDECODECREATEINFO(Structure):
    _fields_ = [
        ('ulWidth', c_ulong),
        ('ulHeight', c_ulong),
        ('ulNumDecodeSurfaces', c_ulong),
        ('CodecType', c_int),
        ('ChromaFormat', c_int),
        ('ulCreationFlags', c_ulong),
        ('bitDepthMinus8', c_ulong),
        ('ulIntraDecodeOnly', c_ulong),
        ('ulMaxWidth', c_ulong),
        ('ulMaxHeight', c_ulong),
        ('Reserved1', c_ulong),
        ('display_area', _ShortRectangle),
        ('OutputFormat', c_int),
        ('DeinterlaceMode', c_int),
        ('ulTargetWidth', c_ulong),
        ('ulTargetHeight', c_ulong),
        ('ulNumOutputSurfaces', c_ulong),
        ('vidLock', c_void_p),
        ('target_rect', _ShortRectangle),
        ('enableHistogram', c_ulong),
        ('enableDecodeFeatures', c_ulong),
        ('Reserved2', c_ulong * 3),
    ]


class CUVIDPROCPARAMS(Structure):
    """How a decoded picture is post-processed when it is mapped."""

    _fields_ = [
        ('progressive_frame', c_int),
        ('second_field', c_int),
        ('top_field_first', c_int),
        ('unpaired_field', c_int),
        ('reserved_flags', c_uint),
        ('reserved_zero', c_uint),
        ('raw_input_dptr', c_ulonglong),
        ('raw_input_pitch', c_uint),
        ('raw_input_format', c_uint),
        ('raw_output_dptr', c_ulonglong),
        ('raw_output_pitch', c_uint),
        ('Reserved1', c_uint),
        ('output_stream', c_void_p),
        ('Reserved', c_uint * 46),
        ('histogram_dptr', c_void_p),
        ('pCuvidProcExt', c_void_p),
    ]


# --------------------------------------------------------------------------
# the libraries
# --------------------------------------------------------------------------

def _bound(library: Any, name: str, *arguments: Any, result: Any = c_int) -> Callable[..., Any]:
    function = getattr(library, name)
    function.argtypes = list(arguments)
    function.restype = result
    return function


class Cuda:
    """The CUDA driver API: one instance per process.

    :meth:`load` initialises the driver on first use. Attribute access gives
    the bound calls under the names the headers use, without the ``_v2``
    suffix the exported symbols carry.
    """

    _instance: Cuda | None = None

    def __init__(self, library_name: str = CUDA_LIBRARY_NAME) -> None:
        library = self.library = _LOADER(library_name)
        pointer, handles = POINTER(c_void_p), POINTER(c_void_p)
        self.cuGetErrorName = _bound(library, 'cuGetErrorName', c_int, POINTER(c_char_p))
        self.cuGetErrorString = _bound(library, 'cuGetErrorString', c_int, POINTER(c_char_p))
        self.cuInit = _bound(library, 'cuInit', c_uint)
        self.cuDeviceGetCount = _bound(library, 'cuDeviceGetCount', POINTER(c_int))
        self.cuDeviceGet = _bound(library, 'cuDeviceGet', POINTER(c_int), c_int)
        self.cuDeviceGetName = _bound(library, 'cuDeviceGetName', c_char_p, c_int, c_int)
        self.cuDevicePrimaryCtxRetain = _bound(
            library, 'cuDevicePrimaryCtxRetain', pointer, c_int)
        self.cuCtxPushCurrent = _bound(library, 'cuCtxPushCurrent_v2', c_void_p)
        self.cuCtxPopCurrent = _bound(library, 'cuCtxPopCurrent_v2', pointer)
        self.cuMemcpy2D = _bound(library, 'cuMemcpy2D_v2', POINTER(CUDA_MEMCPY2D))
        self.cuGLGetDevices = _bound(
            library, 'cuGLGetDevices_v2', POINTER(c_uint), POINTER(c_int), c_uint, c_int)
        self.cuGraphicsGLRegisterImage = _bound(
            library, 'cuGraphicsGLRegisterImage', pointer, c_uint, c_uint, c_uint)
        self.cuGraphicsUnregisterResource = _bound(
            library, 'cuGraphicsUnregisterResource', c_void_p)
        self.cuGraphicsMapResources = _bound(
            library, 'cuGraphicsMapResources', c_uint, handles, c_void_p)
        self.cuGraphicsUnmapResources = _bound(
            library, 'cuGraphicsUnmapResources', c_uint, handles, c_void_p)
        self.cuGraphicsSubResourceGetMappedArray = _bound(
            library, 'cuGraphicsSubResourceGetMappedArray', pointer, c_void_p, c_uint, c_uint)
        self._contexts: dict[int, c_void_p] = {}
        self.check(self.cuInit(0), 'cuInit')

    @classmethod
    def load(cls) -> Cuda:
        """The process-wide driver interface, initialised on first use.

        Raises OSError where the library is absent and :class:`CUDAError`
        where the driver will not initialise.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def error_name(self, status: int) -> str:
        """The driver's name for a ``CUresult``, such as ``CUDA_ERROR_INVALID_VALUE``."""
        name = c_char_p()
        if self.cuGetErrorName(status, ctypes.byref(name)) != CUDA_SUCCESS or not name.value:
            return ''
        return name.value.decode('ascii', 'replace')

    def error_text(self, status: int) -> str:
        """The driver's description of a ``CUresult``, or '' where it has none."""
        text = c_char_p()
        if self.cuGetErrorString(status, ctypes.byref(text)) != CUDA_SUCCESS or not text.value:
            return ''
        return text.value.decode('ascii', 'replace')

    def check(self, status: int, call: str, detail: str = '') -> None:
        """Raise :class:`CUDAError` unless `status` is success."""
        if status != CUDA_SUCCESS:
            described = ': '.join(part for part in (self.error_text(status), detail) if part)
            raise CUDAError(status, call, self.error_name(status), described)

    def devices(self) -> list[int]:
        """Every CUDA device in the machine, as device handles."""
        count = c_int(0)
        self.check(self.cuDeviceGetCount(ctypes.byref(count)), 'cuDeviceGetCount')
        found = []
        for ordinal in range(count.value):
            device = c_int(0)
            self.check(self.cuDeviceGet(ctypes.byref(device), ordinal), 'cuDeviceGet')
            found.append(device.value)
        return found

    def gl_devices(self) -> list[int]:
        """The CUDA devices the current OpenGL context renders on.

        Empty where the context belongs to another vendor's GPU or to a
        software rasteriser, which the driver reports as an error or as no
        devices depending on the platform.
        """
        devices = (c_int * 8)()
        count = c_uint(0)
        status = self.cuGLGetDevices(ctypes.byref(count), devices, len(devices),
                                     CU_GL_DEVICE_LIST_ALL)
        if status != CUDA_SUCCESS:
            return []
        return [int(device) for device in devices[:count.value]]

    def device_name(self, device: int) -> str:
        """The marketing name of a device, such as ``NVIDIA GeForce RTX 3060 Ti``."""
        name = ctypes.create_string_buffer(128)
        self.check(self.cuDeviceGetName(name, len(name), device), 'cuDeviceGetName')
        return name.value.decode('utf-8', 'replace')

    def context(self, device: int) -> c_void_p:
        """The device's primary context, retained for the life of the process.

        The primary context is the one every CUDA library in a process shares
        (PyTorch and CuPy use it too), and the driver counts its users. This
        binding stays one of them once a decoder has been opened: making the
        context takes about 100 ms and destroying it 300 ms, measured on an
        RTX 3060 Ti, and a player opened and closed around each movie would
        otherwise pay both every time, on the thread that draws.
        """
        found = self._contexts.get(device)
        if found is None:
            found = c_void_p()
            self.check(self.cuDevicePrimaryCtxRetain(ctypes.byref(found), device),
                       'cuDevicePrimaryCtxRetain')
            self._contexts[device] = found
        return found

    @contextmanager
    def current(self, context: c_void_p) -> Iterator[None]:
        """Make `context` current on this thread for the scope, then put back the one before."""
        self.check(self.cuCtxPushCurrent(context), 'cuCtxPushCurrent')
        try:
            yield
        finally:
            self.cuCtxPopCurrent(ctypes.byref(c_void_p()))


class Cuvid:
    """NVDEC: the bitstream parser and the hardware decoder. One per process."""

    _instance: Cuvid | None = None

    def __init__(self, library_name: str = CUVID_LIBRARY_NAME) -> None:
        library = self.library = _LOADER(library_name)
        pointer = POINTER(c_void_p)
        self.cuvidCreateVideoParser = _bound(
            library, 'cuvidCreateVideoParser', pointer, POINTER(CUVIDPARSERPARAMS))
        self.cuvidParseVideoData = _bound(
            library, 'cuvidParseVideoData', c_void_p, POINTER(CUVIDSOURCEDATAPACKET))
        self.cuvidDestroyVideoParser = _bound(library, 'cuvidDestroyVideoParser', c_void_p)
        self.cuvidGetDecoderCaps = _bound(
            library, 'cuvidGetDecoderCaps', POINTER(CUVIDDECODECAPS))
        self.cuvidCreateDecoder = _bound(
            library, 'cuvidCreateDecoder', pointer, POINTER(CUVIDDECODECREATEINFO))
        self.cuvidDestroyDecoder = _bound(library, 'cuvidDestroyDecoder', c_void_p)
        self.cuvidDecodePicture = _bound(
            library, 'cuvidDecodePicture', c_void_p, POINTER(CUVIDPICPARAMS))
        self.cuvidMapVideoFrame = _bound(
            library, 'cuvidMapVideoFrame64', c_void_p, c_int, POINTER(c_ulonglong),
            POINTER(c_uint), POINTER(CUVIDPROCPARAMS))
        self.cuvidUnmapVideoFrame = _bound(
            library, 'cuvidUnmapVideoFrame64', c_void_p, c_ulonglong)

    @classmethod
    def load(cls) -> Cuvid:
        """The process-wide decoder interface, loading the library on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
