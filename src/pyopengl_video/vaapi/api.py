"""ctypes over libva: the types, constants and entry points this backend uses.

``libva.so.2`` is the encoder interface on Linux -- Intel parts through the
``iHD`` media driver, AMD parts through Mesa's ``radeonsi`` -- and
``libva-drm.so.2`` opens one against a DRM render node, which is what makes the
path work with no display server.

The binding is hand-written, and a structure declared at the wrong size is a
structure the driver reads past the end of. It does not report that: the calls
come back ``VA_STATUS_ERROR_INVALID_PARAMETER`` at best and produce a
well-formed stream of the wrong thing at worst. So every layout here is checked
against what a C compiler says the headers mean -- ``tests/test_va_abi.py``
against the recording ``tools/record_va_abi.py`` makes -- and the checks run
with no driver and no GPU.

The libva headers are MIT-licensed. They are not vendored here; only the facts
they state about layout and constant values are, which is what ``NOTICES.md``
records.
"""
from __future__ import annotations

import ctypes
import os
from collections.abc import Callable
from typing import Any

#: The libva runtime, and the piece of it that opens a DRM render node.
LIBRARY_NAME = 'libva.so.2'
DRM_LIBRARY_NAME = 'libva-drm.so.2'

# --------------------------------------------------------------- scalar types

VAStatus = ctypes.c_int32
VADisplay = ctypes.c_void_p
VAGenericID = ctypes.c_uint32
VASurfaceID = VAGenericID
VABufferID = VAGenericID
VAConfigID = VAGenericID
VAContextID = VAGenericID
VAProfile = ctypes.c_int32
VAEntrypoint = ctypes.c_int32

#: What ``VA_PADDING_LOW`` expands to, which is the tail of nearly every
#: structure and is what keeps its size stable as the interface grows.
PADDING_LOW = 4
PADDING_LARGE = 32

# ----------------------------------------------------------------- status

VA_STATUS_SUCCESS = 0

# ----------------------------------------------------------------- profiles

VAProfileNone = -1
VAProfileH264Main = 6
VAProfileH264High = 7
VAProfileH264ConstrainedBaseline = 13

VAEntrypointVLD = 1
VAEntrypointEncSlice = 6
VAEntrypointEncSliceLP = 8
VAEntrypointVideoProc = 10

# --------------------------------------------------------------- attributes

VAConfigAttribRTFormat = 0
VAConfigAttribRateControl = 5
VAConfigAttribEncPackedHeaders = 10
VAConfigAttribEncMaxRefFrames = 13

#: An attribute the driver does not answer comes back as this rather than as an
#: error, so a value has to be tested against it before it is believed.
VA_ATTRIB_NOT_SUPPORTED = 0x80000000

VA_RT_FORMAT_YUV420 = 0x00000001
VA_RT_FORMAT_RGB32 = 0x00020000

VA_RC_CBR = 0x00000002
VA_RC_VBR = 0x00000004
VA_RC_CQP = 0x00000010

#: Which packed headers a configuration asks for, as the bits of
#: ``VAConfigAttribEncPackedHeaders``.
VA_ENC_PACKED_HEADER_SEQUENCE = 0x00000001
VA_ENC_PACKED_HEADER_PICTURE = 0x00000002
VA_ENC_PACKED_HEADER_SLICE = 0x00000004
VA_ENC_PACKED_HEADER_MISC = 0x00000008
VA_ENC_PACKED_HEADER_RAW_DATA = 0x00000010

#: Which packed header one buffer *is*, as ``VAEncPackedHeaderType``. These
#: count from one and the attribute bits above are a mask, so the two spellings
#: of "slice" are 3 and 4 -- writing one where the other belongs asks the driver
#: for a header of a kind it will not use, and it reports nothing.
VAEncPackedHeaderSequence = 1
VAEncPackedHeaderPicture = 2
VAEncPackedHeaderSlice = 3
VAEncPackedHeaderRawData = 4

# ------------------------------------------------------------- buffer types

VAEncCodedBufferType = 21
VAEncSequenceParameterBufferType = 22
VAEncPictureParameterBufferType = 23
VAEncSliceParameterBufferType = 24
VAEncPackedHeaderParameterBufferType = 25
VAEncPackedHeaderDataBufferType = 26
VAEncMiscParameterBufferType = 27
VAProcPipelineParameterBufferType = 41

VAEncMiscParameterTypeFrameRate = 0
VAEncMiscParameterTypeRateControl = 1
VAEncMiscParameterTypeHRD = 5

# ---------------------------------------------------------- surface attributes

VASurfaceAttribPixelFormat = 1
VASurfaceAttribMemoryType = 6
VASurfaceAttribExternalBufferDescriptor = 7
VASurfaceAttribUsageHint = 8

VAGenericValueTypeInteger = 1
VAGenericValueTypePointer = 3

VA_SURFACE_ATTRIB_SETTABLE = 0x00000002

VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME = 0x20000000
VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2 = 0x40000000

VA_SURFACE_ATTRIB_USAGE_HINT_ENCODER = 0x00000002
VA_SURFACE_ATTRIB_USAGE_HINT_VPP_READ = 0x00000004
VA_SURFACE_ATTRIB_USAGE_HINT_VPP_WRITE = 0x00000008

VA_EXPORT_SURFACE_READ_ONLY = 0x0001
VA_EXPORT_SURFACE_WRITE_ONLY = 0x0002
VA_EXPORT_SURFACE_SEPARATE_LAYERS = 0x0004
VA_EXPORT_SURFACE_COMPOSED_LAYERS = 0x0008

# ------------------------------------------------------------------- pixels


def fourcc(text: str) -> int:
    """The four-character code `text` as the integer libva carries it in."""
    return int.from_bytes(text.encode('ascii'), 'little')


def fourcc_name(value: int) -> str:
    """The four-character code `value` as text."""
    return value.to_bytes(4, 'little').decode('ascii')


VA_FOURCC_NV12 = fourcc('NV12')
VA_FOURCC_RGBA = fourcc('RGBA')
VA_FOURCC_BGRA = fourcc('BGRA')
VA_FOURCC_ABGR = fourcc('ABGR')
VA_FOURCC_ARGB = fourcc('ARGB')

#: ``DRM_FORMAT_ABGR8888``: red in the low byte, which is what ``GL_RGBA8``
#: exports as. The DRM name reads in the opposite direction from the libva one.
DRM_FORMAT_ABGR8888 = fourcc('AB24')
DRM_FORMAT_ARGB8888 = fourcc('AR24')
DRM_FORMAT_XBGR8888 = fourcc('XB24')

#: What a driver returns for "this layout has no name I can give you".
DRM_FORMAT_MOD_INVALID = (1 << 56) - 1
DRM_FORMAT_MOD_LINEAR = 0

# ------------------------------------------------------------------ pictures

VA_PICTURE_H264_INVALID = 0x00000001
VA_PICTURE_H264_SHORT_TERM_REFERENCE = 0x00000008

VA_PROGRESSIVE = 0x1
VA_FRAME_PICTURE = 0x0

VAProcColorStandardNone = 0
VAProcColorStandardBT709 = 2

#: ``color_range`` in :class:`VAProcColorProperties`. Limited range is what the
#: sequence parameter set this backend writes declares.
VA_SOURCE_RANGE_UNKNOWN = 0
VA_SOURCE_RANGE_REDUCED = 1
VA_SOURCE_RANGE_FULL = 2

#: The low byte of a coded buffer's status holds the quantiser the picture was
#: coded at, which is worth logging when a stream comes out the wrong size.
VA_CODED_BUF_STATUS_PICTURE_AVE_QP_MASK = 0xff


# ------------------------------------------------------------------ structures


class VAConfigAttrib(ctypes.Structure):
    """One capability of a profile and entrypoint, queried or requested."""

    _fields_ = [('type', ctypes.c_int32), ('value', ctypes.c_uint32)]


class _VAGenericValueUnion(ctypes.Union):
    _fields_ = [('i', ctypes.c_int32), ('f', ctypes.c_float),
                ('p', ctypes.c_void_p), ('fn', ctypes.c_void_p)]


class VAGenericValue(ctypes.Structure):
    """A tagged value, which is how a surface attribute carries its payload."""

    _fields_ = [('type', ctypes.c_int32), ('value', _VAGenericValueUnion)]


class VASurfaceAttrib(ctypes.Structure):
    """One property of a surface being created, such as where its memory is."""

    _fields_ = [('type', ctypes.c_int32), ('flags', ctypes.c_uint32),
                ('value', VAGenericValue)]


class VARectangle(ctypes.Structure):
    _fields_ = [('x', ctypes.c_int16), ('y', ctypes.c_int16),
                ('width', ctypes.c_uint16), ('height', ctypes.c_uint16)]


class VACodedBufferSegment(ctypes.Structure):
    """One run of compressed bytes, and where the next one is.

    An encoder may answer in several segments, so reading a coded buffer means
    following ``next`` until it is null.
    """


VACodedBufferSegment._fields_ = [
    ('size', ctypes.c_uint32),
    ('bit_offset', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32),
    ('buf', ctypes.c_void_p),
    ('next', ctypes.POINTER(VACodedBufferSegment)),
    ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
]


class VAPictureH264(ctypes.Structure):
    """One entry of the decoded picture buffer, or the picture being coded.

    picture_id -- the surface the driver reconstructs into, or reads from
    frame_idx -- the picture's frame_num
    flags -- what kind of reference it is, or ``VA_PICTURE_H264_INVALID`` for an
        unused slot; every slot of a reference list has to be filled in
    """

    _fields_ = [
        ('picture_id', VASurfaceID),
        ('frame_idx', ctypes.c_uint32),
        ('flags', ctypes.c_uint32),
        ('TopFieldOrderCnt', ctypes.c_int32),
        ('BottomFieldOrderCnt', ctypes.c_int32),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]

    @classmethod
    def unused(cls) -> VAPictureH264:
        """A slot holding no picture, which is what an empty list is made of."""
        return cls(picture_id=0xFFFFFFFF, frame_idx=0,
                   flags=VA_PICTURE_H264_INVALID)


class _SeqFieldsBits(ctypes.Structure):
    _fields_ = [
        ('chroma_format_idc', ctypes.c_uint32, 2),
        ('frame_mbs_only_flag', ctypes.c_uint32, 1),
        ('mb_adaptive_frame_field_flag', ctypes.c_uint32, 1),
        ('seq_scaling_matrix_present_flag', ctypes.c_uint32, 1),
        ('direct_8x8_inference_flag', ctypes.c_uint32, 1),
        ('log2_max_frame_num_minus4', ctypes.c_uint32, 4),
        ('pic_order_cnt_type', ctypes.c_uint32, 2),
        ('log2_max_pic_order_cnt_lsb_minus4', ctypes.c_uint32, 4),
        ('delta_pic_order_always_zero_flag', ctypes.c_uint32, 1),
    ]


class _SeqFields(ctypes.Union):
    _fields_ = [('bits', _SeqFieldsBits), ('value', ctypes.c_uint32)]


class _VuiFieldsBits(ctypes.Structure):
    _fields_ = [
        ('aspect_ratio_info_present_flag', ctypes.c_uint32, 1),
        ('timing_info_present_flag', ctypes.c_uint32, 1),
        ('bitstream_restriction_flag', ctypes.c_uint32, 1),
        ('log2_max_mv_length_horizontal', ctypes.c_uint32, 5),
        ('log2_max_mv_length_vertical', ctypes.c_uint32, 5),
        ('fixed_frame_rate_flag', ctypes.c_uint32, 1),
        ('low_delay_hrd_flag', ctypes.c_uint32, 1),
        ('motion_vectors_over_pic_boundaries_flag', ctypes.c_uint32, 1),
        ('reserved', ctypes.c_uint32, 16),
    ]


class _VuiFields(ctypes.Union):
    _fields_ = [('bits', _VuiFieldsBits), ('value', ctypes.c_uint32)]


class VAEncSequenceParameterBufferH264(ctypes.Structure):
    """What the driver is told once per sequence, and again at every IDR."""

    _fields_ = [
        ('seq_parameter_set_id', ctypes.c_uint8),
        ('level_idc', ctypes.c_uint8),
        ('intra_period', ctypes.c_uint32),
        ('intra_idr_period', ctypes.c_uint32),
        ('ip_period', ctypes.c_uint32),
        ('bits_per_second', ctypes.c_uint32),
        ('max_num_ref_frames', ctypes.c_uint32),
        ('picture_width_in_mbs', ctypes.c_uint16),
        ('picture_height_in_mbs', ctypes.c_uint16),
        ('seq_fields', _SeqFields),
        ('bit_depth_luma_minus8', ctypes.c_uint8),
        ('bit_depth_chroma_minus8', ctypes.c_uint8),
        ('num_ref_frames_in_pic_order_cnt_cycle', ctypes.c_uint8),
        ('offset_for_non_ref_pic', ctypes.c_int32),
        ('offset_for_top_to_bottom_field', ctypes.c_int32),
        ('offset_for_ref_frame', ctypes.c_int32 * 256),
        ('frame_cropping_flag', ctypes.c_uint8),
        ('frame_crop_left_offset', ctypes.c_uint32),
        ('frame_crop_right_offset', ctypes.c_uint32),
        ('frame_crop_top_offset', ctypes.c_uint32),
        ('frame_crop_bottom_offset', ctypes.c_uint32),
        ('vui_parameters_present_flag', ctypes.c_uint8),
        ('vui_fields', _VuiFields),
        ('aspect_ratio_idc', ctypes.c_uint8),
        ('sar_width', ctypes.c_uint32),
        ('sar_height', ctypes.c_uint32),
        ('num_units_in_tick', ctypes.c_uint32),
        ('time_scale', ctypes.c_uint32),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


class _PicFieldsBits(ctypes.Structure):
    _fields_ = [
        ('idr_pic_flag', ctypes.c_uint32, 1),
        ('reference_pic_flag', ctypes.c_uint32, 2),
        ('entropy_coding_mode_flag', ctypes.c_uint32, 1),
        ('weighted_pred_flag', ctypes.c_uint32, 1),
        ('weighted_bipred_idc', ctypes.c_uint32, 2),
        ('constrained_intra_pred_flag', ctypes.c_uint32, 1),
        ('transform_8x8_mode_flag', ctypes.c_uint32, 1),
        ('deblocking_filter_control_present_flag', ctypes.c_uint32, 1),
        ('redundant_pic_cnt_present_flag', ctypes.c_uint32, 1),
        ('pic_order_present_flag', ctypes.c_uint32, 1),
        ('pic_scaling_matrix_present_flag', ctypes.c_uint32, 1),
    ]


class _PicFields(ctypes.Union):
    _fields_ = [('bits', _PicFieldsBits), ('value', ctypes.c_uint32)]


class VAEncPictureParameterBufferH264(ctypes.Structure):
    """What the driver is told about the picture it is about to code."""

    _fields_ = [
        ('CurrPic', VAPictureH264),
        ('ReferenceFrames', VAPictureH264 * 16),
        ('coded_buf', VABufferID),
        ('pic_parameter_set_id', ctypes.c_uint8),
        ('seq_parameter_set_id', ctypes.c_uint8),
        ('last_picture', ctypes.c_uint8),
        ('frame_num', ctypes.c_uint16),
        ('pic_init_qp', ctypes.c_uint8),
        ('num_ref_idx_l0_active_minus1', ctypes.c_uint8),
        ('num_ref_idx_l1_active_minus1', ctypes.c_uint8),
        ('chroma_qp_index_offset', ctypes.c_int8),
        ('second_chroma_qp_index_offset', ctypes.c_int8),
        ('pic_fields', _PicFields),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


class VAEncSliceParameterBufferH264(ctypes.Structure):
    """The one slice each picture here is coded as, and its reference lists."""

    _fields_ = [
        ('macroblock_address', ctypes.c_uint32),
        ('num_macroblocks', ctypes.c_uint32),
        ('macroblock_info', VABufferID),
        ('slice_type', ctypes.c_uint8),
        ('pic_parameter_set_id', ctypes.c_uint8),
        ('idr_pic_id', ctypes.c_uint16),
        ('pic_order_cnt_lsb', ctypes.c_uint16),
        ('delta_pic_order_cnt_bottom', ctypes.c_int32),
        ('delta_pic_order_cnt', ctypes.c_int32 * 2),
        ('direct_spatial_mv_pred_flag', ctypes.c_uint8),
        ('num_ref_idx_active_override_flag', ctypes.c_uint8),
        ('num_ref_idx_l0_active_minus1', ctypes.c_uint8),
        ('num_ref_idx_l1_active_minus1', ctypes.c_uint8),
        ('RefPicList0', VAPictureH264 * 32),
        ('RefPicList1', VAPictureH264 * 32),
        ('luma_log2_weight_denom', ctypes.c_uint8),
        ('chroma_log2_weight_denom', ctypes.c_uint8),
        ('luma_weight_l0_flag', ctypes.c_uint8),
        ('luma_weight_l0', ctypes.c_int16 * 32),
        ('luma_offset_l0', ctypes.c_int16 * 32),
        ('chroma_weight_l0_flag', ctypes.c_uint8),
        ('chroma_weight_l0', (ctypes.c_int16 * 2) * 32),
        ('chroma_offset_l0', (ctypes.c_int16 * 2) * 32),
        ('luma_weight_l1_flag', ctypes.c_uint8),
        ('luma_weight_l1', ctypes.c_int16 * 32),
        ('luma_offset_l1', ctypes.c_int16 * 32),
        ('chroma_weight_l1_flag', ctypes.c_uint8),
        ('chroma_weight_l1', (ctypes.c_int16 * 2) * 32),
        ('chroma_offset_l1', (ctypes.c_int16 * 2) * 32),
        ('cabac_init_idc', ctypes.c_uint8),
        ('slice_qp_delta', ctypes.c_int8),
        ('disable_deblocking_filter_idc', ctypes.c_uint8),
        ('slice_alpha_c0_offset_div2', ctypes.c_int8),
        ('slice_beta_offset_div2', ctypes.c_int8),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


class VAEncPackedHeaderParameterBuffer(ctypes.Structure):
    """Says how long a packed header is, and whether it is already escaped."""

    _fields_ = [
        ('type', ctypes.c_uint32),
        ('bit_length', ctypes.c_uint32),
        ('has_emulation_bytes', ctypes.c_uint8),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


class _RateControlFlagsBits(ctypes.Structure):
    _fields_ = [
        ('reset', ctypes.c_uint32, 1),
        ('disable_frame_skip', ctypes.c_uint32, 1),
        ('disable_bit_stuffing', ctypes.c_uint32, 1),
        ('mb_rate_control', ctypes.c_uint32, 4),
        ('temporal_id', ctypes.c_uint32, 8),
        ('cfs_I_frames', ctypes.c_uint32, 1),
        ('enable_parallel_brc', ctypes.c_uint32, 1),
        ('enable_dynamic_scaling', ctypes.c_uint32, 1),
        ('frame_tolerance_mode', ctypes.c_uint32, 2),
        ('reserved', ctypes.c_uint32, 12),
    ]


class _RateControlFlags(ctypes.Union):
    _fields_ = [('bits', _RateControlFlagsBits), ('value', ctypes.c_uint32)]


class VAEncMiscParameterRateControl(ctypes.Structure):
    _fields_ = [
        ('bits_per_second', ctypes.c_uint32),
        ('target_percentage', ctypes.c_uint32),
        ('window_size', ctypes.c_uint32),
        ('initial_qp', ctypes.c_uint32),
        ('min_qp', ctypes.c_uint32),
        ('basic_unit_size', ctypes.c_uint32),
        ('rc_flags', _RateControlFlags),
        ('ICQ_quality_factor', ctypes.c_uint32),
        ('max_qp', ctypes.c_uint32),
        ('quality_factor', ctypes.c_uint32),
        ('target_frame_size', ctypes.c_uint32),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


class _FrameRateFlagsBits(ctypes.Structure):
    _fields_ = [('temporal_id', ctypes.c_uint32, 8),
                ('reserved', ctypes.c_uint32, 24)]


class _FrameRateFlags(ctypes.Union):
    _fields_ = [('bits', _FrameRateFlagsBits), ('value', ctypes.c_uint32)]


class VAEncMiscParameterFrameRate(ctypes.Structure):
    """The frame rate, as a numerator in the low half and denominator above it."""

    _fields_ = [
        ('framerate', ctypes.c_uint32),
        ('framerate_flags', _FrameRateFlags),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


class VAEncMiscParameterHRD(ctypes.Structure):
    _fields_ = [
        ('initial_buffer_fullness', ctypes.c_uint32),
        ('buffer_size', ctypes.c_uint32),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


class VAProcColorProperties(ctypes.Structure):
    """How a video processing surface's colour is to be read or written."""

    _fields_ = [
        ('chroma_sample_location', ctypes.c_uint8),
        ('color_range', ctypes.c_uint8),
        ('colour_primaries', ctypes.c_uint8),
        ('transfer_characteristics', ctypes.c_uint8),
        ('matrix_coefficients', ctypes.c_uint8),
        ('reserved', ctypes.c_uint8 * 3),
    ]


class VAProcPipelineParameterBuffer(ctypes.Structure):
    """One video processing pass: a source surface into the current target.

    The reserved tail is shorter on a 64-bit build, because the pointers ahead
    of it are longer; this binding is 64-bit, as every platform it runs on is.
    """

    _fields_ = [
        ('surface', VASurfaceID),
        ('surface_region', ctypes.POINTER(VARectangle)),
        ('surface_color_standard', ctypes.c_int32),
        ('output_region', ctypes.POINTER(VARectangle)),
        ('output_background_color', ctypes.c_uint32),
        ('output_color_standard', ctypes.c_int32),
        ('pipeline_flags', ctypes.c_uint32),
        ('filter_flags', ctypes.c_uint32),
        ('filters', ctypes.POINTER(VABufferID)),
        ('num_filters', ctypes.c_uint32),
        ('forward_references', ctypes.POINTER(VASurfaceID)),
        ('num_forward_references', ctypes.c_uint32),
        ('backward_references', ctypes.POINTER(VASurfaceID)),
        ('num_backward_references', ctypes.c_uint32),
        ('rotation_state', ctypes.c_uint32),
        ('blend_state', ctypes.c_void_p),
        ('mirror_state', ctypes.c_uint32),
        ('additional_outputs', ctypes.POINTER(VASurfaceID)),
        ('num_additional_outputs', ctypes.c_uint32),
        ('input_surface_flag', ctypes.c_uint32),
        ('output_surface_flag', ctypes.c_uint32),
        ('input_color_properties', VAProcColorProperties),
        ('output_color_properties', VAProcColorProperties),
        ('processing_mode', ctypes.c_int32),
        ('output_hdr_metadata', ctypes.c_void_p),
        ('va_reserved', ctypes.c_uint32 * (PADDING_LARGE - 16)),
    ]


class VAImageFormat(ctypes.Structure):
    """How the samples of a mapped surface are laid out."""

    _fields_ = [
        ('fourcc', ctypes.c_uint32),
        ('byte_order', ctypes.c_uint32),
        ('bits_per_pixel', ctypes.c_uint32),
        ('depth', ctypes.c_uint32),
        ('red_mask', ctypes.c_uint32),
        ('green_mask', ctypes.c_uint32),
        ('blue_mask', ctypes.c_uint32),
        ('alpha_mask', ctypes.c_uint32),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


class VAImage(ctypes.Structure):
    """A surface's samples, addressable from the host.

    ``vaDeriveImage`` gives one over a surface's own memory where the driver
    allows it, which is how a test reads back what the hardware wrote without
    a decoder in the way.
    """

    _fields_ = [
        ('image_id', VAGenericID),
        ('format', VAImageFormat),
        ('buf', VABufferID),
        ('width', ctypes.c_uint16),
        ('height', ctypes.c_uint16),
        ('data_size', ctypes.c_uint32),
        ('num_planes', ctypes.c_uint32),
        ('pitches', ctypes.c_uint32 * 3),
        ('offsets', ctypes.c_uint32 * 3),
        ('num_palette_entries', ctypes.c_int32),
        ('entry_bytes', ctypes.c_int32),
        ('component_order', ctypes.c_int8 * 4),
        ('va_reserved', ctypes.c_uint32 * PADDING_LOW),
    ]


class _PrimeObject(ctypes.Structure):
    _fields_ = [('fd', ctypes.c_int), ('size', ctypes.c_uint32),
                ('drm_format_modifier', ctypes.c_uint64)]


class _PrimeLayer(ctypes.Structure):
    _fields_ = [
        ('drm_format', ctypes.c_uint32),
        ('num_planes', ctypes.c_uint32),
        ('object_index', ctypes.c_uint32 * 4),
        ('offset', ctypes.c_uint32 * 4),
        ('pitch', ctypes.c_uint32 * 4),
    ]


class VADRMPRIMESurfaceDescriptor(ctypes.Structure):
    """A surface described as DMA-BUF file descriptors and their layout.

    Handed to :func:`VA.create_surface_from_prime` to import a buffer another
    API allocated, and filled in by ``vaExportSurfaceHandle`` to export one.
    """

    _fields_ = [
        ('fourcc', ctypes.c_uint32),
        ('width', ctypes.c_uint32),
        ('height', ctypes.c_uint32),
        ('num_objects', ctypes.c_uint32),
        ('objects', _PrimeObject * 4),
        ('num_layers', ctypes.c_uint32),
        ('layers', _PrimeLayer * 4),
    ]


#: What ``VAEncMiscParameterBuffer`` is: a type tag and a flexible payload. The
#: concrete structures are built per payload type by :func:`misc_parameter`.
MISC_PARAMETER_TYPES = {
    VAEncMiscParameterTypeRateControl: VAEncMiscParameterRateControl,
    VAEncMiscParameterTypeFrameRate: VAEncMiscParameterFrameRate,
    VAEncMiscParameterTypeHRD: VAEncMiscParameterHRD,
}

_misc_structures: dict[type, type] = {}


def misc_parameter(kind: int, payload: ctypes.Structure) -> ctypes.Structure:
    """Wrap `payload` in the type-tagged envelope a misc parameter buffer is.

    The C declaration ends in a flexible array member, which ctypes has no
    spelling for, so the envelope is built for the payload's own type and cached
    against it.
    """
    structure = _misc_structures.get(type(payload))
    if structure is None:
        structure = type(
            f'VAEncMiscParameter{type(payload).__name__[18:]}',
            (ctypes.Structure,),
            {'_fields_': [('type', ctypes.c_uint32), ('data', type(payload))]},
        )
        _misc_structures[type(payload)] = structure
    return structure(type=kind, data=payload)


# ------------------------------------------------------------------- errors


class VAError(RuntimeError):
    """A libva call failed, named with the status it returned.

    status -- the ``VAStatus`` the call returned
    call -- the entry point that returned it
    context -- what was being attempted, for a message that means something
    """

    def __init__(self, status: int, call: str, context: str = '') -> None:
        self.status = int(status)
        self.call = call
        described = _describe(status)
        detail = f' ({context})' if context else ''
        super().__init__(f'{call} failed{detail}: {described}')


def _describe(status: int) -> str:
    """The driver's own text for a status, or the number when there is none."""
    try:
        text = VA.instance().error_string(status)
    except Exception:  # noqa: BLE001 - describing a failure must not fail
        text = None
    return f'{text} ({status:#x})' if text else f'VAStatus {status:#x}'


# ---------------------------------------------------------------- the library


#: The callback libva hands its own log lines to. Keeping the object alive
#: matters: the driver holds the pointer for as long as the display is open.
MESSAGE_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p)


class VA:
    """The libva entry points, loaded once per process.

    Constructing this loads ``libva.so.2`` and ``libva-drm.so.2``; it raises
    :class:`OSError` where they are not installed, which is what a probe catches.

    The entry points below are bound onto the instance by :meth:`_bind`, from
    the one table that also gives each its argument types; they are declared
    here so that a caller naming one that does not exist is a mistake a checker
    finds rather than an ``AttributeError`` at the far end of a recording.
    """

    _instance: VA | None = None

    vaErrorStr: Callable[..., bytes | None]
    vaQueryVendorString: Callable[..., bytes | None]
    vaDisplayIsValid: Callable[..., int]
    vaInitialize: Callable[..., int]
    vaTerminate: Callable[..., int]
    vaSetInfoCallback: Callable[..., Any]
    vaSetErrorCallback: Callable[..., Any]
    vaMaxNumEntrypoints: Callable[..., int]
    vaQueryConfigEntrypoints: Callable[..., int]
    vaGetConfigAttributes: Callable[..., int]
    vaCreateConfig: Callable[..., int]
    vaDestroyConfig: Callable[..., int]
    vaCreateSurfaces: Callable[..., int]
    vaDestroySurfaces: Callable[..., int]
    vaCreateContext: Callable[..., int]
    vaDestroyContext: Callable[..., int]
    vaCreateBuffer: Callable[..., int]
    vaDestroyBuffer: Callable[..., int]
    vaMapBuffer: Callable[..., int]
    vaUnmapBuffer: Callable[..., int]
    vaBeginPicture: Callable[..., int]
    vaRenderPicture: Callable[..., int]
    vaEndPicture: Callable[..., int]
    vaSyncSurface: Callable[..., int]
    vaDeriveImage: Callable[..., int]
    vaDestroyImage: Callable[..., int]
    vaExportSurfaceHandle: Callable[..., int]
    vaGetDisplayDRM: Callable[..., Any]

    def __init__(self) -> None:
        self.library = ctypes.CDLL(LIBRARY_NAME)
        self.drm_library = ctypes.CDLL(DRM_LIBRARY_NAME)
        self._bind()

    @classmethod
    def instance(cls) -> VA:
        """The process-wide binding, loaded on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _bind(self) -> None:
        """Give every entry point this backend calls its argument types.

        ctypes defaults an unprototyped call's arguments to int, which
        truncates a pointer on a 64-bit build, so nothing here is left
        unprototyped.
        """
        signatures: list[tuple[Any, str, list[Any], Any]] = [
            (self.library, 'vaErrorStr', [VAStatus], ctypes.c_char_p),
            (self.library, 'vaQueryVendorString', [VADisplay], ctypes.c_char_p),
            (self.library, 'vaDisplayIsValid', [VADisplay], ctypes.c_int),
            (self.library, 'vaInitialize',
             [VADisplay, ctypes.POINTER(ctypes.c_int),
              ctypes.POINTER(ctypes.c_int)], VAStatus),
            (self.library, 'vaTerminate', [VADisplay], VAStatus),
            (self.library, 'vaSetInfoCallback',
             [VADisplay, MESSAGE_CALLBACK, ctypes.c_void_p], ctypes.c_void_p),
            (self.library, 'vaSetErrorCallback',
             [VADisplay, MESSAGE_CALLBACK, ctypes.c_void_p], ctypes.c_void_p),
            (self.library, 'vaMaxNumEntrypoints', [VADisplay], ctypes.c_int),
            (self.library, 'vaQueryConfigEntrypoints',
             [VADisplay, VAProfile, ctypes.POINTER(VAEntrypoint),
              ctypes.POINTER(ctypes.c_int)], VAStatus),
            (self.library, 'vaGetConfigAttributes',
             [VADisplay, VAProfile, VAEntrypoint,
              ctypes.POINTER(VAConfigAttrib), ctypes.c_int], VAStatus),
            (self.library, 'vaCreateConfig',
             [VADisplay, VAProfile, VAEntrypoint,
              ctypes.POINTER(VAConfigAttrib), ctypes.c_int,
              ctypes.POINTER(VAConfigID)], VAStatus),
            (self.library, 'vaDestroyConfig', [VADisplay, VAConfigID], VAStatus),
            (self.library, 'vaCreateSurfaces',
             [VADisplay, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
              ctypes.POINTER(VASurfaceID), ctypes.c_uint32,
              ctypes.POINTER(VASurfaceAttrib), ctypes.c_uint32], VAStatus),
            (self.library, 'vaDestroySurfaces',
             [VADisplay, ctypes.POINTER(VASurfaceID), ctypes.c_int], VAStatus),
            (self.library, 'vaCreateContext',
             [VADisplay, VAConfigID, ctypes.c_int, ctypes.c_int, ctypes.c_int,
              ctypes.POINTER(VASurfaceID), ctypes.c_int,
              ctypes.POINTER(VAContextID)], VAStatus),
            (self.library, 'vaDestroyContext', [VADisplay, VAContextID], VAStatus),
            (self.library, 'vaCreateBuffer',
             [VADisplay, VAContextID, ctypes.c_int, ctypes.c_uint32,
              ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(VABufferID)],
             VAStatus),
            (self.library, 'vaDestroyBuffer', [VADisplay, VABufferID], VAStatus),
            (self.library, 'vaMapBuffer',
             [VADisplay, VABufferID, ctypes.POINTER(ctypes.c_void_p)], VAStatus),
            (self.library, 'vaUnmapBuffer', [VADisplay, VABufferID], VAStatus),
            (self.library, 'vaBeginPicture',
             [VADisplay, VAContextID, VASurfaceID], VAStatus),
            (self.library, 'vaRenderPicture',
             [VADisplay, VAContextID, ctypes.POINTER(VABufferID), ctypes.c_int],
             VAStatus),
            (self.library, 'vaEndPicture', [VADisplay, VAContextID], VAStatus),
            (self.library, 'vaSyncSurface', [VADisplay, VASurfaceID], VAStatus),
            (self.library, 'vaDeriveImage',
             [VADisplay, VASurfaceID, ctypes.POINTER(VAImage)], VAStatus),
            (self.library, 'vaDestroyImage', [VADisplay, VAGenericID], VAStatus),
            (self.library, 'vaExportSurfaceHandle',
             [VADisplay, VASurfaceID, ctypes.c_uint32, ctypes.c_uint32,
              ctypes.c_void_p], VAStatus),
            (self.drm_library, 'vaGetDisplayDRM', [ctypes.c_int], VADisplay),
        ]
        for library, name, arguments, result in signatures:
            entry = getattr(library, name)
            entry.argtypes = arguments
            entry.restype = result
            setattr(self, name, entry)

    # ------------------------------------------------------------- helpers

    def error_string(self, status: int) -> str:
        """libva's own description of a status code."""
        text = self.vaErrorStr(status)
        return text.decode('utf-8', 'replace') if text else ''

    def check(self, status: int, call: str, context: str = '') -> None:
        """Raise :class:`VAError` unless `status` says the call succeeded."""
        if status != VA_STATUS_SUCCESS:
            raise VAError(status, call, context)


def render_nodes() -> list[str]:
    """Every DRM render node on this machine, in the order they are numbered.

    A render node is the device an encoder is opened on, and it needs no
    display server -- which is what makes this path work in a container, over
    ssh, and on a machine with no seat at all.
    """
    directory = '/dev/dri'
    try:
        entries = os.listdir(directory)
    except OSError:
        return []
    return sorted(f'{directory}/{name}' for name in entries
                  if name.startswith('renderD'))
