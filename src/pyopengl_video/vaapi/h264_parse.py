"""The H.264 parser: NAL units, parameter sets and slice headers.

A libva decoder is handed the parsed syntax, not the bitstream: the sequence
and picture parameter sets become a picture parameter buffer and an inverse
quantisation matrix, and each slice header becomes a slice parameter buffer.
The driver decodes the macroblocks that follow the header. This module reads
that syntax and holds no libva.

Field names are the syntax elements of ITU-T H.264 (clause 7.3 and annex E),
so the parser can be read beside the specification's syntax tables. Derived
values the decoder needs, such as the visible rectangle, the frame rate and
the size of the decoded picture buffer, are properties of the parsed sets.

Every NAL unit type a progressive High-profile stream uses is read in full.
Interlaced syntax (field pictures, MBAFF) and slice groups are parsed and
reported, and :mod:`pyopengl_video.vaapi.decoder` refuses them.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from math import gcd
from typing import Any

from pyopengl_video.colour import Colour
from pyopengl_video.vaapi.h264 import BitReader, StreamError, rbsp_of

__all__ = [
    'DEFAULT_4X4_INTER', 'DEFAULT_4X4_INTRA', 'DEFAULT_8X8_INTER',
    'DEFAULT_8X8_INTRA', 'FLAT_4X4', 'FLAT_8X8', 'MemoryManagementOperation',
    'NalUnit', 'PictureParameterSet', 'PredWeightTable', 'SequenceParameterSet',
    'SliceHeader', 'StreamError', 'VideoUsability', 'colour_of', 'parse_nal', 'parse_pps',
    'parse_slice_header', 'parse_sps', 'scaling_matrices', 'starts_new_picture',
]

# --------------------------------------------------------------- NAL types

NAL_SLICE = 1
NAL_SLICE_PARTITION_A = 2
NAL_IDR = 5
NAL_SEI = 6
NAL_SPS = 7
NAL_PPS = 8
NAL_AUD = 9
NAL_END_OF_SEQUENCE = 10
NAL_END_OF_STREAM = 11

#: profile_idc values whose SPS carries chroma format, bit depth and scaling
#: syntax (7.3.2.1.1).
HIGH_PROFILES = frozenset({100, 110, 122, 244, 44, 83, 86, 118, 128, 138,
                           139, 134, 135})

#: Profiles that are intra-only when constraint_set3_flag is set, and so never
#: reorder (A.2.8 to A.2.11).
INTRA_PROFILES = frozenset({44, 86, 100, 110, 122, 244})

#: slice_type modulo 5, as named in table 7-6.
SLICE_P, SLICE_B, SLICE_I, SLICE_SP, SLICE_SI = range(5)
SLICE_KINDS = ('P', 'B', 'I', 'SP', 'SI')

#: SampleAspectRatio for aspect_ratio_idc 1 to 16 (table E-1).
ASPECT_RATIOS = {
    1: (1, 1), 2: (12, 11), 3: (10, 11), 4: (16, 11), 5: (40, 33),
    6: (24, 11), 7: (20, 11), 8: (32, 11), 9: (80, 33), 10: (18, 11),
    11: (15, 11), 12: (64, 33), 13: (160, 99), 14: (4, 3), 15: (3, 2),
    16: (2, 1),
}
EXTENDED_SAR = 255

#: MaxDpbMbs by level_idc (table A-1). Level 1b is level_idc 9, or 11 with
#: constraint_set3_flag in Baseline, Main and Extended.
MAX_DPB_MBS = {
    9: 396, 10: 396, 11: 900, 12: 2376, 13: 2376, 20: 2376, 21: 4752,
    22: 8100, 30: 8100, 31: 18000, 32: 20480, 40: 32768, 41: 32768,
    42: 34816, 50: 110400, 51: 184320, 52: 184320, 60: 696320, 61: 696320,
    62: 696320,
}
#: No decoded picture buffer holds more frames than this (A.3.1).
MAX_DPB_FRAMES = 16

# ------------------------------------------------------------ scaling lists

#: Flat_4x4_16 and Flat_8x8_16: the lists in force with no scaling matrix.
FLAT_4X4: tuple[int, ...] = (16,) * 16
FLAT_8X8: tuple[int, ...] = (16,) * 64

#: Default_4x4_Intra and Default_4x4_Inter (table 7-3), in zig-zag order.
DEFAULT_4X4_INTRA: tuple[int, ...] = (6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42)
DEFAULT_4X4_INTER: tuple[int, ...] = (10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34)

#: Default_8x8_Intra and Default_8x8_Inter (table 7-4), in zig-zag order.
DEFAULT_8X8_INTRA: tuple[int, ...] = (
    6, 10, 10, 13, 11, 13, 16, 16, 16, 16, 18, 18, 18, 18, 18, 23,
    23, 23, 23, 23, 23, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27,
    27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31, 31, 31,
    31, 33, 33, 33, 33, 33, 36, 36, 36, 36, 38, 38, 38, 40, 40, 42,
)
DEFAULT_8X8_INTER: tuple[int, ...] = (
    9, 13, 13, 15, 13, 15, 17, 17, 17, 17, 19, 19, 19, 19, 19, 21,
    21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 24, 24, 24, 24,
    24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27, 27,
    27, 28, 28, 28, 28, 28, 30, 30, 30, 30, 32, 32, 32, 33, 33, 35,
)

#: A scaling list as the bitstream gives it: the values, ``'default'`` when the
#: list asks for its default, or None when it is absent and a fall-back rule
#: supplies it.
ScalingList = tuple[int, ...] | str | None


# ---------------------------------------------------------------- NAL units


@dataclasses.dataclass(frozen=True)
class NalUnit:
    """One NAL unit, its header split out.

    data -- the unit as it is in the stream, emulation prevention included,
        which is what a driver is given as slice data
    rbsp -- the payload after the header byte, with emulation prevention
        removed, which is what the syntax is read from
    """

    nal_ref_idc: int
    nal_unit_type: int
    data: bytes
    rbsp: bytes

    @property
    def idr(self) -> bool:
        """Whether this is a slice of an IDR picture."""
        return self.nal_unit_type == NAL_IDR

    @property
    def is_slice(self) -> bool:
        """Whether this is a slice of a primary coded picture, IDR or not."""
        return self.nal_unit_type in (NAL_SLICE, NAL_IDR)


def parse_nal(unit: bytes) -> NalUnit:
    """Split one NAL unit, given without its start code or length prefix."""
    if not unit:
        raise StreamError('an empty NAL unit')
    header = unit[0]
    if header & 0x80:
        raise StreamError(
            f'a NAL unit header {header:#04x} has forbidden_zero_bit set')
    return NalUnit(nal_ref_idc=(header >> 5) & 3, nal_unit_type=header & 0x1F,
                   data=bytes(unit), rbsp=rbsp_of(bytes(unit[1:])))


# ----------------------------------------------------------- parameter sets


@dataclasses.dataclass(frozen=True)
class VideoUsability:
    """The video usability information of annex E, as far as playback needs it.

    The colour fields keep the specification's code points: 2 means
    unspecified for primaries, transfer and matrix.
    """

    aspect_ratio_idc: int = 0
    sar: tuple[int, int] | None = None
    video_format: int = 5
    video_full_range_flag: int = 0
    colour_primaries: int = 2
    transfer_characteristics: int = 2
    matrix_coefficients: int = 2
    chroma_sample_loc_type_top_field: int = 0
    chroma_sample_loc_type_bottom_field: int = 0
    num_units_in_tick: int = 0
    time_scale: int = 0
    fixed_frame_rate_flag: int = 0
    bitstream_restriction_flag: int = 0
    max_num_reorder_frames: int | None = None
    max_dec_frame_buffering: int | None = None


@dataclasses.dataclass(frozen=True)
class SequenceParameterSet:
    """A sequence parameter set (7.3.2.1.1), with the values derived from it."""

    profile_idc: int
    constraint_flags: int
    level_idc: int
    seq_parameter_set_id: int
    chroma_format_idc: int
    separate_colour_plane_flag: int
    bit_depth_luma_minus8: int
    bit_depth_chroma_minus8: int
    qpprime_y_zero_transform_bypass_flag: int
    seq_scaling_matrix_present_flag: int
    scaling_lists: tuple[ScalingList, ...]
    log2_max_frame_num_minus4: int
    pic_order_cnt_type: int
    log2_max_pic_order_cnt_lsb_minus4: int
    delta_pic_order_always_zero_flag: int
    offset_for_non_ref_pic: int
    offset_for_top_to_bottom_field: int
    offset_for_ref_frame: tuple[int, ...]
    max_num_ref_frames: int
    gaps_in_frame_num_value_allowed_flag: int
    pic_width_in_mbs_minus1: int
    pic_height_in_map_units_minus1: int
    frame_mbs_only_flag: int
    mb_adaptive_frame_field_flag: int
    direct_8x8_inference_flag: int
    frame_crop: tuple[int, int, int, int]
    vui: VideoUsability | None

    # ------------------------------------------------------------ derived

    @property
    def constraint_set1_flag(self) -> bool:
        return bool(self.constraint_flags & 0x40)

    @property
    def constraint_set3_flag(self) -> bool:
        return bool(self.constraint_flags & 0x10)

    @property
    def chroma_array_type(self) -> int:
        """ChromaArrayType: 0 for monochrome or separately coded planes."""
        return 0 if self.separate_colour_plane_flag else self.chroma_format_idc

    @property
    def max_frame_num(self) -> int:
        return 1 << (self.log2_max_frame_num_minus4 + 4)

    @property
    def max_pic_order_cnt_lsb(self) -> int:
        return 1 << (self.log2_max_pic_order_cnt_lsb_minus4 + 4)

    @property
    def width_in_mbs(self) -> int:
        return self.pic_width_in_mbs_minus1 + 1

    @property
    def frame_height_in_mbs(self) -> int:
        """FrameHeightInMbs: map units are field macroblock pairs when interlaced."""
        return ((2 - self.frame_mbs_only_flag)
                * (self.pic_height_in_map_units_minus1 + 1))

    @property
    def coded_size(self) -> tuple[int, int]:
        """The size of the macroblock grid, in luma samples."""
        return self.width_in_mbs * 16, self.frame_height_in_mbs * 16

    @property
    def visible(self) -> tuple[int, int, int, int]:
        """The cropped rectangle as ``(x, y, width, height)`` in luma samples.

        The crop offsets are in units of CropUnitX and CropUnitY (7.4.2.1.1),
        which depend on the chroma subsampling and on whether the frame is
        coded as fields.
        """
        if self.chroma_array_type == 0:
            unit_x, unit_y = 1, 2 - self.frame_mbs_only_flag
        else:
            sub_width = 1 if self.chroma_format_idc == 3 else 2
            sub_height = 2 if self.chroma_format_idc == 1 else 1
            unit_x, unit_y = sub_width, sub_height * (2 - self.frame_mbs_only_flag)
        left, right, top, bottom = self.frame_crop
        width, height = self.coded_size
        return (unit_x * left, unit_y * top,
                width - unit_x * (left + right), height - unit_y * (top + bottom))

    @property
    def size(self) -> tuple[int, int]:
        """The picture's visible size, as ``(width, height)``."""
        return self.visible[2], self.visible[3]

    @property
    def frame_rate(self) -> tuple[int, int] | None:
        """Frames per second as ``(numerator, denominator)``, or None if unstated.

        The VUI's tick is a field period, so a frame is two ticks.
        """
        vui = self.vui
        if vui is None or not vui.num_units_in_tick or not vui.time_scale:
            return None
        numerator, denominator = vui.time_scale, 2 * vui.num_units_in_tick
        common = gcd(numerator, denominator)
        return numerator // common, denominator // common

    @property
    def dpb_frames(self) -> int:
        """How many frames the decoded picture buffer holds.

        max_dec_frame_buffering where the VUI states it, otherwise the level's
        limit (A.3.1, item h); never fewer than the stream's reference frames.
        """
        vui = self.vui
        if vui is not None and vui.max_dec_frame_buffering is not None:
            frames = vui.max_dec_frame_buffering
        else:
            limit = MAX_DPB_MBS.get(self._level_for_limits())
            if limit is None:
                frames = MAX_DPB_FRAMES
            else:
                frames = min(limit // (self.width_in_mbs * self.frame_height_in_mbs),
                             MAX_DPB_FRAMES)
        return max(frames, self.max_num_ref_frames, 1)

    def _level_for_limits(self) -> int:
        """level_idc, with level 1b spelled as 11 plus constraint_set3 read as 9."""
        if (self.level_idc == 11 and self.constraint_set3_flag
                and self.profile_idc in (66, 77, 88)):
            return 9
        return self.level_idc

    @property
    def reorder_frames(self) -> int:
        """How many frames may precede a frame in decoding order and follow it
        in output order, which is how long output may be held back."""
        vui = self.vui
        if vui is not None and vui.max_num_reorder_frames is not None:
            return min(vui.max_num_reorder_frames, self.dpb_frames)
        if self.profile_idc in INTRA_PROFILES and self.constraint_set3_flag:
            return 0
        return self.dpb_frames


@dataclasses.dataclass(frozen=True)
class PictureParameterSet:
    """A picture parameter set (7.3.2.2)."""

    pic_parameter_set_id: int
    seq_parameter_set_id: int
    entropy_coding_mode_flag: int
    bottom_field_pic_order_in_frame_present_flag: int
    num_slice_groups_minus1: int
    slice_group_map_type: int
    slice_group_change_rate_minus1: int
    num_ref_idx_l0_default_active_minus1: int
    num_ref_idx_l1_default_active_minus1: int
    weighted_pred_flag: int
    weighted_bipred_idc: int
    pic_init_qp_minus26: int
    pic_init_qs_minus26: int
    chroma_qp_index_offset: int
    deblocking_filter_control_present_flag: int
    constrained_intra_pred_flag: int
    redundant_pic_cnt_present_flag: int
    transform_8x8_mode_flag: int
    pic_scaling_matrix_present_flag: int
    scaling_lists: tuple[ScalingList, ...]
    second_chroma_qp_index_offset: int


def _scaling_list(reader: BitReader, size: int) -> ScalingList:
    """scaling_list() (7.3.2.1.1.1), in zig-zag order."""
    values = []
    last = next_scale = 8
    for index in range(size):
        if next_scale:
            next_scale = (last + reader.se() + 256) % 256
            if index == 0 and next_scale == 0:
                return 'default'
        value = last if next_scale == 0 else next_scale
        values.append(value)
        last = value
    return tuple(values)


def _scaling_lists(reader: BitReader, count: int) -> tuple[ScalingList, ...]:
    """`count` optional lists, six of 16 values and the rest of 64."""
    return tuple(_scaling_list(reader, 16 if index < 6 else 64)
                 if reader.flag() else None
                 for index in range(count))


def _skip_hrd(reader: BitReader) -> None:
    """hrd_parameters() (E.1.2), read past: playback needs none of it."""
    count = reader.ue() + 1
    reader.u(4)                                 # bit_rate_scale
    reader.u(4)                                 # cpb_size_scale
    for _ in range(count):
        reader.ue()                             # bit_rate_value_minus1
        reader.ue()                             # cpb_size_value_minus1
        reader.flag()                           # cbr_flag
    reader.u(20)                                # four delay and offset lengths


def _parse_vui(reader: BitReader) -> VideoUsability:
    """vui_parameters() (E.1.1)."""
    found: dict[str, Any] = {}
    if reader.flag():                           # aspect_ratio_info_present_flag
        idc = reader.u(8)
        found['aspect_ratio_idc'] = idc
        if idc == EXTENDED_SAR:
            found['sar'] = (reader.u(16), reader.u(16))
        else:
            found['sar'] = ASPECT_RATIOS.get(idc)
    if reader.flag():                           # overscan_info_present_flag
        reader.flag()                           # overscan_appropriate_flag
    if reader.flag():                           # video_signal_type_present_flag
        found['video_format'] = reader.u(3)
        found['video_full_range_flag'] = reader.flag()
        if reader.flag():                       # colour_description_present_flag
            found['colour_primaries'] = reader.u(8)
            found['transfer_characteristics'] = reader.u(8)
            found['matrix_coefficients'] = reader.u(8)
    if reader.flag():                           # chroma_loc_info_present_flag
        found['chroma_sample_loc_type_top_field'] = reader.ue()
        found['chroma_sample_loc_type_bottom_field'] = reader.ue()
    if reader.flag():                           # timing_info_present_flag
        found['num_units_in_tick'] = reader.u(32)
        found['time_scale'] = reader.u(32)
        found['fixed_frame_rate_flag'] = reader.flag()
    nal_hrd = reader.flag()
    if nal_hrd:
        _skip_hrd(reader)
    vcl_hrd = reader.flag()
    if vcl_hrd:
        _skip_hrd(reader)
    if nal_hrd or vcl_hrd:
        reader.flag()                           # low_delay_hrd_flag
    reader.flag()                               # pic_struct_present_flag
    if reader.flag():                           # bitstream_restriction_flag
        found['bitstream_restriction_flag'] = 1
        reader.flag()                           # motion_vectors_over_pic_boundaries
        reader.ue()                             # max_bytes_per_pic_denom
        reader.ue()                             # max_bits_per_mb_denom
        reader.ue()                             # log2_max_mv_length_horizontal
        reader.ue()                             # log2_max_mv_length_vertical
        found['max_num_reorder_frames'] = reader.ue()
        found['max_dec_frame_buffering'] = reader.ue()
    return VideoUsability(**found)


def parse_sps(rbsp: bytes) -> SequenceParameterSet:
    """A sequence parameter set from its RBSP (the payload after the NAL header)."""
    reader = BitReader(rbsp)
    profile_idc = reader.u(8)
    constraint_flags = reader.u(8)
    level_idc = reader.u(8)
    seq_parameter_set_id = reader.ue()
    chroma_format_idc, separate_colour_plane_flag = 1, 0
    bit_depth_luma_minus8 = bit_depth_chroma_minus8 = 0
    qpprime_bypass = scaling_present = 0
    scaling_lists: tuple[ScalingList, ...] = ()
    if profile_idc in HIGH_PROFILES:
        chroma_format_idc = reader.ue()
        if chroma_format_idc == 3:
            separate_colour_plane_flag = reader.flag()
        bit_depth_luma_minus8 = reader.ue()
        bit_depth_chroma_minus8 = reader.ue()
        qpprime_bypass = reader.flag()
        scaling_present = reader.flag()
        if scaling_present:
            scaling_lists = _scaling_lists(reader, 8 if chroma_format_idc != 3 else 12)
    log2_max_frame_num_minus4 = reader.ue()
    if log2_max_frame_num_minus4 > MAX_LOG2_FRAME_NUM_MINUS4:
        raise StreamError(
            f'log2_max_frame_num_minus4 {log2_max_frame_num_minus4} is past '
            f'{MAX_LOG2_FRAME_NUM_MINUS4}')
    pic_order_cnt_type = reader.ue()
    log2_max_poc_lsb_minus4 = delta_always_zero = 0
    offset_for_non_ref_pic = offset_for_top_to_bottom_field = 0
    offset_for_ref_frame: tuple[int, ...] = ()
    if pic_order_cnt_type == 0:
        log2_max_poc_lsb_minus4 = reader.ue()
    elif pic_order_cnt_type == 1:
        delta_always_zero = reader.flag()
        offset_for_non_ref_pic = reader.se()
        offset_for_top_to_bottom_field = reader.se()
        offset_for_ref_frame = tuple(reader.se() for _ in range(reader.ue()))
    elif pic_order_cnt_type != 2:
        raise StreamError(f'pic_order_cnt_type {pic_order_cnt_type} is not 0, 1 or 2')
    max_num_ref_frames = reader.ue()
    gaps = reader.flag()
    width_minus1 = reader.ue()
    height_minus1 = reader.ue()
    frame_mbs_only = reader.flag()
    mbaff = 0 if frame_mbs_only else reader.flag()
    direct_8x8 = reader.flag()
    crop = (reader.ue(), reader.ue(), reader.ue(), reader.ue()) if reader.flag() else (0, 0, 0, 0)
    vui = _parse_vui(reader) if reader.flag() else None
    return SequenceParameterSet(
        profile_idc=profile_idc, constraint_flags=constraint_flags,
        level_idc=level_idc, seq_parameter_set_id=seq_parameter_set_id,
        chroma_format_idc=chroma_format_idc,
        separate_colour_plane_flag=separate_colour_plane_flag,
        bit_depth_luma_minus8=bit_depth_luma_minus8,
        bit_depth_chroma_minus8=bit_depth_chroma_minus8,
        qpprime_y_zero_transform_bypass_flag=qpprime_bypass,
        seq_scaling_matrix_present_flag=scaling_present,
        scaling_lists=scaling_lists,
        log2_max_frame_num_minus4=log2_max_frame_num_minus4,
        pic_order_cnt_type=pic_order_cnt_type,
        log2_max_pic_order_cnt_lsb_minus4=log2_max_poc_lsb_minus4,
        delta_pic_order_always_zero_flag=delta_always_zero,
        offset_for_non_ref_pic=offset_for_non_ref_pic,
        offset_for_top_to_bottom_field=offset_for_top_to_bottom_field,
        offset_for_ref_frame=offset_for_ref_frame,
        max_num_ref_frames=max_num_ref_frames,
        gaps_in_frame_num_value_allowed_flag=gaps,
        pic_width_in_mbs_minus1=width_minus1,
        pic_height_in_map_units_minus1=height_minus1,
        frame_mbs_only_flag=frame_mbs_only,
        mb_adaptive_frame_field_flag=mbaff,
        direct_8x8_inference_flag=direct_8x8,
        frame_crop=crop, vui=vui,
    )


def colour_of(sps: SequenceParameterSet) -> Colour:
    """A sequence's colour description, the matrix chosen where it states none."""
    vui = sps.vui
    if vui is None:
        return Colour().resolved(sps.size[1])
    return Colour(matrix=vui.matrix_coefficients, full_range=bool(vui.video_full_range_flag),
                  primaries=vui.colour_primaries, transfer=vui.transfer_characteristics,
                  chroma_location=vui.chroma_sample_loc_type_top_field,
                  ).resolved(sps.size[1])


def _ceil_log2(value: int) -> int:
    """Ceil(Log2(value)) for a positive integer."""
    return (value - 1).bit_length()


def parse_pps(rbsp: bytes,
              sequences: Mapping[int, SequenceParameterSet]) -> PictureParameterSet:
    """A picture parameter set from its RBSP.

    `sequences` are the sequence parameter sets seen so far, by id: the
    number of scaling lists the set may carry depends on its sequence's chroma
    format.
    """
    reader = BitReader(rbsp)
    pic_parameter_set_id = reader.ue()
    seq_parameter_set_id = reader.ue()
    sps = sequences.get(seq_parameter_set_id)
    if sps is None:
        raise StreamError(
            f'picture parameter set {pic_parameter_set_id} names sequence '
            f'parameter set {seq_parameter_set_id}, which the stream has not '
            'defined')
    entropy = reader.flag()
    bottom_field_pic_order = reader.flag()
    groups_minus1 = reader.ue()
    map_type = change_rate_minus1 = 0
    if groups_minus1 > 0:
        map_type = reader.ue()
        if map_type == 0:
            for _ in range(groups_minus1 + 1):
                reader.ue()                     # run_length_minus1
        elif map_type == 2:
            for _ in range(groups_minus1):
                reader.ue()                     # top_left
                reader.ue()                     # bottom_right
        elif map_type in (3, 4, 5):
            reader.flag()                       # slice_group_change_direction_flag
            change_rate_minus1 = reader.ue()
        elif map_type == 6:
            units = reader.ue() + 1
            width = _ceil_log2(groups_minus1 + 1)
            for _ in range(units):
                reader.u(width)                 # slice_group_id
    l0_default = _reference_count(reader.ue())
    l1_default = _reference_count(reader.ue())
    weighted_pred = reader.flag()
    weighted_bipred = reader.u(2)
    qp_minus26 = reader.se()
    qs_minus26 = reader.se()
    chroma_offset = reader.se()
    deblocking = reader.flag()
    constrained_intra = reader.flag()
    redundant = reader.flag()
    transform_8x8 = scaling_present = 0
    scaling_lists: tuple[ScalingList, ...] = ()
    second_chroma_offset = chroma_offset
    if reader.more_rbsp_data():
        transform_8x8 = reader.flag()
        scaling_present = reader.flag()
        if scaling_present:
            eights = (2 if sps.chroma_format_idc != 3 else 6) * transform_8x8
            scaling_lists = _scaling_lists(reader, 6 + eights)
        second_chroma_offset = reader.se()
    return PictureParameterSet(
        pic_parameter_set_id=pic_parameter_set_id,
        seq_parameter_set_id=seq_parameter_set_id,
        entropy_coding_mode_flag=entropy,
        bottom_field_pic_order_in_frame_present_flag=bottom_field_pic_order,
        num_slice_groups_minus1=groups_minus1, slice_group_map_type=map_type,
        slice_group_change_rate_minus1=change_rate_minus1,
        num_ref_idx_l0_default_active_minus1=l0_default,
        num_ref_idx_l1_default_active_minus1=l1_default,
        weighted_pred_flag=weighted_pred, weighted_bipred_idc=weighted_bipred,
        pic_init_qp_minus26=qp_minus26, pic_init_qs_minus26=qs_minus26,
        chroma_qp_index_offset=chroma_offset,
        deblocking_filter_control_present_flag=deblocking,
        constrained_intra_pred_flag=constrained_intra,
        redundant_pic_cnt_present_flag=redundant,
        transform_8x8_mode_flag=transform_8x8,
        pic_scaling_matrix_present_flag=scaling_present,
        scaling_lists=scaling_lists,
        second_chroma_qp_index_offset=second_chroma_offset,
    )


def _resolve(declared: tuple[ScalingList, ...], fallback: list[tuple[int, ...]],
             rule_a: bool) -> list[tuple[int, ...]]:
    """Apply fall-back rule A or B (table 7-2) to the lists a set declared.

    `fallback` is what lists 0, 3, 6 and 7 fall back to when absent: the
    defaults under rule A, the sequence's own lists under rule B. Every other
    absent list copies the list before it of the same kind.
    """
    defaults: list[tuple[int, ...]] = [DEFAULT_4X4_INTRA] * 3 + [DEFAULT_4X4_INTER] * 3 + [
        DEFAULT_8X8_INTRA, DEFAULT_8X8_INTER] * 3
    resolved: list[tuple[int, ...]] = []
    for index, entry in enumerate(declared):
        if entry == 'default':
            resolved.append(defaults[index])
        elif isinstance(entry, tuple):
            resolved.append(entry)
        elif index in (0, 3, 6, 7):
            resolved.append(defaults[index] if rule_a else fallback[index])
        else:
            # The 4x4 lists copy the one before; the 8x8 lists of 4:4:4 copy
            # the one two before, which is the previous list of their kind.
            resolved.append(resolved[index - (1 if index < 6 else 2)])
    return resolved


def scaling_matrices(sps: SequenceParameterSet, pps: PictureParameterSet | None,
                     ) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """The 4x4 and 8x8 scaling lists in force, in zig-zag order.

    Six 4x4 lists (intra Y, Cb, Cr, then inter Y, Cb, Cr), and two 8x8 lists
    (intra and inter luma), or six for 4:4:4.
    """
    eights = 6 if sps.chroma_format_idc == 3 else 2
    count = 6 + eights
    flat: list[tuple[int, ...]] = [FLAT_4X4] * 6 + [FLAT_8X8] * eights
    if sps.seq_scaling_matrix_present_flag:
        sequence = _resolve(sps.scaling_lists, flat, rule_a=True)
    else:
        sequence = flat
    if pps is None or not pps.pic_scaling_matrix_present_flag:
        chosen = sequence
    else:
        # 7.4.2.2: a picture set's absent lists fall back by rule A when the
        # sequence carries no matrix, and by rule B to the sequence's when it does.
        declared = pps.scaling_lists + (None,) * (count - len(pps.scaling_lists))
        chosen = _resolve(declared, sequence,
                          rule_a=not sps.seq_scaling_matrix_present_flag)
    return chosen[:6], chosen[6:count]


# ------------------------------------------------------------- slice headers


@dataclasses.dataclass(frozen=True)
class MemoryManagementOperation:
    """One memory_management_control_operation and its arguments (7.3.3.3)."""

    operation: int
    difference_of_pic_nums_minus1: int = 0
    long_term_pic_num: int = 0
    long_term_frame_idx: int = 0
    max_long_term_frame_idx_plus1: int = 0


@dataclasses.dataclass(frozen=True)
class PredWeightTable:
    """pred_weight_table() (7.3.3.2), with the inferred defaults filled in.

    luma_l0 -- ``(weight, offset)`` for each active entry of list 0
    chroma_l0 -- ``((Cb weight, Cb offset), (Cr weight, Cr offset))`` each
    *_flag -- whether any entry of that list carried explicit weights
    """

    luma_log2_weight_denom: int
    chroma_log2_weight_denom: int
    luma_l0: tuple[tuple[int, int], ...]
    chroma_l0: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    luma_l1: tuple[tuple[int, int], ...]
    chroma_l1: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    luma_weight_l0_flag: bool
    chroma_weight_l0_flag: bool
    luma_weight_l1_flag: bool
    chroma_weight_l1_flag: bool


@dataclasses.dataclass(frozen=True)
class SliceHeader:
    """A slice header (7.3.3), with the parameter sets it refers to.

    header_bits -- where slice_data() begins, in bits from the start of the NAL
        unit, its header byte included, counted with emulation prevention
        removed; this is libva's ``slice_data_bit_offset``
    ref_pic_list_modification_l0/l1 -- ``(modification_of_pic_nums_idc,
        value)`` pairs, the value being abs_diff_pic_num_minus1 or
        long_term_pic_num as the operation takes
    """

    nal_ref_idc: int
    nal_unit_type: int
    first_mb_in_slice: int
    slice_type: int
    pic_parameter_set_id: int
    frame_num: int
    field_pic_flag: int
    bottom_field_flag: int
    idr_pic_id: int
    pic_order_cnt_lsb: int
    delta_pic_order_cnt_bottom: int
    delta_pic_order_cnt: tuple[int, int]
    redundant_pic_cnt: int
    direct_spatial_mv_pred_flag: int
    num_ref_idx_active_override_flag: int
    num_ref_idx_l0_active_minus1: int
    num_ref_idx_l1_active_minus1: int
    ref_pic_list_modification_l0: tuple[tuple[int, int], ...]
    ref_pic_list_modification_l1: tuple[tuple[int, int], ...]
    pred_weight_table: PredWeightTable | None
    no_output_of_prior_pics_flag: int
    long_term_reference_flag: int
    adaptive_ref_pic_marking_mode_flag: int
    memory_management: tuple[MemoryManagementOperation, ...]
    cabac_init_idc: int
    slice_qp_delta: int
    disable_deblocking_filter_idc: int
    slice_alpha_c0_offset_div2: int
    slice_beta_offset_div2: int
    header_bits: int
    sps: SequenceParameterSet = dataclasses.field(repr=False, compare=False)
    pps: PictureParameterSet = dataclasses.field(repr=False, compare=False)

    @property
    def idr(self) -> bool:
        return self.nal_unit_type == NAL_IDR

    @property
    def kind(self) -> str:
        """The slice type's name: ``'P'``, ``'B'``, ``'I'``, ``'SP'`` or ``'SI'``."""
        return SLICE_KINDS[self.slice_type]

    @property
    def is_reference(self) -> bool:
        return self.nal_ref_idc != 0


def _modifications(reader: BitReader) -> tuple[tuple[int, int], ...]:
    """One list's ref_pic_list_modification operations (7.3.3.1)."""
    if not reader.flag():
        return ()
    found: list[tuple[int, int]] = []
    while True:
        idc = reader.ue()
        if idc == 3:
            return tuple(found)
        if idc > 5:
            raise StreamError(f'modification_of_pic_nums_idc {idc} is not defined')
        found.append((idc, reader.ue()))


def _weights(reader: BitReader, count: int, denom: int, chroma_denom: int,
             chroma: bool) -> tuple[tuple[tuple[int, int], ...],
                                    tuple[tuple[tuple[int, int], tuple[int, int]], ...],
                                    bool, bool]:
    """One list of pred_weight_table(), defaults inferred (7.4.3.2)."""
    luma: list[tuple[int, int]] = []
    chromas: list[tuple[tuple[int, int], tuple[int, int]]] = []
    any_luma = any_chroma = False
    default_chroma = ((1 << chroma_denom, 0), (1 << chroma_denom, 0))
    for _ in range(count):
        if reader.flag():
            any_luma = True
            luma.append((reader.se(), reader.se()))
        else:
            luma.append((1 << denom, 0))
        if chroma and reader.flag():
            any_chroma = True
            cb = (reader.se(), reader.se())
            cr = (reader.se(), reader.se())
            chromas.append((cb, cr))
        else:
            chromas.append(default_chroma)
    return tuple(luma), tuple(chromas), any_luma, any_chroma


#: What H.264 allows (7.4.2.1.1, 7.4.2.2 and 7.4.3), so a stream that exceeds
#: them is refused before a loop or a shift is sized from the claim.
MAX_LOG2_FRAME_NUM_MINUS4 = 12
MAX_REFERENCE_INDEX = 31
MAX_WEIGHT_DENOMINATOR = 7


def _reference_count(minus1: int) -> int:
    """A ``num_ref_idx_*_active_minus1`` value, refused where it is past the limit."""
    if minus1 > MAX_REFERENCE_INDEX:
        raise StreamError(
            f'num_ref_idx_active_minus1 {minus1} is past {MAX_REFERENCE_INDEX}')
    return minus1


def _pred_weight_table(reader: BitReader, sps: SequenceParameterSet,
                       l0: int, l1: int | None) -> PredWeightTable:
    denom = reader.ue()
    chroma = sps.chroma_array_type != 0
    chroma_denom = reader.ue() if chroma else 0
    if max(denom, chroma_denom) > MAX_WEIGHT_DENOMINATOR:
        raise StreamError(
            f'a prediction weight denominator of {max(denom, chroma_denom)} is past '
            f'{MAX_WEIGHT_DENOMINATOR}')
    luma_l0, chroma_l0, luma_flag_l0, chroma_flag_l0 = _weights(
        reader, l0, denom, chroma_denom, chroma)
    if l1 is None:
        luma_l1: tuple[tuple[int, int], ...] = ()
        chroma_l1: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = ()
        luma_flag_l1 = chroma_flag_l1 = False
    else:
        luma_l1, chroma_l1, luma_flag_l1, chroma_flag_l1 = _weights(
            reader, l1, denom, chroma_denom, chroma)
    return PredWeightTable(
        luma_log2_weight_denom=denom, chroma_log2_weight_denom=chroma_denom,
        luma_l0=luma_l0, chroma_l0=chroma_l0, luma_l1=luma_l1,
        chroma_l1=chroma_l1, luma_weight_l0_flag=luma_flag_l0,
        chroma_weight_l0_flag=chroma_flag_l0, luma_weight_l1_flag=luma_flag_l1,
        chroma_weight_l1_flag=chroma_flag_l1)


def _marking(reader: BitReader) -> tuple[MemoryManagementOperation, ...]:
    """The adaptive part of dec_ref_pic_marking() (7.3.3.3)."""
    found: list[MemoryManagementOperation] = []
    while True:
        operation = reader.ue()
        if operation == 0:
            return tuple(found)
        if operation > 6:
            raise StreamError(
                f'memory_management_control_operation {operation} is not defined')
        arguments: dict[str, int] = {}
        if operation in (1, 3):
            arguments['difference_of_pic_nums_minus1'] = reader.ue()
        if operation == 2:
            arguments['long_term_pic_num'] = reader.ue()
        if operation in (3, 6):
            arguments['long_term_frame_idx'] = reader.ue()
        if operation == 4:
            arguments['max_long_term_frame_idx_plus1'] = reader.ue()
        found.append(MemoryManagementOperation(operation, **arguments))


def parse_slice_header(unit: NalUnit,
                       sequences: Mapping[int, SequenceParameterSet],
                       pictures: Mapping[int, PictureParameterSet]) -> SliceHeader:
    """The header of a slice NAL unit, read against the parameter sets so far."""
    reader = BitReader(unit.rbsp)
    first_mb = reader.ue()
    raw_type = reader.ue()
    if raw_type > 9:
        raise StreamError(f'slice_type {raw_type} is not defined')
    slice_type = raw_type % 5
    pps_id = reader.ue()
    pps = pictures.get(pps_id)
    if pps is None:
        raise StreamError(
            f'a slice names picture parameter set {pps_id}, which the stream '
            'has not defined')
    sps = sequences.get(pps.seq_parameter_set_id)
    if sps is None:
        raise StreamError(
            f'picture parameter set {pps_id} names sequence parameter set '
            f'{pps.seq_parameter_set_id}, which the stream has not defined')
    if sps.separate_colour_plane_flag:
        reader.u(2)                             # colour_plane_id
    frame_num = reader.u(sps.log2_max_frame_num_minus4 + 4)
    field_pic = bottom_field = 0
    if not sps.frame_mbs_only_flag:
        field_pic = reader.flag()
        if field_pic:
            bottom_field = reader.flag()
    idr = unit.nal_unit_type == NAL_IDR
    idr_pic_id = reader.ue() if idr else 0
    poc_lsb = delta_bottom = 0
    delta = [0, 0]
    in_frame = pps.bottom_field_pic_order_in_frame_present_flag and not field_pic
    if sps.pic_order_cnt_type == 0:
        poc_lsb = reader.u(sps.log2_max_pic_order_cnt_lsb_minus4 + 4)
        if in_frame:
            delta_bottom = reader.se()
    if sps.pic_order_cnt_type == 1 and not sps.delta_pic_order_always_zero_flag:
        delta[0] = reader.se()
        if in_frame:
            delta[1] = reader.se()
    redundant = reader.ue() if pps.redundant_pic_cnt_present_flag else 0
    direct_spatial = reader.flag() if slice_type == SLICE_B else 0
    l0 = pps.num_ref_idx_l0_default_active_minus1
    l1 = pps.num_ref_idx_l1_default_active_minus1
    override = 0
    if slice_type in (SLICE_P, SLICE_SP, SLICE_B):
        override = reader.flag()
        if override:
            l0 = _reference_count(reader.ue())
            if slice_type == SLICE_B:
                l1 = _reference_count(reader.ue())
    modifications_l0: tuple[tuple[int, int], ...] = ()
    modifications_l1: tuple[tuple[int, int], ...] = ()
    if slice_type not in (SLICE_I, SLICE_SI):
        modifications_l0 = _modifications(reader)
        if slice_type == SLICE_B:
            modifications_l1 = _modifications(reader)
    weights = None
    if ((pps.weighted_pred_flag and slice_type in (SLICE_P, SLICE_SP))
            or (pps.weighted_bipred_idc == 1 and slice_type == SLICE_B)):
        weights = _pred_weight_table(reader, sps, l0 + 1,
                                     l1 + 1 if slice_type == SLICE_B else None)
    no_output = long_term = adaptive = 0
    operations: tuple[MemoryManagementOperation, ...] = ()
    if unit.nal_ref_idc:
        if idr:
            no_output = reader.flag()
            long_term = reader.flag()
        else:
            adaptive = reader.flag()
            if adaptive:
                operations = _marking(reader)
    cabac_init = 0
    if pps.entropy_coding_mode_flag and slice_type not in (SLICE_I, SLICE_SI):
        cabac_init = reader.ue()
    qp_delta = reader.se()
    if slice_type in (SLICE_SP, SLICE_SI):
        if slice_type == SLICE_SP:
            reader.flag()                       # sp_for_switch_flag
        reader.se()                             # slice_qs_delta
    deblocking_idc = alpha = beta = 0
    if pps.deblocking_filter_control_present_flag:
        deblocking_idc = reader.ue()
        if deblocking_idc != 1:
            alpha = reader.se()
            beta = reader.se()
    if pps.num_slice_groups_minus1 > 0 and 3 <= pps.slice_group_map_type <= 5:
        units = sps.width_in_mbs * (sps.pic_height_in_map_units_minus1 + 1)
        rate = pps.slice_group_change_rate_minus1 + 1
        bits = 0
        while rate << bits < units + rate:      # Ceil(Log2(units / rate + 1))
            bits += 1
        reader.u(bits)                          # slice_group_change_cycle
    return SliceHeader(
        nal_ref_idc=unit.nal_ref_idc, nal_unit_type=unit.nal_unit_type,
        first_mb_in_slice=first_mb, slice_type=slice_type,
        pic_parameter_set_id=pps_id, frame_num=frame_num,
        field_pic_flag=field_pic, bottom_field_flag=bottom_field,
        idr_pic_id=idr_pic_id, pic_order_cnt_lsb=poc_lsb,
        delta_pic_order_cnt_bottom=delta_bottom,
        delta_pic_order_cnt=(delta[0], delta[1]), redundant_pic_cnt=redundant,
        direct_spatial_mv_pred_flag=direct_spatial,
        num_ref_idx_active_override_flag=override,
        num_ref_idx_l0_active_minus1=l0, num_ref_idx_l1_active_minus1=l1,
        ref_pic_list_modification_l0=modifications_l0,
        ref_pic_list_modification_l1=modifications_l1,
        pred_weight_table=weights, no_output_of_prior_pics_flag=no_output,
        long_term_reference_flag=long_term,
        adaptive_ref_pic_marking_mode_flag=adaptive,
        memory_management=operations, cabac_init_idc=cabac_init,
        slice_qp_delta=qp_delta, disable_deblocking_filter_idc=deblocking_idc,
        slice_alpha_c0_offset_div2=alpha, slice_beta_offset_div2=beta,
        header_bits=reader.position + 8, sps=sps, pps=pps,
    )


def starts_new_picture(previous: SliceHeader, current: SliceHeader) -> bool:
    """Whether `current` is the first slice of a picture after `previous`'s.

    The comparisons of 7.4.1.2.4, which detect the first VCL NAL unit of a
    new primary coded picture.
    """
    if (current.frame_num != previous.frame_num
            or current.pic_parameter_set_id != previous.pic_parameter_set_id
            or current.field_pic_flag != previous.field_pic_flag
            or current.bottom_field_flag != previous.bottom_field_flag
            or (current.nal_ref_idc == 0) != (previous.nal_ref_idc == 0)
            or current.idr != previous.idr
            or (current.idr and current.idr_pic_id != previous.idr_pic_id)):
        return True
    order_type = current.sps.pic_order_cnt_type
    if order_type == 0:
        return (current.pic_order_cnt_lsb != previous.pic_order_cnt_lsb
                or current.delta_pic_order_cnt_bottom
                != previous.delta_pic_order_cnt_bottom)
    if order_type == 1:
        return current.delta_pic_order_cnt != previous.delta_pic_order_cnt
    return False
