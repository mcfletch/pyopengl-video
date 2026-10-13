"""The HEVC parser: NAL units, parameter sets and slice segment headers.

A libva decoder is handed the parsed syntax: the sequence and picture
parameter sets become a picture parameter buffer and a scaling list buffer,
and each slice segment header becomes a slice parameter buffer. This module
reads that syntax and holds no libva.

Field names are the syntax elements of ITU-T H.265 (clause 7.3 and annex E),
so the parser can be read beside the specification's syntax tables. Derived
values the decoder needs, such as the visible rectangle, the frame rate and
the size of the decoded picture buffer, are properties of the parsed sets.

The version 1 syntax is read in full, and so are the range extensions of the
sequence and picture parameter sets, which :mod:`pyopengl_video.vaapi.hevc_decode`
refuses by name. A picture parameter set with a multilayer, 3D or screen
content extension changes the slice segment header in ways not read here, and
its slices raise :class:`StreamError`.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from math import gcd
from typing import Any

from pyopengl_video.colour import Colour
from pyopengl_video.vaapi.h264 import BitReader, StreamError, rbsp_of
from pyopengl_video.vaapi.h264_parse import ASPECT_RATIOS, EXTENDED_SAR

__all__ = [
    'DEFAULT_SCALING_LISTS', 'FLAT_SCALING_LISTS', 'LongTermPicture', 'NalUnit',
    'PictureParameterSet', 'PredWeightTable', 'ProfileTierLevel', 'ScalingLists',
    'SequenceParameterSet', 'ShortTermRefPicSet', 'SliceHeader', 'StreamError',
    'VideoParameterSet', 'VideoUsability', 'colour_of', 'parse_nal', 'parse_pps',
    'parse_slice_header', 'parse_sps', 'parse_vps',
]

# --------------------------------------------------------------- NAL types

#: nal_unit_type values (table 7-1).
NAL_TRAIL_N = 0
NAL_TRAIL_R = 1
NAL_TSA_N = 2
NAL_TSA_R = 3
NAL_STSA_N = 4
NAL_STSA_R = 5
NAL_RADL_N = 6
NAL_RADL_R = 7
NAL_RASL_N = 8
NAL_RASL_R = 9
NAL_BLA_W_LP = 16
NAL_BLA_W_RADL = 17
NAL_BLA_N_LP = 18
NAL_IDR_W_RADL = 19
NAL_IDR_N_LP = 20
NAL_CRA = 21
NAL_VPS = 32
NAL_SPS = 33
NAL_PPS = 34
NAL_AUD = 35
NAL_END_OF_SEQUENCE = 36
NAL_END_OF_BITSTREAM = 37
NAL_FILLER = 38
NAL_PREFIX_SEI = 39
NAL_SUFFIX_SEI = 40

#: The last of the sub-layer non-reference and reference types, which
#: alternate below it: an even type is a picture nothing in its sub-layer
#: predicts from.
NAL_LAST_SUB_LAYER_TYPE = 15
#: The intra random access point types, reserved ones included.
NAL_FIRST_IRAP = 16
NAL_LAST_IRAP = 23
#: Types below this carry slice segments.
NAL_FIRST_NON_VCL = 32

#: slice_type, as named in table 7-7.
SLICE_B, SLICE_P, SLICE_I = range(3)
SLICE_KINDS = ('B', 'P', 'I')

#: general_profile_idc values by name (annex A).
PROFILE_MAIN = 1
PROFILE_MAIN_10 = 2
PROFILE_MAIN_STILL_PICTURE = 3
PROFILE_NAMES = {
    PROFILE_MAIN: 'Main', PROFILE_MAIN_10: 'Main 10',
    PROFILE_MAIN_STILL_PICTURE: 'Main Still Picture',
    4: 'Format Range Extensions', 5: 'High Throughput', 6: 'Multiview Main',
    7: 'Scalable Main', 8: '3D Main', 9: 'Screen Content Coding Extensions',
    10: 'Scalable Format Range Extensions',
    11: 'High Throughput Screen Content Coding Extensions',
}

#: The flags of sps_range_extension() (7.3.2.2.2), in the order coded.
SPS_RANGE_EXTENSION_FLAGS = (
    'transform_skip_rotation_enabled_flag', 'transform_skip_context_enabled_flag',
    'implicit_rdpcm_enabled_flag', 'explicit_rdpcm_enabled_flag',
    'extended_precision_processing_flag', 'intra_smoothing_disabled_flag',
    'high_precision_offsets_enabled_flag', 'persistent_rice_adaptation_enabled_flag',
    'cabac_bypass_alignment_enabled_flag',
)

#: (SubWidthC, SubHeightC) by chroma_format_idc (table 6-1).
CHROMA_SUBSAMPLING = ((1, 1), (2, 2), (2, 1), (1, 1))

# ------------------------------------------------------------ scaling lists

#: Table 7-6: the default 8x8 intra and inter lists, in the up-right diagonal
#: order the stream codes a list in. The 16x16 and 32x32 defaults are the same
#: 64 factors, and every default 4x4 factor is 16 (table 7-5).
DEFAULT_8X8_INTRA: tuple[int, ...] = (
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18,
    17, 18, 18, 17, 18, 21, 19, 20, 21, 20, 19, 21, 24, 22, 22, 24,
    24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35, 35, 31,
    29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115,
)
DEFAULT_8X8_INTER: tuple[int, ...] = (
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18,
    18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24,
    24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28, 28, 28,
    28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91,
)

#: The DC factor of a 16x16 or 32x32 list that states none.
DEFAULT_DC = 16


@dataclasses.dataclass(frozen=True)
class ScalingLists:
    """Every scaling list of one scaling_list_data() (7.3.4), predictions resolved.

    lists -- by sizeId (4x4, 8x8, 16x16, 32x32), then by matrixId: six lists
        of each size (intra Y, Cb, Cr, then inter Y, Cb, Cr) and two of 32x32
        (intra and inter luma). Sixteen factors for 4x4 and 64 for the rest,
        in the up-right diagonal order they are coded in.
    dc -- the DC factor of each 16x16 list, then of each 32x32 list
    """

    lists: tuple[tuple[tuple[int, ...], ...], ...]
    dc: tuple[tuple[int, ...], tuple[int, ...]]


def _uniform(four: tuple[int, ...], intra: tuple[int, ...],
             inter: tuple[int, ...]) -> ScalingLists:
    sixes = (intra,) * 3 + (inter,) * 3
    return ScalingLists(lists=((four,) * 6, sixes, sixes, (intra, inter)),
                        dc=((DEFAULT_DC,) * 6, (DEFAULT_DC,) * 2))


#: The lists in force when scaling lists are enabled and neither parameter
#: set carries any (7.4.5).
DEFAULT_SCALING_LISTS = _uniform((16,) * 16, DEFAULT_8X8_INTRA, DEFAULT_8X8_INTER)
#: The lists in force with scaling lists disabled: every factor 16.
FLAT_SCALING_LISTS = _uniform((16,) * 16, (16,) * 64, (16,) * 64)


def _scaling_list_data(reader: BitReader) -> ScalingLists:
    """scaling_list_data() (7.3.4), each list resolved as 7.4.5 has it."""
    sizes: list[tuple[tuple[int, ...], ...]] = []
    dcs: list[tuple[int, ...]] = []
    for size_id in range(4):
        count = 2 if size_id == 3 else 6
        defaults = DEFAULT_SCALING_LISTS.lists[size_id]
        lists: list[tuple[int, ...]] = []
        dc: list[int] = []
        for matrix in range(count):
            if not reader.flag():               # scaling_list_pred_mode_flag
                delta = reader.ue()             # scaling_list_pred_matrix_id_delta
                if delta > matrix:
                    raise StreamError(
                        f'scaling list {matrix} of size {size_id} is predicted '
                        f'from list {matrix - delta}')
                if delta == 0:
                    lists.append(defaults[matrix])
                    dc.append(DEFAULT_DC)
                else:
                    lists.append(lists[matrix - delta])
                    dc.append(dc[matrix - delta])
                continue
            coefficient = 8
            if size_id > 1:
                coefficient = reader.se() + 8   # scaling_list_dc_coef_minus8
                dc.append(coefficient)
            else:
                dc.append(DEFAULT_DC)
            values = []
            for _ in range(16 if size_id == 0 else 64):
                coefficient = (coefficient + reader.se() + 256) % 256
                values.append(coefficient)
            lists.append(tuple(values))
        sizes.append(tuple(lists))
        if size_id > 1:
            dcs.append(tuple(dc))
    return ScalingLists(lists=tuple(sizes), dc=(dcs[0], dcs[1]))


# ---------------------------------------------------------------- NAL units


@dataclasses.dataclass(frozen=True)
class NalUnit:
    """One NAL unit, its two-byte header split out.

    temporal_id -- TemporalId, which is nuh_temporal_id_plus1 less one
    data -- the unit as it is in the stream, emulation prevention included,
        which is what a driver is given as slice data
    rbsp -- the payload after the header, with emulation prevention removed,
        which is what the syntax is read from
    """

    nal_unit_type: int
    nuh_layer_id: int
    temporal_id: int
    data: bytes
    rbsp: bytes

    @property
    def is_slice(self) -> bool:
        """Whether the unit carries a slice segment."""
        return self.nal_unit_type < NAL_FIRST_NON_VCL


def parse_nal(unit: bytes) -> NalUnit:
    """Split one NAL unit, given without its start code or length prefix."""
    if len(unit) < 2:
        raise StreamError(f'a NAL unit of {len(unit)} bytes has no header')
    if unit[0] & 0x80:
        raise StreamError(
            f'a NAL unit header {unit[0]:#04x} has forbidden_zero_bit set')
    temporal_id_plus1 = unit[1] & 7
    if not temporal_id_plus1:
        raise StreamError('a NAL unit header has nuh_temporal_id_plus1 of zero')
    return NalUnit(nal_unit_type=(unit[0] >> 1) & 0x3F,
                   nuh_layer_id=(unit[0] & 1) << 5 | unit[1] >> 3,
                   temporal_id=temporal_id_plus1 - 1,
                   data=bytes(unit), rbsp=rbsp_of(bytes(unit[2:])))


# ----------------------------------------------------------- parameter sets


@dataclasses.dataclass(frozen=True)
class ProfileTierLevel:
    """The general part of profile_tier_level() (7.3.3).

    general_profile_compatibility_flags -- flag j is bit ``31 - j``
    """

    general_profile_space: int
    general_tier_flag: int
    general_profile_idc: int
    general_profile_compatibility_flags: int
    general_progressive_source_flag: int
    general_interlaced_source_flag: int
    general_non_packed_constraint_flag: int
    general_frame_only_constraint_flag: int
    general_level_idc: int

    def compatible(self, profile: int) -> bool:
        """Whether the stream says it conforms to profile number `profile`."""
        return (self.general_profile_idc == profile
                or bool(self.general_profile_compatibility_flags >> (31 - profile) & 1))

    @property
    def name(self) -> str:
        """The profile in words."""
        return PROFILE_NAMES.get(self.general_profile_idc,
                                 f'general_profile_idc {self.general_profile_idc}')


def _profile_tier_level(reader: BitReader, sub_layers_minus1: int) -> ProfileTierLevel:
    space = reader.u(2)
    tier = reader.flag()
    profile = reader.u(5)
    compatibility = reader.u(32)
    progressive, interlaced, non_packed, frame_only = (reader.flag() for _ in range(4))
    reader.u(32)                                # 43 constraint and reserved bits,
    reader.u(12)                                # then general_inbld_flag
    level = reader.u(8)
    present = [(reader.flag(), reader.flag()) for _ in range(sub_layers_minus1)]
    if sub_layers_minus1:
        reader.u(2 * (8 - sub_layers_minus1))   # reserved_zero_2bits
    for profile_present, level_present in present:
        if profile_present:
            reader.u(32)                        # the sub-layer's 88 profile bits
            reader.u(32)
            reader.u(24)
        if level_present:
            reader.u(8)                         # sub_layer_level_idc
    return ProfileTierLevel(
        general_profile_space=space, general_tier_flag=tier,
        general_profile_idc=profile,
        general_profile_compatibility_flags=compatibility,
        general_progressive_source_flag=progressive,
        general_interlaced_source_flag=interlaced,
        general_non_packed_constraint_flag=non_packed,
        general_frame_only_constraint_flag=frame_only, general_level_idc=level)


def _ordering_info(reader: BitReader, sub_layers_minus1: int,
                   ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """The three per-sub-layer buffering values, one entry for every sub-layer.

    Where the set states only the highest sub-layer's, the lower ones take
    the same values (7.4.3.1).
    """
    stated = range(sub_layers_minus1 + 1) if reader.flag() else (sub_layers_minus1,)
    entries = [(reader.ue(), reader.ue(), reader.ue()) for _ in stated]
    entries = entries * (sub_layers_minus1 + 1) if len(entries) == 1 else entries
    buffering, reorder, latency = zip(*entries, strict=True)
    return tuple(buffering), tuple(reorder), tuple(latency)


def _frame_rate(units_in_tick: int, time_scale: int) -> tuple[int, int] | None:
    if not units_in_tick or not time_scale:
        return None
    common = gcd(time_scale, units_in_tick)
    return time_scale // common, units_in_tick // common


@dataclasses.dataclass(frozen=True)
class VideoParameterSet:
    """A video parameter set (7.3.2.1), as far as its timing information."""

    vps_video_parameter_set_id: int
    vps_max_layers_minus1: int
    vps_max_sub_layers_minus1: int
    vps_temporal_id_nesting_flag: int
    profile_tier_level: ProfileTierLevel
    vps_max_dec_pic_buffering_minus1: tuple[int, ...]
    vps_max_num_reorder_pics: tuple[int, ...]
    vps_max_latency_increase_plus1: tuple[int, ...]
    vps_timing_info_present_flag: int
    vps_num_units_in_tick: int
    vps_time_scale: int

    @property
    def frame_rate(self) -> tuple[int, int] | None:
        """Pictures per second as ``(numerator, denominator)``, or None if unstated."""
        return _frame_rate(self.vps_num_units_in_tick, self.vps_time_scale)


def parse_vps(rbsp: bytes) -> VideoParameterSet:
    """A video parameter set from its RBSP (the payload after the NAL header)."""
    reader = BitReader(rbsp)
    identifier = reader.u(4)
    reader.u(2)                                 # base layer internal, available
    layers_minus1 = reader.u(6)
    sub_layers_minus1 = reader.u(3)
    nesting = reader.flag()
    reader.u(16)                                # vps_reserved_0xffff_16bits
    profile = _profile_tier_level(reader, sub_layers_minus1)
    buffering, reorder, latency = _ordering_info(reader, sub_layers_minus1)
    max_layer_id = reader.u(6)
    for _ in range(reader.ue()):                # vps_num_layer_sets_minus1
        reader.u(max_layer_id + 1)              # layer_id_included_flag
    timing = reader.flag()
    units_in_tick = time_scale = 0
    if timing:
        units_in_tick = reader.u(32)
        time_scale = reader.u(32)
    return VideoParameterSet(
        vps_video_parameter_set_id=identifier, vps_max_layers_minus1=layers_minus1,
        vps_max_sub_layers_minus1=sub_layers_minus1,
        vps_temporal_id_nesting_flag=nesting, profile_tier_level=profile,
        vps_max_dec_pic_buffering_minus1=buffering, vps_max_num_reorder_pics=reorder,
        vps_max_latency_increase_plus1=latency, vps_timing_info_present_flag=timing,
        vps_num_units_in_tick=units_in_tick, vps_time_scale=time_scale)


@dataclasses.dataclass(frozen=True)
class ShortTermRefPicSet:
    """One st_ref_pic_set() (7.3.7), with inter prediction resolved (7.4.8).

    delta_poc_s0 -- DeltaPocS0: the pictures before the current one in output
        order, each as its picture order count less the current picture's,
        nearest first
    used_by_curr_pic_s0 -- whether the current picture may predict from each
    delta_poc_s1, used_by_curr_pic_s1 -- the same for the pictures after
    """

    delta_poc_s0: tuple[int, ...] = ()
    used_by_curr_pic_s0: tuple[int, ...] = ()
    delta_poc_s1: tuple[int, ...] = ()
    used_by_curr_pic_s1: tuple[int, ...] = ()

    @property
    def num_negative_pics(self) -> int:
        return len(self.delta_poc_s0)

    @property
    def num_positive_pics(self) -> int:
        return len(self.delta_poc_s1)

    @property
    def num_delta_pocs(self) -> int:
        return len(self.delta_poc_s0) + len(self.delta_poc_s1)

    @property
    def used(self) -> int:
        """How many of the set's pictures the current picture may predict from."""
        return sum(self.used_by_curr_pic_s0) + sum(self.used_by_curr_pic_s1)


def _st_ref_pic_set(reader: BitReader, index: int,
                    earlier: list[ShortTermRefPicSet], in_slice: bool,
                    ) -> ShortTermRefPicSet:
    """st_ref_pic_set(index), read after the sets `earlier` in the sequence.

    in_slice -- the set is coded in a slice segment header, where it may be
        predicted from any of the sequence's sets
    """
    if index and reader.flag():                 # inter_ref_pic_set_prediction_flag
        delta_idx = reader.ue() + 1 if in_slice else 1
        if delta_idx > index:
            raise StreamError(
                f'reference picture set {index} is predicted from set '
                f'{index - delta_idx}')
        sign = reader.flag()                    # delta_rps_sign
        delta_rps = (1 - 2 * sign) * (reader.ue() + 1)
        source = earlier[index - delta_idx]
        used, use_delta = [], []
        for _ in range(source.num_delta_pocs + 1):
            used.append(reader.flag())          # used_by_curr_pic_flag
            use_delta.append(1 if used[-1] else reader.flag())
        negative, positive = source.num_negative_pics, source.num_positive_pics
        # Equations 7-61 and 7-62. The candidates for each half are the
        # source's other half taken farthest first, then the source picture
        # itself, then the source's same half; each is kept where the shifted
        # distance has the half's sign.
        before = ([(source.delta_poc_s1[j] + delta_rps, negative + j)
                   for j in reversed(range(positive))]
                  + [(delta_rps, source.num_delta_pocs)]
                  + [(source.delta_poc_s0[j] + delta_rps, j) for j in range(negative)])
        after = ([(source.delta_poc_s0[j] + delta_rps, j)
                  for j in reversed(range(negative))]
                 + [(delta_rps, source.num_delta_pocs)]
                 + [(source.delta_poc_s1[j] + delta_rps, negative + j)
                    for j in range(positive)])
        s0 = [(delta, used[j]) for delta, j in before if delta < 0 and use_delta[j]]
        s1 = [(delta, used[j]) for delta, j in after if delta > 0 and use_delta[j]]
    else:
        negative = reader.ue()
        positive = reader.ue()
        if negative > 16 or positive > 16:
            raise StreamError(
                f'a reference picture set of {negative} and {positive} pictures; '
                'no decoded picture buffer holds more than 16')
        s0, s1 = [], []
        delta = 0
        for _ in range(negative):
            delta -= reader.ue() + 1            # delta_poc_s0_minus1
            s0.append((delta, reader.flag()))
        delta = 0
        for _ in range(positive):
            delta += reader.ue() + 1            # delta_poc_s1_minus1
            s1.append((delta, reader.flag()))
    return ShortTermRefPicSet(
        delta_poc_s0=tuple(delta for delta, _ in s0),
        used_by_curr_pic_s0=tuple(used for _, used in s0),
        delta_poc_s1=tuple(delta for delta, _ in s1),
        used_by_curr_pic_s1=tuple(used for _, used in s1))


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
    matrix_coeffs: int = 2
    chroma_sample_loc_type_top_field: int = 0
    chroma_sample_loc_type_bottom_field: int = 0
    field_seq_flag: int = 0
    frame_field_info_present_flag: int = 0
    default_display_window: tuple[int, int, int, int] = (0, 0, 0, 0)
    vui_num_units_in_tick: int = 0
    vui_time_scale: int = 0
    vui_hrd_parameters_present_flag: int = 0
    bitstream_restriction_flag: int = 0


def _skip_sub_layer_hrd(reader: BitReader, count: int, sub_pictures: int) -> None:
    """sub_layer_hrd_parameters() (E.2.3), read past."""
    for _ in range(count):
        reader.ue()                             # bit_rate_value_minus1
        reader.ue()                             # cpb_size_value_minus1
        if sub_pictures:
            reader.ue()                         # cpb_size_du_value_minus1
            reader.ue()                         # bit_rate_du_value_minus1
        reader.flag()                           # cbr_flag


def _skip_hrd(reader: BitReader, sub_layers_minus1: int) -> None:
    """hrd_parameters(1, maxNumSubLayersMinus1) (E.2.2), read past."""
    nal = reader.flag()
    vcl = reader.flag()
    sub_pictures = 0
    if nal or vcl:
        sub_pictures = reader.flag()
        if sub_pictures:
            reader.u(19)                        # tick divisor and three lengths
        reader.u(8)                             # bit_rate_scale, cpb_size_scale
        if sub_pictures:
            reader.u(4)                         # cpb_size_du_scale
        reader.u(15)                            # three delay lengths
    for _ in range(sub_layers_minus1 + 1):
        fixed_general = reader.flag()
        fixed_in_sequence = 1 if fixed_general else reader.flag()
        low_delay = 0
        if fixed_in_sequence:
            reader.ue()                         # elemental_duration_in_tc_minus1
        else:
            low_delay = reader.flag()
        count = 1 if low_delay else reader.ue() + 1
        if nal:
            _skip_sub_layer_hrd(reader, count, sub_pictures)
        if vcl:
            _skip_sub_layer_hrd(reader, count, sub_pictures)


def _parse_vui(reader: BitReader, sub_layers_minus1: int) -> VideoUsability:
    """vui_parameters() (E.2.1)."""
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
            found['matrix_coeffs'] = reader.u(8)
    if reader.flag():                           # chroma_loc_info_present_flag
        found['chroma_sample_loc_type_top_field'] = reader.ue()
        found['chroma_sample_loc_type_bottom_field'] = reader.ue()
    reader.flag()                               # neutral_chroma_indication_flag
    found['field_seq_flag'] = reader.flag()
    found['frame_field_info_present_flag'] = reader.flag()
    if reader.flag():                           # default_display_window_flag
        found['default_display_window'] = (reader.ue(), reader.ue(),
                                           reader.ue(), reader.ue())
    if reader.flag():                           # vui_timing_info_present_flag
        found['vui_num_units_in_tick'] = reader.u(32)
        found['vui_time_scale'] = reader.u(32)
        if reader.flag():                       # vui_poc_proportional_to_timing_flag
            reader.ue()                         # vui_num_ticks_poc_diff_one_minus1
        if reader.flag():                       # vui_hrd_parameters_present_flag
            found['vui_hrd_parameters_present_flag'] = 1
            _skip_hrd(reader, sub_layers_minus1)
    if reader.flag():                           # bitstream_restriction_flag
        found['bitstream_restriction_flag'] = 1
        reader.u(3)                             # tiles fixed, motion vectors, lists
        for _ in range(5):                      # segmentation, two denominators,
            reader.ue()                         # two motion vector lengths
    return VideoUsability(**found)


@dataclasses.dataclass(frozen=True)
class SequenceParameterSet:
    """A sequence parameter set (7.3.2.2), with the values derived from it.

    conformance_window -- the left, right, top and bottom offsets, in chroma
        sample units
    scaling_lists -- the set's own lists, or None where it codes none
    short_term_ref_pic_sets -- the reference picture sets a slice may name
    lt_ref_pic_poc_lsb_sps, used_by_curr_pic_lt_sps_flag -- the long-term
        pictures a slice may name, as picture order count LSBs
    range_extension -- the names of the range extension flags that are set
    """

    sps_video_parameter_set_id: int
    sps_max_sub_layers_minus1: int
    sps_temporal_id_nesting_flag: int
    profile_tier_level: ProfileTierLevel
    sps_seq_parameter_set_id: int
    chroma_format_idc: int
    separate_colour_plane_flag: int
    pic_width_in_luma_samples: int
    pic_height_in_luma_samples: int
    conformance_window: tuple[int, int, int, int]
    bit_depth_luma_minus8: int
    bit_depth_chroma_minus8: int
    log2_max_pic_order_cnt_lsb_minus4: int
    sps_max_dec_pic_buffering_minus1: tuple[int, ...]
    sps_max_num_reorder_pics: tuple[int, ...]
    sps_max_latency_increase_plus1: tuple[int, ...]
    log2_min_luma_coding_block_size_minus3: int
    log2_diff_max_min_luma_coding_block_size: int
    log2_min_luma_transform_block_size_minus2: int
    log2_diff_max_min_luma_transform_block_size: int
    max_transform_hierarchy_depth_inter: int
    max_transform_hierarchy_depth_intra: int
    scaling_list_enabled_flag: int
    sps_scaling_list_data_present_flag: int
    scaling_lists: ScalingLists | None
    amp_enabled_flag: int
    sample_adaptive_offset_enabled_flag: int
    pcm_enabled_flag: int
    pcm_sample_bit_depth_luma_minus1: int
    pcm_sample_bit_depth_chroma_minus1: int
    log2_min_pcm_luma_coding_block_size_minus3: int
    log2_diff_max_min_pcm_luma_coding_block_size: int
    pcm_loop_filter_disabled_flag: int
    short_term_ref_pic_sets: tuple[ShortTermRefPicSet, ...]
    long_term_ref_pics_present_flag: int
    lt_ref_pic_poc_lsb_sps: tuple[int, ...]
    used_by_curr_pic_lt_sps_flag: tuple[int, ...]
    sps_temporal_mvp_enabled_flag: int
    strong_intra_smoothing_enabled_flag: int
    vui: VideoUsability | None
    sps_range_extension_flag: int = 0
    sps_multilayer_extension_flag: int = 0
    sps_3d_extension_flag: int = 0
    sps_scc_extension_flag: int = 0
    range_extension: tuple[str, ...] = ()

    # ------------------------------------------------------------ derived

    @property
    def chroma_array_type(self) -> int:
        """ChromaArrayType: 0 for monochrome or separately coded planes."""
        return 0 if self.separate_colour_plane_flag else self.chroma_format_idc

    @property
    def bit_depth(self) -> int:
        """Bits a luma sample."""
        return 8 + self.bit_depth_luma_minus8

    @property
    def max_pic_order_cnt_lsb(self) -> int:
        return 1 << (self.log2_max_pic_order_cnt_lsb_minus4 + 4)

    @property
    def ctb_log2_size(self) -> int:
        """CtbLog2SizeY: the coding tree block's size as a power of two."""
        return (self.log2_min_luma_coding_block_size_minus3 + 3
                + self.log2_diff_max_min_luma_coding_block_size)

    @property
    def width_in_ctbs(self) -> int:
        """PicWidthInCtbsY."""
        return -(-self.pic_width_in_luma_samples >> self.ctb_log2_size)

    @property
    def height_in_ctbs(self) -> int:
        """PicHeightInCtbsY."""
        return -(-self.pic_height_in_luma_samples >> self.ctb_log2_size)

    @property
    def coded_size(self) -> tuple[int, int]:
        """The size the decoder reconstructs, in luma samples."""
        return self.pic_width_in_luma_samples, self.pic_height_in_luma_samples

    @property
    def visible(self) -> tuple[int, int, int, int]:
        """The conformance window as ``(x, y, width, height)`` in luma samples."""
        unit_x, unit_y = CHROMA_SUBSAMPLING[self.chroma_format_idc]
        left, right, top, bottom = self.conformance_window
        width, height = self.coded_size
        return (unit_x * left, unit_y * top,
                width - unit_x * (left + right), height - unit_y * (top + bottom))

    @property
    def size(self) -> tuple[int, int]:
        """The picture's visible size, as ``(width, height)``."""
        return self.visible[2], self.visible[3]

    @property
    def frame_rate(self) -> tuple[int, int] | None:
        """Pictures per second as ``(numerator, denominator)``, or None if unstated."""
        vui = self.vui
        if vui is None:
            return None
        return _frame_rate(vui.vui_num_units_in_tick, vui.vui_time_scale)

    @property
    def dpb_size(self) -> int:
        """How many pictures the decoded picture buffer holds, the current one included."""
        return self.sps_max_dec_pic_buffering_minus1[-1] + 1

    @property
    def reorder_pics(self) -> int:
        """How many pictures may precede a picture in decoding order and follow
        it in output order."""
        return self.sps_max_num_reorder_pics[-1]

    @property
    def max_latency_pictures(self) -> int | None:
        """SpsMaxLatencyPictures: how many pictures may be output ahead of one
        that precedes them in decoding order, or None where there is no limit."""
        increase = self.sps_max_latency_increase_plus1[-1]
        return self.reorder_pics + increase - 1 if increase else None


def parse_sps(rbsp: bytes) -> SequenceParameterSet:
    """A sequence parameter set from its RBSP (the payload after the NAL header)."""
    reader = BitReader(rbsp)
    video_set = reader.u(4)
    sub_layers_minus1 = reader.u(3)
    nesting = reader.flag()
    profile = _profile_tier_level(reader, sub_layers_minus1)
    identifier = reader.ue()
    chroma_format_idc = reader.ue()
    if chroma_format_idc > 3:
        raise StreamError(f'chroma_format_idc {chroma_format_idc} is not defined')
    separate = reader.flag() if chroma_format_idc == 3 else 0
    width = reader.ue()
    height = reader.ue()
    window = ((reader.ue(), reader.ue(), reader.ue(), reader.ue()) if reader.flag()
              else (0, 0, 0, 0))
    depth_luma = reader.ue()
    depth_chroma = reader.ue()
    log2_poc = reader.ue()
    if log2_poc > 12:
        raise StreamError(
            f'log2_max_pic_order_cnt_lsb_minus4 is {log2_poc}, and 12 is the most')
    buffering, reorder, latency = _ordering_info(reader, sub_layers_minus1)
    min_coding = reader.ue()
    diff_coding = reader.ue()
    min_transform = reader.ue()
    diff_transform = reader.ue()
    depth_inter = reader.ue()
    depth_intra = reader.ue()
    scaling_enabled = reader.flag()
    scaling_present = reader.flag() if scaling_enabled else 0
    scaling = _scaling_list_data(reader) if scaling_present else None
    amp = reader.flag()
    sao = reader.flag()
    pcm = reader.flag()
    pcm_luma = pcm_chroma = pcm_min = pcm_diff = pcm_filter = 0
    if pcm:
        pcm_luma = reader.u(4)
        pcm_chroma = reader.u(4)
        pcm_min = reader.ue()
        pcm_diff = reader.ue()
        pcm_filter = reader.flag()
    count = reader.ue()
    if count > 64:
        raise StreamError(f'{count} short-term reference picture sets; 64 is the most')
    sets: list[ShortTermRefPicSet] = []
    for index in range(count):
        sets.append(_st_ref_pic_set(reader, index, sets, in_slice=False))
    long_term_present = reader.flag()
    long_term_lsb: list[int] = []
    long_term_used: list[int] = []
    if long_term_present:
        for _ in range(reader.ue()):            # num_long_term_ref_pics_sps
            long_term_lsb.append(reader.u(log2_poc + 4))
            long_term_used.append(reader.flag())
    temporal_mvp = reader.flag()
    strong_smoothing = reader.flag()
    vui = _parse_vui(reader, sub_layers_minus1) if reader.flag() else None
    extensions = [0, 0, 0, 0]
    range_extension: list[str] = []
    if reader.flag():                           # sps_extension_present_flag
        extensions = [reader.flag() for _ in range(4)]
        reader.u(4)                             # sps_extension_4bits
        if extensions[0]:
            range_extension = [name for name in SPS_RANGE_EXTENSION_FLAGS
                               if reader.flag()]
    return SequenceParameterSet(
        sps_video_parameter_set_id=video_set, sps_max_sub_layers_minus1=sub_layers_minus1,
        sps_temporal_id_nesting_flag=nesting, profile_tier_level=profile,
        sps_seq_parameter_set_id=identifier, chroma_format_idc=chroma_format_idc,
        separate_colour_plane_flag=separate, pic_width_in_luma_samples=width,
        pic_height_in_luma_samples=height, conformance_window=window,
        bit_depth_luma_minus8=depth_luma, bit_depth_chroma_minus8=depth_chroma,
        log2_max_pic_order_cnt_lsb_minus4=log2_poc,
        sps_max_dec_pic_buffering_minus1=buffering, sps_max_num_reorder_pics=reorder,
        sps_max_latency_increase_plus1=latency,
        log2_min_luma_coding_block_size_minus3=min_coding,
        log2_diff_max_min_luma_coding_block_size=diff_coding,
        log2_min_luma_transform_block_size_minus2=min_transform,
        log2_diff_max_min_luma_transform_block_size=diff_transform,
        max_transform_hierarchy_depth_inter=depth_inter,
        max_transform_hierarchy_depth_intra=depth_intra,
        scaling_list_enabled_flag=scaling_enabled,
        sps_scaling_list_data_present_flag=scaling_present, scaling_lists=scaling,
        amp_enabled_flag=amp, sample_adaptive_offset_enabled_flag=sao,
        pcm_enabled_flag=pcm, pcm_sample_bit_depth_luma_minus1=pcm_luma,
        pcm_sample_bit_depth_chroma_minus1=pcm_chroma,
        log2_min_pcm_luma_coding_block_size_minus3=pcm_min,
        log2_diff_max_min_pcm_luma_coding_block_size=pcm_diff,
        pcm_loop_filter_disabled_flag=pcm_filter,
        short_term_ref_pic_sets=tuple(sets),
        long_term_ref_pics_present_flag=long_term_present,
        lt_ref_pic_poc_lsb_sps=tuple(long_term_lsb),
        used_by_curr_pic_lt_sps_flag=tuple(long_term_used),
        sps_temporal_mvp_enabled_flag=temporal_mvp,
        strong_intra_smoothing_enabled_flag=strong_smoothing, vui=vui,
        sps_range_extension_flag=extensions[0],
        sps_multilayer_extension_flag=extensions[1],
        sps_3d_extension_flag=extensions[2], sps_scc_extension_flag=extensions[3],
        range_extension=tuple(range_extension))


def colour_of(sps: SequenceParameterSet) -> Colour:
    """A sequence's colour description, the matrix chosen where it states none."""
    vui = sps.vui
    if vui is None:
        return Colour().resolved(sps.size[1])
    return Colour(matrix=vui.matrix_coeffs, full_range=bool(vui.video_full_range_flag),
                  primaries=vui.colour_primaries, transfer=vui.transfer_characteristics,
                  chroma_location=vui.chroma_sample_loc_type_top_field,
                  ).resolved(sps.size[1])


@dataclasses.dataclass(frozen=True)
class PictureParameterSet:
    """A picture parameter set (7.3.2.3).

    column_width_minus1, row_height_minus1 -- the tile grid as the set codes
        it: every column and row but the last, and empty for a uniformly
        spaced grid. :meth:`tile_columns` and :meth:`tile_rows` give the whole
        grid for a picture.
    scaling_lists -- the set's own lists, or None where it codes none
    """

    pps_pic_parameter_set_id: int
    pps_seq_parameter_set_id: int
    dependent_slice_segments_enabled_flag: int
    output_flag_present_flag: int
    num_extra_slice_header_bits: int
    sign_data_hiding_enabled_flag: int
    cabac_init_present_flag: int
    num_ref_idx_l0_default_active_minus1: int
    num_ref_idx_l1_default_active_minus1: int
    init_qp_minus26: int
    constrained_intra_pred_flag: int
    transform_skip_enabled_flag: int
    cu_qp_delta_enabled_flag: int
    diff_cu_qp_delta_depth: int
    pps_cb_qp_offset: int
    pps_cr_qp_offset: int
    pps_slice_chroma_qp_offsets_present_flag: int
    weighted_pred_flag: int
    weighted_bipred_flag: int
    transquant_bypass_enabled_flag: int
    tiles_enabled_flag: int
    entropy_coding_sync_enabled_flag: int
    num_tile_columns_minus1: int
    num_tile_rows_minus1: int
    uniform_spacing_flag: int
    column_width_minus1: tuple[int, ...]
    row_height_minus1: tuple[int, ...]
    loop_filter_across_tiles_enabled_flag: int
    pps_loop_filter_across_slices_enabled_flag: int
    deblocking_filter_control_present_flag: int
    deblocking_filter_override_enabled_flag: int
    pps_deblocking_filter_disabled_flag: int
    pps_beta_offset_div2: int
    pps_tc_offset_div2: int
    pps_scaling_list_data_present_flag: int
    scaling_lists: ScalingLists | None
    lists_modification_present_flag: int
    log2_parallel_merge_level_minus2: int
    slice_segment_header_extension_present_flag: int
    pps_range_extension_flag: int = 0
    pps_multilayer_extension_flag: int = 0
    pps_3d_extension_flag: int = 0
    pps_scc_extension_flag: int = 0
    log2_max_transform_skip_block_size_minus2: int = 0
    cross_component_prediction_enabled_flag: int = 0
    chroma_qp_offset_list_enabled_flag: int = 0
    log2_sao_offset_scale_luma: int = 0
    log2_sao_offset_scale_chroma: int = 0

    def tile_columns(self, sps: SequenceParameterSet) -> tuple[int, ...]:
        """The width of every tile column in coding tree blocks (6.5.1)."""
        return _tile_sizes(self.num_tile_columns_minus1 + 1, sps.width_in_ctbs,
                           self.uniform_spacing_flag, self.column_width_minus1)

    def tile_rows(self, sps: SequenceParameterSet) -> tuple[int, ...]:
        """The height of every tile row in coding tree blocks (6.5.1)."""
        return _tile_sizes(self.num_tile_rows_minus1 + 1, sps.height_in_ctbs,
                           self.uniform_spacing_flag, self.row_height_minus1)


def _tile_sizes(count: int, total: int, uniform: int,
                coded_minus1: tuple[int, ...]) -> tuple[int, ...]:
    """Equations 6-3 and 6-4: `count` tiles across `total` coding tree blocks."""
    if uniform:
        return tuple((index + 1) * total // count - index * total // count
                     for index in range(count))
    sizes = [value + 1 for value in coded_minus1]
    if sum(sizes) >= total:
        raise StreamError(
            f'tiles of {sizes} coding tree blocks leave none for the last of '
            f'{count} across a picture of {total}')
    return (*sizes, total - sum(sizes))


def parse_pps(rbsp: bytes) -> PictureParameterSet:
    """A picture parameter set from its RBSP."""
    reader = BitReader(rbsp)
    found: dict[str, Any] = {
        'pps_pic_parameter_set_id': reader.ue(),
        'pps_seq_parameter_set_id': reader.ue(),
        'dependent_slice_segments_enabled_flag': reader.flag(),
        'output_flag_present_flag': reader.flag(),
        'num_extra_slice_header_bits': reader.u(3),
        'sign_data_hiding_enabled_flag': reader.flag(),
        'cabac_init_present_flag': reader.flag(),
        'num_ref_idx_l0_default_active_minus1': reader.ue(),
        'num_ref_idx_l1_default_active_minus1': reader.ue(),
        'init_qp_minus26': reader.se(),
        'constrained_intra_pred_flag': reader.flag(),
        'transform_skip_enabled_flag': reader.flag(),
    }
    if (found['num_ref_idx_l0_default_active_minus1'] > 14
            or found['num_ref_idx_l1_default_active_minus1'] > 14):
        raise StreamError('a picture parameter set with more than 15 active references')
    found['cu_qp_delta_enabled_flag'] = reader.flag()
    found['diff_cu_qp_delta_depth'] = reader.ue() if found['cu_qp_delta_enabled_flag'] else 0
    found['pps_cb_qp_offset'] = reader.se()
    found['pps_cr_qp_offset'] = reader.se()
    found['pps_slice_chroma_qp_offsets_present_flag'] = reader.flag()
    found['weighted_pred_flag'] = reader.flag()
    found['weighted_bipred_flag'] = reader.flag()
    found['transquant_bypass_enabled_flag'] = reader.flag()
    tiles = found['tiles_enabled_flag'] = reader.flag()
    found['entropy_coding_sync_enabled_flag'] = reader.flag()
    columns_minus1 = rows_minus1 = 0
    uniform = across_tiles = 1
    widths: tuple[int, ...] = ()
    heights: tuple[int, ...] = ()
    if tiles:
        columns_minus1 = reader.ue()
        rows_minus1 = reader.ue()
        if columns_minus1 > 19 or rows_minus1 > 21:
            raise StreamError(
                f'{columns_minus1 + 1} tile columns and {rows_minus1 + 1} rows; '
                '20 and 22 are the most')
        uniform = reader.flag()
        if not uniform:
            widths = tuple(reader.ue() for _ in range(columns_minus1))
            heights = tuple(reader.ue() for _ in range(rows_minus1))
        across_tiles = reader.flag()
    found.update(num_tile_columns_minus1=columns_minus1, num_tile_rows_minus1=rows_minus1,
                 uniform_spacing_flag=uniform, column_width_minus1=widths,
                 row_height_minus1=heights,
                 loop_filter_across_tiles_enabled_flag=across_tiles)
    found['pps_loop_filter_across_slices_enabled_flag'] = reader.flag()
    control = found['deblocking_filter_control_present_flag'] = reader.flag()
    override = disabled = beta = tc = 0
    if control:
        override = reader.flag()
        disabled = reader.flag()
        if not disabled:
            beta = reader.se()
            tc = reader.se()
    found.update(deblocking_filter_override_enabled_flag=override,
                 pps_deblocking_filter_disabled_flag=disabled,
                 pps_beta_offset_div2=beta, pps_tc_offset_div2=tc)
    present = found['pps_scaling_list_data_present_flag'] = reader.flag()
    found['scaling_lists'] = _scaling_list_data(reader) if present else None
    found['lists_modification_present_flag'] = reader.flag()
    found['log2_parallel_merge_level_minus2'] = reader.ue()
    found['slice_segment_header_extension_present_flag'] = reader.flag()
    if reader.flag():                           # pps_extension_present_flag
        found['pps_range_extension_flag'] = reader.flag()
        found['pps_multilayer_extension_flag'] = reader.flag()
        found['pps_3d_extension_flag'] = reader.flag()
        found['pps_scc_extension_flag'] = reader.flag()
        reader.u(4)                             # pps_extension_4bits
        if found['pps_range_extension_flag']:   # pps_range_extension() (7.3.2.3.2)
            if found['transform_skip_enabled_flag']:
                found['log2_max_transform_skip_block_size_minus2'] = reader.ue()
            found['cross_component_prediction_enabled_flag'] = reader.flag()
            found['chroma_qp_offset_list_enabled_flag'] = reader.flag()
            if found['chroma_qp_offset_list_enabled_flag']:
                reader.ue()                     # diff_cu_chroma_qp_offset_depth
                for _ in range(reader.ue() + 1):
                    reader.se()                 # cb_qp_offset_list
                    reader.se()                 # cr_qp_offset_list
            found['log2_sao_offset_scale_luma'] = reader.ue()
            found['log2_sao_offset_scale_chroma'] = reader.ue()
    return PictureParameterSet(**found)


# ------------------------------------------------------- slice segment headers


@dataclasses.dataclass(frozen=True)
class LongTermPicture:
    """One long-term entry of a slice's reference picture set (7.4.7.1).

    poc_lsb -- PocLsbLt: the picture's order count, modulo MaxPicOrderCntLsb
    used_by_curr_pic -- UsedByCurrPicLt: the current picture may predict from it
    delta_poc_msb_present_flag -- the entry also states the picture's MSB cycle
    delta_poc_msb_cycle -- DeltaPocMsbCycleLt: how many MaxPicOrderCntLsb
        cycles before the current picture's the picture's order count is
    """

    poc_lsb: int
    used_by_curr_pic: int
    delta_poc_msb_present_flag: int = 0
    delta_poc_msb_cycle: int = 0


@dataclasses.dataclass(frozen=True)
class PredWeightTable:
    """pred_weight_table() (7.3.6.3), as libva takes it.

    delta_luma_weight_l0, luma_offset_l0 -- for each active entry of list 0,
        zero where the entry has no explicit luma weight
    delta_chroma_weight_l0 -- ``(Cb, Cr)`` for each entry
    chroma_offset_l0 -- ChromaOffsetL0 of equation 7-56, ``(Cb, Cr)`` for
        each entry: the offset applied, derived from the coded delta
    """

    luma_log2_weight_denom: int
    delta_chroma_log2_weight_denom: int
    delta_luma_weight_l0: tuple[int, ...] = ()
    luma_offset_l0: tuple[int, ...] = ()
    delta_chroma_weight_l0: tuple[tuple[int, int], ...] = ()
    chroma_offset_l0: tuple[tuple[int, int], ...] = ()
    delta_luma_weight_l1: tuple[int, ...] = ()
    luma_offset_l1: tuple[int, ...] = ()
    delta_chroma_weight_l1: tuple[tuple[int, int], ...] = ()
    chroma_offset_l1: tuple[tuple[int, int], ...] = ()


@dataclasses.dataclass(frozen=True)
class SliceHeader:
    """A slice segment header (7.3.6.1), with the parameter sets it refers to.

    A dependent slice segment carries the fields of the slice it continues,
    with its own address and entry points.

    short_term_ref_pic_set -- the set in force, whether the header codes it or
        names one of the sequence's; empty for an IDR picture
    st_rps_bits -- how many bits the set takes where the header codes it,
        counted with emulation prevention removed; otherwise zero
    long_term -- the long-term entries, those naming the sequence's first
    list_entry_l0, list_entry_l1 -- ref_pic_lists_modification(): indices
        into the reference picture set for each active entry, or empty where
        the list is not modified
    collocated_ref_idx -- None where the slice has no temporal motion vector
        prediction
    slice_deblocking_filter_disabled_flag, slice_beta_offset_div2,
    slice_tc_offset_div2, slice_loop_filter_across_slices_enabled_flag -- the
        picture parameter set's values where the header does not override them
    entry_point_offset_minus1 -- one for each entry point
    header_bytes -- where slice_segment_data() begins, in bytes from the start
        of the NAL unit, its two header bytes included, counted with emulation
        prevention removed; this is libva's ``slice_data_byte_offset``
    header_escapes -- emulation prevention bytes within those header bytes
    """

    nal_unit_type: int
    temporal_id: int
    first_slice_segment_in_pic_flag: int
    no_output_of_prior_pics_flag: int
    slice_pic_parameter_set_id: int
    dependent_slice_segment_flag: int
    slice_segment_address: int
    slice_type: int
    pic_output_flag: int
    colour_plane_id: int
    slice_pic_order_cnt_lsb: int
    short_term_ref_pic_set_sps_flag: int
    short_term_ref_pic_set_idx: int
    short_term_ref_pic_set: ShortTermRefPicSet
    st_rps_bits: int
    long_term: tuple[LongTermPicture, ...]
    slice_temporal_mvp_enabled_flag: int
    slice_sao_luma_flag: int
    slice_sao_chroma_flag: int
    num_ref_idx_active_override_flag: int
    num_ref_idx_l0_active_minus1: int
    num_ref_idx_l1_active_minus1: int
    list_entry_l0: tuple[int, ...]
    list_entry_l1: tuple[int, ...]
    mvd_l1_zero_flag: int
    cabac_init_flag: int
    collocated_from_l0_flag: int
    collocated_ref_idx: int | None
    pred_weight_table: PredWeightTable | None
    five_minus_max_num_merge_cand: int
    slice_qp_delta: int
    slice_cb_qp_offset: int
    slice_cr_qp_offset: int
    cu_chroma_qp_offset_enabled_flag: int
    deblocking_filter_override_flag: int
    slice_deblocking_filter_disabled_flag: int
    slice_beta_offset_div2: int
    slice_tc_offset_div2: int
    slice_loop_filter_across_slices_enabled_flag: int
    entry_point_offset_minus1: tuple[int, ...]
    header_bytes: int
    header_escapes: int
    sps: SequenceParameterSet = dataclasses.field(repr=False, compare=False)
    pps: PictureParameterSet = dataclasses.field(repr=False, compare=False)

    @property
    def idr(self) -> bool:
        """Whether the slice belongs to an IDR picture."""
        return self.nal_unit_type in (NAL_IDR_W_RADL, NAL_IDR_N_LP)

    @property
    def bla(self) -> bool:
        """Whether the slice belongs to a broken link access picture."""
        return NAL_BLA_W_LP <= self.nal_unit_type <= NAL_BLA_N_LP

    @property
    def irap(self) -> bool:
        """Whether the slice belongs to an intra random access point picture."""
        return NAL_FIRST_IRAP <= self.nal_unit_type <= NAL_LAST_IRAP

    @property
    def rasl(self) -> bool:
        """Whether the slice belongs to a random access skipped leading picture."""
        return self.nal_unit_type in (NAL_RASL_N, NAL_RASL_R)

    @property
    def radl(self) -> bool:
        """Whether the slice belongs to a random access decodable leading picture."""
        return self.nal_unit_type in (NAL_RADL_N, NAL_RADL_R)

    @property
    def sub_layer_non_reference(self) -> bool:
        """Whether no picture of the same sub-layer predicts from this one."""
        return self.nal_unit_type <= NAL_LAST_SUB_LAYER_TYPE and not self.nal_unit_type % 2

    @property
    def kind(self) -> str:
        """The slice type's name: ``'B'``, ``'P'`` or ``'I'``."""
        return SLICE_KINDS[self.slice_type]

    @property
    def num_pic_total_curr(self) -> int:
        """NumPicTotalCurr: the pictures the current one may predict from (7-55)."""
        return (self.short_term_ref_pic_set.used
                + sum(entry.used_by_curr_pic for entry in self.long_term))

    @property
    def num_entry_point_offsets(self) -> int:
        return len(self.entry_point_offset_minus1)


def _ceil_log2(value: int) -> int:
    """Ceil(Log2(value)) for a positive integer."""
    return (value - 1).bit_length()


def _long_term(reader: BitReader, sps: SequenceParameterSet) -> tuple[LongTermPicture, ...]:
    """The long-term part of a slice segment header, cycles accumulated (7-52)."""
    candidates = len(sps.lt_ref_pic_poc_lsb_sps)
    from_sps = reader.ue() if candidates else 0     # num_long_term_sps
    if from_sps > candidates:
        raise StreamError(
            f'a slice names {from_sps} of the sequence\'s {candidates} long-term pictures')
    coded = reader.ue()                             # num_long_term_pics
    if from_sps + coded > 16:
        raise StreamError(f'a slice with {from_sps + coded} long-term pictures')
    entries: list[LongTermPicture] = []
    for index in range(from_sps + coded):
        if index < from_sps:
            chosen = reader.u(_ceil_log2(candidates)) if candidates > 1 else 0
            if chosen >= candidates:
                raise StreamError(
                    f'lt_idx_sps {chosen} names none of the sequence\'s '
                    f'{candidates} long-term pictures')
            lsb = sps.lt_ref_pic_poc_lsb_sps[chosen]
            used = sps.used_by_curr_pic_lt_sps_flag[chosen]
        else:
            lsb = reader.u(sps.log2_max_pic_order_cnt_lsb_minus4 + 4)
            used = reader.flag()
        present = reader.flag()                     # delta_poc_msb_present_flag
        cycle = reader.ue() if present else 0       # delta_poc_msb_cycle_lt
        if index not in (0, from_sps):
            cycle += entries[-1].delta_poc_msb_cycle
        entries.append(LongTermPicture(lsb, used, present, cycle))
    return tuple(entries)


def _clip_offset(value: int) -> int:
    """Clip3(-128, 127, value): the range of a chroma offset without the
    high-precision extension."""
    return max(-128, min(127, value))


def _weights(reader: BitReader, count: int, chroma: bool, chroma_denom: int,
             ) -> tuple[tuple[int, ...], tuple[int, ...],
                        tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """One list of pred_weight_table()."""
    luma_flags = [reader.flag() for _ in range(count)]
    chroma_flags = [reader.flag() if chroma else 0 for _ in range(count)]
    weights, offsets = [], []
    chroma_weights: list[tuple[int, int]] = []
    chroma_offsets: list[tuple[int, int]] = []
    for luma_flag, chroma_flag in zip(luma_flags, chroma_flags, strict=True):
        weights.append(reader.se() if luma_flag else 0)
        offsets.append(reader.se() if luma_flag else 0)
        pair_weights, pair_offsets = [0, 0], [0, 0]
        if chroma_flag:
            for component in range(2):
                delta_weight = reader.se()
                delta_offset = reader.se()
                weight = (1 << chroma_denom) + delta_weight
                pair_weights[component] = delta_weight
                # Equation 7-56, with wpOffsetHalfRangeC of 128.
                pair_offsets[component] = _clip_offset(
                    128 + delta_offset - ((128 * weight) >> chroma_denom))
        chroma_weights.append((pair_weights[0], pair_weights[1]))
        chroma_offsets.append((pair_offsets[0], pair_offsets[1]))
    return tuple(weights), tuple(offsets), tuple(chroma_weights), tuple(chroma_offsets)


def _pred_weight_table(reader: BitReader, sps: SequenceParameterSet, l0: int,
                       l1: int | None) -> PredWeightTable:
    denom = reader.ue()
    chroma = sps.chroma_array_type != 0
    delta_chroma_denom = reader.se() if chroma else 0
    if denom > 7 or not 0 <= denom + delta_chroma_denom <= 7:
        raise StreamError(
            f'weighted prediction denominators of {denom} and '
            f'{denom + delta_chroma_denom} bits; 7 is the most')
    chroma_denom = denom + delta_chroma_denom
    found: dict[str, Any] = {}
    for name, count in (('l0', l0), ('l1', l1)):
        if count is None:
            continue
        (found[f'delta_luma_weight_{name}'], found[f'luma_offset_{name}'],
         found[f'delta_chroma_weight_{name}'], found[f'chroma_offset_{name}'],
         ) = _weights(reader, count, chroma, chroma_denom)
    return PredWeightTable(luma_log2_weight_denom=denom,
                           delta_chroma_log2_weight_denom=delta_chroma_denom, **found)


def _escapes_within(data: bytes, rbsp_bytes: int) -> int:
    """How many emulation prevention bytes `data` has in its first `rbsp_bytes`
    bytes, counted as bytes of the payload with the escapes removed."""
    found = 0
    position = data.find(b'\x00\x00\x03')
    while position >= 0 and position + 2 - found < rbsp_bytes:
        found += 1
        position = data.find(b'\x00\x00\x03', position + 3)
    return found


def _unsupported_extension(pps: PictureParameterSet) -> str:
    """The extension of `pps` that changes the slice segment header, or ''."""
    if pps.pps_multilayer_extension_flag:
        return 'a multilayer extension'
    if pps.pps_3d_extension_flag:
        return 'a 3D extension'
    if pps.pps_scc_extension_flag:
        return 'a screen content coding extension'
    return ''


def parse_slice_header(unit: NalUnit,
                       sequences: Mapping[int, SequenceParameterSet],
                       pictures: Mapping[int, PictureParameterSet],
                       independent: SliceHeader | None = None) -> SliceHeader:
    """The header of a slice segment NAL unit, read against the parameter sets so far.

    independent -- the header of the picture's last independent slice segment,
        which a dependent slice segment takes most of its fields from
    """
    reader = BitReader(unit.rbsp)
    kind = unit.nal_unit_type
    first = reader.flag()
    irap = NAL_FIRST_IRAP <= kind <= NAL_LAST_IRAP
    no_output = reader.flag() if irap else 0
    pps_id = reader.ue()
    pps = pictures.get(pps_id)
    if pps is None:
        raise StreamError(
            f'a slice names picture parameter set {pps_id}, which the stream '
            'has not defined')
    sps = sequences.get(pps.pps_seq_parameter_set_id)
    if sps is None:
        raise StreamError(
            f'picture parameter set {pps_id} names sequence parameter set '
            f'{pps.pps_seq_parameter_set_id}, which the stream has not defined')
    extension = _unsupported_extension(pps)
    if extension:
        raise StreamError(
            f'picture parameter set {pps_id} has {extension}, and slices coded '
            'with one are not read here')
    dependent = address = 0
    if not first:
        if pps.dependent_slice_segments_enabled_flag:
            dependent = reader.flag()
        address = reader.u(_ceil_log2(sps.width_in_ctbs * sps.height_in_ctbs))
    found: dict[str, Any]
    if dependent:
        if independent is None:
            raise StreamError(
                f'the dependent slice segment at coding tree block {address} '
                'follows no slice of its picture')
        found = {field.name: getattr(independent, field.name)
                 for field in dataclasses.fields(independent)}
    else:
        found = _independent_fields(reader, kind, sps, pps)
    offsets: tuple[int, ...] = ()
    if pps.tiles_enabled_flag or pps.entropy_coding_sync_enabled_flag:
        count = reader.ue()                     # num_entry_point_offsets
        if count > sps.width_in_ctbs * sps.height_in_ctbs:
            raise StreamError(
                f'{count} entry points in a picture of '
                f'{sps.width_in_ctbs * sps.height_in_ctbs} coding tree blocks')
        if count:
            width = reader.ue() + 1             # offset_len_minus1
            if width > 32:
                raise StreamError(f'entry point offsets of {width} bits')
            offsets = tuple(reader.u(width) for _ in range(count))
    if pps.slice_segment_header_extension_present_flag:
        for _ in range(reader.ue()):            # slice_segment_header_extension_length
            reader.u(8)
    if not reader.flag():                       # byte_alignment()
        raise StreamError('a slice segment header does not end in its alignment bit')
    header_bytes = (reader.position + 7) // 8
    found.update(
        nal_unit_type=kind, temporal_id=unit.temporal_id,
        first_slice_segment_in_pic_flag=first, no_output_of_prior_pics_flag=no_output,
        slice_pic_parameter_set_id=pps_id, dependent_slice_segment_flag=dependent,
        slice_segment_address=address, entry_point_offset_minus1=offsets,
        header_bytes=header_bytes + 2,
        header_escapes=_escapes_within(unit.data[2:], header_bytes),
        sps=sps, pps=pps)
    return SliceHeader(**found)


def _independent_fields(reader: BitReader, kind: int, sps: SequenceParameterSet,
                        pps: PictureParameterSet) -> dict[str, Any]:
    """The part of a slice segment header a dependent slice segment leaves out."""
    reader.u(pps.num_extra_slice_header_bits)   # slice_reserved_flag
    slice_type = reader.ue()
    if slice_type > SLICE_I:
        raise StreamError(f'slice_type {slice_type} is not defined')
    output = reader.flag() if pps.output_flag_present_flag else 1
    colour_plane = reader.u(2) if sps.separate_colour_plane_flag else 0
    poc_lsb = from_sps = set_index = rps_bits = temporal_mvp = 0
    rps = ShortTermRefPicSet()
    long_term: tuple[LongTermPicture, ...] = ()
    if kind not in (NAL_IDR_W_RADL, NAL_IDR_N_LP):
        poc_lsb = reader.u(sps.log2_max_pic_order_cnt_lsb_minus4 + 4)
        sets = sps.short_term_ref_pic_sets
        from_sps = reader.flag()                # short_term_ref_pic_set_sps_flag
        if not from_sps:
            start = reader.position
            rps = _st_ref_pic_set(reader, len(sets), list(sets), in_slice=True)
            rps_bits = reader.position - start
        else:
            if not sets:
                raise StreamError(
                    'a slice names one of the sequence\'s reference picture '
                    'sets, and the sequence has none')
            set_index = reader.u(_ceil_log2(len(sets)))
            if set_index >= len(sets):
                raise StreamError(
                    f'short_term_ref_pic_set_idx {set_index} names none of the '
                    f'sequence\'s {len(sets)} sets')
            rps = sets[set_index]
        if sps.long_term_ref_pics_present_flag:
            long_term = _long_term(reader, sps)
        if sps.sps_temporal_mvp_enabled_flag:
            temporal_mvp = reader.flag()
    sao_luma = sao_chroma = 0
    if sps.sample_adaptive_offset_enabled_flag:
        sao_luma = reader.flag()
        if sps.chroma_array_type:
            sao_chroma = reader.flag()
    override = l0 = l1 = mvd_l1_zero = cabac_init = merge = 0
    collocated_from_l0 = 1
    collocated: int | None = None
    entries_l0: tuple[int, ...] = ()
    entries_l1: tuple[int, ...] = ()
    weights = None
    if slice_type != SLICE_I:
        bidirectional = slice_type == SLICE_B
        l0 = pps.num_ref_idx_l0_default_active_minus1
        l1 = pps.num_ref_idx_l1_default_active_minus1 if bidirectional else 0
        override = reader.flag()
        if override:
            l0 = reader.ue()
            l1 = reader.ue() if bidirectional else 0
        if l0 > 14 or l1 > 14:
            raise StreamError(
                f'a slice with {l0 + 1} and {l1 + 1} active references; 15 is the most')
        total = rps.used + sum(entry.used_by_curr_pic for entry in long_term)
        if pps.lists_modification_present_flag and total > 1:
            width = _ceil_log2(total)
            if reader.flag():                   # ref_pic_list_modification_flag_l0
                entries_l0 = tuple(reader.u(width) for _ in range(l0 + 1))
            if bidirectional and reader.flag():
                entries_l1 = tuple(reader.u(width) for _ in range(l1 + 1))
        if bidirectional:
            mvd_l1_zero = reader.flag()
        if pps.cabac_init_present_flag:
            cabac_init = reader.flag()
        if temporal_mvp:
            if bidirectional:
                collocated_from_l0 = reader.flag()
            collocated = 0
            if (l0 if collocated_from_l0 else l1) > 0:
                collocated = reader.ue()
        if ((pps.weighted_pred_flag and not bidirectional)
                or (pps.weighted_bipred_flag and bidirectional)):
            weights = _pred_weight_table(reader, sps, l0 + 1,
                                         l1 + 1 if bidirectional else None)
        merge = reader.ue()                     # five_minus_max_num_merge_cand
    qp_delta = reader.se()
    cb_offset = cr_offset = 0
    if pps.pps_slice_chroma_qp_offsets_present_flag:
        cb_offset = reader.se()
        cr_offset = reader.se()
    chroma_offset_list = reader.flag() if pps.chroma_qp_offset_list_enabled_flag else 0
    deblocking_override = 0
    if pps.deblocking_filter_override_enabled_flag:
        deblocking_override = reader.flag()
    disabled = pps.pps_deblocking_filter_disabled_flag
    beta, tc = pps.pps_beta_offset_div2, pps.pps_tc_offset_div2
    if deblocking_override:
        disabled = reader.flag()
        beta = tc = 0
        if not disabled:
            beta = reader.se()
            tc = reader.se()
    across_slices = pps.pps_loop_filter_across_slices_enabled_flag
    if across_slices and (sao_luma or sao_chroma or not disabled):
        across_slices = reader.flag()
    return {
        'slice_type': slice_type, 'pic_output_flag': output,
        'colour_plane_id': colour_plane, 'slice_pic_order_cnt_lsb': poc_lsb,
        'short_term_ref_pic_set_sps_flag': from_sps,
        'short_term_ref_pic_set_idx': set_index, 'short_term_ref_pic_set': rps,
        'st_rps_bits': rps_bits, 'long_term': long_term,
        'slice_temporal_mvp_enabled_flag': temporal_mvp,
        'slice_sao_luma_flag': sao_luma, 'slice_sao_chroma_flag': sao_chroma,
        'num_ref_idx_active_override_flag': override,
        'num_ref_idx_l0_active_minus1': l0, 'num_ref_idx_l1_active_minus1': l1,
        'list_entry_l0': entries_l0, 'list_entry_l1': entries_l1,
        'mvd_l1_zero_flag': mvd_l1_zero, 'cabac_init_flag': cabac_init,
        'collocated_from_l0_flag': collocated_from_l0,
        'collocated_ref_idx': collocated, 'pred_weight_table': weights,
        'five_minus_max_num_merge_cand': merge, 'slice_qp_delta': qp_delta,
        'slice_cb_qp_offset': cb_offset, 'slice_cr_qp_offset': cr_offset,
        'cu_chroma_qp_offset_enabled_flag': chroma_offset_list,
        'deblocking_filter_override_flag': deblocking_override,
        'slice_deblocking_filter_disabled_flag': disabled,
        'slice_beta_offset_div2': beta, 'slice_tc_offset_div2': tc,
        'slice_loop_filter_across_slices_enabled_flag': across_slices,
    }
