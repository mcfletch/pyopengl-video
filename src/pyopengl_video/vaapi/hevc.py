"""The HEVC header writer: parameter sets, slice segment headers and picture order.

An HEVC encoder under libva is told what to code and is given the headers that
say so: the video, sequence and picture parameter sets, and each picture's
slice segment header. This module writes them and holds no libva.
:class:`ParameterSets` turns a picture size, a frame rate, a quantiser and the
driver's block sizes and coding tools into the three sets and the slice
segment headers that go with them, and :class:`GroupOfPictures` decides what
each picture is and which pictures it predicts from.
:mod:`pyopengl_video.vaapi.hevc_encode` gives the same values to the driver.

Field names are the syntax elements of ITU-T H.265 (clause 7.3 and annex E).
:mod:`pyopengl_video.vaapi.hevc_parse` reads everything written here.
"""
from __future__ import annotations

import dataclasses

from pyopengl_video.vaapi.h264 import (
    COLOUR_PRIMARIES_BT709,
    MATRIX_BT709,
    MAX_QP,
    TRANSFER_BT709,
    VIDEO_FORMAT_UNSPECIFIED,
    BitWriter,
    annexb,
    checked_frame_rate,
    emulation_prevented,
)
from pyopengl_video.vaapi.hevc_parse import (
    MAX_LUMA_SAMPLES,
    MAX_PICTURE_SIDE,
    NAL_IDR_N_LP,
    NAL_PPS,
    NAL_SPS,
    NAL_TRAIL_R,
    NAL_VPS,
    PROFILE_MAIN,
    PROFILE_MAIN_10,
    SLICE_B,
    SLICE_I,
    SLICE_P,
)

__all__ = ['Blocks', 'GroupOfPictures', 'LEVELS', 'MAX_NUM_MERGE_CAND',
           'ParameterSets', 'Picture', 'Tools', 'level_for', 'nal_unit']

#: chroma_format_idc for 4:2:0, and how many luma samples one unit of a
#: conformance window offset is in each direction at that sampling.
CHROMA_420 = 1
CROP_UNIT = 2

#: How wide slice_pic_order_cnt_lsb is, as the ``minus4`` the syntax carries.
#: Consecutive pictures are one count apart, so any width tells them apart.
LOG2_MAX_POC_LSB_MINUS4 = 4

#: MaxNumMergeCand: how many merging candidates a prediction unit chooses from.
MAX_NUM_MERGE_CAND = 5

#: Levels, as ``(general_level_idc, MaxLumaPs, MaxLumaSr, MaxBR)`` from tables
#: A.8 and A.9. general_level_idc is thirty times the level number, MaxLumaPs
#: is in luma samples a picture, MaxLumaSr in luma samples a second, and MaxBR
#: is the Main tier's, in units of :data:`BITRATE_FACTOR` bits a second.
LEVELS: tuple[tuple[int, int, int, int], ...] = (
    (30, 36864, 552960, 128),
    (60, 122880, 3686400, 1500),
    (63, 245760, 7372800, 3000),
    (90, 552960, 16588800, 6000),
    (93, 983040, 33177600, 10000),
    (120, 2228224, 66846720, 12000),
    (123, 2228224, 133693440, 20000),
    (150, 8912896, 267386880, 25000),
    (153, 8912896, 534773760, 40000),
    (156, 8912896, 1069547520, 60000),
    (180, 35651584, 1069547520, 60000),
    (183, 35651584, 2139095040, 120000),
    (186, 35651584, 4278190080, 240000),
)

#: CpbBrVclFactor for the Main profile (table A.10): what MaxBR counts in.
BITRATE_FACTOR = 1000


def level_for(width: int, height: int, frame_rate: tuple[int, int],
              bitrate: int = 0) -> int:
    """The general_level_idc of the lowest level covering this size, rate and bitrate.

    width, height -- the coded size in luma samples
    bitrate -- bits per second; zero leaves the bitrate out of the choice

    A level covers a picture of at most MaxLumaPs samples whose sides are each
    at most ``sqrt(8 * MaxLumaPs)`` (A.4.1), at most MaxLumaSr samples a
    second and at most the Main tier's MaxBR (A.4.2). The highest level is
    returned where none covers the stream.
    """
    samples = width * height
    numerator, denominator = frame_rate
    for level_idc, max_luma_ps, max_luma_sr, max_br in LEVELS:
        if (samples <= max_luma_ps
                and width * width <= 8 * max_luma_ps
                and height * height <= 8 * max_luma_ps
                and samples * numerator <= max_luma_sr * denominator
                and bitrate <= max_br * BITRATE_FACTOR):
            return level_idc
    return LEVELS[-1][0]


def nal_unit(rbsp: bytes, nal_unit_type: int, temporal_id: int = 0) -> bytes:
    """An RBSP behind its two-byte NAL unit header, escaped (7.3.1.1 and 7.3.1.2).

    Raises :class:`ValueError` for a type outside 0 to 63 or a temporal_id
    outside 0 to 6, which the header has no room for.
    """
    if not 0 <= nal_unit_type <= 63:
        raise ValueError(f'nal_unit_type {nal_unit_type} is outside 0 to 63')
    if not 0 <= temporal_id <= 6:
        raise ValueError(f'temporal_id {temporal_id} is outside 0 to 6')
    return bytes([nal_unit_type << 1, temporal_id + 1]) + emulation_prevented(rbsp)


def _ceil_log2(value: int) -> int:
    """Ceil(Log2(value)) for a positive integer."""
    return (value - 1).bit_length()


@dataclasses.dataclass(frozen=True)
class Blocks:
    """The block sizes a sequence is coded with, each as a power of two.

    coding_tree -- CtbLog2SizeY, the coding tree block
    min_coding -- MinCbLog2SizeY, the smallest coding block
    max_transform, min_transform -- the largest and smallest transform block
    depth_inter, depth_intra -- max_transform_hierarchy_depth_inter and _intra
    """

    coding_tree: int = 5
    min_coding: int = 3
    max_transform: int = 5
    min_transform: int = 2
    depth_inter: int = 2
    depth_intra: int = 2

    def __post_init__(self) -> None:
        if not 4 <= self.coding_tree <= 6:
            raise ValueError(
                f'a coding tree block of {1 << self.coding_tree} samples; the '
                'Main profile has 16, 32 and 64')
        if not 3 <= self.min_coding <= self.coding_tree:
            raise ValueError(
                f'a smallest coding block of {1 << self.min_coding} samples in a '
                f'coding tree block of {1 << self.coding_tree}')
        if not (2 <= self.min_transform < self.min_coding
                and self.min_transform <= self.max_transform <= min(self.coding_tree, 5)):
            raise ValueError(
                f'transform blocks of {1 << self.min_transform} to '
                f'{1 << self.max_transform} samples do not fit coding blocks of '
                f'{1 << self.min_coding} to {1 << self.coding_tree}')
        deepest = self.coding_tree - self.min_transform
        if not (0 <= self.depth_inter <= deepest and 0 <= self.depth_intra <= deepest):
            raise ValueError(
                f'transform hierarchy depths of {self.depth_inter} and '
                f'{self.depth_intra}; {deepest} is the most these block sizes allow')


@dataclasses.dataclass(frozen=True)
class Tools:
    """The optional coding tools a sequence declares, each off unless named.

    amp -- amp_enabled_flag: asymmetric motion partitions
    sample_adaptive_offset -- sample_adaptive_offset_enabled_flag, with both
        of each slice's flags set
    strong_intra_smoothing -- strong_intra_smoothing_enabled_flag
    sign_data_hiding -- sign_data_hiding_enabled_flag
    constrained_intra_pred -- constrained_intra_pred_flag
    transform_skip -- transform_skip_enabled_flag
    cu_qp_delta -- cu_qp_delta_enabled_flag, with a quantisation group the
        size of the coding tree block
    transquant_bypass -- transquant_bypass_enabled_flag
    """

    amp: bool = False
    sample_adaptive_offset: bool = False
    strong_intra_smoothing: bool = False
    sign_data_hiding: bool = False
    constrained_intra_pred: bool = False
    transform_skip: bool = False
    cu_qp_delta: bool = False
    transquant_bypass: bool = False


@dataclasses.dataclass(frozen=True)
class Picture:
    """What one picture is coded as, and what it predicts from.

    idr -- the picture begins a coded video sequence and is coded without
        prediction
    poc -- PicOrderCntVal, which counts pictures from the IDR before
    references -- the picture order counts predicted from, nearest first
    bidirectional -- a predicted picture is a B picture whose second reference
        list repeats its first
    """

    idr: bool
    poc: int
    references: tuple[int, ...] = ()
    bidirectional: bool = False

    @property
    def intra(self) -> bool:
        """Whether the picture predicts from no other."""
        return not self.references

    @property
    def nal_unit_type(self) -> int:
        """IDR_N_LP for an IDR picture, since none follows it in decoding
        order and precedes it in output order; TRAIL_R for the rest."""
        return NAL_IDR_N_LP if self.idr else NAL_TRAIL_R

    @property
    def slice_type(self) -> int:
        if self.intra:
            return SLICE_I
        return SLICE_B if self.bidirectional else SLICE_P


class GroupOfPictures:
    """Decides what each picture of a recording is coded as.

    Pictures are coded in display order. An IDR picture opens the recording,
    follows every `gop` pictures and is coded wherever a caller asks for one.
    Every other picture predicts from the pictures before it, as many as
    `references` and as there have been since the IDR.

    gop -- pictures from one IDR to the next; one makes every picture an IDR
    references -- how many pictures the decoded picture buffer keeps
    bidirectional -- predicted pictures are B pictures (see :class:`Picture`)
    """

    def __init__(self, gop: int, references: int = 1,
                 bidirectional: bool = False) -> None:
        if gop < 1:
            raise ValueError(f'a group of {gop} pictures holds none')
        if references < 1:
            raise ValueError(
                f'a decoded picture buffer of {references} leaves a predicted '
                'picture nothing to predict from')
        self.gop = int(gop)
        self.references = int(references)
        self.bidirectional = bool(bidirectional)
        self._position = 0

    def __repr__(self) -> str:
        return f'<{type(self).__name__} gop {self.gop}, {self.references} references>'

    def next_picture(self, force_idr: bool = False) -> Picture:
        """The next picture to code. Each call advances the structure by one."""
        if force_idr or self._position % self.gop == 0:
            self._position = 0
        poc = self._position
        self._position += 1
        if not poc:
            return Picture(idr=True, poc=0)
        return Picture(
            idr=False, poc=poc,
            references=tuple(range(poc - 1, max(poc - self.references, 0) - 1, -1)),
            bidirectional=self.bidirectional)


class ParameterSets:
    """The parameter sets and slice segment headers of one encoding session.

    The stream is Main profile, Main tier: 8-bit 4:2:0, one slice a picture,
    limited-range BT.709.

    width, height -- the picture in luma samples, both even, and no larger
        coded than the largest picture of any level
    frame_rate -- an exact ``(numerator, denominator)``, both terms above zero
    references -- how many pictures the decoded picture buffer keeps to
        predict from
    init_qp -- the quantiser the picture parameter set starts from, 0 to 51
    bitrate -- bits per second, which the level has to cover; zero leaves it
        out of the level
    blocks -- the block sizes coded with
    tools -- the optional coding tools declared
    alignment -- what the coded width and height are multiples of, each a
        multiple of the smallest coding block; the smallest coding block
        itself by default
    bidirectional -- predicted pictures are B pictures (see :class:`Picture`)

    coded_size -- the picture rounded up to `alignment`, which is what
        pic_width_in_luma_samples and pic_height_in_luma_samples state; the
        conformance window crops it back
    level_idc -- general_level_idc

    :class:`ValueError` is raised here for anything the headers cannot
    state, so that none is written with a value that reads back as another.
    """

    def __init__(self, width: int, height: int, frame_rate: tuple[int, int], *,
                 references: int = 1, init_qp: int = 26, bitrate: int = 0,
                 blocks: Blocks | None = None, tools: Tools | None = None,
                 alignment: tuple[int, int] | None = None,
                 bidirectional: bool = False) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f'{width}x{height} is not a picture size')
        if width % CROP_UNIT or height % CROP_UNIT:
            raise ValueError(
                f'{width}x{height} has an odd dimension, and 4:2:0 chroma is '
                'subsampled by two in each direction')
        if not 0 <= init_qp <= MAX_QP:
            raise ValueError(f'{init_qp} is outside the HEVC quantiser range 0..51')
        if not 1 <= references <= 15:
            raise ValueError(f'{references} reference pictures; 1 to 15 are possible')
        if bitrate < 0:
            raise ValueError(f'a bitrate of {bitrate} bits per second')
        self.width = int(width)
        self.height = int(height)
        self.frame_rate = checked_frame_rate(frame_rate)
        self.references = int(references)
        self.init_qp = int(init_qp)
        self.bitrate = int(bitrate)
        self.blocks = blocks or Blocks()
        self.tools = tools or Tools()
        self.bidirectional = bool(bidirectional)

        smallest = 1 << self.blocks.min_coding
        across, down = alignment or (smallest, smallest)
        if across < 1 or down < 1 or across % smallest or down % smallest:
            raise ValueError(
                f'a coded size in multiples of {across}x{down} is not whole '
                f'coding blocks of {smallest} samples')
        self.coded_size = (-(-self.width // across) * across,
                           -(-self.height // down) * down)
        coded_width, coded_height = self.coded_size
        if (max(self.coded_size) > MAX_PICTURE_SIDE
                or coded_width * coded_height > MAX_LUMA_SAMPLES):
            raise ValueError(
                f'{width}x{height} is coded as {coded_width}x{coded_height}, which '
                'is larger than any level codes')
        #: conf_win_right_offset and conf_win_bottom_offset, in chroma samples.
        self.crop_right = (coded_width - self.width) // CROP_UNIT
        self.crop_bottom = (coded_height - self.height) // CROP_UNIT
        tree = 1 << self.blocks.coding_tree
        self.width_in_ctbs = -(-coded_width // tree)
        self.height_in_ctbs = -(-coded_height // tree)
        self.level_idc = level_for(coded_width, coded_height, self.frame_rate,
                                   self.bitrate)
        self.log2_max_poc_lsb = LOG2_MAX_POC_LSB_MINUS4 + 4

    def __repr__(self) -> str:
        return (f'<{type(self).__name__} {self.width}x{self.height} '
                f'@{self.frame_rate[0]}/{self.frame_rate[1]} level '
                f'{self.level_idc / 30:g}>')

    @property
    def ctbs(self) -> int:
        """PicSizeInCtbsY: the coding tree blocks of one picture."""
        return self.width_in_ctbs * self.height_in_ctbs

    # ------------------------------------------------------- shared syntax

    def _write_profile_tier_level(self, writer: BitWriter) -> None:
        """profile_tier_level(1, 0) (7.3.3): the general part alone."""
        writer.u(0, 2)                          # general_profile_space
        writer.flag(0)                          # general_tier_flag, Main
        writer.u(PROFILE_MAIN, 5)
        # A Main stream is also a Main 10 stream (A.3.3); flag j is bit 31 - j.
        writer.u(1 << (31 - PROFILE_MAIN) | 1 << (31 - PROFILE_MAIN_10), 32)
        writer.flag(1)                          # general_progressive_source_flag
        writer.flag(0)                          # general_interlaced_source_flag
        writer.flag(0)                          # general_non_packed_constraint_flag
        writer.flag(1)                          # general_frame_only_constraint_flag
        writer.u(0, 32)                         # 43 reserved bits and
        writer.u(0, 12)                         # general_inbld_flag
        writer.u(self.level_idc, 8)

    def _write_ordering(self, writer: BitWriter) -> None:
        """The decoded picture buffer's size, with no picture held back."""
        writer.flag(1)                          # sub_layer_ordering_info_present_flag
        writer.ue(self.references)              # max_dec_pic_buffering_minus1
        writer.ue(0)                            # max_num_reorder_pics
        writer.ue(0)                            # max_latency_increase_plus1

    # ------------------------------------------------------------------ VPS

    def vps(self) -> bytes:
        """The video parameter set (7.3.2.1), as a NAL unit."""
        writer = BitWriter()
        writer.u(0, 4)                          # vps_video_parameter_set_id
        writer.flag(1)                          # vps_base_layer_internal_flag
        writer.flag(1)                          # vps_base_layer_available_flag
        writer.u(0, 6)                          # vps_max_layers_minus1
        writer.u(0, 3)                          # vps_max_sub_layers_minus1
        writer.flag(1)                          # vps_temporal_id_nesting_flag
        writer.u(0xFFFF, 16)                    # vps_reserved_0xffff_16bits
        self._write_profile_tier_level(writer)
        self._write_ordering(writer)
        writer.u(0, 6)                          # vps_max_layer_id
        writer.ue(0)                            # vps_num_layer_sets_minus1
        writer.flag(1)                          # vps_timing_info_present_flag
        writer.u(self.frame_rate[1], 32)        # vps_num_units_in_tick
        writer.u(self.frame_rate[0], 32)        # vps_time_scale
        writer.flag(0)                          # vps_poc_proportional_to_timing_flag
        writer.ue(0)                            # vps_num_hrd_parameters
        writer.flag(0)                          # vps_extension_flag
        return nal_unit(writer.rbsp(), NAL_VPS)

    # ------------------------------------------------------------------ SPS

    def sps(self) -> bytes:
        """The sequence parameter set (7.3.2.2), as a NAL unit."""
        blocks, tools = self.blocks, self.tools
        writer = BitWriter()
        writer.u(0, 4)                          # sps_video_parameter_set_id
        writer.u(0, 3)                          # sps_max_sub_layers_minus1
        writer.flag(1)                          # sps_temporal_id_nesting_flag
        self._write_profile_tier_level(writer)
        writer.ue(0)                            # sps_seq_parameter_set_id
        writer.ue(CHROMA_420)
        writer.ue(self.coded_size[0])           # pic_width_in_luma_samples
        writer.ue(self.coded_size[1])           # pic_height_in_luma_samples
        cropped = self.crop_right or self.crop_bottom
        writer.flag(cropped)                    # conformance_window_flag
        if cropped:
            writer.ue(0)                        # conf_win_left_offset
            writer.ue(self.crop_right)
            writer.ue(0)                        # conf_win_top_offset
            writer.ue(self.crop_bottom)
        writer.ue(0)                            # bit_depth_luma_minus8
        writer.ue(0)                            # bit_depth_chroma_minus8
        writer.ue(self.log2_max_poc_lsb - 4)
        self._write_ordering(writer)
        writer.ue(blocks.min_coding - 3)
        writer.ue(blocks.coding_tree - blocks.min_coding)
        writer.ue(blocks.min_transform - 2)
        writer.ue(blocks.max_transform - blocks.min_transform)
        writer.ue(blocks.depth_inter)
        writer.ue(blocks.depth_intra)
        writer.flag(0)                          # scaling_list_enabled_flag
        writer.flag(tools.amp)
        writer.flag(tools.sample_adaptive_offset)
        writer.flag(0)                          # pcm_enabled_flag
        # Set k is the k + 1 pictures before the current one, each one count
        # nearer than the last and each predicted from. A picture names the
        # set as long as the pictures there have been since its IDR.
        writer.ue(self.references)              # num_short_term_ref_pic_sets
        for index in range(self.references):
            if index:
                writer.flag(0)                  # inter_ref_pic_set_prediction_flag
            writer.ue(index + 1)                # num_negative_pics
            writer.ue(0)                        # num_positive_pics
            for _ in range(index + 1):
                writer.ue(0)                    # delta_poc_s0_minus1
                writer.flag(1)                  # used_by_curr_pic_s0_flag
        writer.flag(0)                          # long_term_ref_pics_present_flag
        writer.flag(0)                          # sps_temporal_mvp_enabled_flag
        writer.flag(tools.strong_intra_smoothing)
        writer.flag(1)                          # vui_parameters_present_flag
        self._write_vui(writer)
        writer.flag(0)                          # sps_extension_present_flag
        return nal_unit(writer.rbsp(), NAL_SPS)

    def _write_vui(self, writer: BitWriter) -> None:
        """vui_parameters() (E.2.1): square samples, colour and timing."""
        writer.flag(1)                          # aspect_ratio_info_present_flag
        writer.u(1, 8)                          # aspect_ratio_idc, square
        writer.flag(0)                          # overscan_info_present_flag
        writer.flag(1)                          # video_signal_type_present_flag
        writer.u(VIDEO_FORMAT_UNSPECIFIED, 3)
        writer.flag(0)                          # video_full_range_flag, limited
        writer.flag(1)                          # colour_description_present_flag
        writer.u(COLOUR_PRIMARIES_BT709, 8)
        writer.u(TRANSFER_BT709, 8)
        writer.u(MATRIX_BT709, 8)
        writer.flag(0)                          # chroma_loc_info_present_flag
        writer.flag(0)                          # neutral_chroma_indication_flag
        writer.flag(0)                          # field_seq_flag
        writer.flag(0)                          # frame_field_info_present_flag
        writer.flag(0)                          # default_display_window_flag
        writer.flag(1)                          # vui_timing_info_present_flag
        # One tick is one picture: HEVC has no field period in its time base.
        writer.u(self.frame_rate[1], 32)        # vui_num_units_in_tick
        writer.u(self.frame_rate[0], 32)        # vui_time_scale
        writer.flag(0)                          # vui_poc_proportional_to_timing_flag
        writer.flag(0)                          # vui_hrd_parameters_present_flag
        writer.flag(0)                          # bitstream_restriction_flag

    # ------------------------------------------------------------------ PPS

    def pps(self) -> bytes:
        """The picture parameter set (7.3.2.3), as a NAL unit."""
        tools = self.tools
        writer = BitWriter()
        writer.ue(0)                            # pps_pic_parameter_set_id
        writer.ue(0)                            # pps_seq_parameter_set_id
        writer.flag(0)                          # dependent_slice_segments_enabled_flag
        writer.flag(0)                          # output_flag_present_flag
        writer.u(0, 3)                          # num_extra_slice_header_bits
        writer.flag(tools.sign_data_hiding)
        writer.flag(0)                          # cabac_init_present_flag
        writer.ue(self.references - 1)          # num_ref_idx_l0_default_active_minus1
        writer.ue(self.references - 1 if self.bidirectional else 0)
        writer.se(self.init_qp - 26)            # init_qp_minus26
        writer.flag(tools.constrained_intra_pred)
        writer.flag(tools.transform_skip)
        writer.flag(tools.cu_qp_delta)
        if tools.cu_qp_delta:
            writer.ue(0)                        # diff_cu_qp_delta_depth
        writer.se(0)                            # pps_cb_qp_offset
        writer.se(0)                            # pps_cr_qp_offset
        writer.flag(0)                          # pps_slice_chroma_qp_offsets_present_flag
        writer.flag(0)                          # weighted_pred_flag
        writer.flag(0)                          # weighted_bipred_flag
        writer.flag(tools.transquant_bypass)
        writer.flag(0)                          # tiles_enabled_flag
        writer.flag(0)                          # entropy_coding_sync_enabled_flag
        writer.flag(0)                          # pps_loop_filter_across_slices_enabled_flag
        writer.flag(0)                          # deblocking_filter_control_present_flag
        writer.flag(0)                          # pps_scaling_list_data_present_flag
        writer.flag(0)                          # lists_modification_present_flag
        writer.ue(0)                            # log2_parallel_merge_level_minus2
        writer.flag(0)                          # slice_segment_header_extension_present_flag
        writer.flag(0)                          # pps_extension_present_flag
        return nal_unit(writer.rbsp(), NAL_PPS)

    def headers(self) -> bytes:
        """The three sets as one Annex-B fragment, in the order a stream carries them."""
        return annexb(self.vps(), self.sps(), self.pps())

    # ------------------------------------------------- slice segment headers

    def slice_header(self, picture: Picture, qp: int | None = None) -> bytes:
        """The NAL unit header and slice segment header of `picture` (7.3.6.1).

        The header of the one slice segment each picture is coded as, through
        its byte_alignment(). The slice data follows it in a stream.

        qp -- the quantiser the slice is coded at, 0 to 51, where it is not
            the one the picture parameter set starts from

        Raises :class:`ValueError` for a quantiser outside that range, and
        for a picture the sequence's reference picture sets do not describe:
        an IDR picture with references, another with none, or one with more
        than the buffer keeps.
        """
        slice_qp = self.init_qp if qp is None else int(qp)
        if not 0 <= slice_qp <= MAX_QP:
            raise ValueError(f'{qp} is outside the HEVC quantiser range 0..51')
        if picture.idr and picture.references:
            raise ValueError(
                f'an IDR picture predicting from {len(picture.references)} pictures')
        if not picture.idr and not 1 <= len(picture.references) <= self.references:
            raise ValueError(
                f'picture {picture.poc} predicts from {len(picture.references)} '
                f'pictures, and the sequence has sets of 1 to {self.references}')
        writer = BitWriter()
        writer.flag(1)                          # first_slice_segment_in_pic_flag
        if picture.idr:
            writer.flag(0)                      # no_output_of_prior_pics_flag
        writer.ue(0)                            # slice_pic_parameter_set_id
        writer.ue(picture.slice_type)
        count = len(picture.references)
        if not picture.idr:
            writer.u(picture.poc % (1 << self.log2_max_poc_lsb), self.log2_max_poc_lsb)
            writer.flag(1)                      # short_term_ref_pic_set_sps_flag
            writer.u(count - 1, _ceil_log2(self.references))    # short_term_ref_pic_set_idx
        if self.tools.sample_adaptive_offset:
            writer.flag(1)                      # slice_sao_luma_flag
            writer.flag(1)                      # slice_sao_chroma_flag
        if not picture.intra:
            override = count != self.references
            writer.flag(override)               # num_ref_idx_active_override_flag
            if override:
                writer.ue(count - 1)            # num_ref_idx_l0_active_minus1
                if picture.bidirectional:
                    writer.ue(count - 1)        # num_ref_idx_l1_active_minus1
            if picture.bidirectional:
                writer.flag(0)                  # mvd_l1_zero_flag
            writer.ue(5 - MAX_NUM_MERGE_CAND)   # five_minus_max_num_merge_cand
        writer.se(slice_qp - self.init_qp)      # slice_qp_delta
        # byte_alignment() is a one and then zeroes, as the end of an RBSP is.
        return nal_unit(writer.rbsp(), picture.nal_unit_type)

    # ---------------------------------------------------------- packed forms

    def packed_sequence(self) -> tuple[bytes, int]:
        """The video and sequence parameter sets as one packed header, and its bits."""
        data = annexb(self.vps(), self.sps())
        return data, len(data) * 8

    def packed_picture(self) -> tuple[bytes, int]:
        """The picture parameter set as a packed header, and its length in bits."""
        data = annexb(self.pps())
        return data, len(data) * 8

    def packed_slice_header(self, picture: Picture,
                            qp: int | None = None) -> tuple[bytes, int]:
        """The slice segment header as a packed header, and its length in bits.

        A slice segment header ends on a byte, so every bit of the data is
        syntax.
        """
        data = annexb(self.slice_header(picture, qp))
        return data, len(data) * 8
