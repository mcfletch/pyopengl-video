"""H.264 decoding's control layer: the buffers libva takes, and the picture rules.

:mod:`~pyopengl_video.vaapi.h264_parse` reads the syntax and
:mod:`~pyopengl_video.vaapi.h264_dpb` keeps the decoded picture buffer.
The functions here turn both into the parameter buffers libva takes, and
:class:`H264Decoding` gathers each picture from its slices and decides where it
is decoded and when it is shown. Nothing here holds a driver.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pyopengl_video import codecs
from pyopengl_video.decoder import DecoderError, Sample
from pyopengl_video.mp4 import split_annexb
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.control import DecodeControl, DecodeSession, Format
from pyopengl_video.vaapi.h264_dpb import DecodedPictureBuffer, Frame, Picture
from pyopengl_video.vaapi.h264_parse import (
    NAL_END_OF_SEQUENCE,
    NAL_END_OF_STREAM,
    NAL_IDR,
    NAL_PPS,
    NAL_SLICE,
    NAL_SPS,
    NalUnit,
    PictureParameterSet,
    SequenceParameterSet,
    SliceHeader,
    colour_of,
    parse_nal,
    parse_pps,
    parse_slice_header,
    parse_sps,
    scaling_matrices,
    starts_new_picture,
)

__all__ = ['H264Decoding', 'colour_of', 'iq_matrix', 'picture_parameters',
           'profile_for', 'slice_parameters']

#: profile_idc values by name, for refusing a stream in words a reader knows.
PROFILE_NAMES = {
    66: 'Baseline', 77: 'Main', 88: 'Extended', 100: 'High', 110: 'High 10',
    122: 'High 4:2:2', 244: 'High 4:4:4 Predictive', 44: 'CAVLC 4:4:4 Intra',
    118: 'Multiview High', 128: 'Stereo High',
}

#: The libva profile each decoded profile_idc runs under. Baseline is decoded
#: as Constrained Baseline, which is Baseline without slice groups and
#: redundant pictures; a stream that uses slice groups is refused.
VA_PROFILES = {
    66: api.VAProfileH264ConstrainedBaseline,
    77: api.VAProfileH264Main,
    100: api.VAProfileH264High,
}



def _zigzag(size: int) -> tuple[int, ...]:
    """The frame zig-zag scan of a `size` x `size` block, as raster indices."""
    order = []
    for diagonal in range(2 * size - 1):
        rows = range(max(0, diagonal - size + 1), min(diagonal, size - 1) + 1)
        for row in (rows if diagonal % 2 else reversed(rows)):
            order.append(row * size + diagonal - row)
    return tuple(order)


#: Tables 8-12 and 8-13's frame scans: scan position to raster index.
ZIGZAG_4X4 = _zigzag(4)
ZIGZAG_8X8 = _zigzag(8)


def profile_for(sps: SequenceParameterSet, pps: PictureParameterSet) -> int:
    """The libva profile a stream decodes under; refuses what is not decoded here."""
    if not sps.frame_mbs_only_flag:
        raise DecoderError(
            'the stream is interlaced: it codes pictures as fields or as field '
            'macroblock pairs, and this decoder takes progressive frames only')
    name = PROFILE_NAMES.get(sps.profile_idc, f'profile_idc {sps.profile_idc}')
    if sps.profile_idc not in VA_PROFILES:
        raise DecoderError(
            f'the stream is {name} profile, and VA-API decoding here covers '
            'Constrained Baseline, Main and High (8-bit 4:2:0)')
    if sps.chroma_format_idc != 1 or sps.bit_depth_luma_minus8 or sps.bit_depth_chroma_minus8:
        raise DecoderError(
            f'the stream is {name} profile at chroma format '
            f'{sps.chroma_format_idc} and {8 + sps.bit_depth_luma_minus8} bits, '
            'and this decoder takes 8-bit 4:2:0')
    if pps.num_slice_groups_minus1:
        raise DecoderError(
            'the stream uses slice groups (flexible macroblock ordering), which '
            'VA-API drivers do not decode')
    return VA_PROFILES[sps.profile_idc]


def va_picture(frame: Frame) -> api.VAPictureH264:
    """A reference frame as libva names it."""
    long_term = frame.long_term
    return api.VAPictureH264(
        picture_id=frame.surface,
        frame_idx=frame.long_term_frame_idx if long_term else frame.frame_num,
        flags=(api.VA_PICTURE_H264_LONG_TERM_REFERENCE if long_term
               else api.VA_PICTURE_H264_SHORT_TERM_REFERENCE),
        TopFieldOrderCnt=frame.top_field_order_cnt,
        BottomFieldOrderCnt=frame.bottom_field_order_cnt)


def picture_parameters(picture: Picture, references: Sequence[Frame],
                       surface: int) -> api.VAPictureParameterBufferH264:
    """The picture parameter buffer for `picture`, decoded into `surface`.

    references -- the reference frames in the decoded picture buffer; frames
        inferred from a gap in frame_num have no surface and are left out
    """
    header = picture.header
    sps, pps = header.sps, header.pps
    parameters = api.VAPictureParameterBufferH264()
    current = parameters.CurrPic
    current.picture_id = surface
    current.frame_idx = header.frame_num
    current.flags = 0
    current.TopFieldOrderCnt = picture.top_field_order_cnt
    current.BottomFieldOrderCnt = picture.bottom_field_order_cnt
    decoded = [frame for frame in references if frame.surface is not None]
    for index in range(16):
        parameters.ReferenceFrames[index] = (
            va_picture(decoded[index]) if index < len(decoded)
            else api.VAPictureH264.unused())
    parameters.picture_width_in_mbs_minus1 = sps.pic_width_in_mbs_minus1
    parameters.picture_height_in_mbs_minus1 = sps.pic_height_in_map_units_minus1
    parameters.bit_depth_luma_minus8 = sps.bit_depth_luma_minus8
    parameters.bit_depth_chroma_minus8 = sps.bit_depth_chroma_minus8
    parameters.num_ref_frames = sps.max_num_ref_frames
    bits = parameters.seq_fields.bits
    bits.chroma_format_idc = sps.chroma_format_idc
    bits.residual_colour_transform_flag = sps.separate_colour_plane_flag
    bits.gaps_in_frame_num_value_allowed_flag = sps.gaps_in_frame_num_value_allowed_flag
    bits.frame_mbs_only_flag = sps.frame_mbs_only_flag
    bits.mb_adaptive_frame_field_flag = sps.mb_adaptive_frame_field_flag
    bits.direct_8x8_inference_flag = sps.direct_8x8_inference_flag
    # A.3.3.2: from level 3.1 a bi-predicted block is no smaller than 8x8.
    bits.MinLumaBiPredSize8x8 = int(sps.level_idc >= 31)
    bits.log2_max_frame_num_minus4 = sps.log2_max_frame_num_minus4
    bits.pic_order_cnt_type = sps.pic_order_cnt_type
    bits.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_pic_order_cnt_lsb_minus4
    bits.delta_pic_order_always_zero_flag = sps.delta_pic_order_always_zero_flag
    parameters.pic_init_qp_minus26 = pps.pic_init_qp_minus26
    parameters.pic_init_qs_minus26 = pps.pic_init_qs_minus26
    parameters.chroma_qp_index_offset = pps.chroma_qp_index_offset
    parameters.second_chroma_qp_index_offset = pps.second_chroma_qp_index_offset
    flags = parameters.pic_fields.bits
    flags.entropy_coding_mode_flag = pps.entropy_coding_mode_flag
    flags.weighted_pred_flag = pps.weighted_pred_flag
    flags.weighted_bipred_idc = pps.weighted_bipred_idc
    flags.transform_8x8_mode_flag = pps.transform_8x8_mode_flag
    flags.field_pic_flag = header.field_pic_flag
    flags.constrained_intra_pred_flag = pps.constrained_intra_pred_flag
    flags.pic_order_present_flag = pps.bottom_field_pic_order_in_frame_present_flag
    flags.deblocking_filter_control_present_flag = pps.deblocking_filter_control_present_flag
    flags.redundant_pic_cnt_present_flag = pps.redundant_pic_cnt_present_flag
    flags.reference_pic_flag = int(header.nal_ref_idc != 0)
    parameters.frame_num = header.frame_num
    return parameters


def iq_matrix(sps: SequenceParameterSet,
              pps: PictureParameterSet) -> api.VAIQMatrixBufferH264:
    """The scaling lists in force, in the raster order libva takes them in."""
    fours, eights = scaling_matrices(sps, pps)
    matrix = api.VAIQMatrixBufferH264()
    for index, values in enumerate(fours):
        for position, raster in enumerate(ZIGZAG_4X4):
            matrix.ScalingList4x4[index][raster] = values[position]
    for index, values in enumerate(eights[:2]):
        for position, raster in enumerate(ZIGZAG_8X8):
            matrix.ScalingList8x8[index][raster] = values[position]
    return matrix


def _reference_list(entries: Sequence[Frame | None], target: Any) -> None:
    for index in range(32):
        frame = entries[index] if index < len(entries) else None
        target[index] = (va_picture(frame) if frame is not None and frame.surface is not None
                         else api.VAPictureH264.unused())


def slice_parameters(header: SliceHeader, list0: Sequence[Frame | None],
                     list1: Sequence[Frame | None],
                     size: int) -> api.VASliceParameterBufferH264:
    """The slice parameter buffer for one slice of `size` bytes, NAL header included."""
    parameters = api.VASliceParameterBufferH264()
    parameters.slice_data_size = size
    parameters.slice_data_offset = 0
    parameters.slice_data_flag = api.VA_SLICE_DATA_FLAG_ALL
    parameters.slice_data_bit_offset = header.header_bits
    parameters.first_mb_in_slice = header.first_mb_in_slice
    parameters.slice_type = header.slice_type
    parameters.direct_spatial_mv_pred_flag = header.direct_spatial_mv_pred_flag
    parameters.num_ref_idx_l0_active_minus1 = header.num_ref_idx_l0_active_minus1
    parameters.num_ref_idx_l1_active_minus1 = header.num_ref_idx_l1_active_minus1
    parameters.cabac_init_idc = header.cabac_init_idc
    parameters.slice_qp_delta = header.slice_qp_delta
    parameters.disable_deblocking_filter_idc = header.disable_deblocking_filter_idc
    parameters.slice_alpha_c0_offset_div2 = header.slice_alpha_c0_offset_div2
    parameters.slice_beta_offset_div2 = header.slice_beta_offset_div2
    _reference_list(list0, parameters.RefPicList0)
    _reference_list(list1, parameters.RefPicList1)
    table = header.pred_weight_table
    if table is not None:
        parameters.luma_log2_weight_denom = table.luma_log2_weight_denom
        parameters.chroma_log2_weight_denom = table.chroma_log2_weight_denom
        parameters.luma_weight_l0_flag = int(table.luma_weight_l0_flag)
        parameters.chroma_weight_l0_flag = int(table.chroma_weight_l0_flag)
        parameters.luma_weight_l1_flag = int(table.luma_weight_l1_flag)
        parameters.chroma_weight_l1_flag = int(table.chroma_weight_l1_flag)
        for luma, chroma, weights, offsets, chroma_weights, chroma_offsets in (
            (table.luma_l0, table.chroma_l0, parameters.luma_weight_l0,
             parameters.luma_offset_l0, parameters.chroma_weight_l0,
             parameters.chroma_offset_l0),
            (table.luma_l1, table.chroma_l1, parameters.luma_weight_l1,
             parameters.luma_offset_l1, parameters.chroma_weight_l1,
             parameters.chroma_offset_l1),
        ):
            for index, (weight, offset) in enumerate(luma):
                weights[index], offsets[index] = weight, offset
            for index, pair in enumerate(chroma):
                for component, (weight, offset) in enumerate(pair):
                    chroma_weights[index][component] = weight
                    chroma_offsets[index][component] = offset
    return parameters


# ------------------------------------------------------------ the layer


class _Current:
    """The picture being gathered from its slices."""

    def __init__(self, picture: Picture, surface: int | None, sample: Sample) -> None:
        self.picture = picture
        self.surface = surface
        self.sample = sample
        self.slices: list[tuple[SliceHeader, NalUnit]] = []


class H264Decoding(DecodeControl):
    """H.264 decoding: Constrained Baseline, Main and High, progressive, 8-bit 4:2:0.

    Each sample is one access unit, as Annex-B: a picture and the units before
    it. Pictures are shown in the order their picture order counts give.
    """

    codec = codecs.H264

    def __init__(self, session: DecodeSession, output_slots: int) -> None:
        super().__init__(session, output_slots)
        self._sequences: dict[int, SequenceParameterSet] = {}
        self._pictures: dict[int, PictureParameterSet] = {}
        self._buffer = DecodedPictureBuffer()
        self._format: Format | None = None
        self._current: _Current | None = None
        self._active: SequenceParameterSet | None = None
        self._leading_before: int | None = None
        self._after_reset = False

    def configure(self, headers: bytes) -> None:
        for unit in split_annexb(headers):
            self._parameter_set(parse_nal(unit))

    def _parameter_set(self, unit: NalUnit) -> None:
        if unit.nal_unit_type == NAL_SPS:
            sps = parse_sps(unit.rbsp)
            self._sequences[sps.seq_parameter_set_id] = sps
            if self._active is None:
                self._describe(sps)
        elif unit.nal_unit_type == NAL_PPS:
            pps = parse_pps(unit.rbsp, self._sequences)
            self._pictures[pps.pic_parameter_set_id] = pps

    def _describe(self, sps: SequenceParameterSet) -> None:
        self.size = sps.size
        self.colour = colour_of(sps)

    def decode(self, sample: Sample) -> None:
        for data in split_annexb(sample.data):
            unit = parse_nal(data)
            kind = unit.nal_unit_type
            if kind in (NAL_SPS, NAL_PPS):
                self._parameter_set(unit)
            elif kind in (NAL_SLICE, NAL_IDR):
                header = parse_slice_header(unit, self._sequences, self._pictures)
                if header.redundant_pic_cnt:
                    continue
                current = self._current
                if current is not None and starts_new_picture(current.slices[-1][0], header):
                    self._finish()
                if self._current is None:
                    self._begin(header, sample)
                assert self._current is not None
                self._current.slices.append((header, unit))
            elif kind in (NAL_END_OF_SEQUENCE, NAL_END_OF_STREAM):
                self._finish()
        self._finish()

    def _begin(self, header: SliceHeader, sample: Sample) -> None:
        """Start a picture: activate its sequence, order it, choose its surface."""
        self._activate(header)
        picture = self._buffer.begin(header)
        if self._after_reset:
            self._after_reset = False
            self._leading_before = None if header.idr else picture.poc
        elif header.idr:
            self._leading_before = None
        if self._leading_before is not None and picture.poc < self._leading_before:
            # A leading picture of the group a seek landed in: it refers to
            # pictures before the seek point, which were never decoded.
            self._current = _Current(picture, None, sample)
            return
        self._current = _Current(picture, self.session.acquire(), sample)

    def _activate(self, header: SliceHeader) -> None:
        """Make sure the surfaces suit the sequence this picture belongs to.

        A new sequence of another size or profile can begin only at an IDR.
        What the old one still holds is shown first.
        """
        sps = header.sps
        profile = profile_for(sps, header.pps)
        name = PROFILE_NAMES.get(sps.profile_idc, f'profile_idc {sps.profile_idc}')
        wanted = Format(profile=profile, coded_size=sps.coded_size,
                        surfaces=sps.dpb_frames + 1, name=f'H.264 {name}')
        held = self._format
        if (held is None or (held.profile, held.coded_size) != (profile, sps.coded_size)
                or held.surfaces < wanted.surfaces):
            if held is not None:
                if not header.no_output_of_prior_pics_flag:
                    self._show(self._buffer.flush())
                self._buffer.reset()
            self.session.activate(wanted)
            self._format = wanted
        self._active = sps
        self._describe(sps)

    def _finish(self) -> None:
        """Submit the gathered picture and show what the buffer outputs."""
        current, self._current = self._current, None
        if current is None or current.surface is None:
            return
        picture = current.picture
        batches: list[list[tuple[int, Any]]] = [[
            (api.VAPictureParameterBufferType,
             picture_parameters(picture, self._buffer.reference_frames(), current.surface)),
            (api.VAIQMatrixBufferType, iq_matrix(picture.header.sps, picture.header.pps)),
        ]]
        for header, unit in current.slices:
            list0, list1 = self._buffer.reference_lists(picture, header)
            batches.append([
                (api.VASliceParameterBufferType,
                 slice_parameters(header, list0, list1, len(unit.data))),
                (api.VASliceDataBufferType, unit.data)])
        self.session.submit(current.surface, batches, f'picture {picture.frame_num}')
        self._show(self._buffer.finish(
            picture, surface=current.surface,
            payload=(current.sample.timestamp, current.sample.duration)))

    def _show(self, frames: Sequence[Frame]) -> None:
        sps = self._active
        assert sps is not None
        for frame in frames:
            if frame.surface is None:
                continue
            timestamp, duration = frame.payload
            self.session.output(frame.surface, timestamp=timestamp, duration=duration,
                                visible=sps.visible, colour=self.colour)

    def flush(self) -> None:
        self._finish()
        self._show(self._buffer.flush())

    def reset(self) -> None:
        self._current = None
        self._buffer.reset()
        self._after_reset = True

    def held(self) -> set[int]:
        found = {frame.surface for frame in self._buffer.frames
                 if frame.surface is not None}
        if self._current is not None and self._current.surface is not None:
            found.add(self._current.surface)
        return found
