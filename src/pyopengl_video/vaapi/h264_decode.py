"""H.264 decoding's control layer: the buffers libva takes, and the picture rules.

:mod:`~pyopengl_video.vaapi.h264_parse` reads the syntax and
:mod:`~pyopengl_video.vaapi.h264_dpb` keeps the decoded picture buffer.
The functions here turn both into the parameter buffers libva takes, and
:class:`H264Decoding` gathers each picture from its slices and decides where it
is decoded and when it is shown. Nothing here holds a driver.

Every field of a buffer is assigned with :func:`store`, and every element of
an array in one with :func:`store_each`. ctypes keeps the low bits of an
integer wider than its field, and these read each value back and refuse one
that did not fit, so a driver is never handed another number than the one
parsed.
"""
from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import Any

from pyopengl_video import codecs
from pyopengl_video.decoder import DecoderError, Sample
from pyopengl_video.mp4 import split_annexb
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.control import DecodeControl, DecodeSession, Format
from pyopengl_video.vaapi.h264 import StreamError
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
           'profile_for', 'slice_parameters', 'store', 'store_each']

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


def store(target: ctypes.Structure, **values: int) -> None:
    """Assign each of `values` to the field of `target` it names, and read it back.

    ctypes keeps the low bits of an integer wider than its field and raises
    nothing. A field that does not read back as the value assigned raises
    :class:`DecoderError` naming the field, so that no parsed value reaches a
    driver as another number.
    """
    for name, value in values.items():
        setattr(target, name, value)
        if getattr(target, name) != value:
            raise DecoderError(
                f'{name} is {value}, which does not fit the field '
                f'{type(target).__name__} has for it')


def store_each(target: Any, values: Sequence[int], name: str) -> None:
    """Assign `values` to the first elements of the ctypes array `target`,
    each read back as :func:`store` reads a field back.

    name -- what the array holds, for the :class:`DecoderError`
    """
    if len(values) > len(target):
        raise DecoderError(
            f'{len(values)} values for {name}, and libva has room for {len(target)}')
    for index, value in enumerate(values):
        target[index] = value
        if target[index] != value:
            raise DecoderError(
                f'{name} {index} is {value}, which does not fit the array '
                'libva has for it')


def va_picture(frame: Frame) -> api.VAPictureH264:
    """A reference frame as libva names it."""
    long_term = frame.long_term
    picture = api.VAPictureH264()
    store(picture, picture_id=frame.surface,
          frame_idx=frame.long_term_frame_idx if long_term else frame.frame_num,
          flags=(api.VA_PICTURE_H264_LONG_TERM_REFERENCE if long_term
                 else api.VA_PICTURE_H264_SHORT_TERM_REFERENCE),
          TopFieldOrderCnt=frame.top_field_order_cnt,
          BottomFieldOrderCnt=frame.bottom_field_order_cnt)
    return picture


def picture_parameters(picture: Picture, references: Sequence[Frame],
                       surface: int) -> api.VAPictureParameterBufferH264:
    """The picture parameter buffer for `picture`, decoded into `surface`.

    references -- the reference frames in the decoded picture buffer; frames
        inferred from a gap in frame_num have no surface and are left out
    """
    header = picture.header
    sps, pps = header.sps, header.pps
    parameters = api.VAPictureParameterBufferH264()
    store(parameters.CurrPic, picture_id=surface, frame_idx=header.frame_num, flags=0,
          TopFieldOrderCnt=picture.top_field_order_cnt,
          BottomFieldOrderCnt=picture.bottom_field_order_cnt)
    decoded = [frame for frame in references if frame.surface is not None]
    slots = len(parameters.ReferenceFrames)
    if len(decoded) > slots:
        raise DecoderError(
            f'the picture with frame_num {header.frame_num} keeps {len(decoded)} '
            f'reference frames, and libva describes {slots}')
    for index in range(slots):
        parameters.ReferenceFrames[index] = (
            va_picture(decoded[index]) if index < len(decoded)
            else api.VAPictureH264.unused())
    store(parameters,
          picture_width_in_mbs_minus1=sps.pic_width_in_mbs_minus1,
          picture_height_in_mbs_minus1=sps.pic_height_in_map_units_minus1,
          bit_depth_luma_minus8=sps.bit_depth_luma_minus8,
          bit_depth_chroma_minus8=sps.bit_depth_chroma_minus8,
          num_ref_frames=sps.max_num_ref_frames,
          pic_init_qp_minus26=pps.pic_init_qp_minus26,
          pic_init_qs_minus26=pps.pic_init_qs_minus26,
          chroma_qp_index_offset=pps.chroma_qp_index_offset,
          second_chroma_qp_index_offset=pps.second_chroma_qp_index_offset,
          frame_num=header.frame_num)
    store(parameters.seq_fields.bits,
          chroma_format_idc=sps.chroma_format_idc,
          residual_colour_transform_flag=sps.separate_colour_plane_flag,
          gaps_in_frame_num_value_allowed_flag=sps.gaps_in_frame_num_value_allowed_flag,
          frame_mbs_only_flag=sps.frame_mbs_only_flag,
          mb_adaptive_frame_field_flag=sps.mb_adaptive_frame_field_flag,
          direct_8x8_inference_flag=sps.direct_8x8_inference_flag,
          # A.3.3.2: from level 3.1 a bi-predicted block is no smaller than 8x8.
          MinLumaBiPredSize8x8=int(sps.level_idc >= 31),
          log2_max_frame_num_minus4=sps.log2_max_frame_num_minus4,
          pic_order_cnt_type=sps.pic_order_cnt_type,
          log2_max_pic_order_cnt_lsb_minus4=sps.log2_max_pic_order_cnt_lsb_minus4,
          delta_pic_order_always_zero_flag=sps.delta_pic_order_always_zero_flag)
    store(parameters.pic_fields.bits,
          entropy_coding_mode_flag=pps.entropy_coding_mode_flag,
          weighted_pred_flag=pps.weighted_pred_flag,
          weighted_bipred_idc=pps.weighted_bipred_idc,
          transform_8x8_mode_flag=pps.transform_8x8_mode_flag,
          field_pic_flag=header.field_pic_flag,
          constrained_intra_pred_flag=pps.constrained_intra_pred_flag,
          pic_order_present_flag=pps.bottom_field_pic_order_in_frame_present_flag,
          deblocking_filter_control_present_flag=(
              pps.deblocking_filter_control_present_flag),
          redundant_pic_cnt_present_flag=pps.redundant_pic_cnt_present_flag,
          reference_pic_flag=int(header.nal_ref_idc != 0))
    return parameters


def iq_matrix(sps: SequenceParameterSet,
              pps: PictureParameterSet) -> api.VAIQMatrixBufferH264:
    """The scaling lists in force, in the raster order libva takes them in."""
    fours, eights = scaling_matrices(sps, pps)
    matrix = api.VAIQMatrixBufferH264()
    for index, values in enumerate(fours):
        store_each(matrix.ScalingList4x4[index],
                   _raster(values, ZIGZAG_4X4), '4x4 scaling factor')
    for index, values in enumerate(eights[:2]):
        store_each(matrix.ScalingList8x8[index],
                   _raster(values, ZIGZAG_8X8), '8x8 scaling factor')
    return matrix


def _raster(values: Sequence[int], scan: Sequence[int]) -> list[int]:
    """`values`, given in the order of `scan`, in raster order."""
    raster = [0] * len(scan)
    for position, index in enumerate(scan):
        raster[index] = values[position]
    return raster


def _reference_list(entries: Sequence[Frame], target: Any) -> None:
    """One RefPicList: a picture for each entry, and the rest marked unused."""
    if len(entries) > len(target):
        raise DecoderError(
            f'a reference list of {len(entries)} entries, and libva describes '
            f'{len(target)}')
    for index in range(len(target)):
        target[index] = (va_picture(entries[index]) if index < len(entries)
                         else api.VAPictureH264.unused())


def slice_parameters(header: SliceHeader, list0: Sequence[Frame],
                     list1: Sequence[Frame],
                     size: int) -> api.VASliceParameterBufferH264:
    """The slice parameter buffer for one slice of `size` bytes, NAL header included.

    list0, list1 -- the slice's reference lists, a decoded frame in every
        entry, as :meth:`DecodedPictureBuffer.reference_lists` gives them
    """
    parameters = api.VASliceParameterBufferH264()
    if header.header_bits >= 8 * size:
        raise DecoderError(
            f'a slice header of {header.header_bits} bits in a slice of {size} bytes')
    store(parameters,
          slice_data_size=size, slice_data_offset=0,
          slice_data_flag=api.VA_SLICE_DATA_FLAG_ALL,
          slice_data_bit_offset=header.header_bits,
          first_mb_in_slice=header.first_mb_in_slice,
          slice_type=header.slice_type,
          direct_spatial_mv_pred_flag=header.direct_spatial_mv_pred_flag,
          num_ref_idx_l0_active_minus1=header.num_ref_idx_l0_active_minus1,
          num_ref_idx_l1_active_minus1=header.num_ref_idx_l1_active_minus1,
          cabac_init_idc=header.cabac_init_idc,
          slice_qp_delta=header.slice_qp_delta,
          disable_deblocking_filter_idc=header.disable_deblocking_filter_idc,
          slice_alpha_c0_offset_div2=header.slice_alpha_c0_offset_div2,
          slice_beta_offset_div2=header.slice_beta_offset_div2)
    _reference_list(list0, parameters.RefPicList0)
    _reference_list(list1, parameters.RefPicList1)
    table = header.pred_weight_table
    if table is not None:
        store(parameters,
              luma_log2_weight_denom=table.luma_log2_weight_denom,
              chroma_log2_weight_denom=table.chroma_log2_weight_denom,
              luma_weight_l0_flag=int(table.luma_weight_l0_flag),
              chroma_weight_l0_flag=int(table.chroma_weight_l0_flag),
              luma_weight_l1_flag=int(table.luma_weight_l1_flag),
              chroma_weight_l1_flag=int(table.chroma_weight_l1_flag))
        for luma, chroma, weights, offsets, chroma_weights, chroma_offsets in (
            (table.luma_l0, table.chroma_l0, parameters.luma_weight_l0,
             parameters.luma_offset_l0, parameters.chroma_weight_l0,
             parameters.chroma_offset_l0),
            (table.luma_l1, table.chroma_l1, parameters.luma_weight_l1,
             parameters.luma_offset_l1, parameters.chroma_weight_l1,
             parameters.chroma_offset_l1),
        ):
            store_each(weights, [weight for weight, _ in luma], 'luma weight')
            store_each(offsets, [offset for _, offset in luma], 'luma offset')
            if len(chroma) > len(chroma_weights):
                raise DecoderError(
                    f'chroma weights for {len(chroma)} references, and libva '
                    f'describes {len(chroma_weights)}')
            for index, pair in enumerate(chroma):
                store_each(chroma_weights[index], [weight for weight, _ in pair],
                           'chroma weight')
                store_each(chroma_offsets[index], [offset for _, offset in pair],
                           'chroma offset')
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
        self._described = False
        self._leading_before: int | None = None
        self._after_reset = False

    def configure(self, headers: bytes) -> None:
        for unit in split_annexb(headers):
            self._parameter_set(parse_nal(unit))

    def _parameter_set(self, unit: NalUnit) -> None:
        """Keep a parameter set under its id.

        A set sent again unchanged, as streams do at each key frame, leaves
        the one held in place, so the slices of one picture go on naming the
        same objects.
        """
        if unit.nal_unit_type == NAL_SPS:
            sps = parse_sps(unit.rbsp)
            if self._sequences.get(sps.seq_parameter_set_id) != sps:
                self._sequences[sps.seq_parameter_set_id] = sps
            if not self._described:
                self._describe(sps)
        elif unit.nal_unit_type == NAL_PPS:
            pps = parse_pps(unit.rbsp, self._sequences)
            if self._pictures.get(pps.pic_parameter_set_id) != pps:
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
                self._slice(unit, sample)
            elif kind in (NAL_END_OF_SEQUENCE, NAL_END_OF_STREAM):
                self._finish()
        self._finish()

    def _slice(self, unit: NalUnit, sample: Sample) -> None:
        """Add one slice to its picture, starting the picture at its first."""
        header = parse_slice_header(unit, self._sequences, self._pictures)
        if header.redundant_pic_cnt:
            return
        current = self._current
        if current is not None and starts_new_picture(current.slices[-1][0], header):
            self._finish()
            current = None
        if current is None:
            current = self._begin(header, sample)
        else:
            first = current.picture.header
            if header.sps is not first.sps or header.pps is not first.pps:
                # 7.4.1.2.1: a parameter set does not change within a picture.
                raise StreamError(
                    f'a slice of the picture with frame_num {header.frame_num} is '
                    'coded against other parameter sets than the picture\'s first '
                    'slice')
        current.slices.append((header, unit))

    def _begin(self, header: SliceHeader, sample: Sample) -> _Current:
        """Start a picture: activate its sequence, order it, choose its surface."""
        self._activate(header)
        picture = self._buffer.begin(header)
        # What the buffer outputs here gives up its surface, so it is shown
        # before a surface is chosen for this picture.
        self._show(self._buffer.outputs())
        if self._after_reset:
            self._after_reset = False
            self._leading_before = None if header.idr else picture.poc
        elif header.idr:
            self._leading_before = None
        # A leading picture of the group a seek landed in refers to pictures
        # before the seek point, which were never decoded: it has no surface.
        leading = self._leading_before is not None and picture.poc < self._leading_before
        self._current = _Current(picture, None if leading else self.session.acquire(),
                                 sample)
        return self._current

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
        self._described = True
        self._describe(sps)

    def _finish(self) -> None:
        """Submit the gathered picture and show what the buffer outputs."""
        current, self._current = self._current, None
        if current is None or current.surface is None:
            return
        picture = current.picture
        first = picture.header
        batches: list[list[tuple[int, Any]]] = [[
            (api.VAPictureParameterBufferType,
             picture_parameters(picture, self._buffer.reference_frames(), current.surface)),
            (api.VAIQMatrixBufferType, iq_matrix(first.sps, first.pps)),
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
            payload=(current.sample.timestamp, current.sample.duration,
                     first.sps.visible, self.colour)))

    def _show(self, frames: Sequence[Frame]) -> None:
        """Give the session each decoded frame, as its own sequence described it."""
        for frame in frames:
            if frame.surface is None:
                continue                        # inferred from a gap in frame_num
            timestamp, duration, visible, colour = frame.payload
            self.session.output(frame.surface, timestamp=timestamp, duration=duration,
                                visible=visible, colour=colour)

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
