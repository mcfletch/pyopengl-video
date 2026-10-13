"""HEVC decoding's control layer: the buffers libva takes, and the picture rules.

:mod:`~pyopengl_video.vaapi.hevc_parse` reads the syntax and
:mod:`~pyopengl_video.vaapi.hevc_dpb` keeps the decoded picture buffer.
The functions here turn both into the parameter buffers libva takes, and
:class:`HEVCDecoding` gathers each picture from its slice segments and decides
where it is decoded and when it is shown. Nothing here holds a driver.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pyopengl_video import codecs
from pyopengl_video.decoder import DecoderError, Sample
from pyopengl_video.mp4 import split_annexb
from pyopengl_video.vaapi import api, hevc_api
from pyopengl_video.vaapi.control import DecodeControl, DecodeSession, Format
from pyopengl_video.vaapi.hevc_dpb import DecodedPictureBuffer, Frame, Picture
from pyopengl_video.vaapi.hevc_parse import (
    DEFAULT_SCALING_LISTS,
    FLAT_SCALING_LISTS,
    NAL_END_OF_BITSTREAM,
    NAL_END_OF_SEQUENCE,
    NAL_PPS,
    NAL_SPS,
    SLICE_B,
    SLICE_I,
    NalUnit,
    PictureParameterSet,
    ScalingLists,
    SequenceParameterSet,
    SliceHeader,
    colour_of,
    parse_nal,
    parse_pps,
    parse_slice_header,
    parse_sps,
)

__all__ = ['DIAGONAL_4X4', 'DIAGONAL_8X8', 'HEVCDecoding', 'colour_of', 'format_for',
           'iq_matrix',
           'picture_parameters', 'profile_for', 'scaling_lists_for',
           'slice_parameters']

#: The sample depths decoded, and the libva profile each runs under.
VA_PROFILES = {8: api.VAProfileHEVCMain, 10: api.VAProfileHEVCMain10}

#: chroma_format_idc in words, for refusing a stream.
CHROMA_FORMATS = ('monochrome (4:0:0)', '4:2:0', '4:2:2', '4:4:4')


def _diagonal(size: int) -> tuple[int, ...]:
    """The up-right diagonal scan of a `size` x `size` block (6.5.3), as
    raster indices: each diagonal runs from its lower left to its upper right."""
    order = []
    for diagonal in range(2 * size - 1):
        for column in range(max(0, diagonal - size + 1), min(diagonal, size - 1) + 1):
            order.append((diagonal - column) * size + column)
    return tuple(order)


#: Scan position to raster index, for the 4x4 lists and for the 8x8 grid the
#: larger lists are coded on.
DIAGONAL_4X4 = _diagonal(4)
DIAGONAL_8X8 = _diagonal(8)


def profile_for(sps: SequenceParameterSet, pps: PictureParameterSet) -> int:
    """The libva profile a stream decodes under; refuses what is not decoded here."""
    name = sps.profile_tier_level.name
    for flag, extension in (
        (sps.sps_multilayer_extension_flag or pps.pps_multilayer_extension_flag,
         'a multilayer extension'),
        (sps.sps_3d_extension_flag or pps.pps_3d_extension_flag, 'a 3D extension'),
        (sps.sps_scc_extension_flag or pps.pps_scc_extension_flag,
         'a screen content coding extension'),
    ):
        if flag:
            raise DecoderError(
                f'the stream is {name} profile with {extension}, and VA-API '
                'decoding here covers Main and Main 10')
    if sps.chroma_format_idc != 1 or sps.separate_colour_plane_flag:
        raise DecoderError(
            f'the stream is {name} profile at {CHROMA_FORMATS[sps.chroma_format_idc]}, '
            'and this decoder takes 4:2:0')
    if sps.bit_depth_luma_minus8 != sps.bit_depth_chroma_minus8:
        raise DecoderError(
            f'the stream has {sps.bit_depth} bits a luma sample and '
            f'{8 + sps.bit_depth_chroma_minus8} a chroma sample, and this decoder '
            'takes both at 8 or both at 10')
    if sps.bit_depth not in VA_PROFILES:
        raise DecoderError(
            f'the stream is {name} profile at {sps.bit_depth} bits a sample, and '
            'this decoder takes 8 or 10')
    tools = list(sps.range_extension)
    tools += [tool for tool in ('cross_component_prediction_enabled_flag',
                                'chroma_qp_offset_list_enabled_flag',
                                'log2_max_transform_skip_block_size_minus2',
                                'log2_sao_offset_scale_luma',
                                'log2_sao_offset_scale_chroma')
              if getattr(pps, tool)]
    if tools:
        raise DecoderError(
            f'the stream is {name} profile and uses the range extensions '
            f'({", ".join(tools)}), which the Main and Main 10 decoders here do '
            'not have')
    return VA_PROFILES[sps.bit_depth]


def format_for(sps: SequenceParameterSet, pps: PictureParameterSet) -> Format:
    """The surfaces a sequence is decoded into; refuses what is not decoded here.

    The surfaces are the coded size exactly: radeonsi reconstructs an HEVC
    picture at the size of the context it decodes in, whatever the picture
    parameter buffer says, so a larger surface is decoded as a larger picture.
    """
    return Format(profile=profile_for(sps, pps), coded_size=sps.coded_size,
                  surfaces=sps.dpb_size + 1, depth=sps.bit_depth,
                  name=f'HEVC {"Main 10" if sps.bit_depth == 10 else "Main"}',
                  exact_size=True)


def va_picture(frame: Frame, flags: int = 0) -> hevc_api.VAPictureHEVC:
    """A reference frame as libva names it, with its reference picture set bits."""
    if frame.long_term:
        flags |= hevc_api.VA_PICTURE_HEVC_LONG_TERM_REFERENCE
    return hevc_api.VAPictureHEVC(picture_id=frame.surface, pic_order_cnt=frame.poc,
                                  flags=flags)


def picture_parameters(picture: Picture, headers: Sequence[SliceHeader],
                       surface: int) -> hevc_api.VAPictureParameterBufferHEVC:
    """The picture parameter buffer for `picture`, decoded into `surface`.

    headers -- the header of each of the picture's slice segments
    """
    header = headers[0]
    sps, pps = header.sps, header.pps
    parameters = hevc_api.VAPictureParameterBufferHEVC()
    current = parameters.CurrPic
    current.picture_id = surface
    current.pic_order_cnt = picture.poc
    current.flags = 0
    rps = picture.rps
    references = rps.frames
    if len(references) > hevc_api.REFERENCE_FRAMES:
        raise DecoderError(
            f'picture {picture.poc} keeps {len(references)} reference pictures, '
            f'and libva describes {hevc_api.REFERENCE_FRAMES}')
    for index in range(hevc_api.REFERENCE_FRAMES):
        if index >= len(references):
            parameters.ReferenceFrames[index] = hevc_api.VAPictureHEVC.unused()
            continue
        frame = references[index]
        flags = 0
        if frame in rps.st_curr_before:
            flags = hevc_api.VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE
        elif frame in rps.st_curr_after:
            flags = hevc_api.VA_PICTURE_HEVC_RPS_ST_CURR_AFTER
        elif frame in rps.lt_curr:
            flags = hevc_api.VA_PICTURE_HEVC_RPS_LT_CURR
        parameters.ReferenceFrames[index] = va_picture(frame, flags)
    parameters.pic_width_in_luma_samples = sps.pic_width_in_luma_samples
    parameters.pic_height_in_luma_samples = sps.pic_height_in_luma_samples
    bits = parameters.pic_fields.bits
    bits.chroma_format_idc = sps.chroma_format_idc
    bits.separate_colour_plane_flag = sps.separate_colour_plane_flag
    bits.pcm_enabled_flag = sps.pcm_enabled_flag
    bits.scaling_list_enabled_flag = sps.scaling_list_enabled_flag
    bits.transform_skip_enabled_flag = pps.transform_skip_enabled_flag
    bits.amp_enabled_flag = sps.amp_enabled_flag
    bits.strong_intra_smoothing_enabled_flag = sps.strong_intra_smoothing_enabled_flag
    bits.sign_data_hiding_enabled_flag = pps.sign_data_hiding_enabled_flag
    bits.constrained_intra_pred_flag = pps.constrained_intra_pred_flag
    bits.cu_qp_delta_enabled_flag = pps.cu_qp_delta_enabled_flag
    bits.weighted_pred_flag = pps.weighted_pred_flag
    bits.weighted_bipred_flag = pps.weighted_bipred_flag
    bits.transquant_bypass_enabled_flag = pps.transquant_bypass_enabled_flag
    bits.tiles_enabled_flag = pps.tiles_enabled_flag
    bits.entropy_coding_sync_enabled_flag = pps.entropy_coding_sync_enabled_flag
    bits.pps_loop_filter_across_slices_enabled_flag = (
        pps.pps_loop_filter_across_slices_enabled_flag)
    bits.loop_filter_across_tiles_enabled_flag = pps.loop_filter_across_tiles_enabled_flag
    bits.pcm_loop_filter_disabled_flag = sps.pcm_loop_filter_disabled_flag
    bits.NoPicReorderingFlag = int(sps.reorder_pics == 0)
    bits.NoBiPredFlag = int(all(entry.slice_type != SLICE_B for entry in headers))
    parameters.sps_max_dec_pic_buffering_minus1 = sps.sps_max_dec_pic_buffering_minus1[-1]
    parameters.bit_depth_luma_minus8 = sps.bit_depth_luma_minus8
    parameters.bit_depth_chroma_minus8 = sps.bit_depth_chroma_minus8
    parameters.pcm_sample_bit_depth_luma_minus1 = sps.pcm_sample_bit_depth_luma_minus1
    parameters.pcm_sample_bit_depth_chroma_minus1 = sps.pcm_sample_bit_depth_chroma_minus1
    parameters.log2_min_luma_coding_block_size_minus3 = (
        sps.log2_min_luma_coding_block_size_minus3)
    parameters.log2_diff_max_min_luma_coding_block_size = (
        sps.log2_diff_max_min_luma_coding_block_size)
    parameters.log2_min_transform_block_size_minus2 = (
        sps.log2_min_luma_transform_block_size_minus2)
    parameters.log2_diff_max_min_transform_block_size = (
        sps.log2_diff_max_min_luma_transform_block_size)
    parameters.log2_min_pcm_luma_coding_block_size_minus3 = (
        sps.log2_min_pcm_luma_coding_block_size_minus3)
    parameters.log2_diff_max_min_pcm_luma_coding_block_size = (
        sps.log2_diff_max_min_pcm_luma_coding_block_size)
    parameters.max_transform_hierarchy_depth_intra = sps.max_transform_hierarchy_depth_intra
    parameters.max_transform_hierarchy_depth_inter = sps.max_transform_hierarchy_depth_inter
    parameters.init_qp_minus26 = pps.init_qp_minus26
    parameters.diff_cu_qp_delta_depth = pps.diff_cu_qp_delta_depth
    parameters.pps_cb_qp_offset = pps.pps_cb_qp_offset
    parameters.pps_cr_qp_offset = pps.pps_cr_qp_offset
    parameters.log2_parallel_merge_level_minus2 = pps.log2_parallel_merge_level_minus2
    parameters.num_tile_columns_minus1 = pps.num_tile_columns_minus1
    parameters.num_tile_rows_minus1 = pps.num_tile_rows_minus1
    # The arrays hold every column and row but the twentieth and twenty-second,
    # which a grid that large leaves to be worked out from the picture's size.
    for target, sizes in ((parameters.column_width_minus1, pps.tile_columns(sps)),
                          (parameters.row_height_minus1, pps.tile_rows(sps))):
        for index, size in enumerate(sizes[:len(target)]):
            target[index] = size - 1
    parsing = parameters.slice_parsing_fields.bits
    parsing.lists_modification_present_flag = pps.lists_modification_present_flag
    parsing.long_term_ref_pics_present_flag = sps.long_term_ref_pics_present_flag
    parsing.sps_temporal_mvp_enabled_flag = sps.sps_temporal_mvp_enabled_flag
    parsing.cabac_init_present_flag = pps.cabac_init_present_flag
    parsing.output_flag_present_flag = pps.output_flag_present_flag
    parsing.dependent_slice_segments_enabled_flag = pps.dependent_slice_segments_enabled_flag
    parsing.pps_slice_chroma_qp_offsets_present_flag = (
        pps.pps_slice_chroma_qp_offsets_present_flag)
    parsing.sample_adaptive_offset_enabled_flag = sps.sample_adaptive_offset_enabled_flag
    parsing.deblocking_filter_override_enabled_flag = (
        pps.deblocking_filter_override_enabled_flag)
    parsing.pps_disable_deblocking_filter_flag = pps.pps_deblocking_filter_disabled_flag
    parsing.slice_segment_header_extension_present_flag = (
        pps.slice_segment_header_extension_present_flag)
    parsing.RapPicFlag = int(header.irap)
    parsing.IdrPicFlag = int(header.idr)
    parsing.IntraPicFlag = int(all(entry.slice_type == SLICE_I for entry in headers))
    parameters.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_pic_order_cnt_lsb_minus4
    parameters.num_short_term_ref_pic_sets = len(sps.short_term_ref_pic_sets)
    parameters.num_long_term_ref_pic_sps = len(sps.lt_ref_pic_poc_lsb_sps)
    parameters.num_ref_idx_l0_default_active_minus1 = pps.num_ref_idx_l0_default_active_minus1
    parameters.num_ref_idx_l1_default_active_minus1 = pps.num_ref_idx_l1_default_active_minus1
    parameters.pps_beta_offset_div2 = pps.pps_beta_offset_div2
    parameters.pps_tc_offset_div2 = pps.pps_tc_offset_div2
    parameters.num_extra_slice_header_bits = pps.num_extra_slice_header_bits
    parameters.st_rps_bits = header.st_rps_bits
    return parameters


def scaling_lists_for(sps: SequenceParameterSet, pps: PictureParameterSet) -> ScalingLists:
    """The scaling lists in force for a picture (7.4.3.2.1, 7.4.3.3.1).

    The picture parameter set's where it codes any, then the sequence's, then
    the defaults; every factor 16 where the sequence does not enable lists.
    """
    if not sps.scaling_list_enabled_flag:
        return FLAT_SCALING_LISTS
    return pps.scaling_lists or sps.scaling_lists or DEFAULT_SCALING_LISTS


def iq_matrix(sps: SequenceParameterSet,
              pps: PictureParameterSet) -> hevc_api.VAIQMatrixBufferHEVC:
    """The scaling lists in force, in the raster order libva takes them in."""
    chosen = scaling_lists_for(sps, pps)
    matrix = hevc_api.VAIQMatrixBufferHEVC()
    for target, lists in zip((matrix.ScalingList4x4, matrix.ScalingList8x8,
                              matrix.ScalingList16x16, matrix.ScalingList32x32),
                             chosen.lists, strict=True):
        for index, values in enumerate(lists):
            scan = DIAGONAL_4X4 if len(values) == len(DIAGONAL_4X4) else DIAGONAL_8X8
            for position, raster in enumerate(scan):
                target[index][raster] = values[position]
    matrix.ScalingListDC16x16[:] = chosen.dc[0]
    matrix.ScalingListDC32x32[:] = chosen.dc[1]
    return matrix


def _reference_list(entries: Sequence[Frame], references: Sequence[Frame],
                    target: Any) -> None:
    for index in range(hevc_api.REFERENCE_FRAMES):
        target[index] = (references.index(entries[index]) if index < len(entries)
                         else hevc_api.NO_REFERENCE)


def slice_parameters(header: SliceHeader, list0: Sequence[Frame], list1: Sequence[Frame],
                     references: Sequence[Frame], size: int, *, last: bool,
                     entry_offset: int = 0) -> hevc_api.VASliceParameterBufferHEVC:
    """The slice parameter buffer for one slice segment of `size` bytes, NAL
    header included.

    references -- the frames of the picture's ``ReferenceFrames``, in its
        order, which the two lists are given as indices into
    last -- this is the picture's last slice segment
    entry_offset -- how many entry points the picture's earlier slice
        segments have
    """
    parameters = hevc_api.VASliceParameterBufferHEVC()
    parameters.slice_data_size = size
    parameters.slice_data_offset = 0
    parameters.slice_data_flag = api.VA_SLICE_DATA_FLAG_ALL
    parameters.slice_data_byte_offset = header.header_bytes
    parameters.slice_segment_address = header.slice_segment_address
    _reference_list(list0, references, parameters.RefPicList[0])
    _reference_list(list1, references, parameters.RefPicList[1])
    flags = parameters.LongSliceFlags.fields
    flags.LastSliceOfPic = int(last)
    flags.dependent_slice_segment_flag = header.dependent_slice_segment_flag
    flags.slice_type = header.slice_type
    flags.color_plane_id = header.colour_plane_id
    flags.slice_sao_luma_flag = header.slice_sao_luma_flag
    flags.slice_sao_chroma_flag = header.slice_sao_chroma_flag
    flags.mvd_l1_zero_flag = header.mvd_l1_zero_flag
    flags.cabac_init_flag = header.cabac_init_flag
    flags.slice_temporal_mvp_enabled_flag = header.slice_temporal_mvp_enabled_flag
    flags.slice_deblocking_filter_disabled_flag = header.slice_deblocking_filter_disabled_flag
    flags.collocated_from_l0_flag = header.collocated_from_l0_flag
    flags.slice_loop_filter_across_slices_enabled_flag = (
        header.slice_loop_filter_across_slices_enabled_flag)
    parameters.collocated_ref_idx = (hevc_api.NO_REFERENCE if header.collocated_ref_idx is None
                                     else header.collocated_ref_idx)
    parameters.num_ref_idx_l0_active_minus1 = header.num_ref_idx_l0_active_minus1
    parameters.num_ref_idx_l1_active_minus1 = header.num_ref_idx_l1_active_minus1
    parameters.slice_qp_delta = header.slice_qp_delta
    parameters.slice_cb_qp_offset = header.slice_cb_qp_offset
    parameters.slice_cr_qp_offset = header.slice_cr_qp_offset
    parameters.slice_beta_offset_div2 = header.slice_beta_offset_div2
    parameters.slice_tc_offset_div2 = header.slice_tc_offset_div2
    table = header.pred_weight_table
    if table is not None:
        parameters.luma_log2_weight_denom = table.luma_log2_weight_denom
        parameters.delta_chroma_log2_weight_denom = table.delta_chroma_log2_weight_denom
        for target, values in (
            (parameters.delta_luma_weight_l0, table.delta_luma_weight_l0),
            (parameters.luma_offset_l0, table.luma_offset_l0),
            (parameters.delta_luma_weight_l1, table.delta_luma_weight_l1),
            (parameters.luma_offset_l1, table.luma_offset_l1),
        ):
            target[:len(values)] = values
        for pairs, chroma in (
            (parameters.delta_chroma_weight_l0, table.delta_chroma_weight_l0),
            (parameters.ChromaOffsetL0, table.chroma_offset_l0),
            (parameters.delta_chroma_weight_l1, table.delta_chroma_weight_l1),
            (parameters.ChromaOffsetL1, table.chroma_offset_l1),
        ):
            for index, pair in enumerate(chroma):
                pairs[index][:] = pair
    parameters.five_minus_max_num_merge_cand = header.five_minus_max_num_merge_cand
    parameters.num_entry_point_offsets = header.num_entry_point_offsets
    parameters.entry_offset_to_subset_array = entry_offset
    parameters.slice_data_num_emu_prevn_bytes = header.header_escapes
    return parameters


# ------------------------------------------------------------ the layer


class _Current:
    """The picture being gathered from its slice segments."""

    def __init__(self, picture: Picture, surface: int | None, sample: Sample) -> None:
        self.picture = picture
        self.surface = surface
        self.sample = sample
        self.slices: list[tuple[SliceHeader, NalUnit]] = []
        #: The last slice segment that is not a dependent one.
        self.independent: SliceHeader | None = None


class HEVCDecoding(DecodeControl):
    """HEVC decoding: Main and Main 10, 4:2:0, at 8 and 10 bits.

    Each sample is one access unit, as Annex-B: a picture and the units before
    it. Pictures are shown in the order their picture order counts give.
    After a :meth:`reset`, decoding resumes at the next intra random access
    point, and the leading pictures that predict from before it are dropped.
    """

    codec = codecs.HEVC

    def __init__(self, session: DecodeSession, output_slots: int) -> None:
        super().__init__(session, output_slots)
        self._sequences: dict[int, SequenceParameterSet] = {}
        self._pictures: dict[int, PictureParameterSet] = {}
        self._buffer = DecodedPictureBuffer()
        self._format: Format | None = None
        self._current: _Current | None = None
        self._described = False

    def configure(self, headers: bytes) -> None:
        for data in split_annexb(headers):
            self._parameter_set(parse_nal(data))

    def _parameter_set(self, unit: NalUnit) -> None:
        if unit.nal_unit_type == NAL_SPS:
            sps = parse_sps(unit.rbsp)
            self._sequences[sps.sps_seq_parameter_set_id] = sps
            if not self._described:
                self._describe(sps)
        elif unit.nal_unit_type == NAL_PPS:
            pps = parse_pps(unit.rbsp)
            self._pictures[pps.pps_pic_parameter_set_id] = pps

    def _describe(self, sps: SequenceParameterSet) -> None:
        self.size = sps.size
        self.colour = colour_of(sps)

    def decode(self, sample: Sample) -> None:
        for data in split_annexb(sample.data):
            unit = parse_nal(data)
            if unit.nuh_layer_id:
                continue                        # a layer above the base one
            kind = unit.nal_unit_type
            if kind in (NAL_SPS, NAL_PPS):
                self._parameter_set(unit)
            elif unit.is_slice:
                self._slice(unit, sample)
            elif kind in (NAL_END_OF_SEQUENCE, NAL_END_OF_BITSTREAM):
                self._finish()
                self._show(self._buffer.end_of_sequence())
        self._finish()

    def _slice(self, unit: NalUnit, sample: Sample) -> None:
        """Add one slice segment to its picture, starting the picture at its first."""
        current = self._current
        header = parse_slice_header(unit, self._sequences, self._pictures,
                                    current.independent if current else None)
        if header.first_slice_segment_in_pic_flag:
            self._finish()
            current = self._begin(header, sample)
        if current is None:
            return                              # the picture's first slice never arrived
        current.slices.append((header, unit))
        if not header.dependent_slice_segment_flag:
            current.independent = header

    def _begin(self, header: SliceHeader, sample: Sample) -> _Current:
        """Start a picture: activate its sequence, order it, choose its surface."""
        self._activate(header)
        picture = self._buffer.begin(header)
        # What the buffer outputs here gives up its surface, so it is shown
        # before a surface is chosen for this picture.
        self._show(self._buffer.outputs())
        surface = None if picture.skipped else self.session.acquire()
        self._current = _Current(picture, surface, sample)
        return self._current

    def _activate(self, header: SliceHeader) -> None:
        """Make sure the surfaces suit the sequence this picture belongs to.

        A sequence of another size or depth begins at an intra random access
        point. What the old one still holds is shown first.
        """
        sps = header.sps
        wanted = format_for(sps, header.pps)
        held = self._format
        if (held is None or held.surfaces < wanted.surfaces
                or (held.profile, held.coded_size, held.depth)
                != (wanted.profile, wanted.coded_size, wanted.depth)):
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
        headers = [header for header, _ in current.slices]
        references = picture.rps.frames
        first = headers[0]
        batches: list[list[tuple[int, Any]]] = [[
            (api.VAPictureParameterBufferType,
             picture_parameters(picture, headers, current.surface)),
            (api.VAIQMatrixBufferType, iq_matrix(first.sps, first.pps)),
        ]]
        entry_offset = 0
        list0: list[Frame] = []
        list1: list[Frame] = []
        for index, (header, unit) in enumerate(current.slices):
            if not header.dependent_slice_segment_flag:
                # A dependent slice segment predicts with its slice's lists.
                list0, list1 = self._buffer.reference_lists(picture, header)
            batches.append([
                (api.VASliceParameterBufferType,
                 slice_parameters(header, list0, list1, references, len(unit.data),
                                  last=index == len(current.slices) - 1,
                                  entry_offset=entry_offset)),
                (api.VASliceDataBufferType, unit.data)])
            entry_offset += header.num_entry_point_offsets
        self.session.submit(current.surface, batches, f'picture {picture.poc}')
        self._show(self._buffer.finish(
            picture, surface=current.surface,
            payload=(current.sample.timestamp, current.sample.duration,
                     first.sps.visible, self.colour)))

    def _show(self, frames: Sequence[Frame]) -> None:
        """Give the session each frame, as its own sequence described it."""
        for frame in frames:
            timestamp, duration, visible, colour = frame.payload
            self.session.output(frame.surface, timestamp=timestamp, duration=duration,
                                visible=visible, colour=colour)

    def flush(self) -> None:
        self._finish()
        self._show(self._buffer.flush())

    def reset(self) -> None:
        self._current = None
        self._buffer.reset()

    def held(self) -> set[int]:
        found = {frame.surface for frame in self._buffer.frames}
        if self._current is not None and self._current.surface is not None:
            found.add(self._current.surface)
        return found
