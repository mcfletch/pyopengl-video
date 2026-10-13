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
from pyopengl_video.vaapi.control import (
    DecodeControl,
    DecodeSession,
    Format,
    store,
    store_each,
)
from pyopengl_video.vaapi.h264 import StreamError
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
    picture = hevc_api.VAPictureHEVC()
    store(picture, picture_id=frame.surface, pic_order_cnt=frame.poc, flags=flags)
    return picture


def picture_parameters(picture: Picture, headers: Sequence[SliceHeader],
                       surface: int) -> hevc_api.VAPictureParameterBufferHEVC:
    """The picture parameter buffer for `picture`, decoded into `surface`.

    headers -- the header of each of the picture's slice segments
    """
    header = headers[0]
    sps, pps = header.sps, header.pps
    parameters = hevc_api.VAPictureParameterBufferHEVC()
    store(parameters.CurrPic, picture_id=surface, pic_order_cnt=picture.poc, flags=0)
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
    store(parameters.pic_fields.bits,
          chroma_format_idc=sps.chroma_format_idc,
          separate_colour_plane_flag=sps.separate_colour_plane_flag,
          pcm_enabled_flag=sps.pcm_enabled_flag,
          scaling_list_enabled_flag=sps.scaling_list_enabled_flag,
          transform_skip_enabled_flag=pps.transform_skip_enabled_flag,
          amp_enabled_flag=sps.amp_enabled_flag,
          strong_intra_smoothing_enabled_flag=sps.strong_intra_smoothing_enabled_flag,
          sign_data_hiding_enabled_flag=pps.sign_data_hiding_enabled_flag,
          constrained_intra_pred_flag=pps.constrained_intra_pred_flag,
          cu_qp_delta_enabled_flag=pps.cu_qp_delta_enabled_flag,
          weighted_pred_flag=pps.weighted_pred_flag,
          weighted_bipred_flag=pps.weighted_bipred_flag,
          transquant_bypass_enabled_flag=pps.transquant_bypass_enabled_flag,
          tiles_enabled_flag=pps.tiles_enabled_flag,
          entropy_coding_sync_enabled_flag=pps.entropy_coding_sync_enabled_flag,
          pps_loop_filter_across_slices_enabled_flag=(
              pps.pps_loop_filter_across_slices_enabled_flag),
          loop_filter_across_tiles_enabled_flag=pps.loop_filter_across_tiles_enabled_flag,
          pcm_loop_filter_disabled_flag=sps.pcm_loop_filter_disabled_flag,
          NoPicReorderingFlag=int(sps.reorder_pics == 0),
          NoBiPredFlag=int(all(entry.slice_type != SLICE_B for entry in headers)))
    store(parameters,
          pic_width_in_luma_samples=sps.pic_width_in_luma_samples,
          pic_height_in_luma_samples=sps.pic_height_in_luma_samples,
          sps_max_dec_pic_buffering_minus1=sps.sps_max_dec_pic_buffering_minus1[-1],
          bit_depth_luma_minus8=sps.bit_depth_luma_minus8,
          bit_depth_chroma_minus8=sps.bit_depth_chroma_minus8,
          pcm_sample_bit_depth_luma_minus1=sps.pcm_sample_bit_depth_luma_minus1,
          pcm_sample_bit_depth_chroma_minus1=sps.pcm_sample_bit_depth_chroma_minus1,
          log2_min_luma_coding_block_size_minus3=(
              sps.log2_min_luma_coding_block_size_minus3),
          log2_diff_max_min_luma_coding_block_size=(
              sps.log2_diff_max_min_luma_coding_block_size),
          log2_min_transform_block_size_minus2=(
              sps.log2_min_luma_transform_block_size_minus2),
          log2_diff_max_min_transform_block_size=(
              sps.log2_diff_max_min_luma_transform_block_size),
          log2_min_pcm_luma_coding_block_size_minus3=(
              sps.log2_min_pcm_luma_coding_block_size_minus3),
          log2_diff_max_min_pcm_luma_coding_block_size=(
              sps.log2_diff_max_min_pcm_luma_coding_block_size),
          max_transform_hierarchy_depth_intra=sps.max_transform_hierarchy_depth_intra,
          max_transform_hierarchy_depth_inter=sps.max_transform_hierarchy_depth_inter,
          init_qp_minus26=pps.init_qp_minus26,
          diff_cu_qp_delta_depth=pps.diff_cu_qp_delta_depth,
          pps_cb_qp_offset=pps.pps_cb_qp_offset,
          pps_cr_qp_offset=pps.pps_cr_qp_offset,
          log2_parallel_merge_level_minus2=pps.log2_parallel_merge_level_minus2,
          num_tile_columns_minus1=pps.num_tile_columns_minus1,
          num_tile_rows_minus1=pps.num_tile_rows_minus1,
          log2_max_pic_order_cnt_lsb_minus4=sps.log2_max_pic_order_cnt_lsb_minus4,
          num_short_term_ref_pic_sets=len(sps.short_term_ref_pic_sets),
          num_long_term_ref_pic_sps=len(sps.lt_ref_pic_poc_lsb_sps),
          num_ref_idx_l0_default_active_minus1=pps.num_ref_idx_l0_default_active_minus1,
          num_ref_idx_l1_default_active_minus1=pps.num_ref_idx_l1_default_active_minus1,
          pps_beta_offset_div2=pps.pps_beta_offset_div2,
          pps_tc_offset_div2=pps.pps_tc_offset_div2,
          num_extra_slice_header_bits=pps.num_extra_slice_header_bits,
          st_rps_bits=header.st_rps_bits)
    # The arrays hold every column and row but the twentieth and twenty-second,
    # which a grid that large leaves to be worked out from the picture's size.
    # Each tile has a coding tree block or more, so no size less one is negative.
    for target, sizes, name in (
            (parameters.column_width_minus1, pps.tile_columns(sps), 'tile column width'),
            (parameters.row_height_minus1, pps.tile_rows(sps), 'tile row height')):
        store_each(target, [size - 1 for size in sizes[:len(target)]], name)
    store(parameters.slice_parsing_fields.bits,
          lists_modification_present_flag=pps.lists_modification_present_flag,
          long_term_ref_pics_present_flag=sps.long_term_ref_pics_present_flag,
          sps_temporal_mvp_enabled_flag=sps.sps_temporal_mvp_enabled_flag,
          cabac_init_present_flag=pps.cabac_init_present_flag,
          output_flag_present_flag=pps.output_flag_present_flag,
          dependent_slice_segments_enabled_flag=(
              pps.dependent_slice_segments_enabled_flag),
          pps_slice_chroma_qp_offsets_present_flag=(
              pps.pps_slice_chroma_qp_offsets_present_flag),
          sample_adaptive_offset_enabled_flag=sps.sample_adaptive_offset_enabled_flag,
          deblocking_filter_override_enabled_flag=(
              pps.deblocking_filter_override_enabled_flag),
          pps_disable_deblocking_filter_flag=pps.pps_deblocking_filter_disabled_flag,
          slice_segment_header_extension_present_flag=(
              pps.slice_segment_header_extension_present_flag),
          RapPicFlag=int(header.irap),
          IdrPicFlag=int(header.idr),
          IntraPicFlag=int(all(entry.slice_type == SLICE_I for entry in headers)))
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
            raster = [0] * len(scan)
            for position, place in enumerate(scan):
                raster[place] = values[position]
            store_each(target[index], raster, 'scaling factor')
    store_each(matrix.ScalingListDC16x16, chosen.dc[0], '16x16 DC scaling factor')
    store_each(matrix.ScalingListDC32x32, chosen.dc[1], '32x32 DC scaling factor')
    return matrix


def _reference_list(entries: Sequence[Frame], references: Sequence[Frame],
                    target: Any) -> None:
    """One RefPicList: each entry's place in `references`, the rest naming none."""
    if len(entries) > len(target):
        raise DecoderError(
            f'a reference list of {len(entries)} entries, and libva describes '
            f'{len(target)}')
    for index in range(len(target)):
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

    Raises :class:`DecoderError` for a slice segment whose header leaves no
    slice data in those `size` bytes.
    """
    if header.header_bytes + header.header_escapes >= size:
        raise DecoderError(
            f'a slice segment header of {header.header_bytes} bytes and '
            f'{header.header_escapes} escapes in a slice segment of {size} bytes')
    parameters = hevc_api.VASliceParameterBufferHEVC()
    store(parameters,
          slice_data_size=size, slice_data_offset=0,
          slice_data_flag=api.VA_SLICE_DATA_FLAG_ALL,
          slice_data_byte_offset=header.header_bytes,
          slice_segment_address=header.slice_segment_address,
          collocated_ref_idx=(hevc_api.NO_REFERENCE if header.collocated_ref_idx is None
                              else header.collocated_ref_idx),
          num_ref_idx_l0_active_minus1=header.num_ref_idx_l0_active_minus1,
          num_ref_idx_l1_active_minus1=header.num_ref_idx_l1_active_minus1,
          slice_qp_delta=header.slice_qp_delta,
          slice_cb_qp_offset=header.slice_cb_qp_offset,
          slice_cr_qp_offset=header.slice_cr_qp_offset,
          slice_beta_offset_div2=header.slice_beta_offset_div2,
          slice_tc_offset_div2=header.slice_tc_offset_div2,
          five_minus_max_num_merge_cand=header.five_minus_max_num_merge_cand,
          num_entry_point_offsets=header.num_entry_point_offsets,
          entry_offset_to_subset_array=entry_offset,
          slice_data_num_emu_prevn_bytes=header.header_escapes)
    _reference_list(list0, references, parameters.RefPicList[0])
    _reference_list(list1, references, parameters.RefPicList[1])
    store(parameters.LongSliceFlags.fields,
          LastSliceOfPic=int(last),
          dependent_slice_segment_flag=header.dependent_slice_segment_flag,
          slice_type=header.slice_type,
          color_plane_id=header.colour_plane_id,
          slice_sao_luma_flag=header.slice_sao_luma_flag,
          slice_sao_chroma_flag=header.slice_sao_chroma_flag,
          mvd_l1_zero_flag=header.mvd_l1_zero_flag,
          cabac_init_flag=header.cabac_init_flag,
          slice_temporal_mvp_enabled_flag=header.slice_temporal_mvp_enabled_flag,
          slice_deblocking_filter_disabled_flag=(
              header.slice_deblocking_filter_disabled_flag),
          collocated_from_l0_flag=header.collocated_from_l0_flag,
          slice_loop_filter_across_slices_enabled_flag=(
              header.slice_loop_filter_across_slices_enabled_flag))
    table = header.pred_weight_table
    if table is not None:
        store(parameters,
              luma_log2_weight_denom=table.luma_log2_weight_denom,
              delta_chroma_log2_weight_denom=table.delta_chroma_log2_weight_denom)
        for target, values, name in (
            (parameters.delta_luma_weight_l0, table.delta_luma_weight_l0, 'luma weight'),
            (parameters.luma_offset_l0, table.luma_offset_l0, 'luma offset'),
            (parameters.delta_luma_weight_l1, table.delta_luma_weight_l1, 'luma weight'),
            (parameters.luma_offset_l1, table.luma_offset_l1, 'luma offset'),
        ):
            store_each(target, values, name)
        for pairs, chroma, name in (
            (parameters.delta_chroma_weight_l0, table.delta_chroma_weight_l0,
             'chroma weight'),
            (parameters.ChromaOffsetL0, table.chroma_offset_l0, 'chroma offset'),
            (parameters.delta_chroma_weight_l1, table.delta_chroma_weight_l1,
             'chroma weight'),
            (parameters.ChromaOffsetL1, table.chroma_offset_l1, 'chroma offset'),
        ):
            if len(chroma) > len(pairs):
                raise DecoderError(
                    f'{name}s for {len(chroma)} references, and libva describes '
                    f'{len(pairs)}')
            for index, pair in enumerate(chroma):
                store_each(pairs[index], pair, name)
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
        """Keep a parameter set under its id.

        A set sent again unchanged, as streams do at each key frame, leaves
        the one held in place, so the slice segments of one picture go on
        naming the same objects.
        """
        if unit.nal_unit_type == NAL_SPS:
            sps = parse_sps(unit.rbsp)
            if self._sequences.get(sps.sps_seq_parameter_set_id) != sps:
                self._sequences[sps.sps_seq_parameter_set_id] = sps
            if not self._described:
                self._describe(sps)
        elif unit.nal_unit_type == NAL_PPS:
            pps = parse_pps(unit.rbsp)
            if self._pictures.get(pps.pps_pic_parameter_set_id) != pps:
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
        first = current.picture.header
        if (header.slice_pic_parameter_set_id != first.slice_pic_parameter_set_id
                or header.sps is not first.sps or header.pps is not first.pps):
            # 7.4.7.1: every slice segment of a picture names one picture
            # parameter set, and the surfaces are those of its sequence.
            raise StreamError(
                f'the slice segment at coding tree block {header.slice_segment_address} '
                f'of picture {current.picture.poc} is coded against other parameter '
                'sets than the picture\'s first')
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
        # The picture is the current one before its surface is chosen, so
        # that held() names what it predicts from while the session chooses.
        # Without a surface it is gathered and never submitted.
        current = self._current = _Current(picture, None, sample)
        if not picture.skipped:
            current.surface = self.session.acquire()
        return current

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
        current = self._current
        if current is None or current.surface is None:
            self._current = None
            return
        picture = current.picture
        headers = [header for header, _ in current.slices]
        references = picture.rps.frames
        first = headers[0]
        try:
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
        finally:
            # The picture is the current one, and its references held, until
            # it has been handed over or refused.
            self._current = None
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
        """The surfaces in use: the buffer's, the one the picture being
        gathered is decoded into, and every one its reference picture set
        names."""
        found = {frame.surface for frame in self._buffer.frames}
        current = self._current
        if current is not None:
            found.update(frame.surface for frame in current.picture.rps.frames)
            if current.surface is not None:
                found.add(current.surface)
        return found
