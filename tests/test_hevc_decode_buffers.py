"""The libva HEVC decode buffers and the control layer, with no driver.

A field the driver is told wrongly is not reported: the picture decodes into
the wrong thing. Each buffer is built here from headers whose values are known
and read back field by field, and :class:`HEVCDecoding` is run over the
fixtures against a session that records what it is asked.
"""
import dataclasses
import io
import json
import random
from pathlib import Path

import pytest

from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.decoder import DecoderError, Sample
from pyopengl_video.mp4 import split_annexb
from pyopengl_video.vaapi import api, hevc_api
from pyopengl_video.vaapi.control import Format
from pyopengl_video.vaapi.decoder import _Pool
from pyopengl_video.vaapi.h264 import StreamError, annexb
from pyopengl_video.vaapi.hevc_decode import (
    DIAGONAL_4X4,
    DIAGONAL_8X8,
    HEVCDecoding,
    colour_of,
    format_for,
    iq_matrix,
    picture_parameters,
    profile_for,
    slice_parameters,
)
from pyopengl_video.vaapi.hevc_parse import (
    DEFAULT_8X8_INTER,
    DEFAULT_8X8_INTRA,
    NAL_CRA,
    NAL_END_OF_SEQUENCE,
    NAL_IDR_W_RADL,
    NAL_TRAIL_N,
)
from tests.hevc_syntax import (
    SCALING,
    Stream,
    coded,
    header,
    nal,
    parameter_units,
    picture_set,
    pps_bytes,
    sequence,
    slice_rbsp,
    sps_bytes,
)

DATA = Path(__file__).parent / 'data' / 'decode'
FIXTURES = json.loads((DATA / 'hevc.json').read_text())['streams']

INVALID = hevc_api.VA_PICTURE_HEVC_INVALID
BEFORE = hevc_api.VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE
AFTER = hevc_api.VA_PICTURE_HEVC_RPS_ST_CURR_AFTER
LONG = hevc_api.VA_PICTURE_HEVC_LONG_TERM_REFERENCE
LONG_CURRENT = hevc_api.VA_PICTURE_HEVC_RPS_LT_CURR


def pyramid(pps=None, **sps):
    """Pictures 0, 16, 8 and 4 decoded into surfaces of those numbers plus
    100, with 0 held as a long-term picture from 8 on."""
    sps.setdefault('long_term', ())
    stream = Stream(pps=pps, **sps)
    stream.idr(100)
    stream.decode(116, 16, before=(-16,))
    stream.decode(108, 8, after=(8,), kind='B', long_term={'pics': [(0, 1, None)]})
    stream.decode(104, 4, before=(), after=(4, 12), kind='B',
                  long_term={'pics': [(0, 1, None)]})
    return stream


class TestPictureParameters:
    def build(self, pps=None, slices=None, **sps):
        stream = pyramid(pps=pps, **sps)
        options = slices or [{'kind': 'B'}]
        headers = [stream.header(6, before=(-2,), after=(2, 10), unused=(10,),
                                 long_term={'pics': [(0, 1, None)]}, **entry)
                   for entry in options]
        picture = stream.buffer.begin(headers[0])
        return stream, picture, picture_parameters(picture, headers, 7)

    def test_the_picture_being_decoded(self):
        _, _, parameters = self.build()
        current = parameters.CurrPic
        assert (current.picture_id, current.pic_order_cnt, current.flags) == (7, 6, 0)

    def test_the_references_with_their_sets(self):
        _, _, parameters = self.build()
        found = [(entry.picture_id, entry.pic_order_cnt, entry.flags)
                 for entry in parameters.ReferenceFrames if not entry.flags & INVALID]
        assert found == [(104, 4, BEFORE), (108, 8, AFTER), (100, 0, LONG | LONG_CURRENT),
                         (116, 16, 0)]

    def test_the_slots_holding_no_picture(self):
        _, _, parameters = self.build()
        unused = list(parameters.ReferenceFrames)[4:]
        assert len(unused) == 11
        assert all(entry.picture_id == api.VA_INVALID_SURFACE and entry.flags == INVALID
                   for entry in unused)

    def test_a_picture_standing_in_twice_is_listed_once(self):
        stream = pyramid()
        slice_header = stream.header(6, before=(-2, -3), after=(2,))
        picture = stream.buffer.begin(slice_header)
        assert picture.rps.missing == (3,)
        parameters = picture_parameters(picture, [slice_header], 7)
        found = [entry.picture_id for entry in parameters.ReferenceFrames
                 if not entry.flags & INVALID]
        assert found == [104, 108]

    def test_more_references_than_libva_describes_are_refused(self):
        stream = Stream(ordering=((16, 0, 0),))
        stream.idr(0)
        for poc in range(1, 17):
            stream.decode(poc, poc, before=tuple(range(-1, -poc - 1, -1)))
        slice_header = stream.header(17, before=tuple(range(-1, -17, -1)))
        picture = stream.buffer.begin(slice_header)
        with pytest.raises(DecoderError, match='16 reference pictures'):
            picture_parameters(picture, [slice_header], 99)

    def test_the_sequence_fields(self):
        _, _, parameters = self.build(
            width=160, height=96, diff_cb=2, amp=1, sao=1, temporal_mvp=1,
            strong_smoothing=1, pcm=(7, 5, 0, 1, 1), scaling='default',
            ordering=((5, 2, 0),), st_rps=[{'s0': [(0, 1)]}, {'s0': [(1, 1)]}])
        assert parameters.pic_width_in_luma_samples == 160
        assert parameters.pic_height_in_luma_samples == 96
        bits = parameters.pic_fields.bits
        assert bits.chroma_format_idc == 1
        assert bits.separate_colour_plane_flag == 0
        assert bits.pcm_enabled_flag == 1
        assert bits.scaling_list_enabled_flag == 1
        assert bits.amp_enabled_flag == 1
        assert bits.strong_intra_smoothing_enabled_flag == 1
        assert bits.pcm_loop_filter_disabled_flag == 1
        assert parameters.sps_max_dec_pic_buffering_minus1 == 5
        assert (parameters.bit_depth_luma_minus8, parameters.bit_depth_chroma_minus8) == (0, 0)
        assert parameters.pcm_sample_bit_depth_luma_minus1 == 7
        assert parameters.pcm_sample_bit_depth_chroma_minus1 == 5
        assert parameters.log2_min_luma_coding_block_size_minus3 == 0
        assert parameters.log2_diff_max_min_luma_coding_block_size == 2
        assert parameters.log2_min_transform_block_size_minus2 == 0
        assert parameters.log2_diff_max_min_transform_block_size == 2
        assert parameters.log2_min_pcm_luma_coding_block_size_minus3 == 0
        assert parameters.log2_diff_max_min_pcm_luma_coding_block_size == 1
        assert parameters.max_transform_hierarchy_depth_inter == 1
        assert parameters.max_transform_hierarchy_depth_intra == 2
        assert parameters.log2_max_pic_order_cnt_lsb_minus4 == 4
        assert parameters.num_short_term_ref_pic_sets == 2
        assert parameters.num_long_term_ref_pic_sps == 0
        parsing = parameters.slice_parsing_fields.bits
        assert parsing.long_term_ref_pics_present_flag == 1
        assert parsing.sps_temporal_mvp_enabled_flag == 1
        assert parsing.sample_adaptive_offset_enabled_flag == 1

    def test_ten_bits(self):
        _, _, parameters = self.build(bit_depth=2, profile_idc=2)
        assert (parameters.bit_depth_luma_minus8, parameters.bit_depth_chroma_minus8) == (2, 2)

    def test_the_picture_set_fields(self):
        _, _, parameters = self.build(pps={
            'dependent_slices': 1, 'output_flag_present': 1, 'extra_bits': 2,
            'sign_hiding': 1, 'cabac_init_present': 1, 'refs': (2, 1), 'init_qp': -5,
            'constrained_intra': 1, 'transform_skip': 1, 'cu_qp_delta_depth': 2,
            'chroma_offsets': (3, -4), 'slice_chroma_offsets_present': 1,
            'weighted_pred': 1, 'weighted_bipred': 1, 'transquant_bypass': 1, 'wpp': 1,
            'loop_filter_across_slices': 1, 'lists_modification': 1, 'parallel_merge': 2,
            'deblocking': {'override': 1, 'beta': 2, 'tc': -3}, 'header_extension': 1})
        bits = parameters.pic_fields.bits
        assert bits.transform_skip_enabled_flag == 1
        assert bits.sign_data_hiding_enabled_flag == 1
        assert bits.constrained_intra_pred_flag == 1
        assert bits.cu_qp_delta_enabled_flag == 1
        assert bits.weighted_pred_flag == 1
        assert bits.weighted_bipred_flag == 1
        assert bits.transquant_bypass_enabled_flag == 1
        assert bits.tiles_enabled_flag == 0
        assert bits.entropy_coding_sync_enabled_flag == 1
        assert bits.pps_loop_filter_across_slices_enabled_flag == 1
        assert bits.loop_filter_across_tiles_enabled_flag == 1
        assert parameters.init_qp_minus26 == -5
        assert parameters.diff_cu_qp_delta_depth == 2
        assert (parameters.pps_cb_qp_offset, parameters.pps_cr_qp_offset) == (3, -4)
        assert parameters.log2_parallel_merge_level_minus2 == 2
        assert parameters.num_ref_idx_l0_default_active_minus1 == 2
        assert parameters.num_ref_idx_l1_default_active_minus1 == 1
        assert (parameters.pps_beta_offset_div2, parameters.pps_tc_offset_div2) == (2, -3)
        assert parameters.num_extra_slice_header_bits == 2
        parsing = parameters.slice_parsing_fields.bits
        assert parsing.lists_modification_present_flag == 1
        assert parsing.cabac_init_present_flag == 1
        assert parsing.output_flag_present_flag == 1
        assert parsing.dependent_slice_segments_enabled_flag == 1
        assert parsing.pps_slice_chroma_qp_offsets_present_flag == 1
        assert parsing.deblocking_filter_override_enabled_flag == 1
        assert parsing.pps_disable_deblocking_filter_flag == 0
        assert parsing.slice_segment_header_extension_present_flag == 1

    def test_deblocking_switched_off_by_the_pps(self):
        _, _, parameters = self.build(pps={'deblocking': {'disabled': 1}})
        assert parameters.slice_parsing_fields.bits.pps_disable_deblocking_filter_flag == 1

    def test_one_tile_is_the_whole_picture(self):
        _, _, parameters = self.build(width=160, height=96, diff_cb=2)
        assert (parameters.num_tile_columns_minus1, parameters.num_tile_rows_minus1) == (0, 0)
        assert parameters.column_width_minus1[0] == 4
        assert parameters.row_height_minus1[0] == 2

    def test_a_uniform_tile_grid_is_worked_out(self):
        _, _, parameters = self.build(width=112, height=64,
                                      pps={'tiles': {'columns': 3, 'rows': 2}})
        assert parameters.pic_fields.bits.tiles_enabled_flag == 1
        assert (parameters.num_tile_columns_minus1, parameters.num_tile_rows_minus1) == (2, 1)
        assert list(parameters.column_width_minus1)[:4] == [1, 1, 2, 0]
        assert list(parameters.row_height_minus1)[:3] == [1, 1, 0]

    def test_a_stated_tile_grid(self):
        _, _, parameters = self.build(
            width=160, height=96, diff_cb=2,
            pps={'tiles': {'columns': 3, 'rows': 2, 'widths': (0, 1), 'heights': (1,),
                           'loop_filter': 0}})
        assert list(parameters.column_width_minus1)[:3] == [0, 1, 1]
        assert list(parameters.row_height_minus1)[:2] == [1, 0]
        assert parameters.pic_fields.bits.loop_filter_across_tiles_enabled_flag == 0

    def test_twenty_tile_columns_fill_the_array(self):
        _, _, parameters = self.build(width=320, height=64,
                                      pps={'tiles': {'columns': 20, 'rows': 1}})
        assert parameters.num_tile_columns_minus1 == 19
        assert list(parameters.column_width_minus1) == [0] * 19

    def test_a_trailing_picture_of_b_slices(self):
        _, _, parameters = self.build()
        parsing = parameters.slice_parsing_fields.bits
        assert (parsing.RapPicFlag, parsing.IdrPicFlag, parsing.IntraPicFlag) == (0, 0, 0)
        assert parameters.pic_fields.bits.NoBiPredFlag == 0

    def test_a_picture_with_no_b_slice(self):
        _, _, parameters = self.build(slices=[{'kind': 'P'}, {'kind': 'I', 'first': 0,
                                                              'address': 3}])
        assert parameters.pic_fields.bits.NoBiPredFlag == 1
        assert parameters.slice_parsing_fields.bits.IntraPicFlag == 0

    def test_one_b_slice_among_others(self):
        _, _, parameters = self.build(slices=[{'kind': 'I'}, {'kind': 'B', 'first': 0,
                                                              'address': 3}])
        assert parameters.pic_fields.bits.NoBiPredFlag == 0
        assert parameters.slice_parsing_fields.bits.IntraPicFlag == 0

    def test_random_access_pictures(self):
        stream = Stream()
        idr = stream.header(nal_type=NAL_IDR_W_RADL, kind='I')
        parsing = picture_parameters(stream.buffer.begin(idr), [idr], 0).slice_parsing_fields.bits
        assert (parsing.RapPicFlag, parsing.IdrPicFlag, parsing.IntraPicFlag) == (1, 1, 1)
        cra = stream.header(8, nal_type=NAL_CRA, kind='I')
        parsing = picture_parameters(stream.buffer.begin(cra), [cra], 0).slice_parsing_fields.bits
        assert (parsing.RapPicFlag, parsing.IdrPicFlag, parsing.IntraPicFlag) == (1, 0, 1)

    def test_reordering(self):
        _, _, parameters = self.build()
        assert parameters.pic_fields.bits.NoPicReorderingFlag == 0
        stream = Stream(ordering=((2, 0, 0),))
        idr = stream.header(nal_type=NAL_IDR_W_RADL, kind='I')
        parameters = picture_parameters(stream.buffer.begin(idr), [idr], 0)
        assert parameters.pic_fields.bits.NoPicReorderingFlag == 1

    def test_the_length_of_a_set_coded_in_the_header(self):
        stream, picture, parameters = self.build()
        assert parameters.st_rps_bits == picture.header.st_rps_bits > 0
        named = Stream(st_rps=[{'s0': [(0, 1)]}, {'s0': [(1, 1)]}])
        named.idr(0)
        slice_header = named.header(1, rps=0, kind='P')
        parameters = picture_parameters(named.buffer.begin(slice_header), [slice_header], 1)
        assert parameters.st_rps_bits == 0


#: Table 7-6 as the matrices it is the diagonal scan of.
INTRA_MATRIX = (
    (16, 16, 16, 16, 17, 18, 21, 24),
    (16, 16, 16, 16, 17, 19, 22, 25),
    (16, 16, 17, 18, 20, 22, 25, 29),
    (16, 16, 18, 21, 24, 27, 31, 36),
    (17, 17, 20, 24, 30, 35, 41, 47),
    (18, 19, 22, 27, 35, 44, 54, 65),
    (21, 22, 25, 31, 41, 54, 70, 88),
    (24, 25, 29, 36, 47, 65, 88, 115),
)
INTER_MATRIX = (
    (16, 16, 16, 16, 17, 18, 20, 24),
    (16, 16, 16, 17, 18, 20, 24, 25),
    (16, 16, 17, 18, 20, 24, 25, 28),
    (16, 17, 18, 20, 24, 25, 28, 33),
    (17, 18, 20, 24, 25, 28, 33, 41),
    (18, 20, 24, 25, 28, 33, 41, 54),
    (20, 24, 25, 28, 33, 41, 54, 71),
    (24, 25, 28, 33, 41, 54, 71, 91),
)


def raster(matrix):
    return [value for row in matrix for value in row]


class TestScalingLists:
    def test_the_scans_are_the_specifications(self):
        """6.5.3: each diagonal from its lower left to its upper right."""
        assert DIAGONAL_4X4 == (0, 4, 1, 8, 5, 2, 12, 9, 6, 3, 13, 10, 7, 14, 11, 15)
        assert DIAGONAL_8X8[:10] == (0, 8, 1, 16, 9, 2, 24, 17, 10, 3)
        assert sorted(DIAGONAL_8X8) == list(range(64))

    def test_flat_where_the_sequence_enables_none(self):
        matrix = iq_matrix(sequence(), picture_set(scaling=SCALING))
        for lists in (matrix.ScalingList4x4, matrix.ScalingList8x8,
                      matrix.ScalingList16x16, matrix.ScalingList32x32):
            assert all(value == 16 for row in lists for value in row)
        assert list(matrix.ScalingListDC16x16) == [16] * 6
        assert list(matrix.ScalingListDC32x32) == [16] * 2

    def test_the_defaults_in_raster_order(self):
        matrix = iq_matrix(sequence(scaling='default'), picture_set())
        assert all(value == 16 for row in matrix.ScalingList4x4 for value in row)
        for lists in (matrix.ScalingList8x8, matrix.ScalingList16x16):
            for index in range(6):
                assert list(lists[index]) == raster(INTRA_MATRIX if index < 3 else INTER_MATRIX)
        assert list(matrix.ScalingList32x32[0]) == raster(INTRA_MATRIX)
        assert list(matrix.ScalingList32x32[1]) == raster(INTER_MATRIX)
        assert list(matrix.ScalingListDC16x16) == [16] * 6

    def test_the_default_tables_are_the_scan_of_the_matrices(self):
        assert [raster(INTRA_MATRIX)[index] for index in DIAGONAL_8X8] == list(DEFAULT_8X8_INTRA)
        assert [raster(INTER_MATRIX)[index] for index in DIAGONAL_8X8] == list(DEFAULT_8X8_INTER)

    def test_the_sequences_lists_go_from_coded_to_raster_order(self):
        four = tuple(10 + index for index in range(16))
        eight = tuple(20 + index for index in range(64))
        matrix = iq_matrix(sequence(scaling=SCALING), picture_set())
        for position, index in enumerate(DIAGONAL_4X4):
            assert matrix.ScalingList4x4[0][index] == four[position]
            assert matrix.ScalingList4x4[3][index] == four[::-1][position]
        for position, index in enumerate(DIAGONAL_8X8):
            assert matrix.ScalingList8x8[0][index] == eight[position]
            assert matrix.ScalingList16x16[3][index] == eight[::-1][position]
            assert matrix.ScalingList32x32[1][index] == eight[position]
        assert list(matrix.ScalingListDC16x16) == [40, 40, 16, 50, 40, 40]
        assert list(matrix.ScalingListDC32x32) == [60, 60]

    def test_the_picture_sets_lists_replace_the_sequences(self):
        matrix = iq_matrix(sequence(scaling='default'), picture_set(scaling=SCALING))
        assert matrix.ScalingList8x8[0][0] == 20
        assert list(matrix.ScalingListDC32x32) == [60, 60]
        matrix = iq_matrix(sequence(scaling=SCALING), picture_set())
        assert matrix.ScalingList8x8[0][0] == 20


class TestSliceParameters:
    def build(self, pps=None, sps=None, size=500, last=True, entry_offset=0, **options):
        stream = pyramid(pps=pps, **(sps or {}))
        options.setdefault('kind', 'B')
        options.setdefault('refs', (2, 1))
        slice_header = stream.header(6, before=(-2,), after=(2, 10),
                                     long_term={'pics': [(0, 1, None)]}, **options)
        picture = stream.buffer.begin(slice_header)
        list0, list1 = stream.buffer.reference_lists(picture, slice_header)
        return slice_header, slice_parameters(
            slice_header, list0, list1, picture.rps.frames, size, last=last,
            entry_offset=entry_offset)

    def test_where_the_slice_data_is(self):
        slice_header, parameters = self.build()
        assert parameters.slice_data_size == 500
        assert parameters.slice_data_offset == 0
        assert parameters.slice_data_flag == api.VA_SLICE_DATA_FLAG_ALL
        assert parameters.slice_data_byte_offset == slice_header.header_bytes
        assert parameters.slice_data_num_emu_prevn_bytes == 0

    def test_escapes_in_the_header_are_stated(self):
        slice_header, parameters = self.build(pps={'header_extension': 1}, extension=bytes(7))
        assert parameters.slice_data_num_emu_prevn_bytes == slice_header.header_escapes > 0

    def test_the_lists_are_indices_into_the_reference_frames(self):
        """The references are 4, 8, 16, then the long-term 0."""
        _, parameters = self.build()
        assert list(parameters.RefPicList[0]) == [0, 1, 2] + [0xFF] * 12
        assert list(parameters.RefPicList[1]) == [1, 2] + [0xFF] * 13
        assert parameters.num_ref_idx_l0_active_minus1 == 2
        assert parameters.num_ref_idx_l1_active_minus1 == 1

    def test_a_list_reaching_the_long_term_picture(self):
        _, parameters = self.build(refs=(3, 3))
        assert list(parameters.RefPicList[0])[:4] == [0, 1, 2, 3]
        assert list(parameters.RefPicList[1])[:4] == [1, 2, 0, 3]

    def test_an_intra_slice_has_no_references(self):
        _, parameters = self.build(kind='I', refs=None)
        assert list(parameters.RefPicList[0]) == [0xFF] * 15
        assert list(parameters.RefPicList[1]) == [0xFF] * 15
        assert parameters.LongSliceFlags.fields.slice_type == 2
        assert parameters.collocated_ref_idx == 0xFF

    def test_the_flags(self):
        _, parameters = self.build(
            sps={'sao': 1, 'temporal_mvp': 1},
            pps={'cabac_init_present': 1, 'loop_filter_across_slices': 1,
                 'deblocking': {'override': 1}},
            temporal_mvp=1, sao=(1, 0), mvd_l1_zero=1, cabac_init=1, collocated=(0, 1),
            deblocking={'disabled': 1}, loop_filter_across_slices=0)
        flags = parameters.LongSliceFlags.fields
        assert flags.LastSliceOfPic == 1
        assert flags.dependent_slice_segment_flag == 0
        assert flags.slice_type == 0
        assert flags.color_plane_id == 0
        assert (flags.slice_sao_luma_flag, flags.slice_sao_chroma_flag) == (1, 0)
        assert flags.mvd_l1_zero_flag == 1
        assert flags.cabac_init_flag == 1
        assert flags.slice_temporal_mvp_enabled_flag == 1
        assert flags.slice_deblocking_filter_disabled_flag == 1
        assert flags.collocated_from_l0_flag == 0
        assert flags.slice_loop_filter_across_slices_enabled_flag == 0
        assert parameters.collocated_ref_idx == 1

    def test_flags_at_their_defaults(self):
        _, parameters = self.build(last=False)
        flags = parameters.LongSliceFlags.fields
        assert flags.LastSliceOfPic == 0
        assert flags.collocated_from_l0_flag == 1
        assert flags.slice_temporal_mvp_enabled_flag == 0
        assert parameters.collocated_ref_idx == 0xFF

    def test_the_quantiser_and_filter_fields(self):
        _, parameters = self.build(
            pps={'slice_chroma_offsets_present': 1,
                 'deblocking': {'override': 1, 'beta': 2, 'tc': -3}},
            qp_delta=-7, chroma_offsets=(1, -2), merge=3, address=0,
            deblocking={'beta': -1, 'tc': 4})
        assert parameters.slice_qp_delta == -7
        assert (parameters.slice_cb_qp_offset, parameters.slice_cr_qp_offset) == (1, -2)
        assert (parameters.slice_beta_offset_div2, parameters.slice_tc_offset_div2) == (-1, 4)
        assert parameters.five_minus_max_num_merge_cand == 3

    def test_the_pps_filter_offsets_stand_where_the_slice_states_none(self):
        _, parameters = self.build(pps={'deblocking': {'beta': 2, 'tc': -3}})
        assert (parameters.slice_beta_offset_div2, parameters.slice_tc_offset_div2) == (2, -3)

    def test_explicit_weights(self):
        _, parameters = self.build(pps={'weighted_bipred': 1}, weights={
            'denom': 5, 'chroma_delta': -1,
            'l0': [((3, -4), ((2, 5), (-6, 7))), (None, None), ((1, 1), None)],
            'l1': [(None, ((100, -300), (-100, 300))), ((-5, 6), None)]})
        assert parameters.luma_log2_weight_denom == 5
        assert parameters.delta_chroma_log2_weight_denom == -1
        assert list(parameters.delta_luma_weight_l0)[:4] == [3, 0, 1, 0]
        assert list(parameters.luma_offset_l0)[:4] == [-4, 0, 1, 0]
        assert [list(pair) for pair in parameters.delta_chroma_weight_l0][:2] == [[2, -6], [0, 0]]
        assert [list(pair) for pair in parameters.ChromaOffsetL0][:2] == [[-11, 55], [0, 0]]
        assert list(parameters.delta_luma_weight_l1)[:3] == [0, -5, 0]
        assert list(parameters.luma_offset_l1)[:3] == [0, 6, 0]
        assert list(parameters.delta_chroma_weight_l1[0]) == [100, -100]
        assert list(parameters.ChromaOffsetL1[0]) == [-128, 127]

    def test_without_a_table_the_weights_are_empty(self):
        _, parameters = self.build()
        assert parameters.luma_log2_weight_denom == 0
        assert list(parameters.delta_luma_weight_l0) == [0] * 15

    def test_entry_points_and_the_address(self):
        _, parameters = self.build(pps={'wpp': 1}, first=0, address=5,
                                   entry_points=(9, 300), entry_offset=4)
        assert parameters.slice_segment_address == 5
        assert parameters.num_entry_point_offsets == 2
        assert parameters.entry_offset_to_subset_array == 4

    def test_a_dependent_slice_segment_takes_its_slices_fields(self):
        stream = pyramid(pps={'dependent_slices': 1})
        independent = stream.header(6, before=(-2,), after=(2,), kind='B', qp_delta=5)
        dependent = header(stream.sps, stream.pps, independent=independent, first=0,
                           dependent=1, address=7)
        picture = stream.buffer.begin(independent)
        list0, list1 = stream.buffer.reference_lists(picture, independent)
        parameters = slice_parameters(dependent, list0, list1, picture.rps.frames, 40,
                                      last=True)
        flags = parameters.LongSliceFlags.fields
        assert flags.dependent_slice_segment_flag == 1
        assert flags.slice_type == 0
        assert parameters.slice_qp_delta == 5
        assert parameters.slice_segment_address == 7
        assert parameters.slice_data_byte_offset == dependent.header_bytes
        assert list(parameters.RefPicList[0])[:2] == [0, 0xFF]


class TestProfiles:
    def test_main_decodes_as_main(self):
        assert profile_for(sequence(), picture_set()) == api.VAProfileHEVCMain

    def test_ten_bits_decode_as_main_10(self):
        sps = sequence(bit_depth=2, profile_idc=2)
        assert profile_for(sps, picture_set()) == api.VAProfileHEVCMain10

    def test_the_format_is_the_coded_size_exactly(self):
        sps = sequence(width=120, height=72, conformance_window=(0, 0, 0, 2),
                       ordering=((4, 2, 0),), bit_depth=2, profile_idc=2)
        assert format_for(sps, picture_set()) == Format(
            profile=api.VAProfileHEVCMain10, coded_size=(120, 72), surfaces=6, depth=10,
            name='HEVC Main 10', exact_size=True)

    @pytest.mark.parametrize('options,message', [
        ({'chroma_format_idc': 2, 'profile_idc': 4}, r'Format Range Extensions profile at 4:2:2'),
        ({'chroma_format_idc': 3, 'profile_idc': 4}, '4:4:4'),
        ({'chroma_format_idc': 0, 'profile_idc': 4}, 'monochrome'),
        ({'bit_depth': 4, 'profile_idc': 4}, '12 bits a sample'),
        ({'bit_depth': 2, 'bit_depth_chroma': 0}, '10 bits a luma sample and 8'),
        ({'range_extension': (0, 0, 1, 0, 0, 0, 0, 1, 0), 'profile_idc': 4},
         'range extensions .implicit_rdpcm_enabled_flag, '
         'persistent_rice_adaptation_enabled_flag'),
        ({'extension_flags': (1, 0, 0), 'profile_idc': 7}, 'Scalable Main profile with a multilayer'),
        ({'extension_flags': (0, 1, 0), 'profile_idc': 8}, '3D extension'),
        ({'extension_flags': (0, 0, 1), 'profile_idc': 9}, 'screen content coding'),
    ])
    def test_sequences_not_decoded_are_refused_by_name(self, options, message):
        with pytest.raises(DecoderError, match=message):
            profile_for(sequence(**options), picture_set())

    @pytest.mark.parametrize('options,message', [
        ({'range_extension': {'cross_component': 1}}, 'cross_component_prediction_enabled_flag'),
        ({'range_extension': {'offset_list': [(1, -1)]}}, 'chroma_qp_offset_list_enabled_flag'),
        ({'range_extension': {'sao_scale': (1, 0)}}, 'log2_sao_offset_scale_luma'),
        ({'range_extension': {}, 'transform_skip': 1}, 'log2_max_transform_skip_block_size'),
        ({'extension_flags': (0, 0, 1)}, 'screen content coding'),
    ])
    def test_picture_sets_not_decoded_are_refused_by_name(self, options, message):
        with pytest.raises(DecoderError, match=message):
            profile_for(sequence(), picture_set(**options))

    def test_range_extensions_with_no_tool_switched_on_decode(self):
        sps = sequence(range_extension=(0,) * 9)
        pps = picture_set(range_extension={})
        assert profile_for(sps, pps) == api.VAProfileHEVCMain


class TestColour:
    def test_what_the_stream_states(self):
        colour = colour_of(sequence(vui={'colour': (9, 16, 9), 'full_range': 1,
                                         'chroma_loc': 2}))
        assert (colour.primaries, colour.transfer, colour.matrix) == (9, 16, 9)
        assert colour.full_range
        assert colour.chroma_location == 2

    def test_a_matrix_is_chosen_where_the_stream_states_none(self):
        assert colour_of(sequence()).matrix == 6
        assert colour_of(sequence(width=1280, height=720)).matrix == 1
        assert colour_of(sequence(vui={'timing': (1, 30)})).matrix == 6


class StandInDevice:
    """A device that answers surface limits and nothing else."""

    vendor = 'a stand-in'

    def __init__(self, least):
        self.least = least

    def surface_limits(self, config):  # noqa: ARG002 - the signature of the device's
        return {api.VASurfaceAttribMinWidth: self.least[0],
                api.VASurfaceAttribMinHeight: self.least[1],
                api.VASurfaceAttribMaxWidth: 4096, api.VASurfaceAttribMaxHeight: 4096}


def allocation(format, least=(64, 64)):
    """What a pool would allocate for `format` on a device with that minimum."""
    pool = object.__new__(_Pool)
    pool.device = StandInDevice(least)
    pool.format = format
    pool.coded_size = format.coded_size
    pool.config = api.VAConfigID(0)
    return pool._allocation()  # noqa: SLF001 - the rule under test, without the surfaces a pool makes


class TestSurfacesOfTheCodedSizeExactly:
    """radeonsi decodes an HEVC picture at its context's size, so the context
    cannot be made larger than the picture."""

    def exact(self, size):
        return Format(profile=api.VAProfileHEVCMain, coded_size=size, surfaces=4,
                      name='HEVC Main', exact_size=True)

    def test_a_picture_the_device_reaches_is_allocated_as_coded(self):
        assert allocation(self.exact((120, 72))) == (120, 72)
        assert allocation(self.exact((64, 64))) == (64, 64)

    @pytest.mark.parametrize('size', [(96, 56), (56, 96), (32, 32)])
    def test_a_picture_under_the_minimum_is_refused_by_its_size(self, size):
        with pytest.raises(DecoderError, match=(
                f'{size[0]}x{size[1]} HEVC Main picture is smaller than the 64x64')):
            allocation(self.exact(size))

    def test_other_formats_still_decode_into_a_larger_surface(self):
        loose = dataclasses.replace(self.exact((96, 56)), exact_size=False)
        assert allocation(loose) == (96, 64)

    def test_a_device_with_no_minimum_takes_any_size(self):
        assert allocation(self.exact((32, 16)), least=(0, 0)) == (32, 16)


# ------------------------------------------------------------ the layer


class Session:
    """A decode session with no driver: surfaces are numbers, requests are kept.

    A frame output stays lent until `slots` more have been output, as a
    player holding that many frames would have it.
    """

    def __init__(self, slots=2):
        self.slots = slots
        self.control = None
        self.format = None
        self.formats = []
        self.submitted = []
        self.shown = []
        self.lent = []

    def activate(self, format):
        self.formats.append(format)
        self.format = format
        self.lent = []

    def acquire(self):
        held = self.control.held()
        assert len(held) <= self.format.surfaces
        free = [surface for surface in range(self.format.surfaces + self.slots)
                if surface not in held and surface not in self.lent]
        assert free, 'every surface is held or lent'
        return free[0]

    def submit(self, surface, batches, what):
        picture = batches[0][0][1]
        for entry in picture.ReferenceFrames:
            if not entry.flags & INVALID:
                assert entry.picture_id in self.control.held()
                assert entry.picture_id != surface
        self.submitted.append((surface, batches, what))

    def output(self, surface, *, timestamp, duration, visible, colour):
        assert surface in [target for target, _, _ in self.submitted]
        self.shown.append({'surface': surface, 'timestamp': timestamp, 'duration': duration,
                           'visible': visible, 'colour': colour})
        self.lent = (self.lent + [surface])[-self.slots:]


def layer(slots=2):
    session = Session(slots)
    control = HEVCDecoding(session, slots)
    session.control = control
    return session, control


def run(name, start=0, session_and_control=None):
    """A fixture through the layer from sample `start`; the session and the reader."""
    session, control = session_and_control or layer()
    with AnnexBReader(DATA / f'{name}.h265') as reader:
        for index in range(start, len(reader)):
            control.decode(reader.sample(index, parameter_sets=index == start))
        control.flush()
        order = reader.display_order()
        times = [reader.presentation_time(index) for index in range(len(reader))]
    return session, control, order, times


class TestTheLayerOnFixtures:
    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_every_picture_is_decoded_once_and_shown_in_order(self, name):
        session, control, order, times = run(name)
        assert len(session.submitted) == len(order)
        assert [shown['timestamp'] for shown in session.shown] == sorted(times)
        assert control.held() == set()

    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_the_surfaces_suit_the_sequence(self, name):
        session, control, _, _ = run(name)
        [format] = session.formats
        depth = FIXTURES[name]['depth']
        assert format.depth == depth
        assert format.profile == (api.VAProfileHEVCMain10 if depth == 10
                                  else api.VAProfileHEVCMain)
        assert format.exact_size
        width, height = FIXTURES[name]['size']
        assert format.coded_size == (-(-width // 8) * 8, -(-height // 8) * 8)
        assert control.size == (width, height)
        assert {shown['visible'] for shown in session.shown} == {(0, 0, width, height)}

    def test_a_picture_is_its_parameters_then_each_slice_with_its_data(self):
        session, _, _, _ = run('hevc_slices_wpp')
        with AnnexBReader(DATA / 'hevc_slices_wpp.h265') as reader:
            slices = [unit for unit in split_annexb(reader.sample(1).data)
                      if (unit[0] >> 1) & 0x3F < 32]
        _, batches, what = session.submitted[1]
        assert [kind for kind, _ in batches[0]] == [
            api.VAPictureParameterBufferType, api.VAIQMatrixBufferType]
        assert [[kind for kind, _ in batch] for batch in batches[1:]] == [
            [api.VASliceParameterBufferType, api.VASliceDataBufferType]] * 3
        assert [batch[1][1] for batch in batches[1:]] == slices
        assert [batch[0][1].slice_data_size for batch in batches[1:]] == [
            len(unit) for unit in slices]
        assert what.startswith('picture ')

    def test_the_last_slice_is_marked_and_entry_points_are_counted_on(self):
        session, _, _, _ = run('hevc_slices_wpp')
        parameters = [batch[0][1] for batch in session.submitted[1][1][1:]]
        assert [entry.LongSliceFlags.fields.LastSliceOfPic for entry in parameters] == [0, 0, 1]
        assert [entry.num_entry_point_offsets for entry in parameters] == [1, 1, 1]
        assert [entry.entry_offset_to_subset_array for entry in parameters] == [0, 1, 2]
        assert [entry.slice_segment_address for entry in parameters] == [0, 20, 40]

    def test_reference_lists_name_decoded_pictures(self):
        session, _, _, _ = run('hevc_bpyramid')
        for _, batches, _ in session.submitted:
            picture = batches[0][0][1]
            valid = [index for index, entry in enumerate(picture.ReferenceFrames)
                     if not entry.flags & INVALID]
            for batch in batches[1:]:
                parameters = batch[0][1]
                intra = parameters.LongSliceFlags.fields.slice_type == 2
                for listed, count in ((parameters.RefPicList[0],
                                       parameters.num_ref_idx_l0_active_minus1 + 1),
                                      (parameters.RefPicList[1],
                                       parameters.num_ref_idx_l1_active_minus1 + 1)):
                    used = [index for index in listed if index != 0xFF]
                    assert all(index in valid for index in used)
                    assert intra or len(used) in (count, 0)

    def test_the_scaling_lists_of_the_stream_reach_the_driver(self):
        session, _, _, _ = run('hevc_scaling_lists')
        matrix = session.submitted[0][1][0][1][1]
        # The fixture's lists are 11 + n + 2 * row + 3 * column in raster order.
        assert list(matrix.ScalingList8x8[0])[:9] == [16, 19, 22, 25, 28, 31, 34, 37, 18]
        assert list(matrix.ScalingList4x4[0]) == [16] * 16
        assert list(matrix.ScalingListDC32x32) == [22, 23]

    def test_frames_carry_their_samples_timing_and_colour(self):
        session, control, _, _ = run('hevc_main10')
        assert {shown['duration'] for shown in session.shown} == {3000}
        colour = session.shown[0]['colour']
        assert (colour.primaries, colour.transfer, colour.matrix) == (9, 16, 9)
        assert control.colour == colour


class TestTheLayer:
    def test_the_configuration_says_what_the_stream_is(self):
        session, control = layer()
        with AnnexBReader(DATA / 'hevc_main10.h265') as reader:
            control.configure(reader.parameter_sets())
        assert control.size == (120, 68)
        assert control.colour.matrix == 9
        assert session.formats == []

    def test_a_seek_to_a_cra_picture_drops_its_leading_pictures(self):
        with AnnexBReader(DATA / 'hevc_open_gop.h265') as reader:
            key = reader.sync_samples[1]
            order = reader.display_order()
            times = [reader.presentation_time(index) for index in range(len(reader))]
        session, control = layer()
        run('hevc_open_gop', session_and_control=(session, control))
        before = len(session.submitted)
        session.shown.clear()
        control.reset()
        assert control.held() == set()
        run('hevc_open_gop', start=key, session_and_control=(session, control))
        expected = [times[index] for index in order[order.index(key):]]
        assert [shown['timestamp'] for shown in session.shown] == expected
        assert len(session.submitted) - before == len(expected) == len(order) - key - 3

    def test_decoding_from_between_key_pictures_waits_for_the_next(self):
        with AnnexBReader(DATA / 'hevc_bpyramid.h265') as reader:
            key = reader.sync_samples[1]
            order = reader.display_order()
            times = [reader.presentation_time(index) for index in range(len(reader))]
        session, _, _, _ = run('hevc_bpyramid', start=key - 4)
        assert [shown['timestamp'] for shown in session.shown] == [
            times[index] for index in order[order.index(key):]]

    def test_an_end_of_sequence_shows_what_is_waiting(self):
        session, control = layer()
        with AnnexBReader(DATA / 'hevc_bpyramid.h265') as reader:
            for index in range(3):
                control.decode(reader.sample(index))
            waiting = 3 - len(session.shown)
            assert waiting > 0
            end = annexb(nal(NAL_END_OF_SEQUENCE, b'')[:2])
            control.decode(Sample(end, 0, 0, False, 3))
        assert len(session.shown) == 3
        assert control.held() == set()

    def written(self, *slices, sps=None, pps=None):
        """One sample: parameter sets, then a NAL unit for each of `slices`."""
        sps_options, pps_options = sps or {}, pps or {}
        parsed_sps, parsed_pps = sequence(**sps_options), picture_set(**pps_options)
        units = parameter_units(sps=sps_bytes(**sps_options), pps=pps_bytes(**pps_options))
        for options in slices:
            options = dict(options)
            layer_id = options.pop('layer', 0)
            units.append(nal(options.get('nal_type', 1),
                             slice_rbsp(parsed_sps, parsed_pps, **options), layer=layer_id))
        return Sample(annexb(*units), 0, 3000, True, 0)

    IDR = {'nal_type': NAL_IDR_W_RADL, 'kind': 'I'}  # noqa: RUF012 - a constant the cases read

    def test_a_dependent_slice_segment_is_submitted_with_its_slices_lists(self):
        session, control = layer()
        control.decode(self.written(self.IDR, pps={'dependent_slices': 1}))
        trailing = {'kind': 'P', 'poc_lsb': 1, 'rps': coded(before=(-1,)), 'qp_delta': 4}
        control.decode(self.written(
            trailing, {'first': 0, 'dependent': 1, 'address': 5},
            pps={'dependent_slices': 1}))
        _, batches, _ = session.submitted[1]
        first, second = (batch[0][1] for batch in batches[1:])
        assert second.LongSliceFlags.fields.dependent_slice_segment_flag == 1
        assert second.slice_segment_address == 5
        assert second.slice_qp_delta == 4
        assert list(second.RefPicList[0]) == list(first.RefPicList[0])
        assert second.RefPicList[0][0] == 0
        assert (first.LongSliceFlags.fields.LastSliceOfPic,
                second.LongSliceFlags.fields.LastSliceOfPic) == (0, 1)

    def test_a_slice_whose_pictures_first_never_arrived_is_dropped(self):
        session, control = layer()
        control.decode(self.written({'kind': 'I', 'first': 0, 'address': 3,
                                     'nal_type': NAL_IDR_W_RADL}))
        assert session.submitted == []

    def test_a_layer_above_the_base_is_ignored(self):
        session, control = layer()
        control.decode(self.written(self.IDR, {**self.IDR, 'layer': 1}))
        control.flush()
        assert len(session.submitted) == 1
        assert len(session.submitted[0][1]) == 2

    def test_a_picture_not_for_output_is_decoded_and_not_shown(self):
        session, control = layer()
        pps = {'output_flag_present': 1}
        control.decode(self.written(self.IDR, pps=pps))
        control.decode(self.written(
            {'kind': 'P', 'poc_lsb': 1, 'rps': coded(before=(-1,)), 'pic_output': 0,
             'nal_type': NAL_TRAIL_N}, pps=pps))
        control.flush()
        assert len(session.submitted) == 2
        assert len(session.shown) == 1

    def test_a_sequence_of_another_size_shows_the_first_and_takes_new_surfaces(self):
        session, control = layer()
        control.decode(self.written(self.IDR, sps={'ordering': ((4, 2, 0),)}))
        assert session.shown == []
        control.decode(self.written(self.IDR, sps={'ordering': ((4, 2, 0),), 'width': 128,
                                                   'height': 72}))
        assert [format.coded_size for format in session.formats] == [(64, 64), (128, 72)]
        assert [shown['visible'] for shown in session.shown] == [(0, 0, 64, 64)]
        control.flush()
        assert [shown['visible'] for shown in session.shown] == [(0, 0, 64, 64),
                                                                 (0, 0, 128, 72)]
        assert control.size == (128, 72)

    def test_the_same_sequence_again_keeps_its_surfaces(self):
        session, control = layer()
        control.decode(self.written(self.IDR))
        control.decode(self.written(self.IDR))
        assert len(session.formats) == 1

    @pytest.mark.parametrize('sps,pps,message', [
        ({'chroma_format_idc': 2, 'profile_idc': 4}, {}, '4:2:2'),
        ({'bit_depth': 4, 'profile_idc': 4}, {}, '12 bits'),
        ({'range_extension': (1, 0, 0, 0, 0, 0, 0, 0, 0)}, {}, 'range extensions'),
        ({}, {'range_extension': {'cross_component': 1}}, 'range extensions'),
    ])
    def test_a_stream_not_decoded_is_refused_when_a_picture_uses_it(self, sps, pps, message):
        session, control = layer()
        sample = self.written(self.IDR, sps=sps, pps=pps)
        control.configure(sample.data[:sample.data.index(b'\x00\x00\x00\x01\x26')])
        with pytest.raises(DecoderError, match=message):
            control.decode(sample)
        assert session.formats == []
        assert session.submitted == []


class TestDamagedStreams:
    """Bytes changed in the headers of a fixture: every picture is decoded or
    refused in words, and the layer never asks for a surface it holds."""

    @pytest.mark.parametrize('name', ['hevc_open_gop', 'hevc_scaling_default'])
    def test_damage_is_decoded_or_refused(self, name):
        original = (DATA / f'{name}.h265').read_bytes()
        starts = [index + 3 for index in range(len(original) - 3)
                  if original[index:index + 3] == b'\x00\x00\x01']
        chance = random.Random(name)
        refused = 0
        for _ in range(60):
            data = bytearray(original)
            for _ in range(chance.randint(1, 4)):
                position = chance.choice(starts) + chance.randint(0, 24)
                if position < len(data):
                    data[position] = chance.randint(0, 255)
            try:
                with AnnexBReader(io.BytesIO(bytes(data))) as reader:
                    samples = [reader.sample(index) for index in range(len(reader))]
            except StreamError:
                refused += 1
                samples = [Sample(b'\x00\x00\x00\x01' + unit, 0, 0, False, index)
                           for index, unit in enumerate(split_annexb(bytes(data)))]
            session, control = layer()
            for sample in samples:
                try:
                    control.decode(sample)
                except (StreamError, DecoderError):
                    refused += 1
            try:
                control.flush()
            except (StreamError, DecoderError):
                refused += 1
            assert len(session.shown) <= len(session.submitted)
        assert refused
