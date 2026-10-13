"""The libva decode buffers, built from parsed headers with no driver.

A field the driver is told wrongly is not reported: the picture decodes into
the wrong thing. Each buffer is built here from headers whose values are known
and read back field by field.
"""
import pytest

from pyopengl_video.decoder import DecoderError
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.decoder import SurfaceOrder
from pyopengl_video.vaapi.h264_decode import (
    ZIGZAG_4X4,
    ZIGZAG_8X8,
    iq_matrix,
    picture_parameters,
    profile_for,
    slice_parameters,
)
from pyopengl_video.vaapi.h264_dpb import DecodedPictureBuffer
from pyopengl_video.vaapi.h264_parse import MemoryManagementOperation as Op
from pyopengl_video.vaapi.h264_parse import PredWeightTable
from tests.h264_syntax import header, picture_set, sequence


def sets(**options):
    sps = sequence(max_num_ref_frames=6, gaps=1, **options)
    return sps, picture_set(sps, refs=(2, 1), weighted_bipred=1)


def decoded(dpb, slice_header, surface):
    picture = dpb.begin(slice_header)
    dpb.finish(picture, surface=surface)


class TestPictureParameters:
    def build(self):
        sps, pps = sets()
        dpb = DecodedPictureBuffer()
        decoded(dpb, header(sps, pps, kind='I', idr=True), 10)
        decoded(dpb, header(sps, pps, frame_num=1, poc_lsb=4,
                            memory_management=(Op(4, max_long_term_frame_idx_plus1=1),
                                               Op(6, long_term_frame_idx=0))), 11)
        decoded(dpb, header(sps, pps, frame_num=4, poc_lsb=8), 12)
        current = header(sps, pps, kind='B', frame_num=5, poc_lsb=6, nal_ref_idc=0)
        picture = dpb.begin(current)
        return sps, pps, picture, picture_parameters(picture, dpb.reference_frames(), 13)

    def test_the_picture_being_decoded(self):
        _, _, picture, parameters = self.build()
        current = parameters.CurrPic
        assert current.picture_id == 13
        assert current.frame_idx == 5
        assert current.flags == 0
        assert (current.TopFieldOrderCnt, current.BottomFieldOrderCnt) == (6, 6)
        assert parameters.frame_num == 5
        assert not parameters.pic_fields.bits.reference_pic_flag

    def test_the_references_with_their_kinds(self):
        _, _, _, parameters = self.build()
        found = {entry.picture_id: entry for entry in parameters.ReferenceFrames
                 if not entry.flags & api.VA_PICTURE_H264_INVALID}
        assert sorted(found) == [10, 11, 12]
        assert found[11].flags == api.VA_PICTURE_H264_LONG_TERM_REFERENCE
        assert found[11].frame_idx == 0
        assert found[12].flags == api.VA_PICTURE_H264_SHORT_TERM_REFERENCE
        assert found[12].frame_idx == 4
        assert found[12].TopFieldOrderCnt == 8

    def test_frames_inferred_from_a_gap_are_not_given_to_the_driver(self):
        """frame_num jumped from 1 to 4: frames 2 and 3 have no surface."""
        _, _, _, parameters = self.build()
        ids = [entry.picture_id for entry in parameters.ReferenceFrames]
        assert ids.count(api.VA_INVALID_SURFACE) == 16 - 3

    def test_the_sequence_fields(self):
        sps, _, _, parameters = self.build()
        assert parameters.picture_width_in_mbs_minus1 == sps.pic_width_in_mbs_minus1
        assert parameters.picture_height_in_mbs_minus1 == sps.pic_height_in_map_units_minus1
        assert parameters.num_ref_frames == 6
        bits = parameters.seq_fields.bits
        assert bits.chroma_format_idc == 1
        assert bits.frame_mbs_only_flag == 1
        assert bits.direct_8x8_inference_flag == 1
        assert bits.gaps_in_frame_num_value_allowed_flag == 1
        assert bits.log2_max_frame_num_minus4 == 0
        assert bits.log2_max_pic_order_cnt_lsb_minus4 == 2
        assert bits.pic_order_cnt_type == 0

    def test_the_picture_set_fields(self):
        _, pps, _, parameters = self.build()
        assert parameters.pic_init_qp_minus26 == -3
        assert parameters.chroma_qp_index_offset == 2
        assert parameters.second_chroma_qp_index_offset == 2
        bits = parameters.pic_fields.bits
        assert bits.entropy_coding_mode_flag == 1
        assert bits.weighted_bipred_idc == 1
        assert bits.deblocking_filter_control_present_flag == 1
        assert bits.transform_8x8_mode_flag == pps.transform_8x8_mode_flag

    def test_bi_prediction_below_8x8_is_barred_from_level_31(self):
        sps, pps = sets(level_idc=31)
        dpb = DecodedPictureBuffer()
        picture = dpb.begin(header(sps, pps, kind='I', idr=True))
        assert picture_parameters(picture, [], 0).seq_fields.bits.MinLumaBiPredSize8x8
        sps, pps = sets(level_idc=30)
        picture = DecodedPictureBuffer().begin(header(sps, pps, kind='I', idr=True))
        assert not picture_parameters(picture, [], 0).seq_fields.bits.MinLumaBiPredSize8x8


class TestScalingMatrix:
    def test_flat_without_lists(self):
        sps, pps = sets()
        matrix = iq_matrix(sps, pps)
        assert all(value == 16 for row in matrix.ScalingList4x4 for value in row)
        assert all(value == 16 for row in matrix.ScalingList8x8 for value in row)

    def test_lists_go_from_zig_zag_to_raster_order(self):
        custom4 = tuple(range(1, 17))
        custom8 = tuple(range(1, 65))
        sps = sequence(scaling=[custom4] * 6 + [custom8] * 2)
        matrix = iq_matrix(sps, picture_set(sps))
        for scan_position, raster in enumerate(ZIGZAG_4X4):
            assert matrix.ScalingList4x4[0][raster] == custom4[scan_position]
        for scan_position, raster in enumerate(ZIGZAG_8X8):
            assert matrix.ScalingList8x8[1][raster] == custom8[scan_position]

    def test_the_scans_are_the_specifications(self):
        """Table 8-12 and 8-13: frame zig-zag scans, as raster indices."""
        assert ZIGZAG_4X4 == (0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15)
        assert ZIGZAG_8X8[:10] == (0, 1, 8, 16, 9, 2, 3, 10, 17, 24)
        assert sorted(ZIGZAG_8X8) == list(range(64))


class TestSliceParameters:
    def build(self, **fields):
        sps, pps = sets()
        dpb = DecodedPictureBuffer()
        decoded(dpb, header(sps, pps, kind='I', idr=True), 10)
        decoded(dpb, header(sps, pps, frame_num=1, poc_lsb=8), 11)
        current = header(sps, pps, kind='B', frame_num=2, poc_lsb=4, nal_ref_idc=0,
                         **fields)
        picture = dpb.begin(current)
        list0, list1 = dpb.reference_lists(picture, current)
        return current, slice_parameters(current, list0, list1, size=500)

    def test_where_the_slice_data_is(self):
        current, parameters = self.build()
        assert parameters.slice_data_size == 500
        assert parameters.slice_data_offset == 0
        assert parameters.slice_data_flag == api.VA_SLICE_DATA_FLAG_ALL
        assert parameters.slice_data_bit_offset == current.header_bits

    def test_the_lists_and_their_unused_entries(self):
        _, parameters = self.build()
        assert parameters.slice_type == 1
        assert parameters.num_ref_idx_l0_active_minus1 == 2
        assert parameters.num_ref_idx_l1_active_minus1 == 1
        assert [parameters.RefPicList1[i].picture_id for i in range(2)] == [11, 10]
        for entries, active in ((parameters.RefPicList0, 3), (parameters.RefPicList1, 2)):
            assert all(entries[i].flags & api.VA_PICTURE_H264_INVALID
                       and entries[i].picture_id == api.VA_INVALID_SURFACE
                       for i in range(active, 32))

    def test_every_active_entry_names_a_decoded_surface(self):
        """Two frames are held and list 0 has three entries: the third is the
        list's first frame again, never an entry with no surface."""
        _, parameters = self.build()
        assert [parameters.RefPicList0[i].picture_id for i in range(3)] == [10, 11, 10]
        assert not any(parameters.RefPicList0[i].flags & api.VA_PICTURE_H264_INVALID
                       for i in range(3))

    def test_the_slice_fields(self):
        _, parameters = self.build(slice_qp_delta=-3, cabac_init_idc=2,
                                   disable_deblocking_filter_idc=2,
                                   slice_alpha_c0_offset_div2=-1,
                                   slice_beta_offset_div2=3,
                                   direct_spatial_mv_pred_flag=1)
        assert parameters.slice_qp_delta == -3
        assert parameters.cabac_init_idc == 2
        assert parameters.disable_deblocking_filter_idc == 2
        assert parameters.slice_alpha_c0_offset_div2 == -1
        assert parameters.slice_beta_offset_div2 == 3
        assert parameters.direct_spatial_mv_pred_flag == 1

    def test_explicit_weights(self):
        table = PredWeightTable(
            luma_log2_weight_denom=5, chroma_log2_weight_denom=4,
            luma_l0=((32, 0), (40, -3), (32, 0)),
            chroma_l0=(((16, 0), (16, 0)), ((12, 1), (20, -2)), ((16, 0), (16, 0))),
            luma_l1=((32, 0), (33, 1)),
            chroma_l1=(((16, 0), (16, 0)), ((16, 0), (16, 0))),
            luma_weight_l0_flag=True, chroma_weight_l0_flag=True,
            luma_weight_l1_flag=True, chroma_weight_l1_flag=False)
        _, parameters = self.build(pred_weight_table=table)
        assert parameters.luma_log2_weight_denom == 5
        assert parameters.chroma_log2_weight_denom == 4
        assert parameters.luma_weight_l0_flag == 1
        assert list(parameters.luma_weight_l0)[:3] == [32, 40, 32]
        assert list(parameters.luma_offset_l0)[:3] == [0, -3, 0]
        assert list(parameters.chroma_weight_l0[1]) == [12, 20]
        assert list(parameters.chroma_offset_l0[1]) == [1, -2]
        assert list(parameters.luma_weight_l1)[:2] == [32, 33]
        assert parameters.chroma_weight_l1_flag == 0

    def test_without_a_table_the_weights_are_empty(self):
        _, parameters = self.build()
        assert parameters.luma_weight_l0_flag == 0
        assert parameters.luma_log2_weight_denom == 0


class TestProfiles:
    @pytest.mark.parametrize('profile_idc,constraints,expected', [
        (66, 0x40, api.VAProfileH264ConstrainedBaseline),
        (66, 0x00, api.VAProfileH264ConstrainedBaseline),
        (77, 0x00, api.VAProfileH264Main),
        (100, 0x00, api.VAProfileH264High),
    ])
    def test_each_profile_decodes_with_its_libva_profile(self, profile_idc,
                                                         constraints, expected):
        sps = sequence(profile_idc=profile_idc, constraint_flags=constraints)
        assert profile_for(sps, picture_set(sps)) == expected

    def test_high_10_is_refused_by_name(self):
        sps = sequence(profile_idc=110, bit_depth=2)
        with pytest.raises(DecoderError, match='High 10'):
            profile_for(sps, picture_set(sps))

    def test_4_4_4_is_refused(self):
        sps = sequence(profile_idc=244, chroma_format_idc=3)
        with pytest.raises(DecoderError, match='4:4:4'):
            profile_for(sps, picture_set(sps))

    def test_interlaced_coding_is_refused_by_name(self):
        sps = sequence(frame_mbs_only=0)
        with pytest.raises(DecoderError, match='interlaced'):
            profile_for(sps, picture_set(sps))

    def test_slice_groups_are_refused(self):
        sps = sequence(profile_idc=66)
        with pytest.raises(DecoderError, match='slice groups'):
            profile_for(sps, picture_set(sps, slice_groups=1))


class TestSurfaceChoice:
    """The surface reused is the one OpenGL finished with longest ago."""

    def test_surfaces_start_in_pool_order(self):
        order = SurfaceOrder([7, 8, 9])
        assert order.choose(in_use=set()) == 7
        assert order.choose(in_use={7}) == 8

    def test_a_returned_surface_goes_to_the_back(self):
        order = SurfaceOrder([7, 8, 9])
        order.returned(7)
        assert order.choose(in_use=set()) == 8
        order.returned(8)
        assert order.choose(in_use=set()) == 9
        order.returned(9)
        assert order.choose(in_use=set()) == 7

    def test_none_free_is_none(self):
        assert SurfaceOrder([1, 2]).choose(in_use={1, 2}) is None
