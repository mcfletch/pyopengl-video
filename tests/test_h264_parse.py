"""The H.264 parser: bit reader, parameter sets and slice headers.

No hardware. The streams parsed here are written with :class:`BitWriter`, in
the order of the syntax tables of ITU-T H.264, so each test states a field in
the specification's terms and reads it back through the parser. The encoder's
own :class:`ParameterSets` is parsed as well: a decoder has to read what our
encoders write.
"""
import pytest

from pyopengl_video.vaapi.h264 import (
    BitReader,
    BitWriter,
    GroupOfPictures,
    ParameterSets,
    nal_unit,
    rbsp_of,
)
from pyopengl_video.vaapi.h264_parse import (
    DEFAULT_4X4_INTER,
    DEFAULT_4X4_INTRA,
    DEFAULT_8X8_INTER,
    DEFAULT_8X8_INTRA,
    FLAT_4X4,
    FLAT_8X8,
    MemoryManagementOperation,
    StreamError,
    parse_nal,
    parse_pps,
    parse_slice_header,
    parse_sps,
    scaling_matrices,
    starts_new_picture,
)
from tests.h264_syntax import (
    full_vui,
    picture_set,
    sequence,
    slice_unit,
    sps_bytes,
    write_pps_head,
    write_scaling_list,
)
from tests.h264_syntax import header as make_header

# ---------------------------------------------------------------- bit reader


class TestBitReader:
    def test_fields_read_back_in_the_order_they_were_written(self):
        writer = BitWriter()
        writer.u(5, 3)
        writer.ue(300)
        writer.se(-77)
        writer.flag(1)
        writer.u(0xABCDE, 20)
        reader = BitReader(writer.rbsp())
        assert reader.u(3) == 5
        assert reader.ue() == 300
        assert reader.se() == -77
        assert reader.flag() == 1
        assert reader.u(20) == 0xABCDE

    @pytest.mark.parametrize('value', [0, 1, 2, 3, 7, 8, 254, 255, 65535, 2**31])
    def test_exp_golomb_values_of_every_width(self, value):
        writer = BitWriter()
        writer.ue(value)
        assert BitReader(writer.rbsp()).ue() == value

    @pytest.mark.parametrize('value', [0, 1, -1, 2, -2, 1000, -1000])
    def test_signed_values(self, value):
        writer = BitWriter()
        writer.se(value)
        assert BitReader(writer.rbsp()).se() == value

    def test_a_32_bit_field(self):
        writer = BitWriter()
        writer.u(0xDEADBEEF, 32)
        assert BitReader(writer.rbsp()).u(32) == 0xDEADBEEF

    def test_the_position_counts_bits(self):
        writer = BitWriter()
        writer.u(0, 3)
        writer.ue(3)
        reader = BitReader(writer.rbsp())
        reader.u(3)
        reader.ue()
        assert reader.position == 8

    def test_reading_past_the_end_is_a_stream_error(self):
        reader = BitReader(b'\xff')
        reader.u(8)
        with pytest.raises(StreamError):
            reader.u(1)

    def test_an_exp_golomb_code_longer_than_the_data_is_a_stream_error(self):
        with pytest.raises(StreamError):
            BitReader(b'\x00\x00').ue()

    def test_more_data_stops_at_the_trailing_bits(self):
        writer = BitWriter()
        writer.u(0b101, 3)
        reader = BitReader(writer.rbsp())
        assert reader.more_rbsp_data()
        reader.u(3)
        assert not reader.more_rbsp_data()

    def test_more_data_ignores_trailing_zero_bytes(self):
        writer = BitWriter()
        writer.u(1, 1)
        reader = BitReader(writer.rbsp() + b'\x00\x00')
        reader.u(1)
        assert not reader.more_rbsp_data()

    def test_byte_alignment_is_reported(self):
        reader = BitReader(b'\xff\xff')
        assert reader.byte_aligned()
        reader.u(3)
        assert not reader.byte_aligned()


class TestRBSP:
    def test_emulation_prevention_is_removed(self):
        assert rbsp_of(b'\x00\x00\x03\x01') == b'\x00\x00\x01'

    def test_consecutive_escapes_are_each_removed(self):
        assert rbsp_of(b'\x00\x00\x03\x00\x00\x03\x00') == b'\x00\x00\x00\x00\x00'

    def test_an_escaped_three_keeps_the_data_byte(self):
        assert rbsp_of(b'\x00\x00\x03\x03') == b'\x00\x00\x03'

    def test_a_unit_round_trips_through_the_writer(self):
        payload = bytes([0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 7])
        unit = nal_unit(payload, nal_unit_type=1)
        assert rbsp_of(unit[1:]) == payload


class TestNalUnit:
    def test_the_header_byte_is_split_into_its_fields(self):
        unit = parse_nal(bytes([0x65, 0x88]))
        assert unit.nal_ref_idc == 3
        assert unit.nal_unit_type == 5
        assert unit.idr

    def test_a_non_reference_picture(self):
        unit = parse_nal(bytes([0x01, 0x9a]))
        assert unit.nal_ref_idc == 0
        assert unit.nal_unit_type == 1
        assert not unit.idr

    def test_the_payload_is_unescaped(self):
        unit = parse_nal(bytes([0x41, 0, 0, 3, 1]))
        assert unit.rbsp == b'\x00\x00\x01'
        assert unit.data == bytes([0x41, 0, 0, 3, 1])

    def test_the_forbidden_bit_is_refused(self):
        with pytest.raises(StreamError):
            parse_nal(bytes([0xE5, 0]))

    def test_an_empty_unit_is_refused(self):
        with pytest.raises(StreamError):
            parse_nal(b'')


# --------------------------------------------------- what our encoders write


def encoder_sets(width=1920, height=1080, fps=(30, 1), refs=1):
    sets = ParameterSets(width, height, fps, max_num_ref_frames=refs)
    sps = parse_sps(parse_nal(sets.sps()).rbsp)
    pps = parse_pps(parse_nal(sets.pps()).rbsp, {sps.seq_parameter_set_id: sps})
    return sets, sps, pps


class TestEncoderSequenceParameterSet:
    def test_profile_and_level(self):
        sets, sps, _ = encoder_sets()
        assert sps.profile_idc == 100
        assert sps.level_idc == sets.level_idc

    def test_the_coded_size_is_whole_macroblocks(self):
        _, sps, _ = encoder_sets()
        assert sps.coded_size == (1920, 1088)

    def test_the_visible_size_is_cropped(self):
        _, sps, _ = encoder_sets()
        assert sps.size == (1920, 1080)
        assert sps.visible == (0, 0, 1920, 1080)

    def test_an_odd_number_of_macroblocks_is_cropped_on_both_axes(self):
        _, sps, _ = encoder_sets(width=1000, height=500)
        assert sps.coded_size == (1008, 512)
        assert sps.size == (1000, 500)

    def test_the_counter_widths(self):
        sets, sps, _ = encoder_sets()
        assert sps.max_frame_num == 1 << sets.log2_max_frame_num
        assert sps.max_pic_order_cnt_lsb == 1 << sets.log2_max_poc_lsb
        assert sps.pic_order_cnt_type == 0

    def test_progressive_only(self):
        _, sps, _ = encoder_sets()
        assert sps.frame_mbs_only_flag

    def test_the_colour_description(self):
        _, sps, _ = encoder_sets()
        vui = sps.vui
        assert vui is not None
        assert vui.colour_primaries == 1
        assert vui.transfer_characteristics == 1
        assert vui.matrix_coefficients == 1
        assert not vui.video_full_range_flag

    def test_the_frame_rate(self):
        _, sps, _ = encoder_sets(fps=(30000, 1001))
        assert sps.frame_rate == (30000, 1001)

    def test_no_reordering(self):
        _, sps, _ = encoder_sets(refs=2)
        assert sps.vui.max_num_reorder_frames == 0
        assert sps.vui.max_dec_frame_buffering == 2
        assert sps.max_num_ref_frames == 2
        assert sps.reorder_frames == 0
        assert sps.dpb_frames == 2

    def test_no_scaling_matrices_means_flat(self):
        _, sps, pps = encoder_sets()
        four, eight = scaling_matrices(sps, pps)
        assert four == [FLAT_4X4] * 6
        assert eight == [FLAT_8X8] * 2


class TestEncoderPictureParameterSet:
    def test_the_fields_the_encoder_sets(self):
        _, _, pps = encoder_sets()
        assert pps.pic_parameter_set_id == 0
        assert pps.seq_parameter_set_id == 0
        assert pps.entropy_coding_mode_flag
        assert pps.transform_8x8_mode_flag
        assert pps.deblocking_filter_control_present_flag
        assert pps.pic_init_qp_minus26 == 0
        assert pps.num_ref_idx_l0_default_active_minus1 == 0

    def test_the_second_chroma_offset_defaults_to_the_first(self):
        sets = ParameterSets(64, 64, (30, 1), transform_8x8=False)
        sps = parse_sps(parse_nal(sets.sps()).rbsp)
        pps = parse_pps(parse_nal(sets.pps()).rbsp, {0: sps})
        assert not pps.transform_8x8_mode_flag
        assert pps.second_chroma_qp_index_offset == pps.chroma_qp_index_offset

    def test_a_pps_naming_an_unknown_sps_is_refused(self):
        sets = ParameterSets(64, 64, (30, 1))
        with pytest.raises(StreamError, match='sequence parameter set 0'):
            parse_pps(parse_nal(sets.pps()).rbsp, {})


class TestEncoderSliceHeaders:
    def headers(self, count, gop=4):
        sets, sps, pps = encoder_sets(width=64, height=64)
        structure = GroupOfPictures(gop=gop)
        found = []
        for _ in range(count):
            picture = structure.next_picture()
            data, bits = sets.packed_slice_header(picture)
            unit = parse_nal(data[4:])
            header = parse_slice_header(unit, {0: sps}, {0: pps})
            found.append((picture, header, bits))
        return found

    def test_each_field_reads_back(self):
        for picture, header, _ in self.headers(9):
            assert header.first_mb_in_slice == 0
            assert header.idr == picture.idr
            assert header.frame_num == picture.frame_num
            assert header.pic_order_cnt_lsb == picture.poc % 256
            assert header.slice_type == (2 if picture.intra else 0)
            if picture.idr:
                assert header.idr_pic_id == picture.idr_pic_id

    def test_the_slice_types_are_named(self):
        kinds = [header.kind for _, header, _ in self.headers(5)]
        assert kinds == ['I', 'P', 'P', 'P', 'I']

    def test_the_header_length_is_where_slice_data_starts(self):
        """The packed header's bit count less its start code is the offset."""
        for _, header, bits in self.headers(5):
            assert header.header_bits == bits - 32

    def test_predicted_pictures_override_to_one_reference(self):
        for picture, header, _ in self.headers(5):
            if not picture.intra:
                assert header.num_ref_idx_active_override_flag
                assert header.num_ref_idx_l0_active_minus1 == 0

    def test_the_marking_is_the_sliding_window(self):
        for picture, header, _ in self.headers(5):
            if picture.idr:
                assert not header.no_output_of_prior_pics_flag
                assert not header.long_term_reference_flag
            else:
                assert not header.adaptive_ref_pic_marking_mode_flag
                assert header.memory_management == ()


# ------------------------------------------------ syntax the encoders do not use










class TestSequenceParameterSetSyntax:
    def test_a_full_vui_is_read_past_its_hrd_parameters(self):
        sps = parse_sps(sps_bytes(vui=full_vui))
        vui = sps.vui
        assert vui.sar == (4, 3)
        assert vui.video_full_range_flag
        assert (vui.colour_primaries, vui.transfer_characteristics,
                vui.matrix_coefficients) == (6, 6, 6)
        assert vui.chroma_sample_loc_type_top_field == 2
        assert sps.frame_rate == (24000, 1001)
        assert vui.max_num_reorder_frames == 2
        assert sps.reorder_frames == 2
        assert sps.dpb_frames == 4

    def test_picture_order_count_type_one(self):
        sps = parse_sps(sps_bytes(poc_type=1))
        assert sps.pic_order_cnt_type == 1
        assert sps.offset_for_non_ref_pic == -1
        assert sps.offset_for_ref_frame == (2, 4)

    def test_picture_order_count_type_two(self):
        sps = parse_sps(sps_bytes(poc_type=2))
        assert sps.pic_order_cnt_type == 2

    def test_cropping_is_in_chroma_units(self):
        sps = parse_sps(sps_bytes(crop=(1, 2, 3, 4)))
        assert sps.visible == (2, 6, 64 - 2 - 4, 64 - 6 - 8)

    def test_a_baseline_profile_has_no_chroma_syntax(self):
        sps = parse_sps(sps_bytes(profile_idc=66, constraint_flags=0x40))
        assert sps.profile_idc == 66
        assert sps.chroma_format_idc == 1
        assert sps.constraint_set1_flag

    def test_interlaced_coding_is_reported(self):
        sps = parse_sps(sps_bytes(frame_mbs_only=0))
        assert not sps.frame_mbs_only_flag
        assert sps.mb_adaptive_frame_field_flag
        assert sps.coded_size == (64, 128)

    def test_the_gaps_flag(self):
        assert parse_sps(sps_bytes(gaps=1)).gaps_in_frame_num_value_allowed_flag

    def test_without_vui_the_dpb_is_sized_from_the_level(self):
        """Level 3 holds 8100 macroblocks of frames, 16 frames at most."""
        sps = parse_sps(sps_bytes(level_idc=30, width_mbs=45, height_mbs=36))
        assert sps.dpb_frames == 8100 // (45 * 36)
        assert sps.reorder_frames == sps.dpb_frames

    def test_the_dpb_never_exceeds_sixteen_frames(self):
        sps = parse_sps(sps_bytes(level_idc=51, width_mbs=4, height_mbs=4))
        assert sps.dpb_frames == 16

    def test_the_dpb_holds_the_reference_frames_at_least(self):
        sps = parse_sps(sps_bytes(level_idc=10, width_mbs=22, height_mbs=18,
                                  max_num_ref_frames=3))
        assert sps.dpb_frames == 3


class TestScalingMatrices:
    def test_present_lists_are_kept(self):
        custom4 = list(range(1, 17))
        custom8 = [value % 50 + 4 for value in range(64)]
        sps = parse_sps(sps_bytes(scaling=[custom4] * 6 + [custom8] * 2))
        four, eight = scaling_matrices(sps, None)
        assert four == [tuple(custom4)] * 6
        assert eight == [tuple(custom8)] * 2

    def test_the_default_flag_selects_the_default_lists(self):
        sps = parse_sps(sps_bytes(scaling=['default'] * 8))
        four, eight = scaling_matrices(sps, None)
        assert four == [DEFAULT_4X4_INTRA] * 3 + [DEFAULT_4X4_INTER] * 3
        assert eight == [DEFAULT_8X8_INTRA, DEFAULT_8X8_INTER]

    def test_fall_back_rule_a_for_absent_sequence_lists(self):
        """Absent 0, 3, 6, 7 take the defaults; the others copy the one before."""
        custom = tuple(range(20, 36))
        sps = parse_sps(sps_bytes(scaling=[None, custom, None, None, None, None,
                                           None, None]))
        four, eight = scaling_matrices(sps, None)
        assert four == [DEFAULT_4X4_INTRA, custom, custom,
                        DEFAULT_4X4_INTER, DEFAULT_4X4_INTER, DEFAULT_4X4_INTER]
        assert eight == [DEFAULT_8X8_INTRA, DEFAULT_8X8_INTER]

    def test_fall_back_rule_b_takes_the_sequence_lists(self):
        sequence = tuple(range(40, 56))
        sps = parse_sps(sps_bytes(scaling=[sequence] * 6 + [None, None]))
        writer = BitWriter()
        write_pps_head(writer)
        writer.flag(1)                          # transform_8x8_mode_flag
        writer.flag(1)                          # pic_scaling_matrix_present
        own = tuple(range(60, 76))
        for index in range(8):
            present = index == 1
            writer.flag(present)
            if present:
                write_scaling_list(writer, own)
        writer.se(0)
        pps = parse_pps(writer.rbsp(), {0: sps})
        four, eight = scaling_matrices(sps, pps)
        assert four == [sequence, own, own, sequence, sequence, sequence]
        assert eight == [DEFAULT_8X8_INTRA, DEFAULT_8X8_INTER]

    def test_without_a_sequence_matrix_absent_picture_lists_take_the_defaults(self):
        """7.4.2.2: rule A applies when the sequence has no scaling matrix,
        which is how x264 signals the JVT default matrices."""
        sps = parse_sps(sps_bytes())
        writer = BitWriter()
        write_pps_head(writer)
        writer.flag(1)                          # transform_8x8_mode_flag
        writer.flag(1)                          # pic_scaling_matrix_present
        for _ in range(8):
            writer.flag(0)
        writer.se(0)
        four, eight = scaling_matrices(sps, parse_pps(writer.rbsp(), {0: sps}))
        assert four == [DEFAULT_4X4_INTRA] * 3 + [DEFAULT_4X4_INTER] * 3
        assert eight == [DEFAULT_8X8_INTRA, DEFAULT_8X8_INTER]

    def test_a_picture_set_without_lists_uses_the_sequence_ones(self):
        sequence = tuple(range(40, 56))
        sps = parse_sps(sps_bytes(scaling=[sequence] * 6 + ['default'] * 2))
        writer = BitWriter()
        write_pps_head(writer)
        pps = parse_pps(writer.rbsp(), {0: sps})
        four, _ = scaling_matrices(sps, pps)
        assert four == [sequence] * 6

    def test_the_default_tables_are_whole(self):
        assert len(DEFAULT_4X4_INTRA) == len(DEFAULT_4X4_INTER) == 16
        assert len(DEFAULT_8X8_INTRA) == len(DEFAULT_8X8_INTER) == 64
        assert DEFAULT_4X4_INTRA[0] == 6 and DEFAULT_4X4_INTRA[-1] == 42
        assert DEFAULT_8X8_INTER[0] == 9 and DEFAULT_8X8_INTER[-1] == 35




class TestPictureParameterSetSyntax:
    def test_slice_groups_are_parsed_and_reported(self):
        sps = parse_sps(sps_bytes())
        writer = BitWriter()
        write_pps_head(writer, slice_groups=1)
        pps = parse_pps(writer.rbsp(), {0: sps})
        assert pps.num_slice_groups_minus1 == 1
        assert pps.pic_init_qp_minus26 == -3

    def test_the_signed_offsets(self):
        sps = parse_sps(sps_bytes())
        writer = BitWriter()
        write_pps_head(writer)
        pps = parse_pps(writer.rbsp(), {0: sps})
        assert pps.chroma_qp_index_offset == 2
        assert pps.second_chroma_qp_index_offset == 2


# --------------------------------------------------------------- slice syntax




class TestSliceHeaderSyntax:
    def sets(self, **pps_options):
        sps = parse_sps(sps_bytes(max_num_ref_frames=4))
        writer = BitWriter()
        write_pps_head(writer, **pps_options)
        return {0: sps}, {0: parse_pps(writer.rbsp(), {0: sps})}

    def test_a_b_slice_with_weights_modifications_and_operations(self):
        sequences, pictures = self.sets(weighted_bipred=1, refs=(1, 0))

        def write(writer):
            writer.ue(0)                        # first_mb_in_slice
            writer.ue(6)                        # B, all slices
            writer.ue(0)
            writer.u(5, 4)                      # frame_num
            writer.u(20, 6)                     # pic_order_cnt_lsb
            writer.flag(1)                      # direct_spatial_mv_pred_flag
            writer.flag(1)                      # num_ref_idx_active_override
            writer.ue(2)
            writer.ue(1)
            writer.flag(1)                      # ref_pic_list_modification_flag_l0
            writer.ue(0)
            writer.ue(3)                        # abs_diff_pic_num_minus1
            writer.ue(2)
            writer.ue(1)                        # long_term_pic_num
            writer.ue(3)
            writer.flag(0)                      # ref_pic_list_modification_flag_l1
            writer.ue(5)                        # luma_log2_weight_denom
            writer.ue(4)                        # chroma_log2_weight_denom
            for index in range(3):
                writer.flag(index == 1)
                if index == 1:
                    writer.se(40)
                    writer.se(-3)
                writer.flag(index == 2)
                if index == 2:
                    for _ in range(2):
                        writer.se(12)
                        writer.se(1)
            for _ in range(2):
                writer.flag(0)
                writer.flag(0)
            writer.flag(1)                      # adaptive_ref_pic_marking_mode_flag
            writer.ue(1)
            writer.ue(0)
            writer.ue(3)
            writer.ue(1)
            writer.ue(2)
            writer.ue(0)
            writer.ue(1)                        # cabac_init_idc
            writer.se(-4)                       # slice_qp_delta
            writer.ue(2)
            writer.se(1)
            writer.se(-1)

        header = parse_slice_header(slice_unit(write), sequences, pictures)
        assert header.kind == 'B'
        assert header.frame_num == 5
        assert header.pic_order_cnt_lsb == 20
        assert header.direct_spatial_mv_pred_flag
        assert header.num_ref_idx_l0_active_minus1 == 2
        assert header.num_ref_idx_l1_active_minus1 == 1
        assert header.ref_pic_list_modification_l0 == ((0, 3), (2, 1))
        assert header.ref_pic_list_modification_l1 == ()
        weights = header.pred_weight_table
        assert weights.luma_log2_weight_denom == 5
        assert weights.chroma_log2_weight_denom == 4
        assert weights.luma_l0 == ((32, 0), (40, -3), (32, 0))
        assert weights.chroma_l0[2] == ((12, 1), (12, 1))
        assert weights.chroma_l0[0] == ((16, 0), (16, 0))
        assert weights.luma_weight_l0_flag
        assert weights.chroma_weight_l0_flag
        assert not weights.luma_weight_l1_flag
        assert weights.luma_l1 == ((32, 0), (32, 0))
        assert header.memory_management == (
            MemoryManagementOperation(1, difference_of_pic_nums_minus1=0),
            MemoryManagementOperation(3, difference_of_pic_nums_minus1=1,
                                      long_term_frame_idx=2),
        )
        assert header.cabac_init_idc == 1
        assert header.slice_qp_delta == -4
        assert header.disable_deblocking_filter_idc == 2
        assert header.slice_alpha_c0_offset_div2 == 1
        assert header.slice_beta_offset_div2 == -1

    def test_inactive_counts_come_from_the_picture_set(self):
        sequences, pictures = self.sets(refs=(2, 1), entropy=0)

        def write(writer):
            writer.ue(0)
            writer.ue(5)                        # P
            writer.ue(0)
            writer.u(1, 4)
            writer.u(2, 6)
            writer.flag(0)                      # num_ref_idx_active_override
            writer.flag(0)                      # ref_pic_list_modification_flag_l0
            writer.flag(0)                      # adaptive_ref_pic_marking_mode_flag
            writer.se(0)
            writer.ue(1)                        # deblocking off

        header = parse_slice_header(slice_unit(write), sequences, pictures)
        assert header.num_ref_idx_l0_active_minus1 == 2
        assert header.num_ref_idx_l1_active_minus1 == 1
        assert header.pred_weight_table is None
        assert header.disable_deblocking_filter_idc == 1

    def test_a_non_reference_slice_has_no_marking(self):
        sequences, pictures = self.sets()

        def write(writer):
            writer.ue(0)
            writer.ue(7)                        # I
            writer.ue(0)
            writer.u(1, 4)
            writer.u(2, 6)
            writer.se(0)
            writer.ue(0)
            writer.se(0)
            writer.se(0)

        header = parse_slice_header(slice_unit(write, nal_ref_idc=0),
                                    sequences, pictures)
        assert header.nal_ref_idc == 0
        assert header.memory_management == ()

    def test_an_idr_with_a_long_term_reference(self):
        sequences, pictures = self.sets()

        def write(writer):
            writer.ue(0)
            writer.ue(7)
            writer.ue(0)
            writer.u(0, 4)
            writer.ue(3)                        # idr_pic_id
            writer.u(0, 6)
            writer.flag(1)                      # no_output_of_prior_pics
            writer.flag(1)                      # long_term_reference
            writer.se(0)
            writer.ue(0)
            writer.se(0)
            writer.se(0)

        header = parse_slice_header(slice_unit(write, nal_ref_idc=3,
                                               nal_unit_type=5),
                                    sequences, pictures)
        assert header.idr_pic_id == 3
        assert header.no_output_of_prior_pics_flag
        assert header.long_term_reference_flag

    def test_the_bottom_field_order_delta(self):
        sequences, pictures = self.sets(bottom_field_pic_order=1)

        def write(writer):
            writer.ue(0)
            writer.ue(7)
            writer.ue(0)
            writer.u(0, 4)
            writer.u(4, 6)
            writer.se(-1)                       # delta_pic_order_cnt_bottom
            writer.se(0)
            writer.ue(0)
            writer.se(0)
            writer.se(0)

        header = parse_slice_header(slice_unit(write, nal_ref_idc=0),
                                    sequences, pictures)
        assert header.delta_pic_order_cnt_bottom == -1

    def test_a_slice_naming_an_unknown_picture_set_is_refused(self):
        def write(writer):
            writer.ue(0)
            writer.ue(7)
            writer.ue(9)

        with pytest.raises(StreamError, match='picture parameter set 9'):
            parse_slice_header(slice_unit(write), {}, {})

    def test_a_slice_type_out_of_range_is_refused(self):
        sequences, pictures = self.sets()

        def write(writer):
            writer.ue(0)
            writer.ue(10)
            writer.ue(0)

        with pytest.raises(StreamError, match='slice_type is 10'):
            parse_slice_header(slice_unit(write), sequences, pictures)

    def test_field_pictures_are_reported(self):
        sps = parse_sps(sps_bytes(frame_mbs_only=0))
        writer = BitWriter()
        write_pps_head(writer)
        pps = parse_pps(writer.rbsp(), {0: sps})

        def write(writer):
            writer.ue(0)
            writer.ue(7)
            writer.ue(0)
            writer.u(0, 4)
            writer.flag(1)                      # field_pic_flag
            writer.flag(1)                      # bottom_field_flag
            writer.u(0, 6)
            writer.se(0)
            writer.ue(0)
            writer.se(0)
            writer.se(0)

        header = parse_slice_header(slice_unit(write, nal_ref_idc=0),
                                    {0: sps}, {0: pps})
        assert header.field_pic_flag
        assert header.bottom_field_flag


class TestPictureBoundaries:
    """7.4.1.2.4: what makes a slice the first of a new picture."""

    def sets(self):
        sps = sequence()
        pps = picture_set(sps)
        return lambda **fields: make_header(sps, pps, **fields)

    def test_a_second_slice_of_the_same_picture(self):
        make = self.sets()
        first = make(frame_num=1, poc_lsb=2)
        second = make(frame_num=1, poc_lsb=2, first_mb_in_slice=8)
        assert not starts_new_picture(first, second)

    @pytest.mark.parametrize('change', [
        {'frame_num': 2},
        {'poc_lsb': 4},
        {'nal_ref_idc': 0},
        {'delta_pic_order_cnt_bottom': 1},
        {'pic_parameter_set_id': 1},
    ])
    def test_a_field_that_differs_starts_a_new_picture(self, change):
        make = self.sets()
        first = make(frame_num=1, poc_lsb=2)
        fields = {'frame_num': 1, 'poc_lsb': 2, 'first_mb_in_slice': 8, **change}
        assert starts_new_picture(first, make(**fields))

    def test_two_idr_pictures_differ_by_their_identifier(self):
        make = self.sets()
        first = make(kind='I', idr=True, idr_pic_id=0)
        assert starts_new_picture(first, make(kind='I', idr=True, idr_pic_id=1,
                                              first_mb_in_slice=8))
        assert not starts_new_picture(first, make(kind='I', idr=True,
                                                  first_mb_in_slice=8))

    def test_an_idr_after_a_non_idr_picture(self):
        make = self.sets()
        assert starts_new_picture(make(), make(idr=True, first_mb_in_slice=8))


# --------------------------------------------- limits the specification sets


class TestLimitsTheStreamCannotExceed:
    """A claim past what H.264 allows is refused before anything is sized from it."""

    def test_the_frame_number_width_stops_at_sixteen_bits(self):
        parse_sps(sps_bytes(log2_max_frame_num_minus4=12))
        with pytest.raises(StreamError, match='log2_max_frame_num_minus4'):
            parse_sps(sps_bytes(log2_max_frame_num_minus4=13))

    def test_a_default_reference_count_stops_at_thirty_two(self):
        sps = parse_sps(sps_bytes())
        writer = BitWriter()
        write_pps_head(writer, refs=(31, 0))
        assert parse_pps(writer.rbsp(), {0: sps}).num_ref_idx_l0_default_active_minus1 == 31
        writer = BitWriter()
        write_pps_head(writer, refs=(32, 0))
        with pytest.raises(StreamError, match='num_ref_idx_l0_default_active_minus1'):
            parse_pps(writer.rbsp(), {0: sps})
