"""The H.264 bitstream writer and the parameter sets built with it.

No hardware here: this is the half of the VA-API backend that libva does not do
for the caller, and a field packed one bit off is exactly what no player reports
clearly.  Every set written is read back with the parser below, which follows
the syntax tables of the H.264 specification rather than the writer's own code.
"""
import pytest

from pyopengl_video.vaapi.h264 import (
    BitWriter,
    GroupOfPictures,
    ParameterSets,
    annexb,
    emulation_prevented,
)


class BitReader:
    """Reads back what :class:`BitWriter` produced, by the spec's syntax.

    Written the long way round on purpose: a reader that shared the writer's
    helpers would agree with it about a mistake.
    """

    def __init__(self, data: bytes):
        self.bits = ''.join(format(byte, '08b') for byte in data)
        self.position = 0

    def u(self, count: int) -> int:
        chunk = self.bits[self.position:self.position + count]
        assert len(chunk) == count, 'ran off the end of the bitstream'
        self.position += count
        return int(chunk, 2)

    def flag(self) -> int:
        return self.u(1)

    def ue(self) -> int:
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
        return (1 << zeros) - 1 + (self.u(zeros) if zeros else 0)

    def se(self) -> int:
        value = self.ue()
        magnitude = (value + 1) // 2
        return magnitude if value % 2 else -magnitude


def unescape(data: bytes) -> bytes:
    """Undo emulation prevention, so the reader sees the RBSP the writer built."""
    out = bytearray()
    zeros = 0
    for byte in data:
        if zeros == 2 and byte == 3:
            zeros = 0
            continue
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


# --------------------------------------------------------------- bit writer


def bits_of(writer: BitWriter) -> str:
    """The writer's buffer as a string of '0' and '1', trailing bits included."""
    return ''.join(format(byte, '08b') for byte in writer.rbsp())


class TestBitWriter:
    def test_unsigned_fields_pack_big_endian(self):
        writer = BitWriter()
        writer.u(1, 1)
        writer.u(0, 2)
        writer.u(5, 3)
        assert bits_of(writer).startswith('100101')

    def test_a_field_wider_than_a_byte_spans_bytes(self):
        writer = BitWriter()
        writer.u(0xABCD, 16)
        assert bits_of(writer).startswith('1010101111001101')

    def test_a_value_too_large_for_its_field_is_refused(self):
        writer = BitWriter()
        with pytest.raises(ValueError):
            writer.u(4, 2)

    def test_a_negative_unsigned_field_is_refused(self):
        writer = BitWriter()
        with pytest.raises(ValueError):
            writer.u(-1, 8)

    @pytest.mark.parametrize('value,expected', [
        (0, '1'), (1, '010'), (2, '011'), (3, '00100'), (4, '00101'),
        (5, '00110'), (6, '00111'), (7, '0001000'), (8, '0001001'),
        (14, '0001111'), (15, '000010000'),
    ])
    def test_exp_golomb_matches_the_specification(self, value, expected):
        writer = BitWriter()
        writer.ue(value)
        assert bits_of(writer).startswith(expected)

    @pytest.mark.parametrize('value,mapped', [
        (0, 0), (1, 1), (-1, 2), (2, 3), (-2, 4), (3, 5), (-3, 6),
    ])
    def test_signed_exp_golomb_maps_onto_the_unsigned_code(self, value, mapped):
        signed, unsigned = BitWriter(), BitWriter()
        signed.se(value)
        unsigned.ue(mapped)
        assert bits_of(signed) == bits_of(unsigned)

    def test_a_negative_unsigned_exp_golomb_value_is_refused(self):
        writer = BitWriter()
        with pytest.raises(ValueError):
            writer.ue(-1)

    def test_trailing_bits_close_the_rbsp_on_a_byte_boundary(self):
        writer = BitWriter()
        writer.u(0, 3)
        data = writer.rbsp()
        assert len(data) == 1
        # The stop bit, then zeroes to the end of the byte.
        assert format(data[0], '08b') == '00010000'

    def test_an_already_aligned_rbsp_still_gets_a_whole_stop_byte(self):
        writer = BitWriter()
        writer.u(0xFF, 8)
        data = writer.rbsp()
        assert data == bytes([0xFF, 0x80])

    def test_round_trip_through_the_reader(self):
        writer = BitWriter()
        writer.ue(300)
        writer.se(-77)
        writer.u(1, 1)
        writer.u(1000, 12)
        reader = BitReader(writer.rbsp())
        assert reader.ue() == 300
        assert reader.se() == -77
        assert reader.flag() == 1
        assert reader.u(12) == 1000


# ------------------------------------------------------- emulation prevention


class TestEmulationPrevention:
    @pytest.mark.parametrize('raw,expected', [
        (b'\x00\x00\x00', b'\x00\x00\x03\x00'),
        (b'\x00\x00\x01', b'\x00\x00\x03\x01'),
        (b'\x00\x00\x02', b'\x00\x00\x03\x02'),
        (b'\x00\x00\x03', b'\x00\x00\x03\x03'),
        (b'\x00\x00\x04', b'\x00\x00\x04'),
        (b'\x00\x01\x00', b'\x00\x01\x00'),
    ])
    def test_the_three_byte_patterns_that_must_not_appear(self, raw, expected):
        assert emulation_prevented(raw) == expected

    def test_a_long_run_of_zeroes_is_broken_up_repeatedly(self):
        assert emulation_prevented(b'\x00' * 6) == b'\x00\x00\x03\x00\x00\x03\x00\x00'

    def test_the_escape_is_reversible(self):
        raw = bytes([0, 0, 0, 1, 2, 3, 0, 0, 1, 0, 0, 0, 0])
        assert unescape(emulation_prevented(raw)) == raw

    def test_a_start_code_cannot_survive_the_escape(self):
        assert b'\x00\x00\x01' not in emulation_prevented(b'\xAA\x00\x00\x01\xBB')


class TestAnnexB:
    def test_a_unit_is_prefixed_with_a_four_byte_start_code(self):
        assert annexb(b'\x67\x42') == b'\x00\x00\x00\x01\x67\x42'

    def test_several_units_are_joined(self):
        assert annexb(b'\xAA', b'\xBB') == (
            b'\x00\x00\x00\x01\xAA\x00\x00\x00\x01\xBB')


# ------------------------------------------------------------ parameter sets


def parse_sps(unit: bytes) -> dict:
    """Read a sequence parameter set NAL unit by the specification's syntax."""
    assert unit[0] & 0x1F == 7, 'not an SPS NAL unit'
    reader = BitReader(unescape(unit[1:]))
    found = {'profile_idc': reader.u(8)}
    found['constraint_flags'] = reader.u(8)
    found['level_idc'] = reader.u(8)
    found['seq_parameter_set_id'] = reader.ue()
    if found['profile_idc'] in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138,
                                139, 134, 135):
        found['chroma_format_idc'] = reader.ue()
        if found['chroma_format_idc'] == 3:
            found['separate_colour_plane_flag'] = reader.flag()
        found['bit_depth_luma_minus8'] = reader.ue()
        found['bit_depth_chroma_minus8'] = reader.ue()
        found['qpprime_y_zero_transform_bypass_flag'] = reader.flag()
        assert reader.flag() == 0, 'scaling matrices are not written here'
    found['log2_max_frame_num_minus4'] = reader.ue()
    found['pic_order_cnt_type'] = reader.ue()
    if found['pic_order_cnt_type'] == 0:
        found['log2_max_pic_order_cnt_lsb_minus4'] = reader.ue()
    elif found['pic_order_cnt_type'] == 1:
        raise AssertionError('pic_order_cnt_type 1 is not written here')
    found['max_num_ref_frames'] = reader.ue()
    found['gaps_in_frame_num_value_allowed_flag'] = reader.flag()
    found['pic_width_in_mbs_minus1'] = reader.ue()
    found['pic_height_in_map_units_minus1'] = reader.ue()
    found['frame_mbs_only_flag'] = reader.flag()
    if not found['frame_mbs_only_flag']:
        found['mb_adaptive_frame_field_flag'] = reader.flag()
    found['direct_8x8_inference_flag'] = reader.flag()
    found['frame_cropping_flag'] = reader.flag()
    if found['frame_cropping_flag']:
        found['crop'] = tuple(reader.ue() for _ in range(4))
    found['vui_parameters_present_flag'] = reader.flag()
    if found['vui_parameters_present_flag']:
        found['vui'] = parse_vui(reader)
    return found


def parse_vui(reader: BitReader) -> dict:
    """Read the video usability information that follows an SPS."""
    vui = {}
    if reader.flag():                                  # aspect_ratio_info
        vui['aspect_ratio_idc'] = reader.u(8)
        if vui['aspect_ratio_idc'] == 255:
            vui['sar'] = (reader.u(16), reader.u(16))
    if reader.flag():                                  # overscan_info
        vui['overscan_appropriate_flag'] = reader.flag()
    if reader.flag():                                  # video_signal_type
        vui['video_format'] = reader.u(3)
        vui['video_full_range_flag'] = reader.flag()
        if reader.flag():                              # colour_description
            vui['colour_primaries'] = reader.u(8)
            vui['transfer_characteristics'] = reader.u(8)
            vui['matrix_coefficients'] = reader.u(8)
    if reader.flag():                                  # chroma_loc_info
        vui['chroma_sample_loc_type_top_field'] = reader.ue()
        vui['chroma_sample_loc_type_bottom_field'] = reader.ue()
    if reader.flag():                                  # timing_info
        vui['num_units_in_tick'] = reader.u(32)
        vui['time_scale'] = reader.u(32)
        vui['fixed_frame_rate_flag'] = reader.flag()
    assert reader.flag() == 0, 'no NAL HRD is written here'
    assert reader.flag() == 0, 'no VCL HRD is written here'
    vui['pic_struct_present_flag'] = reader.flag()
    vui['bitstream_restriction_flag'] = reader.flag()
    if vui['bitstream_restriction_flag']:
        vui['motion_vectors_over_pic_boundaries_flag'] = reader.flag()
        vui['max_bytes_per_pic_denom'] = reader.ue()
        vui['max_bits_per_mb_denom'] = reader.ue()
        vui['log2_max_mv_length_horizontal'] = reader.ue()
        vui['log2_max_mv_length_vertical'] = reader.ue()
        vui['max_num_reorder_frames'] = reader.ue()
        vui['max_dec_frame_buffering'] = reader.ue()
    return vui


def rbsp_bits(data: bytes) -> int:
    """How many bits of syntax a raw byte sequence holds, before its stop bit."""
    trailing = data.rstrip(b'\x00')
    assert trailing, 'a raw byte sequence is never all zeroes'
    last = trailing[-1]
    return (len(trailing) - 1) * 8 + (7 - (last & -last).bit_length() + 1)


def parse_pps(unit: bytes) -> dict:
    """Read a picture parameter set NAL unit by the specification's syntax."""
    assert unit[0] & 0x1F == 8, 'not a PPS NAL unit'
    payload = unescape(unit[1:])
    reader = BitReader(payload)
    found = {
        'pic_parameter_set_id': reader.ue(),
        'seq_parameter_set_id': reader.ue(),
        'entropy_coding_mode_flag': reader.flag(),
        'bottom_field_pic_order_in_frame_present_flag': reader.flag(),
        'num_slice_groups_minus1': reader.ue(),
    }
    found['num_ref_idx_l0_default_active_minus1'] = reader.ue()
    found['num_ref_idx_l1_default_active_minus1'] = reader.ue()
    found['weighted_pred_flag'] = reader.flag()
    found['weighted_bipred_idc'] = reader.u(2)
    found['pic_init_qp_minus26'] = reader.se()
    found['pic_init_qs_minus26'] = reader.se()
    found['chroma_qp_index_offset'] = reader.se()
    found['deblocking_filter_control_present_flag'] = reader.flag()
    found['constrained_intra_pred_flag'] = reader.flag()
    found['redundant_pic_cnt_present_flag'] = reader.flag()
    if reader.position < rbsp_bits(payload):
        found['transform_8x8_mode_flag'] = reader.flag()
        found['pic_scaling_matrix_present_flag'] = reader.flag()
        found['second_chroma_qp_index_offset'] = reader.se()
    return found


#: A 1080p60 set, which is the shape most of these assertions are about.
def sets(**changes) -> ParameterSets:
    arguments = dict(width=1920, height=1080, frame_rate=(60, 1),
                     max_num_ref_frames=1, init_qp=26)
    arguments.update(changes)
    return ParameterSets(**arguments)


class TestSequenceParameterSet:
    def test_the_unit_is_an_sps_with_the_right_nal_header(self):
        unit = sets().sps()
        assert unit[0] == 0x67, 'nal_ref_idc 3, nal_unit_type 7'

    def test_it_declares_high_profile(self):
        found = parse_sps(sets().sps())
        assert found['profile_idc'] == 100
        assert found['chroma_format_idc'] == 1, '4:2:0'
        assert found['bit_depth_luma_minus8'] == 0
        assert found['bit_depth_chroma_minus8'] == 0

    def test_the_frame_size_is_stated_in_macroblocks(self):
        found = parse_sps(sets().sps())
        assert found['pic_width_in_mbs_minus1'] == 1920 // 16 - 1
        assert found['pic_height_in_map_units_minus1'] == 1080 // 16 - 1 + 1, (
            '1080 is not a multiple of 16, so the coded height is 1088')

    def test_a_height_that_is_not_a_multiple_of_16_is_cropped_back(self):
        found = parse_sps(sets().sps())
        assert found['frame_cropping_flag'] == 1
        # Cropping is in chroma units, two luma rows each for 4:2:0 frames.
        assert found['crop'] == (0, 0, 0, (1088 - 1080) // 2)

    def test_a_frame_that_fits_the_grid_exactly_is_not_cropped(self):
        found = parse_sps(sets(width=1280, height=720).sps())
        assert found['frame_cropping_flag'] == 0
        assert found['pic_width_in_mbs_minus1'] == 79
        assert found['pic_height_in_map_units_minus1'] == 44

    def test_a_width_that_is_not_a_multiple_of_16_is_cropped_back(self):
        found = parse_sps(sets(width=1918, height=1080).sps())
        assert found['crop'][:2] == (0, (1920 - 1918) // 2)

    def test_progressive_frames_only(self):
        found = parse_sps(sets().sps())
        assert found['frame_mbs_only_flag'] == 1

    def test_the_colour_description_is_limited_range_bt709(self):
        vui = parse_sps(sets().sps())['vui']
        assert vui['video_full_range_flag'] == 0, 'limited range'
        assert vui['colour_primaries'] == 1, 'BT.709'
        assert vui['transfer_characteristics'] == 1
        assert vui['matrix_coefficients'] == 1

    def test_the_frame_rate_is_written_as_a_tick_of_a_field(self):
        vui = parse_sps(sets(frame_rate=(60, 1)).sps())['vui']
        # time_scale counts fields, so it is twice the frame rate numerator.
        assert vui['time_scale'] == 120
        assert vui['num_units_in_tick'] == 1
        assert vui['fixed_frame_rate_flag'] == 1

    def test_a_broadcast_frame_rate_keeps_its_exact_ratio(self):
        vui = parse_sps(sets(frame_rate=(30000, 1001)).sps())['vui']
        assert (vui['time_scale'], vui['num_units_in_tick']) == (60000, 1001)

    def test_the_reordering_depth_is_declared(self):
        vui = parse_sps(sets().sps())['vui']
        assert vui['bitstream_restriction_flag'] == 1
        assert vui['max_num_reorder_frames'] == 0, 'no picture is held back'

    def test_the_level_covers_the_frame_size_and_rate(self):
        assert parse_sps(sets(width=1920, height=1080,
                              frame_rate=(30, 1)).sps())['level_idc'] == 40
        assert parse_sps(sets(width=1920, height=1080,
                              frame_rate=(60, 1)).sps())['level_idc'] == 42
        assert parse_sps(sets(width=640, height=480,
                              frame_rate=(30, 1)).sps())['level_idc'] == 30
        assert parse_sps(sets(width=3840, height=2160,
                              frame_rate=(60, 1)).sps())['level_idc'] == 52

    def test_a_bitrate_the_level_cannot_carry_raises_the_level(self):
        # 1080p30 fits level 4.0 on size and rate alone; 4.0 carries 20 Mbit/s
        # for Main and 25 for High, so 40 Mbit/s has to go higher.
        assert parse_sps(sets(frame_rate=(30, 1), bitrate=8_000_000)
                         .sps())['level_idc'] == 40
        assert parse_sps(sets(frame_rate=(30, 1), bitrate=40_000_000)
                         .sps())['level_idc'] == 41

    def test_a_bitrate_beyond_every_level_still_produces_a_stream(self):
        assert parse_sps(sets(bitrate=10_000_000_000).sps())['level_idc'] == 62

    def test_the_frame_number_field_is_wide_enough_for_the_gop(self):
        found = parse_sps(sets(max_num_ref_frames=4).sps())
        assert found['max_num_ref_frames'] == 4
        assert found['log2_max_frame_num_minus4'] >= 0

    def test_no_start_code_can_appear_inside_the_unit(self):
        for width, height in ((1920, 1080), (256, 256), (64, 64), (3840, 2160)):
            unit = sets(width=width, height=height).sps()
            assert b'\x00\x00\x01' not in unit
            assert b'\x00\x00\x00' not in unit


class TestPictureParameterSet:
    def test_the_unit_is_a_pps_with_the_right_nal_header(self):
        assert sets().pps()[0] == 0x68, 'nal_ref_idc 3, nal_unit_type 8'

    def test_it_refers_to_the_sequence_parameter_set(self):
        found = parse_pps(sets().pps())
        assert found['pic_parameter_set_id'] == 0
        assert found['seq_parameter_set_id'] == 0

    def test_entropy_coding_is_cabac_for_high_profile(self):
        assert parse_pps(sets().pps())['entropy_coding_mode_flag'] == 1

    def test_the_initial_quantiser_is_carried_across(self):
        assert parse_pps(sets(init_qp=26).pps())['pic_init_qp_minus26'] == 0
        assert parse_pps(sets(init_qp=18).pps())['pic_init_qp_minus26'] == -8
        assert parse_pps(sets(init_qp=40).pps())['pic_init_qp_minus26'] == 14

    def test_deblocking_control_is_present_so_a_slice_may_set_it(self):
        assert parse_pps(sets().pps())['deblocking_filter_control_present_flag'] == 1

    def test_one_reference_is_active_by_default(self):
        found = parse_pps(sets().pps())
        assert found['num_ref_idx_l0_default_active_minus1'] == 0
        assert found['num_ref_idx_l1_default_active_minus1'] == 0

    def test_the_8x8_transform_is_declared_when_the_encoder_may_use_it(self):
        found = parse_pps(sets(transform_8x8=True).pps())
        assert found['transform_8x8_mode_flag'] == 1
        assert found['pic_scaling_matrix_present_flag'] == 0
        assert found['second_chroma_qp_index_offset'] == 0

    def test_the_set_ends_early_when_the_8x8_transform_is_off(self):
        assert 'transform_8x8_mode_flag' not in parse_pps(
            sets(transform_8x8=False).pps())


class TestHeaders:
    def test_headers_are_the_two_sets_in_annex_b_order(self):
        built = sets()
        assert built.headers() == annexb(built.sps(), built.pps())

    def test_the_two_sets_are_stable_across_calls(self):
        built = sets()
        assert built.sps() == built.sps()
        assert built.pps() == built.pps()


class TestRefusedSizes:
    @pytest.mark.parametrize('width,height', [(0, 480), (640, 0), (-16, 16)])
    def test_a_degenerate_size_is_refused(self, width, height):
        with pytest.raises(ValueError):
            sets(width=width, height=height)

    def test_an_odd_size_is_refused_because_chroma_is_subsampled(self):
        with pytest.raises(ValueError):
            sets(width=641, height=480)


# ------------------------------------------------------------- slice headers


def parse_slice_header(data: bytes, sets: ParameterSets) -> dict:
    """Read a packed slice header: start code, NAL header, then the syntax.

    Follows section 7.3.3 for the subset this backend writes -- one slice a
    picture, no slice groups, picture order count type zero.
    """
    assert data[:4] == b'\x00\x00\x00\x01', 'a packed header carries a start code'
    unit = unescape(data[4:])
    found = {'nal_ref_idc': (unit[0] >> 5) & 3, 'nal_unit_type': unit[0] & 0x1F}
    reader = BitReader(unit[1:])
    idr = found['nal_unit_type'] == 5
    found['first_mb_in_slice'] = reader.ue()
    found['slice_type'] = reader.ue()
    found['pic_parameter_set_id'] = reader.ue()
    found['frame_num'] = reader.u(sets.log2_max_frame_num)
    if idr:
        found['idr_pic_id'] = reader.ue()
    found['pic_order_cnt_lsb'] = reader.u(sets.log2_max_poc_lsb)
    predicted = found['slice_type'] % 5 == 0
    if predicted:
        found['num_ref_idx_active_override_flag'] = reader.flag()
        if found['num_ref_idx_active_override_flag']:
            found['num_ref_idx_l0_active_minus1'] = reader.ue()
        found['ref_pic_list_modification_flag_l0'] = reader.flag()
    if found['nal_ref_idc']:
        if idr:
            found['no_output_of_prior_pics_flag'] = reader.flag()
            found['long_term_reference_flag'] = reader.flag()
        else:
            found['adaptive_ref_pic_marking_mode_flag'] = reader.flag()
    if predicted:                                  # CABAC, and not an I slice
        found['cabac_init_idc'] = reader.ue()
    found['slice_qp_delta'] = reader.se()
    found['disable_deblocking_filter_idc'] = reader.ue()
    if found['disable_deblocking_filter_idc'] != 1:
        found['slice_alpha_c0_offset_div2'] = reader.se()
        found['slice_beta_offset_div2'] = reader.se()
    found['bits_read'] = reader.position + 8 + 32   # NAL header and start code
    return found


def pictures(gop=8, count=2, **changes):
    structure = GroupOfPictures(gop=gop, **changes)
    return [structure.next_picture() for _ in range(count)]


class TestSliceHeader:
    """The header libva's drivers leave to the caller, on the AMD path.

    Mesa's encoder writes the slice *data* and nothing in front of it, so the
    NAL header and the whole of section 7.3.3 are written here. A field packed
    one bit off makes a picture no player decodes, and reports nothing.
    """

    def test_an_idr_is_a_type_five_unit_at_the_highest_reference_priority(self):
        built = sets()
        header = parse_slice_header(built.slice_header(pictures()[0]), built)
        assert header['nal_unit_type'] == 5
        assert header['nal_ref_idc'] == 3

    def test_a_predicted_picture_is_a_type_one_unit(self):
        built = sets()
        header = parse_slice_header(built.slice_header(pictures()[1]), built)
        assert header['nal_unit_type'] == 1
        assert header['nal_ref_idc'] == 3, 'later pictures predict from it'

    def test_the_whole_picture_is_one_slice(self):
        built = sets()
        header = parse_slice_header(built.slice_header(pictures()[0]), built)
        assert header['first_mb_in_slice'] == 0

    def test_slice_types_say_the_whole_picture_is_of_that_type(self):
        built = sets()
        idr, predicted = pictures()
        assert parse_slice_header(built.slice_header(idr), built)['slice_type'] == 7
        assert parse_slice_header(
            built.slice_header(predicted), built)['slice_type'] == 5

    def test_the_counters_come_from_the_picture(self):
        built = sets()
        first, second = pictures()
        opened = parse_slice_header(built.slice_header(first), built)
        assert opened['frame_num'] == 0
        assert opened['pic_order_cnt_lsb'] == 0
        assert opened['idr_pic_id'] == first.idr_pic_id
        later = parse_slice_header(built.slice_header(second), built)
        assert later['frame_num'] == 1
        assert later['pic_order_cnt_lsb'] == 2

    def test_the_picture_order_count_wraps_within_its_field(self):
        built = sets()
        structure = GroupOfPictures(gop=1000)
        far = [structure.next_picture() for _ in range(200)][-1]
        header = parse_slice_header(built.slice_header(far), built)
        assert far.poc == 398
        assert header['pic_order_cnt_lsb'] == 398 % (1 << built.log2_max_poc_lsb)

    def test_an_idr_marks_itself_as_starting_a_new_sequence(self):
        built = sets()
        header = parse_slice_header(built.slice_header(pictures()[0]), built)
        assert header['no_output_of_prior_pics_flag'] == 0
        assert header['long_term_reference_flag'] == 0, 'references are short term'

    def test_a_predicted_picture_uses_the_sliding_window(self):
        built = sets()
        header = parse_slice_header(built.slice_header(pictures()[1]), built)
        assert header['adaptive_ref_pic_marking_mode_flag'] == 0

    def test_a_predicted_picture_states_its_one_active_reference(self):
        built = sets()
        header = parse_slice_header(built.slice_header(pictures()[1]), built)
        assert header['num_ref_idx_active_override_flag'] == 1
        assert header['num_ref_idx_l0_active_minus1'] == 0
        assert header['ref_pic_list_modification_flag_l0'] == 0

    def test_deblocking_is_on_across_slice_edges(self):
        built = sets()
        header = parse_slice_header(built.slice_header(pictures()[0]), built)
        assert header['disable_deblocking_filter_idc'] == 0
        assert header['slice_alpha_c0_offset_div2'] == 0
        assert header['slice_beta_offset_div2'] == 0

    def test_the_quantiser_is_the_one_the_picture_set_starts_from(self):
        built = sets(init_qp=30)
        header = parse_slice_header(built.slice_header(pictures()[0]), built)
        assert header['slice_qp_delta'] == 0

    def test_a_caller_may_move_the_quantiser_off_the_initial_one(self):
        built = sets(init_qp=30)
        header = parse_slice_header(
            built.slice_header(pictures()[0], qp=22), built)
        assert header['slice_qp_delta'] == 22 - 30

    def test_the_length_in_bits_is_reported_with_the_bytes(self):
        built = sets()
        data, bits = built.packed_slice_header(pictures()[0])
        assert data == built.slice_header(pictures()[0])
        assert bits <= len(data) * 8
        assert bits > len(data) * 8 - 8, 'at most a byte of padding'
        assert parse_slice_header(data, built)['bits_read'] <= bits

    def test_no_start_code_can_appear_inside_a_slice_header(self):
        built = sets()
        for picture in pictures(gop=4, count=4):
            data = built.slice_header(picture)
            assert b'\x00\x00\x01' not in data[4:]

    def test_the_parameter_sets_go_out_together(self):
        built = sets()
        data, bits = built.packed_parameter_sets()
        assert data == annexb(built.sps(), built.pps())
        assert bits == len(data) * 8
