"""The libva VP9 decode buffers, built from parsed headers with no driver.

A field the driver is told wrongly is not reported: the frame decodes into
the wrong picture. Each buffer is built here from headers whose values are
known and read back field by field.
"""
import pytest

from pyopengl_video.colour import MATRIX_BT601, MATRIX_BT709, MATRIX_BT2020
from pyopengl_video.decoder import DecoderError
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.vp9_decode import (
    colour_of,
    filter_levels,
    picture_parameters,
    profile_for,
    quantiser_scales,
    slice_parameters,
)
from pyopengl_video.vaapi.vp9_parse import (
    CS_BT_601,
    CS_BT_709,
    CS_BT_2020,
    CS_SMPTE_170,
    CS_UNKNOWN,
    SEG_LVL_ALT_L,
    SEG_LVL_ALT_Q,
    SEG_LVL_REF_FRAME,
    SEG_LVL_SKIP,
    DecodingState,
    parse_uncompressed_header,
)
from pyopengl_video.vaapi.vp9_tables import AC_QLOOKUP, DC_QLOOKUP
from tests.vp9_syntax import INTER, frame

SLOTS = (20, 21, 22, 23, 24, 25, 26, 27)


def header_of(*frames):
    """The header of the last of `frames`, each parsed after the one before."""
    state = DecodingState()
    for data in frames:
        header = parse_uncompressed_header(data, state)
    return header


class TestProfiles:
    def test_profile_0_decodes_as_8_bit(self):
        assert profile_for(header_of(frame())) == api.VAProfileVP9Profile0

    def test_profile_2_decodes_at_10_bits(self):
        assert profile_for(header_of(frame(profile=2, depth=10))) == api.VAProfileVP9Profile2

    def test_12_bits_are_refused_by_name(self):
        with pytest.raises(DecoderError, match='12 bits'):
            profile_for(header_of(frame(profile=2, depth=12)))

    @pytest.mark.parametrize(('profile', 'subsampling', 'named'), [
        (1, (0, 0), '4:4:4'), (1, (1, 0), '4:2:2'), (1, (0, 1), '4:4:0'),
        (3, (0, 0), '4:4:4')])
    def test_other_chroma_layouts_are_refused_by_name(self, profile, subsampling, named):
        header = header_of(frame(profile=profile, depth=10, subsampling=subsampling))
        with pytest.raises(DecoderError, match=f'profile {profile}.*{named}'):
            profile_for(header)


class TestPictureParameters:
    def inter(self, **options):
        return header_of(frame(size=(122, 70)), frame(
            frame_type=INTER, refs=(5, 0, 3), sign_bias=(0, 1, 1), **options))

    def test_the_frame_size_is_the_stream_s_own(self):
        parameters = picture_parameters(self.inter(), SLOTS)
        assert (parameters.frame_width, parameters.frame_height) == (122, 70)

    def test_every_reference_slot_names_its_surface(self):
        slots = (20, None, 22, 23, None, 25, 26, 27)
        parameters = picture_parameters(self.inter(), slots)
        assert list(parameters.reference_frames) == [
            20, api.VA_INVALID_SURFACE, 22, 23, api.VA_INVALID_SURFACE, 25, 26, 27]

    def test_the_three_references_are_slot_numbers_with_their_sign_bias(self):
        bits = picture_parameters(self.inter(), SLOTS).pic_fields.bits
        assert (bits.last_ref_frame, bits.golden_ref_frame, bits.alt_ref_frame) == (5, 0, 3)
        assert (bits.last_ref_frame_sign_bias, bits.golden_ref_frame_sign_bias,
                bits.alt_ref_frame_sign_bias) == (0, 1, 1)

    def test_the_frame_flags(self):
        header = self.inter(show_frame=0, reset_frame_context=2, high_precision=1,
                            refresh_context=1, parallel=0, context=3)
        bits = picture_parameters(header, SLOTS).pic_fields.bits
        assert (bits.subsampling_x, bits.subsampling_y) == (1, 1)
        assert bits.frame_type == 1
        assert bits.show_frame == 0
        assert bits.error_resilient_mode == 0
        assert bits.intra_only == 0
        assert bits.allow_high_precision_mv == 1
        assert bits.frame_parallel_decoding_mode == 0
        assert bits.reset_frame_context == 2
        assert bits.refresh_frame_context == 1
        assert bits.frame_context_idx == 3
        assert bits.lossless_flag == 0

    def test_a_key_frame(self):
        bits = picture_parameters(header_of(frame(context=2)), SLOTS).pic_fields.bits
        assert bits.frame_type == 0
        assert bits.show_frame == 1
        assert bits.frame_context_idx == 0      # an intra frame decodes with context 0
        assert (bits.last_ref_frame, bits.golden_ref_frame, bits.alt_ref_frame) == (0, 0, 0)

    def test_an_error_resilient_frame(self):
        bits = picture_parameters(self.inter(error_resilient=1), SLOTS).pic_fields.bits
        assert bits.error_resilient_mode == 1
        assert bits.frame_parallel_decoding_mode == 1
        assert bits.refresh_frame_context == 0

    @pytest.mark.parametrize(('raw', 'filter_type'), [
        (1, 0), (0, 1), (2, 2), (3, 3), (None, 4)])
    def test_the_interpolation_filter_is_the_decoded_one(self, raw, filter_type):
        """Regular, smooth, sharp, bilinear, then switchable: not the raw field."""
        bits = picture_parameters(self.inter(filter=raw), SLOTS).pic_fields.bits
        assert bits.mcomp_filter_type == filter_type

    def test_the_loop_filter_and_the_tiles(self):
        header = header_of(frame(size=(1920, 64), level=41, sharpness=6,
                                 tile_columns=(1, 1), tile_rows=2))
        parameters = picture_parameters(header, SLOTS)
        assert (parameters.filter_level, parameters.sharpness_level) == (41, 6)
        assert (parameters.log2_tile_columns, parameters.log2_tile_rows) == (2, 2)

    def test_the_two_header_lengths(self):
        header = header_of(frame(compressed=300))
        parameters = picture_parameters(header, SLOTS)
        assert parameters.frame_header_length_in_bytes == header.uncompressed_size == 14
        assert parameters.first_partition_size == 300

    def test_the_profile_and_bit_depth(self):
        parameters = picture_parameters(header_of(frame(profile=2, depth=10)), SLOTS)
        assert (parameters.profile, parameters.bit_depth) == (2, 10)
        parameters = picture_parameters(header_of(frame()), SLOTS)
        assert (parameters.profile, parameters.bit_depth) == (0, 8)

    def test_lossless(self):
        bits = picture_parameters(header_of(frame(base_q=0)), SLOTS).pic_fields.bits
        assert bits.lossless_flag == 1

    def test_segmentation_with_its_map_probabilities(self):
        header = header_of(frame(segmentation={
            'tree': (1, 2, None, 4, 5, 6, 7), 'pred': (8, None, 10), 'features': {}}))
        parameters = picture_parameters(header, SLOTS)
        bits = parameters.pic_fields.bits
        assert bits.segmentation_enabled == 1
        assert bits.segmentation_update_map == 1
        assert bits.segmentation_temporal_update == 1
        assert list(parameters.mb_segment_tree_probs) == [1, 2, 255, 4, 5, 6, 7]
        assert list(parameters.segment_pred_probs) == [8, 255, 10]

    def test_no_segmentation(self):
        bits = picture_parameters(header_of(frame()), SLOTS).pic_fields.bits
        assert (bits.segmentation_enabled, bits.segmentation_update_map,
                bits.segmentation_temporal_update) == (0, 0, 0)

    def test_an_intra_only_frame(self):
        header = header_of(frame(), frame(frame_type=INTER, show_frame=0, intra_only=1))
        bits = picture_parameters(header, SLOTS).pic_fields.bits
        assert (bits.frame_type, bits.intra_only, bits.show_frame) == (1, 1, 0)


class TestFilterLevels:
    def test_no_loop_filter_gives_no_levels(self):
        header = header_of(frame(level=0, deltas=({0: 5}, {})))
        assert filter_levels(header, 0) == ((0, 0),) * 4

    def test_without_deltas_every_entry_is_the_frame_level(self):
        assert filter_levels(header_of(frame(level=20)), 0) == ((20, 20),) * 4

    def test_deltas_by_reference_and_mode(self):
        """Intra uses its reference delta alone; the rest add a mode delta."""
        header = header_of(frame(level=20, deltas=({0: 3, 1: 0, 2: -4, 3: 7}, {0: 2, 1: -1})))
        assert filter_levels(header, 0) == ((23, 0), (22, 19), (18, 15), (29, 26))

    def test_deltas_count_double_from_level_32(self):
        header = header_of(frame(level=40, deltas=({0: 3, 2: -4, 3: 7}, {0: 2, 1: -1})))
        assert filter_levels(header, 0) == ((46, 0), (44, 38), (36, 30), (58, 52))

    def test_levels_stay_between_0_and_63(self):
        header = header_of(frame(level=60, deltas=({0: 9, 2: -63}, {})))
        levels = filter_levels(header, 0)
        assert levels[0][0] == 63
        assert levels[2] == (0, 0)

    def test_a_segment_s_level_is_added_to_the_frame_s(self):
        header = header_of(frame(level=20, segmentation={
            'features': {(3, SEG_LVL_ALT_L): -6, (4, SEG_LVL_ALT_L): 63}}))
        assert filter_levels(header, 3) == ((14, 14),) * 4
        assert filter_levels(header, 4) == ((63, 63),) * 4
        assert filter_levels(header, 0) == ((20, 20),) * 4

    def test_a_segment_s_level_replaces_the_frame_s_when_absolute(self):
        header = header_of(frame(level=20, segmentation={
            'abs': 1, 'features': {(3, SEG_LVL_ALT_L): 9}}))
        assert filter_levels(header, 3) == ((9, 9),) * 4

    def test_the_doubling_follows_the_frame_s_level_not_the_segment_s(self):
        header = header_of(frame(level=20, deltas=({0: 3}, {}), segmentation={
            'features': {(1, SEG_LVL_ALT_L): 30}}))
        assert filter_levels(header, 1)[0][0] == 53


class TestQuantiserScales:
    def test_the_scales_at_the_frame_s_index(self):
        header = header_of(frame(base_q=60, q_deltas=(-4, 5, -6)))
        assert quantiser_scales(header, 0) == (
            AC_QLOOKUP[8][60], DC_QLOOKUP[8][56], AC_QLOOKUP[8][54], DC_QLOOKUP[8][65])

    def test_known_table_entries(self):
        """The ends of each table, as the specification prints them."""
        assert (DC_QLOOKUP[8][0], DC_QLOOKUP[8][255]) == (4, 1336)
        assert (AC_QLOOKUP[8][0], AC_QLOOKUP[8][255]) == (4, 1828)
        assert (DC_QLOOKUP[10][0], DC_QLOOKUP[10][255]) == (4, 5347)
        assert (AC_QLOOKUP[10][0], AC_QLOOKUP[10][255]) == (4, 7312)
        assert all(len(table) == 256 for tables in (DC_QLOOKUP, AC_QLOOKUP)
                   for table in tables.values())

    def test_ten_bit_frames_use_the_ten_bit_tables(self):
        header = header_of(frame(profile=2, depth=10, base_q=200))
        assert quantiser_scales(header, 0) == (
            AC_QLOOKUP[10][200], DC_QLOOKUP[10][200], AC_QLOOKUP[10][200],
            DC_QLOOKUP[10][200])

    def test_an_index_with_its_delta_stays_between_0_and_255(self):
        header = header_of(frame(base_q=250, q_deltas=(15, 15, 15)))
        assert quantiser_scales(header, 0) == (
            AC_QLOOKUP[8][250], DC_QLOOKUP[8][255], AC_QLOOKUP[8][255], DC_QLOOKUP[8][255])
        header = header_of(frame(base_q=3, q_deltas=(-15, 0, 0)))
        assert quantiser_scales(header, 0)[1] == DC_QLOOKUP[8][0]

    def test_a_segment_s_index_is_added_to_the_frame_s(self):
        header = header_of(frame(base_q=60, segmentation={
            'features': {(2, SEG_LVL_ALT_Q): -24, (5, SEG_LVL_ALT_Q): 255}}))
        assert quantiser_scales(header, 2)[0] == AC_QLOOKUP[8][36]
        assert quantiser_scales(header, 5)[0] == AC_QLOOKUP[8][255]
        assert quantiser_scales(header, 0)[0] == AC_QLOOKUP[8][60]

    def test_a_segment_s_index_replaces_the_frame_s_when_absolute(self):
        header = header_of(frame(base_q=60, q_deltas=(2, 0, 0), segmentation={
            'abs': 1, 'features': {(2, SEG_LVL_ALT_Q): 100}}))
        assert quantiser_scales(header, 2)[:2] == (AC_QLOOKUP[8][100], DC_QLOOKUP[8][102])


class TestSliceParameters:
    def test_the_data_is_the_whole_frame(self):
        parameters = slice_parameters(header_of(frame()), 1234)
        assert parameters.slice_data_size == 1234
        assert parameters.slice_data_offset == 0
        assert parameters.slice_data_flag == api.VA_SLICE_DATA_FLAG_ALL

    def test_without_segmentation_only_the_first_segment_is_filled(self):
        header = header_of(frame(level=20, base_q=60))
        parameters = slice_parameters(header, 100)
        first = parameters.seg_param[0]
        assert [tuple(row) for row in first.filter_level] == [(20, 20)] * 4
        assert (first.luma_ac_quant_scale, first.luma_dc_quant_scale,
                first.chroma_ac_quant_scale, first.chroma_dc_quant_scale) == (
            AC_QLOOKUP[8][60], DC_QLOOKUP[8][60], AC_QLOOKUP[8][60], DC_QLOOKUP[8][60])
        assert first.segment_flags.value == 0
        assert bytes(parameters.seg_param[1]) == bytes(len(bytes(first)))
        assert bytes(parameters.seg_param[7]) == bytes(len(bytes(first)))

    def test_every_segment_carries_its_own_levels_and_scales(self):
        header = header_of(frame(level=20, base_q=60, segmentation={'features': {
            (1, SEG_LVL_ALT_Q): 40, (1, SEG_LVL_ALT_L): -5,
            (6, SEG_LVL_REF_FRAME): 2, (7, SEG_LVL_SKIP): 0}}))
        parameters = slice_parameters(header, 100)
        segment = parameters.seg_param[1]
        assert [tuple(row) for row in segment.filter_level] == [(15, 15)] * 4
        assert segment.luma_ac_quant_scale == AC_QLOOKUP[8][100]
        assert segment.chroma_dc_quant_scale == DC_QLOOKUP[8][100]
        plain = parameters.seg_param[3]
        assert [tuple(row) for row in plain.filter_level] == [(20, 20)] * 4
        assert plain.luma_ac_quant_scale == AC_QLOOKUP[8][60]

    def test_the_reference_and_skip_features(self):
        header = header_of(frame(segmentation={'features': {
            (6, SEG_LVL_REF_FRAME): 2, (7, SEG_LVL_SKIP): 0}}))
        parameters = slice_parameters(header, 100)
        referenced = parameters.seg_param[6].segment_flags.fields
        assert (referenced.segment_reference_enabled, referenced.segment_reference,
                referenced.segment_reference_skipped) == (1, 2, 0)
        skipped = parameters.seg_param[7].segment_flags.fields
        assert (skipped.segment_reference_enabled, skipped.segment_reference_skipped) == (0, 1)


class TestColour:
    @pytest.mark.parametrize(('color_space', 'matrix'), [
        (CS_BT_601, MATRIX_BT601), (CS_SMPTE_170, MATRIX_BT601),
        (CS_BT_709, MATRIX_BT709), (CS_BT_2020, MATRIX_BT2020)])
    def test_the_matrix_a_colour_space_names(self, color_space, matrix):
        assert colour_of(header_of(frame(color_space=color_space))).matrix == matrix

    def test_an_unknown_colour_space_is_chosen_by_height(self):
        small = colour_of(header_of(frame(color_space=CS_UNKNOWN, size=(160, 90))))
        large = colour_of(header_of(frame(color_space=CS_UNKNOWN, size=(1280, 720))))
        assert (small.matrix, large.matrix) == (MATRIX_BT601, MATRIX_BT709)

    def test_the_range(self):
        assert colour_of(header_of(frame(color_range=1))).full_range
        assert not colour_of(header_of(frame(color_range=0))).full_range
