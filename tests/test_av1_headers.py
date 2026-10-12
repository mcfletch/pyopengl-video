"""The AV1 headers the encoder writes, read back by the decoder's parser.

No driver is involved. Every header :mod:`pyopengl_video.vaapi.av1` writes is
parsed with :mod:`pyopengl_video.vaapi.av1_parse`, and each offset libva asks
for is checked by rewriting the bits at it and reading the named field change.
"""
import pytest

from pyopengl_video import obu
from pyopengl_video.vaapi import av1
from pyopengl_video.vaapi.av1_parse import (
    ALL_FRAMES,
    EIGHTTAP,
    IDENTITY,
    INTER_FRAME,
    KEY_FRAME,
    PRIMARY_REF_NONE,
    TX_MODE_LARGEST,
    TX_MODE_SELECT,
    FrameState,
    parse_frame_header,
    parse_sequence_header,
)
from pyopengl_video.vaapi.bits import read_leb128

SIZE = (160, 90)


def sequence(size=SIZE, frame_rate=(30, 1), bitrate=1_000_000, **options):
    return av1.Sequence(*size, frame_rate, bitrate, **options)


def only_unit(data):
    """The one OBU `data` holds."""
    units = list(obu.split(data))
    assert len(units) == 1
    assert units[0].raw == data
    return units[0]


def key_frame(base_q_idx=130):
    return av1.Frame(key=True, number=0, base_q_idx=base_q_idx)


def inter_frame(number=1, base_q_idx=130):
    return av1.Frame(key=False, number=number, base_q_idx=base_q_idx)


def parsed(stream, packed, state=None, padding=b''):
    """The frame header in `packed`, read against `state`."""
    unit = next(obu.split(packed.data))
    return parse_frame_header(unit.payload + padding,
                              parse_sequence_header(only_unit(stream.sequence_header()).payload),
                              state or FrameState())


def after_key_frame(stream):
    """The reference slots as a key frame of `stream` leaves them."""
    state = FrameState()
    header = parsed(stream, stream.frame_header(key_frame()), state)
    state.update(header, parse_sequence_header(only_unit(stream.sequence_header()).payload))
    return state


def rewritten(packed, offset, bits, value):
    """`packed.data` with the `bits` at bit `offset` holding `value`."""
    total = len(packed.data) * 8
    number = int.from_bytes(packed.data, 'big')
    shift = total - offset - bits
    number &= ~(((1 << bits) - 1) << shift)
    number |= value << shift
    return number.to_bytes(len(packed.data), 'big')


def field(data, offset, bits):
    """The `bits` at bit `offset` of `data`."""
    return (int.from_bytes(data, 'big') >> (len(data) * 8 - offset - bits)) & ((1 << bits) - 1)


class TestSequenceHeader:
    def test_it_is_one_sequence_header_obu_with_a_size(self):
        unit = only_unit(sequence().sequence_header())
        assert unit.kind == obu.SEQUENCE_HEADER
        assert not unit.has_extension

    def test_the_profile_and_the_picture_size(self):
        header = parse_sequence_header(only_unit(sequence().sequence_header()).payload)
        assert header.seq_profile == 0
        assert not header.still_picture
        assert not header.reduced_still_picture_header
        assert header.size == SIZE

    def test_eight_bit_420_limited_range_bt709(self):
        color = parse_sequence_header(
            only_unit(sequence().sequence_header()).payload).color_config
        assert color.bit_depth == 8
        assert not color.mono_chrome
        assert (color.subsampling_x, color.subsampling_y) == (1, 1)
        assert color.color_description_present_flag
        assert (color.color_primaries, color.transfer_characteristics,
                color.matrix_coefficients) == (1, 1, 1)
        assert color.color_range == 0
        assert not color.separate_uv_delta_q

    def test_the_tools_no_frame_header_here_has_syntax_for_are_off(self):
        header = parse_sequence_header(only_unit(sequence().sequence_header()).payload)
        for name in ('use_128x128_superblock', 'enable_filter_intra',
                     'enable_intra_edge_filter', 'enable_interintra_compound',
                     'enable_masked_compound', 'enable_warped_motion',
                     'enable_dual_filter', 'enable_jnt_comp', 'enable_ref_frame_mvs',
                     'enable_superres', 'enable_restoration',
                     'film_grain_params_present', 'frame_id_numbers_present_flag',
                     'seq_force_screen_content_tools'):
            assert getattr(header, name) == 0, name
        assert header.enable_order_hint
        assert header.order_hint_bits == av1.ORDER_HINT_BITS
        assert header.enable_cdef

    def test_the_frame_rate_is_the_display_tick(self):
        header = parse_sequence_header(
            only_unit(sequence(frame_rate=(30000, 1001)).sequence_header()).payload)
        assert header.timing_info is not None
        assert header.timing_info.num_units_in_display_tick == 1001
        assert header.timing_info.time_scale == 30000
        assert not header.timing_info.equal_picture_interval
        assert header.decoder_model_info is None

    def test_one_operating_point_at_the_level_of_the_stream(self):
        stream = sequence((1920, 1080), bitrate=8_000_000)
        header = parse_sequence_header(only_unit(stream.sequence_header()).payload)
        assert len(header.operating_points) == 1
        point = header.operating_points[0]
        assert point.idc == 0
        assert (point.seq_level_idx, point.seq_tier) == (stream.level, stream.tier) == (8, 0)

    def test_the_trailing_bit_lies_in_the_last_byte(self):
        payload = only_unit(sequence().sequence_header()).payload
        assert payload[-1] != 0

    @pytest.mark.parametrize('size', [(16, 16), (2, 2), (4096, 2160), (8192, 4352)])
    def test_every_size_reads_back(self, size):
        header = parse_sequence_header(
            only_unit(sequence(size, bitrate=20_000_000).sequence_header()).payload)
        assert header.size == size

    def test_an_odd_size_is_refused(self):
        with pytest.raises(ValueError, match='161x90'):
            sequence((161, 90))


class TestLevel:
    @pytest.mark.parametrize('size, frame_rate, bitrate, expected', [
        ((160, 90), (30, 1), 1_000_000, (0, 0)),
        ((640, 360), (30, 1), 2_000_000, (1, 0)),
        ((1280, 720), (30, 1), 4_000_000, (5, 0)),
        ((1920, 1080), (30, 1), 8_000_000, (8, 0)),
        ((1920, 1080), (60, 1), 8_000_000, (9, 0)),
        ((3840, 2160), (30, 1), 20_000_000, (12, 0)),
        ((3840, 2160), (60, 1), 20_000_000, (13, 0)),
    ])
    def test_the_lowest_level_the_stream_fits(self, size, frame_rate, bitrate, expected):
        assert av1.level_for(*size, frame_rate, bitrate) == expected

    def test_a_bitrate_over_the_main_tier_takes_the_high_tier(self):
        assert av1.level_for(1920, 1080, (30, 1), 25_000_000) == (8, 1)

    def test_a_level_without_a_high_tier_gives_way_to_the_next(self):
        # Level 2.0 carries 1.5 Mbit/s and has no high tier; 2.1 carries 3.
        assert av1.level_for(160, 90, (30, 1), 2_000_000) == (1, 0)

    def test_what_no_level_holds_is_declared_unconstrained(self):
        assert av1.level_for(160, 90, (30, 1), 2_000_000_000) == (av1.LEVEL_MAXIMUM, 0)


class TestQuantiser:
    def test_five_indices_to_one_h264_step(self):
        assert av1.base_q_idx(26) == 130
        assert av1.base_q_idx(51) == 255

    def test_it_never_reaches_the_lossless_index(self):
        assert av1.base_q_idx(0) == 1

    def test_it_is_clamped_to_the_scale(self):
        assert av1.base_q_idx(80) == 255
        assert av1.base_q_idx(-4) == 1

    def test_it_rises_with_the_quantiser(self):
        found = [av1.base_q_idx(qp) for qp in range(52)]
        assert found == sorted(set(found))


class TestKeyFrameHeader:
    def test_a_frame_obu_by_default(self):
        packed = sequence().frame_header(key_frame())
        unit = next(obu.split(packed.data))
        assert unit.kind == obu.FRAME
        assert unit.raw == packed.data

    def test_a_frame_header_obu_when_asked(self):
        packed = sequence().frame_header(key_frame(), frame_obu=False)
        assert only_unit(packed.data).kind == obu.FRAME_HEADER

    def test_it_is_a_shown_key_frame_refreshing_every_slot(self):
        stream = sequence()
        header = parsed(stream, stream.frame_header(key_frame()))
        assert not header.show_existing_frame
        assert header.frame_type == KEY_FRAME
        assert header.show_frame
        assert header.error_resilient_mode
        assert header.refresh_frame_flags == ALL_FRAMES
        assert header.primary_ref_frame == PRIMARY_REF_NONE
        assert header.order_hint == 0

    def test_the_frame_is_the_sequence_size_and_renders_at_the_picture_size(self):
        stream = sequence()
        header = parsed(stream, stream.frame_header(key_frame()))
        assert not header.frame_size_override_flag
        assert header.size == SIZE
        assert (header.render_width, header.render_height) == SIZE

    def test_the_quantiser(self):
        stream = sequence()
        header = parsed(stream, stream.frame_header(key_frame(base_q_idx=77)))
        quantization = header.quantization
        assert quantization.base_q_idx == 77
        assert (quantization.delta_q_y_dc, quantization.delta_q_u_dc,
                quantization.delta_q_u_ac) == (0, 0, 0)
        assert not quantization.using_qmatrix
        assert not header.segmentation.enabled
        assert not header.delta_q_present

    def test_the_loop_filter_rises_with_the_quantiser(self):
        stream = sequence()
        low = parsed(stream, stream.frame_header(key_frame(base_q_idx=20))).loop_filter
        high = parsed(stream, stream.frame_header(key_frame(base_q_idx=250))).loop_filter
        assert low.level == (av1.loop_filter_level(20),) * 4
        assert high.level == (av1.loop_filter_level(250),) * 4
        assert 0 < low.level[0] < high.level[0] <= 63
        assert not high.delta_enabled

    def test_cdef_is_present_and_does_nothing(self):
        stream = sequence()
        cdef = parsed(stream, stream.frame_header(key_frame())).cdef
        assert (cdef.damping_minus_3, cdef.bits) == (0, 0)
        assert (cdef.y_pri_strength, cdef.y_sec_strength) == ((0,), (0,))
        assert (cdef.uv_pri_strength, cdef.uv_sec_strength) == ((0,), (0,))

    @pytest.mark.parametrize('select, expected', [(True, TX_MODE_SELECT),
                                                  (False, TX_MODE_LARGEST)])
    def test_the_transform_mode(self, select, expected):
        stream = sequence(tx_mode_select=select)
        header = parsed(stream, stream.frame_header(key_frame()))
        assert header.tx_mode == expected
        assert stream.tx_mode == expected
        assert not header.reduced_tx_set

    def test_the_whole_header_is_read(self):
        stream = sequence()
        packed = stream.frame_header(key_frame())
        header = parsed(stream, packed)
        prefix = 8 * (1 + av1.OBU_SIZE_BYTES)
        assert packed.size_in_bits == prefix + header.header_bits
        # A frame OBU's header is padded with zero bits to a whole byte.
        assert len(packed.data) * 8 == (packed.size_in_bits + 7) // 8 * 8
        assert field(packed.data, packed.size_in_bits,
                     len(packed.data) * 8 - packed.size_in_bits) == 0


class TestInterFrameHeader:
    def test_it_predicts_from_slot_zero_and_refreshes_every_slot(self):
        stream = sequence()
        header = parsed(stream, stream.frame_header(inter_frame(3)),
                        after_key_frame(stream))
        assert header.frame_type == INTER_FRAME
        assert header.show_frame
        assert not header.error_resilient_mode
        assert header.order_hint == 3
        assert header.primary_ref_frame == 0
        assert header.refresh_frame_flags == ALL_FRAMES
        assert not header.frame_refs_short_signaling
        assert header.ref_frame_idx == (0,) * 7

    def test_the_order_hint_wraps_at_its_width(self):
        stream = sequence()
        header = parsed(stream, stream.frame_header(inter_frame(259)),
                        after_key_frame(stream))
        assert header.order_hint == 259 % (1 << av1.ORDER_HINT_BITS) == 3

    def test_the_size_and_the_render_size(self):
        stream = sequence()
        header = parsed(stream, stream.frame_header(inter_frame()),
                        after_key_frame(stream))
        assert header.size == SIZE
        assert (header.render_width, header.render_height) == SIZE

    def test_single_reference_prediction_with_no_global_motion(self):
        stream = sequence()
        header = parsed(stream, stream.frame_header(inter_frame()),
                        after_key_frame(stream))
        assert not header.reference_select
        assert not header.skip_mode_present
        assert header.gm_type == (IDENTITY,) * 8
        assert not header.is_filter_switchable
        assert header.interpolation_filter == EIGHTTAP
        assert not header.is_motion_mode_switchable
        assert not header.allow_high_precision_mv
        assert not header.disable_cdf_update
        assert not header.disable_frame_end_update_cdf

    def test_the_whole_header_is_read(self):
        stream = sequence()
        packed = stream.frame_header(inter_frame())
        header = parsed(stream, packed, after_key_frame(stream))
        assert packed.size_in_bits == 8 * (1 + av1.OBU_SIZE_BYTES) + header.header_bits


class TestOffsets:
    """Each offset names the field a driver rewrites there."""

    @pytest.fixture(params=['key', 'inter'])
    def case(self, request):
        stream = sequence()
        if request.param == 'key':
            return stream, stream.frame_header(key_frame()), FrameState
        return stream, stream.frame_header(inter_frame()), lambda: after_key_frame(stream)

    def reread(self, case, offset, bits, value, padding=b''):
        stream, packed, state = case
        data = rewritten(packed, offset, bits, value)
        return parse_frame_header(next(obu.split(data)).payload + padding,
                                  parse_sequence_header(
                                      only_unit(stream.sequence_header()).payload),
                                  state())

    def test_the_quantiser_index(self, case):
        _, packed, _ = case
        assert field(packed.data, packed.bit_offset_qindex, 8) == 130
        assert self.reread(case, packed.bit_offset_qindex, 8, 201).quantization.base_q_idx == 201

    def test_segmentation_enabled(self, case):
        _, packed, _ = case
        assert field(packed.data, packed.bit_offset_segmentation, 1) == 0
        # An enabled segmentation has more syntax than the header holds.
        header = self.reread(case, packed.bit_offset_segmentation, 1, 1, padding=bytes(32))
        assert header.segmentation.enabled

    def test_the_loop_filter_parameters(self, case):
        _, packed, _ = case
        header = self.reread(case, packed.bit_offset_loopfilter_params, 6, 47)
        assert header.loop_filter.level[0] == 47

    def test_the_cdef_parameters(self, case):
        _, packed, _ = case
        assert packed.size_in_bits_cdef_params == 16
        header = self.reread(case, packed.bit_offset_cdef_params, 2, 3)
        assert header.cdef.damping_minus_3 == 3
        last = packed.bit_offset_cdef_params + packed.size_in_bits_cdef_params - 2
        assert self.reread(case, last, 2, 2).cdef.uv_sec_strength == (2,)

    def test_the_obu_size(self, case):
        _, packed, _ = case
        assert packed.byte_offset_obu_size == 1
        size, used = read_leb128(packed.data, packed.byte_offset_obu_size)
        assert used == av1.OBU_SIZE_BYTES
        assert size == len(packed.data) - 1 - used

    @pytest.mark.parametrize('size_bytes', [1, 2, 4])
    def test_the_obu_size_takes_the_bytes_the_driver_writes_it_in(self, size_bytes):
        stream = sequence()
        packed = stream.frame_header(key_frame(), size_bytes=size_bytes)
        size, used = read_leb128(packed.data, 1)
        assert used == size_bytes
        assert size == len(packed.data) - 1 - size_bytes
        assert field(packed.data, packed.bit_offset_qindex, 8) == 130
        assert only_unit(packed.data).kind == obu.FRAME

    def test_a_frame_header_obu_counts_its_trailing_bit(self):
        stream = sequence()
        packed = stream.frame_header(key_frame(), frame_obu=False)
        header = parsed(stream, packed)
        assert packed.size_in_bits == 8 * (1 + av1.OBU_SIZE_BYTES) + header.header_bits + 1
        assert field(packed.data, packed.size_in_bits - 1, 1) == 1
        assert field(packed.data, packed.size_in_bits,
                     len(packed.data) * 8 - packed.size_in_bits) == 0


class TestTiles:
    def test_a_small_frame_is_one_tile(self):
        stream = sequence()
        assert (stream.tile_cols, stream.tile_rows) == (1, 1)
        info = parsed(stream, stream.frame_header(key_frame())).tile_info
        assert info.uniform_tile_spacing_flag
        assert (info.tile_cols, info.tile_rows) == (1, 1)
        assert info.width_in_sbs_minus_1 == stream.width_in_sbs_minus_1 == (2,)
        assert info.height_in_sbs_minus_1 == stream.height_in_sbs_minus_1 == (1,)

    @pytest.mark.parametrize('size', [(4096, 2160), (4160, 2160), (8192, 4352)])
    def test_a_frame_wider_than_a_tile_may_be_is_split(self, size):
        stream = sequence(size, bitrate=20_000_000)
        info = parsed(stream, stream.frame_header(key_frame())).tile_info
        assert (info.tile_cols, info.tile_rows) == (stream.tile_cols, stream.tile_rows)
        assert info.width_in_sbs_minus_1 == stream.width_in_sbs_minus_1
        assert info.height_in_sbs_minus_1 == stream.height_in_sbs_minus_1
        assert all(width < 64 for width in info.width_in_sbs_minus_1)
        if info.num_tiles > 1:
            assert info.context_update_tile_id == 0
            assert info.tile_size_bytes == 4

    def test_the_inter_header_of_a_split_frame_reads_back(self):
        stream = sequence((8192, 4352), bitrate=20_000_000)
        header = parsed(stream, stream.frame_header(inter_frame()), after_key_frame(stream))
        assert header.tile_info.num_tiles == stream.tile_cols * stream.tile_rows > 1
        assert header.quantization.base_q_idx == 130
