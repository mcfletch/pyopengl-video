"""The AV1 parser: the sequence header, frame headers and tile groups.

A libva decoder is handed the parsed syntax: the sequence header and a frame's
uncompressed header become one picture parameter buffer, and each tile of a
tile group becomes a slice parameter buffer naming where its bytes are. This
module reads that syntax and holds no libva.

Field names are the syntax elements and variables of the AV1 Bitstream &
Decoding Process Specification (section 5), so the parser can be read beside
its syntax tables. A variable the specification writes in CamelCase is spelled
here in snake_case: ``UpscaledWidth`` is ``upscaled_width``.
:mod:`pyopengl_video.obu` splits a temporal unit into the OBUs whose payloads
are read here.

A frame header is read against what earlier frames left behind: the size, the
order hint, the segmentation, the loop filter deltas, the global motion and the
film grain saved in each of the eight reference slots. :class:`FrameState`
holds them, :func:`parse_frame_header` reads from it, and
:meth:`FrameState.update` is the reference frame update process (7.20).
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from pyopengl_video.vaapi.bits import BitReader
from pyopengl_video.vaapi.h264 import StreamError

__all__ = [
    'Cdef', 'ColorConfig', 'DecoderModelInfo', 'FilmGrain', 'FrameHeader',
    'FrameState', 'LoopFilter', 'LoopRestoration', 'OperatingPoint',
    'Quantization', 'Reader', 'SavedFrame', 'Segmentation', 'SequenceHeader',
    'StreamError', 'Tile', 'TileGroup', 'TileInfo', 'TimingInfo',
    'get_relative_dist', 'parse_frame_header', 'parse_sequence_header',
    'parse_tile_group', 'warp_valid',
]

# ----------------------------------------------------------------- constants

KEY_FRAME, INTER_FRAME, INTRA_ONLY_FRAME, SWITCH_FRAME = range(4)
FRAME_TYPE_NAMES = ('key', 'inter', 'intra-only', 'switch')

NUM_REF_FRAMES = 8
REFS_PER_FRAME = 7
TOTAL_REFS_PER_FRAME = 8
PRIMARY_REF_NONE = 7
ALL_FRAMES = (1 << NUM_REF_FRAMES) - 1

#: The reference frame names, which index ``order_hints`` and ``gm_params``.
(INTRA_FRAME, LAST_FRAME, LAST2_FRAME, LAST3_FRAME, GOLDEN_FRAME, BWDREF_FRAME,
 ALTREF2_FRAME, ALTREF_FRAME) = range(8)

SELECT_SCREEN_CONTENT_TOOLS = 2
SELECT_INTEGER_MV = 2

SUPERRES_NUM = 8
SUPERRES_DENOM_MIN = 9
SUPERRES_DENOM_BITS = 3

MAX_TILE_WIDTH = 4096
MAX_TILE_AREA = 4096 * 2304
MAX_TILE_ROWS = 64
MAX_TILE_COLS = 64

MAX_SEGMENTS = 8
SEG_LVL_MAX = 8
SEG_LVL_ALT_Q = 0
SEG_LVL_REF_FRAME = 5
MAX_LOOP_FILTER = 63
SEGMENTATION_FEATURE_BITS = (8, 6, 6, 6, 6, 3, 0, 0)
SEGMENTATION_FEATURE_SIGNED = (1, 1, 1, 1, 1, 0, 0, 0)
SEGMENTATION_FEATURE_MAX = (255, MAX_LOOP_FILTER, MAX_LOOP_FILTER, MAX_LOOP_FILTER,
                            MAX_LOOP_FILTER, 7, 0, 0)

#: loop_filter_ref_deltas as setup_past_independence leaves them (6.8.2).
DEFAULT_REF_DELTAS = (1, 0, 0, 0, -1, 0, -1, -1)
DEFAULT_MODE_DELTAS = (0, 0)

#: interpolation_filter values; SWITCHABLE is what is_filter_switchable infers.
EIGHTTAP, EIGHTTAP_SMOOTH, EIGHTTAP_SHARP, BILINEAR, SWITCHABLE = range(5)

#: TxMode values.
ONLY_4X4, TX_MODE_LARGEST, TX_MODE_SELECT = range(3)

#: FrameRestorationType values, and what each coded lr_type maps to.
RESTORE_NONE, RESTORE_WIENER, RESTORE_SGRPROJ, RESTORE_SWITCHABLE = range(4)
REMAP_LR_TYPE = (RESTORE_NONE, RESTORE_SWITCHABLE, RESTORE_WIENER, RESTORE_SGRPROJ)
RESTORATION_TILESIZE_MAX = 256

#: GmType values.
IDENTITY, TRANSLATION, ROTZOOM, AFFINE = range(4)
WARPEDMODEL_PREC_BITS = 16
GM_ABS_TRANS_BITS = 12
GM_ABS_TRANS_ONLY_BITS = 9
GM_ABS_ALPHA_BITS = 12
GM_ALPHA_PREC_BITS = 15
GM_TRANS_PREC_BITS = 6
GM_TRANS_ONLY_PREC_BITS = 3
WARP_PARAM_REDUCE_BITS = 6
DIV_LUT_BITS = 8
DIV_LUT_PREC_BITS = 14
#: Div_Lut (7.11.3.7): 2**22 / (256 + f), rounded to nearest.
DIV_LUT = tuple(((1 << (DIV_LUT_BITS + DIV_LUT_PREC_BITS + 1)) // (256 + index) + 1) >> 1
                for index in range(257))

#: gm_params for a reference frame with no global motion.
DEFAULT_GM_PARAMS = (0, 0, 1 << WARPEDMODEL_PREC_BITS, 0, 0, 1 << WARPEDMODEL_PREC_BITS)

CP_BT_709 = 1
CP_UNSPECIFIED = 2
TC_UNSPECIFIED = 2
TC_SRGB = 13
MC_IDENTITY = 0
MC_UNSPECIFIED = 2
CSP_UNKNOWN = 0

#: ref_frame_idx entries the set frame refs process fills in order (7.8).
REF_FRAME_LIST = (LAST2_FRAME, LAST3_FRAME, BWDREF_FRAME, ALTREF2_FRAME, ALTREF_FRAME)


# ------------------------------------------------------------------ reading


class Reader(BitReader):
    """A bit reader with the AV1 descriptors :class:`BitReader` has no use for."""

    def ns(self, count: int) -> int:
        """ns(n): a value from 0 to `count` - 1, the low values a bit shorter."""
        width = count.bit_length()
        spare = (1 << width) - count
        value = self.f(width - 1)
        if value < spare:
            return value
        return (value << 1) - spare + self.f(1)

    def le(self, size: int) -> int:
        """le(n): a little-endian value of `size` bytes."""
        return sum(self.f(8) << (8 * index) for index in range(size))

    def uvlc(self) -> int:
        """uvlc(): a variable length code led by as many zeroes as it has value bits."""
        zeroes = 0
        while not self.flag():
            zeroes += 1
            if zeroes >= 32:
                return (1 << 32) - 1
        return self.f(zeroes) + (1 << zeroes) - 1

    def delta_q(self) -> int:
        """read_delta_q(): a quantiser offset, zero where it is not coded."""
        return self.su(7) if self.flag() else 0

    def subexp(self, symbols: int) -> int:
        """decode_subexp(): a value below `symbols`, small values in fewer bits."""
        index, base, k = 0, 0, 3
        while True:
            bits = k + index - 1 if index else k
            step = 1 << bits
            if symbols <= base + 3 * step:
                return self.ns(symbols - base) + base
            if not self.flag():
                return self.f(bits) + base
            index += 1
            base += step

    def unsigned_subexp_with_ref(self, limit: int, reference: int) -> int:
        """decode_unsigned_subexp_with_ref(): a value below `limit`, coded about `reference`."""
        value = self.subexp(limit)
        if (reference << 1) <= limit:
            return _inverse_recenter(reference, value)
        return limit - 1 - _inverse_recenter(limit - 1 - reference, value)

    def signed_subexp_with_ref(self, low: int, high: int, reference: int) -> int:
        """decode_signed_subexp_with_ref(): a value from `low` to `high` - 1."""
        return self.unsigned_subexp_with_ref(high - low, reference - low) + low


def _inverse_recenter(reference: int, value: int) -> int:
    if value > 2 * reference:
        return value
    if value & 1:
        return reference - ((value + 1) >> 1)
    return reference + (value >> 1)


def _tile_log2(block: int, target: int) -> int:
    """The smallest k with ``block << k`` at least `target`."""
    k = 0
    while (block << k) < target:
        k += 1
    return k


# ---------------------------------------------------------- sequence header


@dataclasses.dataclass(frozen=True)
class TimingInfo:
    """timing_info() (5.5.3)."""

    num_units_in_display_tick: int
    time_scale: int
    equal_picture_interval: int
    num_ticks_per_picture_minus_1: int = 0


@dataclasses.dataclass(frozen=True)
class DecoderModelInfo:
    """decoder_model_info() (5.5.4)."""

    buffer_delay_length_minus_1: int
    num_units_in_decoding_tick: int
    buffer_removal_time_length_minus_1: int
    frame_presentation_time_length_minus_1: int


@dataclasses.dataclass(frozen=True)
class OperatingPoint:
    """One operating point of a sequence header.

    idc -- operating_point_idc: bit t set where temporal layer t is decoded,
        bit 8 + s where spatial layer s is; 0 where the stream has no layers
    """

    idc: int
    seq_level_idx: int
    seq_tier: int = 0
    decoder_model_present: int = 0
    decoder_buffer_delay: int = 0
    encoder_buffer_delay: int = 0
    low_delay_mode_flag: int = 0
    initial_display_delay_minus_1: int | None = None


@dataclasses.dataclass(frozen=True)
class ColorConfig:
    """color_config() (5.5.2)."""

    bit_depth: int
    mono_chrome: int
    color_description_present_flag: int
    color_primaries: int
    transfer_characteristics: int
    matrix_coefficients: int
    color_range: int
    subsampling_x: int
    subsampling_y: int
    chroma_sample_position: int
    separate_uv_delta_q: int

    @property
    def num_planes(self) -> int:
        """NumPlanes: 1 for a monochrome stream, otherwise 3."""
        return 1 if self.mono_chrome else 3


@dataclasses.dataclass(frozen=True)
class SequenceHeader:
    """A sequence header OBU (5.5.1).

    operating_points -- every operating point; the first is the one decoded
    order_hint_bits -- OrderHintBits, 0 where order hints are off
    """

    seq_profile: int
    still_picture: int
    reduced_still_picture_header: int
    timing_info: TimingInfo | None
    decoder_model_info: DecoderModelInfo | None
    initial_display_delay_present_flag: int
    operating_points: tuple[OperatingPoint, ...]
    frame_width_bits_minus_1: int
    frame_height_bits_minus_1: int
    max_frame_width_minus_1: int
    max_frame_height_minus_1: int
    frame_id_numbers_present_flag: int
    delta_frame_id_length_minus_2: int
    additional_frame_id_length_minus_1: int
    use_128x128_superblock: int
    enable_filter_intra: int
    enable_intra_edge_filter: int
    enable_interintra_compound: int
    enable_masked_compound: int
    enable_warped_motion: int
    enable_dual_filter: int
    enable_order_hint: int
    enable_jnt_comp: int
    enable_ref_frame_mvs: int
    seq_force_screen_content_tools: int
    seq_force_integer_mv: int
    order_hint_bits: int
    enable_superres: int
    enable_cdef: int
    enable_restoration: int
    color_config: ColorConfig
    film_grain_params_present: int

    @property
    def size(self) -> tuple[int, int]:
        """The largest frame of the sequence, as (width, height)."""
        return self.max_frame_width_minus_1 + 1, self.max_frame_height_minus_1 + 1

    @property
    def operating_point_idc(self) -> int:
        """OperatingPointIdc of the operating point decoded, which is the first."""
        return self.operating_points[0].idc

    @property
    def equal_picture_interval(self) -> int:
        return self.timing_info.equal_picture_interval if self.timing_info else 0

    @property
    def id_len(self) -> int:
        """idLen: the bits of a frame id, where frame ids are present."""
        return (self.additional_frame_id_length_minus_1
                + self.delta_frame_id_length_minus_2 + 3)


def _timing_info(reader: Reader) -> TimingInfo:
    ticks = reader.f(32)
    scale = reader.f(32)
    equal = reader.flag()
    return TimingInfo(ticks, scale, equal, reader.uvlc() if equal else 0)


def _color_config(reader: Reader, profile: int) -> ColorConfig:
    high_bitdepth = reader.flag()
    if profile == 2 and high_bitdepth:
        bit_depth = 12 if reader.flag() else 10
    else:
        bit_depth = 10 if high_bitdepth else 8
    mono_chrome = 0 if profile == 1 else reader.flag()
    described = reader.flag()
    if described:
        primaries, transfer, matrix = reader.f(8), reader.f(8), reader.f(8)
    else:
        primaries, transfer, matrix = CP_UNSPECIFIED, TC_UNSPECIFIED, MC_UNSPECIFIED
    position = CSP_UNKNOWN
    if mono_chrome:
        return ColorConfig(bit_depth, 1, described, primaries, transfer, matrix,
                           reader.flag(), 1, 1, position, 0)
    if (primaries, transfer, matrix) == (CP_BT_709, TC_SRGB, MC_IDENTITY):
        color_range, subsampling_x, subsampling_y = 1, 0, 0
    else:
        color_range = reader.flag()
        if profile == 0:
            subsampling_x = subsampling_y = 1
        elif profile == 1:
            subsampling_x = subsampling_y = 0
        elif bit_depth == 12:
            subsampling_x = reader.flag()
            subsampling_y = reader.flag() if subsampling_x else 0
        else:
            subsampling_x, subsampling_y = 1, 0
        if subsampling_x and subsampling_y:
            position = reader.f(2)
    return ColorConfig(bit_depth, 0, described, primaries, transfer, matrix,
                       color_range, subsampling_x, subsampling_y, position,
                       reader.flag())


def parse_sequence_header(payload: bytes) -> SequenceHeader:
    """Read a sequence header OBU's payload."""
    reader = Reader(payload)
    profile = reader.f(3)
    still_picture = reader.flag()
    reduced = reader.flag()
    timing: TimingInfo | None = None
    model: DecoderModelInfo | None = None
    display_delay_present = 0
    if reduced:
        points = [OperatingPoint(0, reader.f(5))]
    else:
        if reader.flag():
            timing = _timing_info(reader)
            if reader.flag():
                model = DecoderModelInfo(reader.f(5), reader.f(32), reader.f(5), reader.f(5))
        display_delay_present = reader.flag()
        points = []
        for _ in range(reader.f(5) + 1):
            idc = reader.f(12)
            level = reader.f(5)
            tier = reader.flag() if level > 7 else 0
            present = decoder_delay = encoder_delay = low_delay = 0
            if model is not None:
                present = reader.flag()
                if present:
                    width = model.buffer_delay_length_minus_1 + 1
                    decoder_delay = reader.f(width)
                    encoder_delay = reader.f(width)
                    low_delay = reader.flag()
            display_delay = None
            if display_delay_present and reader.flag():
                display_delay = reader.f(4)
            points.append(OperatingPoint(idc, level, tier, present, decoder_delay,
                                         encoder_delay, low_delay, display_delay))
    width_bits = reader.f(4)
    height_bits = reader.f(4)
    max_width = reader.f(width_bits + 1)
    max_height = reader.f(height_bits + 1)
    frame_ids = 0 if reduced else reader.flag()
    delta_id_length = additional_id_length = 0
    if frame_ids:
        delta_id_length = reader.f(4)
        additional_id_length = reader.f(3)
    superblock = reader.flag()
    filter_intra = reader.flag()
    intra_edge = reader.flag()
    interintra = masked = warped = dual = order_hint = jnt = ref_mvs = 0
    screen_content, integer_mv = SELECT_SCREEN_CONTENT_TOOLS, SELECT_INTEGER_MV
    order_hint_bits = 0
    if not reduced:
        interintra = reader.flag()
        masked = reader.flag()
        warped = reader.flag()
        dual = reader.flag()
        order_hint = reader.flag()
        if order_hint:
            jnt = reader.flag()
            ref_mvs = reader.flag()
        if not reader.flag():                   # seq_choose_screen_content_tools
            screen_content = reader.flag()
        if screen_content > 0 and not reader.flag():    # seq_choose_integer_mv
            integer_mv = reader.flag()
        if order_hint:
            order_hint_bits = reader.f(3) + 1
    superres = reader.flag()
    cdef = reader.flag()
    restoration = reader.flag()
    color = _color_config(reader, profile)
    return SequenceHeader(
        seq_profile=profile, still_picture=still_picture,
        reduced_still_picture_header=reduced, timing_info=timing,
        decoder_model_info=model,
        initial_display_delay_present_flag=display_delay_present,
        operating_points=tuple(points), frame_width_bits_minus_1=width_bits,
        frame_height_bits_minus_1=height_bits, max_frame_width_minus_1=max_width,
        max_frame_height_minus_1=max_height, frame_id_numbers_present_flag=frame_ids,
        delta_frame_id_length_minus_2=delta_id_length,
        additional_frame_id_length_minus_1=additional_id_length,
        use_128x128_superblock=superblock, enable_filter_intra=filter_intra,
        enable_intra_edge_filter=intra_edge, enable_interintra_compound=interintra,
        enable_masked_compound=masked, enable_warped_motion=warped,
        enable_dual_filter=dual, enable_order_hint=order_hint, enable_jnt_comp=jnt,
        enable_ref_frame_mvs=ref_mvs, seq_force_screen_content_tools=screen_content,
        seq_force_integer_mv=integer_mv, order_hint_bits=order_hint_bits,
        enable_superres=superres, enable_cdef=cdef, enable_restoration=restoration,
        color_config=color, film_grain_params_present=reader.flag())


# ------------------------------------------------------------- frame header


@dataclasses.dataclass
class TileInfo:
    """tile_info() (5.9.15).

    mi_col_starts, mi_row_starts -- MiColStarts and MiRowStarts, with the
        frame's MiCols and MiRows as the last entry
    width_in_sbs_minus_1, height_in_sbs_minus_1 -- each tile column's width
        and each tile row's height in superblocks, less one, whether the
        spacing is uniform or coded
    """

    uniform_tile_spacing_flag: int = 1
    tile_cols: int = 1
    tile_rows: int = 1
    tile_cols_log2: int = 0
    tile_rows_log2: int = 0
    mi_col_starts: tuple[int, ...] = ()
    mi_row_starts: tuple[int, ...] = ()
    width_in_sbs_minus_1: tuple[int, ...] = ()
    height_in_sbs_minus_1: tuple[int, ...] = ()
    context_update_tile_id: int = 0
    tile_size_bytes: int = 4

    @property
    def num_tiles(self) -> int:
        return self.tile_cols * self.tile_rows


@dataclasses.dataclass
class Quantization:
    """quantization_params() (5.9.12); the deltas are DeltaQYDc and the rest."""

    base_q_idx: int = 0
    delta_q_y_dc: int = 0
    diff_uv_delta: int = 0
    delta_q_u_dc: int = 0
    delta_q_u_ac: int = 0
    delta_q_v_dc: int = 0
    delta_q_v_ac: int = 0
    using_qmatrix: int = 0
    qm_y: int = 0
    qm_u: int = 0
    qm_v: int = 0


def _no_features() -> tuple[tuple[int, ...], ...]:
    return ((0,) * SEG_LVL_MAX,) * MAX_SEGMENTS


@dataclasses.dataclass
class Segmentation:
    """segmentation_params() (5.9.14).

    feature_enabled, feature_data -- FeatureEnabled and FeatureData, by
        segment and then feature: what the frame codes, or what the frame it
        loads its state from left where it codes none
    """

    enabled: int = 0
    update_map: int = 0
    temporal_update: int = 0
    update_data: int = 0
    feature_enabled: tuple[tuple[int, ...], ...] = dataclasses.field(default_factory=_no_features)
    feature_data: tuple[tuple[int, ...], ...] = dataclasses.field(default_factory=_no_features)
    seg_id_pre_skip: int = 0
    last_active_seg_id: int = 0


@dataclasses.dataclass
class LoopFilter:
    """loop_filter_params() (5.9.11).

    level -- loop_filter_level: luma vertical, luma horizontal, U and V
    ref_deltas, mode_deltas -- the deltas in force after this frame's updates
    """

    level: tuple[int, int, int, int] = (0, 0, 0, 0)
    sharpness: int = 0
    delta_enabled: int = 1
    delta_update: int = 0
    ref_deltas: tuple[int, ...] = DEFAULT_REF_DELTAS
    mode_deltas: tuple[int, ...] = DEFAULT_MODE_DELTAS


@dataclasses.dataclass
class Cdef:
    """cdef_params() (5.9.19).

    y_sec_strength, uv_sec_strength -- the secondary strengths as the
        specification uses them: a coded 3 is 4
    """

    damping_minus_3: int = 0
    bits: int = 0
    y_pri_strength: tuple[int, ...] = (0,)
    y_sec_strength: tuple[int, ...] = (0,)
    uv_pri_strength: tuple[int, ...] = (0,)
    uv_sec_strength: tuple[int, ...] = (0,)


@dataclasses.dataclass
class LoopRestoration:
    """lr_params() (5.9.20).

    frame_restoration_type -- FrameRestorationType of each plane, a
        ``RESTORE_*`` value
    unit_shift -- lr_unit_shift after its extra bit: the luma unit is
        64 << unit_shift samples
    """

    frame_restoration_type: tuple[int, int, int] = (RESTORE_NONE,) * 3
    uses_lr: int = 0
    unit_shift: int = 0
    uv_shift: int = 0


@dataclasses.dataclass(frozen=True)
class FilmGrain:
    """film_grain_params() (5.9.30); every field zero where no grain is applied.

    ar_coeffs_y, ar_coeffs_cb, ar_coeffs_cr -- the coded values, which are the
        coefficients plus 128
    """

    apply_grain: int = 0
    grain_seed: int = 0
    update_grain: int = 0
    film_grain_params_ref_idx: int = 0
    point_y_value: tuple[int, ...] = ()
    point_y_scaling: tuple[int, ...] = ()
    chroma_scaling_from_luma: int = 0
    point_cb_value: tuple[int, ...] = ()
    point_cb_scaling: tuple[int, ...] = ()
    point_cr_value: tuple[int, ...] = ()
    point_cr_scaling: tuple[int, ...] = ()
    grain_scaling_minus_8: int = 0
    ar_coeff_lag: int = 0
    ar_coeffs_y: tuple[int, ...] = ()
    ar_coeffs_cb: tuple[int, ...] = ()
    ar_coeffs_cr: tuple[int, ...] = ()
    ar_coeff_shift_minus_6: int = 0
    grain_scale_shift: int = 0
    cb_mult: int = 0
    cb_luma_mult: int = 0
    cb_offset: int = 0
    cr_mult: int = 0
    cr_luma_mult: int = 0
    cr_offset: int = 0
    overlap_flag: int = 0
    clip_to_restricted_range: int = 0

    @property
    def num_y_points(self) -> int:
        return len(self.point_y_value)

    @property
    def num_cb_points(self) -> int:
        return len(self.point_cb_value)

    @property
    def num_cr_points(self) -> int:
        return len(self.point_cr_value)


GmParams = tuple[tuple[int, ...], ...]


def _default_gm() -> GmParams:
    return (DEFAULT_GM_PARAMS,) * (ALTREF_FRAME + 1)


@dataclasses.dataclass
class FrameHeader:
    """A frame's uncompressed header (5.9.2).

    header_bits -- the header's length in bits, before the padding to a byte
    frame_width, frame_height -- the size the tiles are coded at
    upscaled_width -- the width after super-resolution, which is what is shown
    order_hints -- OrderHints: the order hint of each reference frame, by
        reference frame name
    gm_type, gm_params -- the global motion of each reference frame, by
        reference frame name; entry 0 (INTRA_FRAME) is unused
    interpolation_filter -- ``SWITCHABLE`` where is_filter_switchable is set

    A header with show_existing_frame set stops after frame_to_show_map_idx.
    Its frame type, sizes and film grain are those of the slot it shows.
    """

    show_existing_frame: int = 0
    frame_to_show_map_idx: int = 0
    frame_presentation_time: int | None = None
    display_frame_id: int = 0
    frame_type: int = KEY_FRAME
    show_frame: int = 1
    showable_frame: int = 0
    error_resilient_mode: int = 0
    disable_cdf_update: int = 0
    allow_screen_content_tools: int = 0
    force_integer_mv: int = 0
    current_frame_id: int = 0
    frame_size_override_flag: int = 0
    order_hint: int = 0
    primary_ref_frame: int = PRIMARY_REF_NONE
    buffer_removal_time: dict[int, int] = dataclasses.field(default_factory=dict)
    refresh_frame_flags: int = 0
    ref_order_hint: tuple[int, ...] = ()
    frame_width: int = 0
    frame_height: int = 0
    upscaled_width: int = 0
    use_superres: int = 0
    superres_denom: int = SUPERRES_NUM
    render_width: int = 0
    render_height: int = 0
    mi_cols: int = 0
    mi_rows: int = 0
    allow_intrabc: int = 0
    frame_refs_short_signaling: int = 0
    last_frame_idx: int = 0
    gold_frame_idx: int = 0
    ref_frame_idx: tuple[int, ...] = (0,) * REFS_PER_FRAME
    found_ref: int = 0
    allow_high_precision_mv: int = 0
    is_filter_switchable: int = 0
    interpolation_filter: int = EIGHTTAP
    is_motion_mode_switchable: int = 0
    use_ref_frame_mvs: int = 0
    order_hints: tuple[int, ...] = (0,) * (ALTREF_FRAME + 1)
    disable_frame_end_update_cdf: int = 0
    tile_info: TileInfo = dataclasses.field(default_factory=TileInfo)
    quantization: Quantization = dataclasses.field(default_factory=Quantization)
    segmentation: Segmentation = dataclasses.field(default_factory=Segmentation)
    delta_q_present: int = 0
    delta_q_res: int = 0
    delta_lf_present: int = 0
    delta_lf_res: int = 0
    delta_lf_multi: int = 0
    coded_lossless: int = 0
    all_lossless: int = 0
    loop_filter: LoopFilter = dataclasses.field(default_factory=LoopFilter)
    cdef: Cdef = dataclasses.field(default_factory=Cdef)
    loop_restoration: LoopRestoration = dataclasses.field(default_factory=LoopRestoration)
    tx_mode: int = ONLY_4X4
    reference_select: int = 0
    skip_mode_present: int = 0
    skip_mode_frame: tuple[int, int] = (0, 0)
    allow_warped_motion: int = 0
    reduced_tx_set: int = 0
    gm_type: tuple[int, ...] = (IDENTITY,) * (ALTREF_FRAME + 1)
    gm_params: GmParams = dataclasses.field(default_factory=_default_gm)
    film_grain: FilmGrain = FilmGrain()
    header_bits: int = 0
    temporal_id: int = 0
    spatial_id: int = 0

    @property
    def frame_is_intra(self) -> bool:
        """FrameIsIntra: a key frame or an intra-only frame."""
        return self.frame_type in (KEY_FRAME, INTRA_ONLY_FRAME)

    @property
    def size(self) -> tuple[int, int]:
        """The picture after super-resolution, as (width, height)."""
        return self.upscaled_width, self.frame_height

    @property
    def kind(self) -> str:
        """The frame type in words, for a message."""
        return FRAME_TYPE_NAMES[self.frame_type]


@dataclasses.dataclass(frozen=True)
class SavedFrame:
    """What one reference slot keeps of the frame saved in it (7.20).

    valid -- RefValid: a frame has been saved here since the last key frame
    showable_frame -- the frame may be output by show_existing_frame
    """

    valid: bool = False
    frame_id: int = 0
    upscaled_width: int = 0
    frame_width: int = 0
    frame_height: int = 0
    render_width: int = 0
    render_height: int = 0
    mi_cols: int = 0
    mi_rows: int = 0
    frame_type: int = KEY_FRAME
    showable_frame: int = 0
    subsampling_x: int = 1
    subsampling_y: int = 1
    bit_depth: int = 8
    order_hint: int = 0
    order_hints: tuple[int, ...] = (0,) * (ALTREF_FRAME + 1)
    gm_params: GmParams = dataclasses.field(default_factory=_default_gm)
    loop_filter_ref_deltas: tuple[int, ...] = DEFAULT_REF_DELTAS
    loop_filter_mode_deltas: tuple[int, ...] = DEFAULT_MODE_DELTAS
    feature_enabled: tuple[tuple[int, ...], ...] = dataclasses.field(default_factory=_no_features)
    feature_data: tuple[tuple[int, ...], ...] = dataclasses.field(default_factory=_no_features)
    film_grain: FilmGrain = FilmGrain()


class FrameState:
    """The eight reference slots' saved state, which frame headers are read against.

    slots -- a :class:`SavedFrame` for each slot
    """

    def __init__(self) -> None:
        self.slots: list[SavedFrame] = [SavedFrame() for _ in range(NUM_REF_FRAMES)]

    def reset(self) -> None:
        """Forget every saved frame, as before the first frame of a stream."""
        self.slots = [SavedFrame() for _ in range(NUM_REF_FRAMES)]

    def invalidate(self, slot: int, order_hint: int | None = None) -> None:
        """Mark `slot` as holding no frame, with `order_hint` where one is given."""
        saved = self.slots[slot]
        self.slots[slot] = dataclasses.replace(
            saved, valid=False,
            order_hint=saved.order_hint if order_hint is None else order_hint)

    def update(self, header: FrameHeader, sequence: SequenceHeader) -> None:
        """The reference frame update process (7.20) for a frame just decoded.

        A header that shows an existing key frame loads that frame's state
        first (7.21), which leaves every slot holding what its slot held.
        """
        if header.show_existing_frame:
            if header.frame_type == KEY_FRAME:
                shown = self.slots[header.frame_to_show_map_idx]
                self.slots = [shown for _ in range(NUM_REF_FRAMES)]
            return
        color = sequence.color_config
        saved = SavedFrame(
            valid=True, frame_id=header.current_frame_id,
            upscaled_width=header.upscaled_width, frame_width=header.frame_width,
            frame_height=header.frame_height, render_width=header.render_width,
            render_height=header.render_height, mi_cols=header.mi_cols,
            mi_rows=header.mi_rows, frame_type=header.frame_type,
            showable_frame=header.showable_frame, subsampling_x=color.subsampling_x,
            subsampling_y=color.subsampling_y, bit_depth=color.bit_depth,
            order_hint=header.order_hint, order_hints=header.order_hints,
            gm_params=header.gm_params,
            loop_filter_ref_deltas=header.loop_filter.ref_deltas,
            loop_filter_mode_deltas=header.loop_filter.mode_deltas,
            feature_enabled=header.segmentation.feature_enabled,
            feature_data=header.segmentation.feature_data,
            film_grain=header.film_grain)
        for slot in range(NUM_REF_FRAMES):
            if (header.refresh_frame_flags >> slot) & 1:
                self.slots[slot] = saved


def get_relative_dist(a: int, b: int, order_hint_bits: int) -> int:
    """get_relative_dist(): how far order hint `a` is after `b`, sign extended."""
    if not order_hint_bits:
        return 0
    difference = a - b
    top = 1 << (order_hint_bits - 1)
    return (difference & (top - 1)) - (difference & top)


def _set_frame_refs(header: FrameHeader, sequence: SequenceHeader,
                    state: FrameState) -> tuple[int, ...]:
    """The set frame refs process (7.8): ref_frame_idx from two coded entries."""
    bits = sequence.order_hint_bits
    refs = [-1] * REFS_PER_FRAME
    refs[LAST_FRAME - LAST_FRAME] = header.last_frame_idx
    refs[GOLDEN_FRAME - LAST_FRAME] = header.gold_frame_idx
    used = [False] * NUM_REF_FRAMES
    used[header.last_frame_idx] = used[header.gold_frame_idx] = True
    current = 1 << (bits - 1)
    shifted = [current + get_relative_dist(slot.order_hint, header.order_hint, bits)
               for slot in state.slots]

    def latest_backward() -> int:
        found, hint = -1, 0
        for index, value in enumerate(shifted):
            if not used[index] and value >= current and (found < 0 or value >= hint):
                found, hint = index, value
        return found

    def earliest_backward() -> int:
        found, hint = -1, 0
        for index, value in enumerate(shifted):
            if not used[index] and value >= current and (found < 0 or value < hint):
                found, hint = index, value
        return found

    def latest_forward() -> int:
        found, hint = -1, 0
        for index, value in enumerate(shifted):
            if not used[index] and value < current and (found < 0 or value >= hint):
                found, hint = index, value
        return found

    for name, find in ((ALTREF_FRAME, latest_backward), (BWDREF_FRAME, earliest_backward),
                       (ALTREF2_FRAME, earliest_backward)):
        found = find()
        if found >= 0:
            refs[name - LAST_FRAME] = found
            used[found] = True
    for name in REF_FRAME_LIST:
        if refs[name - LAST_FRAME] < 0:
            found = latest_forward()
            if found >= 0:
                refs[name - LAST_FRAME] = found
                used[found] = True
    # min() takes the first of equal hints, as the specification's loop does.
    earliest = min(range(NUM_REF_FRAMES), key=shifted.__getitem__)
    return tuple(earliest if entry < 0 else entry for entry in refs)


def _superres_params(reader: Reader, sequence: SequenceHeader, header: FrameHeader) -> None:
    """superres_params() and compute_image_size(), from the size in `header`."""
    header.use_superres = reader.flag() if sequence.enable_superres else 0
    header.superres_denom = (reader.f(SUPERRES_DENOM_BITS) + SUPERRES_DENOM_MIN
                             if header.use_superres else SUPERRES_NUM)
    header.upscaled_width = header.frame_width
    header.frame_width = ((header.upscaled_width * SUPERRES_NUM + header.superres_denom // 2)
                          // header.superres_denom)
    header.mi_cols = 2 * ((header.frame_width + 7) >> 3)
    header.mi_rows = 2 * ((header.frame_height + 7) >> 3)


def _frame_size(reader: Reader, sequence: SequenceHeader, header: FrameHeader) -> None:
    if header.frame_size_override_flag:
        header.frame_width = reader.f(sequence.frame_width_bits_minus_1 + 1) + 1
        header.frame_height = reader.f(sequence.frame_height_bits_minus_1 + 1) + 1
    else:
        header.frame_width, header.frame_height = sequence.size
    _superres_params(reader, sequence, header)


def _render_size(reader: Reader, header: FrameHeader) -> None:
    if reader.flag():
        header.render_width = reader.f(16) + 1
        header.render_height = reader.f(16) + 1
    else:
        header.render_width, header.render_height = header.size


def _frame_size_with_refs(reader: Reader, sequence: SequenceHeader,
                          header: FrameHeader, state: FrameState) -> None:
    for index in range(REFS_PER_FRAME):
        if reader.flag():
            saved = state.slots[header.ref_frame_idx[index]]
            header.found_ref = 1
            header.frame_width = saved.upscaled_width
            header.frame_height = saved.frame_height
            header.render_width = saved.render_width
            header.render_height = saved.render_height
            _superres_params(reader, sequence, header)
            return
    _frame_size(reader, sequence, header)
    _render_size(reader, header)


def _tile_info(reader: Reader, sequence: SequenceHeader, header: FrameHeader) -> TileInfo:
    shift = 5 if sequence.use_128x128_superblock else 4
    round_up = (1 << shift) - 1
    sb_cols = (header.mi_cols + round_up) >> shift
    sb_rows = (header.mi_rows + round_up) >> shift
    sb_size = shift + 2
    max_tile_width_sb = MAX_TILE_WIDTH >> sb_size
    max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size)
    min_log2_cols = _tile_log2(max_tile_width_sb, sb_cols)
    max_log2_cols = _tile_log2(1, min(sb_cols, MAX_TILE_COLS))
    max_log2_rows = _tile_log2(1, min(sb_rows, MAX_TILE_ROWS))
    min_log2_tiles = max(min_log2_cols, _tile_log2(max_tile_area_sb, sb_rows * sb_cols))
    info = TileInfo(uniform_tile_spacing_flag=reader.flag())
    col_starts: list[int] = []
    row_starts: list[int] = []
    if info.uniform_tile_spacing_flag:
        cols_log2 = min_log2_cols
        while cols_log2 < max_log2_cols and reader.flag():
            cols_log2 += 1
        width_sb = (sb_cols + (1 << cols_log2) - 1) >> cols_log2
        col_starts = list(range(0, sb_cols, width_sb))
        rows_log2 = max(min_log2_tiles - cols_log2, 0)
        while rows_log2 < max_log2_rows and reader.flag():
            rows_log2 += 1
        height_sb = (sb_rows + (1 << rows_log2) - 1) >> rows_log2
        row_starts = list(range(0, sb_rows, height_sb))
    else:
        widest = 0
        start = 0
        while start < sb_cols:
            col_starts.append(start)
            size = reader.ns(min(sb_cols - start, max_tile_width_sb)) + 1
            widest = max(widest, size)
            start += size
        cols_log2 = _tile_log2(1, len(col_starts))
        if min_log2_tiles > 0:
            max_tile_area_sb = (sb_rows * sb_cols) >> (min_log2_tiles + 1)
        else:
            max_tile_area_sb = sb_rows * sb_cols
        max_tile_height_sb = max(max_tile_area_sb // widest, 1)
        start = 0
        while start < sb_rows:
            row_starts.append(start)
            start += reader.ns(min(sb_rows - start, max_tile_height_sb)) + 1
        rows_log2 = _tile_log2(1, len(row_starts))
    info.tile_cols, info.tile_rows = len(col_starts), len(row_starts)
    info.tile_cols_log2, info.tile_rows_log2 = cols_log2, rows_log2
    info.width_in_sbs_minus_1 = tuple(
        end - start - 1 for start, end in zip(col_starts, [*col_starts[1:], sb_cols], strict=True))
    info.height_in_sbs_minus_1 = tuple(
        end - start - 1 for start, end in zip(row_starts, [*row_starts[1:], sb_rows], strict=True))
    info.mi_col_starts = (*(start << shift for start in col_starts), header.mi_cols)
    info.mi_row_starts = (*(start << shift for start in row_starts), header.mi_rows)
    if cols_log2 > 0 or rows_log2 > 0:
        info.context_update_tile_id = reader.f(cols_log2 + rows_log2)
        info.tile_size_bytes = reader.f(2) + 1
    return info


def _quantization_params(reader: Reader, color: ColorConfig) -> Quantization:
    found = Quantization(base_q_idx=reader.f(8))
    found.delta_q_y_dc = reader.delta_q()
    if color.num_planes > 1:
        if color.separate_uv_delta_q:
            found.diff_uv_delta = reader.flag()
        found.delta_q_u_dc = reader.delta_q()
        found.delta_q_u_ac = reader.delta_q()
        if found.diff_uv_delta:
            found.delta_q_v_dc = reader.delta_q()
            found.delta_q_v_ac = reader.delta_q()
        else:
            found.delta_q_v_dc, found.delta_q_v_ac = found.delta_q_u_dc, found.delta_q_u_ac
    found.using_qmatrix = reader.flag()
    if found.using_qmatrix:
        found.qm_y = reader.f(4)
        found.qm_u = reader.f(4)
        found.qm_v = reader.f(4) if color.separate_uv_delta_q else found.qm_u
    return found


def _segmentation_params(reader: Reader, header: FrameHeader,
                         previous: SavedFrame | None) -> Segmentation:
    """segmentation_params(), over what `previous` left (None after setup_past_independence)."""
    found = Segmentation(enabled=reader.flag())
    if not found.enabled:
        return found
    if previous is not None:
        found.feature_enabled = previous.feature_enabled
        found.feature_data = previous.feature_data
    if header.primary_ref_frame == PRIMARY_REF_NONE:
        found.update_map, found.temporal_update, found.update_data = 1, 0, 1
    else:
        found.update_map = reader.flag()
        if found.update_map:
            found.temporal_update = reader.flag()
        found.update_data = reader.flag()
    if found.update_data:
        enabled: list[tuple[int, ...]] = []
        data: list[tuple[int, ...]] = []
        for _ in range(MAX_SEGMENTS):
            flags, values = [], []
            for feature in range(SEG_LVL_MAX):
                flag = reader.flag()
                value = 0
                if flag:
                    bits = SEGMENTATION_FEATURE_BITS[feature]
                    limit = SEGMENTATION_FEATURE_MAX[feature]
                    if SEGMENTATION_FEATURE_SIGNED[feature]:
                        value = max(-limit, min(limit, reader.su(1 + bits)))
                    else:
                        value = max(0, min(limit, reader.f(bits)))
                flags.append(flag)
                values.append(value)
            enabled.append(tuple(flags))
            data.append(tuple(values))
        found.feature_enabled, found.feature_data = tuple(enabled), tuple(data)
    for segment, features in enumerate(found.feature_enabled):
        for feature, flag in enumerate(features):
            if flag:
                found.last_active_seg_id = segment
                if feature >= SEG_LVL_REF_FRAME:
                    found.seg_id_pre_skip = 1
    return found


def _coded_lossless(header: FrameHeader) -> int:
    """CodedLossless: every segment's quantiser index is 0 with no deltas."""
    quantization, segmentation = header.quantization, header.segmentation
    if (quantization.delta_q_y_dc or quantization.delta_q_u_ac or quantization.delta_q_u_dc
            or quantization.delta_q_v_ac or quantization.delta_q_v_dc):
        return 0
    for segment in range(MAX_SEGMENTS):
        qindex = quantization.base_q_idx
        if segmentation.enabled and segmentation.feature_enabled[segment][SEG_LVL_ALT_Q]:
            qindex = max(0, min(255, qindex + segmentation.feature_data[segment][SEG_LVL_ALT_Q]))
        if qindex:
            return 0
    return 1


def _loop_filter_params(reader: Reader, header: FrameHeader, color: ColorConfig,
                        previous: SavedFrame | None) -> LoopFilter:
    if header.coded_lossless or header.allow_intrabc:
        return LoopFilter()
    found = LoopFilter()
    if previous is not None:
        found.ref_deltas = previous.loop_filter_ref_deltas
        found.mode_deltas = previous.loop_filter_mode_deltas
    first, second = reader.f(6), reader.f(6)
    u = v = 0
    if color.num_planes > 1 and (first or second):
        u, v = reader.f(6), reader.f(6)
    found.level = (first, second, u, v)
    found.sharpness = reader.f(3)
    found.delta_enabled = reader.flag()
    if found.delta_enabled:
        found.delta_update = reader.flag()
        if found.delta_update:
            found.ref_deltas = tuple(
                reader.su(7) if reader.flag() else value for value in found.ref_deltas)
            found.mode_deltas = tuple(
                reader.su(7) if reader.flag() else value for value in found.mode_deltas)
    return found


def _cdef_params(reader: Reader, header: FrameHeader, sequence: SequenceHeader) -> Cdef:
    if header.coded_lossless or header.allow_intrabc or not sequence.enable_cdef:
        return Cdef()
    found = Cdef(damping_minus_3=reader.f(2), bits=reader.f(2))
    y_pri, y_sec, uv_pri, uv_sec = [], [], [], []
    for _ in range(1 << found.bits):
        y_pri.append(reader.f(4))
        strength = reader.f(2)
        y_sec.append(4 if strength == 3 else strength)
        if sequence.color_config.num_planes > 1:
            uv_pri.append(reader.f(4))
            strength = reader.f(2)
            uv_sec.append(4 if strength == 3 else strength)
        else:
            uv_pri.append(0)
            uv_sec.append(0)
    found.y_pri_strength, found.y_sec_strength = tuple(y_pri), tuple(y_sec)
    found.uv_pri_strength, found.uv_sec_strength = tuple(uv_pri), tuple(uv_sec)
    return found


def _lr_params(reader: Reader, header: FrameHeader, sequence: SequenceHeader) -> LoopRestoration:
    found = LoopRestoration()
    if header.all_lossless or header.allow_intrabc or not sequence.enable_restoration:
        return found
    color = sequence.color_config
    types = [RESTORE_NONE] * 3
    uses_chroma = False
    for plane in range(color.num_planes):
        types[plane] = REMAP_LR_TYPE[reader.f(2)]
        if types[plane] != RESTORE_NONE:
            found.uses_lr = 1
            uses_chroma = uses_chroma or plane > 0
    found.frame_restoration_type = (types[0], types[1], types[2])
    if found.uses_lr:
        if sequence.use_128x128_superblock:
            found.unit_shift = reader.flag() + 1
        else:
            found.unit_shift = reader.flag()
            if found.unit_shift:
                found.unit_shift += reader.flag()
        if color.subsampling_x and color.subsampling_y and uses_chroma:
            found.uv_shift = reader.flag()
    return found


def _skip_mode_params(reader: Reader, header: FrameHeader, sequence: SequenceHeader,
                      state: FrameState) -> None:
    """skip_mode_params() (5.9.22): the two references skip mode predicts from."""
    bits = sequence.order_hint_bits
    allowed = False
    if not (header.frame_is_intra or not header.reference_select
            or not sequence.enable_order_hint):
        hints = [state.slots[index].order_hint for index in header.ref_frame_idx]
        forward = backward = -1
        forward_hint = backward_hint = 0
        for index, hint in enumerate(hints):
            distance = get_relative_dist(hint, header.order_hint, bits)
            if distance < 0:
                if forward < 0 or get_relative_dist(hint, forward_hint, bits) > 0:
                    forward, forward_hint = index, hint
            elif distance > 0 and (
                    backward < 0 or get_relative_dist(hint, backward_hint, bits) < 0):
                backward, backward_hint = index, hint
        if forward < 0:
            pass
        elif backward >= 0:
            allowed = True
            pair = (forward, backward)
        else:
            second, second_hint = -1, 0
            for index, hint in enumerate(hints):
                if get_relative_dist(hint, forward_hint, bits) < 0 and (
                        second < 0 or get_relative_dist(hint, second_hint, bits) > 0):
                    second, second_hint = index, hint
            if second >= 0:
                allowed = True
                pair = (forward, second)
        if allowed:
            header.skip_mode_frame = (LAST_FRAME + min(pair), LAST_FRAME + max(pair))
    header.skip_mode_present = reader.flag() if allowed else 0


def _global_motion_params(reader: Reader, header: FrameHeader,
                          previous: SavedFrame | None) -> None:
    """global_motion_params() (5.9.24), coded against the previous frame's models."""
    if header.frame_is_intra:
        return
    prior = previous.gm_params if previous is not None else _default_gm()
    types = [IDENTITY] * (ALTREF_FRAME + 1)
    models = [list(DEFAULT_GM_PARAMS) for _ in range(ALTREF_FRAME + 1)]

    def read(kind: int, ref: int, index: int) -> None:
        abs_bits, prec_bits = GM_ABS_ALPHA_BITS, GM_ALPHA_PREC_BITS
        if index < 2:
            if kind == TRANSLATION:
                coarse = 0 if header.allow_high_precision_mv else 1
                abs_bits = GM_ABS_TRANS_ONLY_BITS - coarse
                prec_bits = GM_TRANS_ONLY_PREC_BITS - coarse
            else:
                abs_bits, prec_bits = GM_ABS_TRANS_BITS, GM_TRANS_PREC_BITS
        prec_diff = WARPEDMODEL_PREC_BITS - prec_bits
        diagonal = index % 3 == 2
        rounding = (1 << WARPEDMODEL_PREC_BITS) if diagonal else 0
        offset = (1 << prec_bits) if diagonal else 0
        limit = 1 << abs_bits
        reference = (prior[ref][index] >> prec_diff) - offset
        models[ref][index] = (
            reader.signed_subexp_with_ref(-limit, limit + 1, reference) << prec_diff) + rounding

    for ref in range(LAST_FRAME, ALTREF_FRAME + 1):
        kind = IDENTITY
        if reader.flag():                       # is_global
            if reader.flag():                   # is_rot_zoom
                kind = ROTZOOM
            else:
                kind = TRANSLATION if reader.flag() else AFFINE
        types[ref] = kind
        if kind >= ROTZOOM:
            read(kind, ref, 2)
            read(kind, ref, 3)
            if kind == AFFINE:
                read(kind, ref, 4)
                read(kind, ref, 5)
            else:
                models[ref][4] = -models[ref][3]
                models[ref][5] = models[ref][2]
        if kind >= TRANSLATION:
            read(kind, ref, 0)
            read(kind, ref, 1)
    header.gm_type = tuple(types)
    header.gm_params = tuple(tuple(model) for model in models)


def _round2signed(value: int, bits: int) -> int:
    if not bits:
        return value
    rounded = (abs(value) + (1 << (bits - 1))) >> bits
    return -rounded if value < 0 else rounded


def warp_valid(params: Sequence[int]) -> bool:
    """warpValid of the setup shear process (7.11.3.6) for one warp model."""
    def clip(value: int) -> int:
        return max(-32768, min(32767, value))

    def reduce(value: int) -> int:
        return _round2signed(value, WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS

    divisor = params[2]
    if divisor <= 0:
        return False
    # The resolve divisor process (7.11.3.7).
    top = divisor.bit_length() - 1
    rest = divisor - (1 << top)
    if top > DIV_LUT_BITS:
        index = (rest + (1 << (top - DIV_LUT_BITS - 1))) >> (top - DIV_LUT_BITS)
    else:
        index = rest << (DIV_LUT_BITS - top)
    shift = top + DIV_LUT_PREC_BITS
    factor = DIV_LUT[index]
    alpha = reduce(clip(params[2] - (1 << WARPEDMODEL_PREC_BITS)))
    beta = reduce(clip(params[3]))
    gamma = reduce(clip(_round2signed((params[4] << WARPEDMODEL_PREC_BITS) * factor, shift)))
    delta = reduce(clip(params[5] - _round2signed(params[3] * params[4] * factor, shift)
                        - (1 << WARPEDMODEL_PREC_BITS)))
    one = 1 << WARPEDMODEL_PREC_BITS
    return 4 * abs(alpha) + 7 * abs(beta) < one and 4 * abs(gamma) + 4 * abs(delta) < one


def _film_grain_params(reader: Reader, header: FrameHeader, sequence: SequenceHeader,
                       state: FrameState) -> FilmGrain:
    if not sequence.film_grain_params_present or not (header.show_frame or header.showable_frame):
        return FilmGrain()
    if not reader.flag():                       # apply_grain
        return FilmGrain()
    seed = reader.f(16)
    update = reader.flag() if header.frame_type == INTER_FRAME else 1
    if not update:
        index = reader.f(3)
        return dataclasses.replace(
            state.slots[index].film_grain, apply_grain=1, grain_seed=seed,
            update_grain=0, film_grain_params_ref_idx=index)
    color = sequence.color_config

    def points() -> tuple[tuple[int, ...], tuple[int, ...]]:
        values, scalings = [], []
        for _ in range(reader.f(4)):
            values.append(reader.f(8))
            scalings.append(reader.f(8))
        return tuple(values), tuple(scalings)

    y_value, y_scaling = points()
    from_luma = 0 if color.mono_chrome else reader.flag()
    cb_value: tuple[int, ...] = ()
    cb_scaling: tuple[int, ...] = ()
    cr_value: tuple[int, ...] = ()
    cr_scaling: tuple[int, ...] = ()
    if not (color.mono_chrome or from_luma
            or (color.subsampling_x == 1 and color.subsampling_y == 1 and not y_value)):
        cb_value, cb_scaling = points()
        cr_value, cr_scaling = points()
    scaling = reader.f(2)
    lag = reader.f(2)
    luma_positions = 2 * lag * (lag + 1)
    chroma_positions = luma_positions + 1 if y_value else luma_positions
    coeffs_y = tuple(reader.f(8) for _ in range(luma_positions)) if y_value else ()
    coeffs_cb = (tuple(reader.f(8) for _ in range(chroma_positions))
                 if from_luma or cb_value else ())
    coeffs_cr = (tuple(reader.f(8) for _ in range(chroma_positions))
                 if from_luma or cr_value else ())
    shift = reader.f(2)
    scale_shift = reader.f(2)
    cb_mult = cb_luma_mult = cb_offset = cr_mult = cr_luma_mult = cr_offset = 0
    if cb_value:
        cb_mult, cb_luma_mult, cb_offset = reader.f(8), reader.f(8), reader.f(9)
    if cr_value:
        cr_mult, cr_luma_mult, cr_offset = reader.f(8), reader.f(8), reader.f(9)
    return FilmGrain(
        apply_grain=1, grain_seed=seed, update_grain=1, point_y_value=y_value,
        point_y_scaling=y_scaling, chroma_scaling_from_luma=from_luma,
        point_cb_value=cb_value, point_cb_scaling=cb_scaling, point_cr_value=cr_value,
        point_cr_scaling=cr_scaling, grain_scaling_minus_8=scaling, ar_coeff_lag=lag,
        ar_coeffs_y=coeffs_y, ar_coeffs_cb=coeffs_cb, ar_coeffs_cr=coeffs_cr,
        ar_coeff_shift_minus_6=shift, grain_scale_shift=scale_shift, cb_mult=cb_mult,
        cb_luma_mult=cb_luma_mult, cb_offset=cb_offset, cr_mult=cr_mult,
        cr_luma_mult=cr_luma_mult, cr_offset=cr_offset, overlap_flag=reader.flag(),
        clip_to_restricted_range=reader.flag())


def _temporal_point_info(reader: Reader, sequence: SequenceHeader) -> int | None:
    model = sequence.decoder_model_info
    if model is None or sequence.equal_picture_interval:
        return None
    return reader.f(model.frame_presentation_time_length_minus_1 + 1)


def _mark_ref_frames(header: FrameHeader, sequence: SequenceHeader, state: FrameState) -> None:
    """mark_ref_frames() (5.9.2): slots whose frame id is out of reach hold no frame."""
    id_len = sequence.id_len
    reach = 1 << (sequence.delta_frame_id_length_minus_2 + 2)
    current = header.current_frame_id
    for index, saved in enumerate(state.slots):
        if current > reach:
            stale = saved.frame_id > current or saved.frame_id < current - reach
        else:
            stale = current < saved.frame_id < (1 << id_len) + current - reach
        if stale:
            state.invalidate(index)


def _show_existing(reader: Reader, header: FrameHeader, sequence: SequenceHeader,
                   state: FrameState) -> FrameHeader:
    header.show_existing_frame = 1
    header.frame_to_show_map_idx = reader.f(3)
    header.frame_presentation_time = _temporal_point_info(reader, sequence)
    if sequence.frame_id_numbers_present_flag:
        header.display_frame_id = reader.f(sequence.id_len)
    saved = state.slots[header.frame_to_show_map_idx]
    header.frame_type = saved.frame_type
    header.show_frame = 1
    header.refresh_frame_flags = ALL_FRAMES if saved.frame_type == KEY_FRAME else 0
    header.current_frame_id = saved.frame_id
    header.upscaled_width, header.frame_width = saved.upscaled_width, saved.frame_width
    header.frame_height = saved.frame_height
    header.render_width, header.render_height = saved.render_width, saved.render_height
    header.mi_cols, header.mi_rows = saved.mi_cols, saved.mi_rows
    header.order_hint, header.order_hints = saved.order_hint, saved.order_hints
    header.gm_params = saved.gm_params
    if sequence.film_grain_params_present:
        header.film_grain = saved.film_grain
    header.header_bits = reader.position
    return header


def parse_frame_header(payload: bytes, sequence: SequenceHeader, state: FrameState, *,
                       temporal_id: int = 0, spatial_id: int = 0) -> FrameHeader:
    """Read the uncompressed header at the start of a frame header or frame OBU.

    state -- what the reference slots hold. The header is read against it, and
        a slot the header declares lost (a shown key frame's, an order hint or
        a frame id that does not match) is marked as holding no frame.
    temporal_id, spatial_id -- the OBU's layer, which selects the buffer
        removal times that are present
    """
    reader = Reader(payload)
    header = FrameHeader(temporal_id=temporal_id, spatial_id=spatial_id)
    color = sequence.color_config
    if sequence.reduced_still_picture_header:
        header.frame_type, header.show_frame, header.showable_frame = KEY_FRAME, 1, 0
        header.error_resilient_mode = 1
    else:
        if reader.flag():
            return _show_existing(reader, header, sequence, state)
        header.frame_type = reader.f(2)
        header.show_frame = reader.flag()
        if header.show_frame:
            header.frame_presentation_time = _temporal_point_info(reader, sequence)
            header.showable_frame = int(header.frame_type != KEY_FRAME)
        else:
            header.showable_frame = reader.flag()
        if header.frame_type == SWITCH_FRAME or (
                header.frame_type == KEY_FRAME and header.show_frame):
            header.error_resilient_mode = 1
        else:
            header.error_resilient_mode = reader.flag()
    intra = header.frame_is_intra
    if header.frame_type == KEY_FRAME and header.show_frame:
        for index in range(NUM_REF_FRAMES):
            state.invalidate(index, 0)
    header.disable_cdf_update = reader.flag()
    if sequence.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS:
        header.allow_screen_content_tools = reader.flag()
    else:
        header.allow_screen_content_tools = sequence.seq_force_screen_content_tools
    if header.allow_screen_content_tools:
        if sequence.seq_force_integer_mv == SELECT_INTEGER_MV:
            header.force_integer_mv = reader.flag()
        else:
            header.force_integer_mv = sequence.seq_force_integer_mv
    if intra:
        header.force_integer_mv = 1
    if sequence.frame_id_numbers_present_flag:
        header.current_frame_id = reader.f(sequence.id_len)
        _mark_ref_frames(header, sequence, state)
    if header.frame_type == SWITCH_FRAME:
        header.frame_size_override_flag = 1
    elif not sequence.reduced_still_picture_header:
        header.frame_size_override_flag = reader.flag()
    header.order_hint = reader.f(sequence.order_hint_bits)
    if not (intra or header.error_resilient_mode):
        header.primary_ref_frame = reader.f(3)
    model = sequence.decoder_model_info
    if model is not None and reader.flag():     # buffer_removal_time_present_flag
        for number, point in enumerate(sequence.operating_points):
            if not point.decoder_model_present:
                continue
            in_temporal = (point.idc >> temporal_id) & 1
            in_spatial = (point.idc >> (spatial_id + 8)) & 1
            if point.idc == 0 or (in_temporal and in_spatial):
                header.buffer_removal_time[number] = reader.f(
                    model.buffer_removal_time_length_minus_1 + 1)
    if header.frame_type == SWITCH_FRAME or (
            header.frame_type == KEY_FRAME and header.show_frame):
        header.refresh_frame_flags = ALL_FRAMES
    else:
        header.refresh_frame_flags = reader.f(8)
    if ((not intra or header.refresh_frame_flags != ALL_FRAMES)
            and header.error_resilient_mode and sequence.enable_order_hint):
        hints = []
        for index in range(NUM_REF_FRAMES):
            hint = reader.f(sequence.order_hint_bits)
            hints.append(hint)
            if hint != state.slots[index].order_hint or not state.slots[index].valid:
                state.invalidate(index, hint)
        header.ref_order_hint = tuple(hints)
    if intra:
        _frame_size(reader, sequence, header)
        _render_size(reader, header)
        if header.allow_screen_content_tools and header.upscaled_width == header.frame_width:
            header.allow_intrabc = reader.flag()
    else:
        if sequence.enable_order_hint:
            header.frame_refs_short_signaling = reader.flag()
            if header.frame_refs_short_signaling:
                header.last_frame_idx = reader.f(3)
                header.gold_frame_idx = reader.f(3)
                header.ref_frame_idx = _set_frame_refs(header, sequence, state)
        refs = list(header.ref_frame_idx)
        for index in range(REFS_PER_FRAME):
            if not header.frame_refs_short_signaling:
                refs[index] = reader.f(3)
            if sequence.frame_id_numbers_present_flag:
                reader.f(sequence.delta_frame_id_length_minus_2 + 2)    # delta_frame_id_minus_1
        header.ref_frame_idx = tuple(refs)
        if header.frame_size_override_flag and not header.error_resilient_mode:
            _frame_size_with_refs(reader, sequence, header, state)
        else:
            _frame_size(reader, sequence, header)
            _render_size(reader, header)
        if not header.force_integer_mv:
            header.allow_high_precision_mv = reader.flag()
        header.is_filter_switchable = reader.flag()
        header.interpolation_filter = SWITCHABLE if header.is_filter_switchable else reader.f(2)
        header.is_motion_mode_switchable = reader.flag()
        if not (header.error_resilient_mode or not sequence.enable_ref_frame_mvs):
            header.use_ref_frame_mvs = reader.flag()
        header.order_hints = (0, *(state.slots[index].order_hint for index in refs))
    if sequence.reduced_still_picture_header or header.disable_cdf_update:
        header.disable_frame_end_update_cdf = 1
    else:
        header.disable_frame_end_update_cdf = reader.flag()
    # load_previous(), or None for setup_past_independence().
    previous = (None if header.primary_ref_frame == PRIMARY_REF_NONE
                else state.slots[header.ref_frame_idx[header.primary_ref_frame]])
    header.tile_info = _tile_info(reader, sequence, header)
    header.quantization = _quantization_params(reader, color)
    header.segmentation = _segmentation_params(reader, header, previous)
    if header.quantization.base_q_idx > 0:
        header.delta_q_present = reader.flag()
    if header.delta_q_present:
        header.delta_q_res = reader.f(2)
        if not header.allow_intrabc:
            header.delta_lf_present = reader.flag()
        if header.delta_lf_present:
            header.delta_lf_res = reader.f(2)
            header.delta_lf_multi = reader.flag()
    header.coded_lossless = _coded_lossless(header)
    header.all_lossless = int(header.coded_lossless
                              and header.frame_width == header.upscaled_width)
    header.loop_filter = _loop_filter_params(reader, header, color, previous)
    header.cdef = _cdef_params(reader, header, sequence)
    header.loop_restoration = _lr_params(reader, header, sequence)
    if not header.coded_lossless:
        header.tx_mode = TX_MODE_SELECT if reader.flag() else TX_MODE_LARGEST
    if not intra:
        header.reference_select = reader.flag()
    _skip_mode_params(reader, header, sequence, state)
    if not (intra or header.error_resilient_mode or not sequence.enable_warped_motion):
        header.allow_warped_motion = reader.flag()
    header.reduced_tx_set = reader.flag()
    _global_motion_params(reader, header, previous)
    header.film_grain = _film_grain_params(reader, header, sequence, state)
    header.header_bits = reader.position
    return header


# --------------------------------------------------------------- tile groups


@dataclasses.dataclass(frozen=True)
class Tile:
    """One tile's coded bytes within the data its tile group was read from."""

    row: int
    column: int
    offset: int
    size: int


@dataclasses.dataclass(frozen=True)
class TileGroup:
    """A tile group's header (5.11.1) and where each of its tiles lies."""

    tg_start: int
    tg_end: int
    tiles: tuple[Tile, ...]


def parse_tile_group(data: bytes, header: FrameHeader, start: int = 0) -> TileGroup:
    """Read the tile group that begins at byte `start` of `data` and runs to its end.

    start -- 0 for a tile group OBU's payload; for a frame OBU, the byte after
        the frame header and its padding
    """
    info = header.tile_info
    reader = Reader(data, start * 8)
    count = info.num_tiles
    first, last = 0, count - 1
    if count > 1 and reader.flag():             # tile_start_and_end_present_flag
        bits = info.tile_cols_log2 + info.tile_rows_log2
        first = reader.f(bits)
        last = reader.f(bits)
        if not first <= last < count:
            raise StreamError(
                f'a tile group names tiles {first} to {last} of a frame with {count}')
    reader.byte_align()
    position = reader.byte_position
    tiles = []
    for number in range(first, last + 1):
        if number == last:
            size = len(data) - position
        else:
            reader.position = position * 8
            size = reader.le(info.tile_size_bytes) + 1
            position += info.tile_size_bytes
        if size <= 0 or position + size > len(data):
            raise StreamError(
                f'tile {number} of {size} bytes runs past the end of its tile group')
        tiles.append(Tile(number // info.tile_cols, number % info.tile_cols, position, size))
        position += size
    return TileGroup(first, last, tuple(tiles))
