"""AV1 headers written bit by bit, and a walk through a stream's frame headers.

The tests build sequence and frame headers here with known values, so the
parser and the control layer are tested on syntax no encoder was asked for.
:class:`Sequence` fixes the sequence-level choices and writes the sequence
header; :func:`frame_header` writes an uncompressed header for it.
"""
import dataclasses

from pyopengl_video import obu as obus
from pyopengl_video.vaapi.av1_parse import (
    INTER_FRAME,
    INTRA_ONLY_FRAME,
    KEY_FRAME,
    PRIMARY_REF_NONE,
    SWITCH_FRAME,
    FrameState,
    parse_frame_header,
    parse_sequence_header,
    parse_tile_group,
)
from pyopengl_video.vaapi.bits import BitWriter, leb128


def obu(kind, payload, *, temporal_id=None, spatial_id=0, size=True):
    """One OBU; with a `temporal_id` it carries the extension byte."""
    extension = temporal_id is not None
    header = bytes([kind << 3 | (0x04 if extension else 0) | (0x02 if size else 0)])
    if extension:
        header += bytes([temporal_id << 5 | spatial_id << 3])
    return header + (leb128(len(payload)) if size else b'') + payload


TEMPORAL_DELIMITER = obu(obus.TEMPORAL_DELIMITER, b'')


@dataclasses.dataclass(frozen=True)
class Sequence:
    """The sequence-level choices of a hand-built stream.

    Screen content tools, every compound and warped tool, CDEF and loop
    restoration are off, so a frame header has none of their syntax. Order
    hints are 7 bits.

    twelve_bit, subsampling -- what a profile 2 sequence with high_bitdepth
        states; `subsampling` is written at 12 bits alone
    """

    profile: int = 0
    size: tuple[int, int] = (64, 48)
    size_bits: int = 8
    high_bitdepth: int = 0
    twelve_bit: int = 0
    subsampling: tuple[int, int] = (1, 1)
    mono_chrome: int = 0
    film_grain: int = 0
    superres: int = 0
    operating_points: tuple[int, ...] = (0,)
    colour: tuple[int, int, int] | None = None
    color_range: int = 0

    def payload(self):
        """The sequence header OBU's payload."""
        writer = BitWriter()
        writer.f(self.profile, 3)
        writer.flag(0)                          # still_picture
        writer.flag(0)                          # reduced_still_picture_header
        writer.flag(0)                          # timing_info_present_flag
        writer.flag(0)                          # initial_display_delay_present_flag
        writer.f(len(self.operating_points) - 1, 5)
        for idc in self.operating_points:
            writer.f(idc, 12)
            writer.f(0, 5)                      # seq_level_idx
        writer.f(self.size_bits - 1, 4)
        writer.f(self.size_bits - 1, 4)
        writer.f(self.size[0] - 1, self.size_bits)
        writer.f(self.size[1] - 1, self.size_bits)
        writer.flag(0)                          # frame_id_numbers_present_flag
        writer.flag(0)                          # use_128x128_superblock
        writer.flag(0)                          # enable_filter_intra
        writer.flag(0)                          # enable_intra_edge_filter
        writer.flag(0)                          # enable_interintra_compound
        writer.flag(0)                          # enable_masked_compound
        writer.flag(0)                          # enable_warped_motion
        writer.flag(0)                          # enable_dual_filter
        writer.flag(1)                          # enable_order_hint
        writer.flag(0)                          # enable_jnt_comp
        writer.flag(0)                          # enable_ref_frame_mvs
        writer.flag(0)                          # seq_choose_screen_content_tools
        writer.flag(0)                          # seq_force_screen_content_tools
        writer.f(6, 3)                          # order_hint_bits_minus_1
        writer.flag(self.superres)
        writer.flag(0)                          # enable_cdef
        writer.flag(0)                          # enable_restoration
        writer.flag(self.high_bitdepth)
        if self.profile == 2 and self.high_bitdepth:
            writer.flag(self.twelve_bit)
        if self.profile != 1:
            writer.flag(self.mono_chrome)
        writer.flag(self.colour is not None)
        if self.colour is not None:
            for value in self.colour:
                writer.f(value, 8)
        if self.mono_chrome:
            writer.flag(self.color_range)
        elif self.colour != (1, 13, 0):
            writer.flag(self.color_range)
            if self.profile == 2 and self.high_bitdepth and self.twelve_bit:
                writer.flag(self.subsampling[0])
                if self.subsampling[0]:
                    writer.flag(self.subsampling[1])
            if self.profile == 0 or (self.profile == 2 and self.twelve_bit
                                     and self.subsampling == (1, 1)):
                writer.f(0, 2)                  # chroma_sample_position
        if not self.mono_chrome:
            writer.flag(0)                      # separate_uv_delta_q
        writer.flag(self.film_grain)
        writer.flag(1)                          # trailing_one_bit
        return writer.bytes()

    def obu(self):
        return obu(obus.SEQUENCE_HEADER, self.payload())

    def parsed(self):
        return parse_sequence_header(self.payload())


#: The sequence a header is written for where none is named.
DEFAULT = Sequence()


def frame_header(sequence=DEFAULT, *, frame_type=KEY_FRAME, show_frame=1,
                 showable_frame=1, error_resilient=0, order_hint=0,
                 primary_ref_frame=PRIMARY_REF_NONE, refresh=0xFF, ref_order_hints=None,
                 short_signaling=None, refs=(0,) * 7, size=None, found_ref=None,
                 superres_denom=None, base_q_idx=20, segments=None, segmentation=None,
                 loop_filter=(0, 0), ref_deltas=None, mode_deltas=None, grain=None,
                 tile_info=None):
    """An uncompressed header as bytes, with its length in bits.

    size -- a frame size, which sets frame_size_override_flag
    found_ref -- for an inter frame with a size override, the reference whose
        size is taken; None codes `size`
    short_signaling -- (last_frame_idx, gold_frame_idx)
    segments -- feature values as {(segment, feature): value}, coded with
        segmentation_update_data
    segmentation -- (update_map, temporal_update, update_data) where the
        frame has a primary reference frame
    ref_deltas, mode_deltas -- {index: value} updates to the loop filter deltas
    grain -- None for no grain; ``'new'`` for one luma point; a slot number to
        take the parameters of that reference; or a dictionary of the points
        to write as (value, scaling) pairs under ``y``, ``cb`` and ``cr``,
        with ``from_luma`` for chroma_scaling_from_luma
    tile_info -- None for one uniformly spaced tile; or a function that is
        given the writer and writes tile_info() from
        uniform_tile_spacing_flag on
    """
    writer = BitWriter()
    intra = frame_type in (KEY_FRAME, INTRA_ONLY_FRAME)
    writer.flag(0)                              # show_existing_frame
    writer.f(frame_type, 2)
    writer.flag(show_frame)
    if not show_frame:
        writer.flag(showable_frame)
    if frame_type == SWITCH_FRAME or (frame_type == KEY_FRAME and show_frame):
        error_resilient = 1
    else:
        writer.flag(error_resilient)
    writer.flag(0)                              # disable_cdf_update
    override = size is not None or found_ref is not None or frame_type == SWITCH_FRAME
    if frame_type != SWITCH_FRAME:
        writer.flag(override)
    writer.f(order_hint, 7)
    if not (intra or error_resilient):
        writer.f(primary_ref_frame, 3)
    else:
        primary_ref_frame = PRIMARY_REF_NONE
    if not (frame_type == SWITCH_FRAME or (frame_type == KEY_FRAME and show_frame)):
        writer.f(refresh, 8)
    else:
        refresh = 0xFF
    if (not intra or refresh != 0xFF) and error_resilient:
        for hint in ref_order_hints or (0,) * 8:
            writer.f(hint, 7)

    def frame_size():
        if override:
            width, height = size or sequence.size
            writer.f(width - 1, sequence.size_bits)
            writer.f(height - 1, sequence.size_bits)
        superres()

    def superres():
        if sequence.superres:
            writer.flag(superres_denom is not None)
            if superres_denom is not None:
                writer.f(superres_denom - 9, 3)

    if intra:
        frame_size()
        writer.flag(0)                          # render_and_frame_size_different
    else:
        writer.flag(short_signaling is not None)
        if short_signaling is not None:
            writer.f(short_signaling[0], 3)
            writer.f(short_signaling[1], 3)
        else:
            for slot in refs:
                writer.f(slot, 3)
        if override and not error_resilient:
            for index in range(7):
                writer.flag(index == found_ref)
                if index == found_ref:
                    superres()
                    break
            else:
                frame_size()
                writer.flag(0)
        else:
            frame_size()
            writer.flag(0)
        writer.flag(0)                          # allow_high_precision_mv
        writer.flag(0)                          # is_filter_switchable
        writer.f(2, 2)                          # interpolation_filter
        writer.flag(0)                          # is_motion_mode_switchable
    writer.flag(1)                              # disable_frame_end_update_cdf
    if tile_info is None:
        writer.flag(1)                          # uniform_tile_spacing_flag
    else:
        tile_info(writer)
    writer.f(base_q_idx, 8)
    writer.flag(0)                              # DeltaQYDc
    if not sequence.mono_chrome:
        writer.flag(0)                          # DeltaQUDc
        writer.flag(0)                          # DeltaQUAc
    writer.flag(0)                              # using_qmatrix
    writer.flag(segments is not None or segmentation is not None)
    if segments is not None or segmentation is not None:
        update_data = 1
        if primary_ref_frame != PRIMARY_REF_NONE:
            update_map, temporal_update, update_data = segmentation or (1, 0, 1)
            writer.flag(update_map)
            if update_map:
                writer.flag(temporal_update)
            writer.flag(update_data)
        if update_data:
            for segment in range(8):
                for feature, (bits, signed) in enumerate(
                        zip((8, 6, 6, 6, 6, 3, 0, 0), (1, 1, 1, 1, 1, 0, 0, 0), strict=True)):
                    value = (segments or {}).get((segment, feature))
                    writer.flag(value is not None)
                    if value is not None and signed:
                        writer.su(value, bits + 1)
                    elif value is not None:
                        writer.f(value, bits)
    if base_q_idx:
        writer.flag(0)                          # delta_q_present
    lossless = not base_q_idx and not any(
        feature == 0 and value > 0 for (_, feature), value in (segments or {}).items())
    if not lossless:
        writer.f(loop_filter[0], 6)
        writer.f(loop_filter[1], 6)
        if not sequence.mono_chrome and any(loop_filter):
            writer.f(1, 6)
            writer.f(2, 6)
        writer.f(3, 3)                          # loop_filter_sharpness
        updates = ref_deltas is not None or mode_deltas is not None
        writer.flag(1)                          # loop_filter_delta_enabled
        writer.flag(updates)
        if updates:
            for count, values in ((8, ref_deltas or {}), (2, mode_deltas or {})):
                for index in range(count):
                    writer.flag(index in values)
                    if index in values:
                        writer.su(values[index], 7)
        writer.flag(1)                          # tx_mode_select
    if not intra:
        writer.flag(0)                          # reference_select
    writer.flag(0)                              # reduced_tx_set
    if not intra:
        for _ in range(7):
            writer.flag(0)                      # is_global
    if sequence.film_grain and (show_frame or showable_frame):
        writer.flag(grain is not None)
        if grain is not None:
            writer.f(0x1234, 16)                # grain_seed
            if frame_type == INTER_FRAME:
                writer.flag(grain == 'new')     # update_grain
            if isinstance(grain, dict):
                _grain_points(writer, sequence, grain)
            elif grain != 'new':
                writer.f(grain, 3)              # film_grain_params_ref_idx
            else:
                writer.f(1, 4)                  # num_y_points
                writer.f(40, 8)
                writer.f(90, 8)
                if not sequence.mono_chrome:
                    writer.flag(1)              # chroma_scaling_from_luma
                writer.f(2, 2)                  # grain_scaling_minus_8
                writer.f(1, 2)                  # ar_coeff_lag
                for value in (120, 130, 140, 150):
                    writer.f(value, 8)          # ar_coeffs_y_plus_128
                if not sequence.mono_chrome:
                    for value in (100, 110, 128, 129, 131, 132, 133, 134, 135, 136):
                        writer.f(value, 8)      # ar_coeffs_cb_plus_128, then cr
                writer.f(1, 2)                  # ar_coeff_shift_minus_6
                writer.f(2, 2)                  # grain_scale_shift
                writer.flag(1)                  # overlap_flag
                writer.flag(0)                  # clip_to_restricted_range
    return writer.bytes(), len(writer)


def _grain_points(writer, sequence, grain):
    """Write film_grain_params() from num_y_points on, for the points in `grain`."""
    luma = grain.get('y', ())
    from_luma = grain.get('from_luma', 0)
    chroma = [grain.get('cb', ()), grain.get('cr', ())]

    def points(found):
        writer.f(len(found), 4)
        for value, scaling in found:
            writer.f(value, 8)
            writer.f(scaling, 8)

    points(luma)
    if not sequence.mono_chrome:
        writer.flag(from_luma)                  # chroma_scaling_from_luma
    if sequence.mono_chrome or from_luma or not luma:
        chroma = [(), ()]
    else:
        for found in chroma:
            points(found)
    writer.f(0, 2)                              # grain_scaling_minus_8
    writer.f(0, 2)                              # ar_coeff_lag
    for found in chroma:
        if from_luma or found:
            # At lag 0 a chroma plane has one coefficient, and only where
            # there are luma points.
            for _ in range(1 if luma else 0):
                writer.f(128, 8)
    writer.f(0, 2)                              # ar_coeff_shift_minus_6
    writer.f(0, 2)                              # grain_scale_shift
    for found in chroma:
        if found:
            writer.f(128, 8)                    # cb_mult or cr_mult
            writer.f(192, 8)                    # cb_luma_mult or cr_luma_mult
            writer.f(256, 9)                    # cb_offset or cr_offset
    writer.flag(0)                              # overlap_flag
    writer.flag(0)                              # clip_to_restricted_range


def ns(writer, value, count):
    """Write `value` as ns(`count`): one of `count` values, the low ones a bit shorter."""
    width = count.bit_length()
    spare = (1 << width) - count
    if value < spare:
        writer.f(value, width - 1)
    else:
        writer.f((value + spare) >> 1, width - 1)
        writer.f((value + spare) & 1, 1)


def stated_tiles(columns, rows, *, sb_cols, sb_rows, context_update_tile_id=0,
                 tile_size_bytes=1, widest=64, tallest=None):
    """A `tile_info` for :func:`frame_header`: tile sizes stated one by one.

    columns, rows -- each tile column's width and each tile row's height, in
        superblocks
    sb_cols, sb_rows -- the frame's size in superblocks
    widest, tallest -- maxTileWidthSb and maxTileHeightSb as tile_info()
        derives them for the frame; `tallest` is the frame's height where
        nothing is given
    """
    def write(writer):
        writer.flag(0)                          # uniform_tile_spacing_flag
        for sizes, total, most in ((columns, sb_cols, widest),
                                   (rows, sb_rows, tallest or sb_rows)):
            start = 0
            for size in sizes:
                ns(writer, size - 1, min(total - start, most))
                start += size
        bits = (len(columns) - 1).bit_length() + (len(rows) - 1).bit_length()
        if bits:
            writer.f(context_update_tile_id, bits)
            writer.f(tile_size_bytes - 1, 2)
    return write


def uniform_tiles(column_increments, row_increments, *, bits=0, context_update_tile_id=0,
                  tile_size_bytes=1):
    """A `tile_info` for :func:`frame_header`: uniformly spaced tiles.

    column_increments, row_increments -- the increment_tile_cols_log2 and
        increment_tile_rows_log2 flags to write
    bits -- TileColsLog2 + TileRowsLog2 as those flags leave them, which is
        how wide context_update_tile_id is
    """
    def write(writer):
        writer.flag(1)                          # uniform_tile_spacing_flag
        for flag in (*column_increments, *row_increments):
            writer.flag(flag)
        if bits:
            writer.f(context_update_tile_id, bits)
            writer.f(tile_size_bytes - 1, 2)
    return write


def show_existing(slot):
    """A frame header OBU's payload that shows the frame in `slot`."""
    writer = BitWriter()
    writer.flag(1)
    writer.f(slot, 3)
    writer.flag(1)                              # trailing_one_bit
    return writer.bytes()


def frame(sequence=DEFAULT, tile=b'\x55' * 6, **options):
    """A frame OBU: the header :func:`frame_header` writes, then one tile."""
    header, _ = frame_header(sequence, **options)
    return obu(obus.FRAME, header + tile)


def unit(*units):
    """A temporal unit: a temporal delimiter, then `units`."""
    return TEMPORAL_DELIMITER + b''.join(units)


def headers_of(samples):
    """Every frame header of a stream, with the sequence header it was read under.

    samples -- the data of each temporal unit, in order

    The reference slots are updated after each frame as a decoder does, so
    each header is read against what the frames before it left.
    """
    state = FrameState()
    sequence = None
    found = []
    for data in samples:
        gathering = None
        for unit_ in obus.split(data):
            if unit_.kind == obus.SEQUENCE_HEADER:
                sequence = parse_sequence_header(unit_.payload)
            elif unit_.kind in (obus.FRAME, obus.FRAME_HEADER) and gathering is None:
                header = parse_frame_header(unit_.payload, sequence, state)
                found.append((header, sequence))
                if header.show_existing_frame:
                    state.update(header, sequence)
                elif unit_.kind == obus.FRAME:
                    group = parse_tile_group(unit_.payload, header, (header.header_bits + 7) // 8)
                    if group.tg_end == header.tile_info.num_tiles - 1:
                        state.update(header, sequence)
                    else:
                        gathering = header
                else:
                    gathering = header
            elif unit_.kind == obus.TILE_GROUP and gathering is not None:
                group = parse_tile_group(unit_.payload, gathering)
                if group.tg_end == gathering.tile_info.num_tiles - 1:
                    state.update(gathering, sequence)
                    gathering = None
    return found
