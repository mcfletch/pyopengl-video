"""AV1 decoding's control layer: the buffers libva takes, and the reference slots.

:mod:`~pyopengl_video.vaapi.av1_parse` reads the syntax and keeps what each
reference slot saved of its frame. The functions here turn a parsed frame into
the parameter buffers libva takes, and :class:`AV1Decoding` gathers each frame
from its OBUs, keeps the surface in each of the eight reference slots and
shows the frames the stream says to show. Nothing here holds a driver.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from pyopengl_video import codecs, obu
from pyopengl_video.decoder import DecoderError, Sample
from pyopengl_video.vaapi import api, av1_api
from pyopengl_video.vaapi.av1_parse import (
    ALTREF_FRAME,
    KEY_FRAME,
    LAST_FRAME,
    MAX_TILE_COLS,
    MAX_TILE_ROWS,
    NUM_REF_FRAMES,
    REFS_PER_FRAME,
    SAMPLINGS,
    FilmGrain,
    FrameHeader,
    FrameState,
    MissingReference,
    SequenceHeader,
    StreamError,
    Tile,
    colour_of,
    parse_frame_header,
    parse_sequence_header,
    parse_tile_group,
    same_sequence,
    warp_valid,
)
from pyopengl_video.vaapi.control import Buffer, DecodeControl, DecodeSession, Format

__all__ = ['AV1Decoding', 'colour_of', 'format_for', 'picture_parameters',
           'tile_parameters']

#: seq_profile values by name, for refusing a stream in words a reader knows.
PROFILE_NAMES = {0: 'Main', 1: 'High', 2: 'Professional'}

#: The luma samples a frame's coded size is a multiple of (MiCols counts 4x4
#: blocks and is even).
BLOCK = 8

#: How many tile widths, and how many tile heights, libva's picture parameters
#: hold. A frame has at most 64 tiles along an axis, and the size of the 64th
#: is not stated: it is what is left of the frame.
STATED_TILE_SIZES = MAX_TILE_COLS - 1


def format_for(sequence: SequenceHeader) -> Format:
    """The surfaces a sequence decodes into; refuses what is not decoded here."""
    name = PROFILE_NAMES.get(sequence.seq_profile, f'seq_profile {sequence.seq_profile}')
    color = sequence.color_config
    if sequence.seq_profile != 0:
        raise DecoderError(
            f'the stream is AV1 {name} profile, and VA-API decoding here covers '
            'Main profile (4:2:0 or monochrome, at 8 or 10 bits)')
    sampling = (color.subsampling_x, color.subsampling_y)
    if not color.mono_chrome and sampling != (1, 1):
        raise DecoderError(
            f'the stream is AV1 {name} profile with {SAMPLINGS[sampling]} sampling, '
            'and the surfaces of that profile are 4:2:0 or monochrome')
    width, height = sequence.size
    # Every slot may hold a frame of its own, and a frame with film grain is
    # two surfaces: the one later frames predict from and the one shown.
    each = 2 if sequence.film_grain_params_present else 1
    return Format(
        profile=api.VAProfileAV1Profile0,
        coded_size=(-(-width // BLOCK) * BLOCK, -(-height // BLOCK) * BLOCK),
        surfaces=(NUM_REF_FRAMES + 1) * each, depth=color.bit_depth,
        name=f'AV1 {name}')


def _film_grain(grain: FilmGrain, target: av1_api.VAFilmGrainStructAV1) -> None:
    if not grain.apply_grain:
        return
    bits = target.film_grain_info_fields.bits
    bits.apply_grain = 1
    bits.chroma_scaling_from_luma = grain.chroma_scaling_from_luma
    bits.grain_scaling_minus_8 = grain.grain_scaling_minus_8
    bits.ar_coeff_lag = grain.ar_coeff_lag
    bits.ar_coeff_shift_minus_6 = grain.ar_coeff_shift_minus_6
    bits.grain_scale_shift = grain.grain_scale_shift
    bits.overlap_flag = grain.overlap_flag
    bits.clip_to_restricted_range = grain.clip_to_restricted_range
    target.grain_seed = grain.grain_seed
    target.num_y_points = grain.num_y_points
    target.num_cb_points = grain.num_cb_points
    target.num_cr_points = grain.num_cr_points
    for values, field in (
        (grain.point_y_value, target.point_y_value),
        (grain.point_y_scaling, target.point_y_scaling),
        (grain.point_cb_value, target.point_cb_value),
        (grain.point_cb_scaling, target.point_cb_scaling),
        (grain.point_cr_value, target.point_cr_value),
        (grain.point_cr_scaling, target.point_cr_scaling),
    ):
        for index, value in enumerate(values):
            field[index] = value
    for coded, field in ((grain.ar_coeffs_y, target.ar_coeffs_y),
                         (grain.ar_coeffs_cb, target.ar_coeffs_cb),
                         (grain.ar_coeffs_cr, target.ar_coeffs_cr)):
        for index, value in enumerate(coded):
            field[index] = value - 128
    target.cb_mult, target.cb_luma_mult = grain.cb_mult, grain.cb_luma_mult
    target.cb_offset = grain.cb_offset
    target.cr_mult, target.cr_luma_mult = grain.cr_mult, grain.cr_luma_mult
    target.cr_offset = grain.cr_offset


def picture_parameters(sequence: SequenceHeader, header: FrameHeader,
                       references: Sequence[int | None], surface: int,
                       display: int | None = None,
                       ) -> av1_api.VADecPictureParameterBufferAV1:
    """The picture parameter buffer for the frame `header` opens.

    references -- the surface in each of the eight reference slots, None for
        a slot that holds no frame
    surface -- where the frame is reconstructed
    display -- where it is written with film grain added, for a frame that
        applies grain
    """
    color = sequence.color_config
    parameters = av1_api.VADecPictureParameterBufferAV1()
    parameters.profile = sequence.seq_profile
    parameters.order_hint_bits_minus_1 = max(sequence.order_hint_bits - 1, 0)
    parameters.bit_depth_idx = (color.bit_depth - 8) >> 1
    parameters.matrix_coefficients = color.matrix_coefficients
    fields = parameters.seq_info_fields.fields
    fields.still_picture = sequence.still_picture
    fields.use_128x128_superblock = sequence.use_128x128_superblock
    fields.enable_filter_intra = sequence.enable_filter_intra
    fields.enable_intra_edge_filter = sequence.enable_intra_edge_filter
    fields.enable_interintra_compound = sequence.enable_interintra_compound
    fields.enable_masked_compound = sequence.enable_masked_compound
    fields.enable_dual_filter = sequence.enable_dual_filter
    fields.enable_order_hint = sequence.enable_order_hint
    fields.enable_jnt_comp = sequence.enable_jnt_comp
    fields.enable_cdef = sequence.enable_cdef
    fields.mono_chrome = color.mono_chrome
    fields.color_range = color.color_range
    fields.subsampling_x = color.subsampling_x
    fields.subsampling_y = color.subsampling_y
    fields.film_grain_params_present = sequence.film_grain_params_present

    parameters.current_frame = surface
    grain = header.film_grain
    parameters.current_display_picture = (
        display if grain.apply_grain and display is not None else api.VA_INVALID_SURFACE)
    parameters.frame_width_minus1 = header.upscaled_width - 1
    parameters.frame_height_minus1 = header.frame_height - 1
    for index in range(NUM_REF_FRAMES):
        held = references[index] if index < len(references) else None
        parameters.ref_frame_map[index] = api.VA_INVALID_SURFACE if held is None else held
    for index in range(REFS_PER_FRAME):
        parameters.ref_frame_idx[index] = header.ref_frame_idx[index]
    parameters.primary_ref_frame = header.primary_ref_frame
    parameters.order_hint = header.order_hint

    segmentation = header.segmentation
    segment = parameters.seg_info.segment_info_fields.bits
    segment.enabled = segmentation.enabled
    segment.update_map = segmentation.update_map
    segment.temporal_update = segmentation.temporal_update
    segment.update_data = segmentation.update_data
    for index, (enabled, data) in enumerate(
            zip(segmentation.feature_enabled, segmentation.feature_data, strict=True)):
        mask = 0
        for feature, (flag, value) in enumerate(zip(enabled, data, strict=True)):
            mask |= flag << feature
            parameters.seg_info.feature_data[index][feature] = value
        parameters.seg_info.feature_mask[index] = mask
    _film_grain(grain, parameters.film_grain_info)

    tiles = header.tile_info
    if tiles.tile_cols > MAX_TILE_COLS or tiles.tile_rows > MAX_TILE_ROWS:
        raise DecoderError(
            f'the frame has {tiles.tile_cols} tile columns and {tiles.tile_rows} tile '
            f'rows, and a driver is told of at most {MAX_TILE_COLS} and {MAX_TILE_ROWS}')
    parameters.tile_cols, parameters.tile_rows = tiles.tile_cols, tiles.tile_rows
    for sizes, field in ((tiles.width_in_sbs_minus_1, parameters.width_in_sbs_minus_1),
                         (tiles.height_in_sbs_minus_1, parameters.height_in_sbs_minus_1)):
        for index, size in enumerate(sizes[:STATED_TILE_SIZES]):
            field[index] = size
    parameters.context_update_tile_id = tiles.context_update_tile_id

    bits = parameters.pic_info_fields.bits
    bits.frame_type = header.frame_type
    bits.show_frame = header.show_frame
    bits.showable_frame = header.showable_frame
    bits.error_resilient_mode = header.error_resilient_mode
    bits.disable_cdf_update = header.disable_cdf_update
    bits.allow_screen_content_tools = header.allow_screen_content_tools
    bits.force_integer_mv = header.force_integer_mv
    bits.allow_intrabc = header.allow_intrabc
    bits.use_superres = header.use_superres
    bits.allow_high_precision_mv = header.allow_high_precision_mv
    bits.is_motion_mode_switchable = header.is_motion_mode_switchable
    bits.use_ref_frame_mvs = header.use_ref_frame_mvs
    bits.disable_frame_end_update_cdf = header.disable_frame_end_update_cdf
    bits.uniform_tile_spacing_flag = tiles.uniform_tile_spacing_flag
    bits.allow_warped_motion = header.allow_warped_motion
    parameters.superres_scale_denominator = header.superres_denom
    parameters.interp_filter = header.interpolation_filter

    loop_filter = header.loop_filter
    parameters.filter_level[0], parameters.filter_level[1] = loop_filter.level[:2]
    parameters.filter_level_u, parameters.filter_level_v = loop_filter.level[2:]
    info = parameters.loop_filter_info_fields.bits
    info.sharpness_level = loop_filter.sharpness
    info.mode_ref_delta_enabled = loop_filter.delta_enabled
    info.mode_ref_delta_update = loop_filter.delta_update
    for index, delta in enumerate(loop_filter.ref_deltas):
        parameters.ref_deltas[index] = delta
    for index, delta in enumerate(loop_filter.mode_deltas):
        parameters.mode_deltas[index] = delta

    quantization = header.quantization
    parameters.base_qindex = quantization.base_q_idx
    parameters.y_dc_delta_q = quantization.delta_q_y_dc
    parameters.u_dc_delta_q = quantization.delta_q_u_dc
    parameters.u_ac_delta_q = quantization.delta_q_u_ac
    parameters.v_dc_delta_q = quantization.delta_q_v_dc
    parameters.v_ac_delta_q = quantization.delta_q_v_ac
    matrix = parameters.qmatrix_fields.bits
    matrix.using_qmatrix = quantization.using_qmatrix
    matrix.qm_y, matrix.qm_u, matrix.qm_v = (
        quantization.qm_y, quantization.qm_u, quantization.qm_v)

    mode = parameters.mode_control_fields.bits
    mode.delta_q_present_flag = header.delta_q_present
    mode.log2_delta_q_res = header.delta_q_res
    mode.delta_lf_present_flag = header.delta_lf_present
    mode.log2_delta_lf_res = header.delta_lf_res
    mode.delta_lf_multi = header.delta_lf_multi
    mode.tx_mode = header.tx_mode
    mode.reference_select = header.reference_select
    mode.reduced_tx_set_used = header.reduced_tx_set
    mode.skip_mode_present = header.skip_mode_present

    cdef = header.cdef
    parameters.cdef_damping_minus_3 = cdef.damping_minus_3
    parameters.cdef_bits = cdef.bits
    # libva takes the secondary strength as its two coded bits, where 3 means 4.
    for primaries, secondaries, field in (
        (cdef.y_pri_strength, cdef.y_sec_strength, parameters.cdef_y_strengths),
        (cdef.uv_pri_strength, cdef.uv_sec_strength, parameters.cdef_uv_strengths),
    ):
        for index, (primary, secondary) in enumerate(zip(primaries, secondaries, strict=True)):
            field[index] = primary << 2 | min(secondary, 3)

    restoration = header.loop_restoration
    restore = parameters.loop_restoration_fields.bits
    (restore.yframe_restoration_type, restore.cbframe_restoration_type,
     restore.crframe_restoration_type) = restoration.frame_restoration_type
    restore.lr_unit_shift = restoration.unit_shift
    restore.lr_uv_shift = restoration.uv_shift

    for ref in range(LAST_FRAME, ALTREF_FRAME + 1):
        warp = parameters.wm[ref - LAST_FRAME]
        warp.wmtype = header.gm_type[ref]
        for index, value in enumerate(header.gm_params[ref]):
            warp.wmmat[index] = value
        warp.invalid = int(not warp_valid(header.gm_params[ref]))
    return parameters


def tile_parameters(tile: Tile) -> av1_api.VASliceParameterBufferAV1:
    """The slice parameter buffer naming one tile's bytes in the data buffer after it."""
    parameters = av1_api.VASliceParameterBufferAV1()
    parameters.slice_data_size = tile.size
    parameters.slice_data_offset = tile.offset
    parameters.slice_data_flag = api.VA_SLICE_DATA_FLAG_ALL
    parameters.tile_row = tile.row
    parameters.tile_column = tile.column
    return parameters


# ------------------------------------------------------------ the layer


@dataclasses.dataclass(frozen=True)
class _Picture:
    """A decoded frame: the surface predicted from, and the one shown.

    The two differ for a frame with film grain, whose grain is added into a
    surface of its own.
    """

    surface: int
    display: int
    size: tuple[int, int]


class _Current:
    """The frame being gathered from its tile groups.

    sequence -- the sequence header the frame's header was read under, which
        its tile groups are decoded under too
    surface -- where the frame is reconstructed
    display -- where it is written with film grain added, for a frame that
        applies grain
    batches -- the buffers to submit: the picture parameters, then each tile
        group's
    """

    def __init__(self, header: FrameHeader, sequence: SequenceHeader, sample: Sample,
                 surface: int, display: int | None,
                 parameters: av1_api.VADecPictureParameterBufferAV1) -> None:
        self.header = header
        self.sequence = sequence
        self.sample = sample
        self.surface = surface
        self.display = display
        self.batches: list[list[Buffer]] = [[(api.VAPictureParameterBufferType, parameters)]]
        self.next_tile = 0


class AV1Decoding(DecodeControl):
    """AV1 decoding: Main profile, 4:2:0 or monochrome, at 8 or 10 bits.

    Each sample is one temporal unit, its OBUs carrying their sizes as MP4 and
    IVF hold them. A frame is shown when the stream shows it: as it is
    decoded, or later by a header that shows the frame in a reference slot.
    The temporal unit's timestamp goes on every frame it shows.

    A frame with film grain is decoded into two surfaces, the grain added in
    the one shown. A monochrome stream decodes into the same two-plane
    surfaces as any other, and its chroma plane is whatever the driver leaves
    there (radeonsi leaves the mid value).

    The size of a frame is its coded size after super-resolution. The render
    size a frame header may state is not applied.

    Refused by name: High and Professional profile, an inter frame whose
    references are another size (reference scaling), frames in a temporal or
    spatial layer above the base, and tile list OBUs (large scale tile).

    A sequence header that differs from the one before starts a new sequence,
    which begins at a key frame that is shown (7.5): any other frame after it
    is a :class:`~pyopengl_video.vaapi.h264.StreamError` while a reference
    slot holds a frame of the sequence before, and so is one that arrives
    while a frame is being gathered.

    After :meth:`reset`, and at the start, frames that refer to a frame never
    decoded are passed over until a key frame has been shown.

    A :meth:`decode` that raises leaves no frame being gathered: the tile
    groups of the frame it was reading are not decoded when they arrive.
    """

    codec = codecs.AV1

    def __init__(self, session: DecodeSession, output_slots: int) -> None:
        super().__init__(session, output_slots)
        self._sequence: SequenceHeader | None = None
        self._sequence_bytes = b''
        #: The sequence header the frames in the reference slots were decoded
        #: under.
        self._slots_sequence: SequenceHeader | None = None
        self._state = FrameState()
        self._slots: list[_Picture | None] = [None] * NUM_REF_FRAMES
        self._format: Format | None = None
        #: The frame whose tile groups are being gathered, its picture
        #: parameters built.
        self._current: _Current | None = None
        #: The surfaces acquired for a frame whose picture parameters are not
        #: built yet, which :meth:`held` counts.
        self._acquired: list[int] = []
        #: Whether the tile groups arriving belong to a frame passed over.
        self._passing = False
        self._recovering = True

    # ------------------------------------------------------------ headers

    def configure(self, headers: bytes) -> None:
        for unit in obu.split(headers):
            if unit.kind == obu.SEQUENCE_HEADER:
                self._sequence_header(unit.payload)

    def _sequence_header(self, payload: bytes) -> None:
        if payload == self._sequence_bytes:
            return
        sequence = parse_sequence_header(payload)
        held = self._sequence
        if (self._current is not None and held is not None
                and not same_sequence(held, sequence)):
            raise StreamError(
                'a sequence header that changes the sequence arrived inside a '
                f'frame, between the header of the {self._current.header.described} '
                'and its last tile group')
        self._sequence = sequence
        self._sequence_bytes = payload
        self.size = sequence.size
        self.colour = colour_of(sequence)

    def _in_operating_point(self, unit: obu.Obu) -> bool:
        """Whether the operating point decoded includes the unit's layer (5.3.1)."""
        idc = self._sequence.operating_point_idc if self._sequence is not None else 0
        if not unit.has_extension or not idc:
            return True
        return bool((idc >> unit.temporal_id) & 1 and (idc >> (unit.spatial_id + 8)) & 1)

    # ----------------------------------------------------------- decoding

    def decode(self, sample: Sample) -> None:
        try:
            for unit in obu.split(sample.data):
                self._unit(unit, sample)
        except BaseException:
            # The frame being gathered is not one a later tile group may
            # finish: its header, or one of its tile groups, was refused.
            self._current = None
            self._passing = False
            raise

    def _unit(self, unit: obu.Obu, sample: Sample) -> None:
        kind = unit.kind
        if kind == obu.SEQUENCE_HEADER:
            self._sequence_header(unit.payload)
        elif kind == obu.TEMPORAL_DELIMITER:
            self._current = None
            self._passing = False
        elif not self._in_operating_point(unit):
            return
        elif kind == obu.FRAME_HEADER:
            # A header that arrives while a frame is being gathered is a
            # copy of that frame's header (5.9.1).
            if self._current is None:
                self._begin(unit, sample)
        elif kind == obu.REDUNDANT_FRAME_HEADER:
            if self._current is None and not self._passing:
                self._begin(unit, sample)
        elif kind == obu.FRAME:
            header = self._begin(unit, sample)
            if header is not None:
                self._tile_group(unit.payload, (header.header_bits + 7) // 8)
        elif kind == obu.TILE_GROUP:
            if not self._passing:
                self._tile_group(unit.payload, 0)
        elif kind == obu.TILE_LIST:
            raise DecoderError(
                'the stream carries a tile list OBU (large scale tile), which '
                'this decoder does not decode')

    def _begin(self, unit: obu.Obu, sample: Sample) -> FrameHeader | None:
        """Read a frame's header: show the frame it names, or start gathering one.

        None for a frame that is passed over: one that refers to a frame never
        decoded, before the first key frame has been shown. Its tile groups
        are passed over as they arrive.

        The frame is gathered once its surfaces are held and its picture
        parameters are built, and not before: a header that is refused on the
        way leaves no frame for a tile group to be added to.
        """
        self._current = None
        self._passing = False
        sequence = self._sequence
        if sequence is None:
            raise StreamError('a frame arrived before any sequence header')
        if unit.temporal_id or unit.spatial_id:
            raise DecoderError(
                f'the stream is scalable: it has a frame in temporal layer '
                f'{unit.temporal_id}, spatial layer {unit.spatial_id}, and this '
                'decoder takes streams with the base layer alone')
        try:
            header = parse_frame_header(unit.payload, sequence, self._state,
                                        temporal_id=unit.temporal_id,
                                        spatial_id=unit.spatial_id)
            self._check_sequence_start(sequence, header)
            if header.show_existing_frame:
                self._show_existing(header, sequence, sample)
                return header
            self._check_references(header)
        except MissingReference:
            if not self._recovering:
                raise
            self._passing = True
            return None
        try:
            self._activate(sequence, header)
            self._check_fits(header)
            surface = self._acquire()
            display = self._acquire() if header.film_grain.apply_grain else None
            parameters = picture_parameters(
                sequence, header,
                [held.surface if held is not None else None for held in self._slots],
                surface, display)
        finally:
            self._acquired = []
        self._current = _Current(header, sequence, sample, surface, display, parameters)
        return header

    def _acquire(self) -> int:
        """A surface to decode into, held from here until its frame is gathered or refused."""
        surface = self.session.acquire()
        self._acquired.append(surface)
        return surface

    def _check_sequence_start(self, sequence: SequenceHeader, header: FrameHeader) -> None:
        """Refuse a sequence that begins anywhere but at a key frame that is shown (7.5).

        Such a key frame replaces every reference slot, so no frame of the
        new sequence predicts from one of the sequence before. With no slot
        holding a frame there is nothing of the sequence before to predict
        from, and any frame may begin.
        """
        before = self._slots_sequence
        if before is None or before is sequence or same_sequence(before, sequence):
            return
        starts = (header.frame_type == KEY_FRAME and header.show_frame
                  and not header.show_existing_frame)
        if not starts and any(held is not None for held in self._slots):
            what = ('header that shows a reference slot' if header.show_existing_frame
                    else header.described)
            raise StreamError(
                f'the sequence header changed before a {what}, and a new sequence '
                'begins at a key frame that is shown')

    def _check_references(self, header: FrameHeader) -> None:
        """Refuse an inter frame whose references are not there or not its size."""
        if header.frame_is_intra:
            return
        for name, slot in enumerate(header.ref_frame_idx, LAST_FRAME):
            held = self._slots[slot]
            if held is None:
                raise MissingReference(
                    f'the {header.described} refers to reference slot {slot}, which '
                    'holds no frame')
            if held.size != header.size:
                raise DecoderError(
                    f'the {header.described} is {header.size[0]}x{header.size[1]} and '
                    f'its reference {name} is {held.size[0]}x{held.size[1]}: the stream '
                    'uses reference scaling, which this decoder does not decode')

    def _check_fits(self, header: FrameHeader) -> None:
        """Refuse a frame larger than the surfaces it would be decoded into."""
        width, height = header.size
        coded = self._format.coded_size if self._format is not None else (0, 0)
        if width > coded[0] or height > coded[1]:
            raise StreamError(
                f'the {header.described} is {width}x{height}, and the surfaces of its '
                f'sequence are {coded[0]}x{coded[1]}')

    def _activate(self, sequence: SequenceHeader, header: FrameHeader) -> None:
        """Make sure the surfaces suit the sequence this frame belongs to.

        A sequence of another size or depth begins at a key frame. Every
        frame of the sequence before has been shown by then, and its
        references are dropped with its surfaces.
        """
        wanted = format_for(sequence)
        held = self._format
        if held is not None and (held.profile, held.coded_size, held.depth, held.surfaces) == (
                wanted.profile, wanted.coded_size, wanted.depth, wanted.surfaces):
            return
        if held is not None and header.frame_type != KEY_FRAME:
            raise StreamError(
                f'the sequence changed to {wanted.coded_size[0]}x{wanted.coded_size[1]} '
                f'at {wanted.depth} bits at {header.kind} frame, and a new sequence '
                'begins at a key frame')
        self._slots = [None] * NUM_REF_FRAMES
        self._state.reset()
        self.session.activate(wanted)
        self._format = wanted

    def _tile_group(self, data: bytes, start: int) -> None:
        """Add a tile group to the frame being gathered; decode the frame at its last tile."""
        current = self._current
        if current is None:
            raise StreamError('a tile group arrived with no frame header before it')
        header = current.header
        group = parse_tile_group(data, header, start)
        if group.tg_start != current.next_tile:
            raise StreamError(
                f'a tile group begins at tile {group.tg_start} where tile '
                f'{current.next_tile} was due')
        current.next_tile = group.tg_end + 1
        batch: list[Buffer] = [(api.VASliceParameterBufferType, tile_parameters(tile))
                               for tile in group.tiles]
        batch.append((api.VASliceDataBufferType, data))
        current.batches.append(batch)
        if current.next_tile == header.tile_info.num_tiles:
            self._current = None
            self._finish(current)

    def _finish(self, current: _Current) -> None:
        """Decode the gathered frame, save it in the slots it refreshes, show it."""
        header = current.header
        self.session.submit(current.surface, current.batches, header.described)
        picture = _Picture(
            current.surface,
            current.surface if current.display is None else current.display, header.size)
        self._state.update(header, current.sequence)
        self._slots_sequence = current.sequence
        for slot in range(NUM_REF_FRAMES):
            if (header.refresh_frame_flags >> slot) & 1:
                self._slots[slot] = picture
        if header.show_frame:
            if header.frame_type == KEY_FRAME:
                self._recovering = False
            self._show(picture, current.sample)

    def _show_existing(self, header: FrameHeader, sequence: SequenceHeader,
                       sample: Sample) -> None:
        """Show the frame in a reference slot again (7.21).

        A key frame shown this way starts a new sequence of references: every
        slot then holds it.
        """
        slot = header.frame_to_show_map_idx
        picture = self._slots[slot]
        if picture is None:
            raise MissingReference(
                f'a frame header shows reference slot {slot}, which holds no frame')
        self._state.update(header, sequence)
        if header.frame_type == KEY_FRAME:
            self._slots = [picture] * NUM_REF_FRAMES
            self._recovering = False
        self._show(picture, sample)

    def _show(self, picture: _Picture, sample: Sample) -> None:
        width, height = picture.size
        self.size = picture.size
        self.session.output(picture.display, timestamp=sample.timestamp,
                            duration=sample.duration, visible=(0, 0, width, height),
                            colour=self.colour)

    # ------------------------------------------------------------ the rest

    def flush(self) -> None:
        """Nothing is held back: every frame is shown as the stream shows it."""
        self._current = None
        self._passing = False

    def reset(self) -> None:
        self._current = None
        self._passing = False
        self._slots = [None] * NUM_REF_FRAMES
        self._state.reset()
        self._recovering = True

    def held(self) -> set[int]:
        found: set[int] = set()
        for picture in self._slots:
            if picture is not None:
                found.update((picture.surface, picture.display))
        found.update(self._acquired)
        current = self._current
        if current is not None:
            found.add(current.surface)
            if current.display is not None:
                found.add(current.display)
        return found
