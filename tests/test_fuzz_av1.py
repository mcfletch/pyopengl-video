"""The AV1 fuzz gate: mutated fixtures from the parser to the driver's buffers.

Every mutated stream either decodes or is refused with the package's own
error, in under a second, and whatever is handed to the session is inside
the range the AV1 Bitstream & Decoding Process Specification gives it and
inside what libva's structures hold. The findings these checks first made are
A1 to A14 of ``plans/SECURITY-REVIEW.md``.
"""
import pytest

from pyopengl_video import obu
from pyopengl_video.nvdec import description
from pyopengl_video.vaapi import api, av1_api, av1_decode
from pyopengl_video.vaapi.av1_parse import (
    KEY_FRAME,
    MAX_TILE_COLS,
    MAX_TILE_ROWS,
    PRIMARY_REF_NONE,
    StreamError,
)
from tests import ivf_fuzz

CASES = 2500
SEED = 20261013

#: How many bytes of a frame or tile group OBU's payload are taken as header.
HEADER_BYTES = 48


def header_regions(data):
    """The byte ranges of a temporal unit that hold headers."""
    regions = []
    position = 0
    try:
        for unit in obu.split(data):
            start = position + len(unit.raw) - len(unit.payload)
            if unit.kind == obu.SEQUENCE_HEADER:
                regions.append((position, position + len(unit.raw)))
            elif unit.kind in (obu.FRAME, obu.FRAME_HEADER, obu.TILE_GROUP):
                regions.append((position, min(start + HEADER_BYTES, position + len(unit.raw))))
            position += len(unit.raw)
    except StreamError:
        pass
    return regions or [(0, len(data))]


# ------------------------------------------------------------ what is parsed


def check_sequence(findings, sequence):
    within = findings.within
    within('seq_profile', sequence.seq_profile, 0, 2)
    if sequence.frame_id_numbers_present_flag:
        within('idLen', sequence.id_len, 3, 16)
    color = sequence.color_config
    sampling = (color.subsampling_x, color.subsampling_y)
    if sequence.seq_profile == 0 and sampling != (1, 1):
        findings.report('Main profile that is not 4:2:0', sampling)
    if not color.mono_chrome and color.matrix_coefficients == 0 and sampling != (0, 0):
        findings.report('the identity matrix with subsampled chroma', sampling)
    within('max_frame_width', sequence.size[0], 1, 1 << 16)
    within('max_frame_height', sequence.size[1], 1, 1 << 16)


def check_header(findings, sequence, header):
    """A frame header that became picture parameters."""
    report, within = findings.report, findings.within
    most_width, most_height = sequence.size
    within('UpscaledWidth', header.upscaled_width, 1, most_width)
    within('FrameWidth', header.frame_width, 1, header.upscaled_width)
    within('FrameHeight', header.frame_height, 1, most_height)
    within('SuperresDenom', header.superres_denom, 8, 16)
    tiles = header.tile_info
    within('TileCols', tiles.tile_cols, 1, MAX_TILE_COLS)
    within('TileRows', tiles.tile_rows, 1, MAX_TILE_ROWS)
    within('context_update_tile_id', tiles.context_update_tile_id, 0, tiles.num_tiles - 1)
    within('TileSizeBytes', tiles.tile_size_bytes, 1, 4)
    superblock = 128 if sequence.use_128x128_superblock else 64
    for size in tiles.width_in_sbs_minus_1:
        within('a tile width in samples', (size + 1) * superblock, superblock, 4096)
    if (len(tiles.width_in_sbs_minus_1), len(tiles.height_in_sbs_minus_1)) != (
            tiles.tile_cols, tiles.tile_rows):
        report('tile sizes that are not one for each tile')
    if header.frame_type != KEY_FRAME and header.frame_is_intra and (
            header.refresh_frame_flags == 0xFF):
        report('an intra-only frame that replaces every slot')
    within('primary_ref_frame', header.primary_ref_frame, 0, PRIMARY_REF_NONE)
    for slot in header.ref_frame_idx:
        within('ref_frame_idx', slot, 0, 7)
    quantization = header.quantization
    for delta in (quantization.delta_q_y_dc, quantization.delta_q_u_dc, quantization.delta_q_u_ac,
                  quantization.delta_q_v_dc, quantization.delta_q_v_ac):
        within('a quantiser delta', delta, -64, 63)
    for level in header.loop_filter.level:
        within('loop_filter_level', level, 0, 63)
    for delta in (*header.loop_filter.ref_deltas, *header.loop_filter.mode_deltas):
        within('a loop filter delta', delta, -64, 63)
    within('cdef strengths', len(header.cdef.y_pri_strength), 1, 8)
    for parameters in header.gm_params[1:]:
        for value in parameters:
            within('a global motion parameter', value, -(1 << 31), (1 << 31) - 1)
    grain = header.film_grain
    if grain.apply_grain:
        within('num_y_points', grain.num_y_points, 0, 14)
        within('num_cb_points', grain.num_cb_points, 0, 10)
        within('num_cr_points', grain.num_cr_points, 0, 10)
        for values in (grain.point_y_value, grain.point_cb_value, grain.point_cr_value):
            if any(later <= earlier for earlier, later in zip(values, values[1:], strict=False)):
                report('film grain points that do not increase', values)
        color = sequence.color_config
        if color.subsampling_x and color.subsampling_y and (
                bool(grain.num_cb_points) != bool(grain.num_cr_points)):
            report('4:2:0 film grain in one chroma plane')
        if not grain.update_grain and (
                grain.film_grain_params_ref_idx not in header.ref_frame_idx):
            report('film grain taken from a slot the frame does not refer to')


# --------------------------------------------------- what the driver is handed


def check_picture(findings, session, surface, batches, parsed):
    """One submitted picture: its parameter buffer, then each tile's."""
    report, within = findings.report, findings.within
    kind, parameters = batches[0][0]
    if len(batches[0]) != 1 or kind != api.VAPictureParameterBufferType:
        report('a picture submitted without its picture parameters')
        return
    sequence, header = parsed(parameters)
    check_sequence(findings, sequence)
    check_header(findings, sequence, header)
    width, height = session.format.coded_size
    within('driver frame width', parameters.frame_width_minus1 + 1, 1, width)
    within('driver frame height', parameters.frame_height_minus1 + 1, 1, height)
    if (parameters.frame_width_minus1 + 1, parameters.frame_height_minus1 + 1) != header.size:
        report('a frame size other than the one parsed', header.size)
    tiles = header.tile_info
    if (parameters.tile_cols, parameters.tile_rows) != (tiles.tile_cols, tiles.tile_rows):
        report('a tile grid other than the one parsed',
               (parameters.tile_cols, parameters.tile_rows))
    within('driver tile_cols', parameters.tile_cols, 1, MAX_TILE_COLS)
    within('driver tile_rows', parameters.tile_rows, 1, MAX_TILE_ROWS)
    within('driver context_update_tile_id', parameters.context_update_tile_id, 0,
           parameters.tile_cols * parameters.tile_rows - 1)
    grain = parameters.film_grain_info
    within('driver num_y_points', grain.num_y_points, 0, 14)
    within('driver num_cb_points', grain.num_cb_points, 0, 10)
    within('driver num_cr_points', grain.num_cr_points, 0, 10)
    if parameters.current_frame != surface:
        report('a picture decoded into another surface than its target')
    if not header.frame_is_intra:
        references = [parameters.ref_frame_map[parameters.ref_frame_idx[index]]
                      for index in range(7)]
        ivf_fuzz.check_references(findings, session, references)
        if surface in references:
            report('a picture decoded into a surface it predicts from', surface)
    for held in parameters.ref_frame_map:
        if held != api.VA_INVALID_SURFACE and held not in session.decoded:
            report('a reference slot naming a surface never decoded', held)
    seen = 0
    for batch in batches[1:]:
        *slices, (data_kind, data) = batch
        if data_kind != api.VASliceDataBufferType:
            report('a tile group with no data buffer')
            continue
        for _, tile in slices:
            if tile.slice_data_size < 1 or (
                    tile.slice_data_offset + tile.slice_data_size > len(data)):
                report('a tile outside the data submitted',
                       (tile.slice_data_offset, tile.slice_data_size, len(data)))
            if (tile.tile_row, tile.tile_column) != divmod(seen, parameters.tile_cols):
                report('a tile out of its place in the grid', (tile.tile_row, tile.tile_column))
            seen += 1
    if seen != parameters.tile_cols * parameters.tile_rows:
        report('a picture with another count of tiles than its grid', seen)
    # A frame with film grain is written a second time, into the surface shown.
    if parameters.current_display_picture != api.VA_INVALID_SURFACE:
        if not header.film_grain.apply_grain:
            report('a display surface for a frame with no film grain')
        session.decoded.add(parameters.current_display_picture)


@pytest.fixture
def run(monkeypatch):
    """A fuzz run of the AV1 layer, every store into its buffers read back."""
    findings = ivf_fuzz.Findings()
    built = []
    build = av1_decode.picture_parameters

    def recording(sequence, header, *arguments):
        parameters = build(sequence, header, *arguments)
        built[:] = [(parameters, sequence, header)]
        return parameters

    def parsed(parameters):
        (made, sequence, header), = built
        assert made is parameters
        return sequence, header

    def check(findings, session, surface, batches):
        check_picture(findings, session, surface, batches, parsed)

    monkeypatch.setattr(av1_decode, 'av1_api', ivf_fuzz.recorded(av1_api, findings.report))
    monkeypatch.setattr(av1_decode, 'picture_parameters', recording)

    def fuzz(count=CASES):
        return ivf_fuzz.run(
            av1_decode.AV1Decoding, ivf_fuzz.streams('av1_*.ivf'), count, SEED,
            header_regions, check, findings,
            others=[('describe', lambda data: description.describe('av1', data))])
    return fuzz


def test_mutated_streams_are_decoded_or_refused_and_stay_in_range(run):
    findings = run()
    assert not findings.found, f'\n{findings}'
    # Enough of the cases reach the driver for the checks on its buffers to mean something.
    assert findings.pictures > CASES


def test_the_fixtures_themselves_are_decoded_with_nothing_found(monkeypatch, run):
    """With no mutation, every sample of every fixture reaches the driver."""
    monkeypatch.setattr(ivf_fuzz, 'mutate', lambda _chooser, samples, *_rest: samples)
    findings = run(60)
    assert not findings.found, f'\n{findings}'
    assert findings.pictures >= 60 * ivf_fuzz.SAMPLES
