"""The AV1 decode buffers and the reference slots, with no driver.

A field the driver is told wrongly is not reported: the frame decodes into the
wrong thing. Each buffer is built here from headers whose values are known and
read back field by field. The control layer is then run over hand-built
streams and the fixtures through a stand-in for the decoder's session, which
records what it was asked to decode and to show.
"""
import ctypes
import json
from pathlib import Path

import pytest

from pyopengl_video import obu as obus
from pyopengl_video.decoder import DecoderError, Sample
from pyopengl_video.ivf import IVFReader
from pyopengl_video.vaapi import api, av1_api
from pyopengl_video.vaapi.av1_decode import (
    AV1Decoding,
    colour_of,
    format_for,
    picture_parameters,
    tile_parameters,
)
from pyopengl_video.vaapi.av1_parse import (
    INTER_FRAME,
    INTRA_ONLY_FRAME,
    KEY_FRAME,
    FrameState,
    StreamError,
    Tile,
    parse_frame_header,
)
from tests.av1_syntax import (
    Sequence,
    frame,
    frame_header,
    headers_of,
    obu,
    show_existing,
    unit,
)

DATA = Path(__file__).parent / 'data' / 'decode'
FIXTURES = json.loads((DATA / 'av1.json').read_text())['streams']


def stream_headers(name):
    with IVFReader(DATA / f'{name}.ivf') as stream:
        return headers_of([stream.sample(index).data for index in range(len(stream))])


def samples_of(name):
    with IVFReader(DATA / f'{name}.ivf') as stream:
        return [stream.sample(index) for index in range(len(stream))]


def parameters_of(name, index, references=(None,) * 8, surface=5, display=None):
    header, sequence = stream_headers(name)[index]
    return header, picture_parameters(sequence, header, references, surface, display)


class TestBitFields:
    """The unions of flags, each field set alone and read back as its bit."""

    def test_sequence_flags(self):
        fields = av1_api._AV1SeqInfoFields()  # noqa: SLF001 - an anonymous union of the VA header, measured here
        fields.fields.use_128x128_superblock = 1
        fields.fields.enable_order_hint = 1
        fields.fields.mono_chrome = 1
        fields.fields.film_grain_params_present = 1
        assert fields.value == (1 << 1 | 1 << 7 | 1 << 10 | 1 << 15)

    def test_picture_flags(self):
        fields = av1_api._AV1PicInfoFields()  # noqa: SLF001 - an anonymous union of the VA header, measured here
        fields.bits.frame_type = 3
        fields.bits.showable_frame = 1
        fields.bits.use_superres = 1
        fields.bits.disable_frame_end_update_cdf = 1
        fields.bits.large_scale_tile = 1
        assert fields.value == (3 | 1 << 3 | 1 << 9 | 1 << 13 | 1 << 16)

    def test_mode_control(self):
        fields = av1_api._AV1ModeControlFields()  # noqa: SLF001 - an anonymous union of the VA header, measured here
        fields.bits.log2_delta_q_res = 3
        fields.bits.log2_delta_lf_res = 3
        fields.bits.tx_mode = 2
        fields.bits.skip_mode_present = 1
        assert fields.value == (3 << 1 | 3 << 4 | 2 << 7 | 1 << 11)

    def test_the_narrow_unions(self):
        loop = av1_api._AV1LoopFilterInfoFields()  # noqa: SLF001 - an anonymous union of the VA header, measured here
        loop.bits.sharpness_level = 7
        loop.bits.mode_ref_delta_update = 1
        assert loop.value == (7 | 1 << 4)
        assert ctypes.sizeof(loop) == 1
        matrix = av1_api._AV1QMatrixFields()  # noqa: SLF001 - an anonymous union of the VA header, measured here
        matrix.bits.using_qmatrix = 1
        matrix.bits.qm_u = 15
        matrix.bits.qm_v = 1
        assert matrix.value == (1 | 15 << 5 | 1 << 9)
        assert ctypes.sizeof(matrix) == 2
        restore = av1_api._AV1LoopRestorationFields()  # noqa: SLF001 - an anonymous union of the VA header, measured here
        restore.bits.cbframe_restoration_type = 3
        restore.bits.lr_unit_shift = 2
        restore.bits.lr_uv_shift = 1
        assert restore.value == (3 << 2 | 2 << 6 | 1 << 8)
        assert ctypes.sizeof(restore) == 2

    def test_segment_and_grain_flags(self):
        segment = av1_api._SegmentInfoFields()  # noqa: SLF001 - an anonymous union of the VA header, measured here
        segment.bits.temporal_update = 1
        segment.bits.update_data = 1
        assert segment.value == (1 << 2 | 1 << 3)
        grain = av1_api._FilmGrainInfoFields()  # noqa: SLF001 - an anonymous union of the VA header, measured here
        grain.bits.grain_scaling_minus_8 = 3
        grain.bits.ar_coeff_shift_minus_6 = 3
        grain.bits.clip_to_restricted_range = 1
        assert grain.value == (3 << 2 | 3 << 6 | 1 << 11)


class TestPictureParameters:
    def test_the_sequence_fields(self):
        _, parameters = parameters_of('av1_altref', 1)
        assert parameters.profile == 0
        assert parameters.order_hint_bits_minus_1 == 6
        assert parameters.bit_depth_idx == 0
        assert parameters.matrix_coefficients == 2
        fields = parameters.seq_info_fields.fields
        assert fields.still_picture == 0
        assert fields.use_128x128_superblock == 0
        assert (fields.enable_filter_intra, fields.enable_intra_edge_filter) == (1, 1)
        assert (fields.enable_interintra_compound, fields.enable_masked_compound) == (0, 1)
        assert (fields.enable_dual_filter, fields.enable_order_hint) == (0, 1)
        assert (fields.enable_jnt_comp, fields.enable_cdef) == (0, 1)
        assert (fields.mono_chrome, fields.color_range) == (0, 0)
        assert (fields.subsampling_x, fields.subsampling_y) == (1, 1)
        assert fields.film_grain_params_present == 0

    def test_ten_bits_and_monochrome(self):
        _, parameters = parameters_of('av1_grain10', 0)
        assert parameters.bit_depth_idx == 1
        _, parameters = parameters_of('av1_monochrome', 0)
        assert parameters.seq_info_fields.fields.mono_chrome == 1

    def test_the_frame_and_its_references(self):
        """av1_global_motion, the third header: slots 0 and 1 hold frames."""
        header, parameters = parameters_of(
            'av1_global_motion', 2, references=(30, 31, None, None, None, None, None, None),
            surface=9)
        assert parameters.current_frame == 9
        assert parameters.current_display_picture == api.VA_INVALID_SURFACE
        assert list(parameters.ref_frame_map) == [30, 31] + [api.VA_INVALID_SURFACE] * 6
        assert list(parameters.ref_frame_idx) == [0, 0, 0, 0, 0, 0, 1]
        assert parameters.primary_ref_frame == 7
        assert parameters.order_hint == header.order_hint == 4
        assert (parameters.frame_width_minus1, parameters.frame_height_minus1) == (159, 89)
        assert parameters.anchor_frames_num == 0

    def test_the_picture_flags(self):
        _, parameters = parameters_of('av1_global_motion', 2)
        bits = parameters.pic_info_fields.bits
        assert (bits.frame_type, bits.show_frame, bits.showable_frame) == (INTER_FRAME, 0, 1)
        assert (bits.error_resilient_mode, bits.disable_cdf_update) == (0, 0)
        assert (bits.allow_screen_content_tools, bits.force_integer_mv) == (1, 0)
        assert (bits.allow_intrabc, bits.use_superres) == (0, 0)
        assert (bits.allow_high_precision_mv, bits.is_motion_mode_switchable) == (0, 1)
        assert (bits.use_ref_frame_mvs, bits.disable_frame_end_update_cdf) == (1, 0)
        assert (bits.uniform_tile_spacing_flag, bits.allow_warped_motion) == (1, 1)
        assert bits.large_scale_tile == 0
        assert parameters.superres_scale_denominator == 8
        assert parameters.interp_filter == 0
        mode = parameters.mode_control_fields.bits
        assert (mode.delta_q_present_flag, mode.log2_delta_q_res) == (1, 2)
        assert (mode.tx_mode, mode.reference_select) == (2, 1)
        assert (mode.reduced_tx_set_used, mode.skip_mode_present) == (0, 1)

    def test_an_intra_only_frame(self):
        _, parameters = parameters_of('av1_intra_only', 13)
        bits = parameters.pic_info_fields.bits
        assert (bits.frame_type, bits.allow_intrabc, bits.force_integer_mv) == (
            INTRA_ONLY_FRAME, 1, 1)

    def test_global_motion(self):
        """ALTREF_FRAME is the last of the seven models; the rest are identity."""
        _, parameters = parameters_of('av1_global_motion', 2)
        last = parameters.wm[6]
        assert last.wmtype == av1_api.VAAV1TransformationRotzoom
        assert list(last.wmmat) == [433152, -628736, 65654, -8192, 8192, 65654, 0, 0]
        assert last.invalid == 0
        for index in range(6):
            assert parameters.wm[index].wmtype == av1_api.VAAV1TransformationIdentity
            assert list(parameters.wm[index].wmmat)[:6] == [0, 0, 1 << 16, 0, 0, 1 << 16]
            assert parameters.wm[index].invalid == 0

    def test_cdef_strengths_pack_primary_over_secondary(self):
        """A secondary strength of 4 goes back to its coded 3."""
        _, parameters = parameters_of('av1_global_motion', 2)
        assert (parameters.cdef_damping_minus_3, parameters.cdef_bits) == (1, 1)
        assert list(parameters.cdef_y_strengths)[:2] == [0 << 2 | 2, 1 << 2 | 2]
        assert list(parameters.cdef_uv_strengths)[:2] == [0 << 2 | 3, 1 << 2 | 0]
        assert list(parameters.cdef_y_strengths)[2:] == [0] * 6

    def test_the_loop_filter(self):
        _, parameters = parameters_of('av1_global_motion', 2)
        assert list(parameters.filter_level) == [7, 10]
        assert (parameters.filter_level_u, parameters.filter_level_v) == (7, 10)
        info = parameters.loop_filter_info_fields.bits
        assert (info.sharpness_level, info.mode_ref_delta_enabled,
                info.mode_ref_delta_update) == (0, 1, 0)
        assert list(parameters.ref_deltas) == [1, 0, 0, 0, -1, 0, -1, -1]
        assert list(parameters.mode_deltas) == [0, 0]

    def test_loop_filter_deltas_as_updated(self):
        sequence = Sequence()
        data, _ = frame_header(sequence, loop_filter=(9, 0), ref_deltas={0: -5, 6: 7},
                               mode_deltas={0: -3})
        parsed = sequence.parsed()
        header = parse_frame_header(data, parsed, FrameState())
        parameters = picture_parameters(parsed, header, (None,) * 8, 1)
        assert list(parameters.ref_deltas) == [-5, 0, 0, 0, -1, 0, 7, -1]
        assert list(parameters.mode_deltas) == [-3, 0]
        info = parameters.loop_filter_info_fields.bits
        assert (info.sharpness_level, info.mode_ref_delta_update) == (3, 1)
        assert list(parameters.filter_level) == [9, 0]

    def test_quantisation(self):
        """av1_switch_frames: every delta coded, U and V apart."""
        _, parameters = parameters_of('av1_switch_frames', 8)
        assert parameters.base_qindex == 106
        assert parameters.y_dc_delta_q == 14
        assert (parameters.u_dc_delta_q, parameters.u_ac_delta_q) == (1, -10)
        assert (parameters.v_dc_delta_q, parameters.v_ac_delta_q) == (-26, -39)
        assert parameters.qmatrix_fields.bits.using_qmatrix == 0
        _, parameters = parameters_of('av1_tiles', 1)
        matrix = parameters.qmatrix_fields.bits
        assert (matrix.using_qmatrix, matrix.qm_y, matrix.qm_u, matrix.qm_v) == (1, 6, 6, 6)

    def test_segmentation(self):
        _, parameters = parameters_of('av1_tiles', 1)
        bits = parameters.seg_info.segment_info_fields.bits
        assert (bits.enabled, bits.update_map, bits.temporal_update, bits.update_data) == (
            1, 1, 0, 1)
        assert [row[0] for row in parameters.seg_info.feature_data] == [
            -44, -35, -23, -7, 0, 10, 22, 32]
        assert list(parameters.seg_info.feature_mask) == [1] * 8
        mode = parameters.mode_control_fields.bits
        assert (mode.delta_lf_present_flag, mode.log2_delta_lf_res, mode.delta_lf_multi) == (
            1, 1, 0)

    def test_the_feature_mask_has_a_bit_for_each_feature(self):
        sequence = Sequence()
        data, _ = frame_header(sequence, segments={(3, 1): -9, (3, 5): 4, (3, 7): 0})
        parsed = sequence.parsed()
        header = parse_frame_header(data, parsed, FrameState())
        parameters = picture_parameters(parsed, header, (None,) * 8, 1)
        assert list(parameters.seg_info.feature_mask) == [0, 0, 0, 1 << 1 | 1 << 5 | 1 << 7,
                                                          0, 0, 0, 0]
        assert list(parameters.seg_info.feature_data[3]) == [0, -9, 0, 0, 0, 4, 0, 0]

    def test_tiles(self):
        _, parameters = parameters_of('av1_tile_sizes', 0)
        assert (parameters.tile_cols, parameters.tile_rows) == (2, 2)
        assert list(parameters.width_in_sbs_minus_1)[:3] == [0, 1, 0]
        assert list(parameters.height_in_sbs_minus_1)[:3] == [0, 0, 0]
        assert parameters.context_update_tile_id == 1
        assert parameters.pic_info_fields.bits.uniform_tile_spacing_flag == 0
        _, parameters = parameters_of('av1_tiles', 0)
        assert list(parameters.width_in_sbs_minus_1)[:2] == [1, 0]

    def test_loop_restoration(self):
        """libva takes FrameRestorationType: a coded lr_type of 1 is 3."""
        _, parameters = parameters_of('av1_switch_frames', 8)
        bits = parameters.loop_restoration_fields.bits
        assert (bits.yframe_restoration_type, bits.cbframe_restoration_type,
                bits.crframe_restoration_type) == (3, 3, 3)
        assert (bits.lr_unit_shift, bits.lr_uv_shift) == (0, 1)
        _, parameters = parameters_of('av1_superres', 1)
        bits = parameters.loop_restoration_fields.bits
        assert (bits.yframe_restoration_type, bits.cbframe_restoration_type) == (1, 0)
        assert bits.lr_unit_shift == 1

    def test_super_resolution(self):
        """The size given is the one after scaling up."""
        _, parameters = parameters_of('av1_superres', 1)
        assert parameters.pic_info_fields.bits.use_superres == 1
        assert parameters.superres_scale_denominator == 12
        assert (parameters.frame_width_minus1, parameters.frame_height_minus1) == (119, 71)
        assert parameters.interp_filter == 4

    def test_film_grain(self):
        header, parameters = parameters_of('av1_grain', 0, surface=3, display=8)
        assert parameters.current_frame == 3
        assert parameters.current_display_picture == 8
        grain = parameters.film_grain_info
        bits = grain.film_grain_info_fields.bits
        assert (bits.apply_grain, bits.chroma_scaling_from_luma) == (1, 0)
        assert (bits.grain_scaling_minus_8, bits.ar_coeff_lag) == (3, 3)
        assert (bits.ar_coeff_shift_minus_6, bits.grain_scale_shift) == (1, 1)
        assert (bits.overlap_flag, bits.clip_to_restricted_range) == (1, 1)
        assert grain.grain_seed == 48612
        assert (grain.num_y_points, grain.num_cb_points, grain.num_cr_points) == (2, 2, 2)
        assert list(grain.point_y_value)[:3] == [0, 255, 0]
        assert list(grain.point_y_scaling)[:3] == [192, 192, 0]
        assert list(grain.point_cb_scaling)[:2] == [128, 128]
        assert list(grain.point_cr_value)[:2] == [0, 255]
        # The stream codes each coefficient plus 128.
        assert list(grain.ar_coeffs_y)[:6] == [4, 1, 3, 0, 1, -3]
        assert grain.ar_coeffs_y[23] == 194 - 128
        assert list(grain.ar_coeffs_cb)[22:] == [92 - 128, 195 - 128, 0]
        assert list(grain.ar_coeffs_cr) == list(grain.ar_coeffs_cb)
        assert (grain.cb_mult, grain.cb_luma_mult, grain.cb_offset) == (128, 192, 256)
        assert (grain.cr_mult, grain.cr_luma_mult, grain.cr_offset) == (128, 192, 256)
        assert header.film_grain.apply_grain == 1

    def test_a_frame_with_no_grain_leaves_the_grain_fields_zero(self):
        _, parameters = parameters_of('av1_grain', 1, surface=3, display=8)
        assert parameters.current_display_picture == api.VA_INVALID_SURFACE
        assert bytes(parameters.film_grain_info) == bytes(
            ctypes.sizeof(av1_api.VAFilmGrainStructAV1))

    def test_a_shear_that_is_no_valid_warp_is_marked(self):
        sequence = Sequence()
        data, _ = frame_header(sequence, frame_type=INTER_FRAME, order_hint=1)
        parsed = sequence.parsed()
        header = parse_frame_header(data, parsed, FrameState())
        header.gm_params = (*header.gm_params[:7], (0, 0, 1 << 16, 1 << 14, 0, 1 << 16))
        parameters = picture_parameters(parsed, header, (None,) * 8, 1)
        assert [warp.invalid for warp in parameters.wm] == [0, 0, 0, 0, 0, 0, 1]


class TestTileParameters:
    def test_a_tile_names_its_bytes_and_its_place(self):
        parameters = tile_parameters(Tile(row=2, column=3, offset=40, size=77))
        assert (parameters.slice_data_size, parameters.slice_data_offset) == (77, 40)
        assert parameters.slice_data_flag == api.VA_SLICE_DATA_FLAG_ALL
        assert (parameters.tile_row, parameters.tile_column) == (2, 3)
        assert parameters.anchor_frame_idx == 0


class TestFormat:
    def test_the_coded_size_is_whole_blocks(self):
        _, sequence = stream_headers('av1_altref')[0]
        found = format_for(sequence)
        assert found.profile == api.VAProfileAV1Profile0
        assert found.coded_size == (120, 72)
        assert (found.depth, found.surfaces) == (8, 9)
        assert found.name == 'AV1 Main'

    def test_film_grain_doubles_the_surfaces(self):
        _, sequence = stream_headers('av1_grain10')[0]
        found = format_for(sequence)
        assert (found.depth, found.surfaces) == (10, 18)
        assert found.coded_size == (96, 56)

    def test_other_profiles_are_refused_by_name(self):
        with pytest.raises(DecoderError, match='High profile'):
            format_for(Sequence(profile=1).parsed())
        with pytest.raises(DecoderError, match='Professional profile'):
            format_for(Sequence(profile=2).parsed())

    def test_the_colour_description(self):
        colour = colour_of(Sequence(colour=(9, 16, 9), color_range=1).parsed())
        assert (colour.primaries, colour.transfer, colour.matrix) == (9, 16, 9)
        assert colour.full_range

    def test_an_unstated_matrix_is_chosen_by_height(self):
        assert colour_of(Sequence().parsed()).matrix == 6
        assert colour_of(Sequence(size=(200, 200), size_bits=10).parsed()).matrix == 6


# ------------------------------------------------------------ the layer


class Session:
    """A stand-in for the decoder's session: surfaces by number, and a record."""

    def __init__(self):
        self.control = None
        self.formats = []
        self.count = 0
        self.submitted = []
        self.shown = []
        self.most_held = 0

    def activate(self, format):
        self.formats.append(format)
        self.count = format.surfaces

    def acquire(self):
        held = self.control.held()
        self.most_held = max(self.most_held, len(held))
        for surface in range(100 * len(self.formats), 100 * len(self.formats) + self.count):
            if surface not in held:
                return surface
        raise AssertionError(f'all {self.count} surfaces are held: {sorted(held)}')

    def submit(self, surface, batches, what):
        self.submitted.append((surface, batches, what))

    def output(self, surface, *, timestamp, duration, visible, colour):  # noqa: ARG002 - the session's interface
        self.shown.append((surface, timestamp, visible))


def layer():
    session = Session()
    control = AV1Decoding(session, 4)
    session.control = control
    return control, session


def sample(data, timestamp=0):
    return Sample(data, timestamp, 1, False, timestamp)


def feed(control, *units, start=0):
    for index, data in enumerate(units, start):
        control.decode(sample(data, index))


def picture_of(submission):
    """The picture parameter buffer of one recorded submission."""
    kind, parameters = submission[1][0][0]
    assert kind == api.VAPictureParameterBufferType
    return parameters


SEQUENCE = Sequence()


class TestSlots:
    def test_a_key_frame_is_decoded_and_shown(self):
        control, session = layer()
        feed(control, unit(SEQUENCE.obu(), frame(tile=b'tile-bytes')))
        assert [found.coded_size for found in session.formats] == [(64, 48)]
        [(surface, batches, what)] = session.submitted
        assert what == 'key frame at order hint 0'
        assert [[kind for kind, _ in batch] for batch in batches] == [
            [api.VAPictureParameterBufferType],
            [api.VASliceParameterBufferType, api.VASliceDataBufferType]]
        assert session.shown == [(surface, 0, (0, 0, 64, 48))]
        assert control.size == (64, 48)
        assert control.held() == {surface}

    def test_the_tile_is_named_within_the_frame_obu(self):
        control, session = layer()
        feed(control, unit(SEQUENCE.obu(), frame(tile=b'tile-bytes')))
        (_, tile), (_, data) = session.submitted[0][1][1]
        assert data[tile.slice_data_offset:tile.slice_data_offset + tile.slice_data_size] == (
            b'tile-bytes')
        assert tile.slice_data_offset + tile.slice_data_size == len(data)
        assert (tile.tile_row, tile.tile_column) == (0, 0)

    def test_an_inter_frame_is_told_the_surface_in_every_slot(self):
        control, session = layer()
        feed(control,
             unit(SEQUENCE.obu(), frame()),
             unit(frame(frame_type=INTER_FRAME, order_hint=1, refresh=0b0110)),
             unit(frame(frame_type=INTER_FRAME, order_hint=2, refresh=0b1000,
                        refs=(1, 2, 0, 0, 0, 0, 0))))
        key, second, third = (entry[0] for entry in session.submitted)
        assert len({key, second, third}) == 3
        assert list(picture_of(session.submitted[1]).ref_frame_map) == [key] * 8
        assert list(picture_of(session.submitted[2]).ref_frame_map) == [
            key, second, second, key, key, key, key, key]
        assert list(picture_of(session.submitted[2]).ref_frame_idx) == [1, 2, 0, 0, 0, 0, 0]
        assert picture_of(session.submitted[2]).current_frame == third
        assert control.held() == {key, second, third}

    def test_a_surface_no_slot_holds_is_decoded_into_again(self):
        control, session = layer()
        units = [unit(SEQUENCE.obu(), frame())]
        units += [unit(frame(frame_type=INTER_FRAME, order_hint=index, refresh=0b10))
                  for index in range(1, 30)]
        feed(control, *units)
        assert len(session.submitted) == len(session.shown) == 30
        assert len({entry[0] for entry in session.submitted}) == 3
        assert len(control.held()) == 2

    def test_a_hidden_frame_is_shown_when_a_header_shows_its_slot(self):
        control, session = layer()
        feed(control,
             unit(SEQUENCE.obu(), frame()),
             unit(frame(frame_type=INTER_FRAME, order_hint=4, refresh=0b100, show_frame=0)),
             unit(frame(frame_type=INTER_FRAME, order_hint=1, refresh=0b10)),
             unit(obu(obus.FRAME_HEADER, show_existing(2))))
        hidden = session.submitted[1][0]
        assert [timestamp for _, timestamp, _ in session.shown] == [0, 2, 3]
        assert session.shown[2][0] == hidden
        assert len(session.submitted) == 3

    def test_a_frame_may_be_shown_twice(self):
        control, session = layer()
        feed(control,
             unit(SEQUENCE.obu(), frame()),
             unit(frame(frame_type=INTER_FRAME, order_hint=1, refresh=0b10)),
             unit(obu(obus.FRAME_HEADER, show_existing(1))))
        assert session.shown[1][0] == session.shown[2][0]
        assert [timestamp for _, timestamp, _ in session.shown] == [0, 1, 2]

    def test_a_key_frame_shown_from_its_slot_fills_every_slot(self):
        """7.21: frames after it refer to it through any slot."""
        control, session = layer()
        feed(control,
             unit(SEQUENCE.obu(), frame()),
             unit(frame(show_frame=0, refresh=0b1, order_hint=8)),
             unit(obu(obus.FRAME_HEADER, show_existing(0))),
             unit(frame(frame_type=INTER_FRAME, order_hint=9, refresh=0, refs=(5,) * 7)))
        hidden_key = session.submitted[1][0]
        assert session.shown[1][0] == hidden_key
        assert list(picture_of(session.submitted[2]).ref_frame_map) == [hidden_key] * 8
        assert hidden_key in control.held()
        assert session.submitted[0][0] not in control.held()

    def test_a_frame_header_and_its_tile_group_may_be_apart(self):
        control, session = layer()
        header, _ = frame_header(SEQUENCE)
        feed(control, unit(SEQUENCE.obu(), obu(obus.FRAME_HEADER, header + b'\x80'),
                           obu(obus.REDUNDANT_FRAME_HEADER, header + b'\x80'),
                           obu(obus.TILE_GROUP, b'tile')))
        [(_, batches, _)] = session.submitted
        (_, tile), (_, data) = batches[1]
        assert (tile.slice_data_offset, tile.slice_data_size, data) == (0, 4, b'tile')
        assert len(session.shown) == 1

    def test_metadata_and_padding_are_passed_over(self):
        control, session = layer()
        feed(control, unit(SEQUENCE.obu(), obu(obus.METADATA, b'\x01\x02\x80'),
                           obu(obus.PADDING, bytes(4)), frame()))
        assert len(session.shown) == 1

    def test_film_grain_is_shown_from_a_second_surface(self):
        sequence = Sequence(film_grain=1)
        control, session = layer()
        feed(control,
             unit(sequence.obu(), frame(sequence, grain='new')),
             unit(frame(sequence, frame_type=INTER_FRAME, order_hint=1, refresh=0b10,
                        grain=0)),
             unit(frame(sequence, frame_type=INTER_FRAME, order_hint=2, refresh=0b100)),
             unit(obu(obus.FRAME_HEADER, show_existing(1))))
        assert session.formats[0].surfaces == 18
        first, second, third = (picture_of(entry) for entry in session.submitted)
        assert first.film_grain_info.film_grain_info_fields.bits.apply_grain == 1
        assert first.current_display_picture not in (first.current_frame,
                                                     api.VA_INVALID_SURFACE)
        assert session.submitted[0][0] == first.current_frame
        assert session.shown[0][0] == first.current_display_picture
        # The frames after predict from the picture without its grain.
        assert list(second.ref_frame_map) == [first.current_frame] * 8
        assert second.film_grain_info.grain_seed == 0x1234
        assert second.film_grain_info.num_y_points == 1
        assert session.shown[1][0] == second.current_display_picture
        assert third.current_display_picture == api.VA_INVALID_SURFACE
        assert session.shown[2][0] == third.current_frame
        # Shown again, the second frame is the surface with its grain.
        assert session.shown[3][0] == second.current_display_picture
        assert control.held() == {
            first.current_frame, first.current_display_picture, second.current_frame,
            second.current_display_picture, third.current_frame}

    def test_a_new_sequence_at_a_key_frame_takes_new_surfaces(self):
        larger = Sequence(size=(48, 64))
        control, session = layer()
        feed(control, unit(SEQUENCE.obu(), frame()),
             unit(larger.obu(), frame(larger)),
             unit(frame(larger, frame_type=INTER_FRAME, order_hint=1, refresh=0)))
        assert [found.coded_size for found in session.formats] == [(64, 48), (48, 64)]
        assert session.shown[1][2] == (0, 0, 48, 64)
        assert list(picture_of(session.submitted[2]).ref_frame_map) == [
            session.submitted[1][0]] * 8
        assert control.size == (48, 64)

    def test_a_sequence_of_another_size_without_a_key_frame_is_an_error(self):
        larger = Sequence(size=(48, 64))
        control, _ = layer()
        feed(control, unit(SEQUENCE.obu(), frame()))
        with pytest.raises(StreamError, match='begins at a key frame'):
            feed(control, unit(larger.obu(), frame(
                larger, frame_type=INTRA_ONLY_FRAME, refresh=0b1)))

    def test_the_same_sequence_header_again_keeps_the_surfaces(self):
        control, session = layer()
        feed(control, unit(SEQUENCE.obu(), frame()), unit(SEQUENCE.obu(), frame()))
        assert len(session.formats) == 1


class TestRecovery:
    def test_frames_before_the_first_key_frame_are_passed_over(self):
        control, session = layer()
        feed(control,
             unit(SEQUENCE.obu(), frame(frame_type=INTER_FRAME, order_hint=1, refresh=0b1)),
             unit(obu(obus.FRAME_HEADER, show_existing(3))),
             unit(frame()))
        assert len(session.submitted) == 1
        assert [timestamp for _, timestamp, _ in session.shown] == [2]

    def test_a_reset_drops_every_reference(self):
        control, session = layer()
        feed(control, unit(SEQUENCE.obu(), frame()))
        control.reset()
        assert control.held() == set()
        feed(control,
             unit(frame(frame_type=INTER_FRAME, order_hint=1, refresh=0b1)),
             unit(SEQUENCE.obu(), frame()),
             unit(frame(frame_type=INTER_FRAME, order_hint=1, refresh=0b1)), start=1)
        assert [timestamp for _, timestamp, _ in session.shown] == [0, 2, 3]

    def test_an_intra_only_frame_decodes_before_any_key_frame(self):
        control, session = layer()
        feed(control,
             unit(SEQUENCE.obu(), frame(frame_type=INTRA_ONLY_FRAME, refresh=0b1)),
             unit(frame(frame_type=INTER_FRAME, order_hint=1, refresh=0, refs=(0,) * 7)),
             unit(frame(frame_type=INTER_FRAME, order_hint=2, refresh=0,
                        refs=(0, 0, 0, 1, 0, 0, 0))))
        assert [timestamp for _, timestamp, _ in session.shown] == [0, 1]

    def test_a_missing_reference_after_a_key_frame_is_a_stream_error(self):
        control, _ = layer()
        feed(control, unit(SEQUENCE.obu(), frame()))
        resilient = frame(frame_type=INTER_FRAME, order_hint=2, error_resilient=1,
                          refresh=0, ref_order_hints=(0, 0, 0, 7, 0, 0, 0, 0),
                          refs=(0, 0, 3, 0, 0, 0, 0))
        with pytest.raises(StreamError, match='reference slot 3'):
            feed(control, unit(resilient))

    def test_showing_an_empty_slot_after_a_key_frame_is_a_stream_error(self):
        control, _ = layer()
        feed(control, unit(SEQUENCE.obu(), frame()))
        control._slots[4] = None  # noqa: SLF001 - a slot emptied as a lost frame leaves it
        with pytest.raises(StreamError, match='shows reference slot 4'):
            feed(control, unit(obu(obus.FRAME_HEADER, show_existing(4))))

    def test_a_frame_left_unfinished_is_dropped_at_the_next_temporal_unit(self):
        control, session = layer()
        header, _ = frame_header(SEQUENCE)
        feed(control, unit(SEQUENCE.obu(), obu(obus.FRAME_HEADER, header + b'\x80')),
             unit(frame()))
        assert len(session.submitted) == 1
        assert [timestamp for _, timestamp, _ in session.shown] == [1]

    def test_flush_shows_nothing_more(self):
        control, session = layer()
        feed(control, unit(SEQUENCE.obu(), frame()))
        control.flush()
        assert len(session.shown) == 1


class TestRefusals:
    def test_a_frame_before_any_sequence_header(self):
        control, _ = layer()
        with pytest.raises(StreamError, match='before any sequence header'):
            feed(control, unit(frame()))

    def test_a_tile_group_with_no_frame_header(self):
        control, _ = layer()
        with pytest.raises(StreamError, match='no frame header'):
            feed(control, unit(SEQUENCE.obu(), obu(obus.TILE_GROUP, b'tile')))

    def test_another_profile(self):
        high = Sequence(profile=1)
        control, _ = layer()
        with pytest.raises(DecoderError, match='High profile'):
            feed(control, unit(high.obu(), frame(high)))

    def test_reference_scaling(self):
        control, _ = layer()
        feed(control, unit(SEQUENCE.obu(), frame()))
        with pytest.raises(DecoderError, match='reference scaling') as refused:
            feed(control, unit(frame(frame_type=INTER_FRAME, order_hint=1, refresh=0,
                                     size=(32, 24))))
        assert '32x24' in str(refused.value)
        assert '64x48' in str(refused.value)

    def test_a_key_frame_of_another_size_is_decoded(self):
        """Only an inter frame's size is held to its references'."""
        control, session = layer()
        feed(control, unit(SEQUENCE.obu(), frame()),
             unit(frame(show_frame=0, refresh=0xFF, size=(32, 24))),
             unit(frame(frame_type=INTER_FRAME, order_hint=1, refresh=0, found_ref=0)))
        assert session.shown[1][2] == (0, 0, 32, 24)
        assert picture_of(session.submitted[2]).frame_width_minus1 == 31
        assert len(session.formats) == 1

    def test_a_scalable_stream(self):
        control, _ = layer()
        layered = Sequence(operating_points=(0x103,))
        header, _ = frame_header(layered)
        with pytest.raises(DecoderError, match='scalable') as refused:
            feed(control, unit(layered.obu(),
                               obu(obus.FRAME, header + bytes(4), temporal_id=1)))
        assert 'temporal layer 1' in str(refused.value)

    def test_a_layer_outside_the_operating_point_is_dropped(self):
        """Operating point 0 here holds temporal layer 0 alone (5.3.1)."""
        control, session = layer()
        base_only = Sequence(operating_points=(0x101, 0x103))
        header, _ = frame_header(base_only)
        feed(control, unit(base_only.obu(),
                           obu(obus.FRAME, header + bytes(4), temporal_id=0),
                           obu(obus.FRAME, header + bytes(4), temporal_id=1)))
        assert len(session.submitted) == 1

    def test_large_scale_tile(self):
        control, _ = layer()
        with pytest.raises(DecoderError, match='large scale tile'):
            feed(control, unit(SEQUENCE.obu(), obu(obus.TILE_LIST, bytes(8))))

    def test_tile_groups_out_of_order(self):
        control, _ = layer()
        header, sequence = stream_headers('av1_tiles')[0]
        with IVFReader(DATA / 'av1_tiles.ivf') as stream:
            first = stream.sample(0).data
        units = list(obus.split(first))
        whole = next(found for found in units if found.kind == obus.FRAME)
        start = (header.header_bits + 7) // 8
        # tile_start_and_end_present_flag, then tiles 2 to 3 where 0 was due.
        group = bytes([0b1_10_11_000]) + b'\x00\x00' + b'x' + b'yy'
        data = b''.join(found.raw for found in units if found.kind != obus.FRAME) + obu(
            obus.FRAME_HEADER, whole.payload[:start]) + obu(obus.TILE_GROUP, group)
        del sequence
        with pytest.raises(StreamError, match='tile 0 was due'):
            control.decode(sample(data))


class TestConfigure:
    def test_the_sequence_header_of_an_av1c_box_states_the_size_and_colour(self):
        control, session = layer()
        sequence = Sequence(size=(200, 120), colour=(1, 1, 1), color_range=1)
        control.configure(sequence.obu() + obu(obus.METADATA, b'\x02\x00\x80'))
        assert control.size == (200, 120)
        assert control.colour.matrix == 1
        assert control.colour.full_range
        assert session.formats == []

    def test_no_configuration_is_no_error(self):
        control, _ = layer()
        control.configure(b'')
        assert control.size == (0, 0)

    def test_a_stream_decodes_from_its_configuration_alone(self):
        control, session = layer()
        control.configure(SEQUENCE.obu())
        feed(control, unit(frame()))
        assert len(session.shown) == 1


class TestFixturesThroughTheLayer:
    """Every fixture, decoded with no driver: what is shown, and what is held."""

    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_one_frame_is_shown_for_each_temporal_unit(self, name):
        control, session = layer()
        samples = samples_of(name)
        for found in samples:
            control.decode(found)
        control.flush()
        assert [timestamp for _, timestamp, _ in session.shown] == [
            found.timestamp for found in samples]
        assert len(session.shown) == len(FIXTURES[name]['md5'])
        assert len(session.formats) == 1
        assert session.most_held < session.formats[0].surfaces

    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_every_reference_given_was_decoded_and_is_still_held(self, name):
        control, session = layer()
        decoded = set()
        submit = session.submit

        def checking(surface, batches, what):
            parameters = batches[0][0][1]
            if parameters.pic_info_fields.bits.frame_type in (KEY_FRAME, INTRA_ONLY_FRAME):
                wanted = set()
            else:
                wanted = {parameters.ref_frame_map[index]
                          for index in parameters.ref_frame_idx}
            assert wanted <= decoded, what
            assert wanted <= control.held(), what
            assert surface not in wanted, what
            decoded.add(surface)
            submit(surface, batches, what)

        session.submit = checking
        for found in samples_of(name):
            control.decode(found)
        assert session.submitted

    def test_hidden_frames_are_decoded_and_not_shown(self):
        """av1_altref has 56 frame headers: 40 frames decoded, 16 shown from a slot."""
        control, session = layer()
        for found in samples_of('av1_altref'):
            control.decode(found)
        assert len(session.submitted) == 40
        assert len(session.shown) == 40
        hidden = [picture_of(entry) for entry in session.submitted
                  if not picture_of(entry).pic_info_fields.bits.show_frame]
        assert len(hidden) == 16
        assert {entry.current_frame for entry in hidden} <= {
            surface for surface, _, _ in session.shown}

    def test_every_tile_of_a_frame_is_given_in_order(self):
        control, session = layer()
        control.decode(samples_of('av1_tiles')[0])
        [(_, batches, _)] = session.submitted
        [tiles] = batches[1:]
        assert [kind for kind, _ in tiles] == [api.VASliceParameterBufferType] * 4 + [
            api.VASliceDataBufferType]
        data = tiles[-1][1]
        places = [(tile.tile_row, tile.tile_column) for _, tile in tiles[:4]]
        assert places == [(0, 0), (0, 1), (1, 0), (1, 1)]
        last = tiles[3][1]
        assert last.slice_data_offset + last.slice_data_size == len(data)

    def test_grain_frames_show_the_second_surface(self):
        control, session = layer()
        for found in samples_of('av1_grain'):
            control.decode(found)
        pictures = [picture_of(entry) for entry in session.submitted]
        grained = [entry for entry in pictures
                   if entry.film_grain_info.film_grain_info_fields.bits.apply_grain]
        assert len(grained) == 20
        for entry in grained:
            assert entry.current_display_picture != entry.current_frame
            assert entry.current_display_picture not in list(entry.ref_frame_map)
        assert {surface for surface, _, _ in session.shown} <= {
            entry.current_display_picture for entry in grained}
