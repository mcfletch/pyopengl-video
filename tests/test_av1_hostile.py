"""AV1 headers a hostile file states, held to the specification's ranges.

Every length, count and index in a sequence header, a frame header and a tile
group is chosen by whoever made the file. Each test here writes one that is
outside what the AV1 Bitstream & Decoding Process Specification allows, and
asserts that the parser refuses it with a
:class:`~pyopengl_video.vaapi.h264.StreamError` before the control layer
hands a driver anything. The identifiers are those of
``plans/SECURITY-REVIEW.md``.
"""
import dataclasses
import time

import pytest

from pyopengl_video import obu as obus
from pyopengl_video.decoder import DecoderError, Sample
from pyopengl_video.vaapi import api, av1_decode, av1_parse
from pyopengl_video.vaapi.av1 import Frame, GroupOfPictures, level_for
from pyopengl_video.vaapi.av1 import Sequence as EncodedSequence
from pyopengl_video.vaapi.av1_decode import AV1Decoding, format_for, picture_parameters
from pyopengl_video.vaapi.av1_parse import (
    INTER_FRAME,
    INTRA_ONLY_FRAME,
    KEY_FRAME,
    FrameState,
    MissingReference,
    SavedFrame,
    StreamError,
    parse_frame_header,
    parse_sequence_header,
    parse_tile_group,
)
from tests.av1_syntax import (
    Sequence,
    frame,
    frame_header,
    obu,
    show_existing,
    stated_tiles,
    uniform_tiles,
    unit,
)

SEQUENCE = Sequence()

#: The longest a hostile input may take, in seconds.
LIMIT = 1.0


class Session:
    """A decode session that records what it is asked, with no driver."""

    def __init__(self):
        self.control = None
        self.formats = []
        self.submitted = []
        self.shown = []

    def activate(self, format):
        self.formats.append(format)

    def acquire(self):
        held = self.control.held()
        base = 100 * len(self.formats)
        return next(surface for surface in range(base, base + self.formats[-1].surfaces)
                    if surface not in held)

    def submit(self, surface, batches, what):
        self.submitted.append((surface, batches, what))

    def output(self, surface, *, timestamp, duration, visible, colour):  # noqa: ARG002 - the session's interface
        self.shown.append((surface, timestamp, visible))


def layer():
    session = Session()
    control = AV1Decoding(session, 4)
    session.control = control
    return control, session


def sample(data, index=0):
    return Sample(data, index, 1, False, index)


def parse(sequence=SEQUENCE, state=None, **options):
    data, _ = frame_header(sequence, **options)
    return parse_frame_header(data, sequence.parsed(), state or FrameState())


def after_key_frame(sequence=SEQUENCE, **options):
    """The reference slots as a shown key frame of `sequence` leaves them."""
    state = FrameState()
    header = parse(sequence, state, **options)
    state.update(header, sequence.parsed())
    return state


# --------------------------------------------------------------------- tiles


class TestTileGrid:
    """A1 and A10: TileCols and TileRows are at most 64 (6.8.14)."""

    @pytest.mark.parametrize('width', [4160, 8192, 16384, 65536])
    def test_more_than_64_stated_columns_is_refused(self, width):
        sequence = Sequence(size=(width, 64), size_bits=16)
        sb_cols = -(-width // 64)
        tiles = stated_tiles([1] * sb_cols, [1], sb_cols=sb_cols, sb_rows=1)
        with pytest.raises(StreamError, match='64 tile columns'):
            parse(sequence, tile_info=tiles)

    def test_more_than_64_stated_rows_is_refused(self):
        sequence = Sequence(size=(64, 4160), size_bits=16)
        tiles = stated_tiles([1], [1] * 65, sb_cols=1, sb_rows=65)
        with pytest.raises(StreamError, match='64 tile rows'):
            parse(sequence, tile_info=tiles)

    def test_64_stated_columns_and_rows_are_the_most(self):
        sequence = Sequence(size=(4096, 4096), size_bits=16)
        tiles = stated_tiles([1] * 64, [1] * 64, sb_cols=64, sb_rows=64, tallest=32)
        info = parse(sequence, tile_info=tiles).tile_info
        assert (info.tile_cols, info.tile_rows) == (64, 64)
        assert info.width_in_sbs_minus_1 == (0,) * 64

    def test_a_refused_grid_costs_no_more_than_the_columns_allowed(self):
        """A 65536-wide frame states 1024 columns; the 65th ends the parse."""
        sequence = Sequence(size=(65536, 65536), size_bits=16)
        tiles = stated_tiles([1] * 1024, [1] * 1024, sb_cols=1024, sb_rows=1024)
        started = time.perf_counter()
        with pytest.raises(StreamError, match='64 tile columns'):
            parse(sequence, tile_info=tiles)
        assert time.perf_counter() - started < LIMIT

    def test_the_grid_of_a_frame_passed_over_is_not_walked(self):
        """An inter frame that opens a stream is passed over, whatever tiles it states.

        The sample states 1024 by 1024 tiles in 2 MB. Its grid is past the
        limit and its references are missing, and neither costs a walk over
        a million tiles.
        """
        sequence = Sequence(size=(65536, 65536), size_bits=16)
        tiles = stated_tiles([1] * 1024, [1] * 1024, sb_cols=1024, sb_rows=1024)
        header, _ = frame_header(sequence, frame_type=INTER_FRAME, error_resilient=1,
                                 refresh=1, tile_info=tiles)
        data = header + b'\x00' + b'\x00\x55' * (1024 * 1024 - 1) + b'\x55'
        control, session = layer()
        started = time.perf_counter()
        control.decode(sample(unit(sequence.obu(), obu(obus.FRAME, data))))
        assert time.perf_counter() - started < LIMIT
        assert session.submitted == []
        assert session.formats == []

    def test_a_frame_with_no_superblocks_is_refused(self):
        """A size taken from a slot that states none leaves no tile to lay out."""
        state = FrameState()
        state.slots = [SavedFrame(valid=True)] * 8
        with pytest.raises(StreamError, match='0x0'):
            parse(state=state, frame_type=INTER_FRAME, found_ref=0, refresh=1)

    def test_context_update_tile_id_names_a_tile_of_the_frame(self):
        """Three columns take two bits, which can name a fourth."""
        sequence = Sequence(size=(192, 64))
        for named in (0, 1, 2):
            tiles = uniform_tiles((1, 1), (), bits=2, context_update_tile_id=named)
            assert parse(sequence, tile_info=tiles).tile_info.context_update_tile_id == named
        with pytest.raises(StreamError, match='context_update_tile_id is 3'):
            parse(sequence, tile_info=uniform_tiles((1, 1), (), bits=2,
                                                    context_update_tile_id=3))

    def test_context_update_tile_id_with_stated_tiles(self):
        sequence = Sequence(size=(192, 64))
        tiles = stated_tiles([1, 1, 1], [1], sb_cols=3, sb_rows=1, context_update_tile_id=3)
        with pytest.raises(StreamError, match='context_update_tile_id is 3'):
            parse(sequence, tile_info=tiles)


class TestTileGroups:
    """A5: a tile group costs the bytes of its size fields, not of its tiles."""

    def header(self):
        sequence = Sequence(size=(4096, 4096), size_bits=16)
        tiles = uniform_tiles((1,) * 6, (1,) * 6, bits=12, tile_size_bytes=4)
        header = parse(sequence, tile_info=tiles)
        assert header.tile_info.num_tiles == 4096
        return header

    def test_the_most_tiles_in_megabytes_of_data_are_found_at_once(self):
        header = self.header()
        first = 8 << 20
        body = b'\x00' + (first - 1).to_bytes(4, 'little') + b'\xa5' * first
        body += b'\x00\x00\x00\x00\x55' * 4094 + b'\x55'
        started = time.perf_counter()
        group = parse_tile_group(body, header)
        assert time.perf_counter() - started < LIMIT
        assert len(group.tiles) == 4096
        assert (group.tiles[0].offset, group.tiles[0].size) == (5, first)
        assert (group.tiles[1].offset, group.tiles[1].size) == (5 + first + 4, 1)
        last = group.tiles[-1]
        assert (last.row, last.column, last.offset + last.size) == (63, 63, len(body))

    def test_a_frame_header_is_read_from_megabytes_of_data_at_once(self):
        data, _ = frame_header(SEQUENCE)
        started = time.perf_counter()
        for _ in range(200):
            parse_frame_header(data + bytes(4 << 20), SEQUENCE.parsed(), FrameState())
        assert time.perf_counter() - started < LIMIT

    def test_a_tile_size_cut_short_is_refused(self):
        header = self.header()
        with pytest.raises(StreamError, match='tile 0'):
            parse_tile_group(b'\x00\x01\x00', header)

    def test_a_tile_group_that_starts_past_its_data_is_refused(self):
        with pytest.raises(StreamError):
            parse_tile_group(b'\x55' * 4, parse(), start=5)

    def test_an_empty_tile_group_is_refused(self):
        with pytest.raises(StreamError, match='tile 0'):
            parse_tile_group(b'', parse())


# ---------------------------------------------------------------- frame size


class TestFrameSize:
    """A2 and D6: a frame is no larger than its sequence allows (6.8.4)."""

    def test_a_stated_size_past_the_maximum_is_refused(self):
        sequence = Sequence(size=(64, 48), size_bits=16)
        for size in ((65, 48), (64, 49), (4096, 2304)):
            with pytest.raises(StreamError, match=f'{size[0]}x{size[1]}'):
                parse(sequence, size=size)
        assert parse(sequence, size=(64, 48)).size == (64, 48)

    def test_a_size_taken_from_a_reference_is_held_to_the_maximum_too(self):
        large = Sequence(size=(128, 96))
        state = after_key_frame(large, tile_info=uniform_tiles((0,), (0,)))
        with pytest.raises(StreamError, match='128x96'):
            parse(SEQUENCE, state, frame_type=INTER_FRAME, found_ref=0, refresh=1)

    def test_the_size_before_super_resolution_is_what_is_held(self):
        sequence = Sequence(size=(64, 48), size_bits=16, superres=1)
        header = parse(sequence, size=(64, 48), superres_denom=16)
        assert (header.upscaled_width, header.frame_width) == (64, 32)
        with pytest.raises(StreamError, match='66x48'):
            parse(sequence, size=(66, 48), superres_denom=16)

    def test_the_driver_is_not_told_a_frame_larger_than_its_surfaces(self):
        sequence = Sequence(size=(64, 48), size_bits=16)
        control, session = layer()
        with pytest.raises(StreamError):
            control.decode(sample(unit(sequence.obu(), frame(sequence, size=(4096, 2304)))))
        assert session.submitted == []

    def test_the_control_layer_holds_a_frame_to_its_surfaces_itself(self, monkeypatch):
        """With a parser that lets the size through, the layer still refuses it."""
        sequence = Sequence(size=(64, 48), size_bits=16)
        monkeypatch.setattr(av1_parse, '_check_frame_size', lambda *_arguments: None)
        control, session = layer()
        with pytest.raises(StreamError, match='surfaces'):
            control.decode(sample(unit(sequence.obu(), frame(sequence, size=(256, 256)))))
        assert session.submitted == []

    def test_an_inter_frame_is_within_the_scaling_its_references_allow(self):
        """Twice and a sixteenth of each reference are the limits (6.8.6)."""
        sequence = Sequence(size=(128, 96), size_bits=16)
        state = after_key_frame(sequence, tile_info=uniform_tiles((0,), (0,)))
        with pytest.raises(StreamError, match='16x16'):
            parse(sequence, state, frame_type=INTER_FRAME, size=(16, 16), refresh=1)
        assert parse(sequence, state, frame_type=INTER_FRAME, size=(64, 48),
                     refresh=1).size == (64, 48)


# ------------------------------------------------------------ the sequence


class TestColourConfig:
    """A3: the sampling a sequence states is one its profile has (6.4.1, 6.4.2)."""

    SRGB = (1, 13, 0)

    def test_main_profile_has_no_444(self):
        with pytest.raises(StreamError, match='4:4:4'):
            Sequence(colour=self.SRGB).parsed()

    @pytest.mark.parametrize('depth', [(0, 0), (1, 0)])
    def test_professional_profile_has_444_at_12_bits_alone(self, depth):
        high, twelve = depth
        with pytest.raises(StreamError, match='4:4:4'):
            Sequence(profile=2, high_bitdepth=high, twelve_bit=twelve,
                     colour=self.SRGB).parsed()

    def test_the_profiles_that_have_444_keep_it(self):
        for sequence in (Sequence(profile=1, colour=self.SRGB),
                         Sequence(profile=2, high_bitdepth=1, twelve_bit=1, colour=self.SRGB)):
            color = sequence.parsed().color_config
            assert (color.subsampling_x, color.subsampling_y) == (0, 0)

    def test_an_identity_matrix_is_444(self):
        with pytest.raises(StreamError, match='identity'):
            Sequence(colour=(9, 16, 0)).parsed()

    @pytest.mark.parametrize('profile', [3, 7])
    def test_a_reserved_profile_is_refused(self, profile):
        payload = bytearray(SEQUENCE.payload())
        payload[0] = (payload[0] & 0x1F) | profile << 5
        with pytest.raises(StreamError, match=f'seq_profile is {profile}'):
            parse_sequence_header(bytes(payload))

    def test_nothing_but_420_and_monochrome_gets_main_profile_surfaces(self):
        """format_for holds the sampling itself, whatever the parser let by."""
        parsed = SEQUENCE.parsed()
        for sampling in ((0, 0), (1, 0), (0, 1)):
            color = dataclasses.replace(parsed.color_config, subsampling_x=sampling[0],
                                        subsampling_y=sampling[1])
            with pytest.raises(DecoderError, match='4:2:0'):
                format_for(dataclasses.replace(parsed, color_config=color))
        assert format_for(Sequence(mono_chrome=1).parsed()).profile == api.VAProfileAV1Profile0


class TestSequenceChanges:
    """A4: a sequence is replaced at a shown key frame and nowhere else (7.5)."""

    def test_another_sequence_without_a_key_frame_is_refused(self):
        """Super-resolution switched on changes how a header is read and no surface."""
        other = Sequence(superres=1)
        control, session = layer()
        control.decode(sample(unit(SEQUENCE.obu(), frame())))
        with pytest.raises(StreamError, match='begins at a key frame'):
            control.decode(sample(unit(other.obu(), frame(
                other, frame_type=INTER_FRAME, error_resilient=1, refresh=1)), 1))
        assert len(session.submitted) == 1
        assert len(session.formats) == 1

    def test_a_monochrome_sequence_over_colour_references_is_refused(self):
        """An inter frame of a monochrome sequence does not predict from colour."""
        mono = Sequence(mono_chrome=1)
        control, session = layer()
        control.decode(sample(unit(SEQUENCE.obu(), frame())))
        with pytest.raises(StreamError):
            control.decode(sample(unit(mono.obu(), frame(
                mono, frame_type=INTER_FRAME, error_resilient=1, refresh=1)), 1))
        assert len(session.submitted) == 1

    def test_a_shown_existing_frame_does_not_begin_a_sequence(self):
        other = Sequence(superres=1)
        control, session = layer()
        control.decode(sample(unit(SEQUENCE.obu(), frame()),))
        control.decode(sample(unit(frame(frame_type=INTER_FRAME, order_hint=1, show_frame=0,
                                         refresh=0b10)), 1))
        with pytest.raises(StreamError, match='begins at a key frame'):
            control.decode(sample(unit(other.obu(), obu(obus.FRAME_HEADER, show_existing(1))), 2))
        assert len(session.shown) == 1

    def test_another_sequence_at_a_hidden_key_frame_is_refused(self):
        mono = Sequence(mono_chrome=1)
        control, session = layer()
        control.decode(sample(unit(SEQUENCE.obu(), frame())))
        with pytest.raises(StreamError, match='begins at a key frame'):
            control.decode(sample(unit(mono.obu(), frame(mono, show_frame=0, refresh=1)), 1))
        assert len(session.submitted) == 1

    def test_another_sequence_at_a_shown_key_frame_is_decoded(self):
        mono = Sequence(mono_chrome=1)
        control, session = layer()
        control.decode(sample(unit(SEQUENCE.obu(), frame())))
        control.decode(sample(unit(mono.obu(), frame(mono)), 1))
        flags = [batches[0][0][1].seq_info_fields.fields.mono_chrome
                 for _, batches, _ in session.submitted]
        assert flags == [0, 1]

    def test_a_sequence_that_changes_inside_a_frame_is_refused(self):
        """The tile group is not decoded under a sequence its header was not read under."""
        mono = Sequence(mono_chrome=1)
        header, _ = frame_header(SEQUENCE)
        control, session = layer()
        with pytest.raises(StreamError, match='inside a frame'):
            control.decode(sample(unit(
                SEQUENCE.obu(), obu(obus.FRAME_HEADER, header + b'\x80'), mono.obu(),
                obu(obus.TILE_GROUP, b'\x55' * 6))))
        assert session.submitted == []
        assert control.held() == set()

    def test_the_same_sequence_inside_a_frame_changes_nothing(self):
        header, _ = frame_header(SEQUENCE)
        control, session = layer()
        control.decode(sample(unit(
            SEQUENCE.obu(), obu(obus.FRAME_HEADER, header + b'\x80'), SEQUENCE.obu(),
            obu(obus.TILE_GROUP, b'\x55' * 6))))
        assert len(session.submitted) == 1

    def test_operating_parameters_alone_do_not_make_another_sequence(self):
        """Two headers that differ in nothing a frame is read with are one sequence."""
        first = SEQUENCE.parsed()
        point = dataclasses.replace(first.operating_points[0], decoder_buffer_delay=9,
                                    encoder_buffer_delay=4, low_delay_mode_flag=1)
        assert av1_parse.same_sequence(
            first, dataclasses.replace(first, operating_points=(point,)))
        assert not av1_parse.same_sequence(first, Sequence(mono_chrome=1).parsed())

    def test_a_key_frame_that_is_never_finished_begins_no_sequence(self):
        """Its header was read and none of its tiles: the slots hold the frames before."""
        other = Sequence(superres=1)
        header, _ = frame_header(other)
        control, session = layer()
        control.decode(sample(unit(SEQUENCE.obu(), frame())))
        control.decode(sample(unit(other.obu(), obu(obus.FRAME_HEADER, header + b'\x80')), 1))
        with pytest.raises(StreamError):
            control.decode(sample(unit(frame(
                other, frame_type=INTER_FRAME, order_hint=1, refresh=1)), 2))
        assert len(session.submitted) == 1

    def test_a_sequence_header_and_then_the_first_again_changes_nothing(self):
        other = Sequence(superres=1)
        control, session = layer()
        control.decode(sample(unit(SEQUENCE.obu(), frame())))
        control.decode(sample(unit(other.obu(), SEQUENCE.obu(), frame(
            frame_type=INTER_FRAME, order_hint=1, refresh=1)), 1))
        assert len(session.submitted) == 2

    def test_after_a_reset_any_sequence_may_begin(self):
        """With no reference held, nothing decoded under the old sequence remains."""
        mono = Sequence(mono_chrome=1)
        control, session = layer()
        control.decode(sample(unit(SEQUENCE.obu(), frame())))
        control.reset()
        control.decode(sample(unit(mono.obu(), frame(mono, show_frame=0, refresh=1)), 1))
        assert len(session.submitted) == 2


# ---------------------------------------------------------------- film grain


class TestFilmGrain:
    """A11: film grain points are counted, ordered and paired (6.8.20)."""

    GRAIN = Sequence(film_grain=1)

    def points(self, count, step=10):
        return [(step * index, 90) for index in range(count)]

    def test_14_luma_points_are_the_most(self):
        grain = parse(self.GRAIN, grain={'y': self.points(14), 'from_luma': 1}).film_grain
        assert grain.num_y_points == 14
        with pytest.raises(StreamError, match='num_y_points is 15'):
            parse(self.GRAIN, grain={'y': self.points(15), 'from_luma': 1})

    @pytest.mark.parametrize('plane', ['cb', 'cr'])
    def test_10_chroma_points_are_the_most(self, plane):
        other = 'cr' if plane == 'cb' else 'cb'
        grain = {'y': self.points(1), plane: self.points(10), other: self.points(1)}
        assert getattr(parse(self.GRAIN, grain=grain).film_grain, f'num_{plane}_points') == 10
        grain[plane] = self.points(11)
        with pytest.raises(StreamError, match=f'num_{plane}_points is 11'):
            parse(self.GRAIN, grain=grain)

    @pytest.mark.parametrize('plane', ['y', 'cb', 'cr'])
    def test_point_values_increase(self, plane):
        grain = {'y': self.points(2), 'cb': self.points(2), 'cr': self.points(2)}
        grain[plane] = [(40, 90), (40, 91)]
        with pytest.raises(StreamError, match=f'point_{plane}_value'):
            parse(self.GRAIN, grain=grain)
        grain[plane] = [(40, 90), (39, 91)]
        with pytest.raises(StreamError, match=f'point_{plane}_value'):
            parse(self.GRAIN, grain=grain)

    def test_420_chroma_has_points_in_both_planes_or_neither(self):
        for chroma in ({'cb': self.points(2)}, {'cr': self.points(2)}):
            with pytest.raises(StreamError, match='num_cb_points'):
                parse(self.GRAIN, grain={'y': self.points(1), **chroma})
        both = parse(self.GRAIN, grain={'y': self.points(1)}).film_grain
        assert (both.num_cb_points, both.num_cr_points) == (0, 0)

    def test_grain_is_taken_from_a_frame_the_frame_refers_to(self):
        state = after_key_frame(self.GRAIN, grain='new')
        taken = parse(self.GRAIN, state, frame_type=INTER_FRAME, refresh=1,
                      refs=(0, 0, 0, 5, 0, 0, 0), grain=5).film_grain
        assert (taken.film_grain_params_ref_idx, taken.num_y_points) == (5, 1)
        with pytest.raises(StreamError, match='film_grain_params_ref_idx is 6'):
            parse(self.GRAIN, state, frame_type=INTER_FRAME, refresh=1,
                  refs=(0, 0, 0, 5, 0, 0, 0), grain=6)

    def test_the_driver_is_given_every_point_parsed(self):
        header = parse(self.GRAIN, grain={'y': self.points(14), 'from_luma': 1})
        grain = picture_parameters(self.GRAIN.parsed(), header, [None] * 8, 5,
                                   6).film_grain_info
        assert grain.num_y_points == 14
        assert list(grain.point_y_value) == [10 * index for index in range(14)]


# ---------------------------------------------------------- reference slots


class TestEmptySlots:
    """A12 and L3: a slot that holds no frame gives a header nothing."""

    def test_an_inter_frame_names_slots_that_hold_frames(self):
        with pytest.raises(MissingReference, match='reference slot 0'):
            parse(frame_type=INTER_FRAME, refresh=1)

    def test_a_missing_reference_is_a_stream_error(self):
        assert issubclass(MissingReference, StreamError)

    def test_every_one_of_the_seven_is_looked_at(self):
        state = after_key_frame()
        state.invalidate(6)
        with pytest.raises(MissingReference, match='reference slot 6'):
            parse(state=state, frame_type=INTER_FRAME, refresh=1,
                  refs=(0, 0, 0, 0, 0, 0, 6))

    @pytest.mark.parametrize('tiles', [
        None, stated_tiles([1], [1], sb_cols=1, sb_rows=1)])
    def test_a_size_is_not_taken_from_an_empty_slot(self, tiles):
        """A slot that holds no frame states a size of 0x0, which no tile divides."""
        with pytest.raises(StreamError, match='holds no frame'):
            parse(frame_type=INTER_FRAME, found_ref=0, refresh=1, tile_info=tiles)

    def test_short_signalling_names_slots_that_hold_frames(self):
        with pytest.raises(MissingReference, match='holds no frame'):
            parse(frame_type=INTER_FRAME, refresh=1, short_signaling=(0, 1))

    def test_an_empty_slot_is_not_shown(self):
        with pytest.raises(MissingReference, match='shows reference slot 3'):
            parse_frame_header(show_existing(3), SEQUENCE.parsed(), FrameState())

    def test_a_frame_not_marked_showable_is_not_shown_again(self):
        state = after_key_frame()
        with pytest.raises(StreamError, match='showable'):
            parse_frame_header(show_existing(3), SEQUENCE.parsed(), state)

    def test_the_accessor_answers_a_slot_that_holds_a_frame(self):
        state = after_key_frame()
        assert state.saved(4, 'a test').frame_width == 64
        state.invalidate(4)
        with pytest.raises(MissingReference, match='a test.*slot 4'):
            state.saved(4, 'a test')

    def test_a_reference_of_other_samples_is_refused(self):
        """A reference matches the frame in bit depth and subsampling (6.8.2)."""
        state = after_key_frame(Sequence(high_bitdepth=1))
        with pytest.raises(StreamError, match='10 bits'):
            parse(state=state, frame_type=INTER_FRAME, refresh=1)

    def test_an_intra_only_frame_leaves_a_slot_alone(self):
        """One that replaced all eight would be a key frame in all but name (6.8.2)."""
        with pytest.raises(StreamError, match='intra-only'):
            parse(frame_type=INTRA_ONLY_FRAME, refresh=0xFF)

    def test_a_lost_reference_before_a_key_frame_is_passed_over(self):
        control, session = layer()
        control.decode(sample(unit(
            SEQUENCE.obu(), frame(frame_type=INTER_FRAME, order_hint=1, refresh=1),
            obu(obus.TILE_GROUP, b'\x55' * 6))))
        control.decode(sample(unit(obu(obus.FRAME_HEADER, show_existing(3))), 1))
        control.decode(sample(unit(frame()), 2))
        assert len(session.submitted) == 1
        assert [timestamp for _, timestamp, _ in session.shown] == [2]


class TestParsersRaiseTheirOwnError:
    """L3: whatever the bytes, a parser answers or raises a StreamError."""

    def headers(self):
        yield frame_header(SEQUENCE)[0], FrameState()
        yield frame_header(SEQUENCE, frame_type=INTER_FRAME, refresh=1)[0], after_key_frame()
        grain = Sequence(film_grain=1)
        yield (frame_header(grain, frame_type=INTER_FRAME, refresh=1, grain='new')[0],
               after_key_frame(grain, grain='new'))

    def test_every_single_bit_flip_of_a_header(self):
        for data, state in self.headers():
            sequence = (Sequence(film_grain=1) if len(data) > 40 else SEQUENCE).parsed()
            for bit in range(len(data) * 8):
                changed = bytearray(data)
                changed[bit >> 3] ^= 0x80 >> (bit & 7)
                kept = FrameState()
                kept.slots = list(state.slots)
                try:
                    header = parse_frame_header(bytes(changed), sequence, kept)
                except StreamError:
                    continue
                if not header.show_existing_frame:
                    parse_tile_group(bytes(changed) + b'\x55' * 8, header,
                                     (header.header_bits + 7) // 8)

    def test_every_single_bit_flip_of_a_sequence_header(self):
        data = Sequence(film_grain=1, colour=(1, 1, 1)).payload()
        for bit in range(len(data) * 8):
            changed = bytearray(data)
            changed[bit >> 3] ^= 0x80 >> (bit & 7)
            try:
                parse_sequence_header(bytes(changed))
            except StreamError:
                continue

    def test_every_truncation_of_a_header(self):
        data, _ = frame_header(SEQUENCE)
        for length in range(len(data)):
            try:
                parse_frame_header(data[:length], SEQUENCE.parsed(), FrameState())
            except StreamError:
                continue


# ------------------------------------------------------------ the layer


class TestHalfBuiltFrames:
    """L1: a frame whose buffers were not built is not there to add tiles to."""

    def test_a_lone_tile_group_after_a_failure_is_not_submitted(self, monkeypatch):
        """The picture parameters fail to build; the next sample is a tile group alone."""
        def failing(*_arguments):
            raise DecoderError('the picture parameters were not built')
        control, session = layer()
        monkeypatch.setattr(av1_decode, 'picture_parameters', failing)
        header, _ = frame_header(SEQUENCE)
        with pytest.raises(DecoderError, match='not built'):
            control.decode(sample(unit(SEQUENCE.obu(), obu(obus.FRAME_HEADER, header + b'\x80'))))
        assert control.held() == set()
        with pytest.raises(StreamError, match='no frame header'):
            control.decode(sample(obu(obus.TILE_GROUP, b'\x55' * 6), 1))
        assert session.submitted == []

    def test_fifteen_grain_points_then_a_lone_tile_group(self):
        """A header refused for its film grain, then a sample that is a tile group alone."""
        grain = Sequence(film_grain=1)
        bad = frame(grain, grain={'y': [(index, 1) for index in range(15)], 'from_luma': 1})
        control, session = layer()
        with pytest.raises(StreamError, match='num_y_points'):
            control.decode(sample(unit(grain.obu(), bad)))
        assert control.held() == set()
        with pytest.raises(StreamError, match='no frame header'):
            control.decode(sample(obu(obus.TILE_GROUP, b'\x55' * 6), 1))
        assert session.submitted == []

    def test_a_failure_in_a_tile_group_drops_the_frame_it_belonged_to(self):
        sequence = Sequence(size=(192, 64))
        header, _ = frame_header(sequence, tile_info=uniform_tiles((1, 1), (), bits=2))
        control, session = layer()
        with pytest.raises(StreamError, match='tile 0'):
            control.decode(sample(unit(
                sequence.obu(), obu(obus.FRAME_HEADER, header + b'\x80'),
                obu(obus.TILE_GROUP, b'\x00\x09\x55'))))
        assert control.held() == set()
        with pytest.raises(StreamError, match='no frame header'):
            control.decode(sample(obu(obus.TILE_GROUP, b'\x00\x00\x55\x00\x55\x55'), 1))
        assert session.submitted == []

    def test_every_tile_is_inside_the_data_submitted(self):
        sequence = Sequence(size=(192, 64))
        header, _ = frame_header(sequence, tile_info=uniform_tiles((1, 1), (), bits=2))
        control, session = layer()
        control.decode(sample(unit(
            sequence.obu(), obu(obus.FRAME, header + b'\x00\x00\x55\x01\x55\x55\x55'))))
        [(_, batches, _)] = session.submitted
        *tiles, (_, data) = batches[1]
        assert [(tile.tile_row, tile.tile_column, tile.slice_data_size)
                for _, tile in tiles] == [(0, 0, 1), (0, 1, 2), (0, 2, 1)]
        assert all(tile.slice_data_offset + tile.slice_data_size <= len(data)
                   for _, tile in tiles)
        parameters = batches[0][0][1]
        assert (parameters.tile_cols, parameters.tile_rows) == (3, 1)
        assert list(parameters.width_in_sbs_minus_1[:3]) == [0, 0, 0]

    def test_64_tile_columns_reach_the_driver_whole(self):
        """libva takes 63 widths: the last column is what is left of the frame."""
        sequence = Sequence(size=(4096, 64), size_bits=16)
        tiles = stated_tiles([1] * 64, [1], sb_cols=64, sb_rows=1)
        header = parse(sequence, tile_info=tiles)
        parameters = picture_parameters(sequence.parsed(), header, [None] * 8, 5)
        assert parameters.tile_cols == 64
        assert list(parameters.width_in_sbs_minus_1) == [0] * 63


# ------------------------------------------------------------- the writers


class TestWriters:
    """A14: the header writers refuse what they cannot write as asked."""

    def test_a_lossless_quantiser_index_is_refused(self):
        """Index 0 is lossless coding, whose header has another syntax."""
        for index in (0, -1, 256):
            with pytest.raises(ValueError, match='base_q_idx'):
                Frame(key=True, number=0, base_q_idx=index)
            with pytest.raises(ValueError, match='base_q_idx'):
                GroupOfPictures(30, index)
        assert Frame(key=True, number=0, base_q_idx=1).base_q_idx == 1
        assert Frame(key=False, number=3, base_q_idx=255).base_q_idx == 255

    @pytest.mark.parametrize('rate', [(30, 0), (0, 1), (-30, 1), (30, -1)])
    def test_a_frame_rate_has_two_positive_terms(self, rate):
        with pytest.raises(ValueError, match='frame rate'):
            EncodedSequence(64, 48, rate)

    def test_a_frame_rate_fits_its_32_bit_fields(self):
        with pytest.raises(ValueError, match='frame rate'):
            EncodedSequence(64, 48, (1 << 32, 1))

    @pytest.mark.parametrize('rate', [(30, 0), (0, 1), (30, -1)])
    def test_a_level_is_not_worked_out_for_no_frame_rate(self, rate):
        with pytest.raises(ValueError, match='frame rate'):
            level_for(64, 48, rate)

    def test_a_negative_bitrate_is_refused(self):
        with pytest.raises(ValueError, match='bitrate'):
            EncodedSequence(64, 48, (30, 1), -1)

    def test_a_frame_number_is_not_negative(self):
        with pytest.raises(ValueError, match='number'):
            Frame(key=False, number=-1, base_q_idx=20)


def test_key_frame_type_is_zero():
    """The tests above read frame types by name; the driver is told the number."""
    assert KEY_FRAME == 0
