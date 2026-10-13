"""VP9 frames a hostile file states, held to the specification's ranges.

Each test writes a frame that is outside what the VP9 Bitstream and Decoding
Process Specification allows, or that states something a driver's structures
cannot hold, and asserts that it is refused before the control layer hands a
driver anything. The identifiers are those of ``plans/SECURITY-REVIEW.md``.
"""
import pytest

from pyopengl_video.decoder import DecoderError, Sample
from pyopengl_video.vaapi import vp9_parse
from pyopengl_video.vaapi.vp9_decode import VP9Decoding, picture_parameters
from pyopengl_video.vaapi.vp9_parse import (
    DecodingState,
    StreamError,
    parse_uncompressed_header,
)
from tests.vp9_syntax import INTER, frame, tile_count


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

    def output(self, surface, **shown):
        self.shown.append((surface, shown['visible']))


def layer():
    session = Session()
    control = VP9Decoding(session, 4)
    session.control = control
    return control, session


def sample(data, index=0):
    return Sample(data, index, 1, False, index)


def parse(*frames):
    """The header of the last of `frames`, each parsed after the one before."""
    state = DecodingState()
    for data in frames:
        header = parse_uncompressed_header(data, state)
    return header


class TestProfiles:
    """A6 and D8: a frame that states no samples of its own has its key frame's profile (7.2)."""

    @pytest.mark.parametrize('profile', [1, 2, 3])
    def test_an_inter_frame_of_another_profile_than_its_references_is_refused(self, profile):
        with pytest.raises(StreamError, match=f'profile {profile}'):
            parse(frame(), frame(profile=profile, frame_type=INTER, refs=(0, 0, 0)))

    def test_an_inter_frame_of_profile_0_after_a_profile_2_key_frame_is_refused(self):
        with pytest.raises(StreamError, match='profile 0'):
            parse(frame(profile=2, depth=10), frame(frame_type=INTER, refs=(0, 0, 0)))

    def test_an_inter_frame_of_its_key_frame_s_profile_is_read(self):
        header = parse(frame(profile=2, depth=10),
                       frame(profile=2, frame_type=INTER, refs=(0, 0, 0)))
        assert (header.profile, header.colour.bit_depth) == (2, 10)

    def test_the_driver_is_told_one_profile_for_a_sequence(self):
        """Profile 2 and then profile 1 stated by inter frames under a profile 0 key frame."""
        control, session = layer()
        control.decode(sample(frame()))
        for index, profile in enumerate((2, 1), 1):
            with pytest.raises(StreamError, match=f'profile {profile}'):
                control.decode(sample(frame(profile=profile, frame_type=INTER, refs=(0, 0, 0)),
                                      index))
            control.decode(sample(frame(), index))
        assert [batches[0][0][1].profile for _, batches, _ in session.submitted] == [0, 0, 0]

    def test_the_control_layer_holds_the_profile_itself(self, monkeypatch):
        """With a parser that lets the profile through, the layer still refuses it."""
        monkeypatch.setattr(vp9_parse, '_check_profile', lambda *_arguments: None)
        control, session = layer()
        control.decode(sample(frame()))
        with pytest.raises(DecoderError, match='profile 2'):
            control.decode(sample(frame(profile=2, frame_type=INTER, refs=(0, 0, 0)), 1))
        assert len(session.submitted) == 1

    def test_an_intra_only_frame_of_another_profile_is_refused_by_the_layer(self):
        control, session = layer()
        control.decode(sample(frame()))
        with pytest.raises(DecoderError):
            control.decode(sample(frame(profile=2, depth=10, frame_type=INTER, show_frame=0,
                                        intra_only=1, refresh=1), 1))
        assert len(session.submitted) == 1


class TestFrameSize:
    """A7 and D8: a dimension of 65536 does not fit the 16 bits a driver is told it in."""

    #: The one increment bit a frame 1024 superblocks wide has to state.
    WIDE = (0,)

    @pytest.mark.parametrize(('size', 'columns'), [
        ((65536, 64), WIDE), ((64, 65536), ()), ((65536, 65536), WIDE)])
    def test_a_key_frame_too_large_to_tell_a_driver_of_is_refused(self, size, columns):
        control, session = layer()
        with pytest.raises(DecoderError, match='65535'):
            control.decode(sample(frame(size=size, tile_columns=columns)))
        assert session.formats == []
        assert session.submitted == []

    def test_the_picture_parameters_refuse_it_themselves(self):
        header = parse(frame(size=(65536, 64), tile_columns=self.WIDE))
        with pytest.raises(DecoderError, match='65535'):
            picture_parameters(header, [None] * 8)

    def test_the_largest_frame_a_driver_is_told_of(self):
        header = parse(frame(size=(65535, 65535), tile_columns=self.WIDE))
        parameters = picture_parameters(header, [None] * 8)
        assert (parameters.frame_width, parameters.frame_height) == (65535, 65535)


class TestTiles:
    """A8 and A9: at most 64 tile columns, and every tile inside the frame."""

    def test_64_tile_columns_are_the_most(self):
        """tile_cols_log2 is at most 6 (7.2.11), and a 32768-wide frame may state 7."""
        assert parse(frame(size=(32768, 64), tile_columns=(1, 1, 1, 0))).tile_cols_log2 == 6
        with pytest.raises(StreamError, match='tile_cols_log2 is 7'):
            parse(frame(size=(32768, 64), tile_columns=(1, 1, 1, 1)))

    def test_the_widest_frame_may_state_8(self):
        with pytest.raises(StreamError, match='tile_cols_log2 is 8'):
            parse(frame(size=(65536, 64), tile_columns=(1, 1, 1, 1)))
        assert parse(frame(size=(65536, 64), tile_columns=(1, 1, 0))).tile_cols_log2 == 6

    def test_a_frame_has_a_byte_of_tile_data(self):
        """init_bool() is given at least a byte (9.2.1)."""
        with pytest.raises(StreamError, match='last tile'):
            parse(frame(tiles=0))
        control, session = layer()
        with pytest.raises(StreamError, match='last tile'):
            control.decode(sample(frame(tiles=0)))
        assert session.submitted == []

    def test_a_tile_size_past_the_frame_is_refused(self):
        """A first tile of 0x7fffffff bytes, stated with twelve bytes left."""
        data = frame(size=(512, 64), tile_columns=(1,),
                     tile_data=b'\x7f\xff\xff\xff' + b'\x55' * 12)
        with pytest.raises(StreamError, match='tile 0 of 2'):
            parse(data)
        control, session = layer()
        with pytest.raises(StreamError, match='tile 0 of 2'):
            control.decode(sample(data))
        assert session.submitted == []

    def test_a_tile_of_no_bytes_is_refused(self):
        with pytest.raises(StreamError, match='tile 0 of 2'):
            parse(frame(size=(512, 64), tile_columns=(1,),
                        tile_data=bytes(4) + b'\x55' * 12))

    def test_a_tile_that_leaves_nothing_for_the_last_is_refused(self):
        with pytest.raises(StreamError, match='last tile'):
            parse(frame(size=(512, 64), tile_columns=(1,),
                        tile_data=b'\x00\x00\x00\x02\x55\x55'))

    def test_a_frame_that_ends_inside_a_tile_size_is_refused(self):
        with pytest.raises(StreamError, match='tile 1 of 4'):
            parse(frame(size=(512, 64), tile_columns=(1,), tile_rows=1,
                        tile_data=b'\x00\x00\x00\x01\x55\x00\x00'))

    def test_the_tiles_of_every_row_and_column_are_walked(self):
        assert tile_count(512, (1,), 2) == 8
        header = parse(frame(size=(512, 64), tile_columns=(1,), tile_rows=2))
        assert (header.tile_cols_log2, header.tile_rows_log2) == (1, 2)


class TestParsersRaiseTheirOwnError:
    """Whatever the bytes, the parser answers or raises a StreamError."""

    def test_every_single_bit_flip_and_truncation_of_a_header(self):
        key = frame(size=(512, 64), tile_columns=(1,), segmentation={
            'tree': (1, 2, 3, None, 5, 6, 7), 'pred': (9, None, 11),
            'features': {(0, 0): -20, (3, 1): 9, (7, 2): 3}})
        inter = frame(size=(512, 64), frame_type=INTER, refs=(0, 1, 2), found_ref=None,
                      tile_columns=(1,))
        for data in (key, inter):
            head = len(data) - 20
            for bit in range(head * 8):
                changed = bytearray(data)
                changed[bit >> 3] ^= 0x80 >> (bit & 7)
                state = DecodingState()
                try:
                    parse_uncompressed_header(key, state)
                    parse_uncompressed_header(bytes(changed), state)
                except StreamError:
                    continue
            for length in range(len(data)):
                state = DecodingState()
                try:
                    parse_uncompressed_header(key, state)
                    parse_uncompressed_header(data[:length], state)
                except StreamError:
                    continue
