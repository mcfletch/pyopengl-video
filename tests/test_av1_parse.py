"""The AV1 parser, against what ffmpeg's trace_headers reads and on hand-built headers.

``ffmpeg -i X -c copy -bsf:v trace_headers -f null -`` prints every field of a
stream's headers. The values it printed for the fixture streams are the
constants here. The hand-built headers of :mod:`tests.av1_syntax` cover the
syntax no fixture has.
"""
import dataclasses
import json
from pathlib import Path

import pytest

from pyopengl_video import obu
from pyopengl_video.ivf import IVFReader
from pyopengl_video.vaapi import av1_parse
from pyopengl_video.vaapi.av1_parse import (
    AFFINE,
    ALTREF_FRAME,
    DEFAULT_GM_PARAMS,
    DEFAULT_REF_DELTAS,
    DIV_LUT,
    EIGHTTAP,
    IDENTITY,
    INTER_FRAME,
    INTRA_ONLY_FRAME,
    KEY_FRAME,
    PRIMARY_REF_NONE,
    RESTORE_NONE,
    RESTORE_SGRPROJ,
    RESTORE_SWITCHABLE,
    RESTORE_WIENER,
    ROTZOOM,
    SWITCH_FRAME,
    SWITCHABLE,
    TX_MODE_LARGEST,
    TX_MODE_SELECT,
    FrameState,
    Reader,
    StreamError,
    get_relative_dist,
    parse_frame_header,
    parse_sequence_header,
    parse_tile_group,
    warp_valid,
)
from pyopengl_video.vaapi.bits import BitWriter
from tests.av1_syntax import Sequence, frame_header, headers_of, show_existing

DATA = Path(__file__).parent / 'data' / 'decode'
FIXTURES = json.loads((DATA / 'av1.json').read_text())['streams']

#: For every frame header of a fixture, in the order the stream holds them,
#: what trace_headers reports: the header's length in bits, then frame_type,
#: show_frame, order_hint, refresh_frame_flags and base_q_idx; or the length,
#: ``'show'`` and frame_to_show_map_idx for a header that shows an existing
#: frame. A field the header infers is written as its inferred value.
TRACED = {
    'av1_altref': [
        (79, 0, 1, 0, 255, 18), (140, 1, 0, 16, 2, 69), (141, 1, 0, 8, 4, 96), (141, 1, 0, 4, 8, 109),
        (141, 1, 0, 2, 16, 116), (137, 1, 1, 1, 32, 118), (4, 'show', 4), (137, 1, 1, 3, 64, 118), (4, 'show', 3),
        (141, 1, 0, 6, 128, 115), (137, 1, 1, 5, 32, 118), (4, 'show', 7), (137, 1, 1, 7, 16, 117), (4, 'show', 2),
        (141, 1, 0, 12, 64, 107), (141, 1, 0, 10, 8, 113), (137, 1, 1, 9, 32, 116), (4, 'show', 3),
        (137, 1, 1, 11, 128, 115), (4, 'show', 6), (141, 1, 0, 14, 16, 112), (137, 1, 1, 13, 4, 114),
        (4, 'show', 4), (137, 1, 1, 15, 32, 114), (4, 'show', 1), (136, 1, 1, 17, 8, 131),
        (136, 1, 1, 18, 128, 134), (136, 1, 1, 19, 64, 131), (94, 0, 1, 0, 255, 57), (140, 1, 0, 16, 2, 58),
        (141, 1, 0, 8, 4, 82), (141, 1, 0, 4, 8, 94), (141, 1, 0, 2, 16, 100), (137, 1, 1, 1, 32, 102),
        (4, 'show', 4), (137, 1, 1, 3, 64, 102), (4, 'show', 3), (141, 1, 0, 6, 128, 99), (137, 1, 1, 5, 32, 101),
        (4, 'show', 7), (137, 1, 1, 7, 16, 100), (4, 'show', 2), (138, 1, 0, 12, 64, 92), (141, 1, 0, 10, 8, 97),
        (137, 1, 1, 9, 32, 99), (4, 'show', 3), (137, 1, 1, 11, 128, 98), (4, 'show', 6), (141, 1, 0, 14, 16, 96),
        (137, 1, 1, 13, 4, 98), (4, 'show', 4), (137, 1, 1, 15, 32, 98), (4, 'show', 1), (136, 1, 1, 17, 8, 97),
        (136, 1, 1, 18, 128, 97), (136, 1, 1, 19, 64, 97),
    ],
    'av1_tiles': [
        (261, 0, 1, 0, 255, 18), (286, 1, 0, 9, 2, 69), (215, 1, 0, 4, 4, 94), (153, 1, 0, 2, 8, 107),
        (208, 1, 1, 1, 16, 113), (4, 'show', 3), (146, 1, 1, 3, 32, 113), (4, 'show', 2), (153, 1, 0, 6, 64, 107),
        (146, 1, 1, 5, 128, 113), (4, 'show', 6), (146, 1, 1, 7, 16, 113), (146, 1, 1, 8, 8, 112), (4, 'show', 1),
        (237, 0, 1, 0, 255, 5), (286, 1, 0, 9, 2, 68), (215, 1, 0, 4, 4, 93), (153, 1, 0, 2, 8, 104),
        (208, 1, 1, 1, 16, 110), (4, 'show', 3), (146, 1, 1, 3, 32, 110), (4, 'show', 2), (153, 1, 0, 6, 64, 104),
        (146, 1, 1, 5, 128, 109), (4, 'show', 6), (146, 1, 1, 7, 16, 109), (146, 1, 1, 8, 8, 109), (4, 'show', 1),
    ],
    'av1_grain': [
        (851, 0, 1, 0, 255, 22), (134, 1, 0, 9, 2, 83), (914, 1, 0, 4, 4, 106), (914, 1, 0, 2, 8, 118),
        (910, 1, 1, 1, 16, 124), (4, 'show', 3), (910, 1, 1, 3, 32, 124), (4, 'show', 2), (914, 1, 0, 6, 64, 118),
        (910, 1, 1, 5, 128, 124), (4, 'show', 6), (910, 1, 1, 7, 16, 124), (910, 1, 1, 8, 8, 124),
        (909, 1, 1, 9, 0, 129), (863, 0, 1, 0, 255, 46), (134, 1, 0, 9, 2, 85), (914, 1, 0, 4, 4, 108),
        (914, 1, 0, 2, 8, 120), (910, 1, 1, 1, 16, 126), (4, 'show', 3), (910, 1, 1, 3, 32, 126), (4, 'show', 2),
        (911, 1, 0, 6, 64, 120), (910, 1, 1, 5, 128, 126), (4, 'show', 6), (910, 1, 1, 7, 16, 126),
        (910, 1, 1, 8, 8, 126), (909, 1, 1, 9, 0, 131),
    ],
    'av1_intra_only': [
        (39, 0, 1, 0, 255, 66), (137, 1, 0, 8, 1, 107), (122, 1, 0, 4, 8, 125), (122, 1, 0, 2, 32, 133),
        (121, 1, 1, 1, 0, 140), (4, 'show', 5), (121, 1, 1, 3, 0, 140), (4, 'show', 3), (122, 1, 0, 6, 32, 133),
        (121, 1, 1, 5, 0, 140), (4, 'show', 5), (121, 1, 1, 7, 0, 140), (4, 'show', 0), (49, 2, 0, 12, 2, 49),
        (122, 1, 0, 10, 16, 133), (121, 1, 1, 9, 0, 140), (4, 'show', 4), (121, 1, 1, 11, 0, 140), (4, 'show', 1),
        (138, 1, 0, 20, 4, 115), (122, 1, 0, 16, 8, 129), (122, 1, 0, 14, 32, 135), (121, 1, 1, 13, 0, 140),
        (4, 'show', 5), (121, 1, 1, 15, 0, 140), (4, 'show', 3), (134, 1, 0, 18, 32, 135), (121, 1, 1, 17, 0, 140),
        (4, 'show', 5), (121, 1, 1, 19, 0, 140), (4, 'show', 2), (49, 2, 0, 24, 1, 50), (122, 1, 0, 22, 16, 135),
        (121, 1, 1, 21, 0, 140), (4, 'show', 4), (121, 1, 1, 23, 0, 140), (4, 'show', 0), (138, 1, 0, 32, 2, 117),
        (122, 1, 0, 28, 8, 129), (122, 1, 0, 26, 32, 135), (121, 1, 1, 25, 0, 140), (4, 'show', 5),
        (121, 1, 1, 27, 0, 140), (4, 'show', 3), (122, 1, 0, 30, 32, 135), (121, 1, 1, 29, 0, 140), (4, 'show', 5),
        (121, 1, 1, 31, 0, 140), (4, 'show', 1), (49, 2, 0, 36, 4, 63), (122, 1, 0, 34, 16, 135),
        (121, 1, 1, 33, 0, 140), (4, 'show', 4), (121, 1, 1, 35, 0, 140), (4, 'show', 2), (138, 1, 0, 38, 1, 126),
        (121, 1, 1, 37, 16, 140), (4, 'show', 0), (132, 1, 1, 39, 8, 140),
    ],
    'av1_switch_frames': [
        (222, 0, 1, 0, 255, 83), (172, 1, 1, 1, 2, 106), (172, 1, 1, 2, 4, 106), (172, 1, 1, 3, 8, 106),
        (172, 1, 1, 4, 1, 106), (172, 1, 1, 5, 2, 106), (172, 1, 1, 6, 4, 106), (172, 1, 1, 7, 8, 106),
        (317, 3, 1, 8, 255, 106), (172, 1, 1, 9, 2, 106), (172, 1, 1, 10, 4, 106), (172, 1, 1, 11, 8, 106),
        (172, 1, 1, 12, 1, 106), (172, 1, 1, 13, 2, 106), (172, 1, 1, 14, 4, 106), (172, 1, 1, 15, 8, 106),
        (317, 3, 1, 16, 255, 106), (172, 1, 1, 17, 2, 106), (172, 1, 1, 18, 4, 106), (172, 1, 1, 19, 8, 106),
        (172, 1, 1, 20, 1, 106), (172, 1, 1, 21, 2, 106), (172, 1, 1, 22, 4, 106), (172, 1, 1, 23, 8, 106),
        (317, 3, 1, 24, 255, 106), (172, 1, 1, 25, 2, 106), (172, 1, 1, 26, 4, 106), (172, 1, 1, 27, 8, 106),
        (172, 1, 1, 28, 1, 106), (172, 1, 1, 29, 2, 106),
    ],
    'av1_global_motion': [
        (88, 0, 1, 0, 255, 10), (143, 1, 0, 8, 2, 42), (224, 1, 0, 4, 4, 64), (279, 1, 0, 2, 8, 75),
        (267, 1, 1, 1, 16, 80), (4, 'show', 3), (192, 1, 1, 3, 32, 85), (4, 'show', 2), (256, 1, 0, 6, 64, 76),
        (204, 1, 1, 5, 128, 86), (4, 'show', 6), (248, 1, 1, 7, 16, 87), (4, 'show', 1), (154, 1, 0, 14, 8, 43),
        (296, 1, 0, 11, 32, 69), (211, 1, 1, 9, 4, 96), (200, 1, 1, 10, 128, 97), (4, 'show', 5),
        (245, 1, 1, 12, 64, 98), (186, 1, 1, 13, 16, 99), (4, 'show', 3), (151, 1, 0, 19, 1, 51),
        (278, 1, 0, 16, 4, 86), (137, 1, 1, 15, 128, 126), (4, 'show', 2), (269, 1, 1, 17, 32, 127),
        (253, 1, 1, 18, 64, 128), (258, 1, 1, 19, 0, 129),
    ],
    'av1_resilient': [
        (99, 0, 1, 0, 255, 11), (297, 1, 0, 9, 2, 45), (297, 1, 0, 4, 4, 68), (297, 1, 0, 2, 8, 80),
        (294, 1, 1, 1, 16, 86), (282, 1, 1, 2, 0, 91), (293, 1, 1, 3, 32, 85), (293, 1, 1, 4, 0, 91),
        (298, 1, 0, 6, 64, 79), (293, 1, 1, 5, 128, 84), (293, 1, 1, 6, 0, 91), (294, 1, 1, 7, 16, 84),
        (294, 1, 1, 8, 8, 83), (293, 1, 1, 9, 0, 91), (111, 0, 1, 0, 255, 5), (285, 1, 0, 9, 2, 35),
        (298, 1, 0, 4, 4, 54), (282, 1, 0, 2, 8, 62), (293, 1, 1, 1, 16, 66), (294, 1, 1, 2, 0, 76),
        (294, 1, 1, 3, 32, 66), (293, 1, 1, 4, 0, 76), (298, 1, 0, 6, 64, 62), (294, 1, 1, 5, 128, 65),
        (293, 1, 1, 6, 0, 76), (293, 1, 1, 7, 16, 65), (294, 1, 1, 8, 8, 65), (293, 1, 1, 9, 0, 76),
    ],
    'av1_superres': [
        (38, 0, 1, 0, 255, 45), (140, 1, 0, 8, 1, 79), (128, 1, 0, 4, 8, 120), (140, 1, 0, 2, 32, 131),
        (125, 1, 1, 1, 0, 140), (4, 'show', 5), (137, 1, 1, 3, 0, 140), (4, 'show', 3), (140, 1, 0, 6, 32, 131),
        (137, 1, 1, 5, 0, 140), (4, 'show', 5), (125, 1, 1, 7, 0, 140), (4, 'show', 0), (138, 1, 1, 9, 8, 120),
        (38, 0, 1, 10, 255, 45), (140, 1, 0, 18, 1, 79), (128, 1, 0, 14, 8, 120), (128, 1, 0, 12, 32, 131),
        (125, 1, 1, 11, 0, 140), (4, 'show', 5), (125, 1, 1, 13, 0, 140), (4, 'show', 3), (128, 1, 0, 16, 32, 131),
        (137, 1, 1, 15, 0, 140), (4, 'show', 5), (125, 1, 1, 17, 0, 140), (4, 'show', 0), (138, 1, 1, 19, 8, 120),
    ],
    'av1_switchable10': [
        (42, 0, 1, 0, 255, 77), (140, 1, 0, 8, 1, 110), (137, 1, 0, 4, 8, 127), (137, 1, 0, 2, 32, 135),
        (136, 1, 1, 1, 64, 140), (4, 'show', 5), (136, 1, 1, 3, 64, 140), (4, 'show', 3), (149, 1, 0, 6, 32, 135),
        (136, 1, 1, 5, 64, 140), (4, 'show', 5), (136, 1, 1, 7, 64, 140), (4, 'show', 0), (135, 1, 1, 9, 8, 140),
        (42, 0, 1, 10, 255, 78), (152, 1, 0, 18, 1, 113), (137, 1, 0, 14, 8, 127), (137, 1, 0, 12, 32, 135),
        (136, 1, 1, 11, 64, 140), (4, 'show', 5), (136, 1, 1, 13, 64, 140), (4, 'show', 3),
        (137, 1, 0, 16, 32, 135), (136, 1, 1, 15, 64, 140), (4, 'show', 5), (136, 1, 1, 17, 64, 140),
        (4, 'show', 0), (135, 1, 1, 19, 8, 140),
    ],
    'av1_monochrome': [
        (61, 0, 1, 0, 255, 13), (113, 1, 0, 5, 2, 51), (113, 1, 0, 2, 4, 76), (109, 1, 1, 1, 8, 88), (4, 'show', 2),
        (110, 1, 1, 3, 16, 88), (110, 1, 1, 4, 32, 87), (108, 1, 1, 5, 0, 100), (61, 0, 1, 0, 255, 13),
        (110, 1, 0, 5, 2, 41), (113, 1, 0, 2, 4, 63), (110, 1, 1, 1, 8, 73), (4, 'show', 2), (110, 1, 1, 3, 16, 73),
        (109, 1, 1, 4, 32, 73), (109, 1, 1, 5, 0, 86),
    ],
    'av1_still': [
        (55, 0, 1, 0, 255, 128),
    ],
    'av1_tile_sizes': [
        (78, 0, 1, 0, 255, 23), (151, 1, 0, 5, 2, 85), (137, 1, 0, 2, 4, 108), (136, 1, 1, 1, 8, 120), (4, 'show', 2),
        (136, 1, 1, 3, 16, 120), (136, 1, 1, 4, 32, 120), (135, 1, 1, 5, 0, 131), (90, 0, 1, 0, 255, 62),
        (139, 1, 0, 5, 2, 84), (140, 1, 0, 2, 4, 107), (136, 1, 1, 1, 8, 119), (4, 'show', 2),
        (136, 1, 1, 3, 16, 119), (136, 1, 1, 4, 32, 119), (135, 1, 1, 5, 0, 130),
    ],
    'av1_grain10': [
        (1054, 0, 1, 0, 255, 16), (133, 1, 0, 9, 2, 50), (1104, 1, 0, 4, 4, 71), (1104, 1, 0, 2, 8, 81),
        (1101, 1, 1, 1, 16, 86), (4, 'show', 3), (1101, 1, 1, 3, 32, 85), (4, 'show', 2), (1104, 1, 0, 6, 64, 80),
        (1100, 1, 1, 5, 128, 85), (4, 'show', 6), (1101, 1, 1, 7, 16, 85), (1101, 1, 1, 8, 8, 84),
        (1100, 1, 1, 9, 0, 91), (1066, 0, 1, 0, 255, 11), (133, 1, 0, 9, 2, 44), (1104, 1, 0, 4, 4, 62),
        (1105, 1, 0, 2, 8, 70), (1100, 1, 1, 1, 16, 74), (4, 'show', 3), (1101, 1, 1, 3, 32, 74), (4, 'show', 2),
        (1101, 1, 0, 6, 64, 70), (1101, 1, 1, 5, 128, 73), (4, 'show', 6), (1100, 1, 1, 7, 16, 73),
        (1101, 1, 1, 8, 8, 73), (1100, 1, 1, 9, 0, 84),
    ],
}


def headers(name):
    with IVFReader(DATA / f'{name}.ivf') as stream:
        return headers_of([stream.sample(index).data for index in range(len(stream))])


def bits(*fields):
    """Bytes holding `fields`, each a (value, width) pair, and a trailing one bit."""
    writer = BitWriter()
    for value, width in fields:
        writer.f(value, width)
    writer.flag(1)
    return writer.bytes()


class TestDescriptors:
    def test_ns_codes_the_low_values_in_one_bit_fewer(self):
        """The specification's table for ns(5): 00, 01, 10, 110, 111."""
        for value, code in enumerate(('00', '01', '10', '110', '111')):
            reader = Reader(int(code.ljust(8, '0'), 2).to_bytes(1, 'big'))
            assert reader.ns(5) == value
            assert reader.position == len(code)

    def test_ns_of_one_value_reads_nothing(self):
        reader = Reader(b'\xff')
        assert reader.ns(1) == 0
        assert reader.position == 0

    def test_le_is_little_endian(self):
        assert Reader(b'\x34\x12\xff').le(2) == 0x1234

    def test_uvlc_counts_its_leading_zeroes(self):
        assert Reader(b'\x80').uvlc() == 0
        assert Reader(bits((0b00101, 5))).uvlc() == 4
        assert Reader(bytes(5)).uvlc() == (1 << 32) - 1

    def test_delta_q_is_zero_unless_coded(self):
        assert Reader(b'\x00').delta_q() == 0
        assert Reader(bits((1, 1), (0b1111110, 7))).delta_q() == -2

    def test_subexp_reads_small_values_in_three_bits(self):
        reader = Reader(bits((0, 1), (0b101, 3)))
        assert reader.subexp(100) == 5
        assert reader.position == 4

    def test_subexp_steps_up_by_its_more_bits(self):
        # One more_bits moves the base to 8, the next to 16; then four bits.
        assert Reader(bits((0b110, 3), (0b0011, 4))).subexp(1000) == 16 + 3

    def test_subexp_ends_in_ns_near_the_limit(self):
        # With 20 symbols the second step leaves 12 values, read as ns(12).
        assert Reader(bits((1, 1), (0b011, 3))).subexp(20) == 8 + 3

    def test_a_value_coded_about_its_reference(self):
        """Even codes go above the reference and odd ones below."""
        for code, value in ((0, 10), (1, 9), (2, 11), (3, 8)):
            reader = Reader(bits((0, 1), (code, 3)))
            assert reader.unsigned_subexp_with_ref(1000, 10) == value

    def test_a_signed_value_coded_about_its_reference(self):
        reader = Reader(bits((0, 1), (1, 3)))
        assert reader.signed_subexp_with_ref(-50, 51, 0) == -1

    def test_reading_past_the_end_is_a_stream_error(self):
        with pytest.raises(StreamError):
            Reader(b'\x00').f(9)

    def test_the_relative_distance_wraps(self):
        assert get_relative_dist(1, 127, 7) == 2
        assert get_relative_dist(127, 1, 7) == -2
        assert get_relative_dist(5, 3, 0) == 0


class TestWarp:
    def test_the_division_table_is_the_specifications(self):
        """Its first row, two entries from the middle, and its last entry."""
        assert DIV_LUT[:11] == (16384, 16320, 16257, 16194, 16132, 16070, 16009,
                                15948, 15888, 15828, 15768)
        assert DIV_LUT[154:156] == (10230, 10205)
        assert DIV_LUT[165] == 9963
        assert DIV_LUT[-1] == 8192
        assert len(DIV_LUT) == 257

    def test_no_motion_is_a_valid_warp(self):
        assert warp_valid(DEFAULT_GM_PARAMS)

    def test_a_small_rotation_is_valid(self):
        assert warp_valid((433152, -628736, 65654, -8192, 8192, 65654))

    def test_a_large_shear_is_not(self):
        assert not warp_valid((0, 0, 1 << 16, 1 << 14, 0, 1 << 16))

    def test_a_model_with_no_scale_is_not(self):
        assert not warp_valid((0, 0, 0, 0, 0, 1 << 16))


class TestSequenceHeader:
    def test_libaom_sequence_header(self):
        """av1_altref: the fields trace_headers prints for its sequence header."""
        _, sequence = headers('av1_altref')[0]
        assert sequence.seq_profile == 0
        assert (sequence.still_picture, sequence.reduced_still_picture_header) == (0, 0)
        assert sequence.timing_info is None
        assert len(sequence.operating_points) == 1
        assert sequence.operating_points[0].idc == 0
        assert sequence.operating_points[0].seq_level_idx == 0
        assert (sequence.frame_width_bits_minus_1, sequence.frame_height_bits_minus_1) == (6, 6)
        assert sequence.size == (120, 68)
        assert sequence.frame_id_numbers_present_flag == 0
        assert sequence.use_128x128_superblock == 0
        assert (sequence.enable_filter_intra, sequence.enable_intra_edge_filter) == (1, 1)
        assert sequence.enable_interintra_compound == 0
        assert sequence.enable_masked_compound == 1
        assert sequence.enable_warped_motion == 1
        assert sequence.enable_dual_filter == 0
        assert (sequence.enable_order_hint, sequence.order_hint_bits) == (1, 7)
        assert (sequence.enable_jnt_comp, sequence.enable_ref_frame_mvs) == (0, 1)
        assert sequence.seq_force_screen_content_tools == 2
        assert sequence.seq_force_integer_mv == 2
        assert (sequence.enable_superres, sequence.enable_cdef,
                sequence.enable_restoration) == (0, 1, 1)
        color = sequence.color_config
        assert (color.bit_depth, color.mono_chrome, color.num_planes) == (8, 0, 3)
        assert color.color_description_present_flag == 0
        assert (color.color_primaries, color.transfer_characteristics,
                color.matrix_coefficients) == (2, 2, 2)
        assert (color.color_range, color.chroma_sample_position) == (0, 0)
        assert (color.subsampling_x, color.subsampling_y) == (1, 1)
        assert color.separate_uv_delta_q == 0
        assert sequence.film_grain_params_present == 0

    def test_rav1e_sequence_header(self):
        """av1_switch_frames: a colour description, forced tools, separate chroma deltas."""
        _, sequence = headers('av1_switch_frames')[0]
        assert sequence.operating_points[0].seq_level_idx == 31
        assert sequence.operating_points[0].seq_tier == 0
        assert (sequence.frame_width_bits_minus_1, sequence.frame_height_bits_minus_1) == (6, 5)
        assert sequence.size == (96, 54)
        assert sequence.seq_force_screen_content_tools == 0
        assert sequence.order_hint_bits == 6
        assert sequence.color_config.color_description_present_flag == 1
        assert sequence.color_config.separate_uv_delta_q == 1
        assert sequence.enable_ref_frame_mvs == 0

    def test_a_reduced_still_picture_header(self):
        """av1_still: no operating points, tools or order hints are coded."""
        header, sequence = headers('av1_still')[0]
        assert (sequence.still_picture, sequence.reduced_still_picture_header) == (1, 1)
        assert sequence.size == (64, 36)
        assert (sequence.enable_order_hint, sequence.order_hint_bits) == (0, 0)
        assert (sequence.enable_filter_intra, sequence.enable_cdef,
                sequence.enable_restoration) == (1, 1, 0)
        assert (header.frame_type, header.show_frame, header.showable_frame) == (KEY_FRAME, 1, 0)
        assert header.error_resilient_mode == 1
        assert header.refresh_frame_flags == 0xFF
        assert header.disable_frame_end_update_cdf == 1

    def test_ten_bits(self):
        _, sequence = headers('av1_grain10')[0]
        assert sequence.color_config.bit_depth == 10
        assert sequence.film_grain_params_present == 1

    def test_monochrome_has_one_plane_and_no_chroma_syntax(self):
        _, sequence = headers('av1_monochrome')[0]
        color = sequence.color_config
        assert (color.mono_chrome, color.num_planes) == (1, 1)
        assert (color.subsampling_x, color.subsampling_y) == (1, 1)
        assert color.separate_uv_delta_q == 0

    def test_frame_ids(self):
        """av1_resilient: libaom's error resilient mode numbers its frames."""
        _, sequence = headers('av1_resilient')[0]
        assert sequence.frame_id_numbers_present_flag == 1
        assert sequence.id_len == 15

    def test_hand_built_sequence_header(self):
        sequence = Sequence(size=(200, 100), high_bitdepth=1, film_grain=1, superres=1,
                            colour=(9, 16, 9), color_range=1).parsed()
        assert sequence.size == (200, 100)
        assert sequence.color_config.bit_depth == 10
        assert (sequence.color_config.color_primaries,
                sequence.color_config.transfer_characteristics,
                sequence.color_config.matrix_coefficients) == (9, 16, 9)
        assert sequence.color_config.color_range == 1
        assert (sequence.enable_superres, sequence.film_grain_params_present) == (1, 1)
        assert sequence.seq_force_screen_content_tools == 0
        assert sequence.order_hint_bits == 7

    def test_srgb_in_high_profile_is_not_subsampled(self):
        color = Sequence(profile=1, colour=(1, 13, 0)).parsed().color_config
        assert (color.subsampling_x, color.subsampling_y, color.color_range) == (0, 0, 1)

    def test_operating_points(self):
        sequence = Sequence(operating_points=(0x103, 0x101)).parsed()
        assert [point.idc for point in sequence.operating_points] == [0x103, 0x101]
        assert sequence.operating_point_idc == 0x103

    def test_timing_and_the_decoder_model(self):
        """Timing, a decoder model, and an operating point with both delays."""
        writer = BitWriter()
        writer.f(0, 3)                          # seq_profile
        writer.f(0, 2)                          # still_picture, reduced
        writer.flag(1)                          # timing_info_present_flag
        writer.f(1001, 32)
        writer.f(30000, 32)
        writer.flag(1)                          # equal_picture_interval
        writer.f(0b00101, 5)                    # num_ticks_per_picture_minus_1 = 4
        writer.flag(1)                          # decoder_model_info_present_flag
        writer.f(9, 5)                          # buffer_delay_length_minus_1
        writer.f(90000, 32)
        writer.f(7, 5)                          # buffer_removal_time_length_minus_1
        writer.f(11, 5)                         # frame_presentation_time_length_minus_1
        writer.flag(1)                          # initial_display_delay_present_flag
        writer.f(0, 5)                          # operating_points_cnt_minus_1
        writer.f(0, 12)
        writer.f(9, 5)                          # seq_level_idx
        writer.flag(1)                          # seq_tier
        writer.flag(1)                          # decoder_model_present_for_this_op
        writer.f(700, 10)
        writer.f(300, 10)
        writer.flag(1)                          # low_delay_mode_flag
        writer.flag(1)                          # initial_display_delay_present_for_this_op
        writer.f(5, 4)
        writer.f(3, 4)                          # frame_width_bits_minus_1
        writer.f(3, 4)
        writer.f(15, 4)
        writer.f(9, 4)
        writer.flag(1)                          # frame_id_numbers_present_flag
        writer.f(3, 4)                          # delta_frame_id_length_minus_2
        writer.f(2, 3)                          # additional_frame_id_length_minus_1
        writer.f(0b101, 3)                      # 128x128, filter_intra, intra_edge
        writer.f(0b11111, 5)                    # interintra, masked, warped, dual, order_hint
        writer.f(0b11, 2)                       # jnt_comp, ref_frame_mvs
        writer.flag(0)                          # seq_choose_screen_content_tools
        writer.flag(1)                          # seq_force_screen_content_tools
        writer.flag(0)                          # seq_choose_integer_mv
        writer.flag(1)                          # seq_force_integer_mv
        writer.f(4, 3)                          # order_hint_bits_minus_1
        writer.f(0b111, 3)                      # superres, cdef, restoration
        writer.f(0b000, 3)                      # high_bitdepth, mono_chrome, description
        writer.flag(1)                          # color_range
        writer.f(2, 2)                          # chroma_sample_position
        writer.flag(1)                          # separate_uv_delta_q
        writer.flag(1)                          # film_grain_params_present
        writer.flag(1)
        sequence = parse_sequence_header(writer.bytes())
        timing = sequence.timing_info
        assert (timing.num_units_in_display_tick, timing.time_scale) == (1001, 30000)
        assert timing.num_ticks_per_picture_minus_1 == 4
        model = sequence.decoder_model_info
        assert (model.buffer_delay_length_minus_1, model.num_units_in_decoding_tick,
                model.buffer_removal_time_length_minus_1,
                model.frame_presentation_time_length_minus_1) == (9, 90000, 7, 11)
        assert sequence.operating_points == (av1_parse.OperatingPoint(
            idc=0, seq_level_idx=9, seq_tier=1, decoder_model_present=1,
            decoder_buffer_delay=700, encoder_buffer_delay=300, low_delay_mode_flag=1,
            initial_display_delay_minus_1=5),)
        assert sequence.size == (16, 10)
        assert sequence.id_len == 3 + 2 + 3
        assert sequence.use_128x128_superblock == 1
        assert (sequence.seq_force_screen_content_tools, sequence.seq_force_integer_mv) == (1, 1)
        assert sequence.order_hint_bits == 5
        assert sequence.color_config.chroma_sample_position == 2
        assert sequence.color_config.separate_uv_delta_q == 1
        assert sequence.film_grain_params_present == 1

    def test_a_header_cut_short_is_a_stream_error(self):
        with pytest.raises(StreamError):
            parse_sequence_header(Sequence().payload()[:3])


class TestTracedHeaders:
    @pytest.mark.parametrize('name', sorted(TRACED))
    def test_every_frame_header_reads_as_ffmpeg_reads_it(self, name):
        found = []
        for header, _ in headers(name):
            if header.show_existing_frame:
                found.append((header.header_bits, 'show', header.frame_to_show_map_idx))
            else:
                found.append((header.header_bits, header.frame_type, header.show_frame,
                              header.order_hint, header.refresh_frame_flags,
                              header.quantization.base_q_idx))
        assert found == TRACED[name]

    def test_every_fixture_is_traced(self):
        assert sorted(TRACED) == sorted(FIXTURES)

    def test_a_hidden_inter_frame(self):
        """av1_tiles, the second header: an alternate reference, hidden."""
        header, _ = headers('av1_tiles')[1]
        assert (header.frame_type, header.show_frame, header.showable_frame) == (INTER_FRAME, 0, 1)
        assert header.error_resilient_mode == 0
        assert header.allow_screen_content_tools == 1
        assert header.force_integer_mv == 0
        assert header.primary_ref_frame == PRIMARY_REF_NONE
        assert header.ref_frame_idx == (0,) * 7
        assert header.frame_refs_short_signaling == 0
        assert header.allow_high_precision_mv == 1
        assert (header.is_filter_switchable, header.interpolation_filter) == (0, EIGHTTAP)
        assert (header.is_motion_mode_switchable, header.use_ref_frame_mvs) == (1, 1)
        assert header.disable_frame_end_update_cdf == 0
        assert (header.tx_mode, header.reference_select) == (TX_MODE_SELECT, 0)
        assert (header.allow_warped_motion, header.reduced_tx_set) == (1, 0)
        assert header.size == (160, 90)
        assert (header.mi_cols, header.mi_rows) == (40, 24)
        assert header.order_hints == (0,) * 8

    def test_uniform_tiles(self):
        header, _ = headers('av1_tiles')[0]
        tiles = header.tile_info
        assert tiles.uniform_tile_spacing_flag == 1
        assert (tiles.tile_cols_log2, tiles.tile_rows_log2) == (1, 1)
        assert (tiles.tile_cols, tiles.tile_rows) == (2, 2)
        assert (tiles.context_update_tile_id, tiles.tile_size_bytes) == (0, 2)
        assert tiles.mi_col_starts == (0, 32, 40)
        assert tiles.mi_row_starts == (0, 16, 24)
        assert tiles.width_in_sbs_minus_1 == (1, 0)
        assert tiles.height_in_sbs_minus_1 == (0, 0)

    def test_tiles_of_coded_sizes(self):
        header, _ = headers('av1_tile_sizes')[0]
        tiles = header.tile_info
        assert tiles.uniform_tile_spacing_flag == 0
        assert tiles.width_in_sbs_minus_1 == (0, 1)
        assert tiles.height_in_sbs_minus_1 == (0, 0)
        assert (tiles.tile_cols, tiles.tile_rows) == (2, 2)
        assert (tiles.context_update_tile_id, tiles.tile_size_bytes) == (1, 2)
        assert tiles.mi_col_starts == (0, 16, 40)

    def test_quantiser_matrices_and_segments(self):
        header, _ = headers('av1_tiles')[0]
        quantization = header.quantization
        assert (quantization.using_qmatrix, quantization.qm_y, quantization.qm_u,
                quantization.qm_v) == (1, 5, 5, 5)
        segmentation = header.segmentation
        assert (segmentation.enabled, segmentation.update_map, segmentation.temporal_update,
                segmentation.update_data) == (1, 1, 0, 1)
        assert [data[0] for data in segmentation.feature_data] == [
            -14, -11, -7, -2, 0, 4, 8, 13]
        assert all(enabled == (1, 0, 0, 0, 0, 0, 0, 0)
                   for enabled in segmentation.feature_enabled)
        assert (segmentation.last_active_seg_id, segmentation.seg_id_pre_skip) == (7, 0)

    def test_delta_q_and_delta_lf(self):
        header, _ = headers('av1_tiles')[0]
        assert (header.delta_q_present, header.delta_q_res) == (1, 2)
        assert (header.delta_lf_present, header.delta_lf_res, header.delta_lf_multi) == (1, 1, 0)

    def test_cdef_strengths(self):
        header, _ = headers('av1_tiles')[0]
        cdef = header.cdef
        assert (cdef.damping_minus_3, cdef.bits) == (0, 2)
        assert cdef.y_pri_strength == (0, 2, 0, 0)
        assert cdef.y_sec_strength == (2, 0, 0, 0)
        assert cdef.uv_pri_strength == (2, 8, 4, 0)
        assert cdef.uv_sec_strength == (2, 0, 0, 0)

    def test_a_coded_secondary_strength_of_three_is_four(self):
        """av1_global_motion, the third header: cdef_uv_sec_strength[0] is coded 3."""
        header, _ = headers('av1_global_motion')[2]
        assert header.cdef.uv_sec_strength == (4, 0)
        assert header.cdef.y_sec_strength == (2, 2)

    def test_loop_filter_levels(self):
        header, _ = headers('av1_grain')[1]
        assert header.loop_filter.level == (4, 4, 14, 12)
        assert header.loop_filter.sharpness == 0
        assert (header.loop_filter.delta_enabled, header.loop_filter.delta_update) == (1, 0)
        assert header.loop_filter.ref_deltas == DEFAULT_REF_DELTAS

    def test_film_grain(self):
        header, _ = headers('av1_grain')[0]
        grain = header.film_grain
        assert (grain.apply_grain, grain.grain_seed, grain.update_grain) == (1, 48612, 1)
        assert (grain.point_y_value, grain.point_y_scaling) == ((0, 255), (192, 192))
        assert grain.chroma_scaling_from_luma == 0
        assert (grain.point_cb_value, grain.point_cb_scaling) == ((0, 255), (128, 128))
        assert (grain.point_cr_value, grain.point_cr_scaling) == ((0, 255), (128, 128))
        assert (grain.grain_scaling_minus_8, grain.ar_coeff_lag) == (3, 3)
        assert grain.ar_coeffs_y == (132, 129, 131, 128, 129, 125, 136, 125, 135, 105, 129,
                                     103, 128, 118, 134, 111, 124, 181, 164, 133, 123, 111,
                                     136, 194)
        assert grain.ar_coeffs_cb == (132, 121, 130, 132, 140, 116, 133, 120, 134, 136, 109,
                                      112, 147, 118, 126, 145, 86, 186, 126, 115, 137, 142,
                                      92, 195, 128)
        assert grain.ar_coeffs_cr == grain.ar_coeffs_cb
        assert (grain.ar_coeff_shift_minus_6, grain.grain_scale_shift) == (1, 1)
        assert (grain.cb_mult, grain.cb_luma_mult, grain.cb_offset) == (128, 192, 256)
        assert (grain.cr_mult, grain.cr_luma_mult, grain.cr_offset) == (128, 192, 256)
        assert (grain.overlap_flag, grain.clip_to_restricted_range) == (1, 1)

    def test_a_frame_never_shown_has_no_film_grain(self):
        """av1_grain, the second header: hidden and not showable."""
        header, _ = headers('av1_grain')[1]
        assert (header.show_frame, header.showable_frame) == (0, 0)
        assert header.film_grain == av1_parse.FilmGrain()

    def test_global_motion(self):
        """av1_global_motion, the third header: ALTREF_FRAME turns and grows.

        trace_headers prints the model's kind. The parameters are what the
        hardware decodes the stream exactly with.
        """
        header, _ = headers('av1_global_motion')[2]
        assert header.gm_type == (IDENTITY,) * 7 + (ROTZOOM,)
        assert header.gm_params[ALTREF_FRAME] == (433152, -628736, 65654, -8192, 8192, 65654)
        assert all(header.gm_params[ref] == DEFAULT_GM_PARAMS for ref in range(ALTREF_FRAME))
        assert (header.reference_select, header.skip_mode_present) == (1, 1)
        assert header.ref_frame_idx == (0, 0, 0, 0, 0, 0, 1)
        assert header.order_hints == (0, 0, 0, 0, 0, 0, 0, 8)

    def test_a_resilient_frame_states_its_references_order_hints(self):
        header, _ = headers('av1_resilient')[1]
        assert header.error_resilient_mode == 1
        assert header.current_frame_id == 19019
        assert header.ref_order_hint == (0,) * 8
        assert header.primary_ref_frame == PRIMARY_REF_NONE
        assert header.use_ref_frame_mvs == 0

    def test_a_switch_frame(self):
        """av1_switch_frames, the ninth header."""
        header, _ = headers('av1_switch_frames')[8]
        assert (header.frame_type, header.show_frame) == (SWITCH_FRAME, 1)
        assert (header.error_resilient_mode, header.frame_size_override_flag) == (1, 1)
        assert header.refresh_frame_flags == 0xFF
        assert header.ref_order_hint == (4, 5, 6, 7, 0, 0, 0, 0)
        assert header.ref_frame_idx == (3, 2, 3, 3, 3, 3, 3)
        assert header.size == (96, 54)
        assert header.order_hints == (0, 7, 6, 7, 7, 7, 7, 7)
        quantization = header.quantization
        assert (quantization.delta_q_y_dc, quantization.diff_uv_delta) == (14, 1)
        assert (quantization.delta_q_u_dc, quantization.delta_q_u_ac) == (1, -10)
        assert (quantization.delta_q_v_dc, quantization.delta_q_v_ac) == (-26, -39)
        assert [data[0] for data in header.segmentation.feature_data] == [
            -10, -9, 10, 15, 0, 0, 0, 0]
        assert header.segmentation.last_active_seg_id == 3
        assert header.loop_filter.level == (9, 48, 5, 9)
        assert header.loop_filter.delta_enabled == 0
        assert (header.tx_mode, header.reduced_tx_set) == (TX_MODE_LARGEST, 1)

    def test_loop_restoration_types_are_remapped(self):
        """lr_type 1 is switchable, 2 is Wiener and 3 is self-guided."""
        header, _ = headers('av1_switch_frames')[8]
        restoration = header.loop_restoration
        assert restoration.frame_restoration_type == (RESTORE_SWITCHABLE,) * 3
        assert (restoration.uses_lr, restoration.unit_shift, restoration.uv_shift) == (1, 0, 1)
        header, _ = headers('av1_superres')[1]
        assert header.loop_restoration.frame_restoration_type == (
            RESTORE_WIENER, RESTORE_NONE, RESTORE_NONE)
        assert header.loop_restoration.unit_shift == 1
        header, _ = headers('av1_altref')[28]
        assert header.loop_restoration.frame_restoration_type == (
            RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_NONE)

    def test_segments_persist_from_the_primary_reference(self):
        """av1_switch_frames, the second header: segmentation_update_data is 0.

        Its features are the key frame's, which trace_headers prints there.
        """
        (key, _), (header, _) = headers('av1_switch_frames')[:2]
        assert (header.segmentation.enabled, header.segmentation.update_map,
                header.segmentation.update_data) == (1, 1, 0)
        assert header.primary_ref_frame == 2
        assert header.segmentation.feature_data == key.segmentation.feature_data
        assert [data[0] for data in header.segmentation.feature_data[:4]] == [-15, -14, 15, 18]
        assert header.segmentation.last_active_seg_id == 3

    def test_an_intra_only_frame(self):
        """av1_intra_only, the fourteenth header."""
        header, _ = headers('av1_intra_only')[13]
        assert (header.frame_type, header.show_frame, header.showable_frame) == (
            INTRA_ONLY_FRAME, 0, 1)
        assert header.refresh_frame_flags == 2
        assert header.primary_ref_frame == PRIMARY_REF_NONE
        assert (header.allow_intrabc, header.force_integer_mv) == (1, 1)
        assert header.loop_filter.level == (0, 0, 0, 0)
        assert header.delta_q_present == 1
        assert header.delta_lf_present == 0

    def test_switchable_interpolation(self):
        header, _ = headers('av1_superres')[1]
        assert (header.is_filter_switchable, header.interpolation_filter) == (1, SWITCHABLE)

    def test_super_resolution(self):
        """av1_superres, the second header: coded_denom 3 is a denominator of 12."""
        header, _ = headers('av1_superres')[1]
        assert (header.use_superres, header.superres_denom) == (1, 12)
        assert (header.frame_width, header.upscaled_width, header.frame_height) == (80, 120, 72)
        assert header.size == (120, 72)
        assert (header.render_width, header.render_height) == (120, 72)
        assert header.mi_cols == 20
        assert header.ref_frame_idx == (2,) * 7
        assert header.primary_ref_frame == 0

    def test_a_shown_existing_frame_takes_its_slots_frame(self):
        """av1_altref, the seventh header shows slot 4: the frame at order hint 2."""
        header, _ = headers('av1_altref')[6]
        assert (header.show_existing_frame, header.frame_to_show_map_idx) == (1, 4)
        assert (header.frame_type, header.order_hint) == (INTER_FRAME, 2)
        assert header.refresh_frame_flags == 0
        assert header.size == (120, 68)


def state_with(sequence, *frames):
    """A state after `frames`, each the options of a :func:`frame_header`."""
    state = FrameState()
    parsed = sequence.parsed()
    for options in frames:
        data, _ = frame_header(sequence, **options)
        state.update(parse_frame_header(data, parsed, state), parsed)
    return state, parsed


class TestHandBuiltHeaders:
    def test_a_key_frame(self):
        sequence = Sequence()
        data, length = frame_header(sequence, base_q_idx=33, loop_filter=(5, 6))
        header = parse_frame_header(data, sequence.parsed(), FrameState())
        assert header.header_bits == length
        assert (header.frame_type, header.show_frame, header.showable_frame) == (KEY_FRAME, 1, 0)
        assert header.error_resilient_mode == 1
        assert header.refresh_frame_flags == 0xFF
        assert header.size == (64, 48)
        assert (header.render_width, header.render_height) == (64, 48)
        assert header.quantization.base_q_idx == 33
        assert header.loop_filter.level == (5, 6, 1, 2)
        assert header.loop_filter.sharpness == 3
        assert header.tx_mode == TX_MODE_SELECT
        assert header.allow_screen_content_tools == 0
        assert header.force_integer_mv == 1
        assert header.coded_lossless == 0

    def test_a_quantiser_index_of_zero_is_lossless(self):
        sequence = Sequence()
        data, length = frame_header(sequence, base_q_idx=0)
        header = parse_frame_header(data, sequence.parsed(), FrameState())
        assert header.header_bits == length
        assert (header.coded_lossless, header.all_lossless) == (1, 1)
        assert header.tx_mode == av1_parse.ONLY_4X4
        assert header.loop_filter.level == (0, 0, 0, 0)

    def test_a_segment_with_its_own_quantiser_is_not_lossless(self):
        sequence = Sequence()
        data, length = frame_header(sequence, base_q_idx=0, segments={(1, 0): 9})
        header = parse_frame_header(data, sequence.parsed(), FrameState())
        assert header.header_bits == length
        assert header.coded_lossless == 0

    def test_a_shown_key_frame_empties_every_slot(self):
        state, parsed = state_with(Sequence(), {}, {'frame_type': INTER_FRAME, 'order_hint': 3})
        assert all(slot.valid for slot in state.slots)
        data, _ = frame_header(Sequence())
        parse_frame_header(data, parsed, state)
        assert not any(slot.valid for slot in state.slots)
        assert all(slot.order_hint == 0 for slot in state.slots)

    def test_an_inter_frame_refreshes_the_slots_it_names(self):
        state, _ = state_with(
            Sequence(), {}, {'frame_type': INTER_FRAME, 'order_hint': 3, 'refresh': 0b0110})
        assert [slot.order_hint for slot in state.slots] == [0, 3, 3, 0, 0, 0, 0, 0]
        assert [slot.frame_type for slot in state.slots][:3] == [KEY_FRAME, INTER_FRAME,
                                                                 INTER_FRAME]

    def test_references_and_their_order_hints(self):
        sequence = Sequence()
        state, parsed = state_with(
            sequence, {},
            {'frame_type': INTER_FRAME, 'order_hint': 8, 'refresh': 0b10, 'show_frame': 0},
            {'frame_type': INTER_FRAME, 'order_hint': 4, 'refresh': 0b100})
        data, length = frame_header(sequence, frame_type=INTER_FRAME, order_hint=6,
                                    refresh=0, refs=(2, 0, 0, 0, 1, 1, 1))
        header = parse_frame_header(data, parsed, state)
        assert header.header_bits == length
        assert header.ref_frame_idx == (2, 0, 0, 0, 1, 1, 1)
        assert header.order_hints == (0, 4, 0, 0, 0, 8, 8, 8)
        assert header.interpolation_filter == av1_parse.EIGHTTAP_SHARP
        assert header.gm_type == (IDENTITY,) * 8
        assert AFFINE not in header.gm_type

    def test_short_signalling_orders_the_references_by_their_hints(self):
        """The set frame refs process (7.8), on slots with hints 0 to 14.

        The frame is at hint 9. Slots 5, 6 and 7 are after it: the latest is
        ALTREF, the nearest BWDREF and the next ALTREF2. LAST2 and LAST3 are
        the latest before it that are not already LAST or GOLDEN.
        """
        sequence = Sequence()
        frames = [{}]
        for slot, hint in enumerate((0, 2, 4, 6, 8, 10, 12, 14)):
            frames.append({'frame_type': INTER_FRAME, 'order_hint': hint,
                           'refresh': 1 << slot, 'show_frame': 0})
        state, parsed = state_with(sequence, *frames)
        data, length = frame_header(sequence, frame_type=INTER_FRAME, order_hint=9,
                                    refresh=0, short_signaling=(4, 1))
        header = parse_frame_header(data, parsed, state)
        assert header.header_bits == length
        assert header.frame_refs_short_signaling == 1
        assert (header.last_frame_idx, header.gold_frame_idx) == (4, 1)
        # LAST, LAST2, LAST3, GOLDEN, BWDREF, ALTREF2, ALTREF
        assert header.ref_frame_idx == (4, 3, 2, 1, 5, 6, 7)

    def test_short_signalling_with_no_later_frames_fills_from_the_earliest(self):
        sequence = Sequence()
        frames = [{}]
        for slot, hint in enumerate((1, 2, 3, 4, 5, 6, 7, 8)):
            frames.append({'frame_type': INTER_FRAME, 'order_hint': hint,
                           'refresh': 1 << slot})
        state, parsed = state_with(sequence, *frames)
        data, _ = frame_header(sequence, frame_type=INTER_FRAME, order_hint=9, refresh=0,
                               short_signaling=(7, 6))
        header = parse_frame_header(data, parsed, state)
        # The five left are the latest forward frames, in turn: 5, 4, 3, 2, 1.
        assert header.ref_frame_idx == (7, 5, 4, 6, 3, 2, 1)

    def test_a_frame_size_taken_from_a_reference(self):
        sequence = Sequence()
        state, parsed = state_with(
            sequence, {'show_frame': 0, 'refresh': 0b1, 'size': (40, 24)},
            {'frame_type': INTRA_ONLY_FRAME, 'refresh': 0b10, 'size': (56, 32)})
        data, length = frame_header(sequence, frame_type=INTER_FRAME, order_hint=2,
                                    refresh=0, refs=(0, 0, 1, 0, 0, 0, 0), found_ref=2)
        header = parse_frame_header(data, parsed, state)
        assert header.header_bits == length
        assert (header.frame_size_override_flag, header.found_ref) == (1, 1)
        assert header.size == (56, 32)
        assert (header.mi_cols, header.mi_rows) == (14, 8)

    def test_a_frame_size_of_its_own(self):
        sequence = Sequence()
        state, parsed = state_with(sequence, {})
        data, length = frame_header(sequence, frame_type=INTER_FRAME, order_hint=2,
                                    refresh=0, size=(30, 20))
        header = parse_frame_header(data, parsed, state)
        assert header.header_bits == length
        assert header.found_ref == 0
        assert header.size == (30, 20)
        assert (header.mi_cols, header.mi_rows) == (8, 6)

    def test_super_resolution_narrows_the_coded_width(self):
        sequence = Sequence(superres=1)
        data, length = frame_header(sequence, superres_denom=16)
        header = parse_frame_header(data, sequence.parsed(), FrameState())
        assert header.header_bits == length
        assert (header.use_superres, header.superres_denom) == (1, 16)
        assert (header.upscaled_width, header.frame_width) == (64, 32)
        assert header.mi_cols == 8
        assert header.size == (64, 48)

    def test_loop_filter_deltas_persist_through_the_primary_reference(self):
        sequence = Sequence()
        state, parsed = state_with(
            sequence, {'ref_deltas': {0: 5, 7: -9}, 'mode_deltas': {1: 3}})
        assert state.slots[2].loop_filter_ref_deltas == (5, 0, 0, 0, -1, 0, -1, -9)
        data, length = frame_header(sequence, frame_type=INTER_FRAME, order_hint=1,
                                    primary_ref_frame=3, refs=(0, 0, 0, 2, 0, 0, 0),
                                    ref_deltas={4: 11})
        header = parse_frame_header(data, parsed, state)
        assert header.header_bits == length
        assert header.loop_filter.delta_update == 1
        assert header.loop_filter.ref_deltas == (5, 0, 0, 0, 11, 0, -1, -9)
        assert header.loop_filter.mode_deltas == (0, 3)

    def test_a_frame_with_no_primary_reference_starts_from_the_defaults(self):
        sequence = Sequence()
        state, parsed = state_with(sequence, {'ref_deltas': {0: 5}})
        data, _ = frame_header(sequence, frame_type=INTER_FRAME, order_hint=1)
        header = parse_frame_header(data, parsed, state)
        assert header.loop_filter.ref_deltas == DEFAULT_REF_DELTAS

    def test_segment_features_are_clipped_and_persist(self):
        sequence = Sequence()
        state, parsed = state_with(
            sequence, {'segments': {(0, 0): -100, (2, 1): 63, (5, 5): 6, (6, 7): 0}})
        saved = state.slots[0]
        assert saved.feature_data[0][0] == -100
        assert saved.feature_data[2][1] == 63
        assert saved.feature_data[5][5] == 6
        assert saved.feature_enabled[6] == (0, 0, 0, 0, 0, 0, 0, 1)
        data, length = frame_header(sequence, frame_type=INTER_FRAME, order_hint=1,
                                    primary_ref_frame=0, segmentation=(0, 0, 0))
        header = parse_frame_header(data, parsed, state)
        assert header.header_bits == length
        segmentation = header.segmentation
        assert (segmentation.enabled, segmentation.update_map, segmentation.update_data) == (
            1, 0, 0)
        assert segmentation.feature_data == saved.feature_data
        assert (segmentation.last_active_seg_id, segmentation.seg_id_pre_skip) == (6, 1)

    def test_a_frame_without_segmentation_saves_none(self):
        sequence = Sequence()
        state, _ = state_with(
            sequence, {'segments': {(0, 0): -100}},
            {'frame_type': INTER_FRAME, 'order_hint': 1, 'primary_ref_frame': 0,
             'refresh': 0b1})
        assert state.slots[0].feature_data[0][0] == 0
        assert state.slots[1].feature_data[0][0] == -100

    def test_film_grain_of_a_new_frame(self):
        sequence = Sequence(film_grain=1)
        data, length = frame_header(sequence, grain='new')
        header = parse_frame_header(data, sequence.parsed(), FrameState())
        assert header.header_bits == length
        grain = header.film_grain
        assert (grain.apply_grain, grain.grain_seed, grain.update_grain) == (1, 0x1234, 1)
        assert (grain.point_y_value, grain.point_y_scaling) == ((40,), (90,))
        assert grain.chroma_scaling_from_luma == 1
        assert (grain.num_cb_points, grain.num_cr_points) == (0, 0)
        assert (grain.grain_scaling_minus_8, grain.ar_coeff_lag) == (2, 1)
        assert grain.ar_coeffs_y == (120, 130, 140, 150)
        assert grain.ar_coeffs_cb == (100, 110, 128, 129, 131)
        assert grain.ar_coeffs_cr == (132, 133, 134, 135, 136)
        assert (grain.ar_coeff_shift_minus_6, grain.grain_scale_shift) == (1, 2)
        assert (grain.overlap_flag, grain.clip_to_restricted_range) == (1, 0)

    def test_film_grain_taken_from_a_reference_keeps_its_own_seed(self):
        sequence = Sequence(film_grain=1)
        state, parsed = state_with(sequence, {'grain': 'new'})
        first = state.slots[3].film_grain
        data, length = frame_header(sequence, frame_type=INTER_FRAME, order_hint=1,
                                    refs=(3,) * 7, grain=3)
        # The writer's seed is the same for every frame, so give the slot another.
        state.slots[3] = dataclasses.replace(
            state.slots[3], film_grain=dataclasses.replace(first, grain_seed=77))
        header = parse_frame_header(data, parsed, state)
        assert header.header_bits == length
        grain = header.film_grain
        assert (grain.apply_grain, grain.update_grain, grain.film_grain_params_ref_idx) == (
            1, 0, 3)
        assert grain.grain_seed == 0x1234
        assert grain.ar_coeffs_y == first.ar_coeffs_y
        assert grain.point_y_scaling == first.point_y_scaling

    def test_monochrome_film_grain_has_no_chroma(self):
        sequence = Sequence(film_grain=1, mono_chrome=1)
        data, length = frame_header(sequence, grain='new', loop_filter=(3, 0))
        header = parse_frame_header(data, sequence.parsed(), FrameState())
        assert header.header_bits == length
        assert header.film_grain.chroma_scaling_from_luma == 0
        assert header.film_grain.ar_coeffs_cb == ()
        assert header.loop_filter.level == (3, 0, 0, 0)

    def test_a_resilient_frame_marks_the_slots_whose_hints_differ(self):
        sequence = Sequence()
        state, parsed = state_with(
            sequence, {}, {'frame_type': INTER_FRAME, 'order_hint': 5, 'refresh': 0b10})
        data, length = frame_header(
            sequence, frame_type=INTER_FRAME, order_hint=6, error_resilient=1, refresh=0,
            ref_order_hints=(0, 5, 9, 0, 0, 0, 0, 0))
        header = parse_frame_header(data, parsed, state)
        assert header.header_bits == length
        assert header.ref_order_hint == (0, 5, 9, 0, 0, 0, 0, 0)
        assert [slot.valid for slot in state.slots] == [True, True, False] + [True] * 5
        assert state.slots[2].order_hint == 9

    def test_showing_an_existing_frame(self):
        sequence = Sequence()
        state, parsed = state_with(
            sequence, {},
            {'frame_type': INTER_FRAME, 'order_hint': 5, 'refresh': 0b10, 'show_frame': 0,
             'size': (40, 30)})
        header = parse_frame_header(show_existing(1), parsed, state)
        assert (header.show_existing_frame, header.frame_to_show_map_idx) == (1, 1)
        assert header.header_bits == 4
        assert (header.frame_type, header.refresh_frame_flags) == (INTER_FRAME, 0)
        assert header.size == (40, 30)
        state.update(header, parsed)
        assert state.slots[0].frame_type == KEY_FRAME

    def test_showing_an_existing_key_frame_refreshes_every_slot(self):
        """7.21: the key frame's state is loaded and saved into all eight slots."""
        sequence = Sequence()
        state, parsed = state_with(
            sequence, {},
            {'frame_type': INTER_FRAME, 'order_hint': 5, 'refresh': 0b1000},
            {'show_frame': 0, 'refresh': 0b100, 'order_hint': 9, 'ref_deltas': {1: 4}})
        header = parse_frame_header(show_existing(2), parsed, state)
        assert (header.frame_type, header.refresh_frame_flags) == (KEY_FRAME, 0xFF)
        state.update(header, parsed)
        assert all(slot is state.slots[2] for slot in state.slots)
        assert state.slots[7].order_hint == 9
        assert state.slots[7].loop_filter_ref_deltas[1] == 4

    def test_a_frame_header_cut_short_is_a_stream_error(self):
        sequence = Sequence()
        data, _ = frame_header(sequence)
        with pytest.raises(StreamError):
            parse_frame_header(data[:3], sequence.parsed(), FrameState())


class TestTileGroups:
    def header(self, name='av1_tiles'):
        return headers(name)[0][0]

    def test_one_tile_is_the_whole_group(self):
        sequence = Sequence()
        data, _ = frame_header(sequence)
        header = parse_frame_header(data, sequence.parsed(), FrameState())
        group = parse_tile_group(bytes(9), header)
        assert (group.tg_start, group.tg_end) == (0, 0)
        assert group.tiles == (av1_parse.Tile(0, 0, 0, 9),)

    def test_a_group_after_a_frame_header_starts_at_the_next_byte(self):
        sequence = Sequence()
        data, _ = frame_header(sequence)
        header = parse_frame_header(data, sequence.parsed(), FrameState())
        start = (header.header_bits + 7) // 8
        group = parse_tile_group(data[:start] + bytes(5), header, start)
        assert group.tiles == (av1_parse.Tile(0, 0, start, 5),)

    def test_each_tile_but_the_last_states_its_size(self):
        """Four tiles with two-byte sizes; the last runs to the end."""
        header = self.header()
        data = (b'\x00'                         # tile_start_and_end_present_flag
                + b'\x02\x00' + b'aaa' + b'\x00\x00' + b'b'
                + b'\x04\x00' + b'ccccc' + b'dddd')
        group = parse_tile_group(data, header)
        assert (group.tg_start, group.tg_end) == (0, 3)
        assert group.tiles == (
            av1_parse.Tile(0, 0, 3, 3), av1_parse.Tile(0, 1, 8, 1),
            av1_parse.Tile(1, 0, 11, 5), av1_parse.Tile(1, 1, 16, 4))
        assert [data[tile.offset:tile.offset + tile.size] for tile in group.tiles] == [
            b'aaa', b'b', b'ccccc', b'dddd']

    def test_a_group_of_some_of_the_tiles(self):
        """tile_start_and_end_present_flag, then tg_start 1 and tg_end 2 in two bits each."""
        header = self.header()
        data = bytes([0b1_01_10_000]) + b'\x01\x00' + b'xx' + b'yyy'
        group = parse_tile_group(data, header)
        assert (group.tg_start, group.tg_end) == (1, 2)
        assert group.tiles == (av1_parse.Tile(0, 1, 3, 2), av1_parse.Tile(1, 0, 5, 3))

    def test_a_tile_that_runs_past_the_end_is_a_stream_error(self):
        header = self.header()
        with pytest.raises(StreamError, match='tile 0'):
            parse_tile_group(b'\x00' + b'\x40\x00' + b'short', header)

    def test_a_group_with_no_bytes_for_its_last_tile_is_a_stream_error(self):
        header = self.header()
        with pytest.raises(StreamError, match='tile 3'):
            parse_tile_group(b'\x00' + (b'\x00\x00' + b'x') * 3, header)

    def test_tiles_named_out_of_order_are_a_stream_error(self):
        header = self.header()
        with pytest.raises(StreamError, match='names tiles'):
            parse_tile_group(bytes([0b1_10_01_000]) + bytes(8), header)

    def test_the_fixtures_tiles_fill_their_groups(self):
        with IVFReader(DATA / 'av1_tile_sizes.ivf') as stream:
            data = stream.sample(0).data
        frame = next(unit for unit in obu.split(data) if unit.kind == obu.FRAME)
        header = self.header('av1_tile_sizes')
        group = parse_tile_group(frame.payload, header, (header.header_bits + 7) // 8)
        assert [(tile.row, tile.column) for tile in group.tiles] == [
            (0, 0), (0, 1), (1, 0), (1, 1)]
        assert group.tiles[-1].offset + group.tiles[-1].size == len(frame.payload)
        # A two-byte size stands before every tile but the last.
        assert [later.offset - earlier.offset - earlier.size
                for earlier, later in zip(group.tiles, group.tiles[1:], strict=False)] == [2, 2, 0]
