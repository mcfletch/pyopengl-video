"""VP9 syntax parsing, checked against ffmpeg's reading of the fixture streams.

``ffmpeg -i X -c copy -bsf:v trace_headers -f null -`` prints every field of
every header as the reference parser reads it. :data:`TRACED` holds what it
printed for the libvpx fixtures in ``tests/data/decode``, so the comparison
runs with no ``ffmpeg``. The headers libvpx does not write, intra-only frames
among them, are written by :mod:`tests.vp9_syntax`.
"""
from pathlib import Path

import pytest

from pyopengl_video.ivf import IVFReader
from pyopengl_video.vaapi.h264 import StreamError
from pyopengl_video.vaapi.vp9_parse import (
    BILINEAR,
    CS_BT_601,
    CS_BT_709,
    CS_RGB,
    CS_UNKNOWN,
    EIGHTTAP,
    EIGHTTAP_SHARP,
    EIGHTTAP_SMOOTH,
    NON_KEY_FRAME,
    SEG_LVL_ALT_L,
    SEG_LVL_ALT_Q,
    SEG_LVL_REF_FRAME,
    SEG_LVL_SKIP,
    SWITCHABLE,
    ColourConfig,
    DecodingState,
    Reference,
    parse_uncompressed_header,
    split_superframe,
)
from tests.vp9_syntax import INTER, frame, show_existing, superframe

DATA = Path(__file__).parent / 'data' / 'decode'

#: By stream, each sample's frames as trace_headers read them. A frame is
#: ('existing', frame_to_show_map_idx), or (frame_type, show_frame,
#: error_resilient_mode, intra_only, refresh_frame_flags, ref_frame_idx,
#: interpolation_filter, loop_filter_level, base_q_idx, tile_cols_log2,
#: tile_rows_log2, header_size_in_bytes, the uncompressed header's bytes).
TRACED = {
    'vp9_altref': [
        [(0, 1, 0, 0, 255, (0, 0, 0), None, 3, 46, 0, 0, 88, 18)],
        [(1, 0, 0, 0, 4, (0, 1, 2), 4, 4, 59, 0, 0, 48, 10), (1, 1, 0, 0, 1, (0, 1, 2), 4, 6, 138, 0, 0, 9, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 8, 138, 0, 0, 6, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 8, 138, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 6, 138, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 14, 138, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 9, 138, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 16, 138, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 14, 138, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 19, 138, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 3, 138, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 7, 138, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 4, (0, 1, 2), 0, 0, 142, 0, 0, 3, 10)],
        [(1, 0, 0, 0, 2, (0, 2, 1), 0, 4, 81, 0, 0, 11, 11), (1, 1, 0, 0, 1, (0, 2, 1), 0, 6, 132, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 2, 1), 4, 10, 130, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 2, 1), 0, 5, 126, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 2, 1), 4, 5, 131, 0, 0, 5, 10)],
        [(1, 1, 0, 0, 1, (0, 2, 1), 4, 7, 125, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 2, 1), 4, 11, 129, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 2, (0, 2, 1), 0, 0, 142, 0, 0, 3, 10)],
        [(0, 1, 0, 0, 255, (0, 0, 0), None, 3, 42, 0, 0, 94, 18)],
        [(1, 0, 0, 0, 4, (0, 1, 2), 1, 4, 48, 0, 0, 73, 11), (1, 1, 0, 0, 1, (0, 1, 2), 4, 9, 141, 0, 0, 15, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 12, 142, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 10, 142, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 11, 142, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 10, 142, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 9, 142, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 10, 126, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 5, 134, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 9, 133, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 9, 133, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 9, 134, 0, 0, 5, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 10, 137, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 9, 136, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 4, (0, 1, 2), 0, 0, 142, 0, 0, 3, 10)],
        [(1, 0, 0, 0, 2, (0, 2, 1), 0, 5, 69, 0, 0, 17, 11), (1, 1, 0, 0, 1, (0, 2, 1), 0, 9, 134, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 2, 1), 0, 5, 128, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 2, 1), 0, 5, 126, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 2, 1), 0, 10, 126, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 2, (0, 2, 1), 0, 0, 142, 0, 0, 3, 10)],
    ],
    'vp9_show_existing': [
        [(0, 1, 0, 0, 255, (0, 0, 0), None, 8, 35, 0, 0, 79, 18)],
        [(1, 0, 0, 0, 8, (0, 1, 2), 0, 6, 43, 0, 0, 16, 11), (1, 0, 0, 0, 16, (0, 1, 3), 0, 8, 91, 0, 0, 10, 11), (1, 0, 0, 0, 32, (0, 1, 4), 0, 9, 100, 0, 0, 3, 11), (1, 1, 0, 0, 1, (0, 1, 5), 4, 9, 124, 0, 0, 4, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 5), 0, 11, 124, 0, 0, 3, 10)],
        [('existing', 5)],
        [(1, 1, 0, 0, 32, (5, 1, 4), 0, 11, 124, 0, 0, 5, 10)],
        [(1, 1, 0, 0, 32, (5, 1, 4), 0, 11, 124, 0, 0, 3, 10)],
        [('existing', 4)],
        [(1, 0, 0, 0, 1, (4, 1, 3), 0, 9, 94, 0, 0, 3, 11), (1, 1, 0, 0, 16, (4, 1, 0), 0, 11, 124, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 16, (4, 1, 0), 0, 11, 124, 0, 0, 3, 10)],
        [('existing', 0)],
        [(1, 1, 0, 0, 1, (0, 1, 3), 0, 7, 124, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 3), 0, 11, 124, 0, 0, 3, 10)],
        [('existing', 3)],
        [(1, 0, 0, 0, 4, (0, 3, 1), 2, 7, 60, 0, 0, 14, 11), (1, 0, 0, 0, 16, (0, 3, 2), 2, 7, 85, 0, 0, 7, 11), (1, 0, 0, 0, 32, (0, 3, 4), 2, 11, 92, 0, 0, 3, 11), (1, 1, 0, 0, 1, (0, 3, 5), 0, 11, 122, 0, 0, 3, 10)],
        [('existing', 5)],
        [(1, 1, 0, 0, 32, (5, 3, 4), 0, 7, 92, 0, 0, 3, 10)],
        [('existing', 4)],
        [(1, 1, 0, 0, 16, (4, 3, 2), 0, 12, 92, 0, 0, 5, 10)],
        [(1, 1, 0, 0, 16, (4, 3, 2), 0, 4, 92, 0, 0, 3, 10)],
        [('existing', 2)],
        [(0, 1, 0, 0, 255, (0, 0, 0), None, 4, 25, 0, 0, 80, 18)],
        [(1, 0, 0, 0, 1, (4, 2, 3), 2, 4, 38, 0, 0, 22, 11), (1, 0, 0, 0, 2, (4, 2, 0), 2, 10, 84, 0, 0, 10, 11), (1, 0, 0, 0, 32, (4, 2, 1), 0, 11, 92, 0, 0, 3, 11), (1, 0, 0, 0, 64, (4, 2, 5), 0, 11, 96, 0, 0, 10, 11), (1, 1, 0, 0, 16, (4, 2, 6), 0, 11, 97, 0, 0, 11, 10)],
        [('existing', 6)],
        [(1, 1, 0, 0, 64, (6, 2, 5), 0, 11, 96, 0, 0, 3, 10)],
        [('existing', 5)],
        [(1, 1, 0, 0, 32, (5, 2, 1), 0, 11, 110, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 32, (5, 2, 1), 0, 11, 96, 0, 0, 3, 10)],
        [('existing', 1)],
        [(1, 0, 0, 0, 16, (1, 2, 0), 0, 11, 84, 0, 0, 5, 11), (1, 0, 0, 0, 32, (1, 2, 4), 0, 11, 84, 0, 0, 3, 11), (1, 1, 0, 0, 2, (1, 2, 5), 0, 11, 94, 0, 0, 6, 10)],
        [('existing', 5)],
        [(1, 1, 0, 0, 32, (5, 2, 4), 2, 11, 93, 0, 0, 3, 10)],
        [('existing', 4)],
        [(1, 1, 0, 0, 16, (4, 2, 0), 2, 11, 93, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 16, (4, 2, 0), 2, 11, 92, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (4, 2, 0), 0, 0, 141, 0, 0, 3, 10)],
        [(1, 0, 0, 0, 2, (4, 0, 2), 0, 7, 12, 0, 0, 16, 11), (1, 0, 0, 0, 8, (4, 0, 1), 0, 7, 36, 0, 0, 7, 11), (1, 1, 0, 0, 16, (4, 0, 3), 2, 11, 97, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 16, (4, 0, 3), 2, 11, 96, 0, 0, 3, 10)],
        [('existing', 3)],
        [(1, 1, 0, 0, 8, (3, 0, 1), 2, 12, 96, 0, 0, 3, 10)],
        [('existing', 1)],
    ],
    'vp9_tiles': [
        [(0, 1, 0, 0, 255, (0, 0, 0), None, 1, 33, 1, 1, 172, 18)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 9, 74, 1, 1, 35, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 10, 171, 1, 1, 9, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 27, 192, 1, 1, 5, 10)],
        [(0, 1, 0, 0, 255, (0, 0, 0), None, 3, 33, 1, 1, 170, 18)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 11, 201, 1, 1, 9, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 19, 196, 1, 1, 6, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 23, 195, 1, 1, 3, 10)],
    ],
    'vp9_profile2': [
        [(0, 1, 0, 0, 255, (0, 0, 0), None, 4, 41, 0, 0, 69, 18)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 6, 82, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 16, 109, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 9, 89, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 16, 88, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 16, 75, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 3, (0, 1, 2), 0, 6, 27, 0, 0, 15, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 7, 67, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 7, 62, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 7, 57, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 6, 53, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 4, 53, 0, 0, 3, 10)],
        [(0, 1, 0, 0, 255, (0, 0, 0), None, 4, 27, 0, 0, 83, 18)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 4, 50, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 6, 45, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 8, 42, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 6, 45, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 6, 41, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 3, (0, 1, 2), 0, 6, 14, 0, 0, 20, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 6, 38, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 6, 35, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 6, 38, 0, 0, 10, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 6, 43, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 6, 43, 0, 0, 3, 10)],
    ],
    'vp9_segmented': [
        [(0, 1, 0, 0, 255, (0, 0, 0), None, 0, 33, 0, 0, 91, 33)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 5, 74, 0, 0, 16, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 19, 152, 0, 0, 6, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 10, 147, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 11, 140, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 18, 143, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 10, 141, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 17, 143, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 13, 142, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 13, 144, 0, 0, 3, 10)],
        [(0, 1, 0, 0, 255, (0, 0, 0), None, 3, 33, 0, 0, 97, 33)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 7, 143, 0, 0, 9, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 10, 144, 0, 0, 5, 11)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 11, 146, 0, 0, 3, 11)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 18, 142, 0, 0, 3, 11)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 20, 144, 0, 0, 5, 11)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 10, 145, 0, 0, 3, 11)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 12, 141, 0, 0, 3, 11)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 4, 10, 147, 0, 0, 3, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 16, 147, 0, 0, 3, 11)],
    ],
    'vp9_lossless': [
        [(0, 1, 0, 0, 255, (0, 0, 0), None, 0, 0, 0, 0, 88, 18)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 0, 0, 0, 0, 51, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 0, 0, 0, 0, 28, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 0, 0, 0, 0, 0, 29, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 1, 0, 0, 0, 0, 27, 10)],
        [(0, 1, 0, 0, 255, (0, 0, 0), None, 0, 0, 0, 0, 85, 18)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 1, 0, 0, 0, 0, 58, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 1, 0, 0, 0, 0, 38, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 2, 0, 0, 0, 0, 21, 10)],
        [(1, 1, 0, 0, 1, (0, 1, 2), 2, 0, 0, 0, 0, 18, 10)],
    ],
    'vp9_resilient': [
        [(0, 1, 1, 0, 255, (0, 0, 0), None, 3, 33, 0, 0, 101, 18)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 0, 10, 74, 0, 0, 3, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 4, 19, 132, 0, 0, 3, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 4, 18, 129, 0, 0, 7, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 4, 17, 126, 0, 0, 5, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 4, 17, 124, 0, 0, 3, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 0, 18, 128, 0, 0, 3, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 4, 16, 122, 0, 0, 3, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 4, 17, 123, 0, 0, 3, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 0, 16, 121, 0, 0, 6, 13)],
        [(0, 1, 1, 0, 255, (0, 0, 0), None, 3, 33, 0, 0, 105, 18)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 0, 16, 118, 0, 0, 6, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 0, 16, 121, 0, 0, 10, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 0, 17, 125, 0, 0, 3, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 4, 15, 116, 0, 0, 3, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 4, 17, 124, 0, 0, 3, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 0, 18, 131, 0, 0, 5, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 4, 15, 116, 0, 0, 7, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 0, 17, 124, 0, 0, 9, 13)],
        [(1, 1, 1, 0, 1, (0, 1, 2), 4, 18, 130, 0, 0, 6, 13)],
    ],
}


def samples(name):
    with IVFReader(DATA / f'{name}.ivf') as stream:
        return [stream.sample(index).data for index in range(len(stream))]


def as_traced(header):
    if header.show_existing_frame:
        return ('existing', header.frame_to_show_map_idx)
    inter = header.frame_type == NON_KEY_FRAME and not header.intra_only
    return (header.frame_type, header.show_frame, header.error_resilient_mode,
            header.intra_only, header.refresh_frame_flags, header.ref_frame_idx,
            header.interpolation_filter if inter else None,
            header.loop_filter.level, header.quantisation.base_q_idx,
            header.tile_cols_log2, header.tile_rows_log2,
            header.header_size_in_bytes, header.uncompressed_size)


def parse_stream(name):
    """Every header of a fixture, grouped by sample."""
    state = DecodingState()
    return [[parse_uncompressed_header(data, state) for data in split_superframe(sample)]
            for sample in samples(name)]


class TestFixtureStreams:
    @pytest.mark.parametrize('name', sorted(TRACED))
    def test_every_header_reads_as_the_reference_parser_read_it(self, name):
        found = [[as_traced(header) for header in sample] for sample in parse_stream(name)]
        assert found == TRACED[name]

    def test_the_streams_cover_hidden_frames_and_repeated_ones(self):
        hidden = [row for sample in TRACED['vp9_altref'] for row in sample if row[1] == 0]
        repeated = [row for sample in TRACED['vp9_show_existing'] for row in sample
                    if row[0] == 'existing']
        assert len(hidden) == 4
        assert len(repeated) == 14

    def test_a_key_frame_states_its_size_and_colour(self):
        header = parse_stream('vp9_altref')[0][0]
        assert header.key_frame
        assert header.profile == 0
        assert header.size == (122, 70)
        assert (header.render_width, header.render_height) == (122, 70)
        assert header.colour == ColourConfig(8, CS_UNKNOWN, 0, 1, 1)

    def test_a_profile_2_key_frame_states_ten_bits(self):
        header = parse_stream('vp9_profile2')[0][0]
        assert header.profile == 2
        assert header.size == (90, 50)
        assert header.colour == ColourConfig(10, CS_UNKNOWN, 0, 1, 1)

    def test_an_inter_frame_takes_its_size_and_colour_from_before(self):
        header = parse_stream('vp9_profile2')[1][0]
        assert header.frame_type == NON_KEY_FRAME
        assert header.size == (90, 50)
        assert header.colour.bit_depth == 10

    def test_the_loop_filter_deltas_of_a_key_frame(self):
        header = parse_stream('vp9_segmented')[0][0]
        assert header.loop_filter.delta_enabled == 1
        assert header.loop_filter.delta_update == 1
        assert header.loop_filter.ref_deltas == (1, 0, -1, -1)
        assert header.loop_filter.mode_deltas == (0, 0)

    def test_segmentation_as_a_key_frame_states_it(self):
        segmentation = parse_stream('vp9_segmented')[0][0].segmentation
        assert segmentation.enabled == 1
        assert segmentation.update_map == 1
        assert segmentation.temporal_update == 0
        assert segmentation.update_data == 1
        assert segmentation.abs_or_delta_update == 0
        assert segmentation.tree_probs == (50, 166, 255, 163, 85, 255, 128)
        assert segmentation.pred_probs == (255, 255, 255)
        assert [row[SEG_LVL_ALT_Q] for row in segmentation.feature_enabled] == [
            1, 1, 1, 0, 1, 0, 0, 0]
        assert [row[SEG_LVL_ALT_Q] for row in segmentation.feature_data] == [
            -24, -20, -13, 0, 14, 0, 0, 0]
        assert not any(row[feature] for row in segmentation.feature_enabled
                       for feature in (SEG_LVL_ALT_L, SEG_LVL_REF_FRAME, SEG_LVL_SKIP))

    def test_segmentation_features_carry_over_to_a_frame_that_states_none(self):
        headers = parse_stream('vp9_segmented')
        later = headers[5][0].segmentation
        assert (later.enabled, later.update_map, later.update_data) == (1, 0, 0)
        assert later.feature_data == headers[0][0].segmentation.feature_data
        assert later.tree_probs == headers[0][0].segmentation.tree_probs

    def test_lossless_frames_are_told_by_their_quantiser(self):
        assert all(header.quantisation.lossless
                   for sample in parse_stream('vp9_lossless') for header in sample)
        assert not parse_stream('vp9_altref')[0][0].quantisation.lossless

    def test_an_error_resilient_frame_reads_as_frame_parallel(self):
        header = parse_stream('vp9_resilient')[1][0]
        assert header.error_resilient_mode == 1
        assert header.refresh_frame_context == 0
        assert header.frame_parallel_decoding_mode == 1
        assert header.reset_frame_context == 0
        assert header.interpolation_filter == EIGHTTAP
        assert header.allow_high_precision_mv == 1


class TestSuperframes:
    def test_a_plain_frame_is_one_frame(self):
        data = frame()
        assert split_superframe(data) == [data]

    def test_no_bytes_hold_no_frames(self):
        assert split_superframe(b'') == []

    def test_an_index_splits_the_sample_into_its_frames(self):
        hidden, shown = frame(show_frame=0, frame_type=INTER, intra_only=1), frame()
        assert split_superframe(superframe(hidden, shown)) == [hidden, shown]

    @pytest.mark.parametrize('size_bytes', [1, 2, 3, 4])
    def test_frame_sizes_take_one_to_four_bytes(self, size_bytes):
        frames = [frame(tiles=count) for count in (3, 40, 5)]
        assert split_superframe(superframe(*frames, size_bytes=size_bytes)) == frames

    def test_a_superframe_of_one_frame(self):
        data = frame()
        assert split_superframe(superframe(data)) == [data]

    def test_a_frame_of_no_bytes_is_left_out(self):
        data = frame()
        assert split_superframe(superframe(b'', data)) == [data]

    def test_a_final_byte_the_opening_byte_does_not_match_is_frame_data(self):
        data = frame() + bytes([0x00, 0x10, 0x00, 0xC9])
        assert split_superframe(data) == [data]

    def test_a_marker_with_no_room_for_its_index_is_frame_data(self):
        assert split_superframe(b'\xc9') == [b'\xc9']

    def test_sizes_past_the_end_of_the_sample_are_refused(self):
        data = b'\x00' * 10 + bytes([0xC8]) + (200).to_bytes(2, 'little') + bytes([0xC8])
        with pytest.raises(StreamError, match='superframe index lists 200 bytes'):
            split_superframe(data)

    def test_the_hidden_and_shown_frames_of_a_libvpx_superframe(self):
        """Sizes as trace_headers printed them for the second sample."""
        sample = samples('vp9_altref')[1]
        assert len(sample) == 1779
        assert [len(data) for data in split_superframe(sample)] == [1428, 345]

    def test_a_libvpx_superframe_of_four_frames(self):
        sample = samples('vp9_show_existing')[1]
        assert [len(data) for data in split_superframe(sample)] == [392, 186, 182, 172]

    def test_a_repeated_frame_is_a_sample_of_one_byte(self):
        sample = samples('vp9_show_existing')[3]
        assert split_superframe(sample) == [sample]
        assert len(sample) == 1


def parse(data, state=None):
    return parse_uncompressed_header(data, state if state is not None else DecodingState())


def decoded(*frames):
    """The state after `frames`, and the header of the last."""
    state = DecodingState()
    header = None
    for data in frames:
        header = parse_uncompressed_header(data, state)
    return state, header


class TestFrameTypes:
    def test_a_key_frame_replaces_every_reference(self):
        state, header = decoded(frame(size=(66, 38)))
        assert header.key_frame and header.intra
        assert header.refresh_frame_flags == 0xFF
        assert state.references == [Reference(66, 38, header.colour)] * 8

    def test_a_frame_shown_again_names_its_slot_and_changes_nothing(self):
        state, _ = decoded(frame())
        before = list(state.references)
        header = parse(show_existing(5), state)
        assert header.show_existing_frame == 1
        assert header.frame_to_show_map_idx == 5
        assert header.show_frame == 1
        assert header.refresh_frame_flags == 0
        assert not header.key_frame
        assert state.references == before

    def test_an_inter_frame_names_its_references(self):
        _, header = decoded(frame(), frame(
            frame_type=INTER, refresh=0b100, refs=(3, 1, 7), sign_bias=(0, 0, 1)))
        assert not header.intra
        assert header.refresh_frame_flags == 0b100
        assert header.ref_frame_idx == (3, 1, 7)
        assert header.ref_frame_sign_bias == (0, 0, 1)

    def test_a_profile_0_intra_only_frame_is_8_bit_bt601(self):
        state, header = decoded(frame(color_space=CS_BT_709), frame(
            frame_type=INTER, show_frame=0, intra_only=1, refresh=0b11, size=(32, 24)))
        assert header.intra_only == 1 and header.intra and not header.key_frame
        assert header.colour == ColourConfig(8, CS_BT_601, 0, 1, 1)
        assert header.size == (32, 24)
        assert state.references[0] == state.references[1] == Reference(32, 24, header.colour)
        assert state.references[2].width == 64

    def test_a_profile_2_intra_only_frame_states_its_colour(self):
        _, header = decoded(frame(
            profile=2, depth=10, frame_type=INTER, show_frame=0, intra_only=1,
            color_space=CS_BT_709, color_range=1, refresh=1))
        assert header.colour == ColourConfig(10, CS_BT_709, 1, 1, 1)

    def test_the_context_reset_is_read_unless_error_resilient(self):
        _, header = decoded(frame(), frame(frame_type=INTER, reset_frame_context=3))
        assert header.reset_frame_context == 3
        _, header = decoded(frame(), frame(
            frame_type=INTER, error_resilient=1, reset_frame_context=3))
        assert header.reset_frame_context == 0

    def test_an_intra_frame_decodes_with_context_0(self):
        _, key = decoded(frame(context=2))
        assert key.frame_context_idx == 2
        assert key.context_index == 0
        _, inter = decoded(frame(), frame(frame_type=INTER, context=3))
        assert inter.context_index == 3
        _, resilient = decoded(frame(), frame(frame_type=INTER, error_resilient=1, context=3))
        assert resilient.context_index == 0

    @pytest.mark.parametrize(('raw', 'expected'), [
        (0, EIGHTTAP_SMOOTH), (1, EIGHTTAP), (2, EIGHTTAP_SHARP), (3, BILINEAR),
        (None, SWITCHABLE)])
    def test_the_interpolation_filter(self, raw, expected):
        _, header = decoded(frame(), frame(frame_type=INTER, filter=raw))
        assert header.interpolation_filter == expected


class TestColour:
    def test_twelve_bits(self):
        assert parse(frame(profile=2, depth=12)).colour.bit_depth == 12

    def test_profile_1_states_its_subsampling(self):
        header = parse(frame(profile=1, subsampling=(1, 0)))
        assert (header.colour.subsampling_x, header.colour.subsampling_y) == (1, 0)

    def test_profile_3_rgb_is_full_range_444(self):
        header = parse(frame(profile=3, depth=10, color_space=CS_RGB))
        assert header.colour == ColourConfig(10, CS_RGB, 1, 0, 0)

    def test_rgb_in_profile_0_is_refused(self):
        with pytest.raises(StreamError, match='RGB'):
            parse(frame(color_space=CS_RGB))

    def test_420_in_profile_1_is_refused(self):
        with pytest.raises(StreamError, match='4:2:0'):
            parse(frame(profile=1, subsampling=(1, 1)))


class TestFrameSize:
    def test_a_render_size_apart_from_the_frame_size(self):
        header = parse(frame(size=(66, 38), render=(132, 76)))
        assert header.size == (66, 38)
        assert (header.render_width, header.render_height) == (132, 76)

    def test_an_inter_frame_takes_the_size_of_the_reference_it_names(self):
        state, _ = decoded(frame(size=(100, 60)))
        state.references[4] = Reference(50, 30, state.colour)
        header = parse(frame(frame_type=INTER, refs=(0, 4, 0), found_ref=1, refresh=0), state)
        assert header.size == (50, 30)

    def test_an_inter_frame_may_state_its_own_size(self):
        _, header = decoded(frame(size=(100, 60)), frame(
            frame_type=INTER, found_ref=None, size=(80, 48), refresh=1))
        assert header.size == (80, 48)

    def test_a_frame_too_small_for_every_reference_is_refused(self):
        with pytest.raises(StreamError, match='less than half the size'):
            decoded(frame(size=(100, 60)),
                    frame(frame_type=INTER, found_ref=None, size=(40, 20)))

    def test_a_reference_slot_holding_nothing_is_refused(self):
        with pytest.raises(StreamError, match='slot 6, which holds no frame'):
            parse(frame(frame_type=INTER, refs=(0, 6, 2)), _only_slot_0())

    def test_a_reference_of_another_bit_depth_is_refused(self):
        state, _ = decoded(frame(profile=2, depth=10))
        state.references[1] = Reference(64, 48, ColourConfig(8))
        with pytest.raises(StreamError, match='slot 1, whose samples'):
            parse(frame(profile=2, frame_type=INTER), state)


def _only_slot_0():
    state, _ = decoded(frame())
    state.references[1:] = [None] * 7
    state.references[2] = state.references[0]
    return state


class TestLoopFilter:
    def test_level_and_sharpness(self):
        header = parse(frame(level=37, sharpness=5))
        assert (header.loop_filter.level, header.loop_filter.sharpness) == (37, 5)

    def test_deltas_start_from_the_defaults(self):
        header = parse(frame())
        assert header.loop_filter.delta_enabled == 0
        assert header.loop_filter.ref_deltas == (1, 0, -1, -1)
        assert header.loop_filter.mode_deltas == (0, 0)

    def test_only_the_deltas_stated_change(self):
        header = parse(frame(deltas=({1: -5, 3: 63}, {1: -63})))
        assert header.loop_filter.delta_update == 1
        assert header.loop_filter.ref_deltas == (1, -5, -1, 63)
        assert header.loop_filter.mode_deltas == (0, -63)

    def test_deltas_carry_over_to_the_next_inter_frame(self):
        _, header = decoded(frame(deltas=({0: 4}, {0: 2})),
                            frame(frame_type=INTER, deltas=({}, {})))
        assert header.loop_filter.delta_enabled == 1
        assert header.loop_filter.delta_update == 0
        assert header.loop_filter.ref_deltas == (4, 0, -1, -1)
        assert header.loop_filter.mode_deltas == (2, 0)

    @pytest.mark.parametrize('independent', [
        {'frame_type': 0},
        {'frame_type': 1, 'error_resilient': 1},
        {'frame_type': 1, 'show_frame': 0, 'intra_only': 1},
    ])
    def test_deltas_return_to_the_defaults_at_an_independent_frame(self, independent):
        _, header = decoded(frame(deltas=({0: 4, 2: 9}, {1: 2})), frame(**independent))
        assert header.loop_filter.ref_deltas == (1, 0, -1, -1)
        assert header.loop_filter.mode_deltas == (0, 0)


class TestQuantisation:
    def test_the_base_index_and_its_deltas(self):
        quantisation = parse(frame(base_q=200, q_deltas=(-15, 3, 0))).quantisation
        assert quantisation.base_q_idx == 200
        assert quantisation.delta_q_y_dc == -15
        assert quantisation.delta_q_uv_dc == 3
        assert quantisation.delta_q_uv_ac == 0
        assert not quantisation.lossless

    def test_lossless_is_a_zero_index_with_no_deltas(self):
        assert parse(frame(base_q=0)).quantisation.lossless
        assert not parse(frame(base_q=0, q_deltas=(0, 0, 1))).quantisation.lossless


#: One feature of each kind, at the ends of its range.
FEATURES = {(0, SEG_LVL_ALT_Q): -255, (1, SEG_LVL_ALT_L): 63,
            (2, SEG_LVL_REF_FRAME): 3, (7, SEG_LVL_SKIP): 0}


class TestSegmentation:
    def test_a_frame_without_it(self):
        segmentation = parse(frame()).segmentation
        assert segmentation.enabled == 0
        assert not segmentation.active(0, SEG_LVL_ALT_Q)

    def test_features_of_every_kind(self):
        segmentation = parse(frame(segmentation={'features': FEATURES, 'abs': 1})).segmentation
        assert segmentation.update_data == 1
        assert segmentation.abs_or_delta_update == 1
        assert segmentation.feature_data[0][SEG_LVL_ALT_Q] == -255
        assert segmentation.feature_data[1][SEG_LVL_ALT_L] == 63
        assert segmentation.feature_data[2][SEG_LVL_REF_FRAME] == 3
        assert segmentation.active(7, SEG_LVL_SKIP)
        assert not segmentation.active(6, SEG_LVL_SKIP)
        assert sum(map(sum, segmentation.feature_enabled)) == 4

    def test_the_map_probabilities_with_a_temporal_update(self):
        segmentation = parse(frame(segmentation={
            'tree': (1, None, 3, 4, None, 6, 7), 'pred': (9, None, 11)})).segmentation
        assert segmentation.update_map == 1
        assert segmentation.temporal_update == 1
        assert segmentation.tree_probs == (1, 255, 3, 4, 255, 6, 7)
        assert segmentation.pred_probs == (9, 255, 11)

    def test_the_map_probabilities_without_one(self):
        segmentation = parse(frame(segmentation={'tree': (1, 2, 3, 4, 5, 6, 7)})).segmentation
        assert segmentation.temporal_update == 0
        assert segmentation.pred_probs == (255, 255, 255)

    def test_features_carry_over_and_survive_a_frame_without_segmentation(self):
        _, header = decoded(
            frame(segmentation={'features': FEATURES, 'abs': 1}),
            frame(frame_type=INTER),
            frame(frame_type=INTER, segmentation={}))
        segmentation = header.segmentation
        assert (segmentation.enabled, segmentation.update_data) == (1, 0)
        assert segmentation.abs_or_delta_update == 1
        assert segmentation.feature_data[0][SEG_LVL_ALT_Q] == -255
        assert segmentation.active(7, SEG_LVL_SKIP)

    def test_an_update_replaces_every_feature(self):
        _, header = decoded(
            frame(segmentation={'features': FEATURES}),
            frame(frame_type=INTER, segmentation={'features': {(3, SEG_LVL_ALT_Q): 9}}))
        assert header.segmentation.feature_data[3][SEG_LVL_ALT_Q] == 9
        assert sum(map(sum, header.segmentation.feature_enabled)) == 1

    def test_an_independent_frame_clears_the_features(self):
        _, header = decoded(
            frame(segmentation={'features': FEATURES, 'abs': 1}),
            frame(frame_type=INTER, error_resilient=1, segmentation={}))
        assert header.segmentation.enabled == 1
        assert header.segmentation.abs_or_delta_update == 0
        assert not any(map(any, header.segmentation.feature_enabled))


class TestTiles:
    @pytest.mark.parametrize(('width', 'bits', 'columns'), [
        (64, (), 0),                # too narrow for two tiles: no bit is read
        (448, (), 0),
        (449, (1,), 1),             # eight superblocks: one increment at most
        (449, (0,), 0),
        (1920, (1, 1), 2),
        (1920, (1, 0), 1),
        (4096, (1, 1, 1, 1), 4),
        (4097, (0,), 1),            # wider than 64 superblocks: two tiles at least
        (4097, (1, 1, 0), 3),
    ])
    def test_tile_columns_by_frame_width(self, width, bits, columns):
        header = parse(frame(size=(width, 64), tile_columns=bits))
        assert header.tile_cols_log2 == columns

    @pytest.mark.parametrize('rows', [0, 1, 2])
    def test_tile_rows(self, rows):
        assert parse(frame(tile_rows=rows)).tile_rows_log2 == rows


class TestHeaderExtent:
    def test_the_two_header_lengths(self):
        data = frame(compressed=37)
        header = parse(data)
        assert header.header_size_in_bytes == 37
        assert header.uncompressed_size == 14
        assert data[header.uncompressed_size:][:3] == bytes([1, 2, 3])

    def test_a_frame_that_does_not_open_with_the_marker_is_refused(self):
        with pytest.raises(StreamError, match='frame marker'):
            parse(frame(marker=1))

    def test_a_wrong_sync_code_is_refused(self):
        with pytest.raises(StreamError, match='sync code is 498343'):
            parse(frame(sync=(0x49, 0x83, 0x43)))

    def test_a_compressed_header_of_no_bytes_is_refused(self):
        with pytest.raises(StreamError, match='compressed header of no bytes'):
            parse(frame(compressed=0))

    def test_a_frame_shorter_than_its_headers_is_refused(self):
        with pytest.raises(StreamError, match='headers alone take'):
            parse(frame(compressed=40, tiles=0)[:-1])

    def test_a_frame_cut_inside_its_header_is_refused(self):
        with pytest.raises(StreamError, match='header ends'):
            parse(frame()[:6])
