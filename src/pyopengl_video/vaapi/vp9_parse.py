"""VP9 syntax: the superframe index and the uncompressed header of a frame.

A VP9 frame opens with an *uncompressed header* of plain bit fields: what the
frame is, its size, which of the eight reference slots it predicts from and
which it replaces, the loop filter, the quantiser, segmentation and the tiles.
Everything after it is entropy coded and is the driver's to read. Several
frames may travel as one sample, a *superframe*, with an index of their sizes
at the end.

Some of a header's values are stated only when they change. The loop filter
deltas and the segmentation features carry over from the frame before, a
frame may take its size from a reference slot, and an intra or error
resilient frame sets all of it back to the defaults. :class:`DecodingState`
is what carries over, and :func:`parse_uncompressed_header` reads one frame's
header against it.

The bytes read are a file's. A header that states something outside what the
specification allows raises :class:`~pyopengl_video.vaapi.h264.StreamError`:
a profile its samples are not of, more than 64 tile columns, a tile that runs
past the frame.

The section numbers are those of the VP9 Bitstream and Decoding Process
Specification, version 0.6.
"""
from __future__ import annotations

import dataclasses

from pyopengl_video.colour import (
    MATRIX_BT601,
    MATRIX_BT709,
    MATRIX_BT2020,
    MATRIX_SMPTE240,
    MATRIX_UNSPECIFIED,
    Colour,
)
from pyopengl_video.vaapi.bits import BitReader
from pyopengl_video.vaapi.h264 import StreamError

__all__ = [
    'ALTREF_FRAME', 'BILINEAR', 'CS_BT_2020', 'CS_BT_601', 'CS_BT_709',
    'CS_RESERVED', 'CS_RGB', 'CS_SMPTE_170', 'CS_SMPTE_240', 'CS_UNKNOWN',
    'ColourConfig', 'DecodingState', 'EIGHTTAP', 'EIGHTTAP_SHARP',
    'EIGHTTAP_SMOOTH', 'FrameHeader', 'GOLDEN_FRAME', 'INTRA_FRAME',
    'KEY_FRAME', 'LAST_FRAME', 'LoopFilter', 'MAX_SEGMENTS', 'NON_KEY_FRAME',
    'NUM_REF_FRAMES', 'Quantisation', 'Reference', 'SEG_LVL_ALT_L',
    'SEG_LVL_ALT_Q', 'SEG_LVL_MAX', 'SEG_LVL_REF_FRAME', 'SEG_LVL_SKIP',
    'SWITCHABLE', 'Segmentation', 'colour_of', 'is_key_frame',
    'parse_uncompressed_header',
    'split_superframe',
]

FRAME_MARKER = 2
SYNC_CODE = (0x49, 0x83, 0x42)

KEY_FRAME = 0
NON_KEY_FRAME = 1

NUM_REF_FRAMES = 8
#: How many of the reference slots an inter frame predicts from.
REFS_PER_FRAME = 3

#: What a block predicts from, which is also the index of a loop filter delta.
INTRA_FRAME = 0
LAST_FRAME = 1
GOLDEN_FRAME = 2
ALTREF_FRAME = 3

#: color_space values (7.2.2).
CS_UNKNOWN = 0
CS_BT_601 = 1
CS_BT_709 = 2
CS_SMPTE_170 = 3
CS_SMPTE_240 = 4
CS_BT_2020 = 5
CS_RESERVED = 6
CS_RGB = 7

#: interpolation_filter values.
EIGHTTAP = 0
EIGHTTAP_SMOOTH = 1
EIGHTTAP_SHARP = 2
BILINEAR = 3
SWITCHABLE = 4
#: The filter each raw_interpolation_filter value names (6.2.7).
LITERAL_TO_TYPE = (EIGHTTAP_SMOOTH, EIGHTTAP, EIGHTTAP_SHARP, BILINEAR)

MAX_SEGMENTS = 8
#: The features a segment may carry: a quantiser, a loop filter level, the
#: reference frame its blocks predict from, and skipping the residual.
SEG_LVL_ALT_Q = 0
SEG_LVL_ALT_L = 1
SEG_LVL_REF_FRAME = 2
SEG_LVL_SKIP = 3
SEG_LVL_MAX = 4
#: How many bits each feature's value takes, and whether a sign follows (6.2.11).
SEGMENTATION_FEATURE_BITS = (8, 6, 2, 0)
SEGMENTATION_FEATURE_SIGNED = (1, 1, 0, 0)

#: The loop filter deltas an intra or error resilient frame starts from (7.2).
DEFAULT_REF_DELTAS = (1, 0, -1, -1)
DEFAULT_MODE_DELTAS = (0, 0)

#: A tile is at least this many 64x64 superblocks wide, and at most that many.
MIN_TILE_WIDTH_B64 = 4
MAX_TILE_WIDTH_B64 = 64

#: The largest tile_cols_log2 (7.2.11): a frame has at most 64 tile columns.
MAX_TILE_COLS_LOG2 = 6

#: How many bytes the size of a tile is stated in (6.4).
TILE_SIZE_BYTES = 4

_NO_FEATURES = ((0,) * SEG_LVL_MAX,) * MAX_SEGMENTS


def split_superframe(data: bytes) -> list[bytes]:
    """The frames of one sample, which is a frame or a superframe (Annex B).

    A superframe ends with an index: a marker byte, the size of each frame,
    and the marker byte again. A sample without one is a single frame. An
    index whose sizes run past the sample is a
    :class:`~pyopengl_video.vaapi.h264.StreamError`.
    """
    if not data:
        return []
    marker = data[-1]
    if marker & 0xE0 != 0xC0:
        return [data]
    count = (marker & 0x07) + 1
    size_bytes = ((marker >> 3) & 0x03) + 1
    index_size = 2 + count * size_bytes
    if len(data) < index_size or data[-index_size] != marker:
        return [data]
    start = len(data) - index_size + 1
    sizes = [int.from_bytes(data[start + n * size_bytes:start + (n + 1) * size_bytes],
                            'little') for n in range(count)]
    if sum(sizes) > len(data) - index_size:
        raise StreamError(
            f'the superframe index lists {sum(sizes)} bytes of frames and the '
            f'sample holds {len(data) - index_size}')
    frames = []
    position = 0
    for size in sizes:
        if size:
            frames.append(data[position:position + size])
        position += size
    return frames


@dataclasses.dataclass(frozen=True)
class ColourConfig:
    """How a frame's samples are stored and what colours they are (6.2.2).

    bit_depth -- bits a sample: 8, 10 or 12
    color_space -- a ``CS_*`` value
    color_range -- 1 when the samples span the full range
    subsampling_x/subsampling_y -- 1 where chroma has half the luma's samples
    """

    bit_depth: int = 8
    color_space: int = CS_UNKNOWN
    color_range: int = 0
    subsampling_x: int = 1
    subsampling_y: int = 1

    @property
    def layout(self) -> tuple[int, int, int]:
        """What a reference has to share with a frame predicted from it."""
        return (self.bit_depth, self.subsampling_x, self.subsampling_y)


@dataclasses.dataclass(frozen=True)
class LoopFilter:
    """The loop filter parameters in force for a frame (6.2.8).

    ref_deltas -- the level adjustment by reference frame, ``INTRA_FRAME`` first
    mode_deltas -- the adjustment by prediction mode: zero motion, then the rest
    """

    level: int = 0
    sharpness: int = 0
    delta_enabled: int = 0
    delta_update: int = 0
    ref_deltas: tuple[int, ...] = DEFAULT_REF_DELTAS
    mode_deltas: tuple[int, ...] = DEFAULT_MODE_DELTAS


@dataclasses.dataclass(frozen=True)
class Quantisation:
    """The quantiser indices of a frame (6.2.9)."""

    base_q_idx: int = 0
    delta_q_y_dc: int = 0
    delta_q_uv_dc: int = 0
    delta_q_uv_ac: int = 0

    @property
    def lossless(self) -> bool:
        """Whether the frame is coded with the lossless transform."""
        return not (self.base_q_idx or self.delta_q_y_dc or self.delta_q_uv_dc
                    or self.delta_q_uv_ac)


@dataclasses.dataclass(frozen=True)
class Segmentation:
    """The segmentation parameters in force for a frame (6.2.11).

    tree_probs/pred_probs -- the probabilities the segment map is coded with
    feature_enabled/feature_data -- by segment, then by ``SEG_LVL_*`` feature
    abs_or_delta_update -- 1 when a feature's value replaces the frame's, 0
        when it is added to it
    """

    enabled: int = 0
    update_map: int = 0
    temporal_update: int = 0
    update_data: int = 0
    abs_or_delta_update: int = 0
    tree_probs: tuple[int, ...] = (255,) * 7
    pred_probs: tuple[int, ...] = (255,) * 3
    feature_enabled: tuple[tuple[int, ...], ...] = _NO_FEATURES
    feature_data: tuple[tuple[int, ...], ...] = _NO_FEATURES

    def active(self, segment: int, feature: int) -> bool:
        """Whether `segment` uses `feature` in this frame."""
        return bool(self.enabled and self.feature_enabled[segment][feature])


@dataclasses.dataclass(frozen=True)
class FrameHeader:
    """One frame's uncompressed header (6.2), with what carried over filled in.

    A header with show_existing_frame set says only which reference slot to
    show again, in frame_to_show_map_idx; its other fields are the defaults.

    width/height -- the frame, in luma samples
    render_width/render_height -- the size the stream asks to be shown at
    refresh_frame_flags -- bit n set when the frame replaces reference slot n
    ref_frame_idx -- the slots of the last, golden and alternate references
    ref_frame_sign_bias -- for each of those, 1 when it lies after the frame
        in display order
    interpolation_filter -- ``EIGHTTAP`` to ``SWITCHABLE``
    frame_context_idx -- the probability context as the header states it
    header_size_in_bytes -- the compressed header's length
    uncompressed_size -- this header's own length in bytes
    """

    profile: int = 0
    show_existing_frame: int = 0
    frame_to_show_map_idx: int = 0
    frame_type: int = NON_KEY_FRAME
    show_frame: int = 1
    error_resilient_mode: int = 0
    intra_only: int = 0
    reset_frame_context: int = 0
    colour: ColourConfig = ColourConfig()
    width: int = 0
    height: int = 0
    render_width: int = 0
    render_height: int = 0
    refresh_frame_flags: int = 0
    ref_frame_idx: tuple[int, ...] = (0, 0, 0)
    ref_frame_sign_bias: tuple[int, ...] = (0, 0, 0)
    allow_high_precision_mv: int = 0
    interpolation_filter: int = EIGHTTAP
    refresh_frame_context: int = 0
    frame_parallel_decoding_mode: int = 0
    frame_context_idx: int = 0
    loop_filter: LoopFilter = LoopFilter()
    quantisation: Quantisation = Quantisation()
    segmentation: Segmentation = Segmentation()
    tile_cols_log2: int = 0
    tile_rows_log2: int = 0
    header_size_in_bytes: int = 0
    uncompressed_size: int = 0

    @property
    def size(self) -> tuple[int, int]:
        """The frame, as (width, height)."""
        return (self.width, self.height)

    @property
    def key_frame(self) -> bool:
        return not self.show_existing_frame and self.frame_type == KEY_FRAME

    @property
    def intra(self) -> bool:
        """Whether the frame predicts from no other: a key or intra-only frame."""
        return self.key_frame or bool(self.intra_only)

    @property
    def context_index(self) -> int:
        """The probability context the frame is decoded with.

        An intra or error resilient frame uses context 0 whatever its header
        states (6.2).
        """
        if self.intra or self.error_resilient_mode:
            return 0
        return self.frame_context_idx


#: The H.273 matrix, primaries and transfer each VP9 colour space names. A
#: colour space not listed states none of them.
COLOUR_SPACES = {
    CS_BT_601: (MATRIX_BT601, 2, 2),
    CS_BT_709: (MATRIX_BT709, 1, 1),
    CS_SMPTE_170: (MATRIX_BT601, 6, 6),
    CS_SMPTE_240: (MATRIX_SMPTE240, 7, 7),
    CS_BT_2020: (MATRIX_BT2020, 9, 2),
}

#: Chroma samples sit on the luma sample above and left of them, which is
#: where VP9 puts them when nothing outside the stream says otherwise (7.2.2).
CHROMA_COLOCATED = 2


def colour_of(header: FrameHeader) -> Colour:
    """A frame's colour description, the matrix chosen where it states none."""
    matrix, primaries, transfer = COLOUR_SPACES.get(
        header.colour.color_space, (MATRIX_UNSPECIFIED, 2, 2))
    return Colour(matrix=matrix, full_range=bool(header.colour.color_range),
                  primaries=primaries, transfer=transfer,
                  chroma_location=CHROMA_COLOCATED).resolved(header.height)


@dataclasses.dataclass(frozen=True)
class Reference:
    """What a later header takes from the frame in a reference slot."""

    width: int
    height: int
    colour: ColourConfig


class DecodingState:
    """What one frame's uncompressed header takes from the frames before it.

    references -- the frame in each reference slot, None in a slot no frame
        has been put in
    colour -- the sample layout and colour of the frames being decoded, which
        an inter frame does not restate
    ref_deltas/mode_deltas -- the loop filter deltas in force
    feature_enabled/feature_data -- the segmentation features in force
    """

    def __init__(self) -> None:
        self.references: list[Reference | None] = [None] * NUM_REF_FRAMES
        self.colour = ColourConfig()
        self.tree_probs: tuple[int, ...] = (255,) * 7
        self.pred_probs: tuple[int, ...] = (255,) * 3
        self.setup_past_independence()

    def setup_past_independence(self) -> None:
        """Set back what an intra or error resilient frame does not inherit (7.2)."""
        self.ref_deltas = list(DEFAULT_REF_DELTAS)
        self.mode_deltas = list(DEFAULT_MODE_DELTAS)
        self.feature_enabled = [[0] * SEG_LVL_MAX for _ in range(MAX_SEGMENTS)]
        self.feature_data = [[0] * SEG_LVL_MAX for _ in range(MAX_SEGMENTS)]
        self.abs_or_delta_update = 0

    def refresh(self, header: FrameHeader) -> None:
        """Put `header`'s frame in the reference slots it replaces (8.10)."""
        stored = Reference(header.width, header.height, header.colour)
        for slot in range(NUM_REF_FRAMES):
            if header.refresh_frame_flags >> slot & 1:
                self.references[slot] = stored


def _opening(reader: BitReader) -> tuple[int, int]:
    """The profile and show_existing_frame, which open every header."""
    if reader.f(2) != FRAME_MARKER:
        raise StreamError('the frame does not open with the VP9 frame marker')
    profile = reader.flag() | reader.flag() << 1
    if profile == 3 and reader.flag():
        raise StreamError('a reserved bit of the VP9 frame header is set')
    return profile, reader.flag()


def is_key_frame(data: bytes) -> bool:
    """Whether the frame `data` is a key frame, which needs no frame before it."""
    reader = BitReader(data[:2])
    _, show_existing_frame = _opening(reader)
    return not show_existing_frame and reader.flag() == KEY_FRAME


def parse_uncompressed_header(data: bytes, state: DecodingState) -> FrameHeader:
    """The uncompressed header that opens the frame `data` (6.2).

    `state` gives what the header leaves unstated and is brought up to date:
    the loop filter deltas and segmentation features now in force, and the
    frame in each reference slot it replaces.

    `data` is the whole frame: the sizes of its tiles are walked, and a frame
    whose tiles do not fit it is refused with its header.
    """
    reader = BitReader(data)
    profile, show_existing_frame = _opening(reader)
    if show_existing_frame:
        return FrameHeader(profile=profile, show_existing_frame=1,
                           frame_to_show_map_idx=reader.f(3))
    values: dict[str, object] = {'profile': profile}
    frame_type = values['frame_type'] = reader.flag()
    show_frame = values['show_frame'] = reader.flag()
    error_resilient = values['error_resilient_mode'] = reader.flag()
    intra_only = 0
    if frame_type == KEY_FRAME:
        _sync_code(reader)
        state.colour = _colour_config(reader, profile)
        size = _frame_size(reader)
        render = _render_size(reader, size)
        values['refresh_frame_flags'] = 0xFF
    else:
        if not show_frame:
            intra_only = reader.flag()
        values['intra_only'] = intra_only
        if not error_resilient:
            values['reset_frame_context'] = reader.f(2)
        if intra_only:
            _sync_code(reader)
            # A profile 0 intra-only frame states no colour: it is 8-bit
            # 4:2:0 BT.601.
            state.colour = (_colour_config(reader, profile) if profile > 0
                            else ColourConfig(color_space=CS_BT_601))
            values['refresh_frame_flags'] = reader.f(8)
            size = _frame_size(reader)
            render = _render_size(reader, size)
        else:
            values['refresh_frame_flags'] = reader.f(8)
            slots = []
            biases = []
            for _ in range(REFS_PER_FRAME):
                slots.append(reader.f(3))
                biases.append(reader.flag())
            values['ref_frame_idx'] = tuple(slots)
            values['ref_frame_sign_bias'] = tuple(biases)
            _check_profile(profile, state.colour)
            size = _frame_size_with_refs(reader, state, slots)
            render = _render_size(reader, size)
            values['allow_high_precision_mv'] = reader.flag()
            values['interpolation_filter'] = (
                SWITCHABLE if reader.flag() else LITERAL_TO_TYPE[reader.f(2)])
    values['colour'] = state.colour
    values['width'], values['height'] = size
    values['render_width'], values['render_height'] = render
    if error_resilient:
        values['frame_parallel_decoding_mode'] = 1
    else:
        values['refresh_frame_context'] = reader.flag()
        values['frame_parallel_decoding_mode'] = reader.flag()
    values['frame_context_idx'] = reader.f(2)
    if frame_type == KEY_FRAME or intra_only or error_resilient:
        state.setup_past_independence()
    values['loop_filter'] = _loop_filter(reader, state)
    values['quantisation'] = Quantisation(
        reader.f(8), _delta_q(reader), _delta_q(reader), _delta_q(reader))
    values['segmentation'] = _segmentation(reader, state)
    cols_log2, rows_log2 = _tile_info(reader, size[0])
    values['tile_cols_log2'], values['tile_rows_log2'] = cols_log2, rows_log2
    compressed = values['header_size_in_bytes'] = reader.f(16)
    if not compressed:
        raise StreamError('the frame states a compressed header of no bytes')
    reader.byte_align()                         # trailing_bits()
    values['uncompressed_size'] = reader.byte_position
    if reader.byte_position + compressed > len(data):
        raise StreamError(
            f'the frame is {len(data)} bytes and its headers alone take '
            f'{reader.byte_position + compressed}')
    _check_tiles(data, reader.byte_position + compressed, 1 << (cols_log2 + rows_log2))
    header = FrameHeader(**values)              # type: ignore[arg-type]  # each value has its field's type
    state.refresh(header)
    return header


def _check_profile(profile: int, colour: ColourConfig) -> None:
    """Refuse an inter frame whose profile is not that of the samples it inherits (7.2).

    An inter frame states a profile and no samples: those are its
    references'. Profiles 2 and 3 are the ones above 8 bits, and profiles 1
    and 3 the ones that are not 4:2:0, so the samples say which profile the
    frame that stated them had.
    """
    high = colour.bit_depth > 8
    other = (colour.subsampling_x, colour.subsampling_y) != (1, 1)
    stated = high << 1 | other
    if profile != stated:
        raise StreamError(
            f'a profile {profile} frame predicts from frames of profile {stated}, '
            f'which have {colour.bit_depth} bits a sample')


def _check_tiles(data: bytes, position: int, count: int) -> None:
    """Refuse a frame whose tiles do not fit it (decode_tiles(), 6.4).

    position -- where the tile data begins, after the compressed header
    count -- how many tiles the frame has

    Every tile but the last states its size in four bytes, and the last takes
    what is left. Each is at least the byte its decoder is started with
    (9.2.1). The driver reads these sizes itself, from the frame it is given.
    """
    for number in range(count - 1):
        field = data[position:position + TILE_SIZE_BYTES]
        if len(field) != TILE_SIZE_BYTES:
            raise StreamError(
                f'the frame ends inside the size of tile {number} of {count}')
        size = int.from_bytes(field, 'big')
        position += TILE_SIZE_BYTES
        if not 1 <= size <= len(data) - position:
            raise StreamError(
                f'tile {number} of {count} states {size} bytes, and the frame has '
                f'{len(data) - position} after it')
        position += size
    if position >= len(data):
        raise StreamError(f'the frame has no byte left for the last tile of its {count}')


def _sync_code(reader: BitReader) -> None:
    found = (reader.f(8), reader.f(8), reader.f(8))
    if found != SYNC_CODE:
        raise StreamError(
            f'the frame sync code is {bytes(found).hex()}, not {bytes(SYNC_CODE).hex()}')


def _colour_config(reader: BitReader, profile: int) -> ColourConfig:
    """color_config() (6.2.2)."""
    bit_depth = (12 if reader.flag() else 10) if profile >= 2 else 8
    color_space = reader.f(3)
    subsampled = profile in (1, 3)
    if color_space != CS_RGB:
        color_range = reader.flag()
        subsampling_x = subsampling_y = 1
        if subsampled:
            subsampling_x, subsampling_y = reader.flag(), reader.flag()
            if subsampling_x and subsampling_y:
                raise StreamError(
                    f'a profile {profile} frame states 4:2:0 sampling, which is '
                    f'profile {profile - 1}')
    else:
        if not subsampled:
            raise StreamError(f'a profile {profile} frame states RGB, which takes 4:4:4')
        color_range = 1
        subsampling_x = subsampling_y = 0
    if subsampled and reader.flag():
        raise StreamError('a reserved bit of the VP9 colour configuration is set')
    return ColourConfig(bit_depth, color_space, color_range, subsampling_x, subsampling_y)


def _frame_size(reader: BitReader) -> tuple[int, int]:
    return (reader.f(16) + 1, reader.f(16) + 1)


def _render_size(reader: BitReader, size: tuple[int, int]) -> tuple[int, int]:
    return _frame_size(reader) if reader.flag() else size


def _frame_size_with_refs(reader: BitReader, state: DecodingState,
                          slots: list[int]) -> tuple[int, int]:
    """frame_size_with_refs() (6.2.5), and the rules its references keep (7.2.5)."""
    references = []
    for slot in slots:
        reference = state.references[slot]
        if reference is None:
            raise StreamError(
                f'the frame predicts from reference slot {slot}, which holds no frame')
        if reference.colour.layout != state.colour.layout:
            raise StreamError(
                f'the frame predicts from reference slot {slot}, whose samples '
                'are laid out differently from its own')
        references.append(reference)
    for reference in references:
        if reader.flag():                       # found_ref
            size = (reference.width, reference.height)
            break
    else:
        size = _frame_size(reader)
    width, height = size
    if not any(2 * width >= reference.width and 2 * height >= reference.height
               and width <= 16 * reference.width and height <= 16 * reference.height
               for reference in references):
        raise StreamError(
            f'a {width}x{height} frame is less than half the size, or more than '
            'sixteen times the size, of each of its references')
    return size


def _loop_filter(reader: BitReader, state: DecodingState) -> LoopFilter:
    """loop_filter_params() (6.2.8)."""
    level = reader.f(6)
    sharpness = reader.f(3)
    delta_enabled = reader.flag()
    delta_update = 0
    if delta_enabled:
        delta_update = reader.flag()
        if delta_update:
            for deltas in (state.ref_deltas, state.mode_deltas):
                for index in range(len(deltas)):
                    if reader.flag():
                        deltas[index] = reader.signed_magnitude(6)
    return LoopFilter(level, sharpness, delta_enabled, delta_update,
                      tuple(state.ref_deltas), tuple(state.mode_deltas))


def _delta_q(reader: BitReader) -> int:
    """read_delta_q() (6.2.10)."""
    return reader.signed_magnitude(4) if reader.flag() else 0


def _probability(reader: BitReader) -> int:
    """read_prob() (6.2.12)."""
    return reader.f(8) if reader.flag() else 255


def _segmentation(reader: BitReader, state: DecodingState) -> Segmentation:
    """segmentation_params() (6.2.11)."""
    update_map = temporal_update = update_data = 0
    enabled = reader.flag()
    if enabled:
        update_map = reader.flag()
        if update_map:
            state.tree_probs = tuple(_probability(reader) for _ in range(7))
            temporal_update = reader.flag()
            state.pred_probs = tuple(
                _probability(reader) if temporal_update else 255 for _ in range(3))
        update_data = reader.flag()
        if update_data:
            state.abs_or_delta_update = reader.flag()
            for segment in range(MAX_SEGMENTS):
                for feature in range(SEG_LVL_MAX):
                    value = 0
                    feature_enabled = reader.flag()
                    if feature_enabled:
                        value = reader.f(SEGMENTATION_FEATURE_BITS[feature])
                        if SEGMENTATION_FEATURE_SIGNED[feature] and reader.flag():
                            value = -value
                    state.feature_enabled[segment][feature] = feature_enabled
                    state.feature_data[segment][feature] = value
    return Segmentation(
        enabled, update_map, temporal_update, update_data, state.abs_or_delta_update,
        state.tree_probs, state.pred_probs,
        tuple(tuple(row) for row in state.feature_enabled),
        tuple(tuple(row) for row in state.feature_data))


def _tile_info(reader: BitReader, width: int) -> tuple[int, int]:
    """tile_info() (6.2.13): log2 of the tile columns, then of the tile rows."""
    sb64_cols = (((width + 7) >> 3) + 7) >> 3
    smallest = 0
    while (MAX_TILE_WIDTH_B64 << smallest) < sb64_cols:
        smallest += 1
    largest = 1
    while (sb64_cols >> largest) >= MIN_TILE_WIDTH_B64:
        largest += 1
    largest -= 1
    columns = smallest
    while columns < largest and reader.flag():
        columns += 1
    if columns > MAX_TILE_COLS_LOG2:
        raise StreamError(
            f'tile_cols_log2 is {columns}, and {MAX_TILE_COLS_LOG2} is the most')
    rows = reader.flag()
    if rows:
        rows += reader.flag()
    return columns, rows
