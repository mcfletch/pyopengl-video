"""The AV1 headers an encoder writes: the sequence header and each frame's header.

A libva AV1 encoder is handed the frame header as a finished bitstream, with
the positions of the fields it may rewrite: the quantiser index, the loop
filter and the CDEF strengths that rate control decides, and the OBU size that
depends on the coded tiles. :class:`Sequence` writes the sequence header OBU
and, for each :class:`Frame` that :class:`GroupOfPictures` decides on, the
frame header with those positions as a :class:`PackedFrameHeader`. Nothing
here holds libva.

The stream is profile 0, 8-bit 4:2:0, limited-range BT.709. Every frame is
shown and coded in display order. A key frame fills all eight reference slots
and so does every inter frame, which predicts from the frame before it: each
slot always holds the last frame coded, and one reconstructed picture is all
an encoder keeps.

Field names are the syntax elements of the AV1 Bitstream & Decoding Process
Specification (section 5), as :mod:`pyopengl_video.vaapi.av1_parse` spells
them. That module reads back what this one writes.
"""
from __future__ import annotations

import dataclasses

from pyopengl_video import obu
from pyopengl_video.vaapi.av1_parse import (
    ALL_FRAMES,
    EIGHTTAP,
    INTER_FRAME,
    KEY_FRAME,
    MAX_LOOP_FILTER,
    MAX_TILE_AREA,
    MAX_TILE_COLS,
    MAX_TILE_ROWS,
    MAX_TILE_WIDTH,
    PRIMARY_REF_NONE,
    REFS_PER_FRAME,
    TX_MODE_LARGEST,
    TX_MODE_SELECT,
)
from pyopengl_video.vaapi.bits import BitWriter, leb128

__all__ = ['COLOUR_PRIMARIES_BT709', 'LEVELS', 'LEVEL_MAXIMUM', 'MATRIX_BT709',
           'OBU_SIZE_BYTES', 'ORDER_HINT_BITS', 'QINDEX_MAX', 'QINDEX_PER_QP',
           'REFERENCE_SLOT', 'SUPERBLOCK', 'TRANSFER_BT709', 'Frame',
           'GroupOfPictures', 'PackedFrameHeader', 'Sequence', 'base_q_idx',
           'level_for', 'loop_filter_level']

#: Samples along a superblock's edge. The 128-sample superblock is left off.
SUPERBLOCK = 64

#: The width of an order hint, which counts frames from the last key frame and
#: wraps at 256.
ORDER_HINT_BITS = 8

#: The reference slot every inter frame names. Each frame fills all eight
#: slots, so any of them holds the frame before.
REFERENCE_SLOT = 0

#: How many bytes a frame header's obu_size is written in where a driver does
#: not state its own. The size is padded to this width, so a driver can write
#: the final value over it without moving what follows.
OBU_SIZE_BYTES = 4

#: The largest quantiser index, and the step from one H.264 quantiser to the next.
QINDEX_MAX = 255
QINDEX_PER_QP = 5

#: Colour, as color_config() states it: BT.709 primaries, transfer and matrix.
COLOUR_PRIMARIES_BT709 = 1
TRANSFER_BT709 = 1
MATRIX_BT709 = 1

#: seq_level_idx for a stream no level holds: "maximum parameters".
LEVEL_MAXIMUM = 31

#: Levels, as ``(seq_level_idx, MaxPicSize, MaxHSize, MaxVSize, MaxDisplayRate,
#: MainMbps, HighMbps)`` from Annex A.3 of the specification, lowest first. The
#: rates are megabits a second, and a level with no high tier has 0 for it.
LEVELS = (
    (0, 147456, 2048, 1152, 4_423_680, 1.5, 0),
    (1, 278784, 2816, 1584, 8_363_520, 3.0, 0),
    (4, 665856, 4352, 2448, 19_975_680, 6.0, 0),
    (5, 1065024, 5504, 3096, 31_950_720, 10.0, 0),
    (8, 2359296, 6144, 3456, 70_778_880, 12.0, 30.0),
    (9, 2359296, 6144, 3456, 141_557_760, 20.0, 50.0),
    (12, 8912896, 8192, 4352, 267_386_880, 30.0, 100.0),
    (13, 8912896, 8192, 4352, 534_773_760, 40.0, 160.0),
    (14, 8912896, 8192, 4352, 1_069_547_520, 60.0, 240.0),
    (16, 35651584, 16384, 8704, 1_069_547_520, 60.0, 240.0),
    (17, 35651584, 16384, 8704, 2_139_095_040, 100.0, 480.0),
    (18, 35651584, 16384, 8704, 4_278_190_080, 160.0, 800.0),
)


def _superblocks(samples: int) -> int:
    """How many superblocks cover `samples`."""
    return -(-samples // SUPERBLOCK)


def level_for(width: int, height: int, frame_rate: tuple[int, int],
              bitrate: int = 0) -> tuple[int, int]:
    """The lowest level covering this frame size, rate and bitrate, as (seq_level_idx, seq_tier).

    The frame is measured in whole superblocks, which is the most an encoder
    codes for a picture of that size. The high tier is taken where the bitrate
    is over the level's main tier and the level has one. A stream no level
    holds is :data:`LEVEL_MAXIMUM`.
    """
    coded_width = _superblocks(width) * SUPERBLOCK
    coded_height = _superblocks(height) * SUPERBLOCK
    samples = coded_width * coded_height
    numerator, denominator = frame_rate
    per_second = samples * numerator / denominator
    megabits = bitrate / 1_000_000
    for level, pic_size, h_size, v_size, display_rate, main, high in LEVELS:
        if (samples > pic_size or coded_width > h_size or coded_height > v_size
                or per_second > display_rate):
            continue
        if megabits <= main:
            return level, 0
        if megabits <= high:
            return level, 1
    return LEVEL_MAXIMUM, 0


def base_q_idx(qp: int) -> int:
    """The AV1 quantiser index for an H.264 quantiser of `qp`.

    Five indices to one H.264 step takes H.264's 0 to 51 onto AV1's 0 to 255,
    which is the ratio Mesa converts between the two scales with. Index 0 is
    lossless coding, whose frame header has a different syntax, so the lowest
    index returned is 1.
    """
    return max(1, min(QINDEX_MAX, qp * QINDEX_PER_QP))


def loop_filter_level(q_index: int) -> int:
    """The deblocking filter level written for a frame coded at `q_index`.

    One level to six quantiser indices: a coarser quantiser leaves stronger
    block edges. A driver that chooses its own levels writes them over this.
    """
    return min(MAX_LOOP_FILTER, q_index // 6)


def _tile_log2(block: int, target: int) -> int:
    """tile_log2(): the smallest k with `block` << k at least `target`."""
    k = 0
    while (block << k) < target:
        k += 1
    return k


@dataclasses.dataclass(frozen=True)
class Frame:
    """What one frame is coded as.

    key -- a key frame: decoding can start here
    number -- frames since the last key frame, 0 for the key frame itself
    base_q_idx -- the quantiser index the frame header states
    """

    key: bool
    number: int
    base_q_idx: int

    @property
    def frame_type(self) -> int:
        return KEY_FRAME if self.key else INTER_FRAME

    @property
    def order_hint(self) -> int:
        """The frame's place in display order, modulo the order hint's width."""
        return self.number % (1 << ORDER_HINT_BITS)

    @property
    def primary_ref_frame(self) -> int:
        """Which reference the frame loads its probabilities from.

        An inter frame continues from the frame before it, LAST_FRAME. A key
        frame starts from the defaults.
        """
        return PRIMARY_REF_NONE if self.key else 0

    @property
    def refresh_frame_flags(self) -> int:
        """The slots the frame is saved in, which is all of them."""
        return ALL_FRAMES

    @property
    def ref_frame_idx(self) -> tuple[int, ...]:
        """The slot each of LAST_FRAME to ALTREF_FRAME names."""
        return (REFERENCE_SLOT,) * REFS_PER_FRAME


@dataclasses.dataclass(frozen=True)
class PackedFrameHeader:
    """A frame header as bytes, and where a driver finds what it rewrites.

    data -- the OBU: its header byte, its obu_size and the uncompressed
        header, which opens a frame OBU or is the whole of a frame header OBU
    size_in_bits -- the length from the start of `data` to the last bit of the
        uncompressed header, and for a frame header OBU the trailing one bit
        after it
    bit_offset_qindex -- where base_q_idx is
    bit_offset_segmentation -- where segmentation_enabled is
    bit_offset_loopfilter_params -- where loop_filter_params() begins
    bit_offset_cdef_params -- where cdef_params() begins
    size_in_bits_cdef_params -- the length of cdef_params()
    byte_offset_obu_size -- where obu_size is

    Every offset counts from the start of `data`.
    """

    data: bytes
    size_in_bits: int
    bit_offset_qindex: int
    bit_offset_segmentation: int
    bit_offset_loopfilter_params: int
    bit_offset_cdef_params: int
    size_in_bits_cdef_params: int
    byte_offset_obu_size: int


class Sequence:
    """The sequence-level choices of one encoding session, and its headers.

    width/height -- the picture size in luma samples, both even
    frame_rate -- an exact ``(numerator, denominator)``
    bitrate -- bits per second, which the level must cover; zero leaves the
        bitrate out of the level
    tx_mode_select -- True where each block chooses its transform size
        (TX_MODE_SELECT), False where the largest that fits is used
        (TX_MODE_LARGEST)

    The derived values a driver is told as well are public: :attr:`level` and
    :attr:`tier`, :attr:`tx_mode`, and the tile layout in :attr:`tile_cols`,
    :attr:`tile_rows`, :attr:`width_in_sbs_minus_1` and
    :attr:`height_in_sbs_minus_1`. A frame is one tile unless it is wider or
    larger than a tile may be, and then it is split evenly into the fewest
    tiles that are allowed.
    """

    def __init__(self, width: int, height: int, frame_rate: tuple[int, int],
                 bitrate: int = 0, *, tx_mode_select: bool = True) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f'{width}x{height} is not a frame size')
        if width % 2 or height % 2:
            raise ValueError(
                f'{width}x{height} has an odd dimension, and 4:2:0 chroma needs both even')
        if width > 1 << 16 or height > 1 << 16:
            raise ValueError(f'{width}x{height} is larger than an AV1 frame may be')
        self.width = int(width)
        self.height = int(height)
        self.frame_rate = (int(frame_rate[0]), int(frame_rate[1]))
        self.bitrate = int(bitrate)
        self.level, self.tier = level_for(self.width, self.height, self.frame_rate,
                                          self.bitrate)
        self.tx_mode = TX_MODE_SELECT if tx_mode_select else TX_MODE_LARGEST
        self._lay_out_tiles()

    def __repr__(self) -> str:
        return (f'<{type(self).__name__} {self.width}x{self.height} '
                f'level index {self.level}, {self.tile_cols}x{self.tile_rows} tiles>')

    def _lay_out_tiles(self) -> None:
        """Choose the fewest uniform tiles tile_info() (5.9.15) allows."""
        sb_cols = _superblocks(self.width)
        sb_rows = _superblocks(self.height)
        sb_size_log2 = SUPERBLOCK.bit_length() - 1
        max_tile_width_sb = MAX_TILE_WIDTH >> sb_size_log2
        max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size_log2)
        min_log2_cols = _tile_log2(max_tile_width_sb, sb_cols)
        min_log2_tiles = max(min_log2_cols, _tile_log2(max_tile_area_sb, sb_rows * sb_cols))
        self._max_log2_cols = _tile_log2(1, min(sb_cols, MAX_TILE_COLS))
        self._max_log2_rows = _tile_log2(1, min(sb_rows, MAX_TILE_ROWS))
        self._cols_log2 = min_log2_cols
        self._rows_log2 = max(min_log2_tiles - min_log2_cols, 0)
        self.width_in_sbs_minus_1 = self._uniform(sb_cols, self._cols_log2)
        self.height_in_sbs_minus_1 = self._uniform(sb_rows, self._rows_log2)
        self.tile_cols = len(self.width_in_sbs_minus_1)
        self.tile_rows = len(self.height_in_sbs_minus_1)

    @staticmethod
    def _uniform(superblocks: int, log2: int) -> tuple[int, ...]:
        """Each tile's extent in superblocks, less one, under uniform spacing."""
        step = (superblocks + (1 << log2) - 1) >> log2
        return tuple(min(step, superblocks - start) - 1
                     for start in range(0, superblocks, step))

    # ------------------------------------------------------ sequence header

    def sequence_header(self) -> bytes:
        """The sequence header OBU (5.5), with a size field.

        This is what the stream carries in front of every key frame and what
        an ``av1C`` box holds.
        """
        writer = BitWriter()
        writer.f(0, 3)                          # seq_profile
        writer.flag(0)                          # still_picture
        writer.flag(0)                          # reduced_still_picture_header
        writer.flag(1)                          # timing_info_present_flag
        writer.f(self.frame_rate[1], 32)        # num_units_in_display_tick
        writer.f(self.frame_rate[0], 32)        # time_scale
        # Frames are shown when their timestamps say, which need not be one
        # tick apart.
        writer.flag(0)                          # equal_picture_interval
        writer.flag(0)                          # decoder_model_info_present_flag
        writer.flag(0)                          # initial_display_delay_present_flag
        writer.f(0, 5)                          # operating_points_cnt_minus_1
        writer.f(0, 12)                         # operating_point_idc[0]
        writer.f(self.level, 5)                 # seq_level_idx[0]
        if self.level > 7:
            writer.flag(self.tier)              # seq_tier[0]
        width_bits = max(1, (self.width - 1).bit_length())
        height_bits = max(1, (self.height - 1).bit_length())
        writer.f(width_bits - 1, 4)             # frame_width_bits_minus_1
        writer.f(height_bits - 1, 4)            # frame_height_bits_minus_1
        writer.f(self.width - 1, width_bits)    # max_frame_width_minus_1
        writer.f(self.height - 1, height_bits)  # max_frame_height_minus_1
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
        writer.f(ORDER_HINT_BITS - 1, 3)        # order_hint_bits_minus_1
        writer.flag(0)                          # enable_superres
        writer.flag(1)                          # enable_cdef
        writer.flag(0)                          # enable_restoration
        # color_config() (5.5.2)
        writer.flag(0)                          # high_bitdepth
        writer.flag(0)                          # mono_chrome
        writer.flag(1)                          # color_description_present_flag
        writer.f(COLOUR_PRIMARIES_BT709, 8)     # color_primaries
        writer.f(TRANSFER_BT709, 8)             # transfer_characteristics
        writer.f(MATRIX_BT709, 8)               # matrix_coefficients
        writer.flag(0)                          # color_range: limited
        writer.f(0, 2)                          # chroma_sample_position: unknown
        writer.flag(0)                          # separate_uv_delta_q
        writer.flag(0)                          # film_grain_params_present
        writer.flag(1)                          # trailing_one_bit
        return obu.Obu(obu.SEQUENCE_HEADER, writer.bytes()).encoded()

    # --------------------------------------------------------- frame header

    def frame_header(self, frame: Frame, *, frame_obu: bool = True,
                     size_bytes: int = OBU_SIZE_BYTES) -> PackedFrameHeader:
        """The header of `frame`, with the offsets a driver is told.

        frame_obu -- True for the opening of a frame OBU, which the driver
            follows with the tile group; False for a whole frame header OBU
        size_bytes -- how many bytes obu_size is written in

        obu_size counts the header alone. A driver that appends the tile group
        to a frame OBU writes the final size at :attr:`PackedFrameHeader.byte_offset_obu_size`.
        """
        writer = BitWriter()
        offsets = self._uncompressed_header(writer, frame)
        header_bits = len(writer)
        if frame_obu:
            writer.byte_align()                 # byte_alignment()
        else:
            writer.flag(1)                      # trailing_one_bit
            header_bits += 1
        payload = writer.bytes()
        kind = obu.FRAME if frame_obu else obu.FRAME_HEADER
        prefix = bytes([kind << 3 | 0x02]) + leb128(len(payload), size_bytes)
        start = len(prefix) * 8
        return PackedFrameHeader(
            data=prefix + payload,
            size_in_bits=start + header_bits,
            bit_offset_qindex=start + offsets['qindex'],
            bit_offset_segmentation=start + offsets['segmentation'],
            bit_offset_loopfilter_params=start + offsets['loop_filter'],
            bit_offset_cdef_params=start + offsets['cdef'],
            size_in_bits_cdef_params=offsets['cdef_end'] - offsets['cdef'],
            byte_offset_obu_size=1)

    def _uncompressed_header(self, writer: BitWriter, frame: Frame) -> dict[str, int]:
        """Write uncompressed_header() (5.9.2), returning where its patched fields are."""
        offsets: dict[str, int] = {}
        writer.flag(0)                          # show_existing_frame
        writer.f(frame.frame_type, 2)
        writer.flag(1)                          # show_frame
        if not frame.key:
            # A shown key frame is error resilient by definition and codes no flag.
            writer.flag(0)                      # error_resilient_mode
        writer.flag(0)                          # disable_cdf_update
        writer.flag(0)                          # frame_size_override_flag
        writer.f(frame.order_hint, ORDER_HINT_BITS)
        if not frame.key:
            writer.f(frame.primary_ref_frame, 3)
            # A shown key frame refreshes every slot and codes no flags.
            writer.f(frame.refresh_frame_flags, 8)
            writer.flag(0)                      # frame_refs_short_signaling
            for slot in frame.ref_frame_idx:
                writer.f(slot, 3)
        self._render_size(writer)
        if not frame.key:
            writer.flag(0)                      # allow_high_precision_mv
            writer.flag(0)                      # is_filter_switchable
            writer.f(EIGHTTAP, 2)               # interpolation_filter
            writer.flag(0)                      # is_motion_mode_switchable
        writer.flag(0)                          # disable_frame_end_update_cdf
        self._tile_info(writer)
        # quantization_params() (5.9.12)
        offsets['qindex'] = len(writer)
        writer.f(frame.base_q_idx, 8)
        writer.flag(0)                          # DeltaQYDc: delta_coded
        writer.flag(0)                          # DeltaQUDc: delta_coded
        writer.flag(0)                          # DeltaQUAc: delta_coded
        writer.flag(0)                          # using_qmatrix
        offsets['segmentation'] = len(writer)
        writer.flag(0)                          # segmentation_enabled
        writer.flag(0)                          # delta_q_present
        # loop_filter_params() (5.9.11)
        offsets['loop_filter'] = len(writer)
        level = loop_filter_level(frame.base_q_idx)
        writer.f(level, 6)                      # loop_filter_level[0]
        writer.f(level, 6)                      # loop_filter_level[1]
        if level:
            writer.f(level, 6)                  # loop_filter_level[2]
            writer.f(level, 6)                  # loop_filter_level[3]
        writer.f(0, 3)                          # loop_filter_sharpness
        writer.flag(0)                          # loop_filter_delta_enabled
        # cdef_params() (5.9.19): one strength of zero, which filters nothing.
        offsets['cdef'] = len(writer)
        writer.f(0, 2)                          # cdef_damping_minus_3
        writer.f(0, 2)                          # cdef_bits
        writer.f(0, 4)                          # cdef_y_pri_strength[0]
        writer.f(0, 2)                          # cdef_y_sec_strength[0]
        writer.f(0, 4)                          # cdef_uv_pri_strength[0]
        writer.f(0, 2)                          # cdef_uv_sec_strength[0]
        offsets['cdef_end'] = len(writer)
        writer.flag(self.tx_mode == TX_MODE_SELECT)     # tx_mode_select
        if not frame.key:
            writer.flag(0)                      # reference_select
        writer.flag(0)                          # reduced_tx_set
        if not frame.key:
            for _ in range(REFS_PER_FRAME):
                writer.flag(0)                  # is_global
        return offsets

    def _render_size(self, writer: BitWriter) -> None:
        """Write render_size() (5.9.6) with the picture size stated.

        The frame is the size the sequence header gives, so the render size
        could be inferred from it. A driver that codes a larger frame than the
        picture, to fill its block size, takes the picture size from here.
        """
        writer.flag(1)                          # render_and_frame_size_different
        writer.f(self.width - 1, 16)            # render_width_minus_1
        writer.f(self.height - 1, 16)           # render_height_minus_1

    def _tile_info(self, writer: BitWriter) -> None:
        """Write tile_info() (5.9.15) for the uniform layout chosen."""
        writer.flag(1)                          # uniform_tile_spacing_flag
        if self._cols_log2 < self._max_log2_cols:
            writer.flag(0)                      # increment_tile_cols_log2
        if self._rows_log2 < self._max_log2_rows:
            writer.flag(0)                      # increment_tile_rows_log2
        if self._cols_log2 or self._rows_log2:
            writer.f(0, self._cols_log2 + self._rows_log2)  # context_update_tile_id
            writer.f(3, 2)                      # tile_size_bytes_minus_1


class GroupOfPictures:
    """Decides what each frame of a recording is coded as.

    A key frame every `gop` frames and wherever a caller asks for one, and
    inter frames between them. Nothing is held back, so frames leave the
    encoder in the order they arrived.

    gop -- frames between key frames; one makes every frame a key frame
    base_q_idx -- the quantiser index each frame header states
    """

    def __init__(self, gop: int, base_q_idx: int) -> None:
        if gop < 1:
            raise ValueError(f'a group of {gop} pictures holds none')
        self.gop = int(gop)
        self.base_q_idx = int(base_q_idx)
        self._position = 0

    def __repr__(self) -> str:
        return f'<{type(self).__name__} gop {self.gop}>'

    def next_frame(self, force_key: bool = False) -> Frame:
        """The next frame to code. Each call advances the structure by one frame."""
        if force_key or self._position % self.gop == 0:
            self._position = 0
        frame = Frame(key=self._position == 0, number=self._position,
                      base_q_idx=self.base_q_idx)
        self._position += 1
        return frame
