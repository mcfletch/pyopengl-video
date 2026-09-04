"""The H.264 control layer: bitstream writing and the parameter sets.

libva is a slice-level interface. Where NVENC is handed a picture and decides
everything about how to code it, a libva encoder is told what to code and the
caller supplies the sequence, picture and slice parameters, the group-of-
pictures structure, and -- because the common drivers ask for them -- the
sequence and picture parameter sets as finished bitstreams.

This module is that half, and it holds no libva at all: a
:class:`BitWriter` for the exponential-Golomb syntax the specification is
written in, and :class:`ParameterSets`, which turns a frame size, a frame rate
and a quantiser into the SPS and PPS NAL units that describe them. The values
it derives -- the level, the coded size in macroblocks, the widths of the
frame-number and picture-order-count fields -- are what the encoder must also
put in the structures it hands the driver, so it reads them from here rather
than deriving them again.

Field names follow the syntax elements of ITU-T H.264, so a reader can put this
beside the specification's syntax tables.
"""
from __future__ import annotations

import dataclasses

#: Bytes per macroblock edge. Every coded dimension is a multiple of this, and a
#: frame that is not is coded larger and cropped back in the SPS.
MACROBLOCK = 16

#: profile_idc for High profile, which is what every part this backend runs on
#: encodes and every player decodes.
PROFILE_HIGH = 100

#: chroma_format_idc for 4:2:0, the only sampling an H.264 encoder here produces.
CHROMA_420 = 1

#: How wide the frame_num and pic_order_cnt_lsb fields are, as the ``minus4``
#: the syntax carries: both count to 256 before wrapping, which the derivation
#: in the specification handles for a sequence of any length.
LOG2_MAX_FRAME_NUM_MINUS4 = 4
LOG2_MAX_POC_LSB_MINUS4 = 4

#: Colour, as the video usability information states it: limited-range BT.709
#: primaries, transfer and matrix. The conversion from RGB must match, and
#: :mod:`pyopengl_video.vaapi.encoder` asks the driver for the same thing.
COLOUR_PRIMARIES_BT709 = 1
TRANSFER_BT709 = 1
MATRIX_BT709 = 1
#: video_format 5, "unspecified", which is what a rendered frame is.
VIDEO_FORMAT_UNSPECIFIED = 5

#: Slice types, as the syntax numbers them: a picture coded without prediction
#: is I, one predicting from earlier pictures alone is P.
SLICE_TYPE_P = 0
SLICE_TYPE_I = 2

#: Levels, as ``(level_idc, MaxMBPS, MaxFS, MaxBR)`` from table A-1 of the
#: specification, in the order a stream should prefer them. MaxBR is in units of
#: 1000 bits per second and is stated for the Baseline/Main/Extended profiles;
#: :func:`level_for` scales it for High, which table A-2 permits at 1.25x.
LEVELS: tuple[tuple[int, int, int, int], ...] = (
    (10, 1485, 99, 64),
    (11, 3000, 396, 192),
    (12, 6000, 396, 384),
    (13, 11880, 396, 768),
    (20, 11880, 396, 2000),
    (21, 19800, 792, 4000),
    (22, 20250, 1620, 4000),
    (30, 40500, 1620, 10000),
    (31, 108000, 3600, 14000),
    (32, 216000, 5120, 20000),
    (40, 245760, 8192, 20000),
    (41, 245760, 8192, 50000),
    (42, 522240, 8704, 50000),
    (50, 589824, 22080, 135000),
    (51, 983040, 36864, 240000),
    (52, 2073600, 36864, 240000),
    (60, 4177920, 139264, 240000),
    (61, 8355840, 139264, 480000),
    (62, 16711680, 139264, 800000),
)

#: What table A-2 multiplies the level's MaxBR by for High profile.
HIGH_PROFILE_BITRATE_FACTOR = 1.25


class BitWriter:
    """Accumulates the bit-packed syntax an H.264 raw byte sequence is made of.

    Bits go in most-significant first, which is the order the specification
    reads them in, and :meth:`rbsp` closes the sequence with the stop bit and
    alignment that ends every NAL unit payload.
    """

    def __init__(self) -> None:
        self._bits: list[int] = []

    def u(self, value: int, count: int) -> None:
        """Write `value` as an unsigned field `count` bits wide -- ``u(n)``."""
        if value < 0:
            raise ValueError(f'{value} is negative, and u({count}) is unsigned')
        if count < 0 or value >> count:
            raise ValueError(f'{value} does not fit in a {count}-bit field')
        for shift in range(count - 1, -1, -1):
            self._bits.append((value >> shift) & 1)

    def flag(self, value: object) -> None:
        """Write a one-bit flag."""
        self.u(1 if value else 0, 1)

    def ue(self, value: int) -> None:
        """Write `value` as an unsigned exponential-Golomb code -- ``ue(v)``."""
        if value < 0:
            raise ValueError(f'{value} is negative, and ue(v) is unsigned')
        code = value + 1
        width = code.bit_length()
        self.u(0, width - 1)
        self.u(code, width)

    def se(self, value: int) -> None:
        """Write `value` as a signed exponential-Golomb code -- ``se(v)``.

        The specification maps the signed value onto an unsigned one, positive
        values to odd codes and negative to even.
        """
        self.ue(2 * value - 1 if value > 0 else -2 * value)

    def __len__(self) -> int:
        """How many bits have been written."""
        return len(self._bits)

    def rbsp(self) -> bytes:
        """The raw byte sequence payload: the bits written, stopped and aligned.

        A stop bit follows the syntax, then zeroes to the next byte boundary. An
        already aligned sequence still gains a whole byte, because the stop bit
        is what says where the syntax ended.
        """
        return self._pack(self._bits + [1])

    def padded(self) -> tuple[bytes, int]:
        """The bits written, padded out to a byte, and how many there were.

        For a header that is not the end of a unit: a driver handed one of
        these carries on writing from the bit the count names, so no stop bit
        is written and the padding is not part of the syntax.
        """
        return self._pack(list(self._bits)), len(self._bits)

    @staticmethod
    def _pack(bits: list[int]) -> bytes:
        while len(bits) % 8:
            bits.append(0)
        return bytes(
            int(''.join(str(bit) for bit in bits[index:index + 8]), 2)
            for index in range(0, len(bits), 8)
        )


def emulation_prevented(payload: bytes) -> bytes:
    """Insert the escape bytes that keep a start code out of a NAL unit.

    Three zero bytes in a row would read as the start of the next unit, so the
    specification requires a ``0x03`` after any two zeroes that precede a byte
    below four. A decoder strips it again.
    """
    out = bytearray()
    zeros = 0
    for byte in payload:
        if zeros == 2 and byte <= 3:
            out.append(3)
            zeros = 0
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


def nal_unit(payload: bytes, nal_unit_type: int, nal_ref_idc: int = 3) -> bytes:
    """Wrap an RBSP in its NAL unit header and escape it.

    nal_ref_idc says how much of a reference the unit is; parameter sets and
    reference pictures use 3, and a picture nothing predicts from uses 0.
    """
    return emulation_prevented(
        bytes([(nal_ref_idc & 3) << 5 | (nal_unit_type & 0x1F)]) + payload)


def annexb(*units: bytes) -> bytes:
    """Join NAL `units` into an Annex-B fragment, four-byte start codes."""
    return b''.join(b'\x00\x00\x00\x01' + unit for unit in units)


def level_for(width: int, height: int, frame_rate: tuple[int, int],
              bitrate: int = 0) -> int:
    """The lowest level that covers this frame size, rate and bitrate.

    A stream that declares a level too low for what it contains is one a
    conforming decoder may refuse, so the frame size in macroblocks, the
    macroblock rate and -- when a bitrate is given -- the bit rate all have to
    fit. The highest level is returned when nothing fits, which is the closest
    a stream can come to describing itself.
    """
    macroblocks = (-(-width // MACROBLOCK)) * (-(-height // MACROBLOCK))
    numerator, denominator = frame_rate
    per_second = macroblocks * numerator / denominator
    kilobits = bitrate / 1000.0
    for level_idc, max_mbps, max_fs, max_br in LEVELS:
        if (macroblocks <= max_fs and per_second <= max_mbps
                and kilobits <= max_br * HIGH_PROFILE_BITRATE_FACTOR):
            return level_idc
    return LEVELS[-1][0]


class ParameterSets:
    """The sequence and picture parameter sets for one encoding session.

    Built once when an encoder opens and written into the stream at every IDR.
    The derived values are public because the driver has to be told the same
    ones: :attr:`level_idc`, the coded size in :attr:`width_in_mbs` and
    :attr:`height_in_mbs`, and the field widths the slice headers use.

    width/height -- the picture size a caller asked for, in luma samples; both
        even, because chroma is subsampled by two in each direction
    frame_rate -- as an exact ``(numerator, denominator)`` ratio
    max_num_ref_frames -- how many pictures the decoded picture buffer holds
    init_qp -- the quantiser the picture parameter set starts from
    bitrate -- bits per second the stream is aiming at, which the level must
        cover; zero leaves the bitrate out of the level
    transform_8x8 -- whether the encoder may use the 8x8 transform, which High
        profile allows and the picture parameter set must declare
    """

    def __init__(self, width: int, height: int, frame_rate: tuple[int, int],
                 max_num_ref_frames: int = 1, init_qp: int = 26,
                 bitrate: int = 0, transform_8x8: bool = True) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f'{width}x{height} is not a frame size')
        if width % 2 or height % 2:
            raise ValueError(
                f'{width}x{height} has an odd dimension, and 4:2:0 chroma is '
                'subsampled by two in each direction')
        if not 0 <= init_qp <= 51:
            raise ValueError(f'{init_qp} is outside the H.264 quantiser range 0..51')
        self.width = int(width)
        self.height = int(height)
        self.frame_rate = (int(frame_rate[0]), int(frame_rate[1]))
        self.max_num_ref_frames = int(max_num_ref_frames)
        self.init_qp = int(init_qp)
        self.bitrate = int(bitrate)
        self.transform_8x8 = bool(transform_8x8)

        self.width_in_mbs = -(-self.width // MACROBLOCK)
        self.height_in_mbs = -(-self.height // MACROBLOCK)
        #: The size the macroblock grid actually codes, which is the picture
        #: size rounded up. The difference is cropped away in the SPS.
        self.coded_size = (self.width_in_mbs * MACROBLOCK,
                           self.height_in_mbs * MACROBLOCK)
        self.level_idc = level_for(self.width, self.height, self.frame_rate,
                                   self.bitrate)
        self.log2_max_frame_num = LOG2_MAX_FRAME_NUM_MINUS4 + 4
        self.log2_max_poc_lsb = LOG2_MAX_POC_LSB_MINUS4 + 4

    def __repr__(self) -> str:
        return (f'<{type(self).__name__} {self.width}x{self.height} '
                f'@{self.frame_rate[0]}/{self.frame_rate[1]} level '
                f'{self.level_idc / 10:g}>')

    # ------------------------------------------------------------------ SPS

    def sps(self) -> bytes:
        """The sequence parameter set, as a NAL unit."""
        writer = BitWriter()
        writer.u(PROFILE_HIGH, 8)
        writer.u(0, 8)                          # constraint_set flags, reserved
        writer.u(self.level_idc, 8)
        writer.ue(0)                            # seq_parameter_set_id
        writer.ue(CHROMA_420)
        writer.ue(0)                            # bit_depth_luma_minus8
        writer.ue(0)                            # bit_depth_chroma_minus8
        writer.flag(0)                          # qpprime_y_zero_transform_bypass
        writer.flag(0)                          # seq_scaling_matrix_present
        writer.ue(LOG2_MAX_FRAME_NUM_MINUS4)
        writer.ue(0)                            # pic_order_cnt_type
        writer.ue(LOG2_MAX_POC_LSB_MINUS4)
        writer.ue(self.max_num_ref_frames)
        writer.flag(0)                          # gaps_in_frame_num_value_allowed
        writer.ue(self.width_in_mbs - 1)
        writer.ue(self.height_in_mbs - 1)
        writer.flag(1)                          # frame_mbs_only_flag
        writer.flag(1)                          # direct_8x8_inference_flag
        self._write_cropping(writer)
        writer.flag(1)                          # vui_parameters_present_flag
        self._write_vui(writer)
        return nal_unit(writer.rbsp(), nal_unit_type=7)

    def _write_cropping(self, writer: BitWriter) -> None:
        """State how much of the coded macroblock grid is not in the picture.

        The offsets are in chroma samples -- two luma samples each way for a
        4:2:0 progressive frame -- which is why an odd picture size is refused.
        """
        coded_width, coded_height = self.coded_size
        right = (coded_width - self.width) // 2
        bottom = (coded_height - self.height) // 2
        writer.flag(right or bottom)
        if right or bottom:
            writer.ue(0)                        # left
            writer.ue(right)
            writer.ue(0)                        # top
            writer.ue(bottom)

    def _write_vui(self, writer: BitWriter) -> None:
        """Video usability information: aspect, colour, timing and reordering."""
        writer.flag(1)                          # aspect_ratio_info_present
        writer.u(1, 8)                          # aspect_ratio_idc, square
        writer.flag(0)                          # overscan_info_present
        writer.flag(1)                          # video_signal_type_present
        writer.u(VIDEO_FORMAT_UNSPECIFIED, 3)
        writer.flag(0)                          # video_full_range_flag, limited
        writer.flag(1)                          # colour_description_present
        writer.u(COLOUR_PRIMARIES_BT709, 8)
        writer.u(TRANSFER_BT709, 8)
        writer.u(MATRIX_BT709, 8)
        writer.flag(0)                          # chroma_loc_info_present
        writer.flag(1)                          # timing_info_present
        # The tick is a field period, so the time scale counts twice the frames.
        writer.u(self.frame_rate[1], 32)        # num_units_in_tick
        writer.u(self.frame_rate[0] * 2, 32)    # time_scale
        writer.flag(1)                          # fixed_frame_rate_flag
        writer.flag(0)                          # nal_hrd_parameters_present
        writer.flag(0)                          # vcl_hrd_parameters_present
        writer.flag(0)                          # pic_struct_present
        writer.flag(1)                          # bitstream_restriction
        writer.flag(1)                          # motion_vectors_over_pic_boundaries
        writer.ue(0)                            # max_bytes_per_pic_denom
        writer.ue(0)                            # max_bits_per_mb_denom
        writer.ue(15)                           # log2_max_mv_length_horizontal
        writer.ue(15)                           # log2_max_mv_length_vertical
        writer.ue(0)                            # max_num_reorder_frames
        writer.ue(self.max_num_ref_frames)      # max_dec_frame_buffering

    # ------------------------------------------------------------------ PPS

    def pps(self) -> bytes:
        """The picture parameter set, as a NAL unit."""
        writer = BitWriter()
        writer.ue(0)                            # pic_parameter_set_id
        writer.ue(0)                            # seq_parameter_set_id
        writer.flag(1)                          # entropy_coding_mode_flag, CABAC
        writer.flag(0)                          # bottom_field_pic_order_present
        writer.ue(0)                            # num_slice_groups_minus1
        writer.ue(0)                            # num_ref_idx_l0_default_active_minus1
        writer.ue(0)                            # num_ref_idx_l1_default_active_minus1
        writer.flag(0)                          # weighted_pred_flag
        writer.u(0, 2)                          # weighted_bipred_idc
        writer.se(self.init_qp - 26)            # pic_init_qp_minus26
        writer.se(0)                            # pic_init_qs_minus26
        writer.se(0)                            # chroma_qp_index_offset
        writer.flag(1)                          # deblocking_filter_control_present
        writer.flag(0)                          # constrained_intra_pred_flag
        writer.flag(0)                          # redundant_pic_cnt_present_flag
        if self.transform_8x8:
            writer.flag(1)                      # transform_8x8_mode_flag
            writer.flag(0)                      # pic_scaling_matrix_present
            writer.se(0)                        # second_chroma_qp_index_offset
        return nal_unit(writer.rbsp(), nal_unit_type=8)

    def headers(self) -> bytes:
        """Both sets as one Annex-B fragment, in the order a stream carries them."""
        return annexb(self.sps(), self.pps())

    # --------------------------------------------------------- slice headers

    def slice_header(self, picture: Picture, qp: int | None = None) -> bytes:
        """The NAL header and slice header for `picture`, as an Annex-B unit.

        A libva driver may write only the coded macroblocks and leave
        everything in front of them to the caller, which is what Mesa's does.
        The bytes end wherever the syntax ends, padded out to a byte; the exact
        bit count is what :meth:`packed_slice_header` returns beside them, and
        it is what the driver needs in order to carry on writing from the right
        bit.

        qp -- the quantiser this picture is coded at, when it differs from the
            one the picture parameter set starts from
        """
        return self.packed_slice_header(picture, qp)[0]

    def packed_slice_header(self, picture: Picture,
                            qp: int | None = None) -> tuple[bytes, int]:
        """The slice header as bytes, and how many bits of it are syntax."""
        writer = BitWriter()
        writer.ue(0)                            # first_mb_in_slice
        writer.ue(SLICE_TYPE_I + 5 if picture.intra else SLICE_TYPE_P + 5)
        writer.ue(0)                            # pic_parameter_set_id
        writer.u(picture.frame_num, self.log2_max_frame_num)
        if picture.idr:
            writer.ue(picture.idr_pic_id)
        writer.u(picture.poc % (1 << self.log2_max_poc_lsb), self.log2_max_poc_lsb)
        if not picture.intra:
            writer.flag(1)                      # num_ref_idx_active_override
            writer.ue(max(len(picture.references), 1) - 1)
            writer.flag(0)                      # ref_pic_list_modification_flag_l0
        if picture.reference:
            if picture.idr:
                writer.flag(0)                  # no_output_of_prior_pics_flag
                writer.flag(0)                  # long_term_reference_flag
            else:
                # The sliding window the specification defaults to, which is
                # what GroupOfPictures keeps its buffer by.
                writer.flag(0)                  # adaptive_ref_pic_marking_mode
        if not picture.intra:
            writer.ue(0)                        # cabac_init_idc
        writer.se((self.init_qp if qp is None else int(qp)) - self.init_qp)
        writer.ue(0)                            # disable_deblocking_filter_idc
        writer.se(0)                            # slice_alpha_c0_offset_div2
        writer.se(0)                            # slice_beta_offset_div2

        payload, bits = writer.padded()
        header = bytes([3 << 5 | (5 if picture.idr else 1)])
        escaped = emulation_prevented(header + payload)
        data = b'\x00\x00\x00\x01' + escaped
        padding = len(payload) * 8 - bits
        return data, len(data) * 8 - padding

    def packed_parameter_sets(self) -> tuple[bytes, int]:
        """Both parameter sets as one packed header, and its length in bits.

        They go out together because a driver takes one sequence header per
        picture and the picture parameter set has to be in the stream before
        the slice that names it.
        """
        data = self.headers()
        return data, len(data) * 8


@dataclasses.dataclass(frozen=True)
class Picture:
    """What one picture is coded as, and what it may predict from.

    idr -- this picture begins a new coded video sequence, and a decoder joining
        the stream here needs nothing earlier
    intra -- coded without prediction from another picture
    reference -- later pictures may predict from this one
    frame_num -- the sequence's picture counter, modulo the width the sequence
        parameter set declares for the field
    poc -- picture order count, which is display order; it counts by two a
        frame, the step the specification uses for a progressive frame
    idr_pic_id -- tells one IDR from the next, which a decoder needs when two
        arrive in a row
    references -- the frame numbers this picture predicts from, most recent
        first, which is the order reference list zero is initialised in
    """

    idr: bool
    intra: bool
    reference: bool
    frame_num: int
    poc: int
    idr_pic_id: int
    references: tuple[int, ...]


class GroupOfPictures:
    """Decides what each picture in a recording is coded as.

    Every picture is a reference and predicts from the one before it, with an
    IDR every `gop` frames and wherever a caller asks for one. Nothing is held
    back, so pictures leave the encoder in the order they arrived and a
    container needs no composition offsets.

    gop -- frames between IDR pictures; one makes every picture an IDR
    max_num_ref_frames -- how many pictures the decoded picture buffer holds,
        which the sequence parameter set must declare as the same number
    log2_max_frame_num -- the width of the frame_num field, which is where it
        wraps back to zero
    """

    def __init__(self, gop: int, max_num_ref_frames: int = 1,
                 log2_max_frame_num: int = LOG2_MAX_FRAME_NUM_MINUS4 + 4) -> None:
        if gop < 1:
            raise ValueError(f'a group of {gop} pictures holds none')
        if max_num_ref_frames < 1:
            raise ValueError(
                f'a decoded picture buffer of {max_num_ref_frames} leaves a '
                'predicted picture nothing to predict from')
        self.gop = int(gop)
        self.max_num_ref_frames = int(max_num_ref_frames)
        self.max_frame_num = 1 << int(log2_max_frame_num)
        self._position = 0
        self._idr_pic_id = 1
        self._buffer: list[int] = []

    def __repr__(self) -> str:
        return (f'<{type(self).__name__} gop {self.gop}, '
                f'{self.max_num_ref_frames} reference frames>')

    def next_picture(self, force_idr: bool = False) -> Picture:
        """The next picture to code, and what it may predict from.

        Calling this is what advances the structure, so it is called once per
        picture submitted and its answer is what the driver is told.
        """
        idr = force_idr or self._position % self.gop == 0
        if idr:
            self._position = 0
            self._buffer.clear()
            # Two IDRs in a row carrying the same identifier read as one picture
            # sent twice, so it alternates.
            self._idr_pic_id ^= 1
        picture = Picture(
            idr=idr,
            intra=idr,
            reference=True,
            frame_num=self._position % self.max_frame_num,
            poc=self._position * 2,
            idr_pic_id=self._idr_pic_id,
            references=tuple(self._buffer),
        )
        # Sliding window: the newest reference joins the buffer and the oldest
        # leaves it, which is what the specification's default marking does.
        self._buffer.insert(0, picture.frame_num)
        del self._buffer[self.max_num_ref_frames:]
        self._position += 1
        return picture
