"""VP9 decoding's control layer: the buffers libva takes, and the reference slots.

:mod:`~pyopengl_video.vaapi.vp9_parse` reads the syntax. The functions here
turn a frame's header into the parameter buffers libva takes, and
:class:`VP9Decoding` keeps the eight reference slots, decides where each frame
is decoded and shows the ones the stream shows. Nothing here holds a driver.
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence

from pyopengl_video import codecs
from pyopengl_video.bounds import FIELD_16
from pyopengl_video.colour import Colour
from pyopengl_video.decoder import DecoderError, Sample
from pyopengl_video.vaapi import api, vp9_api
from pyopengl_video.vaapi.control import Buffer, Checked, DecodeControl, DecodeSession, Format
from pyopengl_video.vaapi.h264 import StreamError
from pyopengl_video.vaapi.vp9_parse import (
    INTRA_FRAME,
    MAX_SEGMENTS,
    NUM_REF_FRAMES,
    SEG_LVL_ALT_L,
    SEG_LVL_ALT_Q,
    SEG_LVL_REF_FRAME,
    SEG_LVL_SKIP,
    DecodingState,
    FrameHeader,
    colour_of,
    is_key_frame,
    parse_uncompressed_header,
    split_superframe,
)
from pyopengl_video.vaapi.vp9_tables import AC_QLOOKUP, DC_QLOOKUP

log = logging.getLogger(__name__)

__all__ = ['VP9Decoding', 'colour_of', 'filter_levels', 'picture_parameters',
           'profile_for', 'quantiser_scales', 'slice_parameters']

#: The libva profile each decoded VP9 profile runs under, and its bit depth.
VA_PROFILES = {
    0: (api.VAProfileVP9Profile0, 8),
    2: (api.VAProfileVP9Profile2, 10),
}

#: Chroma layouts by (subsampling_x, subsampling_y), for refusing one in words.
SAMPLINGS = {(1, 1): '4:2:0', (1, 0): '4:2:2', (0, 1): '4:4:0', (0, 0): '4:4:4'}

MAX_LOOP_FILTER = 63
MAX_Q_INDEX = 255


def _check_size(header: FrameHeader) -> None:
    """Refuse a frame larger than the 16 bits libva is told each dimension in.

    A VP9 frame is up to 65536 samples a side, and 65536 written to such a
    field reads back as 0.
    """
    if header.width > FIELD_16 or header.height > FIELD_16:
        raise DecoderError(
            f'the VP9 frame is {header.width}x{header.height}, and VA-API is told '
            f'of frames up to {FIELD_16} samples a side')


def profile_for(header: FrameHeader) -> int:
    """The libva profile a frame decodes under; refuses what is not decoded here."""
    colour = header.colour
    sampling = SAMPLINGS[(colour.subsampling_x, colour.subsampling_y)]
    if header.profile not in VA_PROFILES:
        raise DecoderError(
            f'the stream is VP9 profile {header.profile} ({sampling} at '
            f'{colour.bit_depth} bits), and VA-API decoding here covers profile 0 '
            '(8-bit 4:2:0) and profile 2 at 10 bits')
    profile, depth = VA_PROFILES[header.profile]
    if colour.bit_depth != depth:
        raise DecoderError(
            f'the stream is VP9 profile {header.profile} at {colour.bit_depth} bits, '
            'and this decoder takes 8 or 10')
    return profile


def picture_parameters(header: FrameHeader, slots: Sequence[int | None],
                       ) -> vp9_api.VADecPictureParameterBufferVP9:
    """The picture parameter buffer for the frame `header` opens.

    slots -- the surface in each of the eight reference slots before this
        frame replaces any, None where a slot is empty

    A frame of more than 65535 samples a side is a
    :class:`~pyopengl_video.decoder.DecoderError`.
    """
    _check_size(header)
    parameters = Checked(vp9_api.VADecPictureParameterBufferVP9())
    parameters.frame_width, parameters.frame_height = header.size
    for index, surface in enumerate(slots):
        parameters.reference_frames[index] = (
            api.VA_INVALID_SURFACE if surface is None else surface)
    colour = header.colour
    segmentation = header.segmentation
    bits = parameters.pic_fields.bits
    bits.subsampling_x = colour.subsampling_x
    bits.subsampling_y = colour.subsampling_y
    bits.frame_type = header.frame_type
    bits.show_frame = header.show_frame
    bits.error_resilient_mode = header.error_resilient_mode
    bits.intra_only = header.intra_only
    bits.allow_high_precision_mv = header.allow_high_precision_mv
    bits.mcomp_filter_type = header.interpolation_filter
    bits.frame_parallel_decoding_mode = header.frame_parallel_decoding_mode
    bits.reset_frame_context = header.reset_frame_context
    bits.refresh_frame_context = header.refresh_frame_context
    bits.frame_context_idx = header.context_index
    bits.segmentation_enabled = segmentation.enabled
    bits.segmentation_temporal_update = segmentation.temporal_update
    bits.segmentation_update_map = segmentation.update_map
    last, golden, alternate = header.ref_frame_idx
    bits.last_ref_frame = last
    bits.golden_ref_frame = golden
    bits.alt_ref_frame = alternate
    (bits.last_ref_frame_sign_bias, bits.golden_ref_frame_sign_bias,
     bits.alt_ref_frame_sign_bias) = header.ref_frame_sign_bias
    bits.lossless_flag = int(header.quantisation.lossless)
    parameters.filter_level = header.loop_filter.level
    parameters.sharpness_level = header.loop_filter.sharpness
    parameters.log2_tile_rows = header.tile_rows_log2
    parameters.log2_tile_columns = header.tile_cols_log2
    parameters.frame_header_length_in_bytes = header.uncompressed_size
    parameters.first_partition_size = header.header_size_in_bytes
    parameters.mb_segment_tree_probs[:] = segmentation.tree_probs
    parameters.segment_pred_probs[:] = segmentation.pred_probs
    parameters.profile = header.profile
    parameters.bit_depth = colour.bit_depth
    return parameters.target


def _clip(value: int, most: int) -> int:
    return max(0, min(most, value))


def _segment_value(header: FrameHeader, segment: int, feature: int, base: int,
                   most: int) -> int:
    """`base`, or what a segment's feature makes of it: replaced or added to."""
    segmentation = header.segmentation
    if not segmentation.active(segment, feature):
        return base
    data = segmentation.feature_data[segment][feature]
    return _clip(data if segmentation.abs_or_delta_update else base + data, most)


def filter_levels(header: FrameHeader, segment: int) -> tuple[tuple[int, int], ...]:
    """A segment's loop filter levels, by reference frame and then by mode (8.8.1).

    The reference frames run from ``INTRA_FRAME`` to ``ALTREF_FRAME``. With
    deltas in force an intra block has one level, the first of its pair. A
    frame whose level is 0 is not filtered, and every level is then 0.
    """
    loop_filter = header.loop_filter
    if not loop_filter.level:
        return ((0, 0),) * 4
    level = _segment_value(header, segment, SEG_LVL_ALT_L, loop_filter.level,
                           MAX_LOOP_FILTER)
    if not loop_filter.delta_enabled:
        return ((level, level),) * 4
    # Deltas count double once the frame's own level reaches 32.
    shift = loop_filter.level >> 5
    levels = []
    for reference, ref_delta in enumerate(loop_filter.ref_deltas):
        if reference == INTRA_FRAME:
            levels.append((_clip(level + (ref_delta << shift), MAX_LOOP_FILTER), 0))
            continue
        first, second = (
            _clip(level + (ref_delta << shift) + (mode_delta << shift), MAX_LOOP_FILTER)
            for mode_delta in loop_filter.mode_deltas)
        levels.append((first, second))
    return tuple(levels)


def quantiser_scales(header: FrameHeader, segment: int) -> tuple[int, int, int, int]:
    """A segment's dequantisation scales: luma AC and DC, then chroma AC and DC (8.6.1)."""
    quantisation = header.quantisation
    index = _segment_value(header, segment, SEG_LVL_ALT_Q, quantisation.base_q_idx,
                           MAX_Q_INDEX)
    dc = DC_QLOOKUP[header.colour.bit_depth]
    ac = AC_QLOOKUP[header.colour.bit_depth]
    return (ac[index],
            dc[_clip(index + quantisation.delta_q_y_dc, MAX_Q_INDEX)],
            ac[_clip(index + quantisation.delta_q_uv_ac, MAX_Q_INDEX)],
            dc[_clip(index + quantisation.delta_q_uv_dc, MAX_Q_INDEX)])


def slice_parameters(header: FrameHeader, size: int) -> vp9_api.VASliceParameterBufferVP9:
    """The slice parameter buffer for a frame of `size` bytes, headers included.

    With segmentation off, the first segment alone is filled and the rest are zero.
    """
    parameters = Checked(vp9_api.VASliceParameterBufferVP9())
    parameters.slice_data_size = size
    parameters.slice_data_offset = 0
    parameters.slice_data_flag = api.VA_SLICE_DATA_FLAG_ALL
    segmentation = header.segmentation
    for segment in range(MAX_SEGMENTS if segmentation.enabled else 1):
        target = parameters.seg_param[segment]
        fields = target.segment_flags.fields
        fields.segment_reference_enabled = int(
            segmentation.active(segment, SEG_LVL_REF_FRAME))
        fields.segment_reference = (
            segmentation.feature_data[segment][SEG_LVL_REF_FRAME]
            if fields.segment_reference_enabled else 0)
        fields.segment_reference_skipped = int(segmentation.active(segment, SEG_LVL_SKIP))
        for reference, levels in enumerate(filter_levels(header, segment)):
            target.filter_level[reference][:] = levels
        (target.luma_ac_quant_scale, target.luma_dc_quant_scale,
         target.chroma_ac_quant_scale, target.chroma_dc_quant_scale) = quantiser_scales(
             header, segment)
    return parameters.target


# ------------------------------------------------------------ the layer


@dataclasses.dataclass(frozen=True)
class _Stored:
    """A decoded frame in a reference slot."""

    surface: int
    size: tuple[int, int]
    colour: Colour


class VP9Decoding(DecodeControl):
    """VP9 decoding: profile 0 (8-bit 4:2:0) and profile 2 at 10 bits.

    Each sample is one frame or one superframe, as IVF, WebM and MP4 hold it.
    A frame is shown as soon as it is decoded, a hidden frame is decoded and
    kept for prediction, and a frame that shows an earlier one again gives the
    session the same surface a second time.

    Decoding starts at a key frame: the frames before the first one, and those
    after a :meth:`reset`, are dropped. A key frame may change the size, the
    profile or the bit depth. Any other frame whose size differs from the key
    frame's (reference scaling) is refused, and so is every frame after it
    until the next key frame. A frame that states another profile than its
    key frame's is refused the same way.

    A frame of more than 65535 samples a side is refused: libva is told each
    dimension in 16 bits.
    """

    codec = codecs.VP9

    def __init__(self, session: DecodeSession, output_slots: int) -> None:
        super().__init__(session, output_slots)
        self._format: Format | None = None
        #: The size, the sample layout and the profile the key frame stated.
        self._sequence: tuple[tuple[int, int], tuple[int, int, int], int] = (
            (0, 0), (8, 1, 1), 0)
        self._restart()

    def _restart(self) -> None:
        """Empty the slots and wait for a key frame."""
        self._state = DecodingState()
        self._slots: list[_Stored | None] = [None] * NUM_REF_FRAMES
        self._current: int | None = None
        self._waiting = True

    def configure(self, headers: bytes) -> None:
        """Accept a container's configuration, which decoding has no use for.

        A ``vpcC`` box restates the profile, bit depth and colour of the key
        frames, and those are read from the key frames themselves.
        """
        del headers

    def decode(self, sample: Sample) -> None:
        for data in split_superframe(sample.data):
            if self._waiting and not is_key_frame(data):
                log.debug('dropped a VP9 frame of sample %d: no key frame precedes it',
                          sample.index)
                continue
            try:
                self._frame(data, sample)
            except BaseException:
                # The slots no longer hold what the stream's next frames
                # predict from.
                self._restart()
                raise

    def _frame(self, data: bytes, sample: Sample) -> None:
        header = parse_uncompressed_header(data, self._state)
        if header.show_existing_frame:
            stored = self._slots[header.frame_to_show_map_idx]
            if stored is None:
                raise StreamError(
                    f'the stream shows reference slot {header.frame_to_show_map_idx}, '
                    'which holds no frame')
            self._show(stored, sample)
            return
        if header.key_frame:
            self._begin_sequence(header)
        else:
            self._check_sequence(header)
        self._waiting = False
        surface = self._current = self.session.acquire()
        slots = [None if stored is None else stored.surface for stored in self._slots]
        batches: list[list[Buffer]] = [
            [(api.VAPictureParameterBufferType, picture_parameters(header, slots))],
            [(api.VASliceParameterBufferType, slice_parameters(header, len(data))),
             (api.VASliceDataBufferType, data)],
        ]
        kind = 'key frame' if header.key_frame else 'frame'
        self.session.submit(surface, batches, f'{kind} of sample {sample.index}')
        stored = _Stored(surface, header.size, colour_of(header))
        for slot in range(NUM_REF_FRAMES):
            if header.refresh_frame_flags >> slot & 1:
                self._slots[slot] = stored
        if header.show_frame:
            self._show(stored, sample)
        self._current = None

    def _begin_sequence(self, header: FrameHeader) -> None:
        """Make the surfaces suit a key frame, and take its size and colour."""
        profile = profile_for(header)
        _check_size(header)
        width, height = header.size
        wanted = Format(
            profile=profile, coded_size=((width + 7) & ~7, (height + 7) & ~7),
            surfaces=NUM_REF_FRAMES + 1, depth=header.colour.bit_depth,
            name=f'VP9 profile {header.profile}')
        if wanted != self._format:
            # A key frame replaces every slot, so nothing of the format
            # before is wanted again.
            self._slots = [None] * NUM_REF_FRAMES
            self.session.activate(wanted)
            self._format = wanted
        self._sequence = (header.size, header.colour.layout, header.profile)
        self.size = header.size
        self.colour = colour_of(header)

    def _check_sequence(self, header: FrameHeader) -> None:
        """Refuse a frame that leaves the size, the samples or the profile of its key frame."""
        size, layout, profile = self._sequence
        if header.size != size:
            raise DecoderError(
                f'a VP9 frame is {header.width}x{header.height} where its key frame '
                f'was {size[0]}x{size[1]}: the stream changes size without a key '
                'frame (reference scaling), which this decoder does not decode')
        if header.colour.layout != layout:
            raise DecoderError(
                f'a VP9 frame has {header.colour.bit_depth} bits a sample where its '
                f'key frame had {layout[0]}: the stream changes its samples '
                'without a key frame, which this decoder does not decode')
        if header.profile != profile:
            raise DecoderError(
                f'a VP9 frame states profile {header.profile} where its key frame '
                f'stated profile {profile}: the stream changes profile without a '
                'key frame, which this decoder does not decode')

    def _show(self, stored: _Stored, sample: Sample) -> None:
        width, height = stored.size
        self.session.output(stored.surface, timestamp=sample.timestamp,
                            duration=sample.duration, visible=(0, 0, width, height),
                            colour=stored.colour)

    def flush(self) -> None:
        """Nothing: a VP9 frame is shown when it is decoded, and none is held back."""

    def reset(self) -> None:
        self._restart()

    def held(self) -> set[int]:
        found = {stored.surface for stored in self._slots if stored is not None}
        if self._current is not None:
            found.add(self._current)
        return found
