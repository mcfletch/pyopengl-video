"""AV1 encoding's control layer: what each frame is, and what the driver is told.

:mod:`pyopengl_video.vaapi.av1` writes the sequence header and the frame
headers and decides the group of pictures. :class:`AV1Encoding` turns those
decisions into the parameter buffers libva takes, keeps track of the
reconstructed picture each frame predicts from, and makes a temporal unit of
what the driver wrote. Nothing here holds a driver.

Drivers differ in how much of the stream they write. Intel's takes the packed
frame header as the header of the frame and rewrites the fields rate control
decides, at the offsets it is given. Mesa's reads the packed headers for the
values it has no parameter for (the picture size, the colour description, the
order hint, the reference slots, the render size) and writes its own sequence
and frame headers from them. Both are given the same buffers, and
:meth:`AV1Encoding.coded` reads the sequence header out of what came back.
"""
from __future__ import annotations

from pyopengl_video import codecs, obu
from pyopengl_video.encoder import EncoderError
from pyopengl_video.vaapi import api, av1_api
from pyopengl_video.vaapi.av1 import (
    OBU_SIZE_BYTES,
    ORDER_HINT_BITS,
    Frame,
    GroupOfPictures,
    PackedFrameHeader,
    Sequence,
    base_q_idx,
    loop_filter_level,
)
from pyopengl_video.vaapi.av1_parse import (
    DEFAULT_GM_PARAMS,
    EIGHTTAP,
    LAST_FRAME,
    NUM_REF_FRAMES,
)
from pyopengl_video.vaapi.control import (
    Buffer,
    CodedPicture,
    EncodeControl,
    EncodeSettings,
    packed_header,
    rate_control_buffers,
)
from pyopengl_video.vaapi.h264 import StreamError

__all__ = ['AV1Encoding', 'PACKED_HEADERS_WRITTEN', 'TX_MODE_SELECT_SUPPORTED']

#: The packed headers this layer supplies, and so the only ones it asks the
#: driver to expect: the sequence header OBU at each key frame and the frame
#: header of every frame.
PACKED_HEADERS_WRITTEN = (api.VA_ENC_PACKED_HEADER_SEQUENCE
                          | api.VA_ENC_PACKED_HEADER_PICTURE)

#: The bit of ``tx_mode_support`` that says a driver codes TX_MODE_SELECT.
TX_MODE_SELECT_SUPPORTED = 0x04

#: The OBUs that carry a frame, one of which every coded picture must hold.
_FRAME_UNITS = (obu.FRAME, obu.FRAME_HEADER)

_DELIMITER = obu.Obu(obu.TEMPORAL_DELIMITER, b'').encoded()


class AV1Encoding(EncodeControl):
    """AV1 Main, frames coded in display order: key and inter, each shown.

    stream -- the sequence-level choices, which writes the headers
    structure -- the group of pictures deciding what each frame is
    obu_size_bytes -- how many bytes the frame header's obu_size is written
        in, which is what the driver states it writes one in

    ``settings.qp`` is on H.264's scale and :func:`pyopengl_video.vaapi.av1.base_q_idx`
    converts it: five quantiser indices to one H.264 step.

    Every frame fills all eight reference slots, so an inter frame predicts
    from the frame before it whichever slot it names and one reconstructed
    picture is kept.
    """

    codec = codecs.AV1
    profile = api.VAProfileAV1Profile0
    title = 'AV1 Main'
    attributes = (api.VAConfigAttribEncPackedHeaders, av1_api.VAConfigAttribEncAV1Ext2)
    max_qp = 255
    references = 1

    def __init__(self, settings: EncodeSettings, stated: dict[int, int]) -> None:
        self.settings = settings
        # Asking for a packed header is a promise to write it, so only the two
        # this layer writes are asked for, and only where the driver takes them.
        self.packed_headers = stated.get(
            api.VAConfigAttribEncPackedHeaders, 0) & PACKED_HEADERS_WRITTEN
        limits = av1_api.VAConfigAttribValEncAV1Ext2()
        limits_stated = av1_api.VAConfigAttribEncAV1Ext2 in stated
        limits.value = stated.get(av1_api.VAConfigAttribEncAV1Ext2, 0)
        self.obu_size_bytes = (limits.bits.obu_size_bytes_minus1 + 1 if limits_stated
                               else OBU_SIZE_BYTES)
        width, height = settings.size
        try:
            self.stream = Sequence(
                width, height, settings.frame_rate, settings.bitrate,
                tx_mode_select=(not limits_stated or bool(
                    limits.bits.tx_mode_support & TX_MODE_SELECT_SUPPORTED)))
        except ValueError as error:
            raise EncoderError(f'AV1 cannot code this picture: {error}') from error
        # The surfaces are the picture's own size. A driver that codes whole
        # blocks pads the picture out to them itself.
        self.coded_size = (self.stream.width, self.stream.height)
        self.structure = GroupOfPictures(settings.gop, base_q_idx(settings.qp))
        self._reference: int | None = None
        self._coded_header = b''

    def next_picture(self, force_key: bool = False) -> CodedPicture:
        frame = self.structure.next_frame(force_key=force_key)
        kind = 'key' if frame.key else 'inter'
        return CodedPicture(keyframe=frame.key, name=f'{kind} frame {frame.number}',
                            detail=frame)

    def buffers(self, picture: CodedPicture, coded: int,
                reconstruction: int) -> list[Buffer]:
        frame: Frame = picture.detail
        found: list[Buffer] = []
        if frame.key:
            found.append((api.VAEncSequenceParameterBufferType, self.sequence_parameters()))
            found += rate_control_buffers(self.settings, frame.base_q_idx, self.max_qp)
            if self.packed_headers & api.VA_ENC_PACKED_HEADER_SEQUENCE:
                found += packed_header(av1_api.VAEncPackedHeaderAV1_SPS,
                                       self.stream.sequence_header(),
                                       has_emulation_bytes=False)
        header = None
        if self.packed_headers & api.VA_ENC_PACKED_HEADER_PICTURE:
            header = self.stream.frame_header(frame, size_bytes=self.obu_size_bytes)
        found.append((api.VAEncPictureParameterBufferType,
                      self.picture_parameters(frame, coded, reconstruction, header)))
        if header is not None:
            found += packed_header(av1_api.VAEncPackedHeaderAV1_PPS, header.data,
                                   has_emulation_bytes=False)
        found.append((api.VAEncSliceParameterBufferType, self.tile_group()))
        self._reference = reconstruction
        return found

    def sequence_parameters(self) -> av1_api.VAEncSequenceParameterBufferAV1:
        """The sequence parameters, which say what the whole stream is."""
        stream, settings = self.stream, self.settings
        parameters = av1_api.VAEncSequenceParameterBufferAV1()
        parameters.seq_profile = 0
        parameters.seq_level_idx = stream.level
        parameters.seq_tier = stream.tier
        parameters.intra_period = settings.gop
        parameters.ip_period = 1
        parameters.bits_per_second = settings.bitrate
        bits = parameters.seq_fields.bits
        bits.enable_order_hint = 1
        bits.enable_cdef = 1
        bits.bit_depth_minus8 = 0
        bits.subsampling_x = 1
        bits.subsampling_y = 1
        parameters.order_hint_bits_minus_1 = ORDER_HINT_BITS - 1
        return parameters

    def picture_parameters(self, frame: Frame, coded: int, reconstruction: int,
                           header: PackedFrameHeader | None = None,
                           ) -> av1_api.VAEncPictureParameterBufferAV1:
        """What the driver is told about the frame it is about to code.

        header -- the packed frame header sent with it, whose offsets say
            where the driver writes what it decides; None where none is sent
        """
        stream = self.stream
        parameters = av1_api.VAEncPictureParameterBufferAV1()
        parameters.frame_width_minus_1 = stream.width - 1
        parameters.frame_height_minus_1 = stream.height - 1
        parameters.reconstructed_frame = reconstruction
        parameters.coded_buf = coded
        # Every slot holds the frame before, or nothing at a key frame.
        held = api.VA_INVALID_SURFACE if frame.key or self._reference is None \
            else self._reference
        for slot in range(NUM_REF_FRAMES):
            parameters.reference_frames[slot] = held
        for index, slot in enumerate(frame.ref_frame_idx):
            parameters.ref_frame_idx[index] = slot
        parameters.primary_ref_frame = frame.primary_ref_frame
        parameters.order_hint = frame.order_hint
        parameters.refresh_frame_flags = frame.refresh_frame_flags
        if not frame.key:
            parameters.ref_frame_ctrl_l0.fields.search_idx0 = LAST_FRAME

        flags = parameters.picture_flags.bits
        flags.frame_type = frame.frame_type
        # A shown key frame is error resilient by definition.
        flags.error_resilient_mode = 1 if frame.key else 0
        flags.enable_frame_obu = 1
        parameters.tile_group_obu_hdr_info.bits.obu_has_size_field = 1

        level = loop_filter_level(frame.base_q_idx)
        parameters.filter_level[0] = level
        parameters.filter_level[1] = level
        parameters.filter_level_u = level
        parameters.filter_level_v = level
        parameters.interpolation_filter = EIGHTTAP
        parameters.base_qindex = frame.base_q_idx
        parameters.min_base_qindex = 1
        parameters.max_base_qindex = self.max_qp
        parameters.mode_control_flags.bits.tx_mode = stream.tx_mode

        parameters.tile_cols = stream.tile_cols
        parameters.tile_rows = stream.tile_rows
        for index, extent in enumerate(stream.width_in_sbs_minus_1):
            parameters.width_in_sbs_minus_1[index] = extent
        for index, extent in enumerate(stream.height_in_sbs_minus_1):
            parameters.height_in_sbs_minus_1[index] = extent
        for model in parameters.wm:
            model.wmtype = av1_api.VAAV1EncTransformationIdentity
            for index, value in enumerate(DEFAULT_GM_PARAMS):
                model.wmmat[index] = value

        if header is not None:
            parameters.byte_offset_frame_hdr_obu_size = header.byte_offset_obu_size
            if self.settings.rate_control != api.VA_RC_CQP:
                # Rate control decides the quantiser, and the loop filter and
                # CDEF strengths that follow from it, after the header is written.
                parameters.bit_offset_qindex = header.bit_offset_qindex
                parameters.bit_offset_segmentation = header.bit_offset_segmentation
                parameters.bit_offset_loopfilter_params = header.bit_offset_loopfilter_params
                parameters.bit_offset_cdef_params = header.bit_offset_cdef_params
                parameters.size_in_bits_cdef_params = header.size_in_bits_cdef_params
                parameters.size_in_bits_frame_hdr_obu = header.size_in_bits
        return parameters

    def tile_group(self) -> av1_api.VAEncTileGroupBufferAV1:
        """The one tile group each frame is coded as, holding every tile."""
        group = av1_api.VAEncTileGroupBufferAV1()
        group.tg_start = 0
        group.tg_end = self.stream.tile_cols * self.stream.tile_rows - 1
        return group

    def coded(self, picture: CodedPicture, data: bytes) -> bytes:
        """The temporal unit for `picture`: a delimiter, then what the driver wrote.

        Each OBU has a size field, which is the form MP4, IVF and a decoder
        take. A key frame's unit carries the sequence header: the driver's
        where it wrote one, since that is the one describing the frames it
        coded, and this layer's otherwise. Padding OBUs and a temporal
        delimiter of the driver's own are left out.
        """
        try:
            units = [unit for unit in obu.split(data)
                     if unit.kind not in (obu.TEMPORAL_DELIMITER, obu.PADDING)]
        except StreamError as error:
            raise EncoderError(
                f'the driver wrote something other than OBUs for {picture.name}: '
                f'{error}') from error
        if not any(unit.kind in _FRAME_UNITS for unit in units):
            raise EncoderError(f'the driver wrote no frame for {picture.name}')
        written = [unit for unit in units if unit.kind == obu.SEQUENCE_HEADER]
        header = b''
        if picture.keyframe:
            header = written[0].encoded() if written else self.stream.sequence_header()
            self._coded_header = header
        return _DELIMITER + header + obu.join(
            unit for unit in units if unit.kind != obu.SEQUENCE_HEADER)

    def headers(self) -> bytes:
        """The sequence header OBU, which is what an ``av1C`` box carries.

        Before the first key frame is coded this is the header this layer
        wrote. Afterwards it is the one the stream carries.
        """
        return self._coded_header or self.stream.sequence_header()
