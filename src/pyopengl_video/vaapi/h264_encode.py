"""H.264 encoding's control layer: what each picture is, and what the driver is told.

:mod:`pyopengl_video.vaapi.h264` writes the parameter sets and slice headers
and decides the group of pictures. :class:`H264Encoding` turns those decisions
into the parameter buffers libva takes and keeps track of the reconstructed
pictures each one predicts from. Nothing here holds a driver.
"""
from __future__ import annotations

from pyopengl_video import codecs
from pyopengl_video.mp4 import split_annexb
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.control import (
    Buffer,
    CodedPicture,
    EncodeControl,
    EncodeSettings,
    packed_header,
    rate_control_buffers,
)
from pyopengl_video.vaapi.h264 import (
    SLICE_TYPE_I,
    SLICE_TYPE_P,
    GroupOfPictures,
    ParameterSets,
    Picture,
    annexb,
)

__all__ = ['H264Encoding', 'PACKED_HEADERS_WRITTEN']

#: The packed headers this layer supplies, and so the only ones it asks the
#: driver to expect. Asking for one is a promise to write it every picture that
#: needs it, and the driver writes none of its own once asked.
#:
#: The slice header is in the list because a libva encoder need not write one:
#: Mesa's writes the coded macroblocks and nothing in front of them, so without
#: this every picture reaches the stream with no NAL header on it and no player
#: finds a picture at all. The sequence header carries both parameter sets,
#: which is how a driver taking one header per picture is given both.
PACKED_HEADERS_WRITTEN = (api.VA_ENC_PACKED_HEADER_SEQUENCE
                          | api.VA_ENC_PACKED_HEADER_SLICE)

NAL_SPS = 7
NAL_PPS = 8


class H264Encoding(EncodeControl):
    """H.264 High, pictures coded in display order: I and P, one slice each.

    sets -- the parameter sets the stream opens with
    structure -- the group of pictures deciding what each picture is
    max_num_ref_frames -- how many pictures a P picture may predict from, which
        is what the driver offers
    """

    codec = codecs.H264
    profile = api.VAProfileH264High
    title = 'H.264 High'
    attributes = (api.VAConfigAttribEncPackedHeaders, api.VAConfigAttribEncMaxRefFrames)

    def __init__(self, settings: EncodeSettings, stated: dict[int, int]) -> None:
        self.settings = settings
        # Only the headers this layer actually writes are asked for. The
        # attribute states everything the driver *can* take, slice headers
        # included, and requesting one is a promise to supply it: a driver told
        # to expect packed slice headers stops writing its own, and every
        # picture then comes out as slice payload with no NAL header on it.
        self.packed_headers = stated.get(
            api.VAConfigAttribEncPackedHeaders, 0) & PACKED_HEADERS_WRITTEN
        # The low half is the size of reference list zero. AMD parts state one,
        # which is exactly what a stream of I and P pictures needs; asking for
        # more is refused by the driver rather than quietly reduced.
        reference_frames = stated.get(api.VAConfigAttribEncMaxRefFrames, 1)
        self.max_num_ref_frames = max(1, reference_frames & 0xFFFF)
        self.references = self.max_num_ref_frames
        width, height = settings.size
        self.sets = ParameterSets(
            width=width, height=height, frame_rate=settings.frame_rate,
            max_num_ref_frames=self.max_num_ref_frames, init_qp=settings.qp,
            bitrate=settings.bitrate, transform_8x8=True)
        self.coded_size = self.sets.coded_size
        self.structure = GroupOfPictures(
            gop=settings.gop, max_num_ref_frames=self.max_num_ref_frames,
            log2_max_frame_num=self.sets.log2_max_frame_num)
        self._references: list[tuple[int, int, int]] = []
        self._coded_sets: dict[int, bytes] = {}

    def next_picture(self, force_key: bool = False) -> CodedPicture:
        picture = self.structure.next_picture(force_idr=force_key)
        return CodedPicture(keyframe=picture.idr, name=f'picture {picture.frame_num}',
                            detail=picture)

    def buffers(self, picture: CodedPicture, coded: int,
                reconstruction: int) -> list[Buffer]:
        detail: Picture = picture.detail
        found: list[Buffer] = []
        if detail.idr:
            found.append((api.VAEncSequenceParameterBufferType, self.sequence_parameters()))
            found += rate_control_buffers(self.settings, self.settings.qp, self.max_qp)
            if self.packed_headers & api.VA_ENC_PACKED_HEADER_SEQUENCE:
                data, bits = self.sets.packed_parameter_sets()
                found += packed_header(api.VAEncPackedHeaderSequence, data, bits)
        found.append((api.VAEncPictureParameterBufferType,
                      self.picture_parameters(detail, coded, reconstruction)))
        if self.packed_headers & api.VA_ENC_PACKED_HEADER_SLICE:
            data, bits = self.sets.packed_slice_header(detail)
            found += packed_header(api.VAEncPackedHeaderSlice, data, bits)
        found.append((api.VAEncSliceParameterBufferType, self.slice_parameters(detail)))
        self._remember(detail, reconstruction)
        return found

    def _remember(self, picture: Picture, reconstruction: int) -> None:
        """Record what a submitted picture reconstructed into."""
        entry = (picture.frame_num, reconstruction, picture.poc)
        if picture.idr:
            self._references = [entry]
            return
        self._references = [entry] + [
            kept for kept in self._references if kept[0] in picture.references
        ][:self.max_num_ref_frames - 1]

    def sequence_parameters(self) -> api.VAEncSequenceParameterBufferH264:
        """The sequence parameters, which say what the whole stream is."""
        sets, settings = self.sets, self.settings
        parameters = api.VAEncSequenceParameterBufferH264()
        parameters.seq_parameter_set_id = 0
        parameters.level_idc = sets.level_idc
        parameters.intra_period = settings.gop
        parameters.intra_idr_period = settings.gop
        parameters.ip_period = 1
        parameters.bits_per_second = settings.bitrate
        parameters.max_num_ref_frames = self.max_num_ref_frames
        parameters.picture_width_in_mbs = sets.width_in_mbs
        parameters.picture_height_in_mbs = sets.height_in_mbs
        bits = parameters.seq_fields.bits
        bits.chroma_format_idc = 1
        bits.frame_mbs_only_flag = 1
        bits.direct_8x8_inference_flag = 1
        bits.log2_max_frame_num_minus4 = sets.log2_max_frame_num - 4
        bits.pic_order_cnt_type = 0
        bits.log2_max_pic_order_cnt_lsb_minus4 = sets.log2_max_poc_lsb - 4

        coded_width, coded_height = sets.coded_size
        right = (coded_width - sets.width) // 2
        bottom = (coded_height - sets.height) // 2
        parameters.frame_cropping_flag = 1 if (right or bottom) else 0
        parameters.frame_crop_right_offset = right
        parameters.frame_crop_bottom_offset = bottom

        parameters.vui_parameters_present_flag = 1
        vui = parameters.vui_fields.bits
        vui.aspect_ratio_info_present_flag = 1
        vui.timing_info_present_flag = 1
        vui.bitstream_restriction_flag = 1
        vui.motion_vectors_over_pic_boundaries_flag = 1
        vui.log2_max_mv_length_horizontal = 15
        vui.log2_max_mv_length_vertical = 15
        vui.fixed_frame_rate_flag = 1
        parameters.aspect_ratio_idc = 1
        parameters.num_units_in_tick = settings.frame_rate[1]
        parameters.time_scale = settings.frame_rate[0] * 2
        return parameters

    def picture_parameters(self, picture: Picture, coded: int,
                           reconstruction: int) -> api.VAEncPictureParameterBufferH264:
        """What the driver is told about the picture it is about to code."""
        parameters = api.VAEncPictureParameterBufferH264()
        parameters.CurrPic.picture_id = reconstruction
        parameters.CurrPic.frame_idx = picture.frame_num
        parameters.CurrPic.flags = 0
        parameters.CurrPic.TopFieldOrderCnt = picture.poc
        parameters.CurrPic.BottomFieldOrderCnt = picture.poc
        for index in range(16):
            parameters.ReferenceFrames[index] = api.VAPictureH264.unused()
        for index, entry in enumerate(self.reference_pictures(picture)):
            parameters.ReferenceFrames[index] = entry
        parameters.coded_buf = coded
        parameters.pic_parameter_set_id = 0
        parameters.seq_parameter_set_id = 0
        parameters.last_picture = 0
        parameters.frame_num = picture.frame_num
        parameters.pic_init_qp = self.settings.qp
        parameters.num_ref_idx_l0_active_minus1 = 0
        parameters.num_ref_idx_l1_active_minus1 = 0
        bits = parameters.pic_fields.bits
        bits.idr_pic_flag = 1 if picture.idr else 0
        bits.reference_pic_flag = 1 if picture.reference else 0
        bits.entropy_coding_mode_flag = 1
        bits.transform_8x8_mode_flag = 1
        bits.deblocking_filter_control_present_flag = 1
        return parameters

    def reference_pictures(self, picture: Picture) -> list[api.VAPictureH264]:
        """The decoded picture buffer as the driver is given it."""
        by_frame_num = {frame_num: (surface, poc)
                        for frame_num, surface, poc in self._references}
        entries = []
        for frame_num in picture.references:
            found = by_frame_num.get(frame_num)
            if found is None:
                continue
            surface, poc = found
            entries.append(api.VAPictureH264(
                picture_id=surface, frame_idx=frame_num,
                flags=api.VA_PICTURE_H264_SHORT_TERM_REFERENCE,
                TopFieldOrderCnt=poc, BottomFieldOrderCnt=poc))
        return entries

    def slice_parameters(self, picture: Picture) -> api.VAEncSliceParameterBufferH264:
        """The one slice each picture is coded as, and its reference list."""
        sets = self.sets
        parameters = api.VAEncSliceParameterBufferH264()
        parameters.macroblock_address = 0
        parameters.num_macroblocks = sets.width_in_mbs * sets.height_in_mbs
        parameters.slice_type = SLICE_TYPE_I if picture.intra else SLICE_TYPE_P
        parameters.pic_parameter_set_id = 0
        parameters.idr_pic_id = picture.idr_pic_id
        parameters.pic_order_cnt_lsb = picture.poc % (1 << sets.log2_max_poc_lsb)
        parameters.num_ref_idx_active_override_flag = 1
        parameters.num_ref_idx_l0_active_minus1 = 0
        parameters.num_ref_idx_l1_active_minus1 = 0
        for index in range(32):
            parameters.RefPicList0[index] = api.VAPictureH264.unused()
            parameters.RefPicList1[index] = api.VAPictureH264.unused()
        for index, entry in enumerate(self.reference_pictures(picture)):
            parameters.RefPicList0[index] = entry
        parameters.slice_qp_delta = 0
        parameters.disable_deblocking_filter_idc = 0
        return parameters

    def coded(self, picture: CodedPicture, data: bytes) -> bytes:
        """Keep the parameter sets a coded key picture turned out to carry.

        A driver may amend what it was handed, since a capability the hardware
        does not have is one the picture parameter set must not claim. The sets
        in the stream are the ones that describe it, and those are what
        :meth:`headers` reports.
        """
        if picture.keyframe:
            for unit in split_annexb(data):
                kind = unit[0] & 0x1F
                if kind in (NAL_SPS, NAL_PPS):
                    self._coded_sets[kind] = unit
        return data

    def headers(self) -> bytes:
        """The parameter sets, as an Annex-B fragment.

        Before the first picture is coded these are the sets this layer built.
        Afterwards they are the ones the stream carries, which is what a
        container has to describe the stream with.
        """
        if len(self._coded_sets) == 2:
            return annexb(self._coded_sets[NAL_SPS], self._coded_sets[NAL_PPS])
        return self.sets.headers()
