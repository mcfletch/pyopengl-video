"""HEVC encoding's control layer: what each picture is, and what the driver is told.

:mod:`pyopengl_video.vaapi.hevc` writes the parameter sets and slice segment
headers and decides the group of pictures. :class:`HEVCEncoding` reads the
driver's attributes into the block sizes and coding tools those headers
declare, turns each picture into the parameter buffers libva takes, and keeps
track of the reconstructed pictures each one predicts from. Nothing here
holds a driver.

A driver is given every header twice: as the fields of its parameter buffers
and as packed headers, the bitstream itself. Which of the two it codes from
is the driver's choice. Mesa's AMD encoder reads the values out of the packed
headers and writes its own parameter sets and slice segment headers from
them, with the block sizes and the picture size its hardware codes; the
conformance window, the picture order count and the reference picture set
reach it through the packed headers alone. :meth:`HEVCEncoding.headers`
therefore returns the sets a coded key picture carries.
"""
from __future__ import annotations

from pyopengl_video import codecs
from pyopengl_video.encoder import EncoderError
from pyopengl_video.mp4 import split_annexb
from pyopengl_video.vaapi import api, hevc_api
from pyopengl_video.vaapi.control import (
    Buffer,
    CodedPicture,
    EncodeControl,
    EncodeSettings,
    packed_header,
    rate_control_buffers,
)
from pyopengl_video.vaapi.h264 import annexb
from pyopengl_video.vaapi.hevc import (
    MAX_NUM_MERGE_CAND,
    Blocks,
    GroupOfPictures,
    ParameterSets,
    Picture,
    Tools,
)
from pyopengl_video.vaapi.hevc_parse import (
    NAL_PPS,
    NAL_SPS,
    NAL_VPS,
    PROFILE_MAIN,
    SLICE_B,
)

__all__ = ['HEVCEncoding', 'HEIGHT_ALIGNMENT', 'MAX_REFERENCES',
           'PACKED_HEADERS_WRITTEN', 'blocks_from', 'packed_headers_for', 'tools_from']

#: The packed headers this layer supplies, and so the only ones it asks the
#: driver to expect: the video and sequence parameter sets as a sequence
#: header, the picture parameter set as a picture header, and each slice
#: segment header.
PACKED_HEADERS_WRITTEN = (api.VA_ENC_PACKED_HEADER_SEQUENCE
                          | api.VA_ENC_PACKED_HEADER_PICTURE
                          | api.VA_ENC_PACKED_HEADER_SLICE)

_PARAMETER_SETS = api.VA_ENC_PACKED_HEADER_SEQUENCE | api.VA_ENC_PACKED_HEADER_PICTURE

#: The most pictures kept to predict from. A decoded picture buffer holds six
#: pictures at every level whatever the picture size (A.4.2), the one being
#: decoded among them.
MAX_REFERENCES = 5

#: What the coded height is a multiple of, where the smallest coding block is
#: smaller. The coded width is a whole number of coding tree blocks. Mesa's
#: AMD encoder writes a sequence parameter set with a size rounded up this
#: way whatever size it is given, and takes the conformance window as it is
#: stated, so a size rounded any other way is cropped to the wrong picture.
HEIGHT_ALIGNMENT = 16

#: The coding tools this layer turns on wherever the driver has them.
_PREFERRED = ('amp', 'sao', 'strong_intra_smoothing')

#: The tools whose syntax this layer writes, by the name
#: ``VAConfigAttribValEncHEVCFeatures`` gives each and the name
#: :class:`~pyopengl_video.vaapi.hevc.Tools` does.
_WRITTEN = {
    'amp': 'amp',
    'sao': 'sample_adaptive_offset',
    'strong_intra_smoothing': 'strong_intra_smoothing',
    'sign_data_hiding': 'sign_data_hiding',
    'constrained_intra_pred': 'constrained_intra_pred',
    'transform_skip': 'transform_skip',
    'cu_qp_delta': 'cu_qp_delta',
    'transquant_bypass': 'transquant_bypass',
}

#: The tools it does not, in words. A driver that always codes with one is refused.
_NOT_WRITTEN = {
    'separate_colour_planes': 'separately coded colour planes',
    'scaling_lists': 'scaling lists',
    'pcm': 'PCM coding blocks',
    'temporal_mvp': 'temporal motion vector prediction',
    'dependent_slices': 'dependent slice segments',
    'weighted_prediction': 'weighted prediction',
    'deblocking_filter_disable': 'a disabled deblocking filter',
}


def packed_headers_for(offered: int) -> int:
    """The packed headers to ask a driver for, of those it offers.

    offered -- ``VAConfigAttribEncPackedHeaders`` as the driver states it

    A slice segment header is read against the parameter sets, so a packed
    one goes only to a driver that also takes those packed. Mesa reads a
    packed slice segment header by the last packed sets it was given, and one
    it has no sets for it reads past the end of, without returning.
    """
    wanted = offered & PACKED_HEADERS_WRITTEN
    if wanted & _PARAMETER_SETS != _PARAMETER_SETS:
        wanted &= ~api.VA_ENC_PACKED_HEADER_SLICE
    return wanted


def tools_from(features: int | None, rate_controlled: bool) -> Tools:
    """The coding tools to declare, from ``VAConfigAttribEncHEVCFeatures``.

    features -- the attribute's value, or None where the driver states none,
        which leaves every optional tool off
    rate_controlled -- the driver chooses the quantiser, so it has to be able
        to change it within a picture where it can

    A tool the driver always uses is declared, one it may use is declared
    where this layer prefers it, and one it never uses is left off.
    :class:`~pyopengl_video.encoder.EncoderError` names a tool the driver
    always uses whose syntax is not written here.
    """
    if features is None:
        return Tools()
    stated = hevc_api.VAConfigAttribValEncHEVCFeatures(value=features).bits
    for name, words in _NOT_WRITTEN.items():
        if getattr(stated, name) == hevc_api.VA_FEATURE_REQUIRED:
            raise EncoderError(
                f'the driver\'s HEVC encoder always codes with {words}, which '
                'the VA-API backend does not write')
    wanted = set(_PREFERRED) | ({'cu_qp_delta'} if rate_controlled else set())
    chosen = {}
    for name, field in _WRITTEN.items():
        support = getattr(stated, name)
        chosen[field] = (support == hevc_api.VA_FEATURE_REQUIRED
                         or (support == hevc_api.VA_FEATURE_SUPPORTED and name in wanted))
    return Tools(**chosen)


def blocks_from(sizes: int | None) -> Blocks:
    """The block sizes to code with, from ``VAConfigAttribEncHEVCBlockSizes``.

    sizes -- the attribute's value, or None where the driver states none,
        which gives :class:`~pyopengl_video.vaapi.hevc.Blocks` as it stands

    The largest coding tree block and transform hierarchy depths the driver
    offers, and the whole range of its coding and transform blocks.
    """
    if sizes is None:
        return Blocks()
    stated = hevc_api.VAConfigAttribValEncHEVCBlockSizes(value=sizes).bits
    try:
        return Blocks(
            coding_tree=stated.log2_max_coding_tree_block_size_minus3 + 3,
            min_coding=stated.log2_min_luma_coding_block_size_minus3 + 3,
            max_transform=stated.log2_max_luma_transform_block_size_minus2 + 2,
            min_transform=stated.log2_min_luma_transform_block_size_minus2 + 2,
            depth_inter=stated.max_max_transform_hierarchy_depth_inter,
            depth_intra=stated.max_max_transform_hierarchy_depth_intra)
    except ValueError as error:
        raise EncoderError(
            f'the driver\'s HEVC encoder states block sizes ({sizes:#x}) that '
            f'no Main profile stream has: {error}') from error


class HEVCEncoding(EncodeControl):
    """HEVC Main, pictures coded in display order: IDR and P, one slice each.

    sets -- the parameter sets and slice segment headers of the stream
    structure -- the group of pictures deciding what each picture is
    bidirectional -- predicted pictures are coded as B pictures whose second
        reference list repeats the first, which is what a driver stating
        ``VA_PREDICTION_DIRECTION_BI_NOT_EMPTY`` codes in place of P pictures
    """

    codec = codecs.HEVC
    profile = api.VAProfileHEVCMain
    title = 'HEVC Main'
    attributes = (api.VAConfigAttribEncPackedHeaders, api.VAConfigAttribEncMaxRefFrames,
                  hevc_api.VAConfigAttribEncHEVCFeatures,
                  hevc_api.VAConfigAttribEncHEVCBlockSizes,
                  hevc_api.VAConfigAttribPredictionDirection)

    def __init__(self, settings: EncodeSettings, stated: dict[int, int]) -> None:
        self.settings = settings
        self.packed_headers = packed_headers_for(
            stated.get(api.VAConfigAttribEncPackedHeaders, 0))
        direction = stated.get(hevc_api.VAConfigAttribPredictionDirection, 0)
        self.bidirectional = bool(direction & hevc_api.VA_PREDICTION_DIRECTION_BI_NOT_EMPTY)
        # The size of reference list zero is the low half and of list one the
        # high half. Both lists of a B picture here hold the same pictures.
        offered = stated.get(api.VAConfigAttribEncMaxRefFrames, 1)
        lists = [offered & 0xFFFF] + ([offered >> 16] if self.bidirectional else [])
        self.references = max(1, min(*lists, MAX_REFERENCES))
        blocks = blocks_from(stated.get(hevc_api.VAConfigAttribEncHEVCBlockSizes))
        tools = tools_from(stated.get(hevc_api.VAConfigAttribEncHEVCFeatures),
                           rate_controlled=settings.rate_control != api.VA_RC_CQP)
        width, height = settings.size
        try:
            self.sets = ParameterSets(
                width, height, settings.frame_rate, references=self.references,
                init_qp=settings.qp, bitrate=settings.bitrate, blocks=blocks,
                tools=tools, bidirectional=self.bidirectional,
                alignment=(1 << blocks.coding_tree,
                           max(1 << blocks.min_coding, HEIGHT_ALIGNMENT)))
            self.structure = GroupOfPictures(
                gop=settings.gop, references=self.references,
                bidirectional=self.bidirectional)
        except ValueError as error:
            raise EncoderError(f'HEVC cannot code this: {error}') from error
        self.coded_size = self.sets.coded_size
        self._surfaces: dict[int, int] = {}
        self._coded_sets: dict[int, bytes] = {}

    def next_picture(self, force_key: bool = False) -> CodedPicture:
        picture = self.structure.next_picture(force_idr=force_key)
        return CodedPicture(keyframe=picture.idr, name=f'picture {picture.poc}',
                            detail=picture)

    def buffers(self, picture: CodedPicture, coded: int,
                reconstruction: int) -> list[Buffer]:
        detail: Picture = picture.detail
        found: list[Buffer] = []
        if detail.idr:
            self._surfaces.clear()
            found.append((api.VAEncSequenceParameterBufferType, self.sequence_parameters()))
            found += rate_control_buffers(self.settings, self.settings.qp, self.max_qp)
            if self.packed_headers & api.VA_ENC_PACKED_HEADER_SEQUENCE:
                found += packed_header(hevc_api.VAEncPackedHeaderHEVC_SPS,
                                       *self.sets.packed_sequence())
        found.append((api.VAEncPictureParameterBufferType,
                      self.picture_parameters(detail, coded, reconstruction)))
        if detail.idr and self.packed_headers & api.VA_ENC_PACKED_HEADER_PICTURE:
            found += packed_header(hevc_api.VAEncPackedHeaderHEVC_PPS,
                                   *self.sets.packed_picture())
        if self.packed_headers & api.VA_ENC_PACKED_HEADER_SLICE:
            found += packed_header(hevc_api.VAEncPackedHeaderHEVC_Slice,
                                   *self.sets.packed_slice_header(detail))
        found.append((api.VAEncSliceParameterBufferType, self.slice_parameters(detail)))
        self._remember(detail, reconstruction)
        return found

    def _remember(self, picture: Picture, reconstruction: int) -> None:
        """Record what a submitted picture reconstructed into."""
        self._surfaces[picture.poc] = reconstruction
        for poc in [poc for poc in self._surfaces if poc <= picture.poc - self.references]:
            del self._surfaces[poc]

    def held(self) -> set[int]:
        """The surfaces holding the pictures the next picture may predict from."""
        return set(self._surfaces.values())

    def reference_pictures(self, picture: Picture) -> list[hevc_api.VAPictureHEVC]:
        """The pictures `picture` predicts from as the driver is given them, nearest first."""
        return [
            hevc_api.VAPictureHEVC(
                picture_id=self._surfaces[poc], pic_order_cnt=poc,
                flags=hevc_api.VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE)
            for poc in picture.references]

    def sequence_parameters(self) -> hevc_api.VAEncSequenceParameterBufferHEVC:
        """The sequence parameters, which say what the whole stream is."""
        sets, settings = self.sets, self.settings
        blocks, tools = sets.blocks, sets.tools
        parameters = hevc_api.VAEncSequenceParameterBufferHEVC()
        parameters.general_profile_idc = PROFILE_MAIN
        parameters.general_level_idc = sets.level_idc
        parameters.general_tier_flag = 0
        parameters.intra_period = settings.gop
        parameters.intra_idr_period = settings.gop
        parameters.ip_period = 1
        parameters.bits_per_second = settings.bitrate
        parameters.pic_width_in_luma_samples, parameters.pic_height_in_luma_samples = (
            sets.coded_size)
        bits = parameters.seq_fields.bits
        bits.chroma_format_idc = 1
        bits.strong_intra_smoothing_enabled_flag = tools.strong_intra_smoothing
        bits.amp_enabled_flag = tools.amp
        bits.sample_adaptive_offset_enabled_flag = tools.sample_adaptive_offset
        bits.low_delay_seq = 1
        parameters.log2_min_luma_coding_block_size_minus3 = blocks.min_coding - 3
        parameters.log2_diff_max_min_luma_coding_block_size = (
            blocks.coding_tree - blocks.min_coding)
        parameters.log2_min_transform_block_size_minus2 = blocks.min_transform - 2
        parameters.log2_diff_max_min_transform_block_size = (
            blocks.max_transform - blocks.min_transform)
        parameters.max_transform_hierarchy_depth_inter = blocks.depth_inter
        parameters.max_transform_hierarchy_depth_intra = blocks.depth_intra

        parameters.vui_parameters_present_flag = 1
        vui = parameters.vui_fields.bits
        vui.aspect_ratio_info_present_flag = 1
        vui.vui_timing_info_present_flag = 1
        parameters.aspect_ratio_idc = 1
        parameters.vui_num_units_in_tick = settings.frame_rate[1]
        parameters.vui_time_scale = settings.frame_rate[0]
        return parameters

    def picture_parameters(self, picture: Picture, coded: int, reconstruction: int,
                           ) -> hevc_api.VAEncPictureParameterBufferHEVC:
        """What the driver is told about the picture it is about to code."""
        sets = self.sets
        tools = sets.tools
        parameters = hevc_api.VAEncPictureParameterBufferHEVC()
        parameters.decoded_curr_pic = hevc_api.VAPictureHEVC(
            picture_id=reconstruction, pic_order_cnt=picture.poc, flags=0)
        for index in range(hevc_api.REFERENCE_FRAMES):
            parameters.reference_frames[index] = hevc_api.VAPictureHEVC.unused()
        for index, entry in enumerate(self.reference_pictures(picture)):
            parameters.reference_frames[index] = entry
        parameters.coded_buf = coded
        parameters.collocated_ref_pic_index = hevc_api.NO_REFERENCE
        parameters.pic_init_qp = sets.init_qp
        parameters.num_ref_idx_l0_default_active_minus1 = sets.references - 1
        parameters.num_ref_idx_l1_default_active_minus1 = (
            sets.references - 1 if self.bidirectional else 0)
        parameters.nal_unit_type = picture.nal_unit_type
        bits = parameters.pic_fields.bits
        bits.idr_pic_flag = picture.idr
        if picture.intra:
            bits.coding_type = hevc_api.CODING_TYPE_I
        elif picture.bidirectional:
            bits.coding_type = hevc_api.CODING_TYPE_B
        else:
            bits.coding_type = hevc_api.CODING_TYPE_P
        bits.reference_pic_flag = 1
        bits.sign_data_hiding_enabled_flag = tools.sign_data_hiding
        bits.constrained_intra_pred_flag = tools.constrained_intra_pred
        bits.transform_skip_enabled_flag = tools.transform_skip
        bits.cu_qp_delta_enabled_flag = tools.cu_qp_delta
        bits.transquant_bypass_enabled_flag = tools.transquant_bypass
        return parameters

    def slice_parameters(self, picture: Picture) -> hevc_api.VAEncSliceParameterBufferHEVC:
        """The one slice segment each picture is coded as, and its reference lists."""
        sets = self.sets
        parameters = hevc_api.VAEncSliceParameterBufferHEVC()
        parameters.slice_segment_address = 0
        parameters.num_ctu_in_slice = sets.ctbs
        parameters.slice_type = picture.slice_type
        parameters.slice_pic_parameter_set_id = 0
        for index in range(hevc_api.REFERENCE_FRAMES):
            parameters.ref_pic_list0[index] = hevc_api.VAPictureHEVC.unused()
            parameters.ref_pic_list1[index] = hevc_api.VAPictureHEVC.unused()
        references = self.reference_pictures(picture)
        bidirectional = picture.slice_type == SLICE_B
        for index, entry in enumerate(references):
            parameters.ref_pic_list0[index] = entry
            if bidirectional:
                parameters.ref_pic_list1[index] = entry
        active_minus1 = max(len(references), 1) - 1
        parameters.num_ref_idx_l0_active_minus1 = active_minus1
        parameters.num_ref_idx_l1_active_minus1 = active_minus1 if bidirectional else 0
        parameters.max_num_merge_cand = MAX_NUM_MERGE_CAND
        parameters.slice_qp_delta = 0
        bits = parameters.slice_fields.bits
        bits.last_slice_of_pic_flag = 1
        bits.slice_sao_luma_flag = sets.tools.sample_adaptive_offset
        bits.slice_sao_chroma_flag = sets.tools.sample_adaptive_offset
        bits.num_ref_idx_active_override_flag = bool(
            references and len(references) != sets.references)
        bits.collocated_from_l0_flag = 1
        return parameters

    def coded(self, picture: CodedPicture, data: bytes) -> bytes:
        """Keep the parameter sets a coded key picture turned out to carry.

        A driver may write its own sets in place of the packed ones, so the
        sets in the stream are the ones that describe it, and those are what
        :meth:`headers` returns.
        """
        if picture.keyframe:
            for unit in split_annexb(data):
                kind = (unit[0] >> 1) & 0x3F
                if kind in (NAL_VPS, NAL_SPS, NAL_PPS):
                    self._coded_sets[kind] = unit
        return data

    def headers(self) -> bytes:
        """The video, sequence and picture parameter sets, as an Annex-B fragment.

        The sets this layer built until a key picture has been coded, and the
        ones the stream carries from then on.
        """
        if len(self._coded_sets) == 3:
            return annexb(*(self._coded_sets[kind] for kind in (NAL_VPS, NAL_SPS, NAL_PPS)))
        return self.sets.headers()
