"""The H.264 fuzz gate: mutated fixtures from the parser to the driver's buffers.

Every mutated stream either decodes or is refused with the package's own
error, in under a second, and whatever is parsed or handed to the session is
inside the range ITU-T H.264 gives it. The ranges are those of Appendix A of
``plans/SECURITY-REVIEW.md``.
"""
import pytest

from pyopengl_video.decoder import DecoderError
from pyopengl_video.vaapi import api, h264_decode
from tests import h26x_fuzz

CASES = 8000
SEED = 20261013

INVALID = api.VA_PICTURE_H264_INVALID
SLICE_P, SLICE_B, SLICE_I, SLICE_SP = 0, 1, 2, 3


# ------------------------------------------------------------ what is parsed


def check_sps(findings, sps):
    within = findings.within
    within('seq_parameter_set_id', sps.seq_parameter_set_id, 0, 31)
    within('chroma_format_idc', sps.chroma_format_idc, 0, 3)
    within('bit_depth_luma_minus8', sps.bit_depth_luma_minus8, 0, 6)
    within('bit_depth_chroma_minus8', sps.bit_depth_chroma_minus8, 0, 6)
    within('log2_max_frame_num_minus4', sps.log2_max_frame_num_minus4, 0, 12)
    within('log2_max_pic_order_cnt_lsb_minus4', sps.log2_max_pic_order_cnt_lsb_minus4, 0, 12)
    within('num_ref_frames_in_pic_order_cnt_cycle', len(sps.offset_for_ref_frame), 0, 255)
    for offset in (sps.offset_for_non_ref_pic, sps.offset_for_top_to_bottom_field,
                   *sps.offset_for_ref_frame):
        within('offset_for_*', offset, -2**31 + 1, 2**31 - 1)
    within('max_num_ref_frames', sps.max_num_ref_frames, 0, 16)
    within('PicSizeInMbs', sps.width_in_mbs * sps.frame_height_in_mbs, 1, 139264)
    if min(sps.size) < 1:
        findings.report('frame cropping leaves no picture', sps.size)
    vui = sps.vui
    if vui is not None and vui.max_dec_frame_buffering is not None:
        within('max_dec_frame_buffering', vui.max_dec_frame_buffering,
               sps.max_num_ref_frames, 16)
        within('max_num_reorder_frames', vui.max_num_reorder_frames, 0,
               vui.max_dec_frame_buffering)
    within('dpb_frames', sps.dpb_frames, 1, 16)
    within('reorder_frames', sps.reorder_frames, 0, sps.dpb_frames)


def check_pps(findings, pps):
    within = findings.within
    within('pic_parameter_set_id', pps.pic_parameter_set_id, 0, 255)
    within('pps seq_parameter_set_id', pps.seq_parameter_set_id, 0, 31)
    within('num_slice_groups_minus1', pps.num_slice_groups_minus1, 0, 7)
    within('slice_group_map_type', pps.slice_group_map_type, 0, 6)
    within('num_ref_idx_l0_default_active_minus1',
           pps.num_ref_idx_l0_default_active_minus1, 0, 31)
    within('num_ref_idx_l1_default_active_minus1',
           pps.num_ref_idx_l1_default_active_minus1, 0, 31)
    within('weighted_bipred_idc', pps.weighted_bipred_idc, 0, 2)
    within('pic_init_qp_minus26', pps.pic_init_qp_minus26, -(26 + 36), 25)
    within('pic_init_qs_minus26', pps.pic_init_qs_minus26, -26, 25)
    within('chroma_qp_index_offset', pps.chroma_qp_index_offset, -12, 12)
    within('second_chroma_qp_index_offset', pps.second_chroma_qp_index_offset, -12, 12)


def check_slice(findings, header, unit):
    within = findings.within
    sps, pps = header.sps, header.pps
    macroblocks = sps.width_in_mbs * sps.frame_height_in_mbs
    within('first_mb_in_slice', header.first_mb_in_slice, 0, macroblocks - 1)
    within('idr_pic_id', header.idr_pic_id, 0, 65535)
    within('redundant_pic_cnt', header.redundant_pic_cnt, 0, 127)
    if header.idr and header.slice_type not in (SLICE_I, 4):
        findings.report('an IDR picture with a predicted slice', header.kind)
    most = 31 if header.field_pic_flag else 15
    lists = []
    if header.slice_type in (SLICE_P, SLICE_SP, SLICE_B):
        lists.append((header.num_ref_idx_l0_active_minus1,
                      header.ref_pic_list_modification_l0))
    if header.slice_type == SLICE_B:
        lists.append((header.num_ref_idx_l1_active_minus1,
                      header.ref_pic_list_modification_l1))
    for active_minus1, modifications in lists:
        within('num_ref_idx_active_minus1', active_minus1, 0, most)
        within('reference list modifications', len(modifications), 0, active_minus1 + 1)
    table = header.pred_weight_table
    if table is not None:
        within('luma_log2_weight_denom', table.luma_log2_weight_denom, 0, 7)
        within('chroma_log2_weight_denom', table.chroma_log2_weight_denom, 0, 7)
        for weight, offset in (*table.luma_l0, *table.luma_l1):
            within('luma weight', weight, -128, 128)
            within('luma offset', offset, -128, 127)
        for pair in (*table.chroma_l0, *table.chroma_l1):
            for weight, offset in pair:
                within('chroma weight', weight, -128, 128)
                within('chroma offset', offset, -128, 127)
    within('marking operations', len(header.memory_management), 0, 66)
    for operation in header.memory_management:
        within('long_term_frame_idx', operation.long_term_frame_idx, 0,
               max(sps.max_num_ref_frames - 1, 0))
        within('max_long_term_frame_idx_plus1', operation.max_long_term_frame_idx_plus1,
               0, sps.max_num_ref_frames)
    within('cabac_init_idc', header.cabac_init_idc, 0, 2)
    within('SliceQPY', 26 + pps.pic_init_qp_minus26 + header.slice_qp_delta,
           -6 * sps.bit_depth_luma_minus8, 51)
    within('disable_deblocking_filter_idc', header.disable_deblocking_filter_idc, 0, 2)
    within('slice_alpha_c0_offset_div2', header.slice_alpha_c0_offset_div2, -6, 6)
    within('slice_beta_offset_div2', header.slice_beta_offset_div2, -6, 6)
    within('header_bits', header.header_bits, 8, min(0xFFFF, 8 * len(unit.data)))


# --------------------------------------------------- what the driver is handed


def check_picture(findings, session, surface, batches):
    """One submitted picture: its parameter buffer, then each slice's."""
    report, within = findings.report, findings.within
    parameters = batches[0][0][1]
    width = parameters.picture_width_in_mbs_minus1 + 1
    height = parameters.picture_height_in_mbs_minus1 + 1
    if (width * 16, height * 16) != session.format.coded_size:
        report('a picture of another size than its surfaces', (width, height))
    references = set()
    for entry in parameters.ReferenceFrames:
        if entry.flags & INVALID:
            if entry.picture_id != api.VA_INVALID_SURFACE:
                report('an unused ReferenceFrames entry naming a surface', entry.picture_id)
            continue
        references.add(entry.picture_id)
        if entry.picture_id == surface:
            report('a picture decoded into a surface it predicts from', surface)
        elif entry.picture_id not in session.decoded:
            report('a reference surface that was never decoded', entry.picture_id)
    within('num_ref_frames', parameters.num_ref_frames, 0, 16)
    within('driver pic_init_qp_minus26', parameters.pic_init_qp_minus26, -26, 25)
    within('driver pic_init_qs_minus26', parameters.pic_init_qs_minus26, -26, 25)
    within('driver chroma_qp_index_offset', parameters.chroma_qp_index_offset, -12, 12)
    within('driver second_chroma_qp_index_offset',
           parameters.second_chroma_qp_index_offset, -12, 12)
    sequence = parameters.seq_fields.bits
    within('driver log2_max_frame_num_minus4', sequence.log2_max_frame_num_minus4, 0, 12)
    within('driver log2_max_pic_order_cnt_lsb_minus4',
           sequence.log2_max_pic_order_cnt_lsb_minus4, 0, 12)
    within('driver weighted_bipred_idc', parameters.pic_fields.bits.weighted_bipred_idc, 0, 2)
    for (_, slice_), (_, data) in batches[1:]:
        if slice_.slice_data_size != len(data) or slice_.slice_data_offset:
            report('a slice that is not its data buffer', slice_.slice_data_size)
        within('slice_data_bit_offset', slice_.slice_data_bit_offset, 8, 8 * len(data))
        within('driver first_mb_in_slice', slice_.first_mb_in_slice, 0, width * height - 1)
        within('driver cabac_init_idc', slice_.cabac_init_idc, 0, 2)
        within('driver disable_deblocking_filter_idc',
               slice_.disable_deblocking_filter_idc, 0, 2)
        within('driver slice_alpha_c0_offset_div2', slice_.slice_alpha_c0_offset_div2, -6, 6)
        within('driver slice_beta_offset_div2', slice_.slice_beta_offset_div2, -6, 6)
        within('driver SliceQPY', 26 + parameters.pic_init_qp_minus26 + slice_.slice_qp_delta,
               0, 51)
        within('driver luma_log2_weight_denom', slice_.luma_log2_weight_denom, 0, 7)
        within('driver chroma_log2_weight_denom', slice_.chroma_log2_weight_denom, 0, 7)
        active = [0, 0]
        if slice_.slice_type in (SLICE_P, SLICE_SP, SLICE_B):
            active[0] = slice_.num_ref_idx_l0_active_minus1 + 1
        if slice_.slice_type == SLICE_B:
            active[1] = slice_.num_ref_idx_l1_active_minus1 + 1
        for count, entries in zip(active, (slice_.RefPicList0, slice_.RefPicList1),
                                  strict=True):
            within('driver num_ref_idx_active', count, 0, 16)
            for index, entry in enumerate(entries):
                unused = bool(entry.flags & INVALID)
                if unused != (entry.picture_id == api.VA_INVALID_SURFACE):
                    report('a reference list entry half marked unused', index)
                elif index < count and unused:
                    report('an active reference list entry with no surface', index)
                elif index >= count and not unused:
                    report('a reference list entry past the active count', index)
                elif not unused and entry.picture_id not in references:
                    report('a reference list entry absent from ReferenceFrames', index)


@pytest.fixture
def findings(monkeypatch):
    """A fuzz run of the H.264 layer, with every parsed structure checked."""
    found = h26x_fuzz.Findings()

    def checked(name, check, with_unit=False):
        parse = getattr(h264_decode, name)

        def parsing(*arguments):
            result = parse(*arguments)
            check(found, result, *arguments[:1] if with_unit else ())
            return result
        monkeypatch.setattr(h264_decode, name, parsing)

    checked('parse_sps', check_sps)
    checked('parse_pps', check_pps)
    checked('parse_slice_header', check_slice, with_unit=True)
    return found


def test_mutated_streams_are_decoded_or_refused_and_stay_in_range(findings):
    ran = h26x_fuzz.run(h264_decode.H264Decoding, h26x_fuzz.streams('*.h264'),
                        CASES, SEED, header=1, check_picture=check_picture,
                        findings=findings)
    assert not ran.found, f'\n{ran}'


def test_the_fixtures_themselves_pass_the_checks(findings):
    for name, units in h26x_fuzz.streams('*.h264').items():
        findings.case = name
        try:
            h26x_fuzz.decode(h264_decode.H264Decoding, units, findings, check_picture)
        except DecoderError:
            assert name == 'interlaced_mbaff.h264'
    assert not findings.found, f'\n{findings}'
