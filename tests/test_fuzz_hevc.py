"""The HEVC fuzz gate: mutated fixtures from the parser to the driver's buffers.

Every mutated stream either decodes or is refused with the package's own
error, in under a second, and whatever is parsed or handed to the session is
inside the range ITU-T H.265 gives it. The ranges are those of Appendix A of
``plans/SECURITY-REVIEW.md``.
"""
import pytest

from pyopengl_video.decoder import DecoderError
from pyopengl_video.vaapi import api, hevc_api, hevc_decode
from pyopengl_video.vaapi.h264 import StreamError
from tests import h26x_fuzz

CASES = 6000
SEED = 20261013

INVALID = hevc_api.VA_PICTURE_HEVC_INVALID
NONE = hevc_api.NO_REFERENCE
SLICE_B, SLICE_P, SLICE_I = 0, 1, 2

#: The widest and tallest picture of any level (A.4.1), in luma samples.
MOST_SIDE = 16888


# ------------------------------------------------------------ what is parsed


def check_rps(findings, rps, most, where):
    within = findings.within
    within(f'{where} reference picture set', rps.num_delta_pocs, 0, most)
    for delta in rps.delta_poc_s0:
        within(f'{where} DeltaPocS0', delta, -2**15, -1)
    for delta in rps.delta_poc_s1:
        within(f'{where} DeltaPocS1', delta, 1, 2**15 - 1)


def check_sps(findings, sps):
    within = findings.within
    within('sps_max_sub_layers_minus1', sps.sps_max_sub_layers_minus1, 0, 6)
    within('sps_seq_parameter_set_id', sps.sps_seq_parameter_set_id, 0, 15)
    within('bit_depth_luma_minus8', sps.bit_depth_luma_minus8, 0, 8)
    within('bit_depth_chroma_minus8', sps.bit_depth_chroma_minus8, 0, 8)
    within('log2_max_pic_order_cnt_lsb_minus4', sps.log2_max_pic_order_cnt_lsb_minus4, 0, 12)
    min_coding = sps.log2_min_luma_coding_block_size_minus3 + 3
    tree = sps.ctb_log2_size
    min_transform = sps.log2_min_luma_transform_block_size_minus2 + 2
    max_transform = min_transform + sps.log2_diff_max_min_luma_transform_block_size
    within('CtbLog2SizeY', tree, 4, 6)
    within('MinCbLog2SizeY', min_coding, 3, tree)
    within('MinTbLog2SizeY', min_transform, 2, min_coding - 1)
    within('MaxTbLog2SizeY', max_transform, min_transform, min(tree, 5))
    within('max_transform_hierarchy_depth_inter', sps.max_transform_hierarchy_depth_inter,
           0, tree - min_transform)
    within('max_transform_hierarchy_depth_intra', sps.max_transform_hierarchy_depth_intra,
           0, tree - min_transform)
    for side in sps.coded_size:
        within('picture side in luma samples', side, 1, MOST_SIDE)
        if side % (1 << min_coding):
            findings.report('a picture side that is not whole coding blocks', side)
    if min(sps.size) < 1:
        findings.report('the conformance window leaves no picture', sps.size)
    for buffering, reorder in zip(sps.sps_max_dec_pic_buffering_minus1,
                                  sps.sps_max_num_reorder_pics, strict=True):
        within('sps_max_dec_pic_buffering_minus1', buffering, 0, 15)
        within('sps_max_num_reorder_pics', reorder, 0, buffering)
    if sps.pcm_enabled_flag:
        within('PcmBitDepthY', sps.pcm_sample_bit_depth_luma_minus1 + 1, 1, sps.bit_depth)
        within('PcmBitDepthC', sps.pcm_sample_bit_depth_chroma_minus1 + 1, 1,
               8 + sps.bit_depth_chroma_minus8)
        smallest = sps.log2_min_pcm_luma_coding_block_size_minus3 + 3
        within('Log2MinIpcmCbSizeY', smallest, 3, min(tree, 5))
        within('Log2MaxIpcmCbSizeY',
               smallest + sps.log2_diff_max_min_pcm_luma_coding_block_size,
               smallest, min(tree, 5))
    within('num_long_term_ref_pics_sps', len(sps.lt_ref_pic_poc_lsb_sps), 0, 32)
    within('num_short_term_ref_pic_sets', len(sps.short_term_ref_pic_sets), 0, 64)
    for rps in sps.short_term_ref_pic_sets:
        check_rps(findings, rps, sps.sps_max_dec_pic_buffering_minus1[-1], 'sequence')


def check_pps(findings, pps):
    within = findings.within
    within('pps_pic_parameter_set_id', pps.pps_pic_parameter_set_id, 0, 63)
    within('pps_seq_parameter_set_id', pps.pps_seq_parameter_set_id, 0, 15)
    within('num_ref_idx_l0_default_active_minus1',
           pps.num_ref_idx_l0_default_active_minus1, 0, 14)
    within('num_ref_idx_l1_default_active_minus1',
           pps.num_ref_idx_l1_default_active_minus1, 0, 14)
    within('init_qp_minus26', pps.init_qp_minus26, -(26 + 48), 25)
    within('diff_cu_qp_delta_depth', pps.diff_cu_qp_delta_depth, 0, 3)
    within('pps_cb_qp_offset', pps.pps_cb_qp_offset, -12, 12)
    within('pps_cr_qp_offset', pps.pps_cr_qp_offset, -12, 12)
    within('pps_beta_offset_div2', pps.pps_beta_offset_div2, -6, 6)
    within('pps_tc_offset_div2', pps.pps_tc_offset_div2, -6, 6)
    within('log2_parallel_merge_level_minus2', pps.log2_parallel_merge_level_minus2, 0, 4)


def check_slice(findings, header, unit):
    within = findings.within
    sps, pps = header.sps, header.pps
    blocks = sps.width_in_ctbs * sps.height_in_ctbs
    within('slice_segment_address', header.slice_segment_address, 0, blocks - 1)
    within('slice_type', header.slice_type, 0, 2)
    if header.irap and header.slice_type != SLICE_I:
        findings.report('an intra random access point with a predicted slice', header.kind)
    most = sps.sps_max_dec_pic_buffering_minus1[-1]
    check_rps(findings, header.short_term_ref_pic_set, most, 'slice')
    within('pictures in the reference picture set',
           header.short_term_ref_pic_set.num_delta_pocs + len(header.long_term), 0, most)
    if header.slice_type != SLICE_I:
        within('num_ref_idx_l0_active_minus1', header.num_ref_idx_l0_active_minus1, 0, 14)
        within('num_ref_idx_l1_active_minus1', header.num_ref_idx_l1_active_minus1, 0, 14)
        within('five_minus_max_num_merge_cand', header.five_minus_max_num_merge_cand, 0, 4)
        if header.collocated_ref_idx is not None:
            within('collocated_ref_idx', header.collocated_ref_idx, 0,
                   header.num_ref_idx_l0_active_minus1 if header.collocated_from_l0_flag
                   else header.num_ref_idx_l1_active_minus1)
        for entry in (*header.list_entry_l0, *header.list_entry_l1):
            within('list_entry', entry, 0, header.num_pic_total_curr - 1)
    within('init_qp_minus26 for the sequence', pps.init_qp_minus26,
           -(26 + 6 * sps.bit_depth_luma_minus8), 25)
    within('SliceQpY', 26 + pps.init_qp_minus26 + header.slice_qp_delta,
           -6 * sps.bit_depth_luma_minus8, 51)
    for offset, base in ((header.slice_cb_qp_offset, pps.pps_cb_qp_offset),
                         (header.slice_cr_qp_offset, pps.pps_cr_qp_offset)):
        within('slice chroma QP offset', offset, -12, 12)
        within('chroma QP offset, slice and picture set', offset + base, -12, 12)
    within('slice_beta_offset_div2', header.slice_beta_offset_div2, -6, 6)
    within('slice_tc_offset_div2', header.slice_tc_offset_div2, -6, 6)
    within('diff_cu_qp_delta_depth for the sequence', pps.diff_cu_qp_delta_depth, 0,
           sps.log2_diff_max_min_luma_coding_block_size)
    within('log2_parallel_merge_level_minus2 for the sequence',
           pps.log2_parallel_merge_level_minus2, 0, sps.ctb_log2_size - 2)
    table = header.pred_weight_table
    if table is not None:
        within('luma_log2_weight_denom', table.luma_log2_weight_denom, 0, 7)
        within('ChromaLog2WeightDenom',
               table.luma_log2_weight_denom + table.delta_chroma_log2_weight_denom, 0, 7)
        for value in (*table.delta_luma_weight_l0, *table.delta_luma_weight_l1,
                      *table.luma_offset_l0, *table.luma_offset_l1):
            within('luma weight or offset', value, -128, 127)
        for pair in (*table.delta_chroma_weight_l0, *table.delta_chroma_weight_l1,
                     *table.chroma_offset_l0, *table.chroma_offset_l1):
            for value in pair:
                within('chroma weight or offset', value, -128, 127)
    columns, rows = pps.tile_columns(sps), pps.tile_rows(sps)
    if min(*columns, *rows) < 1:
        findings.report('a tile with no coding tree block', (columns, rows))
    if pps.tiles_enabled_flag and pps.entropy_coding_sync_enabled_flag:
        entries = len(columns) * sps.height_in_ctbs - 1
    elif pps.tiles_enabled_flag:
        entries = len(columns) * len(rows) - 1
    else:
        entries = sps.height_in_ctbs - 1
    within('num_entry_point_offsets', header.num_entry_point_offsets, 0, entries)
    within('header_bytes', header.header_bytes, 2, len(unit.rbsp) + 2)


# --------------------------------------------------- what the driver is handed


def check_picture(findings, session, surface, batches):
    """One submitted picture: its parameter buffer, then each slice segment's."""
    report, within = findings.report, findings.within
    parameters = batches[0][0][1]
    size = (parameters.pic_width_in_luma_samples, parameters.pic_height_in_luma_samples)
    if size != session.format.coded_size:
        report('a picture of another size than its surfaces', size)
    tree = (parameters.log2_min_luma_coding_block_size_minus3 + 3
            + parameters.log2_diff_max_min_luma_coding_block_size)
    within('driver CtbLog2SizeY', tree, 4, 6)
    across, down = (-(-side >> tree) for side in size)
    valid = set()
    for index, entry in enumerate(parameters.ReferenceFrames):
        if entry.flags & INVALID:
            if entry.picture_id != api.VA_INVALID_SURFACE:
                report('an unused ReferenceFrames entry naming a surface', entry.picture_id)
            continue
        valid.add(index)
        if entry.picture_id == surface:
            report('a picture decoded into a surface it predicts from', surface)
        elif entry.picture_id not in session.decoded:
            report('a reference surface that was never decoded', entry.picture_id)
    within('driver sps_max_dec_pic_buffering_minus1',
           parameters.sps_max_dec_pic_buffering_minus1, 0, 15)
    within('driver init_qp_minus26', parameters.init_qp_minus26,
           -(26 + 6 * parameters.bit_depth_luma_minus8), 25)
    within('driver pps_cb_qp_offset', parameters.pps_cb_qp_offset, -12, 12)
    within('driver pps_cr_qp_offset', parameters.pps_cr_qp_offset, -12, 12)
    within('driver pps_beta_offset_div2', parameters.pps_beta_offset_div2, -6, 6)
    within('driver pps_tc_offset_div2', parameters.pps_tc_offset_div2, -6, 6)
    within('driver diff_cu_qp_delta_depth', parameters.diff_cu_qp_delta_depth, 0,
           parameters.log2_diff_max_min_luma_coding_block_size)
    within('driver log2_parallel_merge_level_minus2',
           parameters.log2_parallel_merge_level_minus2, 0, tree - 2)
    within('driver num_tile_columns_minus1', parameters.num_tile_columns_minus1, 0,
           min(across - 1, 19))
    within('driver num_tile_rows_minus1', parameters.num_tile_rows_minus1, 0,
           min(down - 1, 21))
    for count, sizes, total in (
            (parameters.num_tile_columns_minus1, parameters.column_width_minus1, across),
            (parameters.num_tile_rows_minus1, parameters.row_height_minus1, down)):
        # Every tile but the last is stated, and the last has what is left.
        stated = [sizes[index] + 1 for index in range(count)]
        if sum(stated) >= total:
            report('tiles that leave the last no coding tree block', (stated, total))
    for (_, slice_), (_, data) in batches[1:]:
        if slice_.slice_data_size != len(data) or slice_.slice_data_offset:
            report('a slice that is not its data buffer', slice_.slice_data_size)
        within('slice data offset, escapes included',
               slice_.slice_data_byte_offset + slice_.slice_data_num_emu_prevn_bytes,
               2, len(data))
        within('driver slice_segment_address', slice_.slice_segment_address, 0,
               across * down - 1)
        flags = slice_.LongSliceFlags.fields
        within('driver slice_type', flags.slice_type, 0, 2)
        active = [0, 0]
        if flags.slice_type != SLICE_I:
            active[0] = slice_.num_ref_idx_l0_active_minus1 + 1
            within('driver five_minus_max_num_merge_cand',
                   slice_.five_minus_max_num_merge_cand, 0, 4)
            if slice_.collocated_ref_idx != NONE:
                within('driver collocated_ref_idx', slice_.collocated_ref_idx, 0,
                       slice_.num_ref_idx_l0_active_minus1 if flags.collocated_from_l0_flag
                       else slice_.num_ref_idx_l1_active_minus1)
        if flags.slice_type == SLICE_B:
            active[1] = slice_.num_ref_idx_l1_active_minus1 + 1
        for count, entries in zip(active, slice_.RefPicList, strict=True):
            within('driver num_ref_idx_active', count, 0, 15)
            for index, entry in enumerate(entries):
                if index < count and entry == NONE:
                    report('an active reference list entry naming no picture', index)
                elif index >= count and entry != NONE:
                    report('a reference list entry past the active count', index)
                elif entry != NONE and entry not in valid:
                    report('a reference list entry naming an unused ReferenceFrames '
                           'entry', entry)
        within('driver SliceQpY', 26 + parameters.init_qp_minus26 + slice_.slice_qp_delta,
               -6 * parameters.bit_depth_luma_minus8, 51)
        within('driver slice_cb_qp_offset', slice_.slice_cb_qp_offset, -12, 12)
        within('driver slice_cr_qp_offset', slice_.slice_cr_qp_offset, -12, 12)
        within('driver slice_beta_offset_div2', slice_.slice_beta_offset_div2, -6, 6)
        within('driver slice_tc_offset_div2', slice_.slice_tc_offset_div2, -6, 6)
        within('driver luma_log2_weight_denom', slice_.luma_log2_weight_denom, 0, 7)


@pytest.fixture
def findings(monkeypatch):
    """The findings of a run of the HEVC layer, with every parsed structure checked."""
    found = h26x_fuzz.Findings()

    def checked(name, check, with_unit=False):
        parse = getattr(hevc_decode, name)

        def parsing(*arguments):
            result = parse(*arguments)
            try:
                check(found, result, *arguments[:1] if with_unit else ())
            except StreamError:
                found.report('a tile grid refused after its slice was parsed')
            return result
        monkeypatch.setattr(hevc_decode, name, parsing)

    checked('parse_sps', check_sps)
    checked('parse_pps', check_pps)
    checked('parse_slice_header', check_slice, with_unit=True)
    return found


def test_mutated_streams_are_decoded_or_refused_and_stay_in_range(findings):
    ran = h26x_fuzz.run(hevc_decode.HEVCDecoding, h26x_fuzz.streams('*.h265'),
                        CASES, SEED, header=2, check_picture=check_picture,
                        findings=findings)
    assert not ran.found, f'\n{ran}'


def test_the_fixtures_themselves_pass_the_checks(findings):
    for name, units in h26x_fuzz.streams('*.h265').items():
        findings.case = name
        try:
            h26x_fuzz.decode(hevc_decode.HEVCDecoding, units, findings, check_picture)
        except DecoderError as refused:
            pytest.fail(f'{name} is refused: {refused}')
    assert not findings.found, f'\n{findings}'
