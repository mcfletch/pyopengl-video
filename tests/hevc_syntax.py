"""HEVC syntax written for the tests, in the order of the specification's tables.

Each helper writes one syntax structure of ITU-T H.265 with :class:`BitWriter`,
with keyword arguments for the variations the tests need. The parser's tests
read these back field by field, and the decoded picture buffer's tests parse
them into the headers they drive it with.
"""
from pyopengl_video.vaapi.h264 import BitReader, BitWriter, emulation_prevented
from pyopengl_video.vaapi.hevc_dpb import DecodedPictureBuffer
from pyopengl_video.vaapi.hevc_parse import (
    NAL_IDR_W_RADL,
    NAL_PPS,
    NAL_SPS,
    NAL_TRAIL_R,
    NAL_VPS,
    SLICE_KINDS,
    _st_ref_pic_set,
    parse_nal,
    parse_pps,
    parse_slice_header,
    parse_sps,
)


def nal(kind, rbsp, temporal_id=0, layer=0):
    """An RBSP behind its two-byte NAL unit header, escaped."""
    header = bytes([kind << 1 | layer >> 5, (layer & 0x1F) << 3 | temporal_id + 1])
    return header + emulation_prevented(rbsp)


def write_profile_tier_level(writer, profile_idc=1, level_idc=93, sub_layers_minus1=0,
                             sub_layers=()):
    """profile_tier_level(1, sub_layers_minus1) of 7.3.3.

    sub_layers -- for each sub-layer below the highest, whether it states a
        profile and whether it states a level
    """
    writer.u(0, 2)                              # general_profile_space
    writer.flag(0)                              # general_tier_flag
    writer.u(profile_idc, 5)
    writer.u(1 << (31 - profile_idc), 32)       # general_profile_compatibility_flag
    writer.u(0b1001, 4)                         # progressive, frame only
    writer.u(0, 32)
    writer.u(0, 12)
    writer.u(level_idc, 8)
    stated = list(sub_layers) + [(0, 0)] * (sub_layers_minus1 - len(sub_layers))
    for profile_present, level_present in stated:
        writer.flag(profile_present)
        writer.flag(level_present)
    if sub_layers_minus1:
        writer.u(0, 2 * (8 - sub_layers_minus1))
    for profile_present, level_present in stated:
        if profile_present:
            writer.u(profile_idc, 8)
            writer.u(0, 32)
            writer.u(0, 32)
            writer.u(0, 16)
        if level_present:
            writer.u(level_idc - 3, 8)


def write_ordering(writer, ordering, sub_layers_minus1):
    """The sub-layer ordering information, stated for each sub-layer given."""
    writer.flag(len(ordering) > 1 or sub_layers_minus1 == 0)
    for buffering_minus1, reorder, latency_plus1 in ordering:
        writer.ue(buffering_minus1)
        writer.ue(reorder)
        writer.ue(latency_plus1)


def vps_bytes(*, sub_layers_minus1=0, ordering=((4, 2, 0),), timing=None, layer_sets=0):
    """A VPS in the order of 7.3.2.1; `timing` is (units in a tick, time scale)."""
    writer = BitWriter()
    writer.u(0, 4)                              # vps_video_parameter_set_id
    writer.u(3, 2)                              # base layer internal and available
    writer.u(0, 6)                              # vps_max_layers_minus1
    writer.u(sub_layers_minus1, 3)
    writer.flag(1)                              # vps_temporal_id_nesting_flag
    writer.u(0xFFFF, 16)
    write_profile_tier_level(writer, sub_layers_minus1=sub_layers_minus1)
    write_ordering(writer, ordering, sub_layers_minus1)
    writer.u(0, 6)                              # vps_max_layer_id
    writer.ue(layer_sets)                       # vps_num_layer_sets_minus1
    for _ in range(layer_sets):
        writer.flag(1)                          # layer_id_included_flag
    writer.flag(timing is not None)
    if timing is not None:
        writer.u(timing[0], 32)
        writer.u(timing[1], 32)
        writer.flag(0)                          # vps_poc_proportional_to_timing_flag
        writer.ue(0)                            # vps_num_hrd_parameters
    writer.flag(0)                              # vps_extension_flag
    return writer.rbsp()


def write_scaling_list_data(writer, sizes):
    """scaling_list_data() of 7.3.4.

    sizes -- for each of the four sizes, one entry a list: an integer is
        scaling_list_pred_matrix_id_delta, and ``(dc, values)`` codes the
        list, `dc` being None for the 4x4 and 8x8 sizes
    """
    for size_id, entries in enumerate(sizes):
        for entry in entries:
            writer.flag(not isinstance(entry, int))
            if isinstance(entry, int):
                writer.ue(entry)
                continue
            dc, values = entry
            last = 8
            if size_id > 1:
                writer.se(dc - 8)
                last = dc
            for value in values:
                writer.se((value - last + 128) % 256 - 128)
                last = value


def write_st_ref_pic_set(writer, index, spec, in_slice=False):
    """st_ref_pic_set(index) of 7.3.7.

    spec -- ``{'s0': [(delta_poc_minus1, used), ...], 's1': [...]}`` for a set
        coded outright, or ``{'delta_idx_minus1': n, 'delta_rps': n,
        'used': [...], 'use_delta': [...]}`` for one predicted from another
    """
    inter = 'delta_rps' in spec
    if index:
        writer.flag(inter)
    if inter:
        if in_slice:
            writer.ue(spec.get('delta_idx_minus1', 0))
        writer.flag(spec['delta_rps'] < 0)
        writer.ue(abs(spec['delta_rps']) - 1)
        for used, use_delta in zip(spec['used'], spec['use_delta'], strict=True):
            writer.flag(used)
            if not used:
                writer.flag(use_delta)
        return
    writer.ue(len(spec.get('s0', ())))
    writer.ue(len(spec.get('s1', ())))
    for delta_minus1, used in (*spec.get('s0', ()), *spec.get('s1', ())):
        writer.ue(delta_minus1)
        writer.flag(used)


def write_hrd(writer, sub_layers_minus1=0):
    """hrd_parameters(1, sub_layers_minus1) of E.2.2, with NAL parameters."""
    writer.flag(1)                              # nal_hrd_parameters_present_flag
    writer.flag(0)                              # vcl_hrd_parameters_present_flag
    writer.flag(1)                              # sub_pic_hrd_params_present_flag
    writer.u(0, 19)
    writer.u(0, 8)
    writer.u(0, 4)
    writer.u(0, 15)
    for layer in range(sub_layers_minus1 + 1):
        fixed = layer == 0
        writer.flag(fixed)                      # fixed_pic_rate_general_flag
        if not fixed:
            writer.flag(0)                      # fixed_pic_rate_within_cvs_flag
            writer.flag(0)                      # low_delay_hrd_flag
        else:
            writer.ue(0)                        # elemental_duration_in_tc_minus1
        writer.ue(1)                            # cpb_cnt_minus1
        for _ in range(2):
            writer.ue(10)
            writer.ue(20)
            writer.ue(5)
            writer.ue(6)
            writer.flag(0)


def write_vui(writer, vui, sub_layers_minus1=0):
    """vui_parameters() of E.2.1, from a dict of the parts present."""
    writer.flag('sar' in vui)
    if 'sar' in vui:
        writer.u(255, 8)
        writer.u(vui['sar'][0], 16)
        writer.u(vui['sar'][1], 16)
    writer.flag(0)                              # overscan_info_present_flag
    writer.flag('colour' in vui or 'full_range' in vui)
    if 'colour' in vui or 'full_range' in vui:
        writer.u(5, 3)                          # video_format
        writer.flag(vui.get('full_range', 0))
        writer.flag('colour' in vui)
        if 'colour' in vui:
            for value in vui['colour']:
                writer.u(value, 8)
    writer.flag('chroma_loc' in vui)
    if 'chroma_loc' in vui:
        writer.ue(vui['chroma_loc'])
        writer.ue(vui['chroma_loc'])
    writer.flag(0)                              # neutral_chroma_indication_flag
    writer.flag(vui.get('field_seq', 0))
    writer.flag(0)                              # frame_field_info_present_flag
    writer.flag('display_window' in vui)
    for value in vui.get('display_window', ()):
        writer.ue(value)
    writer.flag('timing' in vui)
    if 'timing' in vui:
        writer.u(vui['timing'][0], 32)
        writer.u(vui['timing'][1], 32)
        writer.flag(1)                          # vui_poc_proportional_to_timing_flag
        writer.ue(0)
        writer.flag(vui.get('hrd', 0))
        if vui.get('hrd'):
            write_hrd(writer, sub_layers_minus1)
    writer.flag(vui.get('restriction', 0))
    if vui.get('restriction'):
        writer.u(0b010, 3)
        for value in (0, 2, 1, 15, 15):
            writer.ue(value)


def sps_bytes(*, width=64, height=64, conformance_window=None, bit_depth=0,
              bit_depth_chroma=None, chroma_format_idc=1, profile_idc=1, level_idc=93,
              log2_max_poc_lsb_minus4=4, sub_layers_minus1=0, sub_layers=(),
              ordering=((4, 2, 0),), min_cb_minus3=0, diff_cb=1, scaling=None,
              amp=0, sao=0, pcm=None, st_rps=(), long_term=None, temporal_mvp=0,
              strong_smoothing=0, vui=None, range_extension=None,
              extension_flags=None, seq_parameter_set_id=0):
    """An SPS in the order of 7.3.2.2, with the variations under test.

    scaling -- ``'default'`` enables the lists and codes none; otherwise the
        `sizes` of :func:`write_scaling_list_data`
    pcm -- (luma depth less one, chroma depth less one, minimum block size
        less three, size range, loop filter disabled)
    long_term -- ``(poc lsb, used)`` for each long-term picture a slice may name
    range_extension -- the nine flags of sps_range_extension()
    extension_flags -- the multilayer, 3D and screen content flags
    """
    writer = BitWriter()
    writer.u(0, 4)                              # sps_video_parameter_set_id
    writer.u(sub_layers_minus1, 3)
    writer.flag(1)                              # sps_temporal_id_nesting_flag
    write_profile_tier_level(writer, profile_idc, level_idc, sub_layers_minus1, sub_layers)
    writer.ue(seq_parameter_set_id)
    writer.ue(chroma_format_idc)
    if chroma_format_idc == 3:
        writer.flag(0)                          # separate_colour_plane_flag
    writer.ue(width)
    writer.ue(height)
    writer.flag(conformance_window is not None)
    for value in conformance_window or ():
        writer.ue(value)
    writer.ue(bit_depth)
    writer.ue(bit_depth if bit_depth_chroma is None else bit_depth_chroma)
    writer.ue(log2_max_poc_lsb_minus4)
    write_ordering(writer, ordering, sub_layers_minus1)
    writer.ue(min_cb_minus3)
    writer.ue(diff_cb)
    writer.ue(0)                                # log2_min_luma_transform_block_size_minus2
    writer.ue(2)                                # log2_diff_max_min_luma_transform_block_size
    writer.ue(1)                                # max_transform_hierarchy_depth_inter
    writer.ue(2)                                # max_transform_hierarchy_depth_intra
    writer.flag(scaling is not None)
    if scaling is not None:
        writer.flag(scaling != 'default')
        if scaling != 'default':
            write_scaling_list_data(writer, scaling)
    writer.flag(amp)
    writer.flag(sao)
    writer.flag(pcm is not None)
    if pcm is not None:
        writer.u(pcm[0], 4)
        writer.u(pcm[1], 4)
        writer.ue(pcm[2])
        writer.ue(pcm[3])
        writer.flag(pcm[4])
    writer.ue(len(st_rps))
    for index, spec in enumerate(st_rps):
        write_st_ref_pic_set(writer, index, spec)
    writer.flag(long_term is not None)
    if long_term is not None:
        writer.ue(len(long_term))
        for lsb, used in long_term:
            writer.u(lsb, log2_max_poc_lsb_minus4 + 4)
            writer.flag(used)
    writer.flag(temporal_mvp)
    writer.flag(strong_smoothing)
    writer.flag(vui is not None)
    if vui is not None:
        write_vui(writer, vui, sub_layers_minus1)
    extended = range_extension is not None or extension_flags is not None
    writer.flag(extended)
    if extended:
        writer.flag(range_extension is not None)
        for flag in extension_flags or (0, 0, 0):
            writer.flag(flag)
        writer.u(0, 4)
        for flag in range_extension or ():
            writer.flag(flag)
    return writer.rbsp()


def pps_bytes(*, pic_parameter_set_id=0, seq_parameter_set_id=0, dependent_slices=0,
              output_flag_present=0, extra_bits=0, sign_hiding=0, cabac_init_present=0,
              refs=(0, 0), init_qp=0, constrained_intra=0, transform_skip=0,
              cu_qp_delta_depth=None, chroma_offsets=(0, 0),
              slice_chroma_offsets_present=0, weighted_pred=0, weighted_bipred=0,
              transquant_bypass=0, tiles=None, wpp=0, loop_filter_across_slices=0,
              deblocking=None, scaling=None, lists_modification=0, parallel_merge=0,
              header_extension=0, range_extension=None, extension_flags=None):
    """A PPS in the order of 7.3.2.3, with the variations under test.

    tiles -- ``{'columns': n, 'rows': n}`` for a uniform grid, with ``'widths'``
        and ``'heights'`` (each less one, all but the last) for a stated one
    deblocking -- ``{'override': flag, 'disabled': flag, 'beta': n, 'tc': n}``
    range_extension -- ``{'cross_component': flag, 'offset_list': [(cb, cr),
        ...] or None, 'sao_scale': (luma, chroma)}``
    """
    writer = BitWriter()
    writer.ue(pic_parameter_set_id)
    writer.ue(seq_parameter_set_id)
    writer.flag(dependent_slices)
    writer.flag(output_flag_present)
    writer.u(extra_bits, 3)
    writer.flag(sign_hiding)
    writer.flag(cabac_init_present)
    writer.ue(refs[0])
    writer.ue(refs[1])
    writer.se(init_qp)
    writer.flag(constrained_intra)
    writer.flag(transform_skip)
    writer.flag(cu_qp_delta_depth is not None)
    if cu_qp_delta_depth is not None:
        writer.ue(cu_qp_delta_depth)
    writer.se(chroma_offsets[0])
    writer.se(chroma_offsets[1])
    writer.flag(slice_chroma_offsets_present)
    writer.flag(weighted_pred)
    writer.flag(weighted_bipred)
    writer.flag(transquant_bypass)
    writer.flag(tiles is not None)
    writer.flag(wpp)
    if tiles is not None:
        writer.ue(tiles['columns'] - 1)
        writer.ue(tiles['rows'] - 1)
        writer.flag('widths' not in tiles)
        for value in (*tiles.get('widths', ()), *tiles.get('heights', ())):
            writer.ue(value)
        writer.flag(tiles.get('loop_filter', 1))
    writer.flag(loop_filter_across_slices)
    writer.flag(deblocking is not None)
    if deblocking is not None:
        writer.flag(deblocking.get('override', 0))
        writer.flag(deblocking.get('disabled', 0))
        if not deblocking.get('disabled', 0):
            writer.se(deblocking.get('beta', 0))
            writer.se(deblocking.get('tc', 0))
    writer.flag(scaling is not None)
    if scaling is not None:
        write_scaling_list_data(writer, scaling)
    writer.flag(lists_modification)
    writer.ue(parallel_merge)
    writer.flag(header_extension)
    extended = range_extension is not None or extension_flags is not None
    writer.flag(extended)
    if extended:
        writer.flag(range_extension is not None)
        for flag in extension_flags or (0, 0, 0):
            writer.flag(flag)
        writer.u(0, 4)
        if range_extension is not None:
            if transform_skip:
                writer.ue(1)                    # log2_max_transform_skip_block_size_minus2
            writer.flag(range_extension.get('cross_component', 0))
            offsets = range_extension.get('offset_list')
            writer.flag(offsets is not None)
            if offsets is not None:
                writer.ue(0)                    # diff_cu_chroma_qp_offset_depth
                writer.ue(len(offsets) - 1)
                for cb, cr in offsets:
                    writer.se(cb)
                    writer.se(cr)
            for value in range_extension.get('sao_scale', (0, 0)):
                writer.ue(value)
    return writer.rbsp()


def _ceil_log2(value):
    return (value - 1).bit_length()


def write_weights(writer, entries, chroma):
    """One list of pred_weight_table(): ``(luma, chroma)`` an entry, either None."""
    for luma, _ in entries:
        writer.flag(luma is not None)
    if chroma:
        for _, pair in entries:
            writer.flag(pair is not None)
    for luma, pair in entries:
        if luma is not None:
            writer.se(luma[0])
            writer.se(luma[1])
        if pair is not None:
            for weight, offset in pair:
                writer.se(weight)
                writer.se(offset)


def slice_rbsp(sps, pps, *, nal_type=NAL_TRAIL_R, first=1, no_output=0, dependent=0,
               address=0, kind='P', pic_output=1, poc_lsb=0, rps=None, long_term=None,
               temporal_mvp=0, sao=(0, 0), refs=None, modification=(None, None),
               mvd_l1_zero=0, cabac_init=0, collocated=(1, 0), weights=None, merge=0,
               qp_delta=0, chroma_offsets=(0, 0), cu_chroma_qp_offset=0, deblocking=None,
               loop_filter_across_slices=1, entry_points=(), extension=b'',
               data=b'\x80\x41\x99'):
    """A slice segment in the order of 7.3.6.1, read against parsed parameter sets.

    rps -- an index into the sequence's sets, or the `spec` of
        :func:`write_st_ref_pic_set` for a set coded in the header; None is
        an empty coded set
    long_term -- ``{'sps': [(lt_idx_sps, msb cycle or None), ...],
        'pics': [(poc lsb, used, msb cycle or None), ...]}``
    refs -- ``(l0, l1)`` active reference counts less one, overriding the PPS
    modification -- list_entry_l0 and list_entry_l1, None where unmodified
    collocated -- (collocated_from_l0_flag, collocated_ref_idx)
    weights -- ``{'denom': n, 'chroma_delta': n, 'l0': [...], 'l1': [...]}``,
        the lists as :func:`write_weights` takes them
    deblocking -- ``{'disabled': flag, 'beta': n, 'tc': n}`` overriding the PPS
    entry_points -- entry_point_offset_minus1 for each
    """
    writer = BitWriter()
    irap = 16 <= nal_type <= 23
    idr = nal_type in (19, 20)
    slice_type = SLICE_KINDS.index(kind)
    writer.flag(first)
    if irap:
        writer.flag(no_output)
    writer.ue(pps.pps_pic_parameter_set_id)
    if not first:
        if pps.dependent_slice_segments_enabled_flag:
            writer.flag(dependent)
        writer.u(address, _ceil_log2(sps.width_in_ctbs * sps.height_in_ctbs))
    if not dependent:
        writer.u(0, pps.num_extra_slice_header_bits)
        writer.ue(slice_type)
        if pps.output_flag_present_flag:
            writer.flag(pic_output)
        total = 0
        if not idr:
            writer.u(poc_lsb, sps.log2_max_pic_order_cnt_lsb_minus4 + 4)
            sets = sps.short_term_ref_pic_sets
            writer.flag(isinstance(rps, int))
            if isinstance(rps, int):
                writer.u(rps, _ceil_log2(len(sets)))
                total = sets[rps].used
            else:
                coded = BitWriter()
                write_st_ref_pic_set(coded, len(sets), rps or {}, in_slice=True)
                write_st_ref_pic_set(writer, len(sets), rps or {}, in_slice=True)
                total = _st_ref_pic_set(BitReader(coded.rbsp()), len(sets), list(sets),
                                        in_slice=True).used
            if sps.long_term_ref_pics_present_flag:
                from_sps = (long_term or {}).get('sps', ())
                pictures = (long_term or {}).get('pics', ())
                candidates = len(sps.lt_ref_pic_poc_lsb_sps)
                if candidates:
                    writer.ue(len(from_sps))
                writer.ue(len(pictures))
                for index, cycle in from_sps:
                    if candidates > 1:
                        writer.u(index, _ceil_log2(candidates))
                    total += sps.used_by_curr_pic_lt_sps_flag[index]
                    writer.flag(cycle is not None)
                    if cycle is not None:
                        writer.ue(cycle)
                for lsb, used, cycle in pictures:
                    writer.u(lsb, sps.log2_max_pic_order_cnt_lsb_minus4 + 4)
                    writer.flag(used)
                    total += used
                    writer.flag(cycle is not None)
                    if cycle is not None:
                        writer.ue(cycle)
            if sps.sps_temporal_mvp_enabled_flag:
                writer.flag(temporal_mvp)
        if sps.sample_adaptive_offset_enabled_flag:
            writer.flag(sao[0])
            writer.flag(sao[1])
        if kind != 'I':
            l0 = pps.num_ref_idx_l0_default_active_minus1
            l1 = pps.num_ref_idx_l1_default_active_minus1
            writer.flag(refs is not None)
            if refs is not None:
                l0, l1 = refs
                writer.ue(l0)
                if kind == 'B':
                    writer.ue(l1)
            if pps.lists_modification_present_flag and total > 1:
                for entries in modification[:2 if kind == 'B' else 1]:
                    writer.flag(entries is not None)
                    for entry in entries or ():
                        writer.u(entry, _ceil_log2(total))
            if kind == 'B':
                writer.flag(mvd_l1_zero)
            if pps.cabac_init_present_flag:
                writer.flag(cabac_init)
            if temporal_mvp:
                if kind == 'B':
                    writer.flag(collocated[0])
                if (l0 if collocated[0] or kind == 'P' else l1) > 0:
                    writer.ue(collocated[1])
            if (pps.weighted_pred_flag and kind == 'P') or (
                    pps.weighted_bipred_flag and kind == 'B'):
                plain = [(None, None)]
                weights = weights or {'denom': 0}
                writer.ue(weights['denom'])
                writer.se(weights.get('chroma_delta', 0))
                write_weights(writer, weights.get('l0', plain * (l0 + 1)), chroma=True)
                if kind == 'B':
                    write_weights(writer, weights.get('l1', plain * (l1 + 1)), chroma=True)
            writer.ue(merge)
        writer.se(qp_delta)
        if pps.pps_slice_chroma_qp_offsets_present_flag:
            writer.se(chroma_offsets[0])
            writer.se(chroma_offsets[1])
        if pps.chroma_qp_offset_list_enabled_flag:
            writer.flag(cu_chroma_qp_offset)
        disabled = pps.pps_deblocking_filter_disabled_flag
        if pps.deblocking_filter_override_enabled_flag:
            writer.flag(deblocking is not None)
            if deblocking is not None:
                disabled = deblocking.get('disabled', 0)
                writer.flag(disabled)
                if not disabled:
                    writer.se(deblocking.get('beta', 0))
                    writer.se(deblocking.get('tc', 0))
        if pps.pps_loop_filter_across_slices_enabled_flag and (
                (sps.sample_adaptive_offset_enabled_flag and any(sao)) or not disabled):
            writer.flag(loop_filter_across_slices)
    if pps.tiles_enabled_flag or pps.entropy_coding_sync_enabled_flag:
        writer.ue(len(entry_points))
        if entry_points:
            width = max(max(entry_points).bit_length(), 1)
            writer.ue(width - 1)
            for offset in entry_points:
                writer.u(offset, width)
    if pps.slice_segment_header_extension_present_flag:
        writer.ue(len(extension))
        for byte in extension:
            writer.u(byte, 8)
    return writer.rbsp() + data


def sequence(**options):
    """A parsed SPS; `options` as for :func:`sps_bytes`."""
    return parse_sps(sps_bytes(**options))


def picture_set(**options):
    """A parsed PPS; `options` as for :func:`pps_bytes`."""
    return parse_pps(pps_bytes(**options))


def slice_unit(sps, pps, *, temporal_id=0, **options):
    """One slice segment as a parsed NAL unit; `options` as for :func:`slice_rbsp`."""
    kind = options.get('nal_type', NAL_TRAIL_R)
    return parse_nal(nal(kind, slice_rbsp(sps, pps, **options), temporal_id))


def header(sps, pps, *, independent=None, temporal_id=0, **options):
    """A parsed slice segment header; `options` as for :func:`slice_rbsp`."""
    return parse_slice_header(
        slice_unit(sps, pps, temporal_id=temporal_id, **options),
        {sps.sps_seq_parameter_set_id: sps}, {pps.pps_pic_parameter_set_id: pps},
        independent)


def parameter_units(sps=None, pps=None, vps=None):
    """The VPS, SPS and PPS NAL units, from each one's RBSP or a default."""
    return [nal(NAL_VPS, vps or vps_bytes()), nal(NAL_SPS, sps or sps_bytes()),
            nal(NAL_PPS, pps or pps_bytes())]


# ------------------------------------------------- what several tests share

FOUR = tuple(10 + index for index in range(16))
EIGHT = tuple(20 + index for index in range(64))
SCALING = [
    [(None, FOUR), 0, 1, (None, FOUR[::-1]), 2, 0],
    [(None, EIGHT), 1, (None, EIGHT[::-1]), 0, 1, 3],
    [(40, EIGHT), 1, 0, (50, EIGHT[::-1]), 4, 1],
    [(60, EIGHT), 1],
]

#: Four sets: one coded outright, two predicted from the set before, one more
#: coded outright.
SETS = [
    {'s0': [(0, 1), (1, 1), (3, 0)], 's1': [(0, 1), (2, 1)]},
    {'delta_rps': -1, 'used': [1, 1, 0, 1, 1, 1], 'use_delta': [1, 1, 0, 1, 1, 1]},
    {'delta_rps': 2, 'used': [1, 0, 1, 0, 1], 'use_delta': [1, 1, 1, 0, 1]},
    {'s0': [(1, 1)]},
]


def coded(before=(), after=(), unused=()):
    """A reference picture set coded in a slice header, from picture order
    count distances; those in `unused` are kept and not predicted from."""
    spec = {'s0': [], 's1': []}
    for name, deltas in (('s0', sorted(before, reverse=True)), ('s1', sorted(after))):
        last = 0
        for delta in deltas:
            spec[name].append((abs(delta - last) - 1, int(delta not in unused)))
            last = delta
    return spec


class Stream:
    """A decoded picture buffer and the parameter sets the test's pictures are
    written against. Each picture's surface is a label the test chooses."""

    def __init__(self, *, pps=None, **sps):
        sps.setdefault('ordering', ((5, 2, 0),))
        self.sps = sequence(**sps)
        self.pps = picture_set(**(pps or {}))
        self.buffer = DecodedPictureBuffer()
        self.pictures = {}

    def header(self, poc_lsb=0, *, before=(), after=(), unused=(), **options):
        if 'rps' not in options and options.get('nal_type') != NAL_IDR_W_RADL:
            options['rps'] = coded(before, after, unused)
        options.setdefault('kind', 'B' if after else 'P' if before else 'I')
        return header(self.sps, self.pps, poc_lsb=poc_lsb, **options)

    def decode(self, name, poc_lsb=0, **options):
        """Decode one picture into the surface `name`; return what is output."""
        picture = self.buffer.begin(self.header(poc_lsb, **options))
        self.pictures[name] = picture
        shown = self.buffer.outputs()
        if not picture.skipped:
            shown += self.buffer.finish(picture, surface=name)
        return [frame.surface for frame in shown]

    def idr(self, name='idr'):
        return self.decode(name, nal_type=NAL_IDR_W_RADL, kind='I')

    def held(self):
        return sorted(str(frame.surface) for frame in self.buffer.frames)
