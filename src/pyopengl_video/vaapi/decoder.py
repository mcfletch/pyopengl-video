"""The VA-API decoder: H.264 samples in, NV12 OpenGL textures out.

The picture never leaves the GPU. Each decode surface is exported once as a
DMA-BUF of two layers, luma as ``R8`` and chroma as ``GR88``, and imported into
OpenGL as two textures (:mod:`pyopengl_video.linux.dmabuf`). Decoding writes
the surface and the textures show it; the conversion pass
(:mod:`pyopengl_video.convert`) turns the two planes into RGBA.

libva is a slice-level interface: the driver reconstructs macroblocks, and the
caller supplies the parsed syntax and the picture management.
:mod:`~pyopengl_video.vaapi.h264_parse` reads the syntax,
:mod:`~pyopengl_video.vaapi.h264_dpb` keeps the decoded picture buffer, and the
functions here turn both into the buffers libva takes. They hold no driver, so
the suite checks them on any machine.

**The OpenGL context has to be an EGL context**, as it does for the encoder.
"""
from __future__ import annotations

import ctypes
import functools
import logging
import os
from collections.abc import Callable, Sequence
from typing import Any

from OpenGL.GL import (
    GL_SYNC_FLUSH_COMMANDS_BIT,
    GL_SYNC_GPU_COMMANDS_COMPLETE,
    GL_TIMEOUT_IGNORED,
    glClientWaitSync,
    glDeleteSync,
    glFenceSync,
)

from pyopengl_video.colour import Colour
from pyopengl_video.decoder import DecodedFrame, Decoder, DecoderError, Sample
from pyopengl_video.linux import dmabuf
from pyopengl_video.mp4 import split_annexb
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.device import Device
from pyopengl_video.vaapi.h264 import StreamError
from pyopengl_video.vaapi.h264_dpb import DecodedPictureBuffer, Frame, Picture
from pyopengl_video.vaapi.h264_parse import (
    NAL_END_OF_SEQUENCE,
    NAL_END_OF_STREAM,
    NAL_IDR,
    NAL_PPS,
    NAL_SLICE,
    NAL_SPS,
    NalUnit,
    PictureParameterSet,
    SequenceParameterSet,
    SliceHeader,
    parse_nal,
    parse_pps,
    parse_slice_header,
    parse_sps,
    scaling_matrices,
    starts_new_picture,
)

log = logging.getLogger(__name__)

__all__ = ['DEFAULT_OUTPUT_SLOTS', 'SurfaceOrder', 'VAAPIDecoder', 'iq_matrix', 'picture_parameters',
           'profile_for', 'slice_parameters']

#: How many decoded frames a caller may hold at once, unless it says otherwise.
#: A player converts each frame as it arrives and holds none for long.
DEFAULT_OUTPUT_SLOTS = 4

#: profile_idc values by name, for refusing a stream in words a reader knows.
PROFILE_NAMES = {
    66: 'Baseline', 77: 'Main', 88: 'Extended', 100: 'High', 110: 'High 10',
    122: 'High 4:2:2', 244: 'High 4:4:4 Predictive', 44: 'CAVLC 4:4:4 Intra',
    118: 'Multiview High', 128: 'Stereo High',
}

#: The libva profile each decoded profile_idc runs under. Baseline is decoded
#: as Constrained Baseline, which is Baseline without slice groups and
#: redundant pictures; a stream that uses slice groups is refused.
VA_PROFILES = {
    66: api.VAProfileH264ConstrainedBaseline,
    77: api.VAProfileH264Main,
    100: api.VAProfileH264High,
}


def as_decoder_error(method: Callable[..., Any]) -> Callable[..., Any]:
    """Let a method raise only :class:`DecoderError`, keeping the cause.

    libva's errors, the DMA-BUF shim's and the parser's each say something
    worth reading, and none is a :class:`DecoderError` in itself, since each
    module serves more than decoding.
    """
    @functools.wraps(method)
    def translating(*arguments: Any, **named: Any) -> Any:
        try:
            return method(*arguments, **named)
        except DecoderError:
            raise
        except (api.VAError, dmabuf.DMABufError, StreamError) as error:
            raise DecoderError(str(error)) from error
    return translating


# ------------------------------------------------------------ the buffers


def _zigzag(size: int) -> tuple[int, ...]:
    """The frame zig-zag scan of a `size` x `size` block, as raster indices."""
    order = []
    for diagonal in range(2 * size - 1):
        rows = range(max(0, diagonal - size + 1), min(diagonal, size - 1) + 1)
        for row in (rows if diagonal % 2 else reversed(rows)):
            order.append(row * size + diagonal - row)
    return tuple(order)


#: Tables 8-12 and 8-13's frame scans: scan position to raster index.
ZIGZAG_4X4 = _zigzag(4)
ZIGZAG_8X8 = _zigzag(8)


def profile_for(sps: SequenceParameterSet, pps: PictureParameterSet) -> int:
    """The libva profile a stream decodes under; refuses what is not decoded here."""
    if not sps.frame_mbs_only_flag:
        raise DecoderError(
            'the stream is interlaced: it codes pictures as fields or as field '
            'macroblock pairs, and this decoder takes progressive frames only')
    name = PROFILE_NAMES.get(sps.profile_idc, f'profile_idc {sps.profile_idc}')
    if sps.profile_idc not in VA_PROFILES:
        raise DecoderError(
            f'the stream is {name} profile, and VA-API decoding here covers '
            'Constrained Baseline, Main and High (8-bit 4:2:0)')
    if sps.chroma_format_idc != 1 or sps.bit_depth_luma_minus8 or sps.bit_depth_chroma_minus8:
        raise DecoderError(
            f'the stream is {name} profile at chroma format '
            f'{sps.chroma_format_idc} and {8 + sps.bit_depth_luma_minus8} bits, '
            'and this decoder takes 8-bit 4:2:0')
    if pps.num_slice_groups_minus1:
        raise DecoderError(
            'the stream uses slice groups (flexible macroblock ordering), which '
            'VA-API drivers do not decode')
    return VA_PROFILES[sps.profile_idc]


def va_picture(frame: Frame) -> api.VAPictureH264:
    """A reference frame as libva names it."""
    long_term = frame.long_term
    return api.VAPictureH264(
        picture_id=frame.surface,
        frame_idx=frame.long_term_frame_idx if long_term else frame.frame_num,
        flags=(api.VA_PICTURE_H264_LONG_TERM_REFERENCE if long_term
               else api.VA_PICTURE_H264_SHORT_TERM_REFERENCE),
        TopFieldOrderCnt=frame.top_field_order_cnt,
        BottomFieldOrderCnt=frame.bottom_field_order_cnt)


def picture_parameters(picture: Picture, references: Sequence[Frame],
                       surface: int) -> api.VAPictureParameterBufferH264:
    """The picture parameter buffer for `picture`, decoded into `surface`.

    references -- the reference frames in the decoded picture buffer; frames
        inferred from a gap in frame_num have no surface and are left out
    """
    header = picture.header
    sps, pps = header.sps, header.pps
    parameters = api.VAPictureParameterBufferH264()
    current = parameters.CurrPic
    current.picture_id = surface
    current.frame_idx = header.frame_num
    current.flags = 0
    current.TopFieldOrderCnt = picture.top_field_order_cnt
    current.BottomFieldOrderCnt = picture.bottom_field_order_cnt
    decoded = [frame for frame in references if frame.surface is not None]
    for index in range(16):
        parameters.ReferenceFrames[index] = (
            va_picture(decoded[index]) if index < len(decoded)
            else api.VAPictureH264.unused())
    parameters.picture_width_in_mbs_minus1 = sps.pic_width_in_mbs_minus1
    parameters.picture_height_in_mbs_minus1 = sps.pic_height_in_map_units_minus1
    parameters.bit_depth_luma_minus8 = sps.bit_depth_luma_minus8
    parameters.bit_depth_chroma_minus8 = sps.bit_depth_chroma_minus8
    parameters.num_ref_frames = sps.max_num_ref_frames
    bits = parameters.seq_fields.bits
    bits.chroma_format_idc = sps.chroma_format_idc
    bits.residual_colour_transform_flag = sps.separate_colour_plane_flag
    bits.gaps_in_frame_num_value_allowed_flag = sps.gaps_in_frame_num_value_allowed_flag
    bits.frame_mbs_only_flag = sps.frame_mbs_only_flag
    bits.mb_adaptive_frame_field_flag = sps.mb_adaptive_frame_field_flag
    bits.direct_8x8_inference_flag = sps.direct_8x8_inference_flag
    # A.3.3.2: from level 3.1 a bi-predicted block is no smaller than 8x8.
    bits.MinLumaBiPredSize8x8 = int(sps.level_idc >= 31)
    bits.log2_max_frame_num_minus4 = sps.log2_max_frame_num_minus4
    bits.pic_order_cnt_type = sps.pic_order_cnt_type
    bits.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_pic_order_cnt_lsb_minus4
    bits.delta_pic_order_always_zero_flag = sps.delta_pic_order_always_zero_flag
    parameters.pic_init_qp_minus26 = pps.pic_init_qp_minus26
    parameters.pic_init_qs_minus26 = pps.pic_init_qs_minus26
    parameters.chroma_qp_index_offset = pps.chroma_qp_index_offset
    parameters.second_chroma_qp_index_offset = pps.second_chroma_qp_index_offset
    flags = parameters.pic_fields.bits
    flags.entropy_coding_mode_flag = pps.entropy_coding_mode_flag
    flags.weighted_pred_flag = pps.weighted_pred_flag
    flags.weighted_bipred_idc = pps.weighted_bipred_idc
    flags.transform_8x8_mode_flag = pps.transform_8x8_mode_flag
    flags.field_pic_flag = header.field_pic_flag
    flags.constrained_intra_pred_flag = pps.constrained_intra_pred_flag
    flags.pic_order_present_flag = pps.bottom_field_pic_order_in_frame_present_flag
    flags.deblocking_filter_control_present_flag = pps.deblocking_filter_control_present_flag
    flags.redundant_pic_cnt_present_flag = pps.redundant_pic_cnt_present_flag
    flags.reference_pic_flag = int(header.nal_ref_idc != 0)
    parameters.frame_num = header.frame_num
    return parameters


def iq_matrix(sps: SequenceParameterSet,
              pps: PictureParameterSet) -> api.VAIQMatrixBufferH264:
    """The scaling lists in force, in the raster order libva takes them in."""
    fours, eights = scaling_matrices(sps, pps)
    matrix = api.VAIQMatrixBufferH264()
    for index, values in enumerate(fours):
        for position, raster in enumerate(ZIGZAG_4X4):
            matrix.ScalingList4x4[index][raster] = values[position]
    for index, values in enumerate(eights[:2]):
        for position, raster in enumerate(ZIGZAG_8X8):
            matrix.ScalingList8x8[index][raster] = values[position]
    return matrix


def _reference_list(entries: Sequence[Frame | None], target: Any) -> None:
    for index in range(32):
        frame = entries[index] if index < len(entries) else None
        target[index] = (va_picture(frame) if frame is not None and frame.surface is not None
                         else api.VAPictureH264.unused())


def slice_parameters(header: SliceHeader, list0: Sequence[Frame | None],
                     list1: Sequence[Frame | None],
                     size: int) -> api.VASliceParameterBufferH264:
    """The slice parameter buffer for one slice of `size` bytes, NAL header included."""
    parameters = api.VASliceParameterBufferH264()
    parameters.slice_data_size = size
    parameters.slice_data_offset = 0
    parameters.slice_data_flag = api.VA_SLICE_DATA_FLAG_ALL
    parameters.slice_data_bit_offset = header.header_bits
    parameters.first_mb_in_slice = header.first_mb_in_slice
    parameters.slice_type = header.slice_type
    parameters.direct_spatial_mv_pred_flag = header.direct_spatial_mv_pred_flag
    parameters.num_ref_idx_l0_active_minus1 = header.num_ref_idx_l0_active_minus1
    parameters.num_ref_idx_l1_active_minus1 = header.num_ref_idx_l1_active_minus1
    parameters.cabac_init_idc = header.cabac_init_idc
    parameters.slice_qp_delta = header.slice_qp_delta
    parameters.disable_deblocking_filter_idc = header.disable_deblocking_filter_idc
    parameters.slice_alpha_c0_offset_div2 = header.slice_alpha_c0_offset_div2
    parameters.slice_beta_offset_div2 = header.slice_beta_offset_div2
    _reference_list(list0, parameters.RefPicList0)
    _reference_list(list1, parameters.RefPicList1)
    table = header.pred_weight_table
    if table is not None:
        parameters.luma_log2_weight_denom = table.luma_log2_weight_denom
        parameters.chroma_log2_weight_denom = table.chroma_log2_weight_denom
        parameters.luma_weight_l0_flag = int(table.luma_weight_l0_flag)
        parameters.chroma_weight_l0_flag = int(table.chroma_weight_l0_flag)
        parameters.luma_weight_l1_flag = int(table.luma_weight_l1_flag)
        parameters.chroma_weight_l1_flag = int(table.chroma_weight_l1_flag)
        for luma, chroma, weights, offsets, chroma_weights, chroma_offsets in (
            (table.luma_l0, table.chroma_l0, parameters.luma_weight_l0,
             parameters.luma_offset_l0, parameters.chroma_weight_l0,
             parameters.chroma_offset_l0),
            (table.luma_l1, table.chroma_l1, parameters.luma_weight_l1,
             parameters.luma_offset_l1, parameters.chroma_weight_l1,
             parameters.chroma_offset_l1),
        ):
            for index, (weight, offset) in enumerate(luma):
                weights[index], offsets[index] = weight, offset
            for index, pair in enumerate(chroma):
                for component, (weight, offset) in enumerate(pair):
                    chroma_weights[index][component] = weight
                    chroma_offsets[index][component] = offset
    return parameters


def colour_of(sps: SequenceParameterSet) -> Colour:
    """A sequence's colour description, the matrix chosen where it states none."""
    vui = sps.vui
    if vui is None:
        return Colour().resolved(sps.size[1])
    return Colour(matrix=vui.matrix_coefficients, full_range=bool(vui.video_full_range_flag),
                  primaries=vui.colour_primaries, transfer=vui.transfer_characteristics,
                  chroma_location=vui.chroma_sample_loc_type_top_field,
                  ).resolved(sps.size[1])


# ------------------------------------------------------------- surfaces


class SurfaceOrder:
    """Which free surface to decode into next: the one returned longest ago.

    A surface OpenGL has just given back may still be read by a conversion
    the GPU has not run yet, and decoding into it waits on that; the surface
    returned longest ago is the one least likely to make the decoder wait.
    """

    def __init__(self, surfaces: Sequence[int]) -> None:
        self._order = list(surfaces)

    def choose(self, in_use: set[int]) -> int | None:
        """The free surface returned longest ago, or None when all are in use."""
        for surface in self._order:
            if surface not in in_use:
                return surface
        return None

    def returned(self, surface: int) -> None:
        """Note that `surface` has just come back from OpenGL."""
        self._order.remove(surface)
        self._order.append(surface)


class _Pool:
    """Decode surfaces and the OpenGL textures over them, for one sequence.

    Every surface is exported and imported once, when the pool is made, so
    decoding never creates a GL object. A fence made when a lent frame comes
    back is waited on before its surface is decoded into again, so the decoder
    never writes a picture OpenGL is still reading.
    """

    def __init__(self, device: Device, profile: int, coded_size: tuple[int, int],
                 count: int) -> None:
        self.device = device
        self.profile = profile
        self.coded_size = coded_size
        #: What is allocated: the coded size, raised to the driver's minimum.
        self.size = coded_size
        self.count = count
        self.surfaces: list[int] = []
        self.textures: dict[int, tuple[int, int]] = {}
        self._images: list[tuple[int, object]] = []
        self._fences: dict[int, Any] = {}
        self.config = api.VAConfigID(0)
        self.context = api.VAContextID(0)
        try:
            self._create()
        except BaseException:
            self.close()
            raise

    def _create(self) -> None:
        va, display = self.device.va, self.device.display
        attribute = api.VAConfigAttrib(type=api.VAConfigAttribRTFormat,
                                       value=api.VA_RT_FORMAT_YUV420)
        self.device.check(
            va.vaCreateConfig(display, self.profile, api.VAEntrypointVLD,
                              ctypes.byref(attribute), 1, ctypes.byref(self.config)),
            'vaCreateConfig', f'H.264 decode, profile {self.profile}')
        self.size = self._allocation()
        width, height = self.size
        format_attribute = api.VASurfaceAttrib()
        format_attribute.type = api.VASurfaceAttribPixelFormat
        format_attribute.flags = api.VA_SURFACE_ATTRIB_SETTABLE
        format_attribute.value.type = api.VAGenericValueTypeInteger
        format_attribute.value.value.i = api.VA_FOURCC_NV12
        surfaces = (api.VASurfaceID * self.count)()
        self.device.check(
            va.vaCreateSurfaces(display, api.VA_RT_FORMAT_YUV420, width, height,
                                surfaces, self.count, ctypes.byref(format_attribute), 1),
            'vaCreateSurfaces', f'{self.count} x {width}x{height} NV12')
        self.surfaces = [int(surface) for surface in surfaces]
        self.device.check(
            va.vaCreateContext(display, self.config, width, height, api.VA_PROGRESSIVE,
                               surfaces, self.count, ctypes.byref(self.context)),
            'vaCreateContext', f'{width}x{height} decode')
        for surface in self.surfaces:
            self.textures[surface] = self._import(surface)
        self.order = SurfaceOrder(self.surfaces)

    def _allocation(self) -> tuple[int, int]:
        """The surface size: the coded size, at least the driver's minimum.

        A picture smaller than the minimum decodes into the top left of a
        larger surface; one larger than the maximum is refused by its size.
        """
        limits = self.limits()
        width, height = self.coded_size
        most = (limits.get(api.VASurfaceAttribMaxWidth, width),
                limits.get(api.VASurfaceAttribMaxHeight, height))
        if width > most[0] or height > most[1]:
            raise DecoderError(
                f'a {width}x{height} picture is larger than the {most[0]}x{most[1]} '
                f'this decoder ({self.device.vendor}) reaches')
        return (max(width, limits.get(api.VASurfaceAttribMinWidth, 0)),
                max(height, limits.get(api.VASurfaceAttribMinHeight, 0)))

    def limits(self) -> dict[int, int]:
        """The configuration's size limits, by surface attribute type."""
        va, display = self.device.va, self.device.display
        count = ctypes.c_uint(0)
        self.device.check(va.vaQuerySurfaceAttributes(display, self.config, None,
                                                      ctypes.byref(count)),
                          'vaQuerySurfaceAttributes', 'counting')
        attributes = (api.VASurfaceAttrib * count.value)()
        self.device.check(va.vaQuerySurfaceAttributes(display, self.config, attributes,
                                                      ctypes.byref(count)),
                          'vaQuerySurfaceAttributes', 'size limits')
        wanted = (api.VASurfaceAttribMinWidth, api.VASurfaceAttribMaxWidth,
                  api.VASurfaceAttribMinHeight, api.VASurfaceAttribMaxHeight)
        return {entry.type: int(entry.value.value.i)
                for entry in list(attributes)[:count.value]
                if entry.type in wanted and entry.value.type == api.VAGenericValueTypeInteger}

    def _import(self, surface: int) -> tuple[int, int]:
        """Export `surface` as DMA-BUF layers and give each an OpenGL name."""
        descriptor = api.VADRMPRIMESurfaceDescriptor()
        self.device.check(
            self.device.va.vaExportSurfaceHandle(
                self.device.display, surface, api.VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                api.VA_EXPORT_SURFACE_READ_ONLY | api.VA_EXPORT_SURFACE_SEPARATE_LAYERS,
                ctypes.byref(descriptor)),
            'vaExportSurfaceHandle', f'surface {surface}')
        try:
            formats = [descriptor.layers[index].drm_format
                       for index in range(descriptor.num_layers)]
            if formats != [api.DRM_FORMAT_R8, api.DRM_FORMAT_GR88]:
                raise DecoderError(
                    f'surface {surface} exported as layers '
                    f'{[api.fourcc_name(value) for value in formats]}, not the R8 '
                    'and GR88 planes of NV12')
            width, height = self.size
            planes = []
            for index, size in enumerate(((width, height), (width // 2, height // 2))):
                layer = descriptor.layers[index]
                source = descriptor.objects[layer.object_index[0]]
                texture, image = dmabuf.import_texture(
                    layer.drm_format, *size, source.drm_format_modifier,
                    (dmabuf.Plane(fd=source.fd, offset=layer.offset[0],
                                  stride=layer.pitch[0]),))
                self._images.append((texture, image))
                planes.append(texture)
            return planes[0], planes[1]
        finally:
            for index in range(descriptor.num_objects):
                fd = descriptor.objects[index].fd
                if fd >= 0:
                    os.close(fd)

    def ready(self, surface: int) -> None:
        """Wait until OpenGL has finished reading `surface` from its last lending."""
        fence = self._fences.pop(surface, None)
        if fence is not None:
            glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED)
            glDeleteSync(fence)

    def returned(self, surface: int) -> None:
        """Note the point in OpenGL's commands where a lent surface came back."""
        if self.context:
            self.order.returned(surface)
            previous = self._fences.pop(surface, None)
            if previous is not None:
                glDeleteSync(previous)
            self._fences[surface] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)

    def close(self) -> None:
        """Give back the textures, the images, the surfaces and the context."""
        for fence in self._fences.values():
            glDeleteSync(fence)
        self._fences = {}
        for texture, image in self._images:
            dmabuf.release_imported_texture(texture, image)
        self._images = []
        self.textures = {}
        va, display = self.device.va, self.device.display
        if display is None:
            return
        if self.context:
            va.vaDestroyContext(display, self.context)
            self.context = api.VAContextID(0)
        self.device.destroy_surfaces(self.surfaces)
        self.surfaces = []
        if self.config:
            va.vaDestroyConfig(display, self.config)
            self.config = api.VAConfigID(0)


class _Current:
    """The picture being gathered from its slices."""

    def __init__(self, picture: Picture, surface: int | None, sample: Sample) -> None:
        self.picture = picture
        self.surface = surface
        self.sample = sample
        self.slices: list[tuple[SliceHeader, NalUnit]] = []


# --------------------------------------------------------------- decoder


class VAAPIDecoder(Decoder):
    """H.264 decoding on Intel and AMD hardware, into OpenGL textures.

    codec -- ``'h264'``
    headers -- the stream's SPS and PPS as Annex-B, when the container keeps
        them apart from the samples
    size -- accepted for the registry; the stream's own size is used
    device -- the DRM render node to open; the default is the first with an
        H.264 decoder. Afterwards, the node that was opened.
    output_slots -- how many frames a caller may hold unreleased at once

    Each sample is one access unit: a picture and the units before it.
    Frames come out in display order as ``'nv12'``
    :class:`~pyopengl_video.decoder.DecodedFrame` objects.
    """

    zero_copy = True

    @as_decoder_error
    def __init__(self, codec: str = 'h264', *, headers: bytes = b'',
                 size: tuple[int, int] | None = None, device: str | None = None,
                 output_slots: int = DEFAULT_OUTPUT_SLOTS, **unknown: object) -> None:
        if unknown:
            raise DecoderError(
                f'the VA-API decoder has no setting named {", ".join(sorted(unknown))}; '
                'it takes headers, size, device and output_slots')
        if codec != 'h264':
            raise DecoderError(f'the VA-API backend decodes h264, not {codec!r}')
        del size
        self.codec = codec
        self.output_slots = int(output_slots)
        self._closed = False
        self._sequences: dict[int, SequenceParameterSet] = {}
        self._pictures: dict[int, PictureParameterSet] = {}
        self._buffer = DecodedPictureBuffer()
        self._pool: _Pool | None = None
        self._retired: list[_Pool] = []
        self._lent: dict[DecodedFrame, tuple[_Pool, int]] = {}
        self._current: _Current | None = None
        self._active: SequenceParameterSet | None = None
        self._leading_before: int | None = None
        self._after_reset = False
        nodes = [device] if device else api.render_nodes()
        self._device = Device.first(nodes, _decodes_h264, 'H.264 decoding')
        self.device = self._device.node
        self.vendor = self._device.vendor
        try:
            self._read_parameter_sets(headers)
        except BaseException:
            self.close()
            raise

    def __repr__(self) -> str:
        width, height = self.size
        return f'<{type(self).__name__} {width}x{height} h264 on {self.device}>'

    def _read_parameter_sets(self, headers: bytes) -> None:
        for unit in split_annexb(headers):
            self._parameter_set(parse_nal(unit))

    def _parameter_set(self, unit: NalUnit) -> None:
        if unit.nal_unit_type == NAL_SPS:
            sps = parse_sps(unit.rbsp)
            self._sequences[sps.seq_parameter_set_id] = sps
            if self._active is None:
                self._describe(sps)
        elif unit.nal_unit_type == NAL_PPS:
            pps = parse_pps(unit.rbsp, self._sequences)
            self._pictures[pps.pic_parameter_set_id] = pps

    def _describe(self, sps: SequenceParameterSet) -> None:
        self.size = sps.size
        self.colour = colour_of(sps)

    # ------------------------------------------------------------ decoding

    @as_decoder_error
    def decode(self, sample: Sample) -> list[DecodedFrame]:
        """Decode one access unit; return the frames now ready, in display order."""
        self._check_open()
        ready: list[DecodedFrame] = []
        for data in split_annexb(sample.data):
            unit = parse_nal(data)
            kind = unit.nal_unit_type
            if kind in (NAL_SPS, NAL_PPS):
                self._parameter_set(unit)
            elif kind in (NAL_SLICE, NAL_IDR):
                header = parse_slice_header(unit, self._sequences, self._pictures)
                if header.redundant_pic_cnt:
                    continue
                current = self._current
                if current is not None and starts_new_picture(current.slices[-1][0], header):
                    ready += self._finish()
                if self._current is None:
                    ready += self._begin(header, sample)
                assert self._current is not None
                self._current.slices.append((header, unit))
            elif kind in (NAL_END_OF_SEQUENCE, NAL_END_OF_STREAM):
                ready += self._finish()
        return ready + self._finish()

    def _begin(self, header: SliceHeader, sample: Sample) -> list[DecodedFrame]:
        """Start a picture: activate its sequence, order it, choose its surface."""
        ready = self._activate(header)
        picture = self._buffer.begin(header)
        if self._after_reset:
            self._after_reset = False
            self._leading_before = None if header.idr else picture.poc
        elif header.idr:
            self._leading_before = None
        if self._leading_before is not None and picture.poc < self._leading_before:
            # A leading picture of the group a seek landed in: it refers to
            # pictures before the seek point, which were never decoded.
            self._current = _Current(picture, None, sample)
            return ready
        surface = self._free_surface()
        assert self._pool is not None
        self._pool.ready(surface)
        self._current = _Current(picture, surface, sample)
        return ready

    def _activate(self, header: SliceHeader) -> list[DecodedFrame]:
        """Make sure the surfaces suit the sequence this picture belongs to.

        A new sequence of another size or profile can begin only at an IDR.
        What the old one still holds is output first, and its surfaces are
        given back once every frame lent from them has been released.
        """
        sps = header.sps
        profile = profile_for(sps, header.pps)
        count = sps.dpb_frames + 1 + self.output_slots
        pool = self._pool
        if (pool is not None and pool.profile == profile
                and pool.coded_size == sps.coded_size and pool.count >= count):
            self._active = sps
            self._describe(sps)
            return []
        ready: list[DecodedFrame] = []
        if pool is not None:
            if header.no_output_of_prior_pics_flag:
                self._buffer.reset()
            else:
                ready = [frame for frame in map(self._lend, self._buffer.flush())
                         if frame is not None]
            self._buffer.reset()
            self._retire(pool)
        log.debug('%s: %d surfaces of %dx%d for profile %d', self.device,
                  count, *sps.coded_size, profile)
        self._pool = _Pool(self._device, profile, sps.coded_size, count)
        self._active = sps
        self._describe(sps)
        return ready

    def _free_surface(self) -> int:
        pool = self._pool
        assert pool is not None
        in_use = {frame.surface for frame in self._buffer.frames}
        in_use |= {surface for owner, surface in self._lent.values() if owner is pool}
        surface = pool.order.choose(in_use)
        if surface is not None:
            return surface
        held = [frame for frame, (owner, _) in self._lent.items() if owner is pool]
        raise DecoderError(
            f'every one of the {pool.count} decode surfaces is in use: '
            f'{len(self._buffer.frames)} by the decoded picture buffer and '
            f'{len(held)} by frames still held ({held}); release frames before '
            'decoding more, or open the decoder with more output_slots')

    def _finish(self) -> list[DecodedFrame]:
        """Submit the gathered picture and return what the buffer outputs."""
        current, self._current = self._current, None
        if current is None:
            return []
        if current.surface is None:
            return []
        pool = self._pool
        assert pool is not None
        picture = current.picture
        device = self._device
        batches = [[device.buffer(pool.context, api.VAPictureParameterBufferType,
                                  picture_parameters(picture,
                                                     self._buffer.reference_frames(),
                                                     current.surface)),
                    device.buffer(pool.context, api.VAIQMatrixBufferType,
                                  iq_matrix(picture.header.sps, picture.header.pps))]]
        for header, unit in current.slices:
            list0, list1 = self._buffer.reference_lists(picture, header)
            data = (ctypes.c_uint8 * len(unit.data)).from_buffer_copy(unit.data)
            batches.append([
                device.buffer(pool.context, api.VASliceParameterBufferType,
                              slice_parameters(header, list0, list1, len(unit.data))),
                device.buffer(pool.context, api.VASliceDataBufferType, data)])
        device.picture(pool.context, current.surface, batches,
                       f'picture {picture.frame_num}')
        output = self._buffer.finish(picture, surface=current.surface,
                                     payload=(current.sample.timestamp,
                                              current.sample.duration))
        return [frame for frame in map(self._lend, output) if frame is not None]

    def _lend(self, frame: Frame) -> DecodedFrame | None:
        """Hand a decoded frame to the caller, once the hardware has finished it."""
        pool = self._pool
        if pool is None or frame.surface is None:
            return None
        surface = frame.surface
        self._device.check(self._device.va.vaSyncSurface(self._device.display, surface),
                           'vaSyncSurface', f'surface {surface}')
        timestamp, duration = frame.payload
        sps = self._active
        assert sps is not None
        decoded = DecodedFrame(
            timestamp=timestamp, duration=duration, planes=pool.textures[surface],
            layout='nv12', texture_size=pool.size, visible=sps.visible,
            colour=self.colour, on_release=self._released)
        self._lent[decoded] = (pool, surface)
        return decoded

    def _released(self, frame: DecodedFrame) -> None:
        owner, surface = self._lent.pop(frame)
        owner.returned(surface)
        if owner in self._retired and not any(
                pool is owner for pool, _ in self._lent.values()):
            self._retired.remove(owner)
            owner.close()

    def _retire(self, pool: _Pool) -> None:
        if any(owner is pool for owner, _ in self._lent.values()):
            self._retired.append(pool)
        else:
            pool.close()

    @as_decoder_error
    def flush(self) -> list[DecodedFrame]:
        """Decode what is gathered and return every frame still held back."""
        self._check_open()
        ready = self._finish()
        return ready + [frame for frame in map(self._lend, self._buffer.flush())
                        if frame is not None]

    def reset(self) -> None:
        """Drop every picture held back. Frames already lent stay valid."""
        self._current = None
        self._buffer.reset()
        self._after_reset = True

    def close(self) -> None:
        """Release the surfaces and their textures, frames lent included."""
        if self._closed:
            return
        self._closed = True
        for frame in list(self._lent):
            frame.release()
        for pool in [*self._retired, *([self._pool] if self._pool else [])]:
            pool.close()
        self._retired = []
        self._pool = None
        self._device.close()

    def _check_open(self) -> None:
        if self._closed:
            raise DecoderError('this decoder has been closed')


def _decodes_h264(device: Device) -> str:
    """'' when a device decodes H.264 High, or why it will not do."""
    if api.VAEntrypointVLD in device.entrypoints(api.VAProfileH264High):
        return ''
    return 'no H.264 decode entrypoint (VAEntrypointVLD)'
