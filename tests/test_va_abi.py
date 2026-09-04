"""The libva declarations, checked against the layout the headers state.

A field at the wrong offset is what this catches. libva hands these structures
straight to a driver that reads them by offset, and a mistake is not reported:
the call succeeds and the picture is coded from a bitrate read out of a frame
rate. These checks put the failure next to the mistake, and they run with no
driver, no GPU and no compiler -- ``tools/record_va_abi.py`` does the compiling,
once, wherever the headers are.
"""
import ctypes
import json
from pathlib import Path

import pytest

from pyopengl_video.vaapi import api

#: Written by ``tools/record_va_abi.py`` from the libva headers.
RECORDED = Path(__file__).parent / 'va_abi.json'
ABI = json.loads(RECORDED.read_text())


def test_the_recording_is_here():
    """Without it nothing below checks anything, which must not pass quietly."""
    assert ABI['sizes'], 'no structures recorded'
    assert ABI['constants'], 'no constants recorded'


@pytest.mark.parametrize('name', sorted(ABI['sizes']))
def test_each_structure_is_the_size_the_headers_state(name):
    assert ctypes.sizeof(getattr(api, name)) == ABI['sizes'][name]


@pytest.mark.parametrize('name', sorted(ABI['offsets']))
def test_every_field_sits_where_the_headers_put_it(name):
    structure = getattr(api, name)
    for field, offset in ABI['offsets'][name].items():
        assert getattr(structure, field).offset == offset, f'{name}.{field}'


@pytest.mark.parametrize('name', sorted(ABI['constants']))
def test_each_constant_has_the_value_the_headers_give_it(name):
    assert getattr(api, name) == ABI['constants'][name]


def test_every_declared_structure_is_recorded():
    """A structure added to the binding and not re-recorded is unchecked."""
    declared = {
        name for name, value in vars(api).items()
        if isinstance(value, type) and issubclass(value, ctypes.Structure)
        and not name.startswith('_')
    }
    assert declared <= set(ABI['sizes']), (
        f'not in tests/va_abi.json: {sorted(declared - set(ABI["sizes"]))}; '
        'run tools/record_va_abi.py')


class TestFlexibleArrayMembers:
    """``VAEncMiscParameterBuffer`` ends in one, which ctypes cannot declare."""

    def test_the_payload_follows_the_type_tag(self):
        wrapped = api.misc_parameter(
            api.VAEncMiscParameterTypeHRD,
            api.VAEncMiscParameterHRD(buffer_size=1234))
        assert type(wrapped).data.offset == 4
        assert wrapped.type == api.VAEncMiscParameterTypeHRD
        assert wrapped.data.buffer_size == 1234

    def test_the_envelope_is_built_once_per_payload_type(self):
        first = api.misc_parameter(0, api.VAEncMiscParameterFrameRate())
        second = api.misc_parameter(0, api.VAEncMiscParameterFrameRate())
        assert type(first) is type(second)

    def test_the_envelope_is_the_tag_plus_the_payload(self):
        payload = api.VAEncMiscParameterRateControl()
        wrapped = api.misc_parameter(api.VAEncMiscParameterTypeRateControl,
                                     payload)
        assert ctypes.sizeof(wrapped) == 4 + ctypes.sizeof(payload)


class TestBitFields:
    """The unions of flags, which are where a miscount is least visible."""

    def test_sequence_flags_pack_into_one_word(self):
        fields = api._SeqFields()
        fields.bits.chroma_format_idc = 1
        fields.bits.frame_mbs_only_flag = 1
        fields.bits.direct_8x8_inference_flag = 1
        fields.bits.log2_max_frame_num_minus4 = 4
        fields.bits.log2_max_pic_order_cnt_lsb_minus4 = 4
        # chroma 1 at bit 0, frame_mbs_only at 2, direct_8x8 at 5,
        # log2_max_frame_num at 6, log2_max_poc_lsb at 12.
        assert fields.value == (1 | 1 << 2 | 1 << 5 | 4 << 6 | 4 << 12)

    def test_picture_flags_pack_into_one_word(self):
        fields = api._PicFields()
        fields.bits.idr_pic_flag = 1
        fields.bits.reference_pic_flag = 1
        fields.bits.entropy_coding_mode_flag = 1
        fields.bits.transform_8x8_mode_flag = 1
        fields.bits.deblocking_filter_control_present_flag = 1
        # idr at 0, reference (two bits) at 1, entropy at 3, weighted_pred at 4,
        # weighted_bipred (two bits) at 5, constrained_intra at 7, and so
        # transform_8x8 at 8 with deblocking control at 9.
        assert fields.value == (1 | 1 << 1 | 1 << 3 | 1 << 8 | 1 << 9)

    def test_the_vui_flag_word_reaches_the_wide_fields(self):
        fields = api._VuiFields()
        fields.bits.log2_max_mv_length_horizontal = 15
        fields.bits.log2_max_mv_length_vertical = 15
        assert fields.value == (15 << 3 | 15 << 8)


class TestFourCC:
    def test_a_code_survives_the_round_trip(self):
        for text in ('NV12', 'RGBA', 'AB24', 'AR24'):
            assert api.fourcc_name(api.fourcc(text)) == text

    def test_the_drm_and_libva_names_read_in_opposite_directions(self):
        """``GL_RGBA8`` exports as DRM ``ABGR8888``, which libva calls RGBA."""
        assert api.DRM_FORMAT_ABGR8888 == api.fourcc('AB24')
        assert api.VA_FOURCC_RGBA == api.fourcc('RGBA')

    def test_the_invalid_modifier_is_the_one_the_specification_names(self):
        assert api.DRM_FORMAT_MOD_INVALID == 0x00FFFFFFFFFFFFFF


class TestUnusedPictures:
    def test_an_empty_reference_slot_says_so(self):
        empty = api.VAPictureH264.unused()
        assert empty.flags & api.VA_PICTURE_H264_INVALID
        assert empty.picture_id == 0xFFFFFFFF


class TestErrors:
    def test_an_error_names_the_call_and_what_was_attempted(self):
        error = api.VAError(-1, 'vaCreateConfig', '1920x1080 H264High')
        assert 'vaCreateConfig' in str(error)
        assert '1920x1080 H264High' in str(error)
        assert error.status == -1

    def test_a_status_of_success_raises_nothing(self):
        # Bound without a display, which check() does not need.
        api.VA.check(api.VA, api.VA_STATUS_SUCCESS, 'vaInitialize')

    def test_any_other_status_raises(self):
        with pytest.raises(api.VAError):
            api.VA.check(api.VA, -1, 'vaInitialize')


class TestRenderNodes:
    def test_they_are_found_under_dev_dri(self):
        for node in api.render_nodes():
            assert node.startswith('/dev/dri/renderD')

    def test_they_come_back_in_a_stable_order(self):
        assert api.render_nodes() == sorted(api.render_nodes())

    def test_a_machine_with_no_drm_devices_answers_with_no_nodes(self, monkeypatch):
        monkeypatch.setattr(api.os, 'listdir',
                            lambda path: (_ for _ in ()).throw(FileNotFoundError))
        assert api.render_nodes() == []
