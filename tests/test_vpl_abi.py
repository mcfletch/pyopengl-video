"""The oneVPL structure declarations, checked against the layout the API states.

A wrong pack directive or a missed reserved array makes a structure the runtime
reads past the end of. It does not report that: every call taking an
``mfxVideoParam`` comes back ``MFX_ERR_INVALID_VIDEO_PARAM`` and names nothing,
which is a long way from the cause. These checks put the failure next to the
mistake.

The sizes and offsets are those of the API as published in the oneVPL headers
(see NOTICES.md), on a 64-bit build. ``tools/record_vpl_abi.py`` regenerates
them from the headers where a C compiler is available.
"""
import ctypes
import json
from pathlib import Path

import pytest

from pyopengl_video.vpl import api

#: Written by ``tools/record_vpl_abi.py`` where a C compiler is available. When
#: it is here it is the authority; the table in the binding is what holds the
#: declarations honest on the machines that have neither.
RECORDED = Path(__file__).parent / 'vpl_abi.json'


@pytest.mark.skipif(not RECORDED.exists(),
                    reason='no recorded ABI here; run tools/record_vpl_abi.py '
                           'where a C compiler and the oneVPL headers are')
def test_the_binding_matches_the_abi_recorded_from_the_headers():
    abi = json.loads(RECORDED.read_text())
    for name, size in abi['sizes'].items():
        assert ctypes.sizeof(getattr(api, name)) == size, f'sizeof {name}'
    for structure, fields in abi['offsets'].items():
        for field, offset in fields.items():
            assert getattr(getattr(api, structure), field).offset == offset, (
                f'{structure}.{field}')


@pytest.mark.parametrize('name, size', sorted(api.EXPECTED_SIZES.items()))
def test_each_structure_is_the_size_the_api_states(name, size):
    assert ctypes.sizeof(getattr(api, name)) == size


@pytest.mark.parametrize('where, offset', sorted(api.EXPECTED_OFFSETS.items()))
def test_each_field_sits_where_the_api_puts_it(where, offset):
    structure, field = where
    assert getattr(getattr(api, structure), field).offset == offset


def test_the_geometry_union_is_what_makes_frame_info_sixty_eight_bytes():
    """The union holds a mfxU64, and pack(4) is what keeps it from padding out.

    Declared with natural alignment the structure measures 80, and a runtime
    handed that answers MFX_ERR_INVALID_VIDEO_PARAM to everything.
    """
    assert api.mfxFrameInfo._pack_ == 4
    assert ctypes.sizeof(api._BufferGeometry) == 12
    assert ctypes.sizeof(api._GeometryUnion) == 12


def test_a_frame_info_round_trips_the_values_written_into_it():
    """Reading back through the anonymous unions reaches the same bytes."""
    info = api.mfxFrameInfo()
    info.FourCC = api.MFX_FOURCC_RGB4
    info.Width, info.Height = 1280, 720
    info.CropW, info.CropH = 1280, 720
    raw = bytes(memoryview(info).cast('B'))
    assert raw[32:36] == b'RGB4'
    assert int.from_bytes(raw[36:38], 'little') == 1280
    assert int.from_bytes(raw[38:40], 'little') == 720


def test_the_video_param_reaches_the_codec_arm_through_its_union():
    params = api.mfxVideoParam()
    params.mfx.CodecId = api.MFX_CODEC_AVC
    params.mfx.FrameInfo.Width = 640
    raw = bytes(memoryview(params).cast('B'))
    # codec union at 16, CodecId 100 into mfxInfoMFX, FrameInfo 32 into it.
    assert raw[116:120] == b'AVC '
    assert int.from_bytes(raw[16 + 32 + 36:16 + 32 + 38], 'little') == 640


def test_fourcc_names_survive_the_round_trip():
    for text in ('NV12', 'RGB4', 'BGR4', 'AVC '):
        assert api.fourcc_name(api.fourcc(text)) == text


def test_status_codes_have_names():
    assert api.STATUS_NAMES[-15] == 'MFX_ERR_INVALID_VIDEO_PARAM'
    assert api.STATUS_NAMES[api.MFX_ERR_MORE_DATA] == 'MFX_ERR_MORE_DATA'


def test_an_error_names_the_call_and_the_status():
    error = api.VPLError(-15, 'MFXVideoENCODE_Init', '1280x720')
    assert 'MFX_ERR_INVALID_VIDEO_PARAM' in str(error)
    assert 'MFXVideoENCODE_Init' in str(error)
    assert '1280x720' in str(error)
    assert error.status == -15


def test_check_passes_a_warning_through_but_raises_on_a_failure():
    assert api.VPL.check(api.MFX_WRN_PARTIAL_ACCELERATION, 'call') == 4
    assert api.VPL.check(0, 'call') == 0
    with pytest.raises(api.VPLError):
        api.VPL.check(-3, 'call')


def test_the_library_is_named_for_the_platform():
    assert api.library_name().startswith('libvpl')
