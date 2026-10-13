"""The hand-written NvEncodeAPI structures against the ABI the C header states.

A structure whose fields are in the wrong place is not an error the driver can
report: it reads a bitrate out of what the binding thinks is a frame rate, or
writes past the end of an allocation. These checks compare every declared
structure against sizes and offsets recorded from the header by
``tools/record_nvenc_abi.py``, so the layout is verified on any machine --
no compiler, no NVIDIA driver and no GPU needed.
"""
import ctypes
import importlib.util
import json
from pathlib import Path

import pytest

from pyopengl_video.nvenc import api

ABI = json.loads((Path(__file__).parent / 'nvenc_abi.json').read_text())


def _recorder():
    """``tools/record_nvenc_abi.py``: how a member is reached, and its image made."""
    path = Path(__file__).parent.parent / 'tools' / 'record_nvenc_abi.py'
    spec = importlib.util.spec_from_file_location('record_nvenc_abi', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECORDER = _recorder()


def test_the_recording_is_here():
    """Without it nothing below checks anything, which must not pass quietly."""
    assert ABI['sizes'], 'no structures recorded'
    assert ABI['bits'], 'no bit-fields recorded'


@pytest.mark.parametrize('name', sorted(ABI['sizes']))
def test_structure_size_matches_the_header(name):
    assert ctypes.sizeof(getattr(api, name)) == ABI['sizes'][name]


@pytest.mark.parametrize('name', sorted(ABI['offsets']))
def test_field_offsets_match_the_header(name):
    structure = getattr(api, name)
    ours = {field: getattr(structure, field).offset for field in ABI['offsets'][name]}
    assert ours == ABI['offsets'][name]


@pytest.mark.parametrize('name', sorted(ABI['nested']))
def test_every_member_of_a_structure_inside_one_sits_where_the_header_puts_it(name):
    structure = getattr(api, name)
    ours = {member: RECORDER.offset_of(structure, member) for member in ABI['nested'][name]}
    assert ours == ABI['nested'][name]


@pytest.mark.parametrize('name', sorted(ABI['bits']))
def test_every_bit_field_covers_the_bits_the_header_gives_it(name):
    """One member set to all ones: the same bytes change as in the C structure."""
    structure = getattr(api, name)
    ours = {member: RECORDER.bit_image(structure, member) for member in ABI['bits'][name]}
    assert ours == ABI['bits'][name]


def test_every_declared_structure_and_union_is_recorded():
    """A structure added to the binding without re-recording the ABI is unchecked."""
    declared = {
        name for name, value in vars(api).items()
        if isinstance(value, type) and issubclass(value, RECORDER.AGGREGATES)
        and value not in RECORDER.AGGREGATES
    }
    assert declared - set(ABI['sizes']) == set()


def test_every_member_of_every_recorded_structure_is_recorded():
    """A field added to a recorded structure, at any depth, is unchecked until re-recorded."""
    missing = []
    for name in ABI['sizes']:
        for member, bit_field in RECORDER.members(getattr(api, name)):
            recorded = (ABI['bits'].get(name, {}) if bit_field
                        else {*ABI['offsets'][name], *ABI['nested'].get(name, {})})
            if member not in recorded:
                missing.append(f'{name}.{member}')
    assert not missing, f'not in tests/nvenc_abi.json: {missing}; run tools/record_nvenc_abi.py'


def test_the_recording_states_which_header_it_came_from():
    assert ABI['header_version'] == '{}.{}'.format(*api.HEADER_VERSION)
    assert 'nv-codec-headers' in ABI['source']
