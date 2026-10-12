"""The NVDEC and CUDA declarations, checked against the layout the headers state.

A field at the wrong offset is not an error the driver can report: the parser
is handed a callback read out of a reserved word, or the decoder is created at
a size read out of a codec. These checks compare every declared structure and
constant with what ``tools/record_nvdec_abi.py`` recorded from the headers, so
the layout is verified with no compiler, no NVIDIA driver and no GPU.

``unsigned long`` is eight bytes on Linux and four on Windows, and two of the
structures are made of it, so the recording holds both layouts and the one
this interpreter uses is the one checked.
"""
import ctypes
import json
from pathlib import Path

import pytest

from pyopengl_video.nvdec import api

#: Written by ``tools/record_nvdec_abi.py`` from the nv-codec-headers.
ABI = json.loads((Path(__file__).parent / 'nvdec_abi.json').read_text())
LAYOUT = ABI['models']['lp64' if ctypes.sizeof(ctypes.c_long) == 8 else 'llp64']


def offset_of(structure, path):
    """Where the binding puts the field at dotted `path`, through nested structures."""
    offset = 0
    for name in path.split('.'):
        offset += getattr(structure, name).offset
        structure = dict((entry[0], entry[1]) for entry in structure._fields_)[name]
    return offset


def test_the_recording_is_here():
    """Without it nothing below checks anything, which must not pass quietly."""
    assert LAYOUT['sizes'], 'no structures recorded'
    assert ABI['constants'], 'no constants recorded'


@pytest.mark.parametrize('name', sorted(LAYOUT['sizes']))
def test_each_structure_is_the_size_the_headers_state(name):
    assert ctypes.sizeof(getattr(api, name)) == LAYOUT['sizes'][name]


@pytest.mark.parametrize('name', sorted(LAYOUT['offsets']))
def test_every_field_sits_where_the_headers_put_it(name):
    structure = getattr(api, name)
    ours = {field: offset_of(structure, field) for field in LAYOUT['offsets'][name]}
    assert ours == LAYOUT['offsets'][name]


@pytest.mark.parametrize('name', sorted(ABI['constants']))
def test_each_constant_has_the_value_the_headers_give_it(name):
    assert getattr(api, name) == ABI['constants'][name]


def test_every_declared_structure_is_recorded():
    """A structure added to the binding and not re-recorded is unchecked.

    The private ones, named with a leading underscore, are anonymous in the
    headers and are measured through the field that holds them.
    """
    declared = {
        name for name, value in vars(api).items()
        if isinstance(value, type) and issubclass(value, ctypes.Structure)
        and value is not ctypes.Structure and not name.startswith('_')
    }
    assert declared <= set(LAYOUT['sizes']), (
        f'not in tests/nvdec_abi.json: {sorted(declared - set(LAYOUT["sizes"]))}; '
        'run tools/record_nvdec_abi.py')


def test_every_field_the_binding_declares_was_measured():
    """A field added to a structure and not re-recorded is unchecked too."""
    for name, recorded in LAYOUT['offsets'].items():
        declared = {entry[0] for entry in getattr(api, name)._fields_ if len(entry) == 2}
        assert declared <= set(recorded), f'{name}: {sorted(declared - set(recorded))}'


def test_both_data_models_are_recorded():
    """Linux and Windows lay two of these structures out differently."""
    assert set(ABI['models']) == {'lp64', 'llp64'}
    sizes = {model: layout['sizes']['CUVIDDECODECREATEINFO']
             for model, layout in ABI['models'].items()}
    assert sizes['llp64'] < sizes['lp64']


def test_the_recording_states_which_headers_it_came_from():
    assert ABI['header_version'] == '{}.{}'.format(*api.HEADER_VERSION)
    assert 'nv-codec-headers' in ABI['source']
