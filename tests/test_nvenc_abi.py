"""The hand-written NvEncodeAPI structures against the ABI the C header states.

A structure whose fields are in the wrong place is not an error the driver can
report: it reads a bitrate out of what the binding thinks is a frame rate, or
writes past the end of an allocation. These checks compare every declared
structure against sizes and offsets recorded from the header by
``tools/record_nvenc_abi.py``, so the layout is verified on any machine --
no compiler, no NVIDIA driver and no GPU needed.
"""
import ctypes
import json
from pathlib import Path

import pytest

from pyopengl_video.nvenc import api

ABI = json.loads((Path(__file__).parent / 'nvenc_abi.json').read_text())


def declared_fields(structure):
    """Field names the binding declares that C can take the offset of."""
    return [entry[0] for entry in structure._fields_ if len(entry) == 2]


@pytest.mark.parametrize('name', sorted(ABI['sizes']))
def test_structure_size_matches_the_header(name):
    assert ctypes.sizeof(getattr(api, name)) == ABI['sizes'][name]


@pytest.mark.parametrize('name', sorted(ABI['offsets']))
def test_field_offsets_match_the_header(name):
    structure = getattr(api, name)
    recorded = ABI['offsets'][name]
    ours = {field: getattr(structure, field).offset for field in declared_fields(structure)}
    assert ours == recorded


def test_every_declared_structure_is_recorded():
    """A structure added to the binding without re-recording the ABI is unchecked."""
    bases = (ctypes.Structure, ctypes.Union)
    declared = {
        name for name, value in vars(api).items()
        if isinstance(value, type) and issubclass(value, bases) and value not in bases
    }
    assert declared - set(ABI['sizes']) == set()


def test_the_recording_states_which_header_it_came_from():
    assert ABI['header_version'] == '{}.{}'.format(*api.HEADER_VERSION)
    assert 'nv-codec-headers' in ABI['source']
