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
import importlib.util
import json
from pathlib import Path

import pytest

from pyopengl_video.nvdec import api

#: Written by ``tools/record_nvdec_abi.py`` from the nv-codec-headers.
ABI = json.loads((Path(__file__).parent / 'nvdec_abi.json').read_text())
LAYOUT = ABI['models']['lp64' if ctypes.sizeof(ctypes.c_long) == 8 else 'llp64']


def _recorder():
    """``tools/record_nvdec_abi.py``: how a member is reached, and its image made."""
    path = Path(__file__).parent.parent / 'tools' / 'record_nvdec_abi.py'
    spec = importlib.util.spec_from_file_location('record_nvdec_abi', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECORDER = _recorder()


def aggregates():
    """Every structure and union the binding declares, by class."""
    return {
        value for value in vars(api).values()
        if isinstance(value, type) and issubclass(value, RECORDER.AGGREGATES)
        and value.__module__ == api.__name__
    }


def inside(kind):
    """The structure and union classes `kind` is made of, itself included."""
    found = {kind}
    for entry in kind._fields_:
        field = entry[1]
        if issubclass(field, ctypes.Array):
            field = field._type_
        if issubclass(field, RECORDER.AGGREGATES):
            found |= inside(field)
    return found


def test_the_recording_is_here():
    """Without it nothing below checks anything, which must not pass quietly."""
    assert LAYOUT['sizes'], 'no structures recorded'
    assert LAYOUT['bits'], 'no bit-fields recorded'
    assert ABI['constants'], 'no constants recorded'


@pytest.mark.parametrize('name', sorted(LAYOUT['sizes']))
def test_each_structure_is_the_size_the_headers_state(name):
    assert ctypes.sizeof(getattr(api, name)) == LAYOUT['sizes'][name]


@pytest.mark.parametrize('name', sorted(LAYOUT['offsets']))
def test_every_field_sits_where_the_headers_put_it(name):
    structure = getattr(api, name)
    ours = {field: getattr(structure, field).offset for field in LAYOUT['offsets'][name]}
    assert ours == LAYOUT['offsets'][name]


@pytest.mark.parametrize('name', sorted(LAYOUT['nested']))
def test_every_member_of_a_structure_inside_one_sits_where_the_headers_put_it(name):
    structure = getattr(api, name)
    ours = {member: RECORDER.offset_of(structure, member) for member in LAYOUT['nested'][name]}
    assert ours == LAYOUT['nested'][name]


@pytest.mark.parametrize('name', sorted(LAYOUT['bits']))
def test_every_bit_field_covers_the_bits_the_headers_give_it(name):
    """One member set to all ones: the same bytes change as in the C structure."""
    structure = getattr(api, name)
    ours = {member: RECORDER.bit_image(structure, member) for member in LAYOUT['bits'][name]}
    assert ours == LAYOUT['bits'][name]


@pytest.mark.parametrize('name', sorted(ABI['constants']))
def test_each_constant_has_the_value_the_headers_give_it(name):
    assert getattr(api, name) == ABI['constants'][name]


def test_every_declared_structure_and_union_is_recorded():
    """One added to the binding and not re-recorded is unchecked.

    A class the headers name is recorded under that name. One they leave
    unnamed, which the binding spells with a leading underscore, is recorded
    through each named structure that holds it.
    """
    covered = set()
    for name in LAYOUT['sizes']:
        covered |= inside(getattr(api, name))
    missing = sorted(kind.__name__ for kind in aggregates() - covered)
    assert not missing, (
        f'not in tests/nvdec_abi.json, by name or inside a recorded structure: '
        f'{missing}; run tools/record_nvdec_abi.py')


def test_every_member_of_every_recorded_structure_is_recorded():
    """A field added to a recorded structure, at any depth, is unchecked until re-recorded."""
    missing = []
    for name in LAYOUT['sizes']:
        for member, bit_field in RECORDER.members(getattr(api, name)):
            recorded = (LAYOUT['bits'].get(name, {}) if bit_field
                        else {*LAYOUT['offsets'][name], *LAYOUT['nested'].get(name, {})})
            if member not in recorded:
                missing.append(f'{name}.{member}')
    assert not missing, f'not in tests/nvdec_abi.json: {missing}; run tools/record_nvdec_abi.py'


def test_both_data_models_record_the_same_members():
    """Linux and Windows lay two of these structures out differently."""
    assert set(ABI['models']) == {'lp64', 'llp64'}
    sizes = {model: layout['sizes']['CUVIDDECODECREATEINFO']
             for model, layout in ABI['models'].items()}
    assert sizes['llp64'] < sizes['lp64']
    linux, windows = ABI['models']['lp64'], ABI['models']['llp64']
    for section in ('sizes', 'offsets', 'nested', 'bits'):
        assert {name: set(members) if isinstance(members, dict) else None
                for name, members in linux[section].items()} == {
                    name: set(members) if isinstance(members, dict) else None
                    for name, members in windows[section].items()}, section


def test_the_recording_states_which_headers_it_came_from():
    assert ABI['header_version'] == '{}.{}'.format(*api.HEADER_VERSION)
    assert 'nv-codec-headers' in ABI['source']


class TestThePictureParameters:
    """What the decode callback reads of a picture before the hardware is handed it."""

    def test_an_av1_reference_is_where_the_driver_was_observed_to_put_it(self):
        """The review read ``ref_frame[i].index`` at 0x1ec + 12 * i of the codec's parameters."""
        av1 = api.CUVIDAV1PICPARAMS
        assert av1.ref_frame.offset + api._AV1Reference.index.offset == 0x1ec  # noqa: SLF001 - unnamed in the headers, measured through the structure that holds it
        assert ctypes.sizeof(api._AV1Reference) == 12  # noqa: SLF001 - as above
        assert av1.ref_frame_map.offset == 0x1d7
        assert av1.primary_ref_frame.offset == 0x1d6

    def test_every_codecs_parameters_begin_where_the_union_does(self):
        union = api.CUVIDPICPARAMS.CodecSpecific
        assert union.size == 4096
        for codec in ('h264', 'hevc', 'vp9', 'av1', 'CodecReserved'):
            assert RECORDER.offset_of(api.CUVIDPICPARAMS, f'CodecSpecific.{codec}') == union.offset


@pytest.fixture
def libraries(nvdec_available):  # noqa: ARG001 - requested to skip where there is no NVDEC
    """The CUDA and decoder bindings over the driver's libraries."""
    return api.Cuda.load(), api.Cuvid.load()


class TestEveryFunctionIsDeclared:
    """Standard 18: ctypes assumes ``int`` arguments for a function nobody declared."""

    @staticmethod
    def functions(holder):
        """The foreign functions among an object's attributes, by name."""
        return {name: value for name, value in vars(holder).items()
                if isinstance(value, ctypes._CFuncPtr)}  # noqa: SLF001 - the base class of every foreign function, which ctypes does not export

    def test_every_bound_function_has_its_argument_and_result_types(self, libraries):
        for binding in libraries:
            functions = self.functions(binding)
            assert functions, f'no foreign function was found on {type(binding).__name__}'
            for name, function in functions.items():
                assert function.argtypes is not None, name
                assert function.restype is ctypes.c_int, name

    def test_no_function_is_reached_past_the_binding(self, libraries):
        """A call made as ``library.name(...)`` would be one with no types."""
        for binding in libraries:
            bound = list(self.functions(binding).values())
            stray = sorted(name for name, function in self.functions(binding.library).items()
                           if not any(function is declared for declared in bound))
            assert not stray, f'called with no declared types: {stray}'
