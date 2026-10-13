"""What the H.264 and HEVC control layers write into libva's structures.

ctypes keeps the low bits of an integer wider than its field and raises
nothing, so a parsed value can reach a driver as another number. Three things
hold that off, and each has its test here: the control layers assign fields
through one function that reads each back; over every fixture, each field of a
buffer that is named as a syntax element is the element's parsed value; and
the function raises for a value that does not fit.
"""
import ast
import ctypes
import inspect

import pytest

from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.decoder import DecoderError
from pyopengl_video.vaapi import api, h264_decode, hevc_api, hevc_decode
from pyopengl_video.vaapi.control import store, store_each
from tests.h26x_fuzz import DATA

RESERVED = ('va_reserved', 'ReservedBits', 'reserved')


class Session:
    """A decode session with no driver: surfaces are numbers."""

    control = None
    format = None

    def activate(self, format):
        self.format = format

    def acquire(self):
        return min(set(range(self.format.surfaces + 1)) - self.control.held())

    def submit(self, surface, batches, what):
        pass

    def output(self, surface, **shown):
        pass


def integers(structure, prefix=''):
    """Every integer field of a ctypes structure by name, bit-fields and the
    members of its unions included."""
    found = {}
    for name, *_ in structure._fields_:
        if name in RESERVED:
            continue
        value = getattr(structure, name)
        if isinstance(value, int):
            found[prefix + name] = value
        elif isinstance(value, ctypes.Union):
            for member, *_ in value._fields_:
                if isinstance(getattr(value, member), ctypes.Structure):
                    found.update(integers(getattr(value, member)))
    return found


def recorded(monkeypatch, module):
    """Record what `module` stores and builds while a test runs.

    Three lists: every ``(structure, field, value)`` given to :func:`store`,
    every ``(array, values)`` given to :func:`store_each`, and every
    ``(arguments, buffer)`` of its picture and slice parameter functions.
    """
    stored, arrays, built = [], [], []

    def storing(target, **values):
        stored.extend((target, name, value) for name, value in values.items())
        store(target, **values)

    def storing_each(target, values, name):
        arrays.append((target, list(values)))
        store_each(target, values, name)

    def building(name):
        build = getattr(module, name)

        def wrapper(*arguments, **named):
            buffer = build(*arguments, **named)
            built.append((arguments, buffer))
            return buffer
        monkeypatch.setattr(module, name, wrapper)

    monkeypatch.setattr(module, 'store', storing)
    monkeypatch.setattr(module, 'store_each', storing_each)
    # HEVC's structures of one reference are built by H.264's function.
    monkeypatch.setattr(h264_decode, 'store', storing)
    building('picture_parameters')
    building('slice_parameters')
    return stored, arrays, built


def through(control_class, path):
    """One fixture through a control layer and a session with no driver."""
    session = Session()
    control = control_class(session, 4)
    session.control = control
    with AnnexBReader(path) as reader:
        for index in range(len(reader)):
            control.decode(reader.sample(index, parameter_sets=index == 0))
    control.flush()


def same_named(buffer, *sources):
    """``(field, written, parsed)`` for each integer field of `buffer` that
    one of `sources` has an integer attribute of the same name for."""
    for name, written in integers(buffer).items():
        for source in sources:
            parsed = getattr(source, name, None)
            if isinstance(parsed, int):
                yield name, written, int(parsed)
                break


H264_STREAMS = sorted(path for path in DATA.glob('*.h264') if 'interlaced' not in path.name)
HEVC_STREAMS = sorted(DATA.glob('*.h265'))


def names(paths):
    return [path.name for path in paths]


class TestEveryFieldOfEveryFixture:
    def check(self, stored, arrays):
        assert len(stored) > 1000
        for target, name, value in stored:
            # A value read back as itself fitted, and from an unsigned field
            # was not negative: no `size - 1` written here wrapped below zero.
            assert getattr(target, name) == value, (type(target).__name__, name)
        for target, values in arrays:
            assert list(target)[:len(values)] == values

    @pytest.mark.parametrize('path', H264_STREAMS, ids=names(H264_STREAMS))
    def test_h264_fields_read_back_as_the_values_parsed(self, monkeypatch, path):
        stored, arrays, built = recorded(monkeypatch, h264_decode)
        through(h264_decode.H264Decoding, path)
        self.check(stored, arrays)
        compared = set()
        for arguments, buffer in built:
            if isinstance(buffer, api.VAPictureParameterBufferH264):
                header = arguments[0].header
                pairs = list(same_named(buffer, header.sps, header.pps, header))
                assert buffer.picture_width_in_mbs_minus1 == header.sps.width_in_mbs - 1
                assert buffer.num_ref_frames == header.sps.max_num_ref_frames
            else:
                header = arguments[0]
                pairs = list(same_named(buffer, header))
                assert buffer.slice_data_bit_offset == header.header_bits
            for name, written, parsed in pairs:
                assert written == parsed, name
                compared.add(name)
        assert len(compared) >= 30

    @pytest.mark.parametrize('path', HEVC_STREAMS, ids=names(HEVC_STREAMS))
    def test_hevc_fields_read_back_as_the_values_parsed(self, monkeypatch, path):
        stored, arrays, built = recorded(monkeypatch, hevc_decode)
        through(hevc_decode.HEVCDecoding, path)
        self.check(stored, arrays)
        compared = set()
        for arguments, buffer in built:
            if isinstance(buffer, hevc_api.VAPictureParameterBufferHEVC):
                header = arguments[1][0]
                sps, pps = header.sps, header.pps
                pairs = list(same_named(buffer, sps, pps, header))
                assert (buffer.sps_max_dec_pic_buffering_minus1
                        == sps.sps_max_dec_pic_buffering_minus1[-1])
                tiles = [size - 1 for size in pps.tile_columns(sps)]
                assert min(tiles) >= 0
                assert list(buffer.column_width_minus1)[:len(tiles)] == tiles[:19]
            else:
                header = arguments[0]
                pairs = [pair for pair in same_named(buffer, header)
                         if pair[0] != 'collocated_ref_idx']
                assert buffer.slice_data_byte_offset == header.header_bytes
                assert buffer.collocated_ref_idx == (
                    hevc_api.NO_REFERENCE if header.collocated_ref_idx is None
                    else header.collocated_ref_idx)
            for name, written, parsed in pairs:
                assert written == parsed, name
                compared.add(name)
        assert len(compared) >= 55


@pytest.mark.parametrize('module', [h264_decode, hevc_decode],
                         ids=['h264_decode', 'hevc_decode'])
def test_no_field_is_assigned_but_through_the_store(module):
    """The functions that build a buffer assign no attribute: each field of a
    structure goes through :func:`store`, which reads it back."""
    direct = []
    builders = [node for node in ast.parse(inspect.getsource(module)).body
                if isinstance(node, ast.FunctionDef)
                and node.name not in ('store', 'store_each')]
    assert len(builders) >= 5
    for builder in builders:
        for node in ast.walk(builder):
            targets = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
            direct += [f'{builder.name}: {ast.unparse(target)}' for target in targets
                       if isinstance(target, ast.Attribute)]
    assert direct == []


class TestTheStore:
    def test_a_value_that_fits_is_stored(self):
        parameters = api.VASliceParameterBufferH264()
        store(parameters, slice_qp_delta=-128, first_mb_in_slice=65535)
        assert (parameters.slice_qp_delta, parameters.first_mb_in_slice) == (-128, 65535)

    @pytest.mark.parametrize(('field', 'value'), [
        ('slice_data_bit_offset', 68024),       # 16 bits: reads back as 2488
        ('first_mb_in_slice', 70000),
        ('num_ref_idx_l0_active_minus1', 300),
        ('slice_qp_delta', 200),
        ('cabac_init_idc', -1),
    ])
    def test_a_value_wider_than_its_field_is_refused(self, field, value):
        parameters = api.VASliceParameterBufferH264()
        with pytest.raises(DecoderError, match=f'{field} is {value}'):
            store(parameters, **{field: value})

    def test_a_bit_field_is_held_to_its_width(self):
        bits = api.VAPictureParameterBufferH264().seq_fields.bits
        with pytest.raises(DecoderError, match='log2_max_frame_num_minus4 is 16'):
            store(bits, log2_max_frame_num_minus4=16)

    def test_an_array_takes_values_that_fit_its_elements(self):
        parameters = hevc_api.VASliceParameterBufferHEVC()
        store_each(parameters.luma_offset_l0, [-128, 127], 'luma offset')
        assert list(parameters.luma_offset_l0)[:3] == [-128, 127, 0]
        with pytest.raises(DecoderError, match='luma offset 1 is -1000'):
            store_each(parameters.luma_offset_l0, [0, -1000], 'luma offset')

    def test_an_array_takes_no_more_values_than_it_has_elements(self):
        parameters = hevc_api.VAPictureParameterBufferHEVC()
        with pytest.raises(DecoderError, match='20 values for tile column width'):
            store_each(parameters.column_width_minus1, [0] * 20, 'tile column width')
