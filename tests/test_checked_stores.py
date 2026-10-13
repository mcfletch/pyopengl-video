"""Stores into the structures a driver reads, each read back.

ctypes keeps the low bits of an integer wider than its field and raises
nothing. No hardware.
"""
import ctypes

import pytest

from pyopengl_video.decoder import DecoderError
from pyopengl_video.vaapi import av1_decode, vp9_decode
from pyopengl_video.vaapi.control import Checked, store, store_each


class _Bits(ctypes.Structure):
    _fields_ = [('flag', ctypes.c_uint32, 1), ('kind', ctypes.c_uint32, 2)]


class _Picture(ctypes.Structure):
    _fields_ = [('width', ctypes.c_uint16), ('delta', ctypes.c_int8),
                ('bits', _Bits), ('sizes', ctypes.c_uint8 * 3),
                ('grid', (ctypes.c_uint8 * 2) * 2)]


class TestChecked:
    def test_a_value_that_fits_is_stored(self):
        picture = Checked(_Picture())
        picture.width, picture.delta = 65535, -128
        picture.bits.kind = 3
        picture.sizes[2] = 255
        picture.grid[1][0] = 7
        stored = picture.target
        assert (stored.width, stored.delta, stored.bits.kind) == (65535, -128, 3)
        assert stored.sizes[2] == 255 and stored.grid[1][0] == 7

    @pytest.mark.parametrize('assign, named', [
        (lambda picture: setattr(picture, 'width', 65536), 'width'),
        (lambda picture: setattr(picture, 'width', -1), 'width'),
        (lambda picture: setattr(picture, 'delta', 128), 'delta'),
        (lambda picture: setattr(picture.bits, 'flag', 2), 'bits.flag'),
        (lambda picture: setattr(picture.bits, 'kind', 4), 'bits.kind'),
        (lambda picture: picture.sizes.__setitem__(1, 256), 'sizes[1]'),
        (lambda picture: picture.grid[1].__setitem__(0, -1), 'grid[1][0]'),
    ])
    def test_a_value_that_does_not_fit_is_refused_by_name(self, assign, named):
        with pytest.raises(DecoderError) as caught:
            assign(Checked(_Picture()))
        assert f'_Picture.{named}' in str(caught.value)

    def test_an_element_past_the_array_is_refused(self):
        with pytest.raises(DecoderError, match=r'sizes\[3\]'):
            Checked(_Picture()).sizes[3] = 1

    def test_a_slice_is_stored_element_by_element(self):
        picture = Checked(_Picture())
        picture.sizes[:2] = [4, 5]
        assert list(picture.target.sizes) == [4, 5, 0]
        assert picture.sizes[:2] == [4, 5]
        with pytest.raises(DecoderError, match=r'sizes\[1\]'):
            picture.sizes[:2] = [4, 256]
        with pytest.raises(DecoderError, match='room for'):
            picture.sizes[:4] = [1, 2, 3, 4]
        with pytest.raises(DecoderError, match='room for'):
            picture.sizes[:2] = [1, 2, 3]

    def test_a_field_the_structure_lacks_is_refused(self):
        with pytest.raises(AttributeError):
            Checked(_Picture()).widht = 1

    def test_a_structure_is_stored_whole(self):
        picture = Checked(_Picture())
        picture.bits = _Bits(1, 2)
        assert picture.target.bits.kind == 2

    def test_what_is_read_is_the_plain_value(self):
        picture = Checked(_Picture())
        picture.width = 12
        assert picture.width == 12 and type(picture.width) is int
        assert len(picture.sizes) == 3


class TestStore:
    def test_fields_are_stored_and_read_back(self):
        picture = _Picture()
        store(picture, width=640, delta=-3)
        assert (picture.width, picture.delta) == (640, -3)
        with pytest.raises(DecoderError, match='width'):
            store(picture, width=70000)

    def test_an_array_takes_as_many_values_as_it_has_room_for(self):
        picture = _Picture()
        store_each(picture.sizes, [1, 2, 3], 'tile size')
        assert list(picture.sizes) == [1, 2, 3]
        with pytest.raises(DecoderError, match='tile size'):
            store_each(picture.sizes, [1, 2, 3, 4], 'tile size')
        with pytest.raises(DecoderError, match='tile size 1'):
            store_each(picture.sizes, [1, 256], 'tile size')


@pytest.mark.parametrize('module, builders', [
    (av1_decode, ('picture_parameters', 'tile_parameters')),
    (vp9_decode, ('picture_parameters', 'slice_parameters')),
])
def test_the_structures_a_driver_reads_are_filled_through_checked_stores(module, builders):
    """Each builder fills a Checked structure and hands back what it wraps."""
    import ast  # noqa: PLC0415 - used by this one test
    import inspect  # noqa: PLC0415 - used by this one test

    tree = ast.parse(inspect.getsource(module))
    for name in builders:
        builder = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == name)
        made = [node for node in ast.walk(builder) if isinstance(node, ast.Call)
                and ast.unparse(node.func).endswith('_api.' + ast.unparse(node.func).split('.')[-1])
                and ast.unparse(node.func).split('.')[-1].startswith('VA')]
        wrapped = [node for node in ast.walk(builder) if isinstance(node, ast.Call)
                   and ast.unparse(node.func) == 'Checked']
        assert made and len(wrapped) == len(made), f'{module.__name__}.{name}'
