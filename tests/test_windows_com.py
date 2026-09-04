"""The COM helpers, and the input-handle base, without any hardware.

These need no GPU and no OpenGL context: they are the parts of the Windows shim
that are ordinary Python, kept testable on purpose.
"""
import sys

import pytest

from pyopengl_video.encoder import EncoderError
from pyopengl_video.inputs import InputHandle
from pyopengl_video.windows import com


def test_a_guid_is_built_from_its_usual_spelling():
    factory = com.guid('770aae78-f26f-4dba-a829-253c83d1b387')
    assert factory.Data1 == 0x770AAE78
    assert factory.Data2 == 0xF26F
    assert factory.Data3 == 0x4DBA
    assert bytes(factory.Data4) == bytes.fromhex('a829253c83d1b387')


def test_guids_compare_by_value_and_hash_together():
    one = com.guid('770aae78-f26f-4dba-a829-253c83d1b387')
    same = com.guid('770aae78-f26f-4dba-a829-253c83d1b387')
    other = com.guid('9b7e4e00-342c-4106-a19f-4f2704f689f0')
    assert one == same and hash(one) == hash(same)
    assert one != other
    assert len({one, same, other}) == 2


def test_a_guid_does_not_compare_equal_to_other_things():
    assert com.guid('770aae78-f26f-4dba-a829-253c83d1b387') != 'not a guid'


def test_check_passes_success_through_and_raises_on_a_failure():
    assert com.check(0, 'CreateThing') == 0
    assert com.check(1, 'CreateThing') == 1        # S_FALSE is not a failure
    with pytest.raises(com.InteropError) as caught:
        com.check(-2005270526, 'IDXGIFactory1::EnumAdapters1')
    assert 'EnumAdapters1' in str(caught.value)
    assert '0x887A0002' in str(caught.value)


def test_an_interop_error_is_an_encoder_error_carrying_the_result():
    error = com.InteropError(-2147024882, 'D3D11CreateDevice', 'out of memory')
    assert isinstance(error, EncoderError)
    assert error.result == 0x8007000E
    assert error.call == 'D3D11CreateDevice'
    assert 'out of memory' in str(error)


def test_releasing_nothing_is_harmless():
    com.release(None)


def test_loading_a_library_that_is_not_there_says_so_as_an_encoder_error():
    with pytest.raises(com.InteropError) as caught:
        com.load('no-such-library-here.dll')
    assert 'no-such-library-here.dll' in str(caught.value)


@pytest.mark.skipif(sys.platform != 'win32', reason='loads a Windows library')
def test_loading_the_same_library_twice_gives_the_same_handle():
    assert com.load('dxgi.dll') is com.load('dxgi.dll')


# --------------------------------------------------------- the handle base

def test_the_base_handle_is_always_available_for_drawing():
    """A texture that belongs to OpenGL alone needs no handover."""
    handle = InputHandle()
    with handle.for_drawing() as held:
        assert held is handle


def test_closing_an_empty_handle_touches_no_opengl():
    """Nothing to give back means nothing is called, so this needs no context."""
    handle = InputHandle()
    handle.close()
    handle.close()
    assert handle.framebuffer == 0


@pytest.mark.skipif(sys.platform != 'win32', reason='WGL types are Windows-only')
def test_a_handle_array_accepts_either_spelling_of_a_handle():
    """The two dispatch implementations return different Python types for it.

    ``wglDXRegisterObjectNV`` hands back an opaque pointer object under the
    compiled layer and a plain integer under ctypes, and both name the same
    object. Building the array from whichever type turned up fails on the
    integer, so the address is what is taken.
    """
    import ctypes

    from pyopengl_video.windows.interop import handle_array

    address = 0x12345678
    from_integer = handle_array(address)
    from_pointer = handle_array(ctypes.c_void_p(address))
    assert len(from_integer) == 1 and len(from_pointer) == 1
    assert from_integer[0] == from_pointer[0] == address


@pytest.mark.skipif(sys.platform != 'win32', reason='WGL types are Windows-only')
def test_a_null_handle_makes_an_array_rather_than_failing():
    import ctypes

    from pyopengl_video.windows.interop import handle_array

    assert handle_array(ctypes.c_void_p())[0] in (0, None)


def test_a_handle_that_does_not_own_its_texture_keeps_it_on_close():
    handle = InputHandle()
    handle.texture = 42
    handle.owns_texture = False
    handle.close()
    assert handle.texture == 42
