"""Reading the WGL extension string.

The driver answers with a space-separated ASCII list, and deciding which names
are in it is arithmetic on bytes -- no context, no device and no Windows.  It
is separated from :func:`~pyopengl_video.windows.interop.wgl_extensions` for
that reason, so the part that can be checked anywhere is checked everywhere:
the surrounding function needs a live WGL context and its tests skip off
Windows, which is where a mistake in the parsing would otherwise hide.
"""

import pytest

from pyopengl_video.windows.interop import parse_extension_string


def test_the_names_come_back_as_a_set():
    found = parse_extension_string(b'WGL_ARB_extensions_string WGL_NV_DX_interop2')
    assert found == {'WGL_ARB_extensions_string', 'WGL_NV_DX_interop2'}


def test_the_extension_the_backend_looks_for_is_found_among_many():
    from pyopengl_video.windows.interop import EXTENSION

    text = b' '.join(
        [b'WGL_ARB_extensions_string', EXTENSION.encode('ascii'), b'WGL_EXT_swap_control']
    )
    assert EXTENSION in parse_extension_string(text)


@pytest.mark.parametrize('text', [b'', None])
def test_no_answer_is_no_extensions(text):
    """No context and a driver that declined mean the same to a caller."""
    assert parse_extension_string(text) == set()


def test_runs_of_spaces_do_not_become_empty_names():
    """A trailing separator is common, and an empty name is not an extension."""
    assert parse_extension_string(b'  WGL_one   WGL_two  ') == {'WGL_one', 'WGL_two'}


def test_a_byte_that_is_not_ascii_does_not_raise():
    """A driver that answers with rubbish is a driver offering nothing usable.

    Decoding is forgiving so that one bad byte costs the name it is in rather
    than the whole answer, and never the process.
    """
    found = parse_extension_string(b'WGL_good \xff\xfe WGL_also_good')
    assert {'WGL_good', 'WGL_also_good'} <= found
