"""What a caller has to catch, across every backend.

`docs/usage.md` tells a caller that one `except EncoderError` is enough to know
whether recording failed. That is a promise about all three backends, and each
keeps it differently: oneVPL's and NVENC's driver errors *are* encoder errors,
and the VA-API backend translates libva's and the DMA-BUF shim's at its own
boundary, because those two serve more than encoding.

No hardware here -- the declarations import on any machine, which is the point:
a backend that arrives without keeping the promise fails this on a laptop.
"""
import pytest

from pyopengl_video.encoder import EncoderError, EncoderUnavailable
from pyopengl_video.linux.dmabuf import DMABufError
from pyopengl_video.nvenc.api import NVENCError
from pyopengl_video.vaapi.api import VAError
from pyopengl_video.vpl.api import VPLError

#: Every error a backend raises at a caller from a driver call.
DRIVER_ERRORS = [NVENCError, VPLError]


@pytest.mark.parametrize('error', DRIVER_ERRORS, ids=lambda e: e.__name__)
def test_a_driver_error_is_an_encoder_error(error):
    """So a caller wanting to know only whether recording failed catches one."""
    assert issubclass(error, EncoderError)


@pytest.mark.parametrize('error', DRIVER_ERRORS, ids=lambda e: e.__name__)
def test_it_is_still_a_runtime_error(error):
    """Widening what catches an error must not narrow it."""
    assert issubclass(error, RuntimeError)


def test_nothing_can_encode_is_an_encoder_error():
    assert issubclass(EncoderUnavailable, EncoderError)


class TestTheOnesThatAreNotEncoderErrors:
    """Two are deliberately not, and are translated where they cross over.

    :class:`~pyopengl_video.linux.dmabuf.DMABufError` comes from a shim shared
    by every Linux backend, and libva decodes as well as encodes, so neither is
    an encoder's error until an encoder is what raised it. Making them
    subclasses would put the wrong name on a decoder's failure later;
    translating at the boundary keeps both readings honest.
    """

    @pytest.mark.parametrize('error', [DMABufError, VAError],
                             ids=lambda e: e.__name__)
    def test_it_stands_on_its_own(self, error):
        assert issubclass(error, RuntimeError)
        assert not issubclass(error, EncoderError)

    def test_the_backend_translates_them(self):
        """The public methods of the VA-API encoder carry the translation."""
        from pyopengl_video.vaapi.encoder import VAAPIEncoder, as_encoder_error

        for name in ('__init__', 'register', 'encode', 'flush'):
            method = getattr(VAAPIEncoder, name)
            assert getattr(method, '__wrapped__', None) is not None, (
                f'VAAPIEncoder.{name} is not translated')

        @as_encoder_error
        def raises(kind):
            raise kind

        for kind in (DMABufError('no export'), VAError(-1, 'vaCreateConfig')):
            with pytest.raises(EncoderError) as raised:
                raises(kind)
            assert raised.value.__cause__ is kind
