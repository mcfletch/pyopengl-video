"""The decoder interface and its registry, exercised without any hardware."""
import pytest

from pyopengl_video import decoder as decoder_module
from pyopengl_video import decoders, open_decoder
from pyopengl_video.backend import Backend, VideoError
from pyopengl_video.colour import Colour
from pyopengl_video.decoder import (
    DecodedFrame,
    Decoder,
    DecoderError,
    DecoderUnavailable,
    Sample,
)
from pyopengl_video.encoder import EncoderError


class FakeDecoder(Decoder):
    """Hands every sample straight back as a frame of no textures."""

    def __init__(self, codec='h264', headers=b'', **options):
        self.codec = codec
        self.headers = headers
        self.options = options
        self.size = (320, 240)
        self.held = []

    def decode(self, sample):
        frame = DecodedFrame(timestamp=sample.timestamp, duration=sample.duration,
                             planes=(), layout='nv12', visible=(0, 0, 320, 240),
                             texture_size=(320, 256), colour=Colour(),
                             on_release=self.held.remove)
        self.held.append(frame)
        return [frame]

    def flush(self):
        return []

    def reset(self):
        pass

    def close(self):
        pass


@pytest.fixture
def only_fake_backend(monkeypatch):
    backend = Backend(name='fake', vendor='none', codecs=frozenset({'h264'}),
                      max_size=(640, 480), zero_copy=False, probe=lambda: True,
                      factory=FakeDecoder)
    monkeypatch.setattr(decoder_module, 'BACKENDS', [backend])
    return backend


class TestErrors:
    def test_one_base_covers_encoding_and_decoding(self):
        assert issubclass(EncoderError, VideoError)
        assert issubclass(DecoderError, VideoError)

    def test_nothing_can_decode_is_a_decoder_error(self):
        assert issubclass(DecoderUnavailable, DecoderError)

    def test_they_are_still_runtime_errors(self):
        assert issubclass(DecoderError, RuntimeError)


@pytest.mark.usefixtures('only_fake_backend')
class TestRegistry:
    def test_decoders_lists_the_available_backends(self):
        assert [backend.name for backend in decoders()] == ['fake']

    def test_open_decoder_builds_the_first_capable_backend(self):
        built = open_decoder('h264', headers=b'sets', size=(320, 240), extra=1)
        assert isinstance(built, FakeDecoder)
        assert built.headers == b'sets'
        assert built.options == {'size': (320, 240), 'extra': 1}

    def test_a_codec_nothing_decodes_is_refused_by_name(self):
        with pytest.raises(DecoderUnavailable, match='av1'):
            open_decoder('av1')

    def test_a_size_beyond_every_backend_is_refused(self):
        with pytest.raises(DecoderUnavailable, match='1920x1080'):
            open_decoder('h264', size=(1920, 1080))

    def test_a_backend_named_that_is_not_here_is_refused(self):
        with pytest.raises(DecoderUnavailable, match='nvdec'):
            open_decoder('h264', backend='nvdec')

    def test_the_unavailable_say_why(self, monkeypatch, only_fake_backend):
        declining = only_fake_backend.replace(
            name='shy', probe=lambda: False, explain=lambda: 'no driver here')
        monkeypatch.setattr(decoder_module, 'BACKENDS', [declining])
        with pytest.raises(DecoderUnavailable, match='shy: no driver here'):
            open_decoder('h264')


class TestDecodedFrames:
    def frame(self, decoder):
        return decoder.decode(Sample(b'', timestamp=3000, duration=1500,
                                     keyframe=True, index=0))[0]

    def test_a_frame_belongs_to_the_decoder_until_released(self):
        decoder = FakeDecoder()
        frame = self.frame(decoder)
        assert decoder.held == [frame]
        frame.release()
        assert decoder.held == []
        assert frame.released

    def test_release_is_safe_twice(self):
        decoder = FakeDecoder()
        frame = self.frame(decoder)
        frame.release()
        frame.release()
        assert decoder.held == []

    def test_a_frame_is_released_at_the_end_of_a_with_block(self):
        decoder = FakeDecoder()
        with self.frame(decoder) as frame:
            assert frame.timestamp == 3000
        assert frame.released

    def test_the_size_is_the_visible_rectangles(self):
        frame = self.frame(FakeDecoder())
        assert frame.size == (320, 240)

    def test_a_decoder_closes_at_the_end_of_a_with_block(self):
        closed = []
        decoder = FakeDecoder()
        decoder.close = lambda: closed.append(True)
        with decoder:
            pass
        assert closed == [True]
