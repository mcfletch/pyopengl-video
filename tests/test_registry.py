"""Backend discovery and selection, exercised without any hardware."""
import pytest

from pyopengl_video import encoder as encoder_module
from pyopengl_video import encoders, open_encoder
from pyopengl_video.encoder import Backend, Encoder, EncoderUnavailable, Packet


class FakeEncoder(Encoder):
    """Records what it was asked for; encodes nothing."""

    def __init__(self, width, height, codec='h264', **options):
        self.size = (width, height)
        self.codec = codec
        self.options = options

    def register(self, texture, target=None):
        return texture

    def encode(self, handle, timestamp, duration=0, force_idr=False):
        return []

    def flush(self):
        return []

    def headers(self):
        return b''

    def close(self):
        pass


@pytest.fixture
def only_fake_backend(monkeypatch):
    """Replace the backend registry with one fake that supports h264 up to 640x480."""
    backend = Backend(
        name='fake', vendor='none', codecs=frozenset({'h264'}),
        max_size=(640, 480), zero_copy=False,
        probe=lambda: True, factory=FakeEncoder,
    )
    monkeypatch.setattr(encoder_module, 'BACKENDS', [backend])
    return backend


def test_encoders_lists_available_backends(only_fake_backend):
    assert [b.name for b in encoders()] == ['fake']


def test_encoders_omits_backends_that_do_not_probe(monkeypatch, only_fake_backend):
    monkeypatch.setattr(encoder_module, 'BACKENDS', [only_fake_backend.replace(probe=lambda: False)])
    assert encoders() == []


def test_open_encoder_builds_the_first_capable_backend(only_fake_backend):
    enc = open_encoder(320, 240, codec='h264', bitrate=1_000_000)
    assert isinstance(enc, FakeEncoder)
    assert enc.size == (320, 240)
    assert enc.options['bitrate'] == 1_000_000


def test_open_encoder_rejects_a_codec_no_backend_has(only_fake_backend):
    with pytest.raises(EncoderUnavailable) as caught:
        open_encoder(320, 240, codec='av1')
    assert 'av1' in str(caught.value)


def test_open_encoder_rejects_a_size_no_backend_can_reach(only_fake_backend):
    with pytest.raises(EncoderUnavailable) as caught:
        open_encoder(1920, 1080, codec='h264')
    assert '1920x1080' in str(caught.value)


def test_open_encoder_names_a_backend_directly(only_fake_backend):
    assert isinstance(open_encoder(320, 240, backend='fake'), FakeEncoder)
    with pytest.raises(EncoderUnavailable):
        open_encoder(320, 240, backend='nonesuch')


def test_packet_carries_what_a_muxer_needs():
    packet = Packet(data=b'\x00\x00\x00\x01', timestamp=1500, duration=1500, keyframe=True)
    assert len(packet) == 4
    assert packet.keyframe


def test_the_nvenc_backend_reports_itself_unavailable_off_linux(monkeypatch):
    """NVIDIA supports the encoder's OpenGL device type on Linux alone."""
    from pyopengl_video import nvenc

    monkeypatch.setattr(nvenc, 'OPENGL_DEVICE_PLATFORM', False)
    assert nvenc.probe() is False


class TestSayingWhyThereIsNoEncoder:
    """"None available" is not an answer a caller can act on.

    A backend often declines for a reason the caller could fix -- a context of
    the wrong kind, a driver package not installed -- and that reason is worth
    more than the empty list it turns into.
    """

    def make(self, monkeypatch, *, available, explanation):
        backend = Backend(
            name='fake', vendor='none', codecs=frozenset({'h264'}),
            max_size=(640, 480), zero_copy=False,
            probe=lambda: available, factory=FakeEncoder,
            explain=lambda: explanation,
        )
        monkeypatch.setattr(encoder_module, 'BACKENDS', [backend])
        return backend

    def test_the_reason_reaches_the_caller(self, monkeypatch):
        self.make(monkeypatch, available=False,
                  explanation='this context cannot export a texture')
        with pytest.raises(EncoderUnavailable,
                           match='this context cannot export a texture'):
            open_encoder(320, 240)

    def test_the_backend_that_gave_it_is_named(self, monkeypatch):
        self.make(monkeypatch, available=False, explanation='no driver here')
        with pytest.raises(EncoderUnavailable, match='fake'):
            open_encoder(320, 240)

    def test_a_backend_that_is_available_explains_nothing(self, monkeypatch):
        self.make(monkeypatch, available=True, explanation='should not appear')
        with open_encoder(320, 240) as found:
            assert isinstance(found, FakeEncoder)

    def test_a_backend_with_nothing_to_say_is_left_out(self, monkeypatch):
        self.make(monkeypatch, available=False, explanation='')
        with pytest.raises(EncoderUnavailable) as raised:
            open_encoder(320, 240)
        assert 'fake:' not in str(raised.value)

    def test_a_backend_that_offers_no_explanation_at_all_is_fine(self,
                                                                 only_fake_backend,
                                                                 monkeypatch):
        """`explain` is optional, so an out-of-tree backend need not have one."""
        assert only_fake_backend.explain is None
        monkeypatch.setattr(encoder_module, 'BACKENDS',
                            [only_fake_backend.replace(probe=lambda: False)])
        with pytest.raises(EncoderUnavailable):
            open_encoder(320, 240)

    def test_an_explanation_that_raises_does_not_hide_the_error(self,
                                                                monkeypatch):
        """Explaining is a courtesy, and must not replace the real answer."""
        backend = Backend(
            name='fake', vendor='none', codecs=frozenset({'h264'}),
            max_size=(640, 480), zero_copy=False, probe=lambda: False,
            factory=FakeEncoder,
            explain=lambda: (_ for _ in ()).throw(RuntimeError('boom')),
        )
        monkeypatch.setattr(encoder_module, 'BACKENDS', [backend])
        with pytest.raises(EncoderUnavailable):
            open_encoder(320, 240)

    def test_asking_for_a_backend_by_name_says_why_that_one_declined(
            self, monkeypatch):
        self.make(monkeypatch, available=False, explanation='no driver here')
        with pytest.raises(EncoderUnavailable, match='no driver here'):
            open_encoder(320, 240, backend='fake')
