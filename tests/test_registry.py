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
