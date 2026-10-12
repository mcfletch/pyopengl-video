"""The backend record: what it says it can do, and what discovery narrows that to."""
import pytest

from pyopengl_video import decoder as decoder_module
from pyopengl_video import decoders, encoders, open_decoder, open_encoder
from pyopengl_video import encoder as encoder_module
from pyopengl_video.backend import Backend, available, describe
from pyopengl_video.decoder import DecoderUnavailable
from pyopengl_video.encoder import EncoderUnavailable


def backend(**changes):
    made = Backend(name='fake', vendor='none', codecs=frozenset({'h264', 'hevc', 'av1'}),
                   max_size=(4096, 4096), zero_copy=True, probe=lambda: True,
                   factory=lambda *arguments, **options: (arguments, options))
    return made.replace(**changes)


class TestSizes:
    def test_a_codec_with_no_limit_of_its_own_takes_the_backends(self):
        assert backend().largest('hevc') == (4096, 4096)

    def test_a_codec_with_its_own_limit_is_held_to_it(self):
        made = backend(max_size=(8192, 4352), sizes={'h264': (4096, 4096)})
        assert made.largest('h264') == (4096, 4096)
        assert made.supports(8192, 4320, 'hevc')
        assert not made.supports(8192, 4320, 'h264')

    def test_a_codec_it_does_not_have_is_not_supported_at_any_size(self):
        assert not backend().supports(16, 16, 'vp9')


class TestWhatTheHardwareOffers:
    def test_a_backend_that_cannot_ask_is_reported_as_registered(self):
        made = backend()
        assert made.on_this_machine() is made

    def test_the_record_is_narrowed_to_the_codecs_the_hardware_has(self):
        found = backend(capabilities=lambda: {'hevc': (8192, 4352),
                                              'h264': (4096, 2304)}).on_this_machine()
        assert found.codecs == frozenset({'hevc', 'h264'})
        assert found.largest('hevc') == (8192, 4352)
        assert found.largest('h264') == (4096, 2304)
        assert found.max_size == (8192, 4352)

    def test_a_codec_the_hardware_has_and_the_backend_does_not_is_left_out(self):
        found = backend(capabilities=lambda: {'vp9': (4096, 4096),
                                              'av1': (4096, 4096)}).on_this_machine()
        assert found.codecs == frozenset({'av1'})

    def test_hardware_with_none_of_its_codecs_is_no_backend(self):
        assert backend(capabilities=dict).on_this_machine() is None
        assert list(available([backend(capabilities=dict)])) == []

    def test_a_backend_that_cannot_answer_is_left_out_and_the_rest_are_found(self):
        def fails():
            raise RuntimeError('the driver went away')

        found = list(available([backend(name='broken', capabilities=fails), backend()]))
        assert [entry.name for entry in found] == ['fake']

    def test_a_backend_that_does_not_probe_is_never_asked_its_codecs(self):
        asked = []
        made = backend(probe=lambda: False, capabilities=lambda: asked.append(1) or {})
        assert list(available([made])) == []
        assert asked == []


class TestDescribing:
    def test_each_backend_is_named_with_its_codecs_and_size(self):
        assert describe([backend(codecs=frozenset({'h264'}))]) == (
            "[('fake', ['h264'], (4096, 4096))]")

    def test_no_backends_is_said_in_a_word(self):
        assert describe([]) == 'none'


class TestOpening:
    @pytest.fixture
    def narrowed(self, monkeypatch):
        made = backend(capabilities=lambda: {'hevc': (1920, 1080)})
        monkeypatch.setattr(encoder_module, 'BACKENDS', [made])
        monkeypatch.setattr(decoder_module, 'BACKENDS', [made])

    @pytest.mark.usefixtures('narrowed')
    def test_discovery_reports_the_codecs_the_hardware_has(self):
        assert [sorted(found.codecs) for found in encoders()] == [['hevc']]
        assert [sorted(found.codecs) for found in decoders()] == [['hevc']]

    @pytest.mark.usefixtures('narrowed')
    def test_a_codec_is_opened_by_any_of_its_names(self):
        arguments, options = open_encoder(1280, 720, codec='h265')
        assert arguments == (1280, 720)
        assert options['codec'] == 'hevc'
        _, options = open_decoder('hvc1')
        assert options['codec'] == 'hevc'

    @pytest.mark.usefixtures('narrowed')
    def test_a_codec_the_hardware_lacks_is_refused_though_the_backend_drives_it(self):
        with pytest.raises(EncoderUnavailable, match="'hevc'"):
            open_encoder(1280, 720, codec='av1')
        with pytest.raises(DecoderUnavailable, match='av1'):
            open_decoder('av1')

    @pytest.mark.usefixtures('narrowed')
    def test_a_size_past_the_codecs_limit_is_refused(self):
        with pytest.raises(EncoderUnavailable, match='3840x2160'):
            open_encoder(3840, 2160, codec='hevc')
