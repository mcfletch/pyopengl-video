"""Discovery of the VA-API backend: the record it publishes, and its probe.

Runs with no GPU and no libva -- what a machine with hardware then does is in
``tests/test_vaapi_encode.py``.
"""
import subprocess
import sys

import pytest

from pyopengl_video import encoder as encoder_module
from pyopengl_video import vaapi
from pyopengl_video.encoder import Backend


@pytest.fixture(autouse=True)
def fresh_probe():
    """Each test asks the machine again rather than reading a cached answer."""
    vaapi.forget_probe()
    yield
    vaapi.forget_probe()


class TestTheRecord:
    def test_it_is_registered_among_the_backends_that_ship(self):
        assert vaapi.BACKEND in encoder_module.BACKENDS

    def test_it_says_what_it_is(self):
        backend = vaapi.BACKEND
        assert isinstance(backend, Backend)
        assert backend.name == 'vaapi'
        assert backend.vendor == 'Intel/AMD'
        assert backend.codecs == frozenset({'h264'})
        assert backend.zero_copy is True

    def test_it_takes_the_sizes_h264_reaches(self):
        assert vaapi.BACKEND.supports(1920, 1080, 'h264')
        assert vaapi.BACKEND.supports(3840, 2160, 'h264')
        assert not vaapi.BACKEND.supports(8192, 4320, 'h264')
        assert not vaapi.BACKEND.supports(1920, 1080, 'hevc')

    def test_importing_the_package_does_not_import_the_encoder(self):
        """Discovery costs one library load, not the whole backend.

        In a fresh interpreter, because this one has long since imported it.
        """
        found = subprocess.run(
            [sys.executable, '-c',
             'import sys, pyopengl_video; '
             "print('pyopengl_video.vaapi.encoder' in sys.modules)"],
            capture_output=True, text=True, check=True)
        assert found.stdout.strip() == 'False'


class TestProbe:
    def test_it_is_false_off_linux(self, monkeypatch):
        monkeypatch.setattr(vaapi, 'SUPPORTED_PLATFORM', False)
        assert vaapi.probe() is False

    def test_it_answers_false_rather_than_raising(self, monkeypatch):
        """Discovery calls this, and one backend must not stop the others.

        Whatever goes wrong -- a driver that will not load, a device that
        answers nonsense -- the answer is that this backend is unavailable.
        """
        monkeypatch.setattr(vaapi, '_context_can_export',
                            lambda: (_ for _ in ()).throw(RuntimeError('boom')))
        assert vaapi.probe() is False

    def test_a_backend_that_cannot_answer_still_leaves_the_others_findable(
            self, monkeypatch):
        from pyopengl_video import encoders

        monkeypatch.setattr(vaapi, '_any_device_encodes',
                            lambda: (_ for _ in ()).throw(RuntimeError('boom')))
        assert 'vaapi' not in [found.name for found in encoders()]

    def test_a_missing_library_is_not_an_error(self, monkeypatch):
        from pyopengl_video.vaapi import api

        def refuse():
            raise OSError('libva.so.2: cannot open shared object file')

        monkeypatch.setattr(api.VA, 'instance', staticmethod(refuse))
        monkeypatch.setattr(vaapi, '_context_can_export', lambda: True)
        assert vaapi._any_device_encodes() is False

    def test_a_machine_with_no_render_node_has_no_encoder(self, monkeypatch):
        from pyopengl_video.vaapi import api

        monkeypatch.setattr(api, 'render_nodes', list)
        monkeypatch.setattr(vaapi, '_context_can_export', lambda: True)
        assert vaapi._any_device_encodes() is False

    def test_a_device_that_will_not_open_is_not_an_encoder(self):
        class Refuses:
            def __getattr__(self, name):
                raise OSError('no such device')

        assert vaapi._device_encodes(Refuses(), '/dev/dri/renderD404') is False

    def test_the_answer_is_kept_rather_than_asked_again(self, monkeypatch):
        """Opening a driver costs enough that discovery must not repeat it."""
        monkeypatch.setattr(vaapi, '_context_can_export', lambda: True)
        asked = []

        def count():
            asked.append(1)
            return True

        monkeypatch.setattr(vaapi, '_any_device_encodes', count)
        assert vaapi.probe() and vaapi.probe() and vaapi.probe()
        assert len(asked) == 1

    def test_forgetting_the_answer_asks_again(self, monkeypatch):
        monkeypatch.setattr(vaapi, '_context_can_export', lambda: True)
        asked = []
        monkeypatch.setattr(vaapi, '_any_device_encodes',
                            lambda: asked.append(1) or True)
        vaapi.probe()
        vaapi.forget_probe()
        vaapi.probe()
        assert len(asked) == 2


class TestTheContextItWouldRecordFrom:
    """A context that cannot export is a context this backend cannot serve."""

    def test_no_current_context_is_not_held_against_the_machine(self,
                                                                monkeypatch):
        from OpenGL import EGL

        monkeypatch.setattr(EGL, 'eglGetCurrentContext', lambda: None)
        assert vaapi._context_can_export() is True

    def test_a_context_that_cannot_export_makes_the_backend_unavailable(
            self, monkeypatch):
        from OpenGL import EGL

        from pyopengl_video.linux import dmabuf

        monkeypatch.setattr(EGL, 'eglGetCurrentContext', lambda: 1)
        monkeypatch.setattr(dmabuf, 'unavailable_because',
                            lambda: 'this OpenGL context is not an EGL context')
        assert vaapi._context_can_export() is False
        assert vaapi.probe() is False
