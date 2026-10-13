"""The decoder's surface pool and session over a stand-in for libva.

No driver and no OpenGL: libva's entry points record what they are asked and
answer as a driver would, and the DMA-BUF import hands out numbers. What is
checked is the sequence of libva calls the package makes, and what it still
holds afterwards.
"""
import os

import pytest

from pyopengl_video.colour import Colour
from pyopengl_video.decoder import DecoderError
from pyopengl_video.glcontext import Owner
from pyopengl_video.linux import dmabuf
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi import decoder as decoder_module
from pyopengl_video.vaapi.control import Format
from pyopengl_video.vaapi.device import Device

SMALL = Format(profile=7, coded_size=(64, 64), surfaces=2, name='H.264 High')
LARGE = Format(profile=7, coded_size=(4096, 4096), surfaces=2, name='H.264 High')


class FakeVA:
    """libva, as far as a decode session calls it."""

    def __init__(self):
        self.log = []
        self.refused_size = None
        self.failing = set()
        self.raising = set()
        self.next_surface = 400
        self.next_context = 0x55
        self.live_contexts = set()
        self.live_surfaces = set()

    def check(self, status, call, context=''):
        if status != api.VA_STATUS_SUCCESS:
            raise api.VAError(status, call, context)

    def _answer(self, name):
        if name in self.raising:
            raise RuntimeError(f'{name} raised')
        return 1 if name in self.failing else 0

    def vaCreateConfig(self, display, profile, entrypoint, attributes, count, config):  # noqa: ARG002 - libva's signature
        config._obj.value = 7  # noqa: SLF001 - what a byref argument points at
        self.log.append('vaCreateConfig')
        return 0

    def vaQuerySurfaceAttributes(self, display, config, attributes, count):  # noqa: ARG002 - libva's signature
        return 0

    def vaCreateSurfaces(self, display, format, width, height, surfaces, count,  # noqa: ARG002 - libva's signature
                         attributes, attribute_count):  # noqa: ARG002 - libva's signature
        self.log.append(f'vaCreateSurfaces {count} x {width}x{height}')
        if (width, height) == self.refused_size:
            return 2
        for index in range(count):
            surfaces[index] = self.next_surface + index
        self.live_surfaces.update(surfaces)
        self.next_surface += 100
        return 0

    def vaCreateContext(self, display, config, width, height, flag, surfaces, count, context):  # noqa: ARG002 - libva's signature
        context._obj.value = self.next_context  # noqa: SLF001 - what a byref argument points at
        self.live_contexts.add(self.next_context)
        self.log.append(f'vaCreateContext {self.next_context:#x}')
        self.next_context += 1
        return 0

    def vaExportSurfaceHandle(self, display, surface, memory, flags, descriptor):  # noqa: ARG002 - libva's signature
        described = descriptor._obj  # noqa: SLF001 - what a byref argument points at
        described.num_objects = 1
        described.objects[0].fd = os.open(os.devnull, os.O_RDONLY)
        described.num_layers = 2
        described.layers[0].drm_format = api.DRM_FORMAT_R8
        described.layers[1].drm_format = api.DRM_FORMAT_GR88
        return 0

    def vaDestroyContext(self, display, context):  # noqa: ARG002 - libva's signature
        self.log.append(f'vaDestroyContext {context.value:#x}')
        self.live_contexts.discard(context.value)
        return self._answer('vaDestroyContext')

    def vaDestroySurfaces(self, display, surfaces, count):  # noqa: ARG002 - libva's signature
        self.log.append(f'vaDestroySurfaces {list(surfaces)}')
        self.live_surfaces.difference_update(surfaces)
        return self._answer('vaDestroySurfaces')

    def vaDestroyConfig(self, display, config):  # noqa: ARG002 - libva's signature
        self.log.append('vaDestroyConfig')
        return self._answer('vaDestroyConfig')

    def vaCreateBuffer(self, display, context, kind, size, count, data, buffer):  # noqa: ARG002 - libva's signature
        value = getattr(context, 'value', context)
        assert value in self.live_contexts, f'a buffer was made in destroyed context {value:#x}'
        buffer._obj.value = 900  # noqa: SLF001 - what a byref argument points at
        return 0

    def vaBeginPicture(self, display, context, target):  # noqa: ARG002 - libva's signature
        value = getattr(context, 'value', context)
        assert value in self.live_contexts, f'a picture was begun in destroyed context {value:#x}'
        assert target in self.live_surfaces, f'a picture was begun into destroyed surface {target}'
        self.log.append(f'vaBeginPicture {value:#x} into {target}')
        return 0

    def vaRenderPicture(self, display, context, buffers, count):  # noqa: ARG002 - libva's signature
        return 0

    def vaEndPicture(self, display, context):  # noqa: ARG002 - libva's signature
        return 0

    def vaDestroyBuffer(self, display, buffer):  # noqa: ARG002 - libva's signature
        return 0

    def vaSyncSurface(self, display, surface):  # noqa: ARG002 - libva's signature
        assert surface in self.live_surfaces, f'destroyed surface {surface} was waited on'
        return self._answer('vaSyncSurface')


class FakeDevice(Device):
    def __init__(self):
        self.va = FakeVA()
        self.display = 0xD15
        self.node = '/dev/dri/fake'
        self.vendor = 'a stand-in'
        self._fd = -1


class Holding:
    def held(self):
        return set()


@pytest.fixture
def imported(monkeypatch):
    """The texture names the stand-in import has handed out and not had back."""
    live = []
    names = iter(range(1, 10_000))

    def import_texture(fourcc, width, height, modifier, planes):  # noqa: ARG001 - the import's signature
        os.fstat(planes[0].fd)               # the descriptor is open while it is imported
        name = next(names)
        live.append(name)
        return name, object()

    monkeypatch.setattr(dmabuf, 'import_texture', import_texture)
    monkeypatch.setattr(dmabuf, 'release_imported_texture',
                        lambda texture, _image: live.remove(texture))
    monkeypatch.setattr(decoder_module, 'glFenceSync', lambda *_arguments: object())
    monkeypatch.setattr(decoder_module, 'glDeleteSync', lambda _fence: None)
    monkeypatch.setattr(decoder_module, 'glClientWaitSync', lambda *_arguments: None)
    return live


@pytest.fixture
def device():
    return FakeDevice()


@pytest.fixture
def session(device, imported):  # noqa: ARG001 - requested so the import is stood in for
    made = decoder_module._Session(device, output_slots=1)  # noqa: SLF001 - the class under test
    made.control = Holding()
    return made


def decode_one(session):
    surface = session.acquire()
    session.submit(surface, [[(api.VAPictureParameterBufferType, b'\0' * 8)]], 'a picture')
    session.output(surface, timestamp=0, duration=0, visible=(0, 0, 64, 64), colour=Colour())
    return session.take()


class TestASizeTheDeviceRefuses:
    """A stream goes from one size to a size the device refuses, and back."""

    def test_no_picture_reaches_the_destroyed_context_or_its_surfaces(self, session, device):
        session.activate(SMALL)
        device.va.refused_size = LARGE.coded_size
        with pytest.raises(api.VAError):
            session.activate(LARGE)
        with pytest.raises(DecoderError, match='could not be made'):
            decode_one(session)
        session.activate(SMALL)
        frame, = decode_one(session)
        assert device.va.log[-1] == 'vaBeginPicture 0x56 into 500'
        frame.release()
        session.close()
        assert not device.va.live_contexts and not device.va.live_surfaces

    def test_the_refused_pool_leaves_nothing_behind(self, session, device, imported):
        session.activate(SMALL)
        device.va.refused_size = LARGE.coded_size
        with pytest.raises(api.VAError):
            session.activate(LARGE)
        assert imported == []
        assert not device.va.live_contexts and not device.va.live_surfaces


class TestSynchronising:
    def test_a_surface_that_will_not_synchronise_is_not_lent(self, session, device):
        session.activate(SMALL)
        device.va.failing.add('vaSyncSurface')
        with pytest.raises(api.VAError, match='vaSyncSurface'):
            decode_one(session)
        assert session.take() == []
        device.va.failing.clear()
        assert decode_one(session)


class TestClosingAPool:
    def pool(self, device):
        return decoder_module._Pool(device, SMALL, 3)  # noqa: SLF001 - the class under test

    @pytest.mark.parametrize('raising', ['vaDestroyContext', 'vaDestroySurfaces',
                                         'vaDestroyConfig'])
    def test_every_release_runs_when_one_raises(self, device, imported, raising):
        pool = self.pool(device)
        device.va.raising.add(raising)
        with pytest.raises(RuntimeError, match=raising):
            pool.close()
        assert device.va.log[-3:] == ['vaDestroyContext 0x55',
                                      'vaDestroySurfaces [400, 401, 402]', 'vaDestroyConfig']
        assert imported == []
        assert pool.closed

    def test_a_second_close_releases_nothing(self, device, imported):  # noqa: ARG002 - requested so the import is stood in for
        pool = self.pool(device)
        pool.close()
        calls = len(device.va.log)
        pool.close()
        assert len(device.va.log) == calls

    def test_the_libva_objects_are_released_when_the_textures_cannot_be(
            self, device, imported):
        """The pool's OpenGL context is not current: its names are another's there."""
        pool = self.pool(device)
        pool._owner = Owner(('egl', 1))  # noqa: SLF001 - a context that is not the current one
        with pytest.raises(DecoderError, match='OpenGL context'):
            pool.close()
        assert not device.va.live_contexts and not device.va.live_surfaces
        assert len(imported) == 6
        assert not pool.closed
        pool._owner = Owner(None)  # noqa: SLF001 - the pool's context made current again
        pool.close()
        assert imported == []
        assert pool.closed

    def test_a_frame_returned_in_another_context_makes_no_fence(self, device, imported,  # noqa: ARG002 - requested so the import is stood in for
                                                                 monkeypatch):
        made = []
        monkeypatch.setattr(decoder_module, 'glFenceSync',
                            lambda *_arguments: made.append(1) or object())
        pool = self.pool(device)
        pool.returned(400)
        assert made == [1]
        pool._owner = Owner(('egl', 1))  # noqa: SLF001 - a context that is not the current one
        pool.returned(401)
        assert made == [1]
        pool._owner = Owner(None)  # noqa: SLF001 - the pool's context made current again
        pool.close()
