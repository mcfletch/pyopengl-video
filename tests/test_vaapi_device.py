"""Opening a libva display: the refusals, which need no hardware to reach."""
import os

import pytest

from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.device import Device


def test_no_render_node_is_refused_by_saying_so():
    with pytest.raises(api.VAError, match='no DRM render node'):
        Device.first([], lambda _device: '', 'H.264 decoding')


def test_each_node_that_fails_is_named_with_its_reason(tmp_path):
    missing = str(tmp_path / 'renderD404')
    with pytest.raises(api.VAError) as caught:
        Device.first([missing], lambda _device: '', 'H.264 decoding')
    assert missing in str(caught.value)
    assert 'H.264 decoding' in str(caught.value)


def test_a_node_that_will_not_open_leaves_nothing_open(tmp_path):
    with pytest.raises(OSError):
        Device(str(tmp_path / 'renderD404'))


def test_a_device_the_caller_objects_to_is_closed_and_named():
    closed = []

    class Opened(Device):
        """A device that opens nothing, to put an objection to."""

        def __init__(self, node, va=None):  # noqa: ARG002 - Device's signature
            self.node = node

        def close(self):
            closed.append(self.node)

    with pytest.raises(api.VAError, match='renderD128: no decoder'):
        Opened.first(['/dev/dri/renderD128'], lambda _device: 'no decoder', 'decoding')
    assert closed == ['/dev/dri/renderD128']


# ------------------------------------------------- a stand-in for libva


class RecordingVA:
    """libva's entry points, recording each call and failing the ones named."""

    def __init__(self, failing=()):
        self.calls = []
        self.failing = set(failing)

    def check(self, status, call, context=''):
        if status != api.VA_STATUS_SUCCESS:
            raise api.VAError(status, call, context)

    def __getattr__(self, name):
        if not name.startswith('va'):
            raise AttributeError(name)

        def call(*_arguments):
            self.calls.append(name)
            if name == 'vaGetDisplayDRM':
                return 0xD15
            if name == 'vaQueryVendorString':
                return b'a stand-in'
            return 1 if name in self.failing else api.VA_STATUS_SUCCESS
        return call


def open_descriptors():
    return set(os.listdir('/proc/self/fd'))


needs_proc = pytest.mark.skipif(not os.path.isdir('/proc/self/fd'),
                                reason='descriptors are counted through /proc')


class TestOpeningFails:
    @needs_proc
    def test_a_display_that_will_not_initialise_is_terminated_before_its_node_closes(self):
        va = RecordingVA(failing={'vaInitialize'})
        before = open_descriptors()
        with pytest.raises(api.VAError, match='vaInitialize'):
            Device(os.devnull, va)
        assert va.calls[-2:] == ['vaInitialize', 'vaTerminate']
        assert open_descriptors() == before

    @needs_proc
    def test_a_node_that_gives_no_display_has_nothing_to_terminate(self):
        va = RecordingVA()
        va.vaGetDisplayDRM = lambda _fd: None
        before = open_descriptors()
        with pytest.raises(api.VAError, match='vaGetDisplayDRM'):
            Device(os.devnull, va)
        assert 'vaTerminate' not in va.calls
        assert open_descriptors() == before

    def test_a_device_whose_acceptance_raises_is_closed(self):
        closed = []

        class Opened(Device):
            def __init__(self, node, va=None):  # noqa: ARG002 - Device's signature
                self.node = node

            def close(self):
                closed.append(self.node)

        def accept(_device):
            raise api.VAError(-1, 'vaQueryConfigProfiles', 'asking')

        with pytest.raises(api.VAError, match='vaQueryConfigProfiles'):
            Opened.first(['/dev/dri/renderD128'], accept, 'decoding')
        assert closed == ['/dev/dri/renderD128']


class TestClosing:
    @needs_proc
    def test_the_node_is_closed_when_terminating_raises(self):
        va = RecordingVA()
        before = open_descriptors()
        device = Device(os.devnull, va)

        def refuse(_display):
            raise RuntimeError('the driver is gone')
        va.vaTerminate = refuse
        with pytest.raises(RuntimeError, match='the driver is gone'):
            device.close()
        assert open_descriptors() == before
        assert device.display is None
        device.close()


class TestPicture:
    def device(self, failing=()):
        return Device(os.devnull, RecordingVA(failing))

    def test_the_calls_of_a_picture_run_in_order_and_the_buffers_are_given_back(self):
        device = self.device()
        device.va.calls.clear()
        device.picture(1, 2, [[10, 11], [12]], 'a picture')
        assert device.va.calls == ['vaBeginPicture', 'vaRenderPicture', 'vaRenderPicture',
                                   'vaEndPicture', *['vaDestroyBuffer'] * 3]
        device.close()

    def test_a_picture_that_was_begun_is_ended_when_rendering_fails(self):
        device = self.device(failing={'vaRenderPicture'})
        device.va.calls.clear()
        with pytest.raises(api.VAError, match='vaRenderPicture'):
            device.picture(1, 2, [[10], [11]], 'a picture')
        assert device.va.calls == ['vaBeginPicture', 'vaRenderPicture', 'vaEndPicture',
                                   'vaDestroyBuffer', 'vaDestroyBuffer']
        device.close()

    def test_a_picture_that_would_not_begin_is_not_ended(self):
        device = self.device(failing={'vaBeginPicture'})
        device.va.calls.clear()
        with pytest.raises(api.VAError, match='vaBeginPicture'):
            device.picture(1, 2, [[10]], 'a picture')
        assert device.va.calls == ['vaBeginPicture', 'vaDestroyBuffer']
        device.close()
