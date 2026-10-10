"""Opening a libva display: the refusals, which need no hardware to reach."""
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
