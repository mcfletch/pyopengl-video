"""The decoder's surface bookkeeping, and a control layer against a stand-in session.

No driver. The session is given pools that record what is asked of them, and
the H.264 control layer is given a session that does the same, so what each
decides is checked on any machine.
"""
import json
from pathlib import Path

import pytest

from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.colour import Colour
from pyopengl_video.decoder import DecoderError
from pyopengl_video.vaapi import decoder as decoder_module
from pyopengl_video.vaapi.control import Format
from pyopengl_video.vaapi.decoder import LAYOUTS, SurfaceOrder
from pyopengl_video.vaapi.h264_decode import H264Decoding

DATA = Path(__file__).parent / 'data' / 'decode'
FIXTURES = json.loads((DATA / 'h264.json').read_text())['streams']


class FakeVA:
    def vaSyncSurface(self, display, surface):  # noqa: ARG002 - libva's signature
        return 0


class FakeDevice:
    node = '/dev/dri/fake'
    display = object()
    va = FakeVA()

    def check(self, status, call, context=''):
        assert status == 0, (call, context)


class FakePool:
    """A pool of numbered surfaces with no driver behind them."""

    def __init__(self, device, format, count):
        self.device = device
        self.format = format
        self.profile = format.profile
        self.coded_size = self.size = format.coded_size
        self.layout = LAYOUTS[format.depth]
        self.count = count
        self.surfaces = list(range(100, 100 + count))
        self.textures = {surface: (surface * 2, surface * 2 + 1) for surface in self.surfaces}
        self.order = SurfaceOrder(self.surfaces)
        self.waited = []
        self.came_back = []
        self.closed = False

    def ready(self, surface):
        self.waited.append(surface)

    def returned(self, surface):
        self.came_back.append(surface)
        self.order.returned(surface)

    def close(self):
        self.closed = True


class Holding:
    """A control layer that holds whatever surfaces it is told to."""

    def __init__(self):
        self.surfaces = set()

    def held(self):
        return set(self.surfaces)


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(decoder_module, 'glFenceSync', lambda *_arguments: None)
    made = decoder_module._Session(FakeDevice(), output_slots=2, make_pool=FakePool)  # noqa: SLF001 - the class under test
    made.control = Holding()
    return made


def show(session, surface, timestamp=0):
    session.output(surface, timestamp=timestamp, duration=1, visible=(0, 0, 64, 48),
                   colour=Colour())


FORMAT = Format(profile=7, coded_size=(64, 48), surfaces=3, name='H.264 High')


class TestActivating:
    def test_the_pool_holds_the_layers_surfaces_and_the_output_slots(self, session):
        session.activate(FORMAT)
        assert session.pool.count == 3 + 2

    def test_the_same_format_again_keeps_the_pool(self, session):
        session.activate(FORMAT)
        pool = session.pool
        session.activate(FORMAT)
        session.activate(Format(profile=7, coded_size=(64, 48), surfaces=2))
        assert session.pool is pool

    @pytest.mark.parametrize('changed', [
        Format(profile=6, coded_size=(64, 48), surfaces=3),
        Format(profile=7, coded_size=(128, 96), surfaces=3),
        Format(profile=7, coded_size=(64, 48), surfaces=3, depth=10),
        Format(profile=7, coded_size=(64, 48), surfaces=4),
    ])
    def test_another_profile_size_depth_or_a_larger_count_replaces_it(self, session, changed):
        session.activate(FORMAT)
        old = session.pool
        session.activate(changed)
        assert session.pool is not old
        assert old.closed

    def test_a_replaced_pool_stays_open_while_a_frame_from_it_is_held(self, session):
        session.activate(FORMAT)
        old = session.pool
        show(session, session.acquire())
        frame, = session.take()
        session.activate(Format(profile=7, coded_size=(128, 96), surfaces=3))
        assert not old.closed
        frame.release()
        assert old.closed

    def test_acquiring_before_anything_is_activated_is_a_decoder_error(self, session):
        with pytest.raises(DecoderError, match='what the stream is'):
            session.acquire()


class TestAcquiring:
    def test_a_surface_the_layer_holds_is_not_handed_out(self, session):
        session.activate(FORMAT)
        session.control.surfaces = {100, 101}
        assert session.acquire() == 102

    def test_a_surface_lent_to_the_caller_is_not_handed_out(self, session):
        session.activate(FORMAT)
        show(session, 100)
        frame, = session.take()
        assert session.acquire() == 101
        frame.release()
        assert session.pool.came_back == [100]

    def test_the_surface_returned_longest_ago_is_chosen_first(self, session):
        session.activate(FORMAT)
        for surface in (100, 101):
            show(session, surface)
        first, second = session.take()
        second.release()
        first.release()
        assert session.acquire() == 102
        session.control.surfaces = {102, 103, 104}
        assert session.acquire() == 101

    def test_the_surface_is_waited_on_before_it_is_decoded_into(self, session):
        session.activate(FORMAT)
        surface = session.acquire()
        assert session.pool.waited == [surface]

    def test_every_surface_in_use_names_who_holds_them(self, session):
        session.activate(FORMAT)
        session.control.surfaces = {100, 101, 102}
        for surface in (103, 104):
            show(session, surface)
        held = session.take()
        with pytest.raises(DecoderError, match='3 by the pictures the decoder keeps '
                                               'and 2 by frames still held'):
            session.acquire()
        for frame in held:
            frame.release()


class TestShowing:
    def test_frames_come_out_in_the_order_they_were_shown_and_once(self, session):
        session.activate(FORMAT)
        for timestamp, surface in enumerate((101, 100, 102)):
            show(session, surface, timestamp)
        frames = session.take()
        assert [frame.timestamp for frame in frames] == [0, 1, 2]
        assert [frame.planes for frame in frames] == [(202, 203), (200, 201), (204, 205)]
        assert session.take() == []

    def test_a_frame_carries_the_pools_layout(self, session):
        session.activate(Format(profile=18, coded_size=(64, 48), surfaces=1, depth=10))
        show(session, 100)
        frame, = session.take()
        assert frame.layout == 'p010'
        assert frame.texture_size == (64, 48)

    def test_a_surface_shown_twice_comes_back_when_both_frames_are_released(self, session):
        """VP9 and AV1 show an existing frame again while the first is still held."""
        session.activate(FORMAT)
        show(session, 100, 0)
        show(session, 100, 1)
        first, second = session.take()
        first.release()
        assert session.pool.came_back == []
        session.control.surfaces = {101, 102, 103, 104}
        with pytest.raises(DecoderError, match='in use'):
            session.acquire()
        second.release()
        assert session.pool.came_back == [100]
        assert session.acquire() == 100

    def test_closing_releases_every_frame_and_every_pool(self, session):
        session.activate(FORMAT)
        show(session, 100)
        frame, = session.take()
        pool = session.pool
        session.close()
        assert frame.released
        assert pool.closed
        assert session.pool is None


# ----------------------------------------------------- the H.264 layer


class RecordingSession:
    """A session that hands out surface numbers and records what a layer does."""

    def __init__(self, layer_surfaces=lambda: set()):
        self.held = layer_surfaces
        self.formats = []
        self.submitted = []
        self.shown = []
        self.count = 0

    def activate(self, format):
        self.formats.append(format)
        self.count = format.surfaces

    def acquire(self):
        free = sorted(set(range(self.count)) - self.held())
        assert free, 'the layer holds more surfaces than its format asked for'
        return free[0]

    def submit(self, surface, batches, what):
        self.submitted.append((surface, [[kind for kind, _ in batch] for batch in batches], what))

    def output(self, surface, *, timestamp, duration, visible, colour):  # noqa: ARG002 - the session's signature
        self.shown.append((surface, timestamp, visible))


def h264_layer():
    session = RecordingSession()
    layer = H264Decoding(session, output_slots=4)
    session.held = layer.held
    return layer, session


@pytest.mark.parametrize('name', sorted(FIXTURES))
def test_every_picture_is_decoded_once_and_shown_in_display_order(name):
    layer, session = h264_layer()
    with AnnexBReader(DATA / f'{name}.h264') as stream:
        times = []
        for index in range(len(stream)):
            layer.decode(stream.sample(index))
            times.append(stream.presentation_time(index))
    layer.flush()
    assert len(session.submitted) == len(times)
    assert [timestamp for _, timestamp, _ in session.shown] == sorted(times)
    assert layer.held() == set()


def test_the_format_asks_for_the_buffer_and_the_picture_being_decoded():
    layer, session = h264_layer()
    with AnnexBReader(DATA / 'high_bpyramid.h264') as stream:
        layer.decode(stream.sample(0))
    wanted, = session.formats
    assert wanted.coded_size == (128, 80)
    assert wanted.depth == 8
    assert wanted.name == 'H.264 High'
    assert layer.size == (120, 68)


def test_a_picture_is_a_parameter_batch_and_then_a_batch_for_each_slice():
    layer, session = h264_layer()
    with AnnexBReader(DATA / 'high_bpyramid.h264') as stream:
        layer.decode(stream.sample(0))
    _, batches, _ = session.submitted[0]
    assert batches == [[0, 1], [4, 5], [4, 5]]       # picture and IQ; two slices


def test_a_reset_drops_what_was_held_and_decoding_resumes_at_a_key_frame():
    layer, session = h264_layer()
    with AnnexBReader(DATA / 'open_gop.h264') as stream:
        key = stream.sync_samples[1]
        for index in range(key):
            layer.decode(stream.sample(index))
        assert layer.held()
        layer.reset()
        assert layer.held() == set()
        shown_before = len(session.shown)
        for index in range(key, len(stream)):
            layer.decode(stream.sample(index, parameter_sets=index == key))
        layer.flush()
        later = sorted(stream.presentation_time(index) for index in range(key, len(stream)))
    assert [timestamp for _, timestamp, _ in session.shown[shown_before:]] == later
