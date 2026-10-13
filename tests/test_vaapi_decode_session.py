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


# ------------------------------------------- what a stream can do to it


REFUSED_SIZE = (4096, 4096)


def refusing_pool(device, format, count):
    """A pool factory standing in for a device that refuses one picture size."""
    if format.coded_size == REFUSED_SIZE:
        raise DecoderError('the device refuses this size')
    return FakePool(device, format, count)


class TestAFormatTheDeviceRefuses:
    """The stream changes to a size the device refuses, then back."""

    @pytest.fixture
    def refused(self, session):
        session._make_pool = refusing_pool  # noqa: SLF001 - the factory the session was given
        session.activate(FORMAT)
        old = session.pool
        with pytest.raises(DecoderError, match='refuses this size'):
            session.activate(Format(profile=7, coded_size=REFUSED_SIZE, surfaces=3))
        return session, old

    def test_the_session_holds_no_pool_and_the_old_one_is_closed(self, refused):
        session, old = refused
        assert old.closed
        assert session.pool is None

    def test_nothing_is_acquired_submitted_or_shown_without_a_pool(self, refused):
        session, _ = refused
        with pytest.raises(DecoderError, match='could not be made'):
            session.acquire()
        with pytest.raises(DecoderError, match='could not be made'):
            session.submit(100, [], 'a picture')
        with pytest.raises(DecoderError, match='could not be made'):
            show(session, 100)

    def test_the_first_format_again_makes_a_new_pool(self, refused):
        session, old = refused
        session.activate(FORMAT)
        assert session.pool is not old
        assert not session.pool.closed
        assert session.acquire() == 100


class TestSurfacesOfAnotherPool:
    def test_showing_a_surface_the_pool_does_not_hold_is_a_decoder_error(self, session):
        session.activate(FORMAT)
        with pytest.raises(DecoderError, match='surface 400'):
            show(session, 400)
        assert session.take() == []

    def test_submitting_to_a_surface_the_pool_does_not_hold_is_a_decoder_error(self, session):
        session.activate(FORMAT)
        with pytest.raises(DecoderError, match='surface 400'):
            session.submit(400, [], 'a picture')


class TestHowManySurfaces:
    def make(self, monkeypatch):
        made = []

        def counting(device, format, count):
            made.append(count)
            return FakePool(device, format, count)
        monkeypatch.setattr(decoder_module, 'glFenceSync', lambda *_arguments: None)
        session = decoder_module._Session(FakeDevice(), output_slots=2, make_pool=counting)  # noqa: SLF001 - the class under test
        return session, made

    @pytest.mark.parametrize('surfaces', [100_001, decoder_module.MAX_FORMAT_SURFACES + 1, 0, -1])
    def test_a_count_no_codec_needs_is_refused_before_a_pool_is_made(self, monkeypatch, surfaces):
        session, made = self.make(monkeypatch)
        with pytest.raises(DecoderError, match='surfaces'):
            session.activate(Format(profile=7, coded_size=(64, 48), surfaces=surfaces))
        assert made == []
        assert session.pool is None

    def test_the_most_any_codec_asks_for_is_accepted(self, monkeypatch):
        session, made = self.make(monkeypatch)
        session.activate(Format(profile=7, coded_size=(64, 48),
                                surfaces=decoder_module.MAX_FORMAT_SURFACES))
        assert made == [decoder_module.MAX_FORMAT_SURFACES + 2]


class TestClosingWhenAReleaseFails:
    def test_every_pool_is_closed_and_every_frame_released(self, session):
        session.activate(FORMAT)
        show(session, 100)
        frame, = session.take()
        first = session.pool
        session.activate(Format(profile=7, coded_size=(128, 96), surfaces=3))
        second = session.pool

        def failing():
            raise DecoderError('the textures belong to another context')
        first.close = failing
        with pytest.raises(DecoderError, match='another context'):
            session.close()
        assert frame.released
        assert second.closed
        assert session.pool is None

    def test_a_pool_that_would_not_close_is_closed_by_the_next_close(self, session):
        session.activate(FORMAT)
        pool = session.pool
        closing = pool.close

        def failing():
            raise DecoderError('the textures belong to another context')
        pool.close = failing
        with pytest.raises(DecoderError):
            session.close()
        pool.close = closing
        session.close()
        assert pool.closed


class TestTranslatingErrors:
    @pytest.mark.parametrize('raised', [
        IndexError('list index out of range'), MemoryError(), AssertionError('no picture'),
        ZeroDivisionError('division by zero'), KeyError(400), OverflowError('too large'),
        TypeError('not a number'), OSError(5, 'the device went away'),
    ])
    def test_anything_a_stream_or_a_driver_causes_is_a_decoder_error(self, raised):
        @decoder_module.as_decoder_error
        def method():
            raise raised
        with pytest.raises(DecoderError) as caught:
            method()
        assert caught.value.__cause__ is raised
        assert type(raised).__name__ in str(caught.value)

    def test_a_decoder_error_passes_through_as_it_is(self):
        original = DecoderError('refused')

        @decoder_module.as_decoder_error
        def method():
            raise original
        with pytest.raises(DecoderError) as caught:
            method()
        assert caught.value is original

    @pytest.mark.parametrize('kind', [KeyboardInterrupt, SystemExit, GeneratorExit])
    def test_an_interruption_is_not_an_error_of_the_stream(self, kind):
        @decoder_module.as_decoder_error
        def method():
            raise kind
        with pytest.raises(kind):
            method()


class Recording:
    def __init__(self, failing=False):
        self.closed = 0
        self.failing = failing

    def close(self):
        self.closed += 1
        if self.failing:
            raise RuntimeError('the session would not close')


class TestClosingTheDecoder:
    def decoder(self, **parts):
        made = object.__new__(decoder_module.VAAPIDecoder)
        made._closed = False  # noqa: SLF001 - a decoder built without a device
        for name, value in parts.items():
            setattr(made, name, value)
        return made

    def test_the_device_is_closed_when_the_session_will_not_close(self):
        session, device = Recording(failing=True), Recording()
        decoder = self.decoder(_session=session, _device=device)
        with pytest.raises(DecoderError, match='would not close'):
            decoder.close()
        assert device.closed == 1

    def test_a_decoder_that_never_finished_opening_closes(self):
        self.decoder().close()

    def test_a_closed_decoder_refuses_to_decode(self):
        decoder = self.decoder(_session=Recording(), _device=Recording())
        decoder.close()
        with pytest.raises(DecoderError, match='closed'):
            decoder.flush()


@pytest.mark.parametrize('slots', [0, -1, decoder_module.MAX_OUTPUT_SLOTS + 1, 2.5, 'four'])
def test_output_slots_outside_their_range_are_refused_before_a_device_is_opened(
        monkeypatch, slots):
    def refuse(*_arguments):
        raise AssertionError('a device was opened for a setting that should be refused')
    monkeypatch.setattr(decoder_module.Device, 'first', refuse)
    with pytest.raises(DecoderError, match='output_slots'):
        decoder_module.VAAPIDecoder('h264', output_slots=slots)


class TestAPoolThatWillNotCloseWhole:
    """Its OpenGL context is not current, so its textures stay until it is."""

    def refusing(self, pool):
        closing = pool.close

        def refuse():
            raise DecoderError('another OpenGL context is current')
        pool.close = refuse
        return closing

    def test_a_replaced_pool_is_kept_for_the_sessions_close(self, session):
        session.activate(FORMAT)
        old = session.pool
        closing = self.refusing(old)
        with pytest.raises(DecoderError, match='another OpenGL context'):
            session.activate(Format(profile=7, coded_size=(128, 96), surfaces=3))
        assert session.pool is None
        old.close = closing
        session.close()
        assert old.closed

    def test_a_retired_pool_whose_last_frame_comes_back_is_kept_too(self, session):
        session.activate(FORMAT)
        old = session.pool
        show(session, session.acquire())
        frame, = session.take()
        session.activate(Format(profile=7, coded_size=(128, 96), surfaces=3))
        closing = self.refusing(old)
        with pytest.raises(DecoderError, match='another OpenGL context'):
            frame.release()
        old.close = closing
        session.close()
        assert old.closed
