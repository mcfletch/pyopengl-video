"""The VA-API encoder's own bookkeeping, over stand-ins for the device and libva.

No driver. The device records the pictures it is asked for and fails the one
it is told to, the control layer counts, and the coded buffer segments are
ctypes structures in this process's memory, so what the encoder does with a
failed submission, a driver's lengths and a release that raises is checked on
any machine.
"""
import ctypes

import pytest

from pyopengl_video.encoder import EncoderError
from pyopengl_video.glcontext import Owner
from pyopengl_video.linux import dmabuf
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi import encoder as encoder_module
from pyopengl_video.vaapi.av1_encode import AV1Encoding
from pyopengl_video.vaapi.control import (
    LARGEST_SIDE,
    CodedPicture,
    EncodeSettings,
    packed_frame_rate,
    rate_control_buffers,
)
from pyopengl_video.vaapi.encoder import VAAPIEncoder, as_encoder_error, colour_pipeline
from pyopengl_video.vaapi.h264_encode import H264Encoding
from pyopengl_video.vaapi.hevc_encode import HEVCEncoding


class Device:
    """A device that records each picture and fails the coded one it is told to."""

    display = 0xD15
    node = '/dev/dri/fake'

    def __init__(self, log):
        self.log = log
        self.fail_next_coded = False
        self.closed = 0

    def buffer(self, context, kind, payload):  # noqa: ARG002 - Device's signature
        return 900

    def picture(self, context, target, batches, what):  # noqa: ARG002 - Device's signature
        if what != 'colour conversion' and self.fail_next_coded:
            self.fail_next_coded = False
            raise api.VAError(1, 'vaEndPicture', what)
        self.log.append((what, target))

    def destroy_surfaces(self, surfaces):
        self.log.append(('vaDestroySurfaces', tuple(surfaces)))

    def close(self):
        self.closed += 1


class Control:
    def __init__(self, log):
        self.log = log
        self.pictures = 0

    def next_picture(self, force_key=False):  # noqa: ARG002 - the control layer's signature
        self.pictures += 1
        return CodedPicture(False, f'picture {self.pictures}')

    def buffers(self, picture, coded, reconstruction):
        self.log.append((picture.name, 'coded', coded, 'reconstruction', reconstruction))
        return []

    def coded(self, picture, data):  # noqa: ARG002 - the control layer's signature
        return data


class VA:
    """libva as the encoder reads a packet and closes: every call succeeds."""

    def __init__(self, log):
        self.log = log
        self.raising = set()

    def check(self, status, call, context=''):
        if status != api.VA_STATUS_SUCCESS:
            raise api.VAError(status, call, context)

    def __getattr__(self, name):
        if not name.startswith('va'):
            raise AttributeError(name)

        def call(*arguments):
            self.log.append((name, *(getattr(value, 'value', value)
                                     for value in arguments[1:2])))
            if name in self.raising:
                raise RuntimeError(f'{name} raised')
            return api.VA_STATUS_SUCCESS
        return call


def inside(encoder):
    """The encoder's own attributes by name, for a test to set up and read back."""
    return vars(encoder)


def counters(encoder):
    return tuple(inside(encoder)[name] for name in ('_slot', '_coded_slot', '_recon_slot'))


@pytest.fixture
def encoder(monkeypatch):
    """An encoder with two sources, three reconstructions and two coded buffers."""
    log = []
    made = object.__new__(VAAPIEncoder)
    made.api = VA(log)
    made.control = Control(log)
    made.size = (64, 64)
    made.frame_rate = (30, 1)
    inside(made).update(
        _device=Device(log), _closed=False, _stopped='', _sources=[10, 11],
        _reconstructions=[20, 21, 22], _coded_buffers=[30, 31], _coded_size=1 << 16,
        _pending=[], _slot=0, _recon_slot=0, _coded_slot=0, _conversion_kept=True,
        _context=api.VAContextID(1), _vpp_context=api.VAContextID(2),
        _config=api.VAConfigID(3), _vpp_config=api.VAConfigID(4),
        _registered=[encoder_module.InputHandle(texture=1, surface=5)])
    monkeypatch.setattr(encoder_module, 'glFinish', lambda: None)
    monkeypatch.setattr(made, '_collect', lambda _pointer: b'coded')
    made.log = log
    return made


class TestASubmissionThatFails:
    """The second of three pictures is refused by the driver."""

    def fail_the_second(self, encoder):
        handle, = inside(encoder)['_registered']
        assert encoder.encode(handle, 1) == []
        inside(encoder)['_device'].fail_next_coded = True
        before = counters(encoder)
        with pytest.raises(EncoderError, match='vaEndPicture'):
            encoder.encode(handle, 2)
        return handle, before

    def test_the_slot_counters_have_not_advanced(self, encoder):
        _, before = self.fail_the_second(encoder)
        assert counters(encoder) == before

    def test_the_encoder_refuses_another_picture(self, encoder):
        handle, _ = self.fail_the_second(encoder)
        submitted = len(encoder.log)
        with pytest.raises(EncoderError, match='stopped.*vaEndPicture'):
            encoder.encode(handle, 3)
        assert len(encoder.log) == submitted, 'nothing more reached the device'
        assert encoder.control.pictures == 2

    def test_the_picture_in_flight_is_still_read_from_its_own_buffer(self, encoder):
        self.fail_the_second(encoder)
        packet, = encoder.flush()
        assert packet.timestamp == 1
        assert ('vaSyncSurface', 10) in encoder.log
        assert ('vaMapBuffer', 30) in encoder.log

    def test_it_still_closes(self, encoder):
        self.fail_the_second(encoder)
        device = inside(encoder)['_device']
        encoder.close()
        assert device.closed == 1

    def test_a_packet_that_cannot_be_read_stops_it_too(self, encoder, monkeypatch):
        handle, = inside(encoder)['_registered']
        encoder.encode(handle, 1)

        def refuse(_pointer):
            raise EncoderError('the coded buffer overflowed')
        monkeypatch.setattr(encoder, '_collect', refuse)
        with pytest.raises(EncoderError, match='overflowed'):
            encoder.encode(handle, 2)
        with pytest.raises(EncoderError, match='stopped'):
            encoder.encode(handle, 3)


# ------------------------------------------------------- the coded buffer


def chain(*segments):
    """Coded buffer segments linked in order: each is (bytes, status)."""
    made = []
    for data, status in segments:
        held = ctypes.create_string_buffer(data, len(data)) if data is not None else None
        segment = api.VACodedBufferSegment()
        segment.size = len(data) if data is not None else 0
        segment.status = status
        segment.buf = ctypes.cast(held, ctypes.c_void_p) if held is not None else None
        made.append((segment, held))
    for (segment, _), (following, _) in zip(made, made[1:], strict=False):
        segment.next = ctypes.cast(ctypes.pointer(following), type(segment.next))
    return made


def collect(made, allocated=64):
    encoder = object.__new__(VAAPIEncoder)
    inside(encoder)['_coded_size'] = allocated
    return encoder._collect(ctypes.c_void_p(ctypes.addressof(made[0][0])))  # noqa: SLF001 - one step of the encode path, tested alone


class TestCollecting:
    def test_the_segments_are_joined_in_order(self):
        assert collect(chain((b'abc', 26), (b'', 0), (b'defg', 30))) == b'abcdefg'

    def test_a_segment_larger_than_the_buffer_is_refused(self):
        made = chain((b'abcd', 0))
        made[0][0].size = 1 << 30
        with pytest.raises(EncoderError, match='1073741824 bytes'):
            collect(made)

    def test_segments_that_sum_past_the_buffer_are_refused(self):
        with pytest.raises(EncoderError, match='not inside the 6 bytes'):
            collect(chain((b'abcd', 0), (b'efgh', 0)), allocated=6)

    def test_a_chain_that_does_not_end_is_refused(self):
        made = chain((b'', 0), (b'', 0))
        first, second = made[0][0], made[1][0]
        second.next = ctypes.cast(ctypes.pointer(first), type(second.next))
        with pytest.raises(EncoderError, match='segments'):
            collect(made)

    def test_a_segment_with_a_length_and_no_bytes_is_refused(self):
        made = chain((None, 0))
        made[0][0].size = 4
        with pytest.raises(EncoderError, match='no address'):
            collect(made)

    def test_a_buffer_mapped_at_no_address_is_refused(self):
        encoder = object.__new__(VAAPIEncoder)
        inside(encoder)['_coded_size'] = 64
        with pytest.raises(EncoderError, match='no address'):
            encoder._collect(ctypes.c_void_p())  # noqa: SLF001 - one step of the encode path, tested alone

    @pytest.mark.parametrize(('status', 'words'), [
        (encoder_module.api.VA_CODED_BUF_STATUS_SLICE_OVERFLOW_MASK | 26, 'overflowed'),
        (encoder_module.api.VA_CODED_BUF_STATUS_BAD_BITSTREAM, 'bad'),
    ])
    def test_a_picture_the_driver_marks_as_spoiled_is_not_a_packet(self, status, words):
        with pytest.raises(EncoderError, match=words):
            collect(chain((b'abcd', 0), (b'efgh', status)))

    def test_the_status_bits_have_the_values_libva_gives_them(self):
        assert encoder_module.api.VA_CODED_BUF_STATUS_SLICE_OVERFLOW_MASK == 0x200
        assert encoder_module.api.VA_CODED_BUF_STATUS_BAD_BITSTREAM == 0x8000


# ----------------------------------------------------------------- closing


class Exported:
    def __init__(self, failing=None):
        self.closed = 0
        self.failing = failing

    def close(self):
        self.closed += 1
        if self.failing:
            raise self.failing


class TestClosing:
    def test_everything_is_released_when_giving_an_input_back_raises(self, encoder):
        device = inside(encoder)['_device']
        first = encoder_module.InputHandle(texture=1, surface=5,
                                           exported=Exported(RuntimeError('EGL_NOT_INITIALIZED')))
        second = encoder_module.InputHandle(texture=2, surface=6, exported=Exported())
        inside(encoder)['_registered'] = [first, second]
        with pytest.raises(EncoderError, match='EGL_NOT_INITIALIZED'):
            encoder.close()
        names = [entry[0] for entry in encoder.log]
        assert names.count('vaDestroySurfaces') == 3       # two inputs, then the encoder's own
        assert names.count('vaDestroyBuffer') == 2
        assert names.count('vaDestroyContext') == 2 and names.count('vaDestroyConfig') == 2
        assert (first.surface, second.surface) == (0, 0)
        assert second.exported is None
        assert device.closed == 1
        assert inside(encoder)['_device'] is None

    @pytest.mark.parametrize('raising', ['vaDestroyBuffer', 'vaDestroyContext',
                                         'vaDestroyConfig', 'vaDestroySurfaces'])
    def test_the_device_is_closed_when_a_libva_release_raises(self, encoder, raising):
        device = inside(encoder)['_device']
        encoder.api.raising.add(raising)
        with pytest.raises(EncoderError, match=raising):
            encoder.close()
        assert device.closed == 1

    def test_a_second_close_does_nothing(self, encoder):
        device = inside(encoder)['_device']
        encoder.close()
        calls = len(encoder.log)
        encoder.close()
        assert len(encoder.log) == calls
        assert device.closed == 1

    def test_an_input_kept_by_another_context_is_given_back_by_the_next_close(self, encoder):
        """The libva side is released; the EGL image waits for its own context."""
        refusing = Exported(dmabuf.DMABufError('another OpenGL context is current'))
        handle = encoder_module.InputHandle(texture=1, surface=5, exported=refusing)
        inside(encoder)['_registered'] = [handle]
        device = inside(encoder)['_device']
        with pytest.raises(EncoderError, match='another OpenGL context'):
            encoder.close()
        assert device.closed == 1 and handle.surface == 0
        assert handle.exported is refusing
        refusing.failing = None
        encoder.close()
        assert handle.exported is None and refusing.closed == 2

    def test_an_encoder_that_never_opened_a_device_closes(self):
        with pytest.raises(EncoderError):
            VAAPIEncoder(64, 64, device='/dev/dri/there-is-no-such-node')

    def test_a_closed_encoder_refuses_to_encode_and_to_flush(self, encoder):
        handle, = inside(encoder)['_registered']
        encoder.close()
        with pytest.raises(EncoderError, match='closed'):
            encoder.encode(handle, 0)
        with pytest.raises(EncoderError, match='closed'):
            encoder.flush()


class TestAnInputsFence:
    def test_a_fence_is_not_deleted_from_another_context(self, monkeypatch):
        deleted = []
        monkeypatch.setattr(encoder_module, 'glDeleteSync', deleted.append)
        handle = encoder_module.InputHandle(texture=1, surface=5, fence='a fence')
        handle.owner = Owner(('egl', 1))
        with pytest.raises(EncoderError, match='OpenGL context'):
            handle.close()
        assert deleted == [] and handle.fence == 'a fence'
        handle.owner = Owner(None)
        handle.close()
        assert deleted == ['a fence'] and handle.fence is None


# ------------------------------------------------------ what a caller asks


class TestTranslatingErrors:
    @pytest.mark.parametrize('raised', [
        IndexError('list index out of range'), MemoryError(), AssertionError('no device'),
        ZeroDivisionError('division by zero'), ValueError('not a frame size'),
        OverflowError('too large'), TypeError('not a number'), OSError(5, 'gone'),
    ])
    def test_anything_a_driver_or_a_setting_causes_is_an_encoder_error(self, raised):
        @as_encoder_error
        def method():
            raise raised
        with pytest.raises(EncoderError) as caught:
            method()
        assert caught.value.__cause__ is raised
        assert type(raised).__name__ in str(caught.value)

    @pytest.mark.parametrize('kind', [KeyboardInterrupt, SystemExit, GeneratorExit])
    def test_an_interruption_passes_through(self, kind):
        @as_encoder_error
        def method():
            raise kind
        with pytest.raises(kind):
            method()


@pytest.fixture
def no_device(monkeypatch):
    """Refuse to open a device, so only what is checked before one is reached passes."""
    def refuse(*_arguments):
        raise AssertionError('a device was opened for settings that should be refused')
    monkeypatch.setattr(encoder_module.Device, 'first', refuse)


@pytest.mark.usefixtures('no_device')
class TestSettingsAreCheckedBeforeADeviceIsOpened:
    @pytest.mark.parametrize(('size', 'words'), [
        ((0, 0), 'outside 1'), ((-16, 720), 'outside 1'), ((1281, 721), 'even'),
        ((70000, 70000), 'outside 1'), ((1048592, 16), 'outside 1'),
        ((64.5, 64), 'whole number'), (('wide', 64), 'whole number'),
    ])
    def test_a_size(self, size, words):
        with pytest.raises(EncoderError, match=words):
            VAAPIEncoder(*size)

    @pytest.mark.parametrize(('setting', 'value'), [
        ('bitrate', -1), ('bitrate', 2**32 + 5), ('bitrate', 3_000_000_000),
        ('bitrate', 1.5), ('gop', -5), ('gop', 2**33), ('qp', -1), ('qp', 52), ('qp', 300),
        ('bframes', -1), ('bframes', 'two'),
    ])
    def test_a_number_outside_its_field(self, setting, value):
        with pytest.raises(EncoderError, match=setting):
            VAAPIEncoder(64, 64, **{setting: value})

    @pytest.mark.parametrize('fps', [(0, 1), (30, 0), (-30, 1), 0, -5, (100003, 1001),
                                     (30, 65537), (2**33, 1)])
    def test_a_frame_rate(self, fps):
        with pytest.raises(EncoderError, match='frame rate'):
            VAAPIEncoder(64, 64, fps=fps)


BASE = {'size': (1280, 720), 'frame_rate': (30, 1), 'bitrate': 4_000_000, 'gop': 30,
        'qp': 26, 'rate_control': api.VA_RC_CBR}
STATED = {api.VAConfigAttribEncPackedHeaders: 0x1f, api.VAConfigAttribEncMaxRefFrames: 1}

REFUSED = {
    'size (0, 0)': {'size': (0, 0)}, 'size (-16, 720)': {'size': (-16, 720)},
    'size odd': {'size': (1281, 721)}, 'size (70000, 70000)': {'size': (70000, 70000)},
    'size (1048592, 16)': {'size': (1048592, 16)},
    'bitrate 0': {'bitrate': 0}, 'bitrate -1': {'bitrate': -1},
    'bitrate 2**32+5': {'bitrate': 2**32 + 5}, 'bitrate 3e9': {'bitrate': 3_000_000_000},
    'gop 0': {'gop': 0}, 'gop -5': {'gop': -5}, 'gop 2**33': {'gop': 2**33},
    'qp -1': {'qp': -1}, 'qp 52': {'qp': 52}, 'qp 300': {'qp': 300},
    'fps (0, 1)': {'frame_rate': (0, 1)}, 'fps (30, 0)': {'frame_rate': (30, 0)},
    'fps (-30, 1)': {'frame_rate': (-30, 1)}, 'fps (100003, 1001)': {'frame_rate': (100003, 1001)},
}


class TestWhatAControlLayerIsGiven:
    """The settings are held to their fields wherever a layer is built from them."""

    @pytest.mark.parametrize('layer', [H264Encoding, HEVCEncoding, AV1Encoding])
    @pytest.mark.parametrize('name', sorted(REFUSED))
    def test_a_setting_that_does_not_fit_is_an_encoder_error(self, layer, name):
        with pytest.raises(EncoderError):
            layer(EncodeSettings(**{**BASE, **REFUSED[name]}), dict(STATED))

    @pytest.mark.parametrize(('rate', 'packed'), [
        ((30, 1), 30), ((30000, 1001), 30000 | 1001 << 16),
        ((90000, 3003), 30000 | 1001 << 16), ((30000, 70000), 3 | 7 << 16),
        ((120, 2), 60),
    ])
    def test_a_frame_rate_is_reduced_before_it_is_packed(self, rate, packed):
        assert packed_frame_rate(rate) == packed
        settings = EncodeSettings(**{**BASE, 'frame_rate': rate})
        buffers = rate_control_buffers(settings, 26, 51)
        found, = (payload.data.framerate for _, payload in buffers
                  if type(payload.data).__name__ == 'VAEncMiscParameterFrameRate')
        assert found == packed

    @pytest.mark.parametrize('rate', [(100003, 1001), (30, 65537), (65536, 1)])
    def test_a_frame_rate_that_does_not_fit_sixteen_bits_a_half_is_refused(self, rate):
        with pytest.raises(EncoderError, match='frame rate'):
            packed_frame_rate(rate)

    def test_the_largest_side_still_fits_its_field_once_rounded_to_a_block(self):
        assert LARGEST_SIDE % 64 == 0
        assert LARGEST_SIDE <= 0xFFFF

    @pytest.mark.parametrize('stated', [0xFFFF, 0xFFFF_FFFF, 17, 200])
    def test_h264_keeps_no_more_references_than_the_codec_has(self, stated):
        layer = H264Encoding(EncodeSettings(**BASE),
                             {**STATED, api.VAConfigAttribEncMaxRefFrames: stated})
        assert layer.references <= 16


class TestTheConversionRectangle:
    def test_the_rectangle_is_the_picture(self):
        region = colour_pipeline(5, (1280, 720)).output_region.contents
        assert (region.x, region.y, region.width, region.height) == (0, 0, 1280, 720)

    @pytest.mark.parametrize('size', [(70000, 70000), (0, 64), (64, -1), (65536, 64)])
    def test_a_size_its_fields_do_not_hold_is_refused(self, size):
        with pytest.raises(EncoderError):
            colour_pipeline(5, size)
