"""What the oneVPL encoder does with its runtime's answers, on any platform.

The backend runs on Windows with Intel hardware. Its decisions run here against
stand-ins for the runtime, Direct3D and the WGL interop
(:mod:`tests.vpl_runtime`): which surface a picture is submitted in, what a
status means, what a packet may be read from, and what is released when a
release fails.
"""
import ctypes
import gc
import warnings

import pytest

from pyopengl_video.encoder import EncoderError
from pyopengl_video.vpl import api
from pyopengl_video.vpl import encoder as vpl_encoder
from pyopengl_video.vpl.allocator import SurfaceAllocator
from pyopengl_video.vpl.encoder import VPLEncoder
from pyopengl_video.windows import d3d11, interop
from tests import vpl_runtime


@pytest.fixture
def machine(monkeypatch):
    return vpl_runtime.Machine().install(monkeypatch)


@pytest.fixture
def encoder(machine):  # noqa: ARG001 - requested so the stand-ins are in place
    built = VPLEncoder(320, 240, fps=30, bitrate=2_000_000)
    yield built
    built.close()


class TestSubmitting:
    def test_a_picture_comes_back_as_the_packet_the_runtime_wrote(self, encoder, machine):
        handle = encoder.new_input()
        [packet] = encoder.encode(handle, 3000)
        assert packet.data == machine.runtime.payload
        assert packet.timestamp == 3000 and packet.keyframe

    def test_each_input_is_submitted_in_a_surface_of_its_own(self, encoder, machine):
        """The runtime keeps the address of a surface it has not finished with."""
        machine.runtime.statuses = [api.MFX_ERR_MORE_DATA, api.MFX_ERR_MORE_DATA]
        first, second = encoder.new_input(), encoder.new_input()
        encoder.encode(first, 0)
        encoder.encode(second, 3000)
        (first_at, first_id, _), (second_at, second_id, _) = machine.runtime.surfaces
        assert first_at != second_at
        assert first_id != second_id

    def test_a_surface_the_runtime_still_holds_is_not_submitted_again(self, encoder):
        handle = encoder.new_input()
        encoder.encode(handle, 0)
        handle.surface.Data.Locked = 1      # the runtime sets this on a surface it keeps
        with pytest.raises(EncoderError, match='still being encoded'):
            encoder.encode(handle, 3000)

    def test_an_input_held_for_drawing_is_not_submitted(self, encoder):
        handle = encoder.new_input()
        with handle.for_drawing():
            with pytest.raises(EncoderError, match='for_drawing'):
                encoder.encode(handle, 0)

    def test_an_input_of_another_encoder_is_refused(self, encoder, machine):  # noqa: ARG002 - the stand-ins serve both encoders
        with VPLEncoder(320, 240) as other:
            foreign = other.new_input()
            with pytest.raises(EncoderError, match='new_input'):
                encoder.encode(foreign, 0)

    def test_a_synchronisation_that_times_out_is_a_failure(self, encoder, machine):
        """A positive status from SyncOperation means the picture is not written yet."""
        machine.runtime.sync_status = api.MFX_WRN_IN_EXECUTION
        handle = encoder.new_input()
        with pytest.raises(api.VPLError, match='MFX_WRN_IN_EXECUTION'):
            encoder.encode(handle, 0)

    def test_nothing_more_is_submitted_after_a_time_out(self, encoder, machine):
        """The runtime may still be writing the one bitstream buffer."""
        machine.runtime.sync_status = api.MFX_WRN_IN_EXECUTION
        handle = encoder.new_input()
        with pytest.raises(api.VPLError):
            encoder.encode(handle, 0)
        machine.runtime.sync_status = api.MFX_ERR_NONE
        submitted = len(machine.named('EncodeFrameAsync'))
        with pytest.raises(EncoderError, match='stopped'):
            encoder.encode(encoder.new_input(), 3000)
        with pytest.raises(EncoderError, match='stopped'):
            encoder.flush()
        assert len(machine.named('EncodeFrameAsync')) == submitted

    def test_a_device_that_stays_busy_is_given_up_on(self, encoder, machine, monkeypatch):
        clock = iter(range(0, 10 ** 9, 1000))
        monkeypatch.setattr(vpl_encoder.time, 'monotonic', lambda: next(clock) / 1000)
        monkeypatch.setattr(vpl_encoder.time, 'sleep', lambda _seconds: None)
        machine.runtime.statuses = [api.MFX_WRN_DEVICE_BUSY] * 1000
        with pytest.raises(api.VPLError, match='MFX_WRN_DEVICE_BUSY'):
            encoder.encode(encoder.new_input(), 0)
        assert len(machine.named('EncodeFrameAsync')) < 20

    def test_a_device_busy_for_a_moment_is_offered_the_picture_again(self, encoder, machine):
        machine.runtime.statuses = [api.MFX_WRN_DEVICE_BUSY, api.MFX_WRN_DEVICE_BUSY]
        [packet] = encoder.encode(encoder.new_input(), 0)
        assert packet.data == machine.runtime.payload

    @pytest.mark.parametrize('offset, length', [(0, 10 ** 6), (10 ** 9, 4), (60, 10 ** 6)])
    def test_a_packet_is_read_from_inside_the_buffer_only(self, encoder, machine,
                                                          offset, length):
        machine.runtime.stated_length = length
        handle = encoder.new_input()
        encoder._bitstream.DataOffset = offset  # noqa: SLF001 - cleared again by the submission
        original = machine.runtime.MFXVideoENCODE_EncodeFrameAsync

        def misreport(session, control, surface, bitstream, sync_point):
            status = original(session, control, surface, bitstream, sync_point)
            bitstream._obj.DataOffset = offset  # noqa: SLF001 - what a byref argument points at
            return status

        machine.runtime.MFXVideoENCODE_EncodeFrameAsync = misreport
        with pytest.raises(EncoderError, match='not inside'):
            encoder.encode(handle, 0)

    def test_flushing_ends_when_the_runtime_has_no_more(self, encoder, machine):
        machine.runtime.statuses = [api.MFX_ERR_MORE_DATA]
        assert encoder.flush() == []


class TestWhatAnEncoderIsBuiltWith:
    @pytest.mark.parametrize('size', [(0, 240), (320, 0), (-320, 240), (65536 + 16, 240),
                                      (320, 2 ** 32 + 240)])
    def test_a_size_the_fields_cannot_hold_is_refused(self, machine, size):  # noqa: ARG002 - requested so the stand-ins are in place
        with pytest.raises(EncoderError, match='picture'):
            VPLEncoder(*size)

    @pytest.mark.parametrize('fps', [0, -30, (30, 0), (0, 1)])
    def test_a_rate_of_no_frames_is_refused(self, machine, fps):  # noqa: ARG002 - requested so the stand-ins are in place
        with pytest.raises(EncoderError, match='frame rate'):
            VPLEncoder(320, 240, fps=fps)

    @pytest.mark.parametrize('options, named', [
        (dict(gop=0), 'gop'), (dict(gop=70000), 'gop'), (dict(bframes=-1), 'bframes'),
        (dict(bframes=70000), 'bframes'), (dict(bitrate=0), 'bitrate'),
        (dict(bitrate=-5), 'bitrate'), (dict(rate_control='constqp', qp=52), 'qp'),
        (dict(rate_control='constqp', qp=-1), 'qp'),
    ])
    def test_a_setting_outside_its_field_is_refused_by_name(self, machine, options, named):  # noqa: ARG002 - requested so the stand-ins are in place
        with pytest.raises(EncoderError, match=named):
            VPLEncoder(320, 240, **options)

    def test_nothing_is_opened_for_settings_that_are_refused(self, machine):
        with pytest.raises(EncoderError):
            VPLEncoder(0, 0)
        assert machine.calls == [] and machine.devices == []

    def test_the_fields_hold_what_was_asked_for(self, machine):
        with VPLEncoder(1920, 1080, fps=(30000, 1001), gop=250, bframes=2,
                        bitrate=200_000_000):
            configured = machine.runtime.configured
        assert configured['width'] == 1920 and configured['height'] == 1088
        assert configured['crop'] == (1920, 1080)
        assert configured['rate'] == (30000, 1001)
        assert configured['gop'] == 250 and configured['distance'] == 3
        assert configured['target'] * configured['multiplier'] == pytest.approx(
            200_000, rel=0.001)

    def test_a_constant_quantiser_is_written_as_quantisers(self, machine):
        """The three fields are the bit rates' own bytes under another name."""
        with VPLEncoder(320, 240, rate_control='constqp', qp=30):
            configured = machine.runtime.configured
        assert configured['method'] == api.MFX_RATECONTROL_CQP
        assert (configured['initial'], configured['target'], configured['most']) == (
            30, 30, 30)
        assert configured['multiplier'] == 0

    def test_an_adapter_that_is_not_intels_is_refused(self, machine):
        machine.context_adapter = vpl_runtime.NVIDIA
        with pytest.raises(EncoderError, match='Intel'):
            VPLEncoder(320, 240)
        assert machine.named('MFXInit') == []

    def test_a_process_of_another_pointer_size_is_refused(self, machine, monkeypatch):  # noqa: ARG002 - requested so the stand-ins are in place
        monkeypatch.setattr(api, 'SUPPORTED_POINTER_SIZE', False)
        with pytest.raises(EncoderError, match='64-bit'):
            VPLEncoder(320, 240)

    def test_the_time_a_flush_gives_a_packet_is_one_frame(self, machine):
        with VPLEncoder(320, 240, fps=30) as built:
            machine.runtime.statuses = [api.MFX_ERR_MORE_DATA]
            assert built.flush() == []
            assert built.frame_duration() == 3000


class TestClosing:
    def test_everything_is_released_in_order(self, machine):
        built = VPLEncoder(320, 240)
        built.new_input()
        del machine.calls[:]
        built.close()
        names = [call[0] for call in machine.calls]
        assert names == ['MFXVideoENCODE_Close', 'MFXClose', 'glDeleteFramebuffers',
                         'wglDXUnregisterObjectNV', 'glDeleteTextures',
                         'Texture.Release', 'wglDXCloseDeviceNV', 'Device.Release']

    @pytest.mark.parametrize('failing', ['glDeleteFramebuffers', 'wglDXUnregisterObjectNV',
                                         'glDeleteTextures', 'wglDXCloseDeviceNV'])
    def test_a_release_that_fails_does_not_strand_the_session(self, machine, failing):
        built = VPLEncoder(320, 240)
        built.new_input()
        machine.fail.add(failing)
        with pytest.raises(RuntimeError, match=failing):
            built.close()
        names = [call[0] for call in machine.calls]
        assert 'MFXClose' in names and 'Device.Release' in names
        assert 'Texture.Release' in names
        assert not built.session

    def test_closing_twice_releases_once(self, machine):
        built = VPLEncoder(320, 240)
        built.close()
        released = list(machine.calls)
        built.close()
        assert machine.calls == released

    def test_an_encoder_that_did_not_open_releases_what_it_had(self, machine):
        machine.runtime.MFXVideoENCODE_Init = lambda *_arguments: -3
        with pytest.raises(api.VPLError, match='MFX_ERR_UNSUPPORTED'):
            VPLEncoder(320, 240)
        names = [call[0] for call in machine.calls]
        assert names[-3:] == ['MFXClose', 'wglDXCloseDeviceNV', 'Device.Release']
        assert 'MFXVideoENCODE_Close' not in names

    def test_a_collected_encoder_gives_back_the_session_and_no_opengl(self, machine):
        """Collection runs on whichever thread and context the collector is in."""
        built = VPLEncoder(320, 240)
        built.new_input()
        del machine.calls[:]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            del built
            gc.collect()
        names = [call[0] for call in machine.calls]
        assert 'MFXVideoENCODE_Close' in names and 'MFXClose' in names
        assert not [name for name in names if name.startswith(('gl', 'wgl'))]
        assert [warning for warning in caught
                if issubclass(warning.category, ResourceWarning)]

    def test_an_encoder_that_was_closed_warns_of_nothing(self, machine):  # noqa: ARG002 - requested so the stand-ins are in place
        built = VPLEncoder(320, 240)
        built.close()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            del built
            gc.collect()
        assert not caught


class TestTheSharedTexture:
    @pytest.fixture
    def shared(self, machine):
        device = machine._device()  # noqa: SLF001 - the stand-in's own factory
        bridge = interop.InteropDevice(device)
        yield bridge.create_texture(64, 64)
        bridge.close()

    def test_it_is_not_closed_while_held_for_drawing(self, shared, machine):
        with shared.for_drawing():
            with pytest.raises(interop.InteropError, match='for_drawing'):
                shared.close()
        assert machine.named('wglDXUnregisterObjectNV') == []
        assert [call[0] for call in machine.calls
                if call[0].startswith('wglDX')][-1] == 'wglDXUnlockObjectsNV'

    def test_a_closed_texture_is_not_locked(self, shared, machine):
        shared.close()
        with pytest.raises(interop.InteropError, match='closed'):
            with shared.for_drawing():
                pass
        assert machine.named('wglDXLockObjectsNV') == []

    def test_closing_twice_releases_once(self, shared, machine):
        shared.close()
        released = list(machine.calls)
        shared.close()
        assert machine.calls == released

    def test_a_texture_that_cannot_be_registered_leaves_nothing_behind(self, machine):
        device = machine._device()  # noqa: SLF001 - the stand-in's own factory
        bridge = interop.InteropDevice(device)
        machine.fail.add('wglDXRegisterObjectNV')
        with pytest.raises(interop.InteropError):
            bridge.create_texture(64, 64)
        names = [call[0] for call in machine.calls]
        assert names == ['glDeleteTextures', 'Texture.Release']

    def test_a_texture_with_no_framebuffer_leaves_nothing_behind(self, machine):
        device = machine._device()  # noqa: SLF001 - the stand-in's own factory
        bridge = interop.InteropDevice(device)
        machine.fail.add('glGenFramebuffers')
        with pytest.raises(RuntimeError, match='attachment'):
            bridge.create_texture(64, 64)
        names = [call[0] for call in machine.calls]
        assert names == ['wglDXRegisterObjectNV', 'wglDXUnregisterObjectNV',
                         'glDeleteTextures', 'Texture.Release']


class TestTheAllocatorsCallbacks:
    """The runtime calls these from its own code, where nothing can be raised."""

    @pytest.fixture
    def allocator(self, machine):
        return SurfaceAllocator(machine._device())  # noqa: SLF001 - the stand-in's own factory

    @staticmethod
    def request(count=3, fourcc=api.MFX_FOURCC_NV12):
        wanted = api.mfxFrameAllocRequest()
        wanted.Info.FourCC = fourcc
        wanted.Info.Width, wanted.Info.Height = 320, 240
        wanted.NumFrameMin = wanted.NumFrameSuggested = count
        return ctypes.pointer(wanted)

    def test_surfaces_that_could_not_all_be_made_are_all_forgotten(self, allocator, machine):
        machine.fail_textures_after = 2
        response = api.mfxFrameAllocResponse()
        status = allocator.allocate(None, self.request(), ctypes.pointer(response))
        assert status == api.MFX_ERR_MEMORY_ALLOC
        assert allocator.owned == {}
        assert len(machine.named('Texture.Release')) == 2

    def test_a_texture_that_has_been_closed_is_not_handed_back(self, allocator, machine):
        texture = machine._device().create_texture(64, 64)  # noqa: SLF001 - the stand-in's own factory
        identifier = allocator.adopt(texture)
        texture.close()
        pair = (ctypes.c_void_p * 2)(1, 1)
        status = allocator.get_handle(None, identifier, ctypes.pointer(pair))
        assert status == api.MFX_ERR_NOT_FOUND
        assert list(pair) == [1, 1]

    @pytest.mark.parametrize('callback, arguments', [
        ('allocate', (None, None, None)),
        ('get_handle', (None, 0x10000, None)),
        ('free', (None, None)),
    ])
    def test_a_callback_that_raises_answers_with_a_failure(self, allocator, machine,
                                                           callback, arguments):
        """ctypes gives the runtime whatever is in the result slot otherwise."""
        allocator.owned[0x10000] = machine._device().create_texture(64, 64)  # noqa: SLF001 - the stand-in's own factory
        status = getattr(allocator, callback)(*arguments)
        assert status == api.MFX_ERR_UNKNOWN

    def test_an_interrupt_inside_a_callback_answers_with_a_failure(self, allocator,
                                                                   monkeypatch):
        def interrupted(*_arguments, **_options):
            raise KeyboardInterrupt

        monkeypatch.setattr(allocator.device, 'create_texture', interrupted)
        response = api.mfxFrameAllocResponse()
        status = allocator.allocate(None, self.request(), ctypes.pointer(response))
        assert status == api.MFX_ERR_UNKNOWN


class TestADirect3DTexture:
    @pytest.mark.parametrize('size', [(0, 64), (64, 0), (-1, 64), (2 ** 32 + 16, 64),
                                      (64, 16385)])
    def test_a_size_the_description_cannot_hold_is_refused(self, size):
        """The description's fields are 32 bits wide and drop what does not fit."""
        device = d3d11.Device.__new__(d3d11.Device)
        with pytest.raises(interop.InteropError, match='picture'):
            device.create_texture(*size)
