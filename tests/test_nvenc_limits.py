"""What the NVENC encoder refuses before the driver sees it, and what it keeps when a call fails.

ctypes stores an integer that does not fit a field by dropping its high bits,
so a bit rate of 2**40 reaches the driver as 0 unless it is refused first, and
a function-table entry the driver left null is a call through address zero.
The argument and function-table cases need no GPU: they are refused before a
session is opened. The rest run on real hardware, with one driver call made
to fail.
"""
import ctypes

import pytest

from pyopengl_video import EncoderError
from pyopengl_video.nvenc import api
from pyopengl_video.nvenc.encoder import MOST_BFRAMES, NVENCEncoder

SIZE = (320, 240)
TICK = 1500

#: A status the driver gives for a picture it did not take.
NV_ENC_ERR_GENERIC = api.STATUS_NAMES.index('ERR_GENERIC')

REFUSED = [
    dict(width=0, height=0), dict(width=-16), dict(width=100000, height=100000),
    dict(height=1.5), dict(width='320'),
    dict(fps=(0, 1)), dict(fps=(1, 0)), dict(fps=0), dict(fps=-30),
    dict(fps=float('nan')), dict(fps=float('inf')), dict(fps=(2**31, 1)),
    dict(bitrate=0), dict(bitrate=-1), dict(bitrate=2**40),
    dict(gop=0), dict(gop=-5), dict(gop=2**40), dict(gop=0xFFFFFFFF),
    dict(bframes=-1), dict(bframes=-10), dict(bframes=MOST_BFRAMES + 1),
    dict(bframes=2**33), dict(bframes=True),
]


class TestArguments:
    """N12: every one is refused by name, and no session is opened to find out."""

    @pytest.fixture(autouse=True)
    def no_session(self, monkeypatch):
        def opened(_self):
            raise AssertionError('a session was opened for arguments that are refused')
        monkeypatch.setattr(NVENCEncoder, '_open_session', opened)
        monkeypatch.setattr(api.NvEncodeAPI, 'load',
                            classmethod(lambda _class: opened(None)))

    @pytest.mark.parametrize('arguments', REFUSED, ids=repr)
    def test_an_argument_no_field_holds_is_refused(self, arguments):
        named = {'width': SIZE[0], 'height': SIZE[1], **arguments}
        with pytest.raises(EncoderError):
            NVENCEncoder(**named)

    def test_the_refusal_names_the_argument_and_its_range(self):
        with pytest.raises(EncoderError, match=f'bframes is 64, outside 0 to {MOST_BFRAMES}'):
            NVENCEncoder(*SIZE, bframes=64)
        with pytest.raises(EncoderError, match='bitrate is 1099511627776, outside 1 to'):
            NVENCEncoder(*SIZE, bitrate=2**40)
        with pytest.raises(EncoderError, match='a frame rate of 1/0 cannot be coded'):
            NVENCEncoder(*SIZE, fps=(1, 0))

    def test_an_encoder_that_was_refused_closes_and_collects_quietly(self, recwarn):
        encoder = NVENCEncoder.__new__(NVENCEncoder)
        with pytest.raises(EncoderError):
            encoder.__init__(*SIZE, gop=0)
        encoder.close()
        encoder.__del__()
        assert not [warning for warning in recwarn if warning.category is ResourceWarning]


class TestTheFunctionTable:
    """N9: an entry the driver left null raises, and is never called through."""

    def test_every_call_of_an_empty_table_raises_naming_its_entry(self):
        bound = api.bind(api.NV_ENCODE_API_FUNCTION_LIST(), (9, 0))
        assert set(bound) == set(api.PROTOTYPES)
        for name, function in bound.items():
            with pytest.raises(api.NVENCError, match=name) as raised:
                function(None, None)
            assert raised.value.name == 'ERR_UNIMPLEMENTED'
            assert 'interface version 9.0' in str(raised.value)

    def test_an_entry_the_driver_filled_in_is_called_through(self):
        seen = []

        @api.PROTOTYPES['nvEncDestroyEncoder']
        def destroy(session):
            seen.append(session)
            return 7

        table = api.NV_ENCODE_API_FUNCTION_LIST()
        table.nvEncDestroyEncoder = ctypes.cast(destroy, ctypes.c_void_p)
        bound = api.bind(table)
        assert bound['nvEncDestroyEncoder'](ctypes.c_void_p(1234)) == 7
        assert seen == [1234]
        with pytest.raises(api.NVENCError, match='nvEncEncodePicture'):
            bound['nvEncEncodePicture'](None, None)

    def test_every_call_this_package_makes_has_its_types(self):
        for name, prototype in api.PROTOTYPES.items():
            assert name in api.FUNCTION_NAMES
            assert prototype._restype_ is not None, name
            assert prototype._argtypes_, name

    @pytest.mark.usefixtures('nvenc_available')
    def test_every_function_of_the_loaded_binding_is_declared(self):
        """Standard 18: ctypes assumes ``int`` arguments for a function nobody declared."""
        library = api.NvEncodeAPI.load()
        exports = {name: value for name, value in vars(library.library).items()
                   if isinstance(value, ctypes._CFuncPtr)}  # noqa: SLF001 - the base class of every foreign function, which ctypes does not export
        assert set(exports) == {'NvEncodeAPIGetMaxSupportedVersion',
                                'NvEncodeAPICreateInstance'}
        for name, function in exports.items():
            assert function.argtypes is not None, name
            assert function.restype is ctypes.c_int, name
        for name in api.PROTOTYPES:
            assert callable(getattr(library, name)), name


@pytest.fixture
def reordering(nvenc_available):  # noqa: ARG001 - requested to skip where there is no NVENC
    """An encoder that holds two pictures back, with an input more than it needs."""
    encoder = NVENCEncoder(*SIZE, fps=60, bframes=2)
    inputs = [encoder.new_input() for _ in range(encoder.input_slots + 1)]
    yield encoder, inputs
    encoder.close()


def failing_once(monkeypatch, library, name, status=NV_ENC_ERR_GENERIC):
    """Make the driver call `name` answer `status` the next time, and work after that."""
    real = getattr(library, name)
    calls = []

    def call(*arguments):
        calls.append(arguments)
        return status if len(calls) == 1 else real(*arguments)

    monkeypatch.setattr(library, name, call)
    return calls


class TestAFailedSubmission:
    """N8: a picture the encoder did not take is not one a later call reads out."""

    def test_the_pictures_after_it_come_out_as_themselves(self, reordering, monkeypatch):
        encoder, inputs = reordering
        spare, cycle = inputs[-1], inputs[:-1]
        packets = []
        for index in range(5):
            packets += encoder.encode(cycle[index % len(cycle)], index * TICK)
        failing_once(monkeypatch, encoder.api, 'nvEncEncodePicture')
        with pytest.raises(api.NVENCError, match='nvEncEncodePicture'):
            encoder.encode(spare, 99999)
        for index in range(5, 11):
            packets += encoder.encode(cycle[index % len(cycle)], index * TICK)
        packets += encoder.flush()
        assert sorted(packet.timestamp for packet in packets) == [
            index * TICK for index in range(11)]
        assert all(len(packet) > 0 for packet in packets)

    def test_its_input_can_be_submitted_again(self, reordering, monkeypatch):
        encoder, inputs = reordering
        failing_once(monkeypatch, encoder.api, 'nvEncEncodePicture')
        with pytest.raises(api.NVENCError):
            encoder.encode(inputs[0], 0)
        packets = encoder.encode(inputs[0], 0) + encoder.flush()
        assert [packet.timestamp for packet in packets] == [0]


class TestAFailedRead:
    """N11: a picture whose read failed is still held, and is read by the next drain."""

    def test_the_picture_comes_out_of_the_flush(self, nvenc_available, monkeypatch):  # noqa: ARG002 - requested to skip where there is no NVENC
        encoder = NVENCEncoder(*SIZE, fps=60)
        try:
            first, second = encoder.new_input(), encoder.new_input()
            assert [packet.timestamp for packet in encoder.encode(first, 0)] == [0]
            failing_once(monkeypatch, encoder.api, 'nvEncLockBitstream')
            with pytest.raises(api.NVENCError, match='nvEncLockBitstream'):
                encoder.encode(second, TICK)
            with pytest.raises(EncoderError, match='still being encoded'):
                encoder.encode(second, 2 * TICK)
            assert [packet.timestamp for packet in encoder.flush()] == [TICK]
        finally:
            encoder.close()


class TestOutputBuffers:
    """N11: every buffer made for compressed pictures is destroyed at close."""

    @pytest.fixture
    def counted(self, nvenc_available, monkeypatch):  # noqa: ARG002 - requested to skip where there is no NVENC
        library = api.NvEncodeAPI.load()
        counts = {'created': 0, 'destroyed': 0, 'unmapped': 0, 'mapped': 0}

        def counting(name, key):
            real = getattr(library, name)

            def call(*arguments):
                status = real(*arguments)
                counts[key] += status == api.NV_ENC_SUCCESS
                return status
            monkeypatch.setattr(library, name, call)

        counting('nvEncCreateBitstreamBuffer', 'created')
        counting('nvEncDestroyBitstreamBuffer', 'destroyed')
        counting('nvEncMapInputResource', 'mapped')
        counting('nvEncUnmapInputResource', 'unmapped')
        return counts

    def test_the_buffers_of_pictures_still_held_are_destroyed(self, counted):
        encoder = NVENCEncoder(*SIZE, fps=60, bframes=2)
        inputs = [encoder.new_input() for _ in range(encoder.input_slots)]
        # A key frame and two B-pictures: the reference that lets them out is not sent.
        for index, handle in enumerate(inputs[:3]):
            encoder.encode(handle, index * TICK)
        held = counted['mapped'] - counted['unmapped']
        assert held > 0, 'the encoder holds no picture back, so close has none to release'
        encoder.close()
        assert counted['mapped'] == counted['unmapped'] == 3
        assert counted['created'] == counted['destroyed'] > 0

    def test_a_buffer_that_could_not_be_had_leaves_no_input_mapped(self, counted, monkeypatch):
        encoder = NVENCEncoder(*SIZE, fps=60)
        try:
            handle = encoder.new_input()

            def no_buffer():
                raise EncoderError('no buffer')
            monkeypatch.setattr(encoder, '_acquire_output', no_buffer)
            with pytest.raises(EncoderError, match='no buffer'):
                encoder.encode(handle, 0)
            assert counted['mapped'] == counted['unmapped'] == 1
        finally:
            encoder.close()


class TestInputs:
    @pytest.fixture
    def encoder(self, nvenc_available):  # noqa: ARG002 - requested to skip where there is no NVENC
        made = NVENCEncoder(*SIZE, fps=60)
        yield made
        made.close()

    def test_closing_the_encoder_closes_its_handles_to_it(self, encoder):
        handle = encoder.new_input()
        encoder.close()
        assert handle.registration is None
        assert handle.released

    @pytest.mark.parametrize('texture', [0, -1, 2**32, 1.5, None])
    def test_a_texture_name_no_field_holds_is_refused(self, encoder, texture):
        with pytest.raises(EncoderError, match='the texture name'):
            encoder.register(texture)

    @pytest.mark.parametrize('timestamp', [-1, 2**64, 1.5])
    def test_a_timestamp_no_field_holds_is_refused(self, encoder, timestamp):
        handle = encoder.new_input()
        with pytest.raises(EncoderError, match='the timestamp'):
            encoder.encode(handle, timestamp)

    def test_registering_with_a_closed_encoder_is_refused(self, encoder):
        encoder.close()
        with pytest.raises(EncoderError, match='closed'):
            encoder.register(1)
