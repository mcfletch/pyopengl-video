"""The VP9 control layer's reference slots, driven with a stand-in session.

No driver runs here. The layer is given frames written by
:mod:`tests.vp9_syntax` and the libvpx fixtures, and the session records which
surface each frame was decoded into, what it was told of the eight slots, and
what was shown.
"""
import json
from pathlib import Path

import pytest

from pyopengl_video.decoder import DecoderError, Sample
from pyopengl_video.ivf import IVFReader
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.vp9_decode import VP9Decoding
from pyopengl_video.vaapi.vp9_parse import (
    DecodingState,
    parse_uncompressed_header,
    split_superframe,
)
from tests.test_vaapi_decode_session import RecordingSession
from tests.vp9_syntax import INTER, frame, show_existing, superframe

DATA = Path(__file__).parent / 'data' / 'decode'
FIXTURES = json.loads((DATA / 'vp9.json').read_text())['streams']

#: The contents of a ``vpcC`` box: version and flags, then profile 0, level
#: 1.0, 8 bits, 4:2:0, BT.709 and no initialisation data.
VPCC = bytes([1, 0, 0, 0, 0, 10, 0x82, 1, 1, 1, 0, 0])


class Session(RecordingSession):
    """A recording session that also keeps the buffers of each frame."""

    def __init__(self):
        super().__init__()
        self.buffers = []

    def submit(self, surface, batches, what):
        super().submit(surface, batches, what)
        self.buffers.append({kind: payload for batch in batches for kind, payload in batch})

    def slots(self, index=-1):
        """The surface the picture parameters of one frame name in each slot."""
        return list(self.buffers[index][api.VAPictureParameterBufferType].reference_frames)


@pytest.fixture
def decoding():
    session = Session()
    layer = VP9Decoding(session, output_slots=4)
    session.held = layer.held
    return layer, session


def sample(data, timestamp=0):
    return Sample(data, timestamp, 1, False, 0)


def test_the_configuration_states_nothing_and_any_is_accepted(decoding):
    layer, session = decoding
    layer.configure(b'')
    layer.configure(VPCC)
    assert layer.size == (0, 0)
    assert session.formats == []


class TestKeyFrames:
    def test_the_first_key_frame_states_the_format(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame(size=(122, 70))))
        format, = session.formats
        assert format.profile == api.VAProfileVP9Profile0
        assert format.coded_size == (128, 72)
        assert format.surfaces == 9             # eight slots and the frame being decoded
        assert format.depth == 8
        assert layer.size == (122, 70)

    def test_a_profile_2_key_frame_asks_for_ten_bit_surfaces(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame(profile=2, depth=10)))
        format, = session.formats
        assert (format.profile, format.depth) == (api.VAProfileVP9Profile2, 10)

    def test_the_colour_comes_from_the_key_frame(self, decoding):
        layer, _ = decoding
        layer.decode(sample(frame(color_space=2, color_range=1)))
        assert (layer.colour.matrix, layer.colour.full_range) == (1, True)

    def test_a_key_frame_is_put_in_every_slot(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame()))
        surface = session.submitted[0][0]
        assert layer.held() == {surface}
        layer.decode(sample(frame(frame_type=INTER, refresh=0)))
        assert session.slots() == [surface] * 8

    def test_a_frame_is_its_parameters_then_its_data(self, decoding):
        layer, session = decoding
        data = frame()
        layer.decode(sample(data))
        assert session.submitted[0][1] == [
            [api.VAPictureParameterBufferType],
            [api.VASliceParameterBufferType, api.VASliceDataBufferType]]
        assert session.buffers[0][api.VASliceDataBufferType] == data
        assert session.buffers[0][api.VASliceParameterBufferType].slice_data_size == len(data)

    def test_a_key_frame_of_the_same_format_keeps_the_surfaces(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame()))
        layer.decode(sample(frame()))
        assert len(session.formats) == 1
        assert len(session.submitted) == 2

    def test_a_key_frame_of_another_size_starts_a_new_format(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame(size=(64, 48))))
        layer.decode(sample(frame(size=(96, 80))))
        assert [format.coded_size for format in session.formats] == [(64, 48), (96, 80)]
        assert layer.size == (96, 80)
        assert layer.held() == {session.submitted[1][0]}

    def test_a_key_frame_of_another_depth_starts_a_new_format(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame()))
        layer.decode(sample(frame(profile=2, depth=10)))
        assert [format.depth for format in session.formats] == [8, 10]


class TestSlots:
    def test_an_inter_frame_replaces_only_the_slots_it_names(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame()))
        layer.decode(sample(frame(frame_type=INTER, refresh=0b00100100)))
        key, inter = (entry[0] for entry in session.submitted)
        assert key != inter
        assert layer.held() == {key, inter}
        layer.decode(sample(frame(frame_type=INTER, refresh=0)))
        assert session.slots() == [key, key, inter, key, key, inter, key, key]

    def test_a_frame_is_told_the_slots_as_they_were_before_it(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame()))
        layer.decode(sample(frame(frame_type=INTER, refresh=0b1)))
        assert session.slots() == [session.submitted[0][0]] * 8

    def test_a_surface_no_slot_holds_is_decoded_into_again(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame()))
        for _ in range(20):
            layer.decode(sample(frame(frame_type=INTER, refresh=0b1)))
        assert len(layer.held()) == 2
        assert {entry[0] for entry in session.submitted} <= set(range(9))

    def test_a_frame_that_replaces_no_slot_is_not_kept(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame()))
        layer.decode(sample(frame(frame_type=INTER, refresh=0)))
        assert layer.held() == {session.submitted[0][0]}
        assert [shown[0] for shown in session.shown] == [entry[0] for entry in session.submitted]

    def test_eight_slots_can_hold_eight_pictures(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame()))
        for slot in range(1, 8):
            layer.decode(sample(frame(frame_type=INTER, refresh=1 << slot)))
        assert len(layer.held()) == 8
        layer.decode(sample(frame(frame_type=INTER, refresh=0)))
        assert sorted(session.slots()) == sorted(layer.held())


class TestShowing:
    def test_a_shown_frame_carries_its_sample_s_time_and_its_own_size(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame(size=(122, 70)), timestamp=77))
        assert session.shown == [(session.submitted[0][0], 77, (0, 0, 122, 70))]

    def test_a_hidden_frame_is_decoded_and_not_shown(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame(), timestamp=0))
        hidden = frame(frame_type=INTER, show_frame=0, refresh=0b100)
        shown = frame(frame_type=INTER, refresh=0b1)
        layer.decode(sample(superframe(hidden, shown), timestamp=5))
        assert len(session.submitted) == 3
        assert [entry[1] for entry in session.shown] == [0, 5]
        assert session.shown[1][0] == session.submitted[2][0]
        assert session.submitted[1][0] in layer.held()

    def test_a_frame_shown_again_is_the_slot_s_surface_and_is_not_decoded(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame(size=(66, 38))))
        layer.decode(sample(frame(frame_type=INTER, show_frame=0, refresh=0b1000)))
        hidden = session.submitted[1][0]
        layer.decode(sample(show_existing(3), timestamp=9))
        assert len(session.submitted) == 2
        assert session.shown[-1] == (hidden, 9, (0, 0, 66, 38))
        assert hidden in layer.held()

    def test_showing_a_slot_again_twice_gives_the_surface_twice(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame()))
        layer.decode(sample(show_existing(0), timestamp=1))
        layer.decode(sample(show_existing(7), timestamp=2))
        assert [shown[0] for shown in session.shown] == [session.submitted[0][0]] * 3

    def test_nothing_is_held_back_for_a_flush(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame()))
        layer.flush()
        assert len(session.shown) == 1


class TestStartingPoints:
    def test_frames_before_the_first_key_frame_are_dropped(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame(frame_type=INTER)))
        layer.decode(sample(show_existing(2)))
        layer.decode(sample(frame(frame_type=INTER, show_frame=0, intra_only=1)))
        assert session.submitted == [] and session.shown == [] and session.formats == []
        layer.decode(sample(frame()))
        assert len(session.submitted) == 1

    def test_a_reset_drops_every_slot_and_waits_for_a_key_frame(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame()))
        layer.decode(sample(frame(frame_type=INTER, refresh=0b10)))
        layer.reset()
        assert layer.held() == set()
        layer.decode(sample(frame(frame_type=INTER)))
        assert len(session.submitted) == 2
        layer.decode(sample(frame()))
        layer.decode(sample(frame(frame_type=INTER, refresh=0)))
        assert len(session.submitted) == 4
        assert session.slots() == [session.submitted[2][0]] * 8


class TestRefusals:
    def test_twelve_bits(self, decoding):
        layer, session = decoding
        with pytest.raises(DecoderError, match='12 bits'):
            layer.decode(sample(frame(profile=2, depth=12)))
        assert session.submitted == []

    def test_444(self, decoding):
        layer, _ = decoding
        with pytest.raises(DecoderError, match='4:4:4'):
            layer.decode(sample(frame(profile=1, subsampling=(0, 0))))

    def test_an_inter_frame_of_another_size_is_reference_scaling(self, decoding):
        layer, session = decoding
        layer.decode(sample(frame(size=(64, 48))))
        with pytest.raises(DecoderError, match=r'48x32.*64x48.*reference scaling'):
            layer.decode(sample(frame(frame_type=INTER, found_ref=None, size=(48, 32))))
        assert len(session.submitted) == 1

    def test_an_intra_only_frame_of_another_size(self, decoding):
        layer, _ = decoding
        layer.decode(sample(frame(size=(64, 48))))
        with pytest.raises(DecoderError, match='without a key frame'):
            layer.decode(sample(frame(frame_type=INTER, show_frame=0, intra_only=1,
                                      size=(32, 32))))

    def test_after_a_refusal_the_layer_waits_for_a_key_frame(self, decoding):
        """The slots no longer match the stream, so nothing predicts from them."""
        layer, session = decoding
        layer.decode(sample(frame(size=(64, 48))))
        with pytest.raises(DecoderError):
            layer.decode(sample(frame(frame_type=INTER, found_ref=None, size=(48, 32))))
        assert layer.held() == set()
        layer.decode(sample(frame(frame_type=INTER)))
        assert len(session.submitted) == 1
        layer.decode(sample(frame(size=(64, 48))))
        assert len(session.submitted) == 2


class TestFixtureStreams:
    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_every_frame_is_decoded_once_and_every_picture_shown(self, name, decoding):
        layer, session = decoding
        state = DecodingState()
        coded = 0
        with IVFReader(DATA / f'{name}.ivf') as stream:
            times = []
            for index in range(len(stream)):
                unit = stream.sample(index)
                headers = [parse_uncompressed_header(data, state)
                           for data in split_superframe(unit.data)]
                coded += sum(not header.show_existing_frame for header in headers)
                times += [unit.timestamp] * sum(header.show_frame for header in headers)
                layer.decode(unit)
        layer.flush()
        assert len(session.submitted) == coded
        assert [timestamp for _, timestamp, _ in session.shown] == times
        assert len(session.shown) == len(FIXTURES[name]['md5'])
        assert len(layer.held()) <= 8

    def test_a_repeated_picture_is_shown_from_the_surface_it_was_decoded_into(self, decoding):
        layer, session = decoding
        with IVFReader(DATA / 'vp9_show_existing.ivf') as stream:
            for index in range(len(stream)):
                layer.decode(stream.sample(index))
        assert len(session.submitted) == 41
        assert len(session.shown) == 40
        assert {shown[0] for shown in session.shown} <= {
            entry[0] for entry in session.submitted}
