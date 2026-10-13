"""Which picture each VP9 reference slot holds, followed with no driver.

NVIDIA's parser shows no picture for a frame that only says to show a
reference slot again, so the decoder shows it, and has to know which decoded
picture the slot holds.
"""
from pathlib import Path

import pytest

from pyopengl_video.ivf import IVFReader
from pyopengl_video.nvdec.vp9 import Frame, Slots
from pyopengl_video.vaapi.h264 import StreamError
from tests.vp9_syntax import INTER, frame, show_existing, superframe

DATA = Path(__file__).parent / 'data' / 'decode'


def test_a_key_frame_is_stored_in_every_slot():
    slots = Slots()
    [key] = slots.frames(frame())
    assert (key.shows_slot, key.refreshes) == (None, 0xFF)
    slots.decoded(key, 5)
    assert [slots.picture(slot) for slot in range(8)] == [5] * 8


def test_an_inter_frame_replaces_the_slots_its_header_names():
    slots = Slots()
    [key] = slots.frames(frame())
    slots.decoded(key, 0)
    [inter] = slots.frames(frame(frame_type=INTER, refresh=0b00010100))
    slots.decoded(inter, 3)
    assert [slots.picture(slot) for slot in range(8)] == [0, 0, 3, 0, 3, 0, 0, 0]


def test_a_frame_stored_nowhere_changes_no_slot():
    slots = Slots()
    [key] = slots.frames(frame())
    slots.decoded(key, 0)
    [inter] = slots.frames(frame(frame_type=INTER, refresh=0))
    slots.decoded(inter, 1)
    assert {slots.picture(slot) for slot in range(8)} == {0}


def test_a_frame_shown_again_names_its_slot_and_is_not_coded():
    slots = Slots()
    [key] = slots.frames(frame())
    slots.decoded(key, 0)
    [again] = slots.frames(show_existing(6))
    assert again == Frame(show_existing(6), shows_slot=6)
    assert again.refreshes == 0


def test_a_superframe_is_its_frames_in_order():
    hidden = frame(frame_type=INTER, show_frame=0, refresh=0b100)
    shown = frame(frame_type=INTER, refresh=0b1)
    slots = Slots()
    [key] = slots.frames(frame())
    slots.decoded(key, 0)
    found = slots.frames(superframe(hidden, shown, show_existing(2)))
    assert [(entry.data, entry.shows_slot, entry.refreshes) for entry in found] == [
        (hidden, None, 0b100), (shown, None, 0b1), (show_existing(2), 2, 0)]


def test_a_slot_no_frame_was_stored_in_holds_no_picture():
    assert Slots().picture(0) is None


def test_an_inter_frame_with_no_key_frame_before_it_is_not_read():
    """As after a seek to a sample that is no key frame."""
    with pytest.raises(StreamError, match='holds no frame'):
        Slots().frames(frame(frame_type=INTER))


def test_forgetting_empties_the_slots_and_what_was_read():
    slots = Slots()
    [key] = slots.frames(frame())
    slots.decoded(key, 0)
    slots.forget()
    assert slots.picture(0) is None
    with pytest.raises(StreamError, match='holds no frame'):
        slots.frames(frame(frame_type=INTER))


def test_losing_the_pictures_keeps_the_frames_that_were_read():
    """A new hardware decoder: the next frame is still read against the last."""
    slots = Slots()
    [key] = slots.frames(frame())
    slots.decoded(key, 0)
    slots.lose_pictures()
    assert slots.picture(0) is None
    [inter] = slots.frames(frame(frame_type=INTER, refresh=0b10))
    slots.decoded(inter, 4)
    assert slots.picture(1) == 4


def test_every_frame_shown_again_in_a_real_stream_has_a_picture():
    """libvpx's layered alternate references, pictures numbered as they are decoded."""
    slots = Slots()
    decoded = shown_again = 0
    with IVFReader(DATA / 'vp9_large_show_existing.ivf') as stream:
        for index in range(len(stream)):
            for entry in slots.frames(stream.sample(index).data):
                if entry.shows_slot is None:
                    slots.decoded(entry, decoded)
                    decoded += 1
                else:
                    shown_again += 1
                    assert slots.picture(entry.shows_slot) is not None
    assert shown_again >= 8
    assert decoded > len(stream) - shown_again        # the hidden frames as well
