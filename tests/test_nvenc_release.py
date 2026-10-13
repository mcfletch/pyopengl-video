"""What an NVENC encoder releases, and when: each case in a process that must exit.

A session that is never ended keeps a worker thread of the driver alive, and
the driver's exit handler waits for it: the process prints its last line and
does not exit (N4 of ``plans/SECURITY-REVIEW.md``). An input handle is a
pointer the driver dereferences, so one from another encoder ends the process
with a segmentation fault (N5). Both are run in a child process
(``tests/nvidia_child.py``) with a time limit, and the test asserts on how the
child ended.
"""
import pytest

from tests import nvidia_child

pytestmark = pytest.mark.usefixtures('nvenc_available')

#: Seconds a child is given. An encoder opens, encodes and closes in under two.
LIMIT = 30.0


#: How many inputs an encoder with no B-pictures asks for: the one it reads and one more.
INPUTS = 2


def lifetime(*steps, bframes=0):
    """The report of a child that ran `steps` on an encoder and then exited."""
    return nvidia_child.run('encoder_lifetime', list(steps), bframes, timeout=LIMIT).clean()


class TestTheProcessExits:
    def test_with_an_encoder_and_its_inputs_never_closed(self):
        lifetime()

    def test_with_pictures_still_held_for_reordering(self):
        lifetime('encode', bframes=2)

    def test_after_a_close_that_no_context_was_current_for(self):
        lifetime('drop-context', 'close')

    def test_when_the_window_system_ended_first(self):
        lifetime('encode', 'terminate')

    def test_when_the_encoder_is_collected_with_no_context_current(self):
        lifetime('encode', 'drop-context', 'collect', bframes=2)

    def test_when_the_encoder_is_collected_on_another_thread(self):
        lifetime('encode', 'collect-on-thread', bframes=2)


class TestClosingInTheWrongContext:
    def test_the_session_is_ended_and_the_inputs_are_kept(self):
        [close] = lifetime('drop-context', 'close')['closes']
        assert close == {'raised': 'EncoderError', 'session': False, 'inputs': INPUTS}

    def test_closing_again_in_the_right_context_deletes_the_inputs(self):
        first, second = lifetime('drop-context', 'close', 'restore-context', 'close')['closes']
        assert first == {'raised': 'EncoderError', 'session': False, 'inputs': INPUTS}
        assert second == {'raised': '', 'session': False, 'inputs': 0}

    def test_closing_in_the_right_context_raises_nothing(self):
        first, second = lifetime('encode', 'close', 'close', bframes=2)['closes']
        assert first == second == {'raised': '', 'session': False, 'inputs': 0}


class TestClosingWithPicturesHeld:
    def test_the_session_is_told_the_stream_ended_before_its_buffers_go(self):
        """A session destroyed after the buffers of pictures it still held reads freed memory.

        The segmentation fault that follows came by the third encoder closed
        that way, in each of five runs.
        """
        rounds = 8
        report = nvidia_child.run('close_holding_pictures', rounds, timeout=LIMIT).clean()
        assert report == {'held': 2 * rounds}


class TestHandles:
    def test_only_an_open_input_of_this_encoder_is_encoded(self):
        """N5: any other handle is refused before the driver is handed it."""
        report = nvidia_child.run('foreign_handles', timeout=LIMIT).clean()
        assert report == {"another encoder's": 'EncoderError',
                          "a closed encoder's": 'EncoderError',
                          'its own': '', 'its own, closed': 'EncoderError'}
