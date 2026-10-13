"""Releasing several resources when one of the releases fails.

A `close()` that stops at its first exception leaves everything after it held,
and the object already says it is closed. No hardware.
"""
import pytest

from pyopengl_video.teardown import release_all


def test_every_step_runs_in_order():
    ran = []
    release_all(lambda: ran.append(1), lambda: ran.append(2), lambda: ran.append(3))
    assert ran == [1, 2, 3]


def test_a_step_that_raises_does_not_stop_the_ones_after_it():
    ran = []

    def fails():
        ran.append('fails')
        raise RuntimeError('no OpenGL context is current')

    with pytest.raises(RuntimeError, match='no OpenGL context'):
        release_all(lambda: ran.append('first'), fails, lambda: ran.append('last'))
    assert ran == ['first', 'fails', 'last']


def test_the_first_failure_is_the_one_raised_and_the_rest_are_logged(caplog):
    def first():
        raise RuntimeError('first')

    def second():
        raise ValueError('second')

    with caplog.at_level('ERROR', logger='pyopengl_video.teardown'):
        with pytest.raises(RuntimeError, match='first'):
            release_all(first, second)
    assert [record.exc_info[1].args for record in caplog.records] == [('second',)]


def test_an_interrupt_still_lets_the_rest_run_and_is_raised():
    ran = []

    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        release_all(interrupted, lambda: ran.append('last'))
    assert ran == ['last']


def test_nothing_to_release_is_nothing_to_do():
    release_all()
