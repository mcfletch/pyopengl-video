"""Counting this process's open file descriptors, to see that a file was closed."""
import os
from pathlib import Path

import pytest

_TABLE = Path('/proc/self/fd')


def open_descriptors():
    """How many file descriptors this process holds; skips where it cannot say."""
    if not _TABLE.is_dir():
        pytest.skip('this platform does not list a process\'s descriptors')
    return len(os.listdir(_TABLE))
