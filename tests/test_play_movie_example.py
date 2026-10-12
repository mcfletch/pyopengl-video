"""``examples/play_movie.py``: its clock and layout, and a real run of it.

The example is what a reader copies, so its arithmetic is held as the
library's is, and the script itself is run once, drawing a frame of a fixture
movie into a hidden window and saving it.
"""
import importlib.util
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / 'examples' / 'play_movie.py'
DATA = Path(__file__).parent / 'data' / 'decode'


def load():
    spec = importlib.util.spec_from_file_location('play_movie', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def example():
    return load()


class TestClock:
    def test_time_advances_while_playing(self, example):
        clock = example.Clock(10.0)
        clock.advance(0.5)
        assert clock.position == pytest.approx(0.5)

    def test_a_paused_clock_holds(self, example):
        clock = example.Clock(10.0, start=2.0)
        clock.paused = True
        clock.advance(1.0)
        assert clock.position == 2.0

    def test_it_stops_at_the_end_and_says_so(self, example):
        clock = example.Clock(3.0)
        clock.advance(5.0)
        assert clock.position == 3.0
        assert clock.finished

    def test_a_looping_clock_starts_again(self, example):
        clock = example.Clock(3.0, loop=True)
        clock.advance(4.0)
        assert clock.position == pytest.approx(1.0)
        assert not clock.finished

    def test_seeking_stays_inside_the_movie(self, example):
        clock = example.Clock(10.0, start=2.0)
        clock.seek(-5.0)
        assert clock.position == 0.0
        clock.seek(50.0)
        assert clock.position == 10.0


class TestFit:
    def test_a_wide_movie_in_a_square_window_is_letterboxed(self, example):
        assert example.fit((1920, 1080), (800, 800)) == pytest.approx((1.0, 1080 / 1920))

    def test_a_tall_movie_in_a_wide_window_is_pillarboxed(self, example):
        assert example.fit((480, 640), (800, 600)) == pytest.approx(((480 / 640) / (800 / 600), 1.0))

    def test_a_movie_the_windows_shape_fills_it(self, example):
        assert example.fit((640, 360), (1280, 720)) == pytest.approx((1.0, 1.0))


def png_size(path):
    data = path.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    width, height = struct.unpack_from('>II', data, 16)
    return width, height


class TestPng:
    def test_rows_are_written_top_first(self, example, tmp_path):
        """glReadPixels hands back the bottom row first."""
        bottom, top = bytes([255, 0, 0]), bytes([0, 0, 255])
        target = tmp_path / 'two.png'
        example.write_png(target, 1, 2, bottom + top)
        data = target.read_bytes()
        length = struct.unpack_from('>I', data, 33)[0]
        raw = zlib.decompress(data[41:41 + length])
        assert raw == b'\x00' + top + b'\x00' + bottom
        assert png_size(target) == (1, 2)


@pytest.mark.usefixtures('decoder_available')
def test_the_example_draws_a_frame_of_a_movie(tmp_path):
    target = tmp_path / 'frame.png'
    finished = subprocess.run(
        [sys.executable, str(SCRIPT), str(DATA / 'high_bpyramid.mp4'),
         '--screenshot', str(target), '--at', '0.5', '--size', '240x136'],
        capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert finished.returncode == 0, finished.stderr
    assert 'frame 15' in finished.stdout
    assert png_size(target) == (240, 136)
