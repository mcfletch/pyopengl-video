"""Opening and playing a movie in each codec and container the package reads.

The decoders' own tests hold every picture to its reference checksum. These
hold the path above them: the file is recognised, the track's codec reaches
the right decoder, and the player shows the same picture for a frame however
it arrived there.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from OpenGL.GL import GL_RGBA, GL_TEXTURE_2D, GL_UNSIGNED_BYTE, glBindTexture, glGetTexImage

from pyopengl_video.playback import Movie
from tests.mp4_build import movie as build_movie
from tests.mp4_build import visual_entry

DATA = Path(__file__).parent / 'data' / 'decode'

#: file -> (codec, the stream's name in its codec's table)
MOVIES = {
    'high_bpyramid.mp4': ('h264', 'high_bpyramid'),
    'high_bpyramid.h264': ('h264', 'high_bpyramid'),
    'hevc_bpyramid.mp4': ('hevc', 'hevc_bpyramid'),
    'hevc_bpyramid.h265': ('hevc', 'hevc_bpyramid'),
    'hevc_main10.h265': ('hevc', 'hevc_main10'),
    'av1_altref.mp4': ('av1', 'av1_altref'),
    'av1_altref.ivf': ('av1', 'av1_altref'),
    'vp9_altref.mp4': ('vp9', 'vp9_altref'),
    'vp9_altref.ivf': ('vp9', 'vp9_altref'),
    'vp9_profile2.ivf': ('vp9', 'vp9_profile2'),
}


def table(codec, name):
    return json.loads((DATA / f'{codec}.json').read_text())['streams'][name]


def picture(texture, size):
    glBindTexture(GL_TEXTURE_2D, texture)
    data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
    glBindTexture(GL_TEXTURE_2D, 0)
    return np.frombuffer(data, np.uint8).reshape(size[1], size[0], 4).copy()


@pytest.mark.parametrize('path', sorted(MOVIES))
def test_a_movie_is_opened_by_its_contents_with_no_gl(path):
    codec, name = MOVIES[path]
    stream = table(codec, name)
    with Movie.open(DATA / path) as movie:
        assert movie.codec == codec
        assert movie.size == tuple(stream['size'])
        assert movie.frames == len(stream['md5'])
        assert movie.duration == pytest.approx(movie.frames / 30, rel=0.05)


@pytest.mark.parametrize('path', sorted(MOVIES))
def test_a_file_object_opens_as_its_path_does(path):
    with (DATA / path).open('rb') as handle, Movie.open(handle) as movie:
        assert movie.codec == MOVIES[path][0]


@pytest.mark.parametrize('path', sorted(MOVIES))
def test_every_frame_plays_and_a_seek_shows_the_same_picture(path, vaapi_decodes):
    codec, _ = MOVIES[path]
    vaapi_decodes(codec)
    with Movie.open(DATA / path) as movie:
        playing = movie.player()
        try:
            forward = [picture(playing.frame_at(index / 30 + 0.001), movie.size)
                       for index in range(movie.frames)]
            assert playing.frame == movie.frames - 1
            for index in (movie.frames // 2, 1, movie.frames - 2, 0):
                again = picture(playing.frame_at(index / 30 + 0.001), movie.size)
                assert playing.frame == index
                assert np.array_equal(again, forward[index]), index
        finally:
            playing.close()
    # The source moves every frame, so a player stuck on one would show it.
    assert all(not np.array_equal(earlier, later)
               for earlier, later in zip(forward, forward[1:], strict=False))
    # The last frame, since a fixture may fade in from black.
    assert (forward[-1][..., 3] == 255).all()
    assert forward[-1][..., :3].std() > 10


def test_an_mp4_whose_video_is_in_another_codec_names_it(tmp_path):
    path = tmp_path / 'old.mp4'
    path.write_bytes(build_movie(visual_entry('mp4v', 64, 48), [bytes(8)]))
    with pytest.raises(ValueError, match='mp4v'):
        Movie.open(path)
