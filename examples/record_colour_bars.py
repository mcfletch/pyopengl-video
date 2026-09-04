"""Record a test card, for checking a recording by eye in any video player.

Every backend has to get three things right that a well-formed file will not
reveal on its own: which way up the picture is, which way round its colour
channels are, and what its black and white levels mean. This draws a frame that
makes each of those a glance rather than an investigation, records it, and
prints what the result should look like.

    python examples/record_colour_bars.py bars.mp4

**What to look for**, playing the file back:

* **The word TOP at the top and BOTTOM at the bottom**, spelled out. OpenGL
  counts rows from the bottom of the frame and every video format counts them
  from the top, so a picture that has come through upside down is the ordinary
  mistake here rather than an exotic one -- and reading the word is quicker and
  surer than reasoning about which colour ought to be where.
* **A red band along the top and a blue band along the bottom**, carrying those
  words. Red and blue are at opposite ends on purpose: if the bands read the
  right way up but their colours have traded places, the red and blue channels
  are the wrong way round rather than the picture being flipped.
* **Eight colour bars**, left to right: white, yellow, cyan, green, magenta,
  red, blue, black. That is the usual order, brightest to darkest. If red and
  blue are exchanged the yellow and cyan bars trade places too, which is easier
  to see than the primaries themselves.
* **A grey ramp** below the bars, black at the left and white at the right, in
  sixteen even steps. The first step should be black and the last white, with no
  crushing at either end: if the darkest steps run together, or the brightest
  do, the stream's range and the player's disagree.
* **A white marker sweeping left to right** across the blue band, once every two
  seconds, so motion and frame pacing are visible.

The frame is drawn with nothing but scissored clears, so the picture depends on
no shader, no texture and no geometry -- only on the path from the colour buffer
to the file.
"""
from __future__ import annotations

import argparse
import sys

import glfw
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DRAW_FRAMEBUFFER,
    GL_NEAREST,
    GL_READ_FRAMEBUFFER,
    GL_SCISSOR_TEST,
    glBindFramebuffer,
    glBlitFramebuffer,
    glClear,
    glClearColor,
    glDisable,
    glEnable,
    glScissor,
    glViewport,
)

from pyopengl_video import encoders, open_encoder
from pyopengl_video.mp4 import MP4Writer

#: The eight bars, brightest to darkest, as a video engineer would order them.
#: Red and blue sit apart on purpose, and so do yellow and cyan: an exchange of
#: those two channels moves four of the eight bars.
BARS = [
    ('white', (1.0, 1.0, 1.0)),
    ('yellow', (1.0, 1.0, 0.0)),
    ('cyan', (0.0, 1.0, 1.0)),
    ('green', (0.0, 1.0, 0.0)),
    ('magenta', (1.0, 0.0, 1.0)),
    ('red', (1.0, 0.0, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
    ('black', (0.0, 0.0, 0.0)),
]

#: How the frame is divided, top to bottom, as fractions of its height.
TOP_BAND = 0.16
BARS_BAND = 0.42
RAMP_BAND = 0.14
MARKER_BAND = 0.10
#: The rest is the bottom band.

GREY_STEPS = 16

#: A block alphabet, five wide and seven tall, holding just the letters the two
#: words need. Drawn as scissored clears like everything else here, so the test
#: card needs no font, no texture and no geometry -- which keeps it a picture of
#: the recording path rather than of the text renderer.
GLYPHS = {
    'T': ('11111', '00100', '00100', '00100', '00100', '00100', '00100'),
    'O': ('01110', '10001', '10001', '10001', '10001', '10001', '01110'),
    'P': ('11110', '10001', '10001', '11110', '10000', '10000', '10000'),
    'B': ('11110', '10001', '10001', '11110', '10001', '10001', '11110'),
    'M': ('10001', '11011', '10101', '10001', '10001', '10001', '10001'),
}
GLYPH_WIDTH, GLYPH_HEIGHT = 5, 7


def fill(x, top, width, height, colour, frame_height):
    """Clear a rectangle given in picture coordinates, top-left origin.

    OpenGL measures from the bottom left and a picture is described from the top
    left, so this is where the two are reconciled -- once, rather than at every
    call site.
    """
    glScissor(int(x), int(frame_height - top - height), int(width), int(height))
    glClearColor(*colour, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)


def text_width(text, pixel):
    """How wide `text` will be drawn, with one blank column between letters."""
    return len(text) * (GLYPH_WIDTH + 1) * pixel - pixel


def draw_text(text, left, top, pixel, colour, frame_height):
    """Draw `text` in block letters, its top-left corner at (`left`, `top`)."""
    for index, letter in enumerate(text):
        rows = GLYPHS[letter]
        origin = left + index * (GLYPH_WIDTH + 1) * pixel
        for row, bits in enumerate(rows):
            run = 0                       # merge neighbouring lit cells into one clear
            for column in range(GLYPH_WIDTH + 1):
                lit = column < GLYPH_WIDTH and bits[column] == '1'
                if lit:
                    run += 1
                elif run:
                    fill(origin + (column - run) * pixel, top + row * pixel,
                         run * pixel, pixel, colour, frame_height)
                    run = 0


def draw_test_card(width, height, phase):
    """Draw one frame of the test card. `phase` runs 0.0 to 1.0 and sweeps the marker."""
    top_height = height * TOP_BAND
    bars_height = height * BARS_BAND
    ramp_height = height * RAMP_BAND
    marker_height = height * MARKER_BAND
    bars_top = top_height
    ramp_top = bars_top + bars_height
    marker_top = ramp_top + ramp_height
    bottom_top = marker_top + marker_height
    bottom_height = height - bottom_top

    glViewport(0, 0, width, height)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glEnable(GL_SCISSOR_TEST)

    # Red at the top and blue at the bottom, each saying which it is. The words
    # settle the orientation on their own; the colours then say whether the red
    # and blue channels arrived the right way round.
    fill(0, 0, width, top_height, (1.0, 0.0, 0.0), height)
    fill(0, bottom_top, width, bottom_height, (0.0, 0.0, 1.0), height)
    label(width, height, 'TOP', 0, top_height)
    label(width, height, 'BOTTOM', bottom_top, bottom_height)

    bar_width = width / len(BARS)
    for index, (_name, colour) in enumerate(BARS):
        fill(index * bar_width, bars_top, bar_width + 1, bars_height, colour, height)

    # A ramp in even steps, so crushed blacks or clipped whites show as steps
    # that have run together rather than as a subtly wrong picture.
    step_width = width / GREY_STEPS
    for step in range(GREY_STEPS):
        level = step / (GREY_STEPS - 1)
        fill(step * step_width, ramp_top, step_width + 1, ramp_height,
             (level, level, level), height)

    # The marker: motion, and something to scrub against. Its lane is dark so a
    # white square reads against it at any brightness.
    fill(0, marker_top, width, marker_height, (0.1, 0.1, 0.1), height)
    marker = marker_height * 0.7
    travel = (width - marker) * phase
    fill(travel, marker_top + (marker_height - marker) / 2, marker, marker,
         (1.0, 1.0, 1.0), height)

    glDisable(GL_SCISSOR_TEST)


def label(width, height, text, band_top, band_height):
    """Centre `text` in a band, as large as it will comfortably sit."""
    pixel = max(1, int(band_height * 0.55) // GLYPH_HEIGHT)
    draw_text(text, (width - text_width(text, pixel)) / 2,
              band_top + (band_height - GLYPH_HEIGHT * pixel) / 2,
              pixel, (1.0, 1.0, 1.0), height)


def capture(framebuffer, width, height):
    """Copy the finished frame into the encoder's surface, the right way up."""
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer)
    glBlitFramebuffer(0, 0, width, height,
                      0, height, width, 0,        # destination Y reversed: the flip
                      GL_COLOR_BUFFER_BIT, GL_NEAREST)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('path', help='the .mp4 to write')
    parser.add_argument('--size', default='1280x720', metavar='WxH')
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--bitrate', type=int, default=20_000_000,
                        help='generous by default, so what you are judging is '
                             'the colour and not the compression')
    options = parser.parse_args(argv)
    width, _, height = options.size.partition('x')
    width, height = int(width), int(height)
    frames = int(round(options.seconds * options.fps))
    sweep = options.fps * 2                       # the marker crosses every two seconds

    if not glfw.init():
        raise SystemExit('GLFW will not start')
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    if sys.platform.startswith('linux'):
        # A Linux encoder is handed the frame as a DMA-BUF exported from the
        # texture, which is an EGL extension; GLFW makes a GLX context by
        # default on X11, and a GLX context cannot export one.
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(width, height, 'colour bars', None, None)
    if not window:
        raise SystemExit('no OpenGL context')
    glfw.make_context_current(window)

    available = encoders()
    if not available:
        raise SystemExit('no hardware encoder is reachable from this OpenGL context')

    with open_encoder(width, height, fps=options.fps, bitrate=options.bitrate,
                      gop=options.fps) as encoder:
        ring = [encoder.new_input() for _ in range(encoder.input_slots)]
        step = encoder.timescale // options.fps
        with MP4Writer(options.path, encoder) as movie:
            for index in range(frames):
                draw_test_card(width, height, (index % sweep) / sweep)
                handle = ring[index % len(ring)]
                with handle.for_drawing():
                    capture(handle.framebuffer, width, height)
                movie.write(encoder.encode(handle, timestamp=index * step))
            movie.write(encoder.flush())

    glfw.terminate()
    print(f'{frames} frames of {width}x{height} at {options.fps} fps '
          f'written to {options.path}')
    print(f'encoded by the {available[0].name!r} backend on {available[0].vendor} '
          'hardware, limited-range BT.709')
    print('\nplaying it back, from the top of the picture down:')
    print('  a red band reading TOP     -- if this is at the bottom, the picture')
    print('                                came through upside down')
    print('  ' + ', '.join(name for name, _ in BARS))
    print(f'  a grey ramp in {GREY_STEPS} steps, black at the left, white at the right')
    print('  a white marker crossing a dark lane, once every two seconds')
    print('  a blue band reading BOTTOM')
    return 0


if __name__ == '__main__':
    sys.exit(main())
