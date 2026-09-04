"""Encoding through Intel's hardware, from an OpenGL frame.

These need a machine whose OpenGL context is running on an Intel GPU with a
video encoder. They skip cleanly anywhere else.
"""
import sys

import numpy as np
import pytest

from pyopengl_video import vpl
from pyopengl_video.encoder import EncoderError

pytestmark = pytest.mark.skipif(sys.platform != 'win32',
                                reason='the oneVPL backend is Windows-only so far')


@pytest.fixture
def encoder(vpl_available):
    """A 320x240 encoder, closed when the test ends."""
    from pyopengl_video.vpl.encoder import VPLEncoder
    built = VPLEncoder(320, 240, fps=30, bitrate=2_000_000)
    yield built
    built.close()


def draw(handle, level):
    """Fill a handle's texture with a flat grey, the way a recorder blits."""
    from OpenGL.GL import (
        GL_COLOR_BUFFER_BIT,
        GL_DRAW_FRAMEBUFFER,
        glBindFramebuffer,
        glClear,
        glClearColor,
    )
    with handle.for_drawing():
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, handle.framebuffer)
        glClearColor(level, level * 0.5, 1.0 - level, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)


def test_the_backend_is_discovered_where_the_library_is(vpl_available):
    from pyopengl_video import encoders
    assert 'vpl' in [backend.name for backend in encoders()]


def test_the_backend_reports_itself_unavailable_off_windows(monkeypatch):
    monkeypatch.setattr(vpl, 'SUPPORTED_PLATFORM', False)
    assert vpl.probe() is False


def test_an_encoder_opens_on_the_adapter_the_context_uses(encoder):
    from pyopengl_video.windows import interop
    assert encoder.adapter == interop.adapter_for_context()
    assert encoder.size == (320, 240)
    assert encoder.zero_copy and encoder.allocates_inputs


def test_it_refuses_a_codec_it_does_not_encode(vpl_available):
    from pyopengl_video.vpl.encoder import VPLEncoder
    with pytest.raises(EncoderError) as caught:
        VPLEncoder(320, 240, codec='av1')
    assert 'av1' in str(caught.value)


def test_it_names_an_unknown_preset_and_says_what_it_knows(vpl_available):
    from pyopengl_video.vpl.encoder import VPLEncoder
    with pytest.raises(EncoderError) as caught:
        VPLEncoder(320, 240, preset='turbo')
    assert 'turbo' in str(caught.value) and 'p4' in str(caught.value)


def test_it_names_an_unknown_rate_control(vpl_available):
    from pyopengl_video.vpl.encoder import VPLEncoder
    with pytest.raises(EncoderError) as caught:
        VPLEncoder(320, 240, rate_control='magic')
    assert 'magic' in str(caught.value) and 'vbr' in str(caught.value)


def test_registering_a_foreign_texture_is_refused_with_the_reason(encoder):
    with pytest.raises(EncoderError) as caught:
        encoder.register(7)
    assert 'new_input' in str(caught.value)


def test_a_new_input_is_a_drawable_texture_of_the_frame_size(encoder):
    handle = encoder.new_input()
    assert handle.texture and handle.framebuffer
    assert handle.size == (320, 240)


def test_the_headers_are_a_sequence_and_a_picture_parameter_set(encoder):
    headers = encoder.headers()
    assert headers.startswith(b'\x00\x00\x00\x01')
    kinds = {byte & 0x1F for byte in _nal_headers(headers)}
    assert 7 in kinds, 'no sequence parameter set'
    assert 8 in kinds, 'no picture parameter set'


def test_the_stream_declares_the_colour_it_was_converted_to(encoder):
    """The hardware picks the RGB-to-YUV matrix, so the stream has to say which.

    Left unstated, a player guesses and the picture comes back with the wrong
    saturation. This asks the runtime what it settled on, which is where a
    rejected declaration would show up.
    """
    import ctypes
    from ctypes import byref, c_void_p

    from pyopengl_video.vpl import api

    signal = api.mfxExtVideoSignalInfo()
    signal.Header.BufferId = api.MFX_EXTBUFF_VIDEO_SIGNAL_INFO
    signal.Header.BufferSz = ctypes.sizeof(signal)
    buffers = (c_void_p * 1)(ctypes.cast(byref(signal), c_void_p))
    settled = api.mfxVideoParam()
    settled.NumExtParam = 1
    settled.ExtParam = ctypes.cast(buffers, ctypes.POINTER(c_void_p))
    encoder.api.check(
        encoder.api.MFXVideoENCODE_GetVideoParam(encoder.session, byref(settled)),
        'MFXVideoENCODE_GetVideoParam(colour)')

    assert signal.ColourDescriptionPresent == 1
    assert signal.ColourPrimaries == 1, 'not BT.709 primaries'
    assert signal.TransferCharacteristics == 1, 'not the BT.709 transfer'
    assert signal.MatrixCoefficients == 1, 'not the BT.709 matrix'
    assert signal.VideoFullRange == 0, 'the recording is limited range'


def test_encoding_frames_produces_a_stream_that_starts_with_a_key_frame(encoder):
    ring = [encoder.new_input() for _ in range(encoder.input_slots)]
    step = encoder.timescale // 30
    packets = []
    for index in range(12):
        handle = ring[index % len(ring)]
        draw(handle, (index % 6) / 6.0)
        packets.extend(encoder.encode(handle, timestamp=index * step))
    packets.extend(encoder.flush())

    assert packets, 'the encoder produced nothing at all'
    assert packets[0].keyframe, 'a stream has to open with a key frame'
    assert sum(len(packet) for packet in packets) > 0
    for packet in packets:
        assert packet.data.startswith(b'\x00\x00\x00\x01')


def test_every_frame_submitted_comes_back_out(encoder):
    ring = [encoder.new_input() for _ in range(encoder.input_slots)]
    step = encoder.timescale // 30
    count = 10
    packets = []
    for index in range(count):
        handle = ring[index % len(ring)]
        draw(handle, (index % 5) / 5.0)
        packets.extend(encoder.encode(handle, timestamp=index * step))
    packets.extend(encoder.flush())
    assert len(packets) == count


def test_timestamps_come_back_as_they_were_given(encoder):
    handle = encoder.new_input()
    step = encoder.timescale // 30
    draw(handle, 0.5)
    packets = encoder.encode(handle, timestamp=7 * step)
    packets.extend(encoder.flush())
    assert packets[0].timestamp == 7 * step


def test_a_forced_key_frame_is_one(encoder):
    ring = [encoder.new_input() for _ in range(encoder.input_slots)]
    step = encoder.timescale // 30
    packets = []
    for index in range(4):
        handle = ring[index % len(ring)]
        draw(handle, index / 4.0)
        packets.extend(encoder.encode(handle, timestamp=index * step,
                                      force_idr=index == 2))
    packets.extend(encoder.flush())
    assert packets[2].keyframe


def test_encoding_after_close_is_refused(encoder):
    handle = encoder.new_input()
    encoder.close()
    with pytest.raises(EncoderError):
        encoder.encode(handle, timestamp=0)


def test_closing_twice_is_harmless(vpl_available):
    from pyopengl_video.vpl.encoder import VPLEncoder
    built = VPLEncoder(160, 128, fps=30)
    built.close()
    built.close()


def _nal_headers(stream):
    """The first byte of each NAL unit in an Annex-B fragment."""
    found = []
    index = 0
    while True:
        start = stream.find(b'\x00\x00\x00\x01', index)
        if start < 0:
            return found
        found.append(stream[start + 4])
        index = start + 4


# The encoder is handed a surface, and a handover that quietly went nowhere
# would still produce a well-formed stream -- of something else. What follows
# checks that the output varies with what was drawn, which is the part no amount
# of structural checking can show.

SIZE = (320, 240)


@pytest.fixture
def blit_source(gl_context):
    """A texture and framebuffer holding a frame, to blit from as a recorder does."""
    from OpenGL.GL import (
        GL_RGBA,
        GL_TEXTURE_2D,
        GL_UNSIGNED_BYTE,
        glBindTexture,
        glTexSubImage2D,
    )

    from pyopengl_video.inputs import (
        create_framebuffer,
        create_rgba_texture,
        delete_framebuffer,
        delete_texture,
    )
    texture = create_rgba_texture(*SIZE)
    framebuffer = create_framebuffer(texture)

    def put(frame):
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SIZE[0], SIZE[1],
                        GL_RGBA, GL_UNSIGNED_BYTE, frame)
        glBindTexture(GL_TEXTURE_2D, 0)
        return framebuffer

    put.framebuffer = framebuffer
    yield put
    delete_framebuffer(framebuffer)
    delete_texture(texture)


def blit_into(handle, framebuffer):
    """Copy a frame into the encoder's surface, turning it over as it goes.

    The same call the OpenGLContext recorder makes: OpenGL's framebuffer starts
    at the bottom left and an encoder reads a surface from its first row, so the
    destination's Y coordinates run backwards.
    """
    from OpenGL.GL import (
        GL_COLOR_BUFFER_BIT,
        GL_DRAW_FRAMEBUFFER,
        GL_NEAREST,
        GL_READ_FRAMEBUFFER,
        glBindFramebuffer,
        glBlitFramebuffer,
        glFinish,
    )
    width, height = SIZE
    with handle.for_drawing():
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, handle.framebuffer)
        glBlitFramebuffer(0, 0, width, height, 0, height, width, 0,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glFinish()


def flat_frame():
    """One mid-grey covering the whole frame."""
    frame = np.empty((SIZE[1], SIZE[0], 4), dtype=np.uint8)
    frame[...] = (128, 128, 128, 255)
    return frame


def noise_frame(seed=1234):
    """Random pixels, which no encoder can predict its way out of."""
    frame = np.empty((SIZE[1], SIZE[0], 4), dtype=np.uint8)
    frame[..., :3] = np.random.default_rng(seed).integers(
        0, 256, size=(SIZE[1], SIZE[0], 3), dtype=np.uint8)
    frame[..., 3] = 255
    return frame


def moving_frame(phase):
    """A diagonal gradient that shifts, so consecutive frames differ."""
    x = (np.arange(SIZE[0], dtype=np.int32)[None, :] + phase * 8) % 256
    y = (np.arange(SIZE[1], dtype=np.int32)[:, None] + phase * 4) % 256
    frame = np.empty((SIZE[1], SIZE[0], 4), dtype=np.uint8)
    frame[..., 0], frame[..., 1] = x, y
    frame[..., 2], frame[..., 3] = (x + y) % 256, 255
    return frame


def test_the_flipping_blit_puts_the_frame_the_right_way_up(vpl_available, blit_source):
    """What OpenGL calls the bottom must end up in the surface's last row.

    OpenGL's framebuffer starts at the bottom left; an encoder reads a surface
    from its first row and calls that the top of the picture. So the recorder's
    blit turns the frame over, and the invariant is stated in OpenGL's own
    terms: paint the *bottom* half of the source in GL coordinates, and it has
    to arrive at the far end of the surface. Without the flip the video comes
    out upside down.
    """
    from OpenGL.GL import (
        GL_COLOR_BUFFER_BIT,
        GL_DRAW_FRAMEBUFFER,
        GL_SCISSOR_TEST,
        glBindFramebuffer,
        glClear,
        glClearColor,
        glDisable,
        glEnable,
        glScissor,
    )

    from pyopengl_video.vpl.encoder import VPLEncoder
    width, height = SIZE
    source = blit_source.framebuffer
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, source)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glEnable(GL_SCISSOR_TEST)
    glScissor(0, 0, width, height // 2)          # y from 0: OpenGL's bottom half
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glDisable(GL_SCISSOR_TEST)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)

    encoder = VPLEncoder(*SIZE, fps=30)
    try:
        handle = encoder.new_input()
        blit_into(handle, source)
        surface = np.frombuffer(handle.read(), dtype=np.uint8).reshape(
            height, width, 4)
    finally:
        encoder.close()

    assert surface[0, :, :3].mean() < 8, "the surface's first row is not the dark half"
    assert surface[-1, :, :3].mean() > 247, "the surface's last row is not the lit half"


def test_a_detailed_frame_costs_more_than_a_flat_one(vpl_available, blit_source):
    """A key frame of noise is far larger than a key frame of one colour."""
    from pyopengl_video.vpl.encoder import VPLEncoder
    sizes = {}
    for name, frame in (('flat', flat_frame()), ('noise', noise_frame())):
        encoder = VPLEncoder(*SIZE, fps=30, bitrate=20_000_000, gop=30)
        try:
            handle = encoder.new_input()
            blit_into(handle, blit_source(frame))
            packets = encoder.encode(handle, timestamp=0)
            packets.extend(encoder.flush())
            sizes[name] = sum(len(packet) for packet in packets)
        finally:
            encoder.close()
    assert sizes['noise'] > sizes['flat'] * 5, sizes


def test_a_still_sequence_costs_less_than_a_moving_one(vpl_available, blit_source):
    """Frames that do not change compress to almost nothing after the first."""
    from pyopengl_video.vpl.encoder import VPLEncoder
    sizes = {}
    for name, still in (('still', True), ('moving', False)):
        encoder = VPLEncoder(*SIZE, fps=30, rate_control='constqp', gop=60)
        try:
            ring = [encoder.new_input() for _ in range(encoder.input_slots)]
            packets = []
            for index in range(16):
                handle = ring[index % len(ring)]
                blit_into(handle, blit_source(moving_frame(0 if still else index)))
                packets.extend(encoder.encode(
                    handle, timestamp=index * encoder.timescale // 30))
            packets.extend(encoder.flush())
            # after the opening key frame, what did the rest of the sequence cost?
            sizes[name] = sum(len(packet) for packet in packets[1:])
        finally:
            encoder.close()
    assert sizes['moving'] > sizes['still'] * 3, sizes


def test_a_recording_is_a_complete_mp4_holding_every_frame(vpl_available, blit_source,
                                                           tmp_path):
    """The whole path: draw, blit, encode, mux, and a file that parses back."""
    import struct

    from pyopengl_video.mp4 import MP4Writer
    from pyopengl_video.vpl.encoder import VPLEncoder
    from tests.test_mp4 import boxes, path_to

    target = tmp_path / 'clip.mp4'
    frames = 24
    encoder = VPLEncoder(*SIZE, fps=24, bitrate=2_000_000, gop=12)
    try:
        ring = [encoder.new_input() for _ in range(encoder.input_slots)]
        with MP4Writer(target, encoder) as recording:
            for index in range(frames):
                handle = ring[index % len(ring)]
                blit_into(handle, blit_source(moving_frame(index)))
                recording.write(encoder.encode(
                    handle, timestamp=index * encoder.timescale // 24))
            recording.write(encoder.flush())
    finally:
        encoder.close()

    movie = target.read_bytes()
    assert list(boxes(movie)) == ['ftyp', 'mdat', 'moov']
    stsz = path_to(movie, 'moov', 'trak', 'mdia', 'minf', 'stbl', 'stsz')
    assert struct.unpack_from('>I', stsz, 8)[0] == frames
    tkhd = path_to(movie, 'moov', 'trak', 'tkhd')
    assert struct.unpack_from('>II', tkhd, 76) == (SIZE[0] << 16, SIZE[1] << 16)
    avcc = path_to(movie, 'moov', 'trak', 'mdia', 'minf', 'stbl', 'stsd', 'avc1', 'avcC')
    assert avcc[0] == 1
    assert avcc[1] in (0x42, 0x4d, 0x64)
