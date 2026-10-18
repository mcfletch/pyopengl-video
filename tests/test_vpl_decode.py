"""Decoding through Intel's hardware, into OpenGL textures.

The decisions the decoder makes without a driver are checked on any machine.
The rest needs an OpenGL context running on an Intel GPU, and skips cleanly
anywhere else. The video processor converts to RGB itself, so a picture is not
bit-exact with the shader path the other backends take: what is held here is the
frame count, the display order, the timestamps, the colour and the orientation,
and the last two through a stream the Intel encoder made from known colours.
"""
import hashlib
import json
import sys
from ctypes import c_void_p
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from OpenGL.GL import (
    GL_PACK_ALIGNMENT,
    GL_RGBA,
    GL_TEXTURE_2D,
    GL_UNSIGNED_BYTE,
    glBindTexture,
    glDeleteTextures,
    glGenTextures,
    glGetTexImage,
    glPixelStorei,
    glTexImage2D,
)
from OpenGL.GL import GL_RGBA8 as RGBA8

from pyopengl_video import DecoderError, decoders, open_decoder, open_encoder, vpl
from pyopengl_video.colour import MATRIX_BT601, MATRIX_BT709, MATRIX_BT2020, Colour
from pyopengl_video.convert import NV12Converter, create_output_texture
from pyopengl_video.decoder import Sample
from pyopengl_video.mp4 import MP4Writer
from pyopengl_video.playback import Movie
from pyopengl_video.vpl import api
from pyopengl_video.vpl.decoder import (
    VPLDecodeError,
    VPLDecoder,
    colour_of_header,
    vpp_signal,
)
from tests.conftest import blit

DATA = Path(__file__).parent / 'data' / 'decode'
TICK = 3000                                  # 30 fps in a 90 kHz timescale
TABLES = {codec: json.loads((DATA / f'{codec}.json').read_text())['streams']
          for codec in ('h264', 'hevc', 'vp9')}
#: Streams whose colour the video processor cannot convert, with what it says.
REFUSED = {'hevc_main10': 'coefficients 9'}
STREAMS = [(codec, name) for codec, table in TABLES.items() for name in sorted(table)
           if name not in REFUSED]


class TestWithNoDriver:
    """What the decoder decides before it asks the hardware anything."""

    def test_a_stream_that_states_no_colour_is_resolved_by_picture_height(self):
        unstated = api.mfxExtVideoSignalInfo()
        assert colour_of_header(unstated) == Colour()
        assert vpp_signal(Colour(), 480).InTransferMatrix == api.MFX_TRANSFERMATRIX_BT601
        assert vpp_signal(Colour(), 1080).InTransferMatrix == api.MFX_TRANSFERMATRIX_BT709

    def test_a_failing_release_does_not_keep_the_device_from_being_given_back(self):
        decoder = object.__new__(VPLDecoder)     # state set directly: building one needs a driver
        interop_device, allocator, device = mock.Mock(), mock.Mock(), mock.Mock()
        interop_device.close.side_effect = RuntimeError('interop')
        vars(decoder).update(
            _closed=False, _lent={}, _processing=True, _decoding=True, session=c_void_p(1),
            api=mock.Mock(), interop=interop_device, allocator=allocator, device=device)
        decoder.api.MFXVideoVPP_Close.side_effect = RuntimeError('video processor')
        with pytest.raises(RuntimeError):
            decoder.close()
        decoder.api.MFXVideoDECODE_Close.assert_called_once()
        decoder.api.MFXClose.assert_called_once()
        interop_device.close.assert_called_once()
        allocator.close.assert_called_once()
        device.close.assert_called_once()
        assert decoder.device is None
        decoder.close()                           # closed: nothing is released twice
        device.close.assert_called_once()

    def test_a_stated_colour_description_is_read_from_the_header(self):
        stated = api.mfxExtVideoSignalInfo()
        stated.ColourDescriptionPresent = 1
        stated.MatrixCoefficients = MATRIX_BT709
        stated.VideoFullRange = 1
        stated.ColourPrimaries, stated.TransferCharacteristics = 1, 13
        assert colour_of_header(stated) == Colour(
            matrix=MATRIX_BT709, full_range=True, primaries=1, transfer=13)

    @pytest.mark.parametrize(('colour', 'matrix', 'nominal'), [
        (Colour(matrix=MATRIX_BT709), api.MFX_TRANSFERMATRIX_BT709, api.MFX_NOMINALRANGE_16_235),
        (Colour(matrix=MATRIX_BT601), api.MFX_TRANSFERMATRIX_BT601, api.MFX_NOMINALRANGE_16_235),
        (Colour(matrix=MATRIX_BT601, full_range=True), api.MFX_TRANSFERMATRIX_BT601,
         api.MFX_NOMINALRANGE_0_255),
    ])
    def test_the_video_processor_is_told_the_streams_matrix_and_range(
            self, colour, matrix, nominal):
        signal = vpp_signal(colour, 720)
        assert (signal.InTransferMatrix, signal.InNominalRange) == (matrix, nominal)
        assert signal.OutNominalRange == api.MFX_NOMINALRANGE_0_255
        assert signal.Header.BufferId == api.MFX_EXTBUFF_VPP_VIDEO_SIGNAL_INFO
        assert signal.Header.BufferSz == 48

    def test_a_matrix_the_video_processor_lacks_is_refused_by_name(self):
        with pytest.raises(DecoderError, match='BT.601 and BT.709.*coefficients 9'):
            vpp_signal(Colour(matrix=MATRIX_BT2020), 1080)

    def test_a_setting_the_decoder_does_not_have_is_refused(self):
        with pytest.raises(DecoderError, match='no setting named speed'):
            VPLDecoder('h264', speed=3)

    def test_a_codec_the_backend_does_not_decode_is_refused(self):
        with pytest.raises(DecoderError, match='not .mpeg2.'):
            VPLDecoder('mpeg2')

    def test_the_decoder_is_unavailable_off_windows(self, monkeypatch):
        monkeypatch.setattr(vpl, 'SUPPORTED_PLATFORM', False)
        assert vpl.decoder_probe() is False
        assert vpl.decoder_explain() == ''

    def test_the_status_of_a_refused_call_is_named(self):
        error = VPLDecodeError(-15, 'MFXVideoDECODE_Init', '1920x1080 h264')
        assert isinstance(error, DecoderError)
        assert 'MFX_ERR_INVALID_VIDEO_PARAM' in str(error)


@pytest.fixture
def decodes(vpl_decoder_available):
    """A function that skips the test unless this part decodes the codec named."""
    def need(codec):
        if codec not in vpl_decoder_available:
            pytest.skip(f'this Intel GPU does not decode {codec}')
    return need


def frames_of(decoder, source, start=0, stop=None):
    """Every frame the samples from `start` decode to, in display order, released."""
    found = []
    stop = len(source) if stop is None else stop
    for index in range(start, stop):
        for frame in decoder.decode(source.sample(index, parameter_sets=index == start)):
            found.append((frame.timestamp, frame.duration, frame.size))
            frame.release()
    for frame in decoder.flush():
        found.append((frame.timestamp, frame.duration, frame.size))
        frame.release()
    return found


def rgb_of(converter, frame):
    """A frame's picture as RGBA bytes, first row the picture's bottom."""
    width, height = frame.size
    target = create_output_texture(width, height)
    try:
        converter.convert(frame, target)
        glBindTexture(GL_TEXTURE_2D, target)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixels = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
        return np.frombuffer(pixels, np.uint8).reshape(height, width, 4).copy()
    finally:
        converter.forget(target)
        glDeleteTextures([target])


@pytest.mark.skipif(sys.platform != 'win32', reason='the oneVPL backend is Windows-only so far')
class TestOnIntelHardware:
    @pytest.mark.usefixtures('vpl_decoder_available')
    def test_the_backend_is_discovered_with_the_codecs_the_part_decodes(self):
        found = {backend.name: backend for backend in decoders()}
        assert 'vpl' in found
        assert found['vpl'].zero_copy
        assert 'h264' in found['vpl'].codecs

    @pytest.mark.parametrize(('codec', 'name'), STREAMS)
    def test_every_picture_of_a_stream_comes_out_in_display_order(
            self, decodes, codec, name):
        decodes(codec)
        spec = TABLES[codec][name]
        with Movie.open(DATA / spec['file']) as movie, open_decoder(
                codec, headers=movie.headers, size=movie.size, backend='vpl') as decoder:
            seen = frames_of(decoder, movie.source)
        stamps = [timestamp for timestamp, _duration, _size in seen]
        assert len(seen) == len(spec['md5'])
        assert stamps == sorted(stamps)
        assert {size for _timestamp, _duration, size in seen} == {tuple(spec['size'])}

    @pytest.mark.usefixtures('vpl_decoder_available')
    def test_frames_carry_the_samples_timestamps_and_durations(self):
        with Movie.open(DATA / 'high_bpyramid.mp4') as movie, open_decoder(
                'h264', headers=movie.headers, backend='vpl') as decoder:
            source = movie.source
            wanted = sorted((source.sample(index).timestamp, source.sample(index).duration)
                            for index in range(len(source)))
            seen = frames_of(decoder, source)
        assert [(stamp, duration) for stamp, duration, _size in seen] == wanted

    @pytest.mark.usefixtures('vpl_decoder_available')
    def test_the_size_is_known_from_the_headers_before_the_first_sample(self):
        with Movie.open(DATA / 'high_bpyramid.mp4') as movie, open_decoder(
                'h264', headers=movie.headers, backend='vpl') as decoder:
            assert decoder.size == movie.size

    @pytest.mark.usefixtures('vpl_decoder_available')
    def test_a_frame_is_an_rgba_texture_the_decoder_lends(self):
        with Movie.open(DATA / 'high_bpyramid.mp4') as movie, open_decoder(
                'h264', headers=movie.headers, backend='vpl') as decoder:
            source = movie.source
            frames = []
            for index in range(len(source)):
                frames += decoder.decode(source.sample(index, parameter_sets=index == 0))
                if frames:
                    break
            frame = frames[0]
            assert frame.layout == 'rgba'
            assert len(frame.planes) == 1
            assert frame.visible[:2] == (0, 0)
            assert frame.visible[2:] == movie.size
            assert not frame.released
            frame.release()
            assert frame.released
            frame.release()                       # safe twice

    @pytest.mark.usefixtures('vpl_decoder_available')
    def test_holding_every_output_texture_is_reported_with_the_frames_held(self):
        with Movie.open(DATA / 'high_bpyramid.mp4') as movie, open_decoder(
                'h264', headers=movie.headers, backend='vpl', output_slots=2) as decoder:
            source = movie.source
            held = []
            with pytest.raises(DecoderError, match='every one of the 2 output textures'):
                for index in range(len(source)):
                    held += decoder.decode(source.sample(index, parameter_sets=index == 0))
            assert len(held) == 2
            for frame in held:
                frame.release()

    @pytest.mark.usefixtures('vpl_decoder_available')
    def test_closing_gives_back_frames_still_held(self):
        movie = Movie.open(DATA / 'high_bpyramid.mp4')
        decoder = open_decoder('h264', headers=movie.headers, backend='vpl')
        held = []
        for index in range(len(movie.source)):
            held += decoder.decode(movie.source.sample(index, parameter_sets=index == 0))
            if held:
                break
        decoder.close()
        assert held
        assert all(frame.released for frame in held)
        with pytest.raises(DecoderError, match='closed'):
            decoder.decode(movie.source.sample(0))
        decoder.close()                           # safe twice
        movie.close()

    @pytest.mark.usefixtures('vpl_decoder_available')
    def test_a_reset_decoder_decodes_from_a_key_frame_the_same_way(self):
        converter = NV12Converter()
        try:
            with Movie.open(DATA / 'open_gop.h264') as movie, open_decoder(
                    'h264', headers=movie.headers, backend='vpl') as decoder:
                source = movie.source

                def run():
                    found = {}
                    for index in range(len(source)):
                        for frame in decoder.decode(
                                source.sample(index, parameter_sets=index == 0)):
                            found[frame.timestamp] = hashlib.md5(
                                rgb_of(converter, frame)).hexdigest()
                            frame.release()
                    for frame in decoder.flush():
                        found[frame.timestamp] = hashlib.md5(rgb_of(converter, frame)).hexdigest()
                        frame.release()
                    return found

                first = run()
                assert len(first) == len(source)
                decoder.reset()
                assert run() == first
        finally:
            converter.close()

    def test_a_matrix_the_video_processor_lacks_refuses_the_stream_by_name(self, decodes):
        decodes('hevc')
        with Movie.open(DATA / 'hevc_main10.h265') as movie, pytest.raises(
                DecoderError, match=REFUSED['hevc_main10']):
            open_decoder('hevc', headers=movie.headers, size=movie.size, backend='vpl')


def colour_bars(width, height, bars):
    """An RGBA picture of vertical `bars`, with red above blue at the left edge.

    The rows are an OpenGL texture's, first row the bottom.
    """
    picture = np.zeros((height, width, 4), np.uint8)
    picture[..., 3] = 255
    each = width // len(bars)
    for index, colour in enumerate(bars):
        picture[:, index * each:(index + 1) * each, :3] = colour
    picture[height // 2:, :each, :3] = (255, 0, 0)
    picture[:height // 2, :each, :3] = (0, 0, 255)
    return picture


BARS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 255, 255),
        (0, 0, 0), (128, 128, 128), (0, 255, 255)]


@pytest.mark.skipif(sys.platform != 'win32', reason='the oneVPL backend is Windows-only so far')
@pytest.mark.usefixtures('vpl_available')
class TestWhatTheEncoderMade:
    """The round trip: the Intel encoder's stream, decoded on the Intel decoder."""

    WIDTH, HEIGHT = 320, 240

    @pytest.fixture
    def stream(self, vpl_decoder_available, request):
        codec = request.param
        if codec not in vpl_decoder_available:
            pytest.skip(f'this Intel GPU does not decode {codec}')
        width, height = self.WIDTH, self.HEIGHT
        unsupported = None
        try:
            encoder = open_encoder(width, height, codec=codec, backend='vpl', fps=30,
                                   bitrate=4_000_000)
        except api.VPLError as error:
            if error.status != api.MFX_ERR_UNSUPPORTED:
                raise
            unsupported = error
        if unsupported is not None:
            pytest.skip(f'this Intel GPU does not encode {codec}: {unsupported}')
        texture = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     colour_bars(width, height, BARS))
        handles = [encoder.new_input() for _ in range(2)]
        packets = []
        for index in range(10):
            handle = handles[index % 2]
            blit(handle, texture, (width, height))
            packets += encoder.encode(handle, timestamp=index * TICK, duration=TICK)
        packets += encoder.flush()
        headers = encoder.headers()
        encoder.close()
        glDeleteTextures([texture])
        return codec, headers, packets

    @pytest.mark.parametrize('stream', ['h264', 'hevc'], indirect=True)
    def test_the_colours_and_the_orientation_survive(self, stream):
        codec, headers, packets = stream
        converter = NV12Converter()
        shown = []
        try:
            with open_decoder(codec, headers=headers, backend='vpl') as decoder:
                for index, packet in enumerate(packets):
                    sample = Sample(packet.data, packet.timestamp, packet.duration,
                                    packet.keyframe, index)
                    for frame in decoder.decode(sample):
                        shown.append(rgb_of(converter, frame))
                        frame.release()
                for frame in decoder.flush():
                    shown.append(rgb_of(converter, frame))
                    frame.release()
        finally:
            converter.close()
        assert len(shown) == len(packets)
        picture = shown[-1].astype(int)
        each = self.WIDTH // len(BARS)
        top, bottom = 3 * self.HEIGHT // 4, self.HEIGHT // 4
        # The test's blit does not flip, so the first row of the texture the
        # encoder reads is the top of the picture it codes: the bars read
        # back as drawn and the left edge reads back with its halves swapped.
        for index, colour in enumerate(BARS[1:], start=1):
            for row in (top, bottom):
                got = picture[row, index * each + each // 2, :3]
                assert np.abs(got - colour).max() <= 8, (index, row, got)
        assert np.abs(picture[top, 10, :3] - (0, 0, 255)).max() <= 8
        assert np.abs(picture[bottom, 10, :3] - (255, 0, 0)).max() <= 8

    def test_a_recording_plays_back_through_a_movie(self, vpl_decoder_available, tmp_path):
        if 'h264' not in vpl_decoder_available:
            pytest.skip('this Intel GPU does not decode h264')
        width, height = self.WIDTH, self.HEIGHT
        texture = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     colour_bars(width, height, BARS))
        path = tmp_path / 'bars.mp4'
        with open_encoder(width, height, backend='vpl', fps=30, bitrate=4_000_000) as encoder:
            handles = [encoder.new_input() for _ in range(2)]
            with MP4Writer(path, encoder) as writer:
                for index in range(10):
                    handle = handles[index % 2]
                    blit(handle, texture, (width, height))
                    writer.write(encoder.encode(handle, timestamp=index * TICK, duration=TICK))
                writer.write(encoder.flush())
        glDeleteTextures([texture])
        with Movie.open(path) as movie, movie.player(backend='vpl') as player:
            assert movie.size == (width, height)
            shown = player.frame_at(4 * TICK / 90000)
            glBindTexture(GL_TEXTURE_2D, shown)
            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            picture = np.frombuffer(glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE),
                                    np.uint8).reshape(height, width, 4).astype(int)
        each = width // len(BARS)
        for row in (height // 4, 3 * height // 4):
            assert np.abs(picture[row, 3 * each // 2, :3] - BARS[1]).max() <= 8
