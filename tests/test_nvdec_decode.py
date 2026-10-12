"""The NVDEC decoder on real hardware, into real OpenGL textures.

H.264 decoding is exact, so every picture of the x264 fixtures must come out
of the GPU with the bytes ffmpeg's software decoder produced, read back from
the textures the decoder hands out. Both ways of getting a picture into a
texture are held to that: the copy on the GPU and the copy through host
memory. NVENC's streams are decoded too, which is the round trip the encode
plan asked for.
"""
import ctypes
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_PIXEL_UNPACK_BUFFER,
    GL_PIXEL_UNPACK_BUFFER_BINDING,
    GL_RGBA,
    GL_STATIC_DRAW,
    GL_TEXTURE_2D,
    GL_TEXTURE_BINDING_2D,
    GL_UNPACK_ALIGNMENT,
    GL_UNPACK_ROW_LENGTH,
    GL_UNPACK_SKIP_PIXELS,
    GL_UNPACK_SKIP_ROWS,
    GL_UNSIGNED_BYTE,
    glBindBuffer,
    glBindTexture,
    glBufferData,
    glDeleteBuffers,
    glDeleteTextures,
    glGenBuffers,
    glGetIntegerv,
    glGetTexImage,
    glPixelStorei,
)

from pyopengl_video import DecoderError, decoders, open_decoder
from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.colour import MATRIX_BT601
from pyopengl_video.convert import NV12Converter, create_output_texture
from pyopengl_video.decoder import Sample
from pyopengl_video.mp4 import MP4Reader, MP4Writer
from pyopengl_video.nvdec import api
from pyopengl_video.nvdec.decoder import NVDECDecoder
from pyopengl_video.nvenc.encoder import NVENCEncoder
from pyopengl_video.playback import Movie
from tests.conftest import blit, decode_all, gradient_frame, md5s

DATA = Path(__file__).parent / 'data' / 'decode'
RECORDED = json.loads((DATA / 'fixtures.json').read_text())
FIXTURES = RECORDED['streams']
INTERLACED = RECORDED['interlaced']
TICK = 3000                                  # 30 fps in a 90 kHz timescale

#: Pixel storage a renderer might leave set, each of which changes an upload.
PACKING = {GL_UNPACK_ALIGNMENT: 2, GL_UNPACK_ROW_LENGTH: 7,
           GL_UNPACK_SKIP_ROWS: 3, GL_UNPACK_SKIP_PIXELS: 5}

pytestmark = pytest.mark.usefixtures('nvdec_available')


def samples_of(name, start=0, stop=None):
    with AnnexBReader(DATA / f'{name}.h264') as stream:
        stop = len(stream) if stop is None else stop
        return [stream.sample(index, parameter_sets=index == start)
                for index in range(start, stop)]


@pytest.fixture(params=[True, False], ids=['on-the-gpu', 'through-host-memory'])
def decoder(request):
    """A decoder for each way a picture reaches its textures."""
    made = NVDECDecoder(interop=request.param)
    yield made
    made.close()


class TestConformance:
    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_every_picture_matches_the_reference_decode(self, decoder, name):
        assert decode_all(decoder, samples_of(name)) == FIXTURES[name]['md5']

    def test_an_mp4_decodes_from_its_sample_entry(self):
        with MP4Reader(DATA / 'high_bpyramid.mp4') as movie:
            track = movie.video
            with open_decoder('h264', headers=track.parameter_sets(),
                              backend='nvdec') as made:
                assert made.size == (track.width, track.height)
                samples = [track.sample(index) for index in range(len(track))]
                assert decode_all(made, samples) == FIXTURES['high_bpyramid']['md5']

    def test_frames_carry_the_samples_timestamps_in_display_order(self, decoder):
        with MP4Reader(DATA / 'high_bpyramid.mp4') as movie:
            track = movie.video
            shown = []
            for index in range(len(track)):
                sample = track.sample(index, parameter_sets=index == 0)
                for frame in decoder.decode(sample):
                    shown.append((frame.timestamp, frame.duration))
                    frame.release()
            for frame in decoder.flush():
                shown.append((frame.timestamp, frame.duration))
                frame.release()
            assert [timestamp for timestamp, _ in shown] == sorted(
                track.presentation_time(index) for index in range(len(track)))
            assert {duration for _, duration in shown} == {track.sample(0).duration}

    def test_a_frame_describes_its_surface_and_the_picture_in_it(self, decoder):
        [first, *rest] = [frame for sample in samples_of('high_bpyramid')[:8]
                          for frame in decoder.decode(sample)]
        try:
            assert first.layout == 'nv12'
            assert first.texture_size == (128, 80)        # whole macroblocks
            assert first.visible == (0, 0, 120, 68)
            assert first.size == decoder.size == (120, 68)
        finally:
            for frame in [first, *rest]:
                frame.release()

    def test_the_colour_is_this_packages_reading_of_the_stream(self, decoder):
        """The fixtures state no matrix, which is BT.601 below 720 rows."""
        for frame in decode_all(decoder, samples_of('baseline_cavlc')):
            del frame
        assert decoder.colour.matrix == MATRIX_BT601
        assert decoder.colour.chroma_location == 0


class TestSeveralMovies:
    def test_four_decoders_in_one_context_each_decode_their_own(self):
        """A game with screens on its walls: one session per movie, interleaved."""
        names = sorted(FIXTURES)
        streams = {name: samples_of(name) for name in names}
        made = {name: NVDECDecoder() for name in names}
        found = {name: [] for name in names}
        try:
            for index in range(max(len(samples) for samples in streams.values())):
                for name in names:
                    if index < len(streams[name]):
                        found[name] += md5s(made[name].decode(streams[name][index]))
            for name in names:
                found[name] += md5s(made[name].flush())
        finally:
            for decoder in made.values():
                decoder.close()
        assert found == {name: FIXTURES[name]['md5'] for name in names}


class TestSeeking:
    def test_decoding_resumes_from_a_key_frame_after_a_reset(self, decoder):
        name = 'high_bpyramid'
        decode_all(decoder, samples_of(name, stop=10))
        decoder.reset()
        with AnnexBReader(DATA / f'{name}.h264') as stream:
            key = stream.sync_samples[1]
            order = stream.display_order()
        found = decode_all(decoder, samples_of(name, start=key))
        assert found == FIXTURES[name]['md5'][order.index(key):]

    def test_pictures_held_back_at_a_reset_are_not_shown_after_it(self, decoder):
        """The parser holds pictures past the samples fed; a seek drops them."""
        name = 'high_bpyramid'
        for sample in samples_of(name, stop=6):
            md5s(decoder.decode(sample))
        decoder.reset()
        assert decode_all(decoder, samples_of(name)) == FIXTURES[name]['md5']

    def test_decoding_starts_at_a_key_frame_that_is_not_an_idr(self, decoder):
        name = 'open_gop'
        with AnnexBReader(DATA / f'{name}.h264') as stream:
            key = stream.sync_samples[1]
            assert stream.sample(key).data[4] & 0x1F != 5
            order = stream.display_order()
        found = decode_all(decoder, samples_of(name, start=key))
        assert found == FIXTURES[name]['md5'][order.index(key):]

    def test_a_stream_decodes_again_after_a_flush(self, decoder):
        name = 'high_cqm'
        assert decode_all(decoder, samples_of(name)) == FIXTURES[name]['md5']
        assert decode_all(decoder, samples_of(name)) == FIXTURES[name]['md5']


class TestSequences:
    def test_a_stream_of_another_size_follows_in_the_same_decoder(self, decoder):
        """A new sequence gets its own hardware decoder and textures."""
        first, second = 'baseline_cavlc', 'high_cqm'
        assert decode_all(decoder, samples_of(first)) == FIXTURES[first]['md5']
        assert decode_all(decoder, samples_of(second)) == FIXTURES[second]['md5']
        assert decoder.size == tuple(FIXTURES[second]['size'])

    def test_frames_of_the_sequence_before_stay_readable(self, decoder):
        first, second = 'baseline_cavlc', 'high_cqm'
        held = []
        for sample in samples_of(first):
            for frame in decoder.decode(sample):
                if len(held) < decoder.output_slots:
                    held.append(frame)
                else:
                    frame.release()
        decoder.reset()
        after = decode_all(decoder, samples_of(second))
        assert md5s(held) == FIXTURES[first]['md5'][:len(held)]
        assert after == FIXTURES[second]['md5']

    def test_the_same_sequence_keeps_its_textures_across_a_reset(self, decoder):
        name = 'baseline_cavlc'
        decode_all(decoder, samples_of(name))
        session = decoder._session  # noqa: SLF001 - that nothing was rebuilt is the claim
        decoder.reset()
        decode_all(decoder, samples_of(name))
        assert decoder._session is session  # noqa: SLF001 - that nothing was rebuilt is the claim


class TestOutputTextures:
    def test_a_decoder_that_has_lent_every_texture_says_which_are_held(self, decoder):
        held = []
        with pytest.raises(DecoderError, match='held'):
            for sample in samples_of('baseline_cavlc'):
                held += decoder.decode(sample)
        assert len(held) >= decoder.output_slots
        for frame in held:
            frame.release()

    def test_released_frames_free_their_textures(self, decoder):
        for _ in range(3):
            decode_all(decoder, samples_of('baseline_cavlc'))
        made = decoder._session.outputs.made  # noqa: SLF001 - how many textures exist is the claim
        assert len(made) <= 1 + 2

    def test_a_frame_still_held_keeps_its_picture_while_more_are_decoded(self, decoder):
        """Each frame is its own copy, so decoding on does not write over it."""
        samples = samples_of('baseline_cavlc')
        held = []
        found = []
        for sample in samples:
            frames = decoder.decode(sample)
            if not held and frames:
                held.append(frames.pop(0))
            found += md5s(frames)
        found += md5s(decoder.flush())
        expected = FIXTURES['baseline_cavlc']['md5']
        assert md5s(held) == expected[:1]
        assert found == expected[1:]

    def test_decoding_after_close_is_an_error(self):
        made = NVDECDecoder()
        made.close()
        with pytest.raises(DecoderError, match='closed'):
            made.decode(samples_of('baseline_cavlc')[0])

    def test_closing_twice_is_safe_and_releases_what_was_lent(self):
        made = NVDECDecoder()
        held = [frame for sample in samples_of('baseline_cavlc')[:3]
                for frame in made.decode(sample)]
        made.close()
        made.close()
        assert all(frame.released for frame in held)

    def test_bytes_that_are_no_stream_produce_no_picture(self, decoder):
        for junk in (b'', b'\x00\x00\x00\x01', b'not a stream at all' * 50):
            assert decoder.decode(Sample(junk, 0, 0, True, 0)) == []
        assert decoder.flush() == []


class TestClosing:
    def test_the_hardware_decoder_is_destroyed_off_the_calling_thread(self, monkeypatch):
        """The driver's destroy waits on the video engine; the drawing thread does not."""
        made = NVDECDecoder()
        cuvid = made._cuvid  # noqa: SLF001 - the driver call whose thread is the claim
        real = cuvid.cuvidDestroyDecoder
        destroyed_on = []

        def destroy(decoder):
            destroyed_on.append(threading.current_thread())
            return real(decoder)
        monkeypatch.setattr(cuvid, 'cuvidDestroyDecoder', destroy)
        decode_all(made, samples_of('baseline_cavlc'))
        made.close()
        for thread in threading.enumerate():
            if thread.name == 'nvdec-destroy':
                thread.join(timeout=30)
        assert len(destroyed_on) == 1
        assert destroyed_on[0] is not threading.current_thread()
        assert not destroyed_on[0].is_alive()

    def test_the_cuda_context_outlives_the_decoder(self):
        """Making it again for the next movie would cost the caller 100 ms."""
        with NVDECDecoder() as first:
            context = first._context.value  # noqa: SLF001 - that it is the same one is the claim
        with NVDECDecoder() as second:
            assert second._context.value == context  # noqa: SLF001 - that it is the same one is the claim


class TestRefusedStreams:
    def test_a_picture_the_hardware_cannot_decode_is_refused_by_name(self, monkeypatch):
        """The refusal is raised in the parser's callback and reaches the caller."""
        made = NVDECDecoder()
        cuvid = made._cuvid  # noqa: SLF001 - the driver call the refusal is made from
        real = cuvid.cuvidGetDecoderCaps

        def narrow(capabilities):
            status = real(capabilities)
            ctypes.cast(capabilities, ctypes.POINTER(api.CUVIDDECODECAPS)).contents.nMaxWidth = 64
            return status
        monkeypatch.setattr(cuvid, 'cuvidGetDecoderCaps', narrow)
        try:
            with pytest.raises(DecoderError, match='128x80 picture is larger than the 64x'):
                made.decode(samples_of('high_bpyramid')[0])
        finally:
            made.close()


class TestInterlaced:
    def test_woven_fields_are_the_frames_as_they_were_coded(self):
        name = 'interlaced_mbaff'
        with NVDECDecoder(deinterlace='weave') as made:
            assert decode_all(made, samples_of(name)) == INTERLACED[name]['md5']

    @pytest.mark.parametrize('mode', ['adaptive', 'bob'])
    def test_a_deinterlaced_stream_shows_a_frame_a_sample(self, mode):
        name = 'interlaced_mbaff'
        samples = samples_of(name)
        shown = []

        def note(frames):
            for frame in frames:
                shown.append(frame.timestamp)
                assert frame.size == tuple(INTERLACED[name]['size'])
                frame.release()

        with NVDECDecoder(deinterlace=mode) as made:
            for sample in samples:
                note(made.decode(sample))
            note(made.flush())
        assert shown == sorted(sample.timestamp for sample in samples)


class TestOpenGLState:
    """Decoding happens inside a renderer's frame, and leaves its state alone."""

    def test_bindings_and_pixel_storage_are_as_the_renderer_left_them(self, decoder):
        bound = create_output_texture(16, 16)
        buffer = int(glGenBuffers(1))
        try:
            glBindBuffer(GL_ARRAY_BUFFER, buffer)
            glBufferData(GL_ARRAY_BUFFER, 1 << 20, None, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer)
            glBindTexture(GL_TEXTURE_2D, bound)
            for name, value in PACKING.items():
                glPixelStorei(name, value)

            found = []
            for sample in samples_of('high_bpyramid'):
                frames = decoder.decode(sample)
                state = (int(glGetIntegerv(GL_TEXTURE_BINDING_2D)),
                         int(glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING)),
                         {name: int(glGetIntegerv(name)) for name in PACKING})
                assert state == (bound, buffer, PACKING)
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
                found += md5s(frames)
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer)
                glBindTexture(GL_TEXTURE_2D, bound)
                for name, value in PACKING.items():
                    glPixelStorei(name, value)
        finally:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            glBindTexture(GL_TEXTURE_2D, 0)
            for name in PACKING:
                glPixelStorei(name, 4 if name == GL_UNPACK_ALIGNMENT else 0)
            glDeleteBuffers(1, [buffer])
            glDeleteTextures([bound])
        # The pictures decoded under that state are the right ones.
        assert found == FIXTURES['high_bpyramid']['md5'][:len(found)]
        assert found


class TestRegistry:
    def test_both_backends_are_discovered_with_the_gpu_copy_first(self):
        names = [backend.name for backend in decoders()]
        assert names.index('nvdec') < names.index('nvdec-copy')

    @pytest.mark.parametrize('name, zero_copy', [('nvdec', True), ('nvdec-copy', False)])
    def test_a_decoder_reports_how_its_pictures_travel(self, name, zero_copy):
        with open_decoder('h264', backend=name) as made:
            assert isinstance(made, NVDECDecoder)
            assert made.zero_copy is zero_copy

    def test_the_decoder_names_the_gpu_it_runs_on(self):
        with NVDECDecoder() as made:
            assert made.device
            assert made.device in repr(made)


def encode(frames, size, upload_texture, **options):
    """Encode RGBA `frames` with NVENC; return the encoder's packets."""
    encoder = NVENCEncoder(*size, fps=30, bitrate=8_000_000, **options)
    source = upload_texture(frames[0])
    handles = [encoder.new_input() for _ in range(encoder.input_slots)]
    packets = []
    try:
        for index, frame in enumerate(frames):
            handle = handles[index % len(handles)]
            upload_texture.update(source, frame)
            blit(handle, source, size)
            packets += encoder.encode(handle, timestamp=index * TICK, duration=TICK)
        return packets + encoder.flush()
    finally:
        for handle in handles:
            handle.close()
        encoder.close()


def read_rgb(texture, size):
    """An output texture's RGB, first row at the top."""
    width, height = size
    glBindTexture(GL_TEXTURE_2D, texture)
    data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
    glBindTexture(GL_TEXTURE_2D, 0)
    picture = np.frombuffer(data, np.uint8).reshape(height, width, 4)
    # The output is the right way up for OpenGL: row 0 is the bottom.
    return picture[::-1, :, :3].astype(int)


@pytest.mark.usefixtures('nvenc_available')
class TestRoundTrip:
    """NVENC's stream, decoded and converted back to RGB."""

    SIZE = (320, 240)

    @pytest.mark.parametrize('bframes', [0, 2])
    def test_what_was_encoded_comes_back(self, decoder, upload_texture, bframes):
        frames = [gradient_frame(*self.SIZE, phase=index) for index in range(12)]
        packets = encode(frames, self.SIZE, upload_texture, gop=10, bframes=bframes)
        converter = NV12Converter()
        target = create_output_texture(*self.SIZE)
        errors = {}

        def compare(frame):
            converter.convert(frame, target)
            frame.release()
            difference = read_rgb(target, self.SIZE) - frames[frame.timestamp // TICK][..., :3]
            errors[frame.timestamp // TICK] = float(np.abs(difference).mean())

        try:
            for index, packet in enumerate(packets):
                sample = Sample(packet.data, packet.timestamp, packet.duration,
                                packet.keyframe, index)
                for frame in decoder.decode(sample):
                    compare(frame)
            for frame in decoder.flush():
                compare(frame)
        finally:
            converter.close()
            glDeleteTextures([target])
        assert sorted(errors) == list(range(len(frames)))
        assert max(errors.values()) < 4.0, errors


@pytest.mark.usefixtures('nvenc_available')
class TestPlayingOnTheGPU:
    @pytest.mark.parametrize('backend', ['nvdec', 'nvdec-copy'])
    def test_a_recording_plays_back_frame_by_frame(self, tmp_path, upload_texture, backend):
        size = (320, 240)
        frames = [gradient_frame(*size, phase=index * 3) for index in range(20)]
        path = tmp_path / 'recorded.mp4'
        encoder = NVENCEncoder(*size, fps=30, bitrate=8_000_000, gop=8)
        texture = upload_texture(frames[0])
        try:
            handle = encoder.register(texture)
            with MP4Writer(path, encoder) as writer:
                for index, frame in enumerate(frames):
                    upload_texture.update(texture, frame)
                    writer.write(encoder.encode(handle, index * TICK, TICK))
                writer.write(encoder.flush())
        finally:
            encoder.close()

        with Movie.open(path) as movie:
            playing = movie.player(backend=backend)
            try:
                for index in [*range(20), 3, 15, 0]:
                    shown = read_rgb(playing.frame_at(index / 30), size)
                    difference = np.abs(shown - frames[index][..., :3]).mean()
                    assert difference < 4.0, (index, difference)
            finally:
                playing.close()
