"""The VA-API decoder on real hardware, into real OpenGL textures.

H.264 decoding is exact, so every picture of the x264 fixtures must come out
of the GPU with the bytes ffmpeg's software decoder produced, read back from
the textures the decoder hands out. Our own encoder's streams are decoded too,
which is the round trip the encode plan asked for.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from OpenGL.GL import (
    GL_RGBA,
    GL_TEXTURE_2D,
    GL_UNSIGNED_BYTE,
    glBindTexture,
    glDeleteTextures,
    glGetTexImage,
)

from pyopengl_video import DecoderError, decoders, open_decoder
from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.convert import NV12Converter, create_output_texture
from pyopengl_video.decoder import Sample
from pyopengl_video.mp4 import MP4Reader
from pyopengl_video.vaapi.decoder import VAAPIDecoder
from pyopengl_video.vaapi.encoder import VAAPIEncoder
from pyopengl_video.vaapi.h264 import BitWriter, annexb, nal_unit
from tests.conftest import blit, decode_all, gradient_frame, md5s
from tests.h264_syntax import sps_bytes, write_pps_head

DATA = Path(__file__).parent / 'data' / 'decode'
FIXTURES = json.loads((DATA / 'fixtures.json').read_text())['streams']


@pytest.fixture
def decoder(vaapi_decoder_available):  # noqa: ARG001 - requested to skip where there is no VA-API decoder
    made = VAAPIDecoder()
    yield made
    made.close()


class TestConformance:
    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_every_picture_matches_the_reference_decode(self, decoder, name):
        with AnnexBReader(DATA / f'{name}.h264') as stream:
            samples = [stream.sample(index) for index in range(len(stream))]
        assert decode_all(decoder, samples) == FIXTURES[name]['md5']

    def test_an_mp4_decodes_from_its_sample_entry(self, vaapi_decoder_available):  # noqa: ARG002 - requested to skip where there is no VA-API decoder
        with MP4Reader(DATA / 'high_bpyramid.mp4') as movie:
            track = movie.video
            with open_decoder('h264', headers=track.parameter_sets(),
                              backend='vaapi') as made:
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
                    shown.append(frame.timestamp)
                    frame.release()
            for frame in decoder.flush():
                shown.append(frame.timestamp)
                frame.release()
            assert shown == sorted(track.presentation_time(i) for i in range(len(track)))


class TestSeveralMovies:
    def test_four_decoders_in_one_context_each_decode_their_own(
            self, vaapi_decoder_available):  # noqa: ARG002 - requested to skip where there is no VA-API decoder
        """A game with screens on its walls: one session per movie, interleaved."""
        names = sorted(FIXTURES)
        streams = [AnnexBReader(DATA / f'{name}.h264') for name in names]
        decoders = [VAAPIDecoder() for _ in names]
        found = {name: [] for name in names}
        try:
            for index in range(max(len(stream) for stream in streams)):
                for name, stream, made in zip(names, streams, decoders, strict=True):
                    if index < len(stream):
                        found[name] += md5s(made.decode(stream.sample(index)))
            for name, made in zip(names, decoders, strict=True):
                found[name] += md5s(made.flush())
        finally:
            for made in decoders:
                made.close()
            for stream in streams:
                stream.close()
        assert found == {name: FIXTURES[name]['md5'] for name in names}


class TestSeeking:
    def test_decoding_resumes_from_a_key_frame_after_a_reset(self, decoder):
        name = 'high_bpyramid'
        with AnnexBReader(DATA / f'{name}.h264') as stream:
            decode_all(decoder, [stream.sample(index) for index in range(10)])
            decoder.reset()
            key = stream.sync_samples[1]
            samples = [stream.sample(key, parameter_sets=True)] + [
                stream.sample(index) for index in range(key + 1, len(stream))]
            order = stream.display_order()
        found = decode_all(decoder, samples)
        shown_from = order.index(key)
        assert found == FIXTURES[name]['md5'][shown_from:]

    def test_an_open_gop_seek_drops_the_pictures_it_cannot_decode(self, decoder):
        """Leading pictures before a non-IDR key frame refer to the group before."""
        name = 'open_gop'
        with AnnexBReader(DATA / f'{name}.h264') as stream:
            key = next(index for index in stream.sync_samples[1:])
            assert not stream.sample(key).data[4] & 0x1F == 5
            samples = [stream.sample(key, parameter_sets=True)] + [
                stream.sample(index) for index in range(key + 1, len(stream))]
            order = stream.display_order()
        decoder.reset()
        found = decode_all(decoder, samples)
        assert found == FIXTURES[name]['md5'][order.index(key):]


class TestSurfaces:
    def test_a_decoder_that_has_lent_every_surface_says_which_are_held(self, decoder):
        with AnnexBReader(DATA / 'baseline_cavlc.h264') as stream:
            held = []
            with pytest.raises(DecoderError, match='held'):
                for index in range(len(stream)):
                    held += decoder.decode(stream.sample(index))
            assert len(held) >= decoder.output_slots
        for frame in held:
            frame.release()

    def test_released_frames_free_their_surfaces(self, decoder):
        with AnnexBReader(DATA / 'baseline_cavlc.h264') as stream:
            for _ in range(3):
                decoder.reset()
                decode_all(decoder, [stream.sample(index) for index in range(len(stream))])

    def test_decoding_after_close_is_an_error(self, vaapi_decoder_available):  # noqa: ARG002 - requested to skip where there is no VA-API decoder
        made = VAAPIDecoder()
        made.close()
        with AnnexBReader(DATA / 'baseline_cavlc.h264') as stream, \
                pytest.raises(DecoderError, match='closed'):
            made.decode(stream.sample(0))

    def test_a_stream_that_is_not_decoded_here_is_refused_by_name(self, decoder):
        """Refused when a picture uses the sequence, not when the set arrives."""
        sps = nal_unit(sps_bytes(frame_mbs_only=0), nal_unit_type=7)
        writer = BitWriter()
        write_pps_head(writer)
        pps = nal_unit(writer.rbsp(), nal_unit_type=8)
        writer = BitWriter()
        writer.ue(0)                            # first_mb_in_slice
        writer.ue(7)                            # I
        writer.ue(0)                            # pic_parameter_set_id
        writer.u(0, 4)                          # frame_num
        writer.flag(0)                          # field_pic_flag: an MBAFF frame
        writer.ue(0)                            # idr_pic_id
        writer.u(0, 6)                          # pic_order_cnt_lsb
        writer.flag(0)
        writer.flag(0)
        writer.se(0)
        writer.ue(1)
        idr = nal_unit(writer.rbsp(), nal_unit_type=5)
        with pytest.raises(DecoderError, match='interlaced'):
            decoder.decode(Sample(annexb(sps, pps, idr), 0, 0, True, 0))


class TestRegistry:
    @pytest.mark.usefixtures('vaapi_decoder_available')
    def test_the_backend_is_discovered(self):
        assert 'vaapi' in [backend.name for backend in decoders()]

    @pytest.mark.usefixtures('vaapi_decoder_available')
    def test_it_reports_zero_copy(self):
        [found] = [backend for backend in decoders() if backend.name == 'vaapi']
        assert found.zero_copy


@pytest.mark.usefixtures('vaapi_available')
class TestRoundTrip:
    """Our own encoder's stream, decoded and converted back to RGB."""

    SIZE = (320, 240)

    def encode(self, frames, upload_texture):
        encoder = VAAPIEncoder(*self.SIZE, fps=30, bitrate=8_000_000, gop=10)
        handle = encoder.new_input()
        source = upload_texture(frames[0])
        packets = []
        try:
            for index, frame in enumerate(frames):
                upload_texture.update(source, frame)
                blit(handle, source, self.SIZE)
                packets += encoder.encode(handle, timestamp=index * 3000, duration=3000)
            packets += encoder.flush()
        finally:
            handle.close()
            encoder.close()
        return packets

    def test_what_was_encoded_comes_back(self, decoder, upload_texture):
        width, height = self.SIZE
        frames = [gradient_frame(width, height, phase=index) for index in range(12)]
        packets = self.encode(frames, upload_texture)
        converter = NV12Converter()
        target = create_output_texture(width, height)
        errors = {}

        def compare(frame):
            converter.convert(frame, target)
            frame.release()
            glBindTexture(GL_TEXTURE_2D, target)
            data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
            glBindTexture(GL_TEXTURE_2D, 0)
            picture = np.frombuffer(data, np.uint8).reshape(height, width, 4)
            original = frames[frame.timestamp // 3000]
            # The output is the right way up for OpenGL: row 0 is the bottom.
            difference = (picture[::-1, :, :3].astype(int)
                          - original[..., :3].astype(int))
            errors[frame.timestamp // 3000] = float(np.abs(difference).mean())

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
