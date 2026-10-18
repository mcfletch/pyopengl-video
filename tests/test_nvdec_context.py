"""The NVDEC decoder and the OpenGL context it was opened in, on real hardware.

A texture name means something only in the share group that made it. A
decoder that made its plane textures in one context and deleted them with a
context outside its share group current would delete that context's textures
of the same names, and one that
decoded with no context current handed out frames whose textures were never
made (N10 of ``plans/SECURITY-REVIEW.md``). Also here: what the decoder does
with a picture the parser shows that it never decoded (N6), with the pictures
of a sequence it refused (N7), and with pixel storage a renderer left set
(N14).
"""
import hashlib
import json
from pathlib import Path

import pytest
from OpenGL.GL import GL_UNPACK_SWAP_BYTES, glGetIntegerv, glIsTexture, glPixelStorei

from pyopengl_video import DecoderError
from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.ivf import IVFReader
from pyopengl_video.nvdec import api
from pyopengl_video.nvdec.decoder import NVDECDecoder
from tests.conftest import decode_all, frame_bytes, md5s
from tests.other_context import contexts  # noqa: F401 - a fixture, requested by name

DATA = Path(__file__).parent / 'data' / 'decode'
H264 = json.loads((DATA / 'h264.json').read_text())['streams']
HEVC = json.loads((DATA / 'hevc.json').read_text())['large']
AV1 = json.loads((DATA / 'av1.json').read_text())['large']

pytestmark = pytest.mark.usefixtures('nvdec_available')


def samples_of(name, extension='h264'):
    with AnnexBReader(DATA / f'{name}.{extension}') as stream:
        return [stream.sample(index, parameter_sets=index == 0)
                for index in range(len(stream))]


@pytest.fixture(params=[True, False], ids=['on-the-gpu', 'through-host-memory'])
def interop(request):
    """Each way a picture reaches its textures."""
    return request.param


class TestDecodingInAnotherContext:
    """N10: the pictures are copied into textures of the decoder's own context."""

    @pytest.mark.parametrize('other', ['none', 'another'])
    def test_decode_and_flush_are_refused_and_work_again_in_the_right_one(
            self, contexts, interop, other):  # noqa: F811 - the fixture imported above
        samples = samples_of('high_bpyramid')
        decoder = NVDECDecoder(interop=interop)
        try:
            found = []
            for sample in samples[:8]:
                found += md5s(decoder.decode(sample))
            contexts.use(None if other == 'none' else contexts.second)
            with pytest.raises(DecoderError, match='shares objects with it is current on this thread'):
                decoder.decode(samples[8])
            with pytest.raises(DecoderError, match='shares objects with it is current on this thread'):
                decoder.flush()
            contexts.use(contexts.first)
            for sample in samples[8:]:
                found += md5s(decoder.decode(sample))
            found += md5s(decoder.flush())
            assert found == H264['high_bpyramid']['md5']
        finally:
            contexts.use(contexts.first)
            decoder.close()


class TestAContextThatShares:
    """The plane textures are the share group's, in whichever of its contexts is current."""

    def test_decoding_carries_on_in_it(self, contexts, interop):  # noqa: F811 - the fixture imported above
        samples = samples_of('high_bpyramid')
        decoder = NVDECDecoder(interop=interop)
        try:
            found = []
            for sample in samples[:8]:
                found += md5s(decoder.decode(sample))
            contexts.use(contexts.sharing)
            for sample in samples[8:]:
                found += md5s(decoder.decode(sample))
            found += md5s(decoder.flush())
            assert found == H264['high_bpyramid']['md5']
        finally:
            contexts.use(contexts.first)
            decoder.close()

    def test_closing_in_it_deletes_the_textures(self, contexts, interop):  # noqa: F811 - the fixture imported above
        decoder = NVDECDecoder(interop=interop)
        frames = [frame for sample in samples_of('high_bpyramid')[:10]
                  for frame in decoder.decode(sample)]
        names = sorted({name for frame in frames for name in frame.planes})
        assert names and all(glIsTexture(name) for name in names)
        contexts.use(contexts.sharing)
        decoder.close()
        assert not any(glIsTexture(name) for name in names)
        contexts.use(contexts.first)
        assert not any(glIsTexture(name) for name in names)
        decoder.close()


class TestClosingInAnotherContext:
    """N10: everything but the textures is released, and the textures are left alone."""

    def test_the_other_contexts_textures_survive_and_a_second_close_finishes(
            self, contexts, interop):  # noqa: F811 - the fixture imported above
        decoder = NVDECDecoder(interop=interop)
        samples = samples_of('high_bpyramid')
        frames = [frame for sample in samples[:10] for frame in decoder.decode(sample)]
        names = sorted({name for frame in frames for name in frame.planes})
        assert names and all(glIsTexture(name) for name in names)
        session = decoder._session  # noqa: SLF001 - the hardware decoder, which the first close must destroy

        contexts.use(contexts.second)
        for name in names:
            contexts.texture_named(name)
        with pytest.raises(DecoderError, match='another OpenGL context is current'):
            decoder.close()
        # The other context's textures of those names are untouched, and the
        # parser and the hardware decoder are gone all the same.
        assert all(glIsTexture(name) for name in names)
        assert not session.decoder
        assert not decoder._parser  # noqa: SLF001 - the parser, which the first close must destroy
        assert all(frame.released for frame in frames)
        with pytest.raises(DecoderError, match='closed'):
            decoder.decode(samples[10])

        contexts.use(contexts.first)
        assert all(glIsTexture(name) for name in names)
        decoder.close()
        assert not any(glIsTexture(name) for name in names)
        decoder.close()

    def test_no_context_at_all_keeps_the_textures_for_a_later_close(
            self, contexts, interop):  # noqa: F811 - the fixture imported above
        decoder = NVDECDecoder(interop=interop)
        frames = [frame for sample in samples_of('high_bpyramid')[:10]
                  for frame in decoder.decode(sample)]
        names = sorted({name for frame in frames for name in frame.planes})
        contexts.use(None)
        with pytest.raises(DecoderError, match='no OpenGL context is current'):
            decoder.close()
        contexts.use(contexts.first)
        assert all(glIsTexture(name) for name in names)
        decoder.close()
        assert not any(glIsTexture(name) for name in names)

    def test_a_frame_released_in_another_context_deletes_nothing_there(
            self, contexts, interop):  # noqa: F811 - the fixture imported above
        """A sequence that was replaced keeps its textures until its frames come back."""
        decoder = NVDECDecoder(interop=interop)
        try:
            held = [frame for sample in samples_of('baseline_cavlc')[:6]
                    for frame in decoder.decode(sample)]
            names = sorted({name for frame in held for name in frame.planes})
            assert names
            decode_all(decoder, samples_of('high_cqm'))
            contexts.use(contexts.second)
            for name in names:
                contexts.texture_named(name)
            for frame in held:
                frame.release()
            assert all(glIsTexture(name) for name in names)
            contexts.use(contexts.first)
            assert all(glIsTexture(name) for name in names)
        finally:
            contexts.use(contexts.first)
            decoder.close()
        assert not any(glIsTexture(name) for name in names)


class TestPixelStorage:
    def test_a_ten_bit_picture_is_uploaded_whatever_byte_order_was_left_set(self):
        """N14: with GL_UNPACK_SWAP_BYTES set, each 16-bit sample would arrive reversed."""
        name = 'hevc_large_main10'
        decoder = NVDECDecoder('hevc', interop=False)
        glPixelStorei(GL_UNPACK_SWAP_BYTES, 1)
        try:
            found = []
            for sample in samples_of(name, 'h265'):
                frames = decoder.decode(sample)
                assert int(glGetIntegerv(GL_UNPACK_SWAP_BYTES)) == 1
                found += md5s(frames)
            found += md5s(decoder.flush())
        finally:
            glPixelStorei(GL_UNPACK_SWAP_BYTES, 0)
            decoder.close()
        assert found == HEVC[name]['md5']


class TestPicturesTheDecoderNeverDecoded:
    """N6: the parser shows what it held for a decoder that has been replaced."""

    @pytest.fixture
    def decoder(self, interop):
        made = NVDECDecoder(interop=interop)
        yield made
        made.close()

    def test_one_the_session_never_decoded_is_not_shown(self, decoder, caplog):
        samples = samples_of('high_bpyramid')
        for sample in samples[:2]:
            md5s(decoder.decode(sample))
        session = decoder._session  # noqa: SLF001 - what the parser's display callback copies from
        never = next(index for index in range(session.sequence.surfaces)
                     if index not in session.decoded)
        shown = api.CUVIDPARSERDISPINFO(picture_index=never, progressive_frame=1)
        with caplog.at_level('WARNING'):
            decoder._show(shown, samples[1].timestamp)  # noqa: SLF001 - the display callback's work, with a picture the parser would show
        assert decoder._ready == []  # noqa: SLF001 - the frames the callback produced
        assert 'has not decoded' in caplog.text

    @pytest.mark.parametrize('index', [-1, 1000])
    def test_one_outside_the_decoders_surfaces_is_never_mapped(self, decoder, index):
        for sample in samples_of('high_bpyramid')[:6]:
            md5s(decoder.decode(sample))
        session = decoder._session  # noqa: SLF001 - as above
        pair = session.outputs.take()
        try:
            with pytest.raises(DecoderError, match=f'shows picture {index} of a decoder made'):
                session.copy_out(api.CUVIDPARSERDISPINFO(picture_index=index), pair)
        finally:
            session.outputs.give(pair)

    def test_the_stream_decodes_on_after_a_picture_was_left_out(self, decoder):
        name = 'high_bpyramid'
        samples = samples_of(name)
        found = []
        for sample in samples[:6]:
            found += md5s(decoder.decode(sample))
        decoder._show(api.CUVIDPARSERDISPINFO(picture_index=1000), 10**9)  # noqa: SLF001 - as above
        for sample in samples[6:]:
            found += md5s(decoder.decode(sample))
        assert found + md5s(decoder.flush()) == H264[name]['md5']


class TestARefusedSequence:
    """N7: its pictures are not handed to the decoder of the sequence before."""

    def test_the_session_before_is_retired_when_a_sequence_is_refused(self, interop):
        decoder = NVDECDecoder('hevc', interop=interop)
        try:
            first = samples_of('hevc_large_bpyramid', 'h265')
            small = samples_of('hevc_bpyramid', 'h265')
            decoded = decode_all(decoder, first[:1])
            assert decoded == HEVC['hevc_large_bpyramid']['md5'][:len(decoded)]
            before = decoder._session  # noqa: SLF001 - the hardware decoder of the first sequence
            with pytest.raises(DecoderError, match='smaller than'):
                decoder.decode(small[0])
            assert decoder._session is None  # noqa: SLF001 - no decoder for a refused sequence
            assert not before.decoder
            for sample in small[1:6]:
                try:
                    assert decoder.decode(sample) == []
                except DecoderError as refused:
                    assert 'before it described a sequence' in str(refused)
            assert not before.decoded
            decoder.reset()
            assert decode_all(decoder, first) == HEVC['hevc_large_bpyramid']['md5']
        finally:
            decoder.close()


class TestEveryOutputTextureLent:
    """A picture that was decoded and could not be shown leaves the references whole."""

    def test_an_av1_stream_decodes_on_once_frames_are_released(self):
        """The gate does not wait for a key frame: nothing the stream predicts from was lost."""
        name = 'av1_large_altref'
        with IVFReader(DATA / f'{name}.ivf') as stream:
            samples = [stream.sample(index) for index in range(len(stream))]
        decoder = NVDECDecoder('av1', output_slots=1)
        held, found, refused = [], [], 0
        try:
            for index, sample in enumerate(samples):
                try:
                    frames = decoder.decode(sample)
                except DecoderError as error:
                    assert 'output textures is in use' in str(error)
                    refused += 1
                    frames = []
                # Read now and kept: a frame held is a texture pair lent.
                found += [hashlib.md5(frame_bytes(frame)).hexdigest() for frame in frames]
                held += frames
                if refused and held or index % 14 == 13:
                    for frame in held:
                        frame.release()
                    held.clear()
            found += md5s(decoder.flush())
        finally:
            for frame in held:
                frame.release()
            decoder.close()
        expected = AV1[name]['md5']
        assert refused, 'no picture met every texture lent'
        # Every picture but the ones that could not be shown, each the right one.
        assert len(found) == len(expected) - refused
        assert [picture for picture in expected if picture in found] == found
