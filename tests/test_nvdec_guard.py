"""What a picture is held to before the hardware decoder is handed it, with no GPU.

``cuvidDecodePicture`` copies as many bytes as a picture states and predicts
from the pictures it names, checking neither. The cases here build pictures
as NVIDIA's parser was seen to (N1 and N2 of ``plans/SECURITY-REVIEW.md``): a
length of 0xFFFFFFF4, a reference index of 0xFF, a reference to a picture that
was never decoded.
"""
import pytest

from pyopengl_video.nvdec import api, guard

SURFACES = 9

#: An AV1 inter frame and switch frame, a key frame and an intra-only frame.
AV1_INTER, AV1_SWITCH, AV1_KEY, AV1_INTRA_ONLY = 1, 3, 0, 2


def picture(index=0, length=100):
    return api.CUVIDPICPARAMS(CurrPicIdx=index, nBitstreamDataLen=length)


def av1(frame_type, references=(0,) * 7, index=0, reconstructed=None):
    made = picture(index)
    parameters = made.CodecSpecific.av1
    parameters.frame_type = frame_type
    parameters.decodePicIdx = index if reconstructed is None else reconstructed
    for entry, reference in zip(parameters.ref_frame, references, strict=True):
        entry.index = reference
    return made


def refusal(made, codec=api.cudaVideoCodec_AV1, decoded=(), fed=1000):
    return guard.refusal(made, codec=codec, surfaces=SURFACES, decoded=set(decoded), fed=fed)


class TestBitstreamLength:
    def test_a_picture_within_the_bytes_fed_is_taken(self):
        assert refusal(picture(length=1000), fed=1000) == ''

    @pytest.mark.parametrize('length', [1001, 0xFFFFFFF4, 0xFFFFFFFF])
    def test_a_picture_longer_than_the_bytes_fed_is_refused(self, length):
        assert refusal(picture(length=length), fed=1000) == (
            f'the parser states {length} bytes of bitstream for a picture, and it '
            'has been handed 1000')


class TestTheTarget:
    @pytest.mark.parametrize('index', [-1, SURFACES, 255, 2**31 - 1])
    def test_a_picture_the_decoder_was_not_made_with_is_refused(self, index):
        assert f'asks for picture {index} of a decoder made with {SURFACES}' in refusal(
            picture(index), codec=api.cudaVideoCodec_H264)

    def test_an_av1_frame_with_film_grain_has_two(self):
        made = av1(AV1_KEY, index=3, reconstructed=4)
        assert guard.targets(api.cudaVideoCodec_AV1, made) == [3, 4]
        assert guard.targets(api.cudaVideoCodec_AV1, av1(AV1_KEY, index=3)) == [3]
        assert guard.targets(api.cudaVideoCodec_H264, made) == [3]

    def test_the_picture_an_av1_frame_is_reconstructed_into_is_held_to_the_range_too(self):
        assert 'asks for picture 200' in refusal(av1(AV1_KEY, index=3, reconstructed=200))


class TestAV1References:
    @pytest.mark.parametrize('frame_type', [AV1_INTER, AV1_SWITCH])
    def test_a_frame_whose_references_were_decoded_is_taken(self, frame_type):
        assert refusal(av1(frame_type, (0, 1, 2, 0, 1, 2, 0)), decoded={0, 1, 2}) == ''

    def test_a_reference_of_0xff_is_refused(self):
        """What the parser names for a frame joined after its key frame."""
        assert 'predicts from picture 255' in refusal(
            av1(AV1_INTER, (0xFF,) * 7), decoded={0, 1})

    def test_a_reference_that_was_never_decoded_is_refused(self):
        found = refusal(av1(AV1_INTER, (0, 1, 2, 3, 0, 1, 5)), decoded={0, 1, 2})
        assert 'predicts from picture 3, 5, which this decoder has not decoded' in found

    @pytest.mark.parametrize('frame_type', [AV1_KEY, AV1_INTRA_ONLY])
    def test_a_frame_that_predicts_from_nothing_names_no_reference(self, frame_type):
        made = av1(frame_type, (0xFF,) * 7)
        assert guard.references(api.cudaVideoCodec_AV1, made) == []
        assert refusal(made) == ''


class TestTheOtherCodecs:
    def test_a_vp9_inter_frame_names_three(self):
        made = picture()
        vp9 = made.CodecSpecific.vp9
        vp9.frameType, vp9.LastRefIdx, vp9.GoldenRefIdx, vp9.AltRefIdx = 1, 2, 3, 4
        assert guard.references(api.cudaVideoCodec_VP9, made) == [2, 3, 4]
        assert 'predicts from picture 3, 4' in refusal(
            made, codec=api.cudaVideoCodec_VP9, decoded={2})
        assert refusal(made, codec=api.cudaVideoCodec_VP9, decoded={2, 3, 4}) == ''

    def test_a_vp9_key_frame_and_an_intra_only_frame_name_none(self):
        key, intra = picture(), picture()
        key.CodecSpecific.vp9.frameType = 0
        intra.CodecSpecific.vp9.frameType, intra.CodecSpecific.vp9.intraOnly = 1, 1
        for made in (key, intra):
            made.CodecSpecific.vp9.LastRefIdx = 7
            assert guard.references(api.cudaVideoCodec_VP9, made) == []

    def test_an_h264_picture_names_the_frames_of_its_picture_buffer(self):
        made = picture()
        buffer = made.CodecSpecific.h264.dpb
        for entry in buffer:
            entry.PicIdx = -1
        buffer[0].PicIdx, buffer[1].PicIdx = 5, 6
        assert guard.references(api.cudaVideoCodec_H264, made) == [5, 6]
        assert 'predicts from picture 6' in refusal(
            made, codec=api.cudaVideoCodec_H264, decoded={5})
        assert refusal(made, codec=api.cudaVideoCodec_H264, decoded={5, 6}) == ''

    def test_an_hevc_picture_names_its_reference_pictures(self):
        made = picture()
        for slot in range(16):
            made.CodecSpecific.hevc.RefPicIdx[slot] = -1
        made.CodecSpecific.hevc.RefPicIdx[2] = 4
        assert guard.references(api.cudaVideoCodec_HEVC, made) == [4]
        assert 'predicts from picture 4' in refusal(made, codec=api.cudaVideoCodec_HEVC)

    def test_a_codec_this_package_does_not_decode_names_none(self):
        assert guard.references(-1, picture()) == []
