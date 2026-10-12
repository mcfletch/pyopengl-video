"""NVENC's codecs: what a session is asked, and what each codec's configuration is told.

No driver. The session query runs against a stand-in for the function table,
and the configuration is an ordinary structure filled in by a function.
"""
import ctypes

import pytest

from pyopengl_video import nvenc
from pyopengl_video.encoder import EncoderError
from pyopengl_video.nvenc import api
from pyopengl_video.nvenc import encoder as nvenc_encoder
from pyopengl_video.nvenc.encoder import configure_codec, session_codecs

UNKNOWN_CODEC = api.GUID(1, 2, 3, (ctypes.c_uint8 * 8)(*range(8)))


class FakeLibrary:
    """The three calls a codec query makes, answering for a GPU with `offered`."""

    def __init__(self, offered, sizes=None):
        self.offered = offered
        self.sizes = sizes or {}
        self.destroyed = []

    def struct_version(self, revision, big=False):  # noqa: ARG002 - the interface's signature
        return revision

    def check(self, status, call, session=None):  # noqa: ARG002 - the interface's signature
        assert status == 0, call
        return status

    def nvEncGetEncodeGUIDCount(self, session, count):  # noqa: ARG002 - the interface's signature
        ctypes.cast(count, ctypes.POINTER(ctypes.c_uint32))[0] = len(self.offered)
        return 0

    def nvEncGetEncodeGUIDs(self, session, guids, size, count):  # noqa: ARG002 - the interface's signature
        assert size == len(self.offered)
        for index, guid in enumerate(self.offered):
            guids[index] = guid
        ctypes.cast(count, ctypes.POINTER(ctypes.c_uint32))[0] = len(self.offered)
        return 0

    def nvEncGetEncodeCaps(self, session, codec, query, value):  # noqa: ARG002 - the interface's signature
        asked = ctypes.cast(query, ctypes.POINTER(api.NV_ENC_CAPS_PARAM))[0].capsToQuery
        width, height = next(size for guid, size in self.sizes.items() if guid == codec)
        ctypes.cast(value, ctypes.POINTER(ctypes.c_int))[0] = (
            width if asked == api.NV_ENC_CAPS_WIDTH_MAX else height)
        return 0

    def nvEncDestroyEncoder(self, session):
        self.destroyed.append(session)
        return 0


SIZES = {api.NV_ENC_CODEC_H264_GUID: (4096, 4096),
         api.NV_ENC_CODEC_HEVC_GUID: (8192, 8192),
         api.NV_ENC_CODEC_AV1_GUID: (8192, 8192)}


class TestWhatASessionOffers:
    def test_each_codec_the_gpu_lists_comes_back_with_its_largest_frame(self):
        library = FakeLibrary([api.NV_ENC_CODEC_H264_GUID, api.NV_ENC_CODEC_HEVC_GUID], SIZES)
        assert session_codecs(library, ctypes.c_void_p(1)) == {
            'h264': (4096, 4096), 'hevc': (8192, 8192)}

    def test_a_codec_this_package_has_no_name_for_is_left_out(self):
        library = FakeLibrary([UNKNOWN_CODEC, api.NV_ENC_CODEC_AV1_GUID], SIZES)
        assert session_codecs(library, ctypes.c_void_p(1)) == {'av1': (8192, 8192)}

    def test_a_gpu_that_lists_nothing_offers_nothing(self):
        assert session_codecs(FakeLibrary([]), ctypes.c_void_p(1)) == {}


class TestDiscovery:
    def test_the_record_lists_every_codec_the_backend_drives(self):
        assert nvenc.BACKEND.codecs == frozenset({'h264', 'hevc', 'av1'})
        assert nvenc.BACKEND.largest('h264') == (4096, 4096)
        assert nvenc.BACKEND.largest('hevc') == (8192, 8192)

    def test_with_no_session_to_ask_every_codec_is_assumed(self, monkeypatch):
        def refuse(library):  # noqa: ARG001 - the signature of the function replaced
            raise api.NVENCError(2, 'nvEncOpenEncodeSessionEx', 'no context')

        monkeypatch.setattr(api.NvEncodeAPI, 'load', classmethod(lambda _cls: FakeLibrary([])))
        monkeypatch.setattr(nvenc_encoder, 'open_session', refuse)
        assert nvenc.session_codecs() == nvenc.ASSUMED_SIZES

    def test_a_session_is_asked_and_then_given_back(self, monkeypatch):
        library = FakeLibrary([api.NV_ENC_CODEC_H264_GUID], SIZES)
        session = ctypes.c_void_p(7)
        monkeypatch.setattr(api.NvEncodeAPI, 'load', classmethod(lambda _cls: library))
        monkeypatch.setattr(nvenc_encoder, 'open_session', lambda _library: session)
        assert nvenc.session_codecs() == {'h264': (4096, 4096)}
        assert library.destroyed == [session]

    def test_a_session_that_will_not_answer_is_given_back_too(self, monkeypatch):
        library = FakeLibrary([api.NV_ENC_CODEC_H264_GUID])       # no sizes: the query fails
        session = ctypes.c_void_p(7)
        monkeypatch.setattr(api.NvEncodeAPI, 'load', classmethod(lambda _cls: library))
        monkeypatch.setattr(nvenc_encoder, 'open_session', lambda _library: session)
        assert nvenc.session_codecs() == nvenc.ASSUMED_SIZES
        assert library.destroyed == [session]


class TestConfiguration:
    def configured(self, codec, frame_rate=(30000, 1001)):
        config = api.NV_ENC_CONFIG()
        configure_codec(config, codec, gop=48, frame_rate=frame_rate)
        return config

    def test_h264_is_high_profile_with_its_sets_repeated(self):
        config = self.configured('h264')
        h264 = config.encodeCodecConfig.h264Config
        assert config.profileGUID == api.NV_ENC_H264_PROFILE_HIGH_GUID
        assert (h264.idrPeriod, h264.chromaFormatIDC, h264.repeatSPSPPS) == (48, 1, 1)

    def test_h264_counts_time_in_fields_and_hevc_in_frames(self):
        h264 = self.configured('h264').encodeCodecConfig.h264Config.h264VUIParameters
        hevc = self.configured('hevc').encodeCodecConfig.hevcConfig.hevcVUIParameters
        assert (h264.numUnitInTicks, h264.timeScale) == (1001, 60000)
        assert (hevc.numUnitInTicks, hevc.timeScale) == (1001, 30000)

    def test_hevc_keeps_the_presets_profile_and_repeats_its_sets(self):
        config = self.configured('hevc')
        hevc = config.encodeCodecConfig.hevcConfig
        assert config.profileGUID == api.GUID()
        assert (hevc.idrPeriod, hevc.chromaFormatIDC, hevc.repeatSPSPPS) == (48, 1, 1)

    @pytest.mark.parametrize('codec', ['h264', 'hevc'])
    def test_a_vui_declares_limited_range_bt709(self, codec):
        arm = self.configured(codec).encodeCodecConfig
        vui = arm.h264Config.h264VUIParameters if codec == 'h264' else (
            arm.hevcConfig.hevcVUIParameters)
        assert vui.videoSignalTypePresentFlag == 1
        assert vui.videoFullRangeFlag == 0
        assert vui.colourDescriptionPresentFlag == 1
        assert (vui.colourPrimaries, vui.transferCharacteristics, vui.colourMatrix) == (1, 1, 1)

    def test_av1_repeats_its_sequence_header_and_declares_limited_range_bt709(self):
        av1 = self.configured('av1').encodeCodecConfig.av1Config
        assert (av1.idrPeriod, av1.chromaFormatIDC, av1.repeatSeqHdr) == (48, 1, 1)
        assert (av1.colorPrimaries, av1.transferCharacteristics,
                av1.matrixCoefficients, av1.colorRange) == (1, 1, 1, 0)
        assert av1.outputAnnexBFormat == 0

    def test_a_codec_it_does_not_encode_is_refused_with_those_it_does(self):
        with pytest.raises(EncoderError, match='av1, h264, hevc'):
            configure_codec(api.NV_ENC_CONFIG(), 'vp9', gop=30, frame_rate=(30, 1))
