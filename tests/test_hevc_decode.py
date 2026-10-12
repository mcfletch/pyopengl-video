"""HEVC decoding on real hardware, into real OpenGL textures.

HEVC decoding is exact, so every picture of the libx265 fixtures must come out
of the GPU with the bytes ffmpeg's software decoder produced, read back from
the textures the decoder hands out: NV12 from the 8-bit streams and P010 from
the 10-bit one.
"""
import ctypes
import json
from pathlib import Path

import pytest

from pyopengl_video import DecoderError, decoders, open_decoder, vaapi
from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.decoder import Sample
from pyopengl_video.mp4 import split_annexb
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.decoder import VAAPIDecoder
from pyopengl_video.vaapi.h264 import annexb
from pyopengl_video.vaapi.hevc_parse import NAL_CRA, NAL_IDR_W_RADL, NAL_RASL_N, NAL_RASL_R
from tests.conftest import decode_all, md5s
from tests.hevc_syntax import (
    nal,
    parameter_units,
    picture_set,
    pps_bytes,
    sequence,
    slice_rbsp,
    sps_bytes,
)

DATA = Path(__file__).parent / 'data' / 'decode'
FIXTURES = json.loads((DATA / 'hevc.json').read_text())['streams']


@pytest.fixture
def decoder(vaapi_decodes):
    vaapi_decodes('hevc')
    made = VAAPIDecoder('hevc')
    yield made
    made.close()


def samples_of(name, start=0):
    with AnnexBReader(DATA / f'{name}.h265') as stream:
        return [stream.sample(index, parameter_sets=index == start)
                for index in range(start, len(stream))]


class TestConformance:
    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_every_picture_matches_the_reference_decode(self, decoder, name):
        assert decode_all(decoder, samples_of(name)) == FIXTURES[name]['md5']

    @pytest.mark.parametrize('name,layout', [('hevc_bpyramid', 'nv12'),
                                             ('hevc_main10', 'p010')])
    def test_the_frames_are_laid_out_for_their_depth(self, decoder, name, layout):
        frames = []
        for sample in samples_of(name)[:6]:
            frames += decoder.decode(sample)
        frames += decoder.flush()
        try:
            assert {frame.layout for frame in frames} == {layout}
            assert {frame.visible for frame in frames} == {(0, 0, 120, 68)}
            assert {frame.texture_size for frame in frames} == {(120, 72)}
        finally:
            for frame in frames:
                frame.release()

    def test_the_colour_is_the_streams(self, decoder):
        frames = decoder.decode(samples_of('hevc_main10')[0]) + decoder.flush()
        try:
            colour = frames[0].colour
            assert (colour.primaries, colour.transfer, colour.matrix) == (9, 16, 9)
            assert not colour.full_range
        finally:
            for frame in frames:
                frame.release()

    def test_a_decoder_opens_from_the_parameter_sets(self, vaapi_decodes):
        vaapi_decodes('hevc')
        with AnnexBReader(DATA / 'hevc_main10.h265') as stream:
            headers = stream.parameter_sets()
        with open_decoder('h265', headers=headers, backend='vaapi') as made:
            assert made.codec == 'hevc'
            assert made.size == (120, 68)
            assert made.colour.matrix == 9
            assert decode_all(made, samples_of('hevc_main10')) == FIXTURES['hevc_main10']['md5']

    def test_frames_carry_the_samples_timestamps_in_display_order(self, decoder):
        with AnnexBReader(DATA / 'hevc_bpyramid.h265') as stream:
            times = [stream.presentation_time(index) for index in range(len(stream))]
        shown = []
        for sample in samples_of('hevc_bpyramid'):
            for frame in decoder.decode(sample):
                shown.append(frame.timestamp)
                frame.release()
        for frame in decoder.flush():
            shown.append(frame.timestamp)
            frame.release()
        assert shown == sorted(times)

    def test_one_decoder_takes_a_stream_of_another_size_and_depth(self, decoder):
        found = decode_all(decoder, samples_of('hevc_open_gop'))
        found += decode_all(decoder, samples_of('hevc_main10'))
        found += decode_all(decoder, samples_of('hevc_scaling_lists'))
        assert found == (FIXTURES['hevc_open_gop']['md5'] + FIXTURES['hevc_main10']['md5']
                         + FIXTURES['hevc_scaling_lists']['md5'])
        assert decoder.size == tuple(FIXTURES['hevc_scaling_lists']['size'])


class TestSeveralMovies:
    def test_six_decoders_in_one_context_each_decode_their_own(self, vaapi_decodes):
        """A game with screens on its walls: one session per movie, interleaved."""
        vaapi_decodes('hevc')
        names = sorted(FIXTURES)
        streams = [AnnexBReader(DATA / f'{name}.h265') for name in names]
        made = [VAAPIDecoder('hevc') for _ in names]
        found = {name: [] for name in names}
        try:
            for index in range(max(len(stream) for stream in streams)):
                for name, stream, each in zip(names, streams, made, strict=True):
                    if index < len(stream):
                        found[name] += md5s(each.decode(stream.sample(index)))
            for name, each in zip(names, made, strict=True):
                found[name] += md5s(each.flush())
        finally:
            for each in made:
                each.close()
            for stream in streams:
                stream.close()
        assert found == {name: FIXTURES[name]['md5'] for name in names}


class TestSeeking:
    def test_a_seek_to_a_cra_picture_drops_its_leading_pictures(self, decoder):
        """The pictures after a CRA picture that are shown before it predict
        from the group before, which a decoder starting there never had."""
        name = 'hevc_open_gop'
        with AnnexBReader(DATA / f'{name}.h265') as stream:
            key = stream.sync_samples[1]
            order = stream.display_order()
            kinds = [[(unit[0] >> 1) & 0x3F for unit in split_annexb(stream.sample(index).data)]
                     for index in range(key, key + 4)]
        assert NAL_CRA in kinds[0]
        assert all(NAL_RASL_N in found or NAL_RASL_R in found for found in kinds[1:])
        decode_all(decoder, samples_of(name)[:key - 2])
        decoder.reset()
        found = decode_all(decoder, samples_of(name, start=key))
        expected = FIXTURES[name]['md5'][order.index(key):]
        assert found == expected
        assert len(found) == len(order) - key - 3

    def test_decoding_resumes_from_an_idr_picture_after_a_reset(self, decoder):
        name = 'hevc_bpyramid'
        with AnnexBReader(DATA / f'{name}.h265') as stream:
            key = stream.sync_samples[1]
            order = stream.display_order()
        decode_all(decoder, samples_of(name)[:10])
        decoder.reset()
        found = decode_all(decoder, samples_of(name, start=key))
        assert found == FIXTURES[name]['md5'][order.index(key):]

    def test_samples_before_a_key_picture_are_passed_over(self, decoder):
        name = 'hevc_main10'
        with AnnexBReader(DATA / f'{name}.h265') as stream:
            key = stream.sync_samples[1]
            order = stream.display_order()
        found = decode_all(decoder, samples_of(name, start=key - 5))
        assert found == FIXTURES[name]['md5'][order.index(key):]

    def test_the_same_stream_decodes_again_after_a_reset(self, decoder):
        name = 'hevc_slices_wpp'
        for _ in range(3):
            decoder.reset()
            assert decode_all(decoder, samples_of(name)) == FIXTURES[name]['md5']


def written(*, sps=None, pps=None):
    """One sample: parameter sets from the options given, and an IDR slice."""
    sps_options, pps_options = sps or {}, pps or {}
    slice_data = slice_rbsp(sequence(**sps_options), picture_set(**pps_options),
                            nal_type=NAL_IDR_W_RADL, kind='I')
    units = parameter_units(sps=sps_bytes(**sps_options), pps=pps_bytes(**pps_options))
    return Sample(annexb(*units, nal(NAL_IDR_W_RADL, slice_data)), 0, 3000, True, 0)


def smallest_surface(decoder):
    """The smallest surface the decoder's device makes for HEVC."""
    device = decoder._device  # noqa: SLF001 - the device the decoder under test opened
    config = api.VAConfigID(0)
    attribute = api.VAConfigAttrib(type=api.VAConfigAttribRTFormat,
                                   value=api.VA_RT_FORMAT_YUV420)
    device.check(
        device.va.vaCreateConfig(device.display, api.VAProfileHEVCMain, api.VAEntrypointVLD,
                                 ctypes.byref(attribute), 1, ctypes.byref(config)),
        'vaCreateConfig', 'HEVC Main decode')
    try:
        limits = device.surface_limits(config)
    finally:
        device.va.vaDestroyConfig(device.display, config)
    return (limits.get(api.VASurfaceAttribMinWidth, 0),
            limits.get(api.VASurfaceAttribMinHeight, 0))


class TestRefusals:
    """Refused when a picture uses the sequence, before the driver is asked."""

    @pytest.mark.parametrize('sps,pps,message', [
        ({'chroma_format_idc': 2, 'profile_idc': 4}, {}, 'Format Range Extensions profile at 4:2:2'),
        ({'chroma_format_idc': 3, 'profile_idc': 4}, {}, '4:4:4'),
        ({'bit_depth': 4, 'profile_idc': 4}, {}, '12 bits a sample'),
        ({'range_extension': (0, 0, 0, 0, 1, 0, 0, 0, 0), 'profile_idc': 4}, {},
         'extended_precision_processing_flag'),
        ({}, {'range_extension': {'cross_component': 1}},
         'cross_component_prediction_enabled_flag'),
        ({'extension_flags': (0, 0, 1), 'profile_idc': 9}, {}, 'screen content coding'),
    ])
    def test_a_stream_that_is_not_decoded_here_is_refused_by_name(self, decoder, sps, pps,
                                                                  message):
        with pytest.raises(DecoderError, match=message):
            decoder.decode(written(sps=sps, pps=pps))
        assert decoder.flush() == []

    def test_a_picture_smaller_than_the_device_reaches_is_refused_by_its_size(self, decoder):
        """Decoded into a larger surface it would come out as another picture."""
        least = smallest_surface(decoder)
        if least[0] <= 32 and least[1] <= 32:
            pytest.skip(f'this device decodes HEVC pictures down to {least[0]}x{least[1]}')
        with pytest.raises(DecoderError, match='32x32 HEVC Main picture is smaller than'):
            decoder.decode(written(sps={'width': 32, 'height': 32}))

    def test_a_slice_naming_no_parameter_set_is_refused(self, decoder):
        sample = written()
        slice_only = b'\x00\x00\x00\x01' + list(split_annexb(sample.data))[-1]
        with pytest.raises(DecoderError, match='picture parameter set 0'):
            decoder.decode(Sample(slice_only, 0, 3000, True, 0))

    def test_decoding_goes_on_after_a_refusal(self, decoder):
        with pytest.raises(DecoderError, match='4:2:2'):
            decoder.decode(written(sps={'chroma_format_idc': 2, 'profile_idc': 4}))
        decoder.reset()
        name = 'hevc_scaling_default'
        assert decode_all(decoder, samples_of(name)) == FIXTURES[name]['md5']

    def test_decoding_after_close_is_an_error(self, vaapi_decodes):
        vaapi_decodes('hevc')
        made = VAAPIDecoder('hevc')
        made.close()
        with pytest.raises(DecoderError, match='closed'):
            made.decode(samples_of('hevc_bpyramid')[0])


class TestSurfaces:
    def test_a_decoder_that_has_lent_every_surface_says_which_are_held(self, decoder):
        held = []
        with pytest.raises(DecoderError, match='held'):
            for sample in samples_of('hevc_slices_wpp'):
                held += decoder.decode(sample)
        assert len(held) >= decoder.output_slots
        for frame in held:
            frame.release()


class TestRegistry:
    def test_the_backend_lists_the_codec(self, vaapi_decodes):
        vaapi_decodes('hevc')
        assert 'hevc' in vaapi.decoder_codecs()
        assert 'vaapi' in [backend.name for backend in decoders()]

    def test_the_codec_opens_by_its_other_names(self, vaapi_decodes):
        vaapi_decodes('hevc')
        for name in ('hevc', 'h265', 'H.265'):
            with open_decoder(name, backend='vaapi') as made:
                assert made.codec == 'hevc'
