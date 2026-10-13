"""What the NVDEC backend decides without a driver: no GPU, no OpenGL context.

The sequence a decoder is built for and what refuses it, the surfaces each
sample depth decodes into, which displayed pictures reach the caller, how the
output textures are lent, where the two planes sit in a mapped picture, and
what discovery says of a machine.
"""
import contextlib
import ctypes

import pytest
from OpenGL.GL import GL_R8, GL_R16, GL_RG8, GL_RG16, GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT

import pyopengl_video
from pyopengl_video import decoder as decoder_module
from pyopengl_video import nvdec
from pyopengl_video.colour import Colour
from pyopengl_video.decoder import DecoderError, Sample
from pyopengl_video.nvdec import api
from pyopengl_video.nvdec.decoder import (
    DEINTERLACE_MODES,
    SURFACES,
    NVDECDecoder,
    Outputs,
    Presentation,
    Sequence,
    plane_copies,
)

DEVICE = 'a test GPU'


def video_format(**changes):
    """A 1080p High stream as the parser reports one, with `changes` applied."""
    format = api.CUVIDEOFORMAT(
        codec=api.cudaVideoCodec_H264, progressive_sequence=1,
        min_num_decode_surfaces=9, coded_width=1920, coded_height=1088,
        chroma_format=api.cudaVideoChromaFormat_420)
    format.display_area.right = 1920
    format.display_area.bottom = 1080
    signal = format.video_signal_description
    signal.matrix_coefficients = signal.color_primaries = 2
    signal.transfer_characteristics = 2
    for name, value in changes.items():
        setattr(format, name, value)
    return format


def deep_format(bits=10, **changes):
    """A stream of `bits` a sample, luma and chroma alike."""
    return video_format(bit_depth_luma_minus8=bits - 8, bit_depth_chroma_minus8=bits - 8,
                        **changes)


def capabilities(**changes):
    """What an RTX 3060 Ti answers for 8-bit 4:2:0 H.264, with `changes` applied."""
    found = api.CUVIDDECODECAPS(
        bIsSupported=1, nOutputFormatMask=1, nMaxWidth=4096, nMaxHeight=4096,
        nMinWidth=48, nMinHeight=16, nMaxMBCount=65536)
    for name, value in changes.items():
        setattr(found, name, value)
    return found


class TestSequence:
    def test_it_holds_what_the_parser_reported(self):
        sequence = Sequence.of(video_format())
        assert sequence.coded_size == (1920, 1088)
        assert sequence.visible == (0, 0, 1920, 1080)
        assert sequence.size == (1920, 1080)
        assert sequence.surfaces == 9
        assert sequence.progressive
        assert sequence.bit_depth == (8, 8)

    def test_the_visible_rectangle_is_x_y_width_height(self):
        format = video_format()
        area = format.display_area
        area.left, area.top, area.right, area.bottom = 8, 4, 1288, 724
        assert Sequence.of(format).visible == (8, 4, 1280, 720)

    def test_the_colour_description_is_the_streams(self):
        format = video_format()
        signal = format.video_signal_description
        signal.matrix_coefficients, signal.color_primaries = 1, 1
        signal.transfer_characteristics, signal.video_full_range_flag = 13, 1
        assert Sequence.of(format).colour == Colour(matrix=1, full_range=True,
                                                    primaries=1, transfer=13)

    def test_two_reports_of_one_sequence_are_equal(self):
        """A parser started after a seek reports the sequence again."""
        assert Sequence.of(video_format()) == Sequence.of(video_format())
        assert Sequence.of(video_format()) != Sequence.of(video_format(coded_height=720))

    def test_a_stream_the_hardware_decodes_is_not_refused(self):
        assert Sequence.of(video_format()).refusal(capabilities(), DEVICE) == ''

    def test_a_vp9_streams_colour_is_left_unstated(self):
        """The parser fills in nothing for VP9, and zeros would read as RGB."""
        format = video_format(codec=api.cudaVideoCodec_VP9)
        signal = format.video_signal_description
        signal.matrix_coefficients = signal.color_primaries = 0
        signal.transfer_characteristics = 0
        assert Sequence.of(format).colour == Colour()

    @pytest.mark.parametrize('coded, surface', [
        ((1920, 1088), (1920, 1088)), ((151, 145), (152, 146)),
        ((150, 131), (150, 132)), ((151, 130), (152, 130))])
    def test_pictures_are_copied_out_at_an_even_size(self, coded, surface):
        format = video_format(coded_width=coded[0], coded_height=coded[1])
        assert Sequence.of(format).surface_size == surface


class TestSurfaceFormats:
    def test_an_8_bit_stream_decodes_to_nv12_in_byte_textures(self):
        surface = Sequence.of(video_format()).surface_format
        assert surface is SURFACES[8]
        assert surface.surface == api.cudaVideoSurfaceFormat_NV12
        assert (surface.luma, surface.chroma, surface.kind) == (GL_R8, GL_RG8, GL_UNSIGNED_BYTE)
        assert (surface.sample_bytes, surface.layout) == (1, 'nv12')

    def test_a_10_bit_stream_decodes_to_p016_in_16_bit_textures(self):
        surface = Sequence.of(deep_format(10)).surface_format
        assert surface is SURFACES[10]
        assert surface.surface == api.cudaVideoSurfaceFormat_P016
        assert (surface.luma, surface.chroma, surface.kind) == (
            GL_R16, GL_RG16, GL_UNSIGNED_SHORT)
        assert (surface.sample_bytes, surface.layout) == (2, 'p010')

    def test_each_format_is_one_bit_of_the_output_mask(self):
        assert [SURFACES[depth].output_bit for depth in (8, 10)] == [1, 2]

    def test_a_stream_with_no_chroma_decodes_as_its_depth_does(self):
        mono = api.cudaVideoChromaFormat_Monochrome
        assert Sequence.of(video_format(chroma_format=mono)).surface_format is SURFACES[8]
        assert Sequence.of(deep_format(10, chroma_format=mono)).surface_format is SURFACES[10]

    def test_chroma_is_half_the_luma_each_way(self):
        assert SURFACES[8].plane_sizes((152, 146)) == ((152, 146), (76, 73))

    @pytest.mark.parametrize('format', [
        deep_format(12),
        video_format(chroma_format=api.cudaVideoChromaFormat_422),
        video_format(chroma_format=api.cudaVideoChromaFormat_444),
        video_format(bit_depth_luma_minus8=2)])
    def test_other_layouts_have_no_surface_format(self, format):
        assert Sequence.of(format).surface_format is None


class TestRefusal:
    def refusal(self, format=None, **changes):
        return Sequence.of(format or video_format()).refusal(capabilities(**changes), DEVICE)

    def test_another_chroma_format_is_named(self):
        format = video_format(chroma_format=api.cudaVideoChromaFormat_444)
        assert self.refusal(format) == (
            'the stream is H.264, 4:4:4 at 8 bits, and this decoder takes 4:2:0 or '
            'monochrome at 8 or 10 bits')

    def test_a_stream_deeper_than_10_bits_is_named_by_its_codec_and_bits(self):
        format = deep_format(12, codec=api.cudaVideoCodec_HEVC)
        assert 'HEVC, 4:2:0 at 12 bits' in self.refusal(format)

    def test_a_10_bit_stream_is_decoded_where_the_hardware_outputs_p016(self):
        assert self.refusal(deep_format(10), nOutputFormatMask=0b11) == ''

    def test_hardware_with_no_decoder_for_it_says_so(self):
        assert self.refusal(bIsSupported=0) == (
            'a test GPU has no hardware decoder for this stream (H.264, 4:2:0 at 8 bits)')

    def test_hardware_that_cannot_output_nv12_says_so(self):
        assert 'to NV12' in self.refusal(nOutputFormatMask=2)

    def test_hardware_that_cannot_output_p016_says_so(self):
        reason = self.refusal(deep_format(10, codec=api.cudaVideoCodec_AV1),
                              nOutputFormatMask=1)
        assert 'AV1, 4:2:0 at 10 bits' in reason and 'to P016' in reason

    def test_a_picture_past_the_largest_is_refused_by_both_sizes(self):
        reason = self.refusal(nMaxWidth=1280, nMaxHeight=720)
        assert '1920x1088' in reason and '1280x720' in reason and 'larger' in reason

    def test_a_picture_under_the_smallest_is_refused_by_both_sizes(self):
        reason = self.refusal(video_format(coded_width=32, coded_height=32))
        assert '32x32' in reason and '48x16' in reason and 'smaller' in reason

    def test_a_refusal_by_size_names_the_codec_the_limit_is_for(self):
        """The smallest picture differs by codec: 144x144 in HEVC on an RTX 3060 Ti."""
        format = video_format(codec=api.cudaVideoCodec_HEVC, coded_width=128, coded_height=80)
        assert self.refusal(format, nMinWidth=144, nMinHeight=144) == (
            'a 128x80 picture is smaller than the 144x144 a test GPU decodes in HEVC')

    def test_macroblocks_are_counted_over_the_whole_picture(self):
        """A VP9 picture is of any size, and its last block is a part of one."""
        format = video_format(codec=api.cudaVideoCodec_VP9, coded_width=150, coded_height=146)
        assert self.refusal(format, nMaxMBCount=99) == (
            'a 150x146 picture is 100 macroblocks, and a test GPU decodes up to 99')
        assert self.refusal(format, nMaxMBCount=100) == ''

    def test_too_many_macroblocks_are_counted(self):
        reason = self.refusal(nMaxMBCount=8000)
        assert '8160 macroblocks' in reason and '8000' in reason


def sample(timestamp, duration=3000):
    return Sample(b'', timestamp, duration, False, 0)


class TestPresentation:
    def test_a_picture_is_shown_for_its_samples_duration(self):
        presentation = Presentation()
        presentation.fed(sample(0, 3000))
        presentation.fed(sample(6000, 1500))
        assert presentation.shown(0) == 3000
        assert presentation.shown(6000) == 1500

    def test_pictures_out_of_decoding_order_keep_their_own_durations(self):
        presentation = Presentation()
        for timestamp, duration in ((0, 10), (9000, 30), (3000, 20)):
            presentation.fed(sample(timestamp, duration))
        assert [presentation.shown(at) for at in (0, 3000, 9000)] == [10, 20, 30]

    def test_a_picture_shown_before_the_first_sample_is_dropped(self):
        """A leading picture after a seek: its references were never decoded."""
        presentation = Presentation()
        presentation.fed(sample(30000))
        presentation.fed(sample(24000))
        presentation.fed(sample(27000))
        assert presentation.shown(24000) is None
        assert presentation.shown(27000) is None
        assert presentation.shown(30000) == 3000

    def test_a_restart_moves_where_decoding_starts(self):
        presentation = Presentation()
        presentation.fed(sample(30000))
        presentation.restart()
        presentation.fed(sample(6000))
        assert presentation.shown(6000) == 3000

    def test_a_restart_forgets_the_samples_fed(self):
        presentation = Presentation()
        presentation.fed(sample(0, 1234))
        presentation.restart()
        presentation.fed(sample(0, 99))
        assert presentation.shown(0) == 99

    def test_a_sample_that_produced_no_picture_is_forgotten(self):
        presentation = Presentation()
        for timestamp in (0, 3000, 6000):
            presentation.fed(sample(timestamp))
        presentation.shown(0)
        presentation.shown(6000)
        assert presentation._durations == {}  # noqa: SLF001 - that nothing is kept is the claim

    def test_a_picture_with_no_sample_behind_it_has_no_duration(self):
        presentation = Presentation()
        presentation.fed(sample(0))
        assert presentation.shown(3000) == 0


class TestOutputs:
    def outputs(self, limit=3):
        made = iter(range(100))
        return Outputs(limit, lambda: next(made))

    def test_a_pair_is_made_when_none_is_free(self):
        outputs = self.outputs()
        assert outputs.made == []
        assert outputs.take() == 0
        assert outputs.made == [0]

    def test_a_pair_given_back_is_lent_again_before_another_is_made(self):
        outputs = self.outputs()
        first = outputs.take()
        outputs.give(first)
        assert outputs.take() == first
        assert outputs.made == [first]

    def test_the_pair_returned_longest_ago_goes_first(self):
        outputs = self.outputs()
        first, second = outputs.take(), outputs.take()
        outputs.give(second)
        outputs.give(first)
        assert outputs.take() == second

    def test_no_more_than_the_limit_are_made(self):
        outputs = self.outputs(limit=2)
        assert [outputs.take(), outputs.take(), outputs.take()] == [0, 1, None]
        assert outputs.made == [0, 1]


class TestPlaneCopies:
    def test_luma_is_the_whole_surface_from_its_first_row(self):
        luma, _chroma = plane_copies(0x1000, 2048, (1920, 1088))
        assert luma.srcMemoryType == api.CU_MEMORYTYPE_DEVICE
        assert (luma.srcDevice, luma.srcPitch) == (0x1000, 2048)
        assert (luma.WidthInBytes, luma.Height) == (1920, 1088)

    def test_chroma_follows_luma_by_the_surfaces_height_in_rows(self):
        _luma, chroma = plane_copies(0x1000, 2048, (1920, 1088))
        assert chroma.srcDevice == 0x1000 + 2048 * 1088
        assert chroma.srcPitch == 2048

    def test_chroma_is_half_as_many_rows_of_two_bytes_a_sample(self):
        _luma, chroma = plane_copies(0, 2048, (1920, 1088))
        assert (chroma.WidthInBytes, chroma.Height) == (1920, 544)

    def test_a_16_bit_plane_is_two_bytes_a_sample(self):
        luma, chroma = plane_copies(0x1000, 4096, (1920, 1088), SURFACES[10])
        assert (luma.WidthInBytes, luma.Height) == (3840, 1088)
        assert (chroma.WidthInBytes, chroma.Height) == (3840, 544)
        assert chroma.srcDevice == 0x1000 + 4096 * 1088

    def test_the_destination_is_left_to_the_transfer(self):
        for plane in plane_copies(0x1000, 2048, (1920, 1088)):
            assert plane.dstMemoryType == 0 and not plane.dstArray and not plane.dstHost


class TestSettings:
    """Refused before the driver is asked for anything, so with no driver too."""

    def test_an_unknown_setting_is_refused_by_name(self):
        with pytest.raises(DecoderError, match='no setting named threads'):
            NVDECDecoder(threads=4)

    def test_another_codec_is_refused_by_name(self):
        with pytest.raises(DecoderError, match="decodes av1, h264, hevc, vp9, not 'vp8'"):
            NVDECDecoder('vp8')

    def test_the_binding_names_the_four_codecs_the_parser_reads(self):
        assert api.CODECS == {'h264': api.cudaVideoCodec_H264, 'hevc': api.cudaVideoCodec_HEVC,
                              'vp9': api.cudaVideoCodec_VP9, 'av1': api.cudaVideoCodec_AV1}
        assert api.CODEC_TITLES[api.cudaVideoCodec_HEVC] == 'HEVC'

    def test_an_unknown_deinterlacer_lists_the_ones_there_are(self):
        with pytest.raises(DecoderError, match='adaptive, bob, weave'):
            NVDECDecoder(deinterlace='blend')

    def test_the_three_deinterlacers_are_the_drivers(self):
        assert DEINTERLACE_MODES == {
            'weave': api.cudaVideoDeinterlaceMode_Weave,
            'bob': api.cudaVideoDeinterlaceMode_Bob,
            'adaptive': api.cudaVideoDeinterlaceMode_Adaptive}


class FakeCuda:
    """The CUDA driver, as far as discovery asks: which GPUs render the context."""

    def __init__(self, gl_devices):
        self._gl_devices = gl_devices

    def gl_devices(self):
        return self._gl_devices


@pytest.fixture
def machine(monkeypatch):
    """Stand in for the driver and the context discovery asks about."""
    def describe(driver='', context='egl', gl_devices=(0,)):
        nvdec.forget_probe()
        monkeypatch.setattr(nvdec, '_ask_driver', lambda: driver)
        monkeypatch.setattr(nvdec, 'context_kind', lambda: context)
        monkeypatch.setattr(api.Cuda, 'load',
                            classmethod(lambda _cls: FakeCuda(list(gl_devices))))
    yield describe
    nvdec.forget_probe()


class TestDiscovery:
    def test_a_context_on_the_nvidia_gpu_has_both_backends(self, machine):
        machine()
        assert nvdec.probe() and nvdec.copy_probe()
        assert nvdec.explain() == nvdec.copy_explain() == ''

    def test_no_context_is_not_held_against_the_machine(self, machine):
        machine(context=None, gl_devices=())
        assert nvdec.probe()

    def test_a_glx_or_wgl_context_serves_as_an_egl_one_does(self, machine):
        machine(context='other')
        assert nvdec.probe()

    def test_a_context_on_another_gpu_leaves_the_copying_backend(self, machine):
        machine(gl_devices=())
        assert not nvdec.probe()
        assert nvdec.copy_probe()
        assert 'does not render on an NVIDIA GPU' in nvdec.explain()
        assert "'nvdec-copy'" in nvdec.explain()
        assert nvdec.copy_explain() == ''

    def test_a_machine_with_no_nvidia_driver_has_neither_and_no_advice(self, machine):
        machine(driver=nvdec.NO_DRIVER)
        assert not nvdec.probe() and not nvdec.copy_probe()
        assert nvdec.explain() == nvdec.copy_explain() == ''

    def test_a_driver_without_its_decoder_library_says_which_package(self, machine):
        machine(driver=nvdec.DECODER_LIBRARY_MISSING)
        assert not nvdec.probe() and not nvdec.copy_probe()
        assert 'libnvidia-decode' in nvdec.explain()
        assert 'libnvidia-decode' in nvdec.copy_explain()

    def test_the_driver_is_asked_once(self, machine, monkeypatch):
        machine()
        asked = []
        monkeypatch.setattr(nvdec, '_ask_driver', lambda: asked.append(1) or '')
        nvdec.probe()
        nvdec.copy_probe()
        assert len(asked) == 1
        nvdec.forget_probe()
        nvdec.probe()
        assert len(asked) == 2

    def test_a_probe_that_cannot_answer_is_unavailable(self, machine, monkeypatch):
        machine()

        def broken():
            raise RuntimeError('the driver fell over')
        monkeypatch.setattr(nvdec, '_ask_driver', broken)
        assert nvdec.probe() is False
        assert nvdec.copy_probe() is False
        assert nvdec.explain() == nvdec.copy_explain() == ''


class AskedCuda(FakeCuda):
    """A driver that also lists its devices and lends a context."""

    def __init__(self, gl_devices, devices=(0,)):
        super().__init__(gl_devices)
        self._devices = list(devices)
        self.contexts = []

    def devices(self):
        return self._devices

    def context(self, device):
        self.contexts.append(device)
        return device

    def current(self, _context):
        return contextlib.nullcontext()

    def check(self, status, call, detail=''):
        if status:
            raise api.CUDAError(status, call, detail=detail)


class FakeCuvid:
    """A decoder library answering for the codecs of one GPU generation."""

    def __init__(self, largest, status=0):
        self.largest = largest
        self.status = status
        self.asked = []

    def cuvidGetDecoderCaps(self, reference):
        capabilities = ctypes.cast(reference, ctypes.POINTER(api.CUVIDDECODECAPS)).contents
        self.asked.append((capabilities.eCodecType, capabilities.eChromaFormat,
                           capabilities.nBitDepthMinus8))
        size = self.largest.get(capabilities.eCodecType)
        if size:
            capabilities.bIsSupported = 1
            capabilities.nMaxWidth, capabilities.nMaxHeight = size
        return self.status


@pytest.fixture
def gpu(monkeypatch):
    """Stand in for a GPU that decodes the codecs given, each to its largest frame."""
    def describe(largest, *, gl_devices=(0,), devices=(0,), context='egl', status=0):
        nvdec.forget_probe()
        cuda, cuvid = AskedCuda(list(gl_devices), devices), FakeCuvid(largest, status)
        monkeypatch.setattr(nvdec, 'context_kind', lambda: context)
        monkeypatch.setattr(api.Cuda, 'load', classmethod(lambda _cls: cuda))
        monkeypatch.setattr(api.Cuvid, 'load', classmethod(lambda _cls: cuvid))
        return cuda, cuvid
    yield describe
    nvdec.forget_probe()


#: A GTX 1050: H.264, HEVC and VP9, and no AV1.
PASCAL = {api.cudaVideoCodec_H264: (4096, 4096), api.cudaVideoCodec_HEVC: (8192, 8192),
          api.cudaVideoCodec_VP9: (8192, 8192)}


class TestTheCodecsAGPUDecodes:
    def test_each_codec_the_gpu_has_is_listed_with_its_largest_frame(self, gpu):
        gpu(PASCAL)
        assert nvdec.decoder_codecs() == {
            'h264': (4096, 4096), 'hevc': (8192, 8192), 'vp9': (8192, 8192)}

    def test_it_is_asked_about_8_bit_4_2_0(self, gpu):
        _cuda, cuvid = gpu(PASCAL)
        nvdec.decoder_codecs()
        assert sorted(cuvid.asked) == sorted(
            (codec, api.cudaVideoChromaFormat_420, 0) for codec in api.CODECS.values())

    def test_the_gpu_is_asked_once(self, gpu):
        _cuda, cuvid = gpu(PASCAL)
        nvdec.decoder_codecs()
        nvdec.decoder_codecs()
        assert len(cuvid.asked) == len(api.CODECS)
        nvdec.forget_probe()
        nvdec.decoder_codecs()
        assert len(cuvid.asked) == 2 * len(api.CODECS)

    def test_the_gpu_asked_is_the_one_the_context_renders_on(self, gpu):
        cuda, _cuvid = gpu(PASCAL, gl_devices=(1,), devices=(0, 1))
        nvdec.decoder_codecs()
        assert cuda.contexts == [1]

    @pytest.mark.parametrize('settings', [{'context': None, 'gl_devices': ()},
                                          {'gl_devices': ()}])
    def test_with_no_context_on_it_the_first_gpu_is_asked(self, gpu, settings):
        cuda, _cuvid = gpu(PASCAL, devices=(0, 1), **settings)
        nvdec.decoder_codecs()
        assert cuda.contexts == [0]

    def test_a_gpu_that_will_not_answer_is_assumed_to_have_them_all(self, gpu):
        gpu(PASCAL, status=999)
        assert nvdec.decoder_codecs() == nvdec.ASSUMED_SIZES
        assert set(nvdec.ASSUMED_SIZES) == set(api.CODECS)
        assert nvdec.ASSUMED_SIZES['h264'] == (4096, 4096)

    def test_discovery_reports_the_codecs_the_gpu_has(self, gpu, monkeypatch):
        gpu(PASCAL)
        monkeypatch.setattr(nvdec, '_ask_driver', lambda: '')
        found = {backend.name: backend for backend in pyopengl_video.decoders()}
        for name in ('nvdec', 'nvdec-copy'):
            assert found[name].codecs == {'h264', 'hevc', 'vp9'}
            assert found[name].largest('h264') == (4096, 4096)
            assert found[name].largest('hevc') == (8192, 8192)

    def test_a_codec_the_gpu_lacks_goes_to_no_nvidia_backend(self, gpu, monkeypatch):
        gpu(PASCAL)
        monkeypatch.setattr(nvdec, '_ask_driver', lambda: '')
        with pytest.raises(pyopengl_video.DecoderUnavailable, match='no decoder here can read av1'):
            pyopengl_video.open_decoder('av1', backend='nvdec')


class TestAskingTheDriver:
    def test_a_missing_cuda_library_is_no_driver(self, monkeypatch):
        def missing(_cls):
            raise OSError('libcuda.so.1: cannot open shared object file')
        monkeypatch.setattr(api.Cuda, 'load', classmethod(missing))
        assert nvdec._ask_driver() == nvdec.NO_DRIVER  # noqa: SLF001 - one step of the probes, tested alone

    def test_a_driver_that_will_not_initialise_gives_its_error(self, monkeypatch):
        def failing(_cls):
            raise api.CUDAError(100, 'cuInit', 'CUDA_ERROR_NO_DEVICE')
        monkeypatch.setattr(api.Cuda, 'load', classmethod(failing))
        assert 'CUDA_ERROR_NO_DEVICE' in nvdec._ask_driver()  # noqa: SLF001 - one step of the probes, tested alone

    def test_a_missing_decoder_library_names_the_package(self, monkeypatch):
        def missing(_cls):
            raise OSError('libnvcuvid.so.1: cannot open shared object file')
        monkeypatch.setattr(api.Cuda, 'load', classmethod(lambda _cls: FakeCuda([0])))
        monkeypatch.setattr(api.Cuvid, 'load', classmethod(missing))
        assert nvdec._ask_driver() == nvdec.DECODER_LIBRARY_MISSING  # noqa: SLF001 - one step of the probes, tested alone


class TestRegistry:
    def names(self):
        return [backend.name for backend in decoder_module.BACKENDS]

    def test_both_backends_are_registered(self):
        assert {'nvdec', 'nvdec-copy'} <= set(self.names())

    def test_every_decoder_that_keeps_the_picture_on_the_gpu_is_preferred_to_the_copy(self):
        backends = decoder_module.BACKENDS
        last_zero_copy = max(index for index, backend in enumerate(backends)
                             if backend.zero_copy)
        assert self.names().index('nvdec-copy') > last_zero_copy

    def test_both_records_list_every_codec_the_parser_reads(self):
        for record in (nvdec.DECODER_BACKEND, nvdec.COPY_DECODER_BACKEND):
            assert record.codecs == {'h264', 'hevc', 'vp9', 'av1'}
            assert record.largest('h264') == (4096, 4096)
            assert record.largest('av1') == (8192, 8192)

    def test_the_records_say_which_one_copies(self):
        assert nvdec.DECODER_BACKEND.zero_copy is True
        assert nvdec.COPY_DECODER_BACKEND.zero_copy is False

    def test_each_backend_builds_the_decoder_its_way(self, monkeypatch):
        built = []
        monkeypatch.setattr('pyopengl_video.nvdec.decoder.NVDECDecoder',
                            lambda **options: built.append(options))
        nvdec.DECODER_BACKEND.factory(codec='h264', headers=b'')
        nvdec.COPY_DECODER_BACKEND.factory(codec='h264', headers=b'')
        assert [options['interop'] for options in built] == [True, False]

    def test_naming_the_backend_where_it_is_unavailable_says_why(self, machine):
        machine(gl_devices=())
        with pytest.raises(pyopengl_video.DecoderUnavailable,
                           match='does not render on an NVIDIA GPU'):
            pyopengl_video.open_decoder('h264', backend='nvdec')
