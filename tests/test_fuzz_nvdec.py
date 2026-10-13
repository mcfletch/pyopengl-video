"""The NVDEC fuzz gate: mutated streams into the hardware decoder, in a child process.

NVIDIA's parser and decoder are handed what a file holds, and a value they do
not check ends the process, so each batch of cases runs in a child
(``tests/nvidia_child.py``): the batch passes when the child exits with
status 0 inside its time limit, having raised nothing but the package's own
errors. A crash is a failed assertion that names the case, and the seed is
fixed, so the case runs again.

The cases are a fixture's samples with header bits flipped, bytes set,
samples truncated, emptied, repeated, reordered or replaced by noise, or
another stream of the codec spliced in, with a reset now and then. The
findings these first made are N1 to N3 of ``plans/SECURITY-REVIEW.md``; the
review's own runs were of several thousand cases a codec.
"""
import pytest

from pyopengl_video import nvdec
from tests import nvidia_child

pytestmark = pytest.mark.usefixtures('nvdec_available')

SEED = 20261013

#: Cases in each batch, which is what keeps the file to about half a minute.
#: AV1 has the most, since its is the parser that the findings were in.
CASES = {'av1': 30}
OTHER_CASES = 12

#: Seconds a batch is given; one that takes longer has hung.
LIMIT = 120.0

#: A fixture, how its pictures reach their textures, and the streams spliced into it.
BATCHES = [
    ('high_bpyramid.h264', True, ['high_cqm.h264', 'interlaced_mbaff.h264', 'open_gop.h264']),
    ('interlaced_mbaff.h264', False, ['baseline_cavlc.h264']),
    ('hevc_large_bpyramid.h265', True, ['hevc_large_open_gop.h265', 'hevc_slices_wpp.h265']),
    ('hevc_large_main10.h265', False, ['hevc_large_bpyramid.h265']),
    ('vp9_large_show_existing.ivf', True, ['vp9_large_altref.ivf', 'vp9_large_odd.ivf',
                                           'vp9_tiles.ivf']),
    ('vp9_large_profile2.ivf', False, ['vp9_large_odd.ivf']),
    ('av1_large_altref.ivf', True, ['av1_large_odd.ivf', 'av1_large_superres.ivf',
                                    'av1_large_grain.ivf']),
    ('av1_large_grain10.ivf', False, ['av1_large_monochrome.ivf']),
    ('av1_large_superres.ivf', True, ['av1_large_odd.ivf']),
]


@pytest.mark.parametrize(('fixture', 'interop', 'others'), BATCHES,
                         ids=[f'{fixture}-{"gpu" if interop else "host"}'
                              for fixture, interop, _ in BATCHES])
def test_mutated_streams_raise_the_packages_errors_and_nothing_else(fixture, interop, others):
    codec = nvidia_child.codec_of(fixture)
    if codec not in nvdec.decoder_codecs():
        pytest.skip(f'this NVIDIA GPU has no {codec} decoder')
    count = CASES.get(codec, OTHER_CASES)
    outcome = nvidia_child.run('fuzz', fixture, SEED, count, interop, others, timeout=LIMIT)
    cases = [line for line in outcome.output.splitlines() if line.startswith('case ')]
    assert outcome.returncode == 0 and outcome.report is not None, (
        f'the child process {outcome.ended} in {cases[-1] if cases else "no case"} of '
        f'{fixture}:\n{outcome.output[-3000:]}')
    report = outcome.report
    assert report['cases'] == count
    assert report['frames'] > 0, 'no case decoded a picture, so nothing reached the hardware'
    assert all(kind.split(':')[0] in ('DecoderError', 'CUDAError') for kind in report['errors'])
