"""Streams that ended the process, each decoded in a child that must exit cleanly.

NVIDIA's AV1 parser builds a picture from whatever it is handed, and the
hardware decoder dereferences what the picture names. Three inputs reached a
segmentation fault that way (N1 to N3 of ``plans/SECURITY-REVIEW.md``): a
stream joined after its key frame, a temporal unit cut short, and a sequence
header with one bit changed. Each runs here in a process of its own
(``tests/nvidia_child.py``), which has to exit with status 0 having raised
nothing but the package's errors.

The changes are those of the inputs the review saved
(:data:`tests.nvidia_child.SAVED`), as offsets into the fixtures in
``tests/data/decode``.
"""
import pytest

from tests import nvidia_child

pytestmark = pytest.mark.usefixtures('nvdec_available')

#: AV1 fixtures, the sample decoding starts at, which is past the first key
#: frame, and how many frames the stream shows from its second.
MID_STREAM = [('av1_large_altref.ivf', 3, 15), ('av1_large_altref.ivf', 1, 15),
              ('av1_large_grain.ivf', 2, 10), ('av1_large_superres.ivf', 5, 10),
              ('av1_large_odd.ivf', 1, 6), ('av1_large_monochrome.ivf', 4, 6)]


@pytest.mark.parametrize('interop', [True, False], ids=['on-the-gpu', 'through-host-memory'])
class TestSavedInputs:
    @pytest.mark.parametrize('name', sorted(nvidia_child.SAVED))
    def test_the_stream_is_refused_and_the_process_lives(self, name, interop):
        fixture, changes = nvidia_child.SAVED[name]
        report = nvidia_child.run('decode', fixture, 0, changes, False, interop).clean()
        assert report['errors'], 'the changed stream decoded as though it were whole'


class TestJoiningAfterTheKeyFrame:
    """N1: the first sample decoded is an inter frame, its references never decoded."""

    @pytest.mark.parametrize(('fixture', 'start', 'shown'), MID_STREAM)
    def test_nothing_is_decoded_until_a_key_frame(self, fixture, start, shown):
        report = nvidia_child.run('decode', fixture, start, None, True).clean()
        assert report == {'frames': shown, 'errors': {}}

    def test_a_stream_decodes_from_its_key_frame_as_it_did(self):
        report = nvidia_child.run('decode', 'av1_large_altref.ivf', 0, None, True).clean()
        assert report['errors'] == {}
        assert report['frames'] > 0


class TestThePictureGuardAlone:
    """N1 and N2 with every sample handed to NVIDIA's parser unread.

    The gate stops these inputs before the parser sees them. With it out of
    the way the parser builds its pictures, and what holds is the reading of
    each picture in the decode callback.
    """

    @pytest.mark.parametrize(('fixture', 'start', 'shown'), MID_STREAM)
    def test_a_picture_that_predicts_from_one_never_decoded_is_refused(
            self, fixture, start, shown):
        report = nvidia_child.run('decode', fixture, start, None, True, True, False).clean()
        assert report['frames'] == shown
        [refusal] = report['errors']
        assert refusal.startswith("DecoderError: a picture NVIDIA's parser built is refused")

    @pytest.mark.parametrize('name', [name for name in sorted(nvidia_child.SAVED)
                                      if name.endswith('(N2)')])
    def test_a_picture_longer_than_the_bytes_fed_is_refused(self, name):
        fixture, changes = nvidia_child.SAVED[name]
        report = nvidia_child.run('decode', fixture, 0, changes, False, True, False).clean()
        assert report['errors']


class TestObusTheParserFaultsOn:
    """N3: an OBU with no payload where NVIDIA's parser reads one.

    A metadata OBU of no bytes before a frame header ended the process inside
    ``cuvidParseVideoData`` once in some thousands of such samples, so no one
    run shows it. What is asserted here is that the stream decodes whole,
    since the gate hands the parser the sample without that OBU; that it is
    left out is held, on any machine, by ``tests/test_nvdec_av1_gate.py``.
    """

    def test_a_stream_with_empty_metadata_obus_decodes_whole(self):
        # The temporal delimiter of each sample, an OBU of no bytes, made a
        # metadata OBU of no bytes.
        fixture = 'av1_large_monochrome.ivf'
        changes = {index: [[0, 0x2a]] for index in range(12)}
        report = nvidia_child.run('decode', fixture, 0, changes).clean()
        assert report == {'frames': 12, 'errors': {}}
