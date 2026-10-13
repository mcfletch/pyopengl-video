"""The VP9 fuzz gate: mutated fixtures from the parser to the driver's buffers.

Every mutated stream either decodes or is refused with the package's own
error, in under a second, and whatever is handed to the session is inside
the range the VP9 Bitstream and Decoding Process Specification gives it and
inside what libva's structures hold. The findings these checks first made are
A6 to A9 of ``plans/SECURITY-REVIEW.md``.
"""
import pytest

from pyopengl_video.nvdec import description
from pyopengl_video.nvdec.vp9 import Slots
from pyopengl_video.vaapi import api, vp9_api, vp9_decode
from pyopengl_video.vaapi.vp9_parse import NON_KEY_FRAME
from tests import ivf_fuzz

CASES = 4000
SEED = 20261013

#: How many bytes at the start of a frame are taken as its header, and how
#: many at the end of a sample as its superframe index.
HEADER_BYTES = 36
INDEX_BYTES = 12

#: The libva profile of each VP9 profile decoded here.
PROFILES = {api.VAProfileVP9Profile0: 0, api.VAProfileVP9Profile2: 2}


def header_regions(data):
    """The byte ranges of a sample that hold a header or the superframe index."""
    return [(0, min(len(data), HEADER_BYTES))] * 5 + [
        (max(len(data) - INDEX_BYTES, 0), len(data))]


def check_tiles(findings, data, position, count):
    """Every tile's size, as decode_tiles() reads them, lies inside the frame."""
    for number in range(count - 1):
        size = int.from_bytes(data[position:position + 4], 'big')
        position += 4 + size
        if size < 1 or position > len(data):
            findings.report('a tile that runs past the frame submitted', (number, size))
            return
    if position >= len(data):
        findings.report('a frame submitted with no byte for its last tile', count)


def check_picture(findings, session, surface, batches, parsed):
    """One submitted picture: its parameter buffer, its slice and its data."""
    report, within = findings.report, findings.within
    (kind, parameters), = batches[0]
    (_, slice_), (_, data) = batches[1]
    if kind != api.VAPictureParameterBufferType:
        report('a picture submitted without its picture parameters')
        return
    header = parsed(parameters)
    width, height = session.format.coded_size
    if (parameters.frame_width, parameters.frame_height) != header.size:
        report('a frame size other than the one parsed', header.size)
    within('driver frame_width', parameters.frame_width, 1, width)
    within('driver frame_height', parameters.frame_height, 1, height)
    if parameters.profile != PROFILES[session.format.profile]:
        report('a picture of another profile than its key frame',
               (parameters.profile, session.format.name))
    if parameters.bit_depth != session.format.depth:
        report('a picture of another bit depth than its surfaces', parameters.bit_depth)
    bits = parameters.pic_fields.bits
    if (bits.subsampling_x, bits.subsampling_y) != (1, 1):
        report('a picture that is not 4:2:0')
    if bits.frame_type == NON_KEY_FRAME and not bits.intra_only:
        references = [parameters.reference_frames[slot] for slot in (
            bits.last_ref_frame, bits.golden_ref_frame, bits.alt_ref_frame)]
        ivf_fuzz.check_references(findings, session, references)
        if surface in references:
            report('a picture decoded into a surface it predicts from', surface)
    for held in parameters.reference_frames:
        if held != api.VA_INVALID_SURFACE and held not in session.decoded:
            report('a reference slot naming a surface never decoded', held)
    within('driver log2_tile_columns', parameters.log2_tile_columns, 0, 6)
    within('driver log2_tile_rows', parameters.log2_tile_rows, 0, 2)
    if (parameters.log2_tile_columns, parameters.log2_tile_rows) != (
            header.tile_cols_log2, header.tile_rows_log2):
        report('a tile grid other than the one parsed')
    if (slice_.slice_data_offset, slice_.slice_data_size) != (0, len(data)):
        report('a slice that is not its data buffer', slice_.slice_data_size)
    headers = parameters.frame_header_length_in_bytes + parameters.first_partition_size
    if parameters.frame_header_length_in_bytes != header.uncompressed_size:
        report('an uncompressed header length other than the one parsed')
    within('driver first_partition_size', parameters.first_partition_size, 1, len(data))
    check_tiles(findings, data, headers,
                1 << (parameters.log2_tile_columns + parameters.log2_tile_rows))
    for segment in slice_.seg_param:
        for levels in segment.filter_level:
            for level in levels:
                within('driver filter level', level, 0, 63)


@pytest.fixture
def run(monkeypatch):
    """A fuzz run of the VP9 layer, every store into its buffers read back."""
    findings = ivf_fuzz.Findings()
    built = []
    build = vp9_decode.picture_parameters

    def recording(header, slots):
        parameters = build(header, slots)
        built[:] = [(parameters, header)]
        return parameters

    def parsed(parameters):
        (made, header), = built
        assert made is parameters
        return header

    def check(findings, session, surface, batches):
        check_picture(findings, session, surface, batches, parsed)

    monkeypatch.setattr(vp9_decode, 'vp9_api', ivf_fuzz.recorded(vp9_api, findings.report))
    monkeypatch.setattr(vp9_decode, 'picture_parameters', recording)

    def fuzz(count=CASES):
        return ivf_fuzz.run(
            vp9_decode.VP9Decoding, ivf_fuzz.streams('vp9_*.ivf'), count, SEED,
            header_regions, check, findings,
            others=[('describe', lambda data: description.describe('vp9', data)),
                    ('the NVDEC slots', lambda data: Slots().frames(data))])
    return fuzz


def test_mutated_streams_are_decoded_or_refused_and_stay_in_range(run):
    findings = run()
    assert not findings.found, f'\n{findings}'
    # Enough of the cases reach the driver for the checks on its buffers to mean something.
    assert findings.pictures > CASES


def test_the_fixtures_themselves_are_decoded_with_nothing_found(monkeypatch, run):
    """With no mutation, every frame of every fixture reaches the driver."""
    monkeypatch.setattr(ivf_fuzz, 'mutate', lambda _chooser, samples, *_rest: samples)
    findings = run(40)
    assert not findings.found, f'\n{findings}'
    assert findings.pictures >= 40 * (ivf_fuzz.SAMPLES - 1)
