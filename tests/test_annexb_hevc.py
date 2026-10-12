"""Raw HEVC files: told from H.264 by content, then split, ordered and timed.

No GPU. The access units and the order they are shown in are checked against
what the reference parser reads from each fixture (``tests/hevc_reference.py``).
"""
import io
import json
from pathlib import Path

import pytest

from pyopengl_video.annexb import AnnexBReader
from pyopengl_video.mp4 import split_annexb
from pyopengl_video.vaapi.h264 import StreamError, annexb
from pyopengl_video.vaapi.hevc_parse import (
    NAL_AUD,
    NAL_CRA,
    NAL_END_OF_SEQUENCE,
    NAL_IDR_W_RADL,
    NAL_PPS,
    NAL_PREFIX_SEI,
    NAL_RASL_N,
    NAL_SPS,
    NAL_SUFFIX_SEI,
    NAL_VPS,
)
from tests.hevc_reference import FIXTURES as REFERENCE
from tests.hevc_reference import SLICE_COLUMNS
from tests.hevc_syntax import (
    nal,
    parameter_units,
    picture_set,
    pps_bytes,
    sequence,
    slice_rbsp,
    sps_bytes,
    vps_bytes,
)

DATA = Path(__file__).parent / 'data' / 'decode'
FIXTURES = json.loads((DATA / 'hevc.json').read_text())['streams']
IRAP = range(16, 24)


def kinds(data):
    return [(unit[0] >> 1) & 0x3F for unit in split_annexb(data)]


def pictures(name):
    """The reference parser's first slice segment of each picture, as a dict."""
    return [dict(zip(SLICE_COLUMNS, row, strict=True)) for row in REFERENCE[name]['slices']
            if row[SLICE_COLUMNS.index('slice_segment_address')] == 0]


class TestTheCodecIsReadFromTheStream:
    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_an_hevc_stream(self, name):
        with AnnexBReader(DATA / f'{name}.h265') as reader:
            assert reader.codec == 'hevc'

    @pytest.mark.parametrize('name', ['high_bpyramid', 'baseline_cavlc'])
    def test_an_h264_stream(self, name):
        with AnnexBReader(DATA / f'{name}.h264') as reader:
            assert reader.codec == 'h264'

    def test_the_file_name_says_nothing(self, tmp_path):
        path = tmp_path / 'movie.h264'
        path.write_bytes((DATA / 'hevc_bpyramid.h265').read_bytes())
        with AnnexBReader(path) as reader:
            assert reader.codec == 'hevc'
            assert len(reader) == 40


class TestFixtures:
    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_every_picture_is_one_access_unit(self, name):
        with AnnexBReader(DATA / f'{name}.h265') as reader:
            assert len(reader) == len(FIXTURES[name]['md5']) == len(pictures(name))

    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_the_size_is_the_cropped_one(self, name):
        with AnnexBReader(DATA / f'{name}.h265') as reader:
            assert reader.size == tuple(FIXTURES[name]['size'])

    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_samples_are_the_bytes_of_the_access_units(self, name):
        data = (DATA / f'{name}.h265').read_bytes()
        with AnnexBReader(DATA / f'{name}.h265') as reader:
            joined = b''.join(reader.sample(index).data for index in range(len(reader)))
        assert joined == data

    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_display_order_is_by_picture_order_count(self, name):
        """A fixture's counts restart at each IDR picture and never reach the
        256 its LSB holds, so the LSB orders the pictures between two."""
        group = -1
        expected = []
        for index, picture in enumerate(pictures(name)):
            group += picture['nal_unit_type'] in (19, 20)
            expected.append((group, picture['slice_pic_order_cnt_lsb'], index))
        with AnnexBReader(DATA / f'{name}.h265') as reader:
            assert reader.display_order() == [index for _, _, index in sorted(expected)]

    @pytest.mark.parametrize('name', sorted(FIXTURES))
    def test_sync_samples_are_the_random_access_pictures(self, name):
        expected = [index for index, picture in enumerate(pictures(name))
                    if picture['nal_unit_type'] in IRAP]
        with AnnexBReader(DATA / f'{name}.h265') as reader:
            assert reader.sync_samples == expected
            assert [reader.sample(index).keyframe for index in range(len(reader))] == [
                index in expected for index in range(len(reader))]

    def test_an_open_gop_stream_has_cra_sync_samples(self):
        with AnnexBReader(DATA / 'hevc_open_gop.h265') as reader:
            types = [kinds(reader.sample(index).data) for index in reader.sync_samples]
            assert reader.sync_samples[0] == 0
            assert len(reader.sync_samples) == 3
            assert all(NAL_CRA in found for found in types[1:])
            assert reader.keyframe_before(reader.sync_samples[1] + 2) == reader.sync_samples[1]

    def test_a_picture_of_three_slices_is_one_sample(self):
        with AnnexBReader(DATA / 'hevc_slices_wpp.h265') as reader:
            found = kinds(reader.sample(1).data)
        assert len([kind for kind in found if kind < 32]) == 3

    def test_a_delimiter_starts_its_access_unit(self):
        with AnnexBReader(DATA / 'hevc_scaling_lists.h265') as reader:
            assert all(kinds(reader.sample(index).data)[0] == NAL_AUD
                       for index in range(len(reader)))

    def test_timestamps_count_frames_in_display_order(self):
        with AnnexBReader(DATA / 'hevc_bpyramid.h265') as reader:
            step = reader.timescale // 30
            shown = sorted(reader.sample(index).timestamp for index in range(len(reader)))
            assert shown == [step * index for index in range(len(reader))]
            first = reader.display_order()[0]
            assert reader.sample(first).timestamp == 0
            assert reader.sample(3).duration == step
            assert reader.presentation_time(3) == reader.sample(3).timestamp

    def test_the_frame_rate_is_the_one_the_stream_states(self):
        with AnnexBReader(DATA / 'hevc_main10.h265') as reader:
            assert reader.frame_rate == (30, 1)
            assert reader.duration == pytest.approx(1.0)

    def test_parameter_sets_are_the_three(self):
        with AnnexBReader(DATA / 'hevc_bpyramid.h265') as reader:
            assert kinds(reader.parameter_sets()) == [NAL_VPS, NAL_SPS, NAL_PPS]

    def test_parameter_sets_can_lead_a_sample(self):
        with AnnexBReader(DATA / 'hevc_bpyramid.h265') as reader:
            sample = reader.sample(25, parameter_sets=True)
            assert kinds(sample.data)[:3] == [NAL_VPS, NAL_SPS, NAL_PPS]
            assert sample.data.endswith(reader.sample(25).data)

    def test_a_file_object_is_read_too(self):
        data = (DATA / 'hevc_open_gop.h265').read_bytes()
        with AnnexBReader(io.BytesIO(data)) as reader:
            assert reader.codec == 'hevc'
            assert len(reader) == len(FIXTURES['hevc_open_gop']['md5'])


def written(*slices, sps=None, vps=None, extra=()):
    """A stream of the parameter sets and one NAL unit for each of `slices`.

    A slice is the options of ``slice_rbsp``; a bytes object is a unit as it is.
    """
    sps_data = sps or sps_bytes(ordering=((4, 2, 0),))
    parsed_sps, parsed_pps = sequence(ordering=((4, 2, 0),)), picture_set()
    units = parameter_units(sps=sps_data, pps=pps_bytes(), vps=vps) + list(extra)
    for options in slices:
        if isinstance(options, bytes):
            units.append(options)
            continue
        kind = options.get('nal_type', 1)
        units.append(nal(kind, slice_rbsp(parsed_sps, parsed_pps, **options)))
    return annexb(*units)


def idr():
    return {'nal_type': NAL_IDR_W_RADL, 'kind': 'I'}


def trailing(poc, back):
    return {'kind': 'P', 'poc_lsb': poc, 'rps': {'s0': [(back - 1, 1)]}}


class TestWrittenStreams:
    def test_the_frame_rate_falls_back_to_the_vps(self):
        stream = written(idr(), vps=vps_bytes(timing=(1, 50)))
        with AnnexBReader(io.BytesIO(stream)) as reader:
            assert reader.frame_rate == (50, 1)

    def test_the_sequences_rate_is_taken_before_the_vps(self):
        stream = written(idr(), vps=vps_bytes(timing=(1, 50)),
                         sps=sps_bytes(vui={'timing': (1001, 24000)}))
        with AnnexBReader(io.BytesIO(stream)) as reader:
            assert reader.frame_rate == (24000, 1001)

    def test_the_frame_rate_falls_back_when_the_stream_states_none(self):
        with AnnexBReader(io.BytesIO(written(idr(), idr()))) as reader:
            assert reader.frame_rate == (25, 1)
            assert len(reader) == 2
            assert reader.sync_samples == [0, 1]

    def test_a_caller_may_state_the_frame_rate(self):
        stream = written(idr(), trailing(1, 1))
        with AnnexBReader(io.BytesIO(stream), frame_rate=(30000, 1001)) as reader:
            assert reader.frame_rate == (30000, 1001)
            assert reader.sample(1).timestamp == 3003

    def test_a_second_slice_segment_stays_in_its_picture(self):
        second = {'kind': 'P', 'poc_lsb': 2, 'first': 0, 'address': 3,
                  'rps': {'s0': [(1, 1)]}}
        stream = written(idr(), trailing(2, 2), second, trailing(1, 1))
        with AnnexBReader(io.BytesIO(stream)) as reader:
            assert len(reader) == 3
            assert len(kinds(reader.sample(1).data)) == 2
            assert reader.display_order() == [0, 2, 1]

    def test_units_that_follow_a_picture_stay_with_it(self):
        suffix = nal(NAL_SUFFIX_SEI, b'\x84\x00\x80')
        end = nal(NAL_END_OF_SEQUENCE, b'')[:2]
        prefix = nal(NAL_PREFIX_SEI, b'\x05\x00\x80')
        stream = written(idr(), suffix, end, prefix, idr())
        with AnnexBReader(io.BytesIO(stream)) as reader:
            assert len(reader) == 2
            assert kinds(reader.sample(0).data)[-2:] == [NAL_SUFFIX_SEI, NAL_END_OF_SEQUENCE]
            assert kinds(reader.sample(1).data) == [NAL_PREFIX_SEI, NAL_IDR_W_RADL]

    def test_leading_pictures_that_cannot_be_decoded_are_ordered_last(self):
        """A stream cut at a CRA picture: its skipped leading picture is
        never shown, and still has a place in the stream."""
        cra = {'nal_type': NAL_CRA, 'kind': 'I', 'poc_lsb': 8}
        rasl = {'nal_type': NAL_RASL_N, 'kind': 'P', 'poc_lsb': 6, 'rps': {'s0': [(3, 1)]}}
        stream = written(cra, rasl, trailing(10, 2), trailing(9, 1))
        with AnnexBReader(io.BytesIO(stream)) as reader:
            assert reader.display_order() == [0, 3, 2, 1]
            assert reader.sync_samples == [0]

    def test_a_parameter_set_sent_again_is_listed_once(self):
        again = parameter_units(sps=sps_bytes(ordering=((4, 2, 0),)), pps=pps_bytes())
        stream = written(idr(), *again, idr())
        with AnnexBReader(io.BytesIO(stream)) as reader:
            assert len(reader) == 2
            assert kinds(reader.sample(1, parameter_sets=True).data)[:3] == [
                NAL_VPS, NAL_SPS, NAL_PPS]
            assert kinds(reader.parameter_sets()) == [NAL_VPS, NAL_SPS, NAL_PPS]

    def test_a_layer_above_the_base_is_carried_and_not_read(self):
        enhancement = nal(1, b'\xff\xff\xff', layer=1)
        stream = written(idr(), enhancement, trailing(1, 1))
        with AnnexBReader(io.BytesIO(stream)) as reader:
            assert len(reader) == 2
            assert len(kinds(reader.sample(0).data)) == 5

    def test_a_stream_with_no_pictures_is_refused(self):
        stream = annexb(*parameter_units())
        with pytest.raises(StreamError, match='no pictures'):
            AnnexBReader(io.BytesIO(stream))

    def test_a_slice_before_its_parameter_sets_is_refused(self):
        sps, pps = sequence(), picture_set()
        stream = annexb(nal(NAL_SPS, sps_bytes()),
                        nal(NAL_IDR_W_RADL, slice_rbsp(sps, pps, nal_type=19, kind='I')))
        with pytest.raises(StreamError, match='picture parameter set'):
            AnnexBReader(io.BytesIO(stream))
