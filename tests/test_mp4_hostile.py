"""MP4 files whose lengths, counts and offsets are not what the file holds.

No GPU. Every case is refused with :class:`MP4Error`, in a time and a memory
that follow the file's size and not the numbers it states.
"""
import gc
import io
import struct
import tracemalloc

import pytest

from pyopengl_video.backend import VideoError
from pyopengl_video.container import ContainerError
from pyopengl_video.mp4 import MP4Error, MP4Reader, VideoTrack, box, full_box
from tests.descriptors import open_descriptors
from tests.mp4_build import audio_entry, movie, visual_entry

U = struct.pack
VP9 = visual_entry('vp09', 64, 64)
#: An avcC record: one 2-byte SPS, one 1-byte PPS, 4-byte NAL lengths.
AVCC = b'\x01\x64\x00\x1f\xff\xe1\x00\x02\x67\x64\x01\x00\x01\x68'
SAMPLE = b'\x00\x00\x00\x01\x65'

#: The most Python memory opening any of these files may take.
ALLOWANCE = 4 << 20


def avc(record, width=64, height=64):
    return visual_entry('avc1', width, height, box('avcC', record))


def built(**changes):
    return movie(changes.pop('entry', VP9), changes.pop('samples', (SAMPLE,)), **changes)


def read_everything(data):
    """Open `data` and read every sample of every track."""
    with MP4Reader(io.BytesIO(data)) as reader:
        for track in reader.tracks:
            for index in range(len(track)):
                if isinstance(track, VideoTrack):
                    track.sample(index, parameter_sets=True)
                else:
                    track.sample(index)
        return reader.tracks


def refused(data, match=None):
    """`data` is refused as an MP4, within the memory allowance."""
    tracemalloc.start()
    try:
        with pytest.raises(MP4Error, match=match):
            read_everything(data)
        assert tracemalloc.get_traced_memory()[1] < ALLOWANCE
    finally:
        tracemalloc.stop()


def test_the_error_is_a_video_error_and_still_a_value_error():
    assert issubclass(MP4Error, VideoError)
    assert issubclass(MP4Error, ContainerError)
    assert issubclass(MP4Error, ValueError)


class TestSampleTablesAreHeldToTheirBoxes:
    """C1: a count sizes nothing until the box or the file is seen to hold it."""

    def test_a_constant_sample_size_with_more_samples_than_the_file_has_bytes(self):
        refused(built(tables={'stsz': full_box('stsz', 0, 0, U('>II', 5, 0xFFFFFFFF))}),
                match='stsz')

    def test_a_size_table_shorter_than_its_count(self):
        refused(built(tables={'stsz': full_box('stsz', 0, 0, U('>II', 0, 0xFFFFFFFF))}),
                match='stsz')

    def test_a_time_run_longer_than_the_track(self):
        """One run of four thousand million samples times a track of one."""
        tracks = read_everything(
            built(tables={'stts': full_box('stts', 0, 0, U('>III', 1, 0xFFFFFFFF, 1))}))
        assert len(tracks[0]) == 1

    def test_a_time_table_shorter_than_its_count(self):
        refused(built(tables={'stts': full_box('stts', 0, 0, U('>I', 0xFFFFFFFF))}),
                match='stts')

    def test_a_composition_run_longer_than_the_track(self):
        tracks = read_everything(
            built(extra=[full_box('ctts', 0, 0, U('>III', 1, 0xFFFFFFFF, 7))]))
        assert tracks[0].presentation_time(0) == 7

    def test_a_composition_table_shorter_than_its_count(self):
        refused(built(extra=[full_box('ctts', 0, 0, U('>I', 0xFFFFFFFF))]), match='ctts')

    def test_a_chunk_run_longer_than_the_track(self):
        tracks = read_everything(
            built(tables={'stsc': full_box('stsc', 0, 0, U('>IIII', 1, 1, 0xFFFFFFFF, 1))}))
        assert len(tracks[0]) == 1

    def test_a_chunk_map_shorter_than_its_count(self):
        refused(built(tables={'stsc': full_box('stsc', 0, 0, U('>I', 0xFFFFFFFF))}),
                match='stsc')

    def test_a_chunk_offset_table_shorter_than_its_count(self):
        refused(built(tables={'stco': full_box('stco', 0, 0, U('>I', 0xFFFFFFFF))}),
                match='stco')

    def test_a_sync_table_shorter_than_its_count(self):
        refused(built(extra=[full_box('stss', 0, 0, U('>I', 0xFFFFFFFF))]), match='stss')

    def test_an_edit_list_shorter_than_its_count(self):
        data = movie(VP9, (SAMPLE,))
        edts = box('edts', full_box('elst', 0, 0, U('>I', 0xFFFFFFFF)))
        at = data.index(b'tkhd') - 4
        trak = at - 8
        moov = data.index(b'moov') - 4
        patched = bytearray(data[:at] + edts + data[at:])
        for position in (trak, moov):
            size = struct.unpack_from('>I', patched, position)[0]
            struct.pack_into('>I', patched, position, size + len(edts))
        refused(bytes(patched), match='elst')

    def test_every_table_naming_twenty_million_samples_in_a_small_file(self):
        """The reviewer's amplification case: 7.6 GB from 553 bytes."""
        count = 20_000_000
        refused(built(tables={
            'stsz': full_box('stsz', 0, 0, U('>II', 5, count)),
            'stts': full_box('stts', 0, 0, U('>III', 1, count, 1)),
            'stsc': full_box('stsc', 0, 0, U('>IIII', 1, 1, count, 1))}))

    def test_the_longest_durations_are_read_as_they_are(self):
        samples = (SAMPLE,) * 3
        tracks = read_everything(built(samples=samples, tables={'stts': full_box(
            'stts', 0, 0, U('>III', 1, 3, 0xFFFFFFFF))}))
        assert tracks[0].presentation_time(2) == 2 * 0xFFFFFFFF


class TestCompactSampleSizes:
    """C6: stz2 states a field size and a count."""

    def test_a_field_size_the_format_does_not_have(self):
        for field in (0, 1, 3, 5, 12, 32):
            refused(built(drop=['stsz'], extra=[full_box(
                'stz2', 0, 0, bytes(3), bytes([field]), U('>I', 1), bytes(8))]),
                match='stz2')

    @pytest.mark.parametrize('field', [4, 8, 16])
    def test_a_count_the_box_does_not_hold(self, field):
        refused(built(drop=['stsz'], extra=[full_box(
            'stz2', 0, 0, bytes(3), bytes([field]), U('>I', 0xFFFFFFFF), b'\x55')]),
            match='stz2')

    def test_nine_one_byte_sizes_with_one_byte_present(self):
        refused(built(drop=['stsz'], extra=[full_box(
            'stz2', 0, 0, bytes(3), bytes([8]), U('>I', 9), b'\x05')]), match='stz2')

    def test_an_empty_box(self):
        refused(built(drop=['stsz'], extra=[full_box('stz2', 0, 0)]), match='stz2')

    @pytest.mark.parametrize(('field', 'packed'), [
        (4, bytes([0x53])), (8, bytes([5, 3])), (16, U('>HH', 5, 3))])
    def test_each_field_size_is_read(self, field, packed):
        samples = (b'abcde', b'xyz')
        with MP4Reader(io.BytesIO(built(samples=samples, drop=['stsz'], extra=[full_box(
                'stz2', 0, 0, bytes(3), bytes([field]), U('>I', 2), packed)]))) as reader:
            assert [reader.video.read(index) for index in range(2)] == list(samples)


class TestChunks:
    """C7: the sample-to-chunk map names chunks that exist, in order."""

    def test_a_first_chunk_of_zero(self):
        refused(built(tables={'stsc': full_box('stsc', 0, 0, U('>IIII', 1, 0, 1, 1))}),
                match='stsc')

    def test_a_first_chunk_of_zero_with_no_chunks(self):
        refused(built(tables={'stsc': full_box('stsc', 0, 0, U('>IIII', 1, 0, 1, 1)),
                              'stco': full_box('stco', 0, 0, U('>I', 0))}))

    def test_a_map_of_no_runs_places_no_samples(self):
        refused(built(tables={'stsc': full_box('stsc', 0, 0, U('>I', 0))}), match='place 0 samples')

    def test_a_first_chunk_past_the_last(self):
        refused(built(tables={'stsc': full_box('stsc', 0, 0, U('>IIII', 1, 2, 1, 1))}),
                match='stsc')

    def test_runs_that_do_not_ascend(self):
        samples = (SAMPLE, SAMPLE)
        first = len(box('ftyp', b'isom', U('>I', 512), b'isom')) + 8
        refused(built(samples=samples, tables={
            'stco': full_box('stco', 0, 0, U('>III', 2, first, first + 5)),
            'stsc': full_box('stsc', 0, 0, U('>I', 2), U('>III', 2, 1, 1),
                             U('>III', 1, 1, 1))}), match='stsc')


class TestSyncSamples:
    """C8: a sync sample is one of the track's samples."""

    @pytest.mark.parametrize('number', [0, 2, 0xFFFFFFFF])
    def test_a_sample_number_outside_the_track(self, number):
        refused(built(extra=[full_box('stss', 0, 0, U('>II', 1, number))]), match='stss')

    def test_numbers_out_of_order_and_repeated_are_put_right(self):
        samples = (SAMPLE,) * 4
        with MP4Reader(io.BytesIO(built(samples=samples, extra=[full_box(
                'stss', 0, 0, U('>I', 3), U('>III', 3, 1, 3))]))) as reader:
            assert reader.video.sync_samples == [0, 2]
            assert reader.video.keyframe_before(1) == 0
            assert reader.video.keyframe_before(3) == 2


class TestWhereTheBytesAre:
    """C4: a size or an offset is inside the file before it is read."""

    @pytest.mark.parametrize('size', [2 ** 40, 2 ** 63, 2 ** 63 + 8, 2 ** 64 - 1])
    def test_a_movie_box_larger_than_the_file(self, size):
        refused(built(assemble=lambda ftyp, mdat, moov:
                      ftyp + mdat + U('>I4sQ', 1, b'moov', size) + moov[8:]), match='moov')

    @pytest.mark.parametrize('size', [0x7FFFFFFF, 0xFFFFFFF0])
    def test_a_32_bit_movie_box_size_past_the_end(self, size):
        refused(built(assemble=lambda ftyp, mdat, moov:
                      ftyp + mdat + U('>I4s', size, b'moov') + moov[8:]), match='moov')

    def test_a_large_size_box_with_twelve_bytes_left(self):
        refused(built(assemble=lambda ftyp, *_: ftyp + U('>I4sI', 1, b'moov', 0)))

    @pytest.mark.parametrize('offset', [2 ** 64 - 1, 2 ** 63 - 2, 2 ** 40])
    def test_a_chunk_offset_outside_the_file(self, offset):
        refused(built(tables={'stco': full_box('co64', 0, 0, U('>IQ', 1, offset))}))

    def test_a_sample_larger_than_the_file(self):
        refused(built(tables={'stsz': full_box('stsz', 0, 0, U('>III', 0, 1, 0xFFFFFFFF))}),
                match='sample 0')

    def test_read_at_refuses_a_span_outside_the_file(self):
        with MP4Reader(io.BytesIO(built())) as reader:
            for offset, size in [(-1, 4), (0, -1), (0, 2 ** 40), (2 ** 64, 1)]:
                with pytest.raises(MP4Error):
                    reader.read_at(offset, size)

    def test_a_sample_index_outside_the_track_is_an_index_error(self):
        with MP4Reader(io.BytesIO(built())) as reader:
            for index in (-1, 1):
                with pytest.raises(IndexError):
                    reader.video.sample(index)


class TestRecordsAreAsLongAsTheySay:
    """C6: a length a record states is all there."""

    def test_an_avcc_record_of_four_bytes(self):
        refused(built(entry=avc(b'\x01\x64\x00\x1f')), match='avcC')

    def test_an_sps_longer_than_its_record(self):
        refused(built(entry=avc(b'\x01\x64\x00\x1f\xff\xe1\xff\xff\x67\x64')), match='avcC')

    def test_a_pps_longer_than_its_record(self):
        refused(built(entry=avc(b'\x01\x64\x00\x1f\xff\xe1\x00\x01\x67\x01\xff\xff\x68')),
                match='avcC')

    def test_two_picture_parameter_sets_are_both_read(self):
        record = b'\x01\x64\x00\x1f\xff\xe1\x00\x02\x67\x64\x02\x00\x01\x68\x00\x02\x68\x69'
        with MP4Reader(io.BytesIO(built(entry=avc(record)))) as reader:
            assert reader.video.pps == [b'\x68', b'\x68\x69']

    def test_an_hvcc_record_that_ends_before_its_arrays(self):
        refused(built(entry=visual_entry('hvc1', 64, 64, box('hvcC', bytes(22) + b'\xff'))),
                match='hvcC')

    def test_an_av1c_record_shorter_than_its_header(self):
        refused(built(entry=visual_entry('av01', 64, 64, box('av1C', b'\x81'))), match='av1C')

    def test_a_sample_entry_shorter_than_its_fixed_part(self):
        refused(built(entry=box('vp09', bytes(12))), match='sample entry')

    def test_a_nal_length_longer_than_its_sample(self):
        refused(built(entry=avc(AVCC), samples=(b'\xff\xff\xff\xff\x65',)), match='sample 0')

    def test_bytes_after_the_last_unit_too_few_for_a_length(self):
        refused(built(entry=avc(AVCC), samples=(b'\x00\x00\x00\x01\x65\x00\x00',)),
                match='sample 0')

    def test_a_whole_sample_is_its_units_behind_start_codes(self):
        sample = b'\x00\x00\x00\x01\x65\x00\x00\x00\x02\x41\x42'
        with MP4Reader(io.BytesIO(built(entry=avc(AVCC), samples=(sample,)))) as reader:
            assert reader.video.sample(0).data == (b'\x00\x00\x00\x01\x65'
                                                   b'\x00\x00\x00\x01\x41\x42')


class TestBoxesThatAreCutShort:
    """C9: nothing but MP4Error comes out of a file's own bytes."""

    def test_a_sample_description_with_a_count_and_no_entry(self):
        refused(built(tables={'stsd': full_box('stsd', 0, 0, U('>I', 1))}), match='stsd')

    def test_an_empty_sample_description(self):
        refused(built(tables={'stsd': box('stsd')}), match='stsd')

    def test_a_media_timescale_of_zero(self):
        refused(built(timescale=0), match='timescale')

    @pytest.mark.parametrize('rate', [float('nan'), float('inf'), -1.0, 1e300])
    def test_an_audio_sample_rate_that_is_no_rate(self, rate):
        refused(built(entry=audio_entry(rate), handler=b'soun'), match='sample rate')

    def test_an_audio_sample_rate_is_read(self):
        with MP4Reader(io.BytesIO(built(entry=audio_entry(48000.0),
                                        handler=b'soun'))) as reader:
            assert reader.audio[0].sample_rate == 48000
            assert reader.audio[0].channels == 2

    @pytest.mark.parametrize('name', ['mvhd', 'tkhd', 'mdhd', 'hdlr', 'stsz', 'stco',
                                      'stsc', 'stts'])
    def test_a_box_with_no_payload(self, name):
        data = bytearray(built())
        at = data.index(name.encode()) - 4
        size = struct.unpack_from('>I', data, at)[0]
        # the same box, emptied, with a free box taking up the room it had
        data[at:at + size] = box(name) + box('free', bytes(size - 16))
        refused(bytes(data), match=name)

    def test_every_truncation_of_a_small_movie_is_refused_or_read(self):
        data = built(entry=avc(AVCC), samples=(SAMPLE, SAMPLE), sync=[0])
        for cut in range(len(data)):
            try:
                read_everything(data[:cut])
            except MP4Error:
                pass


class TestTheFileIsClosed:
    """C14: a reader that cannot be made does not keep its file."""

    def test_a_path_that_is_no_mp4(self, tmp_path):
        path = tmp_path / 'short.mp4'
        path.write_bytes(built(timescale=0))
        gc.collect()
        gc.disable()
        try:
            before = open_descriptors()
            with pytest.raises(MP4Error):
                MP4Reader(path)
            assert open_descriptors() == before
        finally:
            gc.enable()


def test_a_stated_size_of_nought_is_read_as_stated():
    """D10: the sample entry's size is a statement, and the reader repeats it."""
    with MP4Reader(io.BytesIO(built(entry=visual_entry('vp09', 0, 0)))) as reader:
        assert reader.video.size == (0, 0)


class TestATrackNobodyReads:
    """A track's tables are read when its samples are first asked for."""

    def pcm(self, frames=400_000):
        """An audio track of `frames` four-byte samples in one chunk, as PCM is stored."""
        return built(entry=audio_entry(48000.0), handler=b'soun',
                     samples=(bytes(4 * frames),), tables={
                         'stsz': full_box('stsz', 0, 0, U('>II', 4, frames)),
                         'stts': full_box('stts', 0, 0, U('>III', 1, frames, 1)),
                         'stsc': full_box('stsc', 0, 0, U('>IIII', 1, 1, frames, 1))})

    def test_opening_costs_nothing_for_its_samples(self, tmp_path):
        path = tmp_path / 'pcm.mp4'
        path.write_bytes(self.pcm())
        tracemalloc.start()
        try:
            with MP4Reader(path) as reader:
                assert reader.audio[0].sample_rate == 48000
                assert reader.duration > 0
                repr(reader)
                opened = tracemalloc.get_traced_memory()[1]
                tracemalloc.reset_peak()
                assert len(reader.audio[0]) == 400_000
                assert reader.audio[0].sample(399_999).data == bytes(4)
                assert reader.audio[0].keyframe_before(399_999) == 399_999
                indexed = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert opened < 1 << 20
        # Five arrays of four or eight bytes a sample, and the table boxes.
        assert indexed < 40 * 400_000

    def test_a_table_that_states_too_much_is_refused_when_it_is_read(self):
        data = built(tables={'stts': full_box('stts', 0, 0, U('>I', 0xFFFFFFFF))})
        with MP4Reader(io.BytesIO(data)) as reader:
            for use in (lambda: len(reader.video), lambda: reader.video.sample(0),
                        lambda: reader.video.sync_samples,
                        lambda: reader.video.presentation_time(0)):
                with pytest.raises(MP4Error, match='stts'):
                    use()
