"""Which AV1 samples reach NVIDIA's parser: the gate, with no GPU.

The gate reads each sample as this package's AV1 parser does and answers one
of three ways: admitted, passed over while decoding waits for a key frame, or
refused. These cases hold it to admitting every sample of every fixture and
to refusing the inputs that ended the process inside NVIDIA's parser (N3 of
``plans/SECURITY-REVIEW.md``), which are the same inputs
``tests/test_nvdec_hostile.py`` decodes on hardware.
"""
import dataclasses

import pytest

from pyopengl_video import DecoderError, obu
from pyopengl_video.nvdec import av1
from pyopengl_video.nvdec.av1 import Gate
from pyopengl_video.vaapi.av1_parse import (
    FrameState,
    StreamError,
    parse_frame_header,
    parse_sequence_header,
)
from tests import nvidia_child

FIXTURES = sorted(path.name for path in nvidia_child.DATA.glob('av1_*.ivf'))


def samples_of(fixture):
    return [sample.data for sample in nvidia_child.samples_of(fixture)]


def saved(name):
    """The samples of one of the review's saved inputs."""
    fixture, changes = nvidia_child.SAVED[name]
    return [sample.data for sample in
            nvidia_child.changed(nvidia_child.samples_of(fixture), changes)]


def answers(gate, samples):
    """What the gate says of each sample: A admitted, a dot passed over, R refused."""
    found = ''
    for data in samples:
        try:
            found += '.' if gate.admit(data) is None else 'A'
        except DecoderError:
            found += 'R'
    return found


def rebuilt(data, change):
    """The temporal unit `data`, each of its OBUs replaced by what `change` returns for it."""
    return obu.join(unit for original in obu.split(data) for unit in change(original))


def frame_split(unit, copies=(), sequence_from=None):
    """A frame OBU as a frame header OBU, redundant copies of it, and a tile group OBU.

    copies -- a function of the header's payload for each redundant frame
        header put between the two
    sequence_from -- the temporal unit whose sequence header the frame is read with
    """
    sequence = parse_sequence_header(next(
        found.payload for found in obu.split(sequence_from)
        if found.kind == obu.SEQUENCE_HEADER))
    bits = parse_frame_header(unit.payload, sequence, FrameState()).header_bits
    whole, odd = divmod(bits, 8)
    header = bytearray(unit.payload[:whole + bool(odd)])
    # The trailing bits of an OBU: a one, then zeroes to the end of the byte.
    if odd:
        header[-1] |= 0x80 >> odd
    else:
        header.append(0x80)
    return [obu.Obu(obu.FRAME_HEADER, bytes(header)),
            *(obu.Obu(obu.REDUNDANT_FRAME_HEADER, copy(bytes(header))) for copy in copies),
            obu.Obu(obu.TILE_GROUP, unit.payload[whole + bool(odd):])]


class TestWholeStreams:
    @pytest.mark.parametrize('fixture', FIXTURES)
    def test_every_sample_of_a_stream_is_handed_over_as_it_is(self, fixture):
        """The fixtures hold nothing but the OBUs a frame is decoded from."""
        gate, samples = Gate(), samples_of(fixture)
        assert [gate.admit(data) for data in samples] == samples

    def test_a_stream_is_admitted_again_after_a_restart(self):
        gate, samples = Gate(), samples_of('av1_large_altref.ivf')
        answers(gate, samples[:7])
        gate.restart()
        assert answers(gate, samples) == 'A' * len(samples)


class TestWaitingForAKeyFrame:
    #: The second key frame of the stream is its sixteenth sample.
    FIXTURE, KEY = 'av1_large_altref.ivf', 15

    @pytest.mark.parametrize('start', [1, 3, 14])
    def test_samples_before_a_key_frame_are_passed_over(self, start):
        samples = samples_of(self.FIXTURE)
        gate = Gate()
        gate.configure(obu.sequence_header(samples[0]))
        assert answers(gate, samples[start:]) == (
            '.' * (self.KEY - start) + 'A' * (len(samples) - self.KEY))

    def test_nothing_has_started_until_a_key_frame_is_shown(self):
        gate, samples = Gate(), samples_of(self.FIXTURE)
        assert not gate.started
        gate.admit(samples[1])
        assert not gate.started
        gate.admit(samples[0])
        assert gate.started

    def test_a_frame_before_any_sequence_header_is_passed_over(self):
        samples = samples_of(self.FIXTURE)
        assert answers(Gate(), samples[1:3]) == '..'

    def test_an_intra_only_frame_is_not_where_decoding_starts(self):
        """It predicts from nothing, and the frames after it predict from frames before it."""
        samples = samples_of('av1_intra_only.ivf')
        gate = Gate()
        gate.configure(obu.sequence_header(samples[0]))
        assert set(answers(gate, samples[1:])) == {'.'}

    def test_a_sample_that_was_refused_is_followed_to_the_next_key_frame(self):
        samples = samples_of(self.FIXTURE)
        samples[4] = samples[4][:len(samples[4]) // 2]
        assert answers(Gate(), samples) == (
            'AAAAR' + '.' * (self.KEY - 5) + 'A' * (len(samples) - self.KEY))

    def test_losing_the_references_waits_for_a_key_frame(self):
        gate, samples = Gate(), samples_of(self.FIXTURE)
        answers(gate, samples[:5])
        gate.lose()
        assert answers(gate, samples[5:]) == (
            '.' * (self.KEY - 5) + 'A' * (len(samples) - self.KEY))


class TestSavedInputs:
    """Each input that reached a segmentation fault is stopped at a sample it changed."""

    @pytest.mark.parametrize('name', sorted(nvidia_child.SAVED))
    def test_a_changed_sample_is_refused(self, name):
        found = answers(Gate(), saved(name))
        first = found.index('R')
        assert first in nvidia_child.SAVED[name][1]
        # Every sample handed over before it is one the gate read whole.
        assert set(found[:first]) <= {'A'}

    def test_the_bit_that_ended_the_process_inside_the_parser_is_refused_at_once(self):
        gate = Gate()
        with pytest.raises(DecoderError, match='the AV1 sample is refused'):
            gate.admit(saved('one bit of the sequence header, order_hint_bits 7 to 5 (N3)')[0])
        assert not gate.started


class TestStructure:
    @pytest.fixture
    def key(self):
        """The first temporal unit of a stream: a delimiter, a sequence header and a key frame."""
        return samples_of('av1_large_altref.ivf')[0]

    def test_bytes_that_are_no_obus_are_refused(self):
        with pytest.raises(DecoderError, match='the AV1 sample is refused'):
            Gate().admit(b'not a stream at all' * 50)

    def test_an_empty_sample_is_admitted_as_nothing(self):
        assert Gate().admit(b'') == b''

    def test_an_obu_with_no_size_field_is_refused(self, key):
        units = list(obu.split(key))
        last = units[-1]
        bare = bytes([last.raw[0] & ~0x02]) + last.payload
        with pytest.raises(DecoderError, match='no size field'):
            Gate().admit(obu.join(units[:-1]) + bare)

    def test_a_frame_cut_short_is_refused(self, key):
        with pytest.raises(DecoderError):
            Gate().admit(key[:len(key) - 40])

    @pytest.mark.parametrize('kind', [obu.METADATA, obu.PADDING, 0, 9, 10, 14])
    @pytest.mark.parametrize('payload', [b'', b'\x01\xff\xff\xff', b'\xff' * 40])
    def test_an_obu_the_decoding_process_ignores_is_not_handed_over(self, key, kind, payload):
        """A metadata OBU of no bytes before a frame header ends the process in NVIDIA's parser."""
        ignored = obu.Obu(kind, payload).encoded()
        units = [unit.encoded() for unit in obu.split(key)]
        mixed = ignored + units[0] + ignored + b''.join(units[1:]) + ignored
        assert Gate().admit(mixed) == key

    def test_a_sample_of_nothing_but_metadata_is_admitted_as_nothing(self):
        assert Gate().admit(obu.Obu(obu.METADATA, b'\x01').encoded()) == b''

    def test_an_obu_header_is_written_out_in_the_form_a_container_stores(self, key):
        """The reserved bit cleared, and the size in as few bytes as hold it."""
        units = list(obu.split(key))
        odd = b''.join(
            bytes([unit.raw[0] | 0x01]) + bytes([len(unit.payload) & 0x7F | 0x80,
                                                  len(unit.payload) >> 7 | 0x80, 0x00])
            + unit.payload for unit in units)
        assert odd != key
        assert Gate().admit(odd) == key

    def test_a_temporal_delimiter_with_a_payload_is_refused(self, key):
        stuffed = obu.Obu(obu.TEMPORAL_DELIMITER, b'\x00').encoded()
        with pytest.raises(DecoderError, match='it has no payload'):
            Gate().admit(stuffed + key)

    def test_a_tile_list_is_refused(self, key):
        with pytest.raises(DecoderError, match='tile list'):
            Gate().admit(key + obu.Obu(obu.TILE_LIST, b'\0' * 8).encoded())

    def test_a_tile_group_with_no_frame_is_refused(self, key):
        gate = Gate()
        assert gate.admit(key)
        with pytest.raises(DecoderError, match='no frame header before it'):
            gate.admit(obu.Obu(obu.TILE_GROUP, b'\0' * 8).encoded())

    def test_a_frame_in_two_obus_is_admitted(self, key):
        split = rebuilt(key, lambda unit: frame_split(unit, sequence_from=key)
                        if unit.kind == obu.FRAME else [unit])
        assert [unit.kind for unit in obu.split(split)][-2:] == [obu.FRAME_HEADER, obu.TILE_GROUP]
        assert Gate().admit(split) == split

    def test_a_temporal_unit_that_ends_inside_a_frame_is_refused(self, key):
        split = rebuilt(key, lambda unit: frame_split(unit, sequence_from=key)[:1]
                        if unit.kind == obu.FRAME else [unit])
        with pytest.raises(DecoderError, match='ends with 0 of the 1 tiles'):
            Gate().admit(split)

    def test_a_second_frame_header_inside_a_frame_is_refused(self, key):
        def twice(unit):
            if unit.kind != obu.FRAME:
                return [unit]
            header, group = frame_split(unit, sequence_from=key)
            return [header, header, group]
        with pytest.raises(DecoderError, match='a frame begins before the last tile'):
            Gate().admit(rebuilt(key, twice))


class TestWhatFollowsAFrameHeader:
    """A header read to another length than it was written to leaves other bits there."""

    @pytest.fixture
    def key(self):
        return samples_of('av1_large_altref.ivf')[0]

    @staticmethod
    def with_header(key, change):
        """The key frame as a frame header OBU and a tile group, the header's payload changed."""
        def changed(unit):
            if unit.kind != obu.FRAME:
                return [unit]
            header, group = frame_split(unit, sequence_from=key)
            return [obu.Obu(obu.FRAME_HEADER, change(header.payload)), group]
        return rebuilt(key, changed)

    def test_trailing_bits_that_end_the_obu_are_admitted(self, key):
        assert Gate().admit(self.with_header(key, bytes)) is not None
        assert Gate().admit(self.with_header(key, lambda header: header + b'\0\0')) is not None

    @pytest.mark.parametrize('change', [
        lambda header: header[:-1] + bytes([header[-1] & (header[-1] - 1)]),
        lambda header: header[:-1] + bytes([header[-1] | 0x01]) + b'\x00',
        lambda header: header + b'\x80',
    ], ids=['no trailing one', 'a bit after it', 'a second one'])
    def test_anything_else_after_the_header_is_refused(self, key, change):
        with pytest.raises(DecoderError, match='not followed by the trailing bits'):
            Gate().admit(self.with_header(key, change))

    def test_a_frame_obu_whose_header_is_not_padded_with_zeroes_is_refused(self, key):
        sequence = parse_sequence_header(next(
            unit.payload for unit in obu.split(key) if unit.kind == obu.SEQUENCE_HEADER))

        def padded(unit):
            if unit.kind != obu.FRAME:
                return [unit]
            bits = parse_frame_header(unit.payload, sequence, FrameState()).header_bits
            assert bits % 8, 'the header ends on a byte, so there is no padding to set'
            payload = bytearray(unit.payload)
            payload[bits // 8] |= 0x01
            return [obu.Obu(obu.FRAME, bytes(payload))]
        with pytest.raises(DecoderError, match='not followed by zeroes'):
            Gate().admit(rebuilt(key, padded))

    def test_a_frame_obu_that_shows_a_reference_slot_is_refused(self):
        """Only a frame header OBU does, and it has no tile group after it."""
        gate, samples = Gate(), samples_of('av1_large_grain.ivf')
        assert answers(gate, samples[:2]) == 'AA'
        shown = rebuilt(samples[2], lambda unit: [obu.Obu(obu.FRAME, unit.payload)]
                        if unit.kind == obu.FRAME_HEADER else [unit])
        assert shown != samples[2]
        with pytest.raises(DecoderError, match='a frame OBU shows a reference slot'):
            gate.admit(shown)


class TestSequenceHeaders:
    """What NVIDIA's parser reads as another sequence than this package's does."""

    def test_the_reduced_form_of_a_header_that_is_no_still_picture_is_refused(self):
        """One bit of a fixture: the parser here read 1x1 at 8 bits, NVIDIA's 4:4:4 at 16."""
        key = bytearray(samples_of('av1_large_monochrome.ivf')[0])
        assert key[4] == 0x00
        key[4] = 0x08
        with pytest.raises(DecoderError, match='still picture'):
            Gate().admit(bytes(key))

    def test_a_still_picture_in_the_reduced_form_is_admitted(self):
        samples = samples_of('av1_still.ivf')
        assert answers(Gate(), samples) == 'A' * len(samples)

    #: The sequence header of ``av1_large_altref.ivf``: 80 bits, then the
    #: trailing one and seven zeroes.
    WHOLE = bytes.fromhex('00 00 00 03 bc ac 89 b5 f3 00 80')

    def test_the_end_of_a_header_is_found_to_the_bit(self):
        assert self.WHOLE == next(
            unit.payload for unit in obu.split(samples_of('av1_large_altref.ivf')[0])
            if unit.kind == obu.SEQUENCE_HEADER)
        assert av1._sequence_header_bits(  # noqa: SLF001 - where the parser stopped reading, which it does not say
            self.WHOLE, parse_sequence_header(self.WHOLE)) == 80
        # One of 78 bits: the trailing one is the seventh bit of its last byte.
        shorter = next(unit.payload for unit in obu.split(samples_of('av1_altref.ivf')[0])
                       if unit.kind == obu.SEQUENCE_HEADER)
        assert av1._sequence_header_bits(shorter, parse_sequence_header(shorter)) == 78  # noqa: SLF001 - as above
        assert shorter[-1] & 0x03 == 0x02

    @pytest.mark.parametrize('payload', [WHOLE, WHOLE + b'\0', WHOLE + b'\0' * 9])
    def test_a_header_that_ends_in_its_trailing_bits_is_admitted(self, payload):
        header = obu.Obu(obu.SEQUENCE_HEADER, payload).encoded()
        assert Gate().configure(header) == header

    @pytest.mark.parametrize('payload', [
        '00 00 00 03 bc ac 89 b5 f3 00 82',
        '00 00 00 03 bc ac 89 b5 f3 00 80 01',
        '00 00 00 03 bc ac 89 b5 f3 00',
        '00 00 00 03 bc ac 89 b5 f3 00 00',
        '00 00 00 02 bc ac 89 b5 f3 00 80',
        '00 00 00 03 7f ac 89 b5 00 00 80',
        '00 00 00 03 81 ac 89 b5 f3 00 80',
    ], ids=['a bit after the trailing one', 'a byte after it', 'no trailing bits',
            'no trailing one', 'two bits shorter', 'seven bits longer', 'another length'])
    def test_a_header_that_ends_anywhere_else_is_refused(self, payload):
        """Each is read here as a sequence, and by NVIDIA's parser as 4:4:4 at 16 bits.

        The last three are the fixture's header with bits changed in the
        widths of its size fields, which moves where the header ends.
        """
        header = obu.Obu(obu.SEQUENCE_HEADER, bytes.fromhex(payload)).encoded()
        with pytest.raises(DecoderError, match='refused'):
            Gate().configure(header)

    def test_the_reserved_chroma_sample_position_is_refused(self):
        key = samples_of('av1_large_altref.ivf')[0]
        sequence = parse_sequence_header(next(
            unit.payload for unit in obu.split(key) if unit.kind == obu.SEQUENCE_HEADER))
        av1._check_sequence(sequence)  # noqa: SLF001 - the rule, apart from the bits that state it
        reserved = dataclasses.replace(sequence, color_config=dataclasses.replace(
            sequence.color_config, chroma_sample_position=3))
        with pytest.raises(StreamError, match='chroma_sample_position is 3'):
            av1._check_sequence(reserved)  # noqa: SLF001 - as above


class TestRedundantFrameHeaders:
    @pytest.fixture
    def key(self):
        return samples_of('av1_large_altref.ivf')[0]

    @staticmethod
    def with_copies(key, *copies):
        return rebuilt(key, lambda unit: frame_split(unit, copies, sequence_from=key)
                       if unit.kind == obu.FRAME else [unit])

    def test_a_copy_of_the_frames_header_is_admitted(self, key):
        copied = self.with_copies(key, bytes, bytes)
        assert Gate().admit(copied) == copied

    def test_one_that_differs_from_the_frames_header_is_refused(self, key):
        def other(header):
            return bytes([header[0], header[1] ^ 0x10, *header[2:]])
        with pytest.raises(DecoderError, match='a redundant frame header differs'):
            Gate().admit(self.with_copies(key, other))

    def test_one_shorter_than_the_frames_header_is_refused(self, key):
        with pytest.raises(DecoderError, match='a redundant frame header differs'):
            Gate().admit(self.with_copies(key, lambda header: header[:2]))

    def test_one_with_no_frame_being_read_is_refused(self, key):
        gate = Gate()
        stray = obu.Obu(obu.REDUNDANT_FRAME_HEADER, b'\x10\x00').encoded()
        with pytest.raises(DecoderError, match='no frame being read'):
            gate.admit(key + stray)


class TestSequences:
    def test_a_new_sequence_that_begins_with_a_key_frame_is_admitted(self):
        first = samples_of('av1_large_monochrome.ivf')
        second = samples_of('av1_large_superres.ivf')
        assert answers(Gate(), first + second) == 'A' * (len(first) + len(second))

    def test_a_new_sequence_header_before_an_inter_frame_is_refused(self):
        first = samples_of('av1_large_altref.ivf')
        other = obu.sequence_header(samples_of('av1_large_grain10.ivf')[0])
        gate = Gate()
        assert answers(gate, first[:3]) == 'AAA'
        with pytest.raises(DecoderError, match='refused'):
            gate.admit(other + first[3])
        assert not gate.started

    def test_a_sequence_header_that_changes_the_sequence_inside_a_frame_is_refused(self):
        key = samples_of('av1_large_altref.ivf')[0]
        other = next(unit for unit in obu.split(samples_of('av1_large_grain10.ivf')[0])
                     if unit.kind == obu.SEQUENCE_HEADER)

        def interrupted(unit):
            if unit.kind != obu.FRAME:
                return [unit]
            header, group = frame_split(unit, sequence_from=key)
            return [header, dataclasses.replace(other), group]
        with pytest.raises(DecoderError, match='arrived inside a frame'):
            Gate().admit(rebuilt(key, interrupted))


class TestConfiguration:
    def test_a_sequence_header_kept_apart_from_the_samples_is_read(self):
        samples = samples_of('av1_large_altref.ivf')
        gate = Gate()
        gate.configure(obu.sequence_header(samples[0]))
        bare = obu.without(samples[0], obu.SEQUENCE_HEADER)
        assert gate.admit(bare) == bare

    def test_the_parser_is_handed_its_sequence_headers_and_nothing_else(self):
        header = obu.sequence_header(samples_of('av1_large_altref.ivf')[0])
        metadata = obu.Obu(obu.METADATA, b'').encoded()
        assert Gate().configure(metadata + header + metadata) == header

    def test_no_configuration_is_no_fault(self):
        assert Gate().configure(b'') == b''

    @pytest.mark.parametrize('headers', [b'\xff' * 12, b'\x0a\x7f', b'\x0a\x03\xff\xff\xff'])
    def test_one_that_is_no_sequence_header_is_refused(self, headers):
        with pytest.raises(DecoderError, match='the AV1 configuration is refused'):
            Gate().configure(headers)

    def test_one_that_carries_a_frame_is_refused(self):
        with pytest.raises(DecoderError, match='carries a frame'):
            Gate().configure(samples_of('av1_large_altref.ivf')[0])
