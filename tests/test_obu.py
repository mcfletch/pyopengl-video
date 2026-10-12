"""Splitting an AV1 temporal unit into its OBUs, and putting one back together."""
import pytest

from pyopengl_video import obu
from pyopengl_video.vaapi.bits import leb128
from pyopengl_video.vaapi.h264 import StreamError

DELIMITER = bytes([obu.TEMPORAL_DELIMITER << 3 | 0x02, 0])
SEQUENCE = bytes([obu.SEQUENCE_HEADER << 3 | 0x02, 3]) + b'\x00\x10\x20'
FRAME = bytes([obu.FRAME << 3 | 0x02]) + leb128(200) + bytes(range(200))


class TestSplitting:
    def test_each_unit_comes_out_with_its_type_and_payload(self):
        units = list(obu.split(DELIMITER + SEQUENCE + FRAME))
        assert [unit.kind for unit in units] == [
            obu.TEMPORAL_DELIMITER, obu.SEQUENCE_HEADER, obu.FRAME]
        assert [unit.payload for unit in units] == [b'', b'\x00\x10\x20', bytes(range(200))]
        assert [unit.raw for unit in units] == [DELIMITER, SEQUENCE, FRAME]

    def test_an_extension_byte_gives_the_layer(self):
        unit, = obu.split(bytes([obu.FRAME << 3 | 0x06, 0b0100_1000, 2, 9, 9]))
        assert (unit.temporal_id, unit.spatial_id, unit.has_extension) == (2, 1, True)
        assert unit.payload == b'\x09\x09'

    def test_a_unit_with_no_size_runs_to_the_end(self):
        units = list(obu.split(SEQUENCE + bytes([obu.FRAME << 3]) + b'\x01\x02\x03'))
        assert units[-1].payload == b'\x01\x02\x03'

    def test_nothing_is_no_units(self):
        assert list(obu.split(b'')) == []

    def test_a_unit_cut_short_is_a_stream_error(self):
        with pytest.raises(StreamError, match='states 200 bytes and 49 remain'):
            list(obu.split(FRAME[:52]))

    def test_a_forbidden_bit_is_a_stream_error(self):
        with pytest.raises(StreamError, match='forbidden bit'):
            list(obu.split(b'\x80\x00'))

    def test_a_header_cut_at_its_extension_is_a_stream_error(self):
        with pytest.raises(StreamError, match='inside its header'):
            list(obu.split(bytes([obu.FRAME << 3 | 0x06])))


class TestJoining:
    def test_units_go_back_as_they_were(self):
        data = DELIMITER + SEQUENCE + FRAME
        assert obu.join(obu.split(data)) == data

    def test_a_unit_with_no_size_is_given_one(self):
        unit, = obu.split(bytes([obu.FRAME << 3]) + b'\x01\x02\x03')
        assert unit.encoded() == bytes([obu.FRAME << 3 | 0x02, 3]) + b'\x01\x02\x03'

    def test_an_extension_byte_survives(self):
        data = bytes([obu.FRAME << 3 | 0x06, 0b0100_1000, 2, 9, 9])
        assert obu.join(obu.split(data)) == data

    def test_kinds_named_are_left_out(self):
        assert obu.without(DELIMITER + SEQUENCE + FRAME, obu.TEMPORAL_DELIMITER) == (
            SEQUENCE + FRAME)
        assert obu.without(DELIMITER + FRAME, obu.TEMPORAL_DELIMITER, obu.FRAME) == b''


class TestTheSequenceHeader:
    def test_it_is_found_wherever_it_sits(self):
        assert obu.sequence_header(DELIMITER + SEQUENCE + FRAME) == SEQUENCE

    def test_a_unit_without_one_gives_nothing(self):
        assert obu.sequence_header(DELIMITER + FRAME) == b''
