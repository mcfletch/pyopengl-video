"""The most-significant-first bit reader and writer the AV1 and VP9 headers use."""
import time

import pytest

from pyopengl_video.vaapi.bits import BitReader, BitWriter, leb128, read_leb128
from pyopengl_video.vaapi.h264 import StreamError


class TestReading:
    def test_fields_come_out_most_significant_bit_first(self):
        reader = BitReader(bytes([0b1010_0110, 0b1100_0001]))
        assert [reader.f(3), reader.f(5), reader.flag(), reader.f(7)] == [5, 6, 1, 0x41]
        assert reader.remaining == 0

    def test_reading_past_the_end_is_a_stream_error(self):
        reader = BitReader(b'\xff')
        reader.f(6)
        with pytest.raises(StreamError, match='bit 8'):
            reader.f(3)

    def test_a_twos_complement_field_keeps_its_sign(self):
        reader = BitReader(bytes([0b1111_0111]))
        assert reader.su(4) == -1
        assert reader.su(4) == 7

    def test_a_vp9_signed_value_is_a_magnitude_then_a_sign(self):
        reader = BitReader(bytes([0b0101_0101, 0b0100_0000]))
        assert reader.signed_magnitude(4) == 5
        assert reader.signed_magnitude(4) == -10

    def test_aligning_skips_to_the_next_byte_and_no_further(self):
        reader = BitReader(b'\x80\x7f')
        reader.flag()
        reader.byte_align()
        assert reader.byte_position == 1
        reader.byte_align()
        assert reader.byte_position == 1
        assert reader.f(8) == 0x7f

    def test_a_base_128_value_is_read_from_the_current_byte(self):
        reader = BitReader(b'\xff' + leb128(300) + b'\x05')
        reader.f(8)
        assert reader.leb128() == 300
        assert reader.f(8) == 5


class TestBase128:
    @pytest.mark.parametrize('value', [0, 1, 127, 128, 300, 16383, 16384, 2**32 - 1])
    def test_a_value_survives_the_round_trip(self, value):
        encoded = leb128(value)
        assert read_leb128(encoded) == (value, len(encoded))

    def test_a_fixed_size_pads_with_continuation_bytes(self):
        assert leb128(5, size=4) == b'\x85\x80\x80\x00'
        assert read_leb128(leb128(5, size=4)) == (5, 4)

    def test_a_value_too_large_for_its_fixed_size_is_refused(self):
        with pytest.raises(ValueError, match='needs 3 bytes'):
            leb128(1 << 14, size=2)

    def test_a_value_that_runs_off_the_end_is_a_stream_error(self):
        with pytest.raises(StreamError, match='past the end'):
            read_leb128(b'\x80\x80')

    def test_more_than_eight_bytes_is_a_stream_error(self):
        with pytest.raises(StreamError, match='eight bytes'):
            read_leb128(b'\x80' * 9)


class TestWriting:
    def test_what_is_written_reads_back(self):
        writer = BitWriter()
        writer.f(5, 3)
        writer.flag(True)
        writer.su(-3, 5)
        writer.f(1000, 12)
        assert len(writer) == 21
        reader = BitReader(writer.bytes())
        assert [reader.f(3), reader.flag(), reader.su(5), reader.f(12)] == [5, 1, -3, 1000]

    def test_the_last_byte_is_filled_with_zero_bits(self):
        writer = BitWriter()
        writer.f(0b101, 3)
        assert writer.bytes() == bytes([0b1010_0000])

    def test_nothing_written_is_no_bytes(self):
        assert BitWriter().bytes() == b''

    def test_a_value_that_does_not_fit_its_field_is_refused(self):
        with pytest.raises(ValueError, match='4 bits'):
            BitWriter().f(16, 4)

    def test_aligning_pads_to_a_whole_byte(self):
        writer = BitWriter()
        writer.flag(1)
        writer.byte_align()
        writer.f(0xAB, 8)
        assert writer.bytes() == b'\x80\xab'


class TestHostileReads:
    """What a reader is asked for with a count or a position a stream chose."""

    def test_a_negative_count_is_a_stream_error_and_moves_nothing(self):
        reader = BitReader(b'\xff\x00')
        with pytest.raises(StreamError, match='-1 bits'):
            reader.f(-1)
        assert reader.position == 0

    def test_a_position_before_the_data_is_a_stream_error(self):
        with pytest.raises(StreamError, match='bit -8'):
            BitReader(b'\xff', -8)

    def test_a_position_past_the_data_is_a_stream_error(self):
        reader = BitReader(b'\xff')
        with pytest.raises(StreamError, match='bit 9'):
            reader.position = 9
        assert reader.position == 0

    def test_the_end_of_the_data_is_a_position(self):
        reader = BitReader(b'\xff', 8)
        assert reader.remaining == 0

    def test_a_twos_complement_field_has_a_sign_bit(self):
        with pytest.raises(StreamError, match='sign'):
            BitReader(b'\xff').su(0)

    def test_a_base_128_value_starts_on_a_byte(self):
        reader = BitReader(b'\x80\x01\xff')
        reader.f(3)
        with pytest.raises(StreamError, match='byte'):
            reader.leb128()
        assert reader.position == 3

    def test_a_base_128_value_is_at_most_32_bits(self):
        """leb128() (4.10.5) is at most (1 << 32) - 1."""
        assert read_leb128(b'\xff\xff\xff\xff\x0f') == (2**32 - 1, 5)
        with pytest.raises(StreamError, match='32 bits'):
            read_leb128(b'\x80\x80\x80\x80\x10')
        with pytest.raises(StreamError, match='32 bits'):
            read_leb128(b'\xff' * 7 + b'\x7f')

    def test_a_base_128_value_is_not_read_from_before_the_data(self):
        with pytest.raises(StreamError, match='offset -1'):
            read_leb128(b'\x05\x06', -1)

    def test_a_field_is_read_from_the_bytes_it_spans(self):
        """The cost of a field is its own length, wherever in the data it lies."""
        data = bytes(range(256)) * 4
        reader = BitReader(data)
        for start in (0, 3, 8, 13, 801, 8 * len(data) - 37):
            reader.position = start
            expected = (int.from_bytes(data, 'big') >> (8 * len(data) - start - 37)) & (2**37 - 1)
            assert reader.f(37) == expected
            assert reader.position == start + 37

    def test_reading_far_into_large_data_costs_what_is_read(self):
        data = b'\xa5' * (32 << 20)
        started = time.perf_counter()
        reader = BitReader(data, 8 * len(data) - 8 * 4096)
        total = sum(reader.f(8) for _ in range(4096))
        assert total == 0xA5 * 4096
        assert time.perf_counter() - started < 1.0


class TestHostileWrites:
    def test_a_twos_complement_value_outside_its_field_is_refused(self):
        writer = BitWriter()
        for value in (8, -9, 200):
            with pytest.raises(ValueError, match='4 bits'):
                writer.su(value, 4)
        writer.su(7, 4)
        writer.su(-8, 4)
        assert writer.bytes() == bytes([0b0111_1000])
