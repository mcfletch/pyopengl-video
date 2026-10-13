"""What the container readers share: the cursor, positioned reads, the error.

No GPU. A cursor refuses any read its bytes cannot satisfy before it makes
one, and a :class:`FileBytes` refuses any span the file does not hold before
it allocates for it.
"""
import io
import threading

import pytest

from pyopengl_video.backend import VideoError
from pyopengl_video.container import ContainerError, Cursor, FileBytes


class Refused(ContainerError):
    """The error these tests ask for, to see that it is the one raised."""


class Little(Cursor):
    error = Refused
    byteorder = 'little'


class Big(Cursor):
    error = Refused


class TestCursor:
    def test_fields_are_read_in_turn(self):
        cursor = Big(bytes([1, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4]), 'a box')
        assert (cursor.u8(), cursor.u16(), cursor.u32(), cursor.u64()) == (1, 2, 3, 4)
        assert cursor.remaining == 0

    def test_the_byte_order_is_the_containers(self):
        assert Little(b'\x01\x00\x00\x00', 'a header').u32() == 1
        assert Big(b'\x01\x00\x00\x00', 'a header').u32() == 1 << 24

    def test_signed_fields(self):
        assert Big(b'\xff\xff\xff\xfe', 'a box').i32() == -2
        assert Big(b'\xff' * 8, 'a box').i64() == -1

    def test_a_double(self):
        assert Big(b'\x40\xe7\x70\x00\x00\x00\x00\x00', 'a box').f64() == 48000.0

    def test_take_gives_exactly_what_was_asked_for(self):
        cursor = Big(b'abcdef', 'a box')
        assert cursor.take(4) == b'abcd'
        assert cursor.rest() == b'ef'

    @pytest.mark.parametrize('read', ['u8', 'u16', 'u32', 'u64', 'i32', 'i64', 'f64'])
    def test_a_field_past_the_end_is_refused(self, read):
        with pytest.raises(Refused, match='an stsz box'):
            getattr(Big(b'', 'an stsz box'), read)()

    @pytest.mark.parametrize('count', [5, 2 ** 32, 2 ** 70, -1])
    def test_a_length_the_bytes_do_not_hold_is_refused(self, count):
        cursor = Big(b'abcd', 'a record')
        with pytest.raises(Refused, match='a record'):
            cursor.take(count)
        with pytest.raises(Refused, match='a record'):
            cursor.skip(count)
        assert cursor.position == 0

    def test_a_cursor_reads_a_span_of_its_bytes(self):
        cursor = Big(b'abcdef', 'a box', 2, 4)
        assert cursor.remaining == 2
        assert cursor.take(2) == b'cd'
        with pytest.raises(Refused):
            cursor.u8()

    def test_a_table_is_read_whole(self):
        table = Big(b'\x00\x00\x00\x01\x00\x00\x00\x02', 'a table').table('I', 2)
        assert list(table) == [1, 2]

    def test_a_table_longer_than_its_bytes_is_refused_before_it_is_made(self):
        with pytest.raises(Refused, match='a table'):
            Big(b'\x00' * 20, 'a table').table('I', 0xFFFFFFFF)

    def test_a_count_is_held_to_the_entries_that_fit(self):
        assert Big(b'\x00\x00\x00\x02' + bytes(16), 'a table').count(8) == 2
        with pytest.raises(Refused, match='3 entries'):
            Big(b'\x00\x00\x00\x03' + bytes(16), 'a table').count(8)


class TestTheError:
    def test_it_is_a_video_error_and_a_value_error(self):
        assert issubclass(ContainerError, VideoError)
        assert issubclass(ContainerError, ValueError)


@pytest.fixture(params=['path', 'file'])
def opened(request, tmp_path):
    """A FileBytes over the bytes 0 to 255, by path and over a file object."""
    data = bytes(range(256))
    if request.param == 'path':
        path = tmp_path / 'bytes.bin'
        path.write_bytes(data)
        found = FileBytes(path, Refused)
    else:
        found = FileBytes(io.BytesIO(data), Refused)
    yield found
    found.close()


class TestFileBytes:
    def test_it_knows_its_size(self, opened):
        assert opened.size == 256

    def test_a_span_is_read_from_where_it_is(self, opened):
        assert opened.read(10, 4, 'a sample') == bytes([10, 11, 12, 13])
        assert opened.read(0, 0, 'a sample') == b''

    @pytest.mark.parametrize(('offset', 'count'), [
        (250, 10), (256, 1), (0, 257), (-1, 4), (0, -1),
        (0, 2 ** 40), (2 ** 64 - 1, 1), (2 ** 63 - 2, 5)])
    def test_a_span_outside_the_file_is_refused(self, opened, offset, count):
        with pytest.raises(Refused, match='a sample'):
            opened.read(offset, count, 'a sample')

    def test_reading_after_close_is_refused(self, opened):
        opened.close()
        with pytest.raises(Refused, match='closed'):
            opened.read(0, 1, 'a sample')

    def test_a_file_object_stays_its_owners(self):
        handle = io.BytesIO(bytes(8))
        FileBytes(handle, Refused).close()
        assert not handle.closed

    def test_a_file_shortened_after_opening_is_a_refusal(self, tmp_path):
        path = tmp_path / 'bytes.bin'
        path.write_bytes(bytes(64))
        with FileBytes(path, Refused) as found:
            path.write_bytes(bytes(8))
            with pytest.raises(Refused, match='ends'):
                found.read(0, 64, 'a sample')

    def test_a_path_is_read_without_moving_a_shared_position(self, tmp_path):
        path = tmp_path / 'bytes.bin'
        path.write_bytes(bytes(range(256)))
        with FileBytes(path, Refused) as found:
            wrong = []

            def spin(first):
                for step in range(2000):
                    offset = (first + step * 7) % 250
                    if found.read(offset, 4, 'a sample') != bytes(range(offset, offset + 4)):
                        wrong.append(offset)

            threads = [threading.Thread(target=spin, args=(start,)) for start in (0, 100)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert wrong == []


class Interleaving(io.BytesIO):
    """A file whose first seek waits for another thread to seek too.

    With no lock around seek and read, the second thread's seek lands between
    the first thread's seek and its read.
    """

    def __init__(self, data):
        super().__init__(data)
        self.first_seek = threading.Event()
        self.second_seek = threading.Event()
        self.armed = False

    def seek(self, *arguments):
        found = super().seek(*arguments)
        if self.armed:
            if not self.first_seek.is_set():
                self.first_seek.set()
                self.second_seek.wait(0.2)
            else:
                self.second_seek.set()
        return found


def test_two_threads_reading_one_file_object_each_get_their_own_bytes():
    handle = Interleaving(bytes(range(256)))
    found = FileBytes(handle, Refused)
    handle.armed = True
    read = {}

    def first():
        read['first'] = found.read(10, 4, 'a sample')

    def second():
        handle.first_seek.wait(1)
        read['second'] = found.read(200, 4, 'a sample')

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert read == {'first': bytes([10, 11, 12, 13]), 'second': bytes([200, 201, 202, 203])}
