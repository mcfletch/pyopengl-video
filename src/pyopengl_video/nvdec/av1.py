"""Which AV1 samples NVIDIA's parser is handed.

NVIDIA's AV1 parser does not hold a stream to the specification before it
acts on it. A sequence header with one bit changed makes it read past the
sample it was given, and an inter frame at the start of a stream gives the
hardware decoder a picture whose references were never decoded; both end the
process with a segmentation fault, the first inside ``cuvidParseVideoData``
where no callback of this package runs. So an AV1 sample is read here first,
and one this package's parser refuses is never handed over.

:class:`Gate` reads each sample as the specification's decoding process does
(AV1 Bitstream & Decoding Process Specification, sections 5 and 7): every OBU
of the temporal unit, each frame header against the sequence header and the
eight reference slots (:mod:`pyopengl_video.vaapi.av1_parse`), each tile
group's sizes against its bytes. It holds no driver and no OpenGL, and is
checked on any machine.

What the parser is handed is what was read, written out again: the temporal
delimiter, the sequence headers, the frame headers and the tile groups, each
OBU with its header in the one form a container stores. Metadata, padding and
OBUs of a reserved type are left out. The decoding process ignores them, the
gate does not read them, and a metadata OBU with no payload ahead of a frame
header is one of the inputs that end the process inside NVIDIA's parser.

Three answers come out of :meth:`Gate.admit`:

- the sample is handed to the parser, as the bytes :meth:`Gate.admit` returns;
- the sample is passed over, because decoding has not reached a key frame
  that is shown: at the start of a stream, after a reset, and after a sample
  that was refused;
- :class:`~pyopengl_video.decoder.DecoderError`, for a sample that breaks the
  specification. Decoding then waits for the next key frame, since the frames
  between predict from one that was not decoded.
"""
from __future__ import annotations

from pyopengl_video import obu
from pyopengl_video.decoder import DecoderError
from pyopengl_video.vaapi.av1_parse import (
    KEY_FRAME,
    FrameHeader,
    FrameState,
    MissingReference,
    SequenceHeader,
    parse_frame_header,
    parse_sequence_header,
    parse_tile_group,
    same_sequence,
)
from pyopengl_video.vaapi.h264 import StreamError

__all__ = ['Gate']

#: chroma_sample_position 3, which the specification reserves (6.4.2).
CSP_RESERVED = 3

#: obu_has_size_field, in an OBU's first byte.
HAS_SIZE_FIELD = 0x02

#: The OBUs a frame is decoded from, which are the ones NVIDIA's parser is
#: handed. The rest carry nothing the decoding process uses (5.3.1): metadata,
#: padding and the reserved types.
DECODED = frozenset({
    obu.SEQUENCE_HEADER, obu.TEMPORAL_DELIMITER, obu.FRAME_HEADER, obu.TILE_GROUP,
    obu.FRAME, obu.REDUNDANT_FRAME_HEADER,
})


class _Frame:
    """A frame whose tile groups are being read.

    header -- the frame's uncompressed header
    copy -- the header's bits as they were coded, which a redundant frame
        header repeats
    next_tile -- the tile the next tile group begins at
    """

    def __init__(self, header: FrameHeader, payload: bytes) -> None:
        self.header = header
        self.copy = _leading_bits(payload, header.header_bits)
        self.next_tile = 0


def _leading_bits(payload: bytes, count: int) -> bytes:
    """The first `count` bits of `payload`, the rest of the last byte cleared."""
    whole, odd = divmod(count, 8)
    if not odd:
        return payload[:whole]
    return payload[:whole] + bytes([payload[whole] & (0xFF << (8 - odd)) & 0xFF])


def _bits_after(payload: bytes, count: int, stop: int | None = None) -> int:
    """The bits of `payload` from bit `count` to byte `stop`, as one integer."""
    whole, odd = divmod(count, 8)
    rest = int.from_bytes(payload[whole:stop], 'big')
    return rest & ((1 << (8 * len(payload[whole:stop]) - odd)) - 1) if odd else rest


def _check_sequence(sequence: SequenceHeader) -> None:
    """Hold a sequence header to what the specification requires and the parser reads past.

    NVIDIA's parser reads a header that breaks one of these as another
    sequence than this package's does, and the frames after it are then
    checked here against a sequence the hardware is not decoding.
    """
    if sequence.reduced_still_picture_header and not sequence.still_picture:
        raise StreamError(
            'the sequence header has the reduced form of a still picture and '
            'says it is no still picture (6.4.1)')
    if sequence.color_config.chroma_sample_position == CSP_RESERVED:
        raise StreamError(
            f'chroma_sample_position is {CSP_RESERVED}, which is reserved (6.4.2)')


def _ends_in_trailing_bits(payload: bytes, bits: int) -> bool:
    """Whether what follows the first `bits` bits of an OBU's payload is its trailing bits.

    A one, and zeroes to the end of the OBU (5.3.4).
    """
    rest = 8 * len(payload) - bits
    return rest >= 1 and _bits_after(payload, bits) == 1 << (rest - 1)


def _sequence_header_bits(payload: bytes, sequence: SequenceHeader) -> int:
    """How many bits of `payload` the sequence header `sequence` was read from.

    The parser reads a header and does not say where it ended, so the end is
    found from what it reads. A header cut a byte short runs past its end,
    which gives the header's last byte. Its last element is
    film_grain_params_present, one bit (5.5.1), so the header's last bit is
    the last one in that byte whose value changes what is read.
    """
    least, most = 1, len(payload)
    while least < most:
        middle = (least + most) // 2
        if _reads_as(payload[:middle], sequence):
            most = middle
        else:
            least = middle + 1
    last = least - 1
    for bit in range(7, -1, -1):
        changed = bytearray(payload)
        changed[last] ^= 0x80 >> bit
        if not _reads_as(bytes(changed), sequence):
            return 8 * last + bit + 1
    raise StreamError('the sequence header is read from none of its last byte')


def _reads_as(payload: bytes, sequence: SequenceHeader) -> bool:
    """Whether `payload` is read as the sequence header `sequence`."""
    try:
        return parse_sequence_header(payload) == sequence
    except Exception:  # noqa: BLE001 - whatever the parser raises of a header cut short or changed
        return False


def _check_sequence_ending(payload: bytes, sequence: SequenceHeader) -> None:
    """Hold what follows a sequence header to the trailing bits that end its OBU.

    NVIDIA's parser reads a header with anything else there as another
    sequence: a stray bit after the trailing one made 4:2:0 at 8 bits into
    4:4:4 at 16.
    """
    if not _ends_in_trailing_bits(payload, _sequence_header_bits(payload, sequence)):
        raise StreamError(
            'the sequence header is not followed by the trailing bits that end its OBU')


def _check_frame_ending(header: FrameHeader, payload: bytes, kind: int) -> None:
    """Hold what follows a frame's header to the padding the specification puts there.

    A frame header OBU ends in trailing bits. In a frame OBU the header is
    followed by zeroes to the next byte, where the tile group begins (5.10).
    A header read to another length than it was written to leaves something
    else in those bits.
    """
    bits = header.header_bits
    if kind == obu.FRAME:
        if _bits_after(payload, bits, (bits + 7) // 8):
            raise StreamError(
                f'the header of the {header.described} is not followed by zeroes '
                'to the byte its tile group begins at')
    elif not _ends_in_trailing_bits(payload, bits):
        raise StreamError(
            f'the header of the {header.described} is not followed by the '
            'trailing bits that end its OBU')


class Gate:
    """Reads each AV1 sample before NVIDIA's parser does, and refuses what breaks the rules.

    One gate follows one parser: :meth:`restart` is called when the parser is
    replaced, and :meth:`lose` when a sample the gate admitted was refused
    further on.
    """

    def __init__(self) -> None:
        self._sequence: SequenceHeader | None = None
        self._sequence_bytes = b''
        #: The sequence header the frames in the reference slots were decoded
        #: under, None while no slot holds a frame.
        self._slots_sequence: SequenceHeader | None = None
        self._state = FrameState()
        #: Whether a key frame has been shown since the references were last lost.
        self._started = False

    @property
    def started(self) -> bool:
        """Whether decoding has reached a key frame that is shown."""
        return self._started

    def configure(self, headers: bytes) -> bytes:
        """Read the configuration a container keeps apart from the samples.

        Returns it as NVIDIA's parser is handed it: its sequence headers,
        each written out as it was read. Raises
        :class:`~pyopengl_video.decoder.DecoderError` for one that is not a
        run of OBUs with a sequence header this package reads, or that
        carries a frame.
        """
        try:
            units = self._units(headers)
            for unit in units:
                if unit.kind == obu.SEQUENCE_HEADER:
                    self._sequence_header(unit.payload, None)
                elif unit.kind in DECODED - {obu.TEMPORAL_DELIMITER}:
                    raise StreamError('it carries a frame, which belongs in a sample')
        except Exception as error:
            raise DecoderError(f'the AV1 configuration is refused: {error}') from error
        return obu.join(unit for unit in units if unit.kind == obu.SEQUENCE_HEADER)

    def admit(self, data: bytes) -> bytes | None:
        """The sample `data` as NVIDIA's parser is handed it, or None where it is not.

        None for a sample passed over while decoding waits for a key frame
        that is shown. The bytes returned are the OBUs a frame is decoded
        from, each written out as it was read, and are empty for a sample
        that holds none. Raises
        :class:`~pyopengl_video.decoder.DecoderError` for a sample that
        breaks the specification, whatever the parser raised in finding that
        out.
        """
        try:
            return obu.join(self._read(data))
        except MissingReference as error:
            waiting = not self._started
            self.lose()
            if waiting:
                return None
            raise DecoderError(f'the AV1 sample is refused: {error}') from error
        except Exception as error:
            self.lose()
            raise DecoderError(f'the AV1 sample is refused: {error}') from error

    def lose(self) -> None:
        """Hold no reference: decoding waits for the next key frame that is shown."""
        self._state.reset()
        self._slots_sequence = None
        self._started = False

    def restart(self) -> None:
        """Forget the stream, its sequence header included: the parser was replaced."""
        self.lose()
        self._sequence = None
        self._sequence_bytes = b''

    # ------------------------------------------------------------ the reading

    @staticmethod
    def _units(data: bytes) -> list[obu.Obu]:
        """The OBUs of `data`, each of which states its size.

        A container stores every OBU with its size field (the low-overhead
        format), and that is the form NVIDIA's parser reads.
        """
        units = list(obu.split(data))
        for unit in units:
            if not unit.raw[0] & HAS_SIZE_FIELD:
                raise StreamError('an OBU has no size field')
            if unit.kind == obu.TILE_LIST:
                raise StreamError(
                    'the stream carries a tile list OBU (large scale tile), which '
                    'this decoder does not decode')
        return units

    def _read(self, data: bytes) -> list[obu.Obu]:
        """Read one temporal unit, following the reference slots through its frames.

        Returns the OBUs its frames are decoded from, in order.
        """
        frame: _Frame | None = None
        read = []
        for unit in self._units(data):
            kind = unit.kind
            if kind not in DECODED or not self._in_operating_point(unit):
                continue
            read.append(unit)
            if kind == obu.SEQUENCE_HEADER:
                self._sequence_header(unit.payload, frame)
            elif kind == obu.TEMPORAL_DELIMITER:
                if unit.payload:
                    raise StreamError(
                        f'a temporal delimiter carries {len(unit.payload)} bytes, '
                        'and it has no payload')
                if frame is not None:
                    raise StreamError(self._unfinished(frame))
            elif kind in (obu.FRAME_HEADER, obu.FRAME):
                if frame is not None:
                    raise StreamError(
                        f'a frame begins before the last tile of the {frame.header.described}')
                frame = self._begin(unit)
                if frame is not None and kind == obu.FRAME:
                    frame = self._tile_group(frame, unit.payload,
                                             (frame.header.header_bits + 7) // 8)
            elif kind == obu.REDUNDANT_FRAME_HEADER:
                self._redundant(frame, unit.payload)
            elif kind == obu.TILE_GROUP:
                if frame is None:
                    raise StreamError('a tile group arrived with no frame header before it')
                frame = self._tile_group(frame, unit.payload, 0)
        if frame is not None:
            raise StreamError(self._unfinished(frame))
        return read

    @staticmethod
    def _unfinished(frame: _Frame) -> str:
        tiles = frame.header.tile_info.num_tiles
        return (f'the temporal unit ends with {frame.next_tile} of the {tiles} tiles '
                f'of the {frame.header.described}')

    def _sequence_header(self, payload: bytes, frame: _Frame | None) -> None:
        if payload == self._sequence_bytes:
            return
        sequence = parse_sequence_header(payload)
        _check_sequence(sequence)
        _check_sequence_ending(payload, sequence)
        held = self._sequence
        if frame is not None and held is not None and not same_sequence(held, sequence):
            raise StreamError(
                'a sequence header that changes the sequence arrived inside a '
                f'frame, between the header of the {frame.header.described} and '
                'its last tile group')
        self._sequence = sequence
        self._sequence_bytes = payload

    def _in_operating_point(self, unit: obu.Obu) -> bool:
        """Whether the operating point decoded includes the unit's layer (5.3.1)."""
        idc = self._sequence.operating_point_idc if self._sequence is not None else 0
        if not unit.has_extension or not idc:
            return True
        return bool((idc >> unit.temporal_id) & 1 and (idc >> (unit.spatial_id + 8)) & 1)

    def _begin(self, unit: obu.Obu) -> _Frame | None:
        """Read a frame's header; None for a header that shows a reference slot."""
        payload = unit.payload
        sequence = self._sequence
        if sequence is None:
            # A frame cannot be read without one, a key frame's type included.
            raise MissingReference('a frame arrived before any sequence header')
        header = parse_frame_header(payload, sequence, self._state,
                                    temporal_id=unit.temporal_id,
                                    spatial_id=unit.spatial_id)
        starts = (header.frame_type == KEY_FRAME and header.show_frame
                  and not header.show_existing_frame)
        if not self._started and not starts:
            raise MissingReference(
                f'decoding starts at a key frame that is shown, and this is a '
                f'{header.described}')
        before = self._slots_sequence
        if not (starts or before is None or before is sequence
                or same_sequence(before, sequence)):
            raise StreamError(
                f'the sequence header changed before a {header.described}, and a '
                'new sequence begins at a key frame that is shown')
        if header.show_existing_frame and unit.kind == obu.FRAME:
            raise StreamError(
                'a frame OBU shows a reference slot, and a frame header OBU alone does (5.10)')
        _check_frame_ending(header, payload, unit.kind)
        if header.show_existing_frame:
            self._finish(header, sequence)
            return None
        return _Frame(header, payload)

    def _redundant(self, frame: _Frame | None, payload: bytes) -> None:
        """Hold a redundant frame header to being a copy of its frame's header (5.9.1)."""
        if frame is None:
            raise StreamError('a redundant frame header arrived with no frame being read')
        bits = frame.header.header_bits
        if len(payload) * 8 < bits or _leading_bits(payload, bits) != frame.copy:
            raise StreamError(
                f'a redundant frame header differs from the header of the '
                f'{frame.header.described}')
        _check_frame_ending(frame.header, payload, obu.REDUNDANT_FRAME_HEADER)

    def _tile_group(self, frame: _Frame, data: bytes, start: int) -> _Frame | None:
        """Read a tile group of `frame`; None once it was the frame's last."""
        header = frame.header
        group = parse_tile_group(data, header, start)
        if group.tg_start != frame.next_tile:
            raise StreamError(
                f'a tile group begins at tile {group.tg_start} where tile '
                f'{frame.next_tile} was due')
        frame.next_tile = group.tg_end + 1
        if frame.next_tile < header.tile_info.num_tiles:
            return frame
        sequence = self._sequence
        if sequence is None:
            raise StreamError('a tile group arrived before any sequence header')
        self._finish(header, sequence)
        return None

    def _finish(self, header: FrameHeader, sequence: SequenceHeader) -> None:
        """The reference frame update process (7.20) for a frame read whole."""
        self._state.update(header, sequence)
        self._slots_sequence = sequence
        if header.frame_type == KEY_FRAME and header.show_frame:
            self._started = True
