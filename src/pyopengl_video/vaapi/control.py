"""What a codec's control layer is, for the VA-API decoder and encoder.

libva is a slice-level interface. The driver reconstructs or codes blocks; the
caller parses or writes the headers, orders the pictures and keeps the
references. That part is different for every codec and the same for every
driver, so each codec has a *control layer*: one class for decoding and one
for encoding, holding the codec's rules and no driver.

:class:`~pyopengl_video.vaapi.decoder.VAAPIDecoder` and
:class:`~pyopengl_video.vaapi.encoder.VAAPIEncoder` are the other half: the
device, the surfaces, the OpenGL textures and the calls. A decoding layer is
given a :class:`DecodeSession` to ask for surfaces and submit pictures through,
which is the interface a test stands in for.
:mod:`pyopengl_video.vaapi.codecs` lists the layers there are.
"""
from __future__ import annotations

import ctypes
import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Protocol

from pyopengl_video import bounds
from pyopengl_video.colour import Colour
from pyopengl_video.decoder import DecoderError, Sample
from pyopengl_video.encoder import EncoderError
from pyopengl_video.vaapi import api

__all__ = ['Buffer', 'CodedPicture', 'DecodeControl', 'DecodeSession',
           'EncodeControl', 'EncodeSettings', 'Format', 'HRD_BUFFER_SECONDS',
           'LARGEST_BITRATE', 'LARGEST_SIDE', 'QP_RANGE', 'VBR_TARGET_PERCENTAGE',
           'packed_frame_rate', 'packed_header', 'rate_control_buffers']

#: One parameter buffer: its libva buffer type, and a ctypes object or the
#: bytes it holds.
Buffer = tuple[int, Any]

#: What fraction of the ceiling a variable-bitrate stream aims at.
VBR_TARGET_PERCENTAGE = 70

#: How much video the hypothetical reference decoder's buffer holds, as a
#: multiple of one second's bits.
HRD_BUFFER_SECONDS = 2

#: The largest width or height an encoder takes. libva carries a picture size
#: in 16-bit fields, and a control layer rounds it up to its codec's blocks,
#: the largest of which is 64 samples, so this is the largest multiple of 64
#: that 16 bits hold.
LARGEST_SIDE = bounds.FIELD_16 // 64 * 64

#: The largest bit rate, in bits a second. The buffer of the hypothetical
#: reference decoder is :data:`HRD_BUFFER_SECONDS` times this, in a 32-bit field.
LARGEST_BITRATE = bounds.FIELD_32 // HRD_BUFFER_SECONDS

#: The quantiser's range, on the H.264 scale every control layer is given it in.
QP_RANGE = (0, 51)


@dataclasses.dataclass(frozen=True)
class Format:
    """What a decoded sequence needs of the surfaces it is decoded into.

    profile -- the libva profile the sequence decodes under
    coded_size -- the size the decoder reconstructs, which is the visible
        picture rounded up to the codec's block size
    surfaces -- how many pictures the control layer holds at once: its
        references, the pictures waiting for their turn to be shown, and the
        one being decoded
    depth -- bits a sample, 8 or 10
    name -- the profile in words, for messages
    exact_size -- the surfaces have to be `coded_size` exactly. A driver that
        takes a codec's picture size from the size its context was made at
        decodes other pictures into a larger one, so a device whose smallest
        surface is larger than `coded_size` cannot decode the sequence.
    """

    profile: int
    coded_size: tuple[int, int]
    surfaces: int
    depth: int = 8
    name: str = ''
    exact_size: bool = False


class DecodeSession(Protocol):
    """What the decoder gives a control layer to decode through."""

    def activate(self, format: Format) -> None:
        """Make the surfaces suit `format`, replacing them when they do not.

        Every picture of the format before must have been given to
        :meth:`output` or dropped first: the surfaces it was decoded into are
        given back once the frames lent from them are released.
        """

    def acquire(self) -> int:
        """A surface no picture is held in, ready to be decoded into."""

    def submit(self, surface: int, batches: Sequence[Sequence[Buffer]],
               what: str) -> None:
        """Decode one picture into `surface` from its parameter buffers.

        batches -- each is the buffers of one ``vaRenderPicture`` call
        what -- the picture in words, for an error
        """

    def output(self, surface: int, *, timestamp: int, duration: int,
               visible: tuple[int, int, int, int], colour: Colour) -> None:
        """Show the picture in `surface`: it is the next frame in display order."""


class DecodeControl(ABC):
    """One codec's decoding rules: its syntax, picture order and references.

    A layer parses each sample, activates the format its sequence needs,
    decodes each picture into a surface it acquires, and gives the pictures to
    :meth:`DecodeSession.output` in display order.
    """

    #: the codec decoded, by the name :mod:`pyopengl_video.codecs` gives it
    codec: ClassVar[str]

    #: the visible picture size, (0, 0) until the stream has stated it
    size: tuple[int, int] = (0, 0)
    #: how the stream's samples become RGB
    colour: Colour = Colour()

    def __init__(self, session: DecodeSession, output_slots: int) -> None:
        self.session = session
        self.output_slots = output_slots

    @abstractmethod
    def configure(self, headers: bytes) -> None:
        """Read the configuration a container keeps apart from the samples."""

    @abstractmethod
    def decode(self, sample: Sample) -> None:
        """Decode one sample, giving the session each picture now ready."""

    @abstractmethod
    def flush(self) -> None:
        """Give the session every picture still held back, in display order."""

    @abstractmethod
    def reset(self) -> None:
        """Drop every picture held, before decoding from a seek point."""

    @abstractmethod
    def held(self) -> set[int]:
        """The surfaces whose pictures are still referenced or yet to be shown."""


# --------------------------------------------------------------- encoding


@dataclasses.dataclass(frozen=True)
class EncodeSettings:
    """What a caller asked of an encoder, as its control layer is given it.

    size -- the picture, as (width, height)
    frame_rate -- an exact (numerator, denominator)
    bitrate -- bits per second
    gop -- frames between key frames
    qp -- the quantiser on the H.264 scale of 0 to 51; a layer whose codec
        counts differently converts it
    rate_control -- the libva rate control mode in use, ``VA_RC_CBR`` and so on

    Each is held to the field libva carries it in, so a control layer writes
    what it was given: :class:`~pyopengl_video.encoder.EncoderError` for a
    side outside 1 to :data:`LARGEST_SIDE`, a frame rate
    :func:`packed_frame_rate` refuses, a bit rate outside 1 to
    :data:`LARGEST_BITRATE`, a group of fewer than 1 or more than 2**32 - 1
    frames, and a quantiser outside :data:`QP_RANGE`.
    """

    size: tuple[int, int]
    frame_rate: tuple[int, int]
    bitrate: int
    gop: int
    qp: int
    rate_control: int

    def __post_init__(self) -> None:
        if len(self.size) != 2 or len(self.frame_rate) != 2:
            raise EncoderError(f'a size of {self.size!r} at a frame rate of '
                               f'{self.frame_rate!r} is not two pairs of numbers')
        bounds.frame_size(*self.size, most=(LARGEST_SIDE, LARGEST_SIDE), error=EncoderError)
        packed_frame_rate(self.frame_rate)
        bounds.whole(self.bitrate, 'bitrate', 1, LARGEST_BITRATE, EncoderError)
        if type(self.gop) is int and self.gop < 1:
            raise EncoderError(f'gop is {self.gop}, and a group of {self.gop} '
                               'pictures holds none')
        bounds.whole(self.gop, 'gop', 1, bounds.FIELD_32, EncoderError)
        bounds.whole(self.qp, 'qp', *QP_RANGE, EncoderError)
        bounds.whole(self.rate_control, 'rate_control', 0, bounds.FIELD_32, EncoderError)


@dataclasses.dataclass(frozen=True)
class CodedPicture:
    """One picture an encoding layer has decided on.

    keyframe -- decoding can start here
    name -- the picture in words, for an error
    detail -- the layer's own record of the picture
    """

    keyframe: bool
    name: str
    detail: Any = None


class EncodeControl(ABC):
    """One codec's encoding rules: its headers, picture types and references.

    settings -- what the caller asked for
    stated -- the configuration attributes the driver answered, by type; one
        it did not answer is absent
    """

    #: the codec produced, by the name :mod:`pyopengl_video.codecs` gives it
    codec: ClassVar[str]
    #: the libva profile the layer codes under
    profile: ClassVar[int]
    #: the profile in words, for messages
    title: ClassVar[str]
    #: the configuration attributes the layer wants answered
    attributes: ClassVar[tuple[int, ...]] = ()
    #: the largest quantiser the codec has, which rate control is told
    max_qp: ClassVar[int] = 51

    #: the size the encoder reconstructs, at least the picture's
    coded_size: tuple[int, int]
    #: the packed headers the layer writes, as ``VA_ENC_PACKED_HEADER_*`` bits,
    #: already limited to those the driver takes
    packed_headers: int = 0
    #: reconstructed pictures kept to predict from
    references: int = 1

    @abstractmethod
    def __init__(self, settings: EncodeSettings, stated: dict[int, int]) -> None: ...

    @abstractmethod
    def next_picture(self, force_key: bool = False) -> CodedPicture:
        """Decide what the next picture is."""

    @abstractmethod
    def buffers(self, picture: CodedPicture, coded: int,
                reconstruction: int) -> list[Buffer]:
        """The parameter buffers that code `picture`.

        coded -- the buffer the driver writes the result into
        reconstruction -- the surface the decoded picture is kept in
        """

    @abstractmethod
    def coded(self, picture: CodedPicture, data: bytes) -> bytes:
        """The packet for `picture`, from the bytes the driver wrote for it."""

    @abstractmethod
    def headers(self) -> bytes:
        """The stream's configuration, for a container that stores it up front."""


def packed_header(kind: int, data: bytes, bits: int | None = None, *,
                  has_emulation_bytes: bool = True) -> list[Buffer]:
    """One packed header: the buffer describing it, then its bytes.

    The two go together and in that order; a driver reads the description and
    takes the next data buffer as what it described.

    kind -- which header it is, a ``VAEncPackedHeader*`` type
    bits -- its length in bits, where that is not the whole of `data`
    has_emulation_bytes -- True when `data` already carries the escapes that
        keep a start code out of it, so the driver adds none
    """
    parameters = api.VAEncPackedHeaderParameterBuffer()
    parameters.type = kind
    # The driver reads `bit_length` bits of the data buffer that follows.
    parameters.bit_length = bounds.whole(
        len(data) * 8 if bits is None else bits, 'the length of a packed header, in bits',
        0, min(len(data) * 8, bounds.FIELD_32), EncoderError)
    parameters.has_emulation_bytes = int(has_emulation_bytes)
    payload = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    return [(api.VAEncPackedHeaderParameterBufferType, parameters),
            (api.VAEncPackedHeaderDataBufferType, payload)]


def packed_frame_rate(rate: tuple[int, int]) -> int:
    """A frame rate as ``VAEncMiscParameterFrameRate`` carries it.

    rate -- an exact (numerator, denominator), both above zero

    The numerator is the low 16 bits and the denominator the 16 above them; a
    denominator of zero there means the numerator is whole frames a second.
    The ratio is reduced first, so 90000/3003 is packed as 30000/1001.
    :class:`~pyopengl_video.encoder.EncoderError` for a rate either half of
    which, once reduced, does not fit 16 bits.
    """
    numerator, denominator = bounds.frame_rate(rate, EncoderError)
    divisor = math.gcd(numerator, denominator)
    numerator, denominator = numerator // divisor, denominator // divisor
    if numerator > bounds.FIELD_16 or denominator > bounds.FIELD_16:
        raise EncoderError(
            f'a frame rate of {rate[0]}/{rate[1]} cannot be coded: it reduces to '
            f'{numerator}/{denominator}, and libva carries each half in 16 bits')
    return numerator | (denominator << 16 if denominator > 1 else 0)


def rate_control_buffers(settings: EncodeSettings, initial_qp: int,
                         max_qp: int) -> list[Buffer]:
    """Bitrate, frame rate and buffering, as the miscellaneous parameters of a key picture.

    initial_qp -- the quantiser rate control starts from, on the codec's own scale
    max_qp -- the largest quantiser the codec has
    """
    initial_qp = bounds.whole(initial_qp, 'the initial quantiser', 0, max_qp, EncoderError)
    control = api.VAEncMiscParameterRateControl()
    control.bits_per_second = settings.bitrate
    control.target_percentage = (
        100 if settings.rate_control == api.VA_RC_CBR else VBR_TARGET_PERCENTAGE)
    control.window_size = 1000
    control.initial_qp = initial_qp
    control.min_qp = 0
    control.max_qp = max_qp
    rate = api.VAEncMiscParameterFrameRate()
    rate.framerate = packed_frame_rate(settings.frame_rate)
    hrd = api.VAEncMiscParameterHRD()
    hrd.buffer_size = settings.bitrate * HRD_BUFFER_SECONDS
    hrd.initial_buffer_fullness = hrd.buffer_size // 2
    return [
        (api.VAEncMiscParameterBufferType, api.misc_parameter(kind, payload))
        for kind, payload in (
            (api.VAEncMiscParameterTypeRateControl, control),
            (api.VAEncMiscParameterTypeFrameRate, rate),
            (api.VAEncMiscParameterTypeHRD, hrd),
        )
    ]


# --------------------------------------------- stores a driver reads back
#
# ctypes keeps the low bits of an integer wider than its field and raises
# nothing: 256 stored in a byte reads back as 0. A control layer fills the
# structures a driver indexes its own arrays with, so every store into one is
# read back, through `store`, `store_each` or a `Checked` structure.


def store(target: ctypes.Structure, **values: int) -> None:
    """Assign each of `values` to the field of `target` it names, and read it back.

    ctypes keeps the low bits of an integer wider than its field and raises
    nothing. A field that does not read back as the value assigned raises
    :class:`DecoderError` naming the field, so that no parsed value reaches a
    driver as another number.
    """
    for name, value in values.items():
        setattr(target, name, value)
        if getattr(target, name) != value:
            raise DecoderError(
                f'{name} is {value}, which does not fit the field '
                f'{type(target).__name__} has for it')


def store_each(target: Any, values: Sequence[int], name: str) -> None:
    """Assign `values` to the first elements of the ctypes array `target`,
    each read back as :func:`store` reads a field back.

    name -- what the array holds, for the :class:`DecoderError`
    """
    if len(values) > len(target):
        raise DecoderError(
            f'{len(values)} values for {name}, and libva has room for {len(target)}')
    for index, value in enumerate(values):
        target[index] = value
        if target[index] != value:
            raise DecoderError(
                f'{name} {index} is {value}, which does not fit the array '
                'libva has for it')


class Checked:
    """A ctypes structure or array whose every integer store is read back.

    target -- the structure or array to fill; :attr:`target` hands it back

    A field or an element reads as it would on the target, a structure or an
    array inside it as another :class:`Checked`. An integer that does not read
    back as the value stored, and an element past the end of an array, raise
    :class:`~pyopengl_video.decoder.DecoderError` naming the field::

        parameters = Checked(VADecPictureParameterBufferVP9())
        parameters.frame_width = header.width
        parameters.pic_fields.bits.frame_type = header.frame_type
        return parameters.target
    """

    __slots__ = ('_name', 'target')

    def __init__(self, target: Any, name: str | None = None) -> None:
        object.__setattr__(self, 'target', target)
        object.__setattr__(self, '_name', name or type(target).__name__)

    def _read(self, value: Any, name: str) -> Any:
        if isinstance(value, (ctypes.Structure, ctypes.Union, ctypes.Array)):
            return Checked(value, name)
        return value

    @staticmethod
    def _refuse(name: str, value: int) -> DecoderError:
        return DecoderError(f'{name} is {value}, which does not fit the field '
                            'libva has for it')

    def __getattr__(self, name: str) -> Any:
        return self._read(getattr(self.target, name), f'{self._name}.{name}')

    def __setattr__(self, name: str, value: Any) -> None:
        if not hasattr(self.target, name):
            raise AttributeError(f'{type(self.target).__name__} has no field {name!r}')
        setattr(self.target, name, value)
        if isinstance(value, int) and getattr(self.target, name) != value:
            raise self._refuse(f'{self._name}.{name}', value)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return self.target[index]
        return self._read(self.target[index], f'{self._name}[{index}]')

    def __setitem__(self, index: int | slice, value: Any) -> None:
        if isinstance(index, slice):
            # Element by element, so that each is read back and one past the
            # end is refused; a slice of a ctypes array is otherwise silent.
            places = range(*index.indices(len(self.target)))
            values = list(value)
            if index.stop is not None and index.stop > len(self.target) or (
                    len(values) != len(places)):
                raise DecoderError(
                    f'{len(values)} values for {self._name}, which has room for '
                    f'{len(places)} of its {len(self.target)} entries there')
            for place, element in zip(places, values, strict=True):
                self[place] = element
            return
        name = f'{self._name}[{index}]'
        if not 0 <= index < len(self.target):
            raise DecoderError(f'{name} is past the {len(self.target)} entries '
                               'libva has room for')
        self.target[index] = value
        if isinstance(value, int) and self.target[index] != value:
            raise self._refuse(name, value)
