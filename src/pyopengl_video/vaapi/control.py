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
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Protocol

from pyopengl_video.colour import Colour
from pyopengl_video.decoder import Sample
from pyopengl_video.vaapi import api

__all__ = ['Buffer', 'CodedPicture', 'DecodeControl', 'DecodeSession',
           'EncodeControl', 'EncodeSettings', 'Format', 'HRD_BUFFER_SECONDS',
           'VBR_TARGET_PERCENTAGE', 'packed_header', 'rate_control_buffers']

#: One parameter buffer: its libva buffer type, and a ctypes object or the
#: bytes it holds.
Buffer = tuple[int, Any]

#: What fraction of the ceiling a variable-bitrate stream aims at.
VBR_TARGET_PERCENTAGE = 70

#: How much video the hypothetical reference decoder's buffer holds, as a
#: multiple of one second's bits.
HRD_BUFFER_SECONDS = 2


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
    """

    size: tuple[int, int]
    frame_rate: tuple[int, int]
    bitrate: int
    gop: int
    qp: int
    rate_control: int


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
    parameters.bit_length = len(data) * 8 if bits is None else bits
    parameters.has_emulation_bytes = int(has_emulation_bytes)
    payload = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    return [(api.VAEncPackedHeaderParameterBufferType, parameters),
            (api.VAEncPackedHeaderDataBufferType, payload)]


def rate_control_buffers(settings: EncodeSettings, initial_qp: int,
                         max_qp: int) -> list[Buffer]:
    """Bitrate, frame rate and buffering, as the miscellaneous parameters of a key picture.

    initial_qp -- the quantiser rate control starts from, on the codec's own scale
    max_qp -- the largest quantiser the codec has
    """
    control = api.VAEncMiscParameterRateControl()
    control.bits_per_second = settings.bitrate
    control.target_percentage = (
        100 if settings.rate_control == api.VA_RC_CBR else VBR_TARGET_PERCENTAGE)
    control.window_size = 1000
    control.initial_qp = initial_qp
    control.min_qp = 0
    control.max_qp = max_qp
    rate = api.VAEncMiscParameterFrameRate()
    # The numerator goes in the low half and the denominator above it; a
    # denominator of zero means the numerator is whole frames a second.
    numerator, denominator = settings.frame_rate
    rate.framerate = numerator | (denominator << 16 if denominator > 1 else 0)
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
