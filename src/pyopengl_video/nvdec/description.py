"""What this package reads of a stream for itself, on the NVDEC backends.

NVIDIA's parser reads the bitstream, and reports a sequence when its first
picture arrives. Two things a caller is told come from this package's own
parsers instead:

- the picture size before the first sample, from the configuration a
  container keeps apart from its samples
- the colour description. The parser never reports where chroma samples sit,
  and for VP9 it reports no colour description.

:func:`describe` reads both from the bytes of a container's headers or of a
sample: the last sequence parameter set of an H.264 or HEVC access unit, the
sequence header of an AV1 temporal unit, and the uncompressed header of a VP9
key frame.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable

from pyopengl_video import codecs, obu
from pyopengl_video.colour import Colour
from pyopengl_video.mp4 import split_annexb
from pyopengl_video.vaapi import av1_parse, h264_parse, hevc_parse, vp9_parse

__all__ = ['Description', 'READERS', 'describe']


@dataclasses.dataclass(frozen=True)
class Description:
    """A stream's picture as this package read it.

    size -- the visible picture, as (width, height)
    colour -- its colour description, the matrix chosen where the stream
        states none
    """

    size: tuple[int, int]
    colour: Colour


def _h264(data: bytes) -> Description | None:
    found = None
    for unit in split_annexb(data):
        if unit and unit[0] & 0x1F == h264_parse.NAL_SPS:
            sps = h264_parse.parse_sps(h264_parse.parse_nal(unit).rbsp)
            found = Description(sps.size, h264_parse.colour_of(sps))
    return found


def _hevc(data: bytes) -> Description | None:
    found = None
    for unit in split_annexb(data):
        if unit and (unit[0] >> 1) & 0x3F == hevc_parse.NAL_SPS:
            sps = hevc_parse.parse_sps(hevc_parse.parse_nal(unit).rbsp)
            found = Description(sps.size, hevc_parse.colour_of(sps))
    return found


def _av1(data: bytes) -> Description | None:
    found = None
    for unit in obu.split(data):
        if unit.kind == obu.SEQUENCE_HEADER:
            sequence = av1_parse.parse_sequence_header(unit.payload)
            found = Description(sequence.size, av1_parse.colour_of(sequence))
    return found


def _vp9(data: bytes) -> Description | None:
    for frame in vp9_parse.split_superframe(data):
        if vp9_parse.is_key_frame(frame):
            header = vp9_parse.parse_uncompressed_header(frame, vp9_parse.DecodingState())
            return Description(header.size, vp9_parse.colour_of(header))
    return None


#: The reader of each codec's headers and samples.
READERS: dict[str, Callable[[bytes], Description | None]] = {
    codecs.H264: _h264,
    codecs.HEVC: _hevc,
    codecs.AV1: _av1,
    codecs.VP9: _vp9,
}


def describe(codec: str, data: bytes) -> Description | None:
    """The picture `data` describes, or None where it describes none.

    codec -- a name in :data:`READERS`
    data -- a container's headers or one sample

    Raises :class:`~pyopengl_video.vaapi.h264.StreamError` for a description
    these parsers do not read. The hardware may still decode such a stream of
    H.264, HEVC or VP9, and the decoder then reports what NVIDIA's parser
    found. An AV1 sample these parsers do not read is not handed to NVIDIA's
    (:mod:`pyopengl_video.nvdec.av1`).
    """
    return READERS[codec](data) if data else None
