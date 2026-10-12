"""The video codecs this package names, and what each is called elsewhere.

A codec has one name here: ``'h264'``, ``'hevc'``, ``'av1'`` or ``'vp9'``.
:func:`canonical` turns the other spellings a caller may hold into that name:
``'h265'``, ``'avc'``, and the four-character code of an MP4 sample entry such
as ``'hvc1'``, so a track's ``codec`` names its decoder as it stands.

Which of them a machine encodes or decodes is the hardware's to say:
:func:`pyopengl_video.encoders` and :func:`pyopengl_video.decoders` report it.
"""
from __future__ import annotations

import dataclasses

__all__ = ['AV1', 'CODECS', 'Codec', 'H264', 'HEVC', 'VP9', 'canonical',
           'of_sample_entry', 'title']

H264 = 'h264'
HEVC = 'hevc'
AV1 = 'av1'
VP9 = 'vp9'


@dataclasses.dataclass(frozen=True)
class Codec:
    """One codec's names.

    name -- the name this package uses
    title -- the name its specification uses, for messages
    aliases -- other spellings :func:`canonical` accepts
    sample_entries -- the MP4 sample entry codes that hold it, which
        :func:`canonical` accepts too
    """

    name: str
    title: str
    aliases: tuple[str, ...] = ()
    sample_entries: tuple[str, ...] = ()


CODECS: dict[str, Codec] = {
    codec.name: codec for codec in (
        Codec(H264, 'H.264', ('avc', 'h.264', 'x264'), ('avc1', 'avc3')),
        Codec(HEVC, 'HEVC', ('h265', 'h.265', 'x265'), ('hvc1', 'hev1')),
        Codec(AV1, 'AV1', (), ('av01',)),
        Codec(VP9, 'VP9', (), ('vp09',)),
    )
}

_SAMPLE_ENTRIES = {entry: codec.name for codec in CODECS.values()
                   for entry in codec.sample_entries}
_ALIASES = {alias: codec.name for codec in CODECS.values() for alias in codec.aliases}
_ALIASES.update(_SAMPLE_ENTRIES)


def canonical(name: str) -> str:
    """The name this package uses for the codec `name` spells.

    A name that is no codec known here comes back lower-cased, so a backend
    registered by an application is still found under the name it gave.
    """
    lowered = name.lower()
    return _ALIASES.get(lowered, lowered)


def title(name: str) -> str:
    """The codec's name as its specification writes it, for a message."""
    found = CODECS.get(canonical(name))
    return found.title if found else name


def of_sample_entry(code: str) -> str | None:
    """The codec an MP4 sample entry of four-character `code` holds, or None."""
    return _SAMPLE_ENTRIES.get(code)
