"""The codecs the VA-API backend has a control layer for, and which a device offers.

libva names a codec by its profiles, and a device lists the profiles it
decodes (``VAEntrypointVLD``) and the ones it encodes
(``VAEntrypointEncSlice``). :data:`DECODERS` and :data:`ENCODERS` say which
profiles make each codec usable here and where its control layer is;
:func:`capabilities` asks the devices on this machine which of them they offer.

A control layer is imported when a decoder or an encoder is built, so
discovery loads libva and nothing else.
"""
from __future__ import annotations

import dataclasses
import importlib
import logging
from collections.abc import Iterable, Mapping
from typing import Any

from pyopengl_video import codecs
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.device import Device

log = logging.getLogger(__name__)

__all__ = ['DECODERS', 'DEFAULT_MAX_SIZE', 'ENCODERS', 'Support', 'capabilities',
           'support_for']

#: The largest frame assumed of a device that states no limit of its own.
DEFAULT_MAX_SIZE = (4096, 4096)


@dataclasses.dataclass(frozen=True)
class Support:
    """One codec in one direction: the profiles that carry it and its control layer.

    codec -- the codec's name
    profiles -- the libva profiles the control layer runs under. A device
        offering any of them offers the codec; the first it offers is the one
        its size limit is read from.
    entrypoint -- ``VAEntrypointVLD`` to decode, ``VAEntrypointEncSlice`` to encode
    control -- the control layer, as ``'module:Class'``
    """

    codec: str
    profiles: tuple[int, ...]
    entrypoint: int
    control: str

    @property
    def title(self) -> str:
        return codecs.title(self.codec)

    @property
    def action(self) -> str:
        return 'decoder' if self.entrypoint == api.VAEntrypointVLD else 'encoder'

    def offered(self, device: Device) -> list[int]:
        """The profiles of this codec `device` offers the entrypoint for."""
        return [profile for profile in self.profiles
                if self.entrypoint in device.entrypoints(profile)]

    def objection(self, device: Device) -> str:
        """'' when `device` offers the codec, or why it will not do."""
        if self.offered(device):
            return ''
        return f'no {self.title} {self.action}'

    def largest(self, device: Device) -> tuple[int, int] | None:
        """The largest frame `device` takes in this codec, or None when it has none."""
        offered = self.offered(device)
        if not offered:
            return None
        return device.largest(offered[0], self.entrypoint) or DEFAULT_MAX_SIZE

    def load(self) -> Any:
        """The control layer's class."""
        module, _, name = self.control.partition(':')
        return getattr(importlib.import_module(module), name)


def _table(*entries: Support) -> dict[str, Support]:
    return {entry.codec: entry for entry in entries}


DECODERS = _table(
    Support(codecs.H264,
            (api.VAProfileH264High, api.VAProfileH264Main,
             api.VAProfileH264ConstrainedBaseline),
            api.VAEntrypointVLD, 'pyopengl_video.vaapi.h264_decode:H264Decoding'),
    Support(codecs.HEVC, (api.VAProfileHEVCMain, api.VAProfileHEVCMain10),
            api.VAEntrypointVLD, 'pyopengl_video.vaapi.hevc_decode:HEVCDecoding'),
    Support(codecs.AV1, (api.VAProfileAV1Profile0,), api.VAEntrypointVLD,
            'pyopengl_video.vaapi.av1_decode:AV1Decoding'),
    Support(codecs.VP9, (api.VAProfileVP9Profile0, api.VAProfileVP9Profile2),
            api.VAEntrypointVLD, 'pyopengl_video.vaapi.vp9_decode:VP9Decoding'),
)

ENCODERS = _table(
    Support(codecs.H264, (api.VAProfileH264High,), api.VAEntrypointEncSlice,
            'pyopengl_video.vaapi.h264_encode:H264Encoding'),
    Support(codecs.HEVC, (api.VAProfileHEVCMain,), api.VAEntrypointEncSlice,
            'pyopengl_video.vaapi.hevc_encode:HEVCEncoding'),
    Support(codecs.AV1, (api.VAProfileAV1Profile0,), api.VAEntrypointEncSlice,
            'pyopengl_video.vaapi.av1_encode:AV1Encoding'),
)


def support_for(table: Mapping[str, Support], codec: str) -> Support | None:
    """The entry for `codec`, by any of its names."""
    return table.get(codecs.canonical(codec))


def capabilities(table: Mapping[str, Support],
                 nodes: Iterable[str] | None = None) -> dict[str, tuple[int, int]]:
    """The codecs of `table` this machine's devices offer, each with its largest frame.

    nodes -- the render nodes to ask; every one on the machine by default

    A node that will not open, or that will not answer, offers nothing. Never
    raises once libva has loaded; where it does not load, that is an
    :class:`OSError`.
    """
    va = api.VA.instance()
    found: dict[str, tuple[int, int]] = {}
    for node in (api.render_nodes() if nodes is None else nodes):
        try:
            device = Device(node, va)
        except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
            log.debug('%s does not answer as a VA-API device: %s', node, error)
            continue
        try:
            for codec, support in table.items():
                size = support.largest(device)
                if size is None:
                    continue
                known = found.get(codec, (0, 0))
                found[codec] = (max(known[0], size[0]), max(known[1], size[1]))
        except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
            log.debug('%s would not list what it offers: %s', node, error)
        finally:
            device.close()
    return found
