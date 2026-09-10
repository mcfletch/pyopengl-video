"""NVENC backend: NVIDIA's video encoder, fed straight from an OpenGL texture.

The encoder session is opened with NvEncodeAPI's OpenGL device type, so a
texture is handed over by name and no CUDA context is involved. See
:mod:`pyopengl_video.nvenc.api` for the binding and
:mod:`pyopengl_video.nvenc.encoder` for the encoder itself.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

from pyopengl_video.encoder import Backend
from pyopengl_video.nvenc.api import _LOADER, LIBRARY_NAME

log = logging.getLogger(__name__)

#: H.264 tops out here on every part that has NVENC; HEVC goes further, and
#: raising this is part of adding it.
MAX_SIZE = (4096, 4096)

#: Where NvEncodeAPI accepts an OpenGL device, which is what lets a texture be
#: handed over by name. NVIDIA states this as Linux only.
OPENGL_DEVICE_PLATFORM = sys.platform.startswith('linux')


def probe() -> bool:
    """Can this machine encode from an OpenGL texture with NVENC?

    Cheap by design: discovery calls this, so it loads the driver library and
    asks for nothing else. That the library exists does not prove the part has an
    encoder -- building the encoder is what finds that out.

    NVIDIA supports the encoder's OpenGL device type on Linux alone, so
    elsewhere this backend reports itself unavailable however good the hardware
    is. Reaching the encoder from OpenGL on Windows means going through Direct3D
    or CUDA, or reading the frame back through a Pixel Buffer Object; see
    ``plans/WINDOWS-SUPPORT.md``.
    """
    if not OPENGL_DEVICE_PLATFORM:
        log.debug("NVENC's OpenGL device type is supported only on Linux, not on %s",
                  sys.platform)
        return False
    try:
        _LOADER(LIBRARY_NAME)
    except OSError as error:
        log.debug('%s is not loadable: %s', LIBRARY_NAME, error)
        return False
    return True


def _build(width: int, height: int, **options: Any) -> Any:
    """Construct an :class:`~pyopengl_video.nvenc.encoder.NVENCEncoder`.

    Imported here rather than at module scope so that discovery costs one
    ``dlopen`` and nothing more.
    """
    from pyopengl_video.nvenc.encoder import NVENCEncoder
    return NVENCEncoder(width, height, **options)


BACKEND = Backend(
    name='nvenc',
    vendor='NVIDIA',
    codecs=frozenset({'h264'}),
    max_size=MAX_SIZE,
    zero_copy=True,
    probe=probe,
    factory=_build,
)
