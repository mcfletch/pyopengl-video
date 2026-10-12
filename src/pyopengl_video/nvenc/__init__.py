"""NVENC backend: NVIDIA's video encoder, fed straight from an OpenGL texture.

The encoder session is opened with NvEncodeAPI's OpenGL device type, so a
texture is handed over by name and no CUDA context is involved. See
:mod:`pyopengl_video.nvenc.api` for the binding and
:mod:`pyopengl_video.nvenc.encoder` for the encoder itself.

NVENC is a frame-level interface: the driver writes the headers and manages
the references for H.264, HEVC and AV1 alike. Which of the three a GPU encodes
depends on its generation, and :func:`session_codecs` asks.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

from pyopengl_video.encoder import Backend
from pyopengl_video.nvenc.api import _LOADER, CODEC_GUIDS, LIBRARY_NAME

log = logging.getLogger(__name__)

#: The largest frame assumed of each codec before a session has been asked.
#: H.264 tops out at 4096 on every part that has NVENC; HEVC and AV1 go to 8192.
MAX_SIZE = (8192, 8192)
H264_MAX_SIZE = (4096, 4096)
ASSUMED_SIZES = {'h264': H264_MAX_SIZE, 'hevc': MAX_SIZE, 'av1': MAX_SIZE}

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


def session_codecs() -> dict[str, tuple[int, int]]:
    """The codecs the GPU behind the current OpenGL context encodes.

    Each with the largest frame it takes. A session can only be opened against
    a current context on the NVIDIA GPU, so without one the answer is every
    codec the backend drives at the size it is assumed to reach, and building
    the encoder is what refuses one the GPU lacks.
    """
    from pyopengl_video.nvenc import encoder  # noqa: PLC0415 - loaded once a session is wanted

    try:
        library = encoder.api.NvEncodeAPI.load()
        session = encoder.open_session(library)
    except Exception as error:  # noqa: BLE001 - discovery answers, it does not raise
        log.debug('no NVENC session to ask for its codecs: %s', error)
        return dict(ASSUMED_SIZES)
    try:
        return encoder.session_codecs(library, session)
    except Exception as error:  # noqa: BLE001 - discovery answers, it does not raise
        log.debug('the NVENC session would not list its codecs: %s', error)
        return dict(ASSUMED_SIZES)
    finally:
        library.nvEncDestroyEncoder(session)


def _build(width: int, height: int, **options: Any) -> Any:
    """Construct an :class:`~pyopengl_video.nvenc.encoder.NVENCEncoder`.

    Imported here rather than at module scope so that discovery costs one
    ``dlopen`` and nothing more.
    """
    from pyopengl_video.nvenc.encoder import (  # noqa: PLC0415 - loaded when an encoder is built
        NVENCEncoder,
    )
    return NVENCEncoder(width, height, **options)


BACKEND = Backend(
    name='nvenc',
    vendor='NVIDIA',
    codecs=frozenset(CODEC_GUIDS),
    max_size=MAX_SIZE,
    sizes={'h264': H264_MAX_SIZE},
    zero_copy=True,
    probe=probe,
    factory=_build,
    capabilities=session_codecs,
)
