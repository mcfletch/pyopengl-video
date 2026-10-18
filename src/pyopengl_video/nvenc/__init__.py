"""NVENC backend: NVIDIA's video encoder, fed straight from an OpenGL texture.

On Linux the encoder session is opened with NvEncodeAPI's OpenGL device type,
so a texture is handed over by name and no CUDA context is involved. NVIDIA
supports that device type on Linux alone; on Windows the session is opened on a
Direct3D 11 device and the frame reaches it through a texture shared between
Direct3D and OpenGL, see :mod:`pyopengl_video.nvenc.direct3d`. See
:mod:`pyopengl_video.nvenc.api` for the binding and
:mod:`pyopengl_video.nvenc.encoder` for the encoder itself.

NVENC is a frame-level interface: the driver writes the headers and manages
the references for H.264, HEVC and AV1 alike. Which of the three a GPU encodes
depends on its generation, and :func:`session_codecs` asks.
"""
from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Iterator
from typing import Any

from pyopengl_video.encoder import Backend
from pyopengl_video.nvenc.api import _LOADER, CODEC_GUIDS, LIBRARY_NAME

log = logging.getLogger(__name__)

#: What :func:`explain` leaves unsaid on a machine with no NVIDIA GPU: a machine
#: without the hardware needs no advice.
NO_NVIDIA_GPU = 'no NVIDIA GPU is installed'

#: The largest frame assumed of each codec before a session has been asked.
#: H.264 tops out at 4096 on every part that has NVENC; HEVC and AV1 go to 8192.
MAX_SIZE = (8192, 8192)
H264_MAX_SIZE = (4096, 4096)
ASSUMED_SIZES = {'h264': H264_MAX_SIZE, 'hevc': MAX_SIZE, 'av1': MAX_SIZE}

#: Where NvEncodeAPI accepts an OpenGL device, which is what lets a texture be
#: handed over by name. NVIDIA states this as Linux only.
OPENGL_DEVICE_PLATFORM = sys.platform.startswith('linux')

#: Where the frame reaches the encoder as a Direct3D 11 texture shared with
#: OpenGL, which is how NVENC is reached from OpenGL on Windows.
DIRECTX_PLATFORM = sys.platform == 'win32'


def _context_reason() -> str:
    """Why the encoder cannot serve the current OpenGL context, or '' where it can.

    On Windows the context must run on an NVIDIA GPU and share Direct3D
    textures, and no context at all is not held against the machine: discovery
    often runs before a window exists, and the encoder says what is wrong when
    it is built. Linux asks nothing here; building the encoder refuses a
    context that is not on the NVIDIA GPU.
    """
    if not DIRECTX_PLATFORM:
        return ''
    from pyopengl_video.nvenc import direct3d  # noqa: PLC0415 - imports the Windows OpenGL bindings

    return direct3d.unavailable_because()


def probe() -> bool:
    """Can this machine encode from OpenGL with NVENC?

    Cheap by design: discovery calls this, so it loads the driver library and,
    on Windows, asks which GPU the current context renders on. That the library
    exists does not prove the part has an encoder -- building the encoder is
    what finds that out.

    NVIDIA supports the encoder's OpenGL device type on Linux alone. On Windows
    the frame reaches the encoder through Direct3D, which needs the OpenGL
    context to be on an NVIDIA GPU; on a hybrid-graphics laptop that is the
    program's graphics preference in Windows' display settings.
    """
    if not (OPENGL_DEVICE_PLATFORM or DIRECTX_PLATFORM):
        log.debug('NVENC is reached from OpenGL on Linux and Windows, not on %s',
                  sys.platform)
        return False
    try:
        _LOADER(LIBRARY_NAME)
    except OSError as error:
        log.debug('%s is not loadable: %s', LIBRARY_NAME, error)
        return False
    try:
        reason = _context_reason()
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('NVENC could not tell which GPU the context is on: %s', error)
        return False
    if reason:
        log.debug('NVENC cannot serve this OpenGL context: %s', reason)
    return not reason


def explain() -> str:
    """Why ``nvenc`` is unavailable here, in a sentence a caller can act on."""
    try:
        _LOADER(LIBRARY_NAME)
    except OSError:
        return ''
    try:
        reason = _context_reason()
    except Exception as error:  # noqa: BLE001 - explaining must not itself fail
        log.debug('NVENC could not say why it declined: %s', error)
        return ''
    return '' if reason == NO_NVIDIA_GPU else reason


@contextlib.contextmanager
def _discovery_device() -> Iterator[Any]:
    """The device a session is opened on to ask what the GPU encodes.

    None on Linux, where the session is opened on the current OpenGL context.
    On Windows an ``ID3D11Device`` on an NVIDIA GPU, closed on leaving.
    """
    if not DIRECTX_PLATFORM:
        yield None
        return
    from pyopengl_video.nvenc import direct3d  # noqa: PLC0415 - imports the Windows OpenGL bindings
    from pyopengl_video.windows import d3d11  # noqa: PLC0415 - the Windows shim

    with d3d11.Device(direct3d.discovery_adapter(), video=False) as device:
        yield device.pointer


def session_codecs() -> dict[str, tuple[int, int]]:
    """The codecs the GPU behind the current OpenGL context encodes.

    Each with the largest frame it takes. On Linux a session can only be opened
    against a current context on the NVIDIA GPU, and on Windows against a
    Direct3D device on one, so where there is none the answer is every codec
    the backend drives at the size it is assumed to reach, and building the
    encoder is what refuses one the GPU lacks.
    """
    from pyopengl_video.nvenc import encoder  # noqa: PLC0415 - loaded once a session is wanted

    try:
        with _discovery_device() as device:
            library = encoder.api.NvEncodeAPI.load()
            session = encoder.open_session(library, device)
            try:
                return encoder.session_codecs(library, session)
            finally:
                library.nvEncDestroyEncoder(session)
    except Exception as error:  # noqa: BLE001 - discovery answers, it does not raise
        log.debug('no NVENC session to ask for its codecs: %s', error)
        return dict(ASSUMED_SIZES)


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
    explain=explain,
    capabilities=session_codecs,
)
