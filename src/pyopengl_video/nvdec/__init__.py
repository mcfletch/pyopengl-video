"""NVDEC backends: NVIDIA's video decoder, into OpenGL textures.

NVDEC has no OpenGL device type, so a decoded picture lands in CUDA device
memory and is copied into a pair of textures from there. Two backends differ
in how:

- ``nvdec`` registers the textures with CUDA and copies on the GPU. The
  OpenGL context has to render on the NVIDIA GPU that decodes, which
  ``cuGLGetDevices`` answers; an EGL, a GLX and a WGL context all serve.
- ``nvdec-copy`` copies each picture through host memory, for a context that
  renders on another GPU: a laptop whose integrated GPU draws the window.

Both use the CUDA driver API (``libcuda.so.1``, ``nvcuda.dll``) and the decoder
library (``libnvcuvid.so.1``, ``nvcuvid.dll``), which ship with the NVIDIA
driver; the CUDA toolkit is not involved. See
:mod:`pyopengl_video.nvdec.api` for the binding and
:mod:`pyopengl_video.nvdec.decoder` for the decoder.
"""
from __future__ import annotations

import logging
from typing import Any

from pyopengl_video.backend import Backend, context_kind
from pyopengl_video.nvdec import api

log = logging.getLogger(__name__)

#: H.264 on the parts this backend runs on. The hardware's own limit is asked
#: for when a stream's first picture arrives, and a picture past it is refused
#: by its size.
MAX_SIZE = (4096, 4096)

#: What :func:`driver_reason` found, kept because discovery may be called
#: repeatedly and initialising the CUDA driver is not free. Cleared by
#: :func:`forget_probe`, which the tests use.
_probed: dict[str, str] = {}

#: Where to find the decoder library when the CUDA library loads without it.
DECODER_LIBRARY_MISSING = (
    f'{api.CUVID_LIBRARY_NAME} is not loadable, though {api.CUDA_LIBRARY_NAME} is. '
    'It ships with the NVIDIA driver; Debian and Ubuntu package it apart from '
    'the rest, as libnvidia-decode-<version>')

#: Why a context that renders elsewhere cannot be written by CUDA.
NOT_ON_AN_NVIDIA_GPU = (
    'the current OpenGL context does not render on an NVIDIA GPU, so CUDA '
    "cannot write its textures; the 'nvdec-copy' backend decodes on the NVIDIA "
    'GPU and copies each picture through host memory')


def forget_probe() -> None:
    """Discard what the probes cached, so the next call asks again."""
    _probed.clear()


#: What :func:`driver_reason` answers on a machine with no NVIDIA driver. The
#: explanations leave it out: a machine without the hardware needs no advice.
NO_DRIVER = 'no NVIDIA driver is installed'


def driver_reason() -> str:
    """Why NVDEC cannot run in this process, or '' where it can.

    It can where the CUDA and decoder libraries load, the driver initialises
    and it reports a GPU. Quiet, and cheap after the first call.
    """
    if 'driver' not in _probed:
        _probed['driver'] = _ask_driver()
    return _probed['driver']


def _ask_driver() -> str:
    if not api.SUPPORTED_POINTER_SIZE:
        return 'NVDEC is bound here for 64-bit processes'
    try:
        cuda = api.Cuda.load()
    except OSError as error:
        log.debug('%s is not loadable: %s', api.CUDA_LIBRARY_NAME, error)
        return NO_DRIVER
    except api.CUDAError as error:
        return f'the CUDA driver will not initialise: {error}'
    try:
        api.Cuvid.load()
    except (OSError, AttributeError) as error:
        log.debug('%s is not usable: %s', api.CUVID_LIBRARY_NAME, error)
        return DECODER_LIBRARY_MISSING
    try:
        if not cuda.devices():
            return 'the CUDA driver reports no GPU'
    except api.CUDAError as error:
        return f'the CUDA driver would not list its GPUs: {error}'
    return ''


def _context_reason() -> str:
    """Why CUDA cannot write the current context's textures, or '' where it can.

    No context at all is not held against the machine: discovery often runs
    before a window exists, and the decoder says what is wrong when it is
    opened.
    """
    if context_kind() is None:
        return ''
    return '' if api.Cuda.load().gl_devices() else NOT_ON_AN_NVIDIA_GPU


def probe() -> bool:
    """Can this machine decode H.264 into the current context's textures on the GPU?"""
    try:
        return not (driver_reason() or _context_reason())
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('the NVDEC backend could not answer for this machine: %s', error)
        return False


def copy_probe() -> bool:
    """Can this machine decode H.264 on an NVIDIA GPU at all?"""
    try:
        return not driver_reason()
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('the NVDEC backend could not answer for this machine: %s', error)
        return False


def explain() -> str:
    """Why ``nvdec`` is unavailable here, in a sentence a caller can act on."""
    try:
        reason = driver_reason()
        if reason:
            return '' if reason == NO_DRIVER else reason
        return _context_reason()
    except Exception as error:  # noqa: BLE001 - explaining must not itself fail
        log.debug('the NVDEC backend could not say why it declined: %s', error)
    return ''


def copy_explain() -> str:
    """Why ``nvdec-copy`` is unavailable here, in a sentence a caller can act on."""
    try:
        reason = driver_reason()
    except Exception as error:  # noqa: BLE001 - explaining must not itself fail
        log.debug('the NVDEC backend could not say why it declined: %s', error)
        return ''
    return '' if reason == NO_DRIVER else reason


def _build(interop: bool, **options: Any) -> Any:
    """Construct an :class:`~pyopengl_video.nvdec.decoder.NVDECDecoder`.

    Imported here rather than at module scope so that discovery never imports
    OpenGL.
    """
    from pyopengl_video.nvdec.decoder import (  # noqa: PLC0415 - loaded when a decoder is built
        NVDECDecoder,
    )
    return NVDECDecoder(interop=interop, **options)


def _build_interop(**options: Any) -> Any:
    return _build(True, **options)


def _build_copy(**options: Any) -> Any:
    return _build(False, **options)


DECODER_BACKEND = Backend(
    name='nvdec',
    vendor='NVIDIA',
    codecs=frozenset({'h264'}),
    max_size=MAX_SIZE,
    zero_copy=True,
    probe=probe,
    factory=_build_interop,
    explain=explain,
)

COPY_DECODER_BACKEND = Backend(
    name='nvdec-copy',
    vendor='NVIDIA',
    codecs=frozenset({'h264'}),
    max_size=MAX_SIZE,
    zero_copy=False,
    probe=copy_probe,
    factory=_build_copy,
    explain=copy_explain,
)
