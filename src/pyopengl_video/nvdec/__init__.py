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

NVIDIA's parser reads H.264, HEVC, VP9 and AV1, so one decoder serves all
four. Which of them a GPU decodes depends on its generation, and
:func:`decoder_codecs` asks it.
"""
from __future__ import annotations

import ctypes
import logging
from typing import Any

from pyopengl_video import codecs
from pyopengl_video.backend import Backend, context_kind
from pyopengl_video.nvdec import api

log = logging.getLogger(__name__)

#: The largest frame assumed of each codec where the GPU cannot be asked.
#: H.264 tops out at 4096 on every part that has NVDEC; the newer codecs go
#: to 8192. The hardware's own limits, the smallest picture among them, are
#: asked for again when a stream's first picture arrives, and a picture
#: outside them is refused by its size.
MAX_SIZE = (8192, 8192)
H264_MAX_SIZE = (4096, 4096)
ASSUMED_SIZES = {name: H264_MAX_SIZE if name == codecs.H264 else MAX_SIZE
                 for name in api.CODECS}

#: What :func:`driver_reason` found, kept because discovery may be called
#: repeatedly and initialising the CUDA driver is not free. Cleared by
#: :func:`forget_probe`, which the tests use.
_probed: dict[str, str] = {}

#: What :func:`decoder_codecs` found, by CUDA device.
_offered: dict[int, dict[str, tuple[int, int]]] = {}

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
    _offered.clear()


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
    """Can this machine decode into the current context's textures on the GPU?"""
    try:
        return not (driver_reason() or _context_reason())
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('the NVDEC backend could not answer for this machine: %s', error)
        return False


def copy_probe() -> bool:
    """Can this machine decode on an NVIDIA GPU at all?"""
    try:
        return not driver_reason()
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('the NVDEC backend could not answer for this machine: %s', error)
        return False


def _device(cuda: api.Cuda) -> int:
    """The CUDA device a decoder opened now would run on."""
    rendering = cuda.gl_devices() if context_kind() is not None else []
    return (rendering or cuda.devices())[0]


def decoder_codecs() -> dict[str, tuple[int, int]]:
    """The codecs this machine's NVIDIA GPU decodes, each with its largest frame.

    The GPU is the one the current OpenGL context renders on, or the first
    where there is no context or it renders elsewhere. It is asked about
    8-bit 4:2:0, which every codec it decodes covers. Asking needs the
    device's CUDA context, which costs about 100 ms the first time and is
    then kept (:meth:`pyopengl_video.nvdec.api.Cuda.context`); the answer is
    kept too. Where the GPU will not answer, every codec the backend drives
    is listed at the size it is assumed to reach, and opening a decoder is
    what refuses one the GPU lacks.
    """
    try:
        cuda, cuvid = api.Cuda.load(), api.Cuvid.load()
        device = _device(cuda)
        if device not in _offered:
            _offered[device] = _ask_codecs(cuda, cuvid, device)
        return dict(_offered[device])
    except Exception as error:  # noqa: BLE001 - discovery answers, it does not raise
        log.debug('the NVIDIA GPU would not list the codecs it decodes: %s', error)
        return dict(ASSUMED_SIZES)


def _ask_codecs(cuda: api.Cuda, cuvid: api.Cuvid, device: int) -> dict[str, tuple[int, int]]:
    found = {}
    with cuda.current(cuda.context(device)):
        for name, codec in api.CODECS.items():
            capabilities = api.CUVIDDECODECAPS(
                eCodecType=codec, eChromaFormat=api.cudaVideoChromaFormat_420)
            cuda.check(cuvid.cuvidGetDecoderCaps(ctypes.byref(capabilities)),
                       'cuvidGetDecoderCaps', codecs.title(name))
            if capabilities.bIsSupported:
                found[name] = (int(capabilities.nMaxWidth), int(capabilities.nMaxHeight))
    return found


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
    codecs=frozenset(api.CODECS),
    max_size=MAX_SIZE,
    sizes={codecs.H264: H264_MAX_SIZE},
    zero_copy=True,
    probe=probe,
    factory=_build_interop,
    explain=explain,
    capabilities=decoder_codecs,
)

COPY_DECODER_BACKEND = Backend(
    name='nvdec-copy',
    vendor='NVIDIA',
    codecs=frozenset(api.CODECS),
    max_size=MAX_SIZE,
    sizes={codecs.H264: H264_MAX_SIZE},
    zero_copy=False,
    probe=copy_probe,
    factory=_build_copy,
    explain=copy_explain,
    capabilities=decoder_codecs,
)
