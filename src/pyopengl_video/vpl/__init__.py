"""oneVPL backends: Intel's video encoder and decoder, over Direct3D 11.

Intel's encoder does not read OpenGL textures, so the frame reaches it through a
surface that is a Direct3D texture and an OpenGL texture at once -- see
:mod:`pyopengl_video.windows.interop`. The encoder allocates those itself, which
is why :meth:`~pyopengl_video.encoder.Encoder.new_input` is the way to get one
here. The decoder runs the other way: it decodes into Direct3D surfaces,
converts them to RGB with the session's video processor and writes the result
into shared textures that OpenGL then reads. :mod:`pyopengl_video.vpl.api` is
the binding, :mod:`pyopengl_video.vpl.encoder` the encoder and
:mod:`pyopengl_video.vpl.decoder` the decoder.

Windows only for now. The same library drives Intel hardware on Linux over
VA-API, which is the sibling backend rather than this one; see
``plans/GPU-VIDEO-ENCODE.md``.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from typing import Any

from pyopengl_video.backend import Backend, context_kind
from pyopengl_video.vpl.api import LIBRARY_NAME, SUPPORTED_POINTER_SIZE

log = logging.getLogger(__name__)

#: The largest frame assumed of each codec. H.264 stops at 4096 in each
#: direction; HEVC, AV1 and VP9 reach 8192 on the parts that have them. The
#: runtime refuses a size the part does not reach when the encoder is built.
MAX_SIZE = (8192, 8192)
H264_MAX_SIZE = (4096, 4096)

#: The codecs the backend asks the runtime for. Which of them a part encodes
#: depends on its generation, and building the encoder is what finds out.
CODECS = frozenset({'h264', 'hevc', 'av1', 'vp9'})

#: The interop this backend feeds the encoder through exists on Windows alone.
#: Intel hardware on Linux is reached over VA-API instead.
SUPPORTED_PLATFORM = sys.platform == 'win32'


def probe() -> bool:
    """Can this machine encode with Intel hardware, from OpenGL?

    Cheap by design, since discovery calls it: the dispatcher is loaded and
    nothing else is asked. That it loads does not prove the part has an encoder
    or that the renderer is on it -- building the encoder is what finds that
    out, and it says which of the two was missing.
    """
    if not SUPPORTED_PLATFORM:
        log.debug('the Direct3D interop this backend needs is Windows-only, not %s',
                  sys.platform)
        return False
    if not SUPPORTED_POINTER_SIZE:
        log.debug('oneVPL is bound here for 64-bit processes')
        return False
    try:
        ctypes.CDLL(LIBRARY_NAME)
    except OSError as error:
        log.debug('%s is not loadable: %s', LIBRARY_NAME, error)
        return False
    return True


#: What each codec decodes up to, as the largest frame. The runtime refuses a
#: size the part does not reach when the first picture arrives.
DECODER_SIZES = {'h264': H264_MAX_SIZE, 'hevc': MAX_SIZE, 'vp9': MAX_SIZE, 'av1': MAX_SIZE}

#: The Direct3D 11 decoder profiles that mean a codec is decoded, by the
#: identifier D3D11 lists them with. A GPU lists the profiles it decodes.
DECODER_PROFILES = {
    'h264': ('1b81be68-a0c7-11d3-b984-00c04f2e73c5',),
    'hevc': ('5b11d51b-2f4c-4452-bcc3-09f2a1160cc0', '107af0e0-ef1a-4d19-aba8-67a163073d13'),
    'vp9': ('463707f8-a1d0-4585-876d-83aa6d60b89e', 'a4c749ef-6ecf-48aa-8448-50a7a1165ff7'),
    'av1': ('b8be4ccb-cf53-46ba-8d59-d6b8a6da5d2a',),
}

NO_INTEL_GPU = 'no Intel GPU is installed'

#: Why the decoder is unavailable, for the cases that have a fixable reason.
NOT_AN_INTEL_GPU = (
    'the current OpenGL context does not render on an Intel GPU, so the Intel '
    'decoder could not write its textures')

#: The codecs a GPU decodes, by the LUID of the adapter; the DXGI enumeration
#: and the device behind the profile query are not free.
_profiles: dict[bytes, frozenset[str]] = {}


def forget_probe() -> None:
    """Discard what the decoder probes cached, so the next call asks again."""
    _profiles.clear()


def _decoder_adapter() -> tuple[Any, str]:
    """The Intel adapter a decoder would run on, and why there is none otherwise.

    With an OpenGL context current it is the adapter that context renders on,
    which has to be Intel's and has to share Direct3D textures. With none it is
    the first Intel adapter, since discovery often runs before a window exists.
    """
    from pyopengl_video.windows import (  # noqa: PLC0415 - these import the Windows OpenGL bindings
        d3d11,
        interop,
    )
    if context_kind() is None:
        for installed in d3d11.adapters():
            if installed.vendor == 'Intel':
                return installed, ''
        return None, NO_INTEL_GPU
    if not interop.available():
        return None, f'this OpenGL context does not offer {interop.EXTENSION}'
    adapter = interop.adapter_for_context()
    if adapter is None:
        return None, 'this OpenGL driver does not say which GPU the context is on'
    if adapter.vendor != 'Intel':
        return None, NOT_AN_INTEL_GPU
    return adapter, ''


def decoder_probe() -> bool:
    """Can this machine decode with Intel hardware into the current context's textures?"""
    try:
        return probe() and not _decoder_adapter()[1]
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('the oneVPL decoder could not answer for this machine: %s', error)
        return False


def decoder_explain() -> str:
    """Why the decoder is unavailable here, in a sentence a caller can act on."""
    try:
        if not probe():
            return ''
        reason = _decoder_adapter()[1]
    except Exception as error:  # noqa: BLE001 - explaining must not itself fail
        log.debug('the oneVPL decoder could not say why it declined: %s', error)
        return ''
    return '' if reason == NO_INTEL_GPU else reason


def decoder_capabilities() -> dict[str, tuple[int, int]]:
    """The codecs the Intel GPU here decodes, each with its largest frame."""
    from pyopengl_video.windows import (  # noqa: PLC0415 - these import the Windows OpenGL bindings
        com,
        d3d11,
    )
    adapter, _reason = _decoder_adapter()
    if adapter is None:
        return {}
    if adapter.luid not in _profiles:
        with d3d11.Device(adapter) as device:
            offered = {bytes(profile) for profile in device.decoder_profiles()}
        _profiles[adapter.luid] = frozenset(
            codec for codec, names in DECODER_PROFILES.items()
            if any(bytes(com.guid(name)) in offered for name in names))
    return {codec: DECODER_SIZES[codec] for codec in _profiles[adapter.luid]}


def _build_decoder(**options: Any) -> Any:
    """Construct a :class:`~pyopengl_video.vpl.decoder.VPLDecoder`.

    Imported here for the reason :func:`_build` is.
    """
    from pyopengl_video.vpl.decoder import (  # noqa: PLC0415 - loaded when a decoder is built
        VPLDecoder,
    )
    return VPLDecoder(**options)


def _build(width: int, height: int, **options: Any) -> Any:
    """Construct a :class:`~pyopengl_video.vpl.encoder.VPLEncoder`.

    Imported here rather than at module scope so that discovery costs one
    library load and nothing more.
    """
    from pyopengl_video.vpl.encoder import (  # noqa: PLC0415 - loaded when an encoder is built
        VPLEncoder,
    )
    return VPLEncoder(width, height, **options)


BACKEND = Backend(
    name='vpl',
    vendor='Intel',
    codecs=CODECS,
    max_size=MAX_SIZE,
    sizes={'h264': H264_MAX_SIZE},
    zero_copy=True,
    probe=probe,
    factory=_build,
)


DECODER_BACKEND = Backend(
    name='vpl',
    vendor='Intel',
    codecs=frozenset(DECODER_SIZES),
    max_size=MAX_SIZE,
    sizes=DECODER_SIZES,
    zero_copy=True,
    probe=decoder_probe,
    factory=_build_decoder,
    explain=decoder_explain,
    capabilities=decoder_capabilities,
)
