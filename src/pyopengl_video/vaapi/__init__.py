"""VA-API backends: Intel and AMD video encoders and decoders on Linux.

One backend serves both vendors. ``libva`` is the encoder interface on Linux --
Intel parts through the ``iHD`` media driver, AMD parts through Mesa's
``radeonsi`` -- so what differs between them is which driver libva loads, not
what this code does.

The frame reaches the encoder as a DMA-BUF exported from the texture
(:mod:`pyopengl_video.linux.dmabuf`), which means **the OpenGL context has to be
an EGL context**; :func:`probe` says so when it is not. The driver's video
processing block converts RGB to NV12 and the video engine codes it, so nothing
crosses the bus but the compressed result.

libva is a slice-level interface, so the codec decisions are this package's:
each codec has a control layer (:mod:`pyopengl_video.vaapi.control`) holding
them, and :mod:`pyopengl_video.vaapi.encoder` is the plumbing.

Decoding runs the other way (:mod:`pyopengl_video.vaapi.decoder`): the driver
decodes into surfaces exported as DMA-BUFs, which OpenGL imports as textures,
and the codec's control layer supplies the syntax and the picture management
libva leaves to the caller.

Which codecs a machine gets is the device's to say.
:data:`pyopengl_video.vaapi.codecs.ENCODERS` and
:data:`~pyopengl_video.vaapi.codecs.DECODERS` list the codecs with a control
layer, and :func:`encoder_codecs` and :func:`decoder_codecs` the ones the
devices here offer.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

from pyopengl_video.backend import Backend, context_kind
from pyopengl_video.linux import dmabuf
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.codecs import DECODERS, ENCODERS, Support, capabilities

log = logging.getLogger(__name__)

#: The largest frame any codec here reaches on the parts this backend runs on.
#: Discovery replaces it with what the device states for each codec.
MAX_SIZE = (8192, 8192)

#: What H.264 reaches, which is less than the newer codecs do.
H264_MAX_SIZE = (4096, 4096)

#: libva is the Linux interface; the vendors' Windows encoders are reached
#: through Direct3D instead.
SUPPORTED_PLATFORM = sys.platform.startswith('linux')

#: What :func:`probe` found, kept because discovery may be called repeatedly and
#: the answer only changes when the hardware does. Cleared by
#: :func:`forget_probe`, which the tests use.
_probed: dict[str, dict[str, tuple[int, int]]] = {}


def forget_probe() -> None:
    """Discard what :func:`probe` cached, so the next call asks again."""
    _probed.clear()


def probe() -> bool:
    """Can this machine encode from an OpenGL texture with libva?

    Four things have to be true: libva loads, a DRM render node opens, that
    device offers slice encoding for a codec in
    :data:`~pyopengl_video.vaapi.codecs.ENCODERS`, and -- when an OpenGL
    context is current -- that context can export a texture as a DMA-BUF.
    libva is installed on plenty of machines whose GPU has no encoder, so
    asking the device is the only reliable answer.

    Quiet, and cheap after the first call: opening a driver costs enough that
    the answer is kept. A context that cannot export makes this false, because
    such a context genuinely cannot be recorded from; the reason is logged, and
    :func:`pyopengl_video.linux.dmabuf.unavailable_because` states it.
    """
    if not SUPPORTED_PLATFORM:
        log.debug('libva is the Linux encoder interface, and this is %s',
                  sys.platform)
        return False
    try:
        if not _context_can_export():
            return False
        return bool(_offered('encode', ENCODERS))
    except Exception as error:  # noqa: BLE001 - see below
        # Discovery calls this, and a backend that cannot answer is a backend
        # that is not available -- never one that stops a caller finding the
        # others.
        log.debug('the VA-API backend could not answer for this machine: %s',
                  error)
        return False


#: Why a context made through GLX or WGL cannot be served, and what to do.
NOT_EGL = ('this OpenGL context is not an EGL context, so it cannot {} a '
           'DMA-BUF; create the window with '
           'glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)')


def _context_reason(kind: str | None) -> str:
    """Why the context of `kind` cannot export a texture, or '' where it can or there is none."""
    if kind is None:
        return ''
    if kind == 'other':
        return NOT_EGL.format('export a texture as')
    return dmabuf.unavailable_because()


def _context_can_export() -> bool:
    """Whether a current OpenGL context could hand a texture over, if there is one.

    No context at all is not held against the machine: discovery often runs
    before a window exists, and the encoder says what is wrong when one is
    registered.
    """
    reason = _context_reason(context_kind())
    if reason:
        log.debug('the VA-API backend cannot use this context: %s', reason)
        return False
    return True


def _offered(direction: str, table: dict[str, Support]) -> dict[str, tuple[int, int]]:
    """The codecs of `table` the devices here offer, asked once and kept."""
    if direction not in _probed:
        try:
            _probed[direction] = capabilities(table)
        except OSError as error:
            log.debug('%s is not loadable: %s', api.LIBRARY_NAME, error)
            _probed[direction] = {}
    return _probed[direction]


def encoder_codecs() -> dict[str, tuple[int, int]]:
    """The codecs this machine's devices encode, each with its largest frame.

    Empty where libva does not load or no device has an encoder. Quiet, and
    cheap after the first call.
    """
    if not SUPPORTED_PLATFORM:
        return {}
    return dict(_offered('encode', ENCODERS))


def decoder_codecs() -> dict[str, tuple[int, int]]:
    """The codecs this machine's devices decode, each with its largest frame."""
    if not SUPPORTED_PLATFORM:
        return {}
    return dict(_offered('decode', DECODERS))


#: What to install where libva finds no driver with the codecs wanted.
NO_DRIVER = ('libva needs a driver for the GPU beside it -- mesa-va-drivers for '
             'AMD, intel-media-va-driver for Intel -- and `vainfo` will say '
             'whether one is loaded')


def explain() -> str:
    """Why this backend is unavailable here, in a sentence a caller can act on.

    Two of the three reasons are the caller's to fix, and neither is visible
    from the empty list discovery otherwise turns into: an OpenGL context that
    is not an EGL context cannot export a texture, and a machine with `libva`
    but no driver for its GPU has nothing to encode with.
    """
    if not SUPPORTED_PLATFORM:
        return ''
    try:
        reason = _context_reason(context_kind())
        if reason:
            return reason
        if not encoder_codecs():
            return (f'no DRM render node here reports an encoder for any of '
                    f'{", ".join(sorted(ENCODERS))}. {NO_DRIVER}')
    except Exception as error:  # noqa: BLE001 - explaining must not itself fail
        log.debug('the VA-API backend could not say why it declined: %s', error)
    return ''


def _context_is_current() -> bool:
    """Whether an OpenGL context of any binding is current on this thread."""
    return context_kind() is not None


def _build(width: int, height: int, **options: Any) -> Any:
    """Construct a :class:`~pyopengl_video.vaapi.encoder.VAAPIEncoder`.

    Imported here rather than at module scope so that discovery costs one
    library load and nothing more.
    """
    from pyopengl_video.vaapi.encoder import (  # noqa: PLC0415 - loaded when an encoder is built
        VAAPIEncoder,
    )
    return VAAPIEncoder(width, height, **options)


BACKEND = Backend(
    name='vaapi',
    vendor='Intel/AMD',
    codecs=frozenset(ENCODERS),
    max_size=MAX_SIZE,
    sizes={'h264': H264_MAX_SIZE},
    zero_copy=True,
    probe=probe,
    factory=_build,
    explain=explain,
    capabilities=encoder_codecs,
)


# ------------------------------------------------------------------ decoding


def decoder_probe() -> bool:
    """Can this machine decode into OpenGL textures with libva?

    libva loads, a render node offers decoding (``VAEntrypointVLD``) for a
    codec in :data:`~pyopengl_video.vaapi.codecs.DECODERS`, and, when an
    OpenGL context is current, that context can import a DMA-BUF. Quiet, and
    cheap after the first call.
    """
    if not SUPPORTED_PLATFORM:
        return False
    try:
        kind = context_kind()
        if kind == 'other' or (kind == 'egl' and not dmabuf.import_available()):
            log.debug('the VA-API decoder cannot use this context: it cannot '
                      'import a DMA-BUF')
            return False
        return bool(_offered('decode', DECODERS))
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('the VA-API decoder could not answer for this machine: %s', error)
        return False


def decoder_explain() -> str:
    """Why the VA-API decoder is unavailable here, in a sentence a caller can act on."""
    if not SUPPORTED_PLATFORM:
        return ''
    try:
        kind = context_kind()
        if kind == 'other':
            return NOT_EGL.format('import')
        if kind == 'egl' and not dmabuf.import_available():
            return ('this OpenGL context cannot import a DMA-BUF '
                    '(EGL_EXT_image_dma_buf_import); create the window with '
                    'glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)')
        if not decoder_codecs():
            return (f'no DRM render node here reports a decoder for any of '
                    f'{", ".join(sorted(DECODERS))}. {NO_DRIVER}')
    except Exception as error:  # noqa: BLE001 - explaining must not itself fail
        log.debug('the VA-API decoder could not say why it declined: %s', error)
    return ''


def _build_decoder(**options: Any) -> Any:
    """Construct a :class:`~pyopengl_video.vaapi.decoder.VAAPIDecoder`."""
    from pyopengl_video.vaapi.decoder import (  # noqa: PLC0415 - loaded when a decoder is built
        VAAPIDecoder,
    )
    return VAAPIDecoder(**options)


DECODER_BACKEND = Backend(
    name='vaapi',
    vendor='Intel/AMD',
    codecs=frozenset(DECODERS),
    max_size=MAX_SIZE,
    sizes={'h264': H264_MAX_SIZE},
    zero_copy=True,
    probe=decoder_probe,
    factory=_build_decoder,
    explain=decoder_explain,
    capabilities=decoder_codecs,
)
