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
:mod:`pyopengl_video.vaapi.h264` holds them and
:mod:`pyopengl_video.vaapi.encoder` the plumbing.

Decoding runs the other way (:mod:`pyopengl_video.vaapi.decoder`): the driver
decodes into surfaces exported as DMA-BUFs, which OpenGL imports as textures.
:mod:`~pyopengl_video.vaapi.h264_parse` and :mod:`~pyopengl_video.vaapi.h264_dpb`
are the syntax and picture management libva leaves to the caller.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

from pyopengl_video.backend import Backend, context_kind
from pyopengl_video.linux import dmabuf
from pyopengl_video.vaapi import api
from pyopengl_video.vaapi.device import Device

log = logging.getLogger(__name__)

#: H.264 on the parts this backend runs on. A driver that will not go this far
#: refuses the size when the encoder is built, and says so.
MAX_SIZE = (4096, 4096)

#: libva is the Linux interface; the vendors' Windows encoders are reached
#: through Direct3D instead.
SUPPORTED_PLATFORM = sys.platform.startswith('linux')

#: What :func:`probe` found, kept because discovery may be called repeatedly and
#: the answer only changes when the hardware does. Cleared by
#: :func:`forget_probe`, which the tests use.
_probed: dict[str, bool] = {}


def forget_probe() -> None:
    """Discard what :func:`probe` cached, so the next call asks again."""
    _probed.clear()


def probe() -> bool:
    """Can this machine encode H.264 from an OpenGL texture with libva?

    Four things have to be true, and this reports all four honestly: libva
    loads, a DRM render node opens, that device offers H.264 slice encoding,
    and -- when an OpenGL context is current -- that context can export a
    texture as a DMA-BUF. libva is installed on plenty of machines whose GPU
    has no encoder, so asking the device is the only reliable answer.

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
        if 'device' not in _probed:
            _probed['device'] = _any_device_encodes()
        return _probed['device']
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


def _any_device_encodes() -> bool:
    """Does any DRM render node here offer H.264 slice encoding?"""
    try:
        va = api.VA.instance()
    except OSError as error:
        log.debug('%s is not loadable: %s', api.LIBRARY_NAME, error)
        return False
    return any(_device_encodes(va, node) for node in api.render_nodes())


def _device_encodes(va: Any, node: str) -> bool:
    """Does one render node offer H.264 slice encoding? Never raises."""
    return _device_offers(va, node, api.VAEntrypointEncSlice)


def _device_offers(va: Any, node: str, entrypoint: int) -> bool:
    """Does one render node offer `entrypoint` for H.264 High? Never raises."""
    try:
        device = Device(node, va)
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('%s does not answer as a VA-API device: %s', node, error)
        return False
    try:
        return entrypoint in device.entrypoints(api.VAProfileH264High)
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('%s would not list its entrypoints: %s', node, error)
        return False
    finally:
        device.close()


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
        if not _any_device_encodes():
            return (
                'no DRM render node here reports an H.264 encoder. libva needs '
                'a driver for the GPU beside it -- mesa-va-drivers for AMD, '
                'intel-media-va-driver for Intel -- and `vainfo` will say '
                'whether one is loaded'
            )
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
    codecs=frozenset({'h264'}),
    max_size=MAX_SIZE,
    zero_copy=True,
    probe=probe,
    factory=_build,
    explain=explain,
)


# ------------------------------------------------------------------ decoding


def decoder_probe() -> bool:
    """Can this machine decode H.264 into OpenGL textures with libva?

    libva loads, a render node offers H.264 decoding (``VAEntrypointVLD``),
    and, when an OpenGL context is current, that context can import a
    DMA-BUF. Quiet, and cheap after the first call.
    """
    if not SUPPORTED_PLATFORM:
        return False
    try:
        kind = context_kind()
        if kind == 'other' or (kind == 'egl' and not dmabuf.import_available()):
            log.debug('the VA-API decoder cannot use this context: it cannot '
                      'import a DMA-BUF')
            return False
        if 'decoder' not in _probed:
            _probed['decoder'] = _any_device_decodes()
        return _probed['decoder']
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('the VA-API decoder could not answer for this machine: %s', error)
        return False


def _any_device_decodes() -> bool:
    """Does any DRM render node here offer H.264 High decoding?"""
    try:
        va = api.VA.instance()
    except OSError as error:
        log.debug('%s is not loadable: %s', api.LIBRARY_NAME, error)
        return False
    return any(_device_offers(va, node, api.VAEntrypointVLD)
               for node in api.render_nodes())


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
        if not _any_device_decodes():
            return ('no DRM render node here reports an H.264 decoder. libva needs '
                    'a driver for the GPU beside it -- mesa-va-drivers for AMD, '
                    'intel-media-va-driver for Intel -- and `vainfo` will say '
                    'whether one is loaded')
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
    codecs=frozenset({'h264'}),
    max_size=MAX_SIZE,
    zero_copy=True,
    probe=decoder_probe,
    factory=_build_decoder,
    explain=decoder_explain,
)
