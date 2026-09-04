"""VA-API backend: Intel and AMD video encoders on Linux.

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
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
from typing import Any

from pyopengl_video.encoder import Backend

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


def _context_can_export() -> bool:
    """Whether a current OpenGL context could hand a texture over, if there is one.

    No context at all is not held against the machine: discovery often runs
    before a window exists, and the encoder says what is wrong when one is
    registered.
    """
    from pyopengl_video.linux import dmabuf

    try:
        from OpenGL import EGL

        if not EGL.eglGetCurrentContext():
            return True
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('EGL is not usable here, so no context is current: %s', error)
        return True
    reason = dmabuf.unavailable_because()
    if reason:
        log.debug('the VA-API backend cannot use this context: %s', reason)
        return False
    return True


def _any_device_encodes() -> bool:
    """Does any DRM render node here offer H.264 slice encoding?"""
    from pyopengl_video.vaapi import api

    try:
        va = api.VA.instance()
    except OSError as error:
        log.debug('%s is not loadable: %s', api.LIBRARY_NAME, error)
        return False
    return any(_device_encodes(va, node) for node in api.render_nodes())


def _device_encodes(va: Any, node: str) -> bool:
    """Does one render node offer H.264 slice encoding? Never raises."""
    from pyopengl_video.vaapi import api

    silence = api.MESSAGE_CALLBACK(lambda context, message: None)
    fd = -1
    display = None
    try:
        fd = os.open(node, os.O_RDWR)
        display = va.vaGetDisplayDRM(fd)
        if not display:
            return False
        va.vaSetInfoCallback(display, silence, None)
        va.vaSetErrorCallback(display, silence, None)
        major, minor = ctypes.c_int(), ctypes.c_int()
        if va.vaInitialize(display, ctypes.byref(major),
                           ctypes.byref(minor)) != api.VA_STATUS_SUCCESS:
            display = None
            return False
        count = va.vaMaxNumEntrypoints(display)
        if count <= 0:
            return False
        entrypoints = (api.VAEntrypoint * count)()
        found = ctypes.c_int(0)
        if va.vaQueryConfigEntrypoints(display, api.VAProfileH264High,
                                       entrypoints,
                                       ctypes.byref(found)) != api.VA_STATUS_SUCCESS:
            return False
        return api.VAEntrypointEncSlice in list(entrypoints)[:found.value]
    except Exception as error:  # noqa: BLE001 - a probe answers, it does not raise
        log.debug('%s does not answer as a VA-API device: %s', node, error)
        return False
    finally:
        if display is not None:
            va.vaTerminate(display)
        if fd >= 0:
            os.close(fd)


def _build(width: int, height: int, **options: Any) -> Any:
    """Construct a :class:`~pyopengl_video.vaapi.encoder.VAAPIEncoder`.

    Imported here rather than at module scope so that discovery costs one
    library load and nothing more.
    """
    from pyopengl_video.vaapi.encoder import VAAPIEncoder
    return VAAPIEncoder(width, height, **options)


BACKEND = Backend(
    name='vaapi',
    vendor='Intel/AMD',
    codecs=frozenset({'h264'}),
    max_size=MAX_SIZE,
    zero_copy=True,
    probe=probe,
    factory=_build,
)
