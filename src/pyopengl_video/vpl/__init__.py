"""oneVPL backend: Intel's video encoder, fed a Direct3D 11 surface.

Intel's encoder does not read OpenGL textures, so the frame reaches it through a
surface that is a Direct3D texture and an OpenGL texture at once -- see
:mod:`pyopengl_video.windows.interop`. The encoder allocates those itself, which
is why :meth:`~pyopengl_video.encoder.Encoder.new_input` is the way to get one
here. :mod:`pyopengl_video.vpl.api` is the binding and
:mod:`pyopengl_video.vpl.encoder` the encoder.

Windows only for now. The same library drives Intel hardware on Linux over
VA-API, which is the sibling backend rather than this one; see
``plans/GPU-VIDEO-ENCODE.md``.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from typing import Any

from pyopengl_video.encoder import Backend
from pyopengl_video.vpl.api import LIBRARY_NAME

log = logging.getLogger(__name__)

#: H.264 on Intel hardware. Older parts stop at 4096 in each direction, and
#: nothing here goes beyond that.
MAX_SIZE = (4096, 4096)

#: The interop this backend feeds the encoder through exists on Windows alone.
#: Intel hardware on Linux is reached over VA-API instead.
SUPPORTED_PLATFORM = sys.platform == 'win32'


def probe() -> bool:
    """Can this machine encode H.264 with Intel hardware, from OpenGL?

    Cheap by design, since discovery calls it: the dispatcher is loaded and
    nothing else is asked. That it loads does not prove the part has an encoder
    or that the renderer is on it -- building the encoder is what finds that
    out, and it says which of the two was missing.
    """
    if not SUPPORTED_PLATFORM:
        log.debug('the Direct3D interop this backend needs is Windows-only, not %s',
                  sys.platform)
        return False
    try:
        ctypes.CDLL(LIBRARY_NAME)
    except OSError as error:
        log.debug('%s is not loadable: %s', LIBRARY_NAME, error)
        return False
    return True


def _build(width: int, height: int, **options: Any) -> Any:
    """Construct a :class:`~pyopengl_video.vpl.encoder.VPLEncoder`.

    Imported here rather than at module scope so that discovery costs one
    library load and nothing more.
    """
    from pyopengl_video.vpl.encoder import VPLEncoder
    return VPLEncoder(width, height, **options)


BACKEND = Backend(
    name='vpl',
    vendor='Intel',
    codecs=frozenset({'h264'}),
    max_size=MAX_SIZE,
    zero_copy=True,
    probe=probe,
    factory=_build,
)
