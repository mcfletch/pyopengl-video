"""Hardware video encoding of OpenGL colour buffers.

The renderer has already put the frame in GPU memory, and every GPU we support
has a video encoder on the same die. This package hands one to the other:

    from pyopengl_video import open_encoder
    from pyopengl_video.mp4 import MP4Writer

    with open_encoder(1920, 1080, fps=60, bitrate=12_000_000) as encoder:
        handle = encoder.register(texture)
        with MP4Writer('out.mp4', encoder) as movie:
            for frame in range(600):
                render()
                movie.write(encoder.encode(handle, timestamp=frame * 1500))
            movie.write(encoder.flush())

Which backend that gets you depends on the machine; :func:`encoders` says what
is available and what each one can do.
"""
from pyopengl_video import encoder as _encoder
from pyopengl_video.encoder import (
    Backend,
    Encoder,
    EncoderError,
    EncoderUnavailable,
    Packet,
    available_backends,
)

__version__ = '0.2.0a1'

__all__ = [
    'Backend', 'Encoder', 'EncoderError', 'EncoderUnavailable', 'Packet',
    'encoders', 'open_encoder', '__version__',
]


def encoders() -> list[Backend]:
    """The encoder backends this machine can actually run.

    Each record says which codecs it accepts, how large a frame it will take and
    whether it reaches the encoder without copying through host memory.
    """
    return list(available_backends())


def open_encoder(width: int, height: int, *, codec: str = 'h264',
                 backend: str | None = None, **options: object) -> Encoder:
    """Open an encoder for `width` x `height` frames of `codec`.

    backend -- pick one by name instead of taking the best available
    options -- passed to the backend: ``fps``, ``bitrate``, ``preset``,
        ``tuning``, ``gop``, ``bframes``; each backend documents what it honours

    Raises :class:`EncoderUnavailable` when nothing on this machine fits.
    """
    candidates = list(available_backends())
    if backend is not None:
        candidates = [found for found in candidates if found.name == backend]
        if not candidates:
            raise EncoderUnavailable(_why(
                f'no encoder backend named {backend!r} is available here; '
                f'available: '
                f'{[found.name for found in available_backends()] or "none"}',
                only=backend))
    capable = [found for found in candidates if found.supports(width, height, codec)]
    if not capable:
        raise EncoderUnavailable(_why(
            f'no encoder here can produce {width}x{height} {codec}; available: '
            f'{[(found.name, sorted(found.codecs), found.max_size) for found in candidates] or "none"}'))
    return capable[0].factory(width, height, codec=codec, **options)


def _why(message: str, only: str | None = None) -> str:
    """Add what the unavailable backends had to say for themselves.

    Very often one of them declined for a reason the caller can act on -- an
    OpenGL context of the wrong kind, a driver package not installed -- and a
    bare "none available" sends them looking at their hardware instead.
    """
    reasons = _encoder.explanations()
    if only is not None:
        reasons = [line for line in reasons if line.startswith(f'{only}:')]
    return '\n'.join([message, *(f'  {line}' for line in reasons)])


def _register_builtin_backends() -> None:
    """Add the backends that ship with this package to the registry.

    Importing a backend must not require its hardware: each module exposes a
    cheap probe, and the driver library is only opened when an encoder is built.
    The order is the order :func:`open_encoder` prefers them in.
    """
    from pyopengl_video.nvenc import BACKEND as NVENC_BACKEND
    from pyopengl_video.vaapi import BACKEND as VAAPI_BACKEND
    from pyopengl_video.vpl import BACKEND as VPL_BACKEND
    _encoder.BACKENDS.extend([NVENC_BACKEND, VPL_BACKEND, VAAPI_BACKEND])


_register_builtin_backends()
