"""Hardware video encoding of OpenGL colour buffers, and decoding into textures.

The renderer has already put the frame in GPU memory, and every GPU we support
has a video encoder on the same die. This package hands one to the other::

    from pyopengl_video import open_encoder
    from pyopengl_video.mp4 import MP4Writer

    with open_encoder(1920, 1080, fps=60, bitrate=12_000_000) as encoder:
        handles = [encoder.new_input() for _ in range(encoder.input_slots)]
        with MP4Writer('out.mp4', encoder) as movie:
            for frame in range(600):
                handle = handles[frame % len(handles)]
                with handle.for_drawing():
                    render_into(handle.framebuffer)
                movie.write(encoder.encode(handle, timestamp=frame * 1500))
            movie.write(encoder.flush())

Playing a movie back is the reverse: a container reader, a decoder and a
conversion pass into an RGBA texture, which :class:`~pyopengl_video.playback.Movie`
puts together::

    from pyopengl_video.playback import Movie

    movie = Movie.open('intro.mp4')         # any thread: reads the index
    player = movie.player()                 # GL thread: decoder and textures
    texture = player.frame_at(seconds)

Which backend that gets you depends on the machine; :func:`encoders` and
:func:`decoders` say what is available and what each one can do.
"""
from pyopengl_video import codecs as _codecs
from pyopengl_video import decoder as _decoder
from pyopengl_video import encoder as _encoder
from pyopengl_video.backend import Backend, VideoError, describe, with_reasons
from pyopengl_video.decoder import (
    DecodedFrame,
    Decoder,
    DecoderError,
    DecoderUnavailable,
    Sample,
)
from pyopengl_video.encoder import (
    Encoder,
    EncoderError,
    EncoderUnavailable,
    Packet,
    available_backends,
)
from pyopengl_video.nvdec import COPY_DECODER_BACKEND as NVDEC_COPY_DECODER_BACKEND
from pyopengl_video.nvdec import DECODER_BACKEND as NVDEC_DECODER_BACKEND
from pyopengl_video.nvenc import BACKEND as NVENC_BACKEND
from pyopengl_video.vaapi import BACKEND as VAAPI_BACKEND
from pyopengl_video.vaapi import DECODER_BACKEND as VAAPI_DECODER_BACKEND
from pyopengl_video.vpl import BACKEND as VPL_BACKEND
from pyopengl_video.vpl import DECODER_BACKEND as VPL_DECODER_BACKEND

__version__ = '1.0.0a4'

__all__ = [
    'Backend', 'DecodedFrame', 'Decoder', 'DecoderError', 'DecoderUnavailable',
    'Encoder', 'EncoderError', 'EncoderUnavailable', 'Packet', 'Sample',
    'VideoError', 'decoders', 'encoders', 'open_decoder', 'open_encoder',
    '__version__',
]


def encoders() -> list[Backend[Encoder]]:
    """The encoder backends this machine can actually run.

    Each record says which codecs the hardware here encodes through it, how
    large a frame it takes in each (:meth:`~pyopengl_video.backend.Backend.largest`)
    and whether it reaches the encoder without copying through host memory.
    """
    return list(available_backends())


def open_encoder(width: int, height: int, *, codec: str = 'h264',
                 backend: str | None = None, **options: object) -> Encoder:
    """Open an encoder for `width` x `height` frames of `codec`.

    codec -- ``'h264'``, ``'hevc'`` or ``'av1'``, where this machine's
        hardware encodes it; :func:`encoders` says which it does
    backend -- pick one by name instead of taking the best available
    options -- passed to the backend: ``fps``, ``bitrate``, ``preset``,
        ``tuning``, ``gop``, ``bframes``; each backend documents what it honours

    Raises :class:`EncoderUnavailable` when nothing on this machine fits.
    """
    codec = _codecs.canonical(codec)
    present = list(available_backends())
    candidates = present
    if backend is not None:
        candidates = [found for found in present if found.name == backend]
        if not candidates:
            raise EncoderUnavailable(with_reasons(
                f'no encoder backend named {backend!r} is available here; '
                f'available: {[found.name for found in present] or "none"}',
                _encoder.BACKENDS, only=backend))
    capable = [found for found in candidates if found.supports(width, height, codec)]
    if not capable:
        raise EncoderUnavailable(with_reasons(
            f'no encoder here can produce {width}x{height} {codec}; available: '
            f'{describe(candidates)}', _encoder.BACKENDS))
    return capable[0].factory(width, height, codec=codec, **options)


def decoders() -> list[Backend[Decoder]]:
    """The decoder backends this machine can actually run.

    Each record says which codecs the hardware here decodes through it.
    """
    return list(_decoder.available_backends())


def open_decoder(codec: str = 'h264', *, headers: bytes = b'',
                 size: tuple[int, int] | None = None, backend: str | None = None,
                 **options: object) -> Decoder:
    """Open a decoder for a stream of `codec`.

    codec -- ``'h264'``, ``'hevc'``, ``'av1'`` or ``'vp9'``, where this
        machine's hardware decodes it; :func:`decoders` says which it does
    headers -- the stream's configuration, when the container stores it apart
        from the samples: parameter sets as Annex-B for H.264 and HEVC, the
        configuration OBUs for AV1. A decoder can then report
        :attr:`~Decoder.size` before the first sample.
    size -- the picture size, when known, so a backend that cannot reach it is
        passed over; it is also given to the backend
    backend -- pick one by name instead of taking the best available
    options -- passed to the backend, which documents what it honours

    Raises :class:`DecoderUnavailable` when nothing on this machine fits.
    """
    codec = _codecs.canonical(codec)
    present = list(_decoder.available_backends())
    candidates = present
    if backend is not None:
        candidates = [found for found in present if found.name == backend]
        if not candidates:
            raise DecoderUnavailable(with_reasons(
                f'no decoder backend named {backend!r} is available here; '
                f'available: {[found.name for found in present] or "none"}',
                _decoder.BACKENDS, only=backend))
    width, height = size or (0, 0)
    capable = [found for found in candidates if found.supports(width, height, codec)]
    if not capable:
        wanted = f'{width}x{height} {codec}' if size else codec
        raise DecoderUnavailable(with_reasons(
            f'no decoder here can read {wanted}; available: {describe(candidates)}',
            _decoder.BACKENDS))
    if size is not None:
        options['size'] = size
    return capable[0].factory(codec=codec, headers=headers, **options)


def _register_builtin_backends() -> None:
    """Add the backends that ship with this package to the registry.

    Importing a backend must not require its hardware: each module exposes a
    cheap probe, and the driver library is only opened when an encoder is built.
    The order is the order :func:`open_encoder` and :func:`open_decoder` prefer
    them in: the decoders that keep the picture on the GPU come before the one
    that copies it through host memory.
    """
    _encoder.BACKENDS.extend([NVENC_BACKEND, VPL_BACKEND, VAAPI_BACKEND])
    _decoder.BACKENDS.extend([NVDEC_DECODER_BACKEND, VAAPI_DECODER_BACKEND,
                              VPL_DECODER_BACKEND, NVDEC_COPY_DECODER_BACKEND])


_register_builtin_backends()
