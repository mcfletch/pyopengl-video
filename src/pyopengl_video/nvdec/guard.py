"""What a picture is held to before the hardware decoder is handed it.

NVIDIA's parser builds a ``CUVIDPICPARAMS`` for each picture, and
``cuvidDecodePicture`` uses what it names without checking it: it copies
``nBitstreamDataLen`` bytes of bitstream, decodes into picture ``CurrPicIdx``
and predicts from the pictures the codec's parameters list. A stream that is
cut short, or joined after its key frame, gives a picture whose length is a
small negative number stored unsigned, or whose references are pictures the
decoder never produced.

:func:`refusal` reads a picture before the decoder does and says what is
wrong with it. It holds no driver state: the decoder passes in which pictures
it has decoded and how many bytes the parser has been fed, so the rules are
checked on any machine.
"""
from __future__ import annotations

from collections.abc import Collection

from pyopengl_video.nvdec import api

__all__ = ['references', 'refusal']

#: ``frame_type`` of an AV1 key frame and intra-only frame, which predict
#: from no other picture.
AV1_KEY_FRAME, AV1_INTRA_ONLY_FRAME = 0, 2

#: ``frameType`` of a VP9 key frame.
VP9_KEY_FRAME = 0


def references(codec: int, picture: api.CUVIDPICPARAMS) -> list[int]:
    """The pictures `picture` predicts from, as the decoder's picture indexes.

    codec -- the ``cudaVideoCodec`` value the picture's parameters are read as

    H.264 and HEVC mark an entry that names no picture with a negative index,
    and those are left out. AV1 and VP9 have no such mark: every entry of an
    inter frame names a picture.
    """
    specific = picture.CodecSpecific
    if codec == api.cudaVideoCodec_H264:
        return [entry.PicIdx for entry in specific.h264.dpb if entry.PicIdx >= 0]
    if codec == api.cudaVideoCodec_HEVC:
        return [index for index in specific.hevc.RefPicIdx if index >= 0]
    if codec == api.cudaVideoCodec_VP9:
        vp9 = specific.vp9
        if vp9.frameType == VP9_KEY_FRAME or vp9.intraOnly:
            return []
        return [vp9.LastRefIdx, vp9.GoldenRefIdx, vp9.AltRefIdx]
    if codec == api.cudaVideoCodec_AV1:
        av1 = specific.av1
        if av1.frame_type in (AV1_KEY_FRAME, AV1_INTRA_ONLY_FRAME):
            return []
        return [reference.index for reference in av1.ref_frame]
    return []


def targets(codec: int, picture: api.CUVIDPICPARAMS) -> list[int]:
    """The pictures `picture` is decoded into.

    An AV1 frame with film grain is reconstructed into one picture, which
    later frames predict from, and shown from another.
    """
    found = [int(picture.CurrPicIdx)]
    if codec == api.cudaVideoCodec_AV1:
        reconstructed = int(picture.CodecSpecific.av1.decodePicIdx)
        if reconstructed != found[0]:
            found.append(reconstructed)
    return found


def refusal(picture: api.CUVIDPICPARAMS, *, codec: int, surfaces: int,
            decoded: Collection[int], fed: int) -> str:
    """Why the hardware decoder is not handed `picture`, or '' where it is.

    codec -- the ``cudaVideoCodec`` value of the sequence being decoded
    surfaces -- how many pictures the decoder was made with
    decoded -- the pictures it has decoded, by index
    fed -- the bytes handed to the parser that no picture has yet been
        decoded from
    """
    length = int(picture.nBitstreamDataLen)
    if length > fed:
        return (f'the parser states {length} bytes of bitstream for a picture, '
                f'and it has been handed {fed}')
    for target in targets(codec, picture):
        if not 0 <= target < surfaces:
            return (f'the parser asks for picture {target} of a decoder made '
                    f'with {surfaces}')
    missing = sorted({index for index in references(codec, picture)
                      if index not in decoded})
    if missing:
        named = ', '.join(map(str, missing))
        return (f'the picture predicts from picture {named}, which this decoder '
                'has not decoded: the stream was joined after its key frame, or '
                'a picture before this one was refused')
    return ''
