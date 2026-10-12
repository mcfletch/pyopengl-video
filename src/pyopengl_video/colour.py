"""How a decoded picture's YUV samples become RGB.

A stream states its colour as Rec. ITU-T H.273 code points in the H.264 video
usability information: the matrix, whether the samples use the full 8-bit
range, the primaries and transfer function, and where chroma samples sit
relative to luma. :class:`Colour` holds those, and turns the matrix and range
into the affine transform the conversion pass applies.

Primaries and transfer are carried and not converted: BT.709's transfer is
treated as sRGB's, which is how browsers display video, and the output texture
is ``GL_SRGB8_ALPHA8``.
"""
from __future__ import annotations

import dataclasses

__all__ = ['Colour', 'MATRIX_BT601', 'MATRIX_BT709', 'MATRIX_BT2020',
           'MATRIX_IDENTITY', 'MATRIX_UNSPECIFIED']

#: MatrixCoefficients code points (H.273 table 4).
MATRIX_IDENTITY = 0
MATRIX_BT709 = 1
MATRIX_UNSPECIFIED = 2
MATRIX_FCC = 4
MATRIX_BT470BG = 5
MATRIX_BT601 = 6
MATRIX_SMPTE240 = 7
MATRIX_BT2020 = 9
MATRIX_BT2020_CONSTANT = 10

#: (Kr, Kb) for each matrix this conversion handles. BT.2020's constant
#: luminance form is converted as its non-constant form, which differs in
#: saturated colours.
COEFFICIENTS = {
    MATRIX_BT709: (0.2126, 0.0722),
    MATRIX_FCC: (0.30, 0.11),
    MATRIX_BT470BG: (0.299, 0.114),
    MATRIX_BT601: (0.299, 0.114),
    MATRIX_SMPTE240: (0.212, 0.087),
    MATRIX_BT2020: (0.2627, 0.0593),
    MATRIX_BT2020_CONSTANT: (0.2627, 0.0593),
}

#: Pictures this many rows tall or taller are high definition, and a stream
#: that states no matrix is BT.709 there and BT.601 below.
HIGH_DEFINITION_ROWS = 720

#: Chroma sample positions (H.264 figure E-1), as the offset from where a plain
#: half-size sampling puts them, in chroma texels: right, then down.
CHROMA_OFFSETS = {
    0: (0.25, 0.0),
    1: (0.0, 0.0),
    2: (0.25, 0.25),
    3: (0.0, 0.25),
    4: (0.25, -0.25),
    5: (0.0, -0.25),
}


@dataclasses.dataclass(frozen=True)
class Colour:
    """A picture's colour description, as H.273 code points.

    matrix -- MatrixCoefficients; 2 is unspecified, and :meth:`resolved`
        chooses one by picture height
    full_range -- samples span 0 to 255, rather than 16 to 235 for luma and
        16 to 240 for chroma
    primaries -- ColourPrimaries, carried for the caller
    transfer -- TransferCharacteristics, carried for the caller
    chroma_location -- chroma_sample_loc_type of the top field, 0 to 5
    """

    matrix: int = MATRIX_UNSPECIFIED
    full_range: bool = False
    primaries: int = 2
    transfer: int = 2
    chroma_location: int = 0

    def resolved(self, height: int) -> Colour:
        """This description with a matrix chosen when the stream states none.

        BT.601 below 720 rows and BT.709 from 720 up, which is the rule
        players apply. A matrix this module has no coefficients for is
        treated the same way.
        """
        if self.matrix == MATRIX_IDENTITY or self.matrix in COEFFICIENTS:
            return self
        chosen = MATRIX_BT709 if height >= HIGH_DEFINITION_ROWS else MATRIX_BT601
        return dataclasses.replace(self, matrix=chosen)

    def yuv_to_rgb(self, depth: int = 8) -> tuple[tuple[float, float, float, float], ...]:
        """The 3x4 affine transform from normalised ``(Y, Cb, Cr, 1)`` to RGB.

        depth -- bits a sample. The inputs are code values over the largest
            one: code / 255 at 8 bits, which is what an 8-bit texture reads
            as, and code / 1023 at 10.

        The result is non-linear R'G'B' from 0 to 1. A matrix still
        unspecified is taken as BT.601.
        """
        luma_scale, luma_offset, chroma_scale = self._range(depth)
        centre = (1 << (depth - 1)) / ((1 << depth) - 1) * chroma_scale
        if self.matrix == MATRIX_IDENTITY:
            # GBR: green is coded as luma, blue as Cb and red as Cr, each with
            # the luma range.
            return (
                (0.0, 0.0, luma_scale, luma_offset),
                (luma_scale, 0.0, 0.0, luma_offset),
                (0.0, luma_scale, 0.0, luma_offset),
            )
        kr, kb = COEFFICIENTS.get(self.matrix, COEFFICIENTS[MATRIX_BT601])
        kg = 1.0 - kr - kb
        red_cr = 2 * (1 - kr)
        blue_cb = 2 * (1 - kb)
        green_cb = -blue_cb * kb / kg
        green_cr = -red_cr * kr / kg
        rows = []
        for cb, cr in ((0.0, red_cr), (green_cb, green_cr), (blue_cb, 0.0)):
            rows.append((luma_scale, cb * chroma_scale, cr * chroma_scale,
                         luma_offset - (cb + cr) * centre))
        return tuple(rows)

    def _range(self, depth: int = 8) -> tuple[float, float, float]:
        """Luma scale, luma offset and chroma scale for normalised code values.

        Limited range at `depth` bits is the 8-bit range times
        ``2 ** (depth - 8)``: luma from 16 to 235 and chroma from 16 to 240 at
        8 bits, 64 to 940 and 64 to 960 at 10.
        """
        if self.full_range:
            return 1.0, 0.0, 1.0
        largest = ((1 << depth) - 1) / (1 << (depth - 8))
        return largest / 219, -16 / 219, largest / 224

    def chroma_offset(self) -> tuple[float, float]:
        """Where chroma samples sit, in chroma texels from a plain half-size map.

        Rightward and downward in the surface's own orientation, first row at
        the top. An unknown location is taken as 0, the H.264 default.
        """
        return CHROMA_OFFSETS.get(self.chroma_location, CHROMA_OFFSETS[0])
