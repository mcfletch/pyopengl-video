"""YUV to RGB: the matrices, ranges and chroma siting a stream can declare.

No GL. The conversion pass applies :meth:`Colour.yuv_to_rgb` as it stands, so
the arithmetic is held here against values worked out from Rec. ITU-T H.273
and the coefficients each matrix names.
"""
import numpy as np
import pytest

from pyopengl_video.colour import (
    MATRIX_BT601,
    MATRIX_BT709,
    MATRIX_BT2020,
    MATRIX_IDENTITY,
    MATRIX_UNSPECIFIED,
    Colour,
)


def convert(colour, y, u, v):
    """8-bit code values through the affine matrix, as the shader applies it."""
    matrix = np.array(colour.yuv_to_rgb())
    return matrix @ np.array([y / 255, u / 255, v / 255, 1.0])


class TestRanges:
    @pytest.mark.parametrize('matrix', [MATRIX_BT601, MATRIX_BT709, MATRIX_BT2020])
    def test_limited_range_black_and_white(self, matrix):
        colour = Colour(matrix=matrix, full_range=False)
        assert convert(colour, 16, 128, 128) == pytest.approx([0, 0, 0], abs=1e-6)
        assert convert(colour, 235, 128, 128) == pytest.approx([1, 1, 1], abs=1e-6)

    @pytest.mark.parametrize('matrix', [MATRIX_BT601, MATRIX_BT709, MATRIX_BT2020])
    def test_full_range_black_and_white(self, matrix):
        colour = Colour(matrix=matrix, full_range=True)
        assert convert(colour, 0, 128, 128) == pytest.approx([0, 0, 0], abs=1e-6)
        assert convert(colour, 255, 128, 128) == pytest.approx([1, 1, 1], abs=1e-6)


class TestTenBits:
    """A 10-bit stream's code values are four times an 8-bit stream's."""

    @staticmethod
    def convert(colour, y, u, v):
        matrix = np.array(colour.yuv_to_rgb(depth=10))
        return matrix @ np.array([y / 1023, u / 1023, v / 1023, 1.0])

    @pytest.mark.parametrize('matrix', [MATRIX_BT601, MATRIX_BT709, MATRIX_BT2020])
    def test_limited_range_black_and_white(self, matrix):
        colour = Colour(matrix=matrix, full_range=False)
        assert self.convert(colour, 64, 512, 512) == pytest.approx([0, 0, 0], abs=1e-6)
        assert self.convert(colour, 940, 512, 512) == pytest.approx([1, 1, 1], abs=1e-6)

    def test_full_range_black_and_white(self):
        colour = Colour(matrix=MATRIX_BT709, full_range=True)
        assert self.convert(colour, 0, 512, 512) == pytest.approx([0, 0, 0], abs=1e-6)
        assert self.convert(colour, 1023, 512, 512) == pytest.approx([1, 1, 1], abs=1e-6)

    def test_a_colour_matches_the_same_colour_at_eight_bits(self):
        colour = Colour(matrix=MATRIX_BT709)
        assert self.convert(colour, 81 * 4, 90 * 4, 240 * 4) == pytest.approx(
            convert(colour, 81, 90, 240), abs=1e-6)


class TestMatrices:
    def test_bt601_has_the_familiar_coefficients(self):
        """R = 1.164(Y-16) + 1.596(Cr-128), and so on, for limited range."""
        matrix = np.array(Colour(matrix=MATRIX_BT601).yuv_to_rgb())
        assert matrix[0, 0] == pytest.approx(255 / 219)
        assert matrix[0, 2] == pytest.approx(1.402 * 255 / 224, rel=1e-4)
        assert matrix[1, 1] == pytest.approx(-0.344136 * 255 / 224, rel=1e-4)
        assert matrix[1, 2] == pytest.approx(-0.714136 * 255 / 224, rel=1e-4)
        assert matrix[2, 1] == pytest.approx(1.772 * 255 / 224, rel=1e-4)

    def test_bt709_red(self):
        """Full red, coded in limited-range BT.709, comes back as red."""
        kr, kb = 0.2126, 0.0722
        y = 16 + 219 * kr
        cb = 128 + 224 * (-kr / (2 * (1 - kb)))
        cr = 128 + 224 * 0.5
        rgb = convert(Colour(matrix=MATRIX_BT709), y, cb, cr)
        assert rgb == pytest.approx([1, 0, 0], abs=1e-6)

    def test_identity_carries_green_in_luma(self):
        rgb = convert(Colour(matrix=MATRIX_IDENTITY, full_range=True), 200, 50, 100)
        assert rgb == pytest.approx([100 / 255, 200 / 255, 50 / 255])


class TestDefaults:
    def test_an_unspecified_matrix_is_bt601_below_720_rows(self):
        assert Colour(matrix=MATRIX_UNSPECIFIED).resolved(height=576).matrix == MATRIX_BT601

    def test_and_bt709_from_720_rows(self):
        assert Colour(matrix=MATRIX_UNSPECIFIED).resolved(height=720).matrix == MATRIX_BT709

    def test_a_stated_matrix_is_kept(self):
        assert Colour(matrix=MATRIX_BT709).resolved(height=240).matrix == MATRIX_BT709

    def test_an_unknown_matrix_is_treated_as_unspecified(self):
        assert Colour(matrix=99).resolved(height=1080).matrix == MATRIX_BT709


class TestChromaSiting:
    """Offsets in chroma texels, in a surface's own orientation (first row top)."""

    @pytest.mark.parametrize('location,offset', [
        (0, (0.25, 0.0)),        # left, between rows: the H.264 default
        (1, (0.0, 0.0)),         # centre
        (2, (0.25, 0.25)),       # top left
        (3, (0.0, 0.25)),        # top
        (4, (0.25, -0.25)),      # bottom left
        (5, (0.0, -0.25)),       # bottom
    ])
    def test_each_location(self, location, offset):
        assert Colour(chroma_location=location).chroma_offset() == offset

    def test_an_unknown_location_is_the_default(self):
        assert Colour(chroma_location=9).chroma_offset() == (0.25, 0.0)
