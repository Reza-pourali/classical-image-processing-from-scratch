import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classical_image_processing.first_derivative import (
    sobel_gradients,
    gradient_magnitude_direction,
    non_maximum_suppression,
    relative_threshold,
)
from classical_image_processing.second_derivative import (
    laplacian_response,
    zero_crossing_field_green,
    zero_crossing_reverse,
)


class TestEdges(unittest.TestCase):
    def test_nms_reduces_nonzero_gradient_support(self):
        image = np.zeros((50, 50), dtype=float)
        image[:, 25:] = 1.0
        gx, gy = sobel_gradients(image)
        mag, theta = gradient_magnitude_direction(gx, gy)
        nms = non_maximum_suppression(mag, theta, mode="gradient")
        self.assertLessEqual(np.count_nonzero(nms), np.count_nonzero(mag))

    def test_relative_threshold_shape(self):
        image = np.array([[0.0, 1.0], [0.5, 0.2]])
        out = relative_threshold(image, 0.5)
        np.testing.assert_array_equal(out, [[0.0, 1.0], [1.0, 0.0]])

    def test_zero_crossing_variants(self):
        L = np.zeros((5, 5), dtype=float)
        L[2, 2] = 2.0
        L[2, 3] = -1.0
        fg = zero_crossing_field_green(L)
        self.assertEqual(fg[2, 2], 1.0)

        L2 = -L
        rev = zero_crossing_reverse(L2)
        self.assertEqual(rev[2, 2], 1.0)


if __name__ == "__main__":
    unittest.main()
