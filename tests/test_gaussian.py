import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classical_image_processing.gaussian import (
    gaussian_kernel_1d,
    gaussian_kernel_2d,
    gaussian_blur_2d,
    gaussian_blur_separable,
)


class TestGaussian(unittest.TestCase):
    def test_kernels_are_normalized(self):
        self.assertAlmostEqual(float(gaussian_kernel_1d(1.0).sum()), 1.0, places=14)
        self.assertAlmostEqual(float(gaussian_kernel_2d(1.0).sum()), 1.0, places=14)

    def test_2d_and_separable_are_equivalent(self):
        rng = np.random.default_rng(42)
        image = rng.random((40, 50))
        for sigma in [0.5, 1.0, 2.0]:
            a = gaussian_blur_2d(image, sigma)
            b = gaussian_blur_separable(image, sigma)
            self.assertLess(np.max(np.abs(a - b)), 1e-12)


if __name__ == "__main__":
    unittest.main()
