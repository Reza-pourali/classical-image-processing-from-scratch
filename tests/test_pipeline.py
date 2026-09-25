import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classical_image_processing.pipeline import run_pipeline


class TestPipeline(unittest.TestCase):
    def test_shapes_are_preserved(self):
        image = np.zeros((48, 64), dtype=float)
        image[12:36, 20:44] = 1.0
        result = run_pipeline(image, sigma=1.0, threshold_ratio=0.15)

        for value in result.__dict__.values():
            self.assertEqual(value.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
