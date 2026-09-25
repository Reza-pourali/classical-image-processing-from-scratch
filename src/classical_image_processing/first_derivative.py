"""First-derivative edge detection and Non-Maximum Suppression."""

import numpy as np

from .gaussian import conv2d_manual


SOBEL_X = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]],
    dtype=np.float64,
)

SOBEL_Y = np.array(
    [[-1, -2, -1],
     [ 0,  0,  0],
     [ 1,  2,  1]],
    dtype=np.float64,
)


def sobel_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = conv2d_manual(image, SOBEL_X)
    gy = conv2d_manual(image, SOBEL_Y)
    return gx, gy


def gradient_magnitude_direction(
    gx: np.ndarray,
    gy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)
    return magnitude, theta


def non_maximum_suppression(
    magnitude: np.ndarray,
    theta: np.ndarray,
    mode: str = "gradient",
) -> np.ndarray:
    """Thin edges by comparing along quantized gradient/edge directions.

    mode="gradient" reproduces the standard NMS direction used in the report.
    mode="edge" adds 90 degrees for the coursework comparison experiment.
    """
    mag = np.asarray(magnitude, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    if mag.shape != theta.shape:
        raise ValueError("magnitude and theta must have the same shape.")
    if mode not in {"gradient", "edge"}:
        raise ValueError("mode must be 'gradient' or 'edge'.")

    H, W = mag.shape
    out = np.zeros_like(mag)

    angle = np.rad2deg(theta) % 180.0
    if mode == "edge":
        angle = (angle + 90.0) % 180.0

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a = angle[i, j]

            if (0 <= a < 22.5) or (157.5 <= a < 180):
                q, r = mag[i, j + 1], mag[i, j - 1]
            elif 22.5 <= a < 67.5:
                q, r = mag[i + 1, j - 1], mag[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                q, r = mag[i + 1, j], mag[i - 1, j]
            else:
                q, r = mag[i - 1, j - 1], mag[i + 1, j + 1]

            if mag[i, j] >= q and mag[i, j] >= r:
                out[i, j] = mag[i, j]

    return out


def relative_threshold(image: np.ndarray, ratio: float = 0.15) -> np.ndarray:
    if not (0.0 <= ratio <= 1.0):
        raise ValueError("ratio must be in [0, 1].")
    image = np.asarray(image, dtype=np.float64)
    max_val = float(image.max(initial=0.0))
    if max_val == 0.0:
        return np.zeros_like(image)
    return (image >= ratio * max_val).astype(np.float64)
