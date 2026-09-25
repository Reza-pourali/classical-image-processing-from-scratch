"""Image loading and manual RGB-to-grayscale conversion."""

from pathlib import Path

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB-like image data to grayscale using coursework weights.

    For OpenCV-loaded BGR images:
        Gray = 0.299 R + 0.587 G + 0.114 B
    """
    arr = np.asarray(image)

    if arr.ndim == 2:
        gray = arr.astype(np.float64)
        if gray.max(initial=0) > 1.0:
            gray /= 255.0
        return gray

    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("Expected a grayscale or 3-channel image.")

    arr = arr.astype(np.float64)
    if arr.max(initial=0) > 1.0:
        arr /= 255.0

    # Assume BGR when coming from OpenCV.
    b = arr[..., 0]
    g = arr[..., 1]
    r = arr[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def read_gray_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return to_gray(image)
