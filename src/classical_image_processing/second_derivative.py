"""Second-derivative / Laplacian and coursework zero-crossing variants."""

import numpy as np

from .gaussian import conv2d_manual


def laplacian_kernel() -> np.ndarray:
    return np.array(
        [[0,  1, 0],
         [1, -4, 1],
         [0,  1, 0]],
        dtype=np.float64,
    )


def laplacian_response(image: np.ndarray) -> np.ndarray:
    return conv2d_manual(image, laplacian_kernel())


def zero_crossing_field_green(laplacian: np.ndarray) -> np.ndarray:
    """Coursework 'Field Green' rule: positive center, negative 4-neighbor."""
    L = np.asarray(laplacian, dtype=np.float64)
    H, W = L.shape
    out = np.zeros_like(L)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if L[i, j] > 0:
                neighbors = (
                    L[i - 1, j],
                    L[i + 1, j],
                    L[i, j - 1],
                    L[i, j + 1],
                )
                if any(n < 0 for n in neighbors):
                    out[i, j] = 1.0
    return out


def zero_crossing_reverse(laplacian: np.ndarray) -> np.ndarray:
    """Reverse coursework rule: negative center, positive 4-neighbor."""
    L = np.asarray(laplacian, dtype=np.float64)
    H, W = L.shape
    out = np.zeros_like(L)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if L[i, j] < 0:
                neighbors = (
                    L[i - 1, j],
                    L[i + 1, j],
                    L[i, j - 1],
                    L[i, j + 1],
                )
                if any(n > 0 for n in neighbors):
                    out[i, j] = 1.0
    return out
