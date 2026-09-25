"""Gaussian smoothing and convolution.

The coursework report describes:
- manual 2D Gaussian kernel construction,
- kernel normalization,
- convolution without cv2.filter2D/scipy.signal.convolve2d,
- and a separable 1D-X + 1D-Y implementation.

This refactor keeps those ideas but uses NumPy sliding windows for speed.
"""

import numpy as np


def _validate_sigma(sigma: float) -> float:
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    return sigma


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    sigma = _validate_sigma(sigma)
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def gaussian_kernel_2d(sigma: float) -> np.ndarray:
    sigma = _validate_sigma(sigma)
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    y = np.arange(-radius, radius + 1, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="xy")
    kernel = np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def conv2d_manual(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D zero-padded convolution without a ready-made convolution routine."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must both be 2D.")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel dimensions must be odd.")

    ph = kernel.shape[0] // 2
    pw = kernel.shape[1] // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, kernel.shape
    )
    flipped = np.flip(kernel, axis=(0, 1))
    return np.einsum("ijkl,kl->ij", windows, flipped, optimize=True)


def conv1d_manual(
    image: np.ndarray,
    kernel: np.ndarray,
    axis: int,
) -> np.ndarray:
    """1D zero-padded convolution along X (axis=1) or Y (axis=0)."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if image.ndim != 2 or kernel.ndim != 1:
        raise ValueError("image must be 2D and kernel must be 1D.")
    if kernel.size % 2 == 0:
        raise ValueError("kernel length must be odd.")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")

    radius = kernel.size // 2
    if axis == 1:
        padded = np.pad(image, ((0, 0), (radius, radius)), mode="constant")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, kernel.size, axis=1
        )
    else:
        padded = np.pad(image, ((radius, radius), (0, 0)), mode="constant")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, kernel.size, axis=0
        )

    flipped = kernel[::-1]
    return np.einsum("ijk,k->ij", windows, flipped, optimize=True)


def gaussian_blur_2d(image: np.ndarray, sigma: float) -> np.ndarray:
    return conv2d_manual(image, gaussian_kernel_2d(sigma))


def gaussian_blur_separable(image: np.ndarray, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel_1d(sigma)
    x_filtered = conv1d_manual(image, kernel, axis=1)
    return conv1d_manual(x_filtered, kernel, axis=0)
