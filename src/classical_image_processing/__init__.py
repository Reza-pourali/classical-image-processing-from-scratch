"""Classical image-processing building blocks implemented from scratch."""

from .io_utils import read_gray_image, to_gray
from .gaussian import (
    gaussian_kernel_1d,
    gaussian_kernel_2d,
    conv2d_manual,
    conv1d_manual,
    gaussian_blur_2d,
    gaussian_blur_separable,
)
from .first_derivative import (
    sobel_gradients,
    gradient_magnitude_direction,
    non_maximum_suppression,
    relative_threshold,
)
from .second_derivative import (
    laplacian_kernel,
    laplacian_response,
    zero_crossing_field_green,
    zero_crossing_reverse,
)

__all__ = [
    "read_gray_image",
    "to_gray",
    "gaussian_kernel_1d",
    "gaussian_kernel_2d",
    "conv2d_manual",
    "conv1d_manual",
    "gaussian_blur_2d",
    "gaussian_blur_separable",
    "sobel_gradients",
    "gradient_magnitude_direction",
    "non_maximum_suppression",
    "relative_threshold",
    "laplacian_kernel",
    "laplacian_response",
    "zero_crossing_field_green",
    "zero_crossing_reverse",
]
