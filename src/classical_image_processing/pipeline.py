"""End-to-end pipeline used by the examples and tests."""

from dataclasses import dataclass

import numpy as np

from .gaussian import gaussian_blur_2d, gaussian_blur_separable
from .first_derivative import (
    sobel_gradients,
    gradient_magnitude_direction,
    non_maximum_suppression,
    relative_threshold,
)
from .second_derivative import (
    laplacian_response,
    zero_crossing_field_green,
    zero_crossing_reverse,
)


@dataclass
class PipelineResult:
    image: np.ndarray
    smooth: np.ndarray
    gradient_magnitude: np.ndarray
    nms_gradient: np.ndarray
    nms_edge: np.ndarray
    edges_gradient: np.ndarray
    edges_edge: np.ndarray
    laplacian: np.ndarray
    field_green: np.ndarray
    field_green_reverse: np.ndarray


def run_pipeline(
    gray_image: np.ndarray,
    sigma: float = 1.0,
    threshold_ratio: float = 0.15,
) -> PipelineResult:
    image = np.asarray(gray_image, dtype=np.float64)
    smooth = gaussian_blur_2d(image, sigma)

    gx, gy = sobel_gradients(smooth)
    magnitude, theta = gradient_magnitude_direction(gx, gy)

    nms_grad = non_maximum_suppression(magnitude, theta, mode="gradient")
    nms_edge = non_maximum_suppression(magnitude, theta, mode="edge")

    edges_grad = relative_threshold(nms_grad, threshold_ratio)
    edges_edge = relative_threshold(nms_edge, threshold_ratio)

    lap = laplacian_response(smooth)
    fg = zero_crossing_field_green(lap)
    fg_rev = zero_crossing_reverse(lap)

    return PipelineResult(
        image=image,
        smooth=smooth,
        gradient_magnitude=magnitude,
        nms_gradient=nms_grad,
        nms_edge=nms_edge,
        edges_gradient=edges_grad,
        edges_edge=edges_edge,
        laplacian=lap,
        field_green=fg,
        field_green_reverse=fg_rev,
    )
