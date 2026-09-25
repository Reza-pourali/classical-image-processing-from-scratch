"""Generate a licensing-safe synthetic image and run the full pipeline."""

from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classical_image_processing.gaussian import (
    gaussian_blur_2d,
    gaussian_blur_separable,
)
from classical_image_processing.pipeline import run_pipeline


def make_demo_image(height=240, width=360) -> np.ndarray:
    """Create edges, smooth gradients, fine texture, and repeated patterns."""
    y, x = np.mgrid[0:height, 0:width]
    img = np.zeros((height, width), dtype=np.float64)

    # Smooth background gradient
    img += 0.15 + 0.25 * (x / width)

    # Large bright rectangle
    img[35:105, 35:145] += 0.45

    # Dark rectangle
    img[125:210, 55:155] -= 0.12

    # Circular object
    circle = (x - 245) ** 2 + (y - 75) ** 2 <= 42 ** 2
    img[circle] += 0.45

    # Slanted boundary
    img[y > (0.35 * x + 95)] += 0.18

    # Fine checker texture
    checker = (((x // 5) + (y // 5)) % 2).astype(float)
    region = (x > 210) & (y > 135)
    img[region] += 0.10 * checker[region]

    # Thin lines
    img[30:215:18, 170:195] += 0.20

    return np.clip(img, 0.0, 1.0)


def save_gray(path: Path, image: np.ndarray):
    cv2.imwrite(str(path), np.clip(image * 255, 0, 255).astype(np.uint8))


def main():
    root = Path(__file__).resolve().parents[1]
    figures = root / "figures"
    figures.mkdir(exist_ok=True)

    image = make_demo_image()
    save_gray(figures / "synthetic_input.png", image)

    sigmas = [0.5, 1.0, 2.0, 4.0]
    diffs = []

    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    for col, sigma in enumerate(sigmas):
        g2 = gaussian_blur_2d(image, sigma)
        gs = gaussian_blur_separable(image, sigma)
        diff = np.abs(g2 - gs)
        diffs.append(float(diff.max()))

        axes[0, col].imshow(g2, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"2D Gaussian σ={sigma}")
        axes[1, col].imshow(gs, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title("Separable X+Y")
        axes[2, col].imshow(diff, cmap="gray")
        axes[2, col].set_title(f"|diff| max={diff.max():.2e}")

        for row in range(3):
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(figures / "gaussian_2d_vs_separable.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    result = run_pipeline(image, sigma=1.0, threshold_ratio=0.15)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    items = [
        (result.image, "Input"),
        (result.smooth, "Gaussian σ=1"),
        (result.gradient_magnitude, "Gradient Magnitude"),
        (result.edges_gradient, "NMS — Gradient Direction"),
        (result.edges_edge, "NMS — Edge Direction (+90°)"),
        (np.abs(result.edges_gradient - result.edges_edge), "NMS Difference"),
    ]
    for ax, (arr, title) in zip(axes.flat, items):
        ax.imshow(arr, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "first_derivative_pipeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    items = [
        (result.smooth, "Smoothed"),
        (result.laplacian, "Laplacian Response"),
        (result.field_green, "Field Green"),
        (result.field_green_reverse, "Field Green Reverse"),
    ]
    for ax, (arr, title) in zip(axes, items):
        ax.imshow(arr, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "second_derivative_pipeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Raw-array comparison: fixes the original S4 issue.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    items = [
        (result.gradient_magnitude, "Gradient Magnitude"),
        (result.edges_gradient, "NMS Gradient"),
        (result.edges_edge, "NMS Edge Direction"),
        (result.laplacian, "Laplacian"),
        (result.field_green, "Field Green"),
        (result.field_green_reverse, "Field Green Reverse"),
    ]
    for ax, (arr, title) in zip(axes.flat, items):
        ax.imshow(arr, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "all_methods_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fg_diff = np.abs(result.field_green - result.field_green_reverse)
    nms_fg_diff = np.abs(result.edges_gradient - result.field_green)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(fg_diff, cmap="gray")
    axes[0].set_title("Field Green vs Reverse")
    axes[1].imshow(nms_fg_diff, cmap="gray")
    axes[1].set_title("NMS(Gradient) vs Field Green")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "difference_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Maximum 2D-vs-separable differences:")
    for sigma, d in zip(sigmas, diffs):
        print(f"  sigma={sigma}: {d:.3e}")
    print("Generated figures in:", figures)


if __name__ == "__main__":
    main()
