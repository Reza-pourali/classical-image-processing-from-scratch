"""Run the classical image-processing pipeline on a user image."""

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classical_image_processing.io_utils import read_gray_image
from classical_image_processing.pipeline import run_pipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="Gaussian smoothing, first-derivative NMS, and second-derivative zero crossings."
    )
    p.add_argument("image", help="Path to an input image")
    p.add_argument("--downsample", type=int, default=4, help="Integer stride; default 4")
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--threshold-ratio", type=float, default=0.15)
    p.add_argument("--output", default="pipeline_result.png")
    return p.parse_args()


def main():
    args = parse_args()
    if args.downsample < 1:
        raise ValueError("--downsample must be >= 1")

    gray = read_gray_image(args.image)
    gray = gray[::args.downsample, ::args.downsample]

    result = run_pipeline(
        gray,
        sigma=args.sigma,
        threshold_ratio=args.threshold_ratio,
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    items = [
        (result.image, "Input"),
        (result.smooth, "Gaussian Smoothed"),
        (result.gradient_magnitude, "Gradient Magnitude"),
        (result.edges_gradient, "NMS Gradient"),
        (result.laplacian, "Laplacian"),
        (result.field_green, "Field Green"),
    ]
    for ax, (arr, title) in zip(axes.flat, items):
        ax.imshow(arr, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
