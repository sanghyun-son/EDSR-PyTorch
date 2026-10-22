

#!/usr/bin/env python3
"""
Evaluate pretrained EDSR on Set5, Set14, B100, and Urban100
at scales x2, x3, and x4.

Run from the official EDSR-PyTorch/src directory:

    python -m pip install numpy pillow scikit-image pandas
    python evaluate_edsr_table.py

Outputs:
    src/edsr_results_table.csv
    src/edsr_results_full_precision.csv

Uses standard EDSR (32 residual blocks, 256 features), without
self-ensemble. PSNR and SSIM are calculated on the Y channel,
after cropping 'scale' pixels from each border.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = "../dataset"
EXPERIMENT_DIR = ROOT.parent / "experiment"
BENCHMARK_DIR = ROOT.parent / "dataset" / "benchmark"

DATASETS = ["Set5", "Set14", "B100", "Urban100"]
#SCALES = [2, 3, 4]
SCALES = [ 3]

# Published reference values: PSNR / SSIM
PUBLISHED = {
    ("Set5", 2): (37.99, 0.9604),
    ("Set5", 3): (34.37, 0.9270),
    ("Set5", 4): (32.46, 0.8968),

    ("Set14", 2): (33.60, 0.9167),
    ("Set14", 3): (30.22, 0.8417),
    ("Set14", 4): (28.80, 0.7876),

    ("B100", 2): (32.16, 0.8984),
    ("B100", 3): (29.09, 0.8052),
    ("B100", 4): (27.71, 0.7420),

    ("Urban100", 2): (32.93, 0.9351),
    ("Urban100", 3): (28.80, 0.8653),
    ("Urban100", 4): (26.64, 0.8033),
}


# ============================================================
# Image loading and metrics
# ============================================================

def load_rgb(path):
    """Load an image as an RGB uint8 array."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def rgb_to_y(image):
    """
    Convert RGB to the commonly used MATLAB-style Y channel.
    Input and output values use the 0-255 range.
    """
    image = image.astype(np.float64)

    return (
        16.0
        + (
            65.481 * image[..., 0]
            + 128.553 * image[..., 1]
            + 24.966 * image[..., 2]
        ) / 255.0
    )



def calculate_metrics(sr_path, hr_path, scale):
    """
    Compute Y-channel PSNR and SSIM after border cropping.

    If SR and HR differ by one pixel because of odd image dimensions,
    align them at the top-left and crop to their common dimensions.
    """
    sr = load_rgb(sr_path)
    hr = load_rgb(hr_path)

    if sr.shape != hr.shape:
        print(
            f"Warning: aligning dimensions for {sr_path.name}: "
            f"SR={sr.shape[:2]}, HR={hr.shape[:2]}"
        )

        # Use common dimensions. Do not resize either image,
        # because interpolation would alter the measured metrics.
        height = min(sr.shape[0], hr.shape[0])
        width = min(sr.shape[1], hr.shape[1])

        sr = sr[:height, :width]
        hr = hr[:height, :width]

    sr_y = rgb_to_y(sr)
    hr_y = rgb_to_y(hr)

    # Standard scale-dependent border crop.
    if scale > 0:
        sr_y = sr_y[scale:-scale, scale:-scale]
        hr_y = hr_y[scale:-scale, scale:-scale]

    mse = np.mean((sr_y - hr_y) ** 2)

    psnr = (
        float("inf")
        if mse == 0
        else float(10.0 * np.log10(255.0**2 / mse))
    )

    ssim = float(
        structural_similarity(
            hr_y,
            sr_y,
            data_range=255.0,
        )
    )

    return psnr, ssim

# ============================================================
# Dataset and output paths
# ============================================================

def find_hr_images(dataset):
    """Return a dictionary mapping image stems to HR image paths."""
    hr_dir = BENCHMARK_DIR / dataset / "HR"

    if not hr_dir.is_dir():
        raise FileNotFoundError(
            f"HR directory not found: {hr_dir}\n"
            "Check your dataset/benchmark directory structure."
        )

    extensions = {".png", ".bmp", ".jpg", ".jpeg"}

    images = {
        p.stem.lower(): p
        for p in hr_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    }

    if not images:
        raise RuntimeError(f"No HR images found in {hr_dir}")

    return images


def find_sr_images(save_name, dataset, scale):
    """
    Locate EDSR output images such as baby_x2_SR.png in:
        experiment/<save_name>/results-<dataset>/
    """
    result_dir = (
        EXPERIMENT_DIR
        / save_name
        / f"results-{dataset}"
    )

    if not result_dir.is_dir():
        raise FileNotFoundError(
            f"SR result directory not found: {result_dir}"
        )

    extensions = {".png", ".bmp", ".jpg", ".jpeg"}
    expected_suffix = f"_x{scale}_sr"

    images = [
        p for p in result_dir.iterdir()
        if (
            p.is_file()
            and p.suffix.lower() in extensions
            and p.stem.lower().endswith(expected_suffix)
        )
    ]

    if not images:
        raise RuntimeError(
            f"No files ending in {expected_suffix} found in {result_dir}"
        )

    return images


def match_hr_image(sr_path, hr_images, scale):
    """Match a generated filename such as baby_x2_SR.png to baby.png."""
    suffix = f"_x{scale}_sr"
    sr_stem = sr_path.stem.lower()

    if not sr_stem.endswith(suffix):
        raise ValueError(
            f"Unexpected SR filename format: {sr_path.name}"
        )

    original_stem = sr_stem[:-len(suffix)]

    if original_stem in hr_images:
        return hr_images[original_stem]

    # Do not guess if a filename cannot be matched unambiguously.
    possible = [
        path for stem, path in hr_images.items()
        if stem == original_stem
        or stem.startswith(original_stem)
        or original_stem.startswith(stem)
    ]

    if len(possible) == 1:
        return possible[0]

    raise FileNotFoundError(
        f"Could not uniquely match SR image {sr_path.name} "
        f"to an HR image. Expected stem: {original_stem}"
    )


def evaluate_dataset(save_name, dataset, scale):
    """Evaluate all generated images for one dataset and scale."""
    hr_images = find_hr_images(dataset)
    sr_images = find_sr_images(save_name, dataset, scale)

    metrics = []

    for sr_path in sorted(sr_images):
        hr_path = match_hr_image(sr_path, hr_images, scale)

        psnr, ssim = calculate_metrics(
            sr_path=sr_path,
            hr_path=hr_path,
            scale=scale,
        )

        metrics.append((psnr, ssim))

        print(
            f"  {sr_path.name:<30} "
            f"PSNR={psnr:.4f}  SSIM={ssim:.4f}"
        )

    if not metrics:
        raise RuntimeError(
            f"No valid image pairs for {dataset} x{scale}"
        )

    mean_psnr = float(np.mean([m[0] for m in metrics]))
    mean_ssim = float(np.mean([m[1] for m in metrics]))

    return mean_psnr, mean_ssim, len(metrics)


# ============================================================
# Run official EDSR evaluation
# ============================================================

def run_edsr(dataset, scale, save_name):
    """Invoke the official EDSR main.py evaluation."""
    command = [
        sys.executable,
        "main.py",
        "--model", "EDSR",
        "--scale", str(scale),
        "--n_resblocks", "32",
        "--n_feats", "256",
        "--res_scale", "0.1",
        "--data_test", dataset,
        "--dir_data", DATA_DIR,
        "--pre_train", "download",
        "--test_only",
        "--save", save_name,
        "--save_results",
    ]

    print("\n" + "=" * 72)
    print(f"Running EDSR: {dataset}, scale x{scale}")
    print("Command:", " ".join(command))
    print("=" * 72)

    subprocess.run(command, cwd=ROOT, check=True)


# ============================================================
# Main
# ============================================================

def main():
    if not (ROOT / "main.py").is_file():
        raise FileNotFoundError(
            "main.py was not found. Save this script in the official "
            "EDSR-PyTorch/src directory and run it from there."
        )

    all_results = []

    for scale in SCALES:
        for dataset in DATASETS:
            save_name = f"edsr_table_{dataset.lower()}_x{scale}"

            # Generate SR images using the official implementation.
            run_edsr(dataset, scale, save_name)

            # Compute PSNR and SSIM from matched SR/HR image pairs.
            psnr, ssim, image_count = evaluate_dataset(
                save_name=save_name,
                dataset=dataset,
                scale=scale,
            )

            published_psnr, published_ssim = PUBLISHED[
                (dataset, scale)
            ]

            all_results.append({
                "Dataset": dataset,
                "Scale": f"x{scale}",
                "Published PSNR": published_psnr,
                "Published SSIM": published_ssim,
                "Our PSNR": psnr,
                "Our SSIM": ssim,
                "Images evaluated": image_count,
            })

            print(
                f"\nMEAN {dataset} x{scale}: "
                f"PSNR={psnr:.4f}, SSIM={ssim:.4f}; "
                f"{image_count} images"
            )

    df = pd.DataFrame(all_results)

    # Report-ready table with the requested combined metric columns.
    table = pd.DataFrame({
        "Dataset": df["Dataset"],
        "Scale": df["Scale"].str.replace("x", "×", regex=False),
        "Published EDSR PSNR / SSIM": [
            f"{p:.2f} / {s:.4f}"
            for p, s in zip(
                df["Published PSNR"],
                df["Published SSIM"],
            )
        ],
        "Our EDSR PSNR / SSIM": [
            f"{p:.4f} / {s:.4f}"
            for p, s in zip(
                df["Our PSNR"],
                df["Our SSIM"],
            )
        ],
    })

    print("\n\n" + "=" * 100)
    print("FINAL COMBINED EDSR RESULTS")
    print("=" * 100)
    print(table.to_string(index=False))
    print("=" * 100)

    formatted_csv = ROOT / "edsr_results_table.csv"
    precise_csv = ROOT / "edsr_results_full_precision.csv"

    table.to_csv(formatted_csv, index=False)
    df.to_csv(precise_csv, index=False)

    print(f"\nReport-ready table: {formatted_csv}")
    print(f"Full-precision metrics: {precise_csv}")


if __name__ == "__main__":
    main()
