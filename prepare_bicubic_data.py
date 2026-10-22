#!/usr/bin/env python3
"""Generate bicubic-upsampled baselines for UC-Merced ×2, ×3, and ×4.

Expected data layout under --data-dir:
  x2/{train,val,test}/...    -> x1/{train,val,test}/...      (×2, 128 -> 256)
  x3/{train,val,test}/...    -> x3_hr/{train,val,test}/...   (×3, 85 -> 255)
  x4/{train,val,test}/...    -> x1/{train,val,test}/...      (×4, 64 -> 256)

Outputs: bicubic_x2/, bicubic_x3/, bicubic_x4/ with the same split/class folders.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def list_images(root: Path) -> dict[Path, Path]:
    return {
        p.relative_to(root): p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


def split_dir(root: Path, split: str) -> Path | None:
    candidates = [root / split]
    if split == "test":
        candidates.append(root / "tet")  # tolerate a dataset split named "tet"
    return next((p for p in candidates if p.is_dir()), None)


def generate_scale(data_root: Path, scale: int, lr_name: str, hr_name: str,
                   out_name: str, overwrite: bool) -> tuple[int, int, int]:
    lr_root, hr_root = data_root / lr_name, data_root / hr_name
    if not lr_root.is_dir():
        raise FileNotFoundError(f"LR directory not found: {lr_root}")
    if not hr_root.is_dir():
        raise FileNotFoundError(f"HR reference directory not found: {hr_root}")

    written = skipped = missing = 0
    started = time.perf_counter()

    for split in SPLITS:
        lr_dir, hr_dir = split_dir(lr_root, split), split_dir(hr_root, split)
        if lr_dir is None or hr_dir is None:
            print(f"[{out_name}] Skipping '{split}': matching split directory not found.")
            continue

        lr_images, hr_images = list_images(lr_dir), list_images(hr_dir)
        if not lr_images:
            print(f"[{out_name}/{split}] No LR images found.")
            continue

        out_dir = data_root / out_name / split
        split_written = 0
        for rel_path, lr_path in lr_images.items():
            hr_path = hr_images.get(rel_path)
            if hr_path is None:
                missing += 1
                print(f"[WARN] No matching HR image for {lr_path}; expected {hr_dir / rel_path}")
                continue

            out_path = (out_dir / rel_path).with_suffix(".png")
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(lr_path) as im:
                lr = ImageOps.exif_transpose(im).convert("RGB")
            with Image.open(hr_path) as im:
                hr = ImageOps.exif_transpose(im)
                target_size = hr.size

            expected = (lr.width * scale, lr.height * scale)
            if target_size != expected:
                raise ValueError(
                    f"Size mismatch at ×{scale} for {rel_path}: LR={lr.size}, "
                    f"HR={target_size}, expected HR={expected}. Check your paired data."
                )

            # Resize only the LR image; HR is used only to get/verify the target size.
            upsampled = lr.resize(target_size, resample=Image.Resampling.BICUBIC)
            upsampled.save(out_path, format="PNG")
            written += 1
            split_written += 1

        print(f"[{out_name}/{split}] wrote {split_written}; "
              f"LR images={len(lr_images)}, HR pairs available={len(hr_images)}")

    elapsed = time.perf_counter() - started
    print(f"[{out_name}] {elapsed:.2f}s | written={written}, "
          f"existing skipped={skipped}, missing pairs={missing}")
    return written, skipped, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/edsr_data"),
                        help="Directory containing x1, x2, x3, x3_hr, and x4.")
    parser.add_argument("--scales", nargs="+", type=int, choices=(2, 3, 4),
                        default=[2, 3, 4], help="Scales to generate (default: 2 3 4).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite generated files that already exist.")
    args = parser.parse_args()

    data_root = args.data_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    configs = {
        2: ("x2", "x1", "bicubic_x2"),
        3: ("x3", "x3_hr", "bicubic_x3"),
        4: ("x4", "x1", "bicubic_x4"),
    }

    total_start = time.perf_counter()
    totals = [0, 0, 0]
    print(f"Data root: {data_root}")
    for scale in args.scales:
        lr_name, hr_name, out_name = configs[scale]
        print(f"\n=== Bicubic ×{scale}: {lr_name} -> {out_name} "
              f"(reference dimensions from {hr_name}) ===")
        result = generate_scale(data_root, scale, lr_name, hr_name, out_name, args.overwrite)
        totals = [a + b for a, b in zip(totals, result)]

    print("\n=== Finished ===")
    print(f"Images written: {totals[0]}")
    print(f"Existing files skipped: {totals[1]}")
    print(f"Missing LR/HR pairs: {totals[2]}")
    print(f"Total wall time: {time.perf_counter() - total_start:.2f}s")
    print(f"Output root: {data_root}/bicubic_x{{2,3,4}}/")


if __name__ == "__main__":
    main()
