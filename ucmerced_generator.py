#!/usr/bin/env python3
"""
Download UC Merced Land Use from Hugging Face and create aligned x1/x2/x3/x4
datasets with stratified train/val/test splits.

Output layout:
data/
  x1/{train,val,test}/<class_name>/*.jpg   # original 256x256 images
  x2/{train,val,test}/<class_name>/*.jpg   # downsampled by 2
  x3/{train,val,test}/<class_name>/*.jpg   # downsampled by 3
  x4/{train,val,test}/<class_name>/*.jpg   # downsampled by 4
  split_manifest.csv
  class_to_idx.json

The same source images are assigned to the same split at every scale.
Downsampling uses PIL bicubic interpolation. For x3, 256 is not divisible by 3,
so the LR image is 85x85. When training an x3 SR model, crop the HR target to
255x255 (or crop prediction/target consistently) so the dimensions match.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageOps


DATASET_ID = "blanchon/UC_Merced"
SCALES = (1, 2, 3, 4)
SPLIT_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data",
        help="Output data directory (default: <project-root>/data).",
    )
    parser.add_argument(
        "--dataset-id",
        default=DATASET_ID,
        help=f"Hugging Face dataset ID (default: {DATASET_ID}).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing x1/x2/x3/x4 output folders before writing.",
    )
    return parser.parse_args()


def stratified_split(records, seed: int):
    """Split records independently per class; keep each source together at all scales."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for record in records:
        by_class[record["label"]].append(record)

    result = {"train": [], "val": [], "test": []}
    for class_name in sorted(by_class):
        class_records = by_class[class_name][:]
        rng.shuffle(class_records)
        n = len(class_records)

        n_train = int(round(n * SPLIT_RATIOS["train"]))
        n_val = int(round(n * SPLIT_RATIOS["val"]))
        # Assign remainder to test to guarantee totals add to n.
        n_test = n - n_train - n_val

        if min(n_train, n_val, n_test) < 1:
            raise ValueError(
                f"Class {class_name!r} has {n} images, too few for all three splits."
            )

        result["train"].extend(class_records[:n_train])
        result["val"].extend(class_records[n_train:n_train + n_val])
        result["test"].extend(class_records[n_train + n_val:])

    for split_records in result.values():
        rng.shuffle(split_records)
    return result


def downsample_image(image: Image.Image, scale: int) -> Image.Image:
    """Create an LR image with dimensions floor(H/scale) x floor(W/scale)."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    if scale == 1:
        return image.copy()

    width, height = image.size
    lr_width = width // scale
    lr_height = height // scale
    if lr_width < 1 or lr_height < 1:
        raise ValueError(f"Image {image.size} is too small for scale x{scale}.")
    return image.resize((lr_width, lr_height), resample=Image.Resampling.BICUBIC)


def remove_existing_outputs(data_dir: Path):
    import shutil

    for scale in SCALES:
        folder = data_dir / f"x{scale}"
        if folder.exists():
            shutil.rmtree(folder)


def main():
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    existing = [data_dir / f"x{scale}" for scale in SCALES]
    if any(path.exists() for path in existing):
        if not args.overwrite:
            raise FileExistsError(
                "At least one output folder already exists under "
                f"{data_dir}. To avoid mixing old and new data, stop and either "
                "move/delete those folders or rerun with --overwrite."
            )
        remove_existing_outputs(data_dir)

    print(f"Loading dataset: {args.dataset_id}")
    dataset_dict = load_dataset(args.dataset_id)
    if "train" in dataset_dict:
        dataset = dataset_dict["train"]
    else:
        first_split = next(iter(dataset_dict.keys()))
        print(f"No 'train' split found; using dataset split {first_split!r}.")
        dataset = dataset_dict[first_split]

    columns = set(dataset.column_names)
    image_column = "image" if "image" in columns else "img" if "img" in columns else None
    if image_column is None or "label" not in columns:
        raise ValueError(
            f"Expected image/img and label columns, found: {dataset.column_names}"
        )

    label_feature = dataset.features["label"]
    label_names = getattr(label_feature, "names", None)
    if not label_names:
        raise ValueError(
            "Could not read class names from the dataset label feature. "
            "Inspect the dataset schema and adapt this script."
        )

    records = []
    for index, row in enumerate(dataset):
        label_value = row["label"]
        if isinstance(label_value, str):
            class_name = label_value
        else:
            class_name = label_names[int(label_value)]

        image = row[image_column]
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Row {index} image is not a PIL image (got {type(image)!r})."
            )
        image = ImageOps.exif_transpose(image).convert("RGB")

        # Stable filenames make corresponding source images identifiable across scales.
        filename = f"{class_name}_{index:04d}.png"
        records.append({
            "index": index,
            "label": class_name,
            "filename": filename,
            "image": image.copy(),
        })

    if not records:
        raise RuntimeError("The Hugging Face dataset returned no records.")

    # Confirm expected balanced class counts before creating splits.
    class_counts = Counter(record["label"] for record in records)
    print(f"Loaded {len(records)} images across {len(class_counts)} classes.")
    for class_name in sorted(class_counts):
        print(f"  {class_name}: {class_counts[class_name]}")

    splits = stratified_split(records, args.seed)

    # Save the class mapping in alphabetical order for deterministic labels.
    sorted_classes = sorted(class_counts)
    with (data_dir / "class_to_idx.json").open("w", encoding="utf-8") as file:
        json.dump({name: i for i, name in enumerate(sorted_classes)}, file, indent=2)

    manifest_path = data_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "class_label", "filename", "source_index"])
        for split_name in ("train", "val", "test"):
            for record in sorted(splits[split_name], key=lambda r: (r["label"], r["filename"])):
                writer.writerow([
                    split_name, record["label"], record["filename"], record["index"]
                ])

    print("\nWriting original and downsampled images...")
    for split_name in ("train", "val", "test"):
        for record in splits[split_name]:
            class_name = record["label"]
            original = record["image"]

            for scale in SCALES:
                output_dir = data_dir / f"x{scale}" / split_name / class_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_image = downsample_image(original, scale)
                # PNG preserves decoded pixel values, which is preferable for
                # paired HR/LR experiments and PSNR evaluation.
                output_image.save(output_dir / record["filename"], format="PNG")

    print("\nDataset generation complete.")
    print(f"Output directory: {data_dir}")
    print(f"Split manifest:   {manifest_path}")
    print(f"Class mapping:    {data_dir / 'class_to_idx.json'}")
    print("\nCounts:")
    for split_name in ("train", "val", "test"):
        counts = Counter(record["label"] for record in splits[split_name])
        print(
            f"  {split_name:5s}: {len(splits[split_name]):4d} images "
            f"({dict(sorted(counts.items()))})"
        )

    print("\nNote for x3 super-resolution: original 256x256 HR images correspond to")
    print("85x85 LR images. Crop HR targets to 255x255 for exact 3x dimensions.")


if __name__ == "__main__":
    main()
