#!/usr/bin/env python3
"""
Download UC Merced Land Use from Hugging Face and build all datasets in one pass.
Everything is written directly into data/.

Output layout:
data/
  x1/{train,val,test}/<class>/*.png       # standardized 256x256 HR (targets for x2/x4)
  x2/{train,val,test}/<class>/*.png       # LR 128x128
  x3/{train,val,test}/<class>/*.png       # LR  85x85
  x3_hr/{train,val,test}/<class>/*.png    # HR 255x255 (top-left crop of x1)
  x4/{train,val,test}/<class>/*.png       # LR  64x64
  split_manifest.csv                      # split, class, filename, source index, sizes
  class_to_idx.json                       # alphabetical class -> index

Notes
- Stratified 70/10/20 split per class, seeded. Every scale uses the same split.
- All LR images are generated from the standardized HR image (PIL bicubic).
- x3: HR is the 255x255 top-left crop of the 256x256 image; LR is 85x85.
- Training: x2 -> (x2 LR, x1 HR), x3 -> (x3 LR, x3_hr HR), x4 -> (x4 LR, x1 HR).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageOps

DATASET_ID = "blanchon/UC_Merced"
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}
HR_SIZE = 256
X3_HR_SIZE = 255
LR_SIZES = {"x2": 128, "x3": 85, "x4": 64}
OUTPUT_FOLDERS = ("x1", "x2", "x3", "x3_hr", "x4")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=project_root / "data",
                   help="Output directory (default: <project-root>/data).")
    p.add_argument("--dataset-id", default=DATASET_ID)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true",
                   help="Delete and recreate only x1, x2, x3, x3_hr and x4 in the data dir.")
    return p.parse_args()


def stratified_split(records, seed: int):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r in records:
        by_class[r["label"]].append(r)

    result = {s: [] for s in SPLITS}
    for class_name in sorted(by_class):
        items = by_class[class_name][:]
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * SPLIT_RATIOS["train"]))
        n_val = int(round(n * SPLIT_RATIOS["val"]))
        n_test = n - n_train - n_val
        if min(n_train, n_val, n_test) < 1:
            raise ValueError(f"Class {class_name!r} has {n} images, too few to split.")
        result["train"].extend(items[:n_train])
        result["val"].extend(items[n_train:n_train + n_val])
        result["test"].extend(items[n_train + n_val:])

    for s in SPLITS:
        rng.shuffle(result[s])
    return result


def save(img: Image.Image, root: Path, split: str, rel: Path) -> None:
    path = root / split / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")  # PNG keeps exact pixel values for PSNR


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    dirs = {name: data_dir / name for name in OUTPUT_FOLDERS}

    # --- guard against mixing old and new data ---
    existing = [p for p in dirs.values() if p.exists() or p.is_symlink()]
    if existing:
        if not args.overwrite:
            raise FileExistsError(
                "These folders already exist: "
                + ", ".join(str(p) for p in existing)
                + ". Move/delete them or rerun with --overwrite "
                "(only x1, x2, x3, x3_hr and x4 are removed)."
            )
        for p in existing:
            if p.is_symlink() or p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)

    # --- load dataset ---
    print(f"Loading dataset: {args.dataset_id}")
    dd = load_dataset(args.dataset_id)
    split_name = "train" if "train" in dd else next(iter(dd.keys()))
    if split_name != "train":
        print(f"No 'train' split found; using {split_name!r}.")
    dataset = dd[split_name]

    cols = set(dataset.column_names)
    image_col = "image" if "image" in cols else "img" if "img" in cols else None
    if image_col is None or "label" not in cols:
        raise ValueError(f"Expected image/img and label columns, got {dataset.column_names}")
    label_names = getattr(dataset.features["label"], "names", None)
    if not label_names:
        raise ValueError("Could not read class names from the dataset label feature.")

    # --- index records (images are re-read one at a time later to save memory) ---
    records = []
    for index in range(len(dataset)):
        lv = dataset[index]["label"]
        cls = lv if isinstance(lv, str) else label_names[int(lv)]
        records.append({"index": index, "label": cls,
                        "filename": f"{cls}_{index:04d}.png"})
    if not records:
        raise RuntimeError("The dataset returned no records.")

    counts = Counter(r["label"] for r in records)
    print(f"Loaded {len(records)} images across {len(counts)} classes.")

    splits = stratified_split(records, args.seed)

    with (data_dir / "class_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump({n: i for i, n in enumerate(sorted(counts))}, f, indent=2)

    # --- write images ---
    print("\nWriting x1, x2, x3, x3_hr, x4 ...")
    manifest_rows = []
    for s in SPLITS:
        for r in splits[s]:
            rel = Path(r["label"]) / r["filename"]
            img = dataset[r["index"]][image_col]
            original = ImageOps.exif_transpose(img).convert("RGB")

            resized = original.size != (HR_SIZE, HR_SIZE)
            hr256 = (original.resize((HR_SIZE, HR_SIZE), Image.Resampling.BICUBIC)
                     if resized else original.copy())
            save(hr256, dirs["x1"], s, rel)

            save(hr256.resize((LR_SIZES["x2"],) * 2, Image.Resampling.BICUBIC),
                 dirs["x2"], s, rel)
            save(hr256.resize((LR_SIZES["x4"],) * 2, Image.Resampling.BICUBIC),
                 dirs["x4"], s, rel)

            hr255 = hr256.crop((0, 0, X3_HR_SIZE, X3_HR_SIZE))
            save(hr255, dirs["x3_hr"], s, rel)
            save(hr255.resize((LR_SIZES["x3"],) * 2, Image.Resampling.BICUBIC),
                 dirs["x3"], s, rel)

            manifest_rows.append({
                "split": s, "class_label": r["label"], "filename": r["filename"],
                "source_index": r["index"],
                "source_size": f"{original.width}x{original.height}",
                "resized_to_256": resized,
            })

    manifest_rows.sort(key=lambda m: (SPLITS.index(m["split"]), m["class_label"], m["filename"]))
    manifest_path = data_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    # --- summary ---
    print("\nDone. Output in:", data_dir)
    for name in OUTPUT_FOLDERS:
        print(f"  {name}/")
    print(f"  split_manifest.csv, class_to_idx.json")
    print("\nCounts:")
    for s in SPLITS:
        print(f"  {s:5s}: {len(splits[s]):4d} images")
    print("\nHR targets: x2/x4 -> x1 (256x256);  x3 -> x3_hr (255x255).")


if __name__ == "__main__":
    main()
