#!/usr/bin/env python3
"""
Create standardized, paired UC-Merced HR/LR datasets for EDSR.

Reads:
  data/x1/{train,val,test}/<class>/<image>
  (also accepts "tet" instead of "test")

Writes new folders, preserving the original folders:
  data/edsr_data/x1/{train,val,test}/...
  data/edsr_data/x2/{train,val,test}/...
  data/edsr_data/x3/{train,val,test}/...
  data/edsr_data/x4/{train,val,test}/...

x1/x2/x4 targets are 256x256/128x128/64x64.
x3 targets are 255x255 and LR is 85x85.
All LR images are generated from their exact corresponding standardized HR target.
"""
import argparse
from pathlib import Path
from PIL import Image
import csv

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_split(root: Path, split: str) -> Path:
    candidates = [root / split]
    if split == "test":
        candidates.append(root / "tet")
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Missing split {split!r} under {root}; tried {candidates}")


def files_under(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/edsr_data"))
    ap.add_argument("--overwrite", action="store_true",
                    help="Delete and recreate only the output directory")
    args = ap.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} already exists and is not empty. "
                "Choose another --output-dir or pass --overwrite to replace it."
            )
        import shutil
        shutil.rmtree(args.output_dir)

    records = []
    for split in ("train", "val", "test"):
        src_split = find_split(args.data_dir / "x1", split)
        for src_path in files_under(src_split):
            rel = src_path.relative_to(src_split)
            with Image.open(src_path) as im:
                original = im.convert("RGB")
            # Standardize original HR to 256x256. UC-Merced images are normally 256x256;
            # resize only if a source image differs, and record that fact in the manifest.
            resized = original.size != (256, 256)
            hr256 = original.resize((256, 256), Image.Resampling.BICUBIC) if resized else original.copy()

            # Save standardized HR targets under x1.
            x1_path = args.output_dir / "x1" / split / rel
            x1_path.parent.mkdir(parents=True, exist_ok=True)
            hr256.save(x1_path)

            # Save LR inputs separately under x2 and x4; their targets are x1.
            lr128 = hr256.resize((128, 128), Image.Resampling.BICUBIC)
            x2_path = args.output_dir / "x2" / split / rel
            x2_path.parent.mkdir(parents=True, exist_ok=True)
            lr128.save(x2_path)

            lr64 = hr256.resize((64, 64), Image.Resampling.BICUBIC)
            x4_path = args.output_dir / "x4" / split / rel
            x4_path.parent.mkdir(parents=True, exist_ok=True)
            lr64.save(x4_path)

            # x3 requires aligned 255x255 HR -> 85x85 LR.
            hr255 = hr256.crop((0, 0, 255, 255))
            x3_hr_path = args.output_dir / "x3_hr" / split / rel
            x3_hr_path.parent.mkdir(parents=True, exist_ok=True)
            hr255.save(x3_hr_path)
            lr85 = hr255.resize((85, 85), Image.Resampling.BICUBIC)
            x3_lr_path = args.output_dir / "x3" / split / rel
            x3_lr_path.parent.mkdir(parents=True, exist_ok=True)
            lr85.save(x3_lr_path)

            records.append({
                "split": split,
                "relative_path": rel.as_posix(),
                "source_size": f"{original.width}x{original.height}",
                "resized_to_256": resized,
                "x1_hr": "256x256",
                "x2_lr": "128x128",
                "x2_hr": "256x256",
                "x3_lr": "85x85",
                "x3_hr": "255x255",
                "x4_lr": "64x64",
                "x4_hr": "256x256",
            })

    manifest = args.output_dir / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)

    print(f"Created paired dataset at: {args.output_dir}")
    print(f"Images processed: {len(records)}")
    print("Folder meanings:")
    print(f"  HR targets for x2/x4: {args.output_dir}/x1/<split>/...")
    print(f"  LR x2 inputs:        {args.output_dir}/x2/<split>/...")
    print(f"  LR x3 inputs:        {args.output_dir}/x3/<split>/...")
    print(f"  HR x3 targets:       {args.output_dir}/x3_hr/<split>/...")
    print(f"  LR x4 inputs:        {args.output_dir}/x4/<split>/...")
    print("The x2 training script expects x1 HR + x2 LR; use --data-dir data/edsr_data.")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
