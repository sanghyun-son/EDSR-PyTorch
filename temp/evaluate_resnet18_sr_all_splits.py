#!/usr/bin/env python3
"""
Evaluate one already-trained ResNet-18 checkpoint on train, validation, and test
splits of EDSR ×2/×3/×4 and bicubic ×2/×3/×4 reconstruction folders.

This script DOES NOT train or update the model. It loads the existing best.pt
from train_resnet18_x1_evaluate_sr.py and uses identical class mapping,
normalization, and 256×256 evaluation resize.

Expected folders under --data-dir:
  edsr_x2, edsr_x3, edsr_x4, bicubic_x2, bicubic_x3, bicubic_x4
each with train/val/test (or tet) class subfolders.

Checkpoint expected by default:
  outputs/resnet18_x1_classifier/best.pt

Outputs:
  summary_all_splits.csv
  predictions_all_splits.csv
  confusion_matrix_<dataset>_<split>.csv
  classification_report_<dataset>_<split>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
DATASETS = ("edsr_x2", "edsr_x3", "edsr_x4",
            "bicubic_x2", "bicubic_x3", "bicubic_x4")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def find_split(root: Path, split: str) -> Path | None:
    candidates = [root / split]
    if split == "test":
        candidates.append(root / "tet")
    return next((p for p in candidates if p.is_dir()), None)


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def class_name_for(path: Path, split_root: Path, class_to_idx: dict[str, int]) -> str:
    rel = path.relative_to(split_root)
    if len(rel.parts) < 2:
        raise ValueError(f"Expected class subfolders under {split_root}, got {path}")
    name = rel.parts[0]
    if name not in class_to_idx:
        raise ValueError(f"Unknown class '{name}' in {path}; known: {list(class_to_idx)}")
    return name


class ImageFolderWithKey(Dataset):
    def __init__(self, root: Path, class_to_idx: dict[str, int], transform):
        self.root = root
        self.transform = transform
        self.samples = []
        for path in list_images(root):
            class_name = class_name_for(path, root, class_to_idx)
            rel = path.relative_to(root)
            key = str(rel.with_suffix("")).replace("\\", "/")
            self.samples.append((path, class_to_idx[class_name], class_name, key))
        if not self.samples:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, class_name, key = self.samples[index]
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        return self.transform(image), label, str(path), key, class_name


def make_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def metrics(y_true: list[int], y_pred: list[int], n_classes: int):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        cm[yt, yp] += 1
    total = len(y_true)
    accuracy = float(np.trace(cm) / total) if total else 0.0
    per_class = []
    f1s = []
    for idx in range(n_classes):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class.append({
            "class_idx": idx, "support": int(cm[idx, :].sum()),
            "precision": precision, "recall": recall, "f1": f1,
        })
    return accuracy, float(np.mean(f1s)), cm, per_class


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    count = 0
    y_true, y_pred, paths, keys, class_names = [], [], [], [], []
    start = time.perf_counter()
    with torch.no_grad():
        for images, labels, batch_paths, batch_keys, batch_classes in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_n = labels.size(0)
            loss_sum += loss.detach() * batch_n
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum()
            count += batch_n
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            paths.extend(batch_paths)
            keys.extend(batch_keys)
            class_names.extend(batch_classes)
    sync_device(device)
    elapsed = time.perf_counter() - start
    return {
        "loss": float((loss_sum / max(count, 1)).item()),
        "accuracy": float((correct.float() / max(count, 1)).item()),
        "seconds": elapsed,
        "y_true": y_true, "y_pred": y_pred,
        "paths": paths, "keys": keys, "classes": class_names,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/edsr_data"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/resnet18_x1_classifier/best.pt"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/resnet18_x1_all_splits"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    data_root = args.data_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint or "class_to_idx" not in checkpoint:
        raise ValueError(
            "Checkpoint must be best.pt produced by train_resnet18_x1_evaluate_sr.py "
            "(needs model_state_dict and class_to_idx)."
        )
    class_to_idx = {str(k): int(v) for k, v in checkpoint["class_to_idx"].items()}
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    model = make_resnet18(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    summary_rows, prediction_rows = [], []
    output_dir.mkdir(parents=True, exist_ok=True)
    start_all = time.perf_counter()
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Classes: {len(class_names)}")

    for dataset_name in DATASETS:
        base = data_root / dataset_name
        if not base.is_dir():
            print(f"[WARN] Dataset folder missing, skipping: {base}")
            continue
        for split in SPLITS:
            split_root = find_split(base, split)
            if split_root is None:
                print(f"[WARN] Split '{split}' missing under {base}; skipping.")
                continue
            dataset = ImageFolderWithKey(split_root, class_to_idx, transform)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers)
            result = evaluate(model, loader, criterion, device)
            accuracy, macro_f1, cm, report = metrics(
                result["y_true"], result["y_pred"], len(class_names)
            )
            row = {
                "dataset": dataset_name, "split": split, "n_images": len(dataset),
                "loss": result["loss"], "accuracy": accuracy,
                "macro_f1": macro_f1, "evaluation_seconds": result["seconds"],
                "checkpoint_epoch": checkpoint.get("epoch", ""),
            }
            summary_rows.append(row)

            tag = f"{dataset_name}_{split}"
            with (output_dir / f"confusion_matrix_{tag}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["true\\\\pred", *class_names])
                for idx, cm_row in enumerate(cm):
                    writer.writerow([class_names[idx], *cm_row.tolist()])

            report_rows = [{
                "class": class_names[item["class_idx"]],
                "support": item["support"], "precision": item["precision"],
                "recall": item["recall"], "f1": item["f1"],
            } for item in report]
            write_rows(output_dir / f"classification_report_{tag}.csv", report_rows)

            for path, key, class_name, yt, yp in zip(
                result["paths"], result["keys"], result["classes"],
                result["y_true"], result["y_pred"]
            ):
                prediction_rows.append({
                    "dataset": dataset_name, "split": split, "image": path,
                    "relative_key": key, "true_class": class_name,
                    "predicted_class": class_names[yp], "correct": int(yt == yp),
                })

            print(
                f"{split.upper():5s} {dataset_name:10s} | n={len(dataset):4d} "
                f"loss={result['loss']:.4f} accuracy={accuracy:.2%} "
                f"macro-F1={macro_f1:.4f} time={result['seconds']:.1f}s"
            )

    write_rows(output_dir / "summary_all_splits.csv", summary_rows)
    write_rows(output_dir / "predictions_all_splits.csv", prediction_rows)
    print("\\n=== Complete ===")
    print(f"Datasets/splits evaluated: {len(summary_rows)}")
    print(f"Total evaluation time: {time.perf_counter() - start_all:.1f}s")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
