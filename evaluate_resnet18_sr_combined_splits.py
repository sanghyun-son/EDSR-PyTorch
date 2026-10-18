#!/usr/bin/env python3
"""Evaluate a frozen ResNet-18 on each SR dataset, pooling train/val/test splits.

This script DOES NOT train or update the classifier. For each reconstruction
method (EDSR x2/x3/x4 and bicubic x2/x3/x4), it evaluates all available train,
validation, and test images, then calculates one set of metrics over the pooled
predictions.

Expected folders under --data-dir:
  edsr_x2, edsr_x3, edsr_x4, bicubic_x2, bicubic_x3, bicubic_x4
Each should contain train/, val/, and test/ (or tet/) class subfolders.

Outputs under --output-dir:
  summary_combined_splits.csv
  predictions_combined_splits.csv
  confusion_matrix_<dataset>_all_splits.csv
  classification_report_<dataset>_all_splits.csv

IMPORTANT: pooled train/validation/test metrics are descriptive combined-split
results, not an unbiased test-set estimate. Use test-only metrics as the main
held-out generalization result in a report.
"""
from __future__ import annotations

import argparse
import csv
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
DATASETS = ("edsr_x2", "edsr_x3", "edsr_x4", "bicubic_x2", "bicubic_x3", "bicubic_x4")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_split(root: Path, split: str) -> Path | None:
    candidates = [root / split]
    if split == "test":
        candidates.append(root / "tet")
    return next((p for p in candidates if p.is_dir()), None)


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


class ImageFolderWithKey(Dataset):
    def __init__(self, root: Path, class_to_idx: dict[str, int], transform):
        self.root, self.transform, self.samples = root, transform, []
        for path in list_images(root):
            rel = path.relative_to(root)
            if len(rel.parts) < 2:
                raise ValueError(f"Expected class subfolders under {root}; got {path}")
            class_name = rel.parts[0]
            if class_name not in class_to_idx:
                raise ValueError(f"Unknown class '{class_name}' in {path}; known classes: {list(class_to_idx)}")
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


def calculate_metrics(y_true: list[int], y_pred: list[int], n_classes: int):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        cm[yt, yp] += 1
    n = len(y_true)
    accuracy = float(np.trace(cm) / n) if n else 0.0
    report = []
    f1_values = []
    for idx in range(n_classes):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        report.append({
            "class_idx": idx,
            "support": int(cm[idx, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return accuracy, float(np.mean(f1_values)), cm, report


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/edsr_data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/resnet18_x1_classifier/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/resnet18_x1_combined_splits"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_root = args.data_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint or "class_to_idx" not in checkpoint:
        raise ValueError("Checkpoint must contain 'model_state_dict' and 'class_to_idx'.")
    class_to_idx = {str(k): int(v) for k, v in checkpoint["class_to_idx"].items()}
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    model = make_resnet18(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, prediction_rows = [], []
    start_all = time.perf_counter()
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Classes: {len(class_names)}")
    print("Pooling all available train/val/test images per dataset...\n")

    for dataset_name in DATASETS:
        base = data_root / dataset_name
        if not base.is_dir():
            print(f"[WARN] Missing dataset folder, skipping: {base}")
            continue

        all_true, all_pred = [], []
        total_loss, total_images, total_seconds = 0.0, 0, 0.0
        split_counts = {"train": 0, "val": 0, "test": 0}

        for split in SPLITS:
            split_root = find_split(base, split)
            if split_root is None:
                print(f"[WARN] Missing {split} split under {base}; skipping it.")
                continue
            dataset = ImageFolderWithKey(split_root, class_to_idx, transform)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers)
            split_start = time.perf_counter()
            split_loss_sum = 0.0
            split_true, split_pred = [], []
            with torch.no_grad():
                for images, labels, paths, keys, class_names_batch in loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    loss_sum = criterion(logits, labels)
                    predictions = logits.argmax(dim=1)
                    split_loss_sum += float(loss_sum.detach().cpu())
                    true_batch = labels.cpu().tolist()
                    pred_batch = predictions.cpu().tolist()
                    split_true.extend(true_batch)
                    split_pred.extend(pred_batch)
                    for path, key, class_name, yt, yp in zip(paths, keys, class_names_batch, true_batch, pred_batch):
                        prediction_rows.append({
                            "dataset": dataset_name,
                            "split": split,
                            "image": path,
                            "relative_key": key,
                            "true_class": class_name,
                            "predicted_class": class_names[yp],
                            "correct": int(yt == yp),
                        })
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - split_start

            n = len(dataset)
            split_counts[split] = n
            total_images += n
            total_loss += split_loss_sum
            total_seconds += elapsed
            all_true.extend(split_true)
            all_pred.extend(split_pred)
            split_acc = sum(a == b for a, b in zip(split_true, split_pred)) / max(n, 1)
            print(f"  {split.upper():5s}: n={n:4d} accuracy={split_acc:.2%} time={elapsed:.1f}s")

        if total_images == 0:
            continue

        accuracy, macro_f1, cm, report = calculate_metrics(all_true, all_pred, len(class_names))
        row = {
            "dataset": dataset_name,
            "n_images_total": total_images,
            "n_train": split_counts["train"],
            "n_val": split_counts["val"],
            "n_test": split_counts["test"],
            "pooled_cross_entropy_loss": total_loss / total_images,
            "pooled_accuracy": accuracy,
            "pooled_macro_f1": macro_f1,
            "evaluation_seconds_total": total_seconds,
            "checkpoint_epoch": checkpoint.get("epoch", ""),
        }
        summary_rows.append(row)

        with (output_dir / f"confusion_matrix_{dataset_name}_all_splits.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["true\\pred", *class_names])
            for idx, cm_row in enumerate(cm):
                writer.writerow([class_names[idx], *cm_row.tolist()])

        report_rows = [{
            "class": class_names[item["class_idx"]],
            "support": item["support"],
            "precision": item["precision"],
            "recall": item["recall"],
            "f1": item["f1"],
        } for item in report]
        write_rows(output_dir / f"classification_report_{dataset_name}_all_splits.csv", report_rows)
        print(f"  POOLED: n={total_images:4d} loss={total_loss / total_images:.4f} "
              f"accuracy={accuracy:.2%} macro-F1={macro_f1:.4f}\n")

    write_rows(output_dir / "summary_combined_splits.csv", summary_rows)
    write_rows(output_dir / "predictions_combined_splits.csv", prediction_rows)
    print("=== Complete ===")
    print(f"Datasets evaluated: {len(summary_rows)}")
    print(f"Total evaluation time: {time.perf_counter() - start_all:.1f}s")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
