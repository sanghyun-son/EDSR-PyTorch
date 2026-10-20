#!/usr/bin/env python3
"""
Train ResNet-18 on original UC-Merced x1 images and evaluate it on:
  - original x1 test images
  - EDSR ×2, ×3, ×4 reconstructions
  - bicubic ×2, ×3, ×4 reconstructions

Expected layout under --data-dir (default data/edsr_data):
  x1/{train,val,test}/<class>/<image>
  edsr_x2/{train,val,test}/<class>/<image>
  edsr_x3/{train,val,test}/<class>/<image>
  edsr_x4/{train,val,test}/<class>/<image>
  bicubic_x2/{train,val,test}/<class>/<image>
  bicubic_x3/{train,val,test}/<class>/<image>
  bicubic_x4/{train,val,test}/<class>/<image>

A test split named "tet" is accepted as an alternative to "test".
Images are resized to 256x256 and normalized using ImageNet statistics for all
datasets, so the x3 255x255 outputs are handled consistently.

Outputs under --output-dir:
  best.pt                 best checkpoint selected by validation loss
  history.csv             per-epoch training/validation loss and accuracy
  test_summary.csv        metrics for x1 and all six reconstruction methods
  test_predictions.csv    per-image prediction and correctness
  confusion_matrix_<name>.csv
  classification_report_<name>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sync_device(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def find_split(root: Path, split: str) -> Path | None:
    candidates = [root / split]
    if split == "test":
        candidates.append(root / "tet")
    return next((p for p in candidates if p.is_dir()), None)


def list_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_class_mapping(data_root: Path, train_root: Path) -> dict[str, int]:
    mapping_path = data_root / "class_to_idx.json"
    if mapping_path.is_file():
        with mapping_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        mapping = {str(k): int(v) for k, v in raw.items()}
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError(f"Invalid class indices in {mapping_path}: {mapping}")
        return mapping

    class_names = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if not class_names:
        raise RuntimeError(f"No class folders found under {train_root}")
    return {name: idx for idx, name in enumerate(class_names)}


def class_name_for(path: Path, split_root: Path, class_to_idx: dict[str, int]) -> str:
    rel = path.relative_to(split_root)
    if len(rel.parts) < 2:
        raise ValueError(
            f"Expected images under class subfolders, but found {path} relative to {split_root}"
        )
    class_name = rel.parts[0]
    if class_name not in class_to_idx:
        raise ValueError(
            f"Unknown class folder '{class_name}' in {path}. "
            f"Known classes: {list(class_to_idx)}"
        )
    return class_name


class UCMercedImages(Dataset):
    def __init__(
        self,
        split_root: Path,
        class_to_idx: dict[str, int],
        transform,
        expected_classes: set[str] | None = None,
    ):
        self.root = split_root
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples: list[tuple[Path, int, str, str]] = []
        self.key_to_sample: dict[str, tuple[Path, int, str, str]] = {}

        for path in list_images(split_root):
            class_name = class_name_for(path, split_root, class_to_idx)
            rel = path.relative_to(split_root)
            key = str(rel.with_suffix("")).replace("\\", "/")
            sample = (path, class_to_idx[class_name], class_name, key)
            self.samples.append(sample)
            self.key_to_sample[key] = sample

        if not self.samples:
            raise RuntimeError(f"No images found in {split_root}")

        found_classes = {sample[2] for sample in self.samples}
        if expected_classes is not None:
            unexpected = found_classes - expected_classes
            if unexpected:
                raise ValueError(f"Unexpected classes in {split_root}: {sorted(unexpected)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, class_name, key = self.samples[index]
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        return self.transform(image), label, str(path), key, class_name


def make_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def calculate_metrics(y_true: list[int], y_pred: list[int], num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total else 0.0
    per_class = []
    f1_values = []
    for idx in range(num_classes):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = int(cm[idx, :].sum())
        per_class.append({
            "class_idx": idx,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        f1_values.append(f1)
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    return accuracy, macro_f1, cm, per_class


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    count = 0
    all_labels, all_preds, all_paths, all_keys, all_classes = [], [], [], [], []
    start = time.perf_counter()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, paths, keys, class_names in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_n = labels.size(0)
            loss_sum += loss.detach() * batch_n
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum()
            count += batch_n

            if not training:
                all_labels.extend(labels.detach().cpu().tolist())
                all_preds.extend(predictions.detach().cpu().tolist())
                all_paths.extend(list(paths))
                all_keys.extend(list(keys))
                all_classes.extend(list(class_names))

    sync_device(device)
    elapsed = time.perf_counter() - start
    result = {
        "loss": float((loss_sum / max(count, 1)).item()),
        "accuracy": float((correct.float() / max(count, 1)).item()),
        "seconds": elapsed,
    }
    if not training:
        result.update({
            "labels": all_labels,
            "predictions": all_preds,
            "paths": all_paths,
            "keys": all_keys,
            "classes": all_classes,
        })
    return result


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_dataset(model, dataset, loader, criterion, device, class_names, output_dir, tag):
    result = run_epoch(model, loader, criterion, device)
    y_true, y_pred = result["labels"], result["predictions"]
    accuracy, macro_f1, cm, per_class = calculate_metrics(y_true, y_pred, len(class_names))
    result["accuracy"] = accuracy
    result["macro_f1"] = macro_f1

    cm_path = output_dir / f"confusion_matrix_{tag}.csv"
    with cm_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *class_names])
        for idx, row in enumerate(cm):
            writer.writerow([class_names[idx], *row.tolist()])

    report_rows = []
    for row in per_class:
        report_rows.append({
            "class": class_names[row["class_idx"]],
            "support": row["support"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
        })
    write_rows(output_dir / f"classification_report_{tag}.csv", report_rows)

    prediction_rows = []
    for path, key, class_name, true, pred in zip(
        result["paths"], result["keys"], result["classes"], y_true, y_pred
    ):
        prediction_rows.append({
            "dataset": tag,
            "image": path,
            "relative_key": key,
            "true_class": class_name,
            "predicted_class": class_names[pred],
            "correct": int(true == pred),
        })

    # Verify that the reconstruction set contains the same class/image stems as x1 test.
    expected_keys = set(dataset.expected_test_keys) if hasattr(dataset, "expected_test_keys") else None
    summary = {
        "dataset": tag,
        "n_images": len(y_true),
        "loss": result["loss"],
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "evaluation_seconds": result["seconds"],
    }
    return summary, prediction_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/edsr_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/resnet18_x1_classifier"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train from random initialization instead of ImageNet weights.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable random horizontal/vertical flips on x1 training images.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    data_root = args.data_dir.expanduser().resolve()
    x1_train_root = find_split(data_root / "x1", "train")
    if x1_train_root is None:
        raise FileNotFoundError(f"Missing x1 training split under {data_root / 'x1'}")
    class_to_idx = load_class_mapping(data_root, x1_train_root)
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    print(f"Classes ({len(class_names)}): {class_names}")

    train_transform_list = []
    if not args.no_augment:
        train_transform_list += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    train_transform_list += [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    train_transform = transforms.Compose(train_transform_list)
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = UCMercedImages(
        x1_train_root, class_to_idx, train_transform, expected_classes=set(class_names)
    )
    val_root = find_split(data_root / "x1", "val")
    test_root = find_split(data_root / "x1", "test")
    if val_root is None or test_root is None:
        raise FileNotFoundError("Expected x1 val and test (or tet) split folders.")
    val_dataset = UCMercedImages(val_root, class_to_idx, eval_transform, set(class_names))
    x1_test_dataset = UCMercedImages(test_root, class_to_idx, eval_transform, set(class_names))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)
    criterion = nn.CrossEntropyLoss()
    model = make_model(len(class_names), pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.csv"
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    run_start = time.perf_counter()

    print(f"Images: train={len(train_dataset)}, val={len(val_dataset)}, "
          f"x1_test={len(x1_test_dataset)}")
    print(f"Training ResNet-18 for up to {args.epochs} epochs.")

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, device, optimizer)
        val_result = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "train_seconds": train_result["seconds"],
            "val_loss": val_result["loss"],
            "val_accuracy": val_result["accuracy"],
            "val_seconds": val_result["seconds"],
            "epoch_seconds": train_result["seconds"] + val_result["seconds"],
        }
        history.append(row)
        write_rows(history_path, history)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss={row['train_loss']:.4f} acc={row['train_accuracy']:.2%} "
            f"({row['train_seconds']:.1f}s) | "
            f"val loss={row['val_loss']:.4f} acc={row['val_accuracy']:.2%} "
            f"({row['val_seconds']:.1f}s)"
        )

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "args": vars(args),
            }, best_path)
            print(f"  Saved best checkpoint: {best_path}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs (patience={args.patience}).")
                break

    if not best_path.is_file():
        raise RuntimeError("Training did not produce a best checkpoint.")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} "
          f"(validation loss={checkpoint['best_val_loss']:.4f}).")

    # Build the evaluation list. Every reconstruction is evaluated only on the held-out test split.
    eval_specs = [
        ("x1", data_root / "x1", "test"),
        ("edsr_x2", data_root / "edsr_x2", "test"),
        ("edsr_x3", data_root / "edsr_x3", "test"),
        ("edsr_x4", data_root / "edsr_x4", "test"),
        ("bicubic_x2", data_root / "bicubic_x2", "test"),
        ("bicubic_x3", data_root / "bicubic_x3", "test"),
        ("bicubic_x4", data_root / "bicubic_x4", "test"),
    ]

    summaries, all_predictions = [], []
    for tag, base_root, split in eval_specs:
        split_root = find_split(base_root, split)
        if split_root is None:
            print(f"[WARN] Skipping {tag}: test split not found under {base_root}.")
            continue
        dataset = UCMercedImages(split_root, class_to_idx, eval_transform, set(class_names))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)
        summary, predictions = evaluate_dataset(
            model, dataset, loader, criterion, device, class_names, output_dir, tag
        )
        summaries.append(summary)
        all_predictions.extend(predictions)
        print(
            f"TEST {tag:10s} | n={summary['n_images']:4d} "
            f"loss={summary['loss']:.4f} accuracy={summary['accuracy']:.2%} "
            f"macro-F1={summary['macro_f1']:.4f} "
            f"time={summary['evaluation_seconds']:.1f}s"
        )

    write_rows(output_dir / "test_summary.csv", summaries)
    write_rows(output_dir / "test_predictions.csv", all_predictions)

    total_seconds = time.perf_counter() - run_start
    print("\\n=== Complete ===")
    print(f"Best epoch: {best_epoch}; best validation loss: {best_val_loss:.4f}")
    print(f"Total training + evaluation wall time: {total_seconds / 60:.2f} min")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
