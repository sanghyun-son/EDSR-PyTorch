import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/edsr_data"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weights/resnet18_ucmerced_x1_best.pth"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Normalization and augmentation
    normalize = transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((0, 0)),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = args.data_dir / "x1" / "train"
    val_dir = args.data_dir / "x1" / "val"

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=eval_transform,
    )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Train and validation class mappings do not match."
        )

    num_classes = len(train_dataset.classes)

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Classes: {num_classes}")
    print(train_dataset.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    # Train from scratch: no ImageNet weights.
    model = resnet18(weights=None, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        # ---------------- Training ----------------
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            train_count += batch_size

        train_loss = train_loss_sum / train_count
        train_acc = train_correct / train_count

        # ---------------- Validation ----------------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                batch_size = images.size(0)
                val_loss_sum += loss.item() * batch_size
                val_correct += (
                    logits.argmax(dim=1) == labels
                ).sum().item()
                val_count += batch_size

        val_loss = val_loss_sum / val_count
        val_acc = val_correct / val_count

        scheduler.step(val_acc)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.2%} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            stale_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                    "classes": train_dataset.classes,
                    "class_to_idx": train_dataset.class_to_idx,
                },
                args.output,
            )

            print(f"Saved best checkpoint: {args.output}")
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print("Early stopping.")
            break

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()
