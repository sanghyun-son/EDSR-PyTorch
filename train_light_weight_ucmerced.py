
#!/usr/bin/env python3

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from src.model import light_weight
from src.loss.semantic_loss import SemanticLoss


# =========================================================
# Reproducibility and device
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def synchronize_device(device):
    """Wait for asynchronous device operations before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# =========================================================
# Dataset utilities
# =========================================================

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


def image_files(root):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_split(root, split):
    candidates = [root / split]

    if split == "test":
        candidates.append(root / "tet")

    for path in candidates:
        if path.is_dir():
            return path

    raise FileNotFoundError(
        f"Cannot find split '{split}' under {root}"
    )


class PairedScaleUC(Dataset):
    """
    Paired UC Merced dataset.

    x2: LR = x2, HR = x1
    x3: LR = x3, HR = x3_hr
    x4: LR = x4, HR = x1

    Returns the relative LR path so reconstructed images retain
    the original class directory, filename, and file extension.
    """

    def __init__(
        self,
        data_dir,
        split,
        scale,
        augment=False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.scale = scale
        self.augment = augment

        lr_root = self.data_dir / f"x{scale}"
        hr_folder = "x3_hr" if scale == 3 else "x1"

        self.lr_root = find_split(lr_root, split)
        self.hr_root = find_split(
            self.data_dir / hr_folder,
            split,
        )

        self.pairs = []

        for lr_path in image_files(self.lr_root):
            relative_path = lr_path.relative_to(self.lr_root)
            hr_path = self.hr_root / relative_path

            if not hr_path.is_file():
                raise FileNotFoundError(
                    f"HR counterpart not found for:\n"
                    f"  LR: {lr_path}\n"
                    f"  Expected HR: {hr_path}"
                )

            with Image.open(lr_path) as image:
                lr_width, lr_height = image.size

            with Image.open(hr_path) as image:
                hr_width, hr_height = image.size

            expected_size = (
                lr_width * scale,
                lr_height * scale,
            )
            actual_size = (hr_width, hr_height)

            if actual_size != expected_size:
                raise ValueError(
                    f"Image dimensions do not match scale x{scale}:\n"
                    f"  LR: {lr_path} ({lr_width}x{lr_height})\n"
                    f"  HR: {hr_path} ({hr_width}x{hr_height})\n"
                    f"  Expected HR: {expected_size}"
                )

            self.pairs.append((lr_path, hr_path))

        if not self.pairs:
            raise RuntimeError(
                f"No paired images found for scale x{scale}, "
                f"split '{split}'"
            )

        print(
            f"[Dataset] x{scale} | {split}: "
            f"{len(self.pairs)} image pairs"
        )

    @staticmethod
    def load_image(path):
        with Image.open(path) as image:
            image = image.convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0

        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        lr_path, hr_path = self.pairs[index]

        lr = self.load_image(lr_path)
        hr = self.load_image(hr_path)

        if self.augment:
            if random.random() < 0.5:
                lr = TF.hflip(lr)
                hr = TF.hflip(hr)

            if random.random() < 0.5:
                lr = TF.vflip(lr)
                hr = TF.vflip(hr)

            rotations = random.randint(0, 3)

            if rotations:
                lr = torch.rot90(
                    lr, rotations, dims=(-2, -1)
                )
                hr = torch.rot90(
                    hr, rotations, dims=(-2, -1)
                )

        # Example: agricultural/agricultural00.png
        relative_path = str(
            lr_path.relative_to(self.lr_root)
        )

        return lr, hr, relative_path


# =========================================================
# Image metrics
# =========================================================

def calculate_psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2).item()

    if mse == 0:
        return float("inf")

    return -10.0 * np.log10(mse)


def calculate_ssim(sr, hr):
    from skimage.metrics import structural_similarity

    sr_np = (
        sr.detach()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )

    hr_np = (
        hr.detach()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )

    return structural_similarity(
        hr_np,
        sr_np,
        data_range=1.0,
        channel_axis=-1,
    )


def save_image(tensor, path):
    array = (
        tensor.detach()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).round().astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # The original filename and extension are preserved.
    Image.fromarray(array).save(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def average(values):
    return float(np.mean(values)) if values else float("nan")


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def evaluate(model, loader, device, output_dir=None):
    model.eval()
    results = []

    for lr, hr, relative_paths in loader:
        lr = lr.to(device) * 255.0
        hr = hr.to(device) * 255.0

        sr = model(lr).clamp(0, 255)

        bicubic = F.interpolate(
            lr,
            size=hr.shape[-2:],
            mode="bicubic",
            align_corners=False,
        ).clamp(0, 255)

        for i, relative_path in enumerate(relative_paths):
            sr_img = (sr[i] / 255.0).clamp(0, 1)
            hr_img = (hr[i] / 255.0).clamp(0, 1)
            bicubic_img = (bicubic[i] / 255.0).clamp(0, 1)

            results.append({
                "image": relative_path,
                "bicubic_psnr_db": calculate_psnr(
                    bicubic_img, hr_img
                ),
                "bicubic_ssim": calculate_ssim(
                    bicubic_img, hr_img
                ),
                "lightweight_psnr_db": calculate_psnr(
                    sr_img, hr_img
                ),
                "lightweight_ssim": calculate_ssim(
                    sr_img, hr_img
                ),
            })

            if output_dir is not None:
                # Preserve class folder, basename, and extension.
                save_image(
                    sr_img,
                    Path(output_dir) / relative_path,
                )

    return results


# =========================================================
# Training
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train lightweight EDSR on UC Merced "
            "for x2, x3, or x4 super-resolution."
        )
    )

    # Dataset and output
    parser.add_argument(
        "--scale",
        type=int,
        choices=[2, 3, 4],
        required=True,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/edsr_data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    # Lightweight EDSR architecture
    parser.add_argument("--n-resblocks", type=int, default=8)
    parser.add_argument("--n-feats", type=int, default=32)
    parser.add_argument("--res-scale", type=float, default=1.0)

    # Optional semantic loss
    parser.add_argument(
        "--semantic-loss",
        action="store_true",
        help="Add frozen ResNet-18 feature loss.",
    )
    parser.add_argument(
        "--semantic-weights",
        type=Path,
        default=Path(
            "weights/resnet18_ucmerced_x1_best.pth"
        ),
    )
    parser.add_argument(
        "--lambda-semantic",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--feature-layer",
        choices=["layer1", "layer2", "layer3", "layer4"],
        default="layer3",
    )

    args = parser.parse_args()

    if args.n_resblocks < 1 or args.n_feats < 1:
        parser.error(
            "--n-resblocks and --n-feats must be positive"
        )

    if args.epochs < 1 or args.batch_size < 1:
        parser.error(
            "--epochs and --batch-size must be positive"
        )

    if args.lr <= 0 or args.patience < 1:
        parser.error(
            "--lr must be positive and --patience at least 1"
        )

    if args.lambda_semantic < 0:
        parser.error("--lambda-semantic cannot be negative")

    set_seed(args.seed)
    device = get_device()

    # Keep experiment checkpoints/logs separate.
    experiment_name = (
        f"light_weight_semantic_x{args.scale}_{args.seed}"
        if args.semantic_loss
        else f"light_weight_x{args.scale}_{args.seed}"
    )

    if args.output_dir is None:
        args.output_dir = Path("outputs") / experiment_name

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"Experiment       : {experiment_name}")
    print(f"Scale            : x{args.scale}")
    print(f"Device           : {device}")
    print(f"Residual blocks  : {args.n_resblocks}")
    print(f"Feature channels : {args.n_feats}")
    print(f"Semantic loss    : {args.semantic_loss}")
    if args.semantic_loss:
        print(f"Feature layer    : {args.feature_layer}")
        print(f"Semantic weight  : {args.lambda_semantic}")
    print(f"Output directory : {args.output_dir}")
    print("=" * 78)

    # -----------------------------------------------------
    # Datasets and loaders
    # -----------------------------------------------------

    train_set = PairedScaleUC(
        args.data_dir, "train", args.scale, augment=True
    )
    val_set = PairedScaleUC(
        args.data_dir, "val", args.scale, augment=False
    )
    test_set = PairedScaleUC(
        args.data_dir, "test", args.scale, augment=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # Fixed evaluation order, one image per batch.
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model_args = argparse.Namespace(
        scale=[args.scale],
        rgb_range=255,
        n_colors=3,
        n_resblocks=args.n_resblocks,
        n_feats=args.n_feats,
        res_scale=args.res_scale,
    )

    model = light_weight.make_model(model_args).to(device)

    pixel_criterion = nn.L1Loss()

    semantic_criterion = None

    if args.semantic_loss:
        if not args.semantic_weights.is_file():
            raise FileNotFoundError(
                "ResNet-18 weights not found: "
                f"{args.semantic_weights}"
            )

        semantic_criterion = SemanticLoss(
            args.semantic_weights,
            feature_layer=args.feature_layer,
        ).to(device)

        semantic_criterion.eval()

        # The semantic network is frozen; only EDSR is optimized.
        for parameter in semantic_criterion.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # -----------------------------------------------------
    # Checkpointing and training history
    # -----------------------------------------------------

    checkpoint_path = args.output_dir / "best_model.pth"
    history_path = args.output_dir / "training_history.csv"

    best_val_l1 = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    # -----------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        # ---------------- Training ----------------
        model.train()

        total_loss = 0.0
        total_pixel_loss = 0.0
        total_semantic_loss = 0.0

        train_psnr_values = []
        train_ssim_values = []

        synchronize_device(device)
        train_start = time.perf_counter()

        for lr, hr, _ in train_loader:
            lr = lr.to(device) * 255.0
            hr = hr.to(device) * 255.0

            optimizer.zero_grad(set_to_none=True)

            sr = model(lr)
            pixel_loss = pixel_criterion(sr, hr)

            if semantic_criterion is not None:
                semantic_loss_value = semantic_criterion(sr, hr)

                loss = (
                    pixel_loss
                    + args.lambda_semantic * semantic_loss_value
                )

                total_semantic_loss += (
                    semantic_loss_value.detach().item()
                )
            else:
                loss = pixel_loss

            # Metrics are measured on the predictions before the
            # optimizer update for this batch.
            with torch.no_grad():
                sr_metric = (sr.detach() / 255.0).clamp(0, 1)
                hr_metric = (hr / 255.0).clamp(0, 1)

                for i in range(sr.shape[0]):
                    train_psnr_values.append(
                        calculate_psnr(
                            sr_metric[i], hr_metric[i]
                        )
                    )
                    train_ssim_values.append(
                        calculate_ssim(
                            sr_metric[i], hr_metric[i]
                        )
                    )

            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
            total_pixel_loss += pixel_loss.detach().item()

        synchronize_device(device)
        train_seconds = time.perf_counter() - train_start

        avg_train_loss = total_loss / len(train_loader)
        avg_train_pixel = total_pixel_loss / len(train_loader)

        avg_train_semantic = (
            total_semantic_loss / len(train_loader)
            if semantic_criterion is not None
            else 0.0
        )

        avg_train_psnr = average(train_psnr_values)
        avg_train_ssim = average(train_ssim_values)

        # ---------------- Validation ----------------
        model.eval()

        val_pixel_total = 0.0
        val_psnr_values = []
        val_ssim_values = []

        synchronize_device(device)
        val_start = time.perf_counter()

        with torch.no_grad():
            for lr, hr, _ in val_loader:
                lr = lr.to(device) * 255.0
                hr = hr.to(device) * 255.0

                sr = model(lr)
                val_pixel_total += pixel_criterion(sr, hr).item()

                sr_metric = (sr / 255.0).clamp(0, 1)
                hr_metric = (hr / 255.0).clamp(0, 1)

                val_psnr_values.append(
                    calculate_psnr(sr_metric[0], hr_metric[0])
                )
                val_ssim_values.append(
                    calculate_ssim(sr_metric[0], hr_metric[0])
                )

        synchronize_device(device)
        val_seconds = time.perf_counter() - val_start

        avg_val_pixel = val_pixel_total / len(val_loader)
        avg_val_psnr = average(val_psnr_values)
        avg_val_ssim = average(val_ssim_values)

        history_row = {
            "epoch": epoch,
            "train_total_loss": avg_train_loss,
            "train_pixel_l1": avg_train_pixel,
            "train_semantic_loss": avg_train_semantic,
            "train_psnr_db": avg_train_psnr,
            "train_ssim": avg_train_ssim,
            "train_time_seconds": train_seconds,
            "val_pixel_l1": avg_val_pixel,
            "val_psnr_db": avg_val_psnr,
            "val_ssim": avg_val_ssim,
            "val_time_seconds": val_seconds,
        }

        history.append(history_row)

        # One compact summary per epoch.
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train: "
            f"L1={avg_train_pixel:.6f}, "
            f"PSNR={avg_train_psnr:.3f} dB, "
            f"SSIM={avg_train_ssim:.4f}, "
            f"Time={train_seconds:.1f}s | "
            f"Val: "
            f"L1={avg_val_pixel:.6f}, "
            f"PSNR={avg_val_psnr:.3f} dB, "
            f"SSIM={avg_val_ssim:.4f}, "
            f"Time={val_seconds:.1f}s | "
            f"Total={val_seconds+train_seconds:.1f}s"
        )

        # Save the best model based on validation pixel L1.
        if avg_val_pixel < best_val_l1:
            best_val_l1 = avg_val_pixel
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scale": args.scale,
                    "n_resblocks": args.n_resblocks,
                    "n_feats": args.n_feats,
                    "res_scale": args.res_scale,
                    "semantic_loss": args.semantic_loss,
                    "lambda_semantic": args.lambda_semantic,
                    "feature_layer": args.feature_layer,
                    "val_pixel_l1": best_val_l1,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        # Keep the CSV current even if training stops early.
        write_csv(history_path, history)

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping: validation L1 did not improve "
                f"for {args.patience} epochs."
            )
            break

    # -----------------------------------------------------
    # Load best checkpoint and final evaluation
    # -----------------------------------------------------

    print()
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation pixel L1: {best_val_l1:.6f}")
    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Image root matches the requested format:
    # data/edsr_data/light_weight_x{scale}/<split>/<class>/<filename>
    #
    # Note: running a second experiment at the same scale will replace
    # images saved by the first experiment in this shared image folder.
    experiment_name = (
        f"light_weight_semantic_x{args.scale}_{args.seed}"
        if args.semantic_loss
        else f"light_weight_x{args.scale}_{args.seed}"
    )

    image_root = args.data_dir / experiment_name
    for split, loader in (
        ("val", val_loader),
        ("test", test_loader),
    ):
        image_output_dir = image_root / split

        results = evaluate(
            model,
            loader,
            device,
            output_dir=image_output_dir,
        )

        write_csv(
            args.output_dir / f"{split}_metrics.csv",
            results,
        )

        print(f"\n{split.upper()} results ({len(results)} images)")
        print(
            f"  Bicubic PSNR : "
            f"{average([r['bicubic_psnr_db'] for r in results]):.4f} dB"
        )
        print(
            f"  Model PSNR   : "
            f"{average([r['lightweight_psnr_db'] for r in results]):.4f} dB"
        )
        print(
            f"  Bicubic SSIM : "
            f"{average([r['bicubic_ssim'] for r in results]):.6f}"
        )
        print(
            f"  Model SSIM   : "
            f"{average([r['lightweight_ssim'] for r in results]):.6f}"
        )
        print(f"  Images saved : {image_output_dir}")

    print("\nTraining and evaluation complete.")


if __name__ == "__main__":
    main()
