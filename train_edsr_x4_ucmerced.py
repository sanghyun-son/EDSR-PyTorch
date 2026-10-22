#!/usr/bin/env python3
"""
Train the EDSR implementation in this repository for UC-Merced x4 SR, with MPS synchronization and timing improvements.

Expected input layout:
  data/x1/{train,val,test}/<class>/<image>.png     # 256x256 HR targets
  data/x4/{train,val,test}/<class>/<image>.png     # 64x64 LR inputs

If the split folder is named "tet", the script also accepts that as "test".
Outputs:
  outputs/edsr_x4/best.pt
  outputs/edsr_x4/history.csv
  outputs/edsr_x4/metrics.csv
  outputs/edsr_x4/test_summary.csv
  data/edsr_x4/{train,val,test}/<class>/<image>.png
"""
import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def sync_device(device):
    """Synchronize accelerator work so wall-clock timings are meaningful."""
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def find_split(root: Path, split: str) -> Path:
    candidates = [root / split]
    if split == "test":
        candidates.append(root / "tet")
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Could not find split '{split}' under {root}. "
        "Expected train/val/test (or tet instead of test)."
    )


def image_files(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in
                  {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


class PairedUC(Dataset):
    def __init__(self, data_root: Path, split: str, augment: bool = False):
        self.lr_root = find_split(data_root / "x4", split)
        self.hr_root = find_split(data_root / "x1", split)
        self.augment = augment
        self.pairs = []
        hr_paths = {p.relative_to(self.hr_root).as_posix(): p for p in image_files(self.hr_root)}
        for lr in image_files(self.lr_root):
            rel = lr.relative_to(self.lr_root).as_posix()
            hr = hr_paths.get(rel)
            if hr is None:
                # Match by relative path without extension as a convenience.
                key = str(Path(rel).with_suffix(""))
                matches = [p for k, p in hr_paths.items()
                           if str(Path(k).with_suffix("")) == key]
                if len(matches) == 1:
                    hr = matches[0]
            if hr is not None:
                self.pairs.append((lr, hr, Path(rel)))
        if not self.pairs:
            raise RuntimeError(
                f"No LR/HR pairs found for split '{split}'. "
                f"Check that x4 and x1 have the same class/image relative paths."
            )
        # Allow a one-pixel mismatch when odd-sized HR images were downsampled
        # with integer rounding. __getitem__ crops the HR target to exactly 4x LR.
        for lr, hr, rel in self.pairs:
            with Image.open(lr) as a, Image.open(hr) as b:
                expected = (a.size[0] * 4, a.size[1] * 4)
                if abs(expected[0] - b.size[0]) > 1 or abs(expected[1] - b.size[1]) > 1:
                    raise ValueError(
                        f"Unexpected x4 pair {rel}: LR={a.size}, HR={b.size}; "
                        f"expected HR close to {expected} (at most 1 pixel mismatch per dimension). "
                        "Check that x4 LR images are paired with x1 HR targets."
                    )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_path, hr_path, rel = self.pairs[idx]
        with Image.open(lr_path) as im:
            lr_image = im.convert("RGB")
        with Image.open(hr_path) as im:
            hr_image = im.convert("RGB")

        # EDSR x4 requires the target to be exactly four times the LR dimensions.
        # Crop at most one excess row/column caused by odd-size downsampling.
        expected_hr_size = (lr_image.width * 4, lr_image.height * 4)
        if hr_image.size != expected_hr_size:
            if (abs(hr_image.width - expected_hr_size[0]) > 1 or
                    abs(hr_image.height - expected_hr_size[1]) > 1):
                raise ValueError(
                    f"Pair dimensions changed unexpectedly for {rel}: "
                    f"LR={lr_image.size}, HR={hr_image.size}"
                )
            hr_image = hr_image.crop((0, 0, expected_hr_size[0], expected_hr_size[1]))

        lr = np.asarray(lr_image, dtype=np.float32) / 255.0
        hr = np.asarray(hr_image, dtype=np.float32) / 255.0
        lr = torch.from_numpy(lr).permute(2, 0, 1)
        hr = torch.from_numpy(hr).permute(2, 0, 1)
        if self.augment:
            # Paired geometric augmentation; apply the same transform to LR and HR.
            if random.random() < 0.5:
                lr, hr = lr.flip(-1), hr.flip(-1)
            if random.random() < 0.5:
                lr, hr = lr.flip(-2), hr.flip(-2)
            k = random.randrange(4)
            if k:
                lr, hr = torch.rot90(lr, k, (-2, -1)), torch.rot90(hr, k, (-2, -1))
        return lr, hr, rel.as_posix()


def to_uint8_chw(t):
    return (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def metrics_for(pred_u8, target_u8):
    psnr = peak_signal_noise_ratio(target_u8, pred_u8, data_range=255)
    ssim = structural_similarity(target_u8, pred_u8, channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


def evaluate(model, loader, device, output_root=None, split=None):
    model.eval()
    rows = []
    with torch.no_grad():
        for lr, hr, rels in loader:
            lr = lr.to(device) * 255.0
            pred = (model(lr) / 255.0).clamp(0, 1)
            for i, rel in enumerate(rels):
                target_u8 = to_uint8_chw(hr[i])
                pred_u8 = to_uint8_chw(pred[i])
                # Bicubic baseline is generated directly from the LR image.
                lr_u8 = to_uint8_chw(lr[i] / 255.0)
                bicubic_img = Image.fromarray(lr_u8).resize(
                    (target_u8.shape[1], target_u8.shape[0]), Image.Resampling.BICUBIC
                )
                bicubic_u8 = np.asarray(bicubic_img.convert("RGB"))
                edsr_psnr, edsr_ssim = metrics_for(pred_u8, target_u8)
                bic_psnr, bic_ssim = metrics_for(bicubic_u8, target_u8)
                rows.append({
                    "split": split or "",
                    "image": rel,
                    "bicubic_psnr_db": bic_psnr,
                    "bicubic_ssim": bic_ssim,
                    "edsr_psnr_db": edsr_psnr,
                    "edsr_ssim": edsr_ssim,
                })
                if output_root is not None:
                    out_path = output_root / rel
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(pred_u8).save(out_path)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    run_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/edsr_x4"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-resblocks", type=int, default=16)
    parser.add_argument("--n-feats", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
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

    # Import the EDSR model directly from this repository's src/model/edsr.py.
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if not (src_dir / "model" / "edsr.py").is_file():
        raise FileNotFoundError(
            "Could not find src/model/edsr.py. Save this script in the repository root."
        )
    sys.path.insert(0, str(src_dir))
    from model import edsr

    model_args = SimpleNamespace(
        n_resblocks=args.n_resblocks,
        n_feats=args.n_feats,
        scale=[4],
        act="relu",
        res_scale=1.0,
        rgb_range=255,
        n_colors=3,
        shift_mean=True,
        dilation=False,
    )
    model = edsr.make_model(model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    criterion = torch.nn.L1Loss()

    train_ds = PairedUC(args.data_dir, "train", augment=True)
    train_eval_ds = PairedUC(args.data_dir, "train", augment=False)
    val_ds = PairedUC(args.data_dir, "val", augment=False)
    test_ds = PairedUC(args.data_dir, "test", augment=False)
    print(f"Images: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=1, shuffle=False,
                                   num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.workers)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.pt"
    last_checkpoint_path = args.output_dir / "last.pt"
    history_path = args.output_dir / "history.csv"
    history = []
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start_epoch = 1

    if args.resume:
        # Prefer the latest-epoch checkpoint for an actual continuation. Fall
        # back to best.pt for compatibility with runs made by older versions.
        resume_path = last_checkpoint_path if last_checkpoint_path.exists() else checkpoint_path
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_val = float(ckpt.get("best_val", ckpt.get("val_l1", float("inf"))))
            best_epoch = int(ckpt.get("best_epoch", ckpt["epoch"]))
            stale = int(ckpt.get("stale", 0))
            if history_path.exists():
                with history_path.open("r", newline="") as f:
                    history = list(csv.DictReader(f))
            print(
                f"Resuming from {resume_path.name}: next epoch {start_epoch}; "
                f"best epoch={best_epoch}, best validation L1={best_val:.6f}"
            )
        else:
            print("--resume was supplied, but no checkpoint exists; starting from scratch.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        sync_device(device)
        train_start = time.perf_counter()
        total_loss = torch.zeros((), device=device)
        n_seen = 0
        for lr, hr, _ in train_loader:
            lr = lr.to(device) * 255.0
            hr = hr.to(device) * 255.0
            optimizer.zero_grad(set_to_none=True)
            pred = model(lr)
            loss = criterion(pred, hr)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * lr.size(0)
            n_seen += lr.size(0)

        sync_device(device)
        train_seconds = time.perf_counter() - train_start
        train_l1 = (total_loss / max(n_seen, 1)).item()
        model.eval()
        val_start = time.perf_counter()
        val_total = torch.zeros((), device=device)
        val_count = 0
        val_psnr_values = []
        val_ssim_values = []
        val_bicubic_psnr_values = []
        val_bicubic_ssim_values = []
        with torch.no_grad():
            for lr, hr, _ in val_loader:
                lr_device = lr.to(device) * 255.0
                hr_device = hr.to(device) * 255.0
                pred = model(lr_device)
                loss = criterion(pred, hr_device)
                val_total += loss.detach()
                val_count += 1

                # Calculate per-image validation PSNR/SSIM for EDSR and bicubic.
                pred_u8 = to_uint8_chw((pred[0] / 255.0).clamp(0, 1))
                target_u8 = to_uint8_chw(hr[0])
                lr_u8 = to_uint8_chw(lr[0])
                bicubic_u8 = np.asarray(
                    Image.fromarray(lr_u8).resize(
                        (target_u8.shape[1], target_u8.shape[0]),
                        Image.Resampling.BICUBIC
                    ).convert("RGB")
                )
                p, ss = metrics_for(pred_u8, target_u8)
                bp, bs = metrics_for(bicubic_u8, target_u8)
                val_psnr_values.append(p)
                val_ssim_values.append(ss)
                val_bicubic_psnr_values.append(bp)
                val_bicubic_ssim_values.append(bs)

        sync_device(device)
        val_seconds = time.perf_counter() - val_start
        val_l1 = (val_total / max(val_count, 1)).item()
        epoch_seconds = time.perf_counter() - epoch_start
        val_psnr = float(np.mean(val_psnr_values))
        val_ssim = float(np.mean(val_ssim_values))
        val_bicubic_psnr = float(np.mean(val_bicubic_psnr_values))
        val_bicubic_ssim = float(np.mean(val_bicubic_ssim_values))
        history.append({
            "epoch": epoch,
            "train_l1": train_l1,
            "val_l1": val_l1,
            "val_edsr_psnr_db": val_psnr,
            "val_edsr_ssim": val_ssim,
            "val_bicubic_psnr_db": val_bicubic_psnr,
            "val_bicubic_ssim": val_bicubic_ssim,
            "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": train_seconds,
            "validation_seconds": val_seconds,
            "epoch_seconds": epoch_seconds,
        })
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train L1={train_l1:.4f} | val L1={val_l1:.4f} | "
            f"val EDSR PSNR={val_psnr:.3f} dB, SSIM={val_ssim:.5f} | "
            f"val bicubic PSNR={val_bicubic_psnr:.3f} dB, SSIM={val_bicubic_ssim:.5f} | "
            f"time train={train_seconds:.1f}s val={val_seconds:.1f}s epoch={epoch_seconds:.1f}s"
        )

        stop_early = False
        if val_l1 < best_val:
            best_val = val_l1
            best_epoch = epoch
            stale = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "val_l1": val_l1,
                        "best_val": best_val, "best_epoch": best_epoch, "stale": stale,
                        "model_args": vars(args)}, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping; best epoch was {best_epoch}.")
                stop_early = True

        if epoch % 10 == 0:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5

        write_csv(history_path, history)
        # Save the most recent completed epoch independently of the best model.
        # This lets --resume continue from the actual stopping point.
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "val_l1": val_l1,
                    "best_val": best_val, "best_epoch": best_epoch, "stale": stale,
                    "model_args": vars(args)}, last_checkpoint_path)
        if stop_early:
            break

    write_csv(history_path, history)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} (val L1={checkpoint['val_l1']:.4f})")

    evaluation_start = time.perf_counter()
    sync_device(device)
    all_rows = []
    for split, loader in [("train", train_eval_loader), ("val", val_loader), ("test", test_loader)]:
        out_root = args.data_dir / "edsr_x4" / split
        rows = evaluate(model, loader, device, output_root=out_root, split=split)
        all_rows.extend(rows)
        bic_p = np.mean([r["bicubic_psnr_db"] for r in rows])
        bic_s = np.mean([r["bicubic_ssim"] for r in rows])
        eds_p = np.mean([r["edsr_psnr_db"] for r in rows])
        eds_s = np.mean([r["edsr_ssim"] for r in rows])
        print(f"{split:5s} | bicubic PSNR={bic_p:.3f} dB SSIM={bic_s:.5f} | "
              f"EDSR PSNR={eds_p:.3f} dB SSIM={eds_s:.5f}")
    sync_device(device)
    evaluation_seconds = time.perf_counter() - evaluation_start
    write_csv(args.output_dir / "metrics.csv", all_rows)

    test_rows = [r for r in all_rows if r["split"] == "test"]
    summary = []
    for method, pkey, skey in [
        ("bicubic", "bicubic_psnr_db", "bicubic_ssim"),
        ("edsr_x4", "edsr_psnr_db", "edsr_ssim"),
    ]:
        summary.append({
            "method": method,
            "test_mean_psnr_db": float(np.mean([r[pkey] for r in test_rows])),
            "test_mean_ssim": float(np.mean([r[skey] for r in test_rows])),
            "num_test_images": len(test_rows),
        })
    write_csv(args.output_dir / "test_summary.csv", summary)
    print(f"\nSaved reconstructed images under: {args.data_dir / 'edsr_x4'}")
    print(f"Saved metrics: {args.output_dir / 'metrics.csv'}")
    print(f"Saved test summary: {args.output_dir / 'test_summary.csv'}")
    print(f"Final train-split + val + test evaluation time: {evaluation_seconds:.1f}s")
    print(f"Total script wall time: {time.perf_counter() - run_start:.1f}s")


if __name__ == "__main__":
    main()
