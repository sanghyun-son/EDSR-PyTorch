# UC-Merced: EDSR and ResNet-18 Quick Start

Run all commands from the repository root (`EDSR-PyTorch-DL-Project`), in the order below. Internet access is needed to download the dataset and pretrained ResNet-18 weights.

## Install dependencies

```bash
python -m pip install torch torchvision datasets pillow numpy scikit-image
```

## Download the dataset

```bash
mkdir -p data
python ucmerced_generator.py --data-dir data --seed 42
```

This creates `data/x1`, `x2`, `x3`, and `x4`, each with `train`, `val`, and `test` splits. Run this once; the script also creates `data/` automatically if needed.

## Prepare aligned HR/LR pairs

```bash
python prepare_edsr_paired_data.py --data-dir data --output-dir data/edsr_data
```

This standardizes image sizes and creates the paired dataset in `data/edsr_data/`. Wait for it to finish before continuing.

## Generate bicubic comparison images

```bash
python generate_bicubic_ucmerced.py --data-dir data/edsr_data --scales 2 3 4
```

## Train and evaluate the EDSR baselines

```bash
python train_edsr_x2_ucmerced.py --data-dir data/edsr_data
python train_edsr_x3_ucmerced.py --data-dir data/edsr_data
python train_edsr_x4_ucmerced.py --data-dir data/edsr_data
```

Each script trains its model, evaluates the best checkpoint, and exports SR images. Checkpoints and metrics go to `outputs/edsr_x2/`, `outputs/edsr_x3/`, and `outputs/edsr_x4/`; reconstructed images go to `data/edsr_data/edsr_x2/`, `edsr_x3/`, and `edsr_x4/`.

For a short execution check, append `--epochs 2`; this still uses the full dataset and runs final evaluation. To continue an interrupted EDSR run, repeat its command with `--resume`.

## Train ResNet-18 and evaluate scene classification

```bash
python train_resnet18_x1_evaluate_sr.py --data-dir data/edsr_data
```

This trains on the original-resolution images and evaluates held-out test images from the original, EDSR, and bicubic folders. The classifier checkpoint and reports are saved in `outputs/resnet18_x1_classifier/`.

## Optional additional classification reports

```bash
python evaluate_resnet18_sr_all_splits.py --data-dir data/edsr_data --checkpoint outputs/resnet18_x1_classifier/best.pt
python evaluate_resnet18_sr_combined_splits.py --data-dir data/edsr_data --checkpoint outputs/resnet18_x1_classifier/best.pt
```

Reports go to `outputs/resnet18_x1_all_splits/` and `outputs/resnet18_x1_combined_splits/`. Combined-split scores include training images; use the held-out test scores for performance comparisons.

## Existing data and results

Skip download/preparation if their outputs are already complete. Both scripts refuse to replace existing data by default; only add `--overwrite` if you intend to delete and regenerate their output folders. Results from this workflow are stored in `outputs/`, not `experiment/`.
