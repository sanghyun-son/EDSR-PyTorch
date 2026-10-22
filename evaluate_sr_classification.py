
# evaluate_sr_classification.py

from pathlib import Path
import csv
import statistics

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18


# =========================================================
# Configuration
# =========================================================

CHECKPOINT = Path("weights/resnet18_ucmerced_x1_best.pth")

# Original HR test images, organized by class:
# x1/test/agricultural/...
TEST_ROOT = Path("data/edsr_data/x1/test")

# Root containing bicubic_x2, edsr_x2_42, etc.
RECON_ROOT = Path("data/edsr_data")

OUTPUT_ROOT = Path("outputs/sr_classification")

SEEDS = [42, 123, 999]
SCALES = [2, 3, 4]

MODELS = {
    "Bicubic": "bicubic",
    "EDSR": "edsr",
    "LW-EDSR": "light_weight",
    "LW-EDSR-SL": "light_weight_semantic",
}

BATCH_SIZE = 32
NUM_WORKERS = 0
IMAGE_SIZE = 256

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
}

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Match the preprocessing used to train your ResNet-18.
# This corresponds to RGB values normalized from [0, 1] to [-1, 1].
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])


# =========================================================
# Check configuration
# =========================================================

def check_configuration():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {CHECKPOINT}"
        )

    if not TEST_ROOT.is_dir():
        raise FileNotFoundError(
            f"Original test directory not found: {TEST_ROOT}\n"
            "Set TEST_ROOT to the folder containing the 21 "
            "ground-truth class folders."
        )

    if not RECON_ROOT.is_dir():
        raise FileNotFoundError(
            f"Reconstruction root not found: {RECON_ROOT}"
        )


# =========================================================
# Load frozen ResNet-18 classifier
# =========================================================

def load_classifier(checkpoint_path):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if "classes" not in checkpoint:
        raise KeyError("Checkpoint has no 'classes' entry.")

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint has no 'model_state_dict' entry."
        )

    classes = checkpoint["classes"]

    # Support either a list of class names or a dictionary.
    if isinstance(classes, dict):
        if all(isinstance(v, int) for v in classes.values()):
            classes = [
                name for name, idx in sorted(
                    classes.items(), key=lambda item: item[1]
                )
            ]
        else:
            classes = list(classes.keys())

    classes = list(classes)
    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(classes)
    }

    classifier = resnet18(
        weights=None,
        num_classes=len(classes),
    )

    state = checkpoint["model_state_dict"]

    # Support DataParallel checkpoints.
    if state and all(
        key.startswith("module.") for key in state
    ):
        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }

    classifier.load_state_dict(state)

    # Freeze all classifier parameters.
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    classifier.to(DEVICE)

    return classifier, classes, class_to_idx


# =========================================================
# Match original HR images with reconstructed images
# =========================================================

class SRTestDataset(Dataset):
    def __init__(
        self,
        test_root,
        recon_root,
        class_to_idx,
        transform,
    ):
        self.test_root = Path(test_root)
        self.recon_root = Path(recon_root)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = []

        if not self.recon_root.is_dir():
            raise FileNotFoundError(
                f"Reconstruction directory not found:\n"
                f"{self.recon_root}"
            )

        # Index reconstruction files.
        reconstructed_images = sorted(
            p for p in self.recon_root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        stem_index = {}
        for path in reconstructed_images:
            stem_index.setdefault(path.stem, []).append(path)

        # Read the original HR test images.
        hr_images = sorted(
            p for p in self.test_root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not hr_images:
            raise RuntimeError(
                f"No images found in {self.test_root}"
            )

        for hr_path in hr_images:
            relative = hr_path.relative_to(self.test_root)

            # Expected: test/class_name/image.ext
            if len(relative.parts) < 2:
                raise ValueError(
                    "Expected class subfolders under TEST_ROOT. "
                    f"Invalid image path: {hr_path}"
                )

            class_name = relative.parts[0]

            if class_name not in self.class_to_idx:
                raise ValueError(
                    f"Unknown class '{class_name}' in {hr_path}. "
                    "Check the classifier's class mapping and "
                    "the TEST_ROOT directory."
                )

            sr_path = None

            # First try matching the relative path.
            for extension in IMAGE_EXTENSIONS:
                candidate = (
                    self.recon_root
                    / relative.with_suffix(extension)
                )
                if candidate.is_file():
                    sr_path = candidate
                    break

            # If extensions/paths differ, match by unique stem.
            if sr_path is None:
                matches = stem_index.get(hr_path.stem, [])

                if len(matches) == 1:
                    sr_path = matches[0]
                elif len(matches) > 1:
                    raise ValueError(
                        f"Multiple reconstructions match "
                        f"'{hr_path.stem}' in {self.recon_root}. "
                        "Use matching class subfolders and "
                        "unique filenames."
                    )

            if sr_path is None:
                raise FileNotFoundError(
                    f"No reconstructed image found for:\n"
                    f"HR image: {hr_path}\n"
                    f"Reconstruction root: {self.recon_root}"
                )

            self.samples.append((
                sr_path,
                self.class_to_idx[class_name],
            ))

        if len(self.samples) != len(hr_images):
            raise RuntimeError(
                "Not all original test images were matched."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, label


# =========================================================
# Classification metrics
# =========================================================

@torch.inference_mode()
def evaluate_classifier(classifier, loader, num_classes):
    true_labels = []
    predicted_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)

        logits = classifier(images)
        predictions = logits.argmax(dim=1).cpu().tolist()

        predicted_labels.extend(predictions)
        true_labels.extend(labels.tolist())

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    ) * 100.0

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    ) * 100.0

    return accuracy, macro_f1


# =========================================================
# Main evaluation
# =========================================================

def main():
    check_configuration()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    classifier, classes, class_to_idx = load_classifier(
        CHECKPOINT
    )

    print(f"Device: {DEVICE}")
    print("Classifier: frozen ResNet-18")
    print(f"Number of classes: {len(classes)}")
    print(f"Original test directory: {TEST_ROOT}")
    print(f"Reconstruction root: {RECON_ROOT}")

    actual_class_dirs = {
        p.name for p in TEST_ROOT.iterdir() if p.is_dir()
    }
    missing_classes = set(classes) - actual_class_dirs

    if missing_classes:
        raise ValueError(
            "Some classifier classes are missing from TEST_ROOT: "
            f"{sorted(missing_classes)}"
        )

    results = []

    for model_name, model_prefix in MODELS.items():
        for scale in SCALES:

            # Bicubic is deterministic and has no training seed.
            seeds_to_evaluate = (
                [None] if model_name == "Bicubic" else SEEDS
            )

            for seed in seeds_to_evaluate:

                if model_name == "Bicubic":
                    # Assumes bicubic_x2/test, bicubic_x3/test, etc.
                    reconstruction_dir = (
                        RECON_ROOT
                        / f"bicubic_x{scale}"
                        / "test"
                    )
                else:
                    # e.g. edsr_x2_42/test
                    reconstruction_dir = (
                        RECON_ROOT
                        / f"{model_prefix}_x{scale}_{seed}"
                        / "test"
                    )

                seed_text = (
                    "" if seed is None else f", seed={seed}"
                )

                print(
                    f"\nEvaluating {model_name}, "
                    f"x{scale}{seed_text}"
                )
                print(f"Directory: {reconstruction_dir}")

                dataset = SRTestDataset(
                    test_root=TEST_ROOT,
                    recon_root=reconstruction_dir,
                    class_to_idx=class_to_idx,
                    transform=TRANSFORM,
                )

                loader = DataLoader(
                    dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                )

                accuracy, macro_f1 = evaluate_classifier(
                    classifier,
                    loader,
                    num_classes=len(classes),
                )

                print(
                    f"Images: {len(dataset)} | "
                    f"Accuracy: {accuracy:.2f}% | "
                    f"Macro-F1: {macro_f1:.2f}%"
                )

                results.append({
                    "model": model_name,
                    "scale": f"x{scale}",
                    "seed": "N/A" if seed is None else seed,
                    "accuracy_pct": accuracy,
                    "macro_f1_pct": macro_f1,
                    "num_images": len(dataset),
                })

    # -----------------------------------------------------
    # Save individual results
    # -----------------------------------------------------
    per_seed_path = OUTPUT_ROOT / "per_seed_results.csv"

    with per_seed_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    # -----------------------------------------------------
    # Aggregate mean and sample standard deviation
    # -----------------------------------------------------
    summary = []

    for model_name in MODELS:
        for scale in [f"x{s}" for s in SCALES]:
            rows = [
                row for row in results
                if row["model"] == model_name
                and row["scale"] == scale
            ]

            accuracies = [
                row["accuracy_pct"] for row in rows
            ]
            f1_scores = [
                row["macro_f1_pct"] for row in rows
            ]

            if model_name == "Bicubic":
                # One deterministic evaluation per scale.
                accuracy_mean = accuracies[0]
                accuracy_std = None
                f1_mean = f1_scores[0]
                f1_std = None
            else:
                if len(rows) != len(SEEDS):
                    raise RuntimeError(
                        f"Expected {len(SEEDS)} runs for "
                        f"{model_name} {scale}, found {len(rows)}"
                    )

                accuracy_mean = statistics.mean(accuracies)
                accuracy_std = statistics.stdev(accuracies)
                f1_mean = statistics.mean(f1_scores)
                f1_std = statistics.stdev(f1_scores)

            summary.append({
                "model": model_name,
                "scale": scale,
                "accuracy_mean_pct": accuracy_mean,
                "accuracy_std_pct": accuracy_std,
                "macro_f1_mean_pct": f1_mean,
                "macro_f1_std_pct": f1_std,
            })

    summary_path = OUTPUT_ROOT / "summary_results.csv"

    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary)

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------
    def fmt(mean, std):
        if std is None:
            return f"{mean:.2f} (single run)"
        return f"{mean:.2f} +/- {std:.2f}"

    print("\n" + "=" * 88)
    print("CLASSIFICATION SUMMARY: MEAN +/- SAMPLE STD ACROSS SEEDS")
    print("=" * 88)
    print(
        f"{'Model':<14} {'Scale':<7} "
        f"{'Accuracy (%)':<25} {'Macro-F1 (%)':<25}"
    )
    print("-" * 88)

    for row in summary:
        accuracy_text = fmt(
            row["accuracy_mean_pct"],
            row["accuracy_std_pct"],
        )
        f1_text = fmt(
            row["macro_f1_mean_pct"],
            row["macro_f1_std_pct"],
        )

        print(
            f"{row['model']:<14} "
            f"{row['scale']:<7} "
            f"{accuracy_text:<25} "
            f"{f1_text:<25}"
        )

    print("\nSaved individual results:", per_seed_path)
    print("Saved summary results:   ", summary_path)


if __name__ == "__main__":
    main()
