from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SemanticLoss(nn.Module):
    def __init__(
        self,
        weights_path="weights/resnet18_ucmerced_x1_best.pth",
        rgb_range=255.0,
        feature_layer="layer3",
    ):
        super().__init__()

        self.rgb_range = float(rgb_range)

        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"ResNet-18 checkpoint not found: {weights_path}"
            )

        # Load the checkpoint produced by train_resnet18_ucmerced.py.
        checkpoint = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )

        num_classes = len(checkpoint["classes"])

        # Construct ResNet-18 without downloading pretrained weights.
        backbone = resnet18(
            weights=None,
            num_classes=num_classes,
        )
        backbone.load_state_dict(checkpoint["model_state_dict"])

        # Select intermediate convolutional features.
        feature_layers = {
            "layer1": [
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
            ],
            "layer2": [
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
            ],
            "layer3": [
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
            ],
            "layer4": [
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            ],
        }

        if feature_layer not in feature_layers:
            raise ValueError(
                "feature_layer must be one of: "
                "'layer1', 'layer2', 'layer3', 'layer4'"
            )

        self.features = nn.Sequential(
            *feature_layers[feature_layer]
        )

        # Freeze ResNet parameters.
        self.features.requires_grad_(False)
        self.features.eval()

        # Match the normalization used when training ResNet-18.
        self.register_buffer(
            "mean",
            torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1),
        )

    def train(self, mode=True):
        # Keep the frozen feature extractor in evaluation mode.
        super().train(mode)
        self.features.eval()
        return self

    def _prepare(self, image):
        # EDSR inputs/outputs use the [0, 255] intensity range.
        image = (image / self.rgb_range).clamp(0.0, 1.0)
        return (image - self.mean) / self.std

    def forward(self, sr, hr):
        sr = self._prepare(sr)
        hr = self._prepare(hr)

        # Gradients must flow through ResNet to the SR network.
        sr_features = self.features(sr)

        # HR features are fixed targets.
        with torch.no_grad():
            hr_features = self.features(hr)

        return F.l1_loss(sr_features, hr_features)
