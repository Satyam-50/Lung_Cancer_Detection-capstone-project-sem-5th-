"""A separate image-level ResNet classifier for IQ-OTH/NCCD labels."""

from __future__ import annotations

from pathlib import Path

from src.data.iq_oth_dataset import CLASS_NAMES


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def build_classifier(num_classes: int = len(CLASS_NAMES), pretrained: bool = True):
    """Build a ResNet-18 head for the dataset's three image-level classes."""
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_eval_transform(image_size: int = 224):
    """Return the deterministic image transform shared by validation and inference."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )


def load_classifier_checkpoint(checkpoint_path: str | Path, device: str = "cpu"):
    """Load a saved classifier and its label metadata."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = tuple(checkpoint.get("class_names", CLASS_NAMES))
    model = build_classifier(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    image_size = int(checkpoint.get("image_size", 224))
    return model, class_names, image_size
