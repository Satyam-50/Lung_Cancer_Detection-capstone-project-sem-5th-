"""Train an image-level classifier on the IQ-OTH/NCCD folder dataset."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "The IQ-OTHNCCD lung cancer dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "models" / "iq_oth_classifier"
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.iq_oth_dataset import CLASS_NAMES, ImageRecord, list_image_records  # noqa: E402
from src.evaluation.metrics import classification_metrics_from_confusion  # noqa: E402
from src.models.iq_oth_classifier import (  # noqa: E402
    IMAGE_MEAN,
    IMAGE_STD,
    build_classifier,
    build_eval_transform,
)


def resolve_data_root(configured_root: str | None) -> Path:
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    environment_root = os.environ.get("PULMOAI_CLASSIFICATION_DATASET_ROOT")
    return Path(environment_root).expanduser().resolve() if environment_root else DEFAULT_DATA_ROOT


def remove_exact_duplicates(records: list[ImageRecord]) -> tuple[list[ImageRecord], int]:
    """Keep one copy of each image and reject identical pixels with conflicting labels."""
    seen: dict[str, ImageRecord] = {}
    duplicates = 0
    for record in records:
        digest = sha256(record.path.read_bytes()).hexdigest()
        previous = seen.get(digest)
        if previous is None:
            seen[digest] = record
            continue
        if previous.class_name != record.class_name:
            raise ValueError(
                "Identical image files have different class folders: "
                f"{previous.path} and {record.path}. Resolve the label conflict before training."
            )
        duplicates += 1
    return list(seen.values()), duplicates


def split_records(
    records: list[ImageRecord], seed: int, validation_fraction: float = 0.15, test_fraction: float = 0.15
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    """Create a deterministic per-class split; source files lack patient IDs."""
    rng = random.Random(seed)
    grouped = {class_name: [] for class_name in CLASS_NAMES}
    for record in records:
        grouped[record.class_name].append(record)

    train_records: list[ImageRecord] = []
    validation_records: list[ImageRecord] = []
    test_records: list[ImageRecord] = []
    for class_name in CLASS_NAMES:
        samples = sorted(grouped[class_name], key=lambda item: item.path.as_posix().casefold())
        if len(samples) < 3:
            raise ValueError(f"Class {class_name} needs at least 3 images to make train/validation/test splits.")
        rng.shuffle(samples)
        test_count = max(1, round(len(samples) * test_fraction))
        validation_count = max(1, round(len(samples) * validation_fraction))
        while test_count + validation_count >= len(samples):
            if test_count >= validation_count and test_count > 1:
                test_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                break
        test_records.extend(samples[:test_count])
        validation_records.extend(samples[test_count : test_count + validation_count])
        train_records.extend(samples[test_count + validation_count :])

    return train_records, validation_records, test_records


def write_split_manifest(
    output_root: Path,
    dataset_root: Path,
    splits: tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]],
) -> None:
    manifest_path = output_root / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=("split", "class_name", "image_path"))
        writer.writeheader()
        for split_name, records in zip(("train", "validation", "test"), splits):
            for record in records:
                writer.writerow(
                    {
                        "split": split_name,
                        "class_name": record.class_name,
                        "image_path": record.path.relative_to(dataset_root).as_posix(),
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="Folder containing the three class directories.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet weights.")
    args = parser.parse_args()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    data_root = resolve_data_root(args.data)
    records = list_image_records(data_root)
    if not records:
        raise SystemExit(
            f"No IQ-OTH/NCCD images found under {data_root}. "
            "Point --data or PULMOAI_CLASSIFICATION_DATASET_ROOT at the extracted dataset folder."
        )
    records, duplicate_count = remove_exact_duplicates(records)
    class_counts = {
        class_name: sum(record.class_name == class_name for record in records)
        for class_name in CLASS_NAMES
    }
    missing_classes = [class_name for class_name, count in class_counts.items() if count == 0]
    if missing_classes:
        raise SystemExit(f"The dataset is missing class folders for: {', '.join(missing_classes)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    splits = split_records(records, args.seed)
    train_records, validation_records, test_records = splits
    args.output.mkdir(parents=True, exist_ok=True)
    write_split_manifest(args.output, data_root.resolve(), splits)

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    eval_transform = build_eval_transform(args.image_size)

    class ImageFolderRecords(Dataset):
        def __init__(self, samples: list[ImageRecord], transform):
            self.samples = samples
            self.transform = transform
            self.class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES)}

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            record = self.samples[index]
            with Image.open(record.path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            return self.transform(image), self.class_to_idx[record.class_name]

    train_dataset = ImageFolderRecords(train_records, train_transform)
    validation_dataset = ImageFolderRecords(validation_records, eval_transform)
    test_dataset = ImageFolderRecords(test_records, eval_transform)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available in this PyTorch environment.")
    device = torch.device(device_name)

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    model = build_classifier(
        num_classes=len(CLASS_NAMES),
        pretrained=not args.no_pretrained,
    ).to(device)
    train_class_counts = {
        class_name: sum(record.class_name == class_name for record in train_records)
        for class_name in CLASS_NAMES
    }
    counts = torch.tensor([train_class_counts[name] for name in CLASS_NAMES], dtype=torch.float32)
    class_weights = counts.sum() / (len(CLASS_NAMES) * counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        total = 0
        confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
        targets_all = []
        probabilities_all = []
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)
                predictions = logits.argmax(dim=1)
                probabilities = torch.softmax(logits, dim=1)
                total_loss += float(loss.item()) * len(labels)
                total += len(labels)
                targets_all.extend(labels.cpu().tolist())
                probabilities_all.extend(probabilities.cpu().tolist())
                for target, predicted in zip(labels.cpu(), predictions.cpu()):
                    confusion[int(target), int(predicted)] += 1

        summary = classification_metrics_from_confusion(confusion.numpy(), CLASS_NAMES)
        try:
            summary["macro_roc_auc_ovr"] = float(
                roc_auc_score(
                    targets_all,
                    probabilities_all,
                    average="macro",
                    multi_class="ovr",
                    labels=list(range(len(CLASS_NAMES))),
                )
            )
        except ValueError:
            summary["macro_roc_auc_ovr"] = None
        summary["per_class_recall"] = {
            name: summary["per_class_metrics"][name]["recall"] for name in CLASS_NAMES
        }
        return {
            "loss": total_loss / max(1, total),
            **summary,
            "confusion_matrix": confusion.tolist(),
        }

    checkpoint_path = args.output / "best.pt"
    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    print(f"Dataset: {data_root}")
    print(f"Images after exact duplicate removal: {len(records)} (removed {duplicate_count})")
    print(f"Class counts: {class_counts}")
    print(
        "Split counts (train/validation/test): "
        f"{len(train_records)}/{len(validation_records)}/{len(test_records)}"
    )
    print("Note: the source images do not include patient IDs; splits are image-level.")
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        processed = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
            processed += len(labels)

        validation_metrics = evaluate(validation_loader)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={running_loss / max(1, processed):.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_accuracy={validation_metrics['accuracy']:.3f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.3f}"
        )

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": list(CLASS_NAMES),
                    "image_size": args.image_size,
                    "architecture": "resnet18",
                    "pretrained": not args.no_pretrained,
                    "seed": args.seed,
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping: validation macro F1 did not improve.")
                break

    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate(test_loader)
    result = {
        "dataset_root": str(data_root.resolve()),
        "class_names": list(CLASS_NAMES),
        "class_to_index": {name: index for index, name in enumerate(CLASS_NAMES)},
        "class_counts_after_deduplication": class_counts,
        "exact_duplicates_removed": duplicate_count,
        "split_counts": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "split_granularity": "image-level; patient identifiers are not present in source files",
        "validation_metrics_for_best_checkpoint": best_checkpoint["validation_metrics"],
        "held_out_test_metrics": test_metrics,
        "checkpoint": str(checkpoint_path.resolve()),
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Held-out test metrics: {json.dumps(test_metrics)}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Metrics: {args.output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
