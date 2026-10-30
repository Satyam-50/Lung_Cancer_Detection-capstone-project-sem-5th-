"""Fine-tune a small Ultralytics detector on the local annotated CT slices."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "Lung Nodules Detection Dataset Annotations" / "ct_images"
)
OUTPUT_ROOT = PROJECT_ROOT / "models" / "nodule_detector"

sys.path.insert(0, str(PROJECT_ROOT))
from src.data.yolo_dataset import audit_dataset, list_images  # noqa: E402


def resolve_dataset_root() -> Path:
    configured_root = os.environ.get("PULMOAI_DATASET_ROOT")
    return Path(configured_root).expanduser().resolve() if configured_root else DEFAULT_DATASET_ROOT


def write_training_files(dataset_root: Path, audit: dict) -> tuple[Path, Path, Path]:
    """Create manifests that remove exact train/validation image overlap."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    split_dir = OUTPUT_ROOT / "split_manifests"
    split_dir.mkdir(parents=True, exist_ok=True)

    duplicate_train_paths = {
        Path(train_path).resolve()
        for item in audit["shared_exact_images"]
        for train_path in item["train_images"]
    }
    validation_hashes = {
        item["sha256"] for item in audit["shared_exact_images"]
    }
    train_images = [
        image
        for image in list_images(dataset_root, "train")
        if image.resolve() not in duplicate_train_paths
        and sha256(image.read_bytes()).hexdigest() not in validation_hashes
    ]
    val_images = list_images(dataset_root, "val")

    # Refuse to proceed if duplicate labels disagree: the same pixel data must
    # have the same reference box regardless of which split it came from.
    if audit["shared_label_mismatches"]:
        examples = ", ".join(audit["shared_label_mismatches"][:5])
        raise ValueError(
            "Exact train/validation image duplicates have different label files. "
            f"Review the annotations before training. Examples: {examples}"
        )

    train_manifest = split_dir / "train.txt"
    val_manifest = split_dir / "val.txt"
    train_manifest.write_text(
        "\n".join(image.resolve().as_posix() for image in train_images) + "\n",
        encoding="utf-8",
    )
    val_manifest.write_text(
        "\n".join(image.resolve().as_posix() for image in val_images) + "\n",
        encoding="utf-8",
    )

    class_ids = sorted(
        {
            int(class_id)
            for split in audit["splits"].values()
            for class_id in split["class_counts"]
        }
    )
    if not class_ids:
        raise ValueError("No valid YOLO annotations were found in train or validation.")
    expected_ids = list(range(class_ids[-1] + 1))
    if class_ids != expected_ids:
        raise ValueError(
            f"YOLO class IDs must be contiguous from 0. Found {class_ids}; "
            "correct the annotation mapping before training."
        )

    # The downloaded dataset does not include a class-name map. Generic names
    # preserve the numeric IDs without inventing clinical meanings.
    names = "\n".join(f"  {class_id}: class_{class_id}" for class_id in class_ids)
    dataset_yaml = OUTPUT_ROOT / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {json.dumps(dataset_root.resolve().as_posix())}",
                f"train: {json.dumps(train_manifest.resolve().as_posix())}",
                f"val: {json.dumps(val_manifest.resolve().as_posix())}",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return train_manifest, val_manifest, dataset_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=resolve_dataset_root())
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained Ultralytics model.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="For example: cpu, 0, or 0,1.")
    args = parser.parse_args()

    dataset_root = args.dataset.expanduser().resolve()
    audit = audit_dataset(dataset_root)
    if not audit["exists"]:
        raise SystemExit(
            f"Dataset not found at {dataset_root}. Set PULMOAI_DATASET_ROOT or use --dataset."
        )
    if audit["splits"]["train"]["image_count"] == 0 or audit["splits"]["val"]["image_count"] == 0:
        raise SystemExit("Both images/train and images/val must contain images.")
    if audit["splits"]["train"]["missing_label_images"] or audit["splits"]["val"]["missing_label_images"]:
        raise SystemExit(
            "Some images have no matching annotation file. Review the Dataset explorer "
            "and fix/confirm those cases before training."
        )
    if audit["splits"]["train"]["invalid_rows"] or audit["splits"]["val"]["invalid_rows"]:
        raise SystemExit("Invalid YOLO annotation rows were found. Review them before training.")

    train_manifest, val_manifest, dataset_yaml = write_training_files(dataset_root, audit)
    print(f"Dataset: {dataset_root}")
    print(f"Unique training images after leakage removal: {len(train_manifest.read_text(encoding='utf-8').splitlines())}")
    print(f"Unique validation images: {len(val_manifest.read_text(encoding='utf-8').splitlines())}")
    if audit["splits"]["train"]["orphan_labels"]:
        print(
            "Warning: ignoring "
            f"{len(audit['splits']['train']['orphan_labels'])} training label files "
            "that do not have a matching image. The source dataset was left unchanged."
        )
    if audit["shared_exact_images"]:
        print(
            "Removed exact image duplicates from the training manifest: "
            f"{audit['train_overlap_count']}. Validation files remain intact."
        )
    print(f"Generated dataset configuration: {dataset_yaml}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc

    model = YOLO(args.model)
    train_results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        patience=10,
        seed=42,
        deterministic=True,
        pretrained=True,
        project=str(OUTPUT_ROOT),
        name="train",
        plots=True,
        val=True,
    )
    save_dir = Path(getattr(train_results, "save_dir", OUTPUT_ROOT / "train"))
    print(f"Training artifacts: {save_dir}")
    print(f"Best checkpoint: {save_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
