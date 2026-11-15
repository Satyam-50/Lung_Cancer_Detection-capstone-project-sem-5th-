"""Evaluate the best saved nodule detector on its held-out validation manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "models" / "nodule_detector"
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.yolo_dataset import audit_dataset, label_path_for_image, list_images  # noqa: E402
from scripts.train_detector import DEFAULT_DATASET_ROOT  # noqa: E402


def resolve_dataset_root() -> Path:
    configured_root = os.environ.get("PULMOAI_DATASET_ROOT")
    return Path(configured_root).expanduser().resolve() if configured_root else DEFAULT_DATASET_ROOT


def create_local_validation_copy(dataset_root: Path, temporary_root: Path, audit: dict) -> tuple[Path, int]:
    """Copy validation images/labels outside synced folders for reliable evaluation."""
    image_dir = temporary_root / "images" / "val"
    label_dir = temporary_root / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    images = list_images(dataset_root, "val")
    if not images:
        raise SystemExit("No images were found in the detector validation split.")
    for image in images:
        label = label_path_for_image(dataset_root, "val", image)
        if not label.is_file():
            raise SystemExit(f"No label file was found for validation image {image.name}.")
        shutil.copy2(image, image_dir / image.name)
        shutil.copy2(label, label_dir / label.name)

    class_ids = sorted(
        {
            int(class_id)
            for split in audit["splits"].values()
            for class_id in split["class_counts"]
        }
    )
    names = {class_id: f"class_{class_id}" for class_id in class_ids}
    dataset_yaml = temporary_root / "dataset.yaml"
    dataset_yaml.write_text(
        json.dumps(
            {
                "path": str(temporary_root.resolve()),
                "train": "images/val",
                "val": "images/val",
                "names": names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_yaml, len(images)


def latest_checkpoint() -> Path | None:
    candidates = [path for path in OUTPUT_ROOT.glob("train*/weights/best.pt") if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=resolve_dataset_root())
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu", help="For example, cpu or 0.")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    dataset_root = args.dataset.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else latest_checkpoint()
    if checkpoint is None or not checkpoint.is_file():
        raise SystemExit("No trained detector checkpoint was found. Train the detector first.")

    audit = audit_dataset(dataset_root)
    if not audit["exists"]:
        raise SystemExit(f"Detector dataset was not found at {dataset_root}.")
    if audit["splits"]["val"]["missing_label_images"] or audit["splits"]["val"]["invalid_rows"]:
        raise SystemExit("The detector validation data has missing or invalid labels; review Dataset explorer.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Install requirements.txt first.") from exc

    evaluation_dir = OUTPUT_ROOT / "evaluation"
    run_name = f"{checkpoint.parent.parent.name}_validation"
    model = YOLO(str(checkpoint))
    with tempfile.TemporaryDirectory(prefix="pulmoai_detector_validation_") as temp_dir:
        temporary_root = Path(temp_dir)
        dataset_yaml, validation_image_count = create_local_validation_copy(
            dataset_root,
            temporary_root,
            audit,
        )
        results = model.val(
            data=str(dataset_yaml),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            plots=True,
            project=str(evaluation_dir),
            name=run_name,
            exist_ok=True,
            verbose=True,
        )

    names = model.names
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
    names = {int(index): str(name) for index, name in names.items()}
    box = results.box
    metrics = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "validation_image_count": validation_image_count,
    }
    per_class = {}
    for class_id, class_name in names.items():
        if class_id >= len(box.ap_class_index) or class_id not in box.ap_class_index:
            continue
        precision, recall, map50, map50_95 = box.class_result(class_id)
        per_class[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "map50": float(map50),
            "map50_95": float(map50_95),
        }

    matrix = None
    axis_labels = list(names.values())
    confusion = getattr(results, "confusion_matrix", None)
    if confusion is not None and hasattr(confusion, "matrix"):
        matrix = confusion.matrix.astype(int).tolist()
        if len(matrix) == len(axis_labels) + 1:
            axis_labels.append("background")

    save_dir = Path(results.save_dir)
    report = {
        "task": "nodule localization",
        "dataset": str(dataset_root),
        "validation_split": str(dataset_root / "images" / "val"),
        "checkpoint": str(checkpoint),
        "image_size": args.imgsz,
        "device": args.device,
        "metrics": metrics,
        "per_class_metrics": per_class,
        "class_names": names,
        "confusion_matrix": matrix,
        "confusion_matrix_labels": axis_labels,
        "confusion_matrix_image": str(save_dir / "confusion_matrix.png"),
        "normalized_confusion_matrix_image": str(save_dir / "confusion_matrix_normalized.png"),
        "evaluation_output": str(save_dir),
        "note": "Detection uses precision, recall, and mAP; classification accuracy is not a standard detection metric.",
    }
    report_path = OUTPUT_ROOT / "validation_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Checkpoint: {checkpoint}")
    print(f"Validation metrics: {json.dumps(metrics)}")
    print(f"Confusion matrix: {report['confusion_matrix_image']}")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
