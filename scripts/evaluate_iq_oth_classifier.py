"""Recompute held-out classifier metrics from a saved checkpoint and split manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import classification_metrics_from_confusion  # noqa: E402
from src.models.iq_oth_classifier import build_eval_transform, load_classifier_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "iq_oth_classifier",
        help="Folder containing best.pt, metrics.json, and split_manifest.csv.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset

    output = args.output.expanduser().resolve()
    metadata_path = output / "metrics.json"
    manifest_path = output / "split_manifest.csv"
    checkpoint_path = output / "best.pt"
    if not all(path.is_file() for path in (metadata_path, manifest_path, checkpoint_path)):
        raise SystemExit(
            f"Expected best.pt, metrics.json, and split_manifest.csv under {output}. "
            "Train the classifier first."
        )

    result = json.loads(metadata_path.read_text(encoding="utf-8"))
    class_names = tuple(result["class_names"])
    class_to_index = {name: index for index, name in enumerate(class_names)}
    dataset_root = Path(result["dataset_root"])
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available in this PyTorch environment.")

    model, checkpoint_classes, image_size = load_classifier_checkpoint(checkpoint_path, device)
    if tuple(checkpoint_classes) != class_names:
        raise SystemExit("Class order in the checkpoint does not match metrics.json.")
    transform = build_eval_transform(image_size)

    class ImageRows(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            image_path = dataset_root / row["image_path"]
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            return transform(image), class_to_index[row["class_name"]]

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))

    updated_metrics = {}
    for split_name, destination_key in (
        ("validation", "validation_metrics_for_best_checkpoint"),
        ("test", "held_out_test_metrics"),
    ):
        rows = [row for row in manifest if row["split"] == split_name]
        if not rows:
            raise SystemExit(f"No {split_name} images are listed in {manifest_path}.")
        loader = DataLoader(ImageRows(rows), batch_size=args.batch_size, shuffle=False, num_workers=0)
        targets, probabilities = [], []
        model.eval()
        with torch.inference_mode():
            for images, labels in loader:
                logits = model(images.to(device))
                probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy().tolist())
                targets.extend(labels.numpy().tolist())

        predictions = np.asarray(probabilities).argmax(axis=1)
        confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        for target, predicted in zip(targets, predictions):
            confusion[int(target), int(predicted)] += 1
        metrics = classification_metrics_from_confusion(confusion, class_names)
        try:
            metrics["macro_roc_auc_ovr"] = float(
                roc_auc_score(
                    targets,
                    probabilities,
                    average="macro",
                    multi_class="ovr",
                    labels=list(range(len(class_names))),
                )
            )
        except ValueError:
            metrics["macro_roc_auc_ovr"] = None
        metrics["confusion_matrix"] = confusion.tolist()
        metrics["per_class_recall"] = {
            name: metrics["per_class_metrics"][name]["recall"] for name in class_names
        }
        old_metrics = result.get(destination_key, {})
        updated_metrics[destination_key] = {**old_metrics, **metrics}

    result.update(updated_metrics)
    result["evaluation_checkpoint"] = str(checkpoint_path)
    result["evaluation_device"] = device
    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Held-out test metrics: {json.dumps(result['held_out_test_metrics'])}")
    print(f"Updated metrics: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
