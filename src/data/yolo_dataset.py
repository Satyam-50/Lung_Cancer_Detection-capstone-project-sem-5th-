"""Utilities for the project's 2D YOLO-format lung nodule dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val")


def list_images(dataset_root: Path, split: str) -> list[Path]:
    """Return supported images in a split, in stable order."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")
    image_dir = dataset_root / "images" / split
    if not image_dir.is_dir():
        return []
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def label_path_for_image(dataset_root: Path, split: str, image_path: Path) -> Path:
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def parse_yolo_label_file(label_path: Path) -> tuple[list[dict[str, float | int]], list[str]]:
    """Read normalized YOLO boxes and return valid rows plus line-level issues."""
    labels: list[dict[str, float | int]] = []
    issues: list[str] = []
    if not label_path.is_file():
        return labels, issues

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            issues.append(f"line {line_number}: expected 5 values, found {len(parts)}")
            continue
        try:
            class_value = float(parts[0])
            cx, cy, width, height = (float(value) for value in parts[1:])
        except ValueError:
            issues.append(f"line {line_number}: contains a non-numeric value")
            continue

        values = (class_value, cx, cy, width, height)
        if not all(value == value and abs(value) != float("inf") for value in values):
            issues.append(f"line {line_number}: contains a non-finite value")
            continue
        if class_value < 0 or not class_value.is_integer():
            issues.append(f"line {line_number}: class id must be a non-negative integer")
            continue
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
            issues.append(f"line {line_number}: box values must be normalized to [0, 1]")
            continue
        if width <= 0.0 or height <= 0.0:
            issues.append(f"line {line_number}: box width and height must be positive")
            continue

        labels.append(
            {
                "class_id": int(class_value),
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
            }
        )
    return labels, issues


def draw_yolo_boxes(image: Image.Image, labels: list[dict[str, float | int]]) -> Image.Image:
    """Draw normalized reference boxes on a copy of an image."""
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    image_width, image_height = rendered.size

    for label in labels:
        cx = float(label["cx"]) * image_width
        cy = float(label["cy"]) * image_height
        box_width = float(label["width"]) * image_width
        box_height = float(label["height"]) * image_height
        x1 = max(0, round(cx - box_width / 2))
        y1 = max(0, round(cy - box_height / 2))
        x2 = min(image_width - 1, round(cx + box_width / 2))
        y2 = min(image_height - 1, round(cy + box_height / 2))
        class_label = f"Class {int(label['class_id'])}"
        draw.rectangle((x1, y1, x2, y2), outline=(255, 76, 76), width=2)
        text_bounds = draw.textbbox((x1, y1), class_label)
        draw.rectangle(text_bounds, fill=(255, 76, 76))
        draw.text((x1, y1), class_label, fill=(255, 255, 255))

    return rendered


def audit_dataset(dataset_root: Path) -> dict[str, Any]:
    """Summarize files, annotation validity, and exact train/validation overlap."""
    dataset_root = dataset_root.expanduser().resolve()
    result: dict[str, Any] = {
        "root": str(dataset_root),
        "exists": dataset_root.is_dir(),
        "splits": {},
        "shared_exact_images": [],
        "shared_label_mismatches": [],
    }
    if not dataset_root.is_dir():
        return result

    image_hashes: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for split in SPLITS:
        images = list_images(dataset_root, split)
        labels_dir = dataset_root / "labels" / split
        label_files = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
        image_by_stem = {image.stem: image for image in images}
        label_by_stem = {label.stem: label for label in label_files}
        class_counts: Counter[int] = Counter()
        invalid_rows: list[str] = []
        image_sizes: Counter[tuple[int, int]] = Counter()

        for image_path in images:
            try:
                image_hash = sha256(image_path.read_bytes()).hexdigest()
                image_hashes[image_hash][split].append(image_path)
                with Image.open(image_path) as image:
                    image_sizes[image.size] += 1
            except (OSError, ValueError) as exc:
                invalid_rows.append(f"{image_path.name}: image read failed ({exc})")
                continue

            label_path = label_by_stem.get(image_path.stem)
            if label_path is None:
                continue
            rows, issues = parse_yolo_label_file(label_path)
            class_counts.update(int(row["class_id"]) for row in rows)
            invalid_rows.extend(f"{label_path.name} {issue}" for issue in issues)

        result["splits"][split] = {
            "images": images,
            "image_count": len(images),
            "label_count": len(label_files),
            "paired_count": len(set(image_by_stem) & set(label_by_stem)),
            "missing_label_images": sorted(set(image_by_stem) - set(label_by_stem)),
            "orphan_labels": sorted(set(label_by_stem) - set(image_by_stem)),
            "class_counts": dict(sorted(class_counts.items())),
            "invalid_rows": invalid_rows,
            "image_sizes": dict(image_sizes),
        }

    shared_images: list[dict[str, str]] = []
    shared_label_mismatches: list[str] = []
    for image_hash, split_paths in image_hashes.items():
        if "train" not in split_paths or "val" not in split_paths:
            continue
        train_image = split_paths["train"][0]
        val_image = split_paths["val"][0]
        train_label = label_path_for_image(dataset_root, "train", train_image)
        val_label = label_path_for_image(dataset_root, "val", val_image)
        shared_images.append(
            {
                "train_image": str(train_image),
                "train_images": [str(path) for path in split_paths["train"]],
                "val_image": str(val_image),
                "sha256": image_hash,
            }
        )
        train_text = train_label.read_text(encoding="utf-8").strip() if train_label.is_file() else ""
        val_text = val_label.read_text(encoding="utf-8").strip() if val_label.is_file() else ""
        if train_text != val_text:
            shared_label_mismatches.append(train_image.name)

    result["shared_exact_images"] = shared_images
    result["shared_label_mismatches"] = shared_label_mismatches
    result["train_overlap_count"] = sum(
        len(item["train_images"]) for item in shared_images
    )
    overlap_train_paths = {
        Path(path).resolve()
        for item in shared_images
        for path in item["train_images"]
    }
    clean_train_class_counts: Counter[int] = Counter()
    for image_path in result["splits"]["train"]["images"]:
        if image_path.resolve() in overlap_train_paths:
            continue
        label_path = label_path_for_image(dataset_root, "train", image_path)
        rows, _ = parse_yolo_label_file(label_path)
        clean_train_class_counts.update(int(row["class_id"]) for row in rows)
    result["clean_train_class_counts"] = dict(sorted(clean_train_class_counts.items()))
    result["clean_train_count"] = max(
        0,
        result["splits"]["train"]["image_count"] - result["train_overlap_count"],
    )
    result["clean_val_count"] = result["splits"]["val"]["image_count"]
    return result
