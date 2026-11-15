"""Utilities for the image-level IQ-OTH/NCCD class folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


CLASS_NAMES = ("Normal", "Benign", "Malignant")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

_DIRECTORY_NAMES = {
    "normal": "Normal",
    "normalcase": "Normal",
    "normalcases": "Normal",
    "benign": "Benign",
    "benigncase": "Benign",
    "benigncases": "Benign",
    "bengin": "Benign",
    "bengincase": "Benign",
    "bengincases": "Benign",
    "malignant": "Malignant",
    "malignantcase": "Malignant",
    "malignantcases": "Malignant",
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    class_name: str


def _normalized_directory_name(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.casefold())


def find_class_directories(search_root: Path) -> dict[str, Path]:
    """Find the three class folders, including inside nested Kaggle archives."""
    search_root = search_root.expanduser().resolve()
    if not search_root.is_dir():
        return {}

    parents = [search_root]
    parents.extend(path for path in search_root.rglob("*") if path.is_dir())
    for parent in parents:
        found: dict[str, Path] = {}
        try:
            children = parent.iterdir()
            for child in children:
                if not child.is_dir():
                    continue
                class_name = _DIRECTORY_NAMES.get(_normalized_directory_name(child.name))
                if class_name is not None:
                    found.setdefault(class_name, child)
        except OSError:
            continue
        if all(class_name in found for class_name in CLASS_NAMES):
            return {class_name: found[class_name] for class_name in CLASS_NAMES}
    return {}


def list_image_records(search_root: Path) -> list[ImageRecord]:
    """Return supported images from recognized class folders in stable order."""
    class_directories = find_class_directories(search_root)
    records: list[ImageRecord] = []
    for class_name in CLASS_NAMES:
        directory = class_directories.get(class_name)
        if directory is None:
            continue
        records.extend(
            ImageRecord(path=image, class_name=class_name)
            for image in sorted(directory.iterdir(), key=lambda item: item.name.casefold())
            if image.is_file() and image.suffix.casefold() in IMAGE_SUFFIXES
        )
    return records


def count_images_by_class(search_root: Path) -> dict[str, int]:
    """Count supported images for each documented IQ-OTH/NCCD category."""
    counts = {class_name: 0 for class_name in CLASS_NAMES}
    for record in list_image_records(search_root):
        counts[record.class_name] += 1
    return counts
