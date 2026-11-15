"""Image loading helpers shared by the Streamlit inference pages."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_uploaded_ct_image(filename: str, payload: bytes) -> Image.Image:
    """Load a common 2D image or a single DICOM CT file as an RGB image."""
    if Path(filename).suffix.casefold() == ".dcm":
        return _load_dicom(payload)

    with Image.open(BytesIO(payload)) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _load_dicom(payload: bytes) -> Image.Image:
    try:
        import pydicom
        from pydicom.pixels import apply_modality_lut
    except ImportError as exc:
        raise RuntimeError(
            "DICOM support is unavailable. Install the project requirements with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    try:
        dataset = pydicom.dcmread(BytesIO(payload))
        pixels = np.asarray(apply_modality_lut(dataset.pixel_array, dataset))
    except Exception as exc:
        raise ValueError(f"Could not decode this DICOM file: {exc}") from exc

    if pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
        return Image.fromarray(np.asarray(pixels, dtype=np.uint8)).convert("RGB")
    if pixels.ndim == 3:
        # A multiframe DICOM is reduced to its middle frame for 2D models.
        pixels = pixels[pixels.shape[0] // 2]
    if pixels.ndim != 2:
        raise ValueError("Upload one 2D CT slice or a single multiframe DICOM file.")

    pixels = np.asarray(pixels, dtype=np.float32)
    center = _first_dicom_number(getattr(dataset, "WindowCenter", None))
    width = _first_dicom_number(getattr(dataset, "WindowWidth", None))
    if center is not None and width is not None and width > 1:
        lower, upper = center - width / 2, center + width / 2
    else:
        lower, upper = np.nanpercentile(pixels, (1, 99))
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            lower, upper = float(np.nanmin(pixels)), float(np.nanmax(pixels))
        if upper <= lower:
            upper = lower + 1.0

    display = np.clip((pixels - lower) / (upper - lower), 0.0, 1.0)
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        display = 1.0 - display
    grayscale = np.rint(display * 255).astype(np.uint8)
    return Image.fromarray(grayscale, mode="L").convert("RGB")


def _first_dicom_number(value) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)) or hasattr(value, "__iter__") and not isinstance(value, str):
            value = next(iter(value))
        return float(value)
    except (TypeError, ValueError, StopIteration):
        return None
