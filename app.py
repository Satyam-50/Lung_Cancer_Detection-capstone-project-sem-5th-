from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src.data.yolo_dataset import (
    audit_dataset,
    draw_yolo_boxes,
    label_path_for_image,
    list_images,
    parse_yolo_label_file,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "Lung Nodules Detection Dataset Annotations" / "ct_images"
)
DATASET_ROOT = Path(
    os.environ.get("PULMOAI_DATASET_ROOT", str(DEFAULT_DATASET_ROOT))
).expanduser().resolve()
MODEL_OUTPUT_ROOT = PROJECT_ROOT / "models" / "nodule_detector"

st.set_page_config(
    page_title="PulmoAI | 2D Lung Nodule Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600, max_entries=2)
def get_dataset_audit(dataset_root: str) -> dict:
    return audit_dataset(Path(dataset_root))


@st.cache_data(ttl=3600, max_entries=32)
def get_image_and_labels(image_path: str, dataset_root: str, split: str):
    path = Path(image_path)
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    labels_path = label_path_for_image(Path(dataset_root), split, path)
    labels, issues = parse_yolo_label_file(labels_path)
    return image, labels, issues, labels_path.is_file()


@st.cache_resource(max_entries=2)
def load_detector(weights_path: str):
    from ultralytics import YOLO

    return YOLO(weights_path)


def latest_checkpoint() -> Path | None:
    candidates = list(MODEL_OUTPUT_ROOT.glob("train*/weights/best.pt"))
    candidates = [path for path in candidates if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def latest_validation_metrics(checkpoint: Path | None) -> dict[str, float] | None:
    if checkpoint is None:
        return None
    results_path = checkpoint.parent.parent / "results.csv"
    if not results_path.is_file():
        return None
    try:
        frame = pd.read_csv(results_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None
    if frame.empty:
        return None

    row = frame.iloc[-1]
    keys = {
        "Precision": "metrics/precision(B)",
        "Recall": "metrics/recall(B)",
        "mAP@0.5": "metrics/mAP50(B)",
        "mAP@0.5:0.95": "metrics/mAP50-95(B)",
    }
    normalized_columns = {str(column).strip(): column for column in frame.columns}
    metrics = {}
    for label, source_column in keys.items():
        column = normalized_columns.get(source_column)
        if column is not None and pd.notna(row[column]):
            metrics[label] = float(row[column])
    return metrics or None


def render_audit_summary(audit: dict) -> None:
    train = audit["splits"]["train"]
    val = audit["splits"]["val"]
    metrics = st.columns(4)
    metrics[0].metric("Training images", train["image_count"])
    metrics[1].metric("Validation images", val["image_count"])
    metrics[2].metric("Unique train after deduplication", audit["clean_train_count"])
    metrics[3].metric("Exact train/validation overlaps", audit["train_overlap_count"])


def render_reference_image(image: Image.Image, labels: list[dict], caption: str) -> None:
    overlay = draw_yolo_boxes(image, labels)
    st.image(
        overlay,
        caption=caption,
        width="stretch",
    )


st.title("PulmoAI: 2D lung nodule localization")
st.caption(
    "Research prototype using the supplied annotated CT images. It localizes labeled findings; "
    "it does not determine whether a nodule is cancerous."
)

with st.sidebar:
    st.title("PulmoAI")
    st.caption("2D CT detection prototype")
    page = st.radio(
        "Project pages",
        ["Detector", "Dataset explorer", "Research scope", "Training guide"],
        label_visibility="visible",
    )
    checkpoint = latest_checkpoint()
    if checkpoint:
        st.success("Trained detector checkpoint found")
        st.caption(str(checkpoint.relative_to(PROJECT_ROOT)))
    else:
        st.info("No trained YOLO checkpoint yet")

if not DATASET_ROOT.is_dir():
    st.warning("The annotated dataset folder was not found at the configured path.")
    st.code(
        "Set PULMOAI_DATASET_ROOT to the folder containing images/train, images/val, "
        "labels/train, and labels/val.",
        language="text",
    )

audit = get_dataset_audit(str(DATASET_ROOT)) if DATASET_ROOT.is_dir() else None


if page == "Detector":
    st.header("Nodule detector")
    st.write(
        "Choose a validation image to compare the dataset annotation with a trained model, "
        "or upload a CT slice for model inference."
    )

    source_options = ["Validation sample", "Upload CT image"]
    source = st.radio("Input source", source_options, horizontal=True)
    input_image: Image.Image | None = None
    reference_labels: list[dict] = []
    is_reference_sample = False

    if source == "Validation sample":
        validation_images = list_images(DATASET_ROOT, "val") if audit else []
        if not validation_images:
            st.info("No validation images are available. Check the dataset path.")
        else:
            image_names = [image.name for image in validation_images]
            selected_name = st.selectbox("Choose a validation image", image_names)
            selected_image = next(image for image in validation_images if image.name == selected_name)
            input_image, reference_labels, issues, has_label_file = get_image_and_labels(
                str(selected_image), str(DATASET_ROOT), "val"
            )
            is_reference_sample = True
            if has_label_file:
                render_reference_image(
                    input_image,
                    reference_labels,
                    f"Dataset reference boxes · {selected_image.name}",
                )
            else:
                st.image(
                    input_image,
                    caption=selected_image.name,
                    width="stretch",
                )
                st.warning("No matching label file was found for this validation image.")
            if issues:
                st.warning("This image has annotation format issues; review the Dataset explorer.")
            st.caption(
                f"Reference annotations: {len(reference_labels)} boxes. "
                "They are ground truth, not model predictions."
            )
    else:
        uploaded_file = st.file_uploader(
            "Upload a CT slice",
            type=["jpg", "jpeg", "png"],
            help="Use a 2D image. A trained checkpoint is required to produce predictions.",
        )
        if uploaded_file is not None:
            try:
                with Image.open(uploaded_file) as opened:
                    input_image = ImageOps.exif_transpose(opened).convert("RGB")
                st.image(
                    input_image,
                    caption="Uploaded CT slice",
                    width="stretch",
                )
            except OSError:
                st.error("This file could not be opened as an image.")

    if input_image is not None:
        if checkpoint is None:
            st.warning(
                "Inference is unavailable until a YOLO checkpoint is trained. "
                "The boxes above, when present, are dataset annotations only."
            )
        elif importlib.util.find_spec("ultralytics") is None:
            st.error("Ultralytics is missing from this Python environment. Install requirements.txt first.")
        else:
            st.caption("Checkpoint outputs are localization predictions, not a cancer diagnosis.")

        with st.form("detector_form", border=True):
            confidence = st.slider("Minimum detection confidence", 0.05, 0.95, 0.25, 0.05)
            run_detection = st.form_submit_button(
                "Run trained detector",
                disabled=checkpoint is None or importlib.util.find_spec("ultralytics") is None,
                icon=":material/search:",
            )

        if run_detection and checkpoint is not None:
            try:
                with st.spinner("Running the trained detector…"):
                    model = load_detector(str(checkpoint))
                    prediction = model.predict(
                        source=input_image,
                        imgsz=256,
                        conf=confidence,
                        verbose=False,
                    )[0]
                    predicted_bgr = prediction.plot()
                    predicted_image = Image.fromarray(predicted_bgr[:, :, ::-1].copy())
                    st.image(
                        predicted_image,
                        caption="Model predictions",
                        width="stretch",
                    )

                    detections = []
                    boxes = prediction.boxes
                    if boxes is not None:
                        class_names = prediction.names
                        for box in boxes:
                            class_id = int(box.cls.item())
                            if isinstance(class_names, dict):
                                class_name = class_names.get(class_id, f"Class {class_id}")
                            else:
                                class_name = class_names[class_id]
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detections.append(
                                {
                                    "class_id": class_id,
                                    "class_name": class_name,
                                    "confidence": round(float(box.conf.item()), 4),
                                    "x_min": round(x1, 1),
                                    "y_min": round(y1, 1),
                                    "x_max": round(x2, 1),
                                    "y_max": round(y2, 1),
                                }
                            )
                    if detections:
                        st.dataframe(
                            pd.DataFrame(detections),
                            hide_index=True,
                        )
                    else:
                        st.info("No boxes passed the selected confidence threshold.")
                    if is_reference_sample:
                        st.caption(
                            "Compare predictions with the reference boxes above. The validation "
                            "set is small and is not an independent clinical test cohort."
                        )
            except Exception as exc:  # Surface model/config errors without crashing the whole app.
                st.error(f"Detector inference failed: {exc}")


elif page == "Dataset explorer":
    st.header("Dataset and annotations")
    st.caption(str(DATASET_ROOT))
    if audit is None:
        st.info("Place the dataset in the expected folder or set PULMOAI_DATASET_ROOT.")
    else:
        render_audit_summary(audit)
        st.write(
            "The text labels use normalized YOLO bounding boxes: "
            "`class_id x_center y_center width height`."
        )

        if audit["train_overlap_count"]:
            st.warning(
                f"The source split has {audit['train_overlap_count']} exact image copies in both "
                "train and validation. Training removes the train-side copies automatically; "
                "the source files are left unchanged."
            )
        if audit["shared_label_mismatches"]:
            st.error(
                "Some exact duplicate images have different annotation text. "
                "Training is stopped until those labels are reviewed."
            )
        if audit["splits"]["train"]["orphan_labels"]:
            st.warning(
                f"{len(audit['splits']['train']['orphan_labels'])} train label files have no "
                "matching image. They are ignored by the generated training manifest."
            )
        if audit["splits"]["train"]["missing_label_images"] or audit["splits"]["val"]["missing_label_images"]:
            st.warning(
                "Some images have no matching label file. Confirm whether these are intentionally "
                "negative images before training."
            )

        class_rows = []
        for class_id in sorted(
            {
                int(class_id)
                for split_summary in audit["splits"].values()
                for class_id in split_summary["class_counts"]
            }
        ):
            class_rows.append(
                {
                    "class_id": class_id,
                    "source train boxes": audit["splits"]["train"]["class_counts"].get(class_id, 0),
                    "train boxes used": audit["clean_train_class_counts"].get(class_id, 0),
                    "validation boxes": audit["splits"]["val"]["class_counts"].get(class_id, 0),
                    "meaning": "Not documented in the downloaded dataset",
                }
            )
        if class_rows:
            st.subheader("Class counts")
            st.dataframe(
                pd.DataFrame(class_rows),
                hide_index=True,
            )
            st.info(
                "The local dataset copy does not include a class-name file. The app preserves "
                "numeric class IDs and does not assign benign or malignant meanings."
            )

        st.subheader("Inspect a labeled image")
        split = st.selectbox("Split", ["train", "val"])
        images = list_images(DATASET_ROOT, split)
        if images:
            selected_name = st.selectbox(
                "Image",
                [image.name for image in images],
                key="explorer_image",
            )
            selected_image = next(image for image in images if image.name == selected_name)
            image, labels, issues, has_label = get_image_and_labels(
                str(selected_image), str(DATASET_ROOT), split
            )
            if has_label:
                render_reference_image(image, labels, f"Reference boxes · {selected_image.name}")
            else:
                st.image(
                    image,
                    caption=selected_image.name,
                    width="stretch",
                )
            st.write(f"Image size: **{image.width} × {image.height} px** · Boxes: **{len(labels)}**")
            if issues:
                st.error("Annotation issues: " + "; ".join(issues))
        else:
            st.info(f"No images found in the {split} split.")

        invalid_rows = {
            split_name: details["invalid_rows"]
            for split_name, details in audit["splits"].items()
            if details["invalid_rows"]
        }
        if invalid_rows:
            st.error("Invalid annotations were found. Review these before training.")
            st.json(invalid_rows)


elif page == "Research scope":
    st.header("Research-paper alignment and scope")
    st.markdown(
        "Reference: [Nguyen, *A scoping review of deep learning approaches for lung cancer "
        "detection using chest radiographs and computed tomography scans* (2025)]"
        "(https://doi.org/10.1016/j.bea.2024.100138)."
    )
    st.write(
        "The cited paper is a scoping review of published methods, not a paper proposing one "
        "architecture that this project can reproduce. This implementation takes a narrow CT "
        "detection task from that literature and chooses a model that matches the annotations "
        "available in the required dataset."
    )

    st.subheader("Implemented pipeline")
    st.code(
        "2D CT slice (JPG)\n"
        "  → pretrained lightweight YOLO detector\n"
        "  → bounding boxes and numeric class IDs\n"
        "  → validation using precision, recall, and mAP",
        language="text",
    )
    st.write(
        "Transfer learning is used as a practical response to the small dataset. Exact duplicate "
        "images shared by the provided train and validation folders are excluded from the training "
        "manifest to avoid direct image leakage."
    )

    st.subheader("What this dataset supports")
    st.markdown(
        "- **Supported:** 2D nodule/finding localization from image-level bounding boxes.\n"
        "- **Not supported by these labels:** pixel-level U-Net segmentation, benign/malignant "
        "classification, 3D volume analysis, or cancer-risk grading.\n"
        "- **Interpretability:** the interface displays localization boxes and confidence scores; "
        "it does not show a synthetic Grad-CAM or claim a clinical explanation."
    )
    st.warning(
        "This is a research prototype, not a diagnostic system. The validation set contains only "
        "41 unique images and is not an independent external test cohort. Report measured results "
        "from the trained model only, with this limitation stated."
    )


else:
    st.header("Training guide")
    st.write(
        "The dataset is already in YOLO detection layout. Training writes a filtered image-list "
        "manifest that excludes exact copies of validation images from the training split."
    )
    st.code(
        "python -m pip install -r requirements.txt\n"
        "python scripts/train_detector.py",
        language="powershell",
    )
    st.write("The script starts from the lightweight pretrained `yolo26n.pt` detector and uses 256 px inputs.")
    st.write(
        "The best checkpoint is saved under `models/nodule_detector/train*/weights/best.pt`. "
        "The app loads the newest matching checkpoint automatically."
    )
    st.info(
        "The dataset contains class IDs 0 and 1 but does not include their clinical names. "
        "Training uses generic class_0/class_1 labels until the dataset owner confirms their meanings."
    )
    if audit is not None:
        render_audit_summary(audit)
        if audit["splits"]["train"]["orphan_labels"]:
            st.caption(
                f"The source has {len(audit['splits']['train']['orphan_labels'])} unmatched train "
                "label files. The training code ignores them without deleting source files."
            )
    recorded_metrics = latest_validation_metrics(checkpoint)
    if recorded_metrics:
        st.subheader("Latest recorded validation metrics")
        st.caption("Values read from the trained model's results.csv; these are not preset benchmarks.")
        with st.container(horizontal=True):
            for metric_name in ("Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"):
                if metric_name in recorded_metrics:
                    st.metric(metric_name, f"{recorded_metrics[metric_name]:.3f}")
