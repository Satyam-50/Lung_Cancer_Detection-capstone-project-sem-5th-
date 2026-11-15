from __future__ import annotations

import importlib.util
from io import BytesIO
import json
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
from src.data.iq_oth_dataset import (
    CLASS_NAMES as IQ_OTH_CLASS_NAMES,
    list_image_records,
)
from src.models.iq_oth_classifier import build_eval_transform, load_classifier_checkpoint
from src.evaluation.metrics import classification_metrics_from_confusion
from src.utils.image_loader import load_uploaded_ct_image


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "Lung Nodules Detection Dataset Annotations" / "ct_images"
)
DATASET_ROOT = Path(
    os.environ.get("PULMOAI_DATASET_ROOT", str(DEFAULT_DATASET_ROOT))
).expanduser().resolve()
MODEL_OUTPUT_ROOT = PROJECT_ROOT / "models" / "nodule_detector"
DEFAULT_IQ_OTH_ROOT = PROJECT_ROOT / "The IQ-OTHNCCD lung cancer dataset"
IQ_OTH_ROOT = Path(
    os.environ.get("PULMOAI_CLASSIFICATION_DATASET_ROOT", str(DEFAULT_IQ_OTH_ROOT))
).expanduser().resolve()
IQ_OTH_MODEL_OUTPUT_ROOT = PROJECT_ROOT / "models" / "iq_oth_classifier"

st.set_page_config(
    page_title="PulmoAI | Lung CT research",
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


@st.cache_data(ttl=3600, max_entries=48)
def get_iq_oth_image(image_path: str):
    with Image.open(image_path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


@st.cache_data(ttl=3600, max_entries=2)
def get_iq_oth_records(dataset_root: str):
    return [
        {"path": str(record.path), "class_name": record.class_name}
        for record in list_image_records(Path(dataset_root))
    ]


@st.cache_data(ttl=300, max_entries=8)
def load_json_report(report_path: str, modified_at_ns: int) -> dict | None:
    path = Path(report_path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_resource(max_entries=2)
def load_detector(weights_path: str):
    from ultralytics import YOLO

    return YOLO(weights_path)


@st.cache_resource(max_entries=2)
def load_iq_oth_classifier(weights_path: str, _modified_at_ns: int):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names, image_size = load_classifier_checkpoint(weights_path, device)
    return model, class_names, image_size, device


def latest_checkpoint() -> Path | None:
    candidates = list(MODEL_OUTPUT_ROOT.glob("train*/weights/best.pt"))
    candidates = [path for path in candidates if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def latest_iq_oth_checkpoint() -> Path | None:
    checkpoint = IQ_OTH_MODEL_OUTPUT_ROOT / "best.pt"
    return checkpoint if checkpoint.is_file() else None


def latest_validation_metrics(checkpoint: Path | None) -> dict[str, float] | None:
    if checkpoint is None:
        return None
    saved_report = MODEL_OUTPUT_ROOT / "validation_metrics.json"
    if saved_report.is_file():
        report = load_json_report(str(saved_report), saved_report.stat().st_mtime_ns)
        if report and isinstance(report.get("metrics"), dict):
            metrics = report["metrics"]
            return {
                "Precision": float(metrics["precision"]),
                "Recall": float(metrics["recall"]),
                "mAP@0.5": float(metrics["map50"]),
                "mAP@0.5:0.95": float(metrics["map50_95"]),
            }
    results_path = checkpoint.parent.parent / "results.csv"
    if not results_path.is_file():
        return None
    try:
        frame = pd.read_csv(results_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None
    if frame.empty:
        return None

    map_column = "metrics/mAP50-95(B)"
    if map_column in frame and frame[map_column].notna().any():
        row = frame.loc[frame[map_column].idxmax()]
    else:
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
        alt="CT image with dataset reference bounding boxes",
        width="stretch",
    )


st.title("PulmoAI", icon=":material/monitor_heart:")
st.caption("Lung CT analysis workspace · college research prototype")

with st.sidebar:
    st.title("PulmoAI")
    st.caption("Lung CT research dashboard")
    page = st.radio(
        "Workspace",
        [
            "Analyze CT",
            "Model performance",
            "Detector",
            "IQ-OTH/NCCD classifier",
            "Dataset explorer",
            "Research scope",
            "Training guide",
        ],
        label_visibility="visible",
    )
    checkpoint = latest_checkpoint()
    if checkpoint:
        st.badge("Nodule model ready", icon=":material/check_circle:", color="green")
    else:
        st.badge("Nodule model missing", icon=":material/error:", color="red")
    classifier_checkpoint = latest_iq_oth_checkpoint()
    if classifier_checkpoint:
        st.badge("IQ-OTH classifier ready", icon=":material/check_circle:", color="green")
    else:
        st.badge("IQ-OTH classifier missing", icon=":material/error:", color="red")

    st.caption("Models run locally in this app process.")

if not DATASET_ROOT.is_dir() and page in {"Detector", "Dataset explorer", "Training guide"}:
    st.warning("The annotated dataset folder was not found at the configured path.")
    st.code(
        "Set PULMOAI_DATASET_ROOT to the folder containing images/train, images/val, "
        "labels/train, and labels/val.",
        language="text",
    )

audit = get_dataset_audit(str(DATASET_ROOT)) if DATASET_ROOT.is_dir() else None


if page == "Analyze CT":
    st.header("Analyze a CT slice", icon=":material/search:")
    st.write(
        "Upload one chest CT image. PulmoAI runs nodule localization and the IQ-OTH/NCCD "
        "image classifier together, then shows each model's output side by side. Classification "
        "covers the whole slice; it does not assign a cancer label to an individual detected box."
    )

    iq_records = get_iq_oth_records(str(IQ_OTH_ROOT))
    source_options = ["Upload a scan"]
    if iq_records:
        source_options.append("Use an IQ-OTH/NCCD sample")
    if audit and list_images(DATASET_ROOT, "val"):
        source_options.append("Use a nodule validation sample")
    source = st.segmented_control(
        "Image source",
        source_options,
        default=source_options[0],
        key="combined_image_source",
    )

    input_image: Image.Image | None = None
    input_name = "uploaded_ct_slice"
    expected_class: str | None = None
    reference_labels: list[dict] = []
    if source == "Use an IQ-OTH/NCCD sample" and iq_records:
        class_available = [
            class_name
            for class_name in IQ_OTH_CLASS_NAMES
            if any(record["class_name"] == class_name for record in iq_records)
        ]
        selected_class = st.selectbox("Sample category", class_available, key="combined_sample_class")
        class_records = [
            record for record in iq_records if record["class_name"] == selected_class
        ]
        selected_path = st.selectbox(
            "Choose a CT slice",
            [record["path"] for record in class_records],
            format_func=lambda path: Path(path).name,
            key="combined_sample_path",
        )
        input_image = get_iq_oth_image(selected_path)
        input_name = Path(selected_path).name
        expected_class = selected_class
    elif source == "Use a nodule validation sample" and audit:
        validation_images = list_images(DATASET_ROOT, "val")
        selected_image = st.selectbox(
            "Choose a labeled validation slice",
            validation_images,
            format_func=lambda path: path.name,
            key="combined_validation_path",
        )
        input_image, reference_labels, issues, has_label_file = get_image_and_labels(
            str(selected_image),
            str(DATASET_ROOT),
            "val",
        )
        input_name = selected_image.name
        if not has_label_file:
            st.warning("This validation image has no matching reference annotation file.")
        if issues:
            st.warning("The selected reference annotations contain format issues.")
    else:
        uploaded_file = st.file_uploader(
            "CT image or DICOM file",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm"],
            help="Upload one 2D CT slice. DICOM uploads use their stored window settings when available.",
            key="combined_ct_upload",
        )
        if uploaded_file is not None:
            input_name = Path(uploaded_file.name).name
            try:
                input_image = load_uploaded_ct_image(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
            except Exception as exc:
                st.error(f"Could not read this CT image: {exc}")

    if input_image is not None:
        st.image(
            input_image,
            caption=f"Input · {input_name}",
            alt="Uploaded 2D CT slice",
            width="stretch",
        )
        if expected_class:
            st.caption(f"IQ-OTH/NCCD sample folder label: **{expected_class}**")
        if reference_labels or source == "Use a nodule validation sample":
            st.caption(f"Nodule validation reference boxes: **{len(reference_labels)}**")
            if reference_labels:
                with st.expander("Show reference annotations"):
                    render_reference_image(
                        input_image,
                        reference_labels,
                        f"Dataset reference boxes · {input_name}",
                    )

    confidence = st.slider(
        "Nodule box confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.10,
        step=0.05,
        help="Higher values show fewer, higher-scoring candidate boxes.",
        key="combined_confidence",
    )
    detector_available = checkpoint is not None and importlib.util.find_spec("ultralytics") is not None
    classifier_available = (
        classifier_checkpoint is not None
        and importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torchvision") is not None
    )
    can_analyze = input_image is not None and (detector_available or classifier_available)
    run_combined = st.button(
        "Analyze slice",
        type="primary",
        icon=":material/search:",
        disabled=not can_analyze,
    )

    if input_image is None:
        st.caption("Choose a sample or upload a CT image to begin.")
    elif not detector_available and not classifier_available:
        st.warning("No trained model is ready. Train the detector and classifier from Training guide.")

    if run_combined and input_image is not None:
        detections: list[dict] = []
        annotated_image: Image.Image | None = None
        detector_error: str | None = None
        classifier_error: str | None = None
        class_names: tuple[str, ...] = ()
        class_scores: list[float] = []
        device = "not run"

        with st.spinner("Running the available CT models…"):
            if detector_available and checkpoint is not None:
                try:
                    detection_model = load_detector(str(checkpoint))
                    prediction = detection_model.predict(
                        source=input_image,
                        imgsz=256,
                        conf=confidence,
                        verbose=False,
                    )[0]
                    predicted_bgr = prediction.plot()
                    annotated_image = Image.fromarray(predicted_bgr[:, :, ::-1].copy())
                    boxes = prediction.boxes
                    if boxes is not None:
                        model_names = prediction.names
                        for box in boxes:
                            class_id = int(box.cls.item())
                            if isinstance(model_names, dict):
                                class_name = model_names.get(class_id, f"class_{class_id}")
                            else:
                                class_name = model_names[class_id]
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detections.append(
                                {
                                    "Class ID": class_id,
                                    "Dataset class": class_name,
                                    "Model score": round(float(box.conf.item()), 4),
                                    "Left (px)": round(x1, 1),
                                    "Top (px)": round(y1, 1),
                                    "Right (px)": round(x2, 1),
                                    "Bottom (px)": round(y2, 1),
                                }
                            )
                except Exception as exc:
                    detector_error = str(exc)

            if classifier_available and classifier_checkpoint is not None:
                try:
                    import torch

                    classifier_model, class_names, image_size, device = load_iq_oth_classifier(
                        str(classifier_checkpoint), classifier_checkpoint.stat().st_mtime_ns
                    )
                    image_tensor = build_eval_transform(image_size)(input_image).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        class_scores = torch.softmax(
                            classifier_model(image_tensor), dim=1
                        )[0].cpu().tolist()
                except Exception as exc:
                    classifier_error = str(exc)

        image_col, summary_col = st.columns([1.25, 1], gap="large")
        with image_col:
            st.subheader("Nodule localization")
            st.image(
                annotated_image or input_image,
                caption="Predicted boxes" if annotated_image else "Input image",
                alt="CT image with model-predicted candidate nodule boxes",
                width="stretch",
            )
            if detector_error:
                st.error(f"Nodule detector failed: {detector_error}")
            elif not detector_available:
                st.info("Nodule detector checkpoint or Ultralytics is unavailable.")
            elif detections:
                st.success(f"{len(detections)} candidate box(es) found above the selected threshold.")
            else:
                st.info(
                    "No boxes passed this threshold. That does not confirm the scan is nodule-free."
                )

        with summary_col:
            st.subheader("IQ-OTH/NCCD category")
            if classifier_error:
                st.error(f"Image classifier failed: {classifier_error}")
            elif class_scores:
                result_index = max(range(len(class_scores)), key=class_scores.__getitem__)
                top_score = class_scores[result_index]
                st.metric("Highest-scoring dataset label", class_names[result_index])
                st.progress(
                    min(max(top_score, 0.0), 1.0),
                    text=f"Model score: {top_score * 100:.1f}%",
                )
                score_frame = pd.DataFrame(
                    [
                        {"Dataset label": name, "Model score (%)": round(score * 100, 2)}
                        for name, score in zip(class_names, class_scores)
                    ]
                )
                st.dataframe(score_frame, hide_index=True, width="stretch")
                if expected_class:
                    st.caption(f"Reference folder label: **{expected_class}**")
            elif not classifier_available:
                st.info("IQ-OTH/NCCD classifier checkpoint or PyTorch is unavailable.")

            st.caption(
                "The class scores are model outputs, not calibrated probabilities or a medical diagnosis. "
                "The IQ-OTH classifier analyzes the full slice; the detector boxes are not passed into it."
            )

        if detections:
            st.subheader("Candidate box details")
            st.dataframe(pd.DataFrame(detections), hide_index=True, width="stretch")

        result_payload = {
            "input_image": input_name,
            "detector": {
                "checkpoint": str(checkpoint) if checkpoint else None,
                "confidence_threshold": confidence,
                "detections": detections,
                "reference_box_count": len(reference_labels)
                if source == "Use a nodule validation sample"
                else None,
                "class_ids_are_clinically_named": False,
            },
            "iq_oth_classifier": {
                "checkpoint": str(classifier_checkpoint) if classifier_checkpoint else None,
                "dataset_label": class_names[
                    max(range(len(class_scores)), key=class_scores.__getitem__)
                ]
                if class_scores
                else None,
                "model_scores": dict(zip(class_names, class_scores)),
                "reference_folder_label": expected_class,
                "device": device,
            },
            "notice": "Research output only; not a cancer diagnosis or a calibrated risk probability.",
        }
        download_col, _ = st.columns([1, 2])
        with download_col:
            if annotated_image:
                png_buffer = BytesIO()
                annotated_image.save(png_buffer, format="PNG")
                st.download_button(
                    "Download annotated image",
                    data=png_buffer.getvalue(),
                    file_name=f"{Path(input_name).stem}_pulmoai.png",
                    mime="image/png",
                    icon=":material/download:",
                    on_click="ignore",
                )
            st.download_button(
                "Download analysis report",
                data=json.dumps(result_payload, indent=2),
                file_name=f"{Path(input_name).stem}_pulmoai_report.json",
                mime="application/json",
                icon=":material/description:",
                on_click="ignore",
            )

    st.warning(
        "PulmoAI is a college research prototype. The detector marks candidate boxes; the "
        "classifier predicts one IQ-OTH/NCCD folder label. These outputs do not establish "
        "whether a person has cancer."
    )


elif page == "Model performance":
    st.header("Model performance", icon=":material/query_stats:")
    st.write("Metrics are calculated from the saved checkpoints and their held-out data splits.")

    classifier_metrics_path = IQ_OTH_MODEL_OUTPUT_ROOT / "metrics.json"
    classifier_report = (
        load_json_report(str(classifier_metrics_path), classifier_metrics_path.stat().st_mtime_ns)
        if classifier_metrics_path.is_file()
        else None
    )
    test_metrics = (classifier_report or {}).get("held_out_test_metrics", {})
    if test_metrics:
        st.subheader("IQ-OTH/NCCD classifier · held-out test split")
        st.caption(
            f"Test images: {test_metrics.get('sample_count', (classifier_report or {}).get('split_counts', {}).get('test', 'unknown'))} · "
            "The source does not include patient IDs, so this split is image-level."
        )
        class_names = (classifier_report or {}).get("class_names", list(IQ_OTH_CLASS_NAMES))
        confusion = test_metrics.get("confusion_matrix")
        class_metrics = test_metrics.get("per_class_metrics")
        if not class_metrics and confusion:
            class_metrics = classification_metrics_from_confusion(
                confusion,
                class_names,
            )["per_class_metrics"]
        if class_metrics:
            rows = []
            for class_name in class_names:
                values = class_metrics.get(class_name, {})
                rows.append(
                    {
                        "Dataset label": class_name,
                        "Precision": values.get("precision"),
                        "Recall": values.get("recall"),
                        "Specificity": values.get("specificity"),
                        "F1 score": values.get("f1"),
                        "Support": values.get("support"),
                    }
                )
            st.bar_chart(
                pd.DataFrame(rows).set_index("Dataset label")[
                    ["Precision", "Recall", "F1 score"]
                ],
                height=300,
                alt="Precision, recall, and F1 by IQ-OTH/NCCD dataset label",
            )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        if confusion:
            st.subheader("Confusion matrix")
            st.caption("Rows are the true dataset labels; columns are the model predictions.")
            st.dataframe(
                pd.DataFrame(confusion, index=class_names, columns=class_names),
                width="stretch",
            )
        st.caption("Confusion matrix cells show image counts, not percentages.")

        st.subheader("Classifier test metrics")
        st.caption(
            "These scores describe the full held-out test set. They cannot be calculated for one "
            "unlabeled uploaded image. Precision, recall, specificity, and F1 are macro averages."
        )
        primary_metric_cols = st.columns(4)
        for column, label, key in zip(
            primary_metric_cols,
            ("Accuracy", "Precision", "Recall", "Specificity"),
            ("accuracy", "macro_precision", "macro_recall", "macro_specificity"),
        ):
            value = test_metrics.get(key)
            column.metric(label, f"{value * 100:.2f}%" if value is not None else "N/A")
        secondary_metric_cols = st.columns(4)
        for column, label, key in zip(
            secondary_metric_cols,
            ("F1 score", "Weighted F1", "ROC-AUC (OvR)", "Test loss"),
            ("macro_f1", "weighted_f1", "macro_roc_auc_ovr", "loss"),
        ):
            value = test_metrics.get(key)
            column.metric(label, f"{value:.4f}" if value is not None else "N/A")

        with st.expander("What these metrics mean"):
            st.markdown(
                "- **Accuracy:** fraction of all test images classified correctly.\n"
                "- **Precision:** when the model predicts a class, how often that class is correct.\n"
                "- **Recall:** fraction of the true images of a class that the model finds.\n"
                "- **Specificity:** fraction of images from other classes correctly recognized as not this class.\n"
                "- **F1 score:** balance between precision and recall."
            )
    else:
        st.info("Classifier test metrics are not available. Run the classifier training script first.")

    st.subheader("Nodule detector · validation split")
    detector_report_path = MODEL_OUTPUT_ROOT / "validation_metrics.json"
    detector_report = (
        load_json_report(str(detector_report_path), detector_report_path.stat().st_mtime_ns)
        if detector_report_path.is_file()
        else None
    )
    if detector_report:
        detector_metrics = detector_report.get("metrics", {})
        st.caption(
            f"Validation images: {detector_metrics.get('validation_image_count', 'unknown')} · "
            "Detector class IDs are kept generic because the source dataset does not document their names."
        )
        detection_cols = st.columns(4)
        for column, label, key in zip(
            detection_cols,
            ("Precision", "Recall", "mAP @ 0.5", "mAP @ 0.5:0.95"),
            ("precision", "recall", "map50", "map50_95"),
        ):
            value = detector_metrics.get(key)
            column.metric(label, f"{value * 100:.2f}%" if value is not None else "N/A")
        per_class = detector_report.get("per_class_metrics", {})
        if per_class:
            st.dataframe(
                pd.DataFrame(
                    [{"Detector class": name, **values} for name, values in per_class.items()]
                ),
                hide_index=True,
                width="stretch",
            )
        matrix = detector_report.get("confusion_matrix")
        matrix_labels = detector_report.get("confusion_matrix_labels", [])
        if matrix and len(matrix_labels) == len(matrix):
            st.caption("Detector confusion matrix, including the background row/column.")
            st.dataframe(
                pd.DataFrame(matrix, index=matrix_labels, columns=matrix_labels),
                width="stretch",
            )
        matrix_image = Path(detector_report.get("confusion_matrix_image", ""))
        if matrix_image.is_file():
            st.image(
                str(matrix_image),
                caption="Detector validation confusion matrix",
                alt="Confusion matrix for nodule detector validation",
                width="stretch",
            )
        st.caption(
            "Object detectors are assessed with precision, recall, and mean average precision. "
            "Classification accuracy is not the standard detection metric."
        )
    else:
        st.info("Detector validation report is not available yet.")
        if checkpoint:
            st.code("python scripts/evaluate_detector.py", language="powershell")


elif page == "Detector":
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
                    alt="Validation CT image without a matching label file",
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
            "Upload a CT image or DICOM file",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "dcm"],
            help="Use one 2D CT slice. DICOM window settings are used when available.",
        )
        if uploaded_file is not None:
            try:
                input_image = load_uploaded_ct_image(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
                st.image(
                    input_image,
                    caption="Uploaded CT slice",
                    alt="Uploaded CT slice for nodule detection",
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
            confidence = st.slider("Minimum detection confidence", 0.05, 0.95, 0.10, 0.05)
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
                        alt="CT slice with predicted nodule detection boxes",
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


elif page == "IQ-OTH/NCCD classifier":
    st.header("IQ-OTH/NCCD image classifier")
    st.write(
        "This separate pipeline assigns one dataset category to a whole CT slice. "
        "It does not draw nodule boxes or estimate a patient's cancer risk."
    )

    iq_records = get_iq_oth_records(str(IQ_OTH_ROOT))
    if iq_records:
        class_counts = {class_name: 0 for class_name in IQ_OTH_CLASS_NAMES}
        for record in iq_records:
            class_counts[record["class_name"]] += 1
        count_frame = pd.DataFrame(
            [{"Dataset category": name, "Images found": class_counts[name]} for name in IQ_OTH_CLASS_NAMES]
        )
        st.dataframe(count_frame, hide_index=True, width="stretch")
        st.caption(
            "Training index map: 0 = Normal, 1 = Benign, 2 = Malignant. "
            "These are dataset categories, not a clinical diagnosis."
        )
    else:
        st.warning("IQ-OTH/NCCD class folders were not found.")
        st.code(
            "Set PULMOAI_CLASSIFICATION_DATASET_ROOT to the folder that contains "
            "Normal cases, Bengin cases/Benign cases, and Malignant cases.",
            language="text",
        )

    source_options = ["Upload CT image"]
    if iq_records:
        source_options.insert(0, "Choose labeled dataset sample")
    source = st.radio("Image source", source_options, horizontal=True)
    input_image: Image.Image | None = None
    expected_class: str | None = None

    if source == "Choose labeled dataset sample":
        chosen_class = st.selectbox("Dataset category", IQ_OTH_CLASS_NAMES)
        class_records = [record for record in iq_records if record["class_name"] == chosen_class]
        if class_records:
            selected_path = st.selectbox(
                "CT slice",
                [record["path"] for record in class_records],
                format_func=lambda path: Path(path).name,
            )
            input_image = get_iq_oth_image(selected_path)
            expected_class = chosen_class
            st.caption(f"Folder label: **{expected_class}**")
    else:
        uploaded_file = st.file_uploader(
            "Upload a CT image or DICOM file",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "dcm"],
            help="Use a 2D image. A trained IQ-OTH/NCCD classifier is required for model scores.",
        )
        if uploaded_file is not None:
            try:
                input_image = load_uploaded_ct_image(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
            except Exception as exc:
                st.error(f"This file could not be opened as a CT image: {exc}")

    if input_image is not None:
        st.image(
            input_image,
            caption="CT slice" if expected_class is None else f"Labeled sample · {expected_class}",
            alt="Two-dimensional CT slice submitted to the IQ-OTH/NCCD classifier",
            width="stretch",
        )

    if classifier_checkpoint is None:
        st.info("Train the separate IQ-OTH/NCCD classifier to enable model predictions.")
        st.code("python scripts/train_iq_oth_classifier.py", language="powershell")
        st.caption(
            "The detector and classifier are trained separately because one predicts boxes "
            "and the other predicts a whole-image category."
        )

    dependencies_available = (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torchvision") is not None
    )
    with st.form("iq_oth_classification_form", border=True):
        run_classification = st.form_submit_button(
            "Classify CT slice",
            disabled=(input_image is None or classifier_checkpoint is None or not dependencies_available),
            icon=":material/analytics:",
        )

    if classifier_checkpoint is not None and not dependencies_available:
        st.error("PyTorch or Torchvision is missing. Install the project requirements first.")

    if run_classification and input_image is not None and classifier_checkpoint is not None:
        try:
            import torch

            with st.spinner("Running the IQ-OTH/NCCD classifier…"):
                model, class_names, image_size, device = load_iq_oth_classifier(
                    str(classifier_checkpoint), classifier_checkpoint.stat().st_mtime_ns
                )
                image_tensor = build_eval_transform(image_size)(input_image).unsqueeze(0).to(device)
                with torch.inference_mode():
                    scores = torch.softmax(model(image_tensor), dim=1)[0].cpu().tolist()

            result_index = max(range(len(scores)), key=scores.__getitem__)
            st.metric("Highest scoring dataset category", class_names[result_index])
            score_frame = pd.DataFrame(
                [
                    {"Dataset category": class_name, "Model score (%)": round(score * 100, 2)}
                    for class_name, score in zip(class_names, scores)
                ]
            )
            st.dataframe(score_frame, hide_index=True, width="stretch")
            st.caption(
                "Scores show the model's relative output across these three dataset categories. "
                "They are not calibrated probabilities or a medical diagnosis."
            )
            if expected_class is not None:
                st.caption(f"Reference folder label: **{expected_class}**")
        except Exception as exc:  # Show checkpoint and image errors without stopping the app.
            st.error(f"Classification failed: {exc}")


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
                    alt="CT slice without a matching reference label file",
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

    st.subheader("Implemented pipelines")
    st.code(
        "One uploaded 2D CT slice\n"
        "  ├─ YOLO detector → candidate box locations and source class IDs\n"
        "  └─ ResNet-18 → Normal / Benign / Malignant IQ-OTH/NCCD dataset label",
        language="text",
    )
    st.write(
        "The Analyze CT page runs both existing models together for a user-uploaded slice. Their "
        "training stays separate because the localization dataset has boxes and undocumented IDs, "
        "while IQ-OTH/NCCD has only whole-image category labels. Exact copies shared by the "
        "detector's train and validation folders are excluded from its training manifest."
    )

    st.subheader("What this dataset supports")
    st.markdown(
        "- **Detection dataset:** 2D nodule/finding localization from bounding boxes; its class IDs "
        "remain generic because their meanings are undocumented.\n"
        "- **IQ-OTH/NCCD dataset:** image-level classification into its documented Normal, Benign, "
        "and Malignant folders.\n"
        "- **Combined user flow:** both models can analyze one uploaded image in the same page; "
        "their predictions remain independent.\n"
        "- **Separate training:** the first dataset has bounding-box annotations, while IQ-OTH/NCCD "
        "has whole-image category labels, so the labels cannot be merged honestly.\n"
        "- **Not supported:** pixel-level U-Net segmentation, 3D volume analysis, or cancer-risk grading."
    )
    st.warning(
        "This is a research prototype, not a diagnostic system. The nodule detector's validation "
        "set contains only 41 unique images. The IQ-OTH/NCCD image files do not expose patient IDs, "
        "so its reproducible train/validation/test split is image-level and may include slices from "
        "the same patient across splits. Treat both results as exploratory."
    )


else:
    st.header("Training guide")
    st.subheader("Nodule localization")
    st.write(
        "The dataset is already in YOLO detection layout. Training writes a filtered image-list "
        "manifest that excludes exact copies of validation images from the training split."
    )
    st.code(
        "python -m pip install -r requirements.txt\n"
        "python scripts/train_detector.py\n"
        "python scripts/evaluate_detector.py",
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
        st.caption("Measured validation metrics from the saved detector checkpoint.")
        with st.container(horizontal=True):
            for metric_name in ("Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"):
                if metric_name in recorded_metrics:
                    st.metric(metric_name, f"{recorded_metrics[metric_name]:.3f}")

    st.divider()
    st.subheader("IQ-OTH/NCCD image classification")
    st.write(
        "This task predicts one folder category for a whole CT slice. The script accepts the "
        "downloaded folder layout, including the source spelling `Bengin cases`."
    )
    st.code(
        "python scripts/train_iq_oth_classifier.py\n"
        "python scripts/evaluate_iq_oth_classifier.py",
        language="powershell",
    )
    st.write(
        "The script creates a deterministic stratified 70/15/15 image-level split, removes exact "
        "duplicate files, saves the best ResNet-18 checkpoint under "
        "`models/iq_oth_classifier/best.pt`, and writes held-out metrics to "
        "`models/iq_oth_classifier/metrics.json`. Patient IDs are not included in the image files, "
        "so the split cannot guarantee patient-level separation."
    )
    st.caption("Label indices used by this project: 0 = Normal, 1 = Benign, 2 = Malignant.")
