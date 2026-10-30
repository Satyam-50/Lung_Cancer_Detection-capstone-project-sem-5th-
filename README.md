# PulmoAI: 2D lung nodule localization

PulmoAI is a research prototype for finding annotated lung nodules in 2D CT images. The active workflow is aligned to the required **Lung Nodules Detection Dataset Annotations** dataset and its bounding-box labels.

## Research context

The reference is Nguyen's 2025 scoping review, [“A scoping review of deep learning approaches for lung cancer detection using chest radiographs and computed tomography scans”](https://doi.org/10.1016/j.bea.2024.100138). It surveys published approaches; it does not prescribe a single model to reproduce. This project applies the review's CT detection context to the labels actually available here: 2D object localization with a lightweight pretrained YOLO detector.

The model choice is based on the dataset format, not a claim that the review selected YOLO26. The detector reports bounding boxes, numeric class IDs, and confidence scores. It does not diagnose cancer.

## Dataset

Source: [Lung Nodules Detection Dataset Annotations on Kaggle](https://www.kaggle.com/datasets/younesselbrag/lung-nodules-detection-dataset-annotations).

Expected local structure:

```text
Lung_Cancer_Detection/
└── Lung Nodules Detection Dataset Annotations/
    └── ct_images/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
```

The label rows use normalized YOLO box format:

```text
class_id x_center y_center width height
```

The current local copy has 239 training images and 41 validation images. Its label files use class IDs `0` and `1`, but the downloaded folder does not include class names. Until the dataset owner confirms their meanings, the project keeps them as generic `class_0` and `class_1`; it does not call them benign or malignant.

### Dataset integrity notes

- All 41 validation images are exact image-content duplicates of files in the training folder, and their paired annotation text matches. `scripts/train_detector.py` excludes the train-side duplicates when it generates training manifests. The downloaded data is not rewritten.
- There are 10 extra training label files without matching images. The app reports these; training ignores them because no image can be paired with them.
- The validation set is small and the dataset has no separate test or external validation cohort. Treat resulting metrics as exploratory only.

## Install and run

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

If the dataset is stored elsewhere, set its root to the directory containing `images` and `labels`:

```powershell
$env:PULMOAI_DATASET_ROOT = "D:\path\to\ct_images"
streamlit run app.py
```

The **Dataset explorer** page previews the reference boxes and audits the splits. The **Detector** page displays dataset labels as ground truth and runs predictions only after a trained `.pt` checkpoint exists.

## Train the detector

```powershell
python scripts/train_detector.py
```

The script starts from the lightweight pretrained `yolo26n.pt` model, trains at the dataset's 256-pixel image size, and evaluates on the deduplicated validation list. It saves its generated manifests, dataset configuration, weights, and validation outputs under `models/nodule_detector/`. The Streamlit app automatically loads the newest `train*/weights/best.pt` checkpoint.

Optional arguments:

```powershell
python scripts/train_detector.py --epochs 60 --imgsz 256 --batch 8 --device cpu
```

The training script can also receive another dataset root with `--dataset PATH` or the `PULMOAI_DATASET_ROOT` environment variable.

## Scope and limitations

This dataset supports **2D nodule/finding localization**. It does not provide 3D CT volumes, pixel-level nodule masks, or malignancy/outcome labels. Therefore the active project does not train or report:

- U-Net segmentation or Dice scores;
- benign-versus-malignant classification or cancer-risk probabilities;
- 3D analysis, volumetry, or patient-level diagnosis;
- synthetic Grad-CAM explanations or fabricated benchmark metrics.

The earlier LUNA16 files in `DATA/annotations.csv`, the `.h5` weights under `models/v1` and `models/v2`, and the older U-Net/classifier notebooks are retained as separate legacy experiments. They are not loaded by this dataset-specific app workflow.

## Repository map

```text
app.py                    Streamlit detector and dataset explorer
src/data/yolo_dataset.py  YOLO label parsing, visualization, and integrity audit
scripts/train_detector.py Dataset-aware training and deduplicated split generation
DATA/annotations.csv      Legacy LUNA16 3D-coordinate annotations
```
