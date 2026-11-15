# PulmoAI: lung CT analysis research prototype

PulmoAI provides one upload workflow that runs two trained models side by side on a 2D CT slice: YOLO nodule localization and IQ-OTH/NCCD image-level classification. The classifier processes the full slice; it does not classify each detected box. The models train on separate datasets because one dataset has bounding boxes and undocumented class IDs, while the other has whole-image Normal/Benign/Malignant labels. Their labels are not combined.

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

## IQ-OTH/NCCD classification dataset

Source: [The IQ-OTH/NCCD lung cancer dataset on Mendeley Data](https://data.mendeley.com/datasets/bhmdr45bh2/4). The source describes three image-level classes: `Normal`, `Benign`, and `Malignant`. The downloaded Kaggle folder spells the benign directory `Bengin cases`; the app recognizes both spellings.

The classifier discovers these class directories under `The IQ-OTHNCCD lung cancer dataset/`, including the nested folder layout produced by the download. You can also set `PULMOAI_CLASSIFICATION_DATASET_ROOT` to a parent directory containing them. The app maps the labels as `0 = Normal`, `1 = Benign`, and `2 = Malignant` for training.

The source describes 1,190 CT slices from 110 cases, but the local folder export can contain a different image count. The training script reads and reports the actual files found. The JPG files do not preserve patient IDs, so the reproducible stratified train/validation/test split is image-level and cannot guarantee that slices from the same patient stay in one split. Treat its metrics as exploratory.

## Install and run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If the dataset is stored elsewhere, set its root to the directory containing `images` and `labels`:

```powershell
$env:PULMOAI_DATASET_ROOT = "D:\path\to\ct_images"
streamlit run app.py
```

If IQ-OTH/NCCD is stored elsewhere, point the classifier at the directory containing its class folders (or a parent directory containing them):

```powershell
$env:PULMOAI_CLASSIFICATION_DATASET_ROOT = "D:\path\to\iq_oth_dataset"
streamlit run app.py
```

The **Analyze CT** page accepts common image formats and single DICOM files, then runs both available checkpoints in one action. It shows the predicted candidate boxes, the IQ-OTH/NCCD dataset category and model scores, and can download an annotated PNG and JSON report. The **Model performance** page shows classifier accuracy, precision, recall, specificity, F1, ROC-AUC and confusion matrix, along with detector precision, recall, mAP and its confusion matrix. The original **Detector**, **IQ-OTH/NCCD classifier**, **Dataset explorer**, **Research scope**, and **Training guide** pages remain available.

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

Re-evaluate the latest saved detector checkpoint and save its validation report and confusion matrix:

```powershell
python scripts/evaluate_detector.py
```

## Train the IQ-OTH/NCCD classifier

```powershell
python scripts/train_iq_oth_classifier.py
```

By default, this fine-tunes an ImageNet-pretrained ResNet-18 and creates a deterministic stratified 70/15/15 image-level train/validation/test split. The script removes exact duplicate images, selects the checkpoint using validation macro F1, then reports held-out test metrics. Outputs go to `models/iq_oth_classifier/`; the app automatically loads `best.pt` from there.

The training report includes per-class precision, recall, specificity, F1, macro ROC-AUC and a confusion matrix. To recompute those metrics from the saved checkpoint and existing split without retraining:

```powershell
python scripts/evaluate_iq_oth_classifier.py
```

Optional arguments:

```powershell
python scripts/train_iq_oth_classifier.py --data "D:\path\to\iq_oth_dataset" --epochs 20 --batch-size 16 --device cpu
```

If pretrained weights cannot be downloaded, add `--no-pretrained` to train from random initialization. For a new dataset path, `--data PATH` takes priority over `PULMOAI_CLASSIFICATION_DATASET_ROOT`.

## Scope and limitations

The bounding-box dataset supports **2D nodule/finding localization** only; its numeric classes have no documented medical meaning. IQ-OTH/NCCD supplies image-level `Normal`, `Benign`, and `Malignant` folder labels, but the source files do not provide patient IDs for leakage-safe patient-level splitting. Neither workflow supports 3D analysis or makes a clinical diagnosis. The project does not claim:

- U-Net segmentation or Dice scores;
- cancer-risk probabilities from classifier output scores;
- 3D analysis, volumetry, or patient-level diagnosis;
- synthetic Grad-CAM explanations or fabricated benchmark metrics.

Uploaded files are processed by the local app process. DICOM files use their stored modality and display-window metadata where available; multiframe files are reduced to the middle frame because both models accept 2D images.

The earlier LUNA16 files in `DATA/annotations.csv`, the `.h5` weights under `models/v1` and `models/v2`, and the older U-Net/classifier notebooks are retained as separate legacy experiments. They are not loaded by this dataset-specific app workflow.

## Repository map

```text
app.py                    Streamlit combined inference, metrics, and project pages
src/data/yolo_dataset.py  YOLO label parsing, visualization, and integrity audit
src/data/iq_oth_dataset.py IQ-OTH/NCCD class-folder discovery
src/utils/image_loader.py JPG/PNG/TIFF and DICOM slice loading
src/models/iq_oth_classifier.py ResNet-18 model and inference loading
scripts/train_detector.py Dataset-aware training and deduplicated split generation
scripts/evaluate_detector.py Best-checkpoint detector validation report
scripts/train_iq_oth_classifier.py IQ-OTH/NCCD training and held-out evaluation
scripts/evaluate_iq_oth_classifier.py Saved-checkpoint classification metrics
DATA/annotations.csv      Legacy LUNA16 3D-coordinate annotations
```
