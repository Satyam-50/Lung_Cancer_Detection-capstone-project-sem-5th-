import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

def compute_segmentation_metrics(y_true, y_pred, threshold=0.5):
    """
    Computes standard segmentation metrics: Dice Score (F1) and IoU (Jaccard).
    y_true: Binary ground truth array (H, W) or (N, H, W)
    y_pred: Predicted probability array (H, W) or (N, H, W)
    """
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    y_true_f = y_true.astype(np.float32).flatten()
    y_pred_f = y_pred_bin.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    iou = (intersection + 1e-7) / (union - intersection + 1e-7)
    
    return {
        "Dice_Coefficient": float(dice),
        "IoU_Score": float(iou)
    }

def compute_clinical_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Computes clinical classification metrics:
    - Sensitivity / Recall (True Positive Rate)
    - Specificity (True Negative Rate)
    - Precision / Positive Predictive Value (PPV)
    - Negative Predictive Value (NPV)
    - F1-Score
    - ROC-AUC
    """
    y_pred_bin = (np.array(y_pred_probs) >= threshold).astype(int)
    y_true = np.array(y_true).astype(int)
    
    cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except Exception:
        auc = 0.5
        
    return {
        "Accuracy": float(acc),
        "Sensitivity_Recall": float(sensitivity),
        "Specificity": float(specificity),
        "Precision_PPV": float(precision),
        "NPV": float(npv),
        "F1_Score": float(f1),
        "ROC_AUC": float(auc),
        "Confusion_Matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }
