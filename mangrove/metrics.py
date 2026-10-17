"""Full evaluation metric suite (docs/PLAN.md section 6).

All metrics computed per-class + overall. Confusion matrix is the
single source of truth; everything else is derived from it.
"""
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    jaccard_score as iou_score,  # sklearn >= 1.0 renamed iou_score -> jaccard_score
)


def _pad_cm(cm, n):
    return np.pad(cm, ((0, n - cm.shape[0]), (0, n - cm.shape[1])))


def compute_all_metrics(y_true, y_pred, n_classes=4, class_names=None):
    """Return dict with overall + per-class metrics and confusion matrix."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    cm = _pad_cm(confusion_matrix(y_true, y_pred, labels=range(n_classes)), n_classes)
    total = cm.sum()
    if class_names is None:
        class_names = [f"c{i}" for i in range(n_classes)]

    out = {
        "n_pixels": int(total),
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "OA": float(accuracy_score(y_true, y_pred, normalize=True)) if total else 0.0,
        "kappa": float(cohen_kappa_score(y_true, y_pred)) if total else 0.0,
        "mIoU": float(iou_score(y_true, y_pred, average="macro", labels=range(n_classes), zero_division=0)) if total else 0.0,
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", labels=range(n_classes), zero_division=0)) if total else 0.0,
        "per_class": {},
    }

    row_sums = cm.sum(axis=1)   # producer's denominator (actual)
    col_sums = cm.sum(axis=0)   # user's denominator (predicted)
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        pa = tp / row_sums[i] if row_sums[i] else 0.0   # producer's accuracy
        ua = tp / col_sums[i] if col_sums[i] else 0.0   # user's accuracy
        iou = tp / (row_sums[i] + col_sums[i] - tp) if (row_sums[i] + col_sums[i] - tp) else 0.0
        f1 = 2 * pa * ua / (pa + ua) if (pa + ua) else 0.0
        out["per_class"][name] = {
            "support": int(row_sums[i]),
            "PA": float(pa), "UA": float(ua), "IoU": float(iou), "F1": float(f1),
        }
    return out


def cd_metrics(y_true, y_pred, n_classes=4, class_names=None):
    """Change-detection extras: CD accuracy, omission, commission (PLAN 6).

    Omission/commission are computed against the 'changed' super-class
    (loss + gain) so they are scale-free to class balance.
    """
    base = compute_all_metrics(y_true, y_pred, n_classes, class_names)
    names = base["class_names"]
    if "loss" in names and "gain" in names:
        change_true = (y_true == names.index("loss")) | (y_true == names.index("gain"))
        change_pred = (y_pred == names.index("loss")) | (y_pred == names.index("gain"))
        n_changed = int(change_true.sum())
        base["CD_Accuracy_changed"] = float(
            np.mean(change_pred == change_true)) if n_changed else 0.0
        base["Omission_Error"] = float(
            np.mean(~change_pred[change_true])) if n_changed else 0.0
        base["Commission_Error"] = float(
            np.mean(change_pred[~change_true])) if int((~change_true).sum()) else 0.0
    return base
