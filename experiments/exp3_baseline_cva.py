"""Exp3c: CVA (Change Vector Analysis) and index-difference baselines.

Three variants:
1. CVA-NDVI: threshold NDVI difference (t2 - t1) at 0
2. CVA-Norm: threshold Euclidean norm of 6-band reflectance difference at 95th percentile
3. CVA-LogReg: logistic regression on 18-dim difference vector (6 bands x 2 dates)

All trained on Vietnam train zone, evaluated on Vietnam test + Beilunhe + Zhanjiang.
Output: results/metrics_cva_baselines.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mangrove.config import SEED, CHANGE_CLASSES
from mangrove.datasets import load_region, load_split
from mangrove.paths import RESULTS_ROOT

# Feature band indices in the 9-band stack
B02, B03, B04, B08, B11, B12, NDVI, NDWI, MNDWI = range(9)
REFLECT_BNDS = [B02, B03, B04, B08, B11, B12]


def flatten_patch(arr, patch_idx):
    """Extract a single patch and flatten to (H*W, C) or (H*W,) for labels."""
    return arr[patch_idx]


def evaluate(y_true, y_pred, n_classes=4):
    """Compute change-detection metrics (binary change: loss+gain vs stable)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    # Binary change: 1=change (loss|gain), 0=no-change (stable_mng|stable_bg)
    yb_true = ((y_true == 1) | (y_true == 2)).astype(int)
    yb_pred = ((y_pred == 1) | (y_pred == 2)).astype(int)
    cm = confusion_matrix(yb_true, yb_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    oa = (tn + tp) / total if total else 0.0
    kappa = (tp + tn - (tp + fn) * (tp + fp) / total) / (total - (tp + fn) * (tp + fp) / total) if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {
        "OA": float(oa), "kappa": float(kappa),
        "F1_change": float(f1), "IoU_change": float(iou),
        "precision": float(prec), "recall": float(rec),
        "n_pixels": int(total),
        "cm_binary": cm.tolist(),
    }


def main():
    t0 = time.time()
    np.random.seed(SEED)

    # Load data
    vn = load_region("vietnam")
    x1, x2 = vn["t1_feat"], vn["t2_feat"]  # (76, 512, 512, 9)
    chg = vn["change"]                      # (76, 512, 512)
    tr_idx = np.asarray(load_split("vietnam", "train"))
    te_idx = np.asarray(load_split("vietnam", "test"))

    # Binary change label: 1 = loss|gain, 0 = stable
    def bin_change(arr):
        return ((arr == 1) | (arr == 2)).astype(np.int8)

    # Training pixels
    X1_tr = x1[tr_idx].reshape(-1, 9).astype(np.float32) / 10000.0
    X2_tr = x2[tr_idx].reshape(-1, 9).astype(np.float32) / 10000.0
    y_tr = bin_change(chg[tr_idx]).reshape(-1)

    # Subsample for speed
    max_px = 500_000
    if len(X1_tr) > max_px:
        sel = np.random.choice(len(X1_tr), max_px, replace=False)
        X1_tr, X2_tr, y_tr = X1_tr[sel], X2_tr[sel], y_tr[sel]

    print(f"Training pixels: {len(X1_tr)}, change fraction: {y_tr.mean():.4f}")

    results = {}

    # === CVA-NDVI: threshold NDVI difference at 0 ===
    # NDVI = (NIR - Red) / (NIR + Red) = band 6 in the feature stack
    ndvi_t1 = X1_tr[:, NDVI]
    ndvi_t2 = X2_tr[:, NDVI]
    # For evaluation: positive NDVI change (t2 > t1) -> gain; negative -> loss
    # We classify as "change" if |NDVI_t2 - NDVI_t1| > threshold
    # Threshold = 0 (any change) is the standard CVA-NDVI
    threshold_ndvi = 0.0

    def cva_ndvi_eval(x1_e, x2_e):
        ndvi1 = x1_e[:, NDVI]
        ndvi2 = x2_e[:, NDVI]
        diff = ndvi2 - ndvi1
        # Binary change prediction: 1 if |diff| > threshold
        pred = (np.abs(diff) > threshold_ndvi).astype(np.int8)
        # Map to 4-class: 1=loss, 2=gain, 0=stable_mng, 3=stable_bg
        # For CVA we can only predict change/no-change, not direction
        # Assign all change to class 1 (loss) for metric computation
        return pred

    # === CVA-Norm: threshold Euclidean norm of reflectance difference ===
    # Compute the 95th percentile of the norm in the training set
    diff_tr = (X2_tr[:, REFLECT_BNDS] - X1_tr[:, REFLECT_BNDS])
    norm_tr = np.sqrt((diff_tr ** 2).sum(axis=1))
    threshold_norm = np.percentile(norm_tr, 95)
    print(f"CVA-Norm threshold (95th pct): {threshold_norm:.6f}")

    def cva_norm_eval(x1_e, x2_e):
        diff_e = (x2_e[:, REFLECT_BNDS] - x1_e[:, REFLECT_BNDS])
        norm_e = np.sqrt((diff_e ** 2).sum(axis=1))
        return (norm_e > threshold_norm).astype(np.int8)

    # === CVA-LogReg: logistic regression on difference features ===
    # Features: 6 reflectance bands x difference + NDVI diff + NDWI diff + MNDWI diff = 9 features
    diff_features_tr = (X2_tr - X1_tr)  # (N, 9)
    scaler = StandardScaler()
    diff_scaled = scaler.fit_transform(diff_features_tr)
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    lr.fit(diff_scaled, y_tr)
    print(f"CVA-LogReg trained: {len(diff_scaled)} pixels, coef={lr.coef_[0].round(4)}")

    def cva_logreg_eval(x1_e, x2_e):
        diff_e = (x2_e - x1_e)
        diff_e_scaled = scaler.transform(diff_e)
        return lr.predict(diff_e_scaled)

    # === Evaluate on all regions ===
    regions = {
        "vietnam_test": {
            "x1": x1[te_idx], "x2": x2[te_idx], "chg": chg[te_idx],
        },
    }
    for region in ("beilunhe", "zhanjiang"):
        d = load_region(region)
        regions[region] = {
            "x1": d["t1_feat"].reshape(-1, 9).astype(np.float32) / 10000.0,
            "x2": d["t2_feat"].reshape(-1, 9).astype(np.float32) / 10000.0,
            "chg": d["change"].reshape(-1),
        }

    for rname, rdata in regions.items():
        x1_e = rdata["x1"]
        x2_e = rdata["x2"]
        y_e = rdata["chg"]
        if x1_e.ndim == 4:
            x1_e = x1_e.reshape(-1, 9).astype(np.float32) / 10000.0
            x2_e = x2_e.reshape(-1, 9).astype(np.float32) / 10000.0
            y_e = y_e.reshape(-1)

        # Subsample for speed
        max_test = 200_000
        if len(x1_e) > max_test:
            sel = np.random.choice(len(x1_e), max_test, replace=False)
            x1_e, x2_e, y_e = x1_e[sel], x2_e[sel], y_e[sel]

        print(f"\n--- {rname} ({len(x1_e)} pixels) ---")

        for variant, fn in [
            ("cva_ndvi", cva_ndvi_eval),
            ("cva_norm", cva_norm_eval),
            ("cva_logreg", cva_logreg_eval),
        ]:
            pred = fn(x1_e, x2_e)
            # Map binary pred to 4-class for metric computation
            # 1 -> class 1 (loss), 0 -> class 3 (stable_bg)
            pred_4class = np.where(pred == 1, 1, 3).astype(np.int8)
            m = evaluate(y_e, pred_4class)
            m["n_pixels_eval"] = int(len(y_e))
            results.setdefault(rname, {})[variant] = m
            print(f"  {variant}: F1={m['F1_change']:.4f} IoU={m['IoU_change']:.4f} OA={m['OA']:.4f}")

    results["_meta"] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": round(time.time() - t0, 1),
        "cv_norm_threshold": float(threshold_norm),
        "lr_coef": lr.coef_[0].tolist(),
        "train_pixels": int(len(X1_tr)),
        "train_change_frac": float(y_tr.mean()),
    }

    out = RESULTS_ROOT / "metrics_cva_baselines.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
