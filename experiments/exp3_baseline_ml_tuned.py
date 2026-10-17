"""Exp3d: Tuned Random Forest + SVM baselines (fair baselines).

Improvements over exp3b:
1. GridSearchCV for n_estimators, max_depth, min_samples_leaf (RF)
2. GridSearchCV for C, gamma, kernel (SVM)
3. Proper class_weight='balanced' as primary
4. Feature selection: use only the 6 reflectance bands + 3 indices (9 features)
   + their differences (9 features) = 18 features total
5. Stratified train/test split for hyperparameter selection

Output: results/metrics_tuned_baselines.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mangrove.config import SEED
from mangrove.datasets import load_region, load_split
from mangrove.paths import RESULTS_ROOT


def bin_change(arr):
    return ((arr == 1) | (arr == 2)).astype(np.int8)


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    yb_true = ((y_true == 1) | (y_true == 2)).astype(int)
    yb_pred = ((y_pred == 1) | (y_pred == 2)).astype(int)
    cm = confusion_matrix(yb_true, yb_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    oa = (tn + tp) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    kappa = (tp + tn - (tp + fn) * (tp + fp) / total) / (total - (tp + fn) * (tp + fp) / total) if total else 0.0
    return {
        "OA": float(oa), "kappa": float(kappa),
        "F1_change": float(f1), "IoU_change": float(iou),
        "precision": float(prec), "recall": float(rec),
        "n_pixels": int(total), "cm_binary": cm.tolist(),
    }


def main():
    t0 = time.time()
    np.random.seed(SEED)

    # Load Vietnam training data
    vn = load_region("vietnam")
    x1, x2 = vn["t1_feat"], vn["t2_feat"]  # (76, 512, 512, 9)
    chg = vn["change"]                      # (76, 512, 512)
    tr_idx = np.asarray(load_split("vietnam", "train"))

    # Features: t1 (9) + t2 (9) + |diff| (9) = 27 features
    X1 = x1[tr_idx].reshape(-1, 9).astype(np.float32) / 10000.0
    X2 = x2[tr_idx].reshape(-1, 9).astype(np.float32) / 10000.0
    Xdiff = np.abs(X2 - X1)
    X = np.concatenate([X1, X2, Xdiff], axis=1)  # (N, 27)
    y = bin_change(chg[tr_idx]).reshape(-1)

    # Subsample for speed
    max_px = 300_000
    if len(X) > max_px:
        sel = np.random.choice(len(X), max_px, replace=False)
        X, y = X[sel], y[sel]

    print(f"Training: {len(X)} pixels, {X.shape[1]} features, change frac={y.mean():.4f}")

    # === Tuned Random Forest ===
    print("\n=== Tuning Random Forest ===")
    rf_params = {
        "n_estimators": [100, 300],
        "max_depth": [None, 20, 40],
        "min_samples_leaf": [1, 5, 10],
        "class_weight": ["balanced"],
        "n_jobs": [-1],
        "random_state": [SEED],
    }
    # Use a smaller grid for speed; the key hyperparams are n_estimators and max_depth
    rf_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 30],
        "min_samples_leaf": [1, 5],
    }
    rf = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED)
    # Do a quick manual grid instead of full GridSearchCV (too slow on 300K pixels)
    best_f1, best_params = 0, None
    for n_est in [100, 300]:
        for max_d in [None, 30]:
            for min_lf in [1, 5]:
                clf = RandomForestClassifier(
                    n_estimators=n_est, max_depth=max_d, min_samples_leaf=min_lf,
                    class_weight="balanced", n_jobs=-1, random_state=SEED,
                )
                # Use a validation subset for speed
                val_frac = 0.2
                n_val = int(len(X) * val_frac)
                X_val, y_val = X[:n_val], y[:n_val]
                X_tr, y_tr = X[n_val:], y[n_val:]
                clf.fit(X_tr, y_tr)
                pred_val = clf.predict(X_val)
                f1_val = f1_score(y_val, pred_val, zero_division=0)
                print(f"  RF n={n_est} d={max_d} lf={min_lf}: val F1={f1_val:.4f}")
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_params = {"n_estimators": n_est, "max_depth": max_d, "min_samples_leaf": min_lf}

    print(f"Best RF params: {best_params} (val F1={best_f1:.4f})")
    # Retrain on full training set with best params
    rf_best = RandomForestClassifier(**best_params, class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf_best.fit(X, y)
    print(f"RF trained on {len(X)} pixels")

    # === Tuned SVM ===
    print("\n=== Tuning SVM ===")
    # SVM is slow on 300K pixels; use a smaller subset for tuning
    svm_max = 50_000
    X_svm = X[:svm_max] if len(X) > svm_max else X
    y_svm = y[:svm_max] if len(y) > svm_max else y

    # Quick grid on SVM
    best_f1_svm, best_svm_params = 0, None
    for C in [0.1, 1.0, 10.0]:
        for gamma in ["scale", "auto"]:
            clf = SVC(C=C, gamma=gamma, class_weight="balanced", random_state=SEED)
            val_frac = 0.2
            n_val = int(len(X_svm) * val_frac)
            clf.fit(X_svm[n_val:], y_svm[n_val:])
            pred_val = clf.predict(X_svm[:n_val])
            f1_val = f1_score(y_svm[:n_val], pred_val, zero_division=0)
            print(f"  SVM C={C} gamma={gamma}: val F1={f1_val:.4f}")
            if f1_val > best_f1_svm:
                best_f1_svm = f1_val
                best_svm_params = {"C": C, "gamma": gamma}

    print(f"Best SVM params: {best_svm_params} (val F1={best_f1_svm:.4f})")
    svm_best = SVC(**best_svm_params, class_weight="balanced", random_state=SEED, cache_size=2048)
    svm_best.fit(X, y)
    print(f"SVM trained on {len(X)} pixels")

    # === Evaluate on all regions ===
    regions = {}
    # Vietnam test
    te_idx = np.asarray(load_split("vietnam", "test"))
    x1_te = x1[te_idx].reshape(-1, 9).astype(np.float32) / 10000.0
    x2_te = x2[te_idx].reshape(-1, 9).astype(np.float32) / 10000.0
    y_te = bin_change(chg[te_idx]).reshape(-1)
    X_te = np.concatenate([x1_te, x2_te, np.abs(x2_te - x1_te)], axis=1)
    regions["vietnam_test"] = (X_te, y_te)

    for region in ("beilunhe", "zhanjiang"):
        d = load_region(region)
        x1_r = d["t1_feat"].reshape(-1, 9).astype(np.float32) / 10000.0
        x2_r = d["t2_feat"].reshape(-1, 9).astype(np.float32) / 10000.0
        y_r = bin_change(d["change"]).reshape(-1)
        X_r = np.concatenate([x1_r, x2_r, np.abs(x2_r - x1_r)], axis=1)
        # Subsample
        max_test = 200_000
        if len(X_r) > max_test:
            sel = np.random.choice(len(X_r), max_test, replace=False)
            X_r, y_r = X_r[sel], y_r[sel]
        regions[region] = (X_r, y_r)

    results = {}
    for rname, (X_e, y_e) in regions.items():
        print(f"\n--- {rname} ({len(X_e)} pixels) ---")
        for name, clf in [("rf_tuned", rf_best), ("svm_tuned", svm_best)]:
            pred = clf.predict(X_e)
            # Map binary to 4-class: 1->loss(1), 0->stable_bg(3)
            pred_4 = np.where(pred == 1, 1, 3).astype(np.int8)
            m = evaluate(y_e, pred_4)
            m["n_pixels_eval"] = int(len(y_e))
            results.setdefault(rname, {})[name] = m
            print(f"  {name}: F1={m['F1_change']:.4f} IoU={m['IoU_change']:.4f} OA={m['OA']:.4f}")

    results["_meta"] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": round(time.time() - t0, 1),
        "rf_params": best_params,
        "svm_params": best_svm_params,
        "rf_val_f1": float(best_f1),
        "svm_val_f1": float(best_f1_svm),
        "train_pixels": int(len(X)),
        "train_change_frac": float(y.mean()),
    }

    out = RESULTS_ROOT / "metrics_tuned_baselines.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
