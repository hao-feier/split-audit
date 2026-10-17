"""Exp3b: classical ML baselines (PLAN 4.3): Random Forest + SVM.

Pixel-level features: t1 stack + t2 stack + |t1 - t2| (27 features).
Training pixels are subsampled from the vietnam train zone; evaluation
is pixel-level on the independent test regions beilunhe and zhanjiang.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mangrove.config import SEED, CHANGE_CLASSES
from mangrove.datasets import load_region, load_split
from mangrove.metrics import cd_metrics
from mangrove.paths import RESULTS_ROOT

np.random.seed(SEED)

TRAIN_REGION = "vietnam"
TEST_REGIONS = ("beilunhe", "zhanjiang")


def pixel_features(region):
    d = load_region(region)
    x1, x2 = d["t1_feat"], d["t2_feat"]
    return np.concatenate([x1, x2, np.abs(x1 - x2)], axis=-1)  # (N, H, W, 27)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train-pixels", type=int, default=300_000)
    ap.add_argument("--max-test-pixels", type=int, default=200_000)
    ap.add_argument("--cw", choices=["none", "balanced"], default="none",
                    help="class weighting for the pixel baselines: "
                         "fair-control baseline with standard practice weighting)")
    args = ap.parse_args()
    cw = "balanced" if args.cw == "balanced" else None

    Xtr = pixel_features(TRAIN_REGION)
    ytr = load_region(TRAIN_REGION)["change"]
    tr_idx = np.asarray(load_split(TRAIN_REGION, "train"))
    Xtr, ytr = Xtr[tr_idx], ytr[tr_idx]
    Xtr = Xtr.reshape(-1, 27).astype(np.float32) / 10000.0
    ytr = ytr.reshape(-1)
    if len(Xtr) > args.max_train_pixels:
        sel = np.random.choice(len(Xtr), args.max_train_pixels, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]

    out = {}
    for name, clf in (
        ("random_forest", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED, class_weight=cw)),
        ("svm_rbf", SVC(kernel="rbf", probability=False, C=1.0, cache_size=2048, class_weight=cw)),
    ):
        clf.fit(Xtr, ytr)
        per_region = {}
        for region in TEST_REGIONS:
            Xte = pixel_features(region)
            yte = load_region(region)["change"]
            te_idx = np.asarray(load_split(region, "test"))
            Xte, yte = Xte[te_idx], yte[te_idx]
            Xte = Xte.reshape(-1, 27).astype(np.float32) / 10000.0
            yte = yte.reshape(-1)
            if len(Xte) > args.max_test_pixels:
                sel = np.random.choice(len(Xte), args.max_test_pixels, replace=False)
                Xte, yte = Xte[sel], yte[sel]
            p = clf.predict(Xte)
            per_region[region] = cd_metrics(yte, p, 4, CHANGE_CLASSES)
        out[name] = per_region
        print(name, {r: ({k: v[k] for k in ("OA", "kappa", "mIoU", "F1_macro")} if v else None)
                     for r, v in per_region.items()})

    fname = "metrics_ml_baselines.json" if args.cw == "none" else "metrics_ml_baselines_weighted.json"
    (RESULTS_ROOT / fname).write_text(json.dumps(out, indent=2))
    print(f"saved {fname}")


if __name__ == "__main__":
    main()
