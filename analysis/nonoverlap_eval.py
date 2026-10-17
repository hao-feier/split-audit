"""Evaluation on the NON-OVERLAPPING patch subset.

Test patches are 512-px windows on a 256-px stride (50% overlap). The
pixel-pooled regional metrics already de-duplicate pixels? No - patches are
evaluated as the union of their (overlapping) pixel supports, so overlapped
pixels are counted twice in the pooled CM. To bound the effect, we recompute
every model's metrics on a subset of test patches with no pixel overlap
(maximum independent set on the overlap graph) and report the shift.

Output: results/nonoverlap_eval.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.config import CHANGE_CLASSES

MODELS = ["unet", "unet_cbam", "unet_msa", "unet_cbam_msa", "deeplabv3plus",
          "temporal_unet", "temporal_unet_gn", "fc_siam"]
SEEDS = [2026, 2027, 2028]
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
PATCH = 512


def independent_set(rows, cols):
    """Greedy maximum independent set on the overlap graph (grid sorted)."""
    order = sorted(range(len(rows)), key=lambda i: (rows[i], cols[i]))
    chosen, blocked = [], set()
    for i in order:
        if i in blocked:
            continue
        chosen.append(i)
        for j in order:
            if j == i:
                continue
            if abs(rows[i] - rows[j]) < PATCH and abs(cols[i] - cols[j]) < PATCH:
                blocked.add(j)
    return chosen


def cm4(y, p):
    cm = np.zeros((4, 4), dtype=np.int64)
    comb = y.ravel().astype(np.int64) * 4 + p.ravel().astype(np.int64)
    return np.bincount(comb, minlength=16).reshape(4, 4)


def f1ch_miou(cm):
    ious = [cm[i, i] / (cm[i].sum() + cm[:, i].sum() - cm[i, i])
            for i in range(4) if (cm[i].sum() + cm[:, i].sum() - cm[i, i]) > 0]
    mi = float(np.mean(ious)) if ious else 0.0
    chg = [1, 2]
    tp = cm[np.ix_(chg, chg)].sum()
    fn = cm[np.ix_(chg, [0, 3])].sum()
    fp = cm[np.ix_([0, 3], chg)].sum()
    f1 = 2 * tp / max(2 * tp + fn + fp, 1)
    return float(f1), mi


def main():
    out = {}
    for region in REGIONS:
        meta = json.loads((ROOT / "data" / "processed" / region /
                           "metadata.json").read_text())
        patches = np.asarray(meta["patches"], dtype=int)
        idx = np.asarray(json.loads((ROOT / "data" / "splits" /
                                     f"{region}_test.json").read_text()), dtype=int)
        rows, cols = patches[idx, 0], patches[idx, 1]
        indep = independent_set(rows, cols)
        labels = np.load(ROOT / "data" / "processed" / region /
                         "change_labels.npz")["change"]
        per_model = {}
        for m in MODELS:
            cm_full, cm_sub = np.zeros((4, 4)), np.zeros((4, 4))
            ok = True
            for s in SEEDS:
                pf = ROOT / "results" / "predictions" / f"{m}_s{s}_{region}.npy"
                if not pf.exists():
                    ok = False
                    break
                pred = np.load(pf)
                for k in range(pred.shape[0]):
                    i = int(idx[k])
                    cm_full += cm4(labels[i].ravel(), pred[k].ravel())
                    if k in indep:
                        cm_sub += cm4(labels[i].ravel(), pred[k].ravel())
            if not ok:
                continue
            per_model[m] = {
                "n_patches_full": int(len(idx)),
                "n_patches_nonoverlap": int(len(indep)),
                "full": {"F1ch": f1ch_miou(cm_full)[0], "mIoU": f1ch_miou(cm_full)[1]},
                "nonoverlap": {"F1ch": f1ch_miou(cm_sub)[0], "mIoU": f1ch_miou(cm_sub)[1]},
            }
        out[region] = per_model
        print(f"== {region}: {len(idx)} test patches -> {len(indep)} non-overlapping")
        for m, v in per_model.items():
            print(f"  {m:16s} F1ch {v['full']['F1ch']:.3f} -> {v['nonoverlap']['F1ch']:.3f} "
                  f"mIoU {v['full']['mIoU']:.3f} -> {v['nonoverlap']['mIoU']:.3f}")
    (ROOT / "results" / "nonoverlap_eval.json").write_text(json.dumps(out, indent=1))
    print("saved results/nonoverlap_eval.json")


if __name__ == "__main__":
    main()
