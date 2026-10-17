"""Patch-level bootstrap CIs + paired model comparisons.

Inputs: results/predictions/<model>_s<seed>_<region>.npy (+ _labels.npy).
Methods:
  - per (model, region): resample the test PATCHES with replacement
    (B=2000), pool pixels of the resampled patches, recompute
    OA / mIoU / F1_change / omission / commission -> mean + 95% CI.
  - per (model A, model B, region): common patch set, paired bootstrap
    (same resampled patch indices for both models) -> CI of the
    difference + two-sided p = 2*min(P(d<0), P(d>0)).
  - seed aggregation: the 3 seeds share the same test patches; the
    per-seed bootstrap is pooled across seeds (resample from
    3 x n_patches patch-replicas).

Output: results/bootstrap_ci.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "unet_msa", "deeplabv3plus",
          "temporal_unet", "temporal_unet_gn", "fc_siam"]
SEEDS = [2026, 2027, 2028]
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
B = 2000


def block_rng(*key):
    """Per-block deterministic RNG: each (model, region) CI block and each
    paired comparison gets its own seed derived from its key, so results are
    independent of model order / pair order (order-independent auditability)."""
    import zlib
    return np.random.default_rng(zlib.crc32("|".join(map(str, key)).encode()))


def metrics(y, p):
    cm = np.zeros((4, 4), dtype=np.int64)
    for i in range(4):
        for j in range(4):
            cm[i, j] = int(((y == i) & (p == j)).sum())
    oa = np.trace(cm) / cm.sum()
    ious = []
    for i in range(4):
        u = cm[i, :].sum() + cm[:, i].sum() - cm[i, i]
        ious.append(cm[i, i] / u if u else 0.0)
    mIoU = float(np.mean(ious))
    chg = [1, 2]
    tp = cm[np.ix_(chg, chg)].sum()
    fn = cm[np.ix_(chg, [0, 3])].sum()
    fp = cm[np.ix_([0, 3], chg)].sum()
    f1 = 2 * tp / max(2 * tp + fn + fp, 1)
    return {"OA": oa, "mIoU": mIoU, "F1ch": f1,
            "omission": fn / max(tp + fn, 1), "commission": fp / max(tp + fp, 1)}


def patch_cms(y, p):
    """Per-patch 4x4 confusion matrices. y, p: (n_patches, H, W) uint8 in 0..3."""
    n = y.shape[0]
    comb = (y.reshape(n, -1).astype(np.int64) * 4
            + p.reshape(n, -1).astype(np.int64))  # values 0..15
    cms = np.zeros((n, 16), dtype=np.int64)
    for k in range(n):
        cms[k] = np.bincount(comb[k], minlength=16)
    return cms.reshape(n, 4, 4)


def metrics_from_cm(cm):
    """Same metrics as metrics(), computed from a (4,4) confusion matrix.
    A patch-level bootstrap (resample patches, pool pixels) has a pooled CM
    that is exactly the sum of the resampled patches' CMs, so this is
    bit-identical to metrics(y[idx], p[idx]) but avoids re-pooling pixels."""
    oa = np.trace(cm) / cm.sum()
    ious = []
    for i in range(4):
        u = cm[i, :].sum() + cm[:, i].sum() - cm[i, i]
        ious.append(cm[i, i] / u if u else 0.0)
    mIoU = float(np.mean(ious))
    chg = [1, 2]
    tp = cm[np.ix_(chg, chg)].sum()
    fn = cm[np.ix_(chg, [0, 3])].sum()
    fp = cm[np.ix_([0, 3], chg)].sum()
    f1 = 2 * tp / max(2 * tp + fn + fp, 1)
    return {"OA": oa, "mIoU": mIoU, "F1ch": f1,
            "omission": fn / max(tp + fn, 1), "commission": fp / max(tp + fp, 1)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()

    data = {}
    for m in args.models:
        for r in REGIONS:
            ys, ps = [], []
            for s in SEEDS:
                pf = ROOT / "results" / "predictions" / f"{m}_s{s}_{r}.npy"
                lf = ROOT / "results" / "predictions" / f"{m}_s{s}_{r}_labels.npy"
                if not (pf.exists() and lf.exists()):
                    continue
                ps.append(np.load(pf)); ys.append(np.load(lf))
            if ps:
                data[(m, r)] = (np.concatenate(ys), np.concatenate(ps))

    out = {}
    cms = {(m, r): patch_cms(y, p) for (m, r), (y, p) in data.items()}
    for (m, r), (y, p) in data.items():
        n = y.shape[0]
        arr = cms[(m, r)]
        oas, mi, f1s, oms, coms = [], [], [], [], []
        rng = block_rng(m, r)
        for b in range(B):
            idx = rng.integers(0, n, n)
            mc = metrics_from_cm(arr[idx].sum(axis=0))
            oas.append(mc["OA"]); mi.append(mc["mIoU"]); f1s.append(mc["F1ch"])
            oms.append(mc["omission"]); coms.append(mc["commission"])
        def ci(a):
            return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        out[f"{m}|{r}"] = {
            "n_patches": int(n),
            "OA": [float(np.mean(oas)), ci(oas)],
            "mIoU": [float(np.mean(mi)), ci(mi)],
            "F1ch": [float(np.mean(f1s)), ci(f1s)],
            "omission": [float(np.mean(oms)), ci(oms)],
            "commission": [float(np.mean(coms)), ci(coms)],
        }

    # paired comparisons (proposed vs each baseline; also best-vs-best)
    pairs = []
    base = ["unet", "deeplabv3plus", "temporal_unet", "fc_siam", "temporal_unet_gn"]
    for a in base:
        pairs.append(("unet_cbam_msa", a))
    pairs.append(("temporal_unet", "unet_cbam_msa"))
    pairs.append(("fc_siam", "temporal_unet_gn"))
    pairs.append(("fc_siam", "temporal_unet"))
    comp = {}
    for (a, b) in pairs:
        for r in REGIONS:
            if (a, r) not in data or (b, r) not in data:
                continue
            ca, cb = cms[(a, r)], cms[(b, r)]
            n = ca.shape[0]
            d_f1, d_mi = [], []
            rng = block_rng(a, b, r)
            for bb in range(B):
                idx = rng.integers(0, n, n)
                ma = metrics_from_cm(ca[idx].sum(axis=0))
                mb = metrics_from_cm(cb[idx].sum(axis=0))
                d_f1.append(ma["F1ch"] - mb["F1ch"])
                d_mi.append(ma["mIoU"] - mb["mIoU"])
            d_f1, d_mi = np.array(d_f1), np.array(d_mi)
            comp[f"{a} vs {b} | {r}"] = {
                "dF1ch": [float(d_f1.mean()),
                          [float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))]],
                "p_F1ch": float(2 * min((d_f1 < 0).mean(), (d_f1 > 0).mean())),
                "dmIoU": [float(d_mi.mean()),
                          [float(np.percentile(d_mi, 2.5)), float(np.percentile(d_mi, 97.5))]],
                "p_mIoU": float(2 * min((d_mi < 0).mean(), (d_mi > 0).mean())),
            }
    out["_paired"] = comp
    (ROOT / "results" / "bootstrap_ci.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:3000])


if __name__ == "__main__":
    main()
