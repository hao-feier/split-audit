"""Mislocalization analysis: distance of missed / spurious change to the
model's own (or the reference's) change pixels.

For each (model, seed, region) prediction dump:
  * proximity of MISSED change: for every true-change pixel, distance to the
    nearest predicted-change pixel -> the fraction within 20/50/100 m is the
    share of the omission that is spatial mislocalisation (the model predicted
    change a few tens of metres away) rather than a global deficit of the
    change prior.
  * proximity of SPURIOUS change: for every predicted-change pixel, distance
    to the nearest true-change pixel -> boundary displacement vs change
    invented far from the reference.
  * background baseline: fraction of non-change pixels within 20 m of any
    predicted-change pixel (the zero-signal proximity rate).

Aggregated over the 5 main seeds per (model, region). Uses the existing
per-patch prediction dumps (results/predictions/), no retraining.

Output: results/mislocalization.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.paths import RESULTS_ROOT  # noqa: E402

MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]
SEEDS = [2026, 2027, 2028, 2029, 2030]
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
PIX_M = 10.0
THRESH_M = [20.0, 50.0, 100.0]


def analyse(pred, label):
    tc = np.isin(label, [1, 2])
    pc = np.isin(pred, [1, 2])
    n_tc, n_pc = int(tc.sum()), int(pc.sum())
    if n_tc == 0 or n_pc == 0:
        return None
    d2pc = distance_transform_edt(~pc) * PIX_M   # every pixel -> nearest pred change
    d2tc = distance_transform_edt(~tc) * PIX_M   # every pixel -> nearest true change
    out = {"n_true_change": n_tc, "n_pred_change": n_pc,
           "omission": float((tc & ~pc).sum() / max(n_tc, 1)),
           "commission": float((pc & ~tc).sum() / max(n_pc, 1))}
    miss = tc & ~pc
    spur = pc & ~tc
    for t in THRESH_M:
        if miss.sum():
            out[f"missed_within_{int(t)}m_frac"] = float((miss & (d2pc <= t)).sum() / miss.sum())
        if spur.sum():
            out[f"spurious_within_{int(t)}m_frac"] = float((spur & (d2tc <= t)).sum() / spur.sum())
        out[f"bg_within_{int(t)}m_frac"] = float((~tc & (d2pc <= t)).sum() / (~tc).sum())
    return out


def main():
    out = {"config": {"models": MODELS, "seeds": SEEDS, "regions": REGIONS,
                      "thresholds_m": THRESH_M,
                      "note": "missed_within_* = share of omitted true-change pixels within "
                              "t m of any predicted-change pixel; spurious_within_* = share of "
                              "false-change pixels within t m of any true-change pixel; "
                              "bg_within_* = zero-signal proximity of non-change pixels to "
                              "predicted change"},
           "per_model": {}}
    for m in MODELS:
        per_region = {}
        for r in REGIONS:
            runs = []
            for s in SEEDS:
                pf = RESULTS_ROOT / "predictions" / f"{m}_s{s}_{r}.npy"
                lf = RESULTS_ROOT / "predictions" / f"{m}_s{s}_{r}_labels.npy"
                if not (pf.exists() and lf.exists()):
                    continue
                preds, labels = np.load(pf), np.load(lf)
                for k in range(len(preds)):
                    a = analyse(preds[k], labels[k])
                    if a:
                        runs.append(a)
            if not runs:
                continue
            keys = set()
            for a in runs:
                keys.update(k for k in a if k not in ("n_true_change", "n_pred_change"))
            # seeds that predict no change (common under uniform weights) lack
            # the spurious_* keys; average over the seeds that have each key
            agg = {k: round(float(np.mean([a[k] for a in runs if k in a])), 3)
                   for k in sorted(keys)
                   if any(k in a for a in runs)}
            agg["n_runs"] = len(runs)
            per_region[r] = agg
        out["per_model"][m] = per_region
        print(m, {r: {k: v for k, v in d.items() if "missed" in k or "spurious" in k}
                  for r, d in per_region.items()}, flush=True)
    (RESULTS_ROOT / "mislocalization.json").write_text(json.dumps(out, indent=2))
    print("wrote results/mislocalization.json")


if __name__ == "__main__":
    main()
