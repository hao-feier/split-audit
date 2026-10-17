"""Aggregate E3-C: does residual proximity matter when overlap is zero?

The manuscript cannot currently answer this. E3-B had to run the overlap-free
control as a block-interleaved design at 5-6 test patches and reports the sign
but not the magnitude. E3-C runs it size-matched at 128 px with 40/10/10, where
both arms have zero train/test pixel overlap by construction and differ only in
where the evaluation patches sit.

Two contrasts are reported:

  within-model  : the same trained model scored on its scattered test set minus
                  the same model scored on the separated test set. Training is
                  held exactly fixed; only the test arrangement varies.
  between-model : separated-arm test F1 minus scattered-arm test F1, matched on
                  architecture, draw and seed. Positive means the separated
                  (far) arm scores higher; a proximity-driven inflation would
                  instead appear NEGATIVE here, since the scattered arm is the
                  one adjacent to the training zone.

Writes results/e3c_summary.json.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "e3c")
SPL = os.path.join(ROOT, "data", "splits", "e3c_p128.json")
OUT = os.path.join(ROOT, "results", "e3c_summary.json")


def boot_ci(x, n=10000, seed=20261004):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return (round(float(np.percentile(b, 2.5)), 2),
            round(float(np.percentile(b, 97.5)), 2))


def main():
    recs = [json.loads(open(f).read())
            for f in sorted(glob.glob(os.path.join(RES, "metrics_*.json")))]
    if not recs:
        print("no E3-C results yet")
        return
    meta = json.loads(open(SPL).read())

    key = defaultdict(dict)
    for r in recs:
        key[(r["model"], r["draw"], r["seed"])][r["arm"]] = r

    paired_w, paired_b, cells = [], [], []
    for (m, d, s), v in sorted(key.items()):
        if "scattered" not in v or "separated" not in v:
            continue
        sc, sp = v["scattered"], v["separated"]
        # signed as separated - scattered; a proximity effect would be negative
        paired_w.append(100 * (sp["test"]["F1_change"] - sc["test"]["F1_change"]))
        paired_b.append(100 * (sp["test"]["F1_change"] - sc["test"]["F1_change"]))
        cells.append({"model": m, "draw": d, "seed": s,
                      "scattered_F1": sc["test"]["F1_change"],
                      "separated_F1": sp["test"]["F1_change"],
                      "gap_pp": round(100 * (sp["test"]["F1_change"]
                                             - sc["test"]["F1_change"]), 2),
                      "scattered_train_F1": sc.get("train_F1_change"),
                      "separated_train_F1": sp.get("train_F1_change")})

    by_arm = {}
    for arm in ("scattered", "separated"):
        f1 = [v[arm]["test"]["F1_change"] for v in key.values() if arm in v]
        tr = [v[arm]["train_F1_change"] for v in key.values()
              if arm in v and v[arm].get("train_F1_change") is not None]
        by_arm[arm] = {
            "n": len(f1),
            "mean_test_F1": round(float(np.mean(f1)), 4) if f1 else None,
            "median_test_F1": round(float(np.median(f1)), 4) if f1 else None,
            "mean_train_F1": round(float(np.mean(tr)), 4) if tr else None,
        }

    # per-model and per-draw breakdowns
    per_model = {}
    for m in sorted({r["model"] for r in recs}):
        g = [c["gap_pp"] for c in cells if c["model"] == m]
        if g:
            per_model[m] = {"n": len(g), "mean_gap_pp": round(float(np.mean(g)), 2)}
    per_draw = {}
    for d in sorted({r["draw"] for r in recs}):
        g = [c["gap_pp"] for c in cells if c["draw"] == d]
        if g:
            per_draw[str(d)] = {"n": len(g), "mean_gap_pp": round(float(np.mean(g)), 2)}

    w = np.array(paired_w) if paired_w else np.array([0.0])
    lo, hi = boot_ci(w)
    out = {
        "design": meta["design"],
        "patch_size": meta["patch_size"],
        "sizes": meta["sizes"],
        "n_pool": meta["n_pool"],
        "total_shared_pixels_both_arms": meta["total_shared_pixels_both_arms"],
        "min_edge_km": {
            "scattered": [d["scattered"]["min_edge_km"] for d in meta["draws"]],
            "separated": [d["separated"]["min_edge_km"] for d in meta["draws"]],
        },
        "n_paired_cells": len(cells),
        "n_runs": len(recs),
        "gap_separated_minus_scattered_pp": {
            "mean": round(float(w.mean()), 2),
            "median": round(float(np.median(w)), 2),
            "sd": round(float(w.std(ddof=1)), 2) if len(w) > 1 else None,
            "min": round(float(w.min()), 2),
            "max": round(float(w.max()), 2),
            "ci95": [lo, hi],
            "n_negative_cells": int((w < 0).sum()),
            "n_positive_cells": int((w > 0).sum()),
        },
        "by_arm": by_arm,
        "per_model": per_model,
        "per_draw": per_draw,
        "cells": cells,
        "reading": ("signed as separated (far) minus scattered (near). A "
                    "proximity-driven inflation would appear NEGATIVE, because "
                    "the scattered arm is the one adjacent to the training "
                    "zone. A value near zero is therefore the absence of a "
                    "same-signed proximity effect at this geometry, not its "
                    "confirmation."),
    }
    json.dump(out, open(OUT, "w"), indent=2)

    g = out["gap_separated_minus_scattered_pp"]
    print(f"E3-C  n_runs={len(recs)}  paired cells={len(cells)}  "
          f"shared pixels both arms={out['total_shared_pixels_both_arms']}")
    for arm, v in by_arm.items():
        print(f"  {arm:10s} test F1 mean {v['mean_test_F1']:.4f}  "
              f"train F1 mean {v['mean_train_F1']:.4f}  (n={v['n']})")
    print(f"  gap (separated - scattered): mean {g['mean']:+.2f} pp  "
          f"median {g['median']:+.2f}  CI95 {g['ci95']}  "
          f"range [{g['min']:+.2f}, {g['max']:+.2f}]")
    print(f"  cells: {g['n_negative_cells']} negative / {g['n_positive_cells']} positive")
    print("  per model:", {k: v["mean_gap_pp"] for k, v in per_model.items()})
    print("  per draw :", {k: v["mean_gap_pp"] for k, v in per_draw.items()})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()