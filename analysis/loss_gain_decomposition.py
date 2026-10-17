"""Build <root>/loss_gain_decomposition.json: per-class split-protocol contrast.

Pools the 3-split-draw uniform (cw none) counterfactual sweep
(random_split_k{1,2,3}, 10 models x 3 run seeds) and reports the two change
classes SEPARATELY, per arm:
    honest = isolated_test  (test zone geographically disjoint from the training zone)
    leaky  = random_test    (test patches interleaved with the training zone)
for loss (class index 1) and gain (class index 2). This decomposes the pooled
change-F1 gap into a loss-component and a gain-component, so the manuscript can
state where the split-protocol inflation concentrates rather than leaving it as
one merged change class.

Same collapse exclusion as make_random_split_comparison.py (random-test F1ch <
COLLAPSE_T = all-background collapse) so the two tables pool the same runs.

Run:  MANGROVE_RESULTS_ROOT=results/uniform python3 analysis/loss_gain_decomposition.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from mangrove.paths import RESULTS_ROOT
from aggregate_results import binary_change, MODELS

KS = [1, 2, 3]
SEEDS = [2026, 2027, 2028]
COLLAPSE_T = 0.05
# 0 stable_mangrove, 1 loss, 2 gain, 3 stable_background
LOSS, GAIN = 1, 2


def class_f1(cm, c):
    """Per-class F1 for class c from a 4x4 confusion matrix (rows=truth, cols=pred)."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = cm[c, c]
    fn = cm[c, :].sum() - tp
    fp = cm[:, c].sum() - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-12)


def main():
    per, excluded = [], []
    for m in MODELS:
        for k in KS:
            for s in SEEDS:
                f = RESULTS_ROOT / f"random_split_k{k}" / f"metrics_{m}_s{s}.json"
                d = json.loads(f.read_text())
                run = f"k{k}-s{s}"
                if binary_change(d["random_test"]["confusion_matrix"])["F1_change"] < COLLAPSE_T:
                    excluded.append(f"{m}/{run}")
                    continue
                for arm, key in (("honest", "isolated_test"), ("leaky", "random_test")):
                    cm = d[key]["confusion_matrix"]
                    per.append({"model": m, "seed": s, "draw": f"k{k}", "arm": arm,
                                "loss": round(float(class_f1(cm, LOSS)), 4),
                                "gain": round(float(class_f1(cm, GAIN)), 4)})

    def mean(arm, cls):
        vals = [r[cls] for r in per if r["arm"] == arm]
        return round(float(np.mean(vals)), 4)

    lh, ll, gh, gl = mean("honest", "loss"), mean("leaky", "loss"), mean("honest", "gain"), mean("leaky", "gain")
    out = {
        "method": ("per-class F1 (loss=idx1, gain=idx2) pooled over 10 models x 3 split-draws x "
                   "3 run-seeds, uniform (cw none) protocol; honest=isolated_test, leaky=random_test"),
        "n_runs": len(per),
        "loss_honest_mean": lh,
        "loss_leaky_mean": ll,
        "loss_gap_pp": round((ll - lh) * 100, 2),
        "gain_honest_mean": gh,
        "gain_leaky_mean": gl,
        "gain_gap_pp": round((gl - gh) * 100, 2),
        "excluded_collapsed_runs": excluded,
        "per_run": per,
    }
    dest = RESULTS_ROOT / "loss_gain_decomposition.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"saved {dest} (n_runs={len(per)}, excluded={len(excluded)})")
    print(json.dumps({k: out[k] for k in
                      ("loss_honest_mean", "loss_leaky_mean", "loss_gap_pp",
                       "gain_honest_mean", "gain_leaky_mean", "gain_gap_pp")}, indent=1))


if __name__ == "__main__":
    main()
