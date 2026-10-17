r"""Prevalence-free re-scoring of the E3-C proximity control.

E3-C's headline is a change-F1 contrast of +0.56 pp between the separated and
scattered arms. Section~\ref{sec:rand} shows that a hard argmax can report
change-F1 of exactly 0 for a model that ranks change far above chance, so an
F1-only control is not sufficient evidence: if the "proximity adds nothing"
conclusion is an artefact of the threshold, it will not survive on a
prevalence-free metric. This script recomputes the same paired contrast on
ROC-AUC and average precision, and also reports how far each run sits from the
argmax floor.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E3C = os.path.join(ROOT, "results", "e3c")
ORIG = os.path.join(ROOT, "results", "e3c_f1_orig")
OUT = os.path.join(ROOT, "results", "e3c_roc_gap.json")
METRICS = ["F1_change", "roc_auc", "ap", "MCC_change", "bal_acc_change"]


def derive(t, prev):
    """Recover the binary confusion matrix from P, R and prevalence.

    The run records store F1/precision/recall but not the confusion matrix or
    MCC, and MCC is one of the prevalence-dependent metrics we want on the same
    footing as F1. P, R and prevalence pin the binary table exactly (up to the
    pixel count), so everything else follows.
    """
    P = float(t["precision_change"])
    R = float(t["recall_change"])
    N = 1.0
    tp = R * prev * N
    fp = tp * (1 - P) / P if P > 0 else 0.0
    fn = prev * N - tp
    tn = N - tp - fp - fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return round(float((R + prec) / 2), 6), round(float(mcc), 6)


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "metrics_*.json"))):
        r = json.load(open(f))
        out[(r["model"], r["draw"], r["arm"], r["seed"])] = r
    return out


def paired(rows, key):
    """separated - scattered within each (model, draw, seed) cell."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["model"], r["draw"], r["seed"])][r["arm"]] = r
    diffs = {}
    for cell, arms in by.items():
        if "separated" in arms and "scattered" in arms:
            a, b = arms["separated"], arms["scattered"]
            diffs[cell] = 100.0 * (float(a[key]) - float(b[key]))
    return diffs


def summ(v):
    if not v:
        return None
    a = np.array(list(v.values()), float)
    n = len(a)
    se = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"n_cells": n, "mean_pp": round(float(a.mean()), 3),
            "median_pp": round(float(np.median(a)), 3),
            "sd_pp": round(float(a.std(ddof=1)), 3) if n > 1 else None,
            "ci95": [round(float(a.mean() - 1.96 * se), 3),
                     round(float(a.mean() + 1.96 * se), 3)],
            "n_negative": int((a < 0).sum()), "n_positive": int((a > 0).sum()),
            "min_pp": round(float(a.min()), 3), "max_pp": round(float(a.max()), 3)}


def main():
    new = load(E3C)
    have = [r for r in new.values() if "test_scores" in r]
    if len(have) < len(new):
        print(f"WARNING: {len(new) - len(have)} of {len(new)} runs lack "
              f"test_scores (sweep incomplete)")
    rows = []
    for r in have:
        prev = r["test_scores"]["prevalence"]
        bal, mcc = derive(r["test"], prev)
        rows.append({"model": r["model"], "draw": r["draw"], "arm": r["arm"],
                     "seed": r["seed"],
                     "F1_change": r["test"]["F1_change"],
                     "MCC_change": mcc, "bal_acc_change": bal,
                     "roc_auc": r["test_scores"]["roc_auc"],
                     "ap": r["test_scores"]["ap"],
                     "prevalence": prev,
                     "max_p_change": r["test_scores"]["max_p_change"]})

    out = {"n_runs": len(rows), "n_cells": len(paired(rows, "roc_auc")),
           "metric": "separated minus scattered, paired by (model, draw, seed)",
           "gaps": {}}
    for k in ("F1_change", "roc_auc", "ap", "MCC_change", "bal_acc_change"):
        d = paired(rows, k)
        if d and all(v is not None for v in d.values()):
            out["gaps"][k] = summ(d)

    # argmax-floor exposure inside E3-C
    out["argmax_floor"] = {
        "n_runs_max_p_change_below_0p5": sum(
            1 for r in rows if r["max_p_change"] < 0.5),
        "min_max_p_change": round(min(r["max_p_change"] for r in rows), 4),
        "median_max_p_change": round(
            float(np.median([r["max_p_change"] for r in rows])), 4),
    }

    # reproducibility: same-seed re-run drift vs the archived original sweep
    if os.path.isdir(ORIG):
        old = load(ORIG)
        drift = [100 * (new[k]["test"]["F1_change"] - old[k]["test"]["F1_change"])
                 for k in old if k in new]
        if drift:
            out["same_seed_rerun_f1_drift_pp"] = {
                "n": len(drift),
                "mean_pp": round(float(np.mean(drift)), 4),
                "max_abs_pp": round(float(np.max(np.abs(drift))), 4),
                "note": ("training is not bit-deterministic on this GPU; F1 and "
                         "the score-based metrics therefore come from the same "
                         "re-run rather than being spliced across runs"),
            }

    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    out["by_arm"] = {a: {"n": len(v),
                         "F1_change": round(float(np.mean([x["F1_change"] for x in v])), 4),
                         "roc_auc": round(float(np.mean([x["roc_auc"] for x in v])), 4),
                         "ap": round(float(np.mean([x["ap"] for x in v])), 4),
                         "prevalence": round(float(np.mean([x["prevalence"] for x in v])), 5)}
                      for a, v in by_arm.items()}
    json.dump(out, open(OUT, "w"), indent=2)

    print(f"E3-C paired contrast (separated - scattered), n={out['n_cells']} cells")
    for k, v in out["gaps"].items():
        print(f"  {k:16s} mean {v['mean_pp']:+7.3f} pp  "
              f"CI [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]  "
              f"neg/pos {v['n_negative']}/{v['n_positive']}")
    f = out.get("same_seed_rerun_f1_drift_pp")
    if f:
        print(f"\nsame-seed re-run F1 drift: mean {f['mean_pp']:+.4f} pp, "
              f"max |{f['max_abs_pp']:.3f}| pp over {f['n']} runs")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
