#!/usr/bin/env python3
"""Prevalence control.

The interleaved (leaky) test set carries ~2.7x the change prevalence of the
isolated (honest) test set, so part of the F1 inflation may be a prevalence
(case-mix) artifact rather than a spatial effect. This script quantifies that
from the per-run confusion matrices already on disk (no new training).

For every (model, split-seed, run-seed) contrast it reads the two test arms of
the same run -- isolated_test (honest) and random_test (leaky) -- collapses the
4x4 CM to binary change (loss+gain) vs unchanged (stable_mangrove+
stable_background), and reports prevalence, precision, recall, F1 and MCC.
MCC is prevalence-insensitive, so the F1 gap vs the MCC gap isolates how much
of the F1 inflation is a prevalence artifact.

Deterministic (pure aggregation over fixed CMs); no seed required. Output:
results/prevalence_control.json
"""
import json, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "uniform")
MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]
SPLITS = [1, 2, 3]
SEEDS = [2026, 2027, 2028]
# collapsed runs (no-change solution), excluded everywhere else in the paper
EXCLUDED = {("unet", 3, 2026), ("unet_cbam", 3, 2026),
            ("unet_cbam_msa", 3, 2026)}
POS = {1, 2}  # change = loss, gain


def binary_from_cm(cm):
    tp = fp = fn = tn = 0
    for i in range(4):
        for j in range(4):
            v = cm[i][j]
            tpos = i in POS
            ppos = j in POS
            if tpos and ppos:
                tp += v
            elif (not tpos) and ppos:
                fp += v
            elif tpos and (not ppos):
                fn += v
            else:
                tn += v
    return tp, fp, fn, tn


def metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    prev = (tp + fn) / total if total else 0.0
    return dict(prevalence=prev, precision=prec, recall=rec, F1=f1, MCC=mcc)


def arm_metrics(arm_block):
    return metrics(*binary_from_cm(arm_block["confusion_matrix"]))


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def main():
    per_contrast = []
    for m in MODELS:
        for k in SPLITS:
            for s in SEEDS:
                if (m, k, s) in EXCLUDED:
                    continue
                path = os.path.join(R, f"random_split_k{k}",
                                   f"metrics_{m}_s{s}.json")
                d = json.load(open(path))
                honest = arm_metrics(d["isolated_test"])
                leaky = arm_metrics(d["random_test"])
                per_contrast.append(dict(
                    model=m, split=k, seed=s,
                    honest=honest, leaky=leaky,
                    prevalence_ratio=leaky["prevalence"] / honest["prevalence"],
                    F1_gap=leaky["F1"] - honest["F1"],
                    MCC_gap=leaky["MCC"] - honest["MCC"],
                ))

    n = len(per_contrast)
    summary = {
        "n_contrasts": n,
        "excluded_collapsed": sorted(
            f"{m}/k{k}-s{s}" for (m, k, s) in EXCLUDED),
        "honest": {k: round(mean([c["honest"][k] for c in per_contrast]), 4)
                    for k in ["prevalence", "precision", "recall", "F1", "MCC"]},
        "leaky": {k: round(mean([c["leaky"][k] for c in per_contrast]), 4)
                  for k in ["prevalence", "precision", "recall", "F1", "MCC"]},
        "prevalence_ratio_leaky_over_honest": round(
            mean([c["prevalence_ratio"] for c in per_contrast]), 3),
        "F1_gap_pp": round(100 * mean([c["F1_gap"] for c in per_contrast]), 2),
        "MCC_gap_pp": round(100 * mean([c["MCC_gap"] for c in per_contrast]), 2),
        "per_model": {
            m: {
                "n": sum(1 for c in per_contrast if c["model"] == m),
                "honest_F1": round(mean([c["honest"]["F1"] for c in per_contrast
                                         if c["model"] == m]), 4),
                "leaky_F1": round(mean([c["leaky"]["F1"] for c in per_contrast
                                        if c["model"] == m]), 4),
                "F1_gap_pp": round(100 * mean([c["F1_gap"] for c in per_contrast
                                               if c["model"] == m]), 2),
                "honest_MCC": round(mean([c["honest"]["MCC"] for c in per_contrast
                                          if c["model"] == m]), 4),
                "leaky_MCC": round(mean([c["leaky"]["MCC"] for c in per_contrast
                                         if c["model"] == m]), 4),
                "MCC_gap_pp": round(100 * mean([c["MCC_gap"] for c in per_contrast
                                                if c["model"] == m]), 2),
            } for m in MODELS
        },
        "per_contrast": per_contrast,
    }

    out = os.path.join(ROOT, "results", "prevalence_control.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    # console report
    print(f"n contrasts (excl. {len(EXCLUDED)} collapsed): {n}")
    print(f"prevalence  honest={summary['honest']['prevalence']:.4f}  "
          f"leaky={summary['leaky']['prevalence']:.4f}  "
          f"ratio={summary['prevalence_ratio_leaky_over_honest']:.3f}x")
    print(f"F1   honest={summary['honest']['F1']:.4f}  "
          f"leaky={summary['leaky']['F1']:.4f}  "
          f"gap={summary['F1_gap_pp']:+.2f} pp")
    print(f"MCC  honest={summary['honest']['MCC']:.4f}  "
          f"leaky={summary['leaky']['MCC']:.4f}  "
          f"gap={summary['MCC_gap_pp']:+.2f} pp  (prevalence-insensitive)")
    print(f"precision honest={summary['honest']['precision']:.4f} "
          f"leaky={summary['leaky']['precision']:.4f}")
    print(f"recall    honest={summary['honest']['recall']:.4f} "
          f"leaky={summary['leaky']['recall']:.4f}")
    print("\nper-model F1 gap vs MCC gap (pp):")
    for m, v in summary["per_model"].items():
        print(f"  {m:<18} F1 {v['honest_F1']:.3f}->{v['leaky_F1']:.3f} "
              f"({v['F1_gap_pp']:+.1f})  |  MCC {v['honest_MCC']:.3f}->"
              f"{v['leaky_MCC']:.3f} ({v['MCC_gap_pp']:+.1f})")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
