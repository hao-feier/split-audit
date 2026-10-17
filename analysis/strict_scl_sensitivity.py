#!/usr/bin/env python3
"""Strict-SCL-mask sensitivity.

The baseline composites keep SCL 1-7 (saturated/dark/shadow/unclassified) as
"clear"; the strict variant keeps only SCL 4-6 (vegetation / not-vegetated /
water). To isolate the *masking* effect we keep the SAME 76 patches (selected by
the baseline clear-fraction criterion) and vary only the composite, then retrain
unet (split-seed 1, run-seeds 2026-2028) on the same both-arm protocol.

This script compares the honest-vs-leaky inflation gap under the two masks, for
the same model / split-seed / run-seeds, from the per-run confusion matrices on
disk (no new training, no seed):
  strict  = results/strict_scl/random_split_k1/metrics_unet_s{seed}.json
  baseline= results/uniform/random_split_k1/metrics_unet_s{seed}.json
Each file carries isolated_test (honest) and random_test (leaky).

Output: results/strict_scl_sensitivity.json
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRICT = os.path.join(ROOT, "results", "strict_scl", "random_split_k1")
BASELINE = os.path.join(ROOT, "results", "uniform", "random_split_k1")
SEEDS = [2026, 2027, 2028]
MODEL = "unet"
POS = {1, 2}  # change = loss, gain


def binary_from_cm(cm):
    tp = fp = fn = tn = 0
    for i in range(4):
        for j in range(4):
            v = cm[i][j]
            tpos, ppos = i in POS, j in POS
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
    den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    prev = (tp + fn) / total if total else 0.0
    return dict(prevalence=prev, precision=prec, recall=rec, F1=f1, MCC=mcc)


def arm(path, key):
    d = json.load(open(path))
    return metrics(*binary_from_cm(d[key]["confusion_matrix"]))


def main():
    rows = []
    for s in SEEDS:
        sp = os.path.join(STRICT, f"metrics_{MODEL}_s{s}.json")
        bp = os.path.join(BASELINE, f"metrics_{MODEL}_s{s}.json")
        if not (os.path.exists(sp) and os.path.exists(bp)):
            print(f"  SKIP s{s}: strict={os.path.exists(sp)} baseline={os.path.exists(bp)}")
            continue
        s_h, s_l = arm(sp, "isolated_test"), arm(sp, "random_test")
        b_h, b_l = arm(bp, "isolated_test"), arm(bp, "random_test")
        rows.append(dict(
            seed=s,
            strict=dict(honest=s_h, leaky=s_l,
                        F1_gap_pp=round(100 * (s_l["F1"] - s_h["F1"]), 2),
                        MCC_gap_pp=round(100 * (s_l["MCC"] - s_h["MCC"]), 2)),
            baseline=dict(honest=b_h, leaky=b_l,
                          F1_gap_pp=round(100 * (b_l["F1"] - b_h["F1"]), 2),
                          MCC_gap_pp=round(100 * (b_l["MCC"] - b_h["MCC"]), 2)),
        ))
    if not rows:
        print("no runs ready yet; re-run after training completes")
        return

    n = len(rows)

    def avg(field):
        return sum(r["strict"][field] for r in rows) / n

    def bavg(field):
        return sum(r["baseline"][field] for r in rows) / n

    summary = {
        "model": MODEL, "split_seed": 1, "seeds": [r["seed"] for r in rows],
        "note": ("Same 76 patches / split-seed / run-seeds; only the composite "
                 "mask differs (strict SCL 4-6 vs baseline SCL 1-7). honest = "
                 "isolated_test, leaky = random_test."),
        "strict_F1_gap_pp": round(avg("F1_gap_pp"), 2),
        "strict_MCC_gap_pp": round(avg("MCC_gap_pp"), 2),
        "baseline_F1_gap_pp": round(bavg("F1_gap_pp"), 2),
        "baseline_MCC_gap_pp": round(bavg("MCC_gap_pp"), 2),
        "per_seed": rows,
    }

    out = os.path.join(ROOT, "results", "strict_scl_sensitivity.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"=== B4 strict-SCL-mask sensitivity ({MODEL}, split-seed 1) ===")
    print(f"seeds: {summary['seeds']}")
    print(f"F1  gap  baseline={summary['baseline_F1_gap_pp']:+.1f} pp  "
          f"strict={summary['strict_F1_gap_pp']:+.1f} pp")
    print(f"MCC gap  baseline={summary['baseline_MCC_gap_pp']:+.1f} pp  "
          f"strict={summary['strict_MCC_gap_pp']:+.1f} pp")
    for r in rows:
        print(f"  s{r['seed']}: baseline F1 {r['baseline']['honest']['F1']:.3f}->"
              f"{r['baseline']['leaky']['F1']:.3f} ({r['baseline']['F1_gap_pp']:+.1f}) | "
              f"strict F1 {r['strict']['honest']['F1']:.3f}->"
              f"{r['strict']['leaky']['F1']:.3f} ({r['strict']['F1_gap_pp']:+.1f})")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
