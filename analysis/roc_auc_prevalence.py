"""Prevalence-free discrimination check for the split-protocol contrast.

Motivation
----------
The manuscript previously argued that the inflation survives a "prevalence-
robust" metric, using MCC and average precision. Both are prevalence-sensitive:

  * AP's random baseline *is* the prevalence, so AP rises with prevalence even
    at fixed discrimination.
  * MCC depends on PPV, which depends on prevalence.

ROC-AUC is P(score(change) > score(no-change)) computed from the TPR/FPR curve.
For a FIXED model it does not depend on class prevalence: two test sets with
different prevalence but identical TPR/FPR curves have identical ROC-AUC. It is
therefore the metric that can actually answer "does the split arrangement
change the model's discrimination, or only the operating-point economics of a
threshold-dependent metric?"

This script recomputes AP, PR-AUC, ROC-AUC, balanced accuracy and MCC for the
isolated ("honest") and interleaved ("leaky") arms of the same checkpoints, so
the four metrics can be compared on identical footing.

Writes results/prevalence_roc.json.
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             matthews_corrcoef, precision_recall_curve, auc,
                             roc_auc_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# reuse the tested loaders / model factory / score extraction rather than
# re-implementing them (ap_pr_auc.py is the reference implementation)
import importlib.util

_spec = importlib.util.spec_from_file_location("apm", "analysis/ap_pr_auc.py")
_apm = importlib.util.module_from_spec(_spec)
_saved = sys.argv
sys.argv = ["ap_pr_auc.py"]
try:
    _spec.loader.exec_module(_apm)
except SystemExit:
    pass
sys.argv = _saved

build_loader = _apm.build_loader
softmax_scores = _apm.softmax_scores
_apm.TEMPORAL_MODELS = {"temporal_unet", "temporal_unet_gn", "fc_siam",
                        "changeformer"}   # dual-input family, in_ch = 9
load_model = _apm.load_model

R = os.path.join(ROOT, "results", "uniform")
OUT = os.path.join(ROOT, "results", "prevalence_roc.json")

MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "unet_msa", "deeplabv3plus",
          "deeplabv3plus_pre", "temporal_unet", "temporal_unet_gn", "fc_siam",
          "changeformer"]
SPLITS = [1, 2, 3]
SEEDS = [2026, 2027, 2028]


def collapse(pred_argmax_path):
    """True when the run predicted no change pixel at all."""
    if not os.path.exists(pred_argmax_path):
        return False
    a = np.load(pred_argmax_path, allow_pickle=True)
    return bool(((a == 1) | (a == 2)).sum() == 0)


def main():
    # resume: the file already holds a list of completed pair records
    rows = []
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT))
            if isinstance(prev, list):
                rows = prev
        except Exception:
            rows = []
    done = {(r["model"], r["split"], r["seed"]) for r in rows}
    print(f"resuming with {len(rows)} pairs already done", flush=True)
    for m in MODELS:
        temporal = None
        for k in SPLITS:
            for s in SEEDS:
                if (m, k, s) in done:
                    continue
                ck = os.path.join(R, f"random_split_k{k}", "checkpoints",
                                  f"{m}_s{s}.pt")
                if not os.path.exists(ck):
                    continue
                sd = torch.load(ck, map_location="cpu")
                w = next((v for v in sd.values()
                          if hasattr(v, "dim") and v.dim() == 4
                          and v.shape[1] in (9, 27)), None)
                in_ch = int(w.shape[1]) if w is not None else 27
                temporal = in_ch == 9
                model = _apm.MODEL_REGISTRY[m](in_ch=in_ch, out_ch=4).to(_apm.DEVICE)
                model.load_state_dict(sd)
                model.eval()
                rec = {"model": m, "split": k, "seed": s}
                ok = True
                for arm, split in (("honest", "test"), ("leaky", k)):
                    dl = build_loader("vietnam", split, m)
                    score, ybin = softmax_scores(model, dl, temporal)
                    if ybin.sum() < 2 or (len(ybin) - ybin.sum()) < 1:
                        ok = False
                        break
                    thr = 0.5
                    yhat = (score >= thr).astype(int)
                    pred = score
                    rec[arm] = {
                        "prev": round(float(ybin.mean()), 5),
                        "roc_auc": round(float(roc_auc_score(ybin, score)), 4),
                        "ap": round(float(average_precision_score(ybin, score)), 4),
                        "mcc": round(float(matthews_corrcoef(ybin, yhat)), 4),
                        "bal_acc": round(float(balanced_accuracy_score(ybin, yhat)), 4),
                    }
                    prec, rec_, _ = precision_recall_curve(ybin, score)
                    rec[arm]["pr_auc"] = round(float(auc(rec_, prec)), 4)
                if not ok:
                    print(f"  SKIP {m}/k{k}-s{s}: degenerate arm", flush=True)
                    continue
                rows.append(rec)
                print(f"  {m:18s} k{k} s{s}  prev "
                      f"{rec['honest']['prev']*100:.2f}->{rec['leaky']['prev']*100:.2f}%  "
                      f"ROC {rec['honest']['roc_auc']:.4f}->{rec['leaky']['roc_auc']:.4f}  "
                      f"AP {rec['honest']['ap']:.4f}->{rec['leaky']['ap']:.4f}", flush=True)
                with open(OUT, "w") as fh:
                    json.dump(rows, fh, indent=1)

    # ---- aggregate: paired gaps per metric, per model and overall ---------
    METRICS = ["roc_auc", "ap", "pr_auc", "mcc", "bal_acc"]
    gaps = {k: [] for k in METRICS}
    per_model = {}
    for r in rows:
        g = {k: round(100 * (r["leaky"][k] - r["honest"][k]), 2) for k in METRICS}
        per_model.setdefault(r["model"], []).append(g)
        for k in METRICS:
            gaps[k].append(g[k])

    summary = {"n_pairs": len(rows), "metrics": {}}
    for k in METRICS:
        v = np.array(gaps[k], float)
        summary["metrics"][k] = {
            "mean_gap_pp": round(float(v.mean()), 2),
            "min_gap_pp": round(float(v.min()), 2),
            "max_gap_pp": round(float(v.max()), 2),
            "sd_gap_pp": round(float(v.std(ddof=1)), 2) if len(v) > 1 else None,
            "per_model_mean_gap_pp": {
                mm: round(float(np.mean([x[k] for x in gg])), 2)
                for mm, gg in per_model.items()},
        }
    prev_ratio = [r["leaky"]["prev"] / r["honest"]["prev"] for r in rows]
    summary["prevalence_ratio_leaky_over_honest"] = {
        "mean": round(float(np.mean(prev_ratio)), 3),
        "min": round(float(np.min(prev_ratio)), 3),
        "max": round(float(np.max(prev_ratio)), 3),
    }
    summary["pairs"] = rows
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n=== prevalence-free discrimination vs prevalence-scaled metrics ===")
    for k in METRICS:
        d = summary["metrics"][k]
        print(f"  {k:9s} mean gap {d['mean_gap_pp']:+7.2f} pp   "
              f"range [{d['min_gap_pp']:+.2f}, {d['max_gap_pp']:+.2f}]")
    print(f"  prevalence ratio leaky/honest: "
          f"{summary['prevalence_ratio_leaky_over_honest']['mean']:.2f}x")
    print(f"wrote {OUT}  ({len(rows)} pairs)")


if __name__ == "__main__":
    main()