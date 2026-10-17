#!/usr/bin/env python3
"""Prevalence-invariant ranking control (AP/PR-AUC arm).

prevalence_control.py showed the F1 inflation (honest -> leaky) is only ~3 pp
explainable by prevalence (case-mix) via MCC. This goes one step further with
the canonical prevalence-invariant RANKING metric, average precision (AP /
PR-AUC), which is computed from the model's soft change score and is insensitive
to the operating threshold and to the base rate.

For every (model, split-seed, run-seed) contrast in the SAME 42-contrast set as
prevalence_control.py we re-run inference (no new training) with the SAME
checkpoint on BOTH test arms:
  * honest = buffer-isolated vietnam_test          (isolated_test in exp3c)
  * leaky  = interleaved vietnam_test_rand{k}      (random_test  in exp3c)
capture the 4-class softmax, form the binary change score
s = P(loss) + P(gain)  (classes 1,2), and compute average precision of the
binary change task. We also recompute the binary confusion matrix from
softmax-argmax and report binary precision/recall/F1/MCC so the pipeline can be
cross-checked against the on-disk hard metrics (and prevalence_control.json).

Deterministic given the checkpoints + splits; softmax capture is the only new
computation. Output: results/prevalence_ap.json
"""
import json
import math
import os
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, auc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mangrove.config import CHANGE_CLASSES  # noqa: E402
from mangrove.datasets import ChangePatchDataset, load_split  # noqa: E402
from mangrove.models import MODEL_REGISTRY  # noqa: E402
from mangrove.paths import SPLITS_ROOT  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

R = os.path.join(ROOT, "results", "uniform")
MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]
SPLITS = [1, 2, 3]
SEEDS = [2026, 2027, 2028]
EXCLUDED = {("unet", 3, 2026), ("unet_cbam", 3, 2026),
            ("unet_cbam_msa", 3, 2026)}
POS = {1, 2}  # change = loss (1) + gain (2)
TEMPORAL_MODELS = {"temporal_unet"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(x):
    return x / 10000.0


def build_loader(region, split, model_name, batch_size=None):
    """Test loader. `split` is 'test' (isolated) or a rand index (leaky)."""
    concat = model_name not in TEMPORAL_MODELS
    ds = ChangePatchDataset(region, "test", augment=False, concat=concat)
    if split != "test":
        idx = load_split(region, f"test_rand{split}")
        ds.idx = idx
    bs = batch_size or (2 if model_name in TEMPORAL_MODELS else 8)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)


@torch.no_grad()
def softmax_scores(model, dl, temporal):
    model.eval()
    scores, ybins = [], []
    for batch in dl:
        if temporal:
            x1, x2, y, _ = batch
            x1, x2 = normalize(x1).to(DEVICE), normalize(x2).to(DEVICE)
            out = model(x1, x2)
        else:
            x, y, _ = batch
            x = normalize(x).to(DEVICE)
            out = model(x)
        sm = torch.softmax(out, dim=1)
        # per-pixel binary change score = P(loss) + P(gain), flattened over HxW
        s = (sm[:, 1] + sm[:, 2]).cpu().numpy().ravel()
        yy = y.numpy()
        scores.append(s)
        ybins.append(np.isin(yy, list(POS)).astype(int).ravel())
    return np.concatenate(scores), np.concatenate(ybins)


def binary_metrics_from_argmax(model, dl, temporal):
    """Reproduce the hard-prediction binary CM (cross-check vs on-disk)."""
    model.eval()
    tp = fp = fn = tn = 0
    for batch in dl:
        if temporal:
            x1, x2, y, _ = batch
            x1, x2 = normalize(x1).to(DEVICE), normalize(x2).to(DEVICE)
            pred = model(x1, x2).argmax(1).cpu().numpy()
        else:
            x, y, _ = batch
            x = normalize(x).to(DEVICE)
            pred = model(x).argmax(1).cpu().numpy()
        y = y.numpy()
        tpos = np.isin(y, list(POS))
        ppos = np.isin(pred, list(POS))
        tp += int(np.sum(tpos & ppos))
        fp += int(np.sum(~tpos & ppos))
        fn += int(np.sum(tpos & ~ppos))
        tn += int(np.sum(~tpos & ~ppos))
    total = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    prev = (tp + fn) / total if total else 0.0
    return dict(prevalence=prev, precision=prec, recall=rec, F1=f1, MCC=mcc), \
        dict(tp=tp, fp=fp, fn=fn, tn=tn, n=total)


def load_model(model_name):
    temporal = model_name in TEMPORAL_MODELS
    in_ch = 9 if temporal else 27
    model = MODEL_REGISTRY[model_name](in_ch=in_ch, out_ch=4).to(DEVICE)
    return model, temporal


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


OUT_PATH = os.path.join(ROOT, "results", "prevalence_ap.json")


def _partial(per_contrast):
    """Write a resumable partial file (summary recomputed on a full run)."""
    with open(OUT_PATH, "w") as f:
        json.dump({"per_contrast": per_contrast,
                   "_partial": len(per_contrast) < 42}, f, indent=2)


def main():
    per_contrast = []
    done_set = set()
    if os.path.exists(OUT_PATH):
        try:
            prev = json.load(open(OUT_PATH)).get("per_contrast", [])
            per_contrast = [c for c in prev if c.get("honest") and c.get("leaky")]
            done_set = {(c["model"], c["split"], c["seed"]) for c in per_contrast}
            if done_set:
                print(f"[resume] {len(done_set)} contrasts already on disk",
                      flush=True)
        except Exception:
            per_contrast, done_set = [], set()
    n_total = sum(1 for m in MODELS for k in SPLITS for s in SEEDS
                  if (m, k, s) not in EXCLUDED)
    done = len(per_contrast)
    for m in MODELS:
        for k in SPLITS:
            for s in SEEDS:
                if (m, k, s) in EXCLUDED:
                    continue
                if (m, k, s) in done_set:
                    continue
                ckpt = os.path.join(R, f"random_split_k{k}",
                                   "checkpoints", f"{m}_s{s}.pt")
                if not os.path.exists(ckpt):
                    print(f"  SKIP {m}/k{k}-s{s}: no checkpoint {ckpt}", flush=True)
                    continue
                model, temporal = load_model(m)
                state = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state)

                rec = dict(model=m, split=k, seed=s)
                for arm, split in (("honest", "test"), ("leaky", k)):
                    dl = build_loader("vietnam", split, m)
                    score, ybin = softmax_scores(model, dl, temporal)
                    if ybin.sum() < 2 or (len(ybin) - ybin.sum()) < 1:
                        ap = None
                    else:
                        ap = float(average_precision_score(ybin, score))
                    # PR-AUC by explicit integration (should equal AP here)
                    prec, recall, _ = precision_recall_curve(ybin, score)
                    pr_auc = float(auc(recall, prec))
                    bm, cm = binary_metrics_from_argmax(model, dl, temporal)
                    rec[arm] = dict(
                        AP=round(ap, 4) if ap is not None else None,
                        PR_AUC=round(pr_auc, 4),
                        n=cm["n"], n_change=int(ybin.sum()),
                        **{kk: round(vv, 4) for kk, vv in bm.items()},
                    )
                # gap metrics (AP is prevalence-invariant)
                def _g(key):
                    h = rec["honest"].get(key)
                    l = rec["leaky"].get(key)
                    return (l - h) if (h is not None and l is not None) else None
                rec["AP_gap_pp"] = round(100 * _g("AP"), 2)
                rec["F1_gap_pp"] = round(100 * _g("F1"), 2)
                rec["MCC_gap_pp"] = round(100 * _g("MCC"), 2)
                rec["prevalence_ratio"] = round(
                    rec["leaky"]["prevalence"] / rec["honest"]["prevalence"], 3)
                per_contrast.append(rec)
                done += 1
                print(f"[{done}/{n_total}] {m:<18} k{k} s{s}: "
                      f"honest AP={rec['honest']['AP']} F1={rec['honest']['F1']} "
                      f"| leaky AP={rec['leaky']['AP']} F1={rec['leaky']['F1']}",
                      flush=True)
                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                _partial(per_contrast)  # resumable checkpoint

    summary = {
        "n_contrasts": len(per_contrast),
        "excluded_collapsed": sorted(f"{m}/k{k}-s{s}" for (m, k, s) in EXCLUDED),
        "metric_note": ("AP (average precision, a.k.a. PR-AUC) of the binary "
                        "change task from soft score P(loss)+P(gain); prevalence-"
                        "invariant ranking metric."),
        "honest": {kk: round(mean([c["honest"][kk] for c in per_contrast]), 4)
                   for kk in ["prevalence", "F1", "MCC", "AP", "PR_AUC"]},
        "leaky": {kk: round(mean([c["leaky"][kk] for c in per_contrast]), 4)
                  for kk in ["prevalence", "F1", "MCC", "AP", "PR_AUC"]},
        "AP_gap_pp": round(mean([c["AP_gap_pp"] for c in per_contrast]), 2),
        "F1_gap_pp": round(mean([c["F1_gap_pp"] for c in per_contrast]), 2),
        "MCC_gap_pp": round(mean([c["MCC_gap_pp"] for c in per_contrast]), 2),
        "prevalence_ratio_leaky_over_honest": round(
            mean([c["prevalence_ratio"] for c in per_contrast]), 3),
        "per_model": {
            m: {
                "n": sum(1 for c in per_contrast if c["model"] == m),
                "honest_AP": round(mean([c["honest"]["AP"] for c in per_contrast
                                         if c["model"] == m]), 4),
                "leaky_AP": round(mean([c["leaky"]["AP"] for c in per_contrast
                                        if c["model"] == m]), 4),
                "AP_gap_pp": round(mean([c["AP_gap_pp"] for c in per_contrast
                                         if c["model"] == m]), 2),
                "F1_gap_pp": round(mean([c["F1_gap_pp"] for c in per_contrast
                                         if c["model"] == m]), 2),
                "MCC_gap_pp": round(mean([c["MCC_gap_pp"] for c in per_contrast
                                          if c["model"] == m]), 2),
            } for m in MODELS
        },
        "per_contrast": per_contrast,
    }

    out = os.path.join(ROOT, "results", "prevalence_ap.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== AP / PR-AUC (prevalence-invariant) summary ===")
    print(f"n contrasts (excl. {len(EXCLUDED)} collapsed): {len(per_contrast)}")
    print(f"prevalence honest={summary['honest']['prevalence']:.4f}  "
          f"leaky={summary['leaky']['prevalence']:.4f}  "
          f"ratio={summary['prevalence_ratio_leaky_over_honest']:.3f}x")
    print(f"AP    honest={summary['honest']['AP']:.4f}  "
          f"leaky={summary['leaky']['AP']:.4f}  gap={summary['AP_gap_pp']:+.2f} pp")
    print(f"F1    honest={summary['honest']['F1']:.4f}  "
          f"leaky={summary['leaky']['F1']:.4f}  gap={summary['F1_gap_pp']:+.2f} pp")
    print(f"MCC   honest={summary['honest']['MCC']:.4f}  "
          f"leaky={summary['leaky']['MCC']:.4f}  gap={summary['MCC_gap_pp']:+.2f} pp")
    print("\nper-model AP gap vs F1 gap vs MCC gap (pp):")
    for m, v in summary["per_model"].items():
        print(f"  {m:<18} AP {v['honest_AP']:.3f}->{v['leaky_AP']:.3f} "
              f"({v['AP_gap_pp']:+.1f})  |  F1 gap {v['F1_gap_pp']:+.1f}  |  "
              f"MCC gap {v['MCC_gap_pp']:+.1f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
