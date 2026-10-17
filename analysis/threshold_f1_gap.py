"""Threshold/FPR-optimized change-F1 gap: interleaved vs buffer-separated arms.

Reviewer concern (deepseek 2.1, claude M5): the headline change-F1 is scored
at the four-class argmax, an unoptimised operating point under extreme
imbalance; if the interleaved-vs-separated gap were an artefact of that
decision rule it should shrink once the threshold is moved to a fair
operating point.

This script re-scores the SAME trained model (k1 checkpoint) on the honest
test set and the three interleaved test sets, and computes change-F1 at
several operating points on the change score s = P_loss + P_gain:

  - argmax       : the paper's headline operating point (4-class argmax)
  - FPR = 1%     : a fixed, fair high-precision operating point
  - FPR = 5%     : same, looser
  - val-optimal  : threshold chosen to maximise change-F1 on the shared
                   validation set, then applied to each test set (no
                   test-set peeking)
  - F1-optimal   : the F1-maximising threshold on the test set itself
                   (oracle upper bound; reported only as a bound)

If the interleaved-minus-honest gap persists across operating points, the
inflation is not a decision-rule artefact. Inference only (no training).

Output: results/uniform/threshold_f1_gap.json
Run:    python3 analysis/threshold_f1_gap.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from mangrove.datasets import load_region, load_split  # noqa: E402
from mangrove.models import MODEL_REGISTRY, DUAL_INPUT_MODELS  # noqa: E402

MODELS = [
    "unet", "unet_cbam", "unet_cbam_msa", "unet_msa",
    "temporal_unet", "temporal_unet_gn", "deeplabv3plus",
    "deeplabv3plus_pre", "fc_siam", "changeformer",
]


@torch.no_grad()
def get_probs4(net, data, patch_idx, is_temporal):
    """Return full (N, 4) class probabilities for the given patches."""
    t1 = data["t1_feat"][patch_idx]
    t2 = data["t2_feat"][patch_idx]
    parts = []
    if is_temporal:
        a = torch.from_numpy((t1.transpose(0, 3, 1, 2) / 10000.0).copy())
        b = torch.from_numpy((t2.transpose(0, 3, 1, 2) / 10000.0).copy())
        for i in range(0, len(a), 2):
            parts.append(net(a[i:i + 2], b[i:i + 2]))
    else:
        x = np.concatenate([t1, t2, np.abs(t1 - t2)], axis=3)
        xt = torch.from_numpy((x.transpose(0, 3, 1, 2) / 10000.0).copy())
        for i in range(0, len(xt), 4):
            parts.append(net(xt[i:i + 4]))
    logits = torch.cat(parts, dim=0).cpu().numpy()
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def argmax_change_f1(probs, label):
    """Change-F1 under the four-class argmax decision (paper headline point)."""
    cls = probs.argmax(axis=1).ravel()
    pred = (cls == 1) | (cls == 2)
    tp = int(np.sum(pred & (label == 1)))
    fp = int(np.sum(pred & (label == 0)))
    fn = int(np.sum((~pred) & (label == 1)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def f1_curve(s, label):
    """Vectorised change-F1 at every 'top-k' threshold, plus FPR per rank."""
    order = np.argsort(-s, kind="mergesort")
    y = label[order].astype(int)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(len(y) - int(y.sum()), 1)
    rec = tp / n_pos
    prec = tp / (tp + fp)
    f1 = np.where(tp + fp > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    fpr = fp / n_neg
    return f1, fpr, s[order]


def f1_at_threshold(s, label, thr):
    pred = s >= thr
    tp = int(np.sum(pred & (label == 1)))
    fp = int(np.sum(pred & (label == 0)))
    fn = int(np.sum((~pred) & (label == 1)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def f1_at_fpr(f1, fpr, target):
    idx = np.where(fpr >= target)[0]
    if len(idx) == 0:
        return float("nan")
    return float(f1[idx[0]])


def best_f1_and_threshold(f1):
    i = int(np.argmax(f1))
    return float(f1[i])


def main():
    t0 = time.time()
    data = load_region("vietnam")
    splits = {
        "honest": np.asarray(load_split("vietnam", "test")),
        "leaky_k1": np.asarray(load_split("vietnam", "test_rand1")),
        "leaky_k2": np.asarray(load_split("vietnam", "test_rand2")),
        "leaky_k3": np.asarray(load_split("vietnam", "test_rand3")),
    }
    val_idx = np.asarray(load_split("vietnam", "val"))
    labels = {n: ((data["change"][idx] == 1) | (data["change"][idx] == 2))
              .ravel().astype(int) for n, idx in splits.items()}
    label_val = ((data["change"][val_idx] == 1) | (data["change"][val_idx] == 2)) \
        .ravel().astype(int)

    out = {"note": "within-model contrast: k1 checkpoint scored on the honest "
                   "test set and three interleaved test sets; change score = "
                   "P_loss + P_gain; val-optimal threshold chosen on the shared "
                   "validation set (vietnam_val.json).",
           "models": {}}
    for m in MODELS:
        ckpt = ROOT / "results/uniform/random_split_k1/checkpoints" / f"{m}_s2027.pt"
        if not ckpt.exists():
            print(f"[skip] {m}: no {ckpt.name}")
            continue
        is_temporal = m in DUAL_INPUT_MODELS
        in_ch = 9 if is_temporal else 27
        net = MODEL_REGISTRY[m](in_ch=in_ch, out_ch=4)
        net.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
        net.eval()

        # scores + full probs per split (probs needed for the argmax point)
        score = {}
        probs_cache = {}
        for n, idx in splits.items():
            probs = get_probs4(net, data, idx, is_temporal)
            score[n] = (probs[:, 1] + probs[:, 2]).ravel()
            probs_cache[n] = probs
        probs_val = get_probs4(net, data, val_idx, is_temporal)
        s_val = (probs_val[:, 1] + probs_val[:, 2]).ravel()
        val_f1, val_fpr, _ = f1_curve(s_val, label_val)
        thr_val = s_val[int(np.argmax(val_f1))]  # val-optimal threshold (shared)

        rec = {}
        for n in splits:
            s, lab = score[n], labels[n]
            f1, fpr, thr_sorted = f1_curve(s, lab)
            best_i = int(np.argmax(f1))
            rec[n] = {
                "pi": round(float(lab.mean()), 4),
                "argmax_F1": round(argmax_change_f1(probs_cache[n], lab), 4),
                "F1_at_FPR_0.01": round(f1_at_fpr(f1, fpr, 0.01), 4),
                "F1_at_FPR_0.05": round(f1_at_fpr(f1, fpr, 0.05), 4),
                "F1_valopt": round(f1_at_threshold(s, lab, float(thr_val)), 4),
                "F1_optimal_oracle": round(float(f1[best_i]), 4),
            }
        # gaps (interleaved mean minus honest)
        for key in ["argmax_F1", "F1_at_FPR_0.01", "F1_at_FPR_0.05",
                    "F1_valopt", "F1_optimal_oracle"]:
            leaky = float(np.mean([rec[f"leaky_k{i}"][key] for i in (1, 2, 3)]))
            rec[key + "_gap"] = round(leaky - rec["honest"][key], 4)
        out["models"][m] = rec
        print(f"[model] {m:18s} gap: argmax={rec['argmax_F1_gap']:+.3f} "
              f"FPR1%={rec['F1_at_FPR_0.01_gap']:+.3f} "
              f"valopt={rec['F1_valopt_gap']:+.3f} oracle={rec['F1_optimal_oracle_gap']:+.3f}")

    # pooled over models (mean of per-model gaps)
    keys = ["argmax_F1", "F1_at_FPR_0.01", "F1_at_FPR_0.05", "F1_valopt", "F1_optimal_oracle"]
    pooled = {}
    for k in keys:
        pooled[k + "_gap"] = round(float(np.mean([out["models"][m][k + "_gap"]
                                                  for m in out["models"]])), 4)
    out["pooled"] = pooled
    (ROOT / "results/uniform/threshold_f1_gap.json").write_text(json.dumps(out, indent=2))
    print("\n[POOLED GAPS] " + json.dumps(pooled))
    print(f"[saved] results/uniform/threshold_f1_gap.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
