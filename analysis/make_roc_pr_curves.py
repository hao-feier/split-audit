"""Compute ROC/PR curves + pAUC@FPR<=0.01 for honest vs leaky arms (inference only).

Uses unet_cbam_msa (representative model) on k1 split.
Output: results/roc_pr_curves.json  (scalar metrics + downsampled curves)

The figure (figures/fig5_roc_pr.pdf) is drawn from that JSON by
analysis/make_figures.py::fig5_roc_pr, so plotting needs no GPU/checkpoint.
Run: python3 analysis/make_roc_pr_curves.py
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

from mangrove.datasets import load_region, load_split
from mangrove.models import MODEL_REGISTRY, DUAL_INPUT_MODELS

N_CURVE_PTS = 201


@torch.no_grad()
def get_probs(net, data, patch_idx, is_temporal):
    t1 = data["t1_feat"][patch_idx]
    t2 = data["t2_feat"][patch_idx]
    parts = []
    if is_temporal:
        a = torch.from_numpy((t1.transpose(0, 3, 1, 2) / 10000.0).copy())
        b = torch.from_numpy((t2.transpose(0, 3, 1, 2) / 10000.0).copy())
        for i in range(0, len(a), 2):
            parts.append(net(a[i:i+2], b[i:i+2]))
    else:
        x = np.concatenate([t1, t2, np.abs(t1 - t2)], axis=3)
        xt = torch.from_numpy((x.transpose(0, 3, 1, 2) / 10000.0).copy())
        for i in range(0, len(xt), 4):
            parts.append(net(xt[i:i+4]))
    logits = torch.cat(parts, dim=0).cpu().numpy()
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return (probs[:, 1] + probs[:, 2]).ravel()


def roc_curve(prob, label):
    order = np.argsort(-prob, kind="mergesort")
    y = label[order].astype(int)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = np.concatenate([[0], tp / n_pos])
    fpr = np.concatenate([[0], fp / n_neg])
    return tpr, fpr


def pr_curve(prob, label):
    order = np.argsort(-prob, kind="mergesort")
    y = label[order].astype(int)
    n_pos = y.sum()
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return precision, recall


def p_auc_roc(tpr, fpr, fpr_max=0.01):
    mask = fpr <= fpr_max
    if mask.sum() < 2:
        return float("nan")
    fpr_m = fpr[mask]
    tpr_m = tpr[mask]
    idx = np.where(np.diff(fpr_m) != 0)[0]
    return float(sum((fpr_m[i+1]-fpr_m[i]) * 0.5*(tpr_m[i]+tpr_m[i+1]) for i in idx) / fpr_max)


def compute_auc(tpr, fpr):
    idx = np.where(np.diff(fpr) != 0)[0]
    return float(sum((fpr[i+1]-fpr[i]) * 0.5*(tpr[i]+tpr[i+1]) for i in idx))


def thin_curve(x, y, n_pts=N_CURVE_PTS, head=0.02, max_head=80):
    """Downsample a step curve to<= n_pts, keeping detail where x <= head.

    The ROC staircase is extremely dense at low FPR (many tied scores), so the
    head region is itself capped at max_head points; the point nearest the
    pAUC cut-off is always retained so the figure's FPR=0.01 marker is exact.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) <= n_pts:
        return x.tolist(), y.tolist()
    head_idx = np.where(x <= head)[0]
    if len(head_idx) > max_head:
        keep = np.unique(np.linspace(0, len(head_idx) - 1, max_head).round().astype(int))
        head_idx = head_idx[keep]
    cut = int(np.argmin(np.abs(x - 0.01)))          # pAUC boundary
    head_idx = np.unique(np.concatenate([head_idx, [cut]]))
    rest = np.arange(head_idx[-1] + 1 if len(head_idx) else 0, len(x))
    budget = max(n_pts - len(head_idx) - 1, 2)
    rest = rest[np.unique(np.linspace(0, len(rest) - 1, budget).round().astype(int))]
    idx = np.unique(np.concatenate([head_idx, rest]))
    return x[idx].round(6).tolist(), y[idx].round(6).tolist()


def main():
    t0 = time.time()
    data = load_region("vietnam")
    model_name = "unet_cbam_msa"
    seed = 2027
    is_temporal = model_name in DUAL_INPUT_MODELS
    in_ch = 9 if is_temporal else 27

    splits = {
        "honest": np.asarray(load_split("vietnam", "test")),
        "leaky_k1": np.asarray(load_split("vietnam", "test_rand1")),
        "leaky_k2": np.asarray(load_split("vietnam", "test_rand2")),
        "leaky_k3": np.asarray(load_split("vietnam", "test_rand3")),
    }
    labels = {}
    for name, idx in splits.items():
        labels[name] = ((data["change"][idx] == 1) | (data["change"][idx] == 2)).ravel().astype(int)

    ckpt = ROOT / "results/uniform/random_split_k1/checkpoints" / f"{model_name}_s{seed}.pt"
    print(f"[load] {ckpt.name}")
    net = MODEL_REGISTRY[model_name](in_ch=in_ch, out_ch=4)
    net.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
    net.eval()

    results = {}
    for name, idx in splits.items():
        prob = get_probs(net, data, idx, is_temporal)
        label = labels[name]
        pi = label.mean()

        tpr, fpr = roc_curve(prob, label)
        prec, rec = pr_curve(prob, label)
        auc = compute_auc(tpr, fpr)
        p_auc_01 = p_auc_roc(tpr, fpr, 0.01)

        results[name] = {
            "pi": round(float(pi), 6),
            "roc_auc": round(auc, 6),
            "pAUC_FPR_le_0.01": round(p_auc_01, 6),
            "tpr": tpr.tolist(),
            "fpr": fpr.tolist(),
            "precision": prec.tolist(),
            "recall": rec.tolist(),
        }
        print(f"  {name:10s} pi={pi:.4f} AUC={auc:.4f} pAUC@0.01={p_auc_01:.4f}")

    # Save JSON: scalar metrics + downsampled curves (so the figure is
    # reproducible from JSON alone, without re-running inference).
    out = {"model": model_name, "seed": seed,
           "n_curve_pts": N_CURVE_PTS,
           "splits": {}}
    for name, r in results.items():
        out["splits"][name] = {k: v for k, v in r.items()
                               if k not in ("tpr", "fpr", "precision", "recall")}
        f, t = thin_curve(r["fpr"], r["tpr"])
        out["splits"][name]["fpr"], out["splits"][name]["tpr"] = f, t
        rc, pr = thin_curve(r["recall"], r["precision"])
        out["splits"][name]["recall"], out["splits"][name]["precision"] = rc, pr

    out_path = ROOT / "results/roc_pr_curves.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[saved] {out_path}")
    print("[next] python3 analysis/make_figures.py   # draws figures/fig5_roc_pr.pdf")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
