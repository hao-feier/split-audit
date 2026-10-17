"""Evaluate existing uniform-sweep checkpoints on the Dongzhai Bay test region.

Loads 3 representative models (U-Net, U-Net+CBAM+MSA, FC-Siam) x 5 seeds
from results/uniform/checkpoints/, runs inference on dongzhai test patches,
and computes per-patch and pooled metrics.

Output: results/dongzhai_metrics.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.config import REGIONS
from mangrove.datasets import load_region, load_split
from mangrove.models import MODEL_REGISTRY, pick_device, DUAL_INPUT_MODELS
from mangrove.paths import RESULTS_ROOT, PROCESSED_ROOT

MODELS = ["unet", "unet_cbam_msa", "fc_siam"]
SEEDS = [2026, 2027, 2028, 2029, 2030]
CKPT_DIR = RESULTS_ROOT / "uniform" / "checkpoints"


def compute_metrics(pred, label):
    """Compute OA, mIoU, F1ch, omission, commission, kappa from 4-class arrays."""
    pred = pred.astype(np.int32)
    label = label.astype(np.int32)
    n_cls = 4
    # Confusion matrix
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for i in range(n_cls):
        for j in range(n_cls):
            cm[i, j] = np.sum((label == i) & (pred == j))

    # OA
    oa = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0

    # Per-class IoU
    ious = []
    for i in range(n_cls):
        denom = cm[i, i] + cm[i, :].sum() + cm[:, i].sum() - cm[i, i]
        ious.append(cm[i, i] / denom if denom > 0 else 0.0)
    miou = float(np.mean(ious))

    # Cohen's kappa
    n = cm.sum()
    po = np.trace(cm) / n if n > 0 else 0
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n) if n > 0 else 0
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0

    # Binary change metrics (change = loss + gain = classes 1, 2)
    ch_label = ((label == 1) | (label == 2)).astype(np.int32)
    ch_pred = ((pred == 1) | (pred == 2)).astype(np.int32)
    tp = np.sum((ch_pred == 1) & (ch_label == 1))
    fp = np.sum((ch_pred == 1) & (ch_label == 0))
    fn = np.sum((ch_pred == 0) & (ch_label == 1))
    tn = np.sum((ch_pred == 0) & (ch_label == 0))
    f1ch = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iouch = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    omission = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    commission = fp / (fp + tp) if (fp + tp) > 0 else 0.0

    # Per-class F1
    f1s = []
    for i in range(n_cls):
        prec = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0.0
        rec = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)

    return {
        "OA": round(float(oa), 4),
        "mIoU": round(miou, 4),
        "kappa": round(float(kappa), 4),
        "F1ch": round(float(f1ch), 4),
        "IoUch": round(float(iouch), 4),
        "omission": round(float(omission), 4),
        "commission": round(float(commission), 4),
        "F1_per_class": [round(float(f), 4) for f in f1s],
        "IoU_per_class": [round(float(i), 4) for i in ious],
        "n_tp": int(tp), "n_fp": int(fp), "n_fn": int(fn), "n_tn": int(tn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--out", type=str, default=str(RESULTS_ROOT / "dongzhai_metrics.json"))
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    # Load dongzhai test data
    print("Loading dongzhai test patches...")
    data = load_region("dongzhai")
    idx = np.asarray(load_split("dongzhai", "test"))
    print(f"  {len(idx)} test patches, shape {data['t1_feat'].shape}")

    t1 = data["t1_feat"][idx]
    t2 = data["t2_feat"][idx]
    change = data["change"][idx]

    # Pre-compute inputs
    x_static = np.concatenate([t1, t2, np.abs(t1 - t2)], axis=3)  # (N, H, W, 27)
    x_static_t = torch.from_numpy((x_static.transpose(0, 3, 1, 2) / 10000.0).copy())
    a_t = torch.from_numpy((t1.transpose(0, 3, 1, 2) / 10000.0).copy())
    b_t = torch.from_numpy((t2.transpose(0, 3, 1, 2) / 10000.0).copy())

    results = {}
    for model in args.models:
        results[model] = {}
        is_temporal = model in DUAL_INPUT_MODELS
        in_ch = 9 if is_temporal else 27
        for seed in args.seeds:
            ck = CKPT_DIR / f"{model}_s{seed}.pt"
            if not ck.exists():
                print(f"  skip {model} s{seed} (no checkpoint)")
                continue
            net = MODEL_REGISTRY[model](in_ch=in_ch, out_ch=4).to(device)
            net.load_state_dict(torch.load(ck, map_location="cpu"))
            net.eval()
            with torch.no_grad():
                if is_temporal:
                    preds = []
                    for i in range(0, len(a_t), 4):
                        preds.append(net(a_t[i:i+4].to(device), b_t[i:i+4].to(device)).argmax(1).cpu().numpy())
                else:
                    preds = []
                    for i in range(0, len(x_static_t), 8):
                        preds.append(net(x_static_t[i:i+8].to(device)).argmax(1).cpu().numpy())
            pred = np.concatenate(preds).astype(np.uint8)
            m = compute_metrics(pred, change)
            results[model][f"s{seed}"] = m
            print(f"  {model} s{seed}: F1ch={m['F1ch']:.4f} mIoU={m['mIoU']:.4f} OA={m['OA']:.4f}")
            del net
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Pooled per-model stats
    print("\n=== Pooled (mean over seeds) ===")
    pooled = {}
    for model in args.models:
        seeds_data = [results[model][f"s{s}"] for s in args.seeds if f"s{s}" in results[model]]
        if not seeds_data:
            continue
        keys = ["OA", "mIoU", "kappa", "F1ch", "IoUch", "omission", "commission"]
        p = {}
        for k in keys:
            vals = [d[k] for d in seeds_data]
            p[k] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
        pooled[model] = p
        print(f"  {model}: F1ch={p['F1ch']['mean']:.4f}±{p['F1ch']['std']:.4f} "
              f"mIoU={p['mIoU']['mean']:.4f}±{p['mIoU']['std']:.4f} "
              f"OA={p['OA']['mean']:.4f}±{p['OA']['std']:.4f}")

    out = {"per_seed": results, "pooled": pooled,
           "n_patches": len(idx), "region": "dongzhai"}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
