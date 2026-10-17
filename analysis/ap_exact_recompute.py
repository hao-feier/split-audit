"""Exact AP at target prevalence for k1/k2/k3 splits (90-run set).

For each (model, split_k, seed):
  - Inference on leaky patches (test_rand{k}) -> prob_leaky
  - Inference on honest patches (test)         -> prob_honest
  - AP_leaky_at_pi_honest  = exact PR curve at prevalence pi_honest
  - AP_honest_raw          = standard AP (pi = pi_honest already)
  - gap_exact = AP_leaky_at_pi_honest - AP_honest_raw

Compare to linear approx: AP_leaky_raw * (pi_h/pi_l) - AP_honest_raw.

Run: python3 analysis/ap_exact_recompute.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mangrove.datasets import load_region, load_split
from mangrove.models import MODEL_REGISTRY, DUAL_INPUT_MODELS

MODEL_SEEDS = {
    "unet": [2026, 2027, 2028],
    "unet_cbam": [2026, 2027, 2028],
    "unet_cbam_msa": [2026, 2027, 2028],
    "deeplabv3plus_pre": [2026, 2027, 2028],
    "deeplabv3plus_scratch": [2026, 2027, 2028],
    "unet3plus": [2026, 2027, 2028],
    "unet3plus_cbam": [2026, 2027, 2028],
    "changeformer": [2026, 2027, 2028],
    "segformer": [2026, 2027, 2028],
    "temporal_unet_gn": [2026, 2027, 2028],
}
SPLITS = ["k1", "k2", "k3"]


def exact_ap(prob, label, target_pi):
    order = np.argsort(-prob, kind="mergesort")
    y_sorted = (label[order] == 1).astype(int)
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0:
        return float("nan")
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg
    denom = tpr * target_pi + fpr * (1.0 - target_pi)
    precision = np.where(denom > 0, tpr * target_pi / np.maximum(denom, 1e-12), 0.0)
    idx = np.where(np.diff(tpr) != 0)[0]
    ap = 0.0
    for i in idx:
        w = tpr[i + 1] - tpr[i]
        ap += w * 0.5 * (precision[i] + precision[i + 1])
    return float(ap)


def standard_ap(prob, label):
    pi = (label == 1).mean()
    return exact_ap(prob, label, pi)


@torch.no_grad()
def get_probs(net, data, patch_idx, is_temporal):
    t1 = data["t1_feat"][patch_idx]
    t2 = data["t2_feat"][patch_idx]
    logits_parts = []
    if is_temporal:
        a = torch.from_numpy((t1.transpose(0, 3, 1, 2) / 10000.0).copy())
        b = torch.from_numpy((t2.transpose(0, 3, 1, 2) / 10000.0).copy())
        for i in range(0, len(a), 2):
            logits_parts.append(net(a[i:i+2], b[i:i+2]))
    else:
        x = np.concatenate([t1, t2, np.abs(t1 - t2)], axis=3)
        xt = torch.from_numpy((x.transpose(0, 3, 1, 2) / 10000.0).copy())
        for i in range(0, len(xt), 4):
            logits_parts.append(net(xt[i:i+4]))
    logits = torch.cat(logits_parts, dim=0).cpu().numpy()
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return (probs[:, 1] + probs[:, 2]).ravel()


def main():
    device = "cpu"
    data = load_region("vietnam")
    honest_idx = np.asarray(load_split("vietnam", "test"))
    h_labels = ((data["change"][honest_idx] == 1) | (data["change"][honest_idx] == 2)).ravel().astype(int)
    pi_h = float(h_labels.mean())
    print(f"[setup] pi_honest = {pi_h:.6f}, {len(honest_idx)} patches")

    results = []
    n_ok = n_skip = 0
    t0 = time.time()

    for split_k in SPLITS:
        leaky_idx = np.asarray(load_split("vietnam", f"test_rand{split_k[1:]}"))
        l_labels = ((data["change"][leaky_idx] == 1) | (data["change"][leaky_idx] == 2)).ravel().astype(int)
        pi_l = float(l_labels.mean())
        print(f"[split] {split_k}: pi_leaky = {pi_l:.6f}, ratio = {pi_l/pi_h:.2f}x")

        for model_name, seeds in MODEL_SEEDS.items():
            is_temporal = model_name in DUAL_INPUT_MODELS
            in_ch = 9 if is_temporal else 27
            for seed in seeds:
                ckpt = Path("results/uniform") / f"random_split_{split_k}" / "checkpoints" / f"{model_name}_s{seed}.pt"
                if not ckpt.exists():
                    n_skip += 1
                    continue
                net = MODEL_REGISTRY[model_name](in_ch=in_ch, out_ch=4).to(device)
                net.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
                net.eval()

                prob_l = get_probs(net, data, leaky_idx, is_temporal)
                prob_h = get_probs(net, data, honest_idx, is_temporal)

                ap_l_raw = standard_ap(prob_l, l_labels)
                ap_l_exact = exact_ap(prob_l, l_labels, pi_h)
                ap_h_raw = standard_ap(prob_h, h_labels)

                results.append({
                    "split": split_k, "model": model_name, "seed": seed,
                    "pi_honest": round(pi_h, 6), "pi_leaky": round(pi_l, 6),
                    "pi_ratio": round(pi_l / pi_h, 4),
                    "ap_leaky_raw": round(ap_l_raw, 6),
                    "ap_leaky_at_pi_honest": round(ap_l_exact, 6),
                    "ap_honest_raw": round(ap_h_raw, 6),
                    "gap_exact_pp": round(100 * (ap_l_exact - ap_h_raw), 3),
                    "gap_linear_pp": round(100 * (ap_l_raw * pi_h / pi_l - ap_h_raw), 3),
                })
                n_ok += 1
                del net
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"  [{split_k} done] {time.time()-t0:.0f}s elapsed")

    out = {
        "method": "exact PR-curve AP at target prevalence (k1/k2/k3, 90-run set)",
        "n_runs": n_ok, "n_skipped": n_skip,
        "results": results,
    }
    if results:
        ge = [r["gap_exact_pp"] for r in results]
        gl = [r["gap_linear_pp"] for r in results]
        out["summary"] = {
            "exact_gap_mean_pp": round(float(np.mean(ge)), 3),
            "exact_gap_min_pp": round(float(np.min(ge)), 3),
            "exact_gap_max_pp": round(float(np.max(ge)), 3),
            "n_positive_exact": sum(1 for g in ge if g > 0),
            "n_negative_exact": sum(1 for g in ge if g < 0),
            "linear_gap_mean_pp": round(float(np.mean(gl)), 3),
            "linear_gap_min_pp": round(float(np.min(gl)), 3),
            "linear_gap_max_pp": round(float(np.max(gl)), 3),
            "n_positive_linear": sum(1 for g in gl if g > 0),
        }

    out_path = Path("results/ap_exact_recompute_k.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[saved] {out_path}")
    print(f"[runs]  {n_ok} ok, {n_skip} skipped, {time.time()-t0:.0f}s total")
    if out.get("summary"):
        s = out["summary"]
        print(f"[exact]   gap = {s['exact_gap_mean_pp']:+.2f} pp "
              f"(range {s['exact_gap_min_pp']:+.2f} to {s['exact_gap_max_pp']:+.2f}, "
              f"pos {s['n_positive_exact']}/{n_ok}, neg {s['n_negative_exact']}/{n_ok})")
        print(f"[linear]  gap = {s['linear_gap_mean_pp']:+.2f} pp "
              f"(range {s['linear_gap_min_pp']:+.2f} to {s['linear_gap_max_pp']:+.2f}, "
              f"pos {s['n_positive_linear']}/{n_ok})")


if __name__ == "__main__":
    main()
