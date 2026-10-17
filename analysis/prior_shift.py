"""Class-prior shift diagnostic: quantify how much cross-regional degradation
is driven by the difference in change prevalence between training and test regions.

For each model/seed:
  1. Compute P(change) in train, val, and each test region
  2. Calibrate a per-class threshold on the validation zone (argmax F1ch)
  3. Apply the frozen threshold to test regions vs fixed 0.5
  4. Report F1ch / omission / commission under both thresholds

Output: results/prior_shift.json
"""
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
TEST_REGIONS = ["beilunhe", "zhanjiang"]  # dongzhai excluded: 2024 composite clear-fraction only 7-13% (< 0.6 bar used elsewhere)


def change_prevalence(change_arr):
    """Fraction of pixels that are change (class 1 or 2)."""
    return float(np.mean((change_arr == 1) | (change_arr == 2)))


def binary_f1_from_probs(probs_ch, label_ch, threshold):
    """Compute F1, omission, commission from change-class probability and binary label."""
    pred_ch = (probs_ch >= threshold).astype(np.int32)
    label_ch = label_ch.astype(np.int32)
    tp = np.sum((pred_ch == 1) & (label_ch == 1))
    fp = np.sum((pred_ch == 1) & (label_ch == 0))
    fn = np.sum((pred_ch == 0) & (label_ch == 1))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    omission = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    commission = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    return f1, omission, commission


def calibrate_threshold(probs_ch, label_ch, n_grid=50):
    """Find threshold that maximises F1ch on the given data."""
    thresholds = np.linspace(0.05, 0.95, n_grid)
    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        f1, _, _ = binary_f1_from_probs(probs_ch, label_ch, t)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t), float(best_f1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--out", type=str, default=str(RESULTS_ROOT / "prior_shift.json"))
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    # Load all regions
    print("Loading region data...")
    region_data = {}
    for rname in ["vietnam", "beilunhe", "zhanjiang"]:
        data = load_region(rname)
        region_data[rname] = data
        if rname == "vietnam":
            train_idx = np.asarray(load_split(rname, "train"))
            val_idx = np.asarray(load_split(rname, "val"))
            test_idx = np.asarray(load_split(rname, "test"))
        else:
            test_idx = np.asarray(load_split(rname, "test"))
        region_data[rname]["train_idx"] = train_idx if rname == "vietnam" else None
        region_data[rname]["val_idx"] = val_idx if rname == "vietnam" else None
        region_data[rname]["test_idx"] = test_idx
        print(f"  {rname}: {len(data['t1_feat'])} patches, "
              f"change prevalence (all) = {change_prevalence(data['change']):.4f}")

    # Per-region change prevalence
    prevalence = {}
    for rname, data in region_data.items():
        ch = data["change"]
        prevalence[rname] = {}
        if rname == "vietnam":
            prevalence[rname]["train"] = round(change_prevalence(ch[data["train_idx"]]), 4)
            prevalence[rname]["val"] = round(change_prevalence(ch[data["val_idx"]]), 4)
            prevalence[rname]["test"] = round(change_prevalence(ch[data["test_idx"]]), 4)
        else:
            prevalence[rname]["test"] = round(change_prevalence(ch[data["test_idx"]]), 4)

    print("\n=== Change Prevalence P(Y=change) ===")
    for rname, prev in prevalence.items():
        print(f"  {rname}: {prev}")

    # Per-model threshold calibration + evaluation
    results = {"prevalence": prevalence, "per_model": {}}
    for model in args.models:
        results["per_model"][model] = {}
        is_temporal = model in DUAL_INPUT_MODELS
        in_ch = 9 if is_temporal else 27

        for seed in args.seeds:
            ck = CKPT_DIR / f"{model}_s{seed}.pt"
            if not ck.exists():
                continue
            net = MODEL_REGISTRY[model](in_ch=in_ch, out_ch=4).to(device)
            net.load_state_dict(torch.load(ck, map_location="cpu"))
            net.eval()

            def get_probs(t1, t2):
                """Return per-pixel probability of change class (P(loss) + P(gain))."""
                with torch.no_grad():
                    if is_temporal:
                        a = torch.from_numpy((t1.transpose(0, 3, 1, 2) / 10000.0).copy())
                        b = torch.from_numpy((t2.transpose(0, 3, 1, 2) / 10000.0).copy())
                        logits = []
                        for i in range(0, len(a), 4):
                            logits.append(net(a[i:i+4].to(device), b[i:i+4].to(device)))
                    else:
                        x = np.concatenate([t1, t2, np.abs(t1 - t2)], axis=3)
                        xt = torch.from_numpy((x.transpose(0, 3, 1, 2) / 10000.0).copy())
                        logits = []
                        for i in range(0, len(xt), 8):
                            logits.append(net(xt[i:i+8].to(device)))
                logits = torch.cat(logits, dim=0).cpu().numpy()
                probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
                # P(change) = P(loss) + P(gain) = classes 1 + 2
                return probs[:, 1] + probs[:, 2]  # (N, H, W)

            seed_results = {}
            # Calibrate on validation zone
            v = region_data["vietnam"]
            val_probs = get_probs(v["t1_feat"][v["val_idx"]], v["t2_feat"][v["val_idx"]])
            val_label = ((v["change"][v["val_idx"]] == 1) | (v["change"][v["val_idx"]] == 2)).ravel()
            val_probs_flat = val_probs.ravel()
            tau_star, f1_star = calibrate_threshold(val_probs_flat, val_label)
            seed_results["tau_star_val"] = round(tau_star, 4)
            seed_results["f1ch_val_at_tau"] = round(f1_star, 4)
            seed_results["f1ch_val_at_0.5"] = round(binary_f1_from_probs(val_probs_flat, val_label, 0.5)[0], 4)

            # Evaluate on test regions
            for rname in TEST_REGIONS:
                r = region_data[rname]
                t1 = r["t1_feat"][r["test_idx"]]
                t2 = r["t2_feat"][r["test_idx"]]
                label_ch = ((r["change"][r["test_idx"]] == 1) | (r["change"][r["test_idx"]] == 2)).ravel()
                probs_flat = get_probs(t1, t2).ravel()

                f1_05, om_05, com_05 = binary_f1_from_probs(probs_flat, label_ch, 0.5)
                f1_ts, om_ts, com_ts = binary_f1_from_probs(probs_flat, label_ch, tau_star)

                seed_results[rname] = {
                    "F1ch_0.5": round(float(f1_05), 4),
                    "F1ch_tau": round(float(f1_ts), 4),
                    "delta_F1": round(float(f1_ts - f1_05), 4),
                    "omission_0.5": round(float(om_05), 4),
                    "omission_tau": round(float(om_ts), 4),
                    "commission_0.5": round(float(com_05), 4),
                    "commission_tau": round(float(com_ts), 4),
                }
            results["per_model"][model][f"s{seed}"] = seed_results
            del net
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Summary
    print("\n=== Prior Shift Summary ===")
    for model in args.models:
        model_data = results["per_model"][model]
        if not model_data:
            continue
        taus = [d["tau_star_val"] for d in model_data.values()]
        print(f"\n{model}:")
        print(f"  tau* (val): {np.mean(taus):.3f} ± {np.std(taus):.3f}")
        for rname in TEST_REGIONS:
            d05 = [d[rname]["F1ch_0.5"] for d in model_data.values() if rname in d]
            dts = [d[rname]["F1ch_tau"] for d in model_data.values() if rname in d]
            if d05:
                print(f"  {rname}: F1ch(0.5)={np.mean(d05):.4f}  F1ch(τ*)={np.mean(dts):.4f}  Δ={np.mean(dts)-np.mean(d05):+.4f}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
