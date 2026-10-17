"""Out-of-sample within-region threshold gain.

The in-sample number in the manuscript (tau* calibrated on the validation
zone, then evaluated on that same zone) is a selection gain. This script
computes the honest out-of-sample version: calibrate tau* on the Vietnam
validation zone, then apply the frozen tau* to the Vietnam in-region test
zone (a disjoint zone), and report the F1ch gain relative to the fixed 0.5
threshold. This is the number that supports (or refutes) the 'within-region
defect, not prior drift' argument.

Output: results/prior_shift_oos.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.datasets import load_region, load_split
from mangrove.models import MODEL_REGISTRY, pick_device, DUAL_INPUT_MODELS
from mangrove.paths import RESULTS_ROOT

MODELS = ["unet", "unet_cbam_msa", "fc_siam"]
SEEDS = [2026, 2027, 2028, 2029, 2030]
CKPT_DIR = RESULTS_ROOT / "uniform" / "checkpoints"


def binary_f1_from_probs(probs_ch, label_ch, threshold):
    pred_ch = (probs_ch >= threshold).astype(np.int32)
    label_ch = label_ch.astype(np.int32)
    tp = np.sum((pred_ch == 1) & (label_ch == 1))
    fp = np.sum((pred_ch == 1) & (label_ch == 0))
    fn = np.sum((pred_ch == 0) & (label_ch == 1))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return float(f1)


def calibrate_threshold(probs_ch, label_ch, n_grid=80):
    thresholds = np.linspace(0.02, 0.98, n_grid)
    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        f1 = binary_f1_from_probs(probs_ch, label_ch, t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def main():
    device = pick_device()
    print("Device:", device)

    v = load_region("vietnam")
    val_idx = np.asarray(load_split("vietnam", "val"))
    test_idx = np.asarray(load_split("vietnam", "test"))
    ch = v["change"]
    print(f"vietnam: val prevalence={np.mean((ch[val_idx]==1)|(ch[val_idx]==2)):.4f}, "
          f"test prevalence={np.mean((ch[test_idx]==1)|(ch[test_idx]==2)):.4f}")

    results = {}
    for model in MODELS:
        is_temporal = model in DUAL_INPUT_MODELS
        in_ch = 9 if is_temporal else 27
        results[model] = {}
        for seed in SEEDS:
            ck = CKPT_DIR / f"{model}_s{seed}.pt"
            if not ck.exists():
                continue
            net = MODEL_REGISTRY[model](in_ch=in_ch, out_ch=4).to(device)
            net.load_state_dict(torch.load(ck, map_location="cpu"))
            net.eval()

            def get_probs(t1, t2):
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
                    return probs[:, 1] + probs[:, 2]

            # calibrate on validation zone
            val_probs = get_probs(v["t1_feat"][val_idx], v["t2_feat"][val_idx]).ravel()
            val_label = ((ch[val_idx] == 1) | (ch[val_idx] == 2)).ravel()
            tau_star, f1_val_tau = calibrate_threshold(val_probs, val_label)
            f1_val_05 = binary_f1_from_probs(val_probs, val_label, 0.5)

            # evaluate frozen tau* on the DISJOINT in-region test zone
            test_probs = get_probs(v["t1_feat"][test_idx], v["t2_feat"][test_idx]).ravel()
            test_label = ((ch[test_idx] == 1) | (ch[test_idx] == 2)).ravel()
            f1_test_tau = binary_f1_from_probs(test_probs, test_label, tau_star)
            f1_test_05 = binary_f1_from_probs(test_probs, test_label, 0.5)

            results[model][f"s{seed}"] = {
                "tau_star_val": round(tau_star, 4),
                "in_sample_val_gain": round(f1_val_tau - f1_val_05, 4),
                "oos_test_gain": round(f1_test_tau - f1_test_05, 4),
                "f1ch_test_0.5": round(f1_test_05, 4),
                "f1ch_test_tau": round(f1_test_tau, 4),
            }
            del net
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # summary
    print("\n=== Out-of-sample within-region threshold gain ===")
    for model in MODELS:
        if not results[model]:
            continue
        ins = [d["in_sample_val_gain"] for d in results[model].values()]
        oos = [d["oos_test_gain"] for d in results[model].values()]
        taus = [d["tau_star_val"] for d in results[model].values()]
        print(f"  {model}: tau*={np.mean(taus):.3f}  "
              f"in-sample val gain={np.mean(ins):+.4f}  "
              f"OOS test-zone gain={np.mean(oos):+.4f}")

    out = ROOT / "results" / "prior_shift_oos.json"
    out.write_text(json.dumps(results, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
