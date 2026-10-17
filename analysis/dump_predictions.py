"""Dump per-patch argmax predictions for bootstrap CI.

For every (model, seed) checkpoint and every evaluation region, runs
inference on the test patches and stores the per-patch prediction array
(results/predictions/<model>_s<seed>_<region>.npy, shape (n_patches, 512, 512),
uint8 class ids). Labels come from data/processed/<region>/change_labels.npz.

Run AFTER the training batch (GPU contention).
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
from mangrove.paths import PROCESSED_ROOT, RESULTS_ROOT

MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "unet_msa", "deeplabv3plus",
          "temporal_unet", "temporal_unet_gn", "fc_siam"]
REGIONS_EVAL = {"vietnam": "test", "beilunhe": "test", "zhanjiang": "test"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[2026, 2027, 2028])
    args = ap.parse_args()
    device = pick_device()
    out_dir = RESULTS_ROOT / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for seed in args.seeds:
            ck = RESULTS_ROOT / "checkpoints" / f"{model}_s{seed}.pt"
            if not ck.exists():
                print(f"skip {model} s{seed} (no checkpoint)")
                continue
            is_temporal = model in DUAL_INPUT_MODELS
            in_ch = 9 if is_temporal else 27
            net = MODEL_REGISTRY[model](in_ch=in_ch, out_ch=4).to(device)
            net.load_state_dict(torch.load(ck, map_location="cpu"))
            net.eval()
            for region, split in REGIONS_EVAL.items():
                data = load_region(region)
                idx = np.asarray(load_split(region, split))
                with torch.no_grad():
                    if is_temporal:
                        a = torch.from_numpy((data["t1_feat"].transpose(0, 3, 1, 2) / 10000.0).copy())
                        b = torch.from_numpy((data["t2_feat"].transpose(0, 3, 1, 2) / 10000.0).copy())
                        preds = []
                        for i in range(0, len(a), 8):
                            preds.append(net(a[i:i+8].to(device), b[i:i+8].to(device)).argmax(1).cpu().numpy())
                    else:
                        t1 = data["t1_feat"].transpose(0, 3, 1, 2)
                        t2 = data["t2_feat"].transpose(0, 3, 1, 2)
                        x = np.concatenate([t1, t2, np.abs(t1 - t2)], 1)
                        xt = torch.from_numpy((x / 10000.0).copy())
                        preds = []
                        for i in range(0, len(xt), 8):
                            preds.append(net(xt[i:i+8].to(device)).argmax(1).cpu().numpy())
                p = np.concatenate(preds)[idx].astype(np.uint8)
                y = data["change"][idx].astype(np.uint8)
                np.save(out_dir / f"{model}_s{seed}_{region}.npy", p)
                np.save(out_dir / f"{model}_s{seed}_{region}_labels.npy", y)
            print(f"done {model} s{seed}")


if __name__ == "__main__":
    main()
