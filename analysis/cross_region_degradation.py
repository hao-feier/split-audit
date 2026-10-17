"""PLAN 7: quantify cross-region generalization drop.

Per model: within-region test (vietnam buffer test zone) vs independent
regions (beilunhe, zhanjiang). Outputs results/cross_region.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mangrove.config import CHANGE_CLASSES
from mangrove.datasets import load_region, load_split
from mangrove.metrics import cd_metrics
from mangrove.models import MODEL_REGISTRY, pick_device
from mangrove.paths import RESULTS_ROOT, FIGURES_ROOT, ckpt_path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEMPORAL_MODELS = {"temporal_unet"}
MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]


def eval_model(model_name, region, split, seed=2026):
    ck = ckpt_path(model_name, seed)
    if not ck.exists():
        return None
    data = load_region(region)
    idx = np.asarray(load_split(region, split))
    if not len(idx):
        return None
    t1 = np.ascontiguousarray(np.transpose(data["t1_feat"][idx] / 10000.0, (0, 3, 1, 2)))
    t2 = np.ascontiguousarray(np.transpose(data["t2_feat"][idx] / 10000.0, (0, 3, 1, 2)))
    y = data["change"][idx]
    is_temporal = model_name in TEMPORAL_MODELS
    in_ch = 9 if is_temporal else 27
    device = pick_device()
    model = MODEL_REGISTRY[model_name](in_ch=in_ch, out_ch=4).to(device)
    model.load_state_dict(torch.load(ck, map_location="cpu"))
    model.eval()
    preds = []
    with torch.no_grad():
        bs = 8
        if is_temporal:
            for i in range(0, len(idx), bs):
                a = torch.from_numpy(t1[i:i + bs])
                b = torch.from_numpy(t2[i:i + bs])
                preds.append(model(a, b).argmax(1).numpy())
        else:
            x = np.concatenate([t1, t2, np.abs(t1 - t2)], 1)
            for i in range(0, len(x), bs):
                preds.append(model(torch.from_numpy(x[i:i + bs])).argmax(1).numpy())
    p = np.concatenate(preds)
    return cd_metrics(y, p, 4, CHANGE_CLASSES)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = {}
    for m in MODELS:
        ck = ckpt_path(m, args.seed)
        if not ck.exists():
            continue
        row = {}
        for region, split in (("vietnam", "test"), ("beilunhe", "test"), ("zhanjiang", "test")):
            r = eval_model(m, region, split, args.seed)
            if r:
                row[region] = {k: r[k] for k in ("OA", "F1_macro", "mIoU", "kappa", "n_pixels")}
        out[m] = row

    (RESULTS_ROOT / f"cross_region_s{args.seed}.json").write_text(json.dumps(out, indent=2))

    # bar plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = list(out)
    xs = np.arange(len(names))
    width = 0.25
    for k, (region, color) in enumerate(zip(("vietnam", "beilunhe", "zhanjiang"),
                                            ("green", "blue", "red"))):
        vals = [out[m][region]["OA"] if region in out[m] else 0 for m in names]
        ax.bar(xs + k * width, vals, width, label=f"{region} (test)")
    ax.set_xticks(xs + width)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("OA")
    ax.set_title("Cross-region generalization (train: vietnam)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / f"fig7_cross_region_s{args.seed}.png", dpi=200)
    print(json.dumps(out, indent=2))
    print("fig6 saved")


if __name__ == "__main__":
    main()
