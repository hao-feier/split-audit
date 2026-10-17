"""Distance-response under the honest (spatially separated) protocol.

For every (model, seed) checkpoint of the five-model family, runs inference
on the 26 never-trained, non-validation Vietnam patches (10 buffer + 10
buffer-isolated test-zone patches) and computes the per-patch binary change
F1. Results are aggregated against each patch's minimum edge distance to the
40-patch training zone, giving an empirical F1_ch(d) curve per model.

Interpretation: if a model re-uses the spatial structure of the training
zone (the spatial shortcut), per-patch agreement with the reference is
expected to be higher for patches near the training zone and to decay with
distance. A flat curve would indicate no proximity effect remains under the
honest protocol. No retraining is involved; val-zone patches are excluded
(used for early stopping), so no patch evaluated here informed any model
selection.

Output: results/distance_response.json (+ fig_distance_response.pdf)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.datasets import ChangePatchDataset, load_region  # noqa: E402
from mangrove.models import MODEL_REGISTRY, DUAL_INPUT_MODELS  # noqa: E402
from mangrove.paths import RESULTS_ROOT, SPLITS_ROOT  # noqa: E402

MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]
SEEDS = [2026, 2027, 2028, 2029, 2030]
REGION = "vietnam"
BINS = [(0, 2.56), (2.56, 5.12), (5.12, 7.68), (7.68, 10.24), (10.24, 14.0)]


def box_m(i, patch_list):
    r, c = patch_list[i]
    return (r * 10.0, (r + 512) * 10.0, c * 10.0, (c + 512) * 10.0)


def min_edge_km(i, j, patch_list):
    a, b = box_m(i, patch_list), box_m(j, patch_list)
    dc = max(0.0, max(a[2], b[2]) - min(a[3], b[3]))
    dr = max(0.0, max(a[0], b[0]) - min(a[1], b[1]))
    return float(np.hypot(dc, dr)) / 1000.0


def f1ch(cm):
    chg = [1, 2]
    tp = cm[np.ix_(chg, chg)].sum()
    fn = cm[np.ix_(chg, [0, 3])].sum()
    fp = cm[np.ix_([0, 3], chg)].sum()
    return 2 * tp / max(2 * tp + fn + fp, 1e-12)


def cm4(y, p):
    cm = np.zeros((4, 4), dtype=np.int64)
    comb = y.ravel().astype(np.int64) * 4 + p.ravel().astype(np.int64)
    return np.bincount(comb, minlength=16).reshape(4, 4)


@torch.no_grad()
def main():
    device = torch.device("cpu")
    meta = json.loads((ROOT / "data" / "processed" / REGION / "metadata.json").read_text())
    patch_list = list(meta["patches"])
    train = json.loads((SPLITS_ROOT / "vietnam_train.json").read_text())
    val = json.loads((SPLITS_ROOT / "vietnam_val.json").read_text())
    test = json.loads((SPLITS_ROOT / "vietnam_test.json").read_text())
    buffer = json.loads((SPLITS_ROOT / "vietnam_buffer.json").read_text())
    eval_idx = sorted(set(test) | set(buffer))
    assert len(eval_idx) == 26 and not (set(eval_idx) & set(train) | set(eval_idx) & set(val))
    dists = {i: min(min_edge_km(i, t, patch_list) for t in train) for i in eval_idx}

    data = load_region(REGION)
    out = {"config": {
        "models": MODELS, "seeds": SEEDS,
        "eval_patches": f"{len(eval_idx)} = 10 test-zone + 16 buffer (val excluded: "
                        "used for early stopping)",
        "metric": "per-patch binary change F1 (loss+gain vs unchanged)",
        "distance": "min edge-to-edge km from the 40-patch training zone"},
        "patch_distances_km": {str(i): round(d, 2) for i, d in sorted(dists.items())},
        "per_model": {}}
    for m in MODELS:
        is_dual = m in DUAL_INPUT_MODELS
        concat = not is_dual
        ds = ChangePatchDataset(REGION, "test", augment=False, concat=concat,
                                cw_mode="sqrt")
        ds.idx = list(eval_idx)
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        in_ch = 9 if is_dual else 27
        per_patch = {i: [] for i in eval_idx}
        for s in SEEDS:
            ck = RESULTS_ROOT / "checkpoints" / f"{m}_s{s}.pt"
            if not ck.exists():
                continue
            net = MODEL_REGISTRY[m](in_ch=in_ch, out_ch=4).to(device)
            net.load_state_dict(torch.load(ck, map_location="cpu"))
            net.eval()
            for batch in dl:
                if is_dual:
                    x1, x2, y, idx = batch
                    p = net(x1.float() / 10000.0, x2.float() / 10000.0).argmax(1).cpu().numpy()
                else:
                    x, y, idx = batch
                    p = net(x.float() / 10000.0).argmax(1).cpu().numpy()
                for k, i in enumerate(idx.tolist()):
                    per_patch[i].append(f1ch(cm4(np.asarray(y[k]), np.asarray(p[k]))))
        agg = []
        for lo, hi in BINS:
            vals = [v for i, v in per_patch.items()
                    for v in [np.mean(per_patch[i])]
                    if lo <= dists[i] < hi and per_patch[i]]
            # pool over seeds and patches inside the bin
            allv = [v for i in eval_idx if lo <= dists[i] < hi and per_patch[i]
                    for v in per_patch[i]]
            if allv:
                agg.append({"bin_km": [lo, hi], "n_patches": int(sum(
                    1 for i in eval_idx if lo <= dists[i] < hi and per_patch[i])),
                    "n_seed_runs": int(len(allv)),
                    "F1ch_mean": round(float(np.mean(allv)), 3),
                    "F1ch_std": round(float(np.std(allv, ddof=1)), 3) if len(allv) > 1 else 0.0,
                    "per_patch": {str(i): round(float(np.mean(per_patch[i])), 3)
                                  for i in sorted(eval_idx)
                                  if lo <= dists[i] < hi and per_patch[i]}})
        out["per_model"][m] = agg
        print(f"{m}: " + " | ".join(
            f"[{a['bin_km'][0]}-{a['bin_km'][1]}] n={a['n_seed_runs']} "
            f"{a['F1ch_mean']:.3f}" for a in agg), flush=True)
    (RESULTS_ROOT / "distance_response.json").write_text(json.dumps(out, indent=2))
    print("wrote results/distance_response.json")


if __name__ == "__main__":
    main()
