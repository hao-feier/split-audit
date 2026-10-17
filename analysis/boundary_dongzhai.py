"""Boundary-distance / reference-resolution analysis for all test regions
including Dongzhai Bay.

For each model (seed 2026) and each test region:
  1. Compute distance from every pixel to the reference mangrove fringe
     (boundary of the mangrove extent in either epoch)
  2. Split into strata: fringe (≤30 m), near (30–100 m), interior (>100 m)
  3. Report F1ch / omission / commission per stratum
  4. Report fraction of predicted change in each stratum

This extends the existing edge-band analysis (analysis/edge_band.py)
to the Dongzhai region and adds the interior/fringe split.

Output: results/boundary_dongzhai.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.config import REGIONS, PATCH_SIZE
from mangrove.datasets import load_region, load_split
from mangrove.models import MODEL_REGISTRY, pick_device, DUAL_INPUT_MODELS
from mangrove.paths import RESULTS_ROOT, PROCESSED_ROOT

MODELS = ["unet", "unet_cbam_msa", "fc_siam"]
SEED = 2026
CKPT_DIR = RESULTS_ROOT / "uniform" / "checkpoints"
TEST_REGIONS = ["beilunhe", "zhanjiang"]  # dongzhai excluded: 2024 composite clear-fraction only 7-13% (< 0.6 bar used elsewhere)

# Stratum boundaries (meters)
FRINGE_M = 30
NEAR_M = 100


def compute_fringe_distance(mng_t1, mng_t2, patch_h, patch_w):
    """Compute per-pixel distance to the mangrove fringe (boundary of either epoch).
    
    mng_t1, mng_t2: (N, H, W) uint8 arrays (1=mangrove, 0=not)
    Returns: (N, H, W) float32 distance in meters (at 10 m/px)
    """
    # Fringe = boundary of mangrove extent in either epoch
    # A pixel is on the fringe if it's mangrove in one epoch and not the other,
    # or if it's adjacent to such a pixel
    mng_union = ((mng_t1 == 1) | (mng_t2 == 1)).astype(np.uint8)
    
    # Distance to the boundary: for each pixel, distance to the nearest
    # pixel that is at the edge of the mangrove union
    # Edge = mangrove pixel adjacent to non-mangrove pixel
    from scipy.ndimage import binary_erosion, binary_dilation
    
    edge = np.zeros_like(mng_union)
    for i in range(mng_union.shape[0]):
        m = mng_union[i]
        eroded = binary_erosion(m)
        dilated = binary_dilation(m)
        edge[i] = ((m == 1) & (~eroded)).astype(np.uint8)  # mangrove pixels on the boundary
    
    # Distance transform: for each pixel, distance to the nearest edge pixel
    dist = np.zeros((mng_union.shape[0], patch_h, patch_w), dtype=np.float32)
    for i in range(mng_union.shape[0]):
        if edge[i].sum() == 0:
            # No edge in this patch (all mangrove or all non-mangrove)
            dist[i] = 9999.0
        else:
            # Distance to nearest edge pixel (in pixels)
            d = distance_transform_edt(1 - edge[i])
            dist[i] = d * 10.0  # convert to meters (10 m/px)
    
    return dist


def stratum_metrics(pred, label, dist, n_h, n_w):
    """Compute F1ch, omission, commission per stratum."""
    pred_ch = ((pred == 1) | (pred == 2)).ravel()
    label_ch = ((label == 1) | (label == 2)).ravel()
    dist_flat = dist.ravel()
    
    strata = {
        "fringe_30m": dist_flat <= FRINGE_M,
        "near_30_100m": (dist_flat > FRINGE_M) & (dist_flat <= NEAR_M),
        "interior_100m": dist_flat > NEAR_M,
    }
    
    results = {}
    for name, mask in strata.items():
        if mask.sum() == 0:
            results[name] = {"n_pixels": 0}
            continue
        p = pred_ch[mask]
        l = label_ch[mask]
        tp = np.sum((p == 1) & (l == 1))
        fp = np.sum((p == 1) & (l == 0))
        fn = np.sum((p == 0) & (l == 1))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        omission = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        commission = fp / (fp + tp) if (fp + tp) > 0 else 0.0
        
        # Fraction of predicted change in this stratum
        n_pred_ch = p.sum()
        frac_pred = n_pred_ch / pred_ch.sum() if pred_ch.sum() > 0 else 0.0
        
        results[name] = {
            "n_pixels": int(mask.sum()),
            "n_true_change": int(l.sum()),
            "n_pred_change": int(n_pred_ch),
            "F1ch": round(float(f1), 4),
            "omission": round(float(omission), 4),
            "commission": round(float(commission), 4),
            "frac_of_pred_change": round(float(frac_pred), 4),
        }
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--out", type=str, default=str(RESULTS_ROOT / "boundary_dongzhai.json"))
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    results = {"strata": {"fringe": f"≤{FRINGE_M} m", "near": f"{FRINGE_M}–{NEAR_M} m",
                          "interior": f">{NEAR_M} m"}, "per_model": {}}

    for model in args.models:
        results["per_model"][model] = {}
        ck = CKPT_DIR / f"{model}_s{SEED}.pt"
        if not ck.exists():
            print(f"skip {model} (no checkpoint)")
            continue

        is_temporal = model in DUAL_INPUT_MODELS
        in_ch = 9 if is_temporal else 27
        net = MODEL_REGISTRY[model](in_ch=in_ch, out_ch=4).to(device)
        net.load_state_dict(torch.load(ck, map_location="cpu"))
        net.eval()

        for rname in TEST_REGIONS:
            print(f"\n{model} / {rname}:")
            data = load_region(rname)
            idx = np.asarray(load_split(rname, "test"))
            
            t1 = data["t1_feat"][idx]
            t2 = data["t2_feat"][idx]
            change = data["change"][idx]
            
            # Get mangrove presence from labels (class 0 = stable mangrove, class 1 = loss)
            # t1 mangrove = stable_mangrove (0) + loss (1)
            # t2 mangrove = stable_mangrove (0) + gain (2)
            mng_t1 = ((change == 0) | (change == 1)).astype(np.uint8)
            mng_t2 = ((change == 0) | (change == 2)).astype(np.uint8)
            
            # Compute fringe distance
            n_patches = len(idx)
            dist = np.zeros((n_patches, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            for i in range(n_patches):
                edge_mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                m = (mng_t1[i] | mng_t2[i])
                from scipy.ndimage import binary_erosion
                eroded = binary_erosion(m)
                edge_mask = ((m == 1) & (~eroded)).astype(np.uint8)
                if edge_mask.sum() == 0:
                    dist[i] = 9999.0
                else:
                    dist[i] = distance_transform_edt(1 - edge_mask) * 10.0
            
            # Inference
            with torch.no_grad():
                if is_temporal:
                    a = torch.from_numpy((t1.transpose(0, 3, 1, 2) / 10000.0).copy())
                    b = torch.from_numpy((t2.transpose(0, 3, 1, 2) / 10000.0).copy())
                    preds = []
                    for i in range(0, len(a), 4):
                        preds.append(net(a[i:i+4].to(device), b[i:i+4].to(device)).argmax(1).cpu().numpy())
                else:
                    x = np.concatenate([t1, t2, np.abs(t1 - t2)], axis=3)
                    xt = torch.from_numpy((x.transpose(0, 3, 1, 2) / 10000.0).copy())
                    preds = []
                    for i in range(0, len(xt), 8):
                        preds.append(net(xt[i:i+8].to(device)).argmax(1).cpu().numpy())
            pred = np.concatenate(preds).astype(np.uint8)
            
            # Stratum metrics
            m = stratum_metrics(pred, change, dist, PATCH_SIZE, PATCH_SIZE)
            results["per_model"][model][rname] = m
            for sname, sm in m.items():
                if "n_pixels" in sm and sm["n_pixels"] > 0:
                    print(f"  {sname:16s}: F1ch={sm.get('F1ch', 'N/A')} "
                          f"omission={sm.get('omission', 'N/A')} "
                          f"commission={sm.get('commission', 'N/A')} "
                          f"frac_pred={sm.get('frac_of_pred_change', 'N/A')}")

        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
