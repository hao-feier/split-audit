"""Edge-band analysis: quantify how much of the model's
predicted change sits in the 10-30 m boundary band around the reference
mangrove fringe. If commission is boundary-driven, a large fraction of the
predicted change (and of the false change specifically) should lie within a
few 10 m pixels of the reference mangrove extent boundary.

Classes: 0 stable_mangrove, 1 loss, 2 gain, 3 stable_background.
Mangrove fringe = boundary of the reference mangrove extent (mangrove in
t1 OR t2 = labels in {0,1,2}).
"""
import glob, json, os, sys
import numpy as np
from scipy.ndimage import distance_transform_edt

# env override so the analysis can point at the uniform (or any) run set
ROOT = os.environ.get("MANGROVE_PREDICTIONS_ROOT", "results/predictions")
MODELS = {
    "unet_cbam_msa": "U-Net+CBAM+MSA (proposed)",
    "unet": "U-Net",
}
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
SEEDS = ["2026", "2027", "2028", "2029", "2030"]

def fringe_mask(label):
    mang = np.isin(label, [0, 1, 2])
    # boundary = mangrove pixel adjacent (4-conn) to a non-mangrove pixel
    from scipy.ndimage import binary_erosion
    er = binary_erosion(mang, iterations=1)
    boundary = mang & ~er
    return boundary

def band_frac(pred, label):
    pred_change = np.isin(pred, [1, 2])
    ref_change = np.isin(label, [1, 2])
    false_change = pred_change & ~ref_change          # commission pixels
    boundary = fringe_mask(label)
    # distance from every pixel to nearest boundary pixel (inf if none)
    dist = distance_transform_edt(~boundary)
    out = {}
    for name, mask in [("pred_change", pred_change), ("false_change", false_change)]:
        if mask.sum() == 0:
            out[name] = {"n": 0}
            continue
        d = dist[mask]
        rec = {"n": int(mask.sum())}
        for b in (1, 2, 3):
            rec[f"within_{b}px"] = float((d <= b).mean())
        out[name] = rec
    return out

def main():
    res = {}
    for mkey, mname in MODELS.items():
        for region in REGIONS:
            per_seed = []
            for s in SEEDS:
                pf = f"{ROOT}/{mkey}_s{s}_{region}.npy"
                lf = f"{ROOT}/{mkey}_s{s}_{region}_labels.npy"
                try:
                    pred = np.load(pf); lab = np.load(lf)
                except FileNotFoundError:
                    continue
                # per-patch then aggregate
                agg = {k: [] for k in ["pred_change", "false_change"]}
                for i in range(pred.shape[0]):
                    r = band_frac(pred[i], lab[i])
                    for k in agg:
                        if r.get(k, {}).get("n", 0) > 0:
                            agg[k].append(r[k])
                rec = {}
                for k in ["pred_change", "false_change"]:
                    if agg[k]:
                        n = int(np.mean([x["n"] for x in agg[k]]))
                        rec[k] = {
                            "n_mean": n,
                            "within_1px": float(np.mean([x["within_1px"] for x in agg[k]])),
                            "within_2px": float(np.mean([x["within_2px"] for x in agg[k]])),
                            "within_3px": float(np.mean([x["within_3px"] for x in agg[k]])),
                        }
                per_seed.append(rec)
            # aggregate across seeds
            if per_seed:
                out = {}
                for k in ["pred_change", "false_change"]:
                    have = [p[k] for p in per_seed if k in p]
                    if have:
                        out[k] = {
                            "n_mean": int(np.mean([h["n_mean"] for h in have])),
                            "within_1px": float(np.mean([h["within_1px"] for h in have])),
                            "within_2px": float(np.mean([h["within_2px"] for h in have])),
                            "within_3px": float(np.mean([h["within_3px"] for h in have])),
                        }
                res[f"{mname} | {region}"] = out
    print(json.dumps(res, indent=1))
    out_dir = os.environ.get("MANGROVE_RESULTS_ROOT", "results")
    with open(f"{out_dir}/edge_band.json", "w") as f:
        json.dump(res, f, indent=1)

if __name__ == "__main__":
    main()
