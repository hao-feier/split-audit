"""Geometric audit of the 2x2 factorial claim: edge distance and footprint
overlap of the random test patches relative to the training zone.

Computes, for every random-arrangement test patch (R/R: split seeds 1-10;
I/R: draws 1-3), the minimum edge-to-edge distance to the training zone and
whether the 512-px patch footprint overlaps the footprint of at least one
training patch (512-px patches on the 256-px stride grid, 10 m/px).

Output: results/footprint_overlap.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PATCH, STEP = 512, 256
PX2KM = 0.010

def edge_dist_km(a, b):
    """Min edge-to-edge distance (km) between two (r0,c0) patch anchors."""
    ar, ac = a; br, bc = b
    dr = max(0, max(ar - (br + PATCH), br - (ar + PATCH)))
    dc = max(0, max(ac - (bc + PATCH), bc - (ac + PATCH)))
    return (dr**2 + dc**2) ** 0.5 * PX2KM

def overlaps(a, b):
    ar, ac = a; br, bc = b
    return (abs(ar - br) < PATCH) and (abs(ac - bc) < PATCH)

def main():
    meta = json.loads((ROOT / "data" / "processed" / "vietnam" / "metadata.json").read_text())
    patches = np.asarray(meta["patches"])
    train = np.asarray(json.loads((ROOT / "data" / "splits" / "vietnam_train.json").read_text()))
    train_pos = patches[train]

    out = {"config": {"patch_px": PATCH, "step_px": STEP}, "rr": {}, "ir": {}}
    for k in range(1, 11):
        tf = ROOT / "data" / "splits" / f"vietnam_train_rand{k}.json"
        f = ROOT / "data" / "splits" / f"vietnam_test_rand{k}.json"
        if not tf.exists() or not f.exists():
            continue
        # R/R training set = the per-split RANDOM training subset
        rtrain = np.asarray(json.loads(tf.read_text()))
        rtrain_pos = patches[rtrain]
        idx = np.asarray(json.loads(f.read_text()))
        n0 = n_ov = 0
        for i in idx:
            d = min(edge_dist_km(patches[i], t) for t in rtrain_pos)
            n0 += d == 0.0
            n_ov += any(overlaps(patches[i], t) for t in rtrain_pos)
        out["rr"][str(k)] = {"n_patches": int(len(idx)),
                             "n_edge_distance_0": int(n0),
                             "frac_edge_distance_0": round(n0 / len(idx), 4),
                             "n_footprint_overlap": int(n_ov),
                             "frac_footprint_overlap": round(n_ov / len(idx), 4)}
    for k in range(1, 4):
        f = ROOT / "data" / "splits" / f"vietnam_test_ir{k}.json"
        if not f.exists():
            continue
        idx = np.asarray(json.loads(f.read_text()))
        n0 = n_ov = 0
        for i in idx:
            d = min(edge_dist_km(patches[i], t) for t in train_pos)
            n0 += d == 0.0
            n_ov += any(overlaps(patches[i], t) for t in train_pos)
        out["ir"][str(k)] = {"n_patches": int(len(idx)),
                             "n_edge_distance_0": int(n0),
                             "frac_edge_distance_0": round(n0 / len(idx), 4),
                             "n_footprint_overlap": int(n_ov),
                             "frac_footprint_overlap": round(n_ov / len(idx), 4)}

    (ROOT / "results" / "footprint_overlap.json").write_text(json.dumps(out, indent=1))
    tot0 = totd = tovo = totn = 0
    for k, v in out["rr"].items():
        totd += v["n_edge_distance_0"]; totn += v["n_patches"]
    for k, v in out["ir"].items():
        tovo += v["n_footprint_overlap"]; totn += 0
    allrr = [v["frac_edge_distance_0"] for v in out["rr"].values()]
    allovr = [v["frac_footprint_overlap"] for v in out["rr"].values()]
    allir = [v["n_edge_distance_0"] / max(v["n_patches"], 1) for v in out["ir"].values()]
    print(f"R/R: edge-distance-0 frac = {sum(v['n_edge_distance_0'] for v in out['rr'].values())}/{sum(v['n_patches'] for v in out['rr'].values())}")
    print(f"R/R: footprint-overlap frac = {sum(v['n_footprint_overlap'] for v in out['rr'].values())}/{sum(v['n_patches'] for v in out['rr'].values())}")
    print(f"I/R: edge-distance-0 counts per draw = {[v['n_edge_distance_0'] for v in out['ir'].values()]}")
    print("saved results/footprint_overlap.json")

if __name__ == "__main__":
    main()
