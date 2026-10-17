"""Val-test spatial proximity audit for the leaky (R/R) arm.

For each split draw k (1-10), computes the minimum edge-to-edge distance
from every test patch to the nearest validation patch, and whether the
512-px test-patch footprint overlaps the footprint of at least one
validation patch.

This quantifies how spatially proximate the validation set is to the
test set in the interleaved arm, where both are drawn from the same
pool of 60 eligible patches.

Output: results/val_test_proximity.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PATCH, STEP = 512, 256
PX2KM = 0.010


def edge_dist_km(a, b):
    ar, ac = a
    br, bc = b
    dr = max(0, max(ar - (br + PATCH), br - (ar + PATCH)))
    dc = max(0, max(ac - (bc + PATCH), bc - (ac + PATCH)))
    return (dr**2 + dc**2) ** 0.5 * PX2KM


def overlaps(a, b):
    return abs(a[0] - b[0]) < PATCH and abs(a[1] - b[1]) < PATCH


def main():
    meta = json.loads(
        (ROOT / "data" / "processed" / "vietnam" / "metadata.json").read_text()
    )
    patches = np.asarray(meta["patches"])

    out = {
        "config": {"patch_px": PATCH, "step_px": STEP, "region": "vietnam"},
        "rr": {},
    }
    all_d = []
    all_ov = 0
    all_n = 0

    for k in range(1, 11):
        val_f = ROOT / "data" / "splits" / f"vietnam_val_rand{k}.json"
        test_f = ROOT / "data" / "splits" / f"vietnam_test_rand{k}.json"
        if not val_f.exists() or not test_f.exists():
            continue
        val_idx = np.asarray(json.loads(val_f.read_text()))
        test_idx = np.asarray(json.loads(test_f.read_text()))
        val_pos = patches[val_idx]

        dmin_list = []
        n_ov = 0
        for i in test_idx:
            d = min(edge_dist_km(patches[i], v) for v in val_pos)
            dmin_list.append(round(float(d), 2))
            if any(overlaps(patches[i], v) for v in val_pos):
                n_ov += 1
            all_d.append(d)
            all_ov += int(any(overlaps(patches[i], v) for v in val_pos))
            all_n += 1

        out["rr"][str(k)] = {
            "n_test": int(len(test_idx)),
            "n_overlap": int(n_ov),
            "frac_overlap": round(n_ov / len(test_idx), 3),
            "median_min_dist_km": round(float(np.median(dmin_list)), 2),
            "min_dist_km": round(float(min(dmin_list)), 2),
        }

    out["aggregate"] = {
        "n_test": all_n,
        "n_overlap": all_ov,
        "frac_overlap": round(all_ov / all_n, 3),
        "median_min_dist_km": round(float(np.median(all_d)), 2),
        "frac_dist_zero": round(sum(1 for d in all_d if d == 0) / all_n, 3),
    }

    out_path = ROOT / "results" / "val_test_proximity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved to {out_path.relative_to(ROOT)}")
    print(json.dumps(out["aggregate"], indent=2))


if __name__ == "__main__":
    main()
