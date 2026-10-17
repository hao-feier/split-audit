"""Exp2: spatially independent split (PLAN section 3).

Strategy
--------
Cross-region:  vietnam = training region A; beilunhe + zhanjiang =
               spatially independent test regions B and C
               (1600 / 1100 km from the training region).
Within-region: vietnam patches are divided by column into
               [train zone | 2 km buffer | val/test zone];
               the val/test zone is further divided by row with a buffer.
               Buffer patches are used for neither training nor testing.

Outputs
-------
  data/splits/vietnam_{train,val,test}.json        (patch indices)
  data/splits/beilunhe_test.json                   (all patches)
  data/splits/zhanjiang_test.json                  (all patches)
  data/splits/vietnam_buffer.json                  (excluded indices)
  figures/fig1_spatial_split.png                   (map of zones)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mangrove.config import REGIONS, PATCH_SIZE, BUFFER_M
from mangrove.paths import PROCESSED_ROOT, SPLITS_ROOT, FIGURES_ROOT


def patch_centers(region):
    meta = json.loads((PROCESSED_ROOT / region / "metadata.json").read_text())
    idx, cols, rows = [], [], []
    for (r, c) in meta["patches"]:
        idx.append(len(idx))
        cols.append((c + PATCH_SIZE / 2) * 10.0)   # meters on 10m grid
        rows.append((r + PATCH_SIZE / 2) * 10.0)
    return np.asarray(cols), np.asarray(rows), len(idx)


def split_region(region):
    cols, rows, n = patch_centers(region)
    p55, p65 = np.percentile(cols, 55), np.percentile(cols, 65)
    in_train_zone = cols < p55
    in_buffer = (cols >= p55) & (cols <= p65)
    in_vt_zone = cols > p65

    vt = np.where(in_vt_zone)[0]
    vt_rows = rows[vt]
    med_row = np.median(vt_rows)
    buf = BUFFER_M
    val_mask = vt_rows < (med_row - buf)
    test_mask = vt_rows > (med_row + buf)

    train = np.where(in_train_zone)[0]
    val = vt[val_mask]
    test = vt[test_mask]
    buffer = np.where(in_buffer)[0]
    # row-buffer strips inside vt zone
    row_buffer = vt[~val_mask & ~test_mask]

    out = {
        "train": train.tolist(),
        "val": val.tolist(),
        "test": test.tolist(),
        "buffer": np.concatenate([buffer, row_buffer]).tolist(),
    }
    return out, (cols, rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=list(REGIONS.keys()))
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(args.regions), figsize=(8 * len(args.regions), 7))
    if len(args.regions) == 1:
        axes = [axes]

    for ax, region in zip(axes, args.regions):
        if region not in REGIONS or REGIONS[region]["role"] != "train":
            # independent test region: everything is test
            cols, rows, n = patch_centers(region)
            json.dump(list(range(n)), open(SPLITS_ROOT / f"{region}_test.json", "w"))
            ax.scatter(cols, rows, s=2, c="blue")
            ax.set_title(f"{region} (independent test region, n={n})")
            continue

        out, (cols, rows) = split_region(region)
        for name in ("train", "val", "test"):
            json.dump(out[name], open(SPLITS_ROOT / f"{region}_{name}.json", "w"))
        json.dump(out["buffer"], open(SPLITS_ROOT / f"{region}_buffer.json", "w"))

        colors = {"train": "green", "val": "orange", "test": "blue", "buffer": "gray"}
        for name, c in colors.items():
            if out[name]:
                ax.scatter(cols[out[name]], rows[out[name]], s=4, c=c, label=name)
        ax.set_title(f"{region} (train region; buffer={len(out['buffer'])})")
        ax.legend()
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    fig.suptitle("Spatially independent split (2 km buffer between zones)")
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "fig1_spatial_split.png", dpi=200)
    print("splits written to", SPLITS_ROOT)
    for region in args.regions:
        for name in ("train", "val", "test", "buffer"):
            p = SPLITS_ROOT / f"{region}_{name}.json"
            if p.exists():
                print(f"  {region}_{name}: {len(json.loads(p.read_text()))}")


if __name__ == "__main__":
    main()
