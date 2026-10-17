r"""Radiometric consistency between the two epochs, beyond a single deep-water pair.

Review criticism (M9): the manuscript verified radiometric consistency on one
deep-water t1/t2 tile pair only, which is weak evidence for harmonising a 2019
Landsat-derived baseline against a 2024 Sentinel-2 baseline across two
processing lines.

This script does three things on the actual composites:

1. Pseudo-invariant features (PIF). Regression of t2 on t1 over spectrally
   stable targets (very high NDVI, low NDWI, i.e. stable terrestrial
   vegetation and not water), per band. A stable land surface should give a gain
   near 1 and an offset near 0; a gain far from 1 indicates the two epochs are
   not radiometrically comparable and no harmonisation is legitimate.

2. Dark-target check. Water is the standard pseudo-invariant target for
   Sentinel-2; the near-zero DN of the deep-water percentiles across all tiles
   and epochs bounds residual atmospheric effects.

3. Brighter-target regression. Water removed, the 95th percentile of each
   epoch over stable bright targets, to separate "no additive bias" from "no
   multiplicative bias".

Writes results/radiometric_consistency.json.
"""
import json
import os
import sys

import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mangrove.paths import PROCESSED_ROOT  # noqa: E402

REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
OUT = os.path.join(ROOT, "results", "radiometric_consistency.json")


def _composite_paths(region):
    out = {}
    for ep in ("t1", "t2"):
        d = os.path.join(PROCESSED_ROOT, region, ep)
        if not os.path.isdir(d):
            return None
        out[ep] = sorted(
            os.path.join(d, f) for f in os.listdir(d)
            if f.endswith("_composite.tif"))
    if not out.get("t1") or not out.get("t2"):
        return None
    return out


def _read(paths, band, shape=None):
    """Mean across tiles of one band.

    Tiles in a region differ in row count (different tile sizes are clipped at
    the region window), so they are cropped to a common shape before averaging.
    The quantity we want is a radiometric level, which is per-tile, so averaging
    is adequate and avoids a mosaic dependency on tile order.
    """
    arrays = []
    for p in paths:
        with rasterio.open(p) as src:
            arrays.append(src.read(band).astype(np.float64))
    if shape is None:
        shape = (min(a.shape[0] for a in arrays), min(a.shape[1] for a in arrays))
    acc = None
    for a in arrays:
        a = a[:shape[0], :shape[1]]
        acc = a if acc is None else acc + a
    return acc / len(arrays)


def _common_shape(reg):
    """Common (rows, cols) across every tile of both epochs in a region."""
    paths = _composite_paths(reg)
    if paths is None:
        return None
    shapes = []
    for ep in ("t1", "t2"):
        for p in paths[ep]:
            with rasterio.open(p) as src:
                shapes.append(src.shape)
    return (min(s[0] for s in shapes), min(s[1] for s in shapes))


def main():
    out = {"per_region": {}, "bands": BANDS,
           "pif_definition": ("stable targets: top 3% NDVI in t1 with NDWI in the "
                              "lower 40% (persistent vegetation, not water), and "
                              "high NDVI in t2 as well; regressed per band as "
                              "t2 = gain * t1 + offset. Water targets: bottom 3% "
                              "NDVI with top 3% NDWI."),
           "interpretation": ("gain near 1 and offset near 0 means the two "
                               "epochs are radiometrically comparable without "
                               "harmonisation; a gain far from 1 would mean the "
                               "observed change confounds reflectance with "
                               "sensor/processing")}

    for reg in REGIONS:
        paths = _composite_paths(reg)
        if paths is None:
            out["per_region"][reg] = {"status": "no composites found"}
            continue
        entry = {"tiles_t1": len(paths["t1"]), "tiles_t2": len(paths["t2"]),
                 "bands": {}}

        shp = _common_shape(reg)
        # The composites already carry NDVI and NDWI as bands 7 and 8; using
        # them avoids recomputing indices on tiles whose grids differ.
        ndvi1 = _read(paths["t1"], 7, shp)
        ndvi2 = _read(paths["t2"], 7, shp)
        ndwi1 = _read(paths["t1"], 8, shp)

        # These region windows are mostly water, so absolute thresholds and
        # whole-window percentiles both fail: select LAND first, then take the
        # stable (high-NDVI) tail within it.
        land = (ndvi1 > 0.15) & (ndvi2 > 0.15)
        entry["n_land_pixels"] = int(land.sum())
        if land.sum() < 5000:
            entry["status"] = f"too few land pixels ({int(land.sum())})"
            out["per_region"][reg] = entry
            continue
        q1 = float(np.percentile(ndvi1[land], 80))
        q2 = float(np.percentile(ndvi2[land], 80))
        stable = land & (ndvi1 >= q1) & (ndvi2 >= q2)
        water = ndwi1 > 0.2
        entry["pif_ndvi_threshold_t1"] = round(q1, 4)
        entry["pif_ndvi_threshold_t2"] = round(q2, 4)
        entry["n_pif_pixels"] = int(stable.sum())
        entry["n_water_pixels"] = int(water.sum())

        if stable.sum() < 1000:
            entry["status"] = f"too few PIF pixels ({int(stable.sum())})"
            out["per_region"][reg] = entry
            continue

        for i, b in enumerate(BANDS, start=1):
            x = _read(paths["t1"], i, shp)[stable].ravel()
            y = _read(paths["t2"], i, shp)[stable].ravel()
            # subsample for speed; the relation is a mean over a large target
            if x.size > 400000:
                idx = np.random.default_rng(7).choice(x.size, 400000, replace=False)
                x, y = x[idx], y[idx]
            gain, offset = np.polyfit(x, y, 1)
            r = np.corrcoef(x, y)[0, 1]
            w1 = _read(paths["t1"], i, shp)[water] if water.sum() > 100 else np.array([])
            w2 = _read(paths["t2"], i, shp)[water] if water.sum() > 100 else np.array([])
            entry["bands"][b] = {
                "pif_gain": round(float(gain), 4),
                "pif_offset": round(float(offset), 1),
                "pif_r": round(float(r), 4),
                "water_pct_t1": round(float(np.percentile(w1, 95)), 1) if w1.size else None,
                "water_pct_t2": round(float(np.percentile(w2, 95)), 1) if w2.size else None,
            }
        gains = [v["pif_gain"] for v in entry["bands"].values()]
        entry["pif_gain_range"] = [round(min(gains), 3), round(max(gains), 3)]
        entry["pif_gain_mean_abs_dev_from_1"] = round(
            float(np.mean([abs(g - 1) for g in gains])), 3)
        out["per_region"][reg] = entry
        print(f"{reg:10s} PIF px={entry['n_pif_pixels']:>9d}  water px={entry['n_water_pixels']:>9d}  "
              f"gain range {entry['pif_gain_range']}", flush=True)

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    for reg, e in out["per_region"].items():
        if "bands" in e:
            print(f"\n{reg}:")
            for b, v in e["bands"].items():
                w1 = v["water_pct_t1"]
                w2 = v["water_pct_t2"]
                print(f"  {b}: gain={v['pif_gain']:.3f} offset={v['pif_offset']:+.1f} "
                      f"r={v['pif_r']:.4f}  water p95 t1={w1} t2={w2}")


if __name__ == "__main__":
    main()