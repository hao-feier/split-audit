"""Extract GMW v4.1.12 2020 layer (stack band 36) + three-way
source-vs-time agreement.

Pairs (presence masks, per region):
  A: v4.1.12 2019  vs v4.0.19 2020   (different source, 1 year apart)  [paper v1]
  B: v4.1.12 2020  vs v4.0.19 2020   (different source, SAME year)     [new]
  C: v4.1.12 2019  vs v4.1.12 2020   (same source, 1 year apart)       [new]

B vs C vs A separates the source difference from the time difference.

Outputs:
  data/processed/<region>/gmw2020_v4112.tif
  results/label_source_agreement.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.config import REGIONS, GMW_MANGROVE
from mangrove.paths import GMW_ROOT, PROCESSED_ROOT, RESULTS_ROOT

STACK = GMW_ROOT / "gmw_v4112_stack" / "gmw_mng_ext_v4112_stack.vrt"
BAND_2020 = 2020 - 1985 + 1  # = 36
BAND_2019 = 2019 - 1985 + 1  # = 35


def read_band(src, band, grid, transform, epsg):
    arr = np.zeros(grid, dtype="uint8")
    reproject(source=rasterio.band(src, band), destination=arr,
              src_transform=src.transform, src_crs="EPSG:4326",
              dst_transform=transform, dst_crs=f"EPSG:{epsg}",
              resampling=Resampling.nearest, src_nodata=0)
    return (arr == GMW_MANGROVE).astype(bool)


def pair_stats(a, b, mask=None):
    """Presence agreement / IoU / Jaccard of two boolean masks (over mask)."""
    if mask is None:
        mask = np.ones(a.shape, bool)
    n = int(mask.sum())
    inter = int((a & b & mask).sum())
    union = int((a | b & mask).sum())
    agree = int(((a == b) & mask).sum())
    return {
        "pixels": n,
        "agreement": agree / max(n, 1),
        "IoU": inter / max(union, 1),
        "area_a_km2": round(float((a & mask).sum() * 1e-4), 2),
        "area_b_km2": round(float((b & mask).sum() * 1e-4), 2),
        "symdiff_km2": round(float(((a ^ b) & mask).sum() * 1e-4), 2),
    }


def main():
    out = {}
    for region, cfg in REGIONS.items():
        meta = json.loads((PROCESSED_ROOT / region / "metadata.json").read_text())
        H, W = meta["grid"]
        transform = from_origin(meta["grid_origin"][0], meta["grid_origin"][1], 10.0, 10.0)
        epsg = meta["epsg"]
        grid = (H, W)

        with rasterio.open(STACK) as src:
            m2019 = read_band(src, BAND_2019, grid, transform, epsg)
            m2020 = read_band(src, BAND_2020, grid, transform, epsg)

        with rasterio.open(PROCESSED_ROOT / region / "gmw2020_s2.tif") as ds:
            s2020 = (ds.read(1) == 1).astype(bool)

        # save the new layer
        with rasterio.open(PROCESSED_ROOT / region / "gmw2020_v4112.tif", "w",
                           driver="GTiff", height=H, width=W, count=1,
                           dtype="uint8", crs=f"EPSG:{epsg}", transform=transform) as dst:
            dst.write(m2020.astype(np.uint8), 1)

        # patch-union domain: metrics over sampled patches are what the paper
        # evaluates; also report full-region for context
        union = np.zeros((H, W), bool)
        for (r, c) in meta["patches"]:
            union[r:r + 512, c:c + 512] = True

        out[region] = {
            "A_v4112_2019_vs_v4019_2020": {
                "region": pair_stats(m2019, s2020),
                "patch_union": pair_stats(m2019, s2020, mask=union),
            },
            "B_v4112_2020_vs_v4019_2020_same_year": {
                "region": pair_stats(m2020, s2020),
                "patch_union": pair_stats(m2020, s2020, mask=union),
            },
            "C_v4112_2019_vs_v4112_2020_same_source": {
                "region": pair_stats(m2019, m2020),
                "patch_union": pair_stats(m2019, m2020, mask=union),
            },
            "v4112_2020_mangrove_km2": round(float(m2020.sum() * 1e-4), 2),
        }
        print(region, "done")

    (RESULTS_ROOT / "label_source_agreement.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
