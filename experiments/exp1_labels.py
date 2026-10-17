"""Exp1c: GMW -> 10 m presence labels + change labels (UTM grid).

- GMW v4.1.12 timeseries (30m, 4326 VRT) -> UTM 10m presence masks (nearest)
- change labels: 0 stable mangrove / 1 loss / 2 gain / 3 stable background
- independent source: GMW 2020 S2-based layer (native 10m, 4326 tiles) -> UTM 10m
All outputs on one UTM 10m grid per region (same transform -> aligned).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mangrove.config import REGIONS, EPOCHS, GMW_MANGROVE
from mangrove.paths import GMW_ROOT, PROCESSED_ROOT

STACK = GMW_ROOT / "gmw_v4112_stack" / "gmw_mng_ext_v4112_stack.vrt"
BAND_T1 = EPOCHS["t1"]["year"] - 1985 + 1
BAND_T2 = EPOCHS["t2"]["year"] - 1985 + 1


def make_grid(cfg):
    epsg = 32600 + int((cfg["center"][0] + 180) // 6) + 1
    trans = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x1, y1 = trans.transform(cfg["box"][0], cfg["box"][1])
    x2, y2 = trans.transform(cfg["box"][2], cfg["box"][3])
    gx0 = 10 * math.floor(min(x1, x2) / 10)
    y_bottom = 10 * math.floor(min(y1, y2) / 10)
    y_top = max(y1, y2)
    W = int(math.ceil((max(x1, x2) - gx0) / 10))
    H = int(math.ceil((y_top - y_bottom) / 10))
    gy0 = y_bottom + H * 10  # grid top-left corner (northing)
    return (H, W), from_origin(gx0, gy0, 10.0, 10.0), epsg, trans


def read_reproject(src, band, grid, transform, epsg):
    """Read one band from a 4326 source, reproject to the target UTM 10m grid."""
    arr = np.zeros(grid, dtype="uint8")
    reproject(source=rasterio.band(src, band), destination=arr,
              src_transform=src.transform, src_crs="EPSG:4326",
              dst_transform=transform, dst_crs=f"EPSG:{epsg}",
              resampling=Resampling.nearest, src_nodata=0)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    args = ap.parse_args()
    cfg = REGIONS[args.region]
    grid, transform, epsg, trans = make_grid(cfg)
    out_dir = PROCESSED_ROOT / args.region
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(arr, name):
        with rasterio.open(out_dir / name, "w", driver="GTiff", height=grid[0],
                           width=grid[1], count=1, dtype="uint8",
                           crs=f"EPSG:{epsg}", transform=transform) as dst:
            dst.write(arr.astype(np.uint8), 1)

    # 1) GMW timeseries 30m -> UTM 10m
    with rasterio.open(STACK) as src:
        m1 = read_reproject(src, BAND_T1, grid, transform, epsg)
        m2 = read_reproject(src, BAND_T2, grid, transform, epsg)
    m1 = (m1 == GMW_MANGROVE).astype(np.int8)
    m2 = (m2 == GMW_MANGROVE).astype(np.int8)
    change = np.full(grid, 3, dtype=np.int8)
    change[(m1 == 1) & (m2 == 1)] = 0
    change[(m1 == 1) & (m2 == 0)] = 1
    change[(m1 == 0) & (m2 == 1)] = 2

    save(m1, "labels_t1.tif")
    save(m2, "labels_t2.tif")
    save(change, "change_labels.tif")

    stats = {
        "mangrove_t1_km2": round(float(m1.sum() * 1e-4), 1),
        "mangrove_t2_km2": round(float(m2.sum() * 1e-4), 1),
        "loss_km2": round(float((change == 1).sum() * 1e-4), 1),
        "gain_km2": round(float((change == 2).sum() * 1e-4), 1),
        "stable_mangrove_km2": round(float((change == 0).sum() * 1e-4), 1),
    }

    # 2) independent source: GMW 2020 S2-based (native 10m tiles, 4326)
    d2020 = GMW_ROOT / "gmw_2020_v4019"
    if d2020.exists():
        m2020 = np.zeros(grid, dtype=np.int8)
        for lat_top in range(math.ceil(cfg["box"][3]), math.ceil(cfg["box"][1]) - 1, -1):
            for lon_left in range(math.floor(cfg["box"][0]), math.floor(cfg["box"][2]) + 1):
                tpath = d2020 / f"GMW_N{lat_top:02d}E{lon_left:03d}_v4019_mng.tif"
                if not tpath.exists():
                    continue
                with rasterio.open(tpath) as src:
                    m2020 = m2020 | read_reproject(src, 1, grid, transform, epsg)
        save(m2020, "gmw2020_s2.tif")
        stats["gmw2020_s2_mangrove_km2"] = round(float(m2020.sum() * 1e-4), 1)

    (out_dir / "label_stats.json").write_text(json.dumps(stats, indent=2))
    print(stats)
    print("labels saved to", out_dir)


if __name__ == "__main__":
    main()
