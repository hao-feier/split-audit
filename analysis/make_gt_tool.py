"""Build the ground-truth interpretation workbench (gt_tool/).

Human-in-the-loop campaign (the independent reference of the paper):
2400 stratified sample points (800/region) must be visually interpreted for
mangrove PRESENCE at t1 (Dec 2019-Jan 2020) and t2 (Dec 2024-Jan 2025).

This script generates a self-contained web workbench:
  gt_tool/
    index.html   keyboard-driven labeling UI (progress in localStorage)
    points.json  all 2400 points + stratum + thumbnail paths + WGS84 coords
    thumbs/      48x48 (480 m) RGB context crops from the t1/t2 S2 composites
                 (B04,B03,B02; fixed per-region percentile stretch)

The interpreter judges presence from the high-resolution basemap (Google
Earth / Esri World Imagery via the per-point links; use Google Earth
HISTORICAL imagery for t1). The thumbnails are 10 m context, not the
evidence.

Usage:
  python analysis/make_gt_tool.py
  cd gt_tool && python3 -m http.server 8000   # then open http://localhost:8000
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.paths import PROCESSED_ROOT, RESULTS_ROOT
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
CROP = 48          # px (10 m) -> 480 m context window
RGB_BANDS = (2, 1, 0)   # 0-based: B04, B03, B02 in the 9-band composite


def mosaic_rgb(region, epoch):
    """(H, W, 3) true-color mosaic of the per-tile composites on the region grid."""
    import analysis.make_groundtruth_samples as gts
    return gts.mosaic_rgb(region, epoch)[0]


def percentiles(arr, n=200_000):
    a = arr.reshape(-1, 3)
    a = a[np.isfinite(a).all(axis=1)]
    idx = np.random.default_rng(0).choice(len(a), size=min(n, len(a)), replace=False)
    lo = np.percentile(a[idx], 1, axis=0)
    hi = np.percentile(a[idx], 99.5, axis=0)
    return lo, np.maximum(hi, lo + 1)


def crop_jpeg(crop, lo, hi, out_path):
    from PIL import Image
    c = np.clip((crop - lo) / (hi - lo), 0, 1)
    img = Image.fromarray((c * 255).astype(np.uint8))
    img.save(out_path, quality=75)


def main():
    out = ROOT / "gt_tool"
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    points = []

    for region in REGIONS:
        proc = PROCESSED_ROOT / region
        meta = json.loads((proc / "metadata.json").read_text())
        H, W = meta["grid"]
        epsg = meta["epsg"]
        transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

        moses, stretch = {}, {}
        for epoch in ["t1", "t2"]:
            print(f"mosaic {region}/{epoch} ...", flush=True)
            m = mosaic_rgb(region, epoch)
            moses[epoch] = m
            stretch[epoch] = percentiles(m)

        ox, oy = meta["grid_origin"]
        feats = json.loads((ROOT / "data" / "groundtruth" / f"{region}_samples.geojson").read_text())["features"]
        for f in feats:
            p = f["properties"]
            lon, lat = transformer.transform(p["x"], p["y"])
            col = int(round((p["x"] - ox) / 10.0))
            row = int(round((oy - p["y"]) / 10.0))
            rec = {
                "id": p["id"], "region": region,
                "gmw_class": p["gmw_class"], "gmw_class_name": p["gmw_class_name"],
                "x": p["x"], "y": p["y"], "lon": round(lon, 6), "lat": round(lat, 6),
                "t1_mng": p.get("t1_mng"), "t2_mng": p.get("t2_mng"),
            }
            for epoch in ["t1", "t2"]:
                r0 = max(0, row - CROP // 2); c0 = max(0, col - CROP // 2)
                r1 = min(H, r0 + CROP); c1 = min(W, c0 + CROP)
                crop = moses[epoch][r0:r1, c0:c1, RGB_BANDS]
                fn = f"thumbs/{p['id']}_{epoch}.jpg"
                crop_jpeg(crop, *stretch[epoch], out / fn)
                rec[fn.split("/")[1]] = fn
            points.append(rec)

    (out / "points.json").write_text(json.dumps(points, indent=1))
    print(f"wrote gt_tool/points.json ({len(points)} points) + thumbnails")
    print("index.html shipped with the tool; serve with: cd gt_tool && python3 -m http.server 8000")


if __name__ == "__main__":
    main()
