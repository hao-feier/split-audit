"""Exp1b: S2 L2A scenes -> cloud-filtered median composite + spectral indices.

Multi-tile aware: scenes are grouped by S2 tile; each tile is composited
separately onto its own UTM grid, clipped to the region box.

Per (region, epoch, tile):
  data/processed/<region>/<epoch>/<tile>_composite.tif     (9 bands, 10 m UTM)
  data/processed/<region>/<epoch>/<tile>_clear_fraction.tif
  data/processed/<region>/<epoch>/cloud_stats_<tile>.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mangrove.config import REGIONS, EPOCHS
from mangrove.geo import win_bounds_utm
from mangrove.paths import S2_ROOT, PROCESSED_ROOT

# SCL surface classes kept as "clear" (standard S2 L2A scheme, all baselines):
# 1 saturated/defective, 2 dark, 3 shadow, 4 vegetation, 5 not-vegetation,
# 6 water, 7 unclassified. 8-12 = cloud family (low/med/high prob, cirrus,
# snow-ice) -> excluded. 0 = no data (excluded).
# Note: class 1 (saturated/defective) and 7 (unclassified) are retained; their
# contribution is negligible here because the dry-season window has low cloud
# and each tile-epoch is a per-pixel median over >=2 clear scenes, which
# suppresses isolated bad pixels.
# B4 strict-SCL sensitivity override: MANGROVE_CLEAR_SCL="4,5,6" keeps only the
# clean-surface classes (vegetation / not-vegetated / water) and masks the
# saturated/dark/shadow/unclassified pixels out of the composites.
_SCL_ENV = os.environ.get("MANGROVE_CLEAR_SCL")
CLEAR_SCL = {int(x) for x in _SCL_ENV.split(",")} if _SCL_ENV else {1, 2, 3, 4, 5, 6, 7}
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
RES10 = 10.0


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def scene_tiles(region, epoch):
    """Return {tile: [scene_dir, ...]} for downloaded scenes of region/epoch."""
    mani = json.loads((S2_ROOT / "manifest.json").read_text())
    out = defaultdict(list)
    for sc in mani.get(region, {}).get(epoch, []):
        d = S2_ROOT / sc["scene"]
        if all((d / f"{b}.tif").exists() for b in BANDS + ["SCL"]):
            out[sc["scene"].split("_")[1]].append(d)
    return dict(out)


def composite_tile(tile, scene_dirs, region_box, cfg, out_dir, epsg, trans):
    """Composite scenes for one S2 tile onto the tile's native UTM grid.

    Each tile may be in a different UTM zone (e.g. beilunhe spans zones 48/49),
    so the region box is re-projected to the tile's native CRS per scene, and
    the output composite inherits the tile's CRS. exp1_patch later reprojects all
    tile composites onto a single reference grid.
    """
    import gc

    comp_out = out_dir / f"{tile}_composite.tif"
    if comp_out.exists():
        print(f"  {tile}: already exists")
        return

    # Determine the tile's native CRS from the first scene's B02 band, and
    # build a 4326 -> native transformer for window computation.
    with rasterio.open(scene_dirs[0] / "B02.tif") as ref:
        tile_crs = ref.crs
    tile_trans = Transformer.from_crs("EPSG:4326", tile_crs, always_xy=True)
    x1, y1 = tile_trans.transform(region_box[0], region_box[1])
    x2, y2 = tile_trans.transform(region_box[2], region_box[3])
    # The composite array only covers the region window, so its top-left must be
    # the window origin, not the full-tile origin (window_transform).
    with rasterio.open(scene_dirs[0] / "B02.tif") as ref:
        win10_ref = win_bounds_utm(ref, x1, y1, x2, y2)
        out_transform = ref.window_transform(win10_ref)

    acc = []
    clear_fracs = []
    cloud_stats = {}
    win10 = None

    for d in scene_dirs:
        sid = d.name
        with rasterio.open(d / "B02.tif") as src:
            win10 = win_bounds_utm(src, x1, y1, x2, y2)
        h, w = win10.height, win10.width

        with rasterio.open(d / "SCL.tif") as src:
            win20 = win_bounds_utm(src, x1, y1, x2, y2)
            scl = src.read(1, window=win20).astype(np.int8)
            clear20 = np.isin(scl, list(CLEAR_SCL))
            clear10 = zoom(clear20.astype(np.float32), 2.0, order=0).astype(bool)[:h, :w]
            if clear10.shape[0] < h or clear10.shape[1] < w:
                p = np.zeros((h, w), dtype=bool)
                p[:clear10.shape[0], :clear10.shape[1]] = clear10
                clear10 = p
            cloud_stats[sid] = round(float(clear20.mean()), 4)
            clear_fracs.append(clear10)
            del scl, clear20

        stack = []
        for b in BANDS:
            with rasterio.open(d / f"{b}.tif") as src:
                if abs(src.res[0] - RES10) < 1e-6:
                    arr = src.read(1, window=win10).astype(np.float32)
                else:
                    win20b = win_bounds_utm(src, x1, y1, x2, y2)
                    arr20 = src.read(1, window=win20b).astype(np.float32)
                    arr = zoom(arr20, 2.0, order=1)[:h, :w]
                    if arr.shape[0] < h or arr.shape[1] < w:
                        p = np.zeros((h, w), dtype=np.float32)
                        p[:arr.shape[0], :arr.shape[1]] = arr
                        arr = p
                    del arr20
            stack.append(arr)
        stack = np.stack(stack, 0)
        acc.append(np.where(clear10[None], stack, np.nan))
        del stack, clear10

    # Per-band median to avoid a (6, S, H, W) copy + full-stack reduction peak.
    comp = np.empty((6,) + (h, w), dtype=np.float32)
    for i in range(6):
        a = np.stack([s[i] for s in acc], 0)
        comp[i] = np.nan_to_num(np.nanmedian(a, 0), nan=0.0)
        del a
    del acc
    b02, b03, b04, b08, b11, b12 = comp
    ndvi = (b08 - b04) / np.maximum(b08 + b04, 1.0)
    ndwi = (b03 - b08) / np.maximum(b03 + b08, 1.0)
    mndwi = (b03 - b11) / np.maximum(b03 + b11, 1.0)
    final = np.stack([b02, b03, b04, b08, b11, b12,
                      (ndvi + 1) * 5000.0, (ndwi + 1) * 5000.0, (mndwi + 1) * 5000.0], 0).astype(np.float32)
    cf = np.median(np.stack(clear_fracs), axis=0).astype(np.float32)

    # Output on the tile's native grid (CRS + transform).
    profile = dict(driver="GTiff", height=h, width=w, count=9, dtype="float32",
                   crs=tile_crs, transform=out_transform)
    with rasterio.open(comp_out, "w", **profile) as dst:
        for i in range(9):
            dst.write(final[i], i + 1)
    with rasterio.open(out_dir / f"{tile}_clear_fraction.tif", "w", driver="GTiff",
                       height=h, width=w, count=1, dtype="float32",
                       crs=tile_crs, transform=out_transform) as dst:
        dst.write(cf, 1)
    (out_dir / f"cloud_stats_{tile}.json").write_text(json.dumps(cloud_stats, indent=2))
    print(f"  {tile}: composite {final.shape} scenes={len(scene_dirs)}")
    del comp, final, cf, clear_fracs
    gc.collect()
    return cloud_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--epoch", required=True)
    args = ap.parse_args()

    cfg = REGIONS[args.region]
    epsg = utm_epsg(cfg["center"][0], cfg["center"][1])
    trans = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    out_dir = PROCESSED_ROOT / args.region / args.epoch
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = scene_tiles(args.region, args.epoch)
    if not tiles:
        print(f"no downloaded scenes for {args.region}/{args.epoch}")
        sys.exit(1)
    all_stats = {}
    for tile, dirs in sorted(tiles.items()):
        all_stats[tile] = composite_tile(tile, dirs, cfg["box"], cfg, out_dir, epsg, trans)
    (out_dir / "cloud_stats.json").write_text(json.dumps(all_stats, indent=2))
    print(f"done {args.region}/{args.epoch}: {len(tiles)} tiles")


if __name__ == "__main__":
    main()
