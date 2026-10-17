"""Exp1d: extract 512x512 patches from (multi-tile) composites + labels.

Builds a single UTM reference grid (10 m) over the region box, reprojects
all tile composites (bilinear) and label rasters (nearest) onto it, then
slidewindows patches with mangrove-fraction and clear-fraction filters.

Outputs per region:
  data/processed/<region>/t1/features.npz, t2/features.npz
  data/processed/<region>/change_labels.npz
  data/processed/<region>/metadata.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mangrove.config import REGIONS, PATCH_SIZE, PATCH_STEP, MIN_MANGROVE_FRAC, MAX_MANGROVE_FRAC
from mangrove.paths import PROCESSED_ROOT


def crs_of(path):
    with rasterio.open(path) as src:
        return src.crs


def ordered_tiles(region_dir, epoch, dst_crs, pattern="*_composite.tif"):
    """Tile rasters in stitch order: reference-zone tiles first, then name."""
    files = sorted(region_dir.glob(f"{epoch}/{pattern}"))
    return sorted(files, key=lambda f: (crs_of(f).to_epsg() != dst_crs, f.name))


def reproject_to_grid(src_path, grid, transform, dst_crs, bilinear=False):
    """Reproject a single-band raster onto the reference UTM grid.

    dst_crs: target CRS as a string (e.g. 'EPSG:32648') or CRS object.
    Uncovered destination pixels are NaN (bilinear) so that multi-tile
    stitching (`np.where(np.isnan(dst), tile, dst)`) lets later tiles fill
    the gaps of earlier ones.
    """
    arr = np.full((grid[0], grid[1]),
                  np.nan if bilinear else 0,
                  dtype="float32" if bilinear else "uint8")
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=arr,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=dst_crs,
            resampling=Resampling.bilinear if bilinear else Resampling.nearest,
            src_nodata=np.nan if bilinear else 0,
        )
    return arr


def reproject_multiband(src_path, grid, transform, dst_crs):
    """Reproject a 9-band composite onto the reference UTM grid (bilinear).

    Uncovered destination pixels are NaN (not 0) so multi-tile stitching
    can distinguish no-data from a real zero reflectance.
    """
    arr = np.full((grid[0], grid[1], 9), np.nan, dtype="float32")
    with rasterio.open(src_path) as src:
        for i in range(9):
            band = np.full((grid[0], grid[1]), np.nan, dtype="float32")
            reproject(
                source=rasterio.band(src, i + 1), destination=band,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
            )
            arr[:, :, i] = band
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--patch-size", type=int, default=PATCH_SIZE,
                    help="patch side in px; setting it EQUAL to --patch-step "
                         "makes the pool disjoint by construction, which is what "
                         "the E3-C overlap-free control needs")
    ap.add_argument("--patch-step", type=int, default=PATCH_STEP)
    ap.add_argument("--out-tag", default=None,
                    help="write into PROCESSED_ROOT/<region>_<tag>/ instead of "
                         "<region>/, so a non-default geometry does not overwrite "
                         "the primary cached arrays")
    ap.add_argument("--min-clear", type=float, default=0.6)
    args = ap.parse_args()
    cfg = REGIONS[args.region]
    region_dir = PROCESSED_ROOT / (args.region if not args.out_tag
                                   else f"{args.region}_{args.out_tag}")

    # reference grid: region box in UTM, 10 m pixel
    epsg = 32600 + int((cfg["center"][0] + 180) // 6) + 1
    dst_crs = f"EPSG:{epsg}"
    trans = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x1, y1 = trans.transform(cfg["box"][0], cfg["box"][1])
    x2, y2 = trans.transform(cfg["box"][2], cfg["box"][3])
    # snap to 10 m grid using a reference-zone tile's origin (S2 tiles are 10 m aligned;
    # anchoring on a different UTM zone would mix CRS coordinates)
    tiles_t1 = ordered_tiles(region_dir, "t1", dst_crs)
    if not tiles_t1:
        print("no composites found"); sys.exit(1)
    with rasterio.open(tiles_t1[0]) as src:
        origin_x, origin_y = src.transform.c, src.transform.f
    gx0 = origin_x + 10 * np.floor((min(x1, x2) - origin_x) / 10)
    y_bottom = min(y1, y2)
    y_top = max(y1, y2)
    gy_bottom = origin_y + 10 * np.floor((y_bottom - origin_y) / 10)
    W = int(np.ceil((max(x1, x2) - gx0) / 10))
    H = int(np.ceil((y_top - gy_bottom) / 10))
    gy0 = gy_bottom + H * 10  # grid top (northing)
    grid = (H, W)
    transform = from_origin(gx0, gy0, 10.0, 10.0)

    print(f"reference grid: {H}x{W} at EPSG:{epsg}, origin=({gx0:.0f},{gy0:.0f})")

    # t1 features: stitch tile composites (each tile may be in a different UTM zone,
    # so reproject onto the reference grid with the correct target CRS).
    feats1 = np.full((H, W, 9), np.nan, dtype="float32")
    for tf in ordered_tiles(region_dir, "t1", dst_crs):
        a = reproject_multiband(tf, grid, transform, dst_crs)
        feats1 = np.where(np.isnan(feats1), a, feats1)
    feats1 = np.nan_to_num(feats1, nan=0.0)

    feats2 = np.full((H, W, 9), np.nan, dtype="float32")
    for tf in ordered_tiles(region_dir, "t2", dst_crs):
        a = reproject_multiband(tf, grid, transform, dst_crs)
        feats2 = np.where(np.isnan(feats2), a, feats2)
    feats2 = np.nan_to_num(feats2, nan=0.0)

    def stitch_single(path):
        a = reproject_to_grid(path, grid, transform, dst_crs)
        return a

    change = stitch_single(region_dir / "change_labels.tif")
    lbl1 = stitch_single(region_dir / "labels_t1.tif")
    lbl2 = stitch_single(region_dir / "labels_t2.tif")

    cf1 = np.full(grid, np.nan, dtype="float32")
    for tf in ordered_tiles(region_dir, "t1", dst_crs, "*_clear_fraction.tif"):
        a = reproject_to_grid(tf, grid, transform, dst_crs, bilinear=True)
        cf1 = np.where(np.isnan(cf1), a, cf1)
    cf2 = np.full(grid, np.nan, dtype="float32")
    for tf in ordered_tiles(region_dir, "t2", dst_crs, "*_clear_fraction.tif"):
        a = reproject_to_grid(tf, grid, transform, dst_crs, bilinear=True)
        cf2 = np.where(np.isnan(cf2), a, cf2)
    cf1 = np.nan_to_num(cf1, nan=0.0)
    cf2 = np.nan_to_num(cf2, nan=0.0)

    # patch grid
    idx, f1, f2, y, cf = [], [], [], [], []
    PS, PST, MINCLR = args.patch_size, args.patch_step, args.min_clear
    for r in range(0, H - PS + 1, PST):
        for c in range(0, W - PS + 1, PST):
            sl = (slice(r, r + PS), slice(c, c + PS))
            m = (lbl1[sl] == 1) | (lbl2[sl] == 1)
            frac = m.mean()
            if not (MIN_MANGROVE_FRAC <= frac <= MAX_MANGROVE_FRAC):
                continue
            if cf1[sl].mean() < MINCLR or cf2[sl].mean() < MINCLR:
                continue
            idx.append((r, c))
            f1.append(feats1[sl])
            f2.append(feats2[sl])
            y.append(change[sl].astype(np.int8))
            cf.append((float(cf1[sl].mean()), float(cf2[sl].mean())))

    if not idx:
        print("no patches kept"); sys.exit(1)

    f1a = np.stack(f1)
    f2a = np.stack(f2)
    ya = np.stack(y)
    np.savez_compressed(region_dir / "t1" / "features.npz", features=f1a)
    np.savez_compressed(region_dir / "t2" / "features.npz", features=f2a)
    np.savez_compressed(region_dir / "change_labels.npz", change=ya)

    meta = {
        "region": args.region,
        "patch_size": args.patch_size,
        "patch_step": args.patch_step,
        "disjoint_by_construction": args.patch_step >= args.patch_size,
        "epsg": epsg,
        "grid": [H, W],
        "grid_origin": [gx0, gy0],
        "n_patches": len(idx),
        "patches": idx,
        "class_counts": {i: int((ya == i).sum()) for i in range(4)},
        "clear_fraction_mean": [float(np.mean([x[0] for x in cf])), float(np.mean([x[1] for x in cf]))],
    }
    (region_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"region={args.region}: {len(idx)} patches, class_counts={meta['class_counts']}")


if __name__ == "__main__":
    main()
