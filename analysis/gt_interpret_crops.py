"""Build interpretation montages for the 2400 GT points.

Evidence per point:
  Both epochs: dry-season S2 L2A composites (10 m), 2.56 km square
  (256 px), same for t1 and t2 (symmetric evidence). High-res basemap
  (Google Earth / ESRI) is unreachable from this host (ESRI serves
  placeholder tiles from here), so presence is judged from the 10 m
  composite context; a second high-res pass from a machine with Google
  access can be done later via the workbench (gt_tool/index.html).

Output: gt_tool/montages/{region}_{epoch}_{k:03d}.png - 5x5 grids of 25
points (row-major), each cell labelled with its local index - plus
gt_tool/montages/manifest.json mapping montage -> point ids.

Usage:
  python analysis/gt_interpret_crops.py                 # all regions, t1+t2
  python analysis/gt_interpret_crops.py --region beilunhe --epochs t2
"""
import argparse
import concurrent.futures as cf
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
Z = 16
CELL_T2_M = 480          # ESRI, ~2.2 m/px at 21N -> ~216 px
CELL_T1_M = 2560         # S2 10 m -> 256 px
N = 5                    # 5x5 per montage
R = 6378137.0
TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")


def mercator_px(lon, lat, z=Z):
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(lon, lat)
    n = 2 ** z
    px = (x / (2 * math.pi * R) + 0.5) * n
    py = (0.5 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / (2 * math.pi)) * n
    return px, py


def fetch_tile(z, x, y, cache_dir):
    f = cache_dir / f"z{z}_{x}_{y}.jpg"
    if f.exists() and f.stat().st_size > 1000:
        return f
    url = TILE_URL.format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": "gt-interp/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
    except Exception:
        return None
    if len(data) > 1000:
        f.write_bytes(data)
        return f
    return None


def esri_crop(lon, lat, cell_m, cache_dir):
    """(2h+1, 2h+1, 3) uint8 crop of ~cell_m centred on (lon, lat)."""
    px, py = mercator_px(lon, lat)  # float pixel coords in the z16 world
    mpp = 156543.03392 * math.cos(math.radians(lat)) / 2 ** Z
    h = int(cell_m / 2 / mpp)
    cx, cy = int(math.floor(px)), int(math.floor(py))
    x0, y0, x1, y1 = cx - h, cy - h, cx + h, cy + h  # inclusive pixel box
    tx0, ty0, tx1, ty1 = x0 // 256, y0 // 256, x1 // 256, y1 // 256
    out = np.zeros((y1 - y0 + 1, x1 - x0 + 1, 3), "uint8")
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            f = fetch_tile(Z, tx, ty, cache_dir)
            if f is None:
                continue
            t = np.asarray(Image.open(f).convert("RGB"))
            sx0, sx1 = max(x0, tx * 256), min(x1 + 1, tx * 256 + 256)
            sy0, sy1 = max(y0, ty * 256), min(y1 + 1, ty * 256 + 256)
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = t[sy0 - ty * 256:sy1 - ty * 256, sx0 - tx * 256:sx1 - tx * 256]
    return out


def s2_crop(mosaic, row, col, cell_m):
    h = cell_m // 10
    r0, c0 = max(0, row - h // 2), max(0, col - h // 2)
    r1, c1 = r0 + h, c0 + h
    r0, c0 = max(0, r1 - h), max(0, c1 - h)
    return mosaic[r0:r1, c0:c1]


def make_montage(cells, labels, out_path):
    cell = cells[0].shape[0]
    m = Image.new("RGB", (N * cell, N * cell), (255, 0, 0))
    draw = ImageDraw.Draw(m)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for i, (c, lab) in enumerate(zip(cells, labels)):
        r, k = divmod(i, N)
        img = Image.fromarray(c).resize((cell, cell), Image.NEAREST)
        m.paste(img, (k * cell, r * cell))
        draw.rectangle([k * cell, r * cell, k * cell + 54, r * cell + 26], fill=(0, 0, 0))
        draw.text((k * cell + 3, r * cell + 2), lab, fill=(255, 255, 0), font=font)
    m.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="all")
    ap.add_argument("--epochs", default="t1,t2")
    args = ap.parse_args()
    epochs = args.epochs.split(",")
    regions = REGIONS if args.region == "all" else [args.region]
    pts = json.loads((ROOT / "gt_tool" / "points.json").read_text())
    # SHUFFLE per region so GMW stratum (block-ordered in the geojson, and
    # encoded in the point id) is NOT correlated with montage position:
    # the interpretation must stay blind to the reference.
    import random
    by_region = {}
    for r in regions:
        pl = [p for p in pts if p["region"] == r]
        random.Random(2026).shuffle(pl)
        by_region[r] = pl
    outdir = ROOT / "gt_tool" / "montages"
    outdir.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "gt_tool" / "hires"
    cache.mkdir(exist_ok=True)
    manifest = json.loads((outdir / "manifest.json").read_text()) if (outdir / "manifest.json").exists() else {}

    for region in regions:
        moses = {}
        for ep in epochs:
            plist = by_region[region]
            cells, labels, ids = [], [], []
            if ep not in moses:
                mcf = ROOT / "gt_tool" / "mosaic_cache" / f"{region}_{ep}.npy"
                if mcf.exists():
                    moses[ep] = np.load(mcf)
                    print(f"{region}/{ep} mosaic loaded from cache", flush=True)
                else:
                    import analysis.make_groundtruth_samples as gts
                    moses[ep] = gts.mosaic_rgb(region, ep)[0]
                    mcf.parent.mkdir(parents=True, exist_ok=True)
                    np.save(mcf, moses[ep])
                    print(f"{region}/{ep} mosaic built + cached", flush=True)
            meta = json.loads((ROOT / f"data/processed/{region}/metadata.json").read_text())
            ox, oy = meta["grid_origin"]
            for p in plist:
                col = int(round((p["x"] - ox) / 10.0))
                row = int(round((oy - p["y"]) / 10.0))
                c = s2_crop(moses[ep], row, col, CELL_T1_M)
                cells.append(c); labels.append(f"{len(ids):03d}"); ids.append(p["id"])
            for k in range(0, len(cells), N * N):
                fn = f"{region}_{ep}_{k // (N * N):03d}.png"
                make_montage(cells[k:k + N * N], labels[k:k + N * N], outdir / fn)
                manifest[f"{region}/{ep}/{k // (N * N):03d}"] = ids[k:k + N * N]
                print(f"{fn}: {len(ids[k:k + N * N])} cells", flush=True)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("done:", len(manifest), "montages total")


if __name__ == "__main__":
    main()
