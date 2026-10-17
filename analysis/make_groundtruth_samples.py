"""Stratified random sample design for visual interpretation.

Sample domain: patch-union of each region (the model-evaluation domain).
Strata: GMW v4.1.12 change class (0 stable mangrove / 1 loss / 2 gain /
3 stable background). Allocation per region: 800 points = 400 background,
200 stable mangrove, 100 loss, 100 gain (change classes over-sampled vs
their area fraction; the area-based estimator of Olofsson et al. 2014
re-weights by stratum area, so the design is unbiased for area-estimated
accuracy).

Interpreter workflow (human): for each point, judge mangrove PRESENCE at
t1 (Dec 2019-Jan 2020) and at t2 (Dec 2024-Jan 2025) using high-resolution
imagery (e.g. Maxar/PlanetScope via Google Earth). Fill interp_t1 /
interp_t2 in the GeoJSON. The change class is derived:
  (1,1) stable mangrove, (1,0) loss, (0,1) gain, (0,0) stable background.

Outputs:
  data/groundtruth/<region>_samples.geojson   (to be filled by interpreter)
  data/groundtruth/interpretation_aid_<region>.png
  results/groundtruth_design.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.config import REGIONS
from mangrove.paths import PROCESSED_ROOT, RESULTS_ROOT, FIGURES_ROOT

N_PER_STRATUM = {3: 400, 0: 200, 1: 100, 2: 100}
SEED = 2026
RGB = (3, 2, 1)  # B04, B03, B02 as 1-based rasterio band indices


def mosaic_rgb(region, epoch):
    """True-color (B04, B03, B02) RGB mosaic of the per-tile composites onto the
    region reference grid. Same-CRS tiles are sliced exactly (10 m grid-aligned);
    cross-UTM-zone tiles (e.g. zone-48 tiles of the Beilunhe region) are
    reprojected with rasterio.warp, as in exp1_patch.py. Reference-zone tiles
    win in the (minimal) overlap."""
    from experiments.exp1_patch import ordered_tiles, reproject_multiband
    meta = json.loads((PROCESSED_ROOT / region / "metadata.json").read_text())
    H, W = meta["grid"]
    epsg = meta["epsg"]
    dst_crs = f"EPSG:{epsg}"
    transform = from_origin(meta["grid_origin"][0], meta["grid_origin"][1], 10.0, 10.0)
    acc = np.zeros((H, W, 3), "float32")
    cnt = np.zeros((H, W), "int32")
    for f in ordered_tiles(PROCESSED_ROOT / region, epoch, dst_crs):
        with rasterio.open(f) as ds:
            if ds.crs.to_epsg() == epsg:
                b = ds.read([3, 2, 1]).astype("float32").transpose(1, 2, 0)  # (H,W,3) = B04,B03,B02
                c0 = int(round((ds.transform.c - transform.c) / 10.0))
                r0 = int(round((transform.f - ds.transform.f) / 10.0))
                sr0, sc0 = max(0, r0), max(0, c0)
                sr1, sc1 = min(H, r0 + b.shape[0]), min(W, c0 + b.shape[1])
                if sr1 <= sr0 or sc1 <= sc0:
                    continue
                a = b[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0]
                sub = (slice(sr0, sr1), slice(sc0, sc1))
                take = (cnt[sub] == 0)
                acc[sub][take] = a[take]
                cnt[sub][take] = 1
            else:
                a9 = reproject_multiband(f, (H, W), transform, dst_crs)
                a = a9[:, :, [2, 1, 0]]
                take = ~np.isnan(a9.max(axis=2)) & (cnt == 0)
                acc[take] = a[take]
                cnt[take] = 1
    acc = acc / np.maximum(cnt[:, :, None], 1)
    lo, hi = np.percentile(acc[cnt > 0], [2, 98])
    norm = np.clip((acc - lo) / max(hi - lo, 1), 0, 1) * 255.0
    return norm.astype(np.uint8), meta



def main():
    rng = np.random.default_rng(SEED)
    design = {}
    for region in REGIONS:
        meta = json.loads((PROCESSED_ROOT / region / "metadata.json").read_text())
        H, W = meta["grid"]
        with rasterio.open(PROCESSED_ROOT / region / "change_labels.tif") as ds:
            lab = ds.read(1)
        union = np.zeros((H, W), bool)
        for (r, c) in meta["patches"]:
            union[r:r + 512, c:c + 512] = True

        pts = []
        for cls, n in N_PER_STRATUM.items():
            mask = (lab == cls) & union
            avail = int(mask.sum())
            n_take = min(n, avail)
            rs, cs = np.where(mask)
            perm = rng.permutation(len(rs))[:n_take]
            for i in perm:
                pts.append({
                    "type": "Feature",
                    "properties": {
                        "id": f"{region}_{cls}_{i:05d}",
                        "region": region,
                        "gmw_class": int(cls),
                        "gmw_class_name": ["stable_mangrove", "loss", "gain",
                                           "stable_background"][cls],
                        "x": meta["grid_origin"][0] + (cs[i] + 0.5) * 10.0,
                        "y": meta["grid_origin"][1] - (rs[i] + 0.5) * 10.0,
                        "t1_mng": int(((lab == 0) | (lab == 1))[rs[i], cs[i]]),
                        "t2_mng": int(((lab == 0) | (lab == 2))[rs[i], cs[i]]),
                        "interp_t1": None,   # 0/1, fill after interpretation
                        "interp_t2": None,
                    },
                    "geometry": {"type": "Point",
                                 "coordinates": [meta["grid_origin"][0] + (cs[i] + 0.5) * 10.0,
                                                  meta["grid_origin"][1] - (rs[i] + 0.5) * 10.0]},
                })
        gt_dir = ROOT / "data" / "groundtruth"
        gt_dir.mkdir(parents=True, exist_ok=True)
        (gt_dir / f"{region}_samples.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": pts}))

        # interpretation aid: top row full extent (context), bottom row zoom
        # on the mangrove extent. Interpreter works in Google Earth at the
        # point coordinates; this figure is orientation + QA.
        rgb1, _ = mosaic_rgb(region, "t1")
        rgb2, _ = mosaic_rgb(region, "t2")
        cmap = matplotlib.colormaps["tab10"]
        mng = (lab == 0) | (lab == 1) | (lab == 2)
        rs, cs = np.where(mng & union)
        if len(rs):
            r0 = max(0, int(rs.min()) - 256)
            r1 = min(H, int(rs.max()) + 512 + 256)
            c0 = max(0, int(cs.min()) - 256)
            c1 = min(W, int(cs.max()) + 512 + 256)
        else:
            r0, r1, c0, c1 = 0, H, 0, W

        coords = np.array([[p["geometry"]["coordinates"][0],
                            p["geometry"]["coordinates"][1]] for p in pts])
        ox, oy = meta["grid_origin"]
        xs = (coords[:, 0] - ox) / 10.0 + 0.5
        ys = (oy - coords[:, 1]) / 10.0 + 0.5
        colors = {0: "#00c000", 1: "#ff0000", 2: "#0000ff", 3: "#888888"}
        names = {0: "stable mangrove", 1: "loss", 2: "gain", 3: "background"}

        def draw_panel(ax, rgb, labview, sub, title):
            r0p, r1p = sub[0]
            c0p, c1p = sub[1]
            ax.imshow(rgb[r0p:r1p, c0p:c1p], aspect="auto")
            if labview is not None:
                ax.imshow(labview[r0p:r1p, c0p:c1p], cmap=cmap, vmin=0, vmax=3.9, alpha=0.0)  # no-op keep grid
            m = (xs >= sub[1][0]) & (xs < sub[1][1]) & (ys >= sub[0][0]) & (ys < sub[0][1])
            for cls in (0, 1, 2, 3):
                mc = m & np.array([p["properties"]["gmw_class"] == cls for p in pts])
                if mc.any():
                    ax.scatter(xs[mc] - sub[1][0], ys[mc] - sub[0][0], s=8,
                               c=colors[cls], lw=0, label=names[cls])
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        full = ((0, H), (0, W))
        fig, axes = plt.subplots(2, 3, figsize=(16, 11))
        for ax, rgb, title in zip(axes[0], (rgb1, rgb2, None),
                                  ("t1 full (Dec 2019-Jan 2020)", "t2 full (Dec 2024-Jan 2025)",
                                   "GMW 2019->2024 change, full extent")):
            if rgb is None:
                draw_panel(ax, np.zeros_like(rgb1), lab, full, title)
            else:
                draw_panel(ax, rgb, None, full, title)
        zoom = ((r0, r1), (c0, c1))
        for ax, rgb, title in zip(axes[1], (rgb1, rgb2, None),
                                  ("t1 zoom", "t2 zoom", "GMW change + sample points (zoom)")):
            if rgb is None:
                draw_panel(ax, np.zeros_like(rgb1), lab, zoom, title)
            else:
                draw_panel(ax, rgb, None, zoom, title)
        handles, labels = [], []
        for cls in (0, 1, 2, 3):
            h = plt.Line2D([], [], marker="o", ls="", color=colors[cls],
                           markersize=5, label=names[cls])
            handles.append(h)
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
        fig.suptitle(f"Ground-truth interpretation aid — {region} ({len(pts)} points)", y=0.995)
        fig.tight_layout(rect=[0, 0.03, 1, 0.985])
        fig.savefig(FIGURES_ROOT / f"interpretation_aid_{region}.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        design[region] = {
            "n_points": len(pts),
            "allocation": {["stable_mangrove", "loss", "gain", "stable_background"][k]: v
                           for k, v in N_PER_STRATUM.items()},
            "stratum_px_available": {["stable_mangrove", "loss", "gain", "stable_background"][k]:
                                     int(((lab == k) & union).sum())
                                     for k in N_PER_STRATUM},
        }
        print(region, len(pts), "points")

    (RESULTS_ROOT / "groundtruth_design.json").write_text(json.dumps(design, indent=2))
    print("saved data/groundtruth/*.geojson + figures/interpretation_aid_*.png")


if __name__ == "__main__":
    main()
