"""Region vs patch-domain statistics, split fractions, distances.

Computes (per region):
  - region-raster (full grid, unique pixels): per-class px + km2, change fraction
  - patch-union (unique pixels inside the union of patch rects): per-class px + km2
  - patch-sum (all patches, 50% overlap): per-class px  [current Table 1]
  - vietnam: per-zone patch-sum change fractions (train/val/test/buffer)
  - centroid + nearest-edge distances between region grids (haversine)
  - lon/lat extent of each region grid (UTM zone check)

Writes results/audit_regions.json.
"""
import json, math
import numpy as np
import rasterio
from pathlib import Path
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
PX = 10.0  # meters
CLASSES = {0: "stable_mangrove", 1: "loss", 2: "gain", 3: "stable_background"}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def extent_latlon(epsg, origin, H, W):
    tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    x0, y0 = origin
    corners = [(x0, y0), (x0 + W * PX, y0), (x0, y0 - H * PX), (x0 + W * PX, y0 - H * PX)]
    lons, lats = tr.transform([c[0] for c in corners], [c[1] for c in corners])
    return min(lons), max(lons), min(lats), max(lats)


def main():
    out = {"regions": {}, "distances": {}}
    for region in ("vietnam", "beilunhe", "zhanjiang"):
        meta = json.loads((PROCESSED / region / "metadata.json").read_text())
        H, W = meta["grid"]
        epsg, origin = meta["epsg"], meta["grid_origin"]
        patches = np.array(meta["patches"])

        with rasterio.open(PROCESSED / region / "change_labels.tif") as ds:
            lab = ds.read(1)
        assert lab.shape == (H, W), (lab.shape, (H, W))

        region_counts = {c: int((lab == c).sum()) for c in CLASSES}
        union = np.zeros((H, W), bool)
        for r, c in patches:
            union[r:r + 512, c:c + 512] = True
        union_counts = {c: int(((lab == c) & union).sum()) for c in CLASSES}
        patchsum_counts = {}
        for c in CLASSES:
            s = 0
            for r, col in patches:
                s += int((lab[r:r + 512, col:col + 512] == c).sum())
            patchsum_counts[c] = s

        def frac(counts, key):
            tot = sum(counts.values())
            return (counts[1] + counts[2]) / tot

        lmin, lmax, bmin, bmax = extent_latlon(epsg, origin, H, W)
        out["regions"][region] = {
            "epsg": epsg,
            "grid_px": [H, W],
            "grid_km2": H * W * PX * PX / 1e6,
            "lon_range": [round(lmin, 4), round(lmax, 4)],
            "lat_range": [round(bmin, 4), round(bmax, 4)],
            "region_km2": {CLASSES[c]: v * PX * PX / 1e6 for c, v in region_counts.items()},
            "region_change_frac": frac(region_counts, None),
            "union_km2": union.sum() * PX * PX / 1e6,
            "avg_overlap_multiplicity": len(patches) * 512 * 512 / union.sum(),
            "union_km2_by_class": {CLASSES[c]: v * PX * PX / 1e6 for c, v in union_counts.items()},
            "union_change_frac": frac(union_counts, None),
            "patchsum_px": patchsum_counts,
            "patchsum_km2_equiv": {CLASSES[c]: v * PX * PX / 1e6 for c, v in patchsum_counts.items()},
            "patchsum_change_frac": frac(patchsum_counts, None),
        }

    # vietnam per-zone change fractions (patch-sum basis)
    zones = {}
    for z in ("train", "val", "test", "buffer"):
        idx = json.loads((SPLITS / f"vietnam_{z}.json").read_text())
        zones[z] = len(idx)
    out["vietnam_zones_n"] = zones

    # distances between region centroids + nearest edges
    regs = out["regions"]
    centers = {}
    for name, d in regs.items():
        centers[name] = ((d["lon_range"][0] + d["lon_range"][1]) / 2,
                         (d["lat_range"][0] + d["lat_range"][1]) / 2)
    for a, b in (("vietnam", "beilunhe"), ("vietnam", "zhanjiang"), ("beilunhe", "zhanjiang")):
        ca, cb = centers[a], centers[b]
        d_cent = haversine_km(ca[1], ca[0], cb[1], cb[0])
        # nearest edge distance between rectangles (lat/lon approx, km)
        da, db = regs[a], regs[b]
        gap_lon = max(0, max(da["lon_range"][0], db["lon_range"][0]) - min(da["lon_range"][1], db["lon_range"][1]))
        gap_lat = max(0, max(da["lat_range"][0], db["lat_range"][0]) - min(da["lat_range"][1], db["lat_range"][1]))
        km_per_deg_lat = 110.574
        km_per_deg_lon = 111.320 * math.cos(math.radians((ca[1] + cb[1]) / 2))
        d_edge = math.hypot(gap_lon * km_per_deg_lon, gap_lat * km_per_deg_lat)
        out["distances"][f"{a}_to_{b}"] = {
            "centroid_km": round(d_cent, 1),
            "nearest_edge_km": round(d_edge, 1),
            "lon_gap_deg": round(gap_lon, 4), "lat_gap_deg": round(gap_lat, 4),
        }

    (ROOT / "results" / "audit_regions.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
