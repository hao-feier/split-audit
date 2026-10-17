"""Exp0: confirm study regions against GMW v4.1.12 extent (PLAN 1).

Outputs:
  results/exp0_region_stats.json   per-region mangrove area (2019/2024), change area
  figures/fig0_region_selection.png  candidate boxes + mangrove extent
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mangrove.config import REGIONS, EPOCHS
from mangrove.geo import win_bounds
from mangrove.paths import GMW_ROOT, RESULTS_ROOT, FIGURES_ROOT
from scipy import ndimage

STACK = GMW_ROOT / "gmw_v4112_stack" / "gmw_mng_ext_v4112_stack.vrt"
BAND_T1 = EPOCHS["t1"]["year"] - 1985 + 1
BAND_T2 = EPOCHS["t2"]["year"] - 1985 + 1


def region_stats(name):
    cfg = REGIONS[name]
    lon_min, lat_min, lon_max, lat_max = cfg["box"]
    with rasterio.open(STACK) as src:
        win = win_bounds(src, lon_min, lat_min, lon_max, lat_max)
        t1 = src.read(BAND_T1, window=win).astype(np.int8)
        t2 = src.read(BAND_T2, window=win).astype(np.int8)
        res_km = src.res[0] * 111.0
        px_km2 = res_km ** 2
        m1 = (t1 == 1)
        m2 = (t2 == 1)
        m1c = ndimage.binary_erosion(m1)
        m2c = ndimage.binary_erosion(m2)
        lons = src.transform.c + (win.col_off + np.arange(win.width) + 0.5) * src.transform.a
        lats = src.transform.f + (win.row_off + np.arange(win.height) + 0.5) * src.transform.d
        ys, xs = np.where(m1)
        stats = {
            "box": list(cfg["box"]),
            "t1_year": EPOCHS["t1"]["year"], "t2_year": EPOCHS["t2"]["year"],
            "mangrove_t1_km2": round(float(m1.sum() * px_km2), 1),
            "mangrove_t2_km2": round(float(m2.sum() * px_km2), 1),
            "core_t1_km2": round(float(m1c.sum() * px_km2), 1),
            "core_t2_km2": round(float(m2c.sum() * px_km2), 1),
            "loss_km2": round(float((m1c & ~m2c).sum() * px_km2), 1),
            "gain_km2": round(float((m2c & ~m1c).sum() * px_km2), 1),
        }
    return stats, lons, lats, m1


def main():
    out, fig_data = {}, {}
    for name in REGIONS:
        stats, lons, lats, m1 = region_stats(name)
        out[name] = stats
        fig_data[name] = (lons, lats, m1)
    (RESULTS_ROOT / "exp0_region_stats.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(10, 7))
    for name, cfg in REGIONS.items():
        lon_min, lat_min, lon_max, lat_max = cfg["box"]
        color = "red" if cfg["role"] == "train" else "blue"
        ax.add_patch(Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                               fill=False, edgecolor=color, lw=2))
        ax.text(lon_min, lat_max + 0.02, f"{name} ({cfg['role']})", fontsize=10, color=color)
        lons, lats, m1 = fig_data[name]
        ys, xs = np.where(m1)
        step = max(1, len(xs) // 3000)
        ax.scatter(lons[xs[::step]], lats[ys[::step]], s=1, c="green", alpha=0.5)
    ax.set_title("Study regions vs GMW v4.1.12 mangrove extent (2019)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_aspect(1.4)
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "fig0_region_selection.png", dpi=200)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
