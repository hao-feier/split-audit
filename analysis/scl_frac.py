"""SCL classes 1-3 (saturated/dark/shadow) kept as "clear"
contribute a fraction of possibly-contaminated observations to the composites.
Here we quantify, per selected scene, the fraction of region-window pixels in
SCL classes 1, 2, 3 (and 7 unclassified, for reference), so the reader can
bound the contamination of the per-pixel median composites.

Output: results/scl_frac.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.config import REGIONS


def region_window(box):
    """UTM window (in px of a 10 m grid) covering the WGS84 region box."""
    epsg = 32600 + int((box[0] + 180) // 6) + 1
    trans = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x1, y1 = trans.transform(box[0], box[1])
    x2, y2 = trans.transform(box[2], box[3])
    gx0 = 10 * math.floor(min(x1, x2) / 10)
    gy0 = 10 * math.floor(min(y1, y2) / 10)
    return epsg, (int(math.ceil((max(x1, x2) - gx0) / 10)),
                  int(math.ceil((max(y1, y2) - gy0) / 10)))


def scene_scl_fractions(scene_dir, w, h):
    with rasterio.open(scene_dir / "SCL.tif") as src:
        scl = src.read(1)
    counts = np.bincount(scl.ravel(), minlength=13)
    total = counts.sum()
    n1, n2, n3, n7 = counts[1], counts[2], counts[3], counts[7]
    return {
        "SCL1_saturated_frac": float(n1 / total),
        "SCL2_dark_frac": float(n2 / total),
        "SCL3_shadow_frac": float(n3 / total),
        "SCL7_unclassified_frac": float(n7 / total),
        "SCL123_combined_frac": float((n1 + n2 + n3) / total),
    }


def main():
    sel = json.loads((ROOT / "results" / "scene_selection.json").read_text())
    out = {}
    for region, epochs in sel.items():
        epsg, (w, h) = region_window(REGIONS[region]["box"])
        out[region] = {}
        for epoch, tiles in epochs.items():
            out[region][epoch] = {}
            for tile, v in tiles.items():
                if not v.get("ok"):
                    continue
                for scene in v["scenes"]:
                    d = ROOT / "data" / "raw" / "s2" / scene
                    if not d.exists():
                        continue
                    out[region][epoch][scene] = scene_scl_fractions(d, w, h)
    (ROOT / "results" / "scl_frac.json").write_text(json.dumps(out, indent=1))
    print("saved results/scl_frac.json")
    for region, epochs in out.items():
        for epoch, scenes in epochs.items():
            if not scenes:
                continue
            f = [s["SCL123_combined_frac"] for s in scenes.values()]
            f7 = [s["SCL7_unclassified_frac"] for s in scenes.values()]
            print(f"{region:10s} {epoch}: SCL123 max={max(f):.4f} "
                  f"mean={np.mean(f):.4f} | SCL7 max={max(f7):.4f}")


if __name__ == "__main__":
    main()
