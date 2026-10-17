"""Uncertainty on the reported mangrove area-change figures.

Conservative model: the two GMW v4.1.12 presence maps carry independent
per-pixel label error; the best available proxy for per-pixel presence
accuracy is the region's two-source agreement fraction a (Landsat-based
v4.1.12 vs the Sentinel-2-based GMW 2020 layer). We flip t1 / t2 presence
independently with probability 1 - a and propagate the moments of the
resulting area difference exactly (independent Bernoulli sum -> mean and
variance in one pass; normal approximation for the CI).

Output: results/area_uncertainty.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]


def presence(region, name):
    with rasterio.open(ROOT / "data" / "processed" / region / name) as src:
        return (src.read(1) == 1)


def main():
    out = {}
    for region in REGIONS:
        m1 = presence(region, "labels_t1.tif")
        m2 = presence(region, "labels_t2.tif")
        m2020 = presence(region, "gmw2020_s2.tif")
        a = float((m1 == m2020).mean())
        q = 1.0 - a
        p1 = m1.astype(float) * (1 - q) + (1 - m1.astype(float)) * q
        p2 = m2.astype(float) * (1 - q) + (1 - m2.astype(float)) * q
        pl = p1 * (1 - p2)
        pg = (1 - p1) * p2
        vl = pl * (1 - pl)
        vg = pg * (1 - pg)
        e_x = float((pg - pl).sum())
        var_x = float((vg + vl + 2 * pl * pg).sum())
        sd = float(np.sqrt(var_x))
        pix = 1e-4
        base_loss = float((m1 & ~m2).sum()) * pix
        base_gain = float((~m1 & m2).sum()) * pix
        net = base_gain - base_loss
        out[region] = {
            "two_source_agreement_t1_vs_2020": a,
            "flip_prob_per_pixel": q,
            "point_estimate_km2": {"loss": base_loss, "gain": base_gain, "net": net},
            "expected_net_km2": e_x * pix,
            "net_sd_km2": sd * pix,
            "net_ci95_km2": [e_x * pix - 1.96 * sd * pix, e_x * pix + 1.96 * sd * pix],
        }
        print(region, "point:", [round(x, 2) for x in (base_loss, base_gain, net)],
              "95% CI on net:", [round(x, 2) for x in out[region]["net_ci95_km2"]])
    (ROOT / "results" / "area_uncertainty.json").write_text(json.dumps(out, indent=1))
    print("saved results/area_uncertainty.json")


if __name__ == "__main__":
    main()
