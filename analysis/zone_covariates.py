"""Zone-level covariate comparison (is the test zone simply harder?).

For every zone of the Vietnam split (train / val / test / buffer, v2) and the
two independent test regions, reports per-patch mean (+ std over patches) of
mangrove-presence fraction, change fraction, and the three derived indices
(NDVI / NDWI / MNDWI) of the t1 feature stack. A systematic covariate gap
between zones would confound 'removing spatial leakage' with 'the test zone
is a different landscape'; these numbers let the paper state where that
confound is present or absent.

Output: results/zone_covariates.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.datasets import load_region  # noqa: E402
from mangrove.paths import SPLITS_ROOT  # noqa: E402

# t1 feature stack: [B02, B03, B04, B08, B11, B12, NDVI, NDWI, MNDWI]
I_NDVI, I_NDWI, I_MNDWI = 6, 7, 8


def zone_stats(region, idx):
    data = load_region(region)
    ch = data["change"]           # (N, H, W) 0 stable-mng / 1 loss / 2 gain / 3 bg
    f1 = data["t1_feat"]          # (N, H, W, 9)
    rows = []
    for i in idx:
        c = ch[i]
        mang = (c <= 2).mean()            # presence in at least one epoch
        change = np.isin(c, [1, 2]).mean()
        rows.append({
            "mangrove_frac": float(mang),
            "change_frac": float(change),
            "ndvi": float(f1[i, :, :, I_NDVI].mean()) / 10000.0,
            "ndwi": float(f1[i, :, :, I_NDWI].mean()) / 10000.0,
            "mndwi": float(f1[i, :, :, I_MNDWI].mean()) / 10000.0,
        })
    keys = ["mangrove_frac", "change_frac", "ndvi", "ndwi", "mndwi"]
    return {k: {"mean": round(float(np.mean([r[k] for r in rows])), 4),
                "std": round(float(np.std([r[k] for r in rows], ddof=1)), 4)}
            for k in keys} | {"n_patches": len(rows)}


def main():
    out = {}
    for name in ["train", "val", "test", "buffer"]:
        idx = json.loads((SPLITS_ROOT / f"vietnam_{name}.json").read_text())
        out[f"vietnam_{name}"] = zone_stats("vietnam", idx)
    for r in ["beilunhe", "zhanjiang"]:
        idx = json.loads((SPLITS_ROOT / f"{r}_test.json").read_text())
        out[f"{r}_test"] = zone_stats(r, idx)
    (ROOT / "results" / "zone_covariates.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(k, v["n_patches"],
              f"mang={v['mangrove_frac']['mean']:.3f} chg={v['change_frac']['mean']:.4f} "
              f"ndvi={v['ndvi']['mean']:.3f} ndwi={v['ndwi']['mean']:.3f} "
              f"mndwi={v['mndwi']['mean']:.3f}", flush=True)
    print("wrote results/zone_covariates.json")


if __name__ == "__main__":
    main()
