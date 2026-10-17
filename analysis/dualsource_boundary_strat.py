"""Second-source check stratified by distance to the reference mangrove fringe.

Motivation: the agreement-subset re-evaluation (the 'Zhanjiang
weakness' argument) is computed on the pixels where the two label sources
agree on t1 presence. If the inter-source disagreement is concentrated in the
boundary band -- where change detection actually happens -- then that subset
is enriched in interior/exterior core pixels and cannot validate change
detection rankings.

This script quantifies, per region:
  1. Band composition of the inter-source disagreement and agreement sets,
     where band = distance to the reference mangrove fringe
     (v4.1.12 t1 UNION t2 extent, 10 m grid).
  2. Per-model (seed 2026) change F1 on: full test support, the agreement
     subset, the edge band (<=30 m) and the interior (>100 m), plus the true
     change fraction of each stratum.

Bands (10 m/px): edge <=3 px (0-30 m), mid 4-10 px (30-100 m), interior >10 px.

Output: results/dualsource_boundary_strat.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
SEED = 2026
EDGE_PX, MID_PX = 3, 10
# predictions root (env override so the check can run on the uniform sweep)
PRED_DIR = Path(os.environ.get(
    "MANGROVE_PREDICTIONS_ROOT", str(ROOT / "results" / "predictions")))


def read_region(name):
    with rasterio.open(ROOT / "data" / "processed" / name) as src:
        return src.read(1).astype(np.int8)


def f1ch(y, p):
    cm = np.zeros((4, 4), dtype=np.int64)
    comb = y.ravel().astype(np.int64) * 4 + p.ravel().astype(np.int64)
    cm = np.bincount(comb, minlength=16).reshape(4, 4)
    chg = [1, 2]
    tp = cm[np.ix_(chg, chg)].sum()
    fn = cm[np.ix_(chg, [0, 3])].sum()
    fp = cm[np.ix_([0, 3], chg)].sum()
    return 2 * tp / max(2 * tp + fn + fp, 1e-9)


def band_frac(mask, edge, mid, interior):
    tot = int(mask.sum())
    if tot == 0:
        return {"edge": None, "mid": None, "interior": None}
    return {b: round(float((mask & m).sum()) / tot, 4)
            for b, m in [("edge", edge), ("mid", mid), ("interior", interior)]}


def main():
    out = {"config": {"seed": SEED, "edge_px": EDGE_PX, "mid_px": MID_PX,
                      "models": MODELS}, "per_region": {}}
    for region in REGIONS:
        proc = ROOT / "data" / "processed" / region
        meta = json.loads((proc / "metadata.json").read_text())
        patches = np.asarray(meta["patches"])
        pres_a = read_region(region + "/labels_t1.tif")    # v4.1.12 t1 presence
        pres_b = read_region(region + "/gmw2020_s2.tif")   # S2-based 2020 presence
        t2m = read_region(region + "/labels_t2.tif")       # v4.1.12 t2 presence
        extent = (pres_a == 1) | (t2m == 1)
        boundary = extent & ~binary_erosion(extent, iterations=1)
        dist = distance_transform_edt(~boundary)
        edge, interior = dist <= EDGE_PX, dist > MID_PX
        mid = (dist > EDGE_PX) & (dist <= MID_PX)

        dis_sup = (pres_a != pres_b) & ((pres_a == 1) | (pres_b == 1))
        # paper subset: ALL test pixels where the two sources agree
        agr_all = (pres_a == pres_b)
        # stricter stratum: mangrove-related pixels where the sources agree
        agr_sup = agr_all & ((pres_a == 1) | (pres_b == 1))

        test_idx = np.asarray(json.loads(
            (ROOT / "data" / "splits" / f"{region}_test.json").read_text()))
        change = np.load(proc / "change_labels.npz")["change"]
        y_all = change[test_idx]
        assert len(y_all) > 0

        models_out = {}
        for m in MODELS:
            pf = PRED_DIR / f"{m}_s{SEED}_{region}.npy"
            if not pf.exists():
                continue
            p = np.load(pf)
            assert p.shape == y_all.shape, (p.shape, y_all.shape)
            crops = {
                "full": None,
                "agree_all": [agr_all[r:r+512, c:c+512] for r, c in patches[test_idx]],
                "agree_mangrove": [agr_sup[r:r+512, c:c+512] for r, c in patches[test_idx]],
                "edge_30m": [edge[r:r+512, c:c+512] for r, c in patches[test_idx]],
                "interior_100m": [interior[r:r+512, c:c+512] for r, c in patches[test_idx]],
            }
            strat = {}
            for name, cmasks in crops.items():
                yy, pp = [], []
                for i in range(len(p)):
                    s = np.ones(512 * 512, bool) if cmasks is None else cmasks[i].ravel()
                    yy.append(y_all[i].ravel()[s])
                    pp.append(p[i].ravel()[s])
                y, pr = np.concatenate(yy), np.concatenate(pp)
                strat[name] = {"F1ch": round(float(f1ch(y, pr)), 4),
                               "n_pixels": int(len(y))}
            for name, cmasks in crops.items():
                if cmasks is None:
                    strat[name]["true_change_frac"] = round(
                        float(np.isin(y_all, [1, 2]).mean()), 4)
                    continue
                tot = sum(int(c.sum()) for c in cmasks)
                chg = sum(int(np.isin(y_all[i], [1, 2])[c].sum())
                          for i, c in enumerate(cmasks))
                strat[name]["true_change_frac"] = round(chg / tot, 4) if tot else None
            models_out[m] = strat

        # true change fraction of the test support per band
        te = ti = ce = ci = 0
        for i in range(len(y_all)):
            r, c = patches[test_idx[i]]
            ye, yi = y_all[i][edge[r:r+512, c:c+512]], y_all[i][interior[r:r+512, c:c+512]]
            te += ye.size; ce += int(np.isin(ye, [1, 2]).sum())
            ti += yi.size; ci += int(np.isin(yi, [1, 2]).sum())

        out["per_region"][region] = {
            "n_test_patches": int(len(test_idx)),
            "inter_source_disagreement_band_frac": band_frac(dis_sup, edge, mid, interior),
            "inter_source_agreement_band_frac": band_frac(agr_sup, edge, mid, interior),
            "true_change_band_frac_test": {
                "edge_30m": round(ce / te, 4), "interior_100m": round(ci / ti, 4)},
            "models": models_out,
        }

    (PRED_DIR.parent / "dualsource_boundary_strat.json").write_text(
        json.dumps(out, indent=1))

    print(f"{'region':<10} {'disagree|edge30':>15} {'agree|interior':>15} "
          f"{'truechg|edge30':>15} {'truechg|interior':>17}")
    for region, rd in out["per_region"].items():
        print(f"{region:<10} "
              f"{rd['inter_source_disagreement_band_frac']['edge']:15.3f} "
              f"{rd['inter_source_agreement_band_frac']['interior']:15.3f} "
              f"{rd['true_change_band_frac_test']['edge_30m']:15.3f} "
              f"{rd['true_change_band_frac_test']['interior_100m']:17.3f}")
    print("\nper-model F1ch by stratum (seed 2026):")
    for region, rd in out["per_region"].items():
        print(f"  {region}")
        for m, st in rd["models"].items():
            row = "  ".join(f"{k}={st[k]['F1ch']:.3f}" for k in
                            ["full", "agree_all", "agree_mangrove", "edge_30m", "interior_100m"])
            print(f"    {m:<18} {row}")
    print(f"\nsaved {PRED_DIR.parent / 'dualsource_boundary_strat.json'}")


if __name__ == "__main__":
    main()
