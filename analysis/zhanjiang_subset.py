"""Robustness of the model ranking where the reference is
least reliable.

The two label sources (Landsat-based GMW v4.1.12, Sentinel-2-based GMW 2020)
agree least on Zhanjiang (74.1% of t1 presence pixels). We re-evaluate every
model on the SUBSET of test pixels where the two sources AGREE on mangrove
presence (t1 presence in v4112 == presence in the S2-based 2020 layer), and
re-rank. If the "temporal family most robust on Zhanjiang" conclusion holds on
the high-confidence subset, it is not an artefact of the weakest reference;
if it reverses, the ranking is reference-quality dependent.

Output: results/dualsource_subset.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.config import CHANGE_CLASSES

MODELS = ["unet", "unet_cbam", "unet_msa", "unet_cbam_msa", "deeplabv3plus",
          "temporal_unet", "temporal_unet_gn", "fc_siam"]
SEEDS = [2026, 2027, 2028]
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
# predictions root (env override so the check can run on the uniform sweep)
PRED_DIR = Path(os.environ.get(
    "MANGROVE_PREDICTIONS_ROOT", str(ROOT / "results" / "predictions")))


def presence_raster(region, name):
    with rasterio.open(ROOT / "data" / "processed" / region / name) as src:
        return src.read(1)


def cm4(y, p):
    cm = np.zeros((4, 4), dtype=np.int64)
    comb = y.ravel().astype(np.int64) * 4 + p.ravel().astype(np.int64)
    return np.bincount(comb, minlength=16).reshape(4, 4)


def f1ch_miou(cm):
    ious = [cm[i, i] / (cm[i].sum() + cm[:, i].sum() - cm[i, i])
            for i in range(4) if (cm[i].sum() + cm[:, i].sum() - cm[i, i]) > 0]
    mi = float(np.mean(ious)) if ious else 0.0
    chg = [1, 2]
    tp = cm[np.ix_(chg, chg)].sum()
    fn = cm[np.ix_(chg, [0, 3])].sum()
    fp = cm[np.ix_([0, 3], chg)].sum()
    f1 = 2 * tp / max(2 * tp + fn + fp, 1)
    return float(f1), mi, float(fp / max(fp + tp, 1))


def main():
    out = {}
    for region in REGIONS:
        meta = json.loads((ROOT / "data" / "processed" / region /
                           "metadata.json").read_text())
        transform = meta.get("transform")
        patches = np.asarray(meta["patches"], dtype=int)
        idx = np.asarray(json.loads((ROOT / "data" / "splits" /
                                     f"{region}_test.json").read_text()), dtype=int)
        p1 = presence_raster(region, "labels_t1.tif")
        p2020 = presence_raster(region, "gmw2020_s2.tif")
        agree = ((p1 == 1) == (p2020 == 1))

        # region-level agreement over the test-patch support
        supp = np.zeros(p1.shape, bool)
        for k in range(len(idx)):
            i = int(idx[k])
            r, c = patches[i]
            supp[r:r + 512, c:c + 512] = True
        agree_frac = float(agree[supp].mean())

        labels = np.load(ROOT / "data" / "processed" / region /
                         "change_labels.npz")["change"]
        per_model = {}
        for m in MODELS:
            cms_all, cms_sub = [], []
            ok = True
            for s in SEEDS:
                pf = PRED_DIR / f"{m}_s{s}_{region}.npy"
                if not pf.exists():
                    ok = False
                    break
                pred = np.load(pf)
                for k in range(pred.shape[0]):
                    i = int(idx[k])
                    r, c = patches[i]
                    sub = agree[r:r + 512, c:c + 512]
                    ys = labels[i]
                    ps = pred[k]
                    cms_all.append(cm4(ys.ravel(), ps.ravel()))
                    cms_sub.append(cm4(ys[sub].ravel(), ps[sub].ravel()))
            if not ok:
                continue
            cms_all = np.stack(cms_all)
            cms_sub = np.stack(cms_sub)
            fa = f1ch_miou(cms_all.sum(axis=0))
            fs = f1ch_miou(cms_sub.sum(axis=0))
            per_model[m] = {
                "full": {"F1ch": fa[0], "mIoU": fa[1], "commission": fa[2]},
                "agreement_subset": {"F1ch": fs[0], "mIoU": fs[1], "commission": fs[2]},
            }
        out[region] = {
            "t1_presence_agreement_test_support": agree_frac,
            "models": per_model,
        }
        print(f"== {region} (agreement {agree_frac:.3f})")
        for metric in ("F1ch", "mIoU"):
            full_rank = sorted(per_model, key=lambda m: -per_model[m]["full"][metric])
            sub_rank = sorted(per_model, key=lambda m: -per_model[m]["agreement_subset"][metric])
            print(f"  {metric} full:      " + " > ".join(full_rank[:4]))
            print(f"  {metric} subset:    " + " > ".join(sub_rank[:4]))

    (PRED_DIR.parent / "dualsource_subset.json").write_text(json.dumps(out, indent=1))
    print("saved results/dualsource_subset.json")


if __name__ == "__main__":
    main()
