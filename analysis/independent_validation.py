"""PLAN 2.2: independent-source validation.

The GMW 2020 S2-based mask (v4.0.19, Zenodo 12756047 — independent sensor +
processing from the Landsat-based v4.1.12 training labels) is used to:
  1. Quantify label-source agreement (GMW2020-S2 vs GMW-ts 2019/2024).
  2. Evaluate each model's predicted t1 presence (stable_mangrove + loss)
     against the independent mask on test-region pixels.

Outputs:
  results/independent_validation.json
  figures/fig7_independent_source.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mangrove.config import REGIONS, CHANGE_CLASSES
from mangrove.datasets import load_region, load_split
from mangrove.models import MODEL_REGISTRY, pick_device
from mangrove.paths import GMW_ROOT, PROCESSED_ROOT, RESULTS_ROOT, FIGURES_ROOT, ckpt_path
import analysis.figstyle as fs

fs.apply_style()

CLASS_RGB = fs.class_rgb_array()          # (4, 3) float, index == class id
# discrete presence palette: white = non-mangrove, green = mangrove
MNG_CMAP = ListedColormap([[1.0, 1.0, 1.0], list(CLASS_RGB[0])])
REGION_TITLE = {"vietnam": "Vietnam", "beilunhe": "Beilunhe",
                "zhanjiang": "Zhanjiang"}

TEMPORAL_MODELS = {"temporal_unet"}
# GMW 2020 S2-based independent validation mask (VRT mosaic of native 10 m tiles).
GMW2020_VRT = GMW_ROOT / "gmw_2020_v4019" / "GMW2020_S2_vrt.vrt"
# Region-specific GTiff produced by exp1_labels (already on the regional UTM grid).
GMW2020_REGION = lambda region: PROCESSED_ROOT / region / "gmw2020_s2.tif"
MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]


def to_grid(src_path, grid, transform):
    arr = np.zeros(grid, dtype="uint8")
    with rasterio.open(src_path) as src:
        reproject(source=rasterio.band(src, 1), destination=arr,
                   src_transform=src.transform, src_crs=src.crs,
                   dst_transform=transform, dst_crs=src.crs,
                   resampling=Resampling.nearest)
    return arr


def focus_crop(mng, meta, idx, win=1024):
    """A win x win window centred on the densest test patch (by mangrove
    fraction). Mangrove in these estuaries is a thin, fragmented fringe, so a
    full-region or bounding-box view is mostly empty; zooming into the richest,
    model-evaluated area fills the frame and exposes the boundary-level
    agreement/disagreement between the two label sources and the model."""
    H, W = mng.shape
    ps = meta["patch_size"]
    idxt = set(idx.tolist())
    best, bestf = None, -1.0
    for pos, (r, c) in enumerate(meta["patches"]):
        if pos in idxt:
            f = float(mng[r:r + ps, c:c + ps].mean())
            if f > bestf:
                bestf, best = f, (r, c)
    if best is None:
        return (0, H, 0, W)
    cr, cc = best[0] + ps // 2, best[1] + ps // 2
    r0, c0 = max(0, cr - win // 2), max(0, cc - win // 2)
    r1, c1 = min(H, r0 + win), min(W, c0 + win)
    r0, c0 = max(0, r1 - win), max(0, c1 - win)
    return (r0, r1, c0, c1)


def eval_presence(model_name, region, split, t1, t2, y, seed=2026):
    ck = ckpt_path(model_name, seed)
    if not ck.exists():
        return None
    is_temporal = model_name in TEMPORAL_MODELS
    in_ch = 9 if is_temporal else 27
    device = pick_device()
    model = MODEL_REGISTRY[model_name](in_ch=in_ch, out_ch=4).to(device)
    model.load_state_dict(torch.load(ck, map_location="cpu"))
    model.eval()
    preds = []
    with torch.no_grad():
        bs = 8
        if is_temporal:
            a, b = torch.from_numpy(t1), torch.from_numpy(t2)
            for i in range(0, len(a), bs):
                ai, bi = a[i:i + bs].to(device), b[i:i + bs].to(device)
                preds.append(model(ai, bi).argmax(1).cpu().numpy())
        else:
            x = np.concatenate([t1, t2, np.abs(t1 - t2)], 1)
            for i in range(0, len(x), bs):
                xb = torch.from_numpy(x[i:i + bs]).to(device)
                preds.append(model(xb).argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    # predicted mangrove presence at t1: stable_mangrove (0) or loss (1)
    pred_mng = (p == 0) | (p == 1)
    return pred_mng, y


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    seed = args.seed
    out = {}
    fig, axes = plt.subplots(len(REGIONS), 3, figsize=(8.0, 2.9 * len(REGIONS) + 1.3))
    letters = "abc"
    col_titles = ["GMW 2020 S2 mask\n(independent source)",
                  "GMW time-series 2019\n(training labels)",
                  "Proposed model\npredicted $t_1$ presence"]
    for row, (region, cfg) in enumerate(REGIONS.items()):
        meta = json.loads((PROCESSED_ROOT / region / "metadata.json").read_text())
        grid = (meta["grid"][0], meta["grid"][1])
        transform = from_origin(meta["grid_origin"][0], meta["grid_origin"][1], 10.0, 10.0)
        m2020 = to_grid(GMW2020_REGION(region), grid, transform).astype(bool)
        lbl1 = to_grid(PROCESSED_ROOT / region / "labels_t1.tif", grid, transform).astype(bool)
        lbl2 = to_grid(PROCESSED_ROOT / region / "labels_t2.tif", grid, transform).astype(bool)

        # 1) source agreement (mangrove pixels only, where any label says mangrove)
        any_mng = lbl1 | lbl2
        agree1 = ((m2020 == lbl1) & any_mng).sum() / max(any_mng.sum(), 1)
        agree2 = ((m2020 == lbl2) & any_mng).sum() / max(any_mng.sum(), 1)
        iou1 = (m2020 & lbl1).sum() / max((m2020 | lbl1).sum(), 1)
        iou2 = (m2020 & lbl2).sum() / max((m2020 | lbl2).sum(), 1)

        # 2) model presence vs independent mask (test region pixels)
        data = load_region(region)
        idx = np.asarray(load_split(region, "test"))
        t1f = np.ascontiguousarray(np.transpose(data["t1_feat"][idx] / 10000.0, (0, 3, 1, 2)))
        t2f = np.ascontiguousarray(np.transpose(data["t2_feat"][idx] / 10000.0, (0, 3, 1, 2)))
        y = data["change"][idx]
        # independent mask on patch pixels
        ps = meta["patch_size"]
        step = meta["patch_step"]
        mng2020_patches = []
        for (r, c) in meta["patches"]:
            mng2020_patches.append(m2020[r:r + ps, c:c + ps])
        mng2020_patches = np.stack(mng2020_patches)
        # idx are positional indices into metadata["patches"]; align
        mng2020_test = mng2020_patches[idx]

        row_out = {
            "source_agreement_vs_ts2019": round(float(agree1), 4),
            "source_agreement_vs_ts2024": round(float(agree2), 4),
            "IoU_vs_ts2019": round(float(iou1), 4),
            "IoU_vs_ts2024": round(float(iou2), 4),
            "models": {},
        }
        for m in MODELS:
            r = eval_presence(m, region, "test", t1f, t2f, y, seed)
            if r is None:
                continue
            pred_mng, _ = r
            # compare on test patch pixels
            pm = pred_mng.reshape(len(idx), ps, ps)
            im = mng2020_test
            valid = im | pm  # only where either says mangrove
            if valid.sum() == 0:
                continue
            acc = ((pm == im) & valid).sum() / valid.sum()
            prec = (pm & im).sum() / max(pm.sum(), 1)
            rec = (pm & im).sum() / max(im.sum(), 1)
            row_out["models"][m] = {"presence_acc": round(float(acc), 4),
                                    "precision": round(float(prec), 4),
                                    "recall": round(float(rec), 4)}

        out[region] = row_out

        # figure: m2020 vs lbl1 vs proposed model presence, cropped to the
        # evaluated (mangrove-containing) extent so the two label sources and
        # the model prediction can be compared in the same spatial frame.
        r0, r1, c0, c1 = focus_crop(m2020 | lbl1, meta, idx)
        best = ("unet_cbam_msa" if "unet_cbam_msa" in row_out["models"]
                else (list(row_out["models"])[-1] if row_out["models"] else None))
        axes[row][0].imshow(m2020[r0:r1, c0:c1], cmap=MNG_CMAP, vmin=-0.5, vmax=1.5)
        axes[row][0].axis("off")
        axes[row][1].imshow(lbl1[r0:r1, c0:c1], cmap=MNG_CMAP, vmin=-0.5, vmax=1.5)
        axes[row][1].axis("off")
        if best:
            r = eval_presence(best, region, "test", t1f, t2f, y, seed)
            pm = r[0].reshape(len(idx), ps, ps)
            canvas = np.zeros(grid, dtype=bool)
            idxt = idx.tolist()
            for pos, (rr, cc) in enumerate(meta["patches"]):
                if pos in idxt:
                    canvas[rr:rr + ps, cc:cc + ps] = pm[idxt.index(pos)]
            axes[row][2].imshow(canvas[r0:r1, c0:c1], cmap=MNG_CMAP, vmin=-0.5, vmax=1.5)
            axes[row][2].axis("off")
        else:
            axes[row][2].axis("off")
        if row == 0:
            for c in range(3):
                axes[row][c].set_title(col_titles[c], fontsize=10, pad=5)
        axes[row][0].text(
            -0.08, 0.5, f"({letters[row]}) {REGION_TITLE[region]}",
            transform=axes[row][0].transAxes, rotation=90, va="center",
            ha="center", fontsize=11, fontweight="bold")

    handles = [Patch(facecolor=tuple(CLASS_RGB[0]), edgecolor="#555", lw=0.6,
                     label="mangrove"),
               Patch(facecolor="white", edgecolor="#999", lw=0.8,
                     label="non-mangrove / not evaluated")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.01),
               handletextpad=0.5, columnspacing=1.2)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(FIGURES_ROOT / f"fig7_independent_source_s{seed}.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES_ROOT / f"fig7_independent_source_s{seed}.pdf",
                bbox_inches="tight", facecolor="white")
    (RESULTS_ROOT / f"independent_validation_s{seed}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("fig7 + json saved")


if __name__ == "__main__":
    main()
