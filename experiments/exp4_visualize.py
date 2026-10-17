"""Exp4: result visualizations (PLAN §7, 9) — publication quality.

Improvements over the first pass:
  * colour-blind-safe class palette shared with every other figure
    (analysis/figstyle.py) instead of a raw red/green scheme;
  * panel letters (a)/(b), a shared four-class legend, and 300-DPI output;
  * fig2/fig3 combine BOTH independent test regions (Beilunhe, Zhanjiang)
    as panels (a)/(b), matching the manuscript captions;
  * fig3 adds a scale bar + north arrow.

Figures:
  fig2_segmentation_comparison.png  RGB t1/t2 + GMW reference + the three
                                    U-Net variants, panels (a)/(b)
  fig3_change_map.png               full-region predicted change map of the
                                    proposed model, panels (a)/(b)
  fig4_failure_cases.png            worst-prediction patches (lowest change
                                    IoU), two per region

Inference needs a GPU:  python experiments/exp4_visualize.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mangrove.config import CHANGE_CLASSES
from mangrove.datasets import load_region, load_split
from mangrove.models import MODEL_REGISTRY, pick_device
from mangrove.paths import RESULTS_ROOT, FIGURES_ROOT, PROCESSED_ROOT, ckpt_path
import analysis.figstyle as fs

fs.apply_style()

# colour-blind-safe class palette, index == class id (from the shared style)
CLASS_RGB = fs.class_rgb_array()            # (4, 3) float in [0, 1]
CLASS_NAMES = list(CHANGE_CLASSES)          # four class names in id order

TEST_REGIONS = ["beilunhe", "zhanjiang"]
REGION_TITLE = {"beilunhe": "Beilunhe", "zhanjiang": "Zhanjiang"}
FIG2_MODELS = ["unet", "unet_cbam", "unet_cbam_msa"]
FIG2_MODEL_LABEL = {"unet": "U-Net", "unet_cbam": "U-Net+CBAM",
                    "unet_cbam_msa": "U-Net+CBAM+MSA"}
PROPOSED = "unet_cbam_msa"
TEMPORAL_MODELS = {"temporal_unet"}
RES_M = 10.0          # S2 L2A ground sampling distance


def run_inference(model_name, region, seed=2026):
    """Re-run inference on a region's test patches; returns (pred, y) or None."""
    import torch
    ck = ckpt_path(model_name, seed)
    if not ck.exists():
        return None
    data = load_region(region)
    idx = np.asarray(load_split(region, "test"))
    t1 = np.ascontiguousarray(np.transpose(data["t1_feat"][idx] / 10000.0, (0, 3, 1, 2)))
    t2 = np.ascontiguousarray(np.transpose(data["t2_feat"][idx] / 10000.0, (0, 3, 1, 2)))
    y = data["change"][idx]
    is_temporal = model_name in TEMPORAL_MODELS
    in_ch = 9 if is_temporal else 27
    device = pick_device()
    model = MODEL_REGISTRY[model_name](in_ch=in_ch, out_ch=4).to(device)
    model.load_state_dict(torch.load(ck, map_location="cpu"))
    model.eval()
    preds = []
    bs = 16
    with torch.no_grad():
        if is_temporal:
            for i in range(0, len(t1), bs):
                a = torch.from_numpy(t1[i:i + bs]).to(device)
                b = torch.from_numpy(t2[i:i + bs]).to(device)
                preds.append(model(a, b).argmax(1).cpu().numpy())
        else:
            x = np.concatenate([t1, t2, np.abs(t1 - t2)], 1)
            for i in range(0, len(x), bs):
                xb = torch.from_numpy(x[i:i + bs]).to(device)
                preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds), y


def mask_image(mask):
    """(H, W) integer class map -> (H, W, 3) float image in the palette."""
    return CLASS_RGB[mask.astype(int)]


def region_stretch(d1, d2, pct=(2, 98)):
    """Per-channel 2nd-98th percentile stretch over a region's t1+t2 test
    pixels, shared by both epochs so any colour shift between panels is real
    change rather than a stretch artifact."""
    m = np.concatenate([d1[..., [3, 2, 1]].reshape(-1, 3),
                        d2[..., [3, 2, 1]].reshape(-1, 3)], 0)
    lo = np.percentile(m, pct[0], axis=0)
    hi = np.percentile(m, pct[1], axis=0)
    return lo, np.maximum(hi, lo + 1e-3)


def rgb(x, lo, hi):
    """False-colour composite (NIR in the red channel) from /10000
    reflectance under a per-channel
    stretch; display order red=B08(3), green=B04(2), blue=B03(1)."""
    img = x[:, :, [3, 2, 1]].astype(np.float32)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def show(ax, image):
    ax.imshow(np.asarray(image, dtype=float))
    ax.axis("off")


def class_legend(fig, ncol=4, ne=False):
    LEGEND_NAMES = {"stable_mangrove": "Stable mangrove", "loss": "Loss",
                    "gain": "Gain", "stable_background": "Stable background"}
    handles = [Patch(facecolor=CLASS_RGB[i], edgecolor="#555", lw=0.6,
                     label=LEGEND_NAMES.get(CLASS_NAMES[i], CLASS_NAMES[i])
                     ) for i in range(4)]
    if ne:
        handles.append(Patch(facecolor="white", edgecolor="#999", lw=1.0,
                             label="not evaluated"))
    fig.legend(handles=handles, loc="lower center",
               ncol=ncol + (1 if ne else 0), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.01), handletextpad=0.5, columnspacing=1.2)


def rep_patch(y):
    """The test patch with the most change (most informative for comparison)."""
    ch = ((y == 1) | (y == 2)).sum((1, 2))
    return int(np.argmax(ch))


def worst_patches(y, pred, k=3):
    """Test patches with real change, ordered by lowest binary-change IoU."""
    ch_t = (y == 1) | (y == 2)
    ch_p = (pred == 1) | (pred == 2)
    inter = np.logical_and(ch_t, ch_p).sum((1, 2))
    union = np.logical_or(ch_t, ch_p).sum((1, 2))
    iou = np.divide(inter, np.maximum(union, 1))
    iou = np.where(ch_t.sum((1, 2)) > 200, iou, np.nan)
    order = np.argsort(iou)                 # nan sorts last
    return [int(i) for i in order if np.isfinite(iou[i])][:k]


def _meta(region):
    return json.loads((PROCESSED_ROOT / region / "metadata.json").read_text())


def mosaic_img(region, pred, idx):
    """Tile the test-patch predictions at their geographic positions on a white
    canvas (white = not evaluated), then crop to the patch extent. Returns the
    cropped (h, w, 3) float image and the region metadata. Because test patches
    are extracted only where mangrove is present, the evaluated area is a small
    cluster; the crop keeps the map focused on it."""
    meta = _meta(region)
    H, W = meta["grid"]
    ps = meta["patch_size"]
    img = np.ones((H, W, 3), np.float32)               # white = not evaluated
    pos_to_row = {int(p): k for k, p in enumerate(idx)}
    r0, r1, c0, c1, anycov = H, 0, W, 0, False
    for pi, (r, c) in enumerate(meta["patches"]):
        if pi in pos_to_row:
            img[r:r + ps, c:c + ps] = CLASS_RGB[pred[pos_to_row[pi]].astype(int)]
            anycov = True
            r0, r1 = min(r0, r), max(r1, r + ps)
            c0, c1 = min(c0, c), max(c1, c + ps)
    if not anycov:
        return img, meta
    m = 48
    r0, r1 = max(0, r0 - m), min(H, r1 + m)
    c0, c1 = max(0, c0 - m), min(W, c1 + m)
    return img[r0:r1, c0:c1], meta


def add_scalebar(ax, cw, ch, km=None):
    """Scale bar (bottom-left) + north arrow (top-right) in the axis's pixel
    coords, sized to the cropped view (cw x ch)."""
    if km is None:
        km = max(1, round(0.12 * cw * RES_M / 1000.0))
    px = km * 1000.0 / RES_M
    x0, y0 = cw * 0.05, ch * 0.90
    ax.plot([x0, x0 + px], [y0, y0], color="black", lw=2.5, solid_capstyle="butt")
    ax.plot([x0, x0], [y0, y0 + ch * 0.015], color="black", lw=2.5)
    ax.plot([x0 + px, x0 + px], [y0, y0 + ch * 0.015], color="black", lw=2.5)
    ax.text(x0 + px / 2, y0 - ch * 0.02, f"{km:g} km", ha="center", va="top",
            fontsize=9, color="black", bbox=dict(fc="white", ec="none", pad=1.5))
    nx, ny = cw * 0.94, ch * 0.09
    ax.annotate("", xy=(nx, ny - ch * 0.055), xytext=(nx, ny + ch * 0.03),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8))
    ax.text(nx, ny + ch * 0.035, "N", ha="center", va="bottom", fontsize=10,
            color="black")


def savefig(fig, name, dpi=300):
    p = FIGURES_ROOT / f"{name}.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--regions", nargs="*", default=TEST_REGIONS)
    args = ap.parse_args()
    seed = args.seed

    # ---- one inference pass per region, for the models the figures need --
    preds, labels, disp, idxs, stretch = {}, {}, {}, {}, {}
    for region in args.regions:
        data = load_region(region)
        idx = np.asarray(load_split(region, "test"))
        idxs[region] = idx
        labels[region] = data["change"][idx]
        disp[region] = (data["t1_feat"][idx] / 10000.0,
                        data["t2_feat"][idx] / 10000.0)
        stretch[region] = region_stretch(*disp[region])
        preds[region] = {}
        for m in FIG2_MODELS:                    # includes the proposed model
            r = run_inference(m, region, seed)
            if r is not None:
                preds[region][m] = r[0]
        print(f"  {region}: inference for {sorted(preds[region])}")

    regions = [r for r in args.regions
               if preds[r].get(PROPOSED) is not None]
    letters = "ab"

    # ================= fig2: segmentation comparison (regions x 6) ========
    n = len(regions)
    fig, axes = plt.subplots(n, 6, figsize=(8.0, 1.45 * n + 1.25))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["False colour 2019", "False colour 2024",
                 "GMW change\n(reference)"] + \
                 [FIG2_MODEL_LABEL[m] for m in FIG2_MODELS]
    for row, region in enumerate(regions):
        y = labels[region]
        d1, d2 = disp[region]
        si = rep_patch(y)                       # one informative patch/region
        sl, sh = stretch[region]
        show(axes[row, 0], rgb(d1[si], sl, sh))
        show(axes[row, 1], rgb(d2[si], sl, sh))
        show(axes[row, 2], mask_image(y[si]))
        for k, m in enumerate(FIG2_MODELS):
            if m in preds[region]:
                show(axes[row, 3 + k], mask_image(preds[region][m][si]))
        if row == 0:
            for c in range(6):
                axes[row, c].set_title(col_titles[c], fontsize=10, pad=5)
        axes[row, 0].text(
            -0.08, 0.5, f"({letters[row]}) {REGION_TITLE[region]}",
            transform=axes[row, 0].transAxes, rotation=90, va="center",
            ha="center", fontsize=11, fontweight="bold")
    class_legend(fig)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    savefig(fig, "fig2_segmentation_comparison")

    # ================= fig3: full-region change maps (regions x 2) ========
    fig, axes = plt.subplots(1, len(regions), figsize=(8.0, 5.0))
    if len(regions) == 1:
        axes = [axes]
    for col, region in enumerate(regions):
        sub, meta = mosaic_img(region, preds[region][PROPOSED], idxs[region])
        ax = axes[col]
        show(ax, sub)
        add_scalebar(ax, sub.shape[1], sub.shape[0])
        ax.set_title(f"({letters[col]}) {REGION_TITLE[region]}",
                     fontsize=11, fontweight="bold", pad=6)
    class_legend(fig, ne=True)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    savefig(fig, "fig3_change_map")

    # ================= fig4: failure cases (3 per region x 4) =============
    rows = []
    for region in regions:
        bad = worst_patches(labels[region], preds[region][PROPOSED], 2)
        for si in bad:
            rows.append((region, si))
    if rows:
        fig, axes = plt.subplots(len(rows), 4,
                                 figsize=(8.0, 2.0 * len(rows) + 1.25))
        for r, (region, si) in enumerate(rows):
            d1, d2 = disp[region]
            sl, sh = stretch[region]
            show(axes[r, 0], rgb(d1[si], sl, sh))
            show(axes[r, 1], rgb(d2[si], sl, sh))
            show(axes[r, 2], mask_image(labels[region][si]))
            show(axes[r, 3], mask_image(preds[region][PROPOSED][si]))
        # column titles on the top row only
        for c, t in enumerate(["False colour 2019", "False colour 2024",
                               "GMW change (ref.)", "U-Net+CBAM+MSA"]):
            axes[0, c].set_title(t, fontsize=10, pad=5)
        # region label to the left of the first row of each region block
        seen = set()
        for r, (region, si) in enumerate(rows):
            if region not in seen:
                seen.add(region)
                letter = letters[regions.index(region)]
                axes[r, 0].text(
                    -0.08, 0.5, f"({letter}) {REGION_TITLE[region]}",
                    transform=axes[r, 0].transAxes, rotation=90,
                    va="center", ha="center", fontsize=11, fontweight="bold")
        class_legend(fig)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        savefig(fig, "fig4_failure_cases")


if __name__ == "__main__":
    main()
