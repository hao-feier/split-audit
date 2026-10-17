"""Plot fig_distance_response from results/distance_response.json.

Per-patch binary change F1 vs the patch's minimum edge distance to the 40-patch
training zone, for the five main models (10 buffer + 10 test-zone patches, val
excluded). Points: individual patches (mean over the five seeds); the dashed
line joins the per-distance-bin means.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import analysis.figstyle as fs  # noqa: E402
from mangrove.paths import RESULTS_ROOT  # noqa: E402

fs.apply_style()
MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]
LETTERS = "abcde"

d = json.loads((RESULTS_ROOT / "distance_response.json").read_text())
pdist = {int(k): v for k, v in d["patch_distances_km"].items()}

fig, axes = plt.subplots(1, 5, figsize=(11.6, 2.7), sharey=True)
for ax, m, letter in zip(axes, MODELS, LETTERS):
    # per-patch points (5-seed mean per patch)
    xs, ys = [], []
    for binrec in d["per_model"][m]:
        for idx, f1 in binrec["per_patch"].items():
            xs.append(pdist[int(idx)])
            ys.append(float(f1))
    ax.scatter(xs, ys, s=20, color=fs.MODEL_COLOR[m], alpha=0.8,
               edgecolor="k", linewidth=0.4, zorder=3)
    # bin-mean line
    bx = [(b["bin_km"][0] + b["bin_km"][1]) / 2.0 for b in d["per_model"][m]]
    by = [b["F1ch_mean"] for b in d["per_model"][m]]
    ax.plot(bx, by, "--", color=fs.TEXT, lw=1.1, marker="o", ms=4, zorder=2)
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0.0, 0.62)
    fs.strip_top_right(ax)
    fs.panel_title(ax, letter, fs.MODEL_LABEL[m], fs=9)
axes[0].set_ylabel(r"Change $F_1$ (per patch)", fontsize=9)
for ax in axes:
    ax.set_xlabel("Distance to training zone (km)", fontsize=8)
    ax.tick_params(labelsize=8)
# single legend
handles = [
    plt.Line2D([0], [0], marker="o", ls="", color=fs.MODEL_COLOR["unet"],
               ms=6, label="per-patch (5-seed mean)"),
    plt.Line2D([0], [0], ls="--", color=fs.TEXT, lw=1.1, marker="o", ms=4,
               label="bin mean"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.03, 1, 1])
fs.save(fig, "fig_distance_response")
print("fig_distance_response saved")
