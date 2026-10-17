"""Shared publication-quality matplotlib style for all paper figures.

Single source of truth so every figure in the manuscript is visually
consistent (font, sizes, palette, spines, export). Apply with
``figstyle.apply_style()`` and export with ``figstyle.save(fig, name)``
which writes both a vector ``<name>.pdf`` and a 300-DPI ``<name>.png``.

Palettes are the Okabe-Ito colour-blind-safe set; the four change-class
colours follow the conventional green/red/blue/grey scheme of the paper
caption but use colour-blind-safe hues.
"""
from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# Figure output directory, resolved relative to this file so that the figure
# scripts work without importing the `mangrove` package (keeps the figure-only
# checkout in analysis/ + results/ self-contained).  MANGROVE_FIGURES_ROOT
# overrides it for tests.
ROOT = Path(__file__).resolve().parent.parent
FIGURES_ROOT = Path(os.environ.get("MANGROVE_FIGURES_ROOT", str(ROOT / "figures")))

# --------------------------------------------------------------------------
# Page geometry -- the single source of truth for figure canvas width
# --------------------------------------------------------------------------
# main.tex includes every figure at `width=\textwidth`.  If a figure's canvas is
# this wide, LaTeX scales it by exactly 1.0, so the point sizes written in
# make_figures.py ARE the point sizes on the page.  Get this wrong and every
# figure is silently shrunk: the earlier value of 6.46 in was wrong for this
# document class, so the layouts rendered at 6.46/5.4567 = 1.18x too large and
# LaTeX shrank every one of them by 0.845 -- an 8 pt label printed at 6.8 pt.
#
# Measured from the document itself, not assumed:
#   \documentclass[journal=remotesensing,manuscript=article,submit,pdftex,
#                  moreauthors]{Definitions/mdpi}
#   -> \textwidth = 394.35522 pt = 5.4567 in   (1 in = 72.27 TeX pt)
# analysis/figure_qa.py re-measures this by compiling that probe and fails if
# the constant below has drifted from the real layout.
PRINT_W_IN = 5.4567

# --------------------------------------------------------------------------
# rcParams
# --------------------------------------------------------------------------
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.titlepad": 6,
    "axes.labelsize": 9,
    "axes.labelpad": 5,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.6,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    # NOT "tight": tight cropping re-expands the canvas around any artist that
    # sits outside the axes (long footnotes, colourbars placed pastx=1), which
    # silently broke the figsize == \textwidth invariant that make_figures.py
    # relies on.  With "standard" the saved width is exactly figsize[0], so
    # LaTeX scales by 1.0 and the point sizes written in make_figures.py are
    # the point sizes on the page.  Keep all content inside the canvas.
    "savefig.bbox": "standard",
    "pdf.fonttype": 42,   # TrueType (embed)
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "mathtext.fontset": "dejavusans",
}


def apply_style():
    """Apply the publication rcParams and return the module for palettes."""
    mpl.rcParams.update(RC)
    return globals()


# --------------------------------------------------------------------------
# Palettes (Okabe-Ito colour-blind-safe)
# --------------------------------------------------------------------------
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
             "#D55E00", "#CC79A7", "#000000"]

# Deep models, in the paper's Exp order. The proposed model is the hero.
MODEL_ORDER = ["unet", "unet_cbam", "unet_msa", "unet_cbam_msa", "deeplabv3plus",
               "temporal_unet", "temporal_unet_gn", "fc_siam",
               "changeformer", "deeplabv3plus_pre"]
MODEL_LABEL = {
    "unet": "U-Net",
    "unet_cbam": "U-Net+CBAM",
    "unet_cbam_msa": "U-Net+CBAM+MSA",
    "deeplabv3plus": "DeepLabv3+",
    "temporal_unet": "Temporal U-Net",
    "temporal_unet_gn": "Temporal U-Net (GN)",
    "unet_msa": "U-Net+MSA",
    "fc_siam": "FC-Siam",
    "changeformer": "ChangeFormer-style",
    "deeplabv3plus_pre": "DeepLabv3+ (pretrained)",
    "random_forest": "Random Forest",
    "svm_rbf": "SVM (RBF)",
}
MODEL_COLOR = {
    "unet": "#0072B2",          # blue
    "unet_cbam": "#56B4E9",     # sky
    "unet_cbam_msa": "#D55E00", # vermillion  <-- proposed / hero
    "deeplabv3plus": "#009E73", # bluish green
    "temporal_unet": "#CC79A7", # reddish purple
    "temporal_unet_gn": "#000000",  # black (R2 control)
    "unet_msa": "#56B4E9",     # sky (ablation, shares CBAM family)
    "changeformer": "#A6A6A6",  # grey (Transformer baseline)
    "deeplabv3plus_pre": "#2A9D8F",  # dark teal (pretrained control)
    "fc_siam": "#E69F00",      # orange (SOTA baseline)
    "random_forest": "#999999",
    "svm_rbf": "#B0B0B0",
}
PROPOSED = "unet_cbam_msa"

# Evaluation regions.
REGIONS = ["within", "beilunhe", "zhanjiang"]
REGION_LABEL = {"within": "Within-region", "beilunhe": "Beilunhe",
                "zhanjiang": "Zhanjiang"}
REGION_COLOR = {"within": "#0072B2", "beilunhe": "#E69F00",
                "zhanjiang": "#009E73"}

# Four change classes (green / red / blue / grey per the paper caption).
CLASS_ORDER = ["stable_mangrove", "loss", "gain", "stable_background"]
CLASS_SHORT = {"stable_mangrove": "Stable\nmangrove", "loss": "Loss",
               "gain": "Gain", "stable_background": "Stable\nbackground"}
CLASS_COLOR = {
    "stable_mangrove": "#009E73",
    "loss": "#D55E00",
    "gain": "#0072B2",
    "stable_background": "#CCCCCC",
}

# Split zones (fig1).
ZONE_COLOR = {"train": "#0072B2", "val": "#E69F00", "test": "#009E73",
              "buffer": "#BBBBBB"}

GREY = "#888888"
TEXT = "#222222"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def class_rgb_array():
    """(4,3) float array of the four class colours, index = class id."""
    import numpy as np
    return np.array([_hex2rgb(CLASS_COLOR[c]) for c in CLASS_ORDER])


def _hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def strip_top_right(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_title(ax, letter, title, fs=9, color=TEXT, pad=6, loc="left"):
    """Left-aligned panel title that starts with a bold '(a)' marker.

    Keeping the marker inside the title (rather than floating it outside the
    axes) is what makes the label and the title read as one unit, and it stays
    aligned no matter how many lines the title wraps to.

    `title` may contain newlines.  `letter` is given bare ("a", "b", ...).
    """
    letter = str(letter).strip().strip("()").lower()
    ax.set_title(r"$\bf{(%s)}$ %s" % (letter, title), fontsize=fs, color=color,
                 pad=pad, loc=loc)


def save(fig, name, outdir=None, png=True):
    """Save a figure as vector PDF (+ 300-DPI PNG) into FIGURES_ROOT."""
    out = Path(outdir) if outdir else FIGURES_ROOT
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"{name}.pdf"
    fig.savefig(pdf)
    if png:
        fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
    return pdf


def symmetrize_margins(fig, axes):
    """Balance the outer left/right margins of a row of panels.

    `tight_layout` only guarantees that nothing overlaps; it does not balance
    the margins, so a long y-label on the first panel leaves the figure
    visibly left-heavy.  Measure both outer edges and shift/scale the axes so
    the whitespace matches.  Run AFTER tight_layout, and only for panels that
    do not use aspect='equal' (resizing would distort them).

    Returns the (left, right) margins in figure fractions.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W = fig.get_figwidth() * fig.dpi
    left = min(ax.get_window_extent(r).x0 for ax in axes)
    right = W - max(ax.get_window_extent(r).x1 for ax in axes)
    span = W - left - right
    if span <= 0:
        return left / W, right / W
    # Symmetric margin: the LARGER of the two measured gaps is not enough on
    # its own -- the y tick labels of the leftmost panel stick out further
    # still, so keep min(left, right) plus that overhang and squeeze the
    # content into the middle.  Labels are re-measured, so iterate twice.
    lab_l = min((t.get_window_extent(r).x0
                 for ax in axes for t in ax.get_yticklabels()
                 if t.get_text()), default=left)
    lab_r = W - max((t.get_window_extent(r).x1
                     for ax in axes for t in ax.get_yticklabels()
                     if t.get_text()), default=W - right)
    m = max(min(left, right), min(lab_l, lab_r)) / W
    m = min(m, 0.45)
    k = (1.0 - 2.0 * m) * W / span
    for ax in axes:
        p0 = ax.get_position()
        ax.set_position([m + (p0.x0 * W - left) * k / W, p0.y0,
                         p0.width * k, p0.height])
    return m, m
