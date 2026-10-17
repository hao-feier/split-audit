"""Change-label spatial autocorrelation decay (mechanism figure).

Why
---
The paper's headline is that a spatially-interleaved test set inflates
change-F1 by +26-32pp, and that the inflation is a TEST-set effect
(isolated-test << interleaved-test regardless of the training split). That
claim needs a *mechanism*: it rests on the change field being spatially
autocorrelated, so that a test patch near a training patch shares signal.

This script measures that directly on the raw change labels (no model, no
training, no data-amount confound). For each region it computes, for a ladder
of pixel lags, the fraction of pixel pairs at that lag that are BOTH in the
change classes (loss|gain), normalised by the independence expectation p^2
(where p is the overall change fraction). The result is a
"co-change enrichment" E(L) that is ~1 at independence and >>1 where the
change field is spatially structured. A monotone decay of E(L) to ~1 bounds
the autocorrelation range -- the distance over which a train/test split can
leak -- and justifies the 2 km buffer used in the spatially-independent
protocol.

Metric (defensible, model-free):
    C(L)  = (1/N) * sum_i  m_i * m_{i+L}      # co-change fraction at lag L
    E(L)  = C(L) / p^2                          # enrichment over independence
    p     = mean(m)                             # overall change fraction

Averaged over the four cardinal directions (E/W/N/S) for a direction-agnostic
estimate; identical formula per region.

Outputs
-------
    results/autocorr/change_autocorr.json
    figures/fig10_change_autocorr_decay.{pdf,png}
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from mangrove.paths import RESULTS_ROOT, FIGURES_ROOT
from mangrove.datasets import ChangePatchDataset
import matplotlib
matplotlib.use("Agg")
import figstyle
import matplotlib.pyplot as plt

# pixel lags (1 px = 10 m). 8000 m = 8 km covers well past the 2 km buffer.
LAGS_M = [10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000]
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
REGION_LABEL = {"vietnam": "Vietnam (train)",
                "beilunhe": "Beilunhe (test)",
                "zhanjiang": "Zhanjiang (test)"}
REGION_COLOR = {"vietnam": "#0072B2", "beilunhe": "#E69F00",
                "zhanjiang": "#009E73"}


def load_change_masks(region):
    # .change is the full region array regardless of split idx; use "test"
    # (exists for all three regions: the two independent test regions only
    # write a _test.json split).
    ds = ChangePatchDataset(region, "test", augment=False, concat=True)
    lab = ds.change                                  # (N, H, W) in {0,1,2,3}
    return ((lab == 1) | (lab == 2)).astype(np.float32)   # loss | gain


def region_curve(region):
    masks = load_change_masks(region)                # (N, H, W)
    p = float(masks.mean())
    p2 = p * p
    H, W = masks.shape[-2:]
    lags_px = [int(round(m / 10)) for m in LAGS_M]
    lags_px = [l for l in lags_px if 0 < l < min(H, W)]
    # x and y directions (each +/ - is identical by symmetry), averaged
    def two_dir(masks_arr, lag):
        xs = float((masks_arr[:, :, :-lag] * masks_arr[:, :, lag:]).mean())
        ys = float((masks_arr[:, :-lag, :] * masks_arr[:, lag:, :]).mean())
        return float(np.mean([xs, ys])) / p2
    curve = []
    for m, l in zip(LAGS_M, lags_px):
        curve.append({"lag_m": m, "enrichment": two_dir(masks, l)})
    # autocorrelation range: farthest lag where enrichment still >= 2x
    rng_km = [c["lag_m"] for c in curve if c["enrichment"] >= 2.0]
    return {"region": region, "n_patches": int(masks.shape[0]),
            "change_fraction": p,
            "autocorr_range_km": (max(rng_km) / 1000.0) if rng_km else None,
            "curve": curve}


def main():
    figstyle.apply_style()
    out = {}
    for region in REGIONS:
        r = region_curve(region)
        out[region] = r
        print(f"\n[{region}] change_frac={r['change_fraction']:.4f} "
              f"autocorr_range(>=2x)={r['autocorr_range_km']} km")
        for c in r["curve"]:
            print(f"  {c['lag_m']:>6} m  enrichment={c['enrichment']:7.3f}")

    outdir = RESULTS_ROOT / "autocorr"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "change_autocorr.json").write_text(json.dumps(out, indent=2))

    # ---- figure: enrichment vs distance (log-x), one curve per region ----
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for region in REGIONS:
        c = out[region]["curve"]
        x = [p["lag_m"] for p in c]
        y = [p["enrichment"] for p in c]
        ax.plot(x, y, marker="o", lw=1.6,
                color=REGION_COLOR[region], label=REGION_LABEL[region])
    # reference lines
    ax.axhline(1.0, color="#888888", lw=0.9, ls="--")
    ax.text(LAGS_M[-1], 1.02, "independence", ha="right", va="bottom",
            fontsize=7, color="#666666")
    ax.axhline(2.0, color="#CCCCCC", lw=0.8, ls=":")
    ax.text(LAGS_M[0], 2.02, "2x", ha="left", va="bottom", fontsize=7,
            color="#999999")
    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1000, 8000])
    ax.set_xticklabels(["10 m", "100 m", "1 km", "8 km"])
    ax.set_xlabel("separation between two pixels")
    ax.set_ylabel("change co-occurrence\nenrichment over independence")
    ax.set_xlim(LAGS_M[0] * 0.8, LAGS_M[-1] * 1.25)
    ymax = max(out[r]["curve"][0]["enrichment"] for r in REGIONS)
    ax.set_yscale("log")
    ax.set_ylim(0.1, ymax * 1.4)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title("Spatial autocorrelation of the change label")
    figstyle.save(fig, "fig10_change_autocorr_decay")
    print("\nwrote", outdir / "change_autocorr.json")
    print("wrote", FIGURES_ROOT / "fig10_change_autocorr_decay.png")


if __name__ == "__main__":
    main()
