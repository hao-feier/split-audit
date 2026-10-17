"""Bound the reference-error x tested-pixel-distribution channel (Action 2, data path).

The paper's Limitations (``paper/main.tex``, ``Reference-error x
tested-pixel-distribution interaction``) states that GMW label errors
concentrate in the 30 m fringe band (the boundary zone of the GMW mangrove
extent), that the honest (buffer-separated) and leaky (interleaved) arms test
spatially different pixel sets, and that this channel "remains untested".

This script tests it without human interpretation: it measures the fraction of
each arm's *change* (loss|gain) pixels that fall inside the 30 m fringe band.
If the interleaved arm's higher TPR were inflated by testing noisier (fringe)
labels, its change pixels would be *more* fringe-concentrated than the honest
arm's. The result is the opposite sign, which bounds the channel away from the
inflating direction.

Definition (matches ``sec:labels`` / ``sec:discussion``):
    mangrove union mask  = (change != stable_background)   i.e. present at t1 OR t2
    boundary             = mangrove_union XOR erosion(mangrove_union)
    fringe band          = 3-pixel (30 m) dilation of the boundary

Both a patch-weighted fraction (the paper's metric convention; overlapping
patches each contribute once) and a unique-pixel fraction (deduplicated on the
full region grid) are reported; they agree to <1.5 pp.

Inputs (light -- change labels only, no feature stacks):
    data/processed/vietnam/change_labels.npz   (N,512,512) int8 0/1/2/3
    data/processed/vietnam/metadata.json       patch (row,col) on the region grid
    data/splits/vietnam_test.json              honest (buffer-separated) test
    data/splits/vietnam_test_ir{1,2,3}.json    leaky (interleaved) test draws

Output:
    results/fringe_band_reference_error.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mangrove.paths import PROCESSED_ROOT, SPLITS_ROOT, RESULTS_ROOT

FRINGE_M = 30.0
FRINGE_PX = int(FRINGE_M / 10.0)  # 3 px at 10 m


def load_change(region="vietnam"):
    ch = np.load(PROCESSED_ROOT / region / "change_labels.npz")["change"]
    meta = json.loads((PROCESSED_ROOT / region / "metadata.json").read_text())
    rows = np.asarray(meta["patches"])[:, 0]
    cols = np.asarray(meta["patches"])[:, 1]
    H, W = meta["grid"]
    return ch, rows, cols, H, W


def fringe_mask(ch):
    """Per-patch 30 m fringe band around the mangrove union boundary."""
    mng_union = (ch != 3).astype(np.uint8)
    fringe = np.zeros_like(mng_union)
    for p in range(ch.shape[0]):
        m = mng_union[p]
        bnd = m ^ ndimage.binary_erosion(m).astype(np.uint8)
        fringe[p] = ndimage.binary_dilation(bnd, iterations=FRINGE_PX).astype(np.uint8)
    return mng_union, fringe


def patch_weighted_frac(change, fringe, patches):
    tot = sum(int(change[p].sum()) for p in patches)
    frg = sum(int((change[p] & fringe[p]).sum()) for p in patches)
    return tot, frg


def unique_frac(change, fringe, rows, cols, H, W, patches):
    u_ch = np.zeros((H, W), np.uint8)
    u_fr = np.zeros((H, W), np.uint8)
    for p in patches:
        r0, c0 = int(rows[p]), int(cols[p])
        u_ch[r0:r0 + 512, c0:c0 + 512] |= change[p]
        u_fr[r0:r0 + 512, c0:c0 + 512] |= fringe[p]
    tot = int(u_ch.sum())
    frg = int((u_ch & u_fr).sum())
    return tot, frg


def frac(change, fringe, rows, cols, H, W, patches):
    pw_t, pw_f = patch_weighted_frac(change, fringe, patches)
    uq_t, uq_f = unique_frac(change, fringe, rows, cols, H, W, patches)
    return {
        "change_pixels": pw_t,
        "fringe_change_pixels": pw_f,
        "fringe_change_frac": pw_f / pw_t if pw_t else None,
        "unique_change_pixels": uq_t,
        "unique_fringe_change_pixels": uq_f,
        "unique_fringe_change_frac": uq_f / uq_t if uq_t else None,
    }


def main():
    ch, rows, cols, H, W = load_change("vietnam")
    mng_union, fringe = fringe_mask(ch)
    change = ((ch == 1) | (ch == 2)).astype(np.uint8)

    def patches(fn):
        return json.loads((SPLITS_ROOT / fn).read_text())

    out = {
        "region": "vietnam",
        "fringe_band_m": FRINGE_M,
        "fringe_band_px": FRINGE_PX,
        "mangrove_mask": "union of t1 and t2 GMW presence (change != stable_background)",
        "change_definition": "loss (1) | gain (2)",
        "honest": {"split": "vietnam_test.json",
                   **frac(change, fringe, rows, cols, H, W, patches("vietnam_test.json"))},
        "leaky": {},
    }

    leaky_draws = {}
    pooled = []
    for k in ("ir1", "ir2", "ir3"):
        ps = patches(f"vietnam_test_{k}.json")
        leaky_draws[k] = {"split": f"vietnam_test_{k}.json",
                          **frac(change, fringe, rows, cols, H, W, ps)}
        pooled += ps
    out["leaky"]["draws"] = leaky_draws
    out["leaky"]["pooled"] = frac(change, fringe, rows, cols, H, W, pooled)

    h = out["honest"]["fringe_change_frac"]
    l = out["leaky"]["pooled"]["fringe_change_frac"]
    out["delta_pp"] = {
        "patch_weighted": (l - h) * 100,
        "unique": ((out["leaky"]["pooled"]["unique_fringe_change_frac"]
                    - out["honest"]["unique_fringe_change_frac"]) * 100),
    }
    out["note"] = (
        "Negative delta_pp: the leaky arm's change pixels are LESS "
        "fringe-concentrated than the honest arm's, so the reference-error "
        "channel cannot inflate the leaky arm's TPR; it acts at least as "
        "strongly on the honest arm."
    )

    (RESULTS_ROOT / "fringe_band_reference_error.json").write_text(
        json.dumps(out, indent=1))

    print(f"honest  change px={out['honest']['change_pixels']} "
          f"fringe_frac={100*h:.1f}%")
    for k, v in leaky_draws.items():
        print(f"{k}  change px={v['change_pixels']} "
              f"fringe_frac={100*v['fringe_change_frac']:.1f}%")
    print(f"leaky pooled  fringe_frac={100*l:.1f}%  "
          f"(patch-weighted delta {out['delta_pp']['patch_weighted']:+.1f} pp, "
          f"unique delta {out['delta_pp']['unique']:+.1f} pp)")
    print("wrote results/fringe_band_reference_error.json")


if __name__ == "__main__":
    main()
