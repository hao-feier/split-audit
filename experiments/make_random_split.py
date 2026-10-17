"""Random (spatially-correlated) split of the Vietnam training patches.

Purpose
-------
Control group for the spatial-autocorrelation cost (paper "the OA masks the
true capability" claim). The spatially-isolated split (exp2_split_v1.py) assigns
Vietnam patches to train/val/test by COLUMN with a 2 km buffer, so no train
patch is spatially adjacent to a test patch. Most change-detection papers
instead assign patches RANDOMLY, so train and test patches interleave in
space -> accuracy is inflated by spatial autocorrelation.

This script builds that "leaky" random split WITHOUT changing the patch set:
  * pool  = vietnam_train + vietnam_val + vietnam_test  (the 61 non-buffer patches)
  * sizes = the SAME per-set sizes as the isolated split (40 / 10 / 11)
  * assignment = one fixed-seed random permutation of the pool

The ONLY difference from the isolated split is therefore spatial arrangement
(same patches, same counts, same model, same hyperparameters) -> the gap in
change-class metrics between the two is the cost of ignoring spatial
autocorrelation, with data-size as a controlled constant.

Outputs
-------
  data/splits/vietnam_train_rand.json
  data/splits/vietnam_val_rand.json
  data/splits/vietnam_test_rand.json
  figures/fig_rand_split.png   (zoned vs random assignment, side by side)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys_path_hack = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(sys_path_hack))
from mangrove.config import PATCH_SIZE
from mangrove.paths import PROCESSED_ROOT, SPLITS_ROOT, FIGURES_ROOT

REGION = "vietnam"


def patch_centers(region):
    meta = json.loads((PROCESSED_ROOT / region / "metadata.json").read_text())
    cols, rows = [], []
    for (r, c) in meta["patches"]:
        cols.append((c + PATCH_SIZE / 2) * 10.0)
        rows.append((r + PATCH_SIZE / 2) * 10.0)
    return np.asarray(cols), np.asarray(rows)


def load_split(name):
    return json.loads((SPLITS_ROOT / f"{REGION}_{name}.json").read_text())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the random split assignment (independent of run seed)")
    args = ap.parse_args()
    split_seed = args.seed
    tag = "_rand" if split_seed == 42 else f"_rand{split_seed}"
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    sizes = (len(train), len(val), len(test))
    pool = np.asarray(train + val + test)
    # guard: the three sets must be disjoint and cover the non-buffer patches
    assert len(set(train) & set(val)) == 0 and len(set(train) & set(test)) == 0 \
        and len(set(val) & set(test)) == 0, "isolated split sets overlap"

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(pool)
    t, v, te = perm[:sizes[0]], perm[sizes[0]:sizes[0] + sizes[1]], \
        perm[sizes[0] + sizes[1]:]
    out = {"train": t.tolist(), "val": v.tolist(), "test": te.tolist()}
    for name, arr in out.items():
        (SPLITS_ROOT / f"{REGION}_{name}{tag}.json").write_text(json.dumps(arr))

    # sanity: disjoint, and spatially interleaved (train/test column ranges overlap)
    assert len(set(t) & set(v)) == 0 and len(set(t) & set(te)) == 0 \
        and len(set(v) & set(te)) == 0, "random split sets overlap"
    cols, rows = patch_centers(REGION)
    def _rng(idx):
        return float(np.min(cols[idx])), float(np.max(cols[idx]))
    iso_t = _rng(train); iso_te = _rng(test)
    rnd_t = _rng(t); rnd_te = _rng(te)
    iso_overlap = max(0.0, min(iso_t[1], iso_te[1]) - max(iso_t[0], iso_te[0]))
    rnd_overlap = max(0.0, min(rnd_t[1], rnd_te[1]) - max(rnd_t[0], rnd_te[0]))

    # side-by-side plot: isolated (zoned) vs random assignment
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"train": "green", "val": "orange", "test": "blue"}
    for ax, (title, split_map) in zip(axes, [
        ("Isolated (zoned, 2 km buffer)", {"train": train, "val": val, "test": test}),
        (f"Random (seed={split_seed}, interleaved)", out),
    ]):
        for name, c in colors.items():
            idx = np.asarray(split_map[name])
            ax.scatter(cols[idx], rows[idx], s=28, c=c, label=f"{name} ({len(idx)})")
        ax.set_title(title)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.legend()
    fig.suptitle("Vietnam patch assignment: isolated vs random (same 61 patches, same counts)")
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "fig_rand_split.png", dpi=200)

    print(f"pool={len(pool)} sizes={sizes}")
    print(f"isolated: train col {iso_t[0]:.0f}-{iso_t[1]:.0f} | test col {iso_te[0]:.0f}-{iso_te[1]:.0f} | "
          f"col-overlap={iso_overlap:.0f} m")
    print(f"random:   train col {rnd_t[0]:.0f}-{rnd_t[1]:.0f} | test col {rnd_te[0]:.0f}-{rnd_te[1]:.0f} | "
          f"col-overlap={rnd_overlap:.0f} m")
    print(f"train∩test = {len(set(t) & set(te))} (must be 0); "
          f"train∩val = {len(set(t) & set(v))}; val∩test = {len(set(v) & set(te))}")
    print(f"wrote vietnam_{{train,val,test}}{tag}.json + figures/fig_rand_split.png")


if __name__ == "__main__":
    main()
