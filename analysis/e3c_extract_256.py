r"""E3-C: a full-scale, overlap-free interleaved control.

Why this exists
---------------
E3 could not build the control the paper says it needs. At 512 px on a 256 px
stride, patches overlap by construction, so 0 of 4000 reseeded random draws
achieved zero train/test footprint overlap, and the only feasible form was a
block-interleaved design reduced to 5-6 test patches. The manuscript therefore
reports a direction but explicitly not a magnitude.

The obstruction is the patch geometry, and the patch geometry is a choice. At
PATCH_STEP == PATCH_SIZE the extracted patches are disjoint by construction, so
a *random* assignment of the pool is already overlap-free -- at the full 40/10/10
design, with no reduced scale and no block structure. Halving the patch to 256 px
keeps 66 patches after the mangrove filter (vs 85 at 512 px), which is enough for
40/10/10.

The design
----------
Both arms are overlap-free; they differ only in where the test patches sit.

  arm "scattered" : random 40/10/10 over the disjoint pool. Zero overlap by
                    construction. Test patches interleaved through the region.
  arm "separated" : the same 40 train patches, and test/val patches chosen to
                    maximise the minimum edge distance to the training set. Zero
                    overlap by construction as well.

Training size, validation size and test size are identical across arms, and
pixel overlap is identically zero in both. The only difference is the spatial
arrangement of the evaluation patches, which is exactly the level-2 residual
proximity question that E3-B could only answer at reduced scale.

Note what this does NOT need: a buffer band. A buffer exists to suppress shared
footprints; when the pool is disjoint there are no shared footprints to suppress,
so discarding patches would only cost statistical power. Distance-based
selection buys the same separation without discarding anything.

Writes data/splits/e3c_256.json.
"""
import json
import os
import sys

import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from mangrove.config import (MAX_MANGROVE_FRAC, MIN_MANGROVE_FRAC,  # noqa
                             PATCH_SIZE)

REGION = "vietnam"
SIZE = 256                      # px at 10 m -> 2.56 km
STEP = 256                      # == SIZE, so patches are disjoint
N_TRAIN, N_VAL, N_TEST = 40, 10, 10
N_DRAWS = 5
SEED = 20261002
OUT = os.path.join(ROOT, "data", "splits", "e3c_256.json")
PX_KM = 0.01


def _load(path):
    with rasterio.open(path) as s:
        return s.read(1)


def _fracs(m, y, x, size):
    w = m[y:y + size, x:x + size]
    v, c = np.unique(w, return_counts=True)
    d = dict(zip(v.tolist(), c.tolist()))
    n = w.size
    # 4-class change labels: 0 stable mangrove, 1 loss, 2 gain, 3 stable bg
    return ((d.get(0, 0) + d.get(1, 0) + d.get(2, 0)) / n,
            (d.get(1, 0) + d.get(2, 0)) / n)


def build_pool():
    P = os.path.join(ROOT, "data", "processed", REGION)
    ch = _load(os.path.join(P, "change_labels.tif"))
    H, W = ch.shape
    pool = []
    for y in range(0, H - SIZE + 1, STEP):
        for x in range(0, W - SIZE + 1, STEP):
            mf, cf = _fracs(ch, y, x, SIZE)
            if MIN_MANGROVE_FRAC <= mf <= MAX_MANGROVE_FRAC:
                pool.append({"y": int(y), "x": int(x),
                             "mangrove_frac": round(float(mf), 4),
                             "change_frac": round(float(cf), 5)})
    return ch.shape, pool


def share_pixels(a, b):
    """True iff two patches share at least one pixel.

    Patches at (y, x) with side SIZE share pixels iff |dy| < SIZE AND |dx| < SIZE.
    With STEP == SIZE every grid offset is a multiple of SIZE, so two DISTINCT
    grid patches can only satisfy this when they are the same patch. Adjacent
    patches touch (edge distance 0) without sharing a pixel, which is why edge
    distance must not be used as an overlap test.
    """
    return abs(a["y"] - b["y"]) < SIZE and abs(a["x"] - b["x"]) < SIZE


def edge_distance_km(a, b):
    """Gap between patch boundaries in km (0 for touching patches)."""
    dy = abs(a["y"] - b["y"])
    dx = abs(a["x"] - b["x"])
    return max(0, max(dy, dx) - SIZE) * PX_KM


def greedy_farthest(pool_by_idx, train_idx, n_pick, forbidden):
    """Pick n_pick patches maximising the distance to the nearest TRAIN patch.

    Greedy max-min: repeatedly take the candidate whose distance to the training
    set is largest. Deterministic given the pool order, so the split list is
    reproducible without an extra RNG stream.
    """
    chosen = []
    avail = [i for i in pool_by_idx if i not in forbidden]
    while len(chosen) < n_pick and avail:
        best, best_d = None, -1.0
        for i in avail:
            d = min(edge_distance_km(pool_by_idx[i], pool_by_idx[t])
                    for t in train_idx)
            # tie-break on index so the choice is deterministic
            if d > best_d or (d == best_d and best is not None and i < best):
                best, best_d = i, d
        chosen.append(best)
        avail.remove(best)
    return chosen


def main():
    shape, pool = build_pool()
    n = len(pool)
    print(f"disjoint pool at {SIZE} px: {n} patches (grid {shape})")
    print(f"  (512 px on 256 px stride, the current geometry, keeps 85 and is "
          f"{'overlapping' if PATCH_SIZE > 256 else 'disjoint'})")
    if n < N_TRAIN + N_VAL + N_TEST:
        print(f"  INFEASIBLE: need {N_TRAIN + N_VAL + N_TEST}, have {n}")
        sys.exit(1)

    idx = {i: p for i, p in enumerate(pool)}
    rng = np.random.default_rng(SEED)
    draws = []
    for d in range(N_DRAWS):
        perm = rng.permutation(n)
        train = sorted(int(i) for i in perm[:N_TRAIN])
        rest = [int(i) for i in perm[N_TRAIN:]]
        # scattered arm: test/val are the first of the remainder
        sc_val = sorted(rest[:N_VAL])
        sc_test = sorted(rest[N_VAL:N_VAL + N_TEST])
        # separated arm: same train, but test/val taken to be far from train
        forb = set(train)
        sep_val = greedy_farthest(idx, train, N_VAL, forb)
        forb |= set(sep_val)
        sep_test = greedy_farthest(idx, train, N_TEST, forb)

        def stats(name, tr, va, te):
            dtr = [edge_distance_km(idx[a], idx[b])
                   for a in tr for b in te]
            return {"arm": name, "train": tr, "val": va, "test": te,
                    "min_edge_km": round(float(min(dtr)), 3),
                    "median_edge_km": round(float(np.median(dtr)), 3),
                    "max_edge_km": round(float(max(dtr)), 3),
                    "mean_edge_km": round(float(np.mean(dtr)), 3)}

        sc = stats("scattered", train, sc_val, sc_test)
        sep = stats("separated", train, sep_val, sep_test)
        # the decisive property: zero shared pixels in BOTH arms, tested with
        # the proper overlap predicate rather than edge distance
        for a in (sc, sep):
            a["shared_pixels"] = sum(
                1 for t in a["test"] for u in a["train"]
                if share_pixels(idx[t], idx[u]))
        draws.append({"draw": d, "scattered": sc, "separated": sep})
        print(f"  draw {d}: scattered min edge {sc['min_edge_km']:.2f} km "
              f"(shared px {sc['shared_pixels']}) | "
              f"separated min edge {sep['min_edge_km']:.2f} km "
              f"(shared px {sep['shared_pixels']})")

    out = {
        "region": REGION, "patch_size": SIZE, "patch_step": STEP,
        "disjoint_by_construction": STEP >= SIZE,
        "design": "40/10/10 in both arms; zero pixel overlap in both; the arms "
                  "differ only in the spatial arrangement of the test patches",
        "n_pool": n, "n_draws": N_DRAWS, "seed": SEED,
        "pool": pool, "draws": draws,
        "verified": {
            "all_arms_zero_shared_pixels": all(
                a["shared_pixels"] == 0
                for d in draws for a in (d["scattered"], d["separated"])),
            "scattered_min_edge_km": [d["scattered"]["min_edge_km"] for d in draws],
            "separated_min_edge_km": [d["separated"]["min_edge_km"] for d in draws],
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nzero shared pixels in every arm: "
          f"{out['verified']['all_arms_zero_shared_pixels']}")
    print(f"min edge distance  scattered: "
          f"{out['verified']['scattered_min_edge_km']}")
    print(f"min edge distance  separated: "
          f"{out['verified']['separated_min_edge_km']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()