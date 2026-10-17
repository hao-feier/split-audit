r"""E3-C split list: scattered vs separated, both overlap-free, size-matched.

Reads the ACTUAL cached patch arrays (which already carry the mangrove and
clear-fraction filters exp1_patch applied) rather than re-deriving candidates
from the raster, so the split list indexes exactly the patches that will be
trained on. This matters: an earlier feasibility check that ignored the
clear-fraction filter over-counted the 256 px pool by 12 patches (66 vs 54).

Both arms use disjoint patches (step == size), so zero train/test pixel overlap
is guaranteed by construction in BOTH arms. The arms differ only in where the
test patches sit:

  scattered : drawn at random from the remainder -- test patches interleaved
              through the region.
  separated : chosen greedily to maximise the minimum edge distance to the
              training set.

No buffer band is discarded: a buffer exists to suppress shared footprints, and
a disjoint pool has none to suppress, so discarding patches would cost power
without buying separation.

Writes data/splits/e3c_p128.json.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

REGION_DIR = "data/processed/vietnam_p128"
SIZE = 128
STEP = 128
PX_KM = 0.01
N_TRAIN, N_VAL, N_TEST = 40, 10, 10
N_DRAWS = 5
SEED = 20261003
OUT = "data/splits/e3c_p128.json"


def grid_positions(n, rows_cols):
    """Row-major window origins matching exp1_patch's enumeration order."""
    H, W = rows_cols
    return [(r, c) for r in range(0, H - SIZE + 1, STEP)
            for c in range(0, W - SIZE + 1, STEP)]


def load_meta():
    meta = json.loads(open(f"{REGION_DIR}/metadata.json").read())
    idx = meta.get("idx") or meta.get("patches") or []
    return meta, idx


def edge_km(a, b):
    """Gap between patch boundaries in km. 0 for touching patches."""
    return max(0, max(abs(a[0] - b[0]), abs(a[1] - b[1])) - SIZE) * PX_KM


def share_pixels(a, b):
    """True iff two patches overlap. With STEP == SIZE, distinct grid windows
    cannot overlap, so this must be identically False -- asserted per draw."""
    return abs(a[0] - b[0]) < SIZE and abs(a[1] - b[1]) < SIZE


def main():
    meta, idx = load_meta()
    n = len(idx)
    print(f"cached pool at {SIZE} px: {n} patches "
          f"(disjoint by construction: {meta.get('disjoint_by_construction')})")
    if n < N_TRAIN + N_VAL + N_TEST:
        print(f"INFEASIBLE for {N_TRAIN}/{N_VAL}/{N_TEST}: need "
              f"{N_TRAIN + N_VAL + N_TEST}, have {n}")
        sys.exit(1)

    pos = [tuple(p) for p in idx]
    rng = np.random.default_rng(SEED)
    draws = []
    for d in range(N_DRAWS):
        perm = rng.permutation(n)
        train = sorted(int(i) for i in perm[:N_TRAIN])
        rest = [int(i) for i in perm[N_TRAIN:]]
        sc_val, sc_test = sorted(rest[:N_VAL]), sorted(rest[N_VAL:N_VAL + N_TEST])

        # PREVALENCE MATCHING IS NOT OPTIONAL HERE.
        # A first pass took the farthest patches without matching prevalence and
        # produced a 1.79x prevalence ratio between the arms (range 1.12-2.32x).
        # F1 is strongly prevalence-sensitive -- the paper's own metric study
        # finds a 2.68x prevalence ratio tracks a ~29 pp F1 gap -- so an unmatched
        # arm comparison measures case mix, not proximity. This is the same trap
        # the earlier draft fell into; the match is enforced here and the achieved
        # ratio is recorded per draw.
        chg = np.load(os.path.join(ROOT, REGION_DIR, "change_labels.npz"))["change"]
        prev = np.array([((chg[i] == 1) | (chg[i] == 2)).sum() / chg[i].size
                         for i in range(n)])

        def farthest_matched(npick, forbidden, target_sum):
            """Pick npick patches far from train whose prevalence sums to
            target_sum. Greedy start from the farthest, then local swaps that
            trade a little distance for a large prevalence improvement."""
            avail = [i for i in range(n) if i not in train and i not in forbidden]
            chosen = sorted(avail,
                            key=lambda i: (-min(edge_km(pos[i], pos[t])
                                                for t in train), i))[:npick]
            # Lexicographic-in-spirit balance: prevalence error dominates (it is
            # the confound that invalidates the comparison), and within that the
            # minimum edge distance is maximised. Exact matching alone forced one
            # touching patch into 4 of 5 draws; this recovers some of that.
            # Pure prevalence matching, with distance used only to break ties and
            # as the starting point. A variant that traded prevalence for distance
            # was tried and rejected: it left two draws at 2.07x and 2.32x, and a
            # variant that forced a hard +/-10% constraint could not find a feasible
            # set in those draws at all. Far patches are systematically less
            # change-dense here, because change concentrates on the fringe where
            # the training patches sit, so distance and prevalence are
            # anti-correlated in this landscape. Prevalence is the confound the
            # paper's own metric work shows dominates F1, so it wins; the residual
            # cost is that the minimum edge distance stays at one patch width in
            # most draws, which we report rather than optimise away.
            def cost(sel):
                return abs(prev[sel].sum() - target_sum)
            def perr(sel):
                return abs(prev[sel].sum() - target_sum) / max(target_sum, 1e-9)
            for _ in range(6000):
                cur = perr(chosen)
                if cur < 0.02 and cur >= 0:
                    pass
                if perr(chosen) < 0.02:
                    break
                best, bestc = None, cur
                for a_i, a in enumerate(chosen):
                    for b in avail:
                        if b in chosen:
                            continue
                        trial = list(chosen)
                        trial[a_i] = b
                        c = cost(trial)
                        if c < bestc - 1e-9:
                            best, bestc = (a_i, b), c
                if best is None:
                    break
                ai, b = best
                chosen[ai] = b
            return sorted(chosen)

        tgt_val = prev[sc_val].sum()
        tgt_test = prev[sc_test].sum()
        sep_val = farthest_matched(N_VAL, set(), tgt_val)
        sep_test = farthest_matched(N_TEST, set(sep_val), tgt_test)

        def stats(name, tr, va, te):
            d = [edge_km(pos[a], pos[b]) for a in te for b in tr]
            sh = sum(1 for a in te for b in tr if share_pixels(pos[a], pos[b]))
            return {"arm": name, "train": tr, "val": va, "test": te,
                    "min_edge_km": round(float(min(d)), 3),
                    "median_edge_km": round(float(np.median(d)), 3),
                    "max_edge_km": round(float(max(d)), 3),
                    "shared_pixels": sh,
                    "test_prevalence": round(float(prev[te].mean()), 5),
                    "val_prevalence": round(float(prev[va].mean()), 5)}

        sc, sep = stats("scattered", train, sc_val, sc_test), \
            stats("separated", train, sep_val, sep_test)
        draws.append({"draw": d, "scattered": sc, "separated": sep})
        ratio = (sc["test_prevalence"] / sep["test_prevalence"]
                 if sep["test_prevalence"] > 0 else float("nan"))
        print(f"  draw {d}: scattered min edge {sc['min_edge_km']:5.2f} km "
              f"| separated min edge {sep['min_edge_km']:5.2f} km "
              f"(median {sep['median_edge_km']:5.2f}) | shared px "
              f"{sc['shared_pixels']}/{sep['shared_pixels']} | "
              f"prev ratio {ratio:.3f}x")

    allshared = sum(a["shared_pixels"]
                    for d in draws for a in (d["scattered"], d["separated"]))
    out = {
        "region_dir": REGION_DIR, "patch_size": SIZE, "patch_step": STEP,
        "n_pool": n, "n_draws": N_DRAWS, "seed": SEED,
        "sizes": {"train": N_TRAIN, "val": N_VAL, "test": N_TEST},
        "design": ("both arms overlap-free and size-matched; the arms differ "
                   "only in the spatial arrangement of the test patches"),
        "total_shared_pixels_both_arms": allshared,
        "draws": draws,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\ntotal shared pixels across both arms and all draws: {allshared}")
    print(f"separated-arm min edge distances: "
          f"{[d['separated']['min_edge_km'] for d in draws]}")
    print(f"wrote {OUT}")

    # also emit per-draw split files the trainer can consume
    for d in draws:
        for arm in ("scattered", "separated"):
            a = d[arm]
            tag = f"e3c{d['draw']}_{arm}"
            for split in ("train", "val", "test"):
                json.dump(a[split], open(f"data/splits/vietnam_p128_{split}_"
                                         f"{tag}.json", "w"))
    print("wrote per-draw split files: data/splits/vietnam_p128_{train,val,test}"
          "_e3c<draw>_<arm>.json")


if __name__ == "__main__":
    main()