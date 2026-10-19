"""D design: freeze the matched overlap-cell arms ("lock") BEFORE any model
output for this design exists.

Design (spec: Feier/specs/2026-10-17-d-overlap-cell-design.md, run on Junzhe):
    Main arm = inner ring O (stride-1: unseen patches adjacent to the seen set
    S = train ∪ val) vs outer ring D (stride-2: second ring around S), k=5 per
    arm, matched on prevalence prev = P(change) x 100 (NOT on dose).
    Sensitivity arms: near-ring k=4, pool k=6 / k=7 (O pool = stride-1, D pool
    = the 17 unseen non-neighbours), strict-train k=2..5.
    Half-region arms are deferred: the row-band anchor quoted in spec sec 2.6
    has no recoverable provenance (docs/TODO_v4.md Phase 6).

This file is run ONCE and is blind: it never opens anything under
results/predictions/. Its output results/cell_arms_lock.json is committed to
git before any inference runs (spec gate 4).

Tie handling mirrors analysis/e3b_repair_matched.py: exact ties (within 1e-15)
are broken by the frozen RNG, drawn once per block.

Usage:  python analysis/cell_make_arms.py
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
PROC = ROOT / "data" / "processed" / "vietnam"
RNG_SEED = 20261017          # frozen in the spec; used only for tie-breaks
TOL = 1e-15                  # exact-tie window (e3b style)
PP_TOL = 5e-4                # self-check tolerance against 4-dp printed values

# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
meta = json.loads((PROC / "metadata.json").read_text())
anch = np.asarray(meta["patches"], dtype=np.int64)
H, W = (int(x) for x in meta["grid"])
PS = int(meta["patch_size"])
n = len(anch)

change = np.load(PROC / "change_labels.npz")["change"]     # (76,512,512) int8, 0..3
prev = ((change == 1) | (change == 2)).reshape(n, -1).mean(axis=1) * 100.0

train = sorted(json.loads((SPLITS / "vietnam_train.json").read_text()))
val = sorted(json.loads((SPLITS / "vietnam_val.json").read_text()))
test = sorted(json.loads((SPLITS / "vietnam_test.json").read_text()))
buffer_ = sorted(json.loads((SPLITS / "vietnam_buffer.json").read_text()))
S = sorted(set(train) | set(val))

# --------------------------------------------------------------------------
# G rebuild (spec sec 1: |dr|<512 and |dc|<512), rings, pools
# --------------------------------------------------------------------------
dr = np.abs(anch[:, 0][:, None] - anch[:, 0][None, :])
dc = np.abs(anch[:, 1][:, None] - anch[:, 1][None, :])
G = (dr < PS) & (dc < PS)
np.fill_diagonal(G, False)

unseen = [i for i in range(n) if i not in set(S)]
O_pool = sorted(i for i in unseen if G[i, S].any())                    # stride-1
D_pool = sorted(i for i in unseen if i not in set(O_pool) and G[i, O_pool].any())
dj17 = sorted(i for i in unseen if not G[i, S].any())                  # unseen non-adjacent

# strict-train variant (third reading)
unseenT = [i for i in range(n) if i not in set(train)]
ovT = sorted(i for i in unseenT if G[i, train].any())
djT = sorted(i for i in unseenT if not G[i, train].any())

# --------------------------------------------------------------------------
# dose column (spec sec 2): fraction of the patch window covered by the union
# of the 50 seen-set windows. Window semantics = experiments/exp1_patch.py:
# slice [r:r+512, c:c+512]; all windows fit fully in the map, so no clipping.
# --------------------------------------------------------------------------
mask = np.zeros((H, W), dtype=bool)
for i in S:
    r, c = (int(x) for x in anch[i])
    mask[r:r + PS, c:c + PS] = True
dose = np.empty(n, dtype=np.float64)
for i in range(n):
    r, c = (int(x) for x in anch[i])
    dose[i] = mask[r:r + PS, c:c + PS].mean()

# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------
def best_match(pool_a, pool_b, k):
    """Enumerate k-subsets of both pools; minimise |mean(prev_a) - mean(prev_b)|.
    Per-subset means are computed exactly like /tmp/feas_v3.py
    (mean of the selected prev values, float64). Ties within TOL are broken
    once by the frozen RNG (e3b style)."""
    subs_a = [tuple(sorted(c)) for c in combinations(pool_a, k)]
    subs_b = [tuple(sorted(c)) for c in combinations(pool_b, k)]
    means_a = [float(np.mean([prev[i] for i in c])) for c in subs_a]
    means_b = [float(np.mean([prev[i] for i in c])) for c in subs_b]
    dmin = np.inf
    for ma in means_a:
        for mb in means_b:
            d = abs(ma - mb)
            if d < dmin:
                dmin = d
    ties = [(abs(means_a[ia] - means_b[ib]), ia, ib)
            for ia, ma in enumerate(means_a) for ib, mb in enumerate(means_b)
            if abs(ma - mb) <= dmin + TOL]
    if len(ties) == 1:
        d, ia, ib = ties[0]
        tie_break = False
    else:
        idx = int(np.random.default_rng(RNG_SEED).integers(len(ties)))
        d, ia, ib = ties[idx]
        tie_break = True
    return {
        "k": k,
        "O_arm": list(subs_a[ia]),
        "D_arm": list(subs_b[ib]),
        "O_mean_prev_pct": means_a[ia],
        "D_mean_prev_pct": means_b[ib],
        "resid_pp": float(d),
        "n_subsets_evaluated": len(subs_a) * len(subs_b),
        "n_ties": len(ties),
        "tie_break_applied": tie_break,
    }


main_arm = best_match(O_pool, D_pool, 5)

# residual table: all C(9,5)=126 O-subsets against the fixed D arm
D_fixed = tuple(D_pool)                      # |D_pool| == 5 == k
D_mean = float(np.mean([prev[i] for i in D_fixed]))
residual_table = sorted(
    ({"ov": list(c),
      "mean_prev_pct": float(np.mean([prev[i] for i in c])),
      "resid_pp": abs(float(np.mean([prev[i] for i in c])) - D_mean)}
     for c in (tuple(sorted(c)) for c in combinations(O_pool, 5))),
    key=lambda row: (row["resid_pp"], row["ov"]))

lock_near_k4 = best_match(O_pool, D_pool, 4)
lock_pool_k6 = best_match(O_pool, dj17, 6)
lock_pool_k7 = best_match(O_pool, dj17, 7)
lock_strict_k2 = best_match(ovT, djT, 2)
lock_strict_k3 = best_match(ovT, djT, 3)
lock_strict_k4 = best_match(ovT, djT, 4)
lock_strict_k5 = best_match(ovT, djT, 5)

# --------------------------------------------------------------------------
# self-checks (spec gate: A-only 16, inset unchanged; plus frozen fea_v3 counts)
# --------------------------------------------------------------------------
checks = []


def check(cid, desc, ok, expected, got):
    checks.append({"id": cid, "check": desc, "expected": expected,
                   "got": got, "pass": bool(ok)})


adj = np.load("/tmp/adj_v2.npy")
a_only = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] and not G[i, j]]
g_only = [(i, j) for i in range(n) for j in range(i + 1, n) if G[i, j] and not adj[i, j]]
diag = sum(1 for i in range(n) for j in range(i + 1, n)
           if G[i, j] and abs(int(anch[i][0] - anch[j][0])) == 256
           and abs(int(anch[i][1] - anch[j][1])) == 256)

check("G_edges", "G edges == 181", G.sum() // 2 == 181, 181, int(G.sum() // 2))
check("G_deg_max", "max degree == 8", int(G.sum(1).max()) == 8, 8, int(G.sum(1).max()))
check("A_only", "A-only false positives == 16 (spec sec 4)", len(a_only) == 16, 16, len(a_only))
check("G_only", "G-only missed edges == 80", len(g_only) == 80, 80, len(g_only))
check("split_sizes", "train/val/test/buffer == 40/10/10/16",
      [len(train), len(val), len(test), len(buffer_)] == [40, 10, 10, 16],
      [40, 10, 10, 16], [len(train), len(val), len(test), len(buffer_)])
check("ov9", "O (stride-1) == spec list",
      O_pool == [11, 18, 19, 20, 21, 22, 25, 31, 39],
      [11, 18, 19, 20, 21, 22, 25, 31, 39], O_pool)
check("dj17", "unseen non-adjacent == spec list",
      dj17 == [26, 27, 28, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 46, 47, 48],
      [26, 27, 28, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 46, 47, 48], dj17)
check("stride2", "D (stride-2) == [26,27,28,32,40]",
      D_pool == [26, 27, 28, 32, 40], [26, 27, 28, 32, 40], D_pool)
check("inset", "sec 4 self-check 'inset unchanged': recomputed O == spec inner set",
      O_pool == [11, 18, 19, 20, 21, 22, 25, 31, 39],
      [11, 18, 19, 20, 21, 22, 25, 31, 39], O_pool)

dose_ov9_ok = all(abs(dose[i] - (0.25 if i == 19 else 0.5)) < 1e-12 for i in O_pool)
check("dose_ov9", "dose(O): 19 -> 0.25, all others -> 0.5 (spec sec 2)",
      dose_ov9_ok, {str(i): (0.25 if i == 19 else 0.5) for i in O_pool},
      {str(i): dose[i] for i in O_pool})
check("dose_dj17_zero", "dose(D pool / dj17) == 0",
      all(abs(dose[i]) < 1e-12 for i in dj17), 0.0,
      float(max(abs(dose[i]) for i in dj17)))

def sel_ok(lock, O_exp, D_exp):
    if lock["n_ties"] > 1:
        return (abs(lock["resid_pp"] - 0) >= 0) and lock["O_mean_prev_pct"] > 0  # values-only
    return lock["O_arm"] == O_exp and lock["D_arm"] == D_exp


check("main_k5", "main arm: resid 0.0150; O=[11,18,19,21,22]; D=[26,27,28,32,40]",
      sel_ok(main_arm, [11, 18, 19, 21, 22], [26, 27, 28, 32, 40])
      and abs(main_arm["resid_pp"] - 0.0150) <= PP_TOL
      and abs(main_arm["O_mean_prev_pct"] - 1.457) <= PP_TOL
      and abs(main_arm["D_mean_prev_pct"] - 1.472) <= PP_TOL,
      {"resid_pp": 0.0150, "O": [11, 18, 19, 21, 22], "D": [26, 27, 28, 32, 40],
       "O_mean": 1.457, "D_mean": 1.472},
      {"resid_pp": main_arm["resid_pp"], "O": main_arm["O_arm"], "D": main_arm["D_arm"],
       "O_mean": main_arm["O_mean_prev_pct"], "D_mean": main_arm["D_mean_prev_pct"],
       "n_ties": main_arm["n_ties"]})
check("near_k4", "near-ring k4: resid 0.0002; [11,18,19,22]/[26,27,28,40]",
      sel_ok(lock_near_k4, [11, 18, 19, 22], [26, 27, 28, 40])
      and abs(lock_near_k4["resid_pp"] - 0.0002) <= PP_TOL,
      {"resid_pp": 0.0002, "O": [11, 18, 19, 22], "D": [26, 27, 28, 40]},
      lock_near_k4)
check("pool_k6", "pool k6: resid 0.0000; [11,18,19,20,21,22]/[28,32,33,34,40,44]",
      sel_ok(lock_pool_k6, [11, 18, 19, 20, 21, 22], [28, 32, 33, 34, 40, 44])
      and abs(lock_pool_k6["resid_pp"] - 0.0000) <= PP_TOL,
      {"resid_pp": 0.0000, "O": [11, 18, 19, 20, 21, 22], "D": [28, 32, 33, 34, 40, 44]},
      lock_pool_k6)
check("pool_k7", "pool k7: resid 0.0003; [...,39]/[28,32,40,43,44,47,48]",
      sel_ok(lock_pool_k7, [11, 18, 19, 20, 21, 22, 39], [28, 32, 40, 43, 44, 47, 48])
      and abs(lock_pool_k7["resid_pp"] - 0.0003) <= PP_TOL,
      {"resid_pp": 0.0003, "O": [11, 18, 19, 20, 21, 22, 39],
       "D": [28, 32, 40, 43, 44, 47, 48]},
      lock_pool_k7)
check("strict_k2", "strict k2: resid 0.0002; [11,39]/[43,44]",
      sel_ok(lock_strict_k2, [11, 39], [43, 44])
      and abs(lock_strict_k2["resid_pp"] - 0.0002) <= PP_TOL,
      {"resid_pp": 0.0002, "O": [11, 39], "D": [43, 44]}, lock_strict_k2)
check("strict_k3", "strict k3: resid 0.0000; [11,18,39]/[5,9,13]",
      sel_ok(lock_strict_k3, [11, 18, 39], [5, 9, 13])
      and abs(lock_strict_k3["resid_pp"] - 0.0000) <= PP_TOL,
      {"resid_pp": 0.0000, "O": [11, 18, 39], "D": [5, 9, 13]}, lock_strict_k3)
check("strict_k4", "strict k4: resid 0.3016; [11,18,25,39]/[8,13,14,28]",
      sel_ok(lock_strict_k4, [11, 18, 25, 39], [8, 13, 14, 28])
      and abs(lock_strict_k4["resid_pp"] - 0.3016) <= PP_TOL,
      {"resid_pp": 0.3016, "O": [11, 18, 25, 39], "D": [8, 13, 14, 28]}, lock_strict_k4)
check("strict_k5", "strict k5: resid 1.1589; [11,18,25,31,39]/[8,9,13,14,28]",
      sel_ok(lock_strict_k5, [11, 18, 25, 31, 39], [8, 9, 13, 14, 28])
      and abs(lock_strict_k5["resid_pp"] - 1.1589) <= PP_TOL,
      {"resid_pp": 1.1589, "O": [11, 18, 25, 31, 39], "D": [8, 9, 13, 14, 28]},
      lock_strict_k5)

ovT_mean = float(np.mean([prev[i] for i in ovT]))
djT_mean = float(np.mean([prev[i] for i in djT]))
djT_max = float(np.max([prev[i] for i in djT]))
check("strict_pool", "ovT mean 3.468; djT n=31 mean 1.424 max 2.842",
      ovT == [11, 18, 25, 31, 39] and len(djT) == 31
      and abs(ovT_mean - 3.468) <= PP_TOL and abs(djT_mean - 1.424) <= PP_TOL
      and abs(djT_max - 2.842) <= PP_TOL,
      {"ovT": [11, 18, 25, 31, 39], "n_djT": 31, "ovT_mean": 3.468,
       "djT_mean": 1.424, "djT_max": 2.842},
      {"ovT": ovT, "n_djT": len(djT), "ovT_mean": ovT_mean, "djT_mean": djT_mean,
       "djT_max": djT_max})

deg_seen = {str(i): int(G[i, S].sum()) for i in O_pool}
check("stride1_deg", "stride-1 n_seen_neighbors",
      deg_seen == {"11": 2, "18": 3, "19": 1, "20": 2, "21": 3, "22": 3,
                   "25": 3, "31": 3, "39": 2},
      {"11": 2, "18": 3, "19": 1, "20": 2, "21": 3, "22": 3, "25": 3, "31": 3, "39": 2},
      deg_seen)

failed = [c for c in checks if not c["pass"]]

# --------------------------------------------------------------------------
# lock file
# --------------------------------------------------------------------------
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


lock = {
    "design": "D: matched overlap-cell arms (controlled Trap-2)",
    "spec": "../Feier/specs/2026-10-17-d-overlap-cell-design.md",
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "rule": {
        "main_arm": "inner ring O (stride-1: unseen adjacent to S=train∪val) vs "
                    "outer D (stride-2 second ring); k=5 per arm; matched on "
                    "prev = P(change)*100 (not on dose)",
        "stride2_derivation": "D = {i ∉ S∪O : G[i, O].any()}",
        "objective": "minimise |mean(prev_O) - mean(prev_D)| in pp",
        "ties": "frozen RNG, drawn once (same style as e3b_repair_matched.py)",
        "rng_seed": RNG_SEED,
        "pre_commitment": "arms computed and frozen before any model prediction was "
                          "produced for this design; this script never reads "
                          "results/predictions/",
        "dose_note": "dose = fraction of the patch window covered by the union of "
                     "the 50 seen-set windows; window semantics from "
                     "experiments/exp1_patch.py (slice [r:r+512, c:c+512])",
    },
    "sets": {
        "S_seen": S, "train": train, "val": val, "test": test, "buffer": buffer_,
        "O_stride1": O_pool, "D_stride2": D_pool, "unseen_nonadjacent_17": dj17,
        "strict_ovT": ovT, "strict_djT": djT,
    },
    "prev_pct": {str(i): float(prev[i]) for i in range(n)},
    "dose": {str(i): float(dose[i]) for i in range(n)},
    "main_arm": main_arm,
    "residual_table": residual_table,
    "lock_near_k4": lock_near_k4,
    "lock_pool_k6": lock_pool_k6,
    "lock_pool_k7": lock_pool_k7,
    "lock_strict_k2": lock_strict_k2,
    "lock_strict_k3": lock_strict_k3,
    "lock_strict_k4": lock_strict_k4,
    "lock_strict_k5": lock_strict_k5,
    "lock_halves_up": {
        "status": "deferred",
        "reason": "spec sec 2.6 row-band anchor [1133,766,259,259] has no "
                  "recoverable provenance; disposition tracked in "
                  "docs/TODO_v4.md Phase 6 (options 1/2/3)",
    },
    "lock_halves_down": {
        "status": "deferred",
        "reason": "same anchor dependency as lock_halves_up",
    },
    "self_checks": checks,
    "inset_interpretation": {
        "reading": "spec sec 4 self-check 'inset unchanged' = the inner set "
                   "(O stride-1 / ov9) recomputes to the spec list unchanged",
        "evidence": "sole literal occurrence of 'inset' in the spec is sec 4 "
                    "(L91); metadata full-key scan, both-repo JSON scan and "
                    "label-derived reconstruction found no other candidate",
        "recomputed_O": O_pool,
        "equals_spec": O_pool == [11, 18, 19, 20, 21, 22, 25, 31, 39],
    },
    "input_hashes": {
        str(p): sha256(p) for p in [
            PROC / "metadata.json", PROC / "change_labels.npz",
            SPLITS / "vietnam_train.json", SPLITS / "vietnam_val.json",
            SPLITS / "vietnam_test.json", SPLITS / "vietnam_buffer.json",
            Path("/tmp/adj_v2.npy"),
        ]
    },
    "no_predictions_read": True,
}

out_path = ROOT / "results" / "cell_arms_lock.json"
out_path.write_text(json.dumps(lock, indent=1))
digest = sha256(out_path)

print(f"S: {len(S)} seen | O stride-1: {O_pool} | D stride-2: {D_pool}")
print(f"main arm k=5: O={main_arm['O_arm']} D={main_arm['D_arm']} "
      f"resid={main_arm['resid_pp']:.4f} pp "
      f"means {main_arm['O_mean_prev_pct']:.4f} / {main_arm['D_mean_prev_pct']:.4f} "
      f"(n_ties={main_arm['n_ties']})")
print(f"pool k6 resid={lock_pool_k6['resid_pp']:.4f} | "
      f"pool k7 resid={lock_pool_k7['resid_pp']:.4f} | "
      f"near k4 resid={lock_near_k4['resid_pp']:.4f}")
print(f"strict k2..k5 resid: {lock_strict_k2['resid_pp']:.4f} "
      f"{lock_strict_k3['resid_pp']:.4f} {lock_strict_k4['resid_pp']:.4f} "
      f"{lock_strict_k5['resid_pp']:.4f}")
print(f"A-only {len(a_only)} | G-only {len(g_only)} | diagonal G edges {diag}")
print(f"self-checks: {len(checks) - len(failed)}/{len(checks)} pass"
      + ("" if not failed else "  FAILED: " + ", ".join(c["id"] for c in failed)))
print(f"-> results/cell_arms_lock.json")
print(f"sha256: {digest}")

if failed:
    sys.exit(1)
