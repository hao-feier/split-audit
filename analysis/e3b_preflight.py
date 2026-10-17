"""E3-B pre-flight: matched-arm feasibility + locked candidate list.

Run BEFORE the E3-B sweep. Produces
  results/e3b_preflight.json   (machine-readable, referenced by the run log)
  logs/e3b_selection_lock.txt (the frozen 16/26 selection rule + draw)

Two questions, in order.

(a) MATCHED-ARM FEASIBILITY.  The between-arm contrast needs, for every block
    split, a size- and prevalence-matched control arm drawn from the isolated
    protocol's own zones.  B uses only the n_test>=5 block splits (test sets of
    5-7 patches instead of the 2-6 of the original four), and larger test sets
    make exact prevalence matching harder.  For each candidate we report the
    best achievable |delta prevalence| over ALL C(10, n_te) subsets of the 10
    honest-zone test patches, and count how many clear +-0.2 pp.  The matching
    rule is already "minimise |delta| over the full combination set"
    (make_block_split.matched_honest), so this measures the residual of that
    rule rather than assuming an exact solution exists.

    The within-model contrast does NOT depend on this: it scores one trained
    model on the two test sets, so a prevalence residual cannot invalidate it.
    A poor match degrades the between-arm reading only.

(b) LOCKED SELECTION.  26 candidates have n_test>=5.  We keep the 16 whose
    prevalence-match residual is <= 0.5 pp (deterministic; no random draw) and
    hold the rest out untouched for a robustness check. The rule is written to
    logs/e3b_selection_lock.txt by THIS script, i.e. before any model is
    trained and before any result is seen.
    That is deliberate: the protocol-hygiene claim of this paper cannot be made
    while the split list was chosen after looking at outcomes.

Usage:  python analysis/e3b_preflight.py
"""
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SEED = 20261001          # frozen; do not change without recording a new lock file
N_TARGET = 16            # splits to train
MIN_TEST = 5             # stratified inclusion criterion (test-set size)
TOL_PP = 0.2             # prevalence-match tolerance, percentage points

spec = importlib.util.spec_from_file_location(
    "mbs", ROOT / "experiments" / "make_block_split.py")
mbs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mbs)

meta = json.loads((ROOT / "data" / "processed" / "vietnam" / "metadata.json").read_text())
orgs = list(meta["patches"])
load = lambda s: json.loads((ROOT / "data" / "splits" / f"vietnam_{s}.json").read_text())
tr, va, te = load("train"), load("val"), load("test")
pool = sorted(tr + va + te)
bid, _ = mbs.blocks_of(orgs, pool)
blocks = sorted(set(bid.values()))
n_train_blocks = -(-len(tr) // 4)

# ---- enumerate every feasible, genuinely-interleaved block split ----------
seen = {}
for seed in range(1, 20000):
    a = mbs.try_seed(orgs, bid, blocks, pool, seed, n_train_blocks)
    if a is None:
        continue
    sets = {c: sorted(pool[k] for k in a if a[k] == c) for c in ("train", "val", "test")}
    if (mbs.overlaps(orgs, sets["train"], sets["test"])
            or mbs.overlaps(orgs, sets["train"], sets["val"])
            or mbs.overlaps(orgs, sets["val"], sets["test"])):
        continue
    trs = [orgs[i] for i in sets["train"]]
    tes = [orgs[i] for i in sets["test"]]
    r0, r1 = min(o[0] for o in trs), max(o[0] for o in trs)
    c0, c1 = min(o[1] for o in trs), max(o[1] for o in trs)
    inside = sum(1 for o in tes if r0 <= o[0] <= r1 and c0 <= o[1] <= c1)
    if inside != len(sets["test"]):
        continue                      # test not interleaved -> not a "random" arm
    seen.setdefault(tuple(sets["test"]), (seed, sets))

prev = mbs.patch_prevalence()
cands = []
for tsig, (seed, sets) in sorted(seen.items()):
    n_te = len(sets["test"])
    if n_te < MIN_TEST:
        continue
    target = float(np.mean([prev[i] for i in sets["test"]]))
    best, best_cost = None, None
    for combo in combinations(te, n_te):
        m = float(np.mean([prev[i] for i in combo]))
        cost = abs(m - target)
        if best_cost is None or cost < best_cost:
            best, best_cost = sorted(int(x) for x in combo), cost
    cands.append({
        "block_seed": seed,
        "n_train": len(sets["train"]), "n_val": len(sets["val"]), "n_test": n_te,
        "test_idx": list(sets["test"]),
        "train_idx": sets["train"], "val_idx": sets["val"],
        "block_test_prev_pct": 100 * target,
        "matched_test_idx": best,
        "matched_test_prev_pct": 100 * float(np.mean([prev[i] for i in best])),
        "prev_residual_pp": 100 * best_cost,
        "within_tol": bool(100 * best_cost <= TOL_PP),
    })

cands.sort(key=lambda d: (-d["n_test"], d["block_seed"]))
n_ok = sum(c["within_tol"] for c in cands)
res = {
    "rule": {
        "min_test": MIN_TEST, "n_candidates": len(cands), "n_target": N_TARGET,
        "rng_seed": SEED, "tolerance_pp": TOL_PP,
        "max_prev_residual_pp": 0.5,
        "selection": "DOUBLE stratification, no random draw: keep every "
                     "candidate with n_test >= MIN_TEST and achievable "
                     "prevalence-match residual <= max_prev_residual_pp. "
                     "Zero selection freedom, so no split list can be "
                     "accused of being outcome-informed. RNG seed retained "
                     "only for reproducibility of the enumeration order.",
        "sort_order_is_deterministic": True,
    },
    "feasibility": {
        "n_candidates": len(cands),
        "n_within_tol": n_ok,
        "n_above_tol": len(cands) - n_ok,
        "max_residual_pp": max(c["prev_residual_pp"] for c in cands),
        "median_residual_pp": float(np.median([c["prev_residual_pp"] for c in cands])),
        "matched_rule": "minimise |delta prevalence| over all C(10, n_test) "
                        "subsets of the honest-zone test patches (this is the "
                        "rule already implemented in make_block_split.matched_honest)",
    },
    "candidates": cands,
}

MAXR = 0.5
chosen = [c for c in cands if c['prev_residual_pp'] <= MAXR]
holdout = [c for c in cands if c['prev_residual_pp'] > MAXR]
assert len(chosen) == N_TARGET, f"stratified set is {len(chosen)}, expected {N_TARGET}"
res["selected"] = [c["block_seed"] for c in chosen]
res["holdout"] = [c["block_seed"] for c in holdout]

out = ROOT / "results" / "e3b_preflight.json"
out.write_text(json.dumps(res, indent=1))

lock = ROOT / "logs" / "e3b_selection_lock.txt"
lock.parent.mkdir(exist_ok=True)
lock.write_text(
    "E3-B SPLIT SELECTION LOCK\n"
    "=========================\n"
    "Frozen BEFORE any E3-B model was trained and before any result was seen.\n"
    "Generated by analysis/e3b_preflight.py.\n\n"
    f"stratification      : n_test >= {MIN_TEST} (test-set size)\n"
    f"candidates found    : {len(cands)} distinct test sets\n"
    f"RNG seed            : {SEED}  (retained for reproducibility; "
    f"not used in selection)\n"
    f"selection rule      : keep every candidate with prev_residual_pp <= "
    f"{MAXR} pp (deterministic; no random draw),\n"
    f"                      sorted by (-n_test, block_seed)\n"
    f"trained splits ({N_TARGET}) : "
    f"{', '.join(str(c['block_seed']) for c in chosen)}\n"
    f"holdout splits ({len(holdout)}) : "
    f"{', '.join(str(c['block_seed']) for c in holdout)}\n\n"
    "test-set sizes (trained): "
    f"{[c['n_test'] for c in chosen]}\n"
    "test-set sizes (holdout): "
    f"{[c['n_test'] for c in holdout]}\n\n"
    "The holdout is NOT trained on. It exists so that the split list can be\n"
    "shown not to have been chosen on the basis of outcomes.\n")

print(f"candidates with n_test>={MIN_TEST}: {len(cands)}")
print(f"prevalence match within +-{TOL_PP} pp: {n_ok}/{len(cands)}")
print(f"residual |dprev|: median "
      f"{res['feasibility']['median_residual_pp']:.3f} pp, max "
      f"{res['feasibility']['max_residual_pp']:.3f} pp")
print(f"selected ({N_TARGET}): {res['selected']}")
print(f"holdout ({len(holdout)}): {res['holdout']}")
print(f"-> {out}")
print(f"-> {lock}")