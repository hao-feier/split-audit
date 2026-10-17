"""E3-B defect-1 repair: redraw the matched-arm test set disjoint from the block
test set.

Defect
------
For 3 of the 16 locked splits (165, 11289, 12557) the matched arm's test set was
IDENTICAL to the block arm's test set, so both contrasts were identically zero on
those splits. The cause is geometric, not accidental: for those three the block
test set lies entirely inside the honest zone (6 of 6 patches shared), so once
those patches are excluded only 4 honest-zone patches remain and a 6-patch
matched set cannot be formed. They are recorded as geometrically infeasible
rather than repaired.

Repair rule (mechanical, blind to results, drawn ONCE -- no re-selection after
seeing outcomes):
    among all C(10, n_test) subsets of the honest-zone test pool that are
    DISJOINT from the block test set, take the one minimising
    |prevalence(matched) - prevalence(block)|; ties broken by the frozen RNG.
This preserves the prevalence-matching property of the original design and adds
only the disjointness constraint that the design was supposed to have.

Honest status of the repair: it is applied to all 13 geometrically feasible
splits, including those that were already disjoint, so it cannot be a
post-hoc adjustment of individual splits. Results before and after are both
reported (see results/e3b_repair_audit.json).

Usage:  python analysis/e3b_repair_matched.py
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
RNG_SEED = 20261001   # same frozen seed as the pre-flight; used only for tie-breaks

import importlib.util
spec = importlib.util.spec_from_file_location(
    "mbs", ROOT / "experiments" / "make_block_split.py")
mbs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mbs)

hon_pool = json.loads((SPLITS / "vietnam_test.json").read_text())
prev = mbs.patch_prevalence()
meta = json.loads((ROOT / "results" / "e3b_splits.json").read_text())

audit, changed, infeasible = [], 0, []
for rec in meta["splits"]:
    ss = rec["split_seed"]
    blk = json.loads((SPLITS / f"vietnam_test_blk{ss}.json").read_text())
    cur = json.loads((SPLITS / f"vietnam_test_hon{ss}.json").read_text())
    n = len(blk)
    disjoint = [c for c in combinations(hon_pool, n) if not (set(c) & set(blk))]
    entry = {"split_seed": ss, "n_test": n,
             "blk_test": list(blk), "matched_before": list(cur),
             "was_identical": set(cur) == set(blk),
             "shared_with_blk_before": len(set(cur) & set(blk)),
             "n_disjoint_subsets": len(disjoint)}
    if not disjoint:
        entry["status"] = "geometrically_infeasible"
        entry["matched_after"] = None
        infeasible.append(ss)
    else:
        target = float(np.mean([prev[i] for i in blk]))
        best, best_cost = None, None
        for combo in disjoint:                     # deterministic order
            m = float(np.mean([prev[i] for i in combo]))
            cost = abs(m - target)
            if best_cost is None or cost < best_cost - 1e-15:
                best, best_cost = sorted(int(x) for x in combo), cost
        # exact ties (several subsets share the same prevalence mean) are broken
        # by the frozen RNG, drawn once
        ties = [sorted(int(x) for x in c) for c in disjoint
                if abs(abs(float(np.mean([prev[i] for i in c])) - target) - best_cost) < 1e-15]
        if len(ties) > 1:
            best = sorted(ties[int(np.random.default_rng(RNG_SEED).integers(len(ties)))])
        entry["status"] = "repaired" if best != cur else "already_disjoint"
        entry["matched_after"] = best
        entry["prev_residual_after_pp"] = 100 * float(
            abs(np.mean([prev[i] for i in best]) - target))
        entry["block_prev_pct"] = 100 * target
        entry["matched_prev_pct"] = 100 * float(np.mean([prev[i] for i in best]))
        if best != cur:
            (SPLITS / f"vietnam_test_hon{ss}.json").write_text(json.dumps(best))
            changed += 1
        else:
            entry["prev_residual_after_pp"] = entry.get("prev_residual_after_pp")
    audit.append(entry)

ok = [a for a in audit if a["status"] != "geometrically_infeasible"]
res = [a["prev_residual_after_pp"] for a in ok if "prev_residual_after_pp" in a]
out = {
    "rule": {"disjointness": "matched test set must not intersect the block test set",
             "objective": "minimise |prevalence(matched) - prevalence(block)|",
             "ties": "frozen RNG, drawn once",
             "rng_seed": RNG_SEED,
             "pre_commitment": "drawn once before any repaired result was inspected; "
                               "no re-selection on the basis of outcomes"},
    "counts": {"splits": len(audit),
               "repaired": sum(a["status"] == "repaired" for a in audit),
               "already_disjoint": sum(a["status"] == "already_disjoint" for a in audit),
               "geometrically_infeasible": len(infeasible)},
    "infeasible_splits": infeasible,
    "post_repair_residual_pp": {"max": max(res), "mean": float(np.mean(res))},
    "audit": audit,
}
(ROOT / "results" / "e3b_repair_audit.json").write_text(json.dumps(out, indent=1))
print(f"repaired (file rewritten): {changed}")
print(f"already disjoint (untouched): {out['counts']['already_disjoint']}")
print(f"geometrically infeasible: {infeasible}")
print(f"post-repair prevalence residual: max {max(res):.3f} pp, "
      f"mean {np.mean(res):.3f} pp")
print("-> results/e3b_repair_audit.json")