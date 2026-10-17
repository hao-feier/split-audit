"""E3-B analysis: overlap-free block-interleaved contrast, 16 stratified splits.

PRE-REGISTERED INTERPRETATION BANDS
------------------------------------
Written before any E3-B model was trained (the split list was frozen in
logs/e3b_selection_lock.txt by analysis/e3b_preflight.py, which ran first). The
bands below are fixed now so that the wording cannot be chosen after the numbers
are seen. Fill in the numbers; do not move the boundaries.

  BAND A -- "adequately powered null"
    condition: 95% CI of the between-arm contrast CONTAINS 0, AND the CI upper
               bound is < 20 pp.
    reading:   residual proximity, with zero shared pixels, does not reproduce an
               inflation of the headline size at this scale, and the interval is
               now tight enough to say so.
    action:    WITHDRAW the residual mechanism hint from the title. The headline
               becomes a statement about the split protocol only. E3 is reported
               as a bounded null, not as "underpowered".

  BAND B -- "partially reproduced at reduced scale"
    condition: 95% CI EXCLUDES 0, AND the point estimate is < 15 pp.
    reading:   some inflation survives without shared pixels, but it is a
               fraction of the 29-42 pp overlapping-split effect.
    action:    rewrite conclusion 3 to the two-part form: overlap is necessary
               for the full effect, residual proximity contributes a minority
               part. Do NOT restore a mechanism claim.

  BAND C -- "mechanism story must be rewritten"
    condition: 95% CI EXCLUDES 0 AND the point estimate is comparable to the
               overlapping-split headline (>= 20 pp).
    reading:   zero-overlap interleaving reproduces the effect, so shared pixel
               footprint is not necessary and the overlap framing in the title is
               wrong.
    action:    retitle and rebuild Sections 4-6 around proximity rather than
               overlap. This is the outcome that would invalidate the current
               framing, and it is why the band is written down first.

  Band A is the expected outcome given v4 (between-arm -2.2 pp, within-model
  +6.0 pp, both CIs spanning 0), but expectation is not evidence and the band
  assignment is mechanical: it is decided by the computed CI and estimate alone.

Two contrasts are reported, and they answer different questions:
  WITHIN-MODEL: one trained model scored on the block test set vs the matched
    test set. Training is held exactly fixed; only the test arrangement varies.
    This is the cleaner of the two and is the primary reading.
  BETWEEN-ARM: block-trained scored on the block test set, minus matched-trained
    scored on the matched test set. Changes training and test arrangement
    together. Secondary.

NOTE on what the matched arm does and does not remove. Both contrasts compare F1
on two different test sets, so a prevalence difference between those two sets
enters BOTH of them; prevalence matching is what limits that, and the locked
split set keeps the residual at <= 0.461 pp (mean 0.167 pp) against a ~1.3-1.7%
base rate. A within-model contrast is NOT immune to a test-set case-mix
difference -- it is immune only to the TRAINING-side confound. Residuals are
reported per split in the output so this can be checked rather than assumed.

Inference: split seed is the unit. Intervals are percentile bootstrap over the
16 split seeds, resampling whole splits (all architectures and run seeds within
a split move together), because split is the dominant variance component and
the 16 test sets are the independent units. Clustering architectures within a
split is what makes this honest.

Run:  MANGROVE_RESULTS_ROOT=results/uniform python analysis/e3b_analyze.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from aggregate_results import binary_change  # noqa: E402

RESULTS = Path(os.environ.get("MANGROVE_RESULTS_ROOT", ROOT / "results" / "uniform"))
SPLIT_META = json.loads((ROOT / "results" / "e3b_splits.json").read_text())
SEEDS = [2026, 2027]
# The three architectures of the pre-specified E3-B design. Split 62's output
# directory also holds legacy v4 runs (5 architectures) because those runs were
# reused; they are NOT part of E3-B and must not enter its per-model statistics.
DESIGN_ARCHS = {"unet", "unet_cbam_msa", "temporal_unet"}
N_BOOT = 10000
RNG = np.random.default_rng(20261001)

# ---------------------------------------------------------------- load pairs
per_split = {}
for rec in SPLIT_META["splits"]:
    ss = rec["split_seed"]
    entry = {"meta": rec, "within": [], "between": []}
    for blk_p in sorted((RESULTS / f"block_blk{ss}").glob("metrics_*_s*.json")):
        hon_p = (RESULTS / f"block_hon{ss}") / blk_p.name
        if not hon_p.exists():
            continue
        B = json.loads(blk_p.read_text())
        H = json.loads(hon_p.read_text())
        model = B.get("model")
        if model not in DESIGN_ARCHS:
            continue
        # WITHIN-MODEL: block-trained model, block test set minus matched test
        if B.get("block_test") and B.get("partner_test"):
            f_blk = binary_change(B["block_test"]["confusion_matrix"])["F1_change"]
            f_hon = binary_change(B["partner_test"]["confusion_matrix"])["F1_change"]
            entry["within"].append({"model": model, "seed": B.get("seed"),
                                    "blk": f_blk, "hon": f_hon,
                                    "diff": f_blk - f_hon})
        # BETWEEN-ARM: block-trained on block test, minus hon-trained on hon test
        if H.get("model") not in DESIGN_ARCHS:
            continue
        if B.get("block_test") and H.get("block_test"):
            f_bb = binary_change(B["block_test"]["confusion_matrix"])["F1_change"]
            f_hh = binary_change(H["block_test"]["confusion_matrix"])["F1_change"]
            entry["between"].append({"model": model, "seed": B.get("seed"),
                                     "blk": f_bb, "hon": f_hh,
                                     "diff": f_bb - f_hh})
    per_split[ss] = entry

n_runs = sum(len(v["within"]) for v in per_split.values())
n_splits = len([s for s, v in per_split.items() if v["within"]])
print(f"splits with data: {n_splits}/{len(per_split)} | paired runs: {n_runs}")
if n_splits == 0:
    print("no E3-B results yet")
    sys.exit(0)


def boot_split_level(items):
    """Percentile bootstrap resampling whole splits (split = the unit)."""
    by_ss = {}
    for ss, v in per_split.items():
        d = [it["diff"] * 100 for it in items_for(v, items)]
        if d:
            by_ss[ss] = float(np.mean(d))
    keys = sorted(by_ss)
    vals = np.array([by_ss[k] for k in keys])
    draws = [float(np.mean(RNG.choice(vals, size=len(vals), replace=True)))
             for _ in range(N_BOOT)]
    return (float(np.mean(vals)), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), len(vals),
            [by_ss[k] for k in keys])


def items_for(entry, which):
    return entry[which]


for which, name in (("within", "WITHIN-MODEL"), ("between", "BETWEEN-ARM")):
    mu, lo, hi, n, byss = boot_split_level(which)
    per_model = {}
    for m in sorted({it["model"] for v in per_split.values()
                     for it in v[which]}):
        vals = [it["diff"] * 100 for v in per_split.values()
                for it in v[which] if it["model"] == m]
        per_model[m] = {"mean_pp": float(np.mean(vals)), "n": len(vals)}
    n_pos = sum(1 for x in byss if x > 0)
    print(f"\n=== {name} (split-level bootstrap, {n} splits) ===")
    print(f"  mean {mu:+.1f} pp   95% CI [{lo:+.1f}, {hi:+.1f}]")
    print(f"  per-split means: {[round(x, 1) for x in byss]}")
    print(f"  splits with positive mean: {n_pos}/{n}")
    for m, d in per_model.items():
        print(f"  {m:20} {d['mean_pp']:+6.1f} pp  (n={d['n']})")
    # ---- pre-registered band assignment (mechanical) ----
    contains0 = lo <= 0 <= hi
    if contains0 and hi < 20:
        band = ("A", "adequately powered null -- withdraw the residual mechanism "
                     "hint from the title")
    elif (not contains0) and abs(mu) < 15:
        band = ("B", "partially reproduced at reduced scale -- rewrite conclusion 3 "
                     "as two-part, overlap necessary for the full effect")
    elif (not contains0) and abs(mu) >= 20:
        band = ("C", "comparable to the overlapping-split headline -- retitle and "
                     "rebuild around proximity, not overlap")
    else:
        band = ("?", "CI contains 0 but the upper bound is >= 20 pp: still too wide "
                     "to bind; report the interval and do not claim a null")
    print(f"  --> PRE-REGISTERED BAND {band[0]}: {band[1]}")

out = {
    "design": {
        "splits": n_splits, "paired_runs": n_runs,
        "architectures": sorted({it["model"] for v in per_split.values()
                                 for it in v["within"]}),
        "unit_of_inference": "split seed (whole splits resampled)",
        "n_boot": N_BOOT,
        "scale_caveat": ("block arm runs 18-25 train / 5-6 test patches against "
                         "the main protocol's 40/10/10; a zero-overlap interleaved "
                         "split is not achievable at full scale (0 of 4000 "
                         "permutations). No extrapolation to the full protocol."),
    },
    "contrasts": {},
}
for which, name in (("within", "within_model"), ("between", "between_arm")):
    mu, lo, hi, n, byss = boot_split_level(which)
    per_model = {}
    for m in sorted({it["model"] for v in per_split.values()
                     for it in v[which]}):
        vals = [it["diff"] * 100 for v in per_split.values()
                for it in v[which] if it["model"] == m]
        per_model[m] = {"mean_pp": float(np.mean(vals)), "n": len(vals)}
    contains0 = lo <= 0 <= hi
    if contains0 and hi < 20:
        band, act = "A", "withdraw residual mechanism hint from title"
    elif (not contains0) and abs(mu) < 15:
        band, act = "B", "rewrite conclusion 3 as two-part (overlap necessary)"
    elif (not contains0) and abs(mu) >= 20:
        band, act = "C", "retitle around proximity, not overlap"
    else:
        band, act = "?", "interval too wide to bind; report CI, claim nothing"
    out["contrasts"][name] = {
        "mean_pp": mu, "ci95": [lo, hi], "n_splits": n,
        "per_split_means_pp": byss,
        "splits_positive": sum(1 for x in byss if x > 0),
        "per_model": per_model, "band": band, "action": act,
    }
(ROOT / "results" / "e3b_contrast.json").write_text(json.dumps(out, indent=1))
print("\n-> results/e3b_contrast.json")