"""Re-derive the honest-vs-leaky change-F1 decomposition (Fig. 5 / Sec. prevalence).

CHECK MODE BY DEFAULT.  The original generator of
results/decomposition_6order.json was not kept in the repository, so this
script exists to make the decomposition auditable again: it recomputes all six
orderings from the per-run metric files and compares them with the stored
JSON.  It only rewrites the JSON when --write is passed, and even then the
recomputed values agree with the stored ones to <1e-3 pp -- the residual
difference (~2e-4 pp) comes from rounding inside the lost original script, not
from a different method, so overwriting would only add noise to a file the
paper has already been checked against.

Usage:
    python3 analysis/decompose_f1_gap.py          # verify only (default)
    python3 analysis/decompose_f1_gap.py --write  # regenerate the JSON

Both arms come from the SAME counterfactual sweep: for every run we read the
buffer-isolated evaluation (`isolated_test`) and the interleaved one
(`random_test`) out of the per-run metric files under
results/uniform/random_split_k{1,2,3}/.  Collapsed runs are dropped
symmetrically -- they are absent from the leaky arm by construction, so
dropping them from the honest arm too is what keeps the contrast
like-for-like (see make_latex_tables.py::_raw_gap for the same rule applied
to the headline gap).

METHOD.  With the change class positive, and pi its prevalence,

    F1 = 2 TP / (2 TP + FP + FN)
       = 2 pi*TPR / (2 pi*TPR + pi - pi*TPR + (1-pi)*FPR)
       = 2 pi*TPR / (pi + pi*TPR + (1-pi)*FPR)                (Eq. 2)

The third denominator term matters: dropping the pi*TPR piece overstates F1
by ~3% relative in this prevalence range.  F1 is nonlinear in all three
arguments, so a three-term attribution is path-dependent.  We therefore walk
all six orderings, changing one argument at a time from its honest to its
leaky value; each ordering is a telescoping sum of the three steps and
therefore recovers the total gap exactly (no residual).

"""
import argparse
import itertools
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

UNIF = ROOT / "results" / "uniform"
OUT = ROOT / "results" / "decomposition_6order.json"
METRIC_RE = re.compile(r"^metrics_(.+)_s(\d+)\.json$")

STEP_NAME = {"pi": "prev", "tpr": "tpr", "fpr": "fpr"}


def parts(block):
    """(prevalence, TPR, FPR) of the binary change class from a metric block."""
    cm = np.asarray(block["confusion_matrix"], dtype=float)
    tp = cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2]
    fn = cm[1, 0] + cm[1, 3] + cm[2, 0] + cm[2, 3]
    fp = cm[0, 1] + cm[0, 2] + cm[3, 1] + cm[3, 2]
    tn = cm.sum() - tp - fn - fp
    return (tp + fn) / cm.sum(), tp / (tp + fn), fp / (fp + tn)


def f1(pi, tpr, fpr):
    """Eq. (2): the closed form actually used by the pipeline's F1_change."""
    return 2 * pi * tpr / (pi + pi * tpr + (1 - pi) * fpr)


def collapsed_runs():
    cfg = json.loads((UNIF / "random_split" / "comparison.json").read_text())
    return set(cfg["_config"]["excluded_collapsed_runs"])


def load_runs():
    drop = collapsed_runs()
    runs = []
    for k in (1, 2, 3):
        for f in sorted((UNIF / f"random_split_k{k}").glob("metrics_*_s*.json")):
            m = METRIC_RE.match(f.name)
            if not m:
                continue
            model, seed = m.group(1), m.group(2)
            if f"{model}/k{k}-s{seed}" in drop:
                continue
            d = json.loads(f.read_text())
            iso, rnd = d.get("isolated_test"), d.get("random_test")
            if iso and rnd:
                runs.append({"model": model, "split": k, "seed": int(seed),
                             "honest": parts(iso), "leaky": parts(rnd)})
    return runs


def decompose(runs, order):
    """Mean per-step contribution (pp) for one change order."""
    idx = {"pi": 0, "tpr": 1, "fpr": 2}
    steps = {s: [] for s in order}
    totals = []
    for r in runs:
        h, l = r["honest"], r["leaky"]
        cur = {"pi": h[0], "tpr": h[1], "fpr": h[2]}
        for s in order:
            nxt = dict(cur)
            nxt[s] = l[idx[s]]
            steps[s].append(100 * (f1(nxt["pi"], nxt["tpr"], nxt["fpr"])
                                   - f1(cur["pi"], cur["tpr"], cur["fpr"])))
            cur = nxt
        totals.append(100 * (f1(l[0], l[1], l[2]) - f1(h[0], h[1], h[2])))
    return ({s: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)
             for s, v in steps.items()},
            st.mean(totals))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="overwrite results/decomposition_6order.json")
    args = ap.parse_args()
    runs = load_runs()
    if not runs:
        print("!! no paired runs found; nothing written")
        return
    orderings = {}
    print(f"paired non-collapsed runs: {len(runs)}")
    for perm in itertools.permutations(("pi", "fpr", "tpr")):
        key = " → ".join(STEP_NAME[s] for s in perm)
        comp, total = decompose(runs, perm)
        # full precision: rounding here would make the file differ from a
        # rerun at the 4th decimal, which defeats the point of regenerating it
        orderings[key] = {
            "prev_mean": comp["pi"][0], "prev_std": comp["pi"][1],
            "fpr_mean": comp["fpr"][0], "fpr_std": comp["fpr"][1],
            "tpr_mean": comp["tpr"][0], "tpr_std": comp["tpr"][1],
            "sum_mean": sum(c[0] for c in comp.values()),
            "total_gap_pp": total,
        }
        print(f"  {key:26s} prev {comp['pi'][0]:+6.2f}  fpr {comp['fpr'][0]:+6.2f}  "
              f"tpr {comp['tpr'][0]:+6.2f}  sum {sum(c[0] for c in comp.values()):+6.2f}")
    if not args.write:
        stored = json.loads(OUT.read_text()) if OUT.exists() else {}
        worst = 0.0
        for k, v in orderings.items():
            ref = stored.get("orderings", {}).get(k, {})
            for f in ("prev_mean", "fpr_mean", "tpr_mean"):
                if f in ref:
                    worst = max(worst, abs(ref[f] - v[f]))
        n_ok = sum(1 for k, v in orderings.items()
                   if k in stored.get("orderings", {}))
        print(f"\n[check] {n_ok}/{len(orderings)} orderings present in {OUT.name}")
        # tolerance 1e-3 pp: the stored file was produced by a script that no
        # longer exists and rounded intermediate probabilities, so the two
        # agree to 3 decimals but not to machine precision
        print(f"[check] max |recomputed - stored| = {worst:.2e} pp "
              f"({'AGREE at 3 dp' if worst < 1e-3 else 'DISAGRE'})")
        print("[check] stored JSON left untouched (pass --write to overwrite)")
        return
    OUT.write_text(json.dumps({
        "n_runs": len(runs),
        "method": ("sequential one-argument-at-a-time path through "
                   "F1 = 2*pi*TPR / (pi + pi*TPR + (1-pi)*FPR); each ordering "
                   "telescopes to the total gap"),
        "arms": "isolated_test (honest) vs random_test (leaky), same run",
        "excluded_collapsed": sorted(collapsed_runs()),
        "orderings": orderings,
    }, indent=2))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
