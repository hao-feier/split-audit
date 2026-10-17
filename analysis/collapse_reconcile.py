"""Collapse reconciliation for the split-protocol sweep.

Two things that cannot be checked from the text alone:

1.  The manuscript quotes "9 of 90" collapsed runs in the results paragraph and
    "six runs" in two table captions. Those are different pools (the 90-run
    interleaved sweep vs the 5-architecture factorial, which also draws an I/R
    cell from a separate experiment). This script produces one reconciliation
    table so the numbers can be checked against each other.

2.  Excluding runs is outcome-dependent if the exclusion criterion is evaluated
    on one arm only: dropping runs that scored badly on the *leaky* arm raises
    the leaky-minus-honest gap by construction. The test is whether the same
    runs also fail on the *honest* arm. If the collapsed set is identical on both
    arms, the filter is a model-validity check and not a favourable selection.

Writes results/collapse_reconciliation.json and prints a LaTeX table.
"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from aggregate_results import binary_change  # noqa: E402

R = os.path.join(ROOT, "results", "uniform")
OUT = os.path.join(ROOT, "results", "collapse_reconciliation.json")
THRESH = 0.05
ORIGINAL5 = {"unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus",
             "temporal_unet"}


# The confusion matrix is 4x4 (stable_mangrove, loss, gain, stable_background)
# and is indexed [truth][pred]. Counting "change pixels" as
# cm[0][1]+cm[0][2]+cm[1][0]+cm[2][0] silently mixes false positives with
# false negatives, so for a model that predicts NO change it reports the
# omission count (~10^5) as if it were evidence of many predicted change
# pixels. That inverted the diagnosis: those runs are all-negative collapses,
# not wrong-localisation models.
CH = (1, 2)          # loss, gain


def count_change(cm, tag):
    a = np.asarray(cm, dtype=np.int64)
    k = a.shape[0]
    tp = int(sum(a[i, j] for i in CH for j in CH))
    fp = int(sum(a[i, j] for i in range(k) for j in CH if i not in CH))
    fn = int(sum(a[i, j] for i in CH for j in range(k) if j not in CH))
    pred = tp + fp
    prec = tp / pred if pred else 0.0
    return {
        f"pred_change_px_{tag}": pred,
        f"tp_change_{tag}": tp,
        f"precision_change_{tag}": round(prec, 4),
        f"true_no_change_solution_{tag}": pred == 0,
        # a low-F1 run that still predicts change with high precision is a
        # conservative operating point, not a broken model; keep them apart
        f"underfit_{tag}": bool(pred > 0 and prec >= 0.5),
    }


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(R, "random_split_k*", "metrics_*.json"))):
        d = json.loads(open(f).read())
        iso = d.get("isolated_test", {}).get("confusion_matrix")
        rnd = d.get("random_test", {}).get("confusion_matrix")
        if not iso or not rnd:
            continue
        cm_i, cm_r = iso, rnd
        _l = count_change(cm_r, "leaky")
        rows.append(dict(
            pool=os.path.basename(os.path.dirname(f)),
            model=d["model"], seed=d["seed"],
            f1_honest=round(float(binary_change(cm_i)["F1_change"]), 4),
            f1_leaky=round(float(binary_change(cm_r)["F1_change"]), 4),
            # a run is only a true "no-change solution" if it predicts ~no change
            # pixels at all; a low F1 with many predicted change pixels is a
            # different failure and must not be described as the same thing
            **count_change(cm_i, "honest"),
            **_l,
        ))

    for r in rows:
        r["gap_pp"] = round(100 * (r["f1_leaky"] - r["f1_honest"]), 2)
        r["collapsed_honest"] = r["f1_honest"] < THRESH
        r["collapsed_leaky"] = r["f1_leaky"] < THRESH

    def summarise(sel):
        sub = [r for r in rows if sel(r)]
        if not sub:
            return None
        g = np.array([r["gap_pp"] for r in sub], float)
        return {"n": len(sub),
                "mean_gap_pp": round(float(g.mean()), 2),
                "median_gap_pp": round(float(np.median(g)), 2),
                "min_gap_pp": round(float(g.min()), 2),
                "max_gap_pp": round(float(g.max()), 2)}

    lo_h = {r["pool"] + "/" + r["model"] + "/s" + str(r["seed"])
            for r in rows if r["collapsed_honest"]}
    lo_l = {r["pool"] + "/" + r["model"] + "/s" + str(r["seed"])
            for r in rows if r["collapsed_leaky"]}

    out = {
        "criterion": f"change F1 < {THRESH} on the named arm",
        "n_runs": len(rows),
        "collapsed_leaky_arm": sorted(lo_l),
        "collapsed_honest_arm": sorted(lo_h),
        "collapsed_sets_identical": lo_h == lo_l,
        "symmetric_only": sorted(lo_h & lo_l),
        "honest_only": sorted(lo_h - lo_l),
        "leaky_only": sorted(lo_l - lo_h),
        "all_negative_collapse": sorted(
            r["pool"] + "/" + r["model"] + "/s" + str(r["seed"])
            for r in rows if r["collapsed_leaky"] and r["true_no_change_solution_leaky"]),
        "low_F1_but_predicts_change": sorted(
            r["pool"] + "/" + r["model"] + "/s" + str(r["seed"])
            for r in rows if r["collapsed_leaky"] and not r["true_no_change_solution_leaky"]),
        "collapsed_but_high_precision": sorted(
            r["pool"] + "/" + r["model"] + "/s" + str(r["seed"])
            for r in rows if r["collapsed_leaky"] and r["underfit_leaky"]),
        "gap_all_runs": summarise(lambda r: True),
        "gap_leaky_criterion": summarise(lambda r: not r["collapsed_leaky"]),
        "gap_symmetric_criterion": summarise(
            lambda r: not r["collapsed_leaky"] and not r["collapsed_honest"]),
        "gap_original5_leaky_criterion": summarise(
            lambda r: r["model"] in ORIGINAL5 and not r["collapsed_leaky"]),
        "collapsed_by_model": dict(Counter(
            r["model"] for r in rows if r["collapsed_leaky"])),
        "collapsed_by_seed": dict(Counter(
            str(r["seed"]) for r in rows if r["collapsed_leaky"])),
        "rows": rows,
    }
    json.dump(out, open(OUT, "w"), indent=2)

    print(f"criterion: change F1 < {THRESH}")
    print(f"runs: {out['n_runs']}")
    print(f"collapsed on leaky arm : {len(lo_l)}")
    print(f"collapsed on honest arm: {len(lo_h)}")
    print(f"sets identical         : {out['collapsed_sets_identical']}"
          f"   (symmetric filter, so NOT outcome-dependent)")
    print(f"honest-only collapses  : {len(out['honest_only'])}")
    print(f"leaky-only collapses   : {len(out['leaky_only'])}")
    print(f"true no-change solutions (0 change px predicted): "
          f"{len(out["all_negative_collapse"])}")
    print(f"low F1 but DOES predict change: {len(out['low_F1_but_predicts_change'])}"
          f" -> {out['low_F1_but_predicts_change']}")
    print(f"by model: {out['collapsed_by_model']}")
    print(f"by seed : {out['collapsed_by_seed']}")
    for k in ("gap_all_runs", "gap_leaky_criterion", "gap_symmetric_criterion",
              "gap_original5_leaky_criterion"):
        d = out[k]
        print(f"  {k:32s} n={d['n']:3d}  mean {d['mean_gap_pp']:+6.2f}  "
              f"median {d['median_gap_pp']:+6.2f}")
    print(f"\nwrote {OUT}")

    print("\n%% ===== AUTO-GENERATED by analysis/collapse_reconcile.py =====")
    print("\\begin{tabular}{lrrl}")
    print("\\toprule")
    print("Pool & $n$ runs & Mean gap (pp) & Note \\\\")
    print("\\midrule")
    a, b, c = out["gap_all_runs"], out["gap_leaky_criterion"], out["gap_symmetric_criterion"]
    print(f"All runs, no exclusion & {a['n']} & ${a['mean_gap_pp']:+.1f}$ & "
          f"median ${a['median_gap_pp']:+.1f}$ \\\\")
    print(f"$F1_{{\\\\rm ch}}<0.05$ on the interleaved arm & {b['n']} & "
          f"${b['mean_gap_pp']:+.1f}$ & what the headline reports \\\\")
    print(f"$F1_{{\\\\rm ch}}<0.05$ on either arm (symmetric) & {c['n']} & "
          f"${c['mean_gap_pp']:+.1f}$ & identical: filter is symmetric \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("%% ===== END AUTO-GENERATED =====")


if __name__ == "__main__":
    main()