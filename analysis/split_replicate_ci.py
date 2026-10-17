"""Partition-level replication of the spatial-interleaving F1ch gap.

The headline gap is, per (architecture, run-seed, split),
    gap = F1_change(random_test) - F1_change(isolated_test)
i.e. the leaky (interleaved) test arm minus the honest (isolated) test arm,
both scored with the same checkpoint trained on that split's train zone.

This script aggregates the counterfactual over K independently drawn
train/test partitions (results/uniform/random_split_k{1..K}) and reports,
honestly:

  * one cluster mean per partition (average over arch x run-seed, collapses
    excluded) so the reviewer can see the gap is not one lucky partition;
  * an equal-weight-per-partition point estimate with a partition-level 95%
    interval (cluster = partition);
  * the pooled (equal-weight-per-run) estimate, which matches the headline;
  * a small-pool correlation diagnostic (pairwise training-patch overlap
    across partitions) so the interval is read as the spread over random
    partitions of this fixed, small region -- NOT as K spatially disjoint
    regions.

Usage:
    python analysis/split_replicate_ci.py [K]     # K default 10
Writes results/uniform/split_replicate_ci.json
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from aggregate_results import binary_change  # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
COLLAPSE_T = 0.05
SEEDS = [2026, 2027, 2028]
UNIFORM = ROOT / "results" / "uniform"
SPLITS = ROOT / "data" / "splits"


def f1ch(arm):
    if not arm or "confusion_matrix" not in arm:
        return None
    return float(binary_change(arm["confusion_matrix"])["F1_change"])


def collect():
    rows = []
    present = []
    for k in range(1, K + 1):
        d = UNIFORM / f"random_split_k{k}"
        if not d.exists():
            continue
        present.append(k)
        for mf in sorted(d.glob("metrics_*.json")):
            m = json.loads(mf.read_text())
            leaky = f1ch(m.get("random_test"))
            honest = f1ch(m.get("isolated_test"))
            if leaky is None or honest is None:
                continue
            rows.append(dict(
                split=k, arch=m.get("model"), seed=m.get("seed"),
                leaky=leaky, honest=honest, gap=leaky - honest,
                collapsed=bool(leaky < COLLAPSE_T)))
    return rows, present


def train_overlap(present):
    """Pairwise training-patch overlap (out of 40) across partitions."""
    sets = {}
    for k in present:
        p = SPLITS / f"vietnam_train_rand{k}.json"
        if p.exists():
            sets[k] = set(json.loads(p.read_text()))
    if len(sets) < 2:
        return None
    ovs = []
    sizes = []
    for a, b in itertools.combinations(sorted(sets), 2):
        ovs.append(len(sets[a] & sets[b]))
        sizes.append(len(sets[a]))
    return dict(n_pairs=len(ovs), train_size=int(sizes[0]),
                overlap_min=int(min(ovs)), overlap_max=int(max(ovs)),
                overlap_frac_min=round(min(ovs) / sizes[0], 3),
                overlap_frac_max=round(max(ovs) / sizes[0], 3))


def main():
    rows, present = collect()
    if not present:
        print("No random_split_k{1..%d} results found under %s" % (K, UNIFORM))
        sys.exit(1)

    # per-partition cluster means (collapses excluded)
    by_split = {}
    for r in rows:
        if not r["collapsed"]:
            by_split.setdefault(r["split"], []).append(r["gap"])
    cluster = {k: dict(mean=float(np.mean(v)), n=len(v),
                       min=float(np.min(v)), max=float(np.max(v)))
               for k, v in by_split.items()}

    ks = sorted(cluster)
    means = np.array([cluster[k]["mean"] for k in ks])
    n = len(means)
    overall = float(means.mean())
    se = float(means.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    tcrit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    ci_lo = overall - tcrit * se
    ci_hi = overall + tcrit * se

    noncol = [r["gap"] for r in rows if not r["collapsed"]]
    pooled = float(np.mean(noncol)) if noncol else float("nan")

    n_pos = int(sum(1 for k in ks if cluster[k]["mean"] > 0))

    out = dict(
        K_partitions=n,
        partitions_present=ks,
        n_runs_total=len(rows),
        n_noncollapse=int(len(noncol)),
        n_collapse=int(len(rows) - len(noncol)),
        collapse_threshold=COLLAPSE_T,
        partitions_positive=n_pos,
        partitions_positive_of_n=f"{n_pos}/{n}",
        per_partition_mean_pp={str(k): round(cluster[k]["mean"] * 100, 2)
                               for k in ks},
        per_partition_range_pp={str(k): [round(cluster[k]["min"] * 100, 2),
                                         round(cluster[k]["max"] * 100, 2)]
                                for k in ks},
        per_partition_n={str(k): cluster[k]["n"] for k in ks},
        overall_mean_pp=round(overall * 100, 2),
        partition_level_se_pp=round(se * 100, 2),
        t_crit=round(tcrit, 4),
        ci95_lo_pp=round(ci_lo * 100, 2),
        ci95_hi_pp=round(ci_hi * 100, 2),
        pooled_noncollapse_pp=round(pooled * 100, 2),
        pooled_range_pp=[round(float(np.min(noncol)) * 100, 2),
                         round(float(np.max(noncol)) * 100, 2)] if noncol else None,
        small_pool_train_overlap=train_overlap(present),
        note=("interval = spread over K independent random partitions of this "
              "fixed small region; partitions share most training patches "
              "(see small_pool_train_overlap), so read as partition-sampling "
              "variability, not K spatially disjoint regions."),
    )
    outp = UNIFORM / "split_replicate_ci.json"
    outp.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print("\nwrote", outp)


if __name__ == "__main__":
    main()
