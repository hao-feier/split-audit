"""Conservative cluster bootstrap + multiple-comparison
correction + seed-replication audit.

Fixes to the earlier patch-level bootstrap (bootstrap_ci.py, which resampled
overlapping patches as independent units, pooling the three seeds as
"independent replicas"):

1. Cluster unit = spatial cell of 1024 px (10.24 km) over the test-patch
   grid; overlapping patches that straddle a cell border are counted and
   reported (anti-conservative by at most that boundary fraction). The
   "strict" variant (connected components of the overlap graph) is reported
   as a sensitivity where it is non-degenerate.
2. Seed handling: the three seeds' per-patch CMs are resampled TOGETHER with
   their patch (no seed-level resampling), so no independence is assumed
   between seeds of the same patch. This is the conservative "patch as unit,
   seeds pooled within unit" inference.
3. Paired contrasts for the FULL model-pair family per region (F1ch and
   mIoU), with two-sided percentile p-values, corrected by Holm-Bonferroni
   and Benjamini-Hochberg (FDR 0.05) within each (region, metric) family.
4. Seed-replication audit: per-patch F1ch across the three seeds;
   Spearman correlation between seed pairs, and the effective sample size
   n_eff = n / (1 + (k - 1) * rho) for k = 3 correlated replicates.

Output: results/cluster_bootstrap_v2.json
"""
import itertools
import json
import sys
from pathlib import Path

import os
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = Path(os.environ.get("MANGROVE_PREDICTIONS_ROOT",
                                          str(ROOT / "results" / "predictions")))
OUT_DIR = Path(os.environ.get("MANGROVE_RESULTS_ROOT",
                                      str(ROOT / "results")))
sys.path.insert(0, str(ROOT))

MODELS = ["unet", "unet_cbam", "unet_msa", "unet_cbam_msa", "deeplabv3plus",
          "temporal_unet", "temporal_unet_gn", "fc_siam", "changeformer"]
SEEDS = [2026, 2027, 2028, 2029, 2030]
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
B = 10000
CELL = 1024
PATCH = 512
RNG = np.random.default_rng(2026)


def load_patch_grid(region):
    meta = json.loads((ROOT / "data" / "processed" / region / "metadata.json").read_text())
    return np.asarray(meta["patches"], dtype=np.int64)


def split_idx(region):
    return np.asarray(json.loads(
        (ROOT / "data" / "splits" / f"{region}_test.json").read_text()), dtype=np.int64)


def cm4(y, p):
    cm = np.zeros((4, 4), dtype=np.int64)
    comb = y.ravel().astype(np.int64) * 4 + p.ravel().astype(np.int64)
    return np.bincount(comb, minlength=16).reshape(4, 4)


def f1ch_miou(cm):
    ious = [cm[i, i] / (cm[i].sum() + cm[:, i].sum() - cm[i, i])
            for i in range(4) if (cm[i].sum() + cm[:, i].sum() - cm[i, i]) > 0]
    mi = float(np.mean(ious)) if ious else 0.0
    chg = [1, 2]
    tp = cm[np.ix_(chg, chg)].sum()
    fn = cm[np.ix_(chg, [0, 3])].sum()
    fp = cm[np.ix_([0, 3], chg)].sum()
    f1 = 2 * tp / max(2 * tp + fn + fp, 1)
    return float(f1), mi


def overlap_pairs(rows, cols):
    n = len(rows)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(rows[i] - rows[j]) < PATCH and abs(cols[i] - cols[j]) < PATCH:
                pairs.append((i, j))
    return pairs


def strict_clusters(n, pairs):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def cell_clusters(rows, cols):
    keys = {}
    for i, (r, c) in enumerate(zip(rows, cols)):
        keys.setdefault((r // CELL, c // CELL), []).append(i)
    return list(keys.values())


def holm(pvals):
    """Holm-Bonferroni; returns array of adjusted p-values."""
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    cum = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        cum = max(cum, val)
        adj[idx] = min(cum, 1.0)
    return adj


def bh(pvals, q=0.05):
    """Benjamini-Hochberg; returns (adjusted p-values, rejected indices)."""
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = pvals[idx] * m / (rank + 1)
        prev = min(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj, [int(i) for i in order if pvals[i] * m / (list(order).index(i) + 1) <= q]


def main():
    out = {"config": {"B": B, "cell_px": CELL, "seeds": SEEDS,
                      "note": "cluster = spatial cell; seeds resampled with their patch "
                              "(no seed independence assumed); full pairwise family; "
                              "Holm + BH corrections per (region, metric)"}}
    per_region = {}
    for region in REGIONS:
        grid = load_patch_grid(region)
        idx = split_idx(region)
        rows, cols = grid[idx, 0], grid[idx, 1]
        pairs = overlap_pairs(rows, cols)
        strict = strict_clusters(len(idx), pairs)
        cells = cell_clusters(rows, cols)
        cell_id = {}
        for ci, cl in enumerate(cells):
            for i in cl:
                cell_id[i] = ci
        n_boundary = sum(1 for i, j in pairs if cell_id[i] != cell_id[j])

        # per model: (n_patches, n_seeds, 4, 4) CMs; a model enters the
        # family with the seeds that have prediction dumps (>= 3)
        data = {}
        for m in MODELS:
            cms = []
            for s in SEEDS:
                pf = PRED_DIR / f"{m}_s{s}_{region}.npy"
                lf = PRED_DIR / f"{m}_s{s}_{region}_labels.npy"
                if not (pf.exists() and lf.exists()):
                    continue
                cms.append(np.stack([cm4(np.load(lf)[k], np.load(pf)[k])
                                     for k in range(len(idx))]))
            if len(cms) >= 3:
                data[m] = np.stack(cms, axis=1)  # (n, n_seeds, 4, 4)

        reg_out = {
            "n_patches": int(len(idx)),
            "n_overlap_pairs": len(pairs),
            "n_clusters_strict": len(strict),
            "n_clusters_cell": len(cells),
            "boundary_pairs_cell": int(n_boundary),
        }

        # ---------------- cluster bootstrap (patch unit, seeds pooled in-unit)
        def bootstrap(clusters, models):
            res = {m: {"F1ch": [], "mIoU": []} for m in models}
            for b in range(B):
                pick = [RNG.integers(0, len(clusters)) for _ in range(len(clusters))]
                for m in models:
                    cm = np.zeros((4, 4), dtype=np.int64)
                    for k in pick:
                        cm += data[m][clusters[k]].sum(axis=(0, 1))  # sum over patches and 3 seeds
                    f, mi = f1ch_miou(cm)
                    res[m]["F1ch"].append(f)
                    res[m]["mIoU"].append(mi)
            return {m: {k: {"mean": float(np.mean(v)),
                            "ci95": [float(np.percentile(v, 2.5)),
                                     float(np.percentile(v, 97.5))]}
                        for k, v in d.items()} for m, d in res.items()}

        models_present = list(data.keys())
        reg_out["cell_bootstrap"] = bootstrap(cells, models_present)
        if len(strict) > 1:
            reg_out["strict_bootstrap"] = bootstrap(strict, models_present)
        else:
            reg_out["strict_bootstrap"] = {"degenerate_single_cluster": True}

        # ---------------- full pairwise family + corrections
        family = {}
        for metric in ("F1ch", "mIoU"):
            res = {}
            for a, b in itertools.combinations(models_present, 2):
                dvals = []
                for bb in range(B):
                    pick = [RNG.integers(0, len(cells)) for _ in range(len(cells))]
                    ca = np.zeros((4, 4), dtype=np.int64)
                    cb = np.zeros((4, 4), dtype=np.int64)
                    for k in pick:
                        ca += data[a][cells[k]].sum(axis=(0, 1))
                        cb += data[b][cells[k]].sum(axis=(0, 1))
                    ma = f1ch_miou(ca)[0 if metric == "F1ch" else 1]
                    mb = f1ch_miou(cb)[0 if metric == "F1ch" else 1]
                    dvals.append(ma - mb)
                dvals = np.array(dvals)
                p = float(2 * min((dvals < 0).mean(), (dvals > 0).mean()))
                res[f"{a} - {b}"] = {
                    "diff_mean": float(dvals.mean()),
                    "ci95": [float(np.percentile(dvals, 2.5)),
                             float(np.percentile(dvals, 97.5))],
                    "p_raw": p,
                }
            p_raw = np.array([v["p_raw"] for v in res.values()])
            adj_h = holm(p_raw)
            adj_b, rej_b = bh(p_raw)
            for i, k in enumerate(res.keys()):
                res[k]["p_holm"] = float(adj_h[i])
                res[k]["p_bh"] = float(adj_b[i])
                res[k]["bh_rejected"] = bool(i in rej_b)
            family[metric] = res
        reg_out["pairwise_family"] = family

        # ---------------- seed-replication audit
        from scipy.stats import spearmanr
        audit = {}
        for m in models_present:
            nseeds = data[m].shape[1]
            per_patch = np.zeros((len(idx), nseeds))
            for k in range(len(idx)):
                for s_i in range(nseeds):
                    f, _ = f1ch_miou(data[m][k, s_i])
                    per_patch[k, s_i] = f
            rhos = [spearmanr(per_patch[:, a], per_patch[:, b])[0]
                    for a in range(nseeds) for b in range(a + 1, nseeds)]
            rho = float(np.mean(rhos))
            k = nseeds
            n_eff = len(idx) / (1 + (k - 1) * rho) if rho > -1 / (k - 1) else len(idx)
            audit[m] = {
                "rho_seed_spearman": rho,
                "n_patches": int(len(idx)),
                "n_eff_correlated_replicates": float(n_eff),
            }
        reg_out["seed_audit"] = audit
        per_region[region] = reg_out

    out["regions"] = per_region
    (OUT_DIR / "cluster_bootstrap_v2.json").write_text(json.dumps(out, indent=1))
    print("saved results/cluster_bootstrap_v2.json")
    for r, ro in per_region.items():
        print(f"== {r}: patches={ro['n_patches']} cells={ro['n_clusters_cell']} "
              f"strict={ro['n_clusters_strict']} boundary_pairs={ro['boundary_pairs_cell']}")
        for m, a in ro["seed_audit"].items():
            print(f"   {m:16s} rho={a['rho_seed_spearman']:.3f} "
                  f"n_eff={a['n_eff_correlated_replicates']:.1f}")
        for metric in ("F1ch", "mIoU"):
            fam = ro["pairwise_family"][metric]
            sig = [(k, v) for k, v in fam.items() if v["p_bh"] < 0.05]
            print(f"   {metric}: BH-significant pairs (q=0.05):")
            for k, v in sorted(sig, key=lambda kv: kv[1]["p_bh"]):
                print(f"      {k}: diff={v['diff_mean']:+.4f} "
                      f"p_raw={v['p_raw']:.4f} p_bh={v['p_bh']:.4f}")


if __name__ == "__main__":
    main()
