"""Cluster / spatial-block bootstrap CIs.

The M7 patch-level bootstrap (bootstrap_ci.py) resamples overlapping 512-px
patches as independent units; with PATCH_STEP=256 neighbouring patches share
~50% of their pixels, so that CI is anti-conservative. Here the resampling
unit is a spatial cluster:

  strict  = connected components of the patch OVERLAP graph (two patches are
            linked iff their 5.12 km boxes intersect). Minimally valid: every
            dependent pair sits in one cluster.
  cell    = 1024-px (10.24 km) super-cells indexed by patch top-left corner.
            The pragmatic "2-km-block" style; some overlapping pairs straddle
            cell borders -> we count those boundary pairs and report them.

Per (model, region) the 3 seeds' per-patch argmax predictions are pooled
(3 replicas per patch, same cluster) and the cluster CMs are resampled
with replacement (B=10000). Reports 95% percentile CIs for OA / mIoU /
F1ch (4-class models) or OA / F1 / IoU (binary PCC), plus paired
differences (same resampled clusters) for the model comparisons.

Output: results/block_bootstrap.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_4C = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus",
             "temporal_unet", "fc_siam", "temporal_unet_gn"]
MODELS_BIN = ["pcc"]
SEEDS = [2026, 2027, 2028]
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
B = 10000
RNG = np.random.default_rng(2026)
PATCH = 512
CELL = 1024

# key comparisons (A - B)
PAIRS = [("unet_cbam_msa", m) for m in
         ["unet", "unet_cbam", "deeplabv3plus", "temporal_unet", "fc_siam"]]
BIN_PAIRS = [("unet_cbam_msa", "pcc"), ("temporal_unet", "pcc")]


def load_patch_grid(region):
    meta = json.loads((ROOT / "data" / "processed" / region / "metadata.json").read_text())
    return np.asarray(meta["patches"], dtype=np.int64)  # (n, 2) = (row, col)


def split_idx(region):
    return np.asarray(json.loads(
        (ROOT / "data" / "splits" / f"{region}_test.json").read_text()), dtype=np.int64)


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


def cm4(y, p):
    cm = np.zeros((4, 4), dtype=np.int64)
    comb = y.ravel().astype(np.int64) * 4 + p.ravel().astype(np.int64)
    bc = np.bincount(comb, minlength=16)
    return bc.reshape(4, 4)


def cmb(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def m4(cm):
    oa = np.trace(cm) / cm.sum()
    ious = [cm[i, i] / (cm[i].sum() + cm[:, i].sum() - cm[i, i])
            for i in range(4) if (cm[i].sum() + cm[:, i].sum() - cm[i, i]) > 0]
    mi = float(np.mean(ious)) if ious else 0.0
    chg = [1, 2]
    tp = cm[np.ix_(chg, chg)].sum(); fn = cm[np.ix_(chg, [0, 3])].sum()
    fp = cm[np.ix_([0, 3], chg)].sum()
    f1 = 2 * tp / max(2 * tp + fn + fp, 1)
    return {"OA": float(oa), "mIoU": mi, "F1ch": float(f1)}


def mb(cm):
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    n = tn + fp + fn + tp
    f1 = 2 * tp / max(2 * tp + fn + fp, 1)
    iou = tp / max(tp + fn + fp, 1)
    return {"OA": float((tp + tn) / max(n, 1)), "F1ch": float(f1), "IoUch": float(iou)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS_4C + MODELS_BIN)
    args = ap.parse_args()

    out = {"config": {"B": B, "patch_px": PATCH, "cell_px": CELL,
                      "seeds_pooled": SEEDS}}
    per_region = {}
    for region in REGIONS:
        grid = load_patch_grid(region)
        idx = split_idx(region)
        rows, cols = grid[idx, 0], grid[idx, 1]
        pairs = overlap_pairs(rows, cols)
        strict = strict_clusters(len(idx), pairs)
        cells = cell_clusters(rows, cols)
        # boundary pairs: overlapping pairs split across different cells
        cell_id = {}
        for ci, cl in enumerate(cells):
            for i in cl:
                cell_id[i] = ci
        n_boundary = sum(1 for i, j in pairs if cell_id[i] != cell_id[j])
        # data[m] is seed-major: row r -> seed r//n, patch r%n (see the
        # concatenate below). So patch i's three seeds live at rows i, i+n,
        # i+2n -- NOT 3i, 3i+1, 3i+2 (that indexing assumed patch-major).
        n_idx = len(idx)
        rep_clusters = {
            "strict": [[i + s * n_idx for i in cl for s in range(3)] for cl in strict],
            "cell": [[i + s * n_idx for i in cl for s in range(3)] for cl in cells],
        }

        # load pooled predictions per model
        data = {}
        for m in args.models:
            if m in MODELS_BIN:
                rep_cms = []
                for s in SEEDS:
                    pf = ROOT / "results" / "predictions" / f"pcc_s{s}_{region}.npy"
                    lf = ROOT / "results" / "predictions" / f"pcc_s{s}_{region}_labels.npy"
                    if not (pf.exists() and lf.exists()):
                        rep_cms = None
                        break
                    rep_cms.append(np.stack([cmb(np.load(lf)[k], np.load(pf)[k])
                                             for k in range(len(idx))]))
            else:
                rep_cms = []
                for s in SEEDS:
                    pf = ROOT / "results" / "predictions" / f"{m}_s{s}_{region}.npy"
                    lf = ROOT / "results" / "predictions" / f"{m}_s{s}_{region}_labels.npy"
                    if not (pf.exists() and lf.exists()):
                        rep_cms = None
                        break
                    rep_cms.append(np.stack([cm4(np.load(lf)[k], np.load(pf)[k])
                                             for k in range(len(idx))]))
            if rep_cms is not None and rep_cms:
                data[m] = np.concatenate(rep_cms, axis=0)  # (3*n, 4, 4) or (2,2)

        reg_out = {
            "n_patches": int(len(idx)),
            "n_overlap_pairs": len(pairs),
            "n_clusters_strict": len(strict),
            "cluster_sizes_strict": sorted((len(c) for c in strict), reverse=True),
            "n_clusters_cell": len(cells),
            "cluster_sizes_cell": sorted((len(c) for c in cells), reverse=True),
            "boundary_pairs_cell": int(n_boundary),
        }
        for method, cl in rep_clusters.items():
            res = {}
            for m, cms in data.items():
                fn = mb if m in MODELS_BIN else m4
                vals = []
                for b in range(B):
                    pick = [RNG.integers(0, len(cl)) for _ in range(len(cl))]
                    pooled = np.zeros(cms.shape[1:], dtype=np.int64)
                    for k in pick:
                        pooled += cms[cl[k]].sum(axis=0)
                    vals.append(fn(pooled))
                keys = ["OA", "F1ch"] + (["mIoU"] if m not in MODELS_BIN else ["IoUch"])
                res[m] = {}
                for k in keys:
                    arr = np.array([v[k] for v in vals])
                    res[m][k] = {"mean": float(arr.mean()),
                                 "ci95": [float(np.percentile(arr, 2.5)),
                                          float(np.percentile(arr, 97.5))] }
            reg_out[method] = res

            # paired comparisons (same resampled clusters)
            paired = {}
            all_pairs = PAIRS if method == "strict" else PAIRS
            for a, b in PAIRS:
                if a in data and b in data:
                    ca, cb = data[a], data[b]
                    dvals = []
                    for bb in range(B):
                        pick = [RNG.integers(0, len(cl)) for _ in range(len(cl))]
                        pa = np.zeros(ca.shape[1:], dtype=np.int64)
                        pb_ = np.zeros(cb.shape[1:], dtype=np.int64)
                        for k in pick:
                            pa += ca[cl[k]].sum(axis=0)
                            pb_ += cb[cl[k]].sum(axis=0)
                        dvals.append((m4(pa)["F1ch"] - m4(pb_)["F1ch"],
                                      m4(pa)["mIoU"] - m4(pb_)["mIoU"]))
                    df1 = np.array([d[0] for d in dvals])
                    dmi = np.array([d[1] for d in dvals])
                    paired[f"{a} vs {b}"] = {
                        "dF1ch": [float(df1.mean()),
                                  [float(np.percentile(df1, 2.5)), float(np.percentile(df1, 97.5))]],
                        "p_F1ch": float(2 * min((df1 < 0).mean(), (df1 > 0).mean())),
                        "dmIoU": [float(dmi.mean()),
                                  [float(np.percentile(dmi, 2.5)), float(np.percentile(dmi, 97.5))]],
                        "p_mIoU": float(2 * min((dmi < 0).mean(), (dmi > 0).mean())),
                    }
            for a, b in BIN_PAIRS:
                if a in data and b in data:
                    ca, cb = data[a], data[b]
                    dvals = []
                    for bb in range(B):
                        pick = [RNG.integers(0, len(cl)) for _ in range(len(cl))]
                        pa = np.zeros((4, 4), dtype=np.int64)
                        pb_ = np.zeros((2, 2), dtype=np.int64)
                        for k in pick:
                            pa += ca[cl[k]].sum(axis=0)
                            pb_ += cb[cl[k]].sum(axis=0)
                        # binary collapse of the 4-class model
                        chg = [1, 2]
                        bt = np.array([[pa[np.ix_([0, 3], [0, 3])].sum(),
                                        pa[np.ix_([0, 3], chg)].sum()],
                                       [pa[np.ix_(chg, [0, 3])].sum(),
                                        pa[np.ix_(chg, chg)].sum()]])
                        dvals.append(m4(pa)["F1ch"] - mb(pb_)["F1ch"])
                    df1 = np.array(dvals)
                    paired[f"{a} (binary) vs {b}"] = {
                        "dF1ch": [float(df1.mean()),
                                  [float(np.percentile(df1, 2.5)), float(np.percentile(df1, 97.5))]],
                        "p_F1ch": float(2 * min((df1 < 0).mean(), (df1 > 0).mean())),
                    }
            reg_out[f"{method}_paired"] = paired
        per_region[region] = reg_out

    out["regions"] = per_region
    (ROOT / "results" / "block_bootstrap.json").write_text(json.dumps(out, indent=2))
    for r, ro in per_region.items():
        print(f"{r}: patches={ro['n_patches']} strict_clusters={ro['n_clusters_strict']} "
              f"sizes={ro['cluster_sizes_strict']} cell_clusters={ro['n_clusters_cell']} "
              f"boundary_pairs={ro['boundary_pairs_cell']}")
        for m, v in ro["strict"].items():
            print(f"  {m}: F1ch={v['F1ch']['mean']:.4f} "
                  f"CI95=[{v['F1ch']['ci95'][0]:.4f},{v['F1ch']['ci95'][1]:.4f}]")
    print("wrote results/block_bootstrap.json")


if __name__ == "__main__":
    main()
