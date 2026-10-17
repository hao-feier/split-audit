"""Leave-one-spatial-cluster-out sensitivity.

Question: do any of the reported test-region metrics rest on a single
spatial cluster (e.g. Zhanjiang has only 2 cell clusters at the 1024-px
scale used by the cluster bootstrap)? We drop each cluster in turn and
recompute the headline metrics, pooled over seeds (CM summed over the
remaining patches and all seeds) and per seed (so the per-seed
distribution is visible).

Cluster definition: identical to analysis/cluster_bootstrap_v2.py —
1024-px (10.24 km) spatial cells of the test-patch grid (primary); the
connected-component "strict" overlap clusters are also reported where
non-degenerate.

Output: results/leave_one_cluster_out.json (+ appended FINDINGS entry).
"""
import json
import sys
from datetime import date
from pathlib import Path

import os
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = Path(os.environ.get("MANGROVE_PREDICTIONS_ROOT",
                                          str(ROOT / "results" / "predictions")))
OUT_DIR = Path(os.environ.get("MANGROVE_RESULTS_ROOT",
                                      str(ROOT / "results")))
sys.path.insert(0, str(ROOT))

# deep four-class models with per-patch prediction dumps (pcc excluded:
# binary per-epoch control, not part of the four-class family)
MODELS = ["unet", "unet_cbam", "unet_msa", "unet_cbam_msa", "deeplabv3plus",
          "deeplabv3plus_pre", "temporal_unet", "temporal_unet_gn", "fc_siam",
          "changeformer"]
SEEDS = [2026, 2027, 2028, 2029, 2030]
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
CELL = 1024
PATCH = 512


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


def load_patch_cms(model, region, n_patches):
    """(n_patches, n_seeds, 4, 4) true x pred CMs; None if <3 seeds present."""
    cms = []
    for s in SEEDS:
        pf = PRED_DIR / f"{model}_s{s}_{region}.npy"
        lf = PRED_DIR / f"{model}_s{s}_{region}_labels.npy"
        if not (pf.exists() and lf.exists()):
            continue
        stack = [cm4(np.load(lf)[k], np.load(pf)[k]) for k in range(n_patches)]
        cms.append(np.stack(stack))
    if len(cms) < 3:
        return None
    return np.stack(cms, axis=1)


def metrics_from_cm(cm):
    """Headline metrics from a (4,4) true x pred confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    chg, un = [1, 2], [0, 3]
    tp = cm[np.ix_(chg, chg)].sum()
    fn = cm[np.ix_(chg, un)].sum()
    fp = cm[np.ix_(un, chg)].sum()
    ious = []
    for i in range(4):
        denom = cm[i].sum() + cm[:, i].sum() - cm[i, i]
        if denom > 0:
            ious.append(cm[i, i] / denom)
    per_class_f1 = [2 * cm[i, i] / max(cm[i].sum() + cm[:, i].sum(), 1)
                    for i in range(4)]
    return {
        "OA": float(cm.diagonal().sum() / max(total, 1)),
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "F1ch": float(2 * tp / max(2 * tp + fn + fp, 1)),
        "pred_change_frac": float(cm[:, chg].sum() / max(total, 1)),
        "true_change_frac": float(cm[chg, :].sum() / max(total, 1)),
        "omission": float(fn / max(tp + fn, 1)),
        "commission": float(fp / max(tp + fp, 1)),
        "per_class_F1": [float(v) for v in per_class_f1],
    }


def cell_clusters(rows, cols):
    keys = {}
    for i, (r, c) in enumerate(zip(rows, cols)):
        keys.setdefault((r // CELL, c // CELL), []).append(i)
    return list(keys.values())


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


def per_seed_f1ch(data, sel, n_seeds):
    """F1ch per seed over the selected patches (data: (n_patches, n_seeds, 4, 4))."""
    out = []
    for s in range(n_seeds):
        out.append(metrics_from_cm(data[sel, s].sum(axis=0))["F1ch"])
    return out


def loco_for_clusters(data, clusters, names):
    n_seeds = data.shape[1]
    all_idx = np.arange(data.shape[0])
    base = metrics_from_cm(data.sum(axis=(0, 1)))
    res = {
        "baseline": {
            "F1ch": base["F1ch"], "mIoU": base["mIoU"], "OA": base["OA"],
            "per_seed_F1ch": per_seed_f1ch(data, all_idx, n_seeds),
        },
        "drops": {},
    }
    for ci, cl in enumerate(clusters):
        cl_set = set(cl)
        keep = np.array([i for i in all_idx if i not in cl_set], dtype=np.int64)
        if len(keep) == 0:
            res["drops"][f"{names[ci]}"] = {"degenerate": True}
            continue
        m = metrics_from_cm(data[keep].sum(axis=(0, 1)))
        res["drops"][f"{names[ci]}"] = {
            "F1ch": m["F1ch"], "mIoU": m["mIoU"], "OA": m["OA"],
            "delta_F1ch": m["F1ch"] - base["F1ch"],
            "delta_mIoU": m["mIoU"] - base["mIoU"],
            "delta_OA": m["OA"] - base["OA"],
            "per_seed_F1ch": per_seed_f1ch(data, keep, n_seeds),
        }
    return res


def main():
    out = {"config": {
        "cell_px": CELL,
        "seeds": SEEDS,
        "models": MODELS,
        "note": "cluster = 1024-px cell of the test-patch grid (same as "
                "cluster_bootstrap_v2.py); pooled metrics from the CM summed "
                "over remaining patches and all seeds; per-seed F1ch reported "
                "for the per-seed distribution",
    }, "regions": {}}

    for region in REGIONS:
        grid = load_patch_grid(region)
        idx = split_idx(region)
        rows, cols = grid[idx, 0], grid[idx, 1]

        data = {}
        for m in MODELS:
            d = load_patch_cms(m, region, len(idx))
            if d is not None:
                data[m] = d

        cells = cell_clusters(rows, cols)
        strict = strict_clusters(len(idx), overlap_pairs(rows, cols))

        def ckey(cl):
            return (int(cols[cl[0]]), int(rows[cl[0]]))
        cells.sort(key=ckey)
        strict.sort(key=ckey)

        reg = {
            "n_patches": int(len(idx)),
            "n_clusters_cell": len(cells),
            "n_clusters_strict": len(strict),
            "cluster_sizes_cell": [len(c) for c in cells],
            "cluster_sizes_strict": [len(c) for c in strict],
        }

        # per-cell pixel composition (region-level reference = proposed model,
        # pooled over seeds; cluster identity itself is model-agnostic)
        tot = data["unet_cbam_msa"].sum(axis=(0, 1))
        tot_true = tot[[1, 2], :].sum()
        tot_pred = tot[:, [1, 2]].sum()
        for ci, cl in enumerate(cells):
            cm = data["unet_cbam_msa"][cl].sum(axis=(0, 1))
            reg[f"cell_{ci}"] = {
                "n_patches": int(len(cl)),
                "pixel_share": float(cm.sum() / tot.sum()),
                "true_change_share": float(cm[[1, 2], :].sum() / max(tot_true, 1)),
                "pred_change_share": float(cm[:, [1, 2]].sum() / max(tot_pred, 1)),
            }

        for variant, clusters in (("cell", cells), ("strict", strict)):
            if len(clusters) < 2:
                reg[variant] = {"degenerate_single_cluster": True}
                continue
            names = [f"{variant}{i + 1}" for i in range(len(clusters))]
            reg[variant] = {m: loco_for_clusters(d, clusters, names)
                            for m, d in data.items()}

        out["regions"][region] = reg

    (OUT_DIR / "leave_one_cluster_out.json").write_text(json.dumps(out, indent=1))
    print("saved results/leave_one_cluster_out.json")
    for r, ro in out["regions"].items():
        print(f"== {r}: patches={ro['n_patches']} cells={ro['n_clusters_cell']} "
              f"sizes={ro['cluster_sizes_cell']} strict={ro['n_clusters_strict']}")
        for ci in range(ro["n_clusters_cell"]):
            c = ro[f"cell_{ci}"]
            print(f"   cell{ci}: patches={c['n_patches']} pixels={c['pixel_share']:.2f} "
                  f"truechg={c['true_change_share']:.2f} predchg={c['pred_change_share']:.2f}")
        v = ro.get("cell")
        if isinstance(v, dict) and not "degenerate_single_cluster" in v:
            for m in ("unet_cbam_msa", "temporal_unet_gn", "fc_siam", "unet"):
                base = v[m]["baseline"]["F1ch"]
                parts = []
                for name, d in v[m]["drops"].items():
                    parts.append(f"{name}=n/a" if "degenerate" in d
                                 else f"{name} {d['F1ch']:.3f}({d['delta_F1ch']:+.3f})")
                print(f"   cell  {m:16s} full {base:.3f} | " + " | ".join(parts))
        v = ro.get("strict")
        if isinstance(v, dict) and "degenerate_single_cluster" not in v:
            for m in ("unet_cbam_msa", "temporal_unet_gn"):
                base = v[m]["baseline"]["F1ch"]
                parts = []
                for name, d in v[m]["drops"].items():
                    parts.append(f"{name}=n/a" if "degenerate" in d
                                 else f"{name} {d['F1ch']:.3f}({d['delta_F1ch']:+.3f})")
                print(f"   strict {m:16s} full {base:.3f} | " + " | ".join(parts))

    # ---- FINDINGS.md entry -------------------------------------------------
    findings = ROOT / "results" / "FINDINGS.md"
    entry = (
        f"\n## {date.today().isoformat()} — Leave-one-cluster-out sensitivity "
        "\n\n"
        "Drops each 1024-px spatial cell cluster of the test-patch grid (same "
        "definition as the cluster bootstrap) and recomputes pooled and "
        "per-seed F1ch/mIoU/OA for all ten deep four-class models "
        "(`analysis/leave_one_cluster_out.py` → `results/leave_one_cluster_out.json`). "
        "Cluster counts: within 4, Beilunhe 5, Zhanjiang 2; strict overlap-component "
        "clusters: 1 / 3 / 1 (strict variant non-degenerate for Beilunhe only).\n"
    )
    findings.write_text(findings.read_text().rstrip() + "\n" + entry)
    print("appended results/FINDINGS.md")


if __name__ == "__main__":
    main()
