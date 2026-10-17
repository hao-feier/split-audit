"""Cluster-bootstrap sensitivity to the spatial cell size (why
1024 px?).

Re-runs the cluster bootstrap of cluster_bootstrap_v2.py with cell sizes
512 / 1024 / 2048 px (5.12 / 10.24 / 20.48 km) and reports, per region and
per model, the number of clusters, the 95% CI width of the change-F1, and the
Holm-corrected p-values of the key contrasts (proposed vs each other model).
Stable CIs across cell sizes support the 1024 px choice; the cell is a
deliberate multiple of the 512 px patch size (>= 1 patch) so that one
resampling unit spans the patch-patch overlap correlation.

Output: results/bootstrap_cell_sensitivity.json
"""
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
sys.path.insert(0, str(ROOT / "analysis"))
import cluster_bootstrap_v2 as cb  # noqa: E402

CELLS = [512, 1024, 2048]
B = 2000
MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus",
          "temporal_unet", "temporal_unet_gn", "fc_siam", "changeformer"]
RNG = np.random.default_rng(2026)


def run_cell(cell_px):
    cb.CELL = cell_px
    out = {}
    for region in cb.REGIONS:
        grid = cb.load_patch_grid(region)
        idx = cb.split_idx(region)
        rows, cols = grid[idx, 0], grid[idx, 1]
        pairs = cb.overlap_pairs(rows, cols)
        cells = cb.cell_clusters(rows, cols)
        data = {}
        for m in MODELS:
            cms = []
            for s in cb.SEEDS:
                pf = PRED_DIR / f"{m}_s{s}_{region}.npy"
                lf = PRED_DIR / f"{m}_s{s}_{region}_labels.npy"
                if pf.exists() and lf.exists():
                    cms.append(np.stack([cb.cm4(np.load(lf)[k], np.load(pf)[k])
                                         for k in range(len(idx))]))
            if len(cms) >= 3:
                data[m] = np.stack(cms, axis=1)
        models = list(data.keys())
        # precompute per-model per-cell summed CMs (over patches and seeds)
        cell_cm = {m: np.stack([data[m][c].sum(axis=(0, 1)) for c in cells])
                   for m in models}  # (n_models_cells, 4, 4)
        ncl = len(cells)
        picks = np.stack([np.array(RNG.integers(0, ncl, size=ncl)) for _ in range(B)])
        res = {"n_clusters": ncl, "models": {}}
        for m in models:
            v = np.array([cb.f1ch_miou(cell_cm[m][p].sum(axis=0))[0] for p in picks])
            ci = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            res["models"][m] = {"F1ch_mean": round(float(v.mean()), 3),
                                "ci95": [round(c, 3) for c in ci],
                                "ci_width": round(ci[1] - ci[0], 3)}
        if "unet_cbam_msa" in models:
            contra = {}
            for m in models:
                if m == "unet_cbam_msa":
                    continue
                diffs = np.array([
                    cb.f1ch_miou(cell_cm["unet_cbam_msa"][p].sum(axis=0))[0]
                    - cb.f1ch_miou(cell_cm[m][p].sum(axis=0))[0] for p in picks])
                p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
                contra[m] = {"delta_F1ch": round(float(diffs.mean()), 3),
                             "p_raw": round(float(min(p, 1.0)), 4)}
            adj = cb.holm(np.array([v["p_raw"] for v in contra.values()]))
            for (m, v), a in zip(contra.items(), adj):
                v["p_holm"] = round(float(min(a, 1.0)), 4)
            res["contrasts_vs_proposed"] = contra
        out[region] = res
    return out


def main():
    out = {"config": {"cells_px": CELLS, "B": B, "models": MODELS,
                      "note": "cell sizes are multiples of the 512 px patch; "
                              "seeds resampled together with their patch"},
           "per_cell": {}}
    for c in CELLS:
        print(f"cell={c}px ...", flush=True)
        out["per_cell"][str(c)] = run_cell(c)
    (OUT_DIR / "bootstrap_cell_sensitivity.json").write_text(
        json.dumps(out, indent=2))
    print("wrote results/bootstrap_cell_sensitivity.json")


if __name__ == "__main__":
    main()
