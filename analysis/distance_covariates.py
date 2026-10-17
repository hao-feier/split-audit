"""Distance-vs-patch-difficulty control.

The distance-response experiment (analysis/eval_distance_response.py) scores
26 never-trained Vietnam patches at five edge distances from the training
zone. The near-far F1 gradient could be a patch-
difficulty confound (change abundance / mangrove fraction / boundary
complexity) rather than proximity.

This exploratory analysis (no training, no test-region selection):
  1. computes per-patch covariates from the GMW labels: true change fraction,
     stable-mangrove fraction, boundary-pixel density (label edge pixels),
  2. loads the per-patch, per-model binary change F1 from
     results/distance_response.json (five main models, five seeds pooled),
  3. fits pooled OLS  F1ch ~ distance + change_frac + mangrove_frac + boundary
     and reports coefficients, SEs and R^2, plus the plain distance gradient
     before/after covariate adjustment, per model.

Output: results/distance_covariates.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.datasets import load_region  # noqa: E402
from mangrove.paths import SPLITS_ROOT  # noqa: E402

REGION = "vietnam"
CHANGED = (1, 2)


def box_m(i, patch_list):
    r, c = patch_list[i]
    return (r * 10.0, (r + 512) * 10.0, c * 10.0, (c + 512) * 10.0)


def min_edge_km(i, j, patch_list):
    a, b = box_m(i, patch_list), box_m(j, patch_list)
    dc = max(0.0, max(a[2], b[2]) - min(a[3], b[3]))
    dr = max(0.0, max(a[0], b[0]) - min(a[1], b[1]))
    return float(np.hypot(dc, dr)) / 1000.0


def main():
    meta = json.loads((ROOT / "data" / "processed" / REGION / "metadata.json").read_text())
    patch_list = list(meta["patches"])
    train = json.loads((SPLITS_ROOT / "vietnam_train.json").read_text())
    test = json.loads((SPLITS_ROOT / "vietnam_test.json").read_text())
    buffer = json.loads((SPLITS_ROOT / "vietnam_buffer.json").read_text())
    eval_idx = sorted(set(test) | set(buffer))
    assert len(eval_idx) == 26

    data = load_region(REGION)
    # per-patch label stats (data["change"] holds the four-class change label
    # per patch: 0 stable mangrove, 1 loss, 2 gain, 3 stable background)
    cov = {}
    for i in eval_idx:
        lab = np.asarray(data["change"][i])
        chg = np.isin(lab, list(CHANGED))
        mg = lab == 0
        # boundary: change pixels adjacent (4-conn) to a non-change pixel
        pad = np.pad(chg, 1)
        neigh = (pad[1:-1, 2:] | pad[1:-1, :-2] | pad[2:, 1:-1] | pad[:-2, 1:-1])
        boundary = chg & neigh
        cov[i] = {
            "true_change_frac": float(chg.mean()),
            "mangrove_frac": float(mg.mean()),
            "boundary_frac": float(boundary.sum() / max(chg.sum(), 1)) if chg.sum() else 0.0,
            "note_boundary": "change pixels are almost all edge-adjacent (frac=1.0 everywhere); kept descriptively only",
        }
    dists = {i: min(min_edge_km(i, t, patch_list) for t in train) for i in eval_idx}

    dr = json.loads((ROOT / "results" / "distance_response.json").read_text())
    models = dr["config"]["models"]
    # per-patch F1ch per model (pooled over seeds in the JSON's per_patch)
    f1 = {m: {} for m in models}
    for m in models:
        for b in dr["per_model"][m]:
            for pid, v in b["per_patch"].items():
                f1[m][int(pid)] = float(v)

    # pooled OLS: F1ch ~ 1 + distance + change_frac (mangrove_frac is strongly
    # correlated with change_frac, r=0.71; boundary_frac is degenerate, 1.0 for
    # every patch -- both kept as descriptive covariates only)
    Xrows, y = [], []
    for m in models:
        for i in eval_idx:
            Xrows.append([1.0, dists[i], cov[i]["true_change_frac"]])
            y.append(f1[m][i])
    X = np.array(Xrows)
    y = np.array(y)
    beta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    n, p = X.shape
    sigma2 = ss_res / (n - p)
    covmat = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(covmat))
    tvals = beta / se

    # per model: plain distance correlation vs covariate-adjusted (residualize
    # F1 and distance on the three covariates, then correlate)
    per_model = {}
    cov3 = np.array([[1.0, cov[i]["true_change_frac"]] for i in eval_idx])
    for m in models:
        ym = np.array([f1[m][i] for i in eval_idx])
        dm = np.array([dists[i] for i in eval_idx])
        r_plain = float(np.corrcoef(dm, ym)[0, 1])
        e_y = ym - cov3 @ np.linalg.lstsq(cov3, ym, rcond=None)[0]
        e_d = dm - cov3 @ np.linalg.lstsq(cov3, dm, rcond=None)[0]
        r_adj = float(np.corrcoef(e_d, e_y)[0, 1])
        per_model[m] = {"r_distance_plain": round(r_plain, 3),
                        "r_distance_adj": round(r_adj, 3)}

    out = {
        "config": {
            "n_patches": len(eval_idx),
            "n_models": len(models),
            "n_obs": n,
            "note": ("exploratory pooled OLS on per-patch binary change F1 "
                     "(five-seed means per patch); covariates = true change "
                     "fraction (mangrove fraction is r=0.71-collinear with it "
                     "and boundary density is degenerate, frac=1.0 for every "
                     "patch, so both are descriptive only); not a "
                     "test-region selection"),
        },
        "covariates": {
            "change_vs_distance_r": round(float(np.corrcoef(
                [dists[i] for i in eval_idx],
                [cov[i]["true_change_frac"] for i in eval_idx])[0, 1]), 3),
            "mangrove_vs_distance_r": round(float(np.corrcoef(
                [dists[i] for i in eval_idx],
                [cov[i]["mangrove_frac"] for i in eval_idx])[0, 1]), 3),
            "change_vs_mangrove_r": round(float(np.corrcoef(
                [cov[i]["true_change_frac"] for i in eval_idx],
                [cov[i]["mangrove_frac"] for i in eval_idx])[0, 1]), 3),
        },
        "ols": {
            "terms": ["intercept", "distance_km", "true_change_frac"],
            "beta": [round(float(b), 4) for b in beta],
            "se": [round(float(s), 4) for s in se],
            "t": [round(float(v), 2) for v in tvals],
            "r2": round(r2, 3),
        },
        "per_model": per_model,
        "per_patch": {str(i): {
            "distance_km": round(dists[i], 2),
            "true_change_frac": round(cov[i]["true_change_frac"], 4),
            "mangrove_frac": round(cov[i]["mangrove_frac"], 4),
            "f1ch_by_model": {m: round(f1[m][i], 3) for m in models},
        } for i in eval_idx},
    }
    (ROOT / "results" / "distance_covariates.json").write_text(
        json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("covariates", "ols", "per_model")}, indent=1))


if __name__ == "__main__":
    main()
