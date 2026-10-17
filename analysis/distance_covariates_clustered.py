"""Cluster-robust inference for the distance covariate.

`analysis/distance_covariates.py` fits a pooled OLS

    F1ch ~ 1 + distance_km + true_change_frac

over 26 Vietnam evaluation patches x 5 models = 130 observations and reports
homoskedastic standard errors (sigma^2 (X'X)^-1, no clustering). With one
observation per (patch, model) and a patch-constant regressor, the 5 model
replicates of the same patch are not independent draws, so the naive t is
pseudo-replicated, so the inference must be mixed-effects or cluster-robust.

A patch random intercept is NOT identified here: `distance_km` is patch-constant,
so a random intercept for the patch absorbs the regressor exactly (rank-deficient
within/patch-FE). The defensible alternatives are therefore cluster-robust
standard errors at three levels, plus a cell-level regression:

  * naive            G = 130 (all observations independent)  [the published number]
  * patch-clustered  G = 26  (patches independent; this is the level the
                              regressor actually varies at)
  * model-clustered  G = 5   (models independent)
  * cell-clustered   G = 6   (1024-px spatial cells; the paper's own resampling
                              unit for the LOCO / cluster-bootstrap analyses)
  * cell-level OLS   n = 6   (one observation per cell, patch F1 averaged over
                              models; no model dimension at all)

CR1 definition: V = (X'X)^-1 [ sum_g X_g' u_g u_g' X_g ] (X'X)^-1, with the
usual small-sample factor G/(G-1) * (N-1)/(N-k).

This script is inference-only. It trains nothing and evaluates no test region.

Output: results/distance_covariates_clustered.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REGION = "vietnam"
CELL = 1024
CHANGED = (1, 2)
DIST_NAME = "distance_km"
COV_NAME = "true_change_frac"


def box_m(i, patch_list):
    r, c = patch_list[i]
    return (r * 10.0, (r + 512) * 10.0, c * 10.0, (c + 512) * 10.0)


def min_edge_km(i, j, patch_list):
    a, b = box_m(i, patch_list), box_m(j, patch_list)
    dc = max(0.0, max(a[2], b[2]) - min(a[3], b[3]))
    dr = max(0.0, max(a[0], b[0]) - min(a[1], b[1]))
    return float(np.hypot(dc, dr)) / 1000.0


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return beta, resid, XtX_inv, n, k, r2


def cr1(X, resid, groups, beta, XtX_inv, n, k):
    """Cluster-robust (CR1) covariance and t-stats."""
    uniq = sorted(set(groups))
    G = len(uniq)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in uniq:
        m = np.array([gi == g for gi in groups])
        Xg, ug = X[m], resid[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    # CR1 finite-sample factor
    c = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    V = c * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.diag(V))
    return beta, se, beta / se, G


def main():
    from mangrove.paths import SPLITS_ROOT

    meta = json.loads((ROOT / "data" / "processed" / REGION / "metadata.json").read_text())
    patch_list = list(meta["patches"])
    train = json.loads((SPLITS_ROOT / "vietnam_train.json").read_text())
    test = json.loads((SPLITS_ROOT / "vietnam_test.json").read_text())
    buffer = json.loads((SPLITS_ROOT / "vietnam_buffer.json").read_text())
    eval_idx = sorted(set(test) | set(buffer))
    assert len(eval_idx) == 26, len(eval_idx)

    from mangrove.datasets import load_region
    data = load_region(REGION)
    chg_frac = {}
    for i in eval_idx:
        chg_frac[i] = float(np.isin(np.asarray(data["change"][i]), list(CHANGED)).mean())

    dists = {i: min(min_edge_km(i, t, patch_list) for t in train) for i in eval_idx}

    # 1024-px spatial cell of the patch grid (same definition as
    # analysis/leave_one_cluster_out.py)
    cell_of = {}
    for i in eval_idx:
        r, c = patch_list[i]
        cell_of[i] = (r // CELL, c // CELL)

    dr = json.loads((ROOT / "results" / "distance_response.json").read_text())
    models = dr["config"]["models"]
    f1 = {m: {} for m in models}
    for m in models:
        for b in dr["per_model"][m]:
            for pid, v in b["per_patch"].items():
                f1[m][int(pid)] = float(v)

    # ---- pooled design: 26 patches x 5 models ----
    X, y, g_patch, g_model, g_cell = [], [], [], [], []
    for m in models:
        for i in eval_idx:
            X.append([1.0, dists[i], chg_frac[i]])
            y.append(f1[m][i])
            g_patch.append(i)
            g_model.append(models.index(m))
            g_cell.append("%d_%d" % cell_of[i])
    X = np.array(X)
    y = np.array(y)
    beta, resid, XtX_inv, n, k, r2 = ols(X, y)
    se_naive = np.sqrt(np.diag(XtX_inv * (resid @ resid) / (n - k)))

    variants = {}
    variants["naive_G130"] = {
        "G": 130, "n": n,
        "beta": [float(b) for b in beta],
        "se": [float(s) for s in se_naive],
        "t": [float(b / s) for b, s in zip(beta, se_naive)],
    }
    for tag, groups in (("patch_clustered_G26", g_patch),
                        ("model_clustered_G5", g_model),
                        ("cell_clustered_G6", g_cell)):
        b_, se_, t_, G = cr1(X, resid, groups, beta, XtX_inv, n, k)
        variants[tag] = {
            "G": G, "n": n,
            "beta": [float(v) for v in b_],
            "se": [float(v) for v in se_],
            "t": [float(v) for v in t_],
        }

    # ---- cell-level regression: one obs per cell, F1 averaged over models ----
    cells = sorted(set(g_cell))
    Xc, yc, gc = [], [], []
    for c in cells:
        idxs = [i for i in eval_idx if "%d_%d" % cell_of[i] == c]
        Xc.append([1.0, float(np.mean([dists[i] for i in idxs])),
                   float(np.mean([chg_frac[i] for i in idxs]))])
        yc.append(float(np.mean([f1[m][i] for m in models for i in idxs])))
        gc.append(c)
    Xc = np.array(Xc)
    yc = np.array(yc)
    bc, resc, XtX_inv_c, nc, kc, r2c = ols(Xc, yc)
    se_c = np.sqrt(np.diag(XtX_inv_c * (resc @ resc) / (nc - kc)))
    variants["cell_level_n6"] = {
        "G": None, "n": nc,
        "beta": [float(v) for v in bc],
        "se": [float(v) for v in se_c],
        "t": [float(v / s) for v, s in zip(bc, se_c)],
        "r2": float(r2c),
    }

    names = ["intercept", DIST_NAME, COV_NAME]
    summary = {}
    for tag, v in variants.items():
        summary[tag] = {
            "n": v["n"], "G": v["G"], "r2": v.get("r2", r2),
            "distance_coef": v["beta"][1],
            "distance_se": v["se"][1],
            "distance_t": v["t"][1],
            "change_frac_coef": v["beta"][2],
            "change_frac_t": v["t"][2],
        }

    out = {
        "config": {
            "region": REGION,
            "n_patches": len(eval_idx),
            "n_models": len(models),
            "n_obs_pooled": n,
            "cell_px": CELL,
            "n_cells": len(cells),
            "cell_sizes": [int(sum(1 for i in eval_idx if "%d_%d" % cell_of[i] == c))
                           for c in cells],
            "spec": "F1ch ~ 1 + distance_km + true_change_frac",
            "cluster_definitions": {
                "patch": "one cluster per evaluation patch (G=26); the level at "
                         "which the regressors actually vary",
                "model": "one cluster per architecture (G=5)",
                "cell": f"one cluster per {CELL}-px spatial cell of the patch "
                        f"grid (G={len(cells)}); the paper's resampling unit in "
                        "the LOCO / cluster-bootstrap analyses",
            },
            "note": ("A patch random intercept is not identified for this "
                     "specification because distance_km is patch-constant, so a "
                     "patch effect absorbs the regressor; cluster-robust SEs are "
                     "the identified alternative. Inference only, no training, "
                     "no test-region selection."),
        },
        "variants": variants,
        "summary": summary,
        "terms": names,
    }
    (ROOT / "results" / "distance_covariates_clustered.json").write_text(
        json.dumps(out, indent=2))

    print("== distance covariate under alternative inference ==")
    print("   spec: F1ch ~ 1 + distance_km + true_change_frac")
    print("   pooled N = %d  (%d patches x %d models)  cells = %s"
          % (n, len(eval_idx), len(models), out["config"]["cell_sizes"]))
    print()
    print("   %-22s %6s %5s %9s %9s %8s"
          % ("variant", "n", "G", "coef", "se", "t"))
    for tag in ("naive_G130", "patch_clustered_G26", "model_clustered_G5",
                "cell_clustered_G6", "cell_level_n6"):
        s = summary[tag]
        print("   %-22s %6d %5s %9.4f %9.4f %8.2f"
              % (tag, s["n"], str(s["G"]), s["distance_coef"],
                 s["distance_se"], s["distance_t"]))
    print()
    print("   two-sided p (|t| >= 1.96 ~ 0.05):")
    for tag, s in summary.items():
        sig = "significant" if abs(s["distance_t"]) >= 1.96 else "NOT significant"
        print("      %-22s t=%6.2f  -> %s" % (tag, s["distance_t"], sig))
    print()
    print("   change_frac control (distance is not the only correlate):")
    for tag, s in summary.items():
        print("      %-22s coef=%+.4f  t=%6.2f"
              % (tag, s["change_frac_coef"], s["change_frac_t"]))


if __name__ == "__main__":
    main()
