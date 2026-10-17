"""Spatial autocorrelation of the MODEL ERROR field.

The label autocorrelation (fig:autocorr) bounds how far the TRUE change field
is structured. But a split leaks to the extent that the MODEL'S ERROR is
structured: if model errors (missed/false change) are autocorrelated over a
longer range than the label, a 2 km buffer (matched to the label range) is
not guaranteed to suffice. Here we measure the enrichment of co-occurrence
of the binary change error (predicted change XOR reference change) of the
proposed model (seed 2026) over the within-region test zone, with the same
lag estimator as analysis/autocorr_decay.py, for the loss and gain
components separately and pooled.

Output: results/autocorr/error_autocorr.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.paths import PROCESSED_ROOT, RESULTS_ROOT

LAGS_M = [10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000]
MODELS = ["unet_cbam_msa", "unet", "temporal_unet"]
SEED = 2026


def curve(masks):
    p = float(masks.mean())
    if p <= 0:
        return None
    p2 = p * p
    H, W = masks.shape[-2:]
    out = {}
    for m in LAGS_M:
        lag = int(round(m / 10))
        if not (0 < lag < min(H, W)):
            continue
        xs = float((masks[:, :, :-lag] * masks[:, :, lag:]).mean())
        ys = float((masks[:, :-lag, :] * masks[:, lag:, :]).mean())
        out[str(m)] = {"co_frac": (xs + ys) / 2, "enrichment": (xs + ys) / 2 / p2,
                       "p_field": p}
    return out


def main():
    res = {}
    for model in MODELS:
        res[model] = {}
        for region in ["vietnam", "beilunhe", "zhanjiang"]:
            pf = ROOT / "results" / "predictions" / f"{model}_s{SEED}_{region}.npy"
            lf = ROOT / "results" / "predictions" / f"{model}_s{SEED}_{region}_labels.npy"
            if not (pf.exists() and lf.exists()):
                continue
            pred = np.load(pf).astype(np.int8)
            lab = np.load(lf).astype(np.int8)
            pc = np.isin(pred, [1, 2])
            lc = np.isin(lab, [1, 2])
            err = (pc != lc).astype(np.float32)
            loss_err = (pc ^ (lab == 1)).astype(np.float32)
            gain_err = (pc ^ (lab == 2)).astype(np.float32)
            res[model][region] = {
                "error": curve(err),
                "loss_error": curve(loss_err),
                "gain_error": curve(gain_err),
            }
    (ROOT / "results" / "autocorr" / "error_autocorr.json").write_text(
        json.dumps(res, indent=1))
    print("saved results/autocorr/error_autocorr.json")
    for model in MODELS:
        for region, d in res[model].items():
            if "error" in d and d["error"]:
                en = {k: round(v["enrichment"], 1) for k, v in d["error"].items()}
                print(model, region, en)


if __name__ == "__main__":
    main()
