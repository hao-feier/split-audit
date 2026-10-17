"""Bootstrap CIs for the 2x2 train/test factorial decomposition.

For each (split k, seed) run, computes all four factorial cells from
per-run confusion matrices:
  II = f1ch(ir_k{k}/isolated_test)       isolated train, isolated test
  IR = f1ch(ir_k{k}/ir_test)             isolated train, random test
  RI = f1ch(random_split_k{k}/isolated)  random train, isolated test
  RR = f1ch(random_split_k{k}/random)    random train, random test

Then resamples (k, seed) pairs with replacement (B=2000) and computes
the 95% percentile CI for each decomposition component:
  dTrain = mean(RI) - mean(II)
  dTest  = mean(IR) - mean(II)
  dInt   = mean(RR) - mean(RI) - mean(IR) + mean(II)

Uses the 3 seeds common to both arms (2026-2028) and 3 split draws,
giving 9 runs per model.

Output: results/factorial_bootstrap_ci.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]
SEEDS = [2026, 2027, 2028]
KS = [1, 2, 3]
B = 2000
NAMES = {
    "unet": "U-Net",
    "unet_cbam": "U-Net+CBAM",
    "unet_cbam_msa": "proposed (U-Net+CBAM+MSA)",
    "deeplabv3plus": "DeepLabv3+",
    "temporal_unet": "Temporal U-Net",
}


def f1ch(cm):
    """Binary change F1 (loss+gain vs unchanged) from a 4x4 CM."""
    tp = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
    fn = cm[1][0] + cm[1][3] + cm[2][0] + cm[2][3]
    fp = cm[0][1] + cm[0][2] + cm[3][1] + cm[3][2]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def collect_cells(model):
    """Collect (II, IR, RI, RR) per (k, seed) run."""
    runs = []
    for k in KS:
        for s in SEEDS:
            ir_f = ROOT / "results" / f"ir_k{k}" / f"metrics_{model}_s{s}.json"
            rs_f = ROOT / "results" / f"random_split_k{k}" / f"metrics_{model}_s{s}.json"
            if not ir_f.exists() or not rs_f.exists():
                continue
            ir_d = json.loads(ir_f.read_text())
            rs_d = json.loads(rs_f.read_text())
            ii = f1ch(ir_d["isolated_test"]["confusion_matrix"])
            ir = f1ch(ir_d["ir_test"]["confusion_matrix"])
            ri = f1ch(rs_d["isolated_test"]["confusion_matrix"])
            rr = f1ch(rs_d["random_test"]["confusion_matrix"])
            runs.append({"k": k, "seed": s, "II": ii, "IR": ir, "RI": ri, "RR": rr})
    return runs


def bootstrap_decomposition(runs, B=B):
    """Resample (k,seed) runs and compute CI for dTrain, dTest, dInt."""
    n = len(runs)
    ii = np.array([r["II"] for r in runs])
    ir = np.array([r["IR"] for r in runs])
    ri = np.array([r["RI"] for r in runs])
    rr = np.array([r["RR"] for r in runs])

    dtrain_obs = ri.mean() - ii.mean()
    dtest_obs = ir.mean() - ii.mean()
    dint_obs = rr.mean() - ri.mean() - ir.mean() + ii.mean()

    rng = np.random.default_rng(2026)
    dtrain_bs = np.empty(B)
    dtest_bs = np.empty(B)
    dint_bs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        ii_b = ii[idx].mean()
        ir_b = ir[idx].mean()
        ri_b = ri[idx].mean()
        rr_b = rr[idx].mean()
        dtrain_bs[b] = ri_b - ii_b
        dtest_bs[b] = ir_b - ii_b
        dint_bs[b] = rr_b - ri_b - ir_b + ii_b

    def ci(x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return round(lo * 100, 2), round(hi * 100, 2)

    return {
        "n_runs": n,
        "dTrain_pp": round(dtrain_obs * 100, 2),
        "dTrain_ci95_pp": list(ci(dtrain_bs)),
        "dTest_pp": round(dtest_obs * 100, 2),
        "dTest_ci95_pp": list(ci(dtest_bs)),
        "dInt_pp": round(dint_obs * 100, 2),
        "dInt_ci95_pp": list(ci(dint_bs)),
    }


def main():
    out = {
        "config": {
            "models": MODELS,
            "seeds": SEEDS,
            "splits": KS,
            "B": B,
            "note": (
                "95% percentile bootstrap CI over (k, seed) runs. "
                "II from ir_k{k}/isolated_test, IR from ir_k{k}/ir_test, "
                "RI from random_split_k{k}/isolated_test, "
                "RR from random_split_k{k}/random_test."
            ),
        },
        "per_model": {},
    }
    for m in MODELS:
        runs = collect_cells(m)
        if not runs:
            out["per_model"][m] = {"error": "no runs found"}
            continue
        out["per_model"][m] = bootstrap_decomposition(runs)

    out_path = ROOT / "results" / "factorial_bootstrap_ci.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved to {out_path.relative_to(ROOT)}")
    for m in MODELS:
        v = out["per_model"].get(m, {})
        if "error" in v:
            print(f"  {NAMES[m]:<30} {v['error']}")
            continue
        print(
            f"  {NAMES[m]:<30} "
            f"dTrain={v['dTrain_pp']:+.2f} {v['dTrain_ci95_pp']}  "
            f"dTest={v['dTest_pp']:+.2f} {v['dTest_ci95_pp']}  "
            f"dInt={v['dInt_pp']:+.2f} {v['dInt_ci95_pp']}"
        )


if __name__ == "__main__":
    main()
