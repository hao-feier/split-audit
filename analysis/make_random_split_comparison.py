"""Build <root>/random_split/comparison.json for tab:rand (uniform protocol).

Pools the 3-split-draw counterfactual sweep (random_split_k{1,2,3}, 5 models
x 3 run seeds = 45 runs) into the same per-model/per-protocol layout as the
sqrt-era single-draw comparison:
    {model: {"rnd_iso": {run: metrics}, "rnd_rnd": {run: metrics}}}
with run ids "k<split_seed>-s<run_seed>". Runs that collapsed to an
all-background solution under the unweighted recipe (random-test F1ch below
COLLAPSE_T; U-Net family, run-seed 2026 only) are EXCLUDED and listed in
"_config"; the paper (Section sec:rand) reports pooled statistics on the
remaining runs and gives the all-runs sensitivity in the text.

Run:  MANGROVE_RESULTS_ROOT=results/uniform python3 analysis/make_random_split_comparison.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from mangrove.paths import RESULTS_ROOT
from aggregate_results import binary_change

MODELS = ["unet", "unet_cbam", "unet_msa", "unet_cbam_msa", "deeplabv3plus",
          "deeplabv3plus_pre", "temporal_unet", "temporal_unet_gn", "fc_siam",
          "changeformer"]
KS = [1, 2, 3]
SEEDS = [2026, 2027, 2028]
COLLAPSE_T = 0.05  # random-test F1ch below this = all-background collapse


def metrics_of(rec):
    out = dict(binary_change(rec["confusion_matrix"]))
    out["OA"] = rec["OA"]
    out["mIoU"] = rec["mIoU"]
    out["kappa"] = rec["kappa"]
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def main():
    comp, excluded = {}, []
    for m in MODELS:
        comp[m] = {"rnd_iso": {}, "rnd_rnd": {}}
        for k in KS:
            for s in SEEDS:
                f = RESULTS_ROOT / f"random_split_k{k}" / f"metrics_{m}_s{s}.json"
                d = json.loads(f.read_text())
                run = f"k{k}-s{s}"
                if binary_change(d["random_test"]["confusion_matrix"])["F1_change"] < COLLAPSE_T:
                    excluded.append(f"{m}/{run}")
                    continue
                comp[m]["rnd_iso"][run] = metrics_of(d["isolated_test"])
                comp[m]["rnd_rnd"][run] = metrics_of(d["random_test"])
    out = {"_config": {"protocol": "uniform (primary), cw none",
                       "split_seeds": KS, "run_seeds": SEEDS,
                       "collapse_criterion": f"random-test F1ch < {COLLAPSE_T}",
                       "excluded_collapsed_runs": excluded},
           **comp}
    dest = RESULTS_ROOT / "random_split" / "comparison.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"saved {dest}; excluded: {excluded}")


if __name__ == "__main__":
    main()
