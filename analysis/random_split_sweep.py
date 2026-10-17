"""Robustness of the spatial-shortcut quantification across
10 independent random split seeds.

The single-seed counterfactual experiment showed that an interleaved random
split inflates the within-region change F1 by 29--36 pp relative to the
buffer-isolated protocol. This script aggregates the sweep (10 split seeds
x 5 main models x 3 run seeds, results/random_split_k{1..10}) against the
same honest baseline (exp3 test_within_region, the spatial protocol):

  inflation      = rnd_rnd - iso_iso   (full leaky protocol)
  training effect= rnd_iso - iso_iso   (random TRAINING only)

per (split seed, run seed), then pooled over all 30 contrasts per model
(mean / std / range), plus the 10 per-split-seed means so the headline
number is not an artefact of one lucky split.

Writes results/random_split_sweep.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from mangrove.paths import RESULTS_ROOT
from aggregate_results import binary_change

MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]
RUN_SEEDS = [2026, 2027, 2028]
SPLIT_SEEDS = list(range(1, 11))
COLLAPSE_T = 0.05  # random-test F1ch below this = all-background collapse


def fetch(model, sub, key, seeds):
    """{run_seed: F1_change} from results/{sub}/metrics_<model>_s<seed>.json."""
    out = {}
    for s in seeds:
        p = RESULTS_ROOT / (sub or "") / f"metrics_{model}_s{s}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text()).get(key)
        if not rec or "confusion_matrix" not in rec:
            continue
        out[s] = binary_change(rec["confusion_matrix"])["F1_change"]
    return out


def summarize(vals):
    if not vals:
        return None
    v = np.asarray(vals, dtype=float)
    return {"mean_pp": round(float(v.mean()) * 100, 2),
            "std_pp": round(float(v.std(ddof=1)) * 100, 2) if len(v) > 1 else 0.0,
            "min_pp": round(float(v.min()) * 100, 2),
            "max_pp": round(float(v.max()) * 100, 2),
            "n": int(len(v))}


def main():
    out = {"config": {"models": MODELS, "run_seeds": RUN_SEEDS,
                      "split_seeds": SPLIT_SEEDS,
                      "note": "baseline = exp3 test_within_region (spatial protocol); "
                              "contrasts per model = (present) split seeds x 3 run seeds; "
                              "*_excl_collapse blocks exclude all-background-collapse runs "
                              f"(random-test F1ch < {COLLAPSE_T})"}}
    per_model = {}
    for model in MODELS:
        iso = fetch(model, "", "test_within_region", RUN_SEEDS)
        if len(iso) < 3:
            print(f"skip {model} (baseline seeds missing: {sorted(iso)})")
            continue
        infl, train = [], []
        per_split_infl, per_split_train = {}, {}
        infl_x, train_x = [], []
        per_split_infl_x, per_split_train_x = {}, {}
        for k in SPLIT_SEEDS:
            sub = f"random_split_k{k}"
            rr = fetch(model, sub, "random_test", RUN_SEEDS)
            ri = fetch(model, sub, "isolated_test", RUN_SEEDS)
            for s in RUN_SEEDS:
                if s in iso and s in rr:
                    infl.append(rr[s] - iso[s])
                    if rr[s] >= COLLAPSE_T:
                        infl_x.append(rr[s] - iso[s])
                if s in iso and s in ri:
                    train.append(ri[s] - iso[s])
                    if rr.get(s, 0.0) >= COLLAPSE_T:
                        train_x.append(ri[s] - iso[s])
            if rr:
                per_split_infl[str(k)] = round(float(np.mean([rr[s] - iso[s] for s in rr])) * 100, 2)
                excl = [rr[s] - iso[s] for s in rr if rr[s] >= COLLAPSE_T]
                if excl:
                    per_split_infl_x[str(k)] = round(float(np.mean(excl)) * 100, 2)
            if ri:
                per_split_train[str(k)] = round(float(np.mean([ri[s] - iso[s] for s in ri])) * 100, 2)
                excl = [ri[s] - iso[s] for s in ri if rr.get(s, 0.0) >= COLLAPSE_T]
                if excl:
                    per_split_train_x[str(k)] = round(float(np.mean(excl)) * 100, 2)
        per_model[model] = {
            "iso_iso_F1ch": {str(s): round(v, 4) for s, v in iso.items()},
            "inflation_F1ch_pp": {**summarize(infl),
                                  "per_split_mean_pp": per_split_infl},
            "inflation_F1ch_pp_excl_collapse": {**summarize(infl_x),
                                                "per_split_mean_pp": per_split_infl_x},
            "training_effect_F1ch_pp": {**summarize(train),
                                        "per_split_mean_pp": per_split_train},
            "training_effect_F1ch_pp_excl_collapse": {**summarize(train_x),
                                                      "per_split_mean_pp": per_split_train_x},
        }
        i, t = per_model[model]["inflation_F1ch_pp"], per_model[model]["training_effect_F1ch_pp"]
        print(f"{model:18s} inflation {i['mean_pp']:+7.2f} +/- {i['std_pp']:.2f} pp "
              f"[{i['min_pp']:+.1f}, {i['max_pp']:+.1f}]  (n={i['n']}) | "
              f"training effect {t['mean_pp']:+7.2f} +/- {t['std_pp']:.2f} pp")
    out["models"] = per_model
    (RESULTS_ROOT / "random_split_sweep.json").write_text(json.dumps(out, indent=1))
    print("saved results/random_split_sweep.json")


if __name__ == "__main__":
    main()
