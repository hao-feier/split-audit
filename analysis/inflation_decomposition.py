"""Same-test-set inflation decomposition.

Three inflators are quantified for the model comparison on the shared
test sets (beilunhe, zhanjiang):

  1. Winner's curse (best seed): reporting the best of the 3 seeds for
     each model inflates each model's test metric by (max_seed - mean_seed);
     the *gap* to a competitor is inflated by the difference of the two.
  2. Test-set model selection: choosing the winner ON the test set
     (argmax over models of the test F1ch) inflates the winner's gap over
     the runner-up. We contrast it with choosing the model on the vietnam
     VAL zone only (no test data), which is the honest protocol.
  3. Anti-conservative CI: the M7 patch-level bootstrap resamples
     overlapping patches as independent -> CI width is compared against
     the R2 spatial-block bootstrap (block_bootstrap.json).

Also reported: in-distribution vs cross-region drop (vietnam within-region
test vs independent regions) for the headline models.

Inputs: results/metrics_<model>_s<seed>.json, results/bootstrap_ci.json,
        results/block_bootstrap.json
Output: results/inflation_decomposition.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from aggregate_results import binary_change  # noqa: E402

MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "unet_msa", "deeplabv3plus",
          "temporal_unet", "temporal_unet_gn", "fc_siam"]
SEEDS = [2026, 2027, 2028]
REGIONS = ["beilunhe", "zhanjiang"]


def f1ch_from_cm(cm):
    cm = np.asarray(cm, dtype=np.float64)
    chg = [1, 2]
    tp = cm[np.ix_(chg, chg)].sum()
    fn = cm[np.ix_(chg, [0, 3])].sum()
    fp = cm[np.ix_([0, 3], chg)].sum()
    return 2 * tp / max(2 * tp + fn + fp, 1)


def load_runs(model):
    """{region_key: {seed: {OA, mIoU, F1ch}}} from per-seed metrics JSONs."""
    out = {}
    for s in SEEDS:
        f = ROOT / "results" / f"metrics_{model}_s{s}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        blocks = {"val": d.get("val"), "within": d.get("test_within_region"),
                  "beilunhe": (d.get("independent_test") or {}).get("beilunhe"),
                  "zhanjiang": (d.get("independent_test") or {}).get("zhanjiang")}
        for key, blk in blocks.items():
            if not blk or "confusion_matrix" not in blk:
                continue
            out.setdefault(key, {})[s] = {"OA": blk["OA"], "mIoU": blk["mIoU"],
                                           "F1ch": f1ch_from_cm(blk["confusion_matrix"])}
    return out


def stats(vals):
    vals = np.asarray(vals, dtype=float)
    return {"mean": float(vals.mean()), "max": float(vals.max()),
            "min": float(vals.min()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}


def main():
    runs = {m: load_runs(m) for m in MODELS}
    runs = {m: r for m, r in runs.items() if r}

    out = {"winner_curse": {}, "selection_inflation": {}, "ci_width": {},
           "domain_shift": {}}

    # 1) winner's curse per model x eval set
    for m, r in runs.items():
        for key, per_seed in r.items():
            if len(per_seed) < 2:
                continue
            out["winner_curse"].setdefault(m, {})[key] = {
                k: stats([per_seed[s][k] for s in sorted(per_seed)]) for k in ("OA", "mIoU", "F1ch")}

    # 2) selection inflation per test region
    for region in REGIONS:
        cand = {m: r for m, r in runs.items() if region in r and len(r[region]) == 3}
        if len(cand) < 2:
            continue
        mean_f1 = {m: np.mean([r[region][s]["F1ch"] for s in SEEDS]) for m, r in cand.items()}
        by_test = sorted(mean_f1.items(), key=lambda kv: -kv[1])
        w_test, _ = by_test[0]
        r_test, _ = by_test[1]
        # val-based selection (honest protocol)
        val_f1 = {}
        for m, r in cand.items():
            if "val" in r and len(r["val"]) == 3:
                val_f1[m] = np.mean([r["val"][s]["F1ch"] for s in SEEDS])
        by_val = sorted(val_f1.items(), key=lambda kv: -kv[1])
        w_val, _ = by_val[0]
        r_val, _ = by_val[1]
        gap_test = mean_f1[w_test] - mean_f1[r_test]
        gap_valsel = mean_f1[w_val] - mean_f1[r_val]
        # per-seed paired gap under val selection (no test data used at all)
        per_seed_gap = [cand[w_val][region][s]["F1ch"] - cand[r_val][region][s]["F1ch"]
                        for s in SEEDS]
        # block-bootstrap CI for that paired gap (if available)
        bb = None
        bbf = ROOT / "results" / "block_bootstrap.json"
        if bbf.exists():
            bbd = json.loads(bbf.read_text())
            key = f"{w_val} vs {r_val}"
            bb = (bbd.get("regions", {}).get(region, {})
                    .get("strict_paired", {}).get(key))
        out["selection_inflation"][region] = {
            "winner_on_test": w_test, "runner_up_on_test": r_test,
            "gap_test_selection": float(gap_test),
            "winner_on_val": w_val, "runner_up_on_val": r_val,
            "gap_val_selection": float(gap_valsel),
            "inflation": float(gap_test - gap_valsel),
            "per_seed_gap_valsel": [float(g) for g in per_seed_gap],
            "block_ci_valsel": bb,
        }

    # 3) CI width: M7 patch bootstrap vs R2 block bootstrap (strict)
    bcf = ROOT / "results" / "bootstrap_ci.json"
    bbf = ROOT / "results" / "block_bootstrap.json"
    if bcf.exists() and bbf.exists():
        bcd = json.loads(bcf.read_text())   # keys "model|region" -> {metric: [mean, [lo, hi]]}
        bbd = json.loads(bbf.read_text())
        for m in MODELS:
            for region in ["within", "beilunhe", "zhanjiang"]:
                bkey = "vietnam" if region == "within" else region
                m7 = bcd.get(f"{m}|{bkey}", {}).get("F1ch")
                blk = (bbd.get("regions", {}).get(bkey, {}).get("strict", {})
                          .get(m, {}).get("F1ch"))
                if m7 and blk:
                    w7 = m7[1][1] - m7[1][0]
                    wb = blk["ci95"][1] - blk["ci95"][0]
                    out["ci_width"].setdefault(m, {})[region] = {
                        "m7_width": float(w7), "block_width": float(wb),
                        "ratio_block_over_m7": float(wb / w7)}

    # 4) domain shift (in-distribution within-region test vs cross-region)
    for m in dict.fromkeys(["unet_cbam_msa", "temporal_unet", "fc_siam"]):
        r = runs.get(m)
        if not r or "within" not in r:
            continue
        for region in REGIONS:
            if region not in r:
                continue
            within = [r["within"][s]["F1ch"] for s in SEEDS if s in r["within"]]
            cross = [r[region][s]["F1ch"] for s in SEEDS if s in r[region]]
            if within and cross:
                out["domain_shift"].setdefault(m, {})[region] = {
                    "within_mean": float(np.mean(within)), "cross_mean": float(np.mean(cross)),
                    "drop": float(np.mean(within) - np.mean(cross))}

    (ROOT / "results" / "inflation_decomposition.json").write_text(json.dumps(out, indent=2))
    # console summary
    print("== winner's curse (max-seed minus mean-seed F1ch) ==")
    for m, d in out["winner_curse"].items():
        row = " ".join(f"{k}: +{v['F1ch']['max']-v['F1ch']['mean']:.4f}" for k, v in d.items())
        print(f"  {m}: {row}")
    print("== selection inflation (F1ch gap) ==")
    for r, d in out["selection_inflation"].items():
        print(f"  {r}: test-sel gap={d['gap_test_selection']:.4f} "
              f"({d['winner_on_test']} > {d['runner_up_on_test']}) | "
              f"val-sel gap={d['gap_val_selection']:.4f} "
              f"({d['winner_on_val']} > {d['runner_up_on_val']}) | "
              f"inflation={d['inflation']:.4f} | "
              f"per-seed val-sel gap={np.round(d['per_seed_gap_valsel'], 4).tolist()}"
              + (f" | blockCI={d['block_ci_valsel']['dF1ch']}" if d['block_ci_valsel'] else ""))
    print("== CI width ratio (block / M7 patch) ==")
    for m, d in out["ci_width"].items():
        row = " ".join(f"{k}: {v['ratio_block_over_m7']:.2f}x" for k, v in d.items())
        print(f"  {m}: {row}")
    print("== domain shift (within -> cross-region, F1ch) ==")
    for m, d in out["domain_shift"].items():
        row = " ".join(f"{k}: {v['within_mean']:.4f} -> {v['cross_mean']:.4f} (drop {v['drop']:.4f})"
                       for k, v in d.items())
        print(f"  {m}: {row}")
    print("wrote results/inflation_decomposition.json")


if __name__ == "__main__":
    main()
