"""Aggregate the R1 non-overlapping block counterfactual (M3-7a).

Reads
    results/uniform/block_blk<ss>/metrics_<model>_s<seed>.json
    results/uniform/block_hon<ss>/metrics_<model>_s<seed>.json

and reports two contrasts, both of which the block arm was built to support:

  between_arm   blk-trained -> blk-test   MINUS  hon-trained -> hon-test
                the protocol contrast, matching how the 2x2 factorial treats
                II vs RR: it changes training arrangement AND test arrangement
                together, at identical counts.

  within_model  blk-trained -> blk-test   MINUS  blk-trained -> hon-test
                same trained model, two test sets. This isolates the TEST-set
                arrangement alone (interleaved-but-separated vs zoned) with the
                training set held exactly fixed. It is the cleaner answer to
                "does a non-overlapping interleaved test set still score
                higher?", and it is available from every run because each run
                also scores its partner arm.

Both are reported per model, per split seed, and pooled, with the between- and
within-split spread, because with 3-6 test patches per split the per-split
estimates are noisy and the pooled mean alone would overstate precision.

Output: results/uniform/block_counterfactual.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "uniform"
SEEDS_BY_ARM = {}
MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]


def f1ch(block):
    """Binary change F1 (the paper's F1ch) from a metrics block.

    The training scripts call mangrove.metrics.cd_metrics, which returns
    CD_Accuracy_changed / Omission_Error / Commission_Error -- binary
    *accuracy* against the changed super-class, NOT F1. F1ch therefore has to
    be derived from the 4x4 confusion matrix on the loss/gain super-class.
    Validated against the main sweep: identical to the value
    analysis/aggregate_results.py reports for the same run.
    """
    if not block:
        return None
    cm, names = block.get("confusion_matrix"), block.get("class_names")
    if not cm or not names:
        return None
    loss, gain = names.index("loss"), names.index("gain")
    # Canonical change-super-class F1 (loss+gain = "changed"), identical to
    # analysis/aggregate_results.binary_change. The previous strict form
    # (tp = cm[loss][loss] + cm[gain][gain]) treated loss<->gain as errors and
    # double-counted those off-diagonals in both fn and fp -- NOT the paper's
    # F1ch. cm is indexed [gt][pred].
    changed = [loss, gain]
    unchanged = [i for i in range(len(names)) if i not in (loss, gain)]
    tp = sum(cm[i][j] for i in changed for j in changed)
    fn = sum(cm[i][j] for i in changed for j in unchanged)
    fp = sum(cm[i][j] for i in unchanged for j in changed)
    d = 2 * tp + fn + fp
    return float(2 * tp / d) if d else None


def load(sub, model, seed):
    p = RES / sub / f"metrics_{model}_s{seed}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def find_subs(prefix):
    return sorted(d.name for d in RES.glob(prefix + "*") if d.is_dir())


def main():
    blk_subs = find_subs("block_blk")
    hon_subs = {s.replace("block_hon", "block_blk"): s for s in find_subs("block_hon")}
    ss_list = sorted(s.replace("block_blk", "") for s in blk_subs)
    if not ss_list:
        print("no block_blk* subdirs under results/uniform/ -- nothing to do")
        return

    per_run = []
    for ss in ss_list:
        for model in MODELS:
            b_sub, h_sub = f"block_blk{ss}", f"block_hon{ss}"
            if h_sub not in hon_subs.values() and not (RES / h_sub).exists():
                continue
            for seed in (2026, 2027, 2028, 2029, 2030):
                b = load(b_sub, model, seed)
                h = load(h_sub, model, seed)
                if not b or not h:
                    continue
                b_own = f1ch(b.get("block_test"))
                b_partner = f1ch(b.get("partner_test"))
                h_own = f1ch(h.get("block_test"))
                h_partner = f1ch(h.get("partner_test"))
                if b_own is None or h_own is None:
                    continue
                per_run.append({
                    "split_seed": ss, "model": model, "run_seed": seed,
                    "blk_trained_on_blk_test": b_own,
                    "blk_trained_on_hon_test": b_partner,
                    "hon_trained_on_hon_test": h_own,
                    "hon_trained_on_blk_test": h_partner,
                    "between_arm_pp": 100.0 * (b_own - h_own),
                    "within_model_pp": (None if b_partner is None
                                        else 100.0 * (b_own - b_partner)),
                })

    if not per_run:
        print("no complete (arm, model, seed) triples found")
        return

    def agg(key, rows=None):
        rows = rows if rows is not None else per_run
        v = [r[key] for r in rows if r.get(key) is not None]
        if not v:
            return {}
        return {"n": len(v), "mean_pp": float(np.mean(v)),
                "std_pp": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "min_pp": float(np.min(v)), "max_pp": float(np.max(v)),
                "n_positive": int(sum(1 for x in v if x > 0))}

    per_model = {}
    for model in MODELS:
        rows = [r for r in per_run if r["model"] == model]
        if rows:
            per_model[model] = {
                "between_arm": agg("between_arm_pp", rows),
                "within_model": agg("within_model_pp", rows),
                "n_runs": len(rows),
            }

    per_split = {}
    for ss in sorted({r["split_seed"] for r in per_run}):
        rows = [r for r in per_run if r["split_seed"] == ss]
        per_split[ss] = {"between_arm": agg("between_arm_pp", rows),
                         "within_model": agg("within_model_pp", rows),
                         "n_runs": len(rows)}

    # paired sign test on the pooled deltas (the honest resampling unit here is
    # the split seed, not the run seed -- runs within a split share the split)
    for tag, key in (("between_arm", "between_arm_pp"),
                     ("within_model", "within_model_pp")):
        # snapshot the split keys first: we add summary keys to per_split below
        ss_keys = [ss for ss in per_split
                   if isinstance(per_split[ss], dict) and tag in per_split[ss]]
        per_split_means = [per_split[ss][tag].get("mean_pp") for ss in ss_keys]
        per_split_means = [x for x in per_split_means if x is not None]
        per_split[tag + "_split_means"] = per_split_means
        if per_split_means:
            n = len(per_split_means)
            k = sum(1 for x in per_split_means if x > 0)
            per_split[tag + "_sign_test"] = {
                "n_splits": n, "n_positive": k,
                "note": ("exact two-sided sign test over split seeds; with "
                         f"n={n} the smallest attainable p is "
                         f"{2/n if n <= 4 else 0.0:.3f}"),
            }

    out = {
        "config": {
            "arms": {
                "blk": "block-interleaved, zero footprint overlap, one-block "
                       "void band; all test patches inside the train extent",
                "hon": "isolated protocol zones, subsampled to identical counts, "
                       "prevalence-matched; buffer-isolated",
            },
            "split_seeds": ss_list,
            "models": MODELS,
            "note": ("Both arms have identical counts, identical recipe and zero "
                     "overlap. Counts are 19-25 train / 3-6 test against the "
                     "protocol's 40/10/10; that reduction is forced by the "
                     "geometry (see experiments/make_block_split.py) and is why "
                     "the split seed, not the run seed, is the resampling unit."),
        },
        "pooled": {"between_arm": agg("between_arm_pp"),
                   "within_model": agg("within_model_pp")},
        "per_model": per_model,
        "per_split": per_split,
        "per_run": per_run,
    }
    (RES / "block_counterfactual.json").write_text(json.dumps(out, indent=1))

    print("== R1 non-overlapping block counterfactual ==")
    print("   split seeds: %s   models: %d   runs: %d"
          % (",".join(ss_list), len(MODELS), len(per_run)))
    print()
    p = out["pooled"]
    for tag in ("between_arm", "within_model"):
        a = p[tag]
        if not a:
            continue
        print("  %-13s %+6.1f pp  (sd %4.1f, range %+.1f..%+.1f, "
              "%d/%d runs positive)"
              % (tag, a["mean_pp"], a["std_pp"], a["min_pp"], a["max_pp"],
                 a["n_positive"], a["n"]))
    print()
    print("  per split seed:")
    for ss, v in per_split.items():
        if not isinstance(v, dict) or "between_arm" not in v:
            continue
        b, w = v["between_arm"], v["within_model"]
        print("    ss=%-4s between %+6.1f (n=%d)   within %+6.1f (n=%d)"
              % (ss, b.get("mean_pp", float("nan")), b.get("n", 0),
                 w.get("mean_pp", float("nan")), w.get("n", 0)))
    print()
    print("  per model (between_arm / within_model, pp):")
    for m, v in per_model.items():
        b, w = v["between_arm"], v["within_model"]
        print("    %-16s %+6.1f / %+6.1f   (n=%d)"
              % (m, b.get("mean_pp", float("nan")),
                 w.get("mean_pp", float("nan")), v["n_runs"]))
    print()
    print("  -> results/uniform/block_counterfactual.json")


if __name__ == "__main__":
    sys.exit(main())
