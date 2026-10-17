"""Aggregate the I/R factorial cell sweep into results/ir_factorial.json.

Reads results/ir_k{k}/metrics_{model}_s{seed}.json for k in 1..3, seeds
2026..2030, and the five main models. For each model:
  * IR cell  = mean binary-change F1 over the 15 I/R runs (3 draws x 5 seeds)
  * sanity   = isolated_test F1 (should match the main sweep's II cell up to
               run-to-run non-determinism ~0.01)
  * decomposition: dTrain = RI - II, dTest = IR - II, dInt = RR - RI - IR + II
                   (II/RI/RR from results/random_split_sweep.json)

Prints a LaTeX-ready row set for tab:factorial.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.paths import RESULTS_ROOT  # respects MANGROVE_RESULTS_ROOT
MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]
SEEDS = [2026, 2027, 2028, 2029, 2030]
KS = [1, 2, 3]
NAMES = {"unet": "U-Net", "unet_cbam": "U-Net+CBAM",
         "unet_cbam_msa": "proposed (U-Net+CBAM+MSA)",
         "deeplabv3plus": "DeepLabv3+", "temporal_unet": "Temporal U-Net"}
COLLAPSE_T = 0.05  # test F1ch below this = all-background collapse


def f1ch(cm):
    """Binary change F1 (loss+gain vs unchanged) from a 4x4 confusion matrix.

    Canonical definition: change={loss,gain} is one class, so a loss<->gain
    mis-assignment is CORRECT (counted in tp). This matches
    aggregate_results.binary_change, which is what random_split_sweep.py uses
    for the II/RI/RR cells combined with this IR cell in tab:factorial.
    """
    tp = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
    fn = cm[1][0] + cm[1][3] + cm[2][0] + cm[2][3]
    fp = cm[0][1] + cm[0][2] + cm[3][1] + cm[3][2]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    sweep = json.load(open(RESULTS_ROOT / "random_split_sweep.json"))["models"]
    out = {"config": {"k": KS, "seeds": SEEDS, "models": MODELS,
                      "note": "excl_collapse blocks exclude all-background-collapse runs "
                              f"(test F1ch < {COLLAPSE_T}); II unchanged (main sweep)"},
           "per_model": {}}
    missing = []
    for m in MODELS:
        ir_vals, iso_vals = [], []
        for k in KS:
            for s in SEEDS:
                f = RESULTS_ROOT / f"ir_k{k}" / f"metrics_{m}_s{s}.json"
                if not f.exists():
                    missing.append(str(f))
                    continue
                d = json.load(open(f))
                ir_vals.append(f1ch(d["ir_test"]["confusion_matrix"]))
                iso_vals.append(f1ch(d["isolated_test"]["confusion_matrix"]))
        if not ir_vals:
            missing.append(f"ALL {m}")
            continue
        ii = sum(sweep[m]["iso_iso_F1ch"].values()) / len(sweep[m]["iso_iso_F1ch"])
        ri = ii + sweep[m]["training_effect_F1ch_pp"]["mean_pp"] / 100
        rr = ii + sweep[m]["inflation_F1ch_pp"]["mean_pp"] / 100
        ir = st.mean(ir_vals)
        out["per_model"][m] = {
            "n_runs": len(ir_vals),
            "II": round(ii, 4), "IR": round(ir, 4),
            "IR_std": round(st.stdev(ir_vals), 4) if len(ir_vals) > 1 else None,
            "RI": round(ri, 4), "RR": round(rr, 4),
            "isolated_test_mean": round(st.mean(iso_vals), 4),
            "sanity_gap_vs_II": round(st.mean(iso_vals) - ii, 4),
            "dTrain_pp": round((ri - ii) * 100, 2),
            "dTest_pp": round((ir - ii) * 100, 2),
            "dInt_pp": round((rr - ri - ir + ii) * 100, 2),
        }
        # collapse-excluded variant (paper-reported statistics)
        i_x = sweep[m].get("inflation_F1ch_pp_excl_collapse")
        t_x = sweep[m].get("training_effect_F1ch_pp_excl_collapse")
        ir_x = [v for v in ir_vals if v >= COLLAPSE_T]
        if i_x and t_x and ir_x:
            ri_x = ii + t_x["mean_pp"] / 100
            rr_x = ii + i_x["mean_pp"] / 100
            ir_xm = st.mean(ir_x)
            out["per_model"][m]["excl_collapse"] = {
                "n_runs": len(ir_x),
                "II": round(ii, 4), "IR": round(ir_xm, 4),
                "IR_std": round(st.stdev(ir_x), 4) if len(ir_x) > 1 else None,
                "RI": round(ri_x, 4), "RR": round(rr_x, 4),
                "dTrain_pp": round((ri_x - ii) * 100, 2),
                "dTest_pp": round((ir_xm - ii) * 100, 2),
                "dInt_pp": round((rr_x - ri_x - ir_xm + ii) * 100, 2),
            }
    json.dump(out, open(RESULTS_ROOT / "ir_factorial.json", "w"), indent=1)
    if missing:
        print(f"WARNING: {len(missing)} runs missing (sweep still running?):")
        for x in missing[:10]:
            print("  ", x)
    print(f"{'model':<24} {'II':>6} {'IR':>6} {'RI':>6} {'RR':>6} "
          f"{'dTest':>7} {'dTrain':>7} {'dInt':>7}  iso_gap")
    for m, v in out["per_model"].items():
        print(f"{NAMES[m]:<24} {v['II']:6.3f} {v['IR']:6.3f} {v['RI']:6.3f} "
              f"{v['RR']:6.3f} {v['dTest_pp']:+7.2f} {v['dTrain_pp']:+7.2f} "
              f"{v['dInt_pp']:+7.2f}  {v['sanity_gap_vs_II']:+.4f}")
    print("\nLaTeX rows for tab:factorial (I/R, dTest, dInt):")
    for m, v in out["per_model"].items():
        print(f"  {NAMES[m]} & {v['II']:.3f} & {v['IR']:.3f} & {v['RI']:.3f} & "
              f"{v['RR']:.3f} & ${v['dTest_pp']:+.1f}$ & ${v['dTrain_pp']:+.1f}$ & "
              f"${v['dInt_pp']:+.1f} \\\\")


if __name__ == "__main__":
    sys.exit(main())
