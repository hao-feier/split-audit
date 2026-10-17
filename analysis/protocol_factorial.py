#!/usr/bin/env python3
"""B factorial: controlled single-scenario evaluation-protocol effect (design-based).

Honest scope
------------
This does NOT produce a new point estimate. The spatial protocol effect already
exists in the source study as a fixed-model, two-test-zone contrast: each
``results/random_split_k*/metrics_{model}_s{seed}.json`` stores BOTH the leaky
``random_test`` zone and the buffer-isolated ``isolated_test`` zone confusion
matrix for the SAME trained model. This script:

  1. Formalises that as a declared design-based estimand
         Delta_k = mean_{panel} [ F1ch(random_test_k) - F1ch(isolated_test_k) ]
     where the design-based unit of replication is the 10 independent spatial
     partitions (k1..k10), and the panel (model x seed) is a fixed average.
   2. Attaches a proper 95% t interval on the MEAN (df = K-1) and an MDE@80%
      built from the SAME small-sample t quantiles -- the fix for audit F1 (the
      old headline was a bare "descriptive range" with no stated inference).
      Both use the t distribution, not the z, because K is small (10).
   3. Folds the reference / buffer / prevalence axes (already measured in the
      source study, predictions held fixed) into one controlled design, so the
      checklist moves from "10 correlations" to "one controlled design + a
      declared estimand (the mean) + a t interval + an MDE".

CPU-only, reads stored artefacts only, no model loading. Wording guardrail is
carried in the emitted JSON: single scene / sensor / biome; this is the
evaluation-protocol effect on reported F1 in THIS study, not a universal
spatial-inflation constant.

Output: results/protocol_factorial.json
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPRESENTATIVE = "temporal_unet"  # robustness row (per user decision #1)


# -------------------------------------------------------------------------- F1ch
def f1ch_from_cm(cm: list, change_idx=(1, 2)) -> float:
    """Change-class F1 from a 4x4 confusion matrix (rows=true, cols=pred)."""
    cm = [list(r) for r in cm]
    chg = list(change_idx)
    non = [i for i in range(len(cm)) if i not in chg]
    tp = sum(cm[i][j] for i in chg for j in chg)
    fn = sum(cm[i][j] for i in chg for j in non)
    fp = sum(cm[i][j] for i in non for j in chg)
    return float(2 * tp / max(2 * tp + fn + fp, 1))


def per_class_f1(cm: list, idx: int) -> float:
    r = cm[idx]
    tp = r[idx]
    p = tp + sum(r[j] for j in range(len(r)) if j != idx)  # producer (row)
    u = tp + sum(cm[j][idx] for j in range(len(cm)) if j != idx)  # user (col)
    if (p + u) == 0:
        return 0.0
    return float(2 * (tp / p) * (tp / u) / ((tp / p) + (tp / u)))


def change_indices(names) -> tuple:
    if names and ("loss" in names and "gain" in names):
        return (names.index("loss"), names.index("gain"))
    return (1, 2)


# ------------------------------------------------------------------ partitions
def load_partitions():
    """Return {k: {model: {seed: {leaky_f1, honest_f1, f1_loss_leaky, ...}}}}."""
    dirs = sorted(glob.glob(str(RESULTS / "random_split_k*")))
    part = {}
    for d in dirs:
        key = os.path.basename(d)  # e.g. random_split_k1
        m = {}
        for f in sorted(glob.glob(os.path.join(d, "metrics_*.json"))):
            try:
                doc = json.load(open(f))
            except (OSError, json.JSONDecodeError):
                continue
            model = doc.get("model")
            seed = doc.get("seed")
            if not model or seed is None:
                continue
            rt = doc.get("random_test") or {}
            it = doc.get("isolated_test") or {}
            cmr = rt.get("confusion_matrix")
            cmi = it.get("confusion_matrix")
            if not cmr or not cmi or len(cmr) != 4 or len(cmi) != 4:
                continue  # binary / missing -> excluded from 4-class panel
            ch_r = change_indices(rt.get("class_names"))
            ch_i = change_indices(it.get("class_names"))
            rec = {
                "seed": seed,
                "leaky_F1ch": f1ch_from_cm(cmr, ch_r),
                "honest_F1ch": f1ch_from_cm(cmi, ch_i),
                "leaky_F1loss": per_class_f1(cmr, ch_r[0]),
                "leaky_F1gain": per_class_f1(cmr, ch_r[1]),
                "honest_F1loss": per_class_f1(cmi, ch_i[0]),
                "honest_F1gain": per_class_f1(cmi, ch_i[1]),
                "leaky_npx": rt.get("n_pixels"),
                "honest_npx": it.get("n_pixels"),
            }
            m.setdefault(model, {})[seed] = rec
        if m:
            part[key] = m
    return part


def _z_ppf(q: float) -> float:
    """Standard-normal quantile. scipy if present, else Acklam's approximation
    (max |err| ~ 1.2e-9), so the script stays CPU-only and dependency-optional."""
    try:
        from scipy import stats  # type: ignore
        return float(stats.norm.ppf(q))
    except Exception:
        import math
        a = [-3969683028665376e-6, 2209460984245205e-6, -2759285104469687e-6,
             1383577518672690e-6, -3066479806614716e-6, 2506628277459239e-6]
        b = [-5447609879822406e-6, 1615858368580409e-6, -1556989798598866e-6,
             6680131188771972e-6, -1328068155288572e-6]
        c = [-7784894002430293e-9, -3223964580411365e-6, -2400758277161838e-6,
             -2549732539343734e-6, 4374664141464968e-6, 2294681378348223e-6]
        d = [7784695709041462e-9, 3224671290700398e-6, 2445134137142996e-6,
             3754408661907416e-6]
        plow, phigh = 0.02425, 1 - 0.02425
        if q < plow:
            u = math.sqrt(-2 * math.log(q))
            return (((((c[0]*u+c[1])*u+c[2])*u+c[3])*u+c[4])*u+c[5]) / \
                   ((((d[0]*u+d[1])*u+d[2])*u+d[3])*u+1)
        if q > phigh:
            u = math.sqrt(-2 * math.log(1 - q))
            return -(((((c[0]*u+c[1])*u+c[2])*u+c[3])*u+c[4])*u+c[5]) / \
                   ((((d[0]*u+d[1])*u+d[2])*u+d[3])*u+1)
        u = q - 0.5
        uu = u * u
        return (((((a[0]*uu+a[1])*uu+a[2])*uu+a[3])*uu+a[4])*uu+a[5])*u / \
               (((((b[0]*uu+b[1])*uu+b[2])*uu+b[3])*uu+b[4])*uu+1)


def _t_ppf(q: float, df: int) -> float:
    """Upper-tail t quantile (q = 0.975 for a two-sided 95% interval). scipy if
    present, else a Cornish-Fisher expansion in the normal quantile, accurate to
    well under 0.2% for df >= 5 (our df is 9)."""
    try:
        from scipy import stats  # type: ignore
        return float(stats.t.ppf(q, df))
    except Exception:
        z = _z_ppf(q)
        d = float(df)
        return float(z + (z ** 3 + z) / (4 * d)
                     + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * d * d))


def main() -> int:
    part = load_partitions()
    if not part:
        print("no partitions with 4-class two-zone matrices found", flush=True)
        return 2

    per_k = {}          # k -> {panel_leaky, panel_honest, delta, n_models}
    per_k_rep = {}      # k -> delta for REPRESENTATIVE (if present)
    for k, models in part.items():
        d_leak, d_hon, d_rep = [], [], []
        n_models = set()
        for model, seeds in models.items():
            for seed, rec in seeds.items():
                d_leak.append(rec["leaky_F1ch"])
                d_hon.append(rec["honest_F1ch"])
                n_models.add(model)
                if model == REPRESENTATIVE:
                    d_rep.append(rec["leaky_F1ch"] - rec["honest_F1ch"])
        panel_leak = sum(d_leak) / len(d_leak)
        panel_hon = sum(d_hon) / len(d_hon)
        per_k[k] = {
            "n_model_seed": len(d_leak),
            "n_models": len(n_models),
            "models": sorted(n_models),
            "panel_leaky_F1ch": panel_leak,
            "panel_honest_F1ch": panel_hon,
            "delta_pp": (panel_leak - panel_hon) * 100,
        }
        if d_rep:
            per_k_rep[k] = (sum(d_rep) / len(d_rep)) * 100

    ks = sorted(per_k)
    deltas = [per_k[k]["delta_pp"] for k in ks]
    mean_d = sum(deltas) / len(deltas)
    var = sum((x - mean_d) ** 2 for x in deltas) / (len(deltas) - 1) if len(deltas) > 1 else 0.0
    sd = var ** 0.5
    K = len(deltas)
    df = max(K - 1, 1)
    se = sd / (K ** 0.5)
    t_a = _t_ppf(0.975, df)          # two-sided 95%
    t_b = _t_ppf(0.80, df)           # 80% power
    lo = mean_d - t_a * se           # 95% t interval on the MEAN
    hi = mean_d + t_a * se
    mde = (t_a + t_b) * se           # small-sample MDE, same t family as the CI

    # prevalence stratification (panel-averaged over partitions)
    def strat(field):
        vals = []
        for k, models in part.items():
            for model, seeds in models.items():
                for seed, rec in seeds.items():
                    vals.append(rec[field])
        return (sum(vals) / len(vals)) * 100 if vals else None

    leak_loss, hon_loss = strat("leaky_F1loss"), strat("honest_F1loss")
    leak_gain, hon_gain = strat("leaky_F1gain"), strat("honest_F1gain")

    # ---- reference axis (predictions held fixed) from stored artefacts ----
    def read(p):
        try:
            return json.load(open(p))
        except (OSError, json.JSONDecodeError):
            return None

    ref = {"source": "boundary_tolerance.json, dualsource_subset.json (predictions fixed)"}
    bt = read(RESULTS / "boundary_tolerance.json")
    if bt:
        # largest |shift| in F1ch from boundary perturbation, predictions fixed
        worst = 0.0
        for reg, mods in (bt.get("per_region") or {}).items():
            for mod, ops in mods.items():
                base = (ops.get("dilate_0") or {}).get("mean")
                if base is None:
                    continue
                for op, v in ops.items():
                    if op != "dilate_0" and isinstance(v, dict) and "mean" in v:
                        worst = max(worst, abs(v["mean"] - base))
        ref["boundary_perturb_max_abs_shift_F1ch"] = round(worst, 4)
    ds = read(RESULTS / "dualsource_subset.json")
    if ds:
        shifts = []
        for reg, r in ds.items():
            for mod, mm in (r.get("models") or {}).items():
                full = (mm.get("full") or {}).get("F1ch")
                sub = (mm.get("agreement_subset") or {}).get("F1ch")
                if full is not None and sub is not None:
                    shifts.append(sub - full)
        if shifts:
            ref["label_source_agreement_shift_F1ch"] = {
                "min": round(min(shifts), 4), "max": round(max(shifts), 4),
                "n": len(shifts),
            }

    # ---- buffer / grid-quantisation axis from stored artefacts ----
    buf = {"source": "buffer_sensitivity.json (nominal -> realised grid gap)"}
    bs = read(RESULTS / "buffer_sensitivity.json")
    if bs:
        gaps = {}
        for b, v in (bs.get("per_B") or {}).items():
            g = v.get("effective_val_test_gap_km")
            if g is not None:
                gaps[b] = g
        if gaps:
            distinct = sorted(set(gaps.values()))
            buf["nominal_settings"] = len(gaps)
            buf["distinct_realised_gaps_km"] = distinct
            buf["collapse"] = len(gaps) - len(distinct)
            buf["realised_map"] = gaps

    out = {
        "design": {
            "estimand": "Delta = mean over the K independent spatial partitions of "
                        "[mean_panel F1ch(random_test) - mean_panel F1ch(isolated_test)]; "
                        "the panel (model x seed) is a fixed average",
            "inference": "95% t interval on the mean (df = K-1) and MDE@80% from the same "
                         "small-sample t quantiles; the t (not the z) is used because K is small",
            "guardrail": "single scene / sensor / biome; the interval reflects partition-to-"
                         "partition variation WITHIN this scene, not cross-scene "
                         "generalisation; not a universal spatial-inflation constant",
            "n_partitions": K,
            "representative_model": REPRESENTATIVE,
            "classes": {"change": ["loss", "gain"], "non_change": ["stable_mangrove", "stable_background"]},
        },
        "primary": {
            "delta_pp": round(mean_d, 3),
            "ci95_pp": [round(lo, 3), round(hi, 3)],
            "ci_method": "95% t interval on the mean, df = {0}".format(df),
            "se_pp": round(se, 3),
            "sd_pp": round(sd, 3),
            "mde_80_pp": round(mde, 3),
            "mde_method": "MDE@80% = (t_0.975 + t_0.80) * SE, df = {0}".format(df),
            "power_note": ("the interval includes 0: the effect is not distinguishable "
                            "from zero at this precision" if lo <= 0 <= hi
                            else "the interval excludes 0 and the effect exceeds the MDE"),
            "per_partition_pp": {k: round(per_k[k]["delta_pp"], 3) for k in ks},
            "per_partition_n_modelseed": {k: per_k[k]["n_model_seed"] for k in ks},
            "representative_model_pp": {k: round(per_k_rep[k], 3) for k in per_k_rep},
        },
        "prevalence_strata_pp": {
            "loss": {"leaky": round(leak_loss, 3), "honest": round(hon_loss, 3),
                     "delta": round(leak_loss - hon_loss, 3)} if leak_loss is not None else None,
            "gain": {"leaky": round(leak_gain, 3), "honest": round(hon_gain, 3),
                     "delta": round(leak_gain - hon_gain, 3)} if leak_gain is not None else None,
        },
        "reference_axis": ref,
        "buffer_axis": buf,
        "provenance": {
            "partitions": [os.path.basename(d) for d in
                           sorted(glob.glob(str(RESULTS / "random_split_k*")))],
            "read_from": "results/random_split_k*/metrics_*.json (random_test + isolated_test CMs), "
                         "boundary_tolerance.json, dualsource_subset.json, buffer_sensitivity.json",
        },
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    op = RESULTS / "protocol_factorial.json"
    op.write_text(json.dumps(out, indent=1))

    print(f"partitions={K}  Delta={mean_d:+.2f}pp  CI95=[{lo:+.2f},{hi:+.2f}]  "
          f"MDE80={mde:+.2f}pp  (power: {out['primary']['power_note']})")
    print("per-partition Delta(pp): " + ", ".join(f"{k[-1]}={per_k[k]['delta_pp']:+.1f}" for k in ks))
    if per_k_rep:
        print("representative " + REPRESENTATIVE + " Delta(pp): " +
              ", ".join(f"{k[-1]}={per_k_rep[k]:+.1f}" for k in sorted(per_k_rep)))
    ps = out["prevalence_strata_pp"]
    print(f"strata: loss {ps['loss']}  gain {ps['gain']}")
    print("buffer:", buf)
    print("reference:", ref)
    print("wrote", op)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
