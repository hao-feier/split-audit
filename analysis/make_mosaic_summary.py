"""Propagate the de-duplicated (mosaic) evaluation into the table pipeline.

R2 established that the pooled regional metrics double-count pixels: test
patches are 512 px on a 256 px stride, so a pixel is counted once per covering
patch (1.85x-2.50x, up to 4x). `analysis/mosaic_eval.py` recomputes every
model x seed x region confusion matrix on a partition of the test footprints
(nearest covering patch; the three assignment rules agree to <= 0.02 F1ch and
the per-region leader is invariant in 12/12 cases, so the rule is a free
choice).

This script writes a metrics_summary.json with the SAME schema as the pooled
one but with every region block replaced by its mosaic counterpart, so
`analysis/make_latex_tables.py` and the supplementary-table generator run
unchanged. It is a drop-in switch:

    python analysis/make_mosaic_summary.py            # -> results/uniform/metrics_summary_mosaic.json
    MANGROVE_TABLES_METRICS=... make_latex_tables.py  (or edit PRIMARY_METRICS)

It also reports, per region, what actually changed, so the size of the
correction is visible rather than silent.

SCOPE NOTE (important). Only the three main evaluation regions have prediction
dumps. The 120 counterfactual runs behind the spatial-protocol factorial
(§sec:rand) have neither checkpoints nor dumps under results/uniform/, so the
2x2 cells and the 31--36 pp headline stay on pooled numbers. This is a stated
asymmetry, not an oversight: R2 bounded the effect of the correction on the
headline at <= 0.07% of 31 pp, because the honest cell is the *more*
duplicated one (dup 2.22 vs 1.00-1.25 for the interleaved draws). Mixing a
mosaic I/I cell with a pooled R/R cell would be *worse*, not better.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "uniform"
REGION_MAP = {"vietnam": "within", "beilunhe": "beilunhe", "zhanjiang": "zhanjiang"}


def cohen_kappa(cm):
    """Cohen's kappa from a square confusion matrix (equivalent to
    sklearn.metrics.cohen_kappa_score on the underlying labels)."""
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    if n == 0:
        return None
    po = np.trace(cm) / n
    pe = float((cm.sum(axis=1) * cm.sum(axis=0)).sum()) / (n * n)
    return float((po - pe) / (1 - pe)) if pe != 1.0 else None


def main():
    src = json.loads((RES / "metrics_summary.json").read_text())
    mos = json.loads((RES / "mosaic_eval.json").read_text())
    out = {}
    n_filled = 0
    n_missing = []

    for model, seeds in src.items():
        out[model] = {}
        for seed, blocks in seeds.items():
            new = dict(blocks)          # keep _params_M / _epochs
            for reg, key in REGION_MAP.items():
                rec = mos.get(f"{model}_s{seed}_{reg}")
                if rec is None:
                    n_missing.append(f"{model}_s{seed}_{reg}")
                    continue
                nb = rec["nearest_centre"]
                new[key] = {
                    "n_pixels": nb["n_pixels"],
                    "cm": nb["confusion_matrix"],
                    "OA": nb["OA"],
                    "kappa": cohen_kappa(nb["confusion_matrix"]),
                    "kappa_change": None,
                    "F1_macro": nb["mIoU"],
                    "mIoU": nb["mIoU"],
                    "F1_change": nb["F1_change"],
                    "IoU_change": nb["IoU_change"],
                    "precision_change": nb["precision_change"],
                    "recall_change": nb["recall_change"],
                    "Omission": nb["Omission"],
                    "Commission": nb["Commission"],
                    "per_class": {k: {"F1": v, "precision": None,
                                      "recall": None}
                                  for k, v in nb["per_class_F1"].items()},
                }
                n_filled += 1
            out[model][seed] = new

    dst = RES / "metrics_summary_mosaic.json"
    dst.write_text(json.dumps(out, indent=1))
    print("wrote %s" % dst)
    print("   %d model x seed x region blocks replaced by the mosaic; %d missing"
          % (n_filled, len(n_missing)))
    if n_missing:
        print("   missing (kept pooled): %s%s"
              % (", ".join(n_missing[:6]), " ..." if len(n_missing) > 6 else ""))

    # ---- what changed -------------------------------------------------
    print()
    print("== pooled -> mosaic, 5-seed means (primary uniform models) ==")
    for reg, key in REGION_MAP.items():
        print("  %s:" % reg)
        for model in sorted(src):
            pv, mv = [], []
            for seed in src[model]:
                a = src[model][seed].get(key, {}).get("F1_change")
                b = out[model][seed].get(key, {}).get("F1_change")
                if a is not None:
                    pv.append(a)
                if b is not None:
                    mv.append(b)
            if len(pv) < 3:
                continue
            print("    %-22s F1ch %.3f -> %.3f  (%+.3f)   OA %.3f -> %.3f"
                  % (model, np.mean(pv), np.mean(mv), np.mean(mv) - np.mean(pv),
                     np.mean([src[model][s][key]["OA"] for s in src[model]
                              if key in src[model][s]]),
                     np.mean([out[model][s][key]["OA"] for s in src[model]
                              if key in out[model][s]])))
        print()

    # leader invariance
    print("== leader invariance ==")
    for reg, key in REGION_MAP.items():
        p, m = {}, {}
        for model in sorted(src):
            pv = [src[model][s].get(key, {}).get("F1_change") for s in src[model]]
            mv = [out[model][s].get(key, {}).get("F1_change") for s in src[model]]
            if all(x is not None for x in pv) and len(pv) >= 3:
                p[model] = np.mean(pv)
                m[model] = np.mean(mv)
        lp = max(p, key=p.get)
        lm = max(m, key=m.get)
        print("  %-10s pooled=%-22s mosaic=%-22s %s"
              % (reg, lp, lm, "(unchanged)" if lp == lm else "*** CHANGED ***"))


if __name__ == "__main__":
    sys.exit(main())
