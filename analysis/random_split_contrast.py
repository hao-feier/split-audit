"""Compare the spatially-isolated protocol (exp3) vs the random-split control
(exp3c) to quantify the cost of ignoring spatial autocorrelation.

For each model (mean +- std over seeds) it reports change-class metrics under
three protocols, all on the SAME Vietnam patch pool:

  iso_iso  : isolated-trained model  -> isolated (buffer) test   [honest, = exp3 test_within_region]
  rnd_iso  : random-trained model    -> isolated (buffer) test   [effect of random TRAINING only]
  rnd_rnd  : random-trained model    -> random (interleaved) test [full "leaky" protocol]

  inflation      = rnd_rnd - iso_iso   (standard: ignoring spatial autocorrelation)
  training effect= rnd_iso - iso_iso   (isolating the training-assignment effect)

Reuses aggregate_results.binary_change so F1_change / IoU_change / Omission /
Commission are identical to the paper tables.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from mangrove.paths import RESULTS_ROOT
from aggregate_results import binary_change, MODELS, MODEL_LABEL


def load(model, seed, sub=""):
    p = (RESULTS_ROOT / sub / f"metrics_{model}_s{seed}.json")
    return json.loads(p.read_text()) if p.exists() else None


def fetch(model, sub, key, seeds):
    """Per-seed metric dicts read straight from the files (reliable path)."""
    rows = []
    for s in seeds:
        m = load(model, s, sub)
        rec = m.get(key) if m else None
        if not rec or "confusion_matrix" not in rec:
            continue
        d = binary_change(rec["confusion_matrix"])
        d.update({"OA": rec["OA"], "mIoU": rec["mIoU"], "kappa": rec["kappa"]})
        rows.append({"seed": s, **d})
    return rows


def cell(rows, c):
    vals = [r[c] for r in rows if r.get(c) is not None]
    if not vals:
        return "-"
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return f"{mu:.4f}" if sd == 0.0 else f"{mu:.4f}\u00b1{sd:.4f}"


def main():
    seeds = [2026, 2027, 2028]
    cols = ["OA", "mIoU", "F1_change", "IoU_change", "Omission", "Commission"]
    order = [m for m in MODELS if (RESULTS_ROOT / f"metrics_{m}_s2026.json").exists()]

    print("Spatial-autocorrelation control: isolated vs random split (Vietnam, mean+/-std over seeds)")
    print("protocols: iso_iso=honest | rnd_iso=random-train/isolated-test | rnd_rnd=full leaky\n")
    print(f"{'model':<26}{'protocol':<10}" + "".join(f"{c:>16}" for c in cols))

    def prow(label, rows, model):
        print(f"{MODEL_LABEL.get(model, model):<26}{label:<10}" +
              "".join(f"{cell(rows, c):>16}" for c in cols))

    def byseed(rows):
        return {r["seed"]: r for r in rows}

    def pp(lst):
        if not lst:
            return "-", 0
        return f"{float(np.mean(lst)) * 100:+.2f}", len(lst)

    comp = {}
    for model in order:
        iso_rows = fetch(model, "", "test_within_region", seeds)
        rndiso_rows = fetch(model, "random_split", "isolated_test", seeds)
        rndrnd_rows = fetch(model, "random_split", "random_test", seeds)
        print()
        prow("iso_iso", iso_rows, model)
        prow("rnd_iso", rndiso_rows, model)
        prow("rnd_rnd", rndrnd_rows, model)

        iso_b, rndi, rndr = byseed(iso_rows), byseed(rndiso_rows), byseed(rndrnd_rows)
        d_infl = [rndr[s]["F1_change"] - iso_b[s]["F1_change"]
                  for s in seeds if s in iso_b and s in rndr]
        d_train = [rndi[s]["F1_change"] - iso_b[s]["F1_change"]
                   for s in seeds if s in iso_b and s in rndi]
        infl_s, n1 = pp(d_infl)
        train_s, n2 = pp(d_train)
        print(f"{'':<26}{'\u0394 rnd-rnd-iso':<10}{(infl_s + ' pp' if n1 else '-'):>16}"
              f"   (F1_change inflation; n={n1})")
        print(f"{'':<26}{'\u0394 rnd-iso-train':<10}{(train_s + ' pp' if n2 else '-'):>16}"
              f"   (F1_change; random-training-only effect; n={n2})")

        comp[model] = {
            "iso_iso": {str(r["seed"]): {k: v for k, v in r.items() if k != "seed"} for r in iso_rows},
            "rnd_iso": {str(r["seed"]): {k: v for k, v in r.items() if k != "seed"} for r in rndiso_rows},
            "rnd_rnd": {str(r["seed"]): {k: v for k, v in r.items() if k != "seed"} for r in rndrnd_rows},
        }

    (RESULTS_ROOT / "random_split" / "comparison.json").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "random_split" / "comparison.json").write_text(json.dumps(comp, indent=2))
    print("\nwrote results/random_split/comparison.json")


if __name__ == "__main__":
    main()
