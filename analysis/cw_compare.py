"""Class-weight sensitivity, uniform (none) vs sqrt, full sweep.

Reads results/ (sqrt, main sweep) and results/uniform/ (same models/seeds,
--cw-mode none) and reports per model/region: predicted vs reference change
fraction, commission/omission shift, mIoU/F1ch. Also folds in the matched
25-epoch U-Net probes (results/probes/metrics_unet_cwprobe_*.json) for the
inv/sqrt/uniform three-way comparison.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.paths import RESULTS_ROOT
from aggregate_results import MODELS, MODEL_LABEL, binary_change

SEEDS = [2026, 2027, 2028]
REGIONS = [("within", "test_within_region"),
           ("beilunhe", "independent_test.beilunhe"),
           ("zhanjiang", "independent_test.zhanjiang")]


def load(model, seed, sub):
    p = RESULTS_ROOT / sub / f"metrics_{model}_s{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None


def region_metrics(d, path):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur or cur[part] is None:
            return None
        cur = cur[part]
    return cur


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()

    print("Class-weight sensitivity: uniform (none) vs sqrt (main sweep)")
    print("per model/region, mean over seeds: pred-chg% ref-chg% | com sqrt->uniform | F1ch sqrt->uniform | mIoU sqrt->uniform\n")
    for model in args.models:
        for label, path in REGIONS:
            rows = []
            for sub in ("", "uniform"):
                f1s, coms, oms, preds, refs, mis = [], [], [], [], [], []
                for s in SEEDS:
                    d = load(model, s, sub)
                    if not d:
                        continue
                    rec = region_metrics(d, path)
                    if not rec:
                        continue
                    cm = np.asarray(rec["confusion_matrix"], dtype=np.int64)
                    n = cm.sum()
                    bc = binary_change(rec["confusion_matrix"])
                    f1s.append(bc["F1_change"]); coms.append(bc["Commission"]); oms.append(bc["Omission"])
                    preds.append(100 * (cm[:, 1].sum() + cm[:, 2].sum()) / n)
                    refs.append(100 * (cm[1].sum() + cm[2].sum()) / n)
                    mis.append(rec["mIoU"])
                if not f1s:
                    rows.append(None)
                    continue
                rows.append({"pred": np.mean(preds), "ref": np.mean(refs),
                             "com": np.mean(coms), "om": np.mean(oms),
                             "f1": np.mean(f1s), "miou": np.mean(mis)})
            if rows[0] is None or rows[1] is None:
                continue
            a, b = rows  # sqrt, uniform
            print(f"{MODEL_LABEL.get(model, model):26s} {label:10s} "
                  f"pred {a['pred']:.2f}->{b['pred']:.2f}% (ref {a['ref']:.2f}%) | "
                  f"com {a['com']:.3f}->{b['com']:.3f} | "
                  f"F1ch {a['f1']:.3f}->{b['f1']:.3f} | "
                  f"mIoU {a['miou']:.3f}->{b['miou']:.3f}")
        print()

    print("Matched 25-epoch U-Net probes (results/probes/):")
    for f in sorted((RESULTS_ROOT / "probes").glob("metrics_unet_cwprobe_*.json")):
        tag = f.stem.split("cwprobe_")[-1]
        d = json.loads(f.read_text())
        parts = []
        for label, path in REGIONS:
            rec = region_metrics(d, path)
            if not rec:
                continue
            cm = np.asarray(rec["confusion_matrix"], dtype=np.int64)
            n = cm.sum()
            bc = binary_change(rec["confusion_matrix"])
            parts.append(f"{label}: OA {rec['OA']:.3f} mIoU {rec['mIoU']:.3f} "
                         f"F1ch {bc['F1_change']:.3f} com {bc['Commission']:.3f} "
                         f"pred {100*(cm[:,1].sum()+cm[:,2].sum())/n:.2f}%")
        print(f"  {tag:8s} " + " | ".join(parts))


if __name__ == "__main__":
    main()
