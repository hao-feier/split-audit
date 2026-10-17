"""Stage 5 (PLAN 6): aggregate all per-(model, seed) metrics into paper tables.

Reads
  results/metrics_<model>_s<seed>.json   (exp3_train.py, DNNs)
  results/metrics_ml_baselines.json      (exp3b, RF/SVM pixel baselines)

Writes
  results/metrics_summary.json   full per-run records + binary change metrics
  results/table_main.csv         per model x region, mean+-std over seeds
  results/table_main.md          same, markdown (paste into the paper repo)
  results/table_perclass.csv     per-class F1 / IoU (mean over seeds)

Binary change detection metrics (PLAN 6 "Change Detection Accuracy") are
collapsed from the 4-class confusion matrix: changed = {loss, gain},
unchanged = {stable_mangrove, stable_background}. This is the metric that is
not dominated by the ~90% background pixels.
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mangrove.config import CHANGE_CLASSES
from mangrove.paths import RESULTS_ROOT

# 0 stable_mangrove, 1 loss, 2 gain, 3 stable_background
CHANGED = (1, 2)
MODELS = ["unet", "unet_cbam", "unet_msa", "unet_cbam_msa", "deeplabv3plus",
          "temporal_unet", "temporal_unet_gn", "fc_siam", "changeformer",
          "deeplabv3plus_pre"]
MODEL_LABEL = {
    "unet": "U-Net (Exp0)",
    "unet_cbam": "U-Net+CBAM (Exp1)",
    "unet_cbam_msa": "U-Net+CBAM+MSA (Exp2, proposed)",
    "deeplabv3plus": "DeepLabv3+",
    "temporal_unet": "Temporal-UNet",
    "temporal_unet_gn": "Temporal-UNet (GroupNorm)",
    "unet_msa": "U-Net+MSA (ablation)",
    "fc_siam": "FC-Siam (SOTA baseline)",
    "changeformer": "ChangeFormer-style (Transformer)",
    "deeplabv3plus_pre": "DeepLabv3+ (ImageNet-pretrained)",
    "random_forest": "Random Forest (pixel)",
    "svm": "SVM (pixel)",
}
# models whose canonical 3-seed runs belong in the paper tables; tagged R2
# variants (det/nd/nested...) are handled by their own analysis scripts
CANONICAL_SEEDS = (2026, 2027, 2028, 2029, 2030)
EVAL_SETS = [("within", "test_within_region"), ("beilunhe", "independent_test.beilunhe"),
             ("zhanjiang", "independent_test.zhanjiang")]


def binary_change(cm):
    """Collapse a 4x4 confusion matrix to changed/unchanged -> dict of metrics."""
    cm = np.asarray(cm, dtype=np.float64)
    changed = list(CHANGED)
    unchanged = [i for i in range(cm.shape[0]) if i not in changed]
    tp = cm[np.ix_(changed, changed)].sum()
    fn = cm[np.ix_(changed, unchanged)].sum()
    fp = cm[np.ix_(unchanged, changed)].sum()
    tn = cm[np.ix_(unchanged, unchanged)].sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    n = tp + fn + fp + tn
    po = (tp + tn) / max(n, 1)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(n * n, 1)
    kappa = (po - pe) / max(1 - pe, 1e-12)
    return {"F1_change": f1, "IoU_change": iou, "precision_change": prec,
            "recall_change": rec, "kappa_change": kappa,
            "Omission": fn / max(tp + fn, 1), "Commission": fp / max(tp + fp, 1)}


def per_class(rec):
    return {k: {"F1": v["F1"], "IoU": v["IoU"], "PA": v["PA"], "UA": v["UA"],
                "support": v["support"]}
            for k, v in rec.get("per_class", {}).items()}


def dig(rec, path):
    """'independent_test.beilunhe' -> rec['independent_test']['beilunhe']"""
    cur = rec
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def summarize_flat(regions):
    """exp3b ML baselines: {region: metrics} (no within-region split)."""
    out = summarize({})
    for region, rec in regions.items():
        if not isinstance(rec, dict) or "OA" not in rec:
            continue
        label = "within" if region == "vietnam" else region
        out[label] = summarize({"test_within_region": rec})["within"]
    return out


def summarize(block):
    """block = {region: metrics dict}; returns {region: record}"""
    out = {}
    for label, path in EVAL_SETS:
        rec = dig(block, path) if isinstance(block, dict) else None
        if not rec:
            continue
        cm = rec.get("confusion_matrix")
        r = {"OA": rec["OA"], "kappa": rec["kappa"], "mIoU": rec["mIoU"],
             "F1_macro": rec["F1_macro"], "n_pixels": rec.get("n_pixels")}
        if cm:
            r.update(binary_change(cm))
            r["cm"] = cm
        r["per_class"] = per_class(rec)
        out[label] = r
    return out


def collect(include_pilot=False):
    """model -> seed -> {region: record}.

    With --include-pilot the archived seed-2026 pilot runs
    (metrics_<model>_pilot2026.json, 2026-09-20 host) are used for models/seeds
    that have no *-s<seed>.json yet, so draft tables/figures can be produced
    while the 3-seed sweep is still running.
    """
    runs = {}   # model -> seed -> {region: record}
    for f in sorted(RESULTS_ROOT.glob("metrics_*_s*.json")):
        m = re.match(r"metrics_(.+)_s(\d+)\.json$", f.name)
        if not m:
            continue
        model, seed = m.group(1), int(m.group(2))
        if model == "pcc" or model not in MODELS or seed not in CANONICAL_SEEDS:
            continue  # R2 variants (tagged) + binary-only PCC are separate
        rec = json.loads(f.read_text())
        runs.setdefault(model, {})[seed] = {
            **summarize(rec),
            "_params_M": rec.get("params_M"),
            "_epochs": (rec.get("hyperparams") or {}).get("epochs"),
        }
    if include_pilot:
        for f in sorted(RESULTS_ROOT.glob("metrics_*_pilot*.json")):
            m = re.match(r"metrics_(.+)_pilot(\d+)\.json$", f.name)
            if not m:
                continue
            model, seed = m.group(1), int(m.group(2))
            if seed in runs.get(model, {}):
                continue
            rec = json.loads(f.read_text())
            runs.setdefault(model, {})[seed] = {
                **summarize(rec),
                "_params_M": rec.get("params_M"),
                "_epochs": (rec.get("hyperparams") or {}).get("epochs"),
                "_pilot": True,
            }

    # pixel-level ML baselines: nested {model: {region: metrics}}
    ml = RESULTS_ROOT / "metrics_ml_baselines.json"
    if ml.exists():
        d = json.loads(ml.read_text())
        for model, regions in d.items():
            if not isinstance(regions, dict):
                continue
            runs.setdefault(model, {})[2026] = summarize_flat(regions)
    return runs


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


def fmt(mu, sd, nd=4):
    if mu is None:
        return "-"
    return f"{mu:.{nd}f}" if sd in (None, 0.0) else f"{mu:.{nd}f}±{sd:.{nd}f}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-pilot", action="store_true",
                    help="fill in missing (model, seed) entries from the archived "
                         "seed-2026 pilot runs (draft tables while the sweep runs)")
    args = ap.parse_args()
    runs = collect(args.include_pilot)
    (RESULTS_ROOT / "metrics_summary.json").write_text(json.dumps(runs, indent=2))

    metrics = ["OA", "kappa", "mIoU", "F1_macro", "F1_change", "IoU_change",
               "Omission", "Commission"]
    regions = ["within", "beilunhe", "zhanjiang"]
    order = MODELS + [m for m in ("random_forest", "svm") if m in runs]

    rows = []
    for model in order:
        if model not in runs:
            continue
        seeds = sorted(runs[model])
        for region in regions:
            row = {"model": MODEL_LABEL.get(model, model), "key": model,
                   "region": region, "n_seeds": len(seeds)}
            for mk in metrics:
                vals = [runs[model][s].get(region, {}).get(mk) for s in seeds]
                mu, sd = mean_std(vals)
                row[mk] = fmt(mu, sd)
            rows.append(row)

    with open(RESULTS_ROOT / "table_main.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "region", "n_seeds"] + metrics)
        w.writeheader()
        w.writerows({k: v for k, v in r.items() if k in ["model", "region", "n_seeds"] + metrics}
                    for r in rows)

    # markdown: one block per region
    lines = ["# Main results (mean ± std over seeds; pixel-level)",
             "",
             "Train region: Vietnam Red River Delta. `within` = buffer-isolated "
             "held-out zone of the training region; `beilunhe` / `zhanjiang` = "
             "spatially independent test regions.",
             ""]
    for region in regions:
        sub = [r for r in rows if r["region"] == region]
        if not sub:
            continue
        lines += [f"## {region}", "",
                  "| Model | OA | Kappa | mIoU | F1_macro | F1_change | IoU_change | "
                  "Omission | Commission |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in sub:
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["model"], r["OA"], r["kappa"], r["mIoU"], r["F1_macro"],
                r["F1_change"], r["IoU_change"], r["Omission"], r["Commission"]))
        lines.append("")
    (RESULTS_ROOT / "table_main.md").write_text("\n".join(lines))

    # per-class table (mean over seeds)
    pc_rows = []
    for model in order:
        if model not in runs:
            continue
        seeds = sorted(runs[model])
        for region in regions:
            for cls in CHANGE_CLASSES:
                f1 = [runs[model][s].get(region, {}).get("per_class", {}).get(cls, {}).get("F1")
                      for s in seeds]
                iou = [runs[model][s].get(region, {}).get("per_class", {}).get(cls, {}).get("IoU")
                       for s in seeds]
                mu1, sd1 = mean_std(f1)
                mu2, sd2 = mean_std(iou)
                if mu1 is None:
                    continue
                pc_rows.append({"model": MODEL_LABEL.get(model, model), "region": region,
                                "class": cls, "F1": fmt(mu1, sd1), "IoU": fmt(mu2, sd2)})
    with open(RESULTS_ROOT / "table_perclass.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "region", "class", "F1", "IoU"])
        w.writeheader()
        w.writerows(pc_rows)

    print(f"runs: { {m: sorted(s) for m, s in runs.items()} }")
    print("\n".join(lines))
    print("wrote metrics_summary.json, table_main.csv, table_main.md, table_perclass.csv")


if __name__ == "__main__":
    main()
