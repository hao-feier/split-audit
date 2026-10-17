"""Class-weight scheme comparison (inv / sqrt / uniform).

Compares the three weighting schemes for the same U-Net under a recipe
matched to the main sweep (100 epochs, patience 15, batch 8, v2 split,
3 seeds):
  - inv:     results/probes_cw/inv/      (inverse frequency)
  - sqrt:    results/                    (main-sweep U-Net, the default)
  - uniform: results/probes_cw/none/     (no weighting)
and, for the full per-model picture of sqrt vs uniform, the 5-architecture
uniform re-run in results/uniform/).

Reports per model (where available) and eval set: OA / mIoU / F1_change /
omission / commission (binary change = {loss, gain}, classes 1 and 2) plus
the true vs predicted change fraction (the over/under-prediction behaviour).

Output: results/cw_schemes.json
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from aggregate_results import binary_change, CHANGED  # noqa: E402

UNET_MODES = {"inv": ROOT / "results" / "probes_cw" / "inv",
              "sqrt": ROOT / "results",
              "uniform": ROOT / "results" / "probes_cw" / "none"}
SWEEP_MODES = {"sqrt": ROOT / "results", "uniform": ROOT / "results" / "uniform"}
MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]
SEEDS = ["2026", "2027", "2028"]
REGIONS = ["within", "beilunhe", "zhanjiang"]
KEYS = ["OA", "mIoU", "F1ch", "omission", "commission", "true_change_pct", "pred_change_pct"]


def block(d, region):
    if region == "within":
        return d.get("test_within_region")
    return (d.get("independent_test") or {}).get(region)


def metrics_of(blk):
    cm = np.asarray(blk["confusion_matrix"], dtype=np.float64)
    bc = binary_change(blk["confusion_matrix"])
    ch = list(CHANGED)
    tot = cm.sum()
    true_ch = cm[np.ix_(ch, list(range(4)))].sum()
    pred_ch = cm[np.ix_(list(range(4)), ch)].sum()
    return {"OA": blk["OA"], "mIoU": blk["mIoU"], "F1ch": bc["F1_change"],
            "omission": bc["Omission"], "commission": bc["Commission"],
            "true_change_pct": 100.0 * true_ch / tot,
            "pred_change_pct": 100.0 * pred_ch / tot}


def aggregate(root, model, region):
    rows = []
    for s in SEEDS:
        f = root / f"metrics_{model}_s{s}.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text())
        blk = block(d, region)
        if blk is None:
            return None
        rows.append(metrics_of(blk))
    return {k: {"mean": st.mean([r[k] for r in rows]),
                "std": st.pstdev([r[k] for r in rows])} for k in KEYS}


def main():
    out = {"unet_schemes": {}, "sweep_sqrt_vs_uniform": {}}
    for region in REGIONS:
        out["unet_schemes"][region] = {}
        for mode, root in UNET_MODES.items():
            a = aggregate(root, "unet", region)
            if a:
                out["unet_schemes"][region][mode] = a
    for m in MODELS:
        out["sweep_sqrt_vs_uniform"][m] = {}
        for region in REGIONS:
            for mode, root in SWEEP_MODES.items():
                a = aggregate(root, m, region)
                if a:
                    out["sweep_sqrt_vs_uniform"][m].setdefault(region, {})[mode] = a
    (ROOT / "results" / "cw_schemes.json").write_text(json.dumps(out, indent=2))
    f = lambda a, k: f"{a[k]['mean']:.3f}" if a else "-"
    lines = ["| model | region | scheme | mIoU | F1ch | true% | pred% | commission |",
             "|---|---|---|---|---|---|---|---|"]
    for region in REGIONS:
        for mode in UNET_MODES:
            a = out["unet_schemes"][region].get(mode)
            lines.append(f"| unet | {region} | {mode} | {f(a,'mIoU')} | {f(a,'F1ch')} | "
                         f"{f(a,'true_change_pct')} | {f(a,'pred_change_pct')} | {f(a,'commission')} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
