"""E7: boundary-tolerance sweep -- the direct test of the label-resolution tax.

The reference change mask is morphed by k pixels (k*10 m) in four ways and the
change F1 is recomputed against the UNMODIFIED prediction:

  dilate_k   reference change GROWS by k px  -> rewards models that predict
             change just outside the reference boundary (optimistic)
  erode_k    reference change SHRINKS by k px -> punishes exactly the same
             models (pessimistic)
  open_k     remove change pixels within k px of a non-change pixel
  close_k    add change pixels within k px of a change pixel

k in {0,1,2,3,4} px = {0,10,20,30,40} m. The 10 m/30 m mismatch hypothesis
predicts a strong k-dependence. A flat curve falsifies it.

Output: results/boundary_tolerance.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, \
    binary_closing, generate_binary_structure

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus",
          "temporal_unet", "fc_siam"]
SEEDS = [2026, 2027, 2028]
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
KS = [0, 1, 2, 3, 4]           # x10 m
ST = generate_binary_structure(2, 1)   # 4-connectivity, 1 px


def test_idx(region):
    return np.asarray(json.loads(
        (ROOT / "data" / "splits" / f"{region}_test.json").read_text()), int)


def load_pred(model, seed, region):
    p = ROOT / "results" / "predictions" / f"{model}_s{seed}_{region}.npy"
    return np.load(p) if p.exists() else None


def binary_change(a):
    return np.isin(a, [1, 2])


def morph(mask, op, k):
    if k == 0:
        return mask
    it = ST.copy()
    for _ in range(k - 1):
        it = binary_dilation(it, ST)
    if op == "dilate":
        return binary_dilation(mask, it)
    if op == "erode":
        return binary_erosion(mask, it)
    if op == "open":
        return binary_opening(mask, it)
    return binary_closing(mask, it)


def prf(yt, yp):
    tp = float((yt & yp).sum()); fp = float((~yt & yp).sum())
    fn = float((yt & ~yp).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"F1": f1, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "omission": fn / (tp + fn) if tp + fn else 0.0,
            "commission": fp / (tp + fp) if tp + fp else 0.0}


def main():
    global CHANGE_CLASSES
    from mangrove.config import CHANGE_CLASSES
    out = {"config": {"models": MODELS, "seeds": SEEDS, "regions": REGIONS,
                      "k_px": KS, "k_m": [k * 10 for k in KS],
                      "ops": ["dilate", "erode", "open", "close"],
                      "note": "prediction unchanged; only the reference "
                              "change mask is morphed"}}
    for region in REGIONS:
        tidx = test_idx(region)
        labels = np.load(ROOT / "data" / "processed" / region /
                         "change_labels.npz")["change"]
        per_model = {}
        for model in MODELS:
            acc = {}
            for op in out["config"]["ops"]:
                for k in KS:
                    acc[f"{op}_{k}"] = []
            for seed in SEEDS:
                pred = load_pred(model, seed, region)
                if pred is None:
                    continue
                yts, yps = [], []
                for j in range(pred.shape[0]):
                    i = int(tidx[j])
                    yps.append(binary_change(pred[j]))
                    yts.append(binary_change(labels[i]))
                yp = np.concatenate([a.ravel() for a in yps])
                for op in out["config"]["ops"]:
                    for k in KS:
                        yt = np.concatenate(
                            [morph(binary_change(labels[int(tidx[j])]), op, k)
                             .ravel() for j in range(pred.shape[0])])
                        acc[f"{op}_{k}"].append(prf(yt, yp))
            if not acc.get("dilate_0"):
                continue
            per_model[model] = {
                m: {"mean": float(np.mean([r["F1"] for r in v])),
                    "std": float(np.std([r["F1"] for r in v], ddof=1))
                    if len(v) > 1 else 0.0,
                    "precision": float(np.mean([r["precision"] for r in v])),
                    "recall": float(np.mean([r["recall"] for r in v])),
                    "omission": float(np.mean([r["omission"] for r in v])),
                    "commission": float(np.mean([r["commission"] for r in v]))}
                for m, v in acc.items() if v}
        out.setdefault("per_region", {})[region] = per_model

    # pooled summary: mean over models of the F1 shift relative to k=0
    summ = {}
    for region, per_model in out.get("per_region", {}).items():
        rows = {}
        for op in out["config"]["ops"]:
            for k in KS:
                f1s = [v[f"{op}_{k}"]["mean"] for v in per_model.values()
                       if f"{op}_{k}" in v]
                if f1s:
                    rows[f"{op}_{k}"] = {"mean_F1": float(np.mean(f1s)),
                                         "n_models": len(f1s)}
        base = np.mean([v["dilate_0"]["mean"] for v in per_model.values()
                        if "dilate_0" in v])
        summ[region] = {"baseline_F1": base, "per_op_k": rows,
                        "max_abs_shift": max(
                            abs(r["mean_F1"] - base) for r in rows.values()),
                        "range_over_all_ops": [
                            min(r["mean_F1"] for r in rows.values()),
                            max(r["mean_F1"] for r in rows.values())]}
    out["summary"] = summ
    (ROOT / "results" / "boundary_tolerance.json").write_text(
        json.dumps(out, indent=1))

    print("== E7 boundary-tolerance sweep (F1ch vs reference morphology) ==")
    for region, s in summ.items():
        print("\n %-11s baseline F1ch = %.3f  | max |shift| = %.3f  "
              "| full range %.3f-%.3f"
              % (region, s["baseline_F1"], s["max_abs_shift"],
                 *s["range_over_all_ops"]))
        hdr = "  %-9s" % "op" + "".join("%9s" % ("%d m" % (k * 10))
                                        for k in KS)
        print(hdr)
        for op in out["config"]["ops"]:
            print("  %-9s" % op + "".join(
                "%9.3f" % s["per_op_k"].get(f"{op}_{k}", {}).get("mean_F1",
                                                                float("nan"))
                for k in KS))


if __name__ == "__main__":
    main()
