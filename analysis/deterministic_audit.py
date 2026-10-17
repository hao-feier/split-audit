"""Deterministic re-run -> noise floor of the training pipeline.

Runs compared (all unet_cbam_msa, same recipe bs=8 / 100ep / pat15 / sqrt):
  det   : deterministic seeds 2026/2027/2028   (tag "det")
  detc  : deterministic seed 2026 repeat       (tag "detc")
  nd/ndc: non-deterministic seed 2026, x2      (tags "nd" / "ndc")
  main  : the original 3-seed sweep (non-deterministic, no tag)

Quantities:
  bit-rep: max |metric diff| between two runs of the SAME config
      detc vs det  -> should be exactly 0 (pipeline determinism)
      ndc  vs nd   -> the raw CUDA/threading non-determinism of a rerun
  floor  : std across the 3 deterministic seeds = the irreducible
      seed-initialisation noise floor (everything else held constant)
  main   : std across the original 3 non-deterministic seeds (the number
      the paper currently reports as +/-)

Output: results/deterministic_audit.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# The audit now covers all eight main-sweep architectures
# (Stage 4 of experiments/run_arch_extension_gpu.sh produces det / detc / nd runs of each,
# seed 2026). The legacy single-model mode (unet_cbam_msa, 3-seed det runs)
# is kept for the archived R2 results.
MODELS_AUDIT = ["unet", "unet_cbam", "unet_msa", "unet_cbam_msa",
                "deeplabv3plus", "temporal_unet", "temporal_unet_gn", "fc_siam"]
MODEL = "unet_cbam_msa"
BLOCKS = [("val", "val"), ("within", "test_within_region"),
          ("beilunhe", "independent_test.beilunhe"),
          ("zhanjiang", "independent_test.zhanjiang")]


def load(model, tag, seed):
    # artifact naming: metrics_{model}_{tag}_s{seed}.json (tag before seed)
    mid = f"_{tag}" if tag else ""
    f = ROOT / "results" / f"metrics_{model}{mid}_s{seed}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def scalars(d):
    """{block: {metric: value}} incl. binary-change F1 from the CM."""
    out = {}
    for name, path in BLOCKS:
        cur = d
        for p in path.split("."):
            cur = cur.get(p) if isinstance(cur, dict) else None
            if cur is None:
                break
        if not isinstance(cur, dict) or "confusion_matrix" not in cur:
            continue
        cm = np.asarray(cur["confusion_matrix"], dtype=float)
        chg = [1, 2]
        tp = cm[np.ix_(chg, chg)].sum(); fn = cm[np.ix_(chg, [0, 3])].sum()
        fp = cm[np.ix_([0, 3], chg)].sum()
        out[name] = {"OA": cur["OA"], "mIoU": cur["mIoU"], "F1_macro": cur["F1_macro"],
                     "CD_Acc": cur.get("CD_Accuracy_changed"),
                     "F1ch": 2 * tp / max(2 * tp + fn + fp, 1)}
    return out


def max_diff(a, b):
    vals = {}
    for blk in set(a) & set(b):
        for m in set(a[blk]) & set(b[blk]):
            if a[blk][m] is None or b[blk][m] is None:
                continue
            vals[f"{blk}/{m}"] = abs(a[blk][m] - b[blk][m])
    return float(max(vals.values())) if vals else None, vals


def seed_std(runs):
    """runs: {seed: scalars} -> per block/metric std across seeds."""
    out = {}
    blocks = set.intersection(*[set(r) for r in runs.values()]) if runs else set()
    for blk in blocks:
        ms = set.intersection(*[set(r[blk]) for r in runs.values()])
        for m in ms:
            arr = [runs[s][blk][m] for s in sorted(runs)]
            out[f"{blk}/{m}"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return out


def audit_one(model, main_seeds=(2026, 2027, 2028, 2029, 2030),
              det_seeds=(2026, 2027, 2028)):
    det = {st: scalars(load(model, "det", st)) for st in det_seeds
           if load(model, "det", st)}
    det_c = scalars(load(model, "detc", 2026)) if load(model, "detc", 2026) else None
    nd = scalars(load(model, "nd", 2026)) if load(model, "nd", 2026) else None
    nd_c = scalars(load(model, "ndc", 2026)) if load(model, "ndc", 2026) else None
    main = {st: scalars(load(model, None, st)) for st in main_seeds
            if load(model, None, st)}

    out = {"model": model, "n_main_seeds": len(main)}
    if det_c is not None and det.get(2026):
        out["det_repeat_max_abs_diff"] = max_diff(det_c, det[2026])
    if nd_c is not None and nd is not None:
        out["nd_repeat_max_abs_diff"] = max_diff(nd_c, nd)
    elif nd is not None and det.get(2026):
        out["nd_vs_det_max_abs_diff"] = max_diff(nd, det[2026])
    out["noise_floor_det_seeds"] = seed_std(det)
    out["main_sweep_seed_std"] = seed_std(main)
    if det and main:
        out["floor_over_main"] = {}
        for k, v in out["noise_floor_det_seeds"].items():
            m = out.get("main_sweep_seed_std", {}).get(k)
            out["floor_over_main"][k] = float(v / m) if m else None
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS_AUDIT)
    ap.add_argument("--single", action="store_true",
                    help="legacy single-model mode (unet_cbam_msa -> "
                         "results/deterministic_audit.json)")
    args = ap.parse_args()

    if args.single:
        out = audit_one(MODEL)
        (ROOT / "results" / "deterministic_audit.json").write_text(
            json.dumps(out, indent=2))
        print("wrote results/deterministic_audit.json")
        return

    all_out = {}
    for m in args.models:
        o = audit_one(m)
        if o is None:
            continue
        all_out[m] = o
        bits = o.get("det_repeat_max_abs_diff")
        ndd = o.get("nd_vs_det_max_abs_diff") or o.get("nd_repeat_max_abs_diff")
        print(f"{m:18s} det-bit-identical="
              f"{'yes' if bits and bits[0] == 0.0 else ('no' if bits else 'n/a')} "
              f"nd-vs-det max|diff|="
              f"{f'{ndd[0]:.3e}' if ndd else 'n/a'} "
              f"main-seeds={o['n_main_seeds']}")
    (ROOT / "results" / "deterministic_audit_all.json").write_text(
        json.dumps(all_out, indent=2))
    print("wrote results/deterministic_audit_all.json")


if __name__ == "__main__":
    main()
