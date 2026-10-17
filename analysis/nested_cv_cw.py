"""Nested-CV class-weight scheme selection.

M6 picked the class-weight scheme (inv / sqrt / uniform) by looking at the
independent test regions -> selection on the test set. Here the selection
is nested: the scheme is chosen from a spatial 3-fold CV on the vietnam
TRAIN zone only (contiguous column bands, so folds are spatially separated
and the fold metric is an honest generalisation estimate). The selected
scheme is then applied to the full train zone and the 5 models are
evaluated on val + the independent regions.

Phases:
  --build-folds  write results/nested_cv_folds.json (3 column bands)
  --select       read the 9 inner runs (unet_cbam_msa, tag nested_f<k>_<scheme>),
                 pick the scheme with the best mean inner-fold val F1ch
  --final        aggregate the final (tag "nested") 5-model x 3-seed runs,
                 compare with the sqrt main sweep and with the "oracle"
                 scheme (best per model on the TEST data, from M6)

Output: results/nested_cv_cw.json
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from aggregate_results import binary_change  # noqa: E402

SCHEMES = ["inv", "sqrt", "none"]
MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]
SEEDS = [2026, 2027, 2028]
REGIONS = ["within", "beilunhe", "zhanjiang"]
MODEL_INNER = "unet_cbam_msa"


def block_of(d, region):
    if region == "within":
        return d.get("test_within_region")
    return (d.get("independent_test") or {}).get(region)


def f1ch(blk):
    cm = blk["confusion_matrix"]
    bc = binary_change(cm)
    return bc["F1_change"]


def build_folds():
    """Spatial 3-fold CV on the vietnam TRAIN zone.

    Folds are ROW bands with a >= 512 px (5.12 km) gap between the row starts
    of adjacent bands, so no patch in one fold overlaps a patch in another
    (overlap requires |row_start_a - row_start_b| < 512 px AND
    |col_start_a - col_start_b| < 512 px; the row gap alone breaks it).
    Patches whose row start sits in a band gap (rows 3072, 4608) overlap
    both neighbours and are excluded from every fold (3 patches, documented).
    """
    meta = json.loads((ROOT / "data" / "processed" / "vietnam" / "metadata.json").read_text())
    train = json.loads((ROOT / "data" / "splits" / "vietnam_train.json").read_text())
    row_of = {i: meta["patches"][i][0] for i in train}
    bands = [lambda r: r <= 2816, lambda r: 3328 <= r <= 4352, lambda r: r >= 4864]
    folds = []
    all_in = []
    for k, test in enumerate(bands):
        val_idx = sorted([i for i in train if test(row_of[i])])
        all_in += val_idx
        folds.append({"fold": k + 1, "val_idx": val_idx})
    excluded = [i for i in train if i not in all_in]
    for f in folds:
        f["train_idx"] = [i for i in all_in if i not in f["val_idx"]]
        f["val_row_range"] = [min(row_of[i] for i in f["val_idx"]),
                              max(row_of[i] for i in f["val_idx"])]
        f["n_val"] = len(f["val_idx"])
        f["n_train"] = len(f["train_idx"])
    out = ROOT / "results" / "nested_cv_folds.json"
    out.write_text(json.dumps({"folds": folds, "excluded_idx": excluded,
                               "excluded_rows": [row_of[i] for i in excluded]}, indent=2))
    for f in folds:
        (ROOT / "results" / f"nested_cv_folds_f{f['fold']}_train.json").write_text(
            json.dumps(f["train_idx"]))
        (ROOT / "results" / f"nested_cv_folds_f{f['fold']}_val.json").write_text(
            json.dumps(f["val_idx"]))
    for f in folds:
        print(f"fold {f['fold']}: val={f['n_val']} train={f['n_train']} "
              f"val_row_range(px)={f['val_row_range']}")
    print(f"excluded (band-gap) patches: {excluded} rows {[row_of[i] for i in excluded]}")
    print(f"wrote {out}")
    return folds


def select():
    scores = {s: {} for s in SCHEMES}
    for k in (1, 2, 3):
        for scheme in SCHEMES:
            f = ROOT / "results" / f"metrics_{MODEL_INNER}_nested_f{k}_{scheme}_s2026.json"
            if not f.exists():
                print(f"MISSING inner run: {f.name}")
                continue
            d = json.loads(f.read_text())
            blk = d.get("val")
            if blk and "confusion_matrix" in blk:
                scores[scheme][k] = f1ch(blk)
    mean = {s: (st.mean(v.values()) if v else None, v) for s, v in scores.items()}
    valid = {s: v for s, (v, _) in mean.items() if v is not None}
    chosen = max(valid, key=lambda s: valid[s]) if valid else None
    out = {
        "inner_model": MODEL_INNER,
        "fold_scheme_scores_F1ch": {s: v for s, (_, v) in mean.items()},
        "fold_scheme_mean_F1ch": {s: v for s, (v, _) in mean.items()},
        "selected_scheme": chosen,
        "note": "scheme selected on train-zone inner CV only; no val/test data touched",
    }
    p = ROOT / "results" / "nested_cv_cw.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(out)
    p.write_text(json.dumps(old, indent=2))
    for s in SCHEMES:
        v = mean[s]
        print(f"  {s}: mean inner F1ch = {v[0] if v[0] is not None else 'n/a'} {v[1]}")
    print(f"selected scheme: {chosen}")


def aggregate_region(d, region):
    cm = block_of(d, region)
    if not cm or "confusion_matrix" not in cm:
        return None
    bc = binary_change(cm["confusion_matrix"])
    return {"OA": cm["OA"], "mIoU": cm["mIoU"], "F1ch": bc["F1_change"],
            "omission": bc["Omission"], "commission": bc["Commission"]}


def final():
    out = {}
    # final nested runs (selected scheme applied to all 5 models)
    nested = {}
    have = False
    for m in MODELS:
        rows = {}
        for s in SEEDS:
            f = ROOT / "results" / f"metrics_{m}_nested_s{s}.json"
            if f.exists():
                rows[s] = json.loads(f.read_text())
        if len(rows) == 3:
            have = True
            nested[m] = {r: [aggregate_region(rows[s], r) for s in SEEDS]
                         for r in REGIONS}
    if have:
        out["final_nested_by_scheme"] = {
            m: {r: {k: st.mean([nested[m][r][s][k] for s in range(3)])
                     for k in ("OA", "mIoU", "F1ch")} for r in REGIONS}
            for m in nested}
    # oracle scheme selection (M6 data): best of sqrt vs uniform per model,
    # chosen ON THE TEST DATA (the biased protocol, shown for contrast)
    oracle = {}
    for m in MODELS:
        per = {}
        for r in ["beilunhe", "zhanjiang"]:
            vals = {}
            for scheme, prefix in (("sqrt", ""), ("uniform", "uniform")):
                root = ROOT / "results" / ("uniform" if prefix else "")
                rows = []
                for s in SEEDS:
                    f = (ROOT / "results" / ("uniform" if prefix else "")
                         / f"metrics_{m}_s{s}.json")
                    if f.exists():
                        rows.append(aggregate_region(json.loads(f.read_text()), r))
                if rows:
                    vals[scheme] = st.mean([x["F1ch"] for x in rows if x])
            if vals:
                per[r] = vals
        if per:
            oracle[m] = per
    out["oracle_test_selection_F1ch"] = oracle
    # comparison vs the sqrt main sweep
    main = {}
    for m in MODELS:
        rows = {}
        for s in SEEDS:
            f = ROOT / "results" / f"metrics_{m}_s{s}.json"
            if f.exists():
                rows[s] = json.loads(f.read_text())
        if rows:
            main[m] = {r: [aggregate_region(rows[s], r) for s in rows] for r in REGIONS}
    out["main_sweep_sqrt_F1ch"] = {
        m: {r: st.mean([x["F1ch"] for x in v if x]) for r, v in regs.items()}
        for m, regs in main.items()}
    p = ROOT / "results" / "nested_cv_cw.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(out)
    p.write_text(json.dumps(old, indent=2))
    print(json.dumps({k: old[k] for k in ("selected_scheme", "fold_scheme_mean_F1ch",
                                          "final_nested_by_scheme") if k in old}, indent=2))
    print("wrote results/nested_cv_cw.json")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-folds", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    if args.build_folds:
        build_folds()
    if args.select:
        select()
    if args.final:
        final()


if __name__ == "__main__":
    main()
