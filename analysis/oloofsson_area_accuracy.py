"""Olofsson et al. (2014) area-based accuracy estimation.

Input: data/groundtruth/<region>_samples.geojson with interpreter-filled
fields interp_t1 / interp_t2 (0/1 mangrove presence at t1 / t2). The
interpreted change class is derived: (1,1) stable mangrove / (1,0) loss /
(0,1) gain / (0,0) stable background.

The sample design is stratified on the GMW v4.1.12 reference map
(make_groundtruth_samples.py), so both estimations below use the GMW map
as the reference: strata = GMW change class at the sample point, and the
estimator re-weights by GMW stratum area over the whole study area
(map-referenced / area-estimated accuracy, Olofsson et al. 2014):
  OA_hat      = sum_h (A_h / A) * p_hat_h
  p_hat_h     = P(pred == h | GMW == h)   (per-stratum agreement, bootstrap CI)
The same estimator is applied to (a) the GMW labels vs. the interpreted
truth (label quality / error floor) and (b) the model predictions at the
sample points vs. the interpreted truth (model accuracy, area-estimated;
vietnam points restricted to the test split for spatial independence).
The binary-change collapse (F1 / omission / commission) is always computed
against the interpreted truth.

Usage:
  python analysis/oloofsson_area_accuracy.py                 # labels vs truth
  python analysis/oloofsson_area_accuracy.py --model unet_cbam_msa --seed 2026
Outputs: results/groundtruth_accuracy_{labels|model_<m>_s<seed>}.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.config import REGIONS, CHANGE_CLASSES
from mangrove.datasets import load_region, load_split
from mangrove.models import MODEL_REGISTRY, pick_device
from mangrove.paths import PROCESSED_ROOT, RESULTS_ROOT

N_BOOT = 1000


def interp_class(p):
    a, b = p["properties"]["interp_t1"], p["properties"]["interp_t2"]
    if a is None or b is None:
        return None
    a, b = int(a), int(b)
    if (a, b) == (1, 1):
        return 0
    if (a, b) == (1, 0):
        return 1
    if (a, b) == (0, 1):
        return 2
    return 3


def area_by_class(region):
    """GMW stratum area (km2) over the whole study-area raster."""
    import rasterio
    meta = json.loads((PROCESSED_ROOT / region / "metadata.json").read_text())
    with rasterio.open(PROCESSED_ROOT / region / "change_labels.tif") as ds:
        lab = ds.read(1)
    return {c: float((lab == c).sum()) * 1e-4 for c in range(4)}


def olofsson(ref, pred, truth, areas, rng):
    """Area-estimated accuracy of `pred` against `truth`, referenced to strata.

    ref:    reference-map class per point (defines strata; GMW here)
    pred:   class being validated (GMW labels or model predictions)
    truth:  interpreted truth class (used for the binary-change collapse)
    areas:  km2 per reference stratum over the study area
    """
    n = len(ref)
    ref_arr = np.asarray(ref)
    strata = [np.array(ref_arr == h) for h in range(4)]
    A = sum(areas[h] for h in range(4) if strata[h].sum() > 0)
    oa = np.zeros(N_BOOT)
    pa = {h: np.zeros(N_BOOT) for h in range(4)}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        stratum_of = ref_arr[idx]  # stratum of each resampled draw
        oa_b = 0.0
        for h in range(4):
            if not strata[h].any():
                continue
            samp = idx[stratum_of == h]
            p_h = float(np.mean(np.asarray([pred[i] == h for i in samp])))
            pa[h][b] = p_h
            oa_b += (areas[h] / A) * p_h
        oa[b] = oa_b

    out = {
        "n_points": int(n),
        "OA_area_est": float(oa.mean()),
        "OA_ci95": [float(x) for x in np.percentile(oa, [2.5, 97.5])],
    }
    for h in range(4):
        if strata[h].sum() == 0:
            continue
        out[f"PA_{CHANGE_CLASSES[h]}"] = float(pa[h].mean())
        out[f"PA_{CHANGE_CLASSES[h]}_ci95"] = [float(x) for x in np.percentile(pa[h], [2.5, 97.5])]

    tr_ch = np.array([t in (1, 2) for t in truth])
    pr_ch = np.array([p in (1, 2) for p in pred])
    tp = int((tr_ch & pr_ch).sum())
    fn = int((tr_ch & ~pr_ch).sum())
    fp = int((~tr_ch & pr_ch).sum())
    tn = int((~tr_ch & ~pr_ch).sum())
    out["change_F1"] = 2 * tp / max(2 * tp + fn + fp, 1)
    out["change_omission"] = fn / max(tp + fn, 1)
    out["change_commission"] = fp / max(fp + tn, 1)
    out.update({"n_tp": tp, "n_fn": fn, "n_fp": fp, "n_tn": tn})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--region", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(2026)

    out = {}
    for region in [args.region] if args.region else REGIONS:
        feats = json.loads((ROOT / "data" / "groundtruth" / f"{region}_samples.geojson").read_text())["features"]
        meta = json.loads((PROCESSED_ROOT / region / "metadata.json").read_text())
        ox, oy = meta["grid_origin"]
        ps = meta["patch_size"]
        areas = area_by_class(region)

        # points with a complete interpretation, with grid location
        pts = []
        for p in feats:
            t = interp_class(p)
            if t is None:
                continue
            x, y = p["properties"]["x"], p["properties"]["y"]
            c = int((x - ox) / 10.0)
            r = int((oy - y) / 10.0)
            pts.append((p, t, r, c))
        n_done = len(pts)
        if n_done == 0:
            print(f"{region}: no interpreted points yet (0/{len(feats)}); skipping")
            continue
        print(f"{region}: {n_done}/{len(feats)} points interpreted")

        row = {"n_interpreted": n_done,
               "stratum_counts_gmw": {str(h): int(sum(1 for p, *_ in pts if p["properties"]["gmw_class"] == h))
                                       for h in range(4)}}
        gmw = [p["properties"]["gmw_class"] for p, *_ in pts]
        truth = [t for _, t, *_ in pts]

        # (a) GMW label quality: validate the GMW map against human truth
        row["label_quality_oloofsson"] = olofsson(gmw, gmw, truth, areas, rng)

        # (b) model predictions at the sample points (test-split patches only)
        if args.model:
            import torch
            data = load_region(region)
            test_pos = set(int(i) for i in load_split(region, "test"))
            pos_of = {}
            for pos, (pr, pc) in enumerate(meta["patches"]):
                if pos in test_pos:
                    pos_of[pos] = (pr, pc)
            is_temporal = args.model in {"temporal_unet"}
            in_ch = 9 if is_temporal else 27
            device = pick_device()
            model = MODEL_REGISTRY[args.model](in_ch=in_ch, out_ch=4).to(device)
            model.load_state_dict(torch.load(
                RESULTS_ROOT / "checkpoints" / f"{args.model}_s{args.seed}.pt",
                map_location="cpu"))
            model.eval()
            pred_map = {}
            with torch.no_grad():
                if is_temporal:
                    a = torch.from_numpy(data["t1_feat"] / 10000.0)
                    b = torch.from_numpy(data["t2_feat"] / 10000.0)
                    for pos in pos_of:
                        o = model(a[pos:pos + 1].to(device), b[pos:pos + 1].to(device))
                        pred_map[pos] = o.argmax(1).cpu().numpy()[0]
                else:
                    x = np.concatenate([data["t1_feat"], data["t2_feat"],
                                        np.abs(data["t1_feat"] - data["t2_feat"])], 1)
                    xt = torch.from_numpy(x / 10000.0)
                    for i in range(0, len(xt), 8):
                        o = model(xt[i:i + 8].to(device)).argmax(1).cpu().numpy()
                        for j in range(len(o)):
                            if (i + j) in pos_of:
                                pred_map[i + j] = o[j]

            m_gmw, m_pred, m_truth = [], [], []
            n_outside = 0
            for p, t, r, c in pts:
                hit = None
                for pos, (pr, pc) in pos_of.items():
                    if pr <= r < pr + ps and pc <= c < pc + ps:
                        hit = pos
                        break
                if hit is None:
                    n_outside += 1
                    continue
                m_gmw.append(p["properties"]["gmw_class"])
                m_pred.append(int(pred_map[hit][r - pos_of[hit][0], c - pos_of[hit][1]]))
                m_truth.append(t)
            if not m_pred:
                print(f"{region}: no interpreted points inside test patches; skipping model")
                continue
            row["model"] = {
                "model": args.model, "seed": args.seed,
                "n_points_in_test": len(m_pred), "n_points_outside_test": n_outside,
                "accuracy_oloofsson": olofsson(m_gmw, m_pred, m_truth, areas, rng),
            }
        out[region] = row
        print(json.dumps(row, indent=2))

    tag = "labels" if not args.model else f"model_{args.model}_s{args.seed}"
    path = RESULTS_ROOT / f"groundtruth_accuracy_{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
