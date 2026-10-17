"""Label-resolution control experiment + edge-band
null baselines.

Part A (30 m re-aggregation). The GMW change labels are natively 30 m
(nearest reprojected to the aligned UTM 10 m grid, i.e. each 30 m cell covers
exactly one 3x3 block of 10 m pixels). We aggregate every model's 10 m
predictions to the native 30 m grid (3x3 majority) and re-evaluate there.
If the commission is a boundary-resolution artefact, commission must drop
substantially at 30 m while true-change recall stays comparable; if the
models invent change, both metrics degrade together.

Part B (edge-band null baselines). For every region we report the
distance-to-reference-fringe distribution of (a) ALL evaluated pixels (the
random-pixel null), (b) the TRUE change pixels of the reference, and (c) the
predicted change pixels of every model. Only if (c) exceeds (a) and (b) can
"most change lies near the fringe" be read as a boundary-placement effect
rather than an unavoidable prior.

Output: results/resolution_control.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.metrics import cd_metrics, compute_all_metrics
from mangrove.config import CHANGE_CLASSES

MODELS = ["unet", "unet_cbam", "unet_msa", "unet_cbam_msa", "deeplabv3plus",
          "temporal_unet", "temporal_unet_gn", "fc_siam"]
SEEDS = [2026, 2027, 2028]
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
PATCH = 512
CELLS = PATCH // 3  # 170 complete 30 m cells per side (drop <=2 px edges)


def test_idx(region):
    idx = json.loads((ROOT / "data" / "splits" / f"{region}_test.json").read_text())
    return np.asarray(idx, dtype=int)


def load_pred(model, seed, region):
    p = ROOT / "results" / "predictions" / f"{model}_s{seed}_{region}.npy"
    return np.load(p) if p.exists() else None


def load_labels(region):
    return np.load(ROOT / "data" / "processed" / region / "change_labels.npz")["change"]


def block_offset(region):
    meta = json.loads((ROOT / "data" / "processed" / region / "metadata.json").read_text())
    return meta["grid_origin"]  # (easting, northing), multiple of 10 m


def agg30(arr_patch):
    """512x512 -> 170x170 30 m cells (3x3 majority) on the global 30 m grid.

    Returns (agg, keep) where keep is the boolean 512x512 mask of the 10 m
    pixels belonging to complete 30 m cells (used to restrict the 10 m
    evaluation to the same spatial support).
    """
    r, c = int(arr_patch.shape[0]), int(arr_patch.shape[1])
    o_r, o_c = (-(r % 3)) % 3, (-(c % 3)) % 3
    # NOTE: r/c here are LOCAL patch offsets; the global row/col are
    # (grid_offset + local). The 30 m grid is aligned to the region origin,
    # so the local offset is o = (-global) mod 3 = (-(r0 + r)) mod 3.
    return None, None


def agg30_with_origin(arr_patch, glo_r, glo_c):
    o_r = (-glo_r) % 3
    o_c = (-glo_c) % 3
    n_r = (512 - o_r) // 3
    n_c = (512 - o_c) // 3
    block = arr_patch[o_r:o_r + 3 * n_r, o_c:o_c + 3 * n_c].reshape(n_r, 3, n_c, 3)
    keep = np.zeros((512, 512), bool)
    keep[o_r:o_r + 3 * n_r, o_c:o_c + 3 * n_c] = True
    # 4-class majority, ties -> smallest class id (applied identically to
    # predictions and labels, so it is symmetric)
    counts = np.zeros((n_r, n_c, 4), dtype=int)
    for k in range(4):
        counts[..., k] = (block == k).sum(axis=(1, 3))
    agg = counts.argmax(axis=2).astype(np.uint8)
    return agg, keep


def binary_change(arr):
    return np.isin(arr, [1, 2])


def change_metrics_from_cm(cm):
    """Binary change (loss+gain) metrics from a 4x4 confusion matrix."""
    cm = np.asarray(cm, dtype=float)
    tp = cm[1, 1] + cm[2, 1] + cm[1, 2] + cm[2, 2]
    fp = cm[0, 1] + cm[0, 2] + cm[3, 1] + cm[3, 2]
    fn = cm[1, 0] + cm[2, 0] + cm[1, 3] + cm[2, 3]
    tn = cm[0, 0] + cm[3, 3]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    # codebase convention: Commission = FP/(FP+TP) = 1 - precision,
    # Omission = FN/(FN+TP) = 1 - recall (matches metrics_*_s*.json)
    return {"F1_change": float(f1), "IoU_change": float(iou),
            "precision_change": float(prec), "recall_change": float(rec),
            "Omission": float(fn / (fn + tp) if fn + tp else 0.0),
            "Commission": float(fp / (fp + tp) if fp + tp else 0.0),
            "OA": float((tp + tn) / max(tp + fp + fn + tn, 1))}


def main():
    out = {"part_a_30m": {}, "part_b_edge_null": {}}
    for region in REGIONS:
        meta = json.loads((ROOT / "data" / "processed" / region /
                           "metadata.json").read_text())
        patches = np.asarray(meta["patches"], dtype=int)
        tidx = test_idx(region)
        labels = load_labels(region)

        # ---------------- Part A: 30 m re-aggregation
        for model in MODELS:
            rec = {}
            for seed in SEEDS:
                pred = load_pred(model, seed, region)
                if pred is None:
                    continue
                ys10, ps10, ys30, ps30 = [], [], [], []
                for k in range(pred.shape[0]):
                    i = int(tidx[k])
                    p10, l10 = pred[k], labels[i]
                    glo_r, glo_c = int(patches[i][0]), int(patches[i][1])
                    pa, ka = agg30_with_origin(p10.astype(np.int8), glo_r, glo_c)
                    la, _ = agg30_with_origin(l10, glo_r, glo_c)
                    ys10.append(l10[ka].ravel())
                    ps10.append(p10[ka].ravel())
                    ys30.append(la.ravel())
                    ps30.append(pa.ravel())
                ys10, ps10 = np.concatenate(ys10), np.concatenate(ps10)
                ys30, ps30 = np.concatenate(ys30), np.concatenate(ps30)
                from sklearn.metrics import confusion_matrix as _cm
                m10 = change_metrics_from_cm(_cm(ys10, ps10, labels=[0, 1, 2, 3]))
                m30 = change_metrics_from_cm(_cm(ys30, ps30, labels=[0, 1, 2, 3]))
                rec[str(seed)] = {
                    "F1ch_10m": m10["F1_change"], "comm_10m": m10["Commission"],
                    "omiss_10m": m10["Omission"], "IoU_10m": m10["IoU_change"],
                    "F1ch_30m": m30["F1_change"], "comm_30m": m30["Commission"],
                    "omiss_30m": m30["Omission"], "IoU_30m": m30["IoU_change"],
                }
            if rec:
                keys = list(rec[str(SEEDS[0])].keys())
                out["part_a_30m"].setdefault(model, {})[region] = {
                    k: {
                        "mean": float(np.mean([rec[s][k] for s in rec])),
                        "std": float(np.std([rec[s][k] for s in rec])),
                        "per_seed": [rec[s][k] for s in sorted(rec, key=int)],
                    } for k in keys
                }

        # ---------------- Part B: edge-band null baselines (reference only)
        fringe_all, rand_d, true_d = [], [], []
        for i in tidx:
            l = labels[i]
            mang = np.isin(l, [0, 1, 2])
            fr = mang & ~binary_erosion(mang, iterations=1)
            d = distance_transform_edt(~fr)
            rand_d.append(d.ravel())
            true_d.append(d[binary_change(l)].ravel())
        rand_d, true_d = np.concatenate(rand_d), np.concatenate(true_d)
        rec = {}
        for name, d in (("random_pixels", rand_d), ("true_change_pixels", true_d)):
            rec[name] = {
                "n": int(d.size),
                **{f"within_{b*10}m": float((d <= b).mean()) for b in (1, 2, 3)},
            }
        # predicted change per model (pooled seeds x patches)
        for model in MODELS:
            pc, fc = [], []
            for seed in SEEDS:
                pred = load_pred(model, seed, region)
                if pred is None:
                    continue
                for k in range(pred.shape[0]):
                    i = int(tidx[k])
                    p, l = pred[k], labels[i]
                    mang = np.isin(l, [0, 1, 2])
                    fr = mang & ~binary_erosion(mang, iterations=1)
                    d = distance_transform_edt(~fr)
                    pc.append(d[binary_change(p)].ravel())
                    fc.append(d[binary_change(p) & ~binary_change(l)].ravel())
            if pc:
                pc, fc = np.concatenate(pc), np.concatenate(fc)
                rec[f"pred_change_{model}"] = {
                    "n": int(pc.size),
                    **{f"within_{b*10}m": float((pc <= b).mean()) for b in (1, 2, 3)},
                }
                rec[f"false_change_{model}"] = {
                    "n": int(fc.size),
                    **{f"within_{b*10}m": float((fc <= b).mean()) for b in (1, 2, 3)},
                }
        out["part_b_edge_null"][region] = rec

    (ROOT / "results" / "resolution_control.json").write_text(json.dumps(out, indent=1))
    print("saved results/resolution_control.json")
    # compact summary
    for model in MODELS:
        for region in REGIONS:
            r = out["part_a_30m"].get(model, {}).get(region)
            if not r:
                continue
            f10 = r["F1ch_10m"]["mean"]; f30 = r["F1ch_30m"]["mean"]
            c10 = r["comm_10m"]["mean"]; c30 = r["comm_30m"]["mean"]
            o10 = r["omiss_10m"]["mean"]; o30 = r["omiss_30m"]["mean"]
            print(f"{model:18s} {region:10s} F1ch {f10:.3f}->{f30:.3f} "
                  f"comm {c10:.3f}->{c30:.3f} omit {o10:.3f}->{o30:.3f}")
    for region in REGIONS:
        b = out["part_b_edge_null"][region]
        print(region, {k: (round(v["within_20m"], 3) if isinstance(v, dict) else v)
                       for k, v in b.items() if k in ("random_pixels", "true_change_pixels")})


if __name__ == "__main__":
    main()
