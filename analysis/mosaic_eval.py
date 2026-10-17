"""Evaluate on a de-duplicated mosaic, not on the
patch union with 50% overlap.

The published regional metrics pool the per-patch confusion matrices of the
test patches (experiments/exp3_train.py concatenates on the patch axis, then one
CM is formed). Test patches are 512-px windows on a 256-px stride, so a pixel
inside the region is counted once per covering patch: 1.85x-2.50x duplication
(up to 4x at patch interiors). A pixel that is systematically harder or easier
than its neighbours is therefore over-weighted in proportion to how many
patches cover it, which is not a uniform reweighting and can move F1.

analysis/nonoverlap_eval.py bounds the effect by *discarding* patches (a maximum
independent set: 4 of 10 Vietnam, 7 of 18 Beilunhe), which throws away most of
the evidence. This script instead partitions the region: every pixel in the
union of the test footprints is assigned to exactly one test patch (nearest
patch centre, a Voronoi partition), and the CM is formed on that mosaic. No
pixel is dropped and none is counted twice.

Two assignments are reported, because the choice is a judgement call and the
paper should not depend on it:
  * `nearest_centre` - Voronoi by distance to patch centre (primary)
  * `majority`      - majority class among the patches covering each pixel
                       (only defined where coverage >= 2; falls back to
                       nearest_centre where a pixel is covered once)

and, for reference, `nearest_origin`.

The `pooled` column is what the paper currently reports (patch-union sum).

Outputs
    results/mosaic_eval.json          per model x seed x region x assignment
    results/mosaic_eval_summary.json  5-seed means, pooled vs mosaic deltas,
                                      per-region leader re-ranking
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.config import CHANGE_CLASSES  # noqa: E402
from mangrove.paths import RESULTS_ROOT  # noqa: E402
from mangrove.metrics import compute_all_metrics, cd_metrics  # noqa: E402

PATCH = 512
STEP = 256

MODELS = ["unet", "unet_cbam", "unet_msa", "unet_cbam_msa", "deeplabv3plus",
          "deeplabv3plus_pre", "temporal_unet", "temporal_unet_gn", "fc_siam",
          "changeformer"]
SEEDS = [2026, 2027, 2028, 2029, 2030]
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]

# eval split name per region (mirrors analysis/dump_predictions.py REGIONS_EVAL:
# all three regions use the "<region>_test.json" file)
SPLIT = {"vietnam": "test", "beilunhe": "test", "zhanjiang": "test"}


def load_split(region, name):
    return json.loads(
        (ROOT / "data" / "splits" / f"{region}_{name}.json").read_text())


def origins(region, split_name):
    """(row, col) origin of each eval patch, in the order the dumps use."""
    meta = json.loads((ROOT / "data" / "processed" / region
                       / "metadata.json").read_text())
    patches = list(meta["patches"])
    idx = load_split(region, split_name)
    return [tuple(patches[i]) for i in idx]


def cm4(y, p):
    comb = np.asarray(y).ravel().astype(np.int64) * 4 + \
        np.asarray(p).ravel().astype(np.int64)
    return np.bincount(comb, minlength=16).reshape(4, 4).astype(np.int64)


def metrics_from_cm(cm):
    """Derive every reported metric from a 4x4 CM using the project's own code."""
    total = int(cm.sum())
    if total == 0:
        return {}
    m = compute_all_metrics(
        np.repeat(np.arange(4), cm.sum(axis=1)),
        np.repeat(np.arange(4), cm.sum(axis=0)),
        n_classes=4, class_names=list(CHANGE_CLASSES))
    # compute_all_metrics above cannot recover the joint structure from margins,
    # so recompute the binary change metrics directly from the CM.
    loss, gain = 1, 2
    tp = int(cm[loss, loss] + cm[gain, gain])
    fn = int(cm[loss].sum() - cm[loss, loss] + cm[gain].sum() - cm[gain, gain])
    fp = int(cm[:, loss].sum() - cm[loss, loss] + cm[:, gain].sum() - cm[gain, gain])
    f1 = 2 * tp / (2 * tp + fn + fp) if (2 * tp + fn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    iou = tp / (tp + fn + fp) if (tp + fn + fp) else 0.0
    # per-class F1
    per = {}
    for c in range(4):
        t = int(cm[c, c])
        f_ = int(cm[c].sum() - t) + int(cm[:, c].sum() - t)
        per[CHANGE_CLASSES[c]] = 2 * t / (2 * t + f_) if (2 * t + f_) else 0.0
    ious = []
    for c in range(4):
        t = int(cm[c, c])
        u = int(cm[c].sum() + cm[:, c].sum() - t)
        ious.append(t / u if u else 0.0)
    return {
        "n_pixels": total,
        "F1_change": float(f1), "IoU_change": float(iou),
        "precision_change": float(prec), "recall_change": float(rec),
        "Omission": float(1 - rec) if rec else 1.0,
        "Commission": float(1 - prec) if prec else 1.0,
        "OA": float(np.trace(cm) / total),
        "mIoU": float(np.mean(ious)),
        "per_class_F1": per,
        "confusion_matrix": cm.tolist(),
    }


def gather_coords(owner, valid, orgs, r0, c0):
    """Fixed traversal order for the mosaic, shared by every assignment mode.

    Returns (k, ys_off, xs_off, ys_abs, xs_abs) per patch. Both the single-patch
    assignments and the majority assignment MUST emit values in this order.
    Ordering them differently silently misaligns predictions against labels --
    which is exactly the bug this refactor fixes: the majority variant was
    emitted in bbox-row-major order while the labels were in (patch, row-major)
    order, which drove omission AND commission both to ~0.99.
    """
    out = []
    for k in range(len(orgs)):
        m = (owner == k) & valid
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        out.append((k, ys - (orgs[k][0] - r0), xs - (orgs[k][1] - c0), ys, xs))
    return out


def gather(cube, coords):
    """Per-pixel values from a (n_patch, 512, 512) cube, in `coords` order."""
    out = [cube[k][a, b] for k, a, b, _, _ in coords]
    return np.concatenate(out) if out else np.zeros(0, dtype=cube.dtype)


_GEOM_CACHE = {}


def region_geometry(orgs, mode):
    """cover + owner maps for a region. Depends only on the patch origins, so it
    is cached across models and seeds (a ~150x saving)."""
    key = (tuple(orgs), mode)
    if key in _GEOM_CACHE:
        return _GEOM_CACHE[key]
    n = len(orgs)
    r0 = min(o[0] for o in orgs)
    c0 = min(o[1] for o in orgs)
    r1 = max(o[0] for o in orgs) + PATCH
    c1 = max(o[1] for o in orgs) + PATCH
    H, W = r1 - r0, c1 - c0

    cover = np.zeros((H, W), dtype=np.int16)
    for (rr, cc) in orgs:
        cover[rr - r0:rr - r0 + PATCH, cc - c0:cc - c0 + PATCH] += 1

    # Nearest covering patch. The distance update is restricted to each patch's
    # own footprint: a pixel must be owned by a patch that actually covers it,
    # otherwise the within-patch gather below runs off the 512-px edge.
    best = np.full((H, W), np.inf, dtype=np.float32)
    owner = np.full((H, W), -1, dtype=np.int16)
    gy = np.arange(H, dtype=np.float32)[:, None] + r0
    gx = np.arange(W, dtype=np.float32)[None, :] + c0
    for k, (rr, cc) in enumerate(orgs):
        if mode == "nearest_origin":
            d = (gy - rr) ** 2 + (gx - cc) ** 2
        else:
            d = (gy - (rr + PATCH / 2)) ** 2 + (gx - (cc + PATCH / 2)) ** 2
        sub_d = d[rr - r0:rr - r0 + PATCH, cc - c0:cc - c0 + PATCH]
        sub_best = best[rr - r0:rr - r0 + PATCH, cc - c0:cc - c0 + PATCH]
        sub_owner = owner[rr - r0:rr - r0 + PATCH, cc - c0:cc - c0 + PATCH]
        m = sub_d < sub_best
        sub_best[m] = sub_d[m]
        sub_owner[m] = k
    valid = cover > 0
    out = (cover, owner, valid, r0, c0, H, W)
    _GEOM_CACHE[key] = out
    return out


def build_mosaic(preds, labels, orgs, mode, stats_only=False):
    """Partition the union of test footprints; return (cm, stats)."""
    n = len(orgs)
    cover, owner, valid, r0, c0, H, W = region_geometry(orgs, mode)

    coords = gather_coords(owner, valid, orgs, r0, c0)
    ym = gather(labels, coords)
    if mode == "majority":
        votes = np.zeros((4, H, W), dtype=np.int16)
        for k, (rr, cc) in enumerate(orgs):
            win = (slice(rr - r0, rr - r0 + PATCH), slice(cc - c0, cc - c0 + PATCH))
            pk = preds[k]
            for c in range(4):
                votes[(c,) + win][pk == c] += 1
        pm_mosaic = np.argmax(votes, axis=0).astype(np.uint8)
        del votes
        pm_valid = np.concatenate([pm_mosaic[ya, xa] for _, _, _, ya, xa in coords]) \
            if coords else np.zeros(0, dtype=np.uint8)
    else:
        pm_valid = gather(preds, coords)

    cm = cm4(ym, pm_valid)
    stats = {
        "bbox_px": [int(H), int(W)],
        "n_eval_patches": n,
        "n_unique_pixels": int(valid.sum()),
        "n_pixels_if_pooled": int(n * PATCH * PATCH),
        "duplication_factor": round(float(cover[valid].mean()), 4),
        "max_coverage": int(cover.max()),
    }
    if stats_only:
        return None, stats
    return cm, stats


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["nearest_centre"],
                    help="mosaic assignments to compute. nearest_centre is the "
                         "primary partition; add 'majority' (or "
                         "'nearest_origin') for the robustness variants. "
                         "majority is ~4x slower.")
    args = ap.parse_args()
    MODES = tuple(args.modes)
    t0 = time.time()

    out_dir = RESULTS_ROOT / "predictions"
    results = {}
    summary_src = {}
    for region in REGIONS:
        orgs = origins(region, SPLIT[region])
        nreg = 0
        for model in MODELS:
            for seed in SEEDS:
                pf = out_dir / f"{model}_s{seed}_{region}.npy"
                lf = out_dir / f"{model}_s{seed}_{region}_labels.npy"
                if not (pf.exists() and lf.exists()):
                    continue
                preds = np.load(pf)
                labels = np.load(lf)
                if len(preds) != len(orgs):
                    print("  !! patch count mismatch", model, seed, region,
                          len(preds), len(orgs))
                    continue
                key = f"{model}_s{seed}_{region}"
                rec = {"pooled": metrics_from_cm(cm4(labels, preds))}
                stats = None
                for mode in MODES:
                    cm, st = build_mosaic(preds, labels, orgs, mode)
                    rec[mode] = metrics_from_cm(cm)
                    if stats is None:
                        stats = st
                rec["mosaic_stats"] = stats
                results[key] = rec
                summary_src.setdefault(region, {}).setdefault(model, []).append(
                    (seed, rec))
                nreg += 1
        print("  %-10s %3d model-seed records  (%.0fs elapsed)"
              % (region, nreg, time.time() - t0), flush=True)

    (RESULTS_ROOT / "mosaic_eval.json").write_text(json.dumps(results, indent=1))

    # ---- 5-seed means: pooled vs mosaic, and leader re-ranking ----
    summary = {}
    for region, per_model in summary_src.items():
        rows = {}
        for model, recs in per_model.items():
            seeds = sorted(recs)
            if len(seeds) < 3:
                continue
            entry = {"n_seeds": len(seeds)}
            for mode in ("pooled",) + MODES:
                for metric in ("F1_change", "mIoU", "OA", "IoU_change",
                               "Omission", "Commission"):
                    entry[f"{mode}_{metric}"] = float(np.mean(
                        [r[1][mode][metric] for r in recs]))
            entry["delta_F1_change"] = (entry["nearest_centre_F1_change"]
                                        - entry["pooled_F1_change"])
            rows[model] = entry
        order_p = sorted(rows, key=lambda m: -rows[m]["pooled_F1_change"])
        order_m = sorted(rows, key=lambda m: -rows[m]["nearest_centre_F1_change"])
        summary[region] = {
            "per_model": rows,
            "leader_pooled": order_p[0],
            "leader_mosaic": order_m[0],
            "leader_changed": order_p[0] != order_m[0],
            "pooled_ranking": order_p,
            "mosaic_ranking": order_m,
            "mosaic_stats": next(iter(per_model.values()))[0][1]["mosaic_stats"],
            "delta_F1_change_range": [
                min(r["delta_F1_change"] for r in rows.values()),
                max(r["delta_F1_change"] for r in rows.values())],
        }

    (RESULTS_ROOT / "mosaic_eval_summary.json").write_text(
        json.dumps(summary, indent=1))

    print()
    print("  RESULTS_ROOT = %s" % RESULTS_ROOT)
    print("  cw_mode of first record: %s" % (
        (json.loads((RESULTS_ROOT / "metrics_unet_s2026.json").read_text())
         ["hyperparams"].get("cw_mode"))
        if (RESULTS_ROOT / "metrics_unet_s2026.json").exists() else "?"))
    print("== pooled (published, overlapping) vs de-duplicated mosaic ==")
    for region, s in summary.items():
        st = s["mosaic_stats"]
        print()
        print("  %s: %d eval patches, %d unique px vs %d pooled px "
              "(dup %.2fx, max cover %d)"
              % (region, st["n_eval_patches"], st["n_unique_pixels"],
                 st["n_pixels_if_pooled"], st["duplication_factor"],
                 st["max_coverage"]))
        print("    %-24s %7s %7s %8s   %7s %7s"
              % ("model", "pool", "mosaic", "dF1ch", "poolOA", "mosaicOA"))
        for m in s["mosaic_ranking"]:
            r = s["per_model"][m]
            print("    %-24s %7.3f %7.3f %+8.3f   %7.3f %7.3f"
                  % (m, r["pooled_F1_change"], r["nearest_centre_F1_change"],
                     r["delta_F1_change"], r["pooled_OA"],
                     r["nearest_centre_OA"]))
        print("    dF1ch range: %+.3f .. %+.3f   leader pooled=%s mosaic=%s %s"
              % (s["delta_F1_change_range"][0], s["delta_F1_change_range"][1],
                 s["leader_pooled"], s["leader_mosaic"],
                 "*** CHANGED ***" if s["leader_changed"] else "(unchanged)"))


if __name__ == "__main__":
    main()
