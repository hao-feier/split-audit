"""Exp2b: split v2 with a GUARANTEED minimum edge distance between sets.

The v1 row split (median +/- 2 km on patch-centre rows) only
removes the median patch; the nearest val/test patch pair touches (edge
distance 0 km), so early stopping on the val zone could leak model-selection
signal into the within-region test zone.

v2 rule: after the column zones of exp2 (55/65 percentiles of patch-centre
x-coordinates) are assigned, the val/test row boundary is chosen so that
EVERY val-test patch pair is separated by >= B (rectangle edge distance).
Patches that cannot satisfy this are moved to the buffer. Train is never
touched (train <-> (val U test) edge is already 5.12 km >= 2 km).

Modes:
  minimal (default): column zones fixed as in v1; only the row boundary is
                     re-optimised. Keeps the training set identical to v1.
  general:           the train/vt column boundary is also widened when B
                     exceeds the v1 column gap (used for the buffer-width
                     sensitivity analysis, B = 1/4/8 km).

Outputs (default): data/splits/v2/vietnam_{train,val,test,buffer}.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
EXT = 512 * 10 / 2  # patch half-extent, meters (5.12 km side -> 2560 m)
REGION = "vietnam"


def centers(meta):
    p = np.array(meta["patches"])
    return np.stack([(p[:, 1] + 256) * 10.0, (p[:, 0] + 256) * 10.0], axis=1)  # (x, y) m


def rect_edge(a, b):
    # edge-to-edge distance between two 5120 m x 5120 m squares (side = 2*EXT)
    dx = max(0.0, abs(a[0] - b[0]) - 2.0 * EXT)
    dy = max(0.0, abs(a[1] - b[1]) - 2.0 * EXT)
    return math.hypot(dx, dy)


def min_pair_dist(A, B):
    if not A or not B:
        return math.inf
    return min(rect_edge(a, b) for a in A for b in B)


def row_split(vt, C, B):
    """Assign vt patches to val (north) / test (south) / buffer so that
    min edge distance between any val-test pair >= B. Maximises |val|+|test|;
    ties broken by closeness to the v1 sizes (10 val, 11 test)."""
    order = sorted(vt, key=lambda i: C[i][1])
    rows = [C[i][1] for i in order]
    best = None
    for bi in range(1, len(order)):  # boundary between order[bi-1] and order[bi]
        val_c = list(order[:bi])
        test_c = list(order[bi:])
        # iteratively remove the patch with the most violating partners
        removed = set()
        changed = True
        while changed:
            changed = False
            worst = None
            for p in val_c + test_c:
                if p in removed:
                    continue
                part = test_c if p in val_c else val_c
                viol = [q for q in part if q not in removed and rect_edge(C[p], C[q]) < B]
                if viol and (worst is None or len(viol) > len(worst[1])):
                    worst = (p, viol)
            if worst:
                removed.add(worst[0])
                changed = True
        val = [p for p in val_c if p not in removed]
        test = [p for p in test_c if p not in removed]
        buf = [p for p in order if p in removed]
        # primary: stay as close to the v1 sizes (10 val, 11 test) as
        # possible; secondary: maximise total retained patches
        score = (-abs(len(val) - 10) - abs(len(test) - 11), len(val) + len(test))
        if best is None or score > best[0]:
            best = (score, val, test, buf)
    return best[1], best[2], best[3]


def make_split(meta, B, mode="minimal"):
    C = centers(meta)
    cols = C[:, 0]
    p55, p65 = np.percentile(cols, 55), np.percentile(cols, 65)
    if mode == "minimal":
        train = [i for i in range(len(cols)) if cols[i] < p55]
        vt = [i for i in range(len(cols)) if cols[i] > p65]
    elif mode == "hybrid":
        # keep the v1 column zones; move patches that violate the B edge
        # guarantee into the buffer (used for buffer-width sensitivity)
        train = [i for i in range(len(cols)) if cols[i] < p55]
        vt = [i for i in range(len(cols)) if cols[i] > p65]
        changed = True
        while changed:
            changed = False
            keep_train = [i for i in train
                          if all(rect_edge(C[i], C[j]) >= B for j in vt)]
            keep_vt = [i for i in vt
                       if all(rect_edge(C[i], C[j]) >= B for j in keep_train)]
            if len(keep_train) != len(train) or len(keep_vt) != len(vt):
                changed = True
            train, vt = keep_train, keep_vt
    else:  # general: widen column zones as needed for B
        cands = sorted(set(float(x) for x in cols))
        best = None
        for b2 in cands:
            vt = [i for i in range(len(cols)) if cols[i] >= b2]
            train = [i for i in range(len(cols))
                     if all(rect_edge(C[i], C[j]) >= B for j in vt)]
            score = (len(train), len(vt))
            if best is None or score > best[0]:
                best = (score, b2, train, vt)
        _, b2, train, vt = best
    val, test, buf = row_split(vt, C, B)
    colbuf = [i for i in range(len(cols)) if i not in train + vt + val + test + buf]
    return {"train": train, "val": val, "test": test, "buffer": buf + colbuf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer-km", type=float, default=2.0)
    ap.add_argument("--mode", choices=("minimal", "hybrid", "general"), default="minimal")
    ap.add_argument("--outdir", default=str(ROOT / "data" / "splits" / "v2"))
    args = ap.parse_args()

    meta = json.loads((PROCESSED / REGION / "metadata.json").read_text())
    out = make_split(meta, args.buffer_km * 1000.0, args.mode)
    od = Path(args.outdir)
    od.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test", "buffer"):
        (od / f"{REGION}_{name}.json").write_text(json.dumps(out[name]))

    # verification: min edge distance across all set pairs
    C = centers(meta)
    sets = {k: np.array(v) for k, v in out.items()}
    print(f"buffer={args.buffer_km} km mode={args.mode}")
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                d = min_pair_dist([C[i] for i in sets[a]], [C[i] for i in sets[b]])
                print(f"  {a:5s} vs {b:5s}: n={len(sets[a])}x{len(sets[b])}, "
                      f"min edge = {d/1000:.2f} km")
    print(f"  sizes: train={len(sets['train'])} val={len(sets['val'])} "
          f"test={len(sets['test'])} buffer={len(sets['buffer'])} "
          f"total={sum(len(v) for v in sets.values())}")


if __name__ == "__main__":
    main()
