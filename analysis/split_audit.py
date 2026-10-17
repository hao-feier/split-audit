"""Split-geometry audit.

For every Vietnam split under data/splits/ (v1, v2, buf{1,2,4,8}, current
top-level) report:
  - zone sizes (train / val / test / buffer),
  - column spans (km) of the train and val/test zones,
  - min edge-to-edge distance (km) between the train zone and the val/test
    zone, and between val and test (5120 m patch squares, same metric as
    experiments/exp2_split_v2.py),
  - set identity: which splits are the identical patch sets.

Also records the grid-snap effect: with a 256-px (2.56 km) stride and 5120 m
patches, the achievable val-test edge distances are
{0, 2.56, 5.12, ...} km, so a requested buffer width B < 2.56 km yields an
effective 2.56 km gap (B = 1 and B = 2 produce the identical split), and
B = 8 leaves no valid val/test pair.

Output: results/split_audit.json
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
META = ROOT / "data" / "processed" / "vietnam" / "metadata.json"
STRIDE_M = 256 * 10.0      # 2560 m between patch centers
EXT = 512 * 10 / 2         # patch half-extent, 2560 m (side = 5120 m)
NAMES = ["v1", "v2", "buf1", "buf2", "buf4", "buf8", "current"]
ZONES = ("train", "val", "test", "buffer")


def centers():
    meta = json.loads(META.read_text())
    p = np.array(meta["patches"])
    return np.stack([(p[:, 1] + 256) * 10.0, (p[:, 0] + 256) * 10.0], axis=1)  # (x, y) m


def rect_edge(a, b):
    return math.hypot(max(0.0, abs(a[0] - b[0]) - 2.0 * EXT),
                      max(0.0, abs(a[1] - b[1]) - 2.0 * EXT))


def min_pair(A, B):
    if len(A) == 0 or len(B) == 0:
        return None
    return min(rect_edge(a, b) for a in A for b in B) / 1000.0  # km


def load_zone(base, z):
    return json.loads((base / f"vietnam_{z}.json").read_text())


def signature(base):
    return tuple(tuple(sorted(load_zone(base, z))) for z in ZONES)


def audit(name, C):
    base = SPLITS if name == "current" else SPLITS / name
    if not (base / "vietnam_train.json").exists():
        return {"missing": True}
    z = {k: load_zone(base, k) for k in ZONES}
    vt = z["val"] + z["test"]
    span = lambda idx: ([round(float(v), 2) for v in
                         (C[idx][:, 0].min() / 1000, C[idx][:, 0].max() / 1000)] if idx else None)
    return {
        "n": {k: len(z[k]) for k in ZONES},
        "train_cols_km": span(z["train"]),
        "vt_cols_km": span(vt),
        "train_to_vt_gap_km": min_pair(C[z["train"]], C[vt]),
        "val_to_test_gap_km": min_pair(C[z["val"]], C[z["test"]]),
        "_sig": signature(base),
    }


def main():
    C = centers()
    out = {
        "grid": {
            "stride_km": STRIDE_M / 1000.0,
            "patch_side_km": 2.0 * EXT / 1000.0,
            "achievable_val_test_edge_distances_km": [
                round(k * STRIDE_M / 1000.0 - 2.0 * EXT / 1000.0, 2) for k in range(2, 7)],
            "note": "requested buffer B snaps to the next achievable val-test edge "
                    "distance; B=1 and B=2 give the identical split, B=8 leaves no "
                    "valid val/test pair.",
        },
        "splits": {},
    }
    for name in NAMES:
        res = audit(name, C)
        sig = res.pop("_sig", None)
        if "missing" not in res:
            res["train_to_vt_gap_km"] = None if res["train_to_vt_gap_km"] is None else round(res["train_to_vt_gap_km"], 2)
            res["val_to_test_gap_km"] = None if res["val_to_test_gap_km"] is None else round(res["val_to_test_gap_km"], 2)
        out["splits"][name] = res
        out.setdefault("_sigs", {})[name] = sig
    sigs = out.pop("_sigs")
    groups = {}
    for name, sig in sigs.items():
        if sig is not None:
            groups.setdefault(sig, []).append(name)
    out["identical_sets"] = [g for g in groups.values() if len(g) > 1]
    (ROOT / "results" / "split_audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
