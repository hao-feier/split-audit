"""E3-B composition penalty C -- measurement protocol and PRE-COMMITTED USE RULE.

Written BEFORE any C value was computed, so that C cannot be used
post-hoc to rescue a stronger conclusion. The rule below is fixed in advance of
the measurement.

PRE-COMMITTED USE RULE (binding, decided before seeing C)
-------------------------------------------------------
  measured C          reported wording                                    conclusion
  < ~5 pp             "composition penalty is bounded and far below the     Option 3
                      >= 39 pp required to mask a +29 pp inflation, so
                      'the inflation is present but masked' fails on
                      magnitude"
  5 - 20 pp           report the number; state the two-sided comparison      Option 3
                      is not identified under this geometry
  >= ~20 pp           "E3-B has no power against the pre-registered          Option 3
                      effect"; add an explicit design-power statement       + power
                                                                              failure
  NO BRANCH UPGRADES THE CONCLUSION. C is a supplementary bounding
  measurement. Option 3 was reached on the combination-decomposition
  dispersion (5.1 pp within, 19.3 pp between, against a 2-3 pp criterion),
  NOT on C, and no value of C can change that. C exists to close off one
  specific objection ("the inflation is there but the matched arm is
  harder"), not to re-open the strong wording.

MEASUREMENT PROTOCOL (revised before running)
----------------------------------------------
C itself is dominated by the same 5-6-patch geometry, so a single equal-count
subset comparison is ONE noisy draw. Protocol: for every (split, architecture,
run seed) block-trained model, draw 100 random equal-count subsets from the
block-side candidate pool and report the DISTRIBUTION of C (median and 5-95
percentiles), not a point estimate. This also makes the n=1-combination
sensitivity check meaningful, since each combination now carries 100 draws.

  C = F1(block-side equal-count control subset) - F1(matched honest-zone set)
      same model, pure evaluation, no training.

net = I - C, so masking a pre-registered I >= +29 pp behind the observed net
requires C >= 39 pp.

Usage:  MANGROVE_RESULTS_ROOT=results/uniform python analysis/e3b_composition_penalty.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from aggregate_results import binary_change  # noqa: E402
from mangrove.datasets import ChangePatchDataset  # noqa: E402
from mangrove.models import MODEL_REGISTRY, DUAL_INPUT_MODELS  # noqa: E402
from mangrove.paths import SPLITS_ROOT  # noqa: E402

RESULTS = Path(os.environ.get("MANGROVE_RESULTS_ROOT", ROOT / "results" / "uniform"))
REGION = "vietnam"
DESIGN_ARCHS = ["unet", "unet_cbam_msa", "temporal_unet"]
N_DRAWS = 100          # per (split, arch, seed): pre-specified above
MASK_TO_HIDE_I29 = 39.0   # = 29 - (-10); the C that would be required
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


_REGION_CACHE = {}


def _region():
    """load_region() reads the whole feature stack into memory. Constructing a
    ChangePatchDataset per draw (100 per model x 39 models) would reload it
    thousands of times; cache it once."""
    if "d" not in _REGION_CACHE:
        from mangrove.datasets import load_region
        _REGION_CACHE["d"] = load_region(REGION)
    return _REGION_CACHE["d"]


def score(model, idx, concat):
    """Confusion matrix of `model` over the given patch indices.

    ChangePatchDataset returns 3 values when concat=True (x, y, i) and 4 when
    concat=False (x1, x2, y, i); the temporal models take the two epochs
    separately. Handled explicitly rather than assumed.
    """
    d = _region()
    t1, t2, ch = d["t1_feat"], d["t2_feat"], d["change"]
    cm = np.zeros((4, 4), dtype=np.int64)
    with torch.no_grad():
        for i in idx:
            y = ch[i].astype(np.int64)
            if concat:
                a = t1[i].astype(np.float32)
                b = t2[i].astype(np.float32)
                x = torch.from_numpy(
                    np.concatenate([a, b, np.abs(a - b)],
                                   axis=-1).transpose(2, 0, 1).copy())
                out = model(x.unsqueeze(0).to(device) / 10000.0)
            else:
                x1 = torch.from_numpy(
                    t1[i].astype(np.float32).transpose(2, 0, 1).copy())
                x2 = torch.from_numpy(
                    t2[i].astype(np.float32).transpose(2, 0, 1).copy())
                out = model(x1.unsqueeze(0).to(device) / 10000.0,
                            x2.unsqueeze(0).to(device) / 10000.0)
            p = out.argmax(1).cpu().numpy()[0]
            cm += np.bincount(p.ravel() * 4 + y.ravel(),
                              minlength=16).reshape(4, 4)
    return cm


def _unused_score(model, idx, concat):
    """Superseded by the cached version above; kept only to document the
    original (dataset-based) behaviour."""
    ds = ChangePatchDataset(REGION, "test", augment=False, concat=concat,
                            idx_filter=set(idx))
    cm = np.zeros((4, 4), dtype=np.int64)
    with torch.no_grad():
        for k in range(len(ds.idx)):
            item = ds[k]
            if concat:
                x, y = item[0], item[1]
                out = model((x.unsqueeze(0).to(device) / 10000.0))
            else:
                x1, x2, y = item[0], item[1], item[2]
                out = model(x1.unsqueeze(0).to(device) / 10000.0,
                            x2.unsqueeze(0).to(device) / 10000.0)
            p = out.argmax(1).cpu().numpy()[0]
            cm += np.bincount(p.ravel() * 4 + y.numpy().ravel(),
                              minlength=16).reshape(4, 4)
    return cm


def f1(model, idx, concat):
    return binary_change(score(model, idx, concat))["F1_change"]


meta = json.loads((ROOT / "results" / "e3b_splits.json").read_text())
rep = {a["split_seed"]: a for a in
       json.loads((ROOT / "results" / "e3b_repair_audit.json").read_text())["audit"]}
feasible = [r["split_seed"] for r in meta["splits"]
            if rep[r["split_seed"]]["status"] != "geometrically_infeasible"]

rows = []
PROTOCOL_CORRECTION = None
for ss in feasible:
    hon = json.loads((SPLITS_ROOT / f"vietnam_test_hon{ss}.json").read_text())
    blk_tr = json.loads((SPLITS_ROOT / f"vietnam_train_blk{ss}.json").read_text())
    blk_va = json.loads((SPLITS_ROOT / f"vietnam_val_blk{ss}.json").read_text())
    blk_te = set(json.loads((SPLITS_ROOT / f"vietnam_test_blk{ss}.json").read_text()))
    hon_te = set(json.loads((SPLITS_ROOT / "vietnam_test.json").read_text()))
    n = len(hon)

    # Control pool, CORRECTED. The first implementation drew the control subsets
    # from the block arm's TRAIN patches, which the model had been fitted on, so
    # the control scored training data and C was not interpretable (the first run
    # returned C = -9.4 pp with f_matched = 0.24, i.e. a training-vs-unseen
    # comparison). The pool must be UNSEEN by the block-trained model, outside the
    # honest test zone (that is the matched side), and disjoint from the block
    # test set. 40-42 patches qualify per split.
    seen = set(blk_tr) | set(blk_va)
    pool = sorted(set(range(76)) - seen - hon_te - blk_te)
    if PROTOCOL_CORRECTION is None:
        PROTOCOL_CORRECTION = {
            "defect": "control pool was the block arm's own TRAIN patches, which the "
                      "model was fitted on; C compared training against unseen data",
            "first_run_C_median_pp": -9.3526,
            "fix": "control pool = patches unseen by the block-trained model, "
                   "outside the honest test zone, disjoint from the block test set",
            "pool_size_range": [40, 42],
        }
    if len(pool) < n:
        continue
    for arch in DESIGN_ARCHS:
        concat = arch not in DUAL_INPUT_MODELS
        in_ch = 9 if not concat else 27
        for seed in (2026, 2027):
            ck = RESULTS / f"block_blk{ss}" / "checkpoints" / f"{arch}_s{seed}.pt"
            if not ck.exists():
                continue
            net = MODEL_REGISTRY[arch](in_ch=in_ch, out_ch=4).to(device)
            net.load_state_dict(torch.load(ck, map_location=device))
            net.eval()
            f_hon = f1(net, hon, concat)
            rng = np.random.default_rng(20261001 + ss + seed)
            draws = np.array([f1(net, sorted(rng.choice(pool, size=n,
                                                         replace=False).tolist()),
                                concat) for _ in range(N_DRAWS)])
            C = 100.0 * (draws - f_hon)
            rows.append({"split": ss, "arch": arch, "seed": seed,
                         "n_test": n, "f_matched": f_hon,
                         "C_median_pp": float(np.median(C)),
                         "C_p5_pp": float(np.percentile(C, 5)),
                         "C_p95_pp": float(np.percentile(C, 95)),
                         "C_draws_pp": [float(x) for x in C]})
            del net
            torch.cuda.empty_cache()

if not rows:
    print("no rows"); sys.exit(0)

med = np.array([r["C_median_pp"] for r in rows])
C_med = float(np.median(med))

if C_med < 5:
    band = ("<5", "composition penalty is bounded and far below the >= 39 pp "
            "required to mask a +29 pp inflation, so 'the inflation is present "
            "but masked' fails on magnitude")
elif C_med < 20:
    band = ("5-20", "report the number; the two-sided comparison is not "
            "identified under this geometry")
else:
    band = (">=20", "E3-B has no power against the pre-registered effect; "
            "explicit design-power failure")
concl = "Option 3" + (" + design-power failure" if C_med >= 20 else "")

out = {
    "pre_committed_use_rule": {
        "<5pp": "bounded, masking fails on magnitude -> Option 3",
        "5-20pp": "report number, two-sided not identified -> Option 3",
        ">=20pp": "no power vs pre-registered effect -> Option 3 + power failure",
        "invariance": ("no branch upgrades the conclusion; Option 3 was reached on "
                       "combination-dispersion (5.1 pp within / 19.3 pp between vs a "
                       "2-3 pp criterion), not on C"),
    },
    "protocol": {"n_draws_per_model": N_DRAWS,
                 "control_pool": "patches UNSEEN by the block-trained model, "
                                 "outside the honest test zone, disjoint from the "
                                 "block test set",
                 "statistic": "distribution of C, median + 5-95 percentiles",
                 "correction": PROTOCOL_CORRECTION},
    "C_median_pp": C_med,
    "C_p5_pp": float(np.percentile(med, 5)),
    "C_p95_pp": float(np.percentile(med, 95)),
    "C_required_to_mask_I29_pp": MASK_TO_HIDE_I29,
    "C_can_mask_I29": bool(C_med >= MASK_TO_HIDE_I29),
    "measured_band": band[0],
    "wording": band[1],
    "conclusion_under_rule": concl,
    "f_matched_mean": float(np.mean([r["f_matched"] for r in rows])),
    "rows": rows,
}
(ROOT / "results" / "e3b_composition_penalty.json").write_text(json.dumps(out, indent=1))
print(f"C median = {C_med:+.2f} pp   (5-95%: {out['C_p5_pp']:+.2f} to "
      f"{out['C_p95_pp']:+.2f})   rows={len(rows)}")
print(f"required to mask I>=+29: {MASK_TO_HIDE_I29:.0f} pp -> "
      f"{'CANNOT' if not out['C_can_mask_I29'] else 'CAN'}")
print(f"band={band[0]} -> {concl}")
print("-> results/e3b_composition_penalty.json")