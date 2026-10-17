"""E3-B: emit the 16 locked block-interleaved splits + their matched controls.

The split list is NOT chosen here. It was frozen beforehand by
analysis/e3b_preflight.py (double stratification: test-set size >= 5 AND
achievable prevalence-match residual <= 0.5 pp), written to
logs/e3b_selection_lock.txt, and is read back from results/e3b_preflight.json.
This script only *materialises* those frozen choices into
data/splits/vietnam_{train,val,test}_blk<ss>.json and the _hon<ss>.json partner
arms, reusing the exact test/train/val indices and matched subset recorded in
the pre-flight, so the emitted files cannot differ from the locked design.

Zero selection freedom: all 16 candidates that clear both criteria are used, so
there is no draw to be accused of being outcome-informed.

Usage:  python analysis/e3b_make_splits.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "mbs", ROOT / "experiments" / "make_block_split.py")
mbs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mbs)

PRE = ROOT / "results" / "e3b_preflight.json"
LOCK = ROOT / "logs" / "e3b_selection_lock.txt"
SPLITS = ROOT / "data" / "splits"

pre = json.loads(PRE.read_text())
assert LOCK.exists(), "selection lock missing -- run e3b_preflight.py first"
selected = set(pre["selected"])

# rebuild the double-stratified set exactly as the pre-flight defined it
cands = [c for c in pre["candidates"]
         if c["block_seed"] in selected]
assert len(cands) == len(selected), "locked seeds missing from candidate list"

emitted = []
for c in sorted(cands, key=lambda x: (-x["n_test"], x["block_seed"])):
    ss = c["block_seed"]
    for arm, trk, vak, tek in (
            ("blk", "train_idx", "val_idx", "test_idx"),
            ("hon", None, None, "matched_test_idx")):
        if arm == "blk":
            tr, va, te = c["train_idx"], c["val_idx"], c["test_idx"]
        else:
            # matched arm: same counts as the block arm; train/val are drawn
            # from their own zones exactly as make_block_split.matched_honest
            # does, and the test subset is the pre-flight's prevalence-matched
            # subset (already frozen, so it is reused verbatim rather than
            # re-searched).
            import numpy as np
            rng = np.random.default_rng(10_000 + ss)
            trz = json.loads((SPLITS / "vietnam_train.json").read_text())
            vaz = json.loads((SPLITS / "vietnam_val.json").read_text())
            tr = sorted(int(x) for x in
                        np.asarray(trz)[rng.permutation(len(trz))[:c["n_train"]]])
            va = sorted(int(x) for x in
                        np.asarray(vaz)[rng.permutation(len(vaz))[:c["n_val"]]])
            te = c["matched_test_idx"]
        for name, idx in (("train", tr), ("val", va), ("test", te)):
            (SPLITS / f"vietnam_{name}_{arm}{ss}.json").write_text(
                json.dumps([int(i) for i in idx]))
    emitted.append({
        "split_seed": ss, "n_train": c["n_train"], "n_val": c["n_val"],
        "n_test": c["n_test"],
        "block_test_prev_pct": c["block_test_prev_pct"],
        "matched_test_prev_pct": c["matched_test_prev_pct"],
        "prev_residual_pp": c["prev_residual_pp"],
    })
    print(f"blk{ss}/hon{ss}  n=({c['n_train']},{c['n_val']},{c['n_test']})  "
          f"prev {c['block_test_prev_pct']:.3f} vs {c['matched_test_prev_pct']:.3f}  "
          f"resid {c['prev_residual_pp']:.3f} pp")

out = ROOT / "results" / "e3b_splits.json"
out.write_text(json.dumps({"rule": pre["rule"], "splits": emitted}, indent=1))
print(f"\nemitted {len(emitted)} split pairs -> {SPLITS}")
print(f"-> {out}")
res = [e["prev_residual_pp"] for e in emitted]
print(f"prevalence residual across the locked set: max {max(res):.3f} pp, "
      f"mean {sum(res)/len(res):.3f} pp")