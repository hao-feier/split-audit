"""Label-noise sensitivity of the interleaved-vs-honest change-F1 gap.

Reviewer concern (glm / deepseek / claude, shared): every number is agreement
with a single GMW label, and GMW's errors concentrate in the 30 m fringe band
(the mangrove-extent boundary), where the honest arm has *more* change (89.8 %)
than the interleaved arm (82.3 %). If the +29.6 pp gap were an artefact of
testing noisier labels, it should shrink when the unreliable band is removed or
when fringe labels are perturbed to simulate that error.

This re-scores the SAME model (k1 checkpoint, held fixed) on the honest test
set and the three interleaved test sets -- exactly the setup of
threshold_f1_gap.py -- and moves the GMW *label*, not the model:

  baseline        : full GMW change label (reproduces the argmax gap)
  fringe_excl     : score only on non-fringe pixels (30 m band removed from
                    both label and prediction, symmetric)
  drop_{f}        : within the fringe band, omit a fraction f of the change
                    labels (GMW omission of thin/fringe mangrove)
  noise_{f}       : within the fringe band, flip each label with probability f
                    (symmetric GMW noise)

A large residual gap under fringe_excl and under realistic fringe error rates
bounds the reference-error channel away from the inflation; a collapsing gap
would attribute the gap to the unreliable band. Perturbations are applied
independently to each arm (same seed, fixed loop order) so both arms carry the
same simulated error rate.

Inference only. Output: results/uniform/label_noise_sensitivity.json
Run: MANGROVE_RESULTS_ROOT=results/uniform python3 analysis/label_noise_sensitivity.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from mangrove.datasets import load_region, load_split  # noqa: E402
from mangrove.models import MODEL_REGISTRY, DUAL_INPUT_MODELS  # noqa: E402
from threshold_f1_gap import get_probs4  # noqa: E402

MODELS = [
    "unet", "unet_cbam", "unet_cbam_msa", "unet_msa",
    "temporal_unet", "temporal_unet_gn", "deeplabv3plus",
    "deeplabv3plus_pre", "fc_siam", "changeformer",
]

FRINGE_PX = 3          # 30 m at 10 m/px, matches fringe_band_reference_error.py
SEED = 20261013
DROPS = (0.20, 0.40)
NOISE = (0.20,)


def fringe_mask(change):
    """Per-patch 30 m band around the mangrove-union boundary (bool)."""
    mng = (change != 3).astype(np.uint8)
    out = np.zeros_like(mng, dtype=bool)
    for p in range(mng.shape[0]):
        m = mng[p]
        bnd = m ^ ndimage.binary_erosion(m).astype(np.uint8)
        out[p] = ndimage.binary_dilation(bnd, iterations=FRINGE_PX).astype(bool)
    return out


def cf1(pred, lab, mask=None):
    """Micro-averaged change-F1, optionally restricted to a pixel mask."""
    if mask is not None:
        pred = pred & mask
        lab = lab & mask
    tp = int((pred & lab).sum())
    fp = int((pred & ~lab).sum())
    fn = int((~pred & lab).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    t0 = time.time()
    data = load_region("vietnam")
    splits = {
        "honest": np.asarray(load_split("vietnam", "test")),
        "leaky_k1": np.asarray(load_split("vietnam", "test_rand1")),
        "leaky_k2": np.asarray(load_split("vietnam", "test_rand2")),
        "leaky_k3": np.asarray(load_split("vietnam", "test_rand3")),
    }
    rng = np.random.default_rng(SEED)
    out = {
        "note": "k1 checkpoint held fixed; GMW change label perturbed (model "
                "unchanged). fringe band = 3 px (30 m) around the mangrove-union "
                "boundary. drop_f omits f of the fringe change labels; noise_f "
                "flips each fringe label with prob f. gap = interleaved mean "
                "minus honest, per perturbation; pooled = mean over models.",
        "fringe_px": FRINGE_PX,
        "rng_seed": SEED,
        "models": {},
    }
    for m in MODELS:
        ckpt = ROOT / "results/uniform/random_split_k1/checkpoints" / f"{m}_s2027.pt"
        if not ckpt.exists():
            print(f"[skip] {m}: no {ckpt.name}")
            continue
        is_temporal = m in DUAL_INPUT_MODELS
        in_ch = 9 if is_temporal else 27
        net = MODEL_REGISTRY[m](in_ch=in_ch, out_ch=4)
        net.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
        net.eval()

        rec = {}
        for n in splits:
            idx = splits[n]
            probs = get_probs4(net, data, idx, is_temporal)  # (N,4,H,W)
            change = data["change"][idx]                     # (N,H,W)
            label = ((change == 1) | (change == 2)).astype(bool).ravel()
            fringe = fringe_mask(change).ravel()
            pred = ((probs.argmax(axis=1) == 1) | (probs.argmax(axis=1) == 2)).ravel()

            arm = {"baseline": cf1(pred, label),
                   "fringe_excl": cf1(pred, label, ~fringe)}
            for f in DROPS:
                lab = label.copy()
                fc = np.where(fringe & label)[0]
                k = int(f * len(fc))
                if k:
                    lab[rng.choice(fc, size=k, replace=False)] = False
                arm[f"drop_{f:.2f}"] = cf1(pred, lab)
            for f in NOISE:
                flip = fringe & (rng.random(pred.size) < f)
                arm[f"noise_{f:.2f}"] = cf1(pred, label ^ flip)
            rec[n] = arm
        del net

        keys = ["baseline", "fringe_excl",
                *[f"drop_{f:.2f}" for f in DROPS],
                *[f"noise_{f:.2f}" for f in NOISE]]
        for key in keys:
            leaky = float(np.mean([rec[f"leaky_k{i}"][key] for i in (1, 2, 3)]))
            rec[key + "_gap"] = round(leaky - rec["honest"][key], 4)
        out["models"][m] = rec
        print(f"[{m:18s}] gap  base={rec['baseline_gap']:+.3f} "
              f"excl={rec['fringe_excl_gap']:+.3f} "
              f"drop.2={rec['drop_0.20_gap']:+.3f} "
              f"noise.2={rec['noise_0.20_gap']:+.3f}")

    keys = ["baseline", "fringe_excl",
            *[f"drop_{f:.2f}" for f in DROPS],
            *[f"noise_{f:.2f}" for f in NOISE]]
    out["pooled"] = {k + "_gap": round(float(np.mean([out["models"][m][k + "_gap"]
                                                       for m in out["models"]])), 4)
                      for k in keys}
    (ROOT / "results/uniform/label_noise_sensitivity.json").write_text(
        json.dumps(out, indent=2))
    print("\n[POOLED GAPS] " + json.dumps(out["pooled"]))
    print(f"[saved] results/uniform/label_noise_sensitivity.json  "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
