"""Re-evaluate existing v2 seed-2026 checkpoints (NO retraining).

The v2 review-response sweep (data/splits v2, >=2 km edge guarantee) trained
all three seed-2026 checkpoints but the eval step (the metrics JSON) was lost
to an OOM, so results/ has no metrics_{unet,unet_cbam,unet_cbam_msa}_s2026.json.
The .pt checkpoints on disk ARE the v2 models, so loading them and running the
identical evaluate() reproduces exactly the metrics the sweep should have
emitted. This restores 3-seed reproducibility for the main tables.

Run on the GPU host:
    python experiments/reeval_s2026.py
    python experiments/reeval_s2026.py --models unet unet_cbam unet_cbam_msa --seed 2026
Then regenerate the aggregation + tables:
    python analysis/aggregate_results.py
    python analysis/make_latex_tables.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import torch
from torch.utils.data import DataLoader

import exp3_train as T
from mangrove.config import TRAIN
from mangrove.datasets import ChangePatchDataset
from mangrove.models import MODEL_REGISTRY
from mangrove import paths
from mangrove.paths import SPLITS_ROOT

MODELS_DEFAULT = ["unet", "unet_cbam", "unet_cbam_msa"]


def pick_device():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.device("cuda")
    return torch.device("cpu")


def reeval_one(model_name, seed, device):
    is_temporal = model_name in T.TEMPORAL_MODELS
    concat = not is_temporal
    in_ch = T.TEMPORAL_IN_CH if is_temporal else T.STATIC_IN_CH

    ckpt = paths.ckpt_path(model_name, seed)
    if not ckpt.exists():
        print(f"SKIP {model_name} s{seed}: checkpoint {ckpt} missing", flush=True)
        return False

    model = MODEL_REGISTRY[model_name](in_ch=in_ch, out_ch=4)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    bs = TRAIN["batch_size"]

    def ds_loader(region, split):
        ds = ChangePatchDataset(region, split, augment=False, concat=concat)
        return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

    t0 = time.time()
    val_m = T.evaluate(model, ds_loader(T.TRAIN_REGION, "val"),
                       temporal=is_temporal, device=device)
    test_m = T.evaluate(model, ds_loader(T.TRAIN_REGION, "test"),
                        temporal=is_temporal, device=device)
    indep = {}
    for region in T.TEST_REGIONS:
        if not (SPLITS_ROOT / f"{region}_test.json").exists():
            print(f"WARNING {model_name} s{seed}: {region}_test.json missing", flush=True)
            indep[region] = None
            continue
        indep[region] = T.evaluate(model, ds_loader(region, "test"),
                                   temporal=is_temporal, device=device)

    res = {
        "model": model_name,
        "params_M": round(n_params / 1e6, 3),
        "seed": seed,
        "in_ch": in_ch,
        # Mirror the recorded sweep hyperparams exactly (the v2 review-response
        # sweep ran --batch-size 8 / --lr 0.001 / --patience 15 / --cw-mode sqrt,
        # overriding config TRAIN's batch_size=2) so this file matches s2027/s2028.
        "hyperparams": {"epochs": 100, "batch_size": 8, "lr": 0.001,
                        "patience": 15, "seed": seed, "cw_mode": "sqrt"},
        "val": val_m,
        "test_within_region": test_m,
        "independent_test": indep,
        "elapsed_h": round((time.time() - t0) / 3600, 3),
    }
    mp = paths.metrics_path(model_name, seed)
    mp.write_text(json.dumps(res, indent=2))
    def _oa(m):
        return f"{m['OA']:.4f}" if m else "n/a"
    print(f"DONE {model_name} s{seed}: val OA={val_m['OA']:.4f} "
          f"within OA={test_m['OA']:.4f} beilunhe OA={_oa(indep['beilunhe'])} "
          f"zhanjiang OA={_oa(indep['zhanjiang'])} -> {mp.name} "
          f"({res['elapsed_h']} h)", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS_DEFAULT)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    device = pick_device()
    print(f"device={device} models={args.models} seed={args.seed}", flush=True)
    ok = [reeval_one(m, args.seed, device) for m in args.models]
    print(f"re-eval complete: {sum(ok)}/{len(ok)} succeeded", flush=True)


if __name__ == "__main__":
    main()
