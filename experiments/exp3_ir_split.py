"""Exp3d: I/R factorial cell -- spatially isolated TRAIN, RANDOM test.

Completes the 2x2 spatial train/test factorial. Training is
bit-for-bit the main-sweep (honest, I/I) setup: the SAME 40 spatial train
patches, the SAME 10 spatial val patches (early stopping), the SAME
hyperparameters (cw sqrt, batch 8 / 2-for-temporal, 100 epochs, patience 15,
lr 1e-3, cosine). The ONLY difference vs the I/I run is the within-region
test set: 10 patches drawn at a fixed RNG seed from the 26 non-train,
non-val patches (10 buffer-isolated test-zone + 16 buffer patches), so test
patches interleave with the training zone (0--12.8 km edge distances).

Evaluation:
  * ir_test        -> the random test set (the I/R cell)
  * isolated_test  -> the buffer-isolated vietnam test zone (sanity: should
                      reproduce the main-sweep test_within_region for the
                      same seed, up to run-to-run non-determinism ~0.01)

Beilunhe / Zhanjiang are NOT re-evaluated: training is identical to the main
sweep, so the independent-region numbers there are already in
results/metrics_<model>_s<seed>.json.

Outputs: results/ir_k<k>/metrics_<model>_s<seed>.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Pre-parse --threads so OMP_NUM_THREADS is set BEFORE `import torch` (see
# exp3_train.py note: late OMP set still spawns 64 oneDNN threads -> OOM).
_t = None
_a = sys.argv[1:]
if "--threads" in _a and _a.index("--threads") + 1 < len(_a):
    _t = _a[_a.index("--threads") + 1]
for a in _a:
    if a.startswith("--threads="):
        _t = a.split("=", 1)[1]
os.environ.setdefault("OMP_NUM_THREADS", _t or "2")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))  # allow `import exp3_train`

from exp3_train import (  # noqa: E402  (identical training path as the main sweep)
    build_dataloaders, normalize, dice_loss, weighted_ce_det, evaluate,
    STATIC_IN_CH, TEMPORAL_IN_CH, TEMPORAL_MODELS, TRAIN_REGION)
from mangrove.config import SEED, TRAIN  # noqa: E402
from mangrove.datasets import ChangePatchDataset  # noqa: E402
from mangrove.models import MODEL_REGISTRY  # noqa: E402
from mangrove.paths import RESULTS_ROOT, SPLITS_ROOT  # noqa: E402


def ir_test_loader(model_name, ir_idx, num_workers, batch_size):
    """Test dataset built on the 'test' base split with the index set swapped
    to the IR random test patches (same trick as exp3c.rand_dataset)."""
    concat = model_name not in TEMPORAL_MODELS
    ds = ChangePatchDataset(TRAIN_REGION, "test", augment=False, concat=concat,
                            cw_mode="sqrt")
    ds.idx = list(ir_idx)
    bs = batch_size or TRAIN["batch_size"]
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--ir-k", type=int, required=True, choices=[1, 2, 3, 4, 5],
                    help="IR random-test draw (data/splits/vietnam_test_ir<k>.json)")
    ap.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    ap.add_argument("--batch-size", type=int, default=TRAIN["batch_size"])
    ap.add_argument("--lr", type=float, default=TRAIN["lr"])
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num-workers", type=int, default=TRAIN["num_workers"])
    ap.add_argument("--cw-mode", choices=["inv", "sqrt", "none"], default="sqrt")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        torch.set_num_threads(args.threads)

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    model_name = args.model
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    ir_idx = json.loads((SPLITS_ROOT / f"vietnam_test_ir{args.ir_k}.json").read_text())
    sub = RESULTS_ROOT / f"ir_k{args.ir_k}"
    (sub / "checkpoints").mkdir(parents=True, exist_ok=True)
    ckpt = sub / "checkpoints" / f"{model_name}_s{seed}.pt"
    log_path = sub / f"train_log_{model_name}_s{seed}.jsonl"
    metrics_path = sub / f"metrics_{model_name}_s{seed}.json"

    # Identical train/val construction as exp3_train (the I/I main sweep).
    train_dl, val_dl, _test_dl_unused, train_ds = build_dataloaders(
        model_name, num_workers=args.num_workers, batch_size=args.batch_size,
        cw_mode=args.cw_mode, train_idx=None, val_idx=None, gen=gen,
        val_split="val")
    ir_dl = ir_test_loader(model_name, ir_idx, args.num_workers, args.batch_size)
    iso_dl = _test_dl_unused  # the buffer-isolated vietnam test zone

    is_temporal = model_name in TEMPORAL_MODELS
    in_ch = TEMPORAL_IN_CH if is_temporal else STATIC_IN_CH
    model = MODEL_REGISTRY[model_name](in_ch=in_ch, out_ch=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ir] model={model_name} in_ch={in_ch} params={n_params/1e6:.2f}M "
          f"device={device} train={len(train_dl)} val={len(val_dl)} "
          f"ir_test={len(ir_dl)} isolated_test={len(iso_dl)} "
          f"cw_mode={args.cw_mode} seed={seed} k={args.ir_k}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=TRAIN["weight_decay"])
    w_ce = train_ds.class_weights.to(device)
    crit_ce = nn.CrossEntropyLoss(weight=w_ce)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_loss, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        for batch in train_dl:
            if is_temporal:
                x1, x2, y, _ = batch
                x1, x2 = normalize(x1).to(device), normalize(x2).to(device)
                y = y.to(device)
                out = model(x1, x2)
            else:
                x, y, _ = batch
                x = normalize(x).to(device)
                y = y.to(device)
                out = model(x)
            opt.zero_grad()
            loss = crit_ce(out, y) + dice_loss(out, y)
            loss.backward()
            opt.step()
            run_loss += float(loss.detach()) * y.size(0)
            n += y.size(0)
        sched.step()
        val = evaluate(model, val_dl, temporal=is_temporal, device=device)
        val_loss = 1.0 - val["mIoU"]
        dur = time.time() - t0
        rec = {"epoch": epoch, "train_loss": run_loss / n, "val_OA": val["OA"],
               "val_F1": val["F1_macro"], "val_mIoU": val["mIoU"],
               "elapsed_min": round(dur / 60, 1), "lr": opt.param_groups[0]["lr"]}
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if val_loss < best_loss - 1e-4:
            best_loss, bad = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(best_state, ckpt)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    val_m = evaluate(model, val_dl, temporal=is_temporal, device=device)
    ir_m = evaluate(model, ir_dl, temporal=is_temporal, device=device)
    iso_m = evaluate(model, iso_dl, temporal=is_temporal, device=device)
    res = {
        "model": model_name,
        "tag": f"ir_k{args.ir_k}",
        "params_M": round(n_params / 1e6, 3),
        "seed": seed,
        "in_ch": in_ch,
        "ir_k": args.ir_k,
        "ir_test_patches": ir_idx,
        "hyperparams": {k: args.__dict__[k] for k in
                        ("epochs", "batch_size", "lr", "patience", "seed",
                         "cw_mode")},
        "val": val_m,
        "ir_test": ir_m,
        "isolated_test": iso_m,
        "elapsed_h": round((time.time() - t0) / 3600, 2),
    }
    metrics_path.write_text(json.dumps(res, indent=2))
    def _oa(m):
        return f"{m['OA']:.4f}" if m else "n/a"
    print(f"val OA={val_m['OA']:.4f} ir OA={_oa(ir_m)} isolated OA={_oa(iso_m)} "
          f"({res['elapsed_h']} h)", flush=True)


if __name__ == "__main__":
    main()
