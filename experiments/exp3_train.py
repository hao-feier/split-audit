"""Exp3: unified training pipeline for all deep models (PLAN 4-5).

Usage:
  python exp3_train.py --model unet          # Exp0 baseline
  python exp3_train.py --model unet_cbam     # Exp1 + CBAM
  python exp3_train.py --model unet_cbam_msa # Exp2 + CBAM + MSA
  python exp3_train.py --model deeplabv3plus
  python exp3_train.py --model temporal_unet # Plan C temporal fusion

Outputs:
  results/checkpoints/<model>.pt
  results/train_log_<model>.jsonl
  results/metrics_<model>.json   (val + test)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# oneDNN/OpenMP capture the thread count when torch is imported; setting
# OMP_NUM_THREADS later (e.g. in main()) is too late — the process still
# spawns 64 threads and OOMs this 15.9 GB host (2026-09-20). Pre-parse
# --threads so the env var is in place before `import torch`; an explicit
# OMP_NUM_THREADS from the launcher always wins.
_t = None
_a = sys.argv[1:]
if "--threads" in _a and _a.index("--threads") + 1 < len(_a):
    _t = _a[_a.index("--threads") + 1]
for a in _a:
    if a.startswith("--threads="):
        _t = a.split("=", 1)[1]
os.environ.setdefault("OMP_NUM_THREADS", _t or "2")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
# --deterministic (noise-floor audit): cuBLAS needs the workspace
# config before torch's first CUDA call, so pre-parse it here.
if "--deterministic" in _a:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mangrove.config import SEED, TRAIN, CHANGE_CLASSES
from mangrove.datasets import ChangePatchDataset
from mangrove.metrics import cd_metrics
from mangrove.models import MODEL_REGISTRY, DUAL_INPUT_MODELS
from mangrove import paths
from mangrove.paths import RESULTS_ROOT

TEMPORAL_MODELS = set(DUAL_INPUT_MODELS)
# Non-temporal models receive the concatenated feature stack
# [t1 (9), t2 (9), |t1-t2| (9)] = 27 channels.
STATIC_IN_CH = 9 * 3
TEMPORAL_IN_CH = 9
# Train region (spatially independent split with 2 km buffer).
TRAIN_REGION = "vietnam"
# Independent test regions (geographically disjoint from TRAIN_REGION).
TEST_REGIONS = ("beilunhe", "zhanjiang")


def normalize(x):
    return x / 10000.0


def weighted_ce_det(logits, y, weight):
    """Deterministic equivalent of nn.CrossEntropyLoss(weight=weight,
    reduction='mean'): sum over pixels of w[y] * (-log p[y]) divided by the
    sum of the per-pixel target weights. Verified against the fused kernel
    to float32 precision (2026-09-23). Built from log_softmax + gather +
    sum, which all have deterministic CUDA implementations (the fused
    nll_loss2d kernel that the standard CE uses does not)."""
    logp = torch.log_softmax(logits, dim=1)
    l0 = -logp.gather(1, y.unsqueeze(1)).squeeze(1)
    w_pix = weight[y]
    return (l0 * w_pix).sum() / w_pix.sum()


def dice_loss(proba, y, smooth=1.0):
    """Soft multi-class dice loss (proba = logits, y = (B,H,W) int64).

    Complements weighted CE: drives per-class overlap directly, which is a
    much stronger gradient for the rare change classes than CE alone when the
    background dominates (~86% of train pixels).
    """
    p = proba.softmax(1)
    y_one = nn.functional.one_hot(y, p.shape[1]).float().permute(0, 3, 1, 2)
    inter = (p * y_one).sum((2, 3))
    denom = p.sum((2, 3)) + y_one.sum((2, 3))
    return 1.0 - (2.0 * inter + smooth).sum() / (denom + smooth).sum()


def build_dataloaders(model_name, num_workers=2, batch_size=None, cw_mode="inv",
                      train_idx=None, val_idx=None, gen=None, val_split="val"):
    # num_workers=0 on this host: forked workers COW the parent's dataset arrays
    # + per-thread oneDNN workspaces, and 2 workers added ~9 GB on top of the
    # parent's ~10 GB peak (global OOM kill, 2026-09-20). In-process loading
    # costs ~1 GB.
    # train_idx/val_idx (nested-CV): restrict the vietnam train/val
    # splits to explicit patch-index sets (spatial column folds).
    concat = model_name not in TEMPORAL_MODELS
    train_ds = ChangePatchDataset(TRAIN_REGION, "train", augment=True, concat=concat,
                                  cw_mode=cw_mode,
                                  idx_filter=set(train_idx) if train_idx else None)
    # Nested CV: the inner-validation fold is a subset of the
    # TRAIN zone, so the val dataset is built on the same base split.
    val_ds = ChangePatchDataset(TRAIN_REGION, val_split, augment=False, concat=concat,
                                idx_filter=set(val_idx) if val_idx else None)
    test_ds = ChangePatchDataset(TRAIN_REGION, "test", augment=False, concat=concat)
    bs = batch_size or TRAIN["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers,
                          pin_memory=False, generator=gen)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl, train_ds


@torch.no_grad()
def evaluate(model, dl, temporal=False, device=torch.device("cpu")):
    model.eval()
    ys, ps = [], []
    for batch in dl:
        if temporal:
            x1, x2, y, _ = batch
            x1 = normalize(x1).to(device)
            x2 = normalize(x2).to(device)
            proba = model(x1, x2)
        else:
            x, y, _ = batch
            x = normalize(x).to(device)
            proba = model(x)
        ps.append(proba.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return cd_metrics(y, p, 4, CHANGE_CLASSES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    ap.add_argument("--batch-size", type=int, default=TRAIN["batch_size"])
    ap.add_argument("--lr", type=float, default=TRAIN["lr"])
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num-workers", type=int, default=TRAIN["num_workers"])
    # Default = 'sqrt' after the 2026-09-21 probe (unet, matched 25 epochs):
    # 'inv' predicted 9.1% change where the reference has 1.3% (commission 0.91,
    # within-region mIoU 0.388); 'sqrt' predicted 1.6% (commission 0.72,
    # mIoU 0.440) and lifted Beilunhe mIoU 0.424 -> 0.543.
    ap.add_argument("--cw-mode", choices=["inv", "sqrt", "none"], default="sqrt",
                    help="class-weight scheme for the imbalanced 4-class problem")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="run seed; artifacts are suffixed _s<seed> so repeats "
                         "of the same model do not overwrite each other")
    ap.add_argument("--threads", type=int, default=None,
                    help="cap torch intra-op threads (CPU only)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="auto = use the RTX 4080 GPU when torch sees it "
                         "(2026-09-21: project also runs remotely on Fu4080-server)")
    ap.add_argument("--deterministic", action="store_true",
                    help="fully deterministic CUDA training "
                         "(cudnn deterministic + deterministic algos + cuBLAS "
                         "workspace config) to bound the run-to-run noise floor")
    ap.add_argument("--tag", default=None,
                    help="suffix artifact names (e.g. det, nested) so repeated "
                         "configurations never overwrite the main-sweep files")
    ap.add_argument("--train-idx", default=None,
                    help="JSON file: patch indices for the train split (nested CV)")
    ap.add_argument("--val-idx", default=None,
                    help="JSON file: patch indices for the val split (nested CV)")
    ap.add_argument("--val-split", default="val", choices=["train", "val"],
                    help="base split the --val-idx filter is applied to "
                         "(nested CV inner folds live inside the train zone)")
    ap.add_argument("--skip-test-eval", action="store_true",
                    help="skip within-region + independent test evals (nested CV "
                         "inner runs must not touch any held-out data)")
    args = ap.parse_args()
    if args.threads:
        # OMP env must be set before the thread pool spawns (first forward):
        # torch.set_num_threads alone left oneDNN at 32 threads -> OOM.
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        torch.set_num_threads(args.threads)

    # Device: 2026-09-21 the project also runs on Fu4080-server (RTX 4080 Laptop,
    # 12 GB) with a CUDA torch build; ~20x faster per epoch than the 4-core CPU
    # box the pipeline originally targeted.
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    model_name = args.model
    base_name = model_name if args.tag is None else f"{model_name}_{args.tag}"
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)  # deterministic shuffle order across reruns
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Deterministic CE (the fused nll_loss2d kernel used by
        # nn.CrossEntropyLoss has no deterministic CUDA implementation in this
        # torch build; weighted_ce_det is its bit-equivalent replacement).
        # warn_only is still needed for AdaptiveMaxPool2d.backward (CBAM),
        # whose max tie-breaking has no deterministic CUDA kernel; the
        # same-seed repeat (tag "detc") measures the residual empirically.
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("deterministic: use_deterministic_algorithms(warn_only) + deterministic CE")
    ckpt = paths.ckpt_path(base_name, seed)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    log_path = paths.log_path(base_name, seed)
    metrics_path = paths.metrics_path(base_name, seed)

    train_idx = json.loads(Path(args.train_idx).read_text()) if args.train_idx else None
    val_idx = json.loads(Path(args.val_idx).read_text()) if args.val_idx else None
    train_dl, val_dl, test_dl, train_ds = build_dataloaders(
        model_name, num_workers=args.num_workers, batch_size=args.batch_size,
        cw_mode=args.cw_mode, train_idx=train_idx, val_idx=val_idx, gen=gen,
        val_split=args.val_split)
    is_temporal = model_name in TEMPORAL_MODELS
    in_ch = TEMPORAL_IN_CH if is_temporal else STATIC_IN_CH
    model = MODEL_REGISTRY[model_name](in_ch=in_ch, out_ch=4)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={model_name} in_ch={in_ch} params={n_params/1e6:.2f}M "
          f"device={device} train={len(train_dl)} val={len(val_dl)} test={len(test_dl)}",
          flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=TRAIN["weight_decay"])
    # Weighted CE + dice: CE alone let unet/unet_cbam collapse to all-background
    # (bg = 85.8% of train pixels); dice pushes per-class overlap (2026-09-20).
    w_ce = train_ds.class_weights.to(device)
    if args.deterministic:
        crit_ce = lambda out, y: weighted_ce_det(out, y, w_ce)
    else:
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
        # Early-stop on macro IoU: OA saturates on the background class
        # (92%+ of pixels) long before the minority change classes are learned.
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
    if args.skip_test_eval:
        # Nested-CV inner runs: no held-out data may be touched at all.
        test_m, indep = None, {}
    else:
        # Within-region held-out test (vietnam test zone, 2 km buffer-isolated).
        test_m = evaluate(model, test_dl, temporal=is_temporal, device=device)
        # Spatially independent test regions (beilunhe, zhanjiang).
        from mangrove.paths import SPLITS_ROOT
        indep = {}
        for region in TEST_REGIONS:
            if not (SPLITS_ROOT / f"{region}_test.json").exists():
                print(f"WARNING: {region}_test.json missing, skipping independent eval")
                indep[region] = None
                continue
            ds = ChangePatchDataset(region, "test", augment=False, concat=not is_temporal)
            dl = DataLoader(ds, batch_size=TRAIN["batch_size"], shuffle=False,
                            num_workers=args.num_workers)
            indep[region] = evaluate(model, dl, temporal=is_temporal, device=device)
    res = {
        "model": model_name,
        "tag": args.tag,
        "params_M": round(n_params / 1e6, 3),
        "seed": seed,
        "in_ch": in_ch,
        "hyperparams": {k: args.__dict__[k] for k in
                        ("epochs", "batch_size", "lr", "patience", "seed", "cw_mode",
                         "deterministic", "tag")},
        "val": val_m,
        "test_within_region": test_m,
        "independent_test": indep,
        "elapsed_h": round((time.time() - t0) / 3600, 2),
    }
    metrics_path.write_text(json.dumps(res, indent=2))
    def _oa(m):
        return f"{m['OA']:.4f}" if m else "n/a"
    print(f"val OA={val_m['OA']:.4f} within OA={_oa(test_m)} "
          f"beilunhe OA={_oa(indep.get('beilunhe'))} zhanjiang OA={_oa(indep.get('zhanjiang'))} "
          f"({res['elapsed_h']} h)")


if __name__ == "__main__":
    main()
