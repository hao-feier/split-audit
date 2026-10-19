"""Dump full-map (all 76 patches) per-patch argmax predictions for design D.

Derived from analysis/dump_predictions.py (same forward path as training and
as the canonical test dump), per Feier/specs/2026-10-17-d-overlap-cell-design.md §5.

Gate 1 (spec §7): the 15 canonical checkpoints (5 models x seeds {2026,2027,2028})
must reproduce the on-disk canonical test predictions pixel-exactly
(np.array_equal with results/predictions/{model}_s{seed}_vietnam.npy). Inference
runs on the full region exactly once per checkpoint and is kept in memory
(~20 MB uint8 each); if ANY checkpoint mismatches, abort before writing anything.

Only after the gate passes, saves
results/d_predictions/{model}_s{seed}_vietnam_all76.npy (uint8, (76,512,512)).

Run AFTER the training batch (GPU contention). Memory guard: region cache
~1.4 GB + batch 2; no grad; per-batch release; requires MemAvailable > 4 GB.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.datasets import load_region, load_split
from mangrove.models import MODEL_REGISTRY, pick_device, DUAL_INPUT_MODELS
from mangrove.paths import RESULTS_ROOT

REGION = "vietnam"
MODELS = ["unet", "unet_cbam", "unet_cbam_msa", "deeplabv3plus", "temporal_unet"]
SEEDS = [2026, 2027, 2028]


def mem_available_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        return None
    return None


def build_inputs(data, is_temporal):
    if is_temporal:
        a = torch.from_numpy((data["t1_feat"].transpose(0, 3, 1, 2) / 10000.0).copy())
        b = torch.from_numpy((data["t2_feat"].transpose(0, 3, 1, 2) / 10000.0).copy())
        return a, b
    t1 = data["t1_feat"].transpose(0, 3, 1, 2)
    t2 = data["t2_feat"].transpose(0, 3, 1, 2)
    n, c, h, w = t1.shape
    x = np.empty((n, 3 * c, h, w), dtype=t1.dtype)
    x[:, :c] = t1
    x[:, c:2 * c] = t2
    np.subtract(t1, t2, out=x[:, 2 * c:])
    np.abs(x[:, 2 * c:], out=x[:, 2 * c:])
    x /= 10000.0
    return torch.from_numpy(x), None


def predict_all(net, a, b, device, batch):
    preds = []
    with torch.no_grad():
        for i in range(0, len(a), batch):
            if b is None:
                out = net(a[i:i + batch].to(device))
            else:
                out = net(a[i:i + batch].to(device), b[i:i + batch].to(device))
            preds.append(out.argmax(1).cpu().numpy())
            del out
    return np.concatenate(preds).astype(np.uint8)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patches", choices=["all", "test"], default="all")
    args = ap.parse_args()

    avail = mem_available_gb()
    if avail is None:
        print("[mem] /proc/meminfo unreadable; RAM guard skipped")
    else:
        print(f"[mem] MemAvailable = {avail:.1f} GB (guard: > 4 GB)")
        if avail <= 4.0:
            print("[mem] ABORT: insufficient free RAM")
            sys.exit(1)

    device = pick_device()
    print(f"[dev] {device}")
    data = load_region(REGION)
    idx_test = np.asarray(load_split(REGION, "test"))
    print(f"[data] {REGION}: t1_feat {data['t1_feat'].shape}, test patches {len(idx_test)}")

    pred_dir = RESULTS_ROOT / "predictions"
    out_dir = RESULTS_ROOT / "d_predictions"
    n_total = len(args.models) * len(args.seeds)
    arrays = {}
    failures = []
    k = 0
    for model in args.models:
        is_temporal = model in DUAL_INPUT_MODELS
        in_ch = 9 if is_temporal else 27
        for seed in args.seeds:
            k += 1
            ck = RESULTS_ROOT / "checkpoints" / f"{model}_s{seed}.pt"
            if not ck.exists():
                print(f"[abort] missing checkpoint {ck}")
                sys.exit(1)
            t0 = time.time()
            net = MODEL_REGISTRY[model](in_ch=in_ch, out_ch=4).to(device)
            net.load_state_dict(torch.load(ck, map_location="cpu"))
            net.eval()
            a, b = build_inputs(data, is_temporal)
            arr = predict_all(net, a, b, device, args.batch)
            del net, a, b
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            arrays[(model, seed)] = arr
            ref_path = pred_dir / f"{model}_s{seed}_{REGION}.npy"
            if not ref_path.exists():
                failures.append((model, seed, f"reference missing: {ref_path}"))
                print(f"[gate1 {k}/{n_total}] {model} s{seed}: FAIL (reference missing)")
                continue
            ref = np.load(ref_path)
            got = arr[idx_test]
            if got.shape != ref.shape:
                failures.append((model, seed, f"shape {got.shape} != ref {ref.shape}"))
                print(f"[gate1 {k}/{n_total}] {model} s{seed}: FAIL (shape)")
            elif not np.array_equal(got, ref):
                n_bad = int((got != ref).sum())
                failures.append((model, seed, f"{n_bad} pixel mismatches"))
                print(f"[gate1 {k}/{n_total}] {model} s{seed}: FAIL ({n_bad} mismatched px)")
            else:
                dt = time.time() - t0
                print(f"[gate1 {k}/{n_total}] {model} s{seed}: PASS ({len(idx_test)}/{len(idx_test)} identical, {dt:.0f}s)")

    if failures:
        print("\n[GATE 1 FAILED] nothing written:")
        for model, seed, msg in failures:
            print(f"  - {model} s{seed}: {msg}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(arrays)
    for i, ((model, seed), arr) in enumerate(arrays.items(), 1):
        if args.patches == "all":
            out = arr
            name = f"{model}_s{seed}_{REGION}_all76.npy"
        else:
            out = arr[idx_test]
            name = f"{model}_s{seed}_{REGION}_test.npy"
        np.save(out_dir / name, out)
        print(f"[save {i}/{n}] {name} {out.shape} {out.dtype}")
    print("[done] gate 1 PASS, all dumps written")


if __name__ == "__main__":
    main()
