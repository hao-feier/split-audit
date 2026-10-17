"""PCC (post-classification comparison) baseline.

Classical bi-temporal CD: classify each date independently into
mangrove / non-mangrove, then change = (class_t1 != class_t2). Direction
(loss vs gain) is not recoverable, so PCC is scored on the binary
changed/unchanged collapse (the same collapse used for the other models'
F1ch / IoU / omission / commission).

Per-date classifier: plain U-Net (2 classes) on the 9-band single-date
feature stack, trained on vietnam train patches with the GMW presence
labels (presence derived from the change labels: t1 = {0,1}, t2 = {0,2};
verified pixel-identical to labels_t1.tif / labels_t2.tif, 2026-09-23).
Same recipe as the main sweep (100 epochs, patience 15, bs 8, weighted CE
--cw-mode sqrt + dice, 3 seeds).

Outputs:
  results/checkpoints/pcc_t1_s<seed>.pt, pcc_t2_s<seed>.pt
  results/train_log_pcc_t<e>_s<seed>.jsonl
  results/metrics_pcc_s<seed>.json      (binary-change metrics per eval set)
  results/predictions/pcc_s<seed>_<region>.npy (+ _labels.npy), binary 0/1
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mangrove.config import SEED, TRAIN
from mangrove.datasets import load_region
from mangrove.models import UNet, pick_device
from mangrove import paths
from mangrove.paths import PROCESSED_ROOT, RESULTS_ROOT, SPLITS_ROOT

EPOCHES = ("t1", "t2")
# presence class per epoch in the 4-class change scheme:
# 0 stable_mangrove (both), 1 loss (t1 only), 2 gain (t2 only), 3 stable_bg (none)
PRESENCE = {"t1": (0, 1), "t2": (0, 2)}
TRAIN_REGION = "vietnam"
EVAL_REGIONS = ("beilunhe", "zhanjiang")


def dice_loss2(proba, y, smooth=1.0):
    p = proba.softmax(1)
    y_one = nn.functional.one_hot(y, 2).float().permute(0, 3, 1, 2)
    inter = (p * y_one).sum((2, 3))
    denom = p.sum((2, 3)) + y_one.sum((2, 3))
    return 1.0 - (2.0 * inter + smooth).sum() / (denom + smooth).sum()


class SingleDateDataset(Dataset):
    """One date of the vietnam train patches: (9-band features, presence 0/1)."""

    def __init__(self, epoch, idx, augment=True):
        data = load_region(TRAIN_REGION)
        self.feat = data[f"{epoch}_feat"]
        ch = data["change"]
        keep = PRESENCE[epoch]
        self.pres = ((ch == keep[0]) | (ch == keep[1])).astype(np.uint8)
        self.idx = list(idx)
        self.augment = augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        i = self.idx[i]
        x = self.feat[i].astype(np.float32)
        y = self.pres[i].astype(np.int64)
        if self.augment:
            k = np.random.randint(4)
            x, y = np.rot90(x, k).copy(), np.rot90(y, k).copy()
            if np.random.rand() > 0.5:
                x, y = x[:, ::-1].copy(), y[:, ::-1].copy()
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()),
                torch.from_numpy(y.copy()))


def binary_metrics(yb, pb):
    """yb, pb: binary changed/unchanged (0/1) pixel arrays (flattened)."""
    tp = int(((yb == 1) & (pb == 1)).sum())
    fn = int(((yb == 1) & (pb == 0)).sum())
    fp = int(((yb == 0) & (pb == 1)).sum())
    tn = int(((yb == 0) & (pb == 0)).sum())
    n = tp + fn + fp + tn
    f1 = 2 * tp / max(2 * tp + fn + fp, 1)
    iou = tp / max(tp + fn + fp, 1)
    return {"OA": (tp + tn) / max(n, 1), "F1_change": f1, "IoU_change": iou,
            "precision_change": tp / max(tp + fp, 1), "recall_change": tp / max(tp + fn, 1),
            "Omission": fn / max(tp + fn, 1), "Commission": fp / max(tp + fp, 1),
            "cm": [[tn, fp], [fn, tp]]}


@torch.no_grad()
def evaluate(model, feat, pres, idx, device):
    model.eval()
    ps = []
    x = torch.from_numpy((feat.transpose(0, 3, 1, 2) / 10000.0).copy())
    for i in range(0, len(x), 8):
        ps.append(model(x[i:i + 8].to(device)).argmax(1).cpu().numpy() == 1)
    p = np.concatenate(ps)[idx]
    y = pres[idx].astype(bool)
    return binary_metrics(y.ravel(), p.ravel()), p, y


def train_epoch(epoch, seed, args, device):
    torch.manual_seed(seed); np.random.seed(seed)
    train_idx = json.loads((SPLITS_ROOT / f"{TRAIN_REGION}_train.json").read_text())
    val_idx = json.loads((SPLITS_ROOT / f"{TRAIN_REGION}_val.json").read_text())
    data = load_region(TRAIN_REGION)
    ch = data["change"]
    keep = PRESENCE[epoch]
    pres = ((ch == keep[0]) | (ch == keep[1])).astype(np.uint8)
    ds_tr = SingleDateDataset(epoch, train_idx, augment=True)
    pix = ds_tr.pres[np.asarray(ds_tr.idx)].ravel()
    counts = np.bincount(pix, minlength=2)
    freq = np.maximum(counts, 1) / max(counts.sum(), 1)
    cw = getattr(args, "cw_mode", "sqrt")
    if cw == "none":
        w = torch.ones(2, dtype=torch.float32)
    elif cw == "sqrt":
        w = torch.tensor(1.0 / (2 * np.sqrt(freq)), dtype=torch.float32)
    elif cw == "inv":
        w = torch.tensor(1.0 / (2 * freq), dtype=torch.float32)
    else:
        raise ValueError(f"unknown cw_mode {cw}")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(SingleDateDataset(epoch, val_idx, augment=False),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)

    net = UNet(in_ch=9, out_ch=2).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=TRAIN["weight_decay"])
    crit = nn.CrossEntropyLoss(weight=w.to(device))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_loss, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    tag = {"none": "_none", "inv": "_inv"}.get(cw, "")
    log_path = paths.log_path(f"pcc_{epoch}{tag}", seed)
    with open(log_path, "a") as fh:
        for e in range(1, args.epochs + 1):
            net.train()
            run_loss, n = 0.0, 0
            for x, y in dl_tr:
                x, y = x.to(device) / 10000.0, y.to(device)
                opt.zero_grad()
                out = net(x)
                loss = crit(out, y) + dice_loss2(out, y)
                loss.backward(); opt.step()
                run_loss += float(loss.detach()) * y.size(0); n += y.size(0)
            sched.step()
            vm, _, _ = evaluate(net, data[f"{epoch}_feat"], pres, val_idx, device)
            val_loss = 1.0 - vm["F1_change"]
            rec = {"epoch": e, "train_loss": run_loss / n, "val_F1ch": vm["F1_change"],
                   "val_OA": vm["OA"], "elapsed_min": round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec), flush=True)
            fh.write(json.dumps(rec) + "\n")
            if val_loss < best_loss - 1e-4:
                best_loss, bad = val_loss, 0
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"early stop at epoch {e}"); break
    torch.save(best_state, paths.ckpt_path(f"pcc_{epoch}{tag}", seed))
    return best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[2026, 2027, 2028])
    ap.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=TRAIN["lr"])
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--cw-mode", choices=["inv", "sqrt", "none"], default="sqrt")
    args = ap.parse_args()
    tag = {"none": "_none", "inv": "_inv"}.get(args.cw_mode, "")
    device = pick_device()

    for seed in args.seeds:
        states = {}
        for epoch in EPOCHES:
            ck = paths.ckpt_path(f"pcc_{epoch}{tag}", seed)
            if ck.exists():
                states[epoch] = torch.load(ck, map_location="cpu")
                print(f"skip pcc {epoch} s{seed} (checkpoint exists)")
            else:
                states[epoch] = train_epoch(epoch, seed, args, device)
        # evaluate per seed: within-region test + independent regions
        res = {"model": "pcc", "seed": seed, "binary": True,
               "hyperparams": vars(args), "val": None, "test_within_region": None,
               "independent_test": {}}
        data_v = load_region(TRAIN_REGION)
        ch = data_v["change"]
        pres1 = ((ch == 0) | (ch == 1)).astype(np.uint8)
        pres2 = ((ch == 0) | (ch == 2)).astype(np.uint8)
        for region, split, data, p1, p2 in (
                ("within", "test", data_v, pres1, pres2),
                ("beilunhe", "test", load_region("beilunhe"), None, None),
                ("zhanjiang", "test", load_region("zhanjiang"), None, None)):
            if p1 is None:
                c = data["change"]
                p1 = ((c == 0) | (c == 1)).astype(np.uint8)
                p2 = ((c == 0) | (c == 2)).astype(np.uint8)
            rname = "vietnam" if region == "within" else region
            idx = np.asarray(json.loads((SPLITS_ROOT / f"{rname}_{split}.json").read_text()))
            net1 = UNet(in_ch=9, out_ch=2).to(device)
            net1.load_state_dict(states["t1"]); net1.eval()
            net2 = UNet(in_ch=9, out_ch=2).to(device)
            net2.load_state_dict(states["t2"]); net2.eval()
            m1, pred1, _ = evaluate(net1, data["t1_feat"], p1, idx, device)
            m2, pred2, _ = evaluate(net2, data["t2_feat"], p2, idx, device)
            changed_pred = (pred1 != pred2).astype(np.uint8)
            changed_true = ((data["change"] == 1) | (data["change"] == 2)).astype(np.uint8)[idx]
            key = "test_within_region" if region == "within" else f"independent_test.{region}"
            blk = binary_metrics(changed_true.ravel(), changed_pred.ravel())
            blk["n_patches"] = int(len(idx))
            if region == "within":
                res["test_within_region"] = blk
            else:
                res["independent_test"][region] = blk
            out_dir = RESULTS_ROOT / "predictions"
            out_dir.mkdir(parents=True, exist_ok=True)
            name = region if region != "within" else "vietnam"
            np.save(out_dir / f"pcc{tag}_s{seed}_{name}.npy", changed_pred)
            np.save(out_dir / f"pcc{tag}_s{seed}_{name}_labels.npy", changed_true)
            print(f"pcc s{seed} {region}: F1ch={blk['F1_change']:.4f} "
                  f"omission={blk['Omission']:.4f} commission={blk['Commission']:.4f}")
        (RESULTS_ROOT / f"metrics_pcc{tag}_s{seed}.json").write_text(json.dumps(res, indent=2))
        print(f"done pcc seed {seed}")


if __name__ == "__main__":
    main()
