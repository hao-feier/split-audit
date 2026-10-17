r"""E3-C: a size-matched, overlap-free test of residual proximity.

The manuscript states that a zero-overlap interleaved split is unattainable at
40/10/10, so the overlap-free control had to be run as E3-B at reduced scale
(block-interleaved, 5-6 test patches, 3 architectures), and reports the sign of
the contrast but explicitly not its magnitude.

That obstruction is a property of the patch geometry, and the geometry is a
choice. When PATCH_STEP == PATCH_SIZE the pool is disjoint, so a *random*
assignment is already overlap-free. This script runs the missing full-size
control at 128 px:

  arm "scattered" : random 40/10/10. Zero train/test pixel overlap by
                    construction. Test patches interleaved through the region.
  arm "separated" : same 40 training patches; test/val chosen to maximise the
                    minimum edge distance to the training set.

Both arms therefore have zero pixel overlap and identical sizes, and differ only
in where the evaluation patches sit. No buffer band is discarded, because a
disjoint pool has no shared footprints for a buffer to suppress.

Outputs results/e3c/metrics_<model>_d<draw>_<arm>_s<seed>.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.config import CHANGE_CLASSES, TRAIN  # noqa: E402
from mangrove.datasets import ChangePatchDataset  # noqa: E402
from mangrove.metrics import cd_metrics  # noqa: E402
sys.path.insert(0, str(ROOT / "analysis"))
from aggregate_results import binary_change  # noqa: E402
from mangrove.models import DUAL_INPUT_MODELS, MODEL_REGISTRY  # noqa: E402
from mangrove.paths import PROCESSED_ROOT, RESULTS_ROOT, SPLITS_ROOT  # noqa

REGION = "vietnam_p128"
OUT = RESULTS_ROOT / "e3c"
STATIC_IN_CH, DUAL_IN_CH = 9 * 3, 9


def split_file(split, draw, arm):
    return SPLITS_ROOT / f"{REGION}_{split}_e3c{draw}_{arm}.json"


def class_weights(idx, change, mode):
    pix = change[np.asarray(idx)].ravel()
    cnt = np.maximum(np.bincount(pix, minlength=4), 1)
    freq = cnt / cnt.sum()
    if mode == "inv":
        w = 1.0 / (4 * freq)
    elif mode == "sqrt":
        w = 1.0 / (4 * np.sqrt(freq))
    elif mode == "none":
        w = np.ones(4)
    else:
        raise ValueError(mode)
    return torch.tensor(w, dtype=torch.float32)


def make_loader(split, draw, arm, model_name, augment, cw_mode, bs):
    concat = model_name not in DUAL_INPUT_MODELS
    ds = ChangePatchDataset(REGION, split, augment=augment, concat=concat,
                            cw_mode=cw_mode)
    ds.idx = json.loads(split_file(split, draw, arm).read_text())
    if split == "train":
        ds.class_weights = class_weights(ds.idx, ds.change, cw_mode)
    return DataLoader(ds, batch_size=bs, shuffle=augment,
                      num_workers=TRAIN["num_workers"], pin_memory=False), ds


def normalize(x):
    return x / 10000.0


@torch.no_grad()
def evaluate(model, dl, dual, device, want_scores=False):
    """4-class metrics from argmax, plus optionally the per-pixel change score.

    The paper's primary metric is change-F1 from a hard argmax. Section~sec:rand
    shows that argmax can return exactly zero change pixels for a model that is
    genuinely informative, so every headline contrast is also reported under
    prevalence-free metrics. Those need the per-pixel P(loss)+P(gain), which a
    confusion matrix cannot supply, so the score is captured here.
    """
    model.eval()
    ys, ps, sc = [], [], []
    for batch in dl:
        if dual:
            x1, x2, y, _ = batch
            proba = model(normalize(x1).to(device), normalize(x2).to(device))
        else:
            x, y, _ = batch
            proba = model(normalize(x).to(device))
        if want_scores:
            sm = torch.softmax(proba, 1)
            sc.append((sm[:, 1] + sm[:, 2]).cpu().numpy().ravel())
        ps.append(proba.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    out = cd_metrics(np.concatenate(ys), np.concatenate(ps), 4, CHANGE_CLASSES)
    if want_scores:
        yb = np.isin(np.concatenate(ys), [1, 2]).astype(int).ravel()
        out["_scores"] = (np.concatenate(sc), yb)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--draw", type=int, required=True)
    ap.add_argument("--arm", required=True, choices=["scattered", "separated"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=TRAIN["lr"])
    ap.add_argument("--weight-decay", type=float, default=TRAIN["weight_decay"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cw-mode", default="none", choices=["inv", "sqrt", "none"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    dual = args.model in DUAL_INPUT_MODELS
    in_ch = DUAL_IN_CH if dual else STATIC_IN_CH
    bs = args.batch_size if not dual else 2

    tr_dl, _ = make_loader("train", args.draw, args.arm, args.model, True,
                           args.cw_mode, bs)
    va_dl, _ = make_loader("val", args.draw, args.arm, args.model, False,
                           args.cw_mode, bs)
    te_dl, _ = make_loader("test", args.draw, args.arm, args.model, False,
                           args.cw_mode, bs)

    model = MODEL_REGISTRY[args.model](in_ch=in_ch, out_ch=4).to(device)
    w = tr_dl.dataset.class_weights.to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for batch in tr_dl:
            if dual:
                x1, x2, y, _ = batch
                out = model(normalize(x1).to(device), normalize(x2).to(device))
            else:
                x, y, _ = batch
                out = model(normalize(x).to(device))
            y = y.to(device)
            loss = crit(out, y)
            sm = torch.softmax(out, 1)
            p = sm[:, 1:3].sum(1)
            t = ((y == 1) | (y == 2)).float()
            inter = (p * t).sum()
            dice = 1 - (2 * inter + 1.0) / (p.sum() + t.sum() + 1.0)
            (loss + dice).backward()
            opt.step()
            opt.zero_grad()
            tot += float(loss.item())
        sched.step()
        vm = evaluate(model, va_dl, dual, device)
        score = vm["mIoU"]
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Diagnostic: is the test score approaching the resubstitution
    # level? Reported per run so the two arms can be compared on it.
    trm = binary_change(evaluate(model, tr_dl, dual, device)["confusion_matrix"])
    tm4 = evaluate(model, te_dl, dual, device, want_scores=True)
    _sc, _yb = tm4.pop("_scores")
    from sklearn.metrics import average_precision_score, roc_auc_score
    tscore = {
        "roc_auc": round(float(roc_auc_score(_yb, _sc)), 4),
        "ap": round(float(average_precision_score(_yb, _sc)), 4),
        "prevalence": round(float(_yb.mean()), 5),
        "max_p_change": round(float(_sc.max()), 4),
        "mean_p_change": round(float(_sc.mean()), 5),
    }
    # the paper's primary metric is the BINARY change F1, which requires
    # collapsing the 4x4 confusion matrix; cd_metrics does not return it
    tm = binary_change(tm4["confusion_matrix"])
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"model": args.model, "draw": args.draw, "arm": args.arm,
           "seed": args.seed, "region": REGION,
           "n_train": len(tr_dl.dataset.idx), "n_val": len(va_dl.dataset.idx),
           "n_test": len(te_dl.dataset.idx), "epochs_run": ep + 1,
           "best_val_miou": round(float(best), 4),
           "test": {k: (round(float(v), 4) if isinstance(v, (int, float))
                        else v) for k, v in tm.items()},
           "train_F1_change": round(float(trm["F1_change"]), 4),
           "test_scores": tscore,
           "test_4class": {"OA": round(float(tm4["OA"]), 4),
                           "mIoU": round(float(tm4["mIoU"]), 4)}}
    p = OUT / f"metrics_{args.model}_d{args.draw}_{args.arm}_s{args.seed}.json"
    p.write_text(json.dumps(rec, indent=2))
    print(f"{args.model} d{args.draw} {args.arm} s{args.seed}: "
          f"F1ch={tm['F1_change']:.4f} roc={tscore['roc_auc']:.4f} "
          f"ap={tscore['ap']:.4f} ep={ep+1} -> {p.name}")


if __name__ == "__main__":
    main()