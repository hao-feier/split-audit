"""Why do 4 architectures predict exactly zero change pixels on the isolated
test set of split k3 with run-seed 2026, while training looked healthy?

Training was fine (no divergence, best val F1 ~0.45) yet test change-F1 was
exactly 0 with predicted-change = 0. That is the signature of an argmax
threshold effect, not a training collapse: if the change probability mass sits
just below the stable classes everywhere, argmax returns stable for every pixel
while the ranking is still informative. This script measures that margin.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import ap_pr_auc  # noqa: E402
from ap_pr_auc import DEVICE, build_loader, normalize  # noqa: E402

# temporal_unet_gn is a dual-input model too but is absent from ap_pr_auc's set,
# which silently builds it with in_ch=27 and fails to load its 9-channel weights.
ap_pr_auc.TEMPORAL_MODELS.add("temporal_unet_gn")
from mangrove.paths import SPLITS_ROOT  # noqa: E402

R = os.path.join(ROOT, "results", "uniform", "random_split_k3")
CKPT = os.path.join(R, "checkpoints")
CASES = [("unet", 2026), ("unet", 2027), ("unet_cbam", 2026),
         ("unet_cbam_msa", 2026), ("unet_msa", 2026),
         ("temporal_unet_gn", 2026)]


@torch.no_grad()
def probe(model_name, seed, arm):
    temporal = model_name in ap_pr_auc.TEMPORAL_MODELS
    in_ch = 9 if temporal else 27
    from mangrove.models import MODEL_REGISTRY
    model = MODEL_REGISTRY[model_name](in_ch=in_ch, out_ch=4).to(DEVICE)
    ck = os.path.join(CKPT, f"{model_name}_s{seed}.pt")
    model.load_state_dict(torch.load(ck, map_location=DEVICE))
    model.eval()
    dl = build_loader("vietnam", "test" if arm == "isolated" else 3, model_name)
    pmax, pmarg, n, n_gt = [], [], 0, 0
    psc, ybin = [], []
    for batch in dl:
        if temporal:
            x1, x2, y, _ = batch
            out = model(normalize(x1).to(DEVICE), normalize(x2).to(DEVICE))
        else:
            x, y, _ = batch
            out = model(normalize(x).to(DEVICE))
        sm = torch.softmax(out, 1)
        psc.append((sm[:, 1] + sm[:, 2]).cpu().numpy().ravel())
        ybin.append(np.isin(y.numpy(), [1, 2]).astype(int).ravel())
        n += int(sm.shape[0]) * sm.shape[2] * sm.shape[3]
    pmax = np.concatenate(psc)
    ybin = np.concatenate(ybin)
    return dict(
        n_px=int(n),
        n_change_ref=int(ybin.sum()),
        pred_change_px=int(((pmax > 0.5) & (ybin == 0)).sum()
                           + ((pmax > 0.5) & (ybin == 1)).sum()),
        px_pch_ge_0_1=int((pmax >= 0.10).sum()),
        px_pch_ge_0_01=int((pmax >= 0.01).sum()),
        mean_pch=round(float(pmax.mean()), 6),
        max_pch=round(float(pmax.max()), 6),
        p99_pch=round(float(np.percentile(pmax, 99)), 6),
    )


if __name__ == "__main__":
    print(f"{'run':28s} {'arm':9s} {'pred_ch':>8s} {'P(ch)>=.10':>10s} "
          f"{'P(ch)>=.01':>10s} {'mean':>8s} {'p99':>8s} {'max':>8s}")
    for m, s in CASES:
        for arm in ("leaky", "isolated"):
            try:
                r = probe(m, s, arm)
            except Exception as e:
                print(f"{m}/s{s:<20d} {arm:9s} ERROR {type(e).__name__}: {e}")
                continue
            print(f"{m+'/s'+str(s):28s} {arm:9s} {r['pred_change_px']:8d} "
                  f"{r['px_pch_ge_0_1']:10d} {r['px_pch_ge_0_01']:10d} "
                  f"{r['mean_pch']:8.5f} {r['p99_pch']:8.5f} {r['max_pch']:8.5f}")
