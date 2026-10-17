"""The F1 = 0 runs are a decision-rule artifact, not training failures.

collapse_reconcile.py flagged 9 runs with change-F1 < 0.05. Inspecting the
saved checkpoints shows that 7 of them predict EXACTLY zero change pixels under
argmax, yet they rank change pixels far better than chance (ROC-AUC 0.78-0.97,
AP 0.12-0.61 at prevalence 0.034). The reason is visible in the softmax: those
models put 18-24% of their mean probability mass on the change classes but never
let any single pixel's change probability exceed the stable classes, so argmax
returns stable everywhere.

This is the paper's own thesis demonstrated from the inside: under extreme class
imbalance a hard argmax can report F1 = 0 for a model that is genuinely
informative, so an F1 gap between split protocols is not by itself evidence
about model capability. It also means the symmetric collapse filter removes
runs whose exclusion is a metric artifact rather than a model-validity problem.
"""
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import ap_pr_auc  # noqa: E402
from mangrove.models import DUAL_INPUT_MODELS, MODEL_REGISTRY  # noqa: E402

# use the authoritative dual-input set: ap_pr_auc hardcodes a subset
ap_pr_auc.TEMPORAL_MODELS.clear()
ap_pr_auc.TEMPORAL_MODELS.update(DUAL_INPUT_MODELS)
DEVICE = ap_pr_auc.DEVICE
ROC = json.load(open(os.path.join(ROOT, "results", "prevalence_roc.json")))
REC = json.load(open(os.path.join(ROOT, "results", "collapse_reconciliation.json")))
OUT = os.path.join(ROOT, "results", "argmax_floor_effect.json")

# softmax margin probe: the 7 all-negative runs + 2 reference runs
PROBE = [("unet", 3, 2026), ("unet_cbam", 3, 2026), ("unet_cbam_msa", 3, 2026),
         ("temporal_unet_gn", 3, 2026), ("changeformer", 1, 2026),
         ("changeformer", 2, 2026), ("changeformer", 2, 2028),
         ("changeformer", 3, 2026),
         ("unet", 3, 2027), ("unet_msa", 3, 2026), ("unet_cbam", 3, 2027)]
ID = r"random_split_k(\d+)/(.+)/s(\d+)"


def parse(x):
    m = re.match(ID, x)
    return (m.group(2), int(m.group(1)), int(m.group(3)))


@torch.no_grad()
def softmax_profile(model_name, split, seed):
    temporal = model_name in DUAL_INPUT_MODELS
    in_ch = 9 if temporal else 27
    model = MODEL_REGISTRY[model_name](in_ch=in_ch, out_ch=4).to(DEVICE)
    ck = os.path.join(ROOT, "results", "uniform", f"random_split_k{split}",
                      "checkpoints", f"{model_name}_s{seed}.pt")
    model.load_state_dict(torch.load(ck, map_location=DEVICE))
    model.eval()
    dl = ap_pr_auc.build_loader("vietnam", split, model_name)
    pch = []
    for batch in dl:
        if temporal:
            x1, x2, _, _ = batch
            out = model(ap_pr_auc.normalize(x1).to(DEVICE),
                        ap_pr_auc.normalize(x2).to(DEVICE))
        else:
            x, _, _ = batch
            out = model(ap_pr_auc.normalize(x).to(DEVICE))
        sm = torch.softmax(out, 1)
        pch.append((sm[:, 1] + sm[:, 2]).cpu().numpy().ravel())
    p = np.concatenate(pch)
    return dict(mean_p_change=round(float(p.mean()), 5),
                max_p_change=round(float(p.max()), 5),
                p99_p_change=round(float(np.percentile(p, 99)), 5),
                px_above_0p5=int((p > 0.5).sum()),
                px_above_0p3=int((p > 0.3).sum()),
                n_px=int(p.size))


def main():
    allneg = {parse(x) for x in REC["all_negative_collapse"]}
    under = {parse(x) for x in REC["low_F1_but_predicts_change"]}
    roc = {(c["model"], c["split"], c["seed"]): c["leaky"]
           for c in ROC["pairs"]}

    probes, zero_f1 = [], []
    for model, split, seed in PROBE:
        k = (model, split, seed)
        try:
            prof = softmax_profile(model, split, seed)
        except Exception as e:                      # pragma: no cover
            prof = {"error": f"{type(e).__name__}: {e}"}
        prof.update(model=model, split=split, seed=seed,
                    arm="interleaved(random)",
                    cls=("all-negative" if k in allneg else
                         "high-precision-underfit" if k in under else "healthy"))
        if "roc_auc" in roc.get(k, {}):
            prof["roc_auc"] = round(roc[k]["roc_auc"], 4)
            prof["ap"] = round(roc[k]["ap"], 4)
            prof["prevalence"] = round(roc[k]["prev"], 5)
        probes.append(prof)

    # aggregate: every run in the sweep whose on-disk F1 is exactly 0
    for c in ROC["pairs"]:
        k = (c["model"], c["split"], c["seed"])
        cm = None
        f = os.path.join(ROOT, "results", "uniform", f"random_split_k{c['split']}",
                         f"metrics_{c['model']}_s{c['seed']}.json")
        if os.path.exists(f):
            d = json.load(open(f))
            cm = d.get("random_test", {}).get("confusion_matrix")
        if cm is None:
            continue
        a = np.asarray(cm)
        tp = sum(a[i, j] for i in (1, 2) for j in (1, 2))
        if tp == 0:
            zero_f1.append(dict(model=c["model"], split=c["split"], seed=c["seed"],
                                roc_auc=round(c["leaky"]["roc_auc"], 4),
                                ap=round(c["leaky"]["ap"], 4),
                                prevalence=round(c["leaky"]["prev"], 5)))
    rest = [c["leaky"] for c in ROC["pairs"]
            if (c["model"], c["split"], c["seed"]) not in allneg | under]

    out = {
        "claim": ("A hard argmax can report change-F1 exactly 0 for a model that "
                  "ranks change pixels far above chance; the collapse set is a "
                  "decision-rule artifact, not a training failure."),
        "n_zero_tp_runs": len(zero_f1),
        "zero_tp_runs": zero_f1,
        "zero_tp_roc_auc": {
            "mean": round(float(np.mean([r["roc_auc"] for r in zero_f1])), 4),
            "min": round(float(np.min([r["roc_auc"] for r in zero_f1])), 4),
            "max": round(float(np.max([r["roc_auc"] for r in zero_f1])), 4)},
        "zero_tp_ap": {
            "mean": round(float(np.mean([r["ap"] for r in zero_f1])), 4),
            "min": round(float(np.min([r["ap"] for r in zero_f1])), 4),
            "max": round(float(np.max([r["ap"] for r in zero_f1])), 4)},
        "noncollapsed_roc_auc_mean": round(float(np.mean(
            [r["roc_auc"] for r in rest])), 4),
        "ap_over_prevalence_zero_tp": round(float(np.mean(
            [r["ap"] / r["prevalence"] for r in zero_f1])), 1),
        "softmax_profiles": probes,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    z = out["zero_tp_runs"]
    print(f"runs with zero change-class TP : {len(z)}")
    print(f"  their ROC-AUC  : mean {out['zero_tp_roc_auc']['mean']} "
          f"(range {out['zero_tp_roc_auc']['min']}-{out['zero_tp_roc_auc']['max']})")
    print(f"  their AP       : mean {out['zero_tp_ap']['mean']} "
          f"(range {out['zero_tp_ap']['min']}-{out['zero_tp_ap']['max']}), "
          f"{out['ap_over_prevalence_zero_tp']}x the random baseline")
    print(f"  non-collapsed : ROC-AUC mean {out['noncollapsed_roc_auc_mean']}")
    print("\nsoftmax profile (interleaved arm):")
    print(f"{'run':30s} {'cls':26s} {'meanP(ch)':>10s} {'maxP(ch)':>9s} {'>0.5':>7s}")
    for p in probes:
        if "error" in p:
            print(f"{p['model']}/s{p['seed']}: {p['error']}")
            continue
        print(f"{p['model']+'/k'+str(p['split'])+'/s'+str(p['seed']):30s} "
              f"{p['cls']:26s} {p['mean_p_change']:10.5f} {p['max_p_change']:9.5f} "
              f"{p['px_above_0p5']:7d}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
