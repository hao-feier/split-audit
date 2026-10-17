"""Buffer-width sensitivity of the within-region (honest) numbers.

Aggregates the buffer re-runs (results/buf{1,2,4}/: U-Net and U-Net+CBAM+MSA,
3 seeds) and attaches the effective val-test edge distance from
results/split_audit.json. The 256-px grid snaps requested buffer widths to
achievable distances, so B=1 and B=2 are the identical split (near-exact
replicate runs) and B=4 is the only materially different setting (5.12 km
gap, 4 extra buffer patches). The independent test regions (Beilunhe,
Zhanjiang) are geographically fixed and are NOT part of this sensitivity.

Output: results/buffer_sensitivity.json
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from aggregate_results import binary_change  # noqa: E402

MODELS = ["unet", "unet_cbam_msa"]
SEEDS = ["2026", "2027", "2028"]
BS = ["1", "2", "4"]


def collect(B, model, seed):
    f = ROOT / "results" / f"buf{B}" / f"metrics_{model}_s{seed}.json"
    d = json.loads(f.read_text())
    blk = d["test_within_region"]
    bc = binary_change(blk["confusion_matrix"])
    return {"OA": blk["OA"], "mIoU": blk["mIoU"], "F1ch": bc["F1_change"],
            "omission": bc["Omission"], "commission": bc["Commission"]}


def main():
    audit = {}
    af = ROOT / "results" / "split_audit.json"
    if af.exists():
        audit = json.loads(af.read_text())["splits"]
    out = {"per_B": {}}
    for B in BS:
        row = {"model": {}}
        for m in MODELS:
            rows = [collect(B, m, s) for s in SEEDS]
            row["model"][m] = {
                k: {"mean": st.mean([r[k] for r in rows]),
                    "std": st.pstdev([r[k] for r in rows]),
                    "per_seed": [round(r[k], 4) for r in rows]}
                for k in rows[0]}
        gap = audit.get(f"buf{B}", {}).get("val_to_test_gap_km")
        row["effective_val_test_gap_km"] = gap
        row["n_buffer_patches"] = audit.get(f"buf{B}", {}).get("n", {}).get("buffer")
        out["per_B"][f"B={B}"] = row
    out["identical_sets"] = audit.get("identical_sets", [])
    out["note"] = ("B=1 and B=2 produce the identical split (grid snap, see "
                   "split_audit.json): their runs are near-exact replicates, "
                   "differing only by cuDNN process non-determinism (~0.05% "
                   "of pixels). B=4 is the only materially different setting.")
    (ROOT / "results" / "buffer_sensitivity.json").write_text(json.dumps(out, indent=2))
    # compact markdown
    lines = ["| B (km) | eff. val-test gap (km) | U-Net F1ch | U-Net mIoU | proposed F1ch | proposed mIoU |",
             "|---|---|---|---|---|---|"]
    for B in BS:
        r = out["per_B"][f"B={B}"]
        a, b = r["model"]["unet"], r["model"]["unet_cbam_msa"]
        lines.append(f"| {B} | {r['effective_val_test_gap_km']} | "
                     f"{a['F1ch']['mean']:.3f}±{a['F1ch']['std']:.3f} | {a['mIoU']['mean']:.3f}±{a['mIoU']['std']:.3f} | "
                     f"{b['F1ch']['mean']:.3f}±{b['F1ch']['std']:.3f} | {b['mIoU']['mean']:.3f}±{b['mIoU']['std']:.3f} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
