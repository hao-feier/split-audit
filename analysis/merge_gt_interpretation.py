"""Merge interpreter output (gt_tool/interpretation.json) into the sample GeoJSONs.

The interpretation workbench (gt_tool/index.html, built by make_gt_tool.py)
exports labels as {id: {t1, t2}} with values 0 (no mangrove) / 1 (mangrove) /
2 (unsure). This script writes them into
data/groundtruth/<region>_samples.geojson as interp_t1 / interp_t2, mapping
2 -> null so the Olofsson estimator (oloofsson_area_accuracy.py), which
excludes points with a missing epoch, drops unsure points automatically.
Raw values stay in interpretation.json.

Usage:
  python analysis/merge_gt_interpretation.py                 # default file
  python analysis/merge_gt_interpretation.py --file gt_tool/interpretation.json
  python analysis/merge_gt_interpretation.py --dry-run

Originals are backed up to data/groundtruth/backup/ before the first merge.
Writes results/gt_interpretation_status.json (counts + stratum cross-tab).
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGIONS = ["vietnam", "beilunhe", "zhanjiang"]
GT = ROOT / "data" / "groundtruth"
CLASS_NAMES = ["stable_mangrove", "loss", "gain", "stable_background"]
INTERP_NAMES = ["stable_mangrove", "loss", "gain", "stable_background", "unsure"]


def interp_class(t1, t2):
    if t1 == 2 or t2 == 2:
        return "unsure"
    return {(1, 1): 0, (1, 0): 1, (0, 1): 2, (0, 0): 3}[(t1, t2)]


def load_labels(path):
    d = json.loads(path.read_text())
    labels = d.get("labels", d)
    out = {}
    for pid, l in labels.items():
        t1, t2 = l.get("t1"), l.get("t2")
        if t1 is None or t2 is None:
            continue  # workbench only exports complete points, but be safe
        out[pid] = (int(t1), int(t2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "gt_tool" / "interpretation.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    labels = load_labels(Path(args.file))
    if not labels:
        sys.exit("no complete labels found in " + args.file)

    status = {"merged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "source": str(args.file), "n_labeled": len(labels), "regions": {}}
    (GT / "backup").mkdir(exist_ok=True)
    for region in REGIONS:
        f = GT / f"{region}_samples.geojson"
        geo = json.loads(f.read_text())
        n_tot = n_new = n_usable = n_unsure = 0
        tab = {g: [0, 0, 0, 0, 0] for g in range(4)}
        for feat in geo["features"]:
            p = feat["properties"]
            n_tot += 1
            if p["id"] not in labels:
                continue
            t1, t2 = labels[p["id"]]
            n_new += 1
            ic = interp_class(t1, t2)
            tab[p["gmw_class"]][4 if ic == "unsure" else ic] += 1
            if ic == "unsure":
                n_unsure += 1
                p["interp_t1"] = None if t1 == 2 else t1
                p["interp_t2"] = None if t2 == 2 else t2
            else:
                n_usable += 1
                p["interp_t1"] = t1
                p["interp_t2"] = t2
        if n_new and not args.dry_run:
            bak = GT / "backup" / f"{region}_samples.geojson.bak"
            if not bak.exists():
                shutil.copy2(f, bak)
            f.write_text(json.dumps(geo, indent=1))
        status["regions"][region] = {
            "n_points": n_tot, "n_interpreted": n_new, "n_usable": n_usable,
            "n_unsure": n_unsure,
            "stratum_x_interp": {str(g): {"n_stratum": sum(tab[g]), **dict(zip(INTERP_NAMES, tab[g]))}
                                 for g in range(4)},
        }
        print(f"{region}: {n_new}/{n_tot} interpreted, {n_usable} usable, {n_unsure} unsure")

    n_us = sum(r["n_usable"] for r in status["regions"].values())
    n_un = sum(r["n_unsure"] for r in status["regions"].values())
    print(f"total: {len(labels)} labeled -> {n_us} usable for the Olofsson estimator, {n_un} unsure")
    if args.dry_run:
        print("dry run: geojsons not written")
        return
    (ROOT / "results" / "gt_interpretation_status.json").write_text(
        json.dumps(status, indent=1))
    print("wrote results/gt_interpretation_status.json (backups in data/groundtruth/backup/)")


if __name__ == "__main__":
    main()
