"""Assemble interpretation.json from per-montage judgments.

Inputs:
  gt_tool/montages/manifest.json   montage key -> list of point ids (row-major)
  gt_tool/interp_raw.json          {montage_key: {"<cell index>": 0|1|2|3, ...}}
      written by the interpreter as they work through the montages.

Cell indices exist in two conventions and both are read:
  per-montage  every montage numbers its own cells 000..len(ids)-1
  global       cell indices run continuously across montages in manifest
               order, so montage 001 starts at len(ids of montage 000)
The convention is decided per montage from the observed index range, so a
partially filled file is never silently truncated to its first montage.

Output: gt_tool/interpretation.json in the workbench format
  {"tool": ..., "labels": {point_id: {"t1": v, "t2": v, "ts": ...}}}
so it can be merged by merge_gt_interpretation.py (2 -> null) or imported
into gt_tool/index.html for a QC pass.

Usage: python analysis/gt_interpret_assemble.py [--report]
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MONT = ROOT / "gt_tool" / "montages" / "manifest.json"
RAW = ROOT / "gt_tool" / "interp_raw.json"
OUT = ROOT / "gt_tool" / "interpretation.json"


def _cells(cell_lab, n_ids, offset):
    """Map one montage's cell labels onto point indices under either convention.

    `offset` is the number of ids in all preceding montages, so the global
    convention is `i = j - offset` and the per-montage convention is `i = j`.
    Both are scored on every montage and the one that lands more cells in
    range wins; scoring both is required because applying only the global rule
    drops every per-montage file from montage 001 onward (and vice versa).
    Ties go to per-montage, which is the convention the workbench writes.
    """
    per_montage, globl = {}, {}
    for key, val in cell_lab.items():
        try:
            j = int(key)
        except ValueError:
            continue
        if 0 <= j < n_ids:
            per_montage[j] = val
        i = j - offset
        if 0 <= i < n_ids:
            globl[i] = val
    return per_montage if len(per_montage) >= len(globl) else globl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    manifest = json.loads(MONT.read_text())
    raw = json.loads(RAW.read_text()) if RAW.exists() else {}

    labels = {}
    n_cells = 0
    offset = 0
    for key, ids in manifest.items():
        region, ep, k = key.split("/")
        cell_lab = raw.get(key, {})
        for i, v in _cells(cell_lab, len(ids), offset).items():
            pid = ids[i]
            n_cells += 1
            labels.setdefault(pid, {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            labels[pid][ep] = int(v)
        offset += len(ids)
    complete = {pid: l for pid, l in labels.items() if "t1" in l and "t2" in l}

    out = {"tool": "gt_workbench", "version": 1,
           "exported": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_labeled": len(complete), "labels": complete}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT.name}: {len(complete)} complete points "
          f"({n_cells} cell judgments total)")

    if args.report or True:
        from collections import Counter
        cov = Counter()
        offset = 0
        for key, ids in manifest.items():
            region, ep, k = key.split("/")
            done = len(_cells(raw.get(key, {}), len(ids), offset))
            offset += len(ids)
            prev_d, prev_n = cov.get((region, ep), (0, 0))
            cov[(region, ep)] = (prev_d + done, prev_n + len(ids))
        total = tot = 0
        for region in ["vietnam", "beilunhe", "zhanjiang"]:
            for ep in ["t1", "t2"]:
                d, n = cov.get((region, ep), (0, 0))
                total += d; tot += n
                print(f"  {region}/{ep}: {d}/{n}")
        print(f"  overall: {total}/{tot} cells, {len(complete)}/"
              f"{tot // 2} points complete (both epochs judged)")


if __name__ == "__main__":
    main()
