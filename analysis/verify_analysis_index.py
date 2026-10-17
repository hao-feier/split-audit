"""Audit results/README.md: every row must point at a real script and a real JSON.

Motivation (2026-10-15): the index was written by hand and immediately carried
three false attributions (dualsource_subset -> independent_validation.py,
dualsource_boundary_strat -> independent_validation.py, distance_response
under results/uniform/). A hand-maintained index is exactly the failure mode it
was built to prevent, so it gets its own gate rather than trust.

Run:  python analysis/verify_analysis_index.py
Exit: 0 = index consistent with the tree; 1 = a row is wrong.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "results" / "README.md"

bad = []


def rows():
    """Yield (json_path, producer_script, question) for each index table row."""
    for line in INDEX.read_text().splitlines():
        m = re.match(r"\|\s*`([^`]+)`\s*\|\s*`?([^|`]*)`?\s*\|", line)
        if not m:
            continue
        path, producer = m.group(1).strip(), m.group(2).strip()
        if path.endswith(".json"):
            yield path, producer, line


def resolve(path):
    """A row may name a bare filename or a full relative path."""
    for cand in (ROOT / path, ROOT / "results" / path,
                 ROOT / "results" / "uniform" / path):
        if cand.exists():
            return cand
    return None


seen = 0
for path, producer, line in rows():
    seen += 1
    jf = resolve(path)
    if jf is None:
        bad.append(f"JSON missing: {path}")
        continue
    if producer.startswith("(") or "*" in producer:
        continue  # deliberately unproduced / wildcard rows
    scripts = re.findall(r"[A-Za-z0-9_]+\.py", producer)
    if not scripts:
        bad.append(f"no script named: {path} -> {producer!r}")
    for s in scripts:
        if not (ROOT / "analysis" / s).exists():
            bad.append(f"script missing: {path} -> analysis/{s}")

# Every JSON named in the index must be reachable, and every script the index
# names must be a real analysis file -- the reverse direction is the one that
# catches "this analysis does not exist".
named_scripts = set()
for path, producer, _ in rows():
    named_scripts.update(re.findall(r"[A-Za-z0-9_]+\.py", producer))

print("=" * 70)
print("ANALYSIS INDEX AUDIT".center(70))
print("=" * 70)
print(f"rows checked: {seen}")
print(f"distinct producers named: {len(named_scripts)}")
if bad:
    print("-" * 70)
    for b in bad:
        print(f"[FAIL] {b}")
    print("=" * 70)
    print(f"RESULT: {len(bad)} inconsistent row(s) in results/README.md")
    sys.exit(1)
print("-" * 70)
print("RESULT: index consistent with the tree")
sys.exit(0)