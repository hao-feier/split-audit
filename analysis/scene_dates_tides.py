"""Acquisition dates + tidal-phase metadata.

Lists every scene used in the dry-season composites with its acquisition
date, and characterises the tidal state of each date (lunar age ->
spring/neap classification, M2-dominated semidiurnal regime of the Gulf of
Tonkin / Beibu Gulf). A per-pixel MEDIAN composite over several scenes on
different days averages over different tidal states, which mitigates (does
not eliminate) water-level-driven spectral variation in the mangrove fringes.
Reporting the dates and the range of tidal states covered lets the reader
assess this directly.

Output: results/scene_dates_tides.json
"""
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mangrove.config import REGIONS

SYNODIC = 29.530588853  # days
NEW_MOON_REF = datetime(2000, 1, 6, 18, 14)  # known new moon (UTC)


def lunar_age(d):
    return ((d - NEW_MOON_REF).total_seconds() / 86400.0) % SYNODIC


def spring_neap(age):
    """Spring tides near new (0) and full (14.77); neaps at quarters."""
    for spring in (0.0, SYNODIC / 2):
        dist = min(abs(age - spring), SYNODIC - abs(age - spring))
        if dist <= 3.0:
            return "spring"
        if dist <= 6.0:
            return "mid"
    return "neap"


def main():
    sel = json.loads((ROOT / "results" / "scene_selection.json").read_text())
    out = {"regions": {}, "summary": {}}
    all_dates = []
    for region, epochs in sel.items():
        out["regions"][region] = {}
        for epoch, tiles in epochs.items():
            rec = {}
            for tile, v in tiles.items():
                if not v.get("ok"):
                    continue
                for scene in v["scenes"]:
                    m = re.match(r"S2[AB]_([0-9A-Z]{5})_(\d{8})_", scene)
                    date = datetime.strptime(m.group(2), "%Y%m%d")
                    age = lunar_age(date)
                    rec[scene] = {
                        "date": date.strftime("%Y-%m-%d"),
                        "lunar_age_days": round(age, 2),
                        "tidal_state": spring_neap(age),
                        "cf_region": v["median_cf_region"],
                        "cf_mangrove": v["median_cf_mangrove"],
                    }
                    all_dates.append((region, epoch, date, age))
            out["regions"][region][epoch] = rec
    # per epoch: range of tidal states covered by the selected scenes
    for region, epochs in out["regions"].items():
        for epoch in list(epochs.keys()):
            rec = out["regions"][region][epoch]
            states = sorted({v["tidal_state"] for v in rec.values()})
            span_days = (max(datetime.strptime(v["date"], "%Y-%m-%d") for v in rec.values())
                         - min(datetime.strptime(v["date"], "%Y-%m-%d") for v in rec.values())
                         ).days
            out["regions"][region][epoch + "_tidal_summary"] = {
                "n_scenes": len(rec),
                "date_range": f"{min(v['date'] for v in rec.values())} .. "
                              f"{max(v['date'] for v in rec.values())}",
                "scene_span_days": span_days,
                "tidal_states_covered": states,
            }
    (ROOT / "results" / "scene_dates_tides.json").write_text(json.dumps(out, indent=1))
    print("saved results/scene_dates_tides.json")
    for region in out["regions"]:
        for k, v in out["regions"][region].items():
            if k.endswith("tidal_summary"):
                print(region, k, v)
    _ = all_dates
    # epoch-pair overlap: how many scenes per epoch per region
    for region, epochs in sel.items():
        for epoch, tiles in epochs.items():
            n = sum(len(v["scenes"]) for v in tiles.values() if v.get("ok"))
            print(region, epoch, "scenes:", n)


if __name__ == "__main__":
    main()
