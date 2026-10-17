"""Tile-by-tile Sentinel-2 L2A processing baseline verification.

Queries the Element 84 STAC API for every Sentinel-2 scene in the
project manifest (data/raw/s2/manifest.json) and reports the
``s2:processing_baseline`` and ``s2:reflectance_conversion_factor``
for each.

This verifies (or refutes) the paper's claim that "the AWS COG
collection appears to serve both epochs on the same scale" by
showing the original processing baseline versions.

Output: results/s2_processing_baseline.json
"""
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
STAC_URL = (
    "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items"
)


def query_scene(scene_id: str) -> dict:
    """Query the STAC API for a single scene and extract key metadata."""
    try:
        r = requests.get(STAC_URL, params={"ids": scene_id}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("numberMatched", 0) == 0:
            return {"scene": scene_id, "error": "not found in STAC"}
        f = data["features"][0]
        props = f["properties"]
        return {
            "scene": scene_id,
            "datetime": props.get("datetime"),
            "created": props.get("created"),
            "platform": props.get("platform"),
            "cloud_cover_pct": props.get("eo:cloud_cover"),
            "processing_baseline": props.get("s2:processing_baseline"),
            "reflectance_conversion_factor": props.get(
                "s2:reflectance_conversion_factor"
            ),
            "mgrs": props.get("grid:code"),
        }
    except Exception as e:
        return {"scene": scene_id, "error": str(e)}


def main():
    manifest = json.loads((ROOT / "data" / "raw" / "s2" / "manifest.json").read_text())
    out = {"config": {"source": "Element 84 STAC API", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, "regions": {}}

    for region in ["vietnam", "beilunhe", "zhanjiang"]:
        if region not in manifest:
            continue
        out["regions"][region] = {}
        for epoch in ["t1", "t2"]:
            scenes = manifest[region].get(epoch, [])
            results = []
            for s in scenes:
                scene_id = s["scene"]
                results.append(query_scene(scene_id))
                time.sleep(0.3)
            out["regions"][region][epoch] = results

    # Summary
    baselines = set()
    for region_data in out["regions"].values():
        for epoch_data in region_data.values():
            for r in epoch_data:
                if "processing_baseline" in r:
                    baselines.add(r["processing_baseline"])
    out["summary"] = {
        "unique_processing_baselines": sorted(baselines),
        "n_scenes": sum(
            len(ep) for reg in out["regions"].values() for ep in reg.values()
        ),
    }

    out_path = ROOT / "results" / "s2_processing_baseline.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved to {out_path.relative_to(ROOT)}")
    print(f"Unique processing baselines: {out['summary']['unique_processing_baselines']}")
    print(f"Total scenes: {out['summary']['n_scenes']}")
    for region, epochs in out["regions"].items():
        for epoch, scenes in epochs.items():
            pb = set(r.get("processing_baseline") for r in scenes if "processing_baseline" in r)
            print(f"  {region} {epoch}: baselines={pb}")


if __name__ == "__main__":
    main()
