"""Exp1a: fetch Sentinel-2 L2A from AWS open data (STAC query + S3 download).

For each region x epoch, queries the element84 STAC API, selects scenes
spread across the dry-season window, and downloads B02/B03/B04/B08/B11/B12/SCL.
Outputs: data/raw/s2/<scene_id>/*.tif + data/raw/s2/manifest.json
Idempotent: skips files already present.
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mangrove.config import REGIONS, EPOCHS, S2_BANDS
from mangrove.paths import S2_ROOT, LOGS_ROOT

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
BAND_ASSET = {
    "B02": "blue", "B03": "green", "B04": "red",
    "B08": "nir", "B11": "swir16", "B12": "swir22", "SCL": "scl",
}


def stac_query(bbox, dt_start, dt_end, limit=60):
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox),
        "datetime": f"{dt_start}T00:00:00Z/{dt_end}T23:59:59Z",
        "limit": limit,
    }
    for attempt in range(3):
        try:
            r = requests.post(STAC_URL, json=payload, timeout=60)
            r.raise_for_status()
            return r.json().get("features", [])
        except Exception as e:
            print(f"  STAC retry {attempt+1}: {e}")
            time.sleep(5 * (attempt + 1))
    return []


def select_scenes(features, n):
    """Pick n scenes: spread across the window AND across S2 tiles (>=1 per tile)."""
    feats = [f for f in features if f.get("assets", {}).get("scl")]
    if not feats:
        return []
    feats.sort(key=lambda f: f["properties"].get("datetime", ""))
    by_tile = {}
    for f in feats:
        by_tile.setdefault(f["id"].split("_")[1], []).append(f)
    # main tiles = those with the most scenes (corner tiles have few)
    main = sorted(by_tile.items(), key=lambda kv: -len(kv[1]))
    keep_tiles = [t for t, v in main if len(v) >= max(2, n // 2) or len(main) == 1]
    if not keep_tiles:
        keep_tiles = [main[0][0]]
    per_tile = max(1, n // len(keep_tiles))
    out = []
    for t in keep_tiles:
        v = by_tile[t]
        idx = [round(i * (len(v) - 1) / max(per_tile - 1, 1)) for i in range(per_tile)]
        for i in sorted(set(idx)):
            out.append(v[i])
    # alternate satellites within the pick
    a = [f for f in out if f["id"].startswith("S2A")]
    b = [f for f in out if f["id"].startswith("S2B")]
    merged, ai, bi = [], 0, 0
    while ai < len(a) or bi < len(b):
        if ai < len(a):
            merged.append(a[ai]); ai += 1
        if bi < len(b):
            merged.append(b[bi]); bi += 1
    return merged[:max(n, len(out))]


def download_band(url, dest):
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    for attempt in range(3):
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code == 200:
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            tmp.rename(dest)
            return True
        time.sleep(5 * (attempt + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=list(REGIONS))
    ap.add_argument("--epoch", required=True, choices=list(EPOCHS))
    ap.add_argument("--n-scenes", type=int, default=4)
    args = ap.parse_args()

    cfg = REGIONS[args.region]
    ep = EPOCHS[args.epoch]
    bbox = cfg["box"]
    print(f"[{args.region}/{args.epoch}] querying STAC {ep['window']}")
    feats = stac_query(bbox, *ep["window"])
    print(f"  {len(feats)} candidate scenes")
    selected = select_scenes(feats, args.n_scenes)
    print(f"  selected: {[f['id'] for f in selected]}")

    manifest = S2_ROOT / "manifest.json"
    mani = json.loads(manifest.read_text()) if manifest.exists() else {}
    downloaded, failed = 0, 0

    for f in selected:
        sid = f["id"]
        d = S2_ROOT / sid
        d.mkdir(parents=True, exist_ok=True)
        url = f["assets"]["scl"]["href"]
        base = url.rsplit("/", 1)[0]
        print(f"  {sid} ({f['properties'].get('datetime', '')[:10]})")
        jobs = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            for band, asset in BAND_ASSET.items():
                fn = f"{band}.tif"
                dest = d / fn
                url_b = f"{base}/{fn}"
                jobs.append(ex.submit(download_band, url_b, dest))
            for j in as_completed(jobs):
                if j.result():
                    downloaded += 1
                else:
                    failed += 1
        mani.setdefault(args.region, {}).setdefault(args.epoch, []).append({
            "scene": sid,
            "datetime": f["properties"].get("datetime"),
            "bands": list(BAND_ASSET.keys()),
        })

    manifest.write_text(json.dumps(mani, indent=2))
    print(f"  done: {downloaded} bands ok, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
