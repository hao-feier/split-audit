"""Global config: seeds, bands, regions, training hyperparameters.

Everything stochastic goes through SEED. Region boxes are WGS84 lon/lat
(checked against GMW extent in exp0).
"""

SEED = 2026

# S2 L2A bands to pull from AWS (band id -> native resolution, m)
S2_BANDS = {
    "B02": 10,   # Blue
    "B03": 10,   # Green
    "B04": 10,   # Red
    "B08": 10,   # NIR
    "B11": 20,   # SWIR1
    "B12": 20,   # SWIR2
    "SCL": 20,   # Scene classification (cloud masking only, not a feature)
}

# Feature stack order used by all models (6 reflectance + 3 indices)
FEATURE_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NDWI", "MNDWI"]
RGB_VIS = ["B04", "B03", "B02"]  # true color for visualization

# Study regions (WGS84).
# Region A (train): Vietnam Red River Delta mangroves, UTM zone 48.
# Region B (test): Beilunhe estuary, Guangxi, UTM zone 49 (~1600 km from A).
# Region C (test): Zhanjiang, Guangdong, UTM zone 49 (secondary independent site).
# Region D (test): Dongzhai Bay, Hainan, UTM zone 49 (~350 km from C).
# All boxes verified against GMW v4.1.12 extent (exp0 region survey).
REGIONS = {
    "vietnam": {
        "center": (106.75, 20.85),
        "box": (106.5, 20.5, 107.0, 21.2),
        "role": "train",
    },
    "beilunhe": {
        "center": (108.4, 21.45),
        "box": (108.1, 21.15, 108.7, 21.75),
        "role": "test",
    },
    "zhanjiang": {
        "center": (110.45, 21.25),
        "box": (110.05, 21.0, 110.85, 21.5),
        "role": "test",
    },
    "dongzhai": {
        "center": (110.6, 19.95),
        "box": (110.53, 19.83, 110.70, 20.10),
        "role": "test",
    },
}

# Time epochs: dry-season windows (low tide / minimum water extent).
# China south coast: Nov-Jan. We composite each window with a cloud filter.
EPOCHS = {
    "t1": {"year": 2019, "window": ("2019-12-01", "2020-01-31")},
    "t2": {"year": 2024, "window": ("2024-12-01", "2025-01-31")},
}

# Change classes (GMW 2019 vs 2024): 0 stable mangrove, 1 loss, 2 gain, 3 stable non-mangrove
CHANGE_CLASSES = ["stable_mangrove", "loss", "gain", "stable_background"]

# GMW extent class values (v4.1.12 raster stack): 0 background, 1 mangrove
GMW_MANGROVE = 1

# Patching
PATCH_SIZE = 512        # px at 10 m -> 5.12 km
PATCH_STEP = 256        # px stride (50% overlap)
MIN_MANGROVE_FRAC = 0.05   # keep patch if >=5% mangrove (t1 or t2)
MAX_MANGROVE_FRAC = 0.85   # skip water/urban-dominated patches

# Training (CPU-tuned). epochs raised 40->100 on 2026-09-20: with
# weighted-CE+dice loss and patience 15, models need more room before
# early stop (previous 40-epoch runs collapsed to all-background).
TRAIN = dict(
    # 512x512 patches: ~1.35 GiB activations per batch; batch 8 peaks at
    # ~12.8 GiB and OOMs this 15.9 GiB host (measured, 2026-09-20).
    batch_size=2,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    # forked workers COW the ~1.5 GiB dataset arrays -> global OOM (2026-09-20)
    num_workers=0,
)

# Spatial split: buffer between train and test zones (meters)
BUFFER_M = 2000

def region_dir(name: str):
    return PROCESSED_ROOT / name
