"""Patch-level PyTorch datasets for change-detection training.

Data layout per region (see experiments/exp1_data_prep.py):
  data/processed/<region>/
    t1/features.npz   (N, H, W, C) float32  -- t1 feature stack
    t1/labels.npz     (N, H, W) uint8       -- GMW presence t1
    t2/features.npz   (N, H, W, C) float32
    t2/labels.npz     (N, H, W) uint8
    change_labels.npz (N, H, W) uint8       -- 0 stable-mng / 1 loss / 2 gain / 3 stable-bg
    metadata.json     -- geo refs, split ids
Splits: data/splits/<region>_train.json etc. (list of patch indices).
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .paths import PROCESSED_ROOT, SPLITS_ROOT


def _load_npz(path, key):
    """Load a compressed .npz and return the underlying array (not the lazy NpzFile)."""
    with np.load(path) as fh:
        return fh[key]


_REGION_CACHE = {}


def load_region(region):
    """Load all arrays for a region. Returns dict with keys t1_feat, t2_feat, change, meta.

    Each features.npz stores key ``features`` (N,H,W,C float32); change_labels.npz
    stores key ``change`` (N,H,W int8). Arrays are materialised eagerly so that
    downstream fancy indexing works directly on numpy arrays.

    Cached per region: train/val/test datasets for the same region share the
    underlying arrays (~1.4 GB per copy on this data — 3 independent copies
    were OOM-killing the training process, 2026-09-20).
    """
    if region in _REGION_CACHE:
        return _REGION_CACHE[region]
    base = PROCESSED_ROOT / region
    out = {}
    out["t1_feat"] = _load_npz(base / "t1" / "features.npz", "features")
    out["t2_feat"] = _load_npz(base / "t2" / "features.npz", "features")
    out["change"] = _load_npz(base / "change_labels.npz", "change")
    out["meta"] = json.loads((base / "metadata.json").read_text())
    _REGION_CACHE[region] = out
    return out


def load_split(region, name):
    """name in {train, val, test}. Returns list of patch indices."""
    path = SPLITS_ROOT / f"{region}_{name}.json"
    return json.loads(path.read_text())


class ChangePatchDataset(Dataset):
    """Static change-detection dataset: features = [t1 stack, t2 stack, |t1-t2|]."""

    def __init__(self, region, split, augment=True, idx_filter=None, concat=True,
                 cw_mode="inv"):
        """cw_mode: how the (extremely imbalanced) class weights are derived.
        'inv'  = inverse frequency  w_c = 1/(4 f_c)  (2026-09-20 default; change
                 classes get ~50x the background weight, which later showed up as
                 3-5x change over-prediction)
        'sqrt' = milder w_c = 1/(4 sqrt(f_c)) (~7x change/background ratio)
        'none' = uniform weights
        """
        data = load_region(region)
        self.idx = load_split(region, split)
        self.concat = concat
        if idx_filter is not None:
            self.idx = [i for i in self.idx if i in idx_filter]
        self.t1 = data["t1_feat"]
        self.t2 = data["t2_feat"]
        self.change = data["change"]
        self.augment = augment and split == "train"
        # class balance weights computed on the training split
        if split == "train":
            pix = self.change[np.asarray(self.idx)].ravel()
            counts = np.bincount(pix, minlength=4)
            counts = np.maximum(counts, 1)
            freq = counts / counts.sum()
            if cw_mode == "inv":
                w = 1.0 / (4 * freq)
            elif cw_mode == "sqrt":
                w = 1.0 / (4 * np.sqrt(freq))
            elif cw_mode == "none":
                w = np.ones(4)
            else:
                raise ValueError(f"unknown cw_mode {cw_mode}")
            self.class_weights = torch.tensor(w, dtype=torch.float32)
        else:
            self.class_weights = torch.ones(4, dtype=torch.float32)

    def __len__(self):
        return len(self.idx)

    def _aug_pair(self, x1, x2, y):
        """Random 90-deg rotation + horizontal flip applied consistently."""
        k = np.random.randint(4)
        x1, x2, y = np.rot90(x1, k).copy(), np.rot90(x2, k).copy(), np.rot90(y, k).copy()
        if np.random.rand() > 0.5:
            x1, x2, y = x1[:, ::-1].copy(), x2[:, ::-1].copy(), y[:, ::-1].copy()
        return x1, x2, y

    def __getitem__(self, i):
        i = self.idx[i]
        x1 = self.t1[i].astype(np.float32)
        x2 = self.t2[i].astype(np.float32)
        y = self.change[i].astype(np.int64)
        if self.augment:
            x1, x2, y = self._aug_pair(x1, x2, y)
        if self.concat:
            x = np.concatenate([x1, x2, np.abs(x1 - x2)], axis=-1)
            return (
                torch.from_numpy(x.transpose(2, 0, 1).copy()),
                torch.from_numpy(y.copy()),
                i,
            )
        return (
            torch.from_numpy(x1.transpose(2, 0, 1).copy()),
            torch.from_numpy(x2.transpose(2, 0, 1).copy()),
            torch.from_numpy(y.copy()),
            i,
        )


class StaticPatchDataset(Dataset):
    """Single-date segmentation (used for MANGO-style pretraining)."""

    def __init__(self, root, images, masks, augment=True):
        from PIL import Image
        self.images = images
        self.masks = masks
        self.root = root
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = np.asarray(Image.open(self.root / self.images[i]).convert("RGB"), dtype=np.float32) / 255.0
        msk = np.asarray(Image.open(self.root / self.masks[i]).convert("L"), dtype=np.int64)
        if self.augment:
            k = np.random.randint(4)
            img, msk = np.rot90(img, k), np.rot90(msk, k)
            if np.random.rand() > 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
        return torch.from_numpy(img.transpose(2, 0, 1).copy()), torch.from_numpy(msk.copy())
