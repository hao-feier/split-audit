"""Geospatial helpers (rasterio 1.4.x compatible)."""
import math

import numpy as np

from rasterio.windows import from_bounds


def win_bounds(src, lon_min, lat_min, lon_max, lat_max):
    """Window for a lon/lat box on an EPSG:4326 dataset, clipped to bounds."""
    w = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
    col_off = max(0, math.floor(w.col_off))
    row_off = max(0, math.floor(w.row_off))
    col_off2 = min(src.width, math.ceil(w.col_off + w.width))
    row_off2 = min(src.height, math.ceil(w.row_off + w.height))
    if col_off2 <= col_off or row_off2 <= row_off:
        raise ValueError("window outside dataset")
    return w.from_bounds(col_off, row_off, col_off - col_off2, row_off - row_off2) if False else \
        type(w)(col_off=col_off, row_off=row_off, width=col_off2 - col_off, height=row_off2 - row_off)


def win_bounds_utm(src, x1, y1, x2, y2):
    """Window for a UTM box on a UTM dataset, clipped to bounds."""
    w = from_bounds(x1, y1, x2, y2, src.transform)
    col_off = max(0, math.floor(w.col_off))
    row_off = max(0, math.floor(w.row_off))
    col_off2 = min(src.width, math.ceil(w.col_off + w.width))
    row_off2 = min(src.height, math.ceil(w.row_off + w.height))
    if col_off2 <= col_off or row_off2 <= row_off:
        raise ValueError("window outside dataset")
    return type(w)(col_off=col_off, row_off=row_off, width=col_off2 - col_off, height=row_off2 - row_off)


def downscale_majority(arr, factor=3):
    """factor x factor majority vote (30m -> 10m). Odd sizes padded with 0."""
    h, w = arr.shape
    hp, wp = (h + factor - 1) // factor * factor, (w + factor - 1) // factor * factor
    if hp != h or wp != w:
        p = arr.__class__.__call__((hp, wp), arr.dtype) if hasattr(arr, "__class__") else None
        import numpy as np
        p = np.zeros((hp, wp), dtype=arr.dtype)
        p[:h, :w] = arr
        arr = p
    blocks = arr.reshape(hp // factor, factor, wp // factor, factor)
    vote = blocks.sum(axis=(1, 3))
    return (vote >= (factor * factor) // 2).astype(arr.dtype)


def upscale_replicate(arr, factor=3):
    """30m -> 10m: each 30m pixel becomes a factor x factor block (exact grid nesting)."""
    return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
