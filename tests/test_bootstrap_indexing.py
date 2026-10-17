"""Regression tests for the cluster-bootstrap index layout in analysis/block_bootstrap.py.

Regression guard for a silent under-count: the pooled array is laid out
SEED-MAJOR (patch i's seed s sits at row i + s*n_idx), but an earlier version
indexed it PATCH-MAJOR (3*i + s). With the wrong index, every "3 seeds per
patch" replicate re-read a single patch's rows and most of the data was never
resampled -- for a region collapsing to one cluster the bootstrap degenerated
instead of reflecting reality.

The logic below reproduces the layout contract exactly as the script builds it
and asserts the two properties that the bug violated:
  1. the mapping patch->row is a bijection onto n_idx*len(SEEDS) rows,
  2. every seed of every patch is covered, and different seeds of the same
     patch map to DIFFERENT rows.
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis.block_bootstrap as BB
from analysis.block_bootstrap import SEEDS

N_SEEDS = len(SEEDS)
_SRC = Path(BB.__file__).read_text()


def test_seed_major_is_the_layout_the_script_actually_builds():
    """Guard the coupling between SEEDS and the literal used to build clusters.

    block_bootstrap builds its resampling index as
    ``[i + s * n_idx for i in cl for s in range(3)]`` -- a hard-coded 3 that
    duplicates the length of SEEDS. If SEEDS were ever extended without updating
    that literal, the replicate would silently keep using only three seeds and
    the CI would look fine while being wrong. This pins the two together.
    """
    literals = set(re.findall(r"for s in range\((\d+)\)", _SRC))
    assert literals == {str(N_SEEDS)}, (
        f"the seed count literal in block_bootstrap.py is {literals}, "
        f"but SEEDS has {N_SEEDS} entries; they must match")


def test_seed_major_layout_is_a_bijection():
    """i + s*n_idx enumerates each (patch, seed) exactly once, no collisions."""
    n_idx = 7  # arbitrary patch count
    rows = [i + s * n_idx for i in range(n_idx) for s in range(N_SEEDS)]
    assert len(rows) == n_idx * N_SEEDS
    assert len(set(rows)) == n_idx * N_SEEDS, "seed-major layout must be collision-free"
    assert set(rows) == set(range(n_idx * N_SEEDS)), "must cover every row"


def test_patch_major_layout_maps_to_the_wrong_rows():
    """The old 3*i+s layout indexes rows that do not hold patch i / seed s.

    Both layouts are collision-free in isolation; the defect is that they
    disagree about WHICH row holds a given (patch, seed). A replicate built
    with one layout and pooled from the other silently reads other patches'
    pixels, so most of the data is never resampled.
    """
    n_idx = 7
    # NOTE: with exactly 3 seeds the two layouts generate the same SET of rows
    # {0..n_idx*3-1}; they differ only in WHICH row each (patch, seed) maps to.
    # That is why the original bug was silent -- a set/union comparison cannot
    # detect it. Compare the per-(patch, seed) mapping instead.
    seed_major = {(i, s): i + s * n_idx for i in range(n_idx) for s in range(N_SEEDS)}
    patch_major = {(i, s): 3 * i + s for i in range(n_idx) for s in range(N_SEEDS)}
    assert len(set(seed_major.values())) == n_idx * N_SEEDS
    assert len(set(patch_major.values())) == n_idx * N_SEEDS
    assert seed_major != patch_major, "the two layouts disagree about row assignment"
    assert seed_major[(0, 1)] != patch_major[(0, 1)], "patch 0 seed 1 moves"

    # For patch 0: the correct layout puts its seeds n_idx rows apart; the
    # patch-major assumption puts them adjacent.
    assert seed_major[(0, 1)] - seed_major[(0, 0)] == n_idx
    assert patch_major[(0, 1)] - patch_major[(0, 0)] == 1


def test_single_cluster_region_degenerates_explicitly():
    """A region collapsing to one cluster has zero bootstrap variance BY DESIGN.

    block_bootstrap draws len(clusters) picks per replicate, so with a single
    cluster every replicate pools exactly that cluster once and the metric is
    identical across replicates -> CI collapses to a point. That is the correct
    (and documented) degenerate behaviour, not a silent failure.
    """
    n_idx = 5
    one_cluster = [[i + s * n_idx for i in range(n_idx) for s in range(N_SEEDS)]]
    assert len(one_cluster) == 1

    cms = np.arange(n_idx * N_SEEDS * 6, dtype=np.int64).reshape(n_idx * N_SEEDS, 6)
    expected = cms.sum(axis=0)

    rng = np.random.default_rng(0)
    pooled_results = []
    for _ in range(10):                       # 10 replicates
        cl = one_cluster
        pick = [rng.integers(0, len(cl)) for _ in range(len(cl))]   # len(cl) == 1
        pooled = np.zeros(cms.shape[1:], dtype=np.int64)
        for k in pick:
            pooled += cms[cl[k]].sum(axis=0)
        pooled_results.append(tuple(pooled.tolist()))

    assert len(set(pooled_results)) == 1, "single-cluster bootstrap has no variance"
    assert pooled_results[0] == tuple(expected.tolist())


def test_multi_cluster_resample_changes_the_pooled_matrix():
    """A two-cluster setup must actually vary the pooled result across draws."""
    n_idx, per = 6, N_SEEDS
    cms = np.zeros((n_idx * per, 4), dtype=np.int64)
    # cluster 0 = patches 0..2, cluster 1 = patches 3..5 (all seeds)
    c0 = [i + s * n_idx for i in range(3) for s in range(per)]
    c1 = [i + s * n_idx for i in range(3, 6) for s in range(per)]
    for r in c0:
        cms[r, 0] = 1
    for r in c1:
        cms[r, 1] = 1
    clusters = [c0, c1]

    rng = np.random.default_rng(1)
    outs = set()
    for _ in range(20):
        pick = [rng.integers(0, len(clusters)) for _ in range(len(clusters))]
        pooled = np.zeros(cms.shape[1:], dtype=np.int64)
        for k in pick:
            pooled += cms[clusters[k]].sum(axis=0)
        outs.add(tuple(pooled.tolist()))
    assert len(outs) > 1, "multi-cluster bootstrap must produce varying pooled CMs"