"""Unit tests for the multiple-comparison corrections in analysis/cluster_bootstrap_v2.

Holm-Bonferroni and Benjamini-Hochberg govern which model comparisons the
manuscript calls significant, so their adjustment must be exactly right. The
tests use the textbook worked examples and the defining properties, rather than
re-deriving the code's own arithmetic.

Definitions used below (m = number of hypotheses):
  Holm   adj_(i) = max over j<=i of (m - j + 1) * p_(j), applied to the
          ascending order, then capped at 1.
  BH     adj_(i) = min over j>=i of m * p_(j) / j, applied from the largest
          p-value down, then capped at 1.
Both must preserve the ordering of the original p-values and never decrease
them.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.cluster_bootstrap_v2 import bh, holm


def test_holm_textbook_example():
    """p = {0.01, 0.04, 0.03, 0.005}, m = 4 -> {0.04, 0.06, 0.09, 0.02}.

    Ascending order is 0.005, 0.01, 0.03, 0.04:
      rank 1: (4-1+1)*0.005 = 0.020
      rank 2: (4-2+1)*0.01  = 0.030 -> running max 0.030
      rank 3: (4-3+1)*0.03  = 0.060
      rank 4: (4-4+1)*0.04  = 0.040 -> running max 0.060
    Mapped back to input positions (0.01, 0.04, 0.03, 0.005) this gives
    (0.03, 0.06, 0.06, 0.02).
    """
    p = [0.01, 0.04, 0.03, 0.005]
    adj = holm(p)
    expected = np.array([0.03, 0.06, 0.06, 0.02])
    assert np.allclose(adj, expected), f"got {adj}, expected {expected}"


def test_holm_is_monotone_and_bounded():
    rng = np.random.default_rng(7)
    p = rng.random(40)
    adj = holm(p)
    assert np.all(adj >= p - 1e-12), "adjustment may only increase p-values"
    assert np.all(adj <= 1.0 + 1e-12), "adjusted p-values must stay <= 1"
    # The correct invariant is WEAK monotonicity along the sorted p order.
    # It is not "argsort is unchanged": Holm caps at 1.0, so many large
    # p-values tie exactly at 1.0 and their relative order carries no meaning.
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12), "adjusted p must not decrease along the sorted order"


def test_holm_single_and_edge_cases():
    assert np.allclose(holm([0.03]), [0.03]), "m=1 leaves p unchanged"
    assert np.allclose(holm([1.0]), [1.0])
    assert np.allclose(holm([0.0]), [0.0])
    # All equal: each rank scales by a decreasing factor, so the largest
    # adjustment lands on the last one and the running max caps them equal.
    adj = holm([0.02] * 3)
    assert np.allclose(adj, [0.06, 0.06, 0.06])


def test_bh_textbook_example():
    """p = {0.01, 0.02, 0.03, 0.04, 0.05}, m = 5 at q = 0.05.

    Descending pass on the ascending order (rank 1..5), m*p/j:
      5*0.01/1 = 0.05, 5*0.02/2 = 0.05, 5*0.03/3 = 0.05,
      5*0.04/4 = 0.05, 5*0.05/5 = 0.05
    Running minimum from the top keeps everything at 0.05, and all five are
    rejected at q = 0.05.
    """
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    adj, rejected = bh(p, q=0.05)
    assert np.allclose(adj, 0.05)
    assert sorted(rejected) == [0, 1, 2, 3, 4]


def test_bh_step_up_pattern():
    """A classic FDR example where BH rejects the two smallest p-values only.

    p = {0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216}
    with m = 10, q = 0.05 rejects the first two and stops.
    """
    p = [0.001, 0.008, 0.039, 0.041, 0.042,
         0.06, 0.074, 0.205, 0.212, 0.216]
    adj, rejected = bh(p, q=0.05)
    assert rejected == [0, 1], f"expected the first two rejections, got {rejected}"
    # Adjusted values are non-decreasing in the sorted p order.
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_bh_is_less_conservative_than_holm():
    """BH controls FDR, Holm controls FWER: for any input BH <= Holm.

    This is the property that lets the manuscript describe one procedure as
    more permissive than the other; if it ever inverted, the prose would be
    wrong even though both implementations stayed "valid".
    """
    rng = np.random.default_rng(11)
    for _ in range(200):
        p = rng.random(rng.integers(2, 12))
        a_holm, _ = bh(p, q=0.05)
        assert np.all(a_holm <= holm(p) + 1e-12)


def test_bh_rejects_none_when_all_p_are_large():
    p = [0.6, 0.7, 0.8, 0.9]
    _, rejected = bh(p, q=0.05)
    assert rejected == []


def test_both_survive_degenerate_input():
    for fn in (holm,):
        assert np.all(np.isfinite(fn([0.0, 1.0])))
    adj, rejected = bh([0.0, 1.0], q=0.05)
    assert np.all(np.isfinite(adj))
    assert 0 in rejected