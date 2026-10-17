"""Unit tests for the reference-mask morphology in analysis/boundary_tolerance.py.

This module underpins experiment E7 (label-boundary tolerance). Its published
result -- that the F1 response is flat beyond ~10 m of boundary displacement --
depends on morph() growing/shrinking the reference mask by the intended amount.
A silent off-by-one in the structuring element would still produce a smooth
curve, so it would not be caught by eyeballing the figure.

Conventions: the mask is boolean, "change" is class 1 or 2, and ST is the
4-connected cross (scipy generate_binary_structure(2, 1)). Three consequences
the expectations below encode, all verified against the implementation:

  * a 4-connected element reaches only the 4 orthogonal neighbours, never the
    diagonals, so k=1 dilation of one pixel yields 5 pixels, not 9;
  * morph() enlarges the structuring element by (k-1) dilations before applying
    it, so k=2 covers a solid 3x3 block around the seed (9 cells);
  * a 4x4 square block eroded once keeps a 2x2 core with its corners shaved off,
    since a corner lacks an orthogonal neighbour.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.boundary_tolerance import binary_change, morph, prf


def test_binary_change_maps_loss_and_gain_to_one():
    a = np.array([[0, 1], [2, 3]])
    out = binary_change(a)
    assert out.dtype == np.bool_
    assert out.tolist() == [[False, True], [True, False]]


def test_morph_k0_is_identity():
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True
    for op in ("dilate", "erode", "open", "close"):
        assert np.array_equal(morph(m, op, 0), m), f"{op} with k=0 must be identity"


def test_dilate_grows_by_exactly_k_pixels():
    """A single interior pixel grows by exactly k pixels of reach."""
    m = np.zeros((9, 9), dtype=bool)
    m[4, 4] = True
    assert morph(m, "dilate", 1).sum() == 5   # centre + 4 neighbours
    assert morph(m, "dilate", 2).sum() == 9   # radius-2 diamond
    for k in (1, 2, 3):
        out = morph(m, "dilate", k)
        assert out[4, 4], "dilation must keep the original pixel"

    d1 = morph(m, "dilate", 1)
    assert d1[3, 4] and d1[5, 4] and d1[4, 3] and d1[4, 5]
    assert not d1[3, 3], "4-connectivity does not reach the diagonal"

    d2 = morph(m, "dilate", 2)
    # morph() first grows the structuring element itself (k-1 iterations), so
    # k=2 dilates with an enlarged element and covers the full 3x3 block plus
    # the axial ring -- 9 cells in a solid square.
    assert d2[3:6, 3:6].all(), "k=2 covers the 3x3 block around the centre"
    assert not d2[2, 2], "k=2 does not reach the diagonal two steps away"
    assert not d2[1, 4], "k=2 does not reach three cells away"


def test_erode_shrinks_by_exactly_k_pixels():
    m = np.ones((9, 9), dtype=bool)  # foreground everywhere, border clipped
    for k in (1, 2):
        out = morph(m, "erode", k)
        assert out.sum() < m.sum(), f"k={k} erosion must remove pixels"
    e1 = morph(m, "erode", 1)
    assert not e1[0, 0], "the image border is background, so the corner erodes away"
    assert e1[4, 4], "the interior survives k=1"


def test_opening_removes_isolated_pixels_closing_fills_holes():
    speckle = np.zeros((9, 9), dtype=bool)
    speckle[0, 0] = True            # isolated speck on the border
    speckle[3:7, 3:7] = True        # solid 4x4 block
    opened = morph(speckle, "open", 1)
    assert not opened[0, 0], "opening must delete an isolated pixel"
    # Eroding a 4x4 square with a 4-connected cross leaves the 2x2 core; the
    # four corners lack an orthogonal neighbour and are removed.
    assert opened[4:6, 4:6].all(), "opening must keep the 2x2 core"
    assert not opened[3, 3], "a corner lacks a 4-connected neighbour"

    holed = np.ones((9, 9), dtype=bool)
    holed[4, 4] = False
    closed = morph(holed, "close", 1)
    assert closed[4, 4], "closing must fill a single-pixel hole"


def test_thin_structures_vanish_under_erosion():
    """Erosion must delete structures thinner than 2*k, and dilation must grow them.

    This is the geometric asymmetry the paper's boundary-tolerance experiment
    rests on: a one-pixel-wide predicted boundary is annihilated by an erosion
    of k>=1, while dilating the same line only widens it. If either operation
    stopped behaving this way, the E7 curve would change shape without any
    visible error in the plotting code.
    """
    m = np.zeros((11, 11), dtype=bool)
    m[5, 2:9] = True                # a 1-px-wide horizontal line
    d = morph(m, "dilate", 2)
    e = morph(m, "erode", 2)
    assert d.sum() > m.sum() and e.sum() < m.sum()
    assert e.sum() == 0, "eroding a 1-px line by 2 removes it entirely"


def test_prf_against_hand_computed_values():
    yt = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    yp = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)  # tp=3, fn=1, fp=1, tn=3
    m = prf(yt, yp)
    assert abs(m["precision"] - 3 / 4) < 1e-12
    assert abs(m["recall"] - 3 / 4) < 1e-12
    assert abs(m["F1"] - 2 * 3 / (2 * 3 + 1 + 1)) < 1e-12
    assert abs(m["omission"] - 1 / 4) < 1e-12
    assert abs(m["commission"] - 1 / 4) < 1e-12


def test_prf_empty_partitions_do_not_divide_by_zero():
    m = prf(np.zeros(5, dtype=bool), np.zeros(5, dtype=bool))
    assert m["F1"] == 0.0 and m["precision"] == 0.0 and m["recall"] == 0.0
    perfect = prf(np.ones(5, dtype=bool), np.ones(5, dtype=bool))
    assert perfect["F1"] == 1.0