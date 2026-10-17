"""Regression tests for the canonical binary-change F1 (analysis/aggregate_results.binary_change).

Regression guard for a class of defect that appeared twice in this project:
an aggregator using a *strict* (true loss-vs-gain) F1 while reporting it under
the name "F1_change". The canonical metric treats {loss, gain} as one "changed"
class, so a confusion-matrix off-diagonal entry between loss and gain is a TRUE
POSITIVE for change detection. The tests below pin that semantic with
hand-computed expectations.

Confusion-matrix convention: cm[true][pred]; classes 0=stable_mangrove,
1=loss, 2=gain, 3=stable_background.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.aggregate_results import binary_change


def _cm(rows):
    """Build a 4x4 cm from [[count]*4 for each true class]."""
    return np.array(rows, dtype=np.float64)


def test_loss_vs_gain_confusion_counts_as_true_positive():
    """cm[1][2] (true loss predicted gain) is a CHANGE hit, not a miss.

    Hand-computed with CHANGED={1,2}:
      tp = cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2] = 6+4+2+8 = 20
      fn = cm[1,0]+cm[1,3]+cm[2,0]+cm[2,3] = 3+1+2+2 = 8
      fp = cm[0,1]+cm[0,2]+cm[3,1]+cm[3,2] = 2+1+1+3 = 7
      tn = remaining stable-stable cells = 89 - 20 - 8 - 7 = 54
    So precision = 20/27, recall = 20/28, F1 = 2*20/(2*20+7+8) = 40/55.
    """
    cm = _cm([
        [18, 2, 1, 0],   # true stable_mangrove
        [3,  6, 4, 1],   # true loss
        [2,  2, 8, 2],   # true gain
        [16, 1, 3, 20],  # true stable_background
    ])
    m = binary_change(cm)

    tp, fn, fp = 20.0, 8.0, 7.0
    assert abs(m["recall_change"] - tp / (tp + fn)) < 1e-12
    assert abs(m["precision_change"] - tp / (tp + fp)) < 1e-12
    assert abs(m["F1_change"] - 2 * tp / (2 * tp + fp + fn)) < 1e-12
    # A strict-F1 implementation would score cm[1,2] and cm[2,1] as misses,
    # giving tp=14, fn=14, fp=7 -> F1 = 28/49 ~= 0.571. Guard against that.
    assert m["F1_change"] > 0.72, "loss<->gain confusion must count as a change hit"


def test_perfect_and_inverted_predictions():
    perfect = np.zeros((4, 4), dtype=np.float64)
    for i, n in ((0, 18), (1, 6), (2, 8), (3, 20)):
        perfect[i, i] = n
    m = binary_change(perfect)
    assert abs(m["F1_change"] - 1.0) < 1e-12
    assert abs(m["precision_change"] - 1.0) < 1e-12
    assert abs(m["recall_change"] - 1.0) < 1e-12

    # Mirroring the columns swaps loss<->gain while leaving both stable classes
    # untouched. Under the canonical {1,2}="changed" semantics that is STILL a
    # perfect change detector -- which is exactly why the strict interpretation
    # (loss vs gain as distinct correct answers) would be the wrong metric.
    mirrored = perfect[:, ::-1].copy()
    m_mir = binary_change(mirrored)
    assert m_mir["F1_change"] == 1.0

    # A genuinely inverted change prediction puts all true changes on stable
    # classes and all stable pixels on change classes.
    inverted = np.zeros((4, 4), dtype=np.float64)
    for i, n in ((0, 18), (1, 6), (2, 8), (3, 20)):
        inverted[i, (i + 2) % 4] = n  # 0->2, 1->3, 2->0, 3->1
    m_inv = binary_change(inverted)
    assert m_inv["F1_change"] == 0.0


def test_no_change_and_no_unchanged_degenerate_cases():
    """Degenerate partitions must not raise or produce NaN.

    max(x, 1) guards in binary_change mean a partition with an empty side
    returns 0.0 rather than dividing by zero.
    """
    only_change = np.zeros((4, 4), dtype=np.float64)
    only_change[1, 1] = 5
    only_change[2, 2] = 5
    m = binary_change(only_change)
    assert m["F1_change"] == 1.0  # nothing to miss, nothing to false-alarm

    only_unchanged = np.zeros((4, 4), dtype=np.float64)
    only_unchanged[0, 0] = 9
    only_unchanged[3, 3] = 7
    m2 = binary_change(only_unchanged)
    assert m2["F1_change"] == 0.0  # no predicted change at all
    assert np.isfinite(m2["F1_change"]) and np.isfinite(m2["kappa_change"])


def test_omission_commission_direction():
    """Omission = change we missed; Commission = change we invented.

    10 true changes, 6 found  -> omission 4/10 = 0.4.
    4 false alarms            -> commission 4/(tp+fp) = 4/10 = 0.4.
    """
    cm = _cm([
        [0, 0, 0, 0],
        [4, 6, 0, 0],   # true loss: 4 missed, 6 hit
        [0, 0, 0, 0],
        [0, 4, 0, 0],   # true background, 4 false alarms
    ])
    m = binary_change(cm)
    assert abs(m["Omission"] - 4 / 10) < 1e-12
    assert abs(m["Commission"] - 4 / 10) < 1e-12  # tp+fp = 6+4 = 10
    assert abs(m["recall_change"] - 0.6) < 1e-12
    assert abs(m["precision_change"] - 0.6) < 1e-12