"""Unit tests for the F1-gap decomposition in analysis/decompose_f1_gap.

The manuscript attributes the split-inflation gap to prevalence (pi), recall
(TPR) and false-alarm rate (FPR) by swapping one component at a time from the
isolated (honest) test to the random (leaky) test. Two properties make the
decomposition trustworthy and are asserted here with hand-built inputs:

  1. CONSERVATION -- the per-step contributions always sum to the total gap,
     for every ordering of the three components. The per-step values depend on
     the order, but their sum must not.
  2. AGREEMENT WITH THE CLOSED FORM -- f1(pi, tpr, fpr) must reproduce the
     F1_change the pipeline computes from the same confusion matrix.

Property 1 is the one that catches a sign error, a dropped term, or a step that
is silently overwritten instead of accumulated.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.decompose_f1_gap import decompose, f1, parts


def _block(cm):
    """Wrap a 4x4 confusion matrix in the metric-block dict parts() expects."""
    return {"confusion_matrix": np.asarray(cm, dtype=float)}


def test_f1_closed_form_matches_pipeline():
    """f1() must agree with binary_change() on the same confusion matrix.

    Imported from aggregate_results so the two code paths are compared rather
    than reimplemented.
    """
    from analysis.aggregate_results import binary_change

    cm = np.array([
        [18, 2, 1, 0],
        [3, 6, 4, 1],
        [2, 2, 8, 2],
        [16, 1, 3, 20],
    ], dtype=float)
    pi, tpr, fpr = parts(_block(cm))
    assert abs(f1(pi, tpr, fpr) - binary_change(cm)["F1_change"]) < 1e-12


def test_parts_reads_the_binary_change_class():
    """Prevalence, TPR and FPR must come from the {loss,gain} change block.

    For the 4x4 matrix below the row sums are 21, 14, 14 and 40, so the total
    is 89. Within the change class: tp = 6+4+2+8 = 20, fn = 3+1+2+2 = 8.
    Across it: fp = 2+1+1+3 = 7, and tn = 89 - 20 - 8 - 7 = 54.
    """
    cm = np.array([
        [18, 2, 1, 0],
        [3, 6, 4, 1],
        [2, 2, 8, 2],
        [16, 1, 3, 20],
    ], dtype=float)
    pi, tpr, fpr = parts(_block(cm))
    tp, fn, fp = 20.0, 8.0, 7.0
    tn = cm.sum() - tp - fn - fp
    assert tn == 54.0
    assert abs(pi - (tp + fn) / cm.sum()) < 1e-12
    assert abs(tpr - tp / (tp + fn)) < 1e-12
    assert abs(fpr - fp / (fp + tn)) < 1e-12


def _sample_runs(n=4):
    """(honest, leaky) triples differing in ALL THREE components.

    Every pair must move prevalence, recall and FPR together. An earlier version
    held FPR fixed across the two arms, which made the fpr step contribute a
    constant zero -- so a mutation that zeroed that step entirely went unnoticed.
    Conservation is only a real check when each term can be non-zero.
    """
    honest = [(0.20, 0.70, 0.05), (0.25, 0.65, 0.08),
              (0.18, 0.75, 0.04), (0.30, 0.60, 0.10)]
    leaky = [(0.28, 0.90, 0.02), (0.33, 0.85, 0.05),
             (0.26, 0.92, 0.01), (0.38, 0.80, 0.07)]
    return [{"honest": h, "leaky": l} for h, l in zip(honest, leaky)]


def test_decomposition_conserves_the_total_gap_for_every_order():
    """Sum of the three step contributions == total gap, for all 6 orders."""
    import itertools

    runs = _sample_runs()
    orders = list(itertools.permutations(["pi", "tpr", "fpr"]))
    assert len(orders) == 6
    for order in orders:
        steps, total = decompose(runs, order)
        step_sum = sum(v[0] for v in steps.values())   # means; pp
        assert abs(step_sum - total) < 1e-9, (
            f"order {order}: steps sum to {step_sum} but the total gap is {total}")


def test_decomposition_is_exact_for_a_single_run():
    """With one run the step means are the step values, so the identity is exact."""
    runs = [{"honest": (0.20, 0.70, 0.05), "leaky": (0.20, 0.90, 0.05)}]
    steps, total = decompose(runs, ("pi", "tpr", "fpr"))
    assert abs(steps["pi"][0]) < 1e-12, "unchanged prevalence contributes nothing"
    assert steps["fpr"][0] == 0.0
    assert abs(steps["tpr"][0] - total) < 1e-12
    assert steps["tpr"][1] == 0.0, "a single run has no standard deviation"


def test_identification_order_moves_share_not_total():
    """The order changes each step's size but never the total gap."""
    import itertools

    runs = _sample_runs()
    totals = []
    for order in itertools.permutations(["pi", "tpr", "fpr"]):
        steps, total = decompose(runs, order)
        totals.append(total)
        assert abs(sum(v[0] for v in steps.values()) - total) < 1e-9
    assert max(totals) - min(totals) < 1e-9, "total gap must be order-invariant"


def test_every_step_contributes_a_nonzero_share():
    """Each component must move between the arms, so no step is trivially zero.

    Conservation alone is vacuous if one term is always 0: a mutation that
    zeroed a step would still satisfy "the steps sum to the total". This
    asserts the fixture actually exercises all three components, and that the
    step shares are all strictly positive.
    """
    runs = _sample_runs()
    for r in runs:
        h, l = r["honest"], r["leaky"]
        assert h[0] != l[0], "prevalence must differ between the arms"
        assert h[1] != l[1], "recall must differ between the arms"
        assert h[2] != l[2], "FPR must differ between the arms"

    steps, total = decompose(runs, ("pi", "tpr", "fpr"))
    for name in ("pi", "tpr", "fpr"):
        assert steps[name][0] > 0.0, f"the {name} step contributes nothing here"


def test_identical_arms_give_zero_gap():
    runs = [{"honest": (0.22, 0.68, 0.06), "leaky": (0.22, 0.68, 0.06)}]
    for order in (("pi", "tpr", "fpr"), ("fpr", "tpr", "pi")):
        steps, total = decompose(runs, order)
        assert abs(total) < 1e-12
        assert all(abs(v[0]) < 1e-12 for v in steps.values())