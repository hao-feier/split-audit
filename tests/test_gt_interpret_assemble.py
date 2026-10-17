"""Regression tests for analysis/gt_interpret_assemble._cells.

Regression guard for a defect found2026-10-15: the assembler read every montage
with per-montage cell indices (000..len(ids)-1) while the raw file in the
repository numbers cells continuously across montages, so 150 of the 175
recorded judgments were unreachable and the coverage report counted only the
first montage. Both conventions are legal depending on how the interpreter
worked, so the mapping must be decided per montage, not assumed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.gt_interpret_assemble import _cells


def test_per_montage_indexing():
    """Indices restart at 000 in every montage."""
    raw = {"000": 1, "001": 0, "024": 3}
    assert _cells(raw, 25, 0) == {0: 1, 1: 0, 24: 3}


def test_per_montage_indexing_survives_a_nonzero_offset():
    """Regression: scoring only the global rule would drop montage 001 onward.

    A per-montage file has 000..024 in EVERY montage, so at offset 25 the
    global mapping maps every key below range and must not win.
    """
    raw = {"000": 1, "001": 2, "024": 3}
    assert _cells(raw, 25, 25) == {0: 1, 1: 2, 24: 3}


def test_global_indexing_offsets_by_prior_montages():
    """Montage 001 starts at 025 when montage 000 held 25 ids."""
    raw = {"025": 2, "030": 0, "049": 1}
    assert _cells(raw, 25, 25) == {0: 2, 5: 0, 24: 1}


def test_out_of_range_cells_are_dropped_not_wrapped():
    """A global index beyond the montage must not alias into another montage."""
    assert _cells({"000": 1, "099": 3}, 25, 0) == {0: 1}
    assert _cells({"049": 1}, 25, 25) == {24: 1}


def test_non_numeric_and_out_of_range_keys_are_ignored():
    assert _cells({"abc": 1, "-1": 2, "1000": 3}, 25, 0) == {}


def test_partially_filled_montage_keeps_its_own_cells():
    """The defect case: only the tail of a global run, first montage empty."""
    assert _cells({"150": 0, "174": 2}, 25, 150) == {0: 0, 24: 2}


def test_both_conventions_readable_across_a_mixed_file():
    """Montage 000 global, montage 001 per-montage: both survive."""
    assert _cells({"000": 0, "024": 1}, 25, 0) == {0: 0, 24: 1}
    assert _cells({"000": 5, "010": 6}, 25, 25) == {0: 5, 10: 6}


@pytest.mark.parametrize("bad", [None, "3", 3.5])
def test_non_integer_values_pass_through_to_caller_validation(bad):
    """Values are not filtered here; the caller casts and records them."""
    assert _cells({"000": bad}, 25, 0) == {0: bad}