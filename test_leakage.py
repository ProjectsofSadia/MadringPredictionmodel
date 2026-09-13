"""Leakage tests.

The frames built here are synthetic scaffolding to exercise the shift(1) logic.
They are NOT Formula 1 data and are never written to data/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import FEATURES, LINEAR_FEATURES, PROVENANCE, add_form_features  # noqa: E402

ALLOWED_SOURCES = {"Q", "FP1", "FP2", "FP3", "GRID", "PRIOR_RACES"}


def test_every_feature_has_provenance():
    assert set(FEATURES) == set(PROVENANCE)


def test_no_feature_comes_from_the_race_being_predicted():
    bad = {f: src for f, src in PROVENANCE.items() if src not in ALLOWED_SOURCES}
    assert not bad, f"features sourced from a disallowed session: {bad}"


def test_linear_features_are_a_subset():
    assert set(LINEAR_FEATURES).issubset(set(FEATURES))


def _panel(values):
    n = len(values)
    return pd.DataFrame(
        {
            "driver": ["AAA"] * n,
            "team": ["TeamX"] * n,
            "round": range(1, n + 1),
            "race_date": pd.date_range("2026-03-08", periods=n, freq="14D"),
            "race_pace_delta_pct": values,
            "quali_gap_pct": np.linspace(0.1, 0.5, n),
            "grid_position": np.arange(1, n + 1, dtype=float),
        }
    )


def test_first_race_has_no_form_feature():
    out = add_form_features(_panel([0.5, -0.2, 0.1, 0.3]))
    assert np.isnan(out.iloc[0]["team_recent_race_pace_delta"])
    assert out.iloc[0]["races_of_history"] == 0


def test_future_results_do_not_change_earlier_form_features():
    base = add_form_features(_panel([0.5, -0.2, 0.1, 0.3]))
    altered = add_form_features(_panel([0.5, -0.2, 0.1, 99.0]))  # last race changed
    cols = [
        "team_recent_race_pace_delta",
        "driver_recent_race_pace_delta",
        "driver_quali_to_race_conversion",
        "races_of_history",
    ]
    pd.testing.assert_frame_equal(
        base.iloc[:-1][cols].reset_index(drop=True),
        altered.iloc[:-1][cols].reset_index(drop=True),
    )


def test_unlabelled_prediction_row_still_gets_prior_form():
    panel = _panel([0.5, -0.2, 0.1, np.nan])  # last row = the race being predicted
    out = add_form_features(panel)
    assert np.isfinite(out.iloc[-1]["team_recent_race_pace_delta"])
    assert out.iloc[-1]["races_of_history"] == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
