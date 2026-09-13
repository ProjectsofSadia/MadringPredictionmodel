"""Mechanical smoke tests for the model and simulator code paths.

All inputs here are synthetic random arrays used only to prove the code runs and
the shapes are right. They are NOT Formula 1 data, they never touch data/, and no
number produced here appears in any prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import run_walk_forward, residual_pool  # noqa: E402
from features import FEATURES  # noqa: E402
from race_simulator import simulate, summarise  # noqa: E402

rng = np.random.default_rng(0)


def _synthetic_panel(n_races=9, n_drivers=20):
    rows = []
    for r in range(1, n_races + 1):
        for d in range(n_drivers):
            q = rng.normal(1.0, 0.6)
            rows.append(
                {
                    "driver": f"D{d:02d}",
                    "team": f"T{d // 2:02d}",
                    "round": r,
                    "race_date": pd.Timestamp("2026-03-08") + pd.Timedelta(days=14 * r),
                    "quali_gap_pct": q,
                    "quali_gap_to_teammate_pct": rng.normal(0, 0.2),
                    "quali_session_reached": rng.integers(1, 4),
                    "quali_missing": 0,
                    "grid_position": float(d + 1),
                    "grid_minus_quali": 0.0,
                    "fp2_longrun_pace_delta_pct": q + rng.normal(0, 0.3),
                    "fp2_longrun_variability_s": abs(rng.normal(0.4, 0.1)),
                    "fp2_longrun_laps": rng.integers(5, 15),
                    "fp3_longrun_pace_delta_pct": np.nan,
                    "fp3_longrun_variability_s": np.nan,
                    "fp3_longrun_laps": np.nan,
                    "longrun_missing": 0,
                    "team_recent_race_pace_delta": q + rng.normal(0, 0.2),
                    "driver_recent_race_pace_delta": q + rng.normal(0, 0.2),
                    "driver_quali_to_race_conversion": rng.normal(0, 0.1),
                    "races_of_history": float(r - 1),
                    "race_pace_delta_pct": 0.8 * q + rng.normal(0, 0.35),
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_runs_and_selects_a_model():
    panel = _synthetic_panel()
    wf = run_walk_forward(panel)
    assert set(wf["summary"]) == {
        "B0_null", "B1_qualifying", "B2_recent_form", "B3_ridge", "XGBoost"
    }
    assert wf["selected_model"] in wf["summary"]
    assert len(wf["folds"]) >= 3
    resid = residual_pool(panel, wf["oof"], wf["selected_model"])
    assert len(resid) > 50 and np.isfinite(resid).all()


def test_all_features_present_in_panel():
    panel = _synthetic_panel(n_races=2, n_drivers=4)
    assert set(FEATURES).issubset(panel.columns)


def _priors():
    return {
        "degradation_pct_per_lap": {"pooled": 0.05},
        "lap_noise_s": {"value": 0.35},
        "pit_loss_s": {"value": 21.0},
        "lap1_gap_s": {"value": 0.9},
        "lap1_position_delta": {"samples": list(rng.normal(0, 1.2, 400))},
        "safety_car": {
            "p_race_has_sc": 0.5,
            "mean_periods_when_present": 1.0,
            "mean_lap_share_when_present": 0.08,
        },
    }


def _assumptions():
    return {
        "following_gap_s": 1.0,
        "dirty_air_penalty_s": 0.25,
        "pass_pace_threshold_s": 0.35,
        "pass_time_cost_s": 0.15,
        "pit_lap_jitter": 3,
        "sc_gap_s": 1.0,
        "sc_pit_loss_discount": 0.55,
    }


def test_simulator_shapes_and_probabilities():
    D, S = 22, 400
    pred = np.sort(rng.normal(0, 0.6, D))
    out = simulate(
        pred_pace_pct=pred,
        residuals=rng.normal(0, 0.4, 300),
        grid_position=np.arange(1, D + 1, dtype=float),
        dnf_per_race=np.full(D, 0.08),
        ref_race_lap_s=95.0,
        n_laps=57,
        priors=_priors(),
        assumptions=_assumptions(),
        pass_prob=0.25,
        n_sims=S,
        seed=1,
    )
    fin = out["finish_positions"]
    assert fin.shape == (S, D)
    for s in range(0, S, 97):
        assert sorted(fin[s]) == list(range(1, D + 1))

    rows = summarise(fin, [f"D{i:02d}" for i in range(D)], np.arange(1, D + 1, dtype=float))
    assert abs(sum(r["win_pct"] for r in rows) - 100) < 1e-6
    assert abs(sum(r["podium_pct"] for r in rows) - 300) < 1e-6
    assert all(1 <= r["expected_finish"] <= D for r in rows)


def test_overtaking_case_changes_outcomes():
    D, S = 22, 400
    pred = np.sort(rng.normal(0, 0.6, D))
    kw = dict(
        pred_pace_pct=pred,
        residuals=rng.normal(0, 0.4, 300),
        grid_position=np.arange(1, D + 1, dtype=float),
        dnf_per_race=np.full(D, 0.05),
        ref_race_lap_s=95.0,
        n_laps=40,
        priors=_priors(),
        assumptions=_assumptions(),
        n_sims=S,
        seed=1,
    )
    low = simulate(pass_prob=0.10, **kw)["finish_positions"].mean(axis=0)
    high = simulate(pass_prob=0.50, **kw)["finish_positions"].mean(axis=0)
    assert not np.allclose(low, high)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
