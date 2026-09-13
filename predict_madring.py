"""Assemble Madring's pre-race feature row and predict expected race pace.

The Madring rows carry a NaN target. They are appended to the historical panel
BEFORE form features are computed, so the EWMA shift(1) gives them form built
only from earlier races - the same code path the training rows use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA, MADRING_ROUND, SEASON
from features import add_form_features, load_madring_grid, practice_features, quali_features
from fetch_data import load_session


def madring_block() -> tuple[pd.DataFrame, dict]:
    q = quali_features(SEASON, MADRING_ROUND)
    if q.empty:
        raise RuntimeError("Madring qualifying data unavailable - cannot build features")

    p = practice_features(SEASON, MADRING_ROUND)
    block = q.merge(p, on="driver", how="left") if not p.empty else q.copy()

    grid = load_madring_grid(DATA / "madring_grid.csv", q)
    block = block.merge(
        grid[["driver_code", "grid_position", "team", "status", "source_url", "retrieved_utc"]]
        .rename(columns={"driver_code": "driver", "team": "grid_team"}),
        on="driver",
        how="left",
    )

    missing_grid = block.loc[block["grid_position"].isna(), "driver"].tolist()
    if missing_grid:
        raise RuntimeError(
            f"No grid entry resolved for {missing_grid}. Fix data/madring_grid.csv "
            "before running - the grid is an external documented input and must not "
            "be inferred from qualifying."
        )

    for col in (
        "fp2_longrun_pace_delta_pct",
        "fp3_longrun_pace_delta_pct",
        "fp2_longrun_laps",
        "fp3_longrun_laps",
        "fp2_longrun_variability_s",
        "fp3_longrun_variability_s",
    ):
        if col not in block.columns:
            block[col] = np.nan
    block["longrun_missing"] = (
        block["fp2_longrun_pace_delta_pct"].isna()
        & block["fp3_longrun_pace_delta_pct"].isna()
    ).astype(int)

    block["race_pace_delta_pct"] = np.nan
    block["round"] = MADRING_ROUND
    block["season"] = SEASON

    qs = load_session(SEASON, MADRING_ROUND, "Q")
    block["race_date"] = pd.Timestamp(qs.date) + pd.Timedelta(days=1)

    pole_lap_s = float(
        qs.laps.pick_quicklaps().pick_fastest()["LapTime"].total_seconds()
    )
    meta = {
        "pole_lap_s": pole_lap_s,
        "grid_source_url": str(grid["source_url"].iat[0]),
        "grid_retrieved_utc": str(grid["retrieved_utc"].iat[0]),
        "n_drivers": int(len(block)),
        "drivers_without_quali_time": block.loc[
            block["quali_missing"] == 1, "driver"
        ].tolist(),
        "practice_sessions_with_longrun_data": [
            s
            for s, col in (("FP2", "fp2_longrun_pace_delta_pct"),
                           ("FP3", "fp3_longrun_pace_delta_pct"))
            if block[col].notna().any()
        ],
    }
    return block, meta


def full_panel(history_panel: pd.DataFrame, madring: pd.DataFrame) -> pd.DataFrame:
    """Recompute form features across history + Madring so the prediction row's
    form comes from strictly earlier races via the same shift(1) code path."""
    cols = sorted(set(history_panel.columns) | set(madring.columns))
    combined = pd.concat(
        [history_panel.reindex(columns=cols), madring.reindex(columns=cols)],
        ignore_index=True,
    )
    drop = [
        c
        for c in (
            "team_recent_race_pace_delta",
            "driver_recent_race_pace_delta",
            "driver_quali_to_race_conversion",
            "races_of_history",
            "grid_minus_quali",
        )
        if c in combined.columns
    ]
    combined = combined.drop(columns=drop)
    return add_form_features(combined)


def predict_pace(model, madring_rows: pd.DataFrame) -> np.ndarray:
    """Predicted pace delta, re-centred so the field mean is zero.

    The target is defined relative to the field median of the same race, so the
    prediction is only meaningful as a relative quantity."""
    raw = np.asarray(model.predict(madring_rows), dtype=float)
    return raw - np.nanmean(raw)
