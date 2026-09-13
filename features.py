"""Feature engineering. Every feature is declared in PROVENANCE with the session
it comes from; tests/test_leakage.py enforces that nothing is sourced from the
race session being predicted.

Provenance values:
    'Q'            qualifying (Saturday)
    'FP2' / 'FP3'  practice long runs (Friday/Saturday)
    'GRID'         official starting grid (published before the race)
    'PRIOR_RACES'  races strictly before this one
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DRIVER_SHRINKAGE_RACES,
    DROP_FIRST_LAPS_OF_PRACTICE_STINT,
    FORM_HALFLIFE_RACES,
    MIN_LONGRUN_STINT_LAPS,
    SEASON,
)
from fetch_data import load_session

PROVENANCE: dict[str, str] = {
    "quali_gap_pct": "Q",
    "quali_gap_to_teammate_pct": "Q",
    "quali_session_reached": "Q",
    "quali_missing": "Q",
    "grid_position": "GRID",
    "grid_minus_quali": "GRID",
    "fp2_longrun_pace_delta_pct": "FP2",
    "fp2_longrun_variability_s": "FP2",
    "fp2_longrun_laps": "FP2",
    "fp3_longrun_pace_delta_pct": "FP3",
    "fp3_longrun_variability_s": "FP3",
    "fp3_longrun_laps": "FP3",
    "longrun_missing": "FP2",
    "team_recent_race_pace_delta": "PRIOR_RACES",
    "driver_recent_race_pace_delta": "PRIOR_RACES",
    "driver_quali_to_race_conversion": "PRIOR_RACES",
    "races_of_history": "PRIOR_RACES",
}

FEATURES = list(PROVENANCE.keys())

# Features used by the linear baselines (they cannot take NaN natively).
LINEAR_FEATURES = [
    "quali_gap_pct",
    "grid_position",
    "team_recent_race_pace_delta",
    "driver_recent_race_pace_delta",
]


# --------------------------------------------------------------------------
# qualifying
# --------------------------------------------------------------------------
def quali_features(year: int, rnd: int) -> pd.DataFrame:
    s = load_session(year, rnd, "Q")
    if s is None:
        return pd.DataFrame()

    res = s.results.copy()
    if not {"Q1", "Q2", "Q3"}.issubset(res.columns):
        return pd.DataFrame()

    times = pd.concat(
        [res[q].dt.total_seconds() for q in ("Q1", "Q2", "Q3")], axis=1
    )
    times.columns = ["q1", "q2", "q3"]
    best = times.min(axis=1, skipna=True)
    reached = times.notna().values.cumsum(axis=1).max(axis=1)

    out = pd.DataFrame(
        {
            "driver": res["Abbreviation"].astype(str).values,
            "team": res["TeamName"].astype(str).values,
            "quali_best_s": best.values,
            "quali_session_reached": np.where(
                times["q3"].notna().values, 3, np.where(times["q2"].notna().values, 2, 1)
            ),
        }
    )
    out.loc[best.isna().values, "quali_session_reached"] = 0
    out["quali_missing"] = out["quali_best_s"].isna().astype(int)

    session_best = out["quali_best_s"].min()
    out["quali_gap_pct"] = 100.0 * (out["quali_best_s"] - session_best) / session_best

    # teammate delta (same-car comparison)
    out["quali_gap_to_teammate_pct"] = np.nan
    for team, grp in out.groupby("team"):
        if grp["quali_best_s"].notna().sum() == 2:
            a, b = grp.index[0], grp.index[1]
            ta, tb = out.at[a, "quali_best_s"], out.at[b, "quali_best_s"]
            out.at[a, "quali_gap_to_teammate_pct"] = 100.0 * (ta - tb) / tb
            out.at[b, "quali_gap_to_teammate_pct"] = 100.0 * (tb - ta) / ta

    out["season"], out["round"] = year, rnd
    return out


# --------------------------------------------------------------------------
# practice long runs
# --------------------------------------------------------------------------
def _longrun_table(session) -> pd.DataFrame:
    """Per-driver representative high-fuel stint from one practice session."""
    laps = session.laps.pick_wo_box().pick_track_status("1", how="equals")
    laps = laps[laps["LapTime"].notna()].copy()
    if "IsAccurate" in laps.columns:
        laps = laps[laps["IsAccurate"].fillna(False)]
    if laps.empty:
        return pd.DataFrame()
    laps["LapTimeS"] = laps["LapTime"].dt.total_seconds()

    rows = []
    for (drv, stint), grp in laps.groupby(["Driver", "Stint"], dropna=True):
        grp = grp.sort_values("LapNumber")
        grp = grp.iloc[DROP_FIRST_LAPS_OF_PRACTICE_STINT:]
        if len(grp) < MIN_LONGRUN_STINT_LAPS:
            continue
        rows.append(
            {
                "driver": str(drv),
                "laps": int(len(grp)),
                "median_s": float(grp["LapTimeS"].median()),
                "std_s": float(grp["LapTimeS"].std(ddof=0)),
            }
        )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(["driver", "laps"], ascending=[True, False])
    best = df.groupby("driver").head(1).reset_index(drop=True)
    ref = best["median_s"].median()
    best["pace_delta_pct"] = 100.0 * (best["median_s"] - ref) / ref
    return best


def practice_features(year: int, rnd: int) -> pd.DataFrame:
    """FP2/FP3 long-run features. Missing sessions stay NaN - never imputed."""
    frames = []
    for ident, prefix in (("FP2", "fp2"), ("FP3", "fp3")):
        s = load_session(year, rnd, ident)
        if s is None:
            continue
        tbl = _longrun_table(s)
        if tbl.empty:
            continue
        tbl = tbl.rename(
            columns={
                "pace_delta_pct": f"{prefix}_longrun_pace_delta_pct",
                "std_s": f"{prefix}_longrun_variability_s",
                "laps": f"{prefix}_longrun_laps",
            }
        )[
            [
                "driver",
                f"{prefix}_longrun_pace_delta_pct",
                f"{prefix}_longrun_variability_s",
                f"{prefix}_longrun_laps",
            ]
        ]
        frames.append(tbl)

    if not frames:
        return pd.DataFrame(columns=["driver"])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="driver", how="outer")
    return out


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------
def grid_from_race_session(year: int, rnd: int) -> pd.DataFrame:
    """Historical grid. GridPosition is pre-race information (published Saturday),
    so using it for a past race is not leakage. For Madring it does not exist yet
    and comes from data/madring_grid.csv instead."""
    s = load_session(year, rnd, "R")
    if s is None or "GridPosition" not in s.results.columns:
        return pd.DataFrame(columns=["driver", "grid_position"])
    res = s.results
    return pd.DataFrame(
        {
            "driver": res["Abbreviation"].astype(str).values,
            "grid_position": res["GridPosition"].astype(float).values,
        }
    ).replace({"grid_position": {0.0: np.nan}})


def load_madring_grid(path, quali: pd.DataFrame) -> pd.DataFrame:
    """Read the official provisional grid CSV and resolve driver codes against the
    qualifying session's own abbreviations (no hard-coded three-letter codes)."""
    grid = pd.read_csv(path)
    required = {
        "driver",
        "team",
        "grid_position",
        "qualifying_position",
        "grid_penalty",
        "status",
        "source_url",
        "retrieved_utc",
    }
    missing = required - set(grid.columns)
    if missing:
        raise ValueError(f"madring_grid.csv missing columns: {sorted(missing)}")

    code_by_last = {}
    if not quali.empty:
        for code in quali["driver"].astype(str):
            code_by_last[code.upper()] = code

    def resolve(row) -> str:
        last = str(row["driver"]).split()[-1].upper()
        for key, code in code_by_last.items():
            if last.startswith(key[:3]) or key.startswith(last[:3]):
                return code
        return last[:3]

    grid["driver_code"] = grid.apply(resolve, axis=1)
    dupes = grid["driver_code"][grid["driver_code"].duplicated()].tolist()
    if dupes:
        raise ValueError(
            f"Ambiguous driver-code resolution for {dupes}. Fix data/madring_grid.csv "
            "by adding an explicit driver_code column entry."
        )
    return grid


# --------------------------------------------------------------------------
# form features (strictly prior races)
# --------------------------------------------------------------------------
def _ewma_prior(values: pd.Series, halflife: float) -> pd.Series:
    """EWMA of strictly previous observations (shift(1) BEFORE the mean)."""
    return values.shift(1).ewm(halflife=halflife, adjust=False).mean()


def add_form_features(panel: pd.DataFrame) -> pd.DataFrame:
    """panel must contain: driver, team, round, race_date, race_pace_delta_pct,
    quali_gap_pct. Rows for the race being predicted may have a NaN target; those
    rows still receive form features computed from earlier races only."""
    df = panel.sort_values(["race_date", "driver"]).copy()

    df["team_recent_race_pace_delta"] = (
        df.groupby("team", group_keys=False)["race_pace_delta_pct"]
        .apply(lambda s: _ewma_prior(s, FORM_HALFLIFE_RACES))
    )
    df["driver_recent_race_pace_delta_raw"] = (
        df.groupby("driver", group_keys=False)["race_pace_delta_pct"]
        .apply(lambda s: _ewma_prior(s, FORM_HALFLIFE_RACES))
    )
    df["races_of_history"] = (
        df.groupby("driver")["race_pace_delta_pct"]
        .apply(lambda s: s.shift(1).notna().cumsum())
        .reset_index(level=0, drop=True)
        .astype(float)
    )

    # shrink driver form toward team form when the driver has little history
    w = df["races_of_history"] / (df["races_of_history"] + DRIVER_SHRINKAGE_RACES)
    df["driver_recent_race_pace_delta"] = (
        w * df["driver_recent_race_pace_delta_raw"].fillna(df["team_recent_race_pace_delta"])
        + (1 - w) * df["team_recent_race_pace_delta"]
    )

    conv = df["race_pace_delta_pct"] - df["quali_gap_pct"]
    df["_conv"] = conv
    df["driver_quali_to_race_conversion"] = (
        df.groupby("driver", group_keys=False)["_conv"]
        .apply(lambda s: _ewma_prior(s, FORM_HALFLIFE_RACES))
    )

    # identical definition for training and prediction rows: drivers without a
    # qualifying time rank last
    df["grid_minus_quali"] = df["grid_position"] - df.groupby("round")["quali_gap_pct"].rank(
        method="min", na_option="bottom"
    )
    df = df.drop(columns=["_conv", "driver_recent_race_pace_delta_raw"])
    return df


def build_panel(
    history: pd.DataFrame, rounds: list[int], season: int = SEASON
) -> pd.DataFrame:
    """Join targets + pre-race features for every historical race."""
    blocks = []
    for rnd in rounds:
        q = quali_features(season, rnd)
        if q.empty:
            print(f"  R{rnd:02d}: no qualifying data, race skipped")
            continue
        p = practice_features(season, rnd)
        g = grid_from_race_session(season, rnd)

        block = q.merge(p, on="driver", how="left").merge(g, on="driver", how="left")
        tgt = history[history["round"] == rnd][
            ["driver", "race_pace_delta_pct", "race_date", "n_eligible_laps"]
        ]
        block = block.merge(tgt, on="driver", how="left")
        if block["race_date"].isna().all():
            continue
        block["race_date"] = block["race_date"].ffill().bfill()
        blocks.append(block)

    if not blocks:
        return pd.DataFrame()

    panel = pd.concat(blocks, ignore_index=True)
    for col in (
        "fp2_longrun_pace_delta_pct",
        "fp3_longrun_pace_delta_pct",
        "fp2_longrun_laps",
        "fp3_longrun_laps",
        "fp2_longrun_variability_s",
        "fp3_longrun_variability_s",
    ):
        if col not in panel.columns:
            panel[col] = np.nan
    panel["longrun_missing"] = (
        panel["fp2_longrun_pace_delta_pct"].isna()
        & panel["fp3_longrun_pace_delta_pct"].isna()
    ).astype(int)

    panel = add_form_features(panel)
    return panel
