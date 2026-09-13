"""Build the historical driver-race dataset and the race_pace_delta_pct target.

Target definition (frozen for V1):

    race_pace_delta_pct = 100 * (rep_lap_driver - field_reference) / field_reference

where rep_lap_driver is the MEDIAN of that driver's eligible race laps and
field_reference is the MEDIAN across drivers of rep_lap_driver in the same race.

Eligible lap = accurate, not deleted, not a pit in/out lap, green flag for the
whole lap (TrackStatus == '1'), lap number >= 3, not within the first 2 laps of a
stint, and within 107% of that driver's own race median.

V1 does NOT apply a gap-to-car-ahead traffic filter (user decision). Residual
traffic contamination is a documented limitation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DROP_FIRST_LAPS_OF_STINT,
    MIN_ELIGIBLE_LAPS,
    MIN_LAP_NUMBER,
    OUTLIER_RATIO,
    SEASON,
)
from fetch_data import load_session


def _seconds(td: pd.Series) -> pd.Series:
    return td.dt.total_seconds()


def eligible_race_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Apply every eligibility rule except the driver-relative outlier cut."""
    df = laps.pick_wo_box().pick_track_status("1", how="equals")
    df = df[df["LapTime"].notna()].copy()

    if "IsAccurate" in df.columns:
        df = df[df["IsAccurate"].fillna(False)]
    if "Deleted" in df.columns:
        df = df[~df["Deleted"].fillna(False)]

    df = df[df["LapNumber"] >= MIN_LAP_NUMBER]

    # drop the first N laps of each stint (out-lap warm-up, tyre switch-on)
    if "TyreLife" in df.columns and df["TyreLife"].notna().any():
        df = df[df["TyreLife"].fillna(99) > DROP_FIRST_LAPS_OF_STINT]
    else:
        first = df.groupby(["Driver", "Stint"])["LapNumber"].transform("min")
        df = df[df["LapNumber"] >= first + DROP_FIRST_LAPS_OF_STINT]

    df["LapTimeS"] = _seconds(df["LapTime"])
    return df


def _driver_rep_pace(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drv, grp in df.groupby("Driver"):
        med0 = grp["LapTimeS"].median()
        kept = grp[grp["LapTimeS"] <= med0 * OUTLIER_RATIO]
        if len(kept) < MIN_ELIGIBLE_LAPS:
            continue
        rows.append(
            {
                "driver": drv,
                "team": str(kept["Team"].mode().iat[0]) if kept["Team"].notna().any() else None,
                "rep_lap_s": float(kept["LapTimeS"].median()),
                "n_eligible_laps": int(len(kept)),
                "lap_std_s": float(kept["LapTimeS"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def race_targets(year: int, rnd: int) -> tuple[pd.DataFrame, dict]:
    """Return (driver-race target rows, race metadata) for one race."""
    s = load_session(year, rnd, "R")
    if s is None:
        return pd.DataFrame(), {"round": rnd, "available": False}

    laps = s.laps
    elig = eligible_race_laps(laps)
    rep = _driver_rep_pace(elig)
    if rep.empty:
        return pd.DataFrame(), {"round": rnd, "available": False, "reason": "no eligible laps"}

    field_ref = float(rep["rep_lap_s"].median())
    rep["race_pace_delta_pct"] = 100.0 * (rep["rep_lap_s"] - field_ref) / field_ref
    rep["season"] = year
    rep["round"] = rnd
    rep["race_date"] = pd.Timestamp(s.date)
    rep["event"] = str(s.event["EventName"])

    meta = {
        "round": rnd,
        "available": True,
        "event": str(s.event["EventName"]),
        "format": str(s.event["EventFormat"]),
        "race_date": str(s.date),
        "field_reference_lap_s": field_ref,
        "n_drivers_with_target": int(len(rep)),
        "total_laps": int(laps["LapNumber"].max()),
        "sc_laps": int(
            laps.loc[laps["TrackStatus"].astype(str).str.contains("4", na=False), "LapNumber"]
            .nunique()
        ),
        "vsc_laps": int(
            laps.loc[laps["TrackStatus"].astype(str).str.contains("6", na=False), "LapNumber"]
            .nunique()
        ),
    }
    return rep, meta


def build_history(rounds: list[int], season: int = SEASON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Driver-race targets for every usable race, plus a race-meta frame."""
    frames, metas = [], []
    for rnd in rounds:
        df, meta = race_targets(season, rnd)
        metas.append(meta)
        if not df.empty:
            frames.append(df)
            print(
                f"  R{rnd:02d} {meta['event'][:26]:26s} "
                f"drivers={meta['n_drivers_with_target']:2d} "
                f"ref={meta['field_reference_lap_s']:.3f}s"
            )
        else:
            print(f"  R{rnd:02d} no target rows ({meta.get('reason', 'unavailable')})")

    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    meta_df = pd.DataFrame(metas)
    if not history.empty:
        history = history.sort_values(["race_date", "driver"]).reset_index(drop=True)
    return history, meta_df


def target_summary(history: pd.DataFrame) -> dict:
    if history.empty:
        return {}
    y = history["race_pace_delta_pct"]
    return {
        "rows": int(len(history)),
        "races": int(history["round"].nunique()),
        "drivers": int(history["driver"].nunique()),
        "target_mean_pct": float(y.mean()),
        "target_std_pct": float(y.std(ddof=0)),
        "target_min_pct": float(y.min()),
        "target_max_pct": float(y.max()),
        "target_mad_pct": float(np.abs(y - y.median()).mean()),
    }
