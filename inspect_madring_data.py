"""
Step 1: inspect what FastF1 actually exposes for the 2026 Spanish GP (Madring)
weekend and for the 2026 races that would form the training set.

This script does NOT model anything. It only reports what exists, what is
missing, and what is unusable, so the methodology can be finalised against
reality rather than against assumptions.

Run:  python src/inspect_madring_data.py
Out:  data/inspection_report.json  + a printed summary
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import fastf1
import pandas as pd

warnings.filterwarnings("ignore")

SEASON = 2026
MADRING_ROUND = 14
WEEKEND_SESSIONS = ["FP1", "FP2", "FP3", "Q"]
HISTORY_ROUNDS = range(1, 14)  # rounds already raced in 2026

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
DATA = ROOT / "data"


def setup() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))


def load(year: int, rnd: int, ident: str):
    """Load a session, returning None if the API has nothing for it."""
    try:
        s = fastf1.get_session(year, rnd, ident)
        s.load(laps=True, telemetry=False, weather=True, messages=True)
        if s.laps is None or len(s.laps) == 0:
            return None
        return s
    except Exception as exc:  # noqa: BLE001 - we want the reason, not a crash
        print(f"  ! {year} R{rnd} {ident}: {type(exc).__name__}: {exc}")
        return None


def null_rates(laps: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        if c in laps.columns:
            out[c] = round(float(laps[c].isna().mean()), 4)
        else:
            out[c] = "COLUMN_ABSENT"
    return out


def stint_summary(laps: pd.DataFrame) -> dict:
    """Longest green-flag, non-box stint per driver - the long-run evidence."""
    clean = laps.pick_wo_box().pick_track_status("1", how="equals")
    rows = []
    for (drv, stint), grp in clean.groupby(["Driver", "Stint"], dropna=True):
        grp = grp.dropna(subset=["LapTime"])
        if len(grp) == 0:
            continue
        rows.append(
            {
                "driver": drv,
                "stint": float(stint),
                "laps": int(len(grp)),
                "compound": str(grp["Compound"].mode().iat[0])
                if grp["Compound"].notna().any()
                else None,
                "median_lap_s": round(
                    float(grp["LapTime"].dt.total_seconds().median()), 3
                ),
                "lap_std_s": round(
                    float(grp["LapTime"].dt.total_seconds().std(ddof=0)), 3
                )
                if len(grp) > 1
                else None,
            }
        )
    if not rows:
        return {"drivers_with_stints": 0, "longest_stints": []}
    df = pd.DataFrame(rows).sort_values(["driver", "laps"], ascending=[True, False])
    longest = df.groupby("driver").head(1).sort_values("laps", ascending=False)
    return {
        "drivers_with_stints": int(df["driver"].nunique()),
        "stints_5plus_laps": int((df["laps"] >= 5).sum()),
        "longest_stints": longest.to_dict(orient="records"),
    }


def inspect_session(year: int, rnd: int, ident: str) -> dict:
    print(f"\n=== {year} R{rnd} {ident} ===")
    s = load(year, rnd, ident)
    if s is None:
        print("  no usable data returned")
        return {"available": False}

    laps = s.laps
    report = {
        "available": True,
        "event": str(s.event["EventName"]),
        "session_date_utc": str(s.date),
        "n_laps": int(len(laps)),
        "n_drivers": int(laps["Driver"].nunique()),
        "columns_present": sorted(laps.columns.tolist()),
        "null_rates": null_rates(
            laps,
            [
                "LapTime",
                "Sector1Time",
                "Sector2Time",
                "Sector3Time",
                "Compound",
                "TyreLife",
                "FreshTyre",
                "TrackStatus",
                "Position",
                "SpeedI1",
                "SpeedST",
                "Deleted",
                "IsAccurate",
            ],
        ),
        "compounds_seen": sorted(
            [c for c in laps["Compound"].dropna().unique().tolist()]
        ),
        "track_status_codes_seen": sorted(
            set("".join(laps["TrackStatus"].dropna().astype(str).tolist()))
        ),
        "accurate_lap_share": round(float(laps["IsAccurate"].mean()), 4)
        if "IsAccurate" in laps
        else None,
        "stints": stint_summary(laps),
    }

    # weather
    try:
        w = s.weather_data
        report["weather"] = {
            "rows": int(len(w)),
            "columns": sorted(w.columns.tolist()),
            "any_rainfall": bool(w["Rainfall"].any()) if "Rainfall" in w else None,
            "air_temp_c": [
                round(float(w["AirTemp"].min()), 1),
                round(float(w["AirTemp"].max()), 1),
            ]
            if "AirTemp" in w
            else None,
            "track_temp_c": [
                round(float(w["TrackTemp"].min()), 1),
                round(float(w["TrackTemp"].max()), 1),
            ]
            if "TrackTemp" in w
            else None,
        }
    except Exception as exc:  # noqa: BLE001
        report["weather"] = {"error": f"{type(exc).__name__}: {exc}"}

    # results (quali classification / grid)
    try:
        res = s.results
        keep = [
            c
            for c in ["Abbreviation", "TeamName", "Position", "Q1", "Q2", "Q3",
                      "GridPosition", "ClassifiedPosition", "Status"]
            if c in res.columns
        ]
        report["results"] = {
            "rows": int(len(res)),
            "columns_kept": keep,
            "no_time_set": [
                str(r["Abbreviation"])
                for _, r in res.iterrows()
                if all(pd.isna(r.get(q)) for q in ("Q1", "Q2", "Q3"))
            ]
            if {"Q1", "Q2", "Q3"}.issubset(res.columns)
            else None,
        }
        if keep:
            print(res[keep].to_string())
    except Exception as exc:  # noqa: BLE001
        report["results"] = {"error": f"{type(exc).__name__}: {exc}"}

    # telemetry availability probe on one driver, one lap
    try:
        fast = laps.pick_quicklaps().pick_fastest()
        tel = fast.get_car_data()
        report["telemetry_probe"] = {
            "driver": str(fast["Driver"]),
            "samples": int(len(tel)),
            "channels": sorted(tel.columns.tolist()),
        }
    except Exception as exc:  # noqa: BLE001
        report["telemetry_probe"] = {"error": f"{type(exc).__name__}: {exc}"}

    print(
        f"  laps={report['n_laps']} drivers={report['n_drivers']} "
        f"compounds={report['compounds_seen']} "
        f"stints>=5laps={report['stints'].get('stints_5plus_laps')}"
    )
    return report


def inspect_history() -> dict:
    """How much usable 2026 race-pace data exists for training."""
    out = {}
    for rnd in HISTORY_ROUNDS:
        s = load(SEASON, rnd, "R")
        if s is None:
            out[rnd] = {"available": False}
            continue
        laps = s.laps
        green = laps.pick_wo_box().pick_track_status("1", how="equals")
        green = green.dropna(subset=["LapTime"])
        per_driver = green.groupby("Driver")["LapTime"].size()
        out[rnd] = {
            "available": True,
            "event": str(s.event["EventName"]),
            "format": str(s.event["EventFormat"]),
            "date_utc": str(s.date),
            "n_laps_total": int(len(laps)),
            "n_green_clean_laps": int(len(green)),
            "drivers_with_25plus_clean_laps": int((per_driver >= 25).sum()),
            "median_clean_laps_per_driver": float(per_driver.median()),
            "sc_or_vsc_present": bool(
                laps["TrackStatus"]
                .dropna()
                .astype(str)
                .str.contains("4|6", regex=True)
                .any()
            ),
        }
        print(
            f"  R{rnd:02d} {out[rnd]['event'][:28]:28s} "
            f"format={out[rnd]['format']:18s} "
            f"clean_laps={out[rnd]['n_green_clean_laps']:5d} "
            f"drivers>=25={out[rnd]['drivers_with_25plus_clean_laps']}"
        )
    return out


def main() -> None:
    setup()

    print("2026 calendar as exposed by FastF1:")
    sched = fastf1.get_event_schedule(SEASON)
    print(
        sched.loc[
            sched["RoundNumber"] > 0,
            ["RoundNumber", "Location", "EventName", "EventFormat"],
        ].to_string(index=False)
    )

    report = {
        "fastf1_version": fastf1.__version__,
        "madring_weekend": {
            ident: inspect_session(SEASON, MADRING_ROUND, ident)
            for ident in WEEKEND_SESSIONS
        },
    }

    print("\n=== 2026 completed races (candidate training set) ===")
    report["history_2026"] = inspect_history()

    path = DATA / "inspection_report.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
