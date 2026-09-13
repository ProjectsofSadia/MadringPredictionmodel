"""Every simulator input that can be estimated from data is estimated here from
the 2026 races. Nothing is Madring-specific: Madring has never hosted an F1 race.

Each returned value carries how it was produced so the frozen prediction JSON can
label it 'estimated-from-data' rather than 'assumption'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from build_dataset import eligible_race_laps
from fetch_data import load_session


def _stint_detrended_std(laps: pd.DataFrame) -> list[float]:
    """Lap-to-lap noise with the tyre-age trend removed, per driver-stint."""
    out = []
    for (_, _), grp in laps.groupby(["Driver", "Stint"], dropna=True):
        if len(grp) < 6 or grp["TyreLife"].isna().all():
            continue
        x = grp["TyreLife"].astype(float).values
        y = grp["LapTimeS"].values
        if np.std(x) == 0:
            continue
        b, a = np.polyfit(x, y, 1)
        out.append(float(np.std(y - (a + b * x), ddof=0)))
    return out


def _deg_slopes(laps: pd.DataFrame, ref: float) -> list[tuple[str, float]]:
    """Lap-time trend vs tyre age within stints, expressed in PERCENT of the race
    reference lap per lap of tyre age, so it pools across circuits."""
    out = []
    for (_, _, comp), grp in laps.groupby(["Driver", "Stint", "Compound"], dropna=True):
        if len(grp) < 8 or grp["TyreLife"].isna().all():
            continue
        x = grp["TyreLife"].astype(float).values
        y = grp["LapTimeS"].values
        if np.std(x) == 0:
            continue
        b, _ = np.polyfit(x, y, 1)
        out.append((str(comp), 100.0 * float(b) / ref))
    return out


def estimate_priors(rounds: list[int], season: int) -> dict:
    lap_noise, deg, pit_loss, gaps_lap1, lap1_delta = [], [], [], [], []
    race_to_pole, sc_flags, sc_period_counts, sc_lap_shares = [], [], [], []
    starts: dict[str, int] = {}
    retirements: dict[str, int] = {}

    for rnd in rounds:
        s = load_session(season, rnd, "R")
        if s is None:
            continue
        laps = s.laps
        elig = eligible_race_laps(laps)
        if elig.empty:
            continue
        rep = elig.groupby("Driver")["LapTimeS"].median()
        ref = float(rep.median())

        lap_noise += _stint_detrended_std(elig)
        deg += _deg_slopes(elig, ref)

        # pit-lane time loss: (in-lap + out-lap) - 2 x that driver's normal lap
        box = laps.pick_box_laps()
        for drv, grp in box.groupby("Driver"):
            if drv not in rep.index:
                continue
            t = grp["LapTime"].dropna().dt.total_seconds()
            if len(t) >= 2:
                pairs = t.values[: len(t) // 2 * 2].reshape(-1, 2).sum(axis=1)
                pit_loss += [float(p - 2 * rep[drv]) for p in pairs if 5 < p - 2 * rep[drv] < 60]

        # lap-1 spread and position shuffle
        try:
            res = s.results
            grid = dict(zip(res["Abbreviation"].astype(str), res["GridPosition"].astype(float)))
        except Exception:  # noqa: BLE001
            grid = {}
        l1 = laps[laps["LapNumber"] == 1][["Driver", "Time", "Position"]].dropna()
        if len(l1) > 5:
            t1 = np.sort(l1["Time"].dt.total_seconds().values)
            gaps_lap1 += list(np.diff(t1))
            for _, r in l1.iterrows():
                g = grid.get(str(r["Driver"]))
                if g and g > 0 and pd.notna(r["Position"]):
                    lap1_delta.append(float(r["Position"]) - float(g))

        # race pace vs pole
        q = load_session(season, rnd, "Q")
        if q is not None:
            try:
                pole = q.laps.pick_quicklaps().pick_fastest()["LapTime"].total_seconds()
                race_to_pole.append(ref / float(pole))
            except Exception:  # noqa: BLE001
                pass

        # safety car / VSC
        ts = laps["TrackStatus"].astype(str)
        total_laps = int(laps["LapNumber"].max())
        sc_laps = laps.loc[ts.str.contains("4", na=False), "LapNumber"].nunique()
        sc_flags.append(1 if sc_laps > 0 else 0)
        if sc_laps > 0:
            sc_lap_shares.append(sc_laps / max(total_laps, 1))
            try:
                status = s.track_status
                sc_period_counts.append(int((status["Status"].astype(str) == "4").sum()))
            except Exception:  # noqa: BLE001
                sc_period_counts.append(1)

        # DNFs by team
        try:
            for _, r in s.results.iterrows():
                team = str(r["TeamName"])
                starts[team] = starts.get(team, 0) + 1
                if str(r.get("ClassifiedPosition", "")) in {"R", "D", "E", "W", "N"}:
                    retirements[team] = retirements.get(team, 0) + 1
        except Exception:  # noqa: BLE001
            pass

    deg_df = pd.DataFrame(deg, columns=["compound", "pct_per_lap"])
    deg_by_compound = (
        deg_df.groupby("compound")["pct_per_lap"].median().to_dict() if len(deg_df) else {}
    )

    field_dnf = (
        sum(retirements.values()) / sum(starts.values()) if starts else 0.0
    )
    dnf_by_team = {}
    for team, n in starts.items():
        r = retirements.get(team, 0)
        # Beta-style shrinkage toward the field rate with 10 pseudo-entries
        dnf_by_team[team] = float((r + 10 * field_dnf) / (n + 10))

    return {
        "lap_noise_s": {
            "value": float(np.median(lap_noise)) if lap_noise else None,
            "n": len(lap_noise),
            "source": "estimated-from-data (within-stint detrended lap-time std, 2026 races)",
        },
        "degradation_pct_per_lap": {
            "by_compound": {k: float(v) for k, v in deg_by_compound.items()},
            "pooled": float(deg_df["pct_per_lap"].median()) if len(deg_df) else None,
            "n": int(len(deg_df)),
            "source": "estimated-from-data (lap time vs TyreLife within stints, 2026 races). "
                      "This is an estimated lap-time trend, NOT measured tyre degradation.",
        },
        "pit_loss_s": {
            "value": float(np.median(pit_loss)) if pit_loss else None,
            "n": len(pit_loss),
            "source": "estimated-from-data (in-lap + out-lap minus two normal laps, 2026 races)",
        },
        "lap1_gap_s": {
            "value": float(np.median(gaps_lap1)) if gaps_lap1 else None,
            "n": len(gaps_lap1),
            "source": "estimated-from-data (consecutive car gaps at end of lap 1, 2026 races)",
        },
        "lap1_position_delta": {
            "samples": [float(x) for x in lap1_delta],
            "n": len(lap1_delta),
            "source": "estimated-from-data (position after lap 1 minus grid, 2026 races)",
        },
        "race_pace_to_pole_ratio": {
            "value": float(np.median(race_to_pole)) if race_to_pole else None,
            "n": len(race_to_pole),
            "source": "estimated-from-data (race field-median clean lap / pole lap, 2026 races)",
        },
        "safety_car": {
            "p_race_has_sc": float(np.mean(sc_flags)) if sc_flags else None,
            "mean_periods_when_present": float(np.mean(sc_period_counts))
            if sc_period_counts
            else None,
            "mean_lap_share_when_present": float(np.mean(sc_lap_shares))
            if sc_lap_shares
            else None,
            "n_races": len(sc_flags),
            "source": "documented-prior from 2026 season TrackStatus code '4'. "
                      "No Madring-specific estimate is possible - the circuit is new.",
        },
        "dnf_probability_per_race": {
            "by_team": dnf_by_team,
            "field_rate": float(field_dnf),
            "source": "estimated-from-data (2026 classifications, shrunk toward field rate)",
        },
    }
