"""FastF1 access layer: cache setup, safe session loading, availability gate.

Nothing in this module invents data. If a session is not available from the API,
it returns None and the caller decides what to do.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import fastf1

from config import CACHE, MADRING_ROUND, SEASON

warnings.filterwarnings("ignore")


def setup_cache() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))


def load_session(year: int, rnd: int, ident: str, *, telemetry: bool = False):
    """Load one session. Returns None if the API has no usable lap data for it."""
    try:
        s = fastf1.get_session(year, rnd, ident)
        s.load(laps=True, telemetry=telemetry, weather=True, messages=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {year} R{rnd} {ident}: {type(exc).__name__}: {exc}")
        return None
    try:
        if s.laps is None or len(s.laps) == 0:
            return None
    except Exception:  # noqa: BLE001
        return None
    return s


def event_format(year: int, rnd: int) -> str:
    sched = fastf1.get_event_schedule(year)
    row = sched.loc[sched["RoundNumber"] == rnd]
    return str(row["EventFormat"].iat[0]) if len(row) else "unknown"


@dataclass
class AvailabilityReport:
    madring: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)

    @property
    def quali_ok(self) -> bool:
        return bool(self.madring.get("Q", {}).get("available"))

    @property
    def practice_sessions_ok(self) -> list[str]:
        return [
            ident
            for ident in ("FP1", "FP2", "FP3")
            if self.madring.get(ident, {}).get("available")
        ]

    @property
    def usable_history_rounds(self) -> list[int]:
        return [r for r, v in self.history.items() if v.get("available")]

    def blocking_problems(self) -> list[str]:
        problems = []
        if not self.quali_ok:
            problems.append(
                "Madring qualifying data is not available from FastF1. "
                "quali_gap_pct, quali_gap_to_teammate_pct and quali_session_reached "
                "cannot be built, which removes the strongest feature block. STOP."
            )
        if len(self.usable_history_rounds) < 6:
            problems.append(
                f"Only {len(self.usable_history_rounds)} of the 13 completed 2026 races "
                "returned usable race laps. Walk-forward validation on 2026 alone is not "
                "credible below ~6 usable races. STOP or switch to a pooled window."
            )
        return problems

    def warnings(self) -> list[str]:
        out = []
        if not self.practice_sessions_ok:
            out.append(
                "No Madring practice session returned lap data: the FP2/FP3 long-run "
                "feature block will be all-NaN for the prediction row."
            )
        elif "FP2" not in self.practice_sessions_ok:
            out.append("Madring FP2 unavailable: fp2 long-run features will be NaN.")
        elif "FP3" not in self.practice_sessions_ok:
            out.append("Madring FP3 unavailable: fp3 long-run features will be NaN.")
        return out


def summarise_session(s) -> dict:
    laps = s.laps
    return {
        "available": True,
        "n_laps": int(len(laps)),
        "n_drivers": int(laps["Driver"].nunique()),
        "compounds": sorted(laps["Compound"].dropna().unique().tolist()),
        "track_status_codes": sorted(
            set("".join(laps["TrackStatus"].dropna().astype(str).tolist()))
        ),
    }


def check_availability(history_rounds: list[int]) -> AvailabilityReport:
    """The gate. Run before any dataset construction."""
    setup_cache()
    rep = AvailabilityReport()

    print("Checking Madring weekend sessions...")
    for ident in ("FP1", "FP2", "FP3", "Q"):
        s = load_session(SEASON, MADRING_ROUND, ident)
        rep.madring[ident] = summarise_session(s) if s is not None else {"available": False}
        state = "ok" if rep.madring[ident]["available"] else "MISSING"
        extra = (
            f" laps={rep.madring[ident]['n_laps']} drivers={rep.madring[ident]['n_drivers']}"
            if rep.madring[ident]["available"]
            else ""
        )
        print(f"  {ident}: {state}{extra}")

    print("Checking 2026 completed races...")
    for rnd in history_rounds:
        s = load_session(SEASON, rnd, "R")
        if s is None:
            rep.history[rnd] = {"available": False}
            print(f"  R{rnd:02d}: MISSING")
            continue
        rep.history[rnd] = {
            "available": True,
            "event": str(s.event["EventName"]),
            "format": str(s.event["EventFormat"]),
            "date_utc": str(s.date),
            "n_laps": int(len(s.laps)),
        }
        print(f"  R{rnd:02d}: ok  {rep.history[rnd]['event'][:30]}")

    return rep
