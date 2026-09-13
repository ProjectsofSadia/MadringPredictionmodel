"""Central configuration. Every threshold used anywhere in the pipeline lives here
so that the README and the frozen prediction JSON can quote it exactly."""

from __future__ import annotations

from pathlib import Path

# Works whether this file sits in <repo>/src/config.py or flat in <repo>/config.py,
# and finds the data files whether they are in <repo>/data/ or next to the code.
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if _HERE.name == "src" else _HERE
SRC = _HERE


def _locate(filename: str, default: Path) -> Path:
    for candidate in (ROOT / "data", ROOT, SRC):
        if (candidate / filename).exists():
            return candidate
    return default


DATA = _locate("madring_grid.csv", ROOT / "data")
ASSUMPTIONS_PATH = _locate("simulation_assumptions.yaml", SRC) / "simulation_assumptions.yaml"

CACHE = ROOT / "cache"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

SEASON = 2026
MADRING_ROUND = 14
HISTORY_ROUNDS = list(range(1, 14))  # rounds already raced in 2026 before Madrid

SEED = 42
N_SIMULATIONS = 10_000

# --- lap eligibility (target construction) -------------------------------
MIN_LAP_NUMBER = 3  # drop the start and the lap-1/2 scramble
DROP_FIRST_LAPS_OF_STINT = 2  # out-lap warm-up + tyre switch-on
OUTLIER_RATIO = 1.07  # drop laps slower than 107% of the driver's own race median
MIN_ELIGIBLE_LAPS = 20  # below this, the driver-race produces no target

# --- practice long runs --------------------------------------------------
MIN_LONGRUN_STINT_LAPS = 5
DROP_FIRST_LAPS_OF_PRACTICE_STINT = 2

# --- form features -------------------------------------------------------
FORM_HALFLIFE_RACES = 3.0
DRIVER_SHRINKAGE_RACES = 3.0  # weight driver form vs team form for low-history drivers

# --- validation ----------------------------------------------------------
MIN_TRAIN_ROUNDS = 4  # first validated round is round 5

# V1 decision (user, 2026-09-12): Option A training window = 2026 only.
# fetch_data/build_dataset take a `seasons` argument so Options B and C can be
# added later by passing more seasons; nothing else in the pipeline changes.
TRAINING_SEASONS = [2026]
