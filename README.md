# MADRING Race Prediction Simulator

Can pre-race information from the 2026 Spanish Grand Prix weekend, plus 2026 season form,
be used to estimate race outcomes at a circuit that has never hosted a Formula 1 race?

This is V1: a deliberately small, defensible pipeline built to produce a **timestamped
pre-race prediction** for the inaugural race at Madring on 13 September 2026.

**Nothing in this repository contains a hand-written prediction number.** The results
sections below stay empty until `run_pipeline.py` has actually run. If you are reading
this and the numbers are missing, the pipeline has not been run on your machine yet.

---

## Why I built it

Most public F1 "prediction" projects predict the winner directly from a classifier trained
on finishing positions, which mostly learns the current championship order. I wanted the
opposite: a model that predicts a continuous, physically interpretable quantity (race pace),
with a separate simulation layer that turns pace into outcome probabilities, and an honest
answer to whether the machine-learning step adds anything at all over a qualifying-based
baseline.

Madring is the interesting case because it removes the crutch. The circuit is new, so no
model can lean on circuit history. Everything has to come from current form and this
weekend's own sessions.

---

## Data

**MEASURED / PUBLIC DATA** (FastF1, which serves Formula 1's public timing and broadcast car
feed - not team telemetry):

- Lap times, sector times, speed-trap values, stint and tyre-compound information, tyre age,
  track status, and session weather logs for 2026 rounds 1-13 and the Madring weekend.

**EXTERNAL DOCUMENTED INPUTS** (not from FastF1, recorded with source and retrieval time):

- `data/madring_grid.csv` - the official provisional starting grid including penalties and
  the two cars that set no qualifying time. The grid is not inferred from qualifying.
- `data/madring_race_inputs.yaml` - race lap count and circuit length, with the source-level
  disagreement about lap count documented rather than silently resolved.

FastF1 does **not** expose battery state of charge, MGU-K deployment, energy recovery, brake
torque or any team-side channel, and none of those appear anywhere in this project.

---

## Prediction target

```
race_pace_delta_pct = 100 * (driver's median eligible clean race lap - field reference) / field reference
```

Field reference = the median across drivers of that same quantity in the same race.

A lap is eligible if it is accurate, not deleted, not a pit in- or out-lap, run entirely
under green flag (`TrackStatus == '1'`), lap 3 or later, more than two laps into its stint,
and within 107% of that driver's own race median. A driver needs at least 20 eligible laps
to contribute a target.

Percent rather than seconds, because lap times differ by tens of seconds between circuits and
one second means something different at each. The field median rather than the winner's pace,
because the winner frequently manages pace and is a noisy anchor.

V1 does **not** apply a gap-to-car-ahead traffic filter. Residual traffic contamination is a
known limitation (see below).

---

## Features

All pre-race. Every feature is declared in `PROVENANCE` in `src/features.py` with the session
it comes from, and `tests/test_leakage.py` fails if anything is sourced from the race being
predicted.

**DERIVED FEATURES**

- Qualifying: gap to session best (%), gap to teammate (%), session reached, missing-time flag
- Grid: starting position, grid minus qualifying rank (penalty size)
- Practice long runs: FP2 and FP3 representative high-fuel stint pace delta (%), variability,
  lap count, and a missing flag
- Season form (strictly prior races, EWMA with `shift(1)`): team race pace, driver race pace
  shrunk toward team form, qualifying-to-race conversion, races of history

Driver and team identity are deliberately **not** features. With ~250 rows a model given IDs
memorises the 2026 pecking order instead of learning a mapping.

Race-day weather is excluded: FastF1 only logs weather during a session, so using race-session
weather on historical rows would be leakage.

---

## Baselines and model

Every model is evaluated on identical walk-forward folds:

- **B0 null** - predict the field median for everyone
- **B1 qualifying** - OLS on qualifying gap
- **B2 recent form** - team EWMA prior race pace
- **B3 ridge** - ridge on qualifying gap, grid, and team/driver form
- **XGBoost** - `max_depth` 2-3, learning rate 0.03-0.05, `min_child_weight` 5-10,
  `subsample`/`colsample_bytree` 0.8, `reg_lambda` 1-5, pseudo-Huber loss, 12-configuration
  tuning grid searched with a nested walk-forward split inside the training data only

**Selection rule: lowest out-of-fold MAE wins, XGBoost included or not.** Ties within 0.01
percentage points go to the simpler model. If ridge or qualifying-only wins, that model feeds
the simulator and this README says so.

### Validation methodology

Expanding-window walk-forward by race date. Fold k trains on every race before round k and
validates on round k. Whole races only, no shuffling, no random splits. First validated round
is round 5, so that form features have history behind them. Reported per fold and aggregated:
MAE, RMSE, R², within-race Spearman rank correlation, and skill score against the null model.

### Results: model validation

<!-- populated from outputs/predictions_madring_<timestamp>.json -->
_Empty until `run_pipeline.py` has been run. See `figures/01_model_validation.png` and the
`validation` block of the frozen prediction JSON._

---

## Uncertainty

The model output is an expected pace, not a distribution. Spread comes from the out-of-fold
residuals of the selected model, bootstrapped inside the simulator. No sigma is chosen by hand
anywhere in this project.

---

## Monte Carlo simulation

10,000 lap-by-lap simulations, fixed seed. Per simulation each driver's underlying pace is the
model prediction plus a bootstrapped residual, held constant for the race. Then: an empirical
lap-1 position shuffle, lap-to-lap noise, an estimated tyre-age lap-time trend, pit stops at
drawn laps with an estimated pit-lane loss, a queue model for track position, a per-lap DNF
hazard from 2026 team retirement rates, and Safety Car periods from a 2026-season prior.

Values estimated from 2026 race data (`src/priors.py`): lap-to-lap noise, degradation trend by
compound, pit-lane time loss, lap-1 gaps and position changes, race-pace-to-pole ratio, Safety
Car frequency, team DNF rates.

Values that are assumptions (`src/simulation_assumptions.yaml`): following gap, dirty-air
penalty, pass pace threshold, pass time cost, pit-lap jitter, Safety Car compression gap and
pit-loss discount.

**Overtaking difficulty at Madring cannot be estimated.** No F1 race has been run there. The
entire simulation is therefore repeated under LOW (p=0.10), BASE (p=0.25) and HIGH (p=0.50)
per-lap pass probability, and the headline result states how much win and podium probability
move across that range.

### Results: simulated outcomes

<!-- populated from outputs/predictions_madring_<timestamp>.json -->
_Empty until `run_pipeline.py` has been run. See `figures/02_win_probability.png`,
`figures/03_grid_vs_expected_finish.png`, and `results_by_overtaking_case` in the frozen JSON._

Output columns: win %, podium %, top-10 %, expected finish, median and P10-P90 finish,
expected positions gained, and the Monte Carlo standard error on each probability.

---

## Limitations

- **SIMULATED OUTCOMES are not predictions of the result.** They are probabilities conditional
  on a list of stated assumptions, several of which cannot be validated for this circuit.
- No traffic filter in the target, so a fast car that spent the race stuck behind a slower one
  has an artificially slow measured race pace in the training data.
- Degradation is an estimated lap-time trend against tyre age, not measured tyre degradation,
  and it is confounded with fuel burn.
- Strategy is a single uniform stop plan with jitter. No undercut modelling, no team-specific
  calls, no tyre allocation tracking.
- Training window is 2026 only (~250 driver-race rows, 8 of 13 weekends with FP2/FP3 data).
  Small. Pooling 2022-2025 is a V2 experiment, not an assumption.
- Safety Car frequency is a season prior applied to a circuit with no history.
- Two drivers set no qualifying time, so their strongest feature is missing and their
  predictions carry more uncertainty than the interval alone suggests.
- The model has never seen this circuit. Nothing in it knows what Madring rewards.

---

## How to reproduce

```bash
pip install -r requirements.txt
python src/inspect_madring_data.py      # what FastF1 actually has for this weekend
python -m pytest tests -q               # leakage + mechanical tests
python run_pipeline.py                  # full pipeline, writes the frozen prediction
```

`run_pipeline.py --inspect-only` runs the availability gate alone. The pipeline refuses to
produce a prediction if Madring qualifying is unavailable or fewer than six 2026 races return
usable race laps.

Seeds are fixed (`SEED = 42` in `src/config.py`). The frozen prediction file records the model
used, validation metrics, feature list, seed, simulation count, grid source, data retrieval
timestamp, every assumption, and the git commit hash.

**The prediction file is frozen at the moment it is written.** It is not edited after the race.
Any later model improvement produces a new file.

---

## Next steps (V2, after the prediction is frozen)

- Evaluate pooled training windows B (2022-2026 with recency weighting) and C (era-stable
  linear part plus 2026 residual correction) on the same folds
- Post-race scorecard: predicted vs actual race pace delta per driver, and whether the actual
  finishing position fell inside the simulated P10-P90 band
- Recalibrate the overtaking parameter against what the race actually showed
- SHAP explanations for individual drivers
- Gap-to-car-ahead traffic filter, once there is enough data to afford the lost laps
