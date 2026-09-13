# MADRING Race Prediction Simulator — Methodology (for approval)

Status: **approved and implemented as V1. No model has been trained yet on this machine and
no prediction exists until `run_pipeline.py` has run.**

V1 scope decisions (2026-09-13):

- Training window: **Option A, 2026 only**. Options B and C are V2; `config.TRAINING_SEASONS`
  and the `seasons` arguments exist so they can be added without restructuring the pipeline.
- Traffic filter: **not enabled**. Target stays the median of eligible clean laps; residual
  traffic contamination is a documented limitation.
- Scope: defensible minimum, prioritising a genuine pre-race timestamp. No SHAP, no pooled
  window experiments, three figures rather than seven, simple strategy and Safety Car models.
- Model selection: lowest out-of-fold MAE on identical folds, XGBoost included or not.

Sections 2 (three training windows) and 11 (full simulator) below describe the full design;
what V1 implements is the subset listed above.

Question: can pre-race information from the 2026 Spanish GP weekend at Madring, plus
2026 season form, be used to estimate race outcomes with honest uncertainty?

---

## 0. What I verified, and what I could not

**Verified directly (FastF1 3.8.3 installed and inspected):**

- The `Laps` schema really contains: `LapTime`, `LapNumber`, `Stint`, `PitInTime`,
  `PitOutTime`, `Sector1/2/3Time`, `SpeedI1`, `SpeedI2`, `SpeedFL`, `SpeedST`,
  `Compound`, `TyreLife`, `FreshTyre`, `Team`, `TrackStatus`, `Position`, `Deleted`,
  `DeletedReason`, `IsAccurate`, `FastF1Generated`.
- Filtering helpers that exist: `pick_wo_box`, `pick_box_laps`, `pick_track_status`,
  `pick_quicklaps`, `pick_accurate`, `pick_not_deleted`, `pick_compounds`, `pick_tyre`.
- `TrackStatus` codes: `1` clear, `2` yellow, `4` Safety Car, `5` red flag,
  `6` VSC deployed, `7` VSC ending. This is the mechanism used to exclude SC/VSC/red-flag
  laps and to build a Safety Car prior.
- `SessionResults` exposes `Q1`, `Q2`, `Q3`, `Position`, `GridPosition`,
  `ClassifiedPosition`, `Status`, `TeamName`, `Abbreviation`.
- Weather is a **measured session log** (`AirTemp`, `TrackTemp`, `Humidity`, `Pressure`,
  `Rainfall`, `WindSpeed`, `WindDirection`). There is **no forecast** in FastF1.
- 2026 calendar: Madrid is **round 14**, `conventional` format, race **Sun 13 Sep 2026,
  13:00 UTC**. Rounds 1–13 are already raced.
- Of those 13 completed rounds, **5 are sprint weekends** (China, Miami, Canada, Silverstone,
  Zandvoort) with only FP1, and **8 are conventional** (Melbourne, Suzuka, Monaco, Barcelona,
  Austria, Spa, Hungary, Monza). This matters a lot; see §2.

**Could not verify from here:** this sandbox blocks `livetiming.formula1.com` and the
Ergast/Jolpica mirrors (403 `host_not_allowed`). So I cannot confirm that Madring FP1/FP2/FP3/Q
timing data has actually landed in the API yet, or how complete it is. That is the one thing
that has to be checked on your machine before anything else.

`src/inspect_madring_data.py` does exactly that and nothing else. Run it first. Everything
below is conditional on what it reports.

**Known facts that must be treated as external, documented inputs (not invented, not from FastF1):**

- Race distance: sources disagree. Formula1.com lists 5.416 km / 57 laps / 308.524 km;
  other outlets list 5.474 km / 53 or 57 laps. Take the lap count from the FIA event
  documents or the official timing sheet and record the source in
  `data/madring_race_inputs.yaml`. Do not guess.
- Starting grid: qualifying classification is not the grid. Grid penalties, pit-lane starts,
  and non-starters (Bearman did not set a time after his FP3 crash; Stroll did not set a time)
  must come from the official provisional starting grid and be stored in
  `data/madring_grid.csv` with the source URL and retrieval timestamp.

---

## 1. Scope, honestly stated

Madring has never hosted an F1 race. There is zero circuit history. So the model cannot know
anything circuit-specific beyond what this weekend's own sessions show. The project is
therefore:

> current car/driver form + this weekend's practice and qualifying evidence
> → expected race pace → simulated race outcomes

That is a narrower claim than "F1 race prediction", and the README should say so in the first
paragraph. It is also the only claim the data supports.

---

## 2. Training set design (the hardest decision)

2026 is a **new regulation era**: new chassis rules with active aerodynamics, new power units
with a roughly 50/50 combustion/electric split and no MGU-H, and new tyre construction. Race
pace structure, degradation behaviour and energy-management-driven lap-time shape are not the
same as 2022–2025. Pooling five seasons blindly would be the single biggest defensibility hole
in the project.

There is also a supply problem. Only 13 races exist in 2026, i.e. roughly 250–260 driver-race
rows, of which only **8 weekends have FP2/FP3 long-run data** because the other 5 are sprints.

I do not want to assert which window is right. I want the validation to decide it. Three
candidate training sets, all evaluated on **identical 2026 validation folds**:

- **A — 2026 only.** In-era, small (~250 rows, ~160 with full practice features).
- **B — 2022–2026 pooled, with an `is_2026_era` flag and exponential recency weighting**
  (half-life ≈ 1 season, applied as `sample_weight`). More data, risk of stale relationships.
- **C — 2022–2026 pooled, but with the pre-2026 rows used only to fit the simple,
  era-stable part of the mapping** (qualifying gap → race pace gap), with 2026 residual
  correction on top.

Whichever wins on the 2026 walk-forward folds is the one shipped, and the losing options are
reported in the README. If A wins, that is a real finding and is more interesting than a bigger
model.

Why pooling is defensible at all: both the target and the features are **relative, within-race
normalised** quantities. "How much of a team's qualifying advantage carries into race pace" is
far more era-stable than absolute lap time. But that is a hypothesis, and §6 tests it.

---

## 3. Unit of observation

One row per **driver-race**. Features describe what was knowable before lights out. Target
describes race pace observed in that race.

---

## 4. Target variable

**`race_pace_delta_pct`** — the driver's representative clean race pace relative to the field
in the same race, in percent:

```
race_pace_delta_pct = 100 * (rep_lap_time_driver - field_reference_lap_time) / field_reference_lap_time
```

Negative = faster than the field reference.

**Representative driver pace** = median of that driver's eligible race laps, where eligible means:

- `LapTime` present and `IsAccurate` is true
- not a pit in-lap or out-lap (`pick_wo_box`)
- `TrackStatus == '1'` for the whole lap (excludes yellow, SC, VSC, red flag)
- lap number ≥ 3 (removes the start and the lap-1 scramble)
- not the first 2 laps of a stint (out-lap warm-up and tyre switch-on)
- lap not `Deleted`
- lap time within 107% of that driver's own race median (removes damage laps, lift-and-coast
  after a mistake, and the worst traffic laps)
- optional traffic filter, see below

A driver needs **≥ 20 eligible laps** to produce a target; otherwise the row is dropped from
training (not imputed). DNFs and heavily disrupted races contribute no target.

**Field reference** = the **median across drivers** of `rep_lap_time_driver` in that race.
Not the winner's pace: the winner often manages pace and is a noisy, non-stationary anchor. The
field median is robust and does not depend on any single car.

**Why median and not a lower quantile:** a low quantile (e.g. P20 of clean laps) is closer to
"free-air potential pace" but is heavily influenced by single tow laps and post-SC restarts.
Median is the stable choice. I will still compute P20 as a secondary target and report whether
the model's ranking changes materially — that is a one-line robustness check, not a second
model.

**Traffic:** FastF1 does not expose gap-to-car-ahead. It can be derived: for each lap number,
compare the `Time` column (session time at lap end) between drivers on the same lap and use
`Position` to find the car ahead. Laps where the gap to the car ahead is under ~1.2 s get
flagged `in_traffic`. I propose computing this, reporting how many laps it removes, and
**only enabling it if it does not destroy sample size** at street circuits. Decision deferred
to the data.

**Why percent, not seconds:** lap times span roughly 70 s to 105 s across the calendar. One
second means very different things at Monaco and Spa. Percent normalisation is what makes
cross-circuit training legitimate at all.

**Fuel correction:** not applied. All drivers run a comparable fuel profile over the same race
distance, so the common fuel effect largely cancels in a within-race relative measure. This is
an explicit simplification, stated in the README, not a silent one.

---

## 5. Features (pre-race only), with the reason for each

Each feature is tagged with its earliest availability. Nothing from the race session is ever a
feature.

**From qualifying (Saturday):**

- `quali_gap_pct` — driver's best qualifying lap vs session best, in percent. Strongest single
  signal of current car+driver performance. Best available lap across Q1/Q2/Q3.
- `quali_gap_to_teammate_pct` — same-car comparison. Separates driver form from car form, which
  the model otherwise cannot see without ID features.
- `quali_session_reached` (1/2/3) — captures whether the Q-time was set with a low-fuel
  Q3-spec run or a Q1 run, which changes what the gap means.
- `quali_missing` (bool) — real and relevant this weekend: two cars set no time.
- `grid_position` — from the **official starting grid**, including penalties. Track position is
  decisive at a tight new street-style circuit; this is also the simulator's starting state.
- `grid_minus_quali` — penalty size, i.e. how far the grid is displaced from raw pace.

**From practice long runs (Friday/Saturday):**

- `fp2_longrun_pace_delta_pct`, `fp3_longrun_pace_delta_pct` — median lap of the driver's
  longest eligible high-fuel stint (≥ 5 consecutive green, non-box laps, first 2 laps of the
  stint dropped), expressed as a percent delta vs the field median of the same quantity. This
  is the only pre-race observation that is actually a *race-pace* observation.
- `longrun_pace_variability` — standard deviation of those stint laps. Consistency proxy and
  partly a data-quality proxy.
- `longrun_laps_count` — how much evidence supports the two features above. Lets the model
  discount thin evidence instead of trusting a 5-lap run as much as a 15-lap run.
- `longrun_compound` — categorical (SOFT/MEDIUM/HARD, relative naming as FastF1 gives it, not
  C1–C6). A long-run delta is meaningless without knowing the compound.
- `longrun_missing` (bool) — Norris lost most of FP2 to a gearbox problem, Lindblad crashed in
  FP2, Bearman crashed in FP3. Missingness is informative and must be modelled, not hidden.
  XGBoost handles NaN natively; the flag makes it explicit.

Sprint weekends in the training set have no FP2/FP3. Those rows will carry NaN on the practice
block. This is a real reason to prefer option A/B with native NaN handling over any imputation
scheme.

**From season form (strictly prior races only):**

- `team_recent_race_pace_delta` — exponentially weighted mean of the team's
  `race_pace_delta_pct` over previous races (half-life ≈ 3 races).
- `driver_recent_race_pace_delta` — same at driver level, shrunk toward the team value with a
  weight proportional to the number of prior races the driver has (protects rookies:
  Lindblad, Bortoleto, the Cadillac pairing).
- `driver_quali_to_race_conversion` — prior mean of (`race_pace_delta_pct` −
  `quali_gap_pct`). Captures "qualifies better than it races" and vice versa.
- `races_of_history` — how many prior races back the form features. Same logic as
  `longrun_laps_count`.

**Weather (measured, pre-race sessions only):**

- `fp_track_temp_mean`, `quali_track_temp_mean`, `track_temp_delta_fp2_to_q`,
  `rain_in_any_practice` (bool). Justification is thermal sensitivity of tyre behaviour and
  the fact that practice run plans are invalidated by a wet session.
- **Race-day weather is excluded as a feature.** FastF1 only logs weather *during* a session,
  so using race-session weather on historical rows would be textbook leakage. If you want a
  wet-race scenario at Madring, it belongs in the simulator as an explicitly labelled scenario
  input, not in the model.

**Circuit characteristics:**

Madring has no history, so any circuit feature must be computable **from the current weekend's
own sessions**. Candidates: pole lap time, `pct_full_throttle` from the pole-lap telemetry,
median top speed (`SpeedST`), estimated pit-lane time loss. I propose including **at most two**
and dropping them if walk-forward validation does not improve. With ~250 rows, every extra
column is a real overfitting cost.

**Deliberately excluded:**

- Driver and team one-hot IDs or label encodings. With 20 drivers and ~250 rows the model would
  memorise the 2026 pecking order rather than learn a mapping, and it would generalise to
  nothing. The form features carry the same information in a continuous, decaying way.
- Championship standings and points. Mostly a lagging restatement of pace, adds collinearity.
- Anything derived from the race session: finishing position, race fastest lap, actual pit
  stops, actual tyre allocation used, actual SC timing.

---

## 6. Leakage register and how it is enforced

Explicit rules:

1. Every feature column must be traceable to a session whose **end time is strictly before the
   Madring race start**, and for historical rows, before that row's race start.
2. Rolling form features are computed after sorting by race date and applying `shift(1)` per
   entity. An expanding window, never a full-season aggregate.
3. Whole races are assigned to train or validation. Never split a race across folds — the
   target is normalised within the race, so splitting leaks the reference.
4. No hyperparameter, threshold, or feature choice is ever selected using Madring's outcome.
   The race has not happened; once it has, nothing gets re-tuned.
5. The field-reference normalisation is computed per race from that race's own laps. This is
   target construction, not feature leakage, but it means the model predicts a **relative**
   quantity, and the Madring predictions should be re-centred so the field mean is ~0 before
   feeding the simulator.

Enforcement: `tests/test_leakage.py` asserts, for every row, that
`feature_source_session_end < race_start`, using a small per-feature provenance map declared in
`src/features.py`. If a feature is added without a provenance entry, the test fails. This is
cheap and it is the single most convincing thing in the repo for an engineer reading it.

---

## 7. Validation protocol

**Walk-forward, expanding window, by race date.**

- Sort driver-race rows by race date.
- Minimum training block: races 1–4 of 2026 (needed for form features) plus, for options B/C,
  all pre-2026 rows.
- Fold k: train on everything before race k, validate on race k. Step forward one race.
- Headline evaluation window: **2026 rounds 5–13** (9 folds), because that is the era Madring
  belongs to. If options B/C are used, pre-2026 races are training material only, never
  validation.
- No shuffling, no random splits, no k-fold.

Reported per fold and aggregated:

- MAE and RMSE in percent of lap time (and also converted to seconds at a 92 s reference lap so
  it is interpretable to a non-ML audience).
- R², reported but not headline — with a narrow target distribution it flatters or punishes
  arbitrarily.
- **Skill score** vs the null model: `1 - MAE_model / MAE_null`. This is the number that
  actually answers "did the model learn anything".
- Spearman rank correlation between predicted and actual pace order within each race. For a
  race simulator, getting the *order* right matters more than the absolute delta.

---

## 8. Baselines (evaluated on exactly the same folds)

- **B0 — null.** Predict 0 for everyone (field-median pace). MAE of this is the bar everything
  else is measured against.
- **B1 — qualifying only.** OLS `race_pace_delta_pct ~ a + b * quali_gap_pct`, coefficients fit
  on training folds only.
- **B2 — recent form only.** Predict the team's EWMA race pace delta.
- **B3 — ridge on (quali gap, team form, grid).** A genuinely competitive linear baseline; if
  XGBoost cannot beat this, the honest conclusion is to ship B3.

If XGBoost does not beat B1/B2/B3, that gets reported prominently, the simpler model feeds the
simulator, and slide 4 of the Instagram sequence says so. A negative result here is a better
post than a fake positive one.

---

## 9. XGBoost specification

- `XGBRegressor`, `objective='reg:pseudohubererror'` (robust to the heavy tails that damaged
  cars and one-off disasters create) with `reg:squarederror` reported as a cross-check.
- Deliberately small capacity for a ~250-row problem: `max_depth` 2–3,
  `learning_rate` 0.03–0.05, `min_child_weight` 5–10, `subsample` 0.8, `colsample_bytree` 0.8,
  `reg_lambda` 1–5, `n_estimators` chosen by early stopping inside each training block.
- **Tuning budget capped at ~20 configurations**, searched with a nested walk-forward split
  *inside* the training data only. The search space and the cap are written into the README.
  Over-tuning a dataset this small is how you manufacture a fake result.
- `random_state` fixed; all seeds recorded in `models/model_card.json` along with the chosen
  hyperparameters, training window, and fold metrics.
- Feature importance: gain-based, plus permutation importance on the validation folds (gain is
  biased toward high-cardinality continuous features). SHAP on the Madring rows for the
  per-driver explanation slide, which is the single best visual for a non-technical audience.

---

## 10. Uncertainty

The point prediction is the expected pace. The spread comes from how wrong the model actually
was, not from a chosen number.

- Collect **out-of-fold residuals** from the 2026 walk-forward folds.
- Check for heteroscedasticity: residual spread vs grid position, vs `longrun_laps_count`,
  vs `races_of_history`. If a clear pattern exists, use segmented residual pools; otherwise one
  pool.
- Check normality (QQ plot). If residuals are heavy-tailed — likely — the simulator draws each
  driver's pace by **bootstrap resampling the empirical residual distribution**, not from a
  fitted Gaussian. No sigma is invented anywhere.
- Drivers with `quali_missing` or no long-run data get residuals drawn from the corresponding
  high-uncertainty segment if one is identified, which is the honest way to say "we know less
  about this car today".

---

## 11. Monte Carlo race simulator

Separate module, no model fitting inside it. Lap-by-lap, 10,000 runs, fixed seed.

Per simulation:

1. **Underlying pace.** Draw `pace_delta_i` = XGBoost mean + bootstrap residual, once per
   driver per simulation, held constant through the race (this is "how good the car really is
   today").
2. **Anchor to seconds.** `base_lap_i = ref_race_lap * (1 + pace_delta_i/100)`, where
   `ref_race_lap` for Madring is estimated as `pole_lap * r`, with `r` = the median ratio of
   race field-median clean lap to pole lap across 2026 races. Derived from data, reported with
   its spread, labelled as an estimate.
3. **Lap-to-lap noise.** σ estimated from within-driver-race dispersion of eligible laps across
   2026 races (typical magnitude a few tenths). Data-derived.
4. **Tyre degradation.** Linear per-compound degradation estimated by regressing eligible lap
   time on `TyreLife` within stints across 2026 races, optionally blended with Madring's own
   practice long runs if the inspection shows enough laps. Reported as *estimated lap-time
   trend*, never as "measured tyre degradation".
5. **Fuel.** Folded into the same stint/lap-number trend rather than modelled separately,
   because the two are not cleanly separable from lap times alone. Stated as a limitation.
6. **Pit stops.** Pit-lane time loss estimated from Madring's own practice in-/out-laps if
   available, else a documented prior; number of stops chosen by a simple stint-length
   optimisation given estimated degradation and pit loss, with jitter on the pit lap. **No
   team-specific strategy, no undercut modelling.** Explicitly flagged as the simulator's
   biggest simplification.
7. **Track position and overtaking.** A car behind that is faster by more than a threshold τ
   passes with probability p per lap when within striking distance; otherwise it is held to the
   car ahead's pace plus a dirty-air penalty. **τ, p and the dirty-air penalty cannot be
   calibrated for Madring — no F1 race has happened there.** So they are run as a **sensitivity
   sweep** (e.g. p ∈ {0.10, 0.25, 0.50}) and the headline numbers are reported as a band across
   that sweep, with the base case stated. Context that justifies a low-p base case: the circuit
   is widely described as overtaking-limited, and the junior-category races there were reported
   as processional. That is journalism, not data, and it is cited as such.
8. **Lap 1.** Empirical distribution of position change from grid to end of lap 1, estimated per
   grid slot across 2026 races. This matters disproportionately when passing is hard.
9. **DNF.** Per-lap hazard from 2026 team-level retirement rates, shrunk toward the field rate
   with a Beta prior (13 races is a small sample). Constant hazard per lap, no separate
   mechanical/collision split.
10. **Safety Car.** No Madring-specific estimate is possible. Prior built from FastF1
    `TrackStatus` codes `4` and `6` across 2026 races: P(at least one SC), expected number of
    periods, and typical duration. Report both the all-2026 prior and a street/semi-street
    subset prior, run the base case on one, and show the other in the sensitivity panel. SC
    effect: gaps compress, and un-pitted cars get a cheaper stop. Deliberately simple.

Every one of these ten items gets a line in `src/simulation_assumptions.yaml` with its value,
its source (estimated-from-data / external-document / assumption), and the sensitivity range.
That file is what makes the project defensible to someone who does this professionally.

---

## 12. Outputs

Per driver, from the 10,000 runs:

- Win %, Podium %, Top-10 %
- Expected finishing position, median, and P10–P90 interval
- Expected positions gained/lost vs grid
- **Monte Carlo standard error** on each probability (`sqrt(p(1-p)/N)`, ~±0.5 pp at p = 0.3),
  so nobody reads "23.4 %" as precision it does not have
- The same table recomputed at the extremes of the overtaking and SC sensitivity sweep, so the
  README can state how much the conclusion depends on the untestable assumptions

Nothing is populated until the simulation has run. There are no placeholder percentages
anywhere in this repo.

**Post-race scorecard** (run Monday, after the race): predicted vs actual `race_pace_delta_pct`
per driver, predicted finishing distribution vs actual result, and whether the simulated
outcome fell inside the P10–P90 band. One honest slide about what the model got wrong is worth
more to a motorsport engineering audience than the prediction itself.

---

## 13. Repository layout

Your proposed structure, with four additions:

```
madring-race-prediction/
├── data/
│   ├── madring_grid.csv              # official grid + source URL + timestamp
│   ├── madring_race_inputs.yaml      # lap count, race distance, sources
│   └── inspection_report.json
├── cache/                            # FastF1 cache (gitignored)
├── figures/
├── models/
│   └── model_card.json               # seeds, hyperparameters, fold metrics, training window
├── outputs/
│   └── predictions_madring_<UTC timestamp>.json
├── src/
│   ├── inspect_madring_data.py
│   ├── fetch_data.py
│   ├── build_dataset.py
│   ├── features.py                   # includes the feature→session provenance map
│   ├── train_baseline.py
│   ├── train_xgboost.py
│   ├── evaluate.py
│   ├── predict_madring.py
│   ├── race_simulator.py
│   ├── simulation_assumptions.yaml
│   └── plots.py
├── tests/
│   └── test_leakage.py
├── docs/METHODOLOGY.md
├── run_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

Timestamp and git-commit the prediction file **before** 13:00 UTC on Sunday. A pre-race
prediction that appears after the race is worth nothing, and the commit hash is the proof.

---

## 14. What this project will not claim

- Not team telemetry. FastF1 exposes the public broadcast timing and car feed.
- Not measured tyre degradation. Estimated lap-time trend versus tyre age.
- Not real strategy. A simplified stop model applied uniformly.
- No battery SOC, MGU-K deployment, ERS state, brake temperature or torque — FastF1 does not
  expose them.
- Not a prediction of the winner. A probability distribution over outcomes, conditional on a
  list of stated assumptions.

---

## 15. Decisions I need from you before I write any implementation

1. **Training window.** Confirm you want all three options (A, B, C) evaluated on identical
   folds, rather than picking one up front. It costs implementation time but it is the
   difference between a defensible project and an assumed one.
2. **Traffic filter.** Enable the derived gap-to-car-ahead filter in the target definition, or
   keep the target simple (median of eligible laps) and list traffic contamination as a stated
   limitation? I lean simple, given the sample size.
3. **Scope under time pressure.** The race is Sunday 13:00 UTC. Full scope — three training
   windows, sensitivity sweeps, SHAP, seven figures — is not a few hours of work. If the goal
   is a genuine pre-race prediction, I would cut to: option A training window, one baseline set,
   XGBoost, residual bootstrap, simulator with the base-case assumptions, and three figures;
   then extend to full scope afterwards with the prediction file already committed. Your call
   on which matters more, the pre-race timestamp or the complete build.
4. **Grid source.** Confirm you will paste the official provisional starting grid (with
   penalties and non-starters) into `data/madring_grid.csv`, since FastF1's `GridPosition`
   comes from the race session and does not exist yet.

Run `src/inspect_madring_data.py` first and send me what it prints. If the Madring sessions
have not landed in the API, or the long-run stints are too thin, parts of §5 change and I would
rather adjust the design than paper over it.
