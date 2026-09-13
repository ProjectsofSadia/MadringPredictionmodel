"""End-to-end V1 pipeline.

    python run_pipeline.py                 # full run
    python run_pipeline.py --inspect-only  # availability gate only
    python run_pipeline.py --race-laps 53  # override the documented lap count

Order: availability gate -> dataset -> features -> walk-forward validation ->
model selection -> residual bootstrap -> Monte Carlo -> figures -> frozen JSON.

The pipeline refuses to produce a prediction if the gate fails. No number in the
output is written by hand.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from build_dataset import build_history, target_summary  # noqa: E402
from config import (  # noqa: E402
    ASSUMPTIONS_PATH,
    DATA,
    FIGURES,
    HISTORY_ROUNDS,
    MADRING_ROUND,
    MODELS,
    N_SIMULATIONS,
    OUTPUTS,
    SEASON,
    SEED,
)
from evaluate import refit_selected, residual_pool, run_walk_forward  # noqa: E402
from features import FEATURES, PROVENANCE, build_panel  # noqa: E402
from fetch_data import check_availability, setup_cache  # noqa: E402
from plots import plot_grid_vs_expected, plot_model_validation, plot_win_probability  # noqa: E402
from predict_madring import full_panel, madring_block, predict_pace  # noqa: E402
from priors import estimate_priors  # noqa: E402
from race_simulator import simulate, summarise  # noqa: E402


def git_hash() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-only", action="store_true")
    ap.add_argument("--race-laps", type=int, default=None)
    ap.add_argument("--n-sims", type=int, default=N_SIMULATIONS)
    args = ap.parse_args()

    for d in (FIGURES, MODELS, OUTPUTS, DATA):
        d.mkdir(parents=True, exist_ok=True)
    setup_cache()
    retrieval_started = datetime.now(timezone.utc).isoformat()

    # ---- 1. availability gate -------------------------------------------
    print("\n[1/9] availability gate")
    avail = check_availability(HISTORY_ROUNDS)
    problems = avail.blocking_problems()
    for w in avail.warnings():
        print(f"  WARNING: {w}")
    if problems:
        print("\nSTOPPING. Blocking data problems:")
        for p in problems:
            print(f"  - {p}")
        return 2
    if args.inspect_only:
        return 0

    # ---- 2. historical targets ------------------------------------------
    print("\n[2/9] building 2026 driver-race targets")
    history, race_meta = build_history(avail.usable_history_rounds, SEASON)
    if history.empty:
        print("STOPPING: no historical targets could be built.")
        return 2
    tsum = target_summary(history)
    print(f"  {tsum['rows']} rows / {tsum['races']} races / "
          f"target sd = {tsum['target_std_pct']:.3f} pct")

    # ---- 3. features -----------------------------------------------------
    print("\n[3/9] building leakage-safe features")
    panel = build_panel(history, avail.usable_history_rounds, SEASON)
    if panel.empty:
        print("STOPPING: feature panel empty.")
        return 2

    # ---- 4. Madring feature row -----------------------------------------
    print("\n[4/9] building Madring pre-race features")
    mad_block, mad_meta = madring_block()
    combined = full_panel(panel, mad_block)
    train_panel = combined[combined["round"] != MADRING_ROUND].copy()
    madring_rows = combined[combined["round"] == MADRING_ROUND].copy()
    print(f"  {len(madring_rows)} drivers, pole lap {mad_meta['pole_lap_s']:.3f}s, "
          f"long-run data from {mad_meta['practice_sessions_with_longrun_data'] or 'none'}")

    # ---- 5. walk-forward validation -------------------------------------
    print("\n[5/9] walk-forward validation (identical folds for every model)")
    wf = run_walk_forward(train_panel)
    for name, m in wf["summary"].items():
        print(f"  {name:16s} MAE={m['mae_pct']:.4f}  RMSE={m['rmse_pct']:.4f}  "
              f"skill={m['skill_vs_null']:+.3f}  rho={m['mean_within_race_spearman']:.3f}")
    selected = wf["selected_model"]
    print(f"  selected: {selected}")

    resid = residual_pool(train_panel, wf["oof"], selected)
    model = refit_selected(train_panel, selected)
    pred = predict_pace(model, madring_rows)
    madring_rows["predicted_pace_delta_pct"] = pred

    # ---- 6. simulator priors --------------------------------------------
    print("\n[6/9] estimating simulator priors from 2026 races")
    priors = estimate_priors(avail.usable_history_rounds, SEASON)
    ratio = priors["race_pace_to_pole_ratio"]["value"]
    if ratio is None:
        print("STOPPING: could not estimate race-pace-to-pole ratio.")
        return 2
    ref_race_lap = mad_meta["pole_lap_s"] * ratio
    print(f"  reference race lap = {ref_race_lap:.3f}s (pole x {ratio:.4f})")

    inputs_path = DATA / "madring_race_inputs.yaml"
    if not inputs_path.exists():
        print(f"STOPPING: {inputs_path} not found. It is a documented external input.")
        return 2
    inputs = yaml.safe_load(inputs_path.read_text())
    n_laps = args.race_laps or int(inputs["race_laps"])
    assumptions = yaml.safe_load(ASSUMPTIONS_PATH.read_text())

    dnf_by_team = priors["dnf_probability_per_race"]["by_team"]
    field_dnf = priors["dnf_probability_per_race"]["field_rate"]
    dnf = np.array(
        [dnf_by_team.get(str(t), field_dnf) for t in madring_rows["team"]], dtype=float
    )

    # ---- 7. Monte Carlo --------------------------------------------------
    print(f"\n[7/9] {args.n_sims:,} simulations x 3 overtaking cases ({n_laps} laps)")
    drivers = madring_rows["driver"].astype(str).tolist()
    grid = madring_rows["grid_position"].astype(float).values
    cases = {}
    for case, p in assumptions["overtaking_cases"].items():
        out = simulate(
            pred_pace_pct=pred,
            residuals=resid,
            grid_position=grid,
            dnf_per_race=dnf,
            ref_race_lap_s=ref_race_lap,
            n_laps=n_laps,
            priors=priors,
            assumptions=assumptions,
            pass_prob=float(p),
            n_sims=args.n_sims,
            seed=SEED,
        )
        cases[case] = summarise(out["finish_positions"], drivers, grid)
        top = cases[case][0]
        print(f"  {case:5s} (p={p}): highest win probability "
              f"{top['driver']} {top['win_pct']:.1f}%")

    base = cases["BASE"]

    # ---- 8. figures ------------------------------------------------------
    print("\n[8/9] figures")
    f1 = plot_model_validation(wf["summary"], selected)
    f2 = plot_win_probability(base, args.n_sims, "BASE")
    f3 = plot_grid_vs_expected(base, args.n_sims)
    for f in (f1, f2, f3):
        print(f"  {f}")

    # ---- 9. freeze -------------------------------------------------------
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sens = {
        r["driver"]: {
            case: round(next(x["win_pct"] for x in cases[case] if x["driver"] == r["driver"]), 2)
            for case in cases
        }
        for r in base[:8]
    }
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_retrieval_started_utc": retrieval_started,
        "race": {
            "season": SEASON,
            "round": MADRING_ROUND,
            "event": "Spanish Grand Prix (Madring)",
            "race_start_utc": inputs["race_start_utc"],
            "race_laps": n_laps,
            "race_laps_source": inputs["race_laps_source"],
        },
        "git_commit": git_hash(),
        "random_seed": SEED,
        "n_simulations": args.n_sims,
        "training_window": f"{SEASON} only (Option A)",
        "training_rounds_used": avail.usable_history_rounds,
        "dataset": tsum,
        "target_definition": (
            "race_pace_delta_pct = 100*(driver median eligible clean race lap - "
            "field median of the same quantity)/field median; no traffic filter in V1"
        ),
        "features": FEATURES,
        "feature_provenance": PROVENANCE,
        "model_selected": selected,
        "model_selection_rule": wf["selection_rule"],
        "validation": {
            "scheme": "expanding-window walk-forward by race, whole races per fold",
            "folds": wf["folds"],
            "metrics": wf["summary"],
            "xgb_params_per_fold": wf["xgb_params_per_fold"],
        },
        "uncertainty": {
            "method": "bootstrap of out-of-fold residuals of the selected model",
            "n_residuals": int(len(resid)),
            "residual_sd_pct": float(np.std(resid, ddof=0)),
        },
        "grid_source": {
            "url": mad_meta["grid_source_url"],
            "retrieved_utc": mad_meta["grid_retrieved_utc"],
            "note": "official provisional starting grid, not inferred from qualifying",
        },
        "madring_inputs": {
            "pole_lap_s": mad_meta["pole_lap_s"],
            "reference_race_lap_s": ref_race_lap,
            "drivers_without_quali_time": mad_meta["drivers_without_quali_time"],
            "practice_sessions_with_longrun_data": mad_meta[
                "practice_sessions_with_longrun_data"
            ],
        },
        "priors_estimated_from_data": priors,
        "assumptions": assumptions,
        "predicted_pace_delta_pct": {
            d: float(p) for d, p in zip(drivers, pred)
        },
        "results_by_overtaking_case": cases,
        "headline_case": "BASE",
        "win_probability_sensitivity": sens,
        "disclaimer": (
            "Simulated probabilities from a simplified model, not a prediction of the "
            "result. Public FastF1 timing data only - no team telemetry, no measured "
            "tyre degradation, no knowledge of team strategy."
        ),
    }

    out_path = OUTPUTS / f"predictions_madring_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    (MODELS / "model_card.json").write_text(
        json.dumps(
            {
                "selected_model": selected,
                "seed": SEED,
                "features": FEATURES,
                "validation": wf["summary"],
                "xgb_params_per_fold": wf["xgb_params_per_fold"],
                "training_window": "2026 only (Option A)",
            },
            indent=2,
            default=str,
        )
    )
    madring_rows.to_csv(OUTPUTS / f"madring_features_{stamp}.csv", index=False)
    pd.DataFrame(base).to_csv(OUTPUTS / f"madring_simulation_base_{stamp}.csv", index=False)

    print(f"\n[9/9] frozen prediction written: {out_path}")
    print("Commit this file BEFORE the race. Any later change goes in a NEW file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
