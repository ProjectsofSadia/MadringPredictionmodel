"""Walk-forward validation. Every model sees exactly the same folds.

Fold k: train on all races before round k, validate on round k.
No shuffling, no random splits, whole races only.

Model selection rule (user, V1): whichever model has the best out-of-fold MAE
feeds the simulator, XGBoost included or not. Ties inside 0.01 pct-points are
broken toward the simpler model, in the order the models are listed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import MIN_TRAIN_ROUNDS
from train_baseline import baseline_models
from train_xgboost import XGBModel

TIE_TOLERANCE = 0.01  # pct-points of lap time


def walk_forward_folds(panel: pd.DataFrame) -> list[tuple[pd.Index, pd.Index, int]]:
    labelled = panel[panel["race_pace_delta_pct"].notna()]
    rounds = sorted(labelled["round"].unique())
    folds = []
    for i, rnd in enumerate(rounds):
        if i < MIN_TRAIN_ROUNDS:
            continue
        train = labelled[labelled["round"] < rnd].index
        val = labelled[labelled["round"] == rnd].index
        if len(train) >= 20 and len(val) > 0:
            folds.append((train, val, int(rnd)))
    return folds


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    err = yhat - y
    out = {
        "mae_pct": float(np.mean(np.abs(err))),
        "rmse_pct": float(np.sqrt(np.mean(err**2))),
        "n": int(len(y)),
    }
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return out


def run_walk_forward(panel: pd.DataFrame) -> dict:
    folds = walk_forward_folds(panel)
    if not folds:
        raise RuntimeError("no walk-forward folds could be built - too few usable races")

    models = baseline_models() + [XGBModel(tune=True)]
    oof = {m.name: pd.Series(index=panel.index, dtype=float) for m in models}
    per_fold: dict[str, list] = {m.name: [] for m in models}
    chosen_params: dict = {}

    for train_idx, val_idx, rnd in folds:
        Xtr, ytr = panel.loc[train_idx], panel.loc[train_idx, "race_pace_delta_pct"]
        Xva, yva = panel.loc[val_idx], panel.loc[val_idx, "race_pace_delta_pct"]

        for m in models:
            model = m.__class__(**({"tune": True} if isinstance(m, XGBModel) else {}))
            if isinstance(model, XGBModel):
                model.fit(Xtr, ytr, rounds=Xtr["round"])
                chosen_params[rnd] = model.best_params_
            else:
                model.fit(Xtr, ytr)
            pred = model.predict(Xva)
            oof[m.name].loc[val_idx] = pred
            fm = metrics(yva.values, pred)
            fm["round"] = rnd
            per_fold[m.name].append(fm)

    summary = {}
    y_all = panel["race_pace_delta_pct"]
    for name, series in oof.items():
        mask = series.notna() & y_all.notna()
        m = metrics(y_all[mask].values, series[mask].values)
        rhos = []
        for _, _, rnd in folds:
            sub = panel[(panel["round"] == rnd) & mask]
            if len(sub) >= 5 and series[sub.index].std(ddof=0) > 0:
                rho = spearmanr(sub["race_pace_delta_pct"], series[sub.index]).statistic
                if np.isfinite(rho):
                    rhos.append(float(rho))
        m["mean_within_race_spearman"] = float(np.mean(rhos)) if rhos else float("nan")
        m["per_fold_mae"] = {f["round"]: round(f["mae_pct"], 4) for f in per_fold[name]}
        summary[name] = m

    null_mae = summary["B0_null"]["mae_pct"]
    for name, m in summary.items():
        m["skill_vs_null"] = float(1.0 - m["mae_pct"] / null_mae) if null_mae > 0 else np.nan

    order = [m.name for m in models]  # simplest first
    best_mae = min(summary[n]["mae_pct"] for n in order)
    best = next(n for n in order if summary[n]["mae_pct"] <= best_mae + TIE_TOLERANCE)

    return {
        "folds": [(int(r), len(t), len(v)) for t, v, r in folds],
        "summary": summary,
        "oof": pd.DataFrame(oof),
        "selected_model": best,
        "xgb_params_per_fold": chosen_params,
        "selection_rule": (
            "lowest out-of-fold MAE on identical walk-forward folds; ties within "
            f"{TIE_TOLERANCE} pct-points broken toward the simpler model"
        ),
    }


def residual_pool(panel: pd.DataFrame, oof: pd.DataFrame, model_name: str) -> np.ndarray:
    """Out-of-fold residuals (actual - predicted) of the selected model.
    The simulator bootstraps from these; no sigma is ever chosen by hand."""
    pred = oof[model_name]
    mask = pred.notna() & panel["race_pace_delta_pct"].notna()
    return (panel.loc[mask, "race_pace_delta_pct"] - pred[mask]).values.astype(float)


def refit_selected(panel: pd.DataFrame, model_name: str):
    """Refit the selected model on every labelled historical row."""
    labelled = panel[panel["race_pace_delta_pct"].notna()]
    X, y = labelled, labelled["race_pace_delta_pct"]
    if model_name == "XGBoost":
        m = XGBModel(tune=True).fit(X, y, rounds=X["round"])
    else:
        m = next(b for b in baseline_models() if b.name == model_name).fit(X, y)
    return m
