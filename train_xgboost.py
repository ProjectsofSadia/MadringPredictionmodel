"""Deliberately small XGBoost regressor.

Capacity is capped because the 2026-only dataset is ~250 driver-race rows. The
tuning grid is 12 configurations, searched with a nested walk-forward split
INSIDE the training data only. Madring is never used for selection.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from config import SEED
from features import FEATURES

# 12 configurations. Kept small on purpose: over-searching a 250-row dataset
# manufactures validation performance that does not exist.
PARAM_GRID = [
    dict(max_depth=d, learning_rate=lr, min_child_weight=mcw, reg_lambda=lam)
    for d, lr, mcw, lam in itertools.product([2, 3], [0.03, 0.05], [5, 10], [3.0])
] + [
    dict(max_depth=2, learning_rate=0.05, min_child_weight=5, reg_lambda=1.0),
    dict(max_depth=2, learning_rate=0.05, min_child_weight=5, reg_lambda=5.0),
    dict(max_depth=3, learning_rate=0.03, min_child_weight=10, reg_lambda=1.0),
    dict(max_depth=3, learning_rate=0.03, min_child_weight=10, reg_lambda=5.0),
]

FIXED = dict(
    n_estimators=400,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:pseudohubererror",
    huber_slope=1.0,
    random_state=SEED,
    n_jobs=2,
    tree_method="hist",
)


class XGBModel:
    name = "XGBoost"
    uses = FEATURES

    def __init__(self, tune: bool = True):
        self.tune = tune
        self.best_params_: dict = {}

    # -- inner walk-forward selection -------------------------------------
    def _inner_score(self, params: dict, X: pd.DataFrame, y: pd.Series,
                     rounds: pd.Series) -> float:
        uniq = sorted(rounds.unique())
        if len(uniq) < 4:
            return np.inf
        errs = []
        for cut in uniq[max(2, len(uniq) - 3):]:  # last up to 3 rounds as inner folds
            tr = rounds < cut
            va = rounds == cut
            if tr.sum() < 20 or va.sum() == 0:
                continue
            m = XGBRegressor(**FIXED, **params)
            m.fit(X.loc[tr, self.uses], y[tr], verbose=False)
            errs.append(np.mean(np.abs(m.predict(X.loc[va, self.uses]) - y[va])))
        return float(np.mean(errs)) if errs else np.inf

    def fit(self, X: pd.DataFrame, y: pd.Series, rounds: pd.Series | None = None):
        params = dict(max_depth=2, learning_rate=0.05, min_child_weight=5, reg_lambda=3.0)
        if self.tune and rounds is not None:
            scored = [(self._inner_score(p, X, y, rounds), p) for p in PARAM_GRID]
            scored = [s for s in scored if np.isfinite(s[0])]
            if scored:
                params = min(scored, key=lambda t: t[0])[1]
        self.best_params_ = params
        self.model_ = XGBRegressor(**FIXED, **params)
        self.model_.fit(X[self.uses], y, verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X[self.uses])

    def importance(self) -> dict:
        gain = self.model_.get_booster().get_score(importance_type="gain")
        return {k: float(v) for k, v in sorted(gain.items(), key=lambda t: -t[1])}
