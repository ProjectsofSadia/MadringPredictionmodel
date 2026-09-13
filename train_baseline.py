"""Baselines. All share the same fit(X_df, y)/predict(X_df) interface as the
XGBoost model so evaluate.py can run them on identical folds.

B0 null            predict 0 (the field-median pace) for everyone
B1 qualifying      OLS on quali_gap_pct
B2 recent form     predict the team's EWMA prior race pace delta
B3 ridge           ridge on a small pre-race feature set

Missing-data strategy for the linear models (B1/B3): median imputation with the
median computed on the TRAINING FOLD ONLY, plus a missing indicator column.
XGBoost keeps NaN natively and never sees an imputed value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from features import LINEAR_FEATURES


class _Base:
    name = "base"
    uses = []

    def fit(self, X: pd.DataFrame, y: pd.Series):  # noqa: D102
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: D102
        raise NotImplementedError


class _Imputer:
    """Training-fold-only median imputation + explicit missing flags."""

    def fit(self, X: pd.DataFrame):
        self.cols_ = list(X.columns)
        self.medians_ = X.median(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.cols_)
        flags = X.isna().astype(float).values
        filled = X.fillna(self.medians_).fillna(0.0).values
        return np.hstack([filled, flags])


class NullModel(_Base):
    name = "B0_null"
    uses: list[str] = []

    def fit(self, X, y):
        self.const_ = 0.0  # field-median pace by construction of the target
        return self

    def predict(self, X):
        return np.full(len(X), self.const_)


class QualifyingModel(_Base):
    name = "B1_qualifying"
    uses = ["quali_gap_pct"]

    def fit(self, X, y):
        self.imp_ = _Imputer().fit(X[self.uses])
        self.m_ = LinearRegression().fit(self.imp_.transform(X[self.uses]), y)
        return self

    def predict(self, X):
        return self.m_.predict(self.imp_.transform(X[self.uses]))


class RecentFormModel(_Base):
    name = "B2_recent_form"
    uses = ["team_recent_race_pace_delta"]

    def fit(self, X, y):
        self.fallback_ = float(np.nanmean(y)) if len(y) else 0.0
        return self

    def predict(self, X):
        return X["team_recent_race_pace_delta"].fillna(self.fallback_).values


class RidgeModel(_Base):
    name = "B3_ridge"
    uses = LINEAR_FEATURES

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.imp_ = _Imputer().fit(X[self.uses])
        Z = self.imp_.transform(X[self.uses])
        self.sc_ = StandardScaler().fit(Z)
        self.m_ = Ridge(alpha=self.alpha).fit(self.sc_.transform(Z), y)
        return self

    def predict(self, X):
        Z = self.imp_.transform(X[self.uses])
        return self.m_.predict(self.sc_.transform(Z))


def baseline_models() -> list[_Base]:
    return [NullModel(), QualifyingModel(), RecentFormModel(), RidgeModel(alpha=1.0)]
