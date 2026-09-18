"""Climatology-only baseline.

Predicts insufficient-rainfall probability for day T purely from the
historical (train-period) base rate of "insufficient" for T's calendar
month — it uses no information about current/recent weather at all. This
is the bar the ML models must clear: if Random Forest / XGBoost can't beat
this, they aren't adding value over "just look at the calendar."
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ClimatologyBaseline:
    def __init__(self) -> None:
        self.month_base_rate_: pd.Series | None = None
        self.month_mean_rainfall_: pd.Series | None = None

    def fit(self, train_labeled_df: pd.DataFrame) -> "ClimatologyBaseline":
        valid = train_labeled_df.dropna(subset=["insufficient_rainfall_next_7_days"])
        month = valid["date_of_record"].dt.month
        month_rate = valid.groupby(month)["insufficient_rainfall_next_7_days"].mean().reindex(range(1, 13))
        # a month with too few surviving (non-dropped) train rows can have
        # no rate at all — fall back to the overall train rate rather than
        # emit NaN probabilities for any test row in that month.
        overall_rate = float(valid["insufficient_rainfall_next_7_days"].mean())
        self.month_base_rate_ = month_rate.fillna(overall_rate)

        valid_reg = train_labeled_df.dropna(subset=["rainfall_next_7_days"])
        month_reg = valid_reg["date_of_record"].dt.month
        month_mean = valid_reg.groupby(month_reg)["rainfall_next_7_days"].mean().reindex(range(1, 13))
        overall_mean = float(valid_reg["rainfall_next_7_days"].mean())
        self.month_mean_rainfall_ = month_mean.fillna(overall_mean)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        month = df["date_of_record"].dt.month
        return month.map(self.month_base_rate_).to_numpy(dtype=float)

    def predict_rainfall_mm(self, df: pd.DataFrame) -> np.ndarray:
        month = df["date_of_record"].dt.month
        return month.map(self.month_mean_rainfall_).to_numpy(dtype=float)
