"""Climatology-only baseline.

Predicts insufficient-rainfall probability for day T purely from the
historical (train-period) base rate of "insufficient" for T's (district,
calendar month) — it uses no information about current/recent weather at
all. Grouping by district as well as month matters as soon as more than
one district is in play: Kerala's monsoon base rate must never be used to
score a Rajasthan row. For single-district data this reduces to exactly
the old month-only behavior, since district is constant. This is the bar
the ML models must clear: if Random Forest / XGBoost can't beat this,
they aren't adding value over "just look at the calendar for this place."
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ClimatologyBaseline:
    def __init__(self) -> None:
        self.rate_: pd.Series | None = None
        self.mean_rainfall_: pd.Series | None = None
        self.global_rate_: float | None = None
        self.global_mean_rainfall_: float | None = None

    def fit(self, train_labeled_df: pd.DataFrame) -> "ClimatologyBaseline":
        valid = train_labeled_df.dropna(subset=["insufficient_rainfall_next_7_days"])
        month = valid["date_of_record"].dt.month
        rate = valid.groupby([valid["district"], month])["insufficient_rainfall_next_7_days"].mean()
        rate.index.names = ["district", "month"]
        # a (district, month) with too few surviving train rows falls back
        # to that district's overall rate, then the global rate, rather
        # than emitting NaN probabilities.
        district_overall = valid.groupby("district")["insufficient_rainfall_next_7_days"].mean()
        self.global_rate_ = float(valid["insufficient_rainfall_next_7_days"].mean())
        all_districts = valid["district"].unique()
        full_index = pd.MultiIndex.from_product([all_districts, range(1, 13)], names=["district", "month"])
        rate = rate.reindex(full_index)
        fb = pd.Series(rate.index.get_level_values("district").map(district_overall).to_numpy(), index=rate.index)
        self.rate_ = rate.fillna(fb).fillna(self.global_rate_)

        valid_reg = train_labeled_df.dropna(subset=["rainfall_next_7_days"])
        month_reg = valid_reg["date_of_record"].dt.month
        mean_rain = valid_reg.groupby([valid_reg["district"], month_reg])["rainfall_next_7_days"].mean()
        mean_rain.index.names = ["district", "month"]
        district_overall_reg = valid_reg.groupby("district")["rainfall_next_7_days"].mean()
        self.global_mean_rainfall_ = float(valid_reg["rainfall_next_7_days"].mean())
        mean_rain = mean_rain.reindex(full_index)
        fb_reg = pd.Series(
            mean_rain.index.get_level_values("district").map(district_overall_reg).to_numpy(), index=mean_rain.index
        )
        self.mean_rainfall_ = mean_rain.fillna(fb_reg).fillna(self.global_mean_rainfall_)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # A district with ZERO valid train rows never gets an index entry
        # at all in self.rate_ (not even NaN) — reindex introduces it as
        # NaN, so a final global fallback is needed here too, not just in
        # fit(), or such a district's test rows would come back NaN.
        month = df["date_of_record"].dt.month
        key = pd.MultiIndex.from_arrays([df["district"], month])
        return self.rate_.reindex(key).fillna(self.global_rate_).to_numpy(dtype=float)

    def predict_rainfall_mm(self, df: pd.DataFrame) -> np.ndarray:
        month = df["date_of_record"].dt.month
        key = pd.MultiIndex.from_arrays([df["district"], month])
        return self.mean_rainfall_.reindex(key).fillna(self.global_mean_rainfall_).to_numpy(dtype=float)
