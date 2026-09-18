"""Feature engineering and target construction for the rainfall-risk model.

Hard rule throughout this module: every feature for prediction day T uses
only information available up to and including day T. The only exception
is `rainfall_next_7_days` / `insufficient_rainfall_next_7_days`, which are
explicitly the TARGETS (T+1..T+7) and must never be used as inputs.

Two things are deliberately NOT computed here because they must be fit on
the training split only (see src/ml/train.py):
  - monthly rainfall climatology (compute_monthly_climatology / attach)
  - the risk threshold table (compute_risk_threshold_table / label)
Both take an explicit `reference_df` argument for exactly this reason —
passing the full dataset instead of the train slice would leak future
rainfall statistics into the labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    DRY_DAY_THRESHOLD_MM,
    FORECAST_HORIZON_DAYS,
    IMPUTE_MISSING_RAINFALL_AS_ZERO,
)

MONSOON_PHASES = ["winter", "pre_monsoon", "sw_monsoon", "post_monsoon"]

# IMD's four-season convention (generalizes across any Indian district,
# not tuned to Kerala specifically).
_MONTH_TO_PHASE = {
    12: "winter", 1: "winter", 2: "winter",
    3: "pre_monsoon", 4: "pre_monsoon", 5: "pre_monsoon",
    6: "sw_monsoon", 7: "sw_monsoon", 8: "sw_monsoon", 9: "sw_monsoon",
    10: "post_monsoon", 11: "post_monsoon",
}

RAW_WEATHER_COLS = ["avg_temp", "min_temp", "max_temp", "wind_speed", "air_pressure"]

FEATURE_COLUMNS = [
    "rainfall_lag_1", "rainfall_lag_3", "rainfall_lag_7",
    "rainfall_sum_last_3", "rainfall_sum_last_7", "rainfall_sum_last_14", "rainfall_sum_last_30",
    "rainfall_rolling_mean_7", "rainfall_rolling_mean_14", "rainfall_rolling_mean_30",
    "consecutive_dry_days",
    "climatology_mean_rainfall_month", "climatology_median_rainfall_month",
    "avg_temp", "min_temp", "max_temp", "air_pressure", "wind_speed",
    "avg_temp_lag_3",
    "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
    "monsoon_phase_winter", "monsoon_phase_pre_monsoon",
    "monsoon_phase_sw_monsoon", "monsoon_phase_post_monsoon",
    "latitude", "longitude", "elevation",
    "rainfall_missing_frac_last_7",
    "temp_missing_flag", "pressure_missing_flag", "wind_missing_flag",
]


def _consecutive_dry_days(rainfall: pd.Series) -> pd.Series:
    """Trailing run length (ending the day BEFORE each row) of known days
    with rainfall < DRY_DAY_THRESHOLD_MM. The walk stops (returns whatever
    it has so far) at the first NaN or first wet day looking backward, so
    an unknown/missing day never gets silently counted as dry.
    """
    is_dry = rainfall < DRY_DAY_THRESHOLD_MM
    is_known = rainfall.notna()

    out = np.zeros(len(rainfall), dtype=float)
    run = 0
    # values as of "yesterday" relative to the row we're filling in
    dry_vals = is_dry.to_numpy()
    known_vals = is_known.to_numpy()
    for i in range(len(rainfall)):
        out[i] = run  # streak ending the day before row i
        if i < len(rainfall):
            if not known_vals[i]:
                run = 0
            elif dry_vals[i]:
                run += 1
            else:
                run = 0
    return pd.Series(out, index=rainfall.index)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date = df["date_of_record"]
    df["month"] = date.dt.month
    day_of_year = date.dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    phase = df["month"].map(_MONTH_TO_PHASE)
    for p in MONSOON_PHASES:
        df[f"monsoon_phase_{p}"] = (phase == p).astype(int)
    return df


def impute_missing_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """Record which days had no rainfall reading, then (if configured)
    impute them as 0mm. See config.IMPUTE_MISSING_RAINFALL_AS_ZERO for the
    evidence behind this. Must run before any lag/rolling/target logic so
    those all see the same (imputed) series consistently.
    """
    df = df.copy()
    df["rainfall_was_missing"] = df["rainfall"].isna().astype(int)
    if IMPUTE_MISSING_RAINFALL_AS_ZERO:
        df["rainfall"] = df["rainfall"].fillna(0.0)
    return df


def add_rainfall_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must be one district's daily series, sorted ascending by date,
    with one row per calendar day (gaps present as NaN rows), already
    passed through impute_missing_rainfall so `rainfall_was_missing`
    exists.
    """
    df = df.copy()
    if "rainfall_was_missing" not in df.columns:
        df = impute_missing_rainfall(df)
    rain = df["rainfall"]
    was_missing = df["rainfall_was_missing"]

    df["rainfall_lag_1"] = rain.shift(1)
    df["rainfall_lag_3"] = rain.shift(3)
    df["rainfall_lag_7"] = rain.shift(7)

    # shift(1) first so the window is T-1..T-N (never includes day T itself)
    past = rain.shift(1)
    for window in (3, 7, 14, 30):
        df[f"rainfall_sum_last_{window}"] = past.rolling(
            window, min_periods=window
        ).sum()
    for window in (7, 14, 30):
        df[f"rainfall_rolling_mean_{window}"] = past.rolling(
            window, min_periods=window
        ).mean()

    df["consecutive_dry_days"] = _consecutive_dry_days(rain).values

    df["rainfall_missing_frac_last_7"] = (
        was_missing.shift(1).rolling(7, min_periods=7).mean()
    )

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_temp_lag_3"] = df["avg_temp"].shift(3)
    df["temp_missing_flag"] = df["avg_temp"].isna().astype(int)
    df["pressure_missing_flag"] = df["air_pressure"].isna().astype(int)
    df["wind_missing_flag"] = df["wind_speed"].isna().astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full leakage-safe feature build on one district's daily series."""
    df = df.sort_values("date_of_record").reset_index(drop=True)
    df = add_temporal_features(df)
    df = add_rainfall_history_features(df)
    df = add_weather_features(df)
    return df


def _forward_window_sum(series: pd.Series, window: int) -> pd.Series:
    """At position i: sum of series[i .. i+window-1] (window days STARTING
    at i, inclusive) — all `window` values must be non-NaN or the result
    is NaN. Implemented via rolling-sum on the reversed series.
    """
    reversed_roll = (
        pd.Series(series.to_numpy()[::-1])
        .rolling(window, min_periods=window)
        .sum()
    )
    return pd.Series(reversed_roll.to_numpy()[::-1], index=series.index)


def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """rainfall_next_7_days = sum of rainfall over T+1..T+FORECAST_HORIZON_DAYS.

    NaN when any of those 7 days is missing rainfall — we do not fabricate
    the target. Rows with NaN target are dropped later, before training.
    """
    df = df.copy()
    # sum of rain[i .. i+6], then shift back one position so row i holds
    # the sum for i+1 .. i+7 instead of i .. i+6
    fwd_sum_from_T = _forward_window_sum(df["rainfall"], FORECAST_HORIZON_DAYS)
    df["rainfall_next_7_days"] = fwd_sum_from_T.shift(-1)
    return df


def compute_monthly_climatology(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median daily rainfall by calendar month, from `reference_df`
    only (pass the TRAIN split). Returns a DataFrame indexed by month.
    """
    stats = (
        reference_df.dropna(subset=["rainfall"])
        .assign(month=reference_df["date_of_record"].dt.month)
        .groupby("month")["rainfall"]
        .agg(climatology_mean_rainfall_month="mean", climatology_median_rainfall_month="median")
    )
    return stats.reindex(range(1, 13))


def attach_climatology(df: pd.DataFrame, climatology: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    month = df["date_of_record"].dt.month if "month" not in df else df["month"]
    df = df.merge(climatology, left_on=month, right_index=True, how="left")
    return df


def compute_risk_threshold_table(reference_df: pd.DataFrame, percentile: float) -> pd.Series:
    """Per-month `percentile`-th percentile of rainfall_next_7_days, from
    `reference_df` only (pass the TRAIN split, after compute_target and
    with NaN targets already present — they're dropped here).
    """
    valid = reference_df.dropna(subset=["rainfall_next_7_days"])
    month = valid["date_of_record"].dt.month
    table = valid.groupby(month)["rainfall_next_7_days"].quantile(percentile / 100.0)
    table = table.reindex(range(1, 13))
    # a month with too few surviving (non-dropped) train rows can have no
    # threshold at all — fall back to the overall train percentile rather
    # than silently NaN-ing out every row in that month downstream.
    overall = float(valid["rainfall_next_7_days"].quantile(percentile / 100.0))
    return table.fillna(overall)


def label_insufficient_rainfall(df: pd.DataFrame, threshold_table: pd.Series) -> pd.DataFrame:
    df = df.copy()
    month = df["date_of_record"].dt.month
    thresh = month.map(threshold_table)
    df["rainfall_risk_threshold_mm"] = thresh
    df["insufficient_rainfall_next_7_days"] = np.where(
        df["rainfall_next_7_days"].isna(),
        np.nan,
        (df["rainfall_next_7_days"] < thresh).astype(float),
    )
    return df


class RainfallFeaturePreprocessor:
    """Bundles everything that must be learned from the TRAINING split only:
    monthly climatology, the per-month risk threshold table, and median
    imputation values. `fit` must only ever be called with training rows.

    `transform` is safe to call on train/val/test/live data alike — it
    only applies parameters already learned by `fit`.
    """

    def __init__(self, risk_threshold_percentile: float):
        self.risk_threshold_percentile = risk_threshold_percentile
        self.feature_columns = list(FEATURE_COLUMNS)
        self.climatology_table_: pd.DataFrame | None = None
        self.risk_threshold_table_: pd.Series | None = None
        self.impute_values_: pd.Series | None = None

    def fit(self, train_features_df: pd.DataFrame) -> "RainfallFeaturePreprocessor":
        self.climatology_table_ = compute_monthly_climatology(train_features_df)
        self.risk_threshold_table_ = compute_risk_threshold_table(
            train_features_df, self.risk_threshold_percentile
        )
        attached = attach_climatology(train_features_df, self.climatology_table_)
        self.impute_values_ = attached[self.feature_columns].median(numeric_only=True)
        return self

    def transform(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.climatology_table_ is None:
            raise RuntimeError("Preprocessor must be fit() before transform().")
        df = attach_climatology(features_df, self.climatology_table_)
        if "rainfall_next_7_days" in df.columns:
            df = label_insufficient_rainfall(df, self.risk_threshold_table_)
        X = df[self.feature_columns].fillna(self.impute_values_)
        return X, df
