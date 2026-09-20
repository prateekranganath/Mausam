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


def impute_missing_rainfall(
    df: pd.DataFrame, impute_as_zero: bool = IMPUTE_MISSING_RAINFALL_AS_ZERO
) -> pd.DataFrame:
    """Record which days had no rainfall reading, then (if configured)
    impute them as 0mm. See config.IMPUTE_MISSING_RAINFALL_AS_ZERO for the
    Thiruvananthapuram-specific evidence this default is based on — it did
    NOT replicate nationally (checked across 143 districts: temperature-on-
    missing-days effect flips sign roughly as often as not), so a pooled
    multi-district model should pass impute_as_zero=False explicitly rather
    than inherit this single-district-justified default. Must run before
    any lag/rolling/target logic so those all see the same series.
    """
    df = df.copy()
    df["rainfall_was_missing"] = df["rainfall"].isna().astype(int)
    if impute_as_zero:
        df["rainfall"] = df["rainfall"].fillna(0.0)
    return df


def add_rainfall_history_features(
    df: pd.DataFrame, impute_as_zero: bool = IMPUTE_MISSING_RAINFALL_AS_ZERO
) -> pd.DataFrame:
    """df must be one district's daily series, sorted ascending by date,
    with one row per calendar day (gaps present as NaN rows), already
    passed through impute_missing_rainfall so `rainfall_was_missing`
    exists.
    """
    df = df.copy()
    if "rainfall_was_missing" not in df.columns:
        df = impute_missing_rainfall(df, impute_as_zero=impute_as_zero)
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


def build_features(
    df: pd.DataFrame, impute_as_zero: bool = IMPUTE_MISSING_RAINFALL_AS_ZERO
) -> pd.DataFrame:
    """Run the full leakage-safe feature build on one district's daily series."""
    df = df.sort_values("date_of_record").reset_index(drop=True)
    df = add_temporal_features(df)
    df = add_rainfall_history_features(df, impute_as_zero=impute_as_zero)
    df = add_weather_features(df)
    return df


def build_features_multi_district(
    df: pd.DataFrame, impute_as_zero: bool = IMPUTE_MISSING_RAINFALL_AS_ZERO
) -> pd.DataFrame:
    """Same as build_features, but for a DataFrame covering MULTIPLE
    districts stacked together (e.g. from
    src.data.district.get_all_districts_daily_series). Lag/rolling/target
    logic must never see a window that crosses a district boundary, so
    each district's rows are processed independently before concatenating
    back — grouping by row order alone would silently leak one district's
    tail into another's head.
    """
    parts = [
        build_features(group, impute_as_zero=impute_as_zero)
        for _, group in df.groupby("district", sort=False)
    ]
    return pd.concat(parts, ignore_index=True)


def compute_target_multi_district(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-district equivalent of compute_target — see
    build_features_multi_district for why grouping is required."""
    parts = [
        compute_target(group) for _, group in df.groupby("district", sort=False)
    ]
    return pd.concat(parts, ignore_index=True)


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
    """Mean/median daily rainfall by (district, calendar month), from
    `reference_df` only (pass the TRAIN split). Grouping by district as
    well as month is what makes this safe to use for a pooled multi-
    district model — Kerala's monsoon climatology must never be applied
    to Rajasthan's rows. For single-district data this reduces to exactly
    the old month-only behavior, since district is constant.

    Returns a DataFrame indexed by (district, month).
    """
    stats = (
        reference_df.dropna(subset=["rainfall"])
        .assign(month=reference_df["date_of_record"].dt.month)
        .groupby(["district", "month"])["rainfall"]
        .agg(climatology_mean_rainfall_month="mean", climatology_median_rainfall_month="median")
    )
    return stats


def attach_climatology(df: pd.DataFrame, climatology: pd.DataFrame) -> pd.DataFrame:
    """Left-join climatology onto df by (district, month). Falls back to
    the district's overall mean/median (across all its months) if a
    specific (district, month) combo wasn't in the training data, and
    NaN (caught by the preprocessor's median imputer) only if the
    district itself is entirely unseen.
    """
    df = df.copy()
    month = df["date_of_record"].dt.month if "month" not in df else df["month"]
    key = pd.MultiIndex.from_arrays([df["district"], month])
    joined = climatology.reindex(key)
    joined.index = df.index

    district_fallback = climatology.groupby(level="district").mean()
    for col in ["climatology_mean_rainfall_month", "climatology_median_rainfall_month"]:
        values = joined[col]
        missing = values.isna()
        if missing.any():
            fb = df.loc[missing, "district"].map(district_fallback[col])
            values = values.copy()
            values[missing] = fb.to_numpy()
        df[col] = values.to_numpy()
    return df


def compute_risk_threshold_table(reference_df: pd.DataFrame, percentile: float) -> pd.Series:
    """Per-(district, month) `percentile`-th percentile of
    rainfall_next_7_days, from `reference_df` only (pass the TRAIN split,
    after compute_target). Grouping by district too — see
    compute_monthly_climatology's docstring for why this matters once
    more than one district is in play.
    """
    valid = reference_df.dropna(subset=["rainfall_next_7_days"])
    month = valid["date_of_record"].dt.month
    table = valid.groupby([valid["district"], month])["rainfall_next_7_days"].quantile(
        percentile / 100.0
    )
    table.index.names = ["district", "month"]

    # a (district, month) with too few surviving train rows can have no
    # threshold at all — fall back to that district's overall percentile,
    # then to the global percentile, rather than silently NaN-ing out
    # every row for that combination downstream.
    district_overall = valid.groupby("district")["rainfall_next_7_days"].quantile(percentile / 100.0)
    global_overall = float(valid["rainfall_next_7_days"].quantile(percentile / 100.0))

    all_districts = valid["district"].unique()
    full_index = pd.MultiIndex.from_product([all_districts, range(1, 13)], names=["district", "month"])
    table = table.reindex(full_index)
    district_fallback = pd.Series(
        table.index.get_level_values("district").map(district_overall).to_numpy(),
        index=table.index,
    )
    table = table.fillna(district_fallback)
    table = table.fillna(global_overall)
    return table


def label_insufficient_rainfall(df: pd.DataFrame, threshold_table: pd.Series) -> pd.DataFrame:
    df = df.copy()
    month = df["date_of_record"].dt.month
    key = pd.MultiIndex.from_arrays([df["district"], month])
    thresh = pd.Series(threshold_table.reindex(key).to_numpy(), index=df.index)
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


def compute_daily_climatology(
    reference_df: pd.DataFrame, smooth_days: int = 7, value_column: str = "rainfall"
) -> pd.DataFrame:
    """Day-of-year mean and standard deviation of rainfall, per district.

    Used by the active/break spell detector, which needs a standardised
    anomaly and therefore a per-day SD as well as a mean — something
    compute_monthly_climatology does not provide, and a monthly mean is
    too coarse for anyway (early June and late June differ enormously in
    a monsoon district).

    Smoothed over +/- `smooth_days` because a raw per-calendar-day mean is
    hopelessly noisy: with ~11 years of record each day-of-year has only
    ~11 observations, and daily rainfall is among the most skewed
    variables in meteorology. A +/-7-day window pools ~165 observations per
    point while staying far narrower than the seasonal cycle it has to
    resolve.

    The window wraps around the year end, so 1 January borrows from late
    December rather than being estimated from half a window.

    Takes `reference_df` explicitly, like the other climatology helpers
    here, so a caller cannot accidentally fit it on data that includes the
    period being evaluated.
    """
    frame = reference_df[["district", "date_of_record", value_column]].copy()
    frame["date_of_record"] = pd.to_datetime(frame["date_of_record"])
    # 29 February is folded onto 28 February: it has a quarter of the
    # samples of any other day, and its own statistics would be noise.
    day_of_year = frame["date_of_record"].dt.dayofyear
    is_leap = frame["date_of_record"].dt.is_leap_year
    frame["doy"] = np.where(is_leap & (day_of_year > 59), day_of_year - 1, day_of_year).clip(1, 365)

    records = []
    for district, group in frame.groupby("district", sort=False):
        by_doy = {doy: g[value_column].to_numpy(dtype=float) for doy, g in group.groupby("doy")}
        for doy in range(1, 366):
            offsets = [((doy - 1 + d) % 365) + 1 for d in range(-smooth_days, smooth_days + 1)]
            pooled = np.concatenate([by_doy[o] for o in offsets if o in by_doy]) if by_doy else np.array([])
            pooled = pooled[~np.isnan(pooled)]
            records.append(
                {
                    "district": district,
                    "doy": doy,
                    "climatology_daily_mean": float(np.mean(pooled)) if len(pooled) else np.nan,
                    # ddof=1: this is a sample of years, not the population.
                    "climatology_daily_sd": float(np.std(pooled, ddof=1)) if len(pooled) > 1 else np.nan,
                    "climatology_n_observations": int(len(pooled)),
                }
            )
    return pd.DataFrame.from_records(records).set_index(["district", "doy"])


def day_of_year_no_leap(dates: pd.Series) -> pd.Series:
    """Day-of-year with 29 February folded onto 28 February, matching
    compute_daily_climatology's index so lookups line up in leap years."""
    dates = pd.to_datetime(dates)
    day_of_year = dates.dt.dayofyear
    shifted = np.where(dates.dt.is_leap_year & (day_of_year > 59), day_of_year - 1, day_of_year)
    return pd.Series(np.clip(shifted, 1, 365), index=dates.index)
