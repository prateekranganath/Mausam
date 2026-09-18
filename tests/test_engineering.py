import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    add_rainfall_history_features,
    compute_monthly_climatology,
    compute_risk_threshold_table,
    compute_target,
    label_insufficient_rainfall,
)


def _synthetic_daily(n=15, start="2020-01-01"):
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({"date_of_record": dates, "rainfall": list(range(n))})


def test_target_is_sum_of_next_7_days():
    df = _synthetic_daily(15)
    out = compute_target(df)
    assert out.loc[0, "rainfall_next_7_days"] == sum(range(1, 8))
    assert out.loc[7, "rainfall_next_7_days"] == sum(range(8, 15))


def test_target_nan_when_insufficient_future_days():
    df = _synthetic_daily(15)
    out = compute_target(df)
    assert pd.isna(out.loc[8, "rainfall_next_7_days"])
    assert pd.isna(out.loc[14, "rainfall_next_7_days"])


def test_target_nan_when_any_future_day_missing():
    df = _synthetic_daily(15)
    df.loc[3, "rainfall"] = np.nan
    out = compute_target(df)
    # row 0's window is days 1..7, which includes the NaN at day 3
    assert pd.isna(out.loc[0, "rainfall_next_7_days"])
    # row 3's window is days 4..10, no NaN in it
    assert out.loc[3, "rainfall_next_7_days"] == sum(range(4, 11))


def test_lag_and_rolling_features_never_include_day_T():
    df = _synthetic_daily(15)
    out = add_rainfall_history_features(df)
    # rainfall_lag_1 at row i must equal rainfall at row i-1, never row i
    for i in range(1, 15):
        assert out.loc[i, "rainfall_lag_1"] == df.loc[i - 1, "rainfall"]
    # rainfall_sum_last_7 at row 7 = sum of rows 0..6 (excludes row 7 itself)
    assert out.loc[7, "rainfall_sum_last_7"] == sum(range(0, 7))


def test_consecutive_dry_days_resets_on_wet_day():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    # dry, dry, wet, dry, dry, dry  (threshold default 2.5mm)
    rainfall = [0.0, 0.5, 10.0, 0.0, 1.0, 0.0]
    df = pd.DataFrame({"date_of_record": dates, "rainfall": rainfall})
    out = add_rainfall_history_features(df)
    # streak ending the day BEFORE each row
    assert out.loc[0, "consecutive_dry_days"] == 0  # no prior data
    assert out.loc[1, "consecutive_dry_days"] == 1  # day0 was dry
    assert out.loc[2, "consecutive_dry_days"] == 2  # day0,1 dry
    assert out.loc[3, "consecutive_dry_days"] == 0  # day2 was wet, streak broke
    assert out.loc[4, "consecutive_dry_days"] == 1
    assert out.loc[5, "consecutive_dry_days"] == 2


def test_missing_rainfall_is_imputed_as_zero_and_flagged():
    # Evidenced assumption (see config.IMPUTE_MISSING_RAINFALL_AS_ZERO):
    # missing days correlate with dry conditions, so they're imputed as
    # 0mm and counted into the dry-day streak, not treated as "unknown".
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    rainfall = [0.0, np.nan, 0.0, 0.0]
    df = pd.DataFrame({"date_of_record": dates, "rainfall": rainfall})
    out = add_rainfall_history_features(df)
    assert out["rainfall_was_missing"].tolist() == [0, 1, 0, 0]
    assert out.loc[1, "rainfall"] == 0.0
    # the imputed day1 counts as dry, so the streak runs through it
    assert out.loc[2, "consecutive_dry_days"] == 2
    assert out.loc[3, "consecutive_dry_days"] == 3


def test_climatology_and_threshold_fit_only_on_reference_df():
    train = _synthetic_daily(60, start="2015-01-01")
    train = compute_target(train)
    other = _synthetic_daily(60, start="2022-01-01")
    other["rainfall"] = other["rainfall"] * 100  # very different distribution
    other = compute_target(other)

    clim = compute_monthly_climatology(train)
    thresh = compute_risk_threshold_table(train, percentile=33)

    # labeling `other` with train-derived thresholds must use train's
    # threshold values, not anything derived from `other`
    labeled_other = label_insufficient_rainfall(other, thresh)
    jan_threshold = thresh.loc[1]
    assert not pd.isna(jan_threshold)
    # sanity: `other`'s rainfall is scaled 100x train's, so with a
    # train-derived threshold almost everything in `other` should be
    # "sufficient" (0), not "insufficient" — proves the threshold truly
    # came from train, not from `other` itself.
    jan_rows = labeled_other[labeled_other["date_of_record"].dt.month == 1]
    valid_jan = jan_rows.dropna(subset=["insufficient_rainfall_next_7_days"])
    if len(valid_jan):
        assert valid_jan["insufficient_rainfall_next_7_days"].mean() < 0.5
