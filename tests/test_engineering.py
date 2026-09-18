import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    add_rainfall_history_features,
    attach_climatology,
    build_features_multi_district,
    compute_monthly_climatology,
    compute_risk_threshold_table,
    compute_target,
    compute_target_multi_district,
    label_insufficient_rainfall,
)


def _synthetic_daily(n=15, start="2020-01-01", district="TestDist"):
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {"date_of_record": dates, "rainfall": list(range(n)), "district": district}
    )


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
    jan_threshold = thresh.loc[("TestDist", 1)]
    assert not pd.isna(jan_threshold)
    # sanity: `other`'s rainfall is scaled 100x train's, so with a
    # train-derived threshold almost everything in `other` should be
    # "sufficient" (0), not "insufficient" — proves the threshold truly
    # came from train, not from `other` itself.
    jan_rows = labeled_other[labeled_other["date_of_record"].dt.month == 1]
    valid_jan = jan_rows.dropna(subset=["insufficient_rainfall_next_7_days"])
    if len(valid_jan):
        assert valid_jan["insufficient_rainfall_next_7_days"].mean() < 0.5


def test_climatology_is_never_shared_across_districts():
    # District A: heavy rain every day. District B: bone dry every day.
    # A's climatology/threshold must never leak into B's rows or vice versa.
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    wet = pd.DataFrame({"date_of_record": dates, "rainfall": 50.0, "district": "Wet"})
    dry = pd.DataFrame({"date_of_record": dates, "rainfall": 0.0, "district": "Dry"})
    train = pd.concat([wet, dry], ignore_index=True)
    train = compute_target_multi_district(train)

    clim = compute_monthly_climatology(train)
    thresh = compute_risk_threshold_table(train, percentile=33)

    assert clim.loc[("Wet", 1), "climatology_mean_rainfall_month"] == pytest.approx(50.0)
    assert clim.loc[("Dry", 1), "climatology_mean_rainfall_month"] == pytest.approx(0.0)
    assert thresh.loc[("Wet", 1)] > thresh.loc[("Dry", 1)]

    labeled = label_insufficient_rainfall(train, thresh)
    # Dry district's own (0mm) rainfall should never register as
    # "insufficient" against ITS OWN threshold (which is also ~0) —
    # if Wet's threshold had leaked in, every Dry row would flag positive.
    dry_labeled = labeled[(labeled["district"] == "Dry") & labeled["insufficient_rainfall_next_7_days"].notna()]
    assert dry_labeled["insufficient_rainfall_next_7_days"].mean() < 0.5


def test_build_features_multi_district_does_not_leak_across_boundary():
    # District A ends with 7 wet days; District B starts right after in
    # the concatenated frame. B's rainfall_sum_last_7 at its first valid
    # row must not include any of A's rainfall.
    dates_a = pd.date_range("2020-01-01", periods=35, freq="D")
    dates_b = pd.date_range("2020-01-01", periods=35, freq="D")
    weather = {"avg_temp": 25.0, "min_temp": 20.0, "max_temp": 30.0, "wind_speed": 2.0, "air_pressure": 1010.0}
    a = pd.DataFrame({"date_of_record": dates_a, "rainfall": 100.0, "district": "A", **weather})
    b = pd.DataFrame({"date_of_record": dates_b, "rainfall": 0.0, "district": "B", **weather})
    combined = pd.concat([a, b], ignore_index=True)

    out = build_features_multi_district(combined)
    b_rows = out[out["district"] == "B"].reset_index(drop=True)
    # B is all zeros, so its 7-day trailing sum must be 0 once defined —
    # any contamination from A's 100mm days would make this nonzero.
    valid = b_rows["rainfall_sum_last_7"].dropna()
    assert (valid == 0).all()
