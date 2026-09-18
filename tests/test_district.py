import numpy as np
import pandas as pd
import pytest

from src.data.district import (
    AmbiguousDistrictError,
    DistrictNotFoundError,
    get_district_daily_series,
    resolve_district,
)


def _fake_clean_df():
    # Two physical stations sharing the SAME station_name but different
    # coordinates, mirroring the real Thiruvananthapuram data-quality
    # finding — aggregation must not collapse them by name alone.
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "date_of_record": d, "district": "TestDist", "state": "TS",
            "station_name": "Sta", "station_id": "Sta__1.0__1.0",
            "latitude": 1.0, "longitude": 1.0, "elevation": 10,
            "rainfall": float(i), "avg_temp": 20.0, "min_temp": 15.0,
            "max_temp": 25.0, "wind_speed": 2.0, "air_pressure": 1000.0,
        })
        rows.append({
            "date_of_record": d, "district": "TestDist", "state": "TS",
            "station_name": "Sta", "station_id": "Sta__2.0__2.0",
            "latitude": 2.0, "longitude": 2.0, "elevation": 20,
            "rainfall": float(i) + 10, "avg_temp": 22.0, "min_temp": 17.0,
            "max_temp": 27.0, "wind_speed": 3.0, "air_pressure": 1002.0,
        })
    # a second, unrelated district that should never leak in
    rows.append({
        "date_of_record": dates[0], "district": "Other", "state": "TS",
        "station_name": "X", "station_id": "X__5.0__5.0",
        "latitude": 5.0, "longitude": 5.0, "elevation": 5,
        "rainfall": 999.0, "avg_temp": 40.0, "min_temp": 30.0,
        "max_temp": 45.0, "wind_speed": 1.0, "air_pressure": 990.0,
    })
    return pd.DataFrame(rows)


def test_aggregates_across_same_named_stations():
    df = _fake_clean_df()
    daily = get_district_daily_series(df, "TestDist")
    # day0: rainfall values 0 and 10 -> mean 5.0
    assert daily.loc[0, "rainfall"] == pytest.approx(5.0)
    assert (daily["district"] == "TestDist").all()
    assert "Other" not in daily["district"].values


def test_missing_reading_excluded_from_mean_not_zero():
    df = _fake_clean_df()
    # blank out one station's rainfall on day0
    df.loc[(df["date_of_record"] == df["date_of_record"].min()) & (df["station_id"] == "Sta__1.0__1.0"), "rainfall"] = np.nan
    daily = get_district_daily_series(df, "TestDist")
    # only the remaining station's value (10.0) should be averaged, not (0+10)/2 nor 10/2
    assert daily.loc[0, "rainfall"] == pytest.approx(10.0)


def test_gap_days_are_present_as_nan_not_dropped():
    df = _fake_clean_df()
    # remove all rows for day index 2 entirely (a full reporting gap)
    gap_date = df["date_of_record"].unique()[2]
    df = df[df["date_of_record"] != gap_date]
    daily = get_district_daily_series(df, "TestDist")
    assert len(daily) == 5  # still one row per calendar day
    gap_row = daily[daily["date_of_record"] == gap_date]
    assert gap_row["rainfall"].isna().all()


def test_district_not_found_raises():
    df = _fake_clean_df()
    with pytest.raises(DistrictNotFoundError):
        get_district_daily_series(df, "NoSuchDistrict")


def test_case_insensitive_lookup():
    df = _fake_clean_df()
    d, s = resolve_district(df, "testdist")
    assert d == "TestDist"
    assert s == "TS"


def test_ambiguous_district_requires_state():
    df = _fake_clean_df()
    extra = df[df["district"] == "TestDist"].copy()
    extra["state"] = "OTHERSTATE"
    df2 = pd.concat([df, extra], ignore_index=True)
    with pytest.raises(AmbiguousDistrictError):
        resolve_district(df2, "TestDist")
    # disambiguated with state= should work
    d, s = resolve_district(df2, "TestDist", state="TS")
    assert s == "TS"
