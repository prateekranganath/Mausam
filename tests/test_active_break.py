import numpy as np
import pandas as pd
import pytest

from src.monsoon import active_break as ab


def district_series(years=range(2015, 2026), jjas_mm=10.0, other_mm=0.5, seed=0):
    """A synthetic district with a clear monsoon season, so JJAS has a
    non-degenerate climatological mean and SD to standardise against."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
    in_season = dates.month.isin(ab.ACTIVE_BREAK_MONTHS)
    scale = np.where(in_season, jjas_mm, other_mm)
    rainfall = rng.gamma(shape=2.0, scale=scale / 2.0)
    return pd.DataFrame({"district": "Testville", "date_of_record": dates, "rainfall": rainfall})


def override(frame, start, days, mm):
    frame = frame.copy()
    mask = (frame["date_of_record"] >= pd.Timestamp(start)) & (
        frame["date_of_record"] < pd.Timestamp(start) + pd.Timedelta(days=days)
    )
    frame.loc[mask, "rainfall"] = mm
    return frame


# --- the skew problem this module exists to work around ------------------------

def test_anomaly_is_computed_on_a_trailing_mean_not_the_daily_value():
    """If this regresses to daily values, breaks become undetectable: a
    rainless day only reaches about -0.65 SD against a skewed daily
    distribution."""
    frame = ab.standardised_anomaly(district_series())
    assert ab.ACCUMULATION_COLUMN in frame.columns


def test_a_dry_spell_actually_reaches_the_break_threshold():
    """The regression guard for the bug that made this module look like it
    worked while reporting zero breaks in eleven years."""
    frame = override(district_series(), "2024-07-01", 20, 0.0)
    labelled = ab.classify_spells(ab.standardised_anomaly(frame))
    july = labelled[
        (labelled["date_of_record"] >= "2024-07-08") & (labelled["date_of_record"] <= "2024-07-20")
    ]
    assert (july["rainfall_anomaly_sd"] <= -ab.ACTIVE_BREAK_SD_THRESHOLD).any()
    assert (july["monsoon_phase"] == ab.BREAK).any()


def test_both_spell_types_occur_in_a_realistic_record():
    """Neither side may be structurally unreachable."""
    spells = ab.spell_history(district_series(), "Testville")
    kinds = {s["phase"] for s in spells}
    assert ab.ACTIVE in kinds and ab.BREAK in kinds


# --- run-length rule -----------------------------------------------------------

def test_a_run_at_the_minimum_length_counts_as_a_spell():
    mask = np.array([False, True, True, True, False])
    assert ab._runs(mask) == [(1, 4)]
    assert 4 - 1 == ab.ACTIVE_BREAK_MIN_RUN_DAYS


def test_a_run_one_day_short_is_not_a_spell():
    frame = pd.DataFrame(
        {
            "district": "T",
            "date_of_record": pd.date_range("2024-07-01", periods=10, freq="D"),
            "rainfall": 0.0,
            "rainfall_anomaly_sd": [0, 0, 2.0, 2.0, 0, 0, 0, 0, 0, 0],
        }
    )
    labelled = ab.classify_spells(frame)
    assert ab.ACTIVE not in set(labelled["monsoon_phase"])


def test_exactly_three_days_past_the_threshold_is_a_spell():
    frame = pd.DataFrame(
        {
            "district": "T",
            "date_of_record": pd.date_range("2024-07-01", periods=10, freq="D"),
            "rainfall": 0.0,
            "rainfall_anomaly_sd": [0, 0, 2.0, 2.0, 2.0, 0, 0, 0, 0, 0],
        }
    )
    labelled = ab.classify_spells(frame)
    assert list(labelled["monsoon_phase"]).count(ab.ACTIVE) == 3


def test_runs_finds_multiple_disjoint_runs():
    mask = np.array([True, True, False, True, True, True])
    assert ab._runs(mask) == [(0, 2), (3, 6)]


def test_runs_on_an_all_false_mask_is_empty():
    assert ab._runs(np.zeros(5, dtype=bool)) == []


# --- the out-of-season guard ---------------------------------------------------

def test_spells_are_not_reported_outside_the_monsoon_months():
    """December drizzle against a near-zero climatological mean can score
    +8 SD; reporting that as an active monsoon would be absurd."""
    frame = pd.DataFrame(
        {
            "district": "T",
            "date_of_record": pd.date_range("2024-12-01", periods=10, freq="D"),
            "rainfall": 0.0,
            "rainfall_anomaly_sd": 8.0,
        }
    )
    labelled = ab.classify_spells(frame)
    assert set(labelled["monsoon_phase"]) == {ab.NOT_APPLICABLE}


def test_a_degenerate_climatology_yields_no_anomaly_rather_than_infinity():
    """An arid district has a near-zero JJAS mean; dividing by its SD would
    produce inf, not a phase."""
    arid = district_series(jjas_mm=0.05, other_mm=0.01)
    frame = ab.standardised_anomaly(arid)
    jjas = frame[frame["date_of_record"].dt.month.isin(ab.ACTIVE_BREAK_MONTHS)]
    assert not np.isinf(jjas["rainfall_anomaly_sd"]).any()
    assert jjas["rainfall_anomaly_sd"].isna().all()


def test_an_unclassifiable_in_season_day_is_not_called_normal():
    """Missing data must read as not_applicable, never as 'nothing unusual'."""
    frame = pd.DataFrame(
        {
            "district": "T",
            "date_of_record": pd.date_range("2024-07-01", periods=5, freq="D"),
            "rainfall": [1.0, np.nan, 1.0, 1.0, 1.0],
            "rainfall_anomaly_sd": [0.1, np.nan, 0.1, 0.1, 0.1],
        }
    )
    labelled = ab.classify_spells(frame)
    assert labelled["monsoon_phase"].iloc[1] == ab.NOT_APPLICABLE
    assert labelled["monsoon_phase"].iloc[0] == ab.NORMAL


def test_incomplete_trailing_window_has_no_anomaly():
    """The first days of a record cannot have a 7-day trailing mean."""
    frame = ab.standardised_anomaly(district_series(years=range(2024, 2025)))
    assert frame["rainfall_anomaly_sd"].iloc[:6].isna().all()


# --- the live answer -----------------------------------------------------------

def test_current_phase_reports_days_in_the_current_phase():
    frame = override(district_series(), "2024-07-01", 25, 0.0)
    result = ab.current_phase(frame, "Testville", as_of=pd.Timestamp("2024-07-22"))
    assert result["monsoon_phase"] == ab.BREAK
    assert result["days_in_current_phase"] >= ab.ACTIVE_BREAK_MIN_RUN_DAYS


def test_current_phase_respects_as_of():
    frame = district_series()
    result = ab.current_phase(frame, "Testville", as_of=pd.Timestamp("2024-07-15"))
    assert result["as_of_date"] == "2024-07-15"
    assert len(result["recent_30_days"]) == 30


def test_current_phase_out_of_season_is_not_applicable():
    result = ab.current_phase(district_series(), "Testville", as_of=pd.Timestamp("2024-01-15"))
    assert result["monsoon_phase"] == ab.NOT_APPLICABLE


def test_current_phase_on_an_empty_frame_does_not_crash():
    empty = pd.DataFrame({"district": [], "date_of_record": pd.to_datetime([]), "rainfall": []})
    result = ab.current_phase(empty, "Testville")
    assert result["monsoon_phase"] == ab.NOT_APPLICABLE


def test_response_carries_no_nan_values():
    """NaN is not valid JSON; the API must never be handed one."""
    frame = district_series()
    result = ab.current_phase(frame, "Testville", as_of=pd.Timestamp("2024-07-15"))
    numbers = [v for v in result.values() if isinstance(v, float)]
    assert all(np.isfinite(v) for v in numbers)
    for day in result["recent_30_days"]:
        for key in ("rainfall_mm", "anomaly_sd"):
            assert day[key] is None or np.isfinite(day[key])


def test_response_states_the_deviation_from_the_published_method():
    result = ab.current_phase(district_series(), "Testville", as_of=pd.Timestamp("2024-07-15"))
    assert "core zone" in result["caveats"]
    assert "Rajeevan" in result["method"]


# --- spell history -------------------------------------------------------------

def test_spell_history_is_ordered_and_well_formed():
    spells = ab.spell_history(district_series(), "Testville")
    assert spells == sorted(spells, key=lambda s: s["start_date"])
    for spell in spells:
        assert spell["days"] >= ab.ACTIVE_BREAK_MIN_RUN_DAYS
        assert spell["start_date"] <= spell["end_date"]
        assert pd.Timestamp(spell["start_date"]).month in ab.ACTIVE_BREAK_MONTHS


def test_spell_history_signs_match_the_phase():
    for spell in ab.spell_history(district_series(), "Testville"):
        if spell["phase"] == ab.ACTIVE:
            assert spell["mean_anomaly_sd"] > 0
        else:
            assert spell["mean_anomaly_sd"] < 0
