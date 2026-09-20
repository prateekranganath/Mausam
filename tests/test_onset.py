import numpy as np
import pandas as pd
import pytest

from src.monsoon import onset as on


def series(year=2024, start="01-01", days=300, rainfall=0.0):
    """A district-shaped daily frame with a constant rainfall baseline."""
    dates = pd.date_range(f"{year}-{start}", periods=days, freq="D")
    return pd.DataFrame({"date_of_record": dates, "rainfall": np.full(days, float(rainfall))})


def set_rain(frame, start_date, values):
    frame = frame.copy()
    begin = frame.index[frame["date_of_record"] == pd.Timestamp(start_date)][0]
    for offset, value in enumerate(values):
        frame.loc[begin + offset, "rainfall"] = value
    return frame


def wet(n, mm=20.0):
    return [mm] * n


def dry(n):
    return [0.0] * n


# A threshold is passed explicitly throughout so these tests exercise the
# rule itself, not the percentile calibration (which validate_onset.py owns).
THRESHOLD = 100.0


def detect(frame, **kw):
    kw.setdefault("threshold_mm", THRESHOLD)
    kw.setdefault("year", 2024)
    return on.detect_onset(frame, "Testville", on.SOUTHWEST, **kw)


# --- the trigger stage ---------------------------------------------------------

def test_a_sustained_wet_spell_is_detected_as_onset():
    frame = set_rain(series(), "2024-06-01", wet(20))
    result = detect(frame)
    assert result.status == on.ONSET_CONFIRMED
    assert result.onset_date == "2024-06-01"
    assert result.trigger_rainy_days >= 5


def test_onset_date_is_the_first_rainy_day_not_the_window_start():
    """A qualifying window needs only 5 of 7 days wet, so it can begin on a
    dry day. Announcing a dry day as the day the monsoon arrived would be
    plainly wrong to anyone standing there."""
    frame = set_rain(series(), "2024-06-01", wet(20))
    result = detect(frame)
    onset = pd.Timestamp(result.onset_date)
    rain_that_day = frame.loc[frame["date_of_record"] == onset, "rainfall"].iloc[0]
    assert rain_that_day >= 2.5


def test_drizzle_does_not_trigger_onset_even_on_seven_rainy_days():
    """7 rainy days at 3mm clears the rainy-day count but not the total.
    Both conditions exist so that a wet-but-feeble week cannot qualify."""
    frame = set_rain(series(), "2024-06-01", wet(30, mm=3.0))
    assert detect(frame).status != on.ONSET_CONFIRMED


def test_one_torrential_day_does_not_trigger_onset():
    """Total alone would pass easily; the rainy-day count is what rejects it."""
    frame = set_rain(series(), "2024-06-01", [500.0] + dry(29))
    assert detect(frame).status != on.ONSET_CONFIRMED


def test_a_window_with_a_missing_day_is_not_evaluated():
    """A gap understates the total and can only depress the rainy-day count,
    so the window is rejected rather than judged on partial evidence."""
    frame = set_rain(series(), "2024-06-01", wet(20))
    frame.loc[frame["date_of_record"] == pd.Timestamp("2024-06-03"), "rainfall"] = np.nan
    result = detect(frame)
    # It should skip past the gap and onset later, not onset on 06-01.
    assert result.onset_date != "2024-06-01"


# --- the persistence stage: rejecting false onsets -----------------------------

def test_a_pre_monsoon_burst_followed_by_drought_is_rejected():
    """The whole reason persistence is a separate stage: a wet week then a
    dry fortnight is a pre-monsoon system, not monsoon onset."""
    frame = series()
    frame = set_rain(frame, "2024-05-10", wet(7))   # burst
    frame = set_rain(frame, "2024-05-17", dry(20))  # then nothing
    result = detect(frame)
    assert "2024-05-10" in result.rejected_candidates
    assert result.onset_date != "2024-05-10"


def test_the_scan_resumes_after_a_rejected_candidate():
    """A false onset must not end the season's search."""
    frame = series()
    frame = set_rain(frame, "2024-05-10", wet(7))   # false onset
    frame = set_rain(frame, "2024-05-17", dry(20))
    frame = set_rain(frame, "2024-06-10", wet(25))  # the real thing
    result = detect(frame)
    assert result.status == on.ONSET_CONFIRMED
    assert result.onset_date == "2024-06-10"
    assert result.rejected_candidates == ["2024-05-10"]


def test_a_dry_run_one_day_short_of_the_limit_still_confirms():
    """Boundary: ONSET_MAX_DRY_RUN_DAYS is 7, so a 6-day run must pass."""
    frame = series()
    frame = set_rain(frame, "2024-06-01", wet(7))
    frame = set_rain(frame, "2024-06-08", dry(6) + wet(4))
    result = detect(frame)
    assert result.status == on.ONSET_CONFIRMED
    assert result.persistence_longest_dry_run_days == 6


def test_a_dry_run_at_the_limit_rejects():
    frame = series()
    frame = set_rain(frame, "2024-06-01", wet(7))
    frame = set_rain(frame, "2024-06-08", dry(7) + wet(3))
    assert "2024-06-01" in detect(frame).rejected_candidates


def test_missing_days_are_not_counted_as_dry():
    """Unknown is not dry. Counting it as dry would manufacture false
    onsets out of gaps in the record."""
    assert on._longest_dry_run(np.array([0.0, 0.0, np.nan, 0.0, 0.0])) == 2


# --- the live states: onset_likely vs onset_confirmed --------------------------

def test_onset_is_only_likely_while_the_persistence_window_is_incomplete():
    """The single most important distinction in this module: rain has
    arrived but we cannot yet know whether it stays."""
    frame = set_rain(series(), "2024-06-01", wet(20))
    result = detect(frame, as_of=pd.Timestamp("2024-06-12"))
    assert result.status == on.ONSET_LIKELY
    assert result.onset_date == "2024-06-01"
    assert result.persistence_longest_dry_run_days is None


def test_the_same_season_confirms_once_the_window_completes():
    frame = set_rain(series(), "2024-06-01", wet(20))
    likely = detect(frame, as_of=pd.Timestamp("2024-06-12"))
    confirmed = detect(frame, as_of=pd.Timestamp("2024-06-25"))
    assert likely.status == on.ONSET_LIKELY
    assert confirmed.status == on.ONSET_CONFIRMED
    assert confirmed.onset_date == likely.onset_date  # same date, firmer claim


def test_pre_onset_while_the_window_is_open_and_nothing_has_happened():
    result = detect(series(), as_of=pd.Timestamp("2024-06-15"))
    assert result.status == on.PRE_ONSET


def test_no_onset_detected_once_the_window_has_closed_dry():
    result = detect(series(days=330), as_of=pd.Timestamp("2024-08-15"))
    assert result.status == on.NO_ONSET_DETECTED


def test_onset_never_detected_before_the_scan_window_opens():
    """April rain is not monsoon onset, however heavy."""
    frame = set_rain(series(), "2024-04-01", wet(30))
    assert detect(frame, as_of=pd.Timestamp("2024-04-30")).status in {
        on.OUTSIDE_SEASON, on.PRE_ONSET, on.NO_ONSET_DETECTED
    }
    assert detect(frame).onset_date != "2024-04-01"


# --- district-relative threshold ----------------------------------------------

def test_threshold_is_relative_to_the_district():
    """A wet district must get a higher bar than a dry one; that is the
    entire reason the threshold is a percentile."""
    wet_district = series(days=365, rainfall=20.0)
    dry_district = series(days=365, rainfall=0.5)
    assert on.trigger_threshold_mm(wet_district) > on.trigger_threshold_mm(dry_district)


def test_threshold_never_falls_below_the_absolute_floor():
    """In an arid district the 85th percentile can be ~0, which would let a
    single drizzly week count as monsoon onset."""
    from src.config import ONSET_ABSOLUTE_FLOOR_MM

    assert on.trigger_threshold_mm(series(days=365, rainfall=0.0)) == ONSET_ABSOLUTE_FLOOR_MM


def test_threshold_on_an_empty_record_falls_back_rather_than_crashing():
    from src.config import ONSET_ABSOLUTE_FLOOR_MM

    empty = pd.DataFrame({"date_of_record": pd.to_datetime([]), "rainfall": []})
    assert on.trigger_threshold_mm(empty) == ONSET_ABSOLUTE_FLOOR_MM


# --- both monsoons -------------------------------------------------------------

def test_a_northeast_monsoon_district_is_identified_from_its_own_rainfall():
    """Tamil Nadu-style districts get most rain in Oct-Dec; an SW-only rule
    would report them as having no monsoon at all."""
    frame = series(days=365)
    frame = set_rain(frame, "2024-10-15", wet(40))
    assert on.dominant_season(frame) == on.NORTHEAST


def test_a_southwest_monsoon_district_is_identified_from_its_own_rainfall():
    frame = series(days=365)
    frame = set_rain(frame, "2024-07-01", wet(40))
    assert on.dominant_season(frame) == on.SOUTHWEST


def test_northeast_onset_is_detected_in_its_own_window():
    frame = series(days=365)
    frame = set_rain(frame, "2024-10-15", wet(25))
    result = on.detect_onset(frame, "Testville", on.NORTHEAST, year=2024, threshold_mm=THRESHOLD)
    assert result.status == on.ONSET_CONFIRMED
    assert result.onset_date == "2024-10-15"


# --- climatology and anomaly ---------------------------------------------------

def multi_year(onset_month_days):
    parts = []
    for year, month_day in onset_month_days.items():
        frame = series(year=year, days=365)
        frame = set_rain(frame, f"{year}-{month_day}", wet(30))
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def test_climatology_medians_only_confirmed_onsets():
    frame = multi_year({y: "06-01" for y in range(2015, 2024)})
    climatology = on.onset_climatology(frame, "Testville", on.SOUTHWEST)
    assert climatology["n_seasons"] == 9
    assert climatology["median_day_of_year"] == pd.Timestamp("2015-06-01").dayofyear


def test_climatology_withholds_a_median_below_the_minimum_seasons():
    """Two seasons is not a climatology; reporting a median would look
    authoritative while meaning almost nothing."""
    frame = multi_year({2015: "06-01", 2016: "06-01"})
    climatology = on.onset_climatology(frame, "Testville", on.SOUTHWEST)
    assert climatology["median_day_of_year"] is None
    assert climatology["n_seasons"] < climatology["min_seasons_required"]


def test_anomaly_is_reported_against_the_districts_own_median():
    onsets = {y: "06-01" for y in range(2015, 2025)}
    onsets[2025] = "06-21"  # 20 days late
    frame = multi_year(onsets)
    status = on.onset_status(frame, "Testville", season=on.SOUTHWEST, as_of=pd.Timestamp("2025-08-01"))
    assert status["anomaly_days"] == 20
    assert "late" in status["anomaly_label"]


def test_status_moves_to_post_onset_well_after_onset():
    frame = multi_year({y: "06-01" for y in range(2015, 2026)})
    recent = on.onset_status(frame, "Testville", on.SOUTHWEST, as_of=pd.Timestamp("2025-06-20"))
    later = on.onset_status(frame, "Testville", on.SOUTHWEST, as_of=pd.Timestamp("2025-08-20"))
    assert recent["status"] == on.ONSET_CONFIRMED
    assert later["status"] == on.POST_ONSET


def test_status_response_says_it_is_not_the_imd_criterion():
    """A user must not be able to read this output as an IMD declaration."""
    frame = multi_year({y: "06-01" for y in range(2015, 2026)})
    status = on.onset_status(frame, "Testville", on.SOUTHWEST, as_of=pd.Timestamp("2025-08-20"))
    assert "NOT IMD" in status["not_imd_criterion"]
    assert "925 hPa" in status["not_imd_criterion"]


def test_status_reports_the_threshold_it_actually_used():
    frame = multi_year({y: "06-01" for y in range(2015, 2026)})
    status = on.onset_status(frame, "Testville", on.SOUTHWEST, as_of=pd.Timestamp("2025-08-20"))
    assert status["climatology"]["trigger_threshold_mm"] > 0
    assert str(status["climatology"]["trigger_threshold_mm"]) in status["method"]
