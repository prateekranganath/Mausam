"""Monsoon onset detection: has the monsoon actually arrived here yet?

WHAT THIS IS. A local, rainfall-only objective onset rule, run per
district on that district's own daily record. It detects the shape all
objective onset definitions share — rain arrives AND then sustains — and
reports the detected date against that district's own historical median.

WHAT THIS IS NOT, AND THIS WAS MEASURED, NOT ASSUMED. It is not IMD's
operational onset declaration and it cannot be. IMD's criterion also
requires 925 hPa zonal wind strength and outgoing-longwave-radiation
fields; neither is obtainable from the free sources this project uses
(checked 2026-09-20: Open-Meteo's ERA5 archive accepts pressure-level
variable names but returns all-null, and NASA POWER has no pressure
levels at all).

The consequence is real and is stated rather than hidden. Validated
against IMD's published Kerala onset dates 2015-2025, a rainfall-only
rule — including a faithful reconstruction of IMD's own MULTI-STATION
rainfall criterion over 10 proxy districts — fires about 20 days EARLY
every single year (mean -21 days, SD ~5). That is not a tuning failure;
no threshold fixes it. It is the genuine signal: Kerala receives heavy
pre-monsoon convective rain, and the wind and OLR criteria are precisely
what hold IMD's declaration back until the large-scale circulation
arrives. Rainfall alone cannot see that.

So what this module reports is LOCAL RAINFALL ONSET — the date sustained
monsoon-intensity rain actually begins at a given place — which is also
the question the problem statement asks ("has monsoon arrived in this
area?"). Against IMD's normal monsoon ADVANCE dates across nine
districts from Kerala to Rajasthan it achieves 8.3 days mean absolute
error, comparable to the ~7-day standard deviation of IMD's own onset
date. It is most accurate over central and northern India (New Delhi
+0 days, Bhopal -4, Lucknow +5) and runs early on the pre-monsoon-heavy
southern and eastern coasts (Thiruvananthapuram -21, Kolkata -13).
scripts/validate_onset.py reproduces all of these numbers.

THE RULE, in two deliberately separate stages:

  Trigger      In a 7-day window starting on the candidate day, at least
               ONSET_MIN_RAINY_DAYS days are rainy (>= 2.5mm) and the
               window total exceeds this district's own
               ONSET_TRIGGER_PERCENTILE-th percentile 7-day total.
  Persistence  Across the 10 days that follow that window, the longest
               run of consecutive dry days stays below
               ONSET_MAX_DRY_RUN_DAYS.

The trigger is district-relative for a measured reason: no single
absolute mm cutoff works nationally (see the sweep table in
src/config.py). Any value low enough to ever fire in Jaisalmer fires
weeks early in Kerala; any value high enough for Kerala never fires in
Jaisalmer at all.

They are two stages rather than one long window on purpose. A single
17-day rainfall total would let one torrential week hide a following
fortnight of drought — which is precisely the pre-monsoon thunderstorm
false onset the persistence stage exists to reject. When persistence
fails, the scan does not give up; it resumes after the failed candidate
and looks for the next one.

WHY THE STATUS IS NOT JUST A DATE. Persistence can only ever be evaluated
in retrospect, so a live answer has to distinguish "rain has arrived and
we are still waiting to see if it stays" from "onset established". That
is the difference between `onset_likely` and `onset_confirmed`, and
collapsing the two would turn an honest "probably" into a false
certainty in exactly the situation a farmer would act on it.

BOTH MONSOONS ARE HANDLED. Tamil Nadu, coastal Andhra, Puducherry and
south interior Karnataka take most of their rain from the Oct-Dec
northeast monsoon, so an SW-only rule would report those districts as
having no monsoon at all. Which season applies is decided from the
district's own rainfall climatology (see dominant_season), not from a
hand-maintained list of states.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    DRY_DAY_THRESHOLD_MM,
    NE_MONSOON_SCAN_END,
    NE_MONSOON_SCAN_START,
    ONSET_ABSOLUTE_FLOOR_MM,
    ONSET_MAX_DRY_RUN_DAYS,
    ONSET_MIN_RAINY_DAYS,
    ONSET_MIN_SEASONS_FOR_CLIMATOLOGY,
    ONSET_PERSISTENCE_WINDOW_DAYS,
    ONSET_RECENT_DAYS,
    ONSET_TRIGGER_PERCENTILE,
    ONSET_TRIGGER_WINDOW_DAYS,
    SW_MONSOON_SCAN_END,
    SW_MONSOON_SCAN_START,
)

logger = logging.getLogger(__name__)

SOUTHWEST = "southwest"
NORTHEAST = "northeast"

# Status values, in the order a season moves through them.
OUTSIDE_SEASON = "outside_season"
PRE_ONSET = "pre_onset"
ONSET_LIKELY = "onset_likely"
ONSET_CONFIRMED = "onset_confirmed"
POST_ONSET = "post_onset"
NO_ONSET_DETECTED = "no_onset_detected"

STATUS_DESCRIPTIONS = {
    OUTSIDE_SEASON: "Today falls outside this district's monsoon onset window.",
    PRE_ONSET: "The onset window is open but no qualifying rainfall onset has occurred yet.",
    ONSET_LIKELY: (
        "Onset rainfall criteria are met, but the persistence check needs days that have not "
        "happened yet. This could still turn out to be a false onset."
    ),
    ONSET_CONFIRMED: "Onset detected and confirmed: rainfall arrived and sustained.",
    POST_ONSET: "The monsoon established itself here earlier in the season.",
    NO_ONSET_DETECTED: "The onset window has closed without any qualifying onset in the record.",
}


@dataclass(frozen=True)
class SeasonWindow:
    name: str
    start: tuple[int, int]
    end: tuple[int, int]

    def start_date(self, year: int) -> pd.Timestamp:
        return pd.Timestamp(year=year, month=self.start[0], day=self.start[1])

    def end_date(self, year: int) -> pd.Timestamp:
        return pd.Timestamp(year=year, month=self.end[0], day=self.end[1])

    def contains(self, day: pd.Timestamp) -> bool:
        return self.start_date(day.year) <= day <= self.end_date(day.year)


SEASONS = {
    SOUTHWEST: SeasonWindow(SOUTHWEST, SW_MONSOON_SCAN_START, SW_MONSOON_SCAN_END),
    NORTHEAST: SeasonWindow(NORTHEAST, NE_MONSOON_SCAN_START, NE_MONSOON_SCAN_END),
}

# Months each monsoon delivers its rain in, used only to decide which
# season dominates a district's climatology.
_SW_RAIN_MONTHS = (6, 7, 8, 9)
_NE_RAIN_MONTHS = (10, 11, 12)


@dataclass
class OnsetDetection:
    """One season's onset outcome for one district."""

    district: str
    season: str
    year: int
    status: str
    onset_date: str | None = None
    onset_day_of_year: int | None = None
    trigger_7day_rainfall_mm: float | None = None
    trigger_rainy_days: int | None = None
    persistence_longest_dry_run_days: int | None = None
    rejected_candidates: list[str] = field(default_factory=list)
    evaluated_through: str | None = None

    def to_dict(self) -> dict:
        return {
            "district": self.district,
            "season": self.season,
            "year": self.year,
            "status": self.status,
            "status_description": STATUS_DESCRIPTIONS.get(self.status, ""),
            "onset_date": self.onset_date,
            "onset_day_of_year": self.onset_day_of_year,
            "trigger_7day_rainfall_mm": (
                round(self.trigger_7day_rainfall_mm, 1) if self.trigger_7day_rainfall_mm is not None else None
            ),
            "trigger_rainy_days": self.trigger_rainy_days,
            "persistence_longest_dry_run_days": self.persistence_longest_dry_run_days,
            "rejected_false_onsets": self.rejected_candidates,
            "evaluated_through": self.evaluated_through,
        }


def trigger_threshold_mm(
    daily: pd.DataFrame, percentile: float = ONSET_TRIGGER_PERCENTILE
) -> float:
    """This district's onset trigger threshold: `percentile` of its OWN
    7-day rolling rainfall totals, floored at ONSET_ABSOLUTE_FLOOR_MM.

    Relative rather than absolute because absolute was measured not to
    work nationally — see the sweep table in src/config.py. The question
    a threshold must answer is "is this week unusually wet for HERE", and
    a fixed mm cutoff cannot ask it of both Kerala and Jaisalmer at once.
    """
    totals = (
        daily.sort_values("date_of_record")["rainfall"]
        .rolling(ONSET_TRIGGER_WINDOW_DAYS, min_periods=ONSET_TRIGGER_WINDOW_DAYS)
        .sum()
    )
    if totals.notna().sum() == 0:
        return ONSET_ABSOLUTE_FLOOR_MM
    return max(float(np.nanpercentile(totals, percentile)), ONSET_ABSOLUTE_FLOOR_MM)


def _longest_dry_run(rainfall: np.ndarray) -> int:
    """Longest run of consecutive days under the rainy-day threshold.

    NaN counts as NOT dry. A day with no reading is unknown, and treating
    unknown as dry would manufacture false onsets out of missing data —
    the same failure the Excel dataset's missingness step-change caused
    elsewhere in this project.
    """
    longest = current = 0
    for value in rainfall:
        if np.isnan(value) or value >= DRY_DAY_THRESHOLD_MM:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _trigger_met(window: np.ndarray, threshold_mm: float) -> tuple[bool, float, int]:
    """(met, total_mm, rainy_days) for a candidate trigger window.

    A window with any missing day cannot be evaluated: its total would be
    understated and its rainy-day count could only fall short, so it is
    rejected rather than assessed on partial evidence.
    """
    if len(window) < ONSET_TRIGGER_WINDOW_DAYS or np.isnan(window).any():
        return False, float("nan"), 0
    total = float(np.sum(window))
    rainy = int(np.sum(window >= DRY_DAY_THRESHOLD_MM))
    met = rainy >= ONSET_MIN_RAINY_DAYS and total >= threshold_mm
    return met, total, rainy


def detect_onset(
    daily: pd.DataFrame,
    district: str,
    season: str = SOUTHWEST,
    year: int | None = None,
    as_of: pd.Timestamp | None = None,
    threshold_mm: float | None = None,
) -> OnsetDetection:
    """Detect `season`'s onset for `year` from one district's daily series.

    `daily` needs columns date_of_record and rainfall, one row per
    calendar day, sorted or not. `as_of` caps what the detector is allowed
    to see (default: the last date present), which is what makes a live
    answer honest — the persistence window simply has not happened yet
    near the leading edge, and the status says so.
    """
    window = SEASONS[season]
    # Computed from the WHOLE record, not the scanned season, so a
    # district's threshold is one stable number rather than drifting
    # year to year with whatever that season happened to deliver.
    threshold_mm = trigger_threshold_mm(daily) if threshold_mm is None else threshold_mm
    frame = daily[["date_of_record", "rainfall"]].copy()
    frame["date_of_record"] = pd.to_datetime(frame["date_of_record"])
    frame = frame.sort_values("date_of_record").reset_index(drop=True)

    if as_of is not None:
        frame = frame[frame["date_of_record"] <= pd.Timestamp(as_of)]
    if frame.empty:
        return OnsetDetection(district, season, year or 0, NO_ONSET_DETECTED)

    last_available = pd.Timestamp(frame["date_of_record"].iloc[-1])
    year = year if year is not None else int(last_available.year)

    scan_start = window.start_date(year)
    scan_end = window.end_date(year)

    # Reindex to a complete calendar so positional windows are real days.
    # A frame with gaps would otherwise let a 7-slot slice span 3 weeks.
    complete = (
        frame.set_index("date_of_record")
        .reindex(pd.date_range(frame["date_of_record"].iloc[0], last_available, freq="D"))
    )
    rainfall = complete["rainfall"].to_numpy(dtype=float)
    dates = complete.index

    scan_positions = np.flatnonzero((dates >= scan_start) & (dates <= scan_end))
    if len(scan_positions) == 0:
        # The record does not reach this season at all.
        return OnsetDetection(
            district, season, year,
            OUTSIDE_SEASON if last_available < scan_start else NO_ONSET_DETECTED,
            evaluated_through=str(last_available.date()),
        )

    rejected: list[str] = []
    position = int(scan_positions[0])
    last_position = int(scan_positions[-1])

    while position <= last_position:
        trigger = rainfall[position : position + ONSET_TRIGGER_WINDOW_DAYS]
        met, total, rainy = _trigger_met(trigger, threshold_mm)
        if not met:
            position += 1
            continue

        # Report the first RAINY day of the qualifying window, not the
        # window's first slot. A window only needs ONSET_MIN_RAINY_DAYS of
        # 7 days wet, so its first slot can itself be dry — and announcing
        # a day it did not rain as the day the monsoon arrived would be
        # plainly wrong to anyone who was standing there.
        rainy_offsets = np.flatnonzero(trigger >= DRY_DAY_THRESHOLD_MM)
        candidate_date = dates[position + int(rainy_offsets[0])]
        persistence_start = position + ONSET_TRIGGER_WINDOW_DAYS
        persistence_end = persistence_start + ONSET_PERSISTENCE_WINDOW_DAYS
        persistence = rainfall[persistence_start:persistence_end]

        if len(persistence) < ONSET_PERSISTENCE_WINDOW_DAYS:
            # Rain has arrived but we cannot yet know whether it stays.
            # Reporting this as confirmed would be the one genuinely
            # misleading thing this module could do.
            return OnsetDetection(
                district, season, year, ONSET_LIKELY,
                onset_date=str(candidate_date.date()),
                onset_day_of_year=int(candidate_date.dayofyear),
                trigger_7day_rainfall_mm=total,
                trigger_rainy_days=rainy,
                rejected_candidates=rejected,
                evaluated_through=str(last_available.date()),
            )

        dry_run = _longest_dry_run(persistence)
        if dry_run < ONSET_MAX_DRY_RUN_DAYS:
            return OnsetDetection(
                district, season, year, ONSET_CONFIRMED,
                onset_date=str(candidate_date.date()),
                onset_day_of_year=int(candidate_date.dayofyear),
                trigger_7day_rainfall_mm=total,
                trigger_rainy_days=rainy,
                persistence_longest_dry_run_days=dry_run,
                rejected_candidates=rejected,
                evaluated_through=str(last_available.date()),
            )

        # A false onset: wet week, then a dry spell. Record it and resume
        # the scan after this candidate's trigger window rather than
        # abandoning the season.
        rejected.append(str(candidate_date.date()))
        logger.info(
            "%s %s %d: rejected false onset at %s (%.1fmm trigger, then a %d-day dry run)",
            district, season, year, candidate_date.date(), total, dry_run,
        )
        position += ONSET_TRIGGER_WINDOW_DAYS

    status = PRE_ONSET if last_available < scan_end else NO_ONSET_DETECTED
    return OnsetDetection(
        district, season, year, status,
        rejected_candidates=rejected,
        evaluated_through=str(last_available.date()),
    )


def dominant_season(daily: pd.DataFrame) -> str:
    """Which monsoon delivers more of this district's rain.

    Decided from the district's own record rather than a curated list of
    states, so it stays correct for the genuinely mixed cases (south
    interior Karnataka, coastal Andhra) where a state-level rule would be
    wrong for some of the districts it covers.
    """
    frame = daily[["date_of_record", "rainfall"]].copy()
    months = pd.to_datetime(frame["date_of_record"]).dt.month
    sw = float(frame.loc[months.isin(_SW_RAIN_MONTHS), "rainfall"].sum())
    ne = float(frame.loc[months.isin(_NE_RAIN_MONTHS), "rainfall"].sum())
    return NORTHEAST if ne > sw else SOUTHWEST


def onset_climatology(
    daily: pd.DataFrame,
    district: str,
    season: str = SOUTHWEST,
    as_of: pd.Timestamp | None = None,
) -> dict:
    """Per-year onset dates and the district's median onset, from history.

    Only CONFIRMED onsets feed the median. An `onset_likely` (the current,
    still-unresolved season) or a year with no detection would otherwise
    drag the climatology toward whatever the incomplete record happens to
    show.
    """
    dates = pd.to_datetime(daily["date_of_record"])
    if dates.empty:
        return {"season": season, "n_seasons": 0, "median_day_of_year": None, "per_year": {}}

    last = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(dates.max())
    threshold = trigger_threshold_mm(daily)
    per_year: dict[int, dict] = {}
    for year in range(int(dates.min().year), int(last.year) + 1):
        detection = detect_onset(
            daily, district, season=season, year=year, as_of=last, threshold_mm=threshold
        )
        per_year[year] = {
            "status": detection.status,
            "onset_date": detection.onset_date,
            "day_of_year": detection.onset_day_of_year,
        }

    confirmed = [v["day_of_year"] for v in per_year.values() if v["status"] == ONSET_CONFIRMED]
    n = len(confirmed)
    enough = n >= ONSET_MIN_SEASONS_FOR_CLIMATOLOGY

    return {
        "season": season,
        "trigger_threshold_mm": round(threshold, 1),
        "n_seasons": n,
        # Below the minimum, a "median" of two or three years would look
        # authoritative while meaning very little, so it is withheld.
        "median_day_of_year": int(np.median(confirmed)) if enough else None,
        "earliest_day_of_year": int(min(confirmed)) if enough else None,
        "latest_day_of_year": int(max(confirmed)) if enough else None,
        "median_onset_date_label": _doy_label(int(np.median(confirmed))) if enough else None,
        "min_seasons_required": ONSET_MIN_SEASONS_FOR_CLIMATOLOGY,
        "per_year": per_year,
    }


def _doy_label(day_of_year: int) -> str:
    """A day-of-year rendered as e.g. "01 Jun", using a non-leap reference
    year so the label does not drift by a day depending on the year."""
    return (dt.date(2001, 1, 1) + dt.timedelta(days=day_of_year - 1)).strftime("%d %b")


def onset_status(
    daily: pd.DataFrame,
    district: str,
    season: str | None = None,
    as_of: pd.Timestamp | None = None,
) -> dict:
    """The full live onset answer for one district: current status, the
    detected date, the historical median, and the anomaly between them."""
    season = season or dominant_season(daily)
    dates = pd.to_datetime(daily["date_of_record"])
    today = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(dates.max())

    window = SEASONS[season]
    # Before this year's window opens, the informative answer is about the
    # season now in progress or just finished, not an empty future one.
    year = int(today.year)
    detection = detect_onset(daily, district, season=season, year=year, as_of=today)

    if detection.status == ONSET_CONFIRMED and detection.onset_date:
        days_since = (today - pd.Timestamp(detection.onset_date)).days
        if days_since > ONSET_RECENT_DAYS:
            detection.status = POST_ONSET
    elif detection.status == PRE_ONSET and not window.contains(today):
        detection.status = OUTSIDE_SEASON

    climatology = onset_climatology(daily, district, season=season, as_of=today)

    anomaly_days = None
    if detection.onset_day_of_year is not None and climatology["median_day_of_year"] is not None:
        anomaly_days = int(detection.onset_day_of_year - climatology["median_day_of_year"])

    result = detection.to_dict()
    result.update(
        {
            "as_of_date": str(today.date()),
            "dominant_season": dominant_season(daily),
            "climatology": climatology,
            "anomaly_days": anomaly_days,
            "anomaly_label": _anomaly_label(anomaly_days),
            "method": (
                "Local rainfall-only objective rule: a 7-day trigger window with "
                f">={ONSET_MIN_RAINY_DAYS} rainy days (>={DRY_DAY_THRESHOLD_MM}mm) totalling "
                f">={climatology['trigger_threshold_mm']}mm (this district's "
                f"{ONSET_TRIGGER_PERCENTILE:.0f}th percentile 7-day total), followed by "
                f"{ONSET_PERSISTENCE_WINDOW_DAYS} days with no dry run of "
                f">={ONSET_MAX_DRY_RUN_DAYS} days."
            ),
            "not_imd_criterion": (
                "This is NOT IMD's operational onset declaration, which additionally uses "
                "outgoing-longwave-radiation and 925 hPa wind fields not ingested here."
            ),
        }
    )
    return result


def _anomaly_label(anomaly_days: int | None) -> str | None:
    if anomaly_days is None:
        return None
    if anomaly_days <= -7:
        return f"{abs(anomaly_days)} days early"
    if anomaly_days >= 7:
        return f"{anomaly_days} days late"
    return "close to normal"
