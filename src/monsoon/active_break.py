"""Active and break spells: within the monsoon, is it raining or paused?

Onset answers "has the monsoon arrived". This answers the question that
matters for the rest of the season: the Indian monsoon does not rain
steadily from June to September, it alternates between ACTIVE spells of
vigorous rain and BREAK spells of near-drought that can last two weeks.
A break during the flowering stage of a kharif crop does far more damage
than the seasonal total suggests, which is why the phase is worth
reporting separately from any rainfall amount.

METHOD, after Rajeevan et al. (2010), "Active and break spells of the
Indian summer monsoon": standardise rainfall against a smoothed
day-of-year climatology, then call a spell ACTIVE where the anomaly
stays above +1 SD for at least 3 consecutive days and BREAK where it
stays below -1 SD for at least 3 consecutive days.

WHAT IS STANDARDISED IS A 7-DAY TRAILING MEAN, NOT THE DAILY VALUE, AND
THAT IS NOT A STYLISTIC CHOICE. Single-district daily rainfall is far too
right-skewed for a symmetric +/-1 SD rule: most days are dry and a few are
torrential, so the mean sits well above the median and a completely
RAINLESS day still only scores about -0.65 SD. Measured over JJAS
2015-2026, the lowest daily z-score reachable at all was -0.68 (Nagpur),
-0.69 (Thiruvananthapuram) and -0.64 (Bhopal) -- so with daily values a
BREAK spell is not merely rare, it is ARITHMETICALLY IMPOSSIBLE, and an
earlier version of this module duly reported zero breaks in eleven years
while appearing to work. Averaging over 7 days cuts the skew from ~3.1-3.6
to ~1.6-1.7 and makes the two sides roughly symmetric (Nagpur: 12.0% of
days below -1 SD against 13.2% above). The paper sidesteps the same
problem a different way, by averaging over the whole core zone rather than
over time; a per-district product cannot do that.

The mean is TRAILING rather than centred so that the identical code path
works in real time -- a centred window would need days that have not
happened yet.

TWO DELIBERATE DEVIATIONS FROM THE PAPER, both of which make results
noisier than the published ones and neither of which is hidden:

  1. The paper defines spells over the monsoon CORE ZONE as one region,
     averaging out local convective noise. This runs per district,
     because a district-level product has to answer per district. A
     single district's daily rainfall is far noisier than a regional
     mean, so short spells here are less reliable than the paper's.
  2. The paper uses a long high-quality gridded rainfall record. This
     uses ~11 years of NASA POWER reanalysis, so the day-of-year SD is
     estimated from fewer years.
  3. Time-averaging replaces the paper's space-averaging, for the skew
     reason above. A spell here therefore means ">= 3 consecutive days on
     which the trailing week was anomalous", a slightly smoother and
     slower-responding signal than the paper's.

JJAS ONLY, AND THIS IS A CORRECTNESS ISSUE RATHER THAN A PREFERENCE.
Outside the monsoon months the climatological daily mean approaches zero
in most of India, and a standardised anomaly with a near-zero denominator
explodes: 4mm of December drizzle in Rajasthan can score +8 SD and would
be reported as a violently "active monsoon" in midwinter. The detector
therefore returns `not_applicable` outside ACTIVE_BREAK_MONTHS rather
than a number that would be arithmetically valid and physically absurd.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import (
    ACTIVE_BREAK_ACCUMULATION_DAYS,
    ACTIVE_BREAK_CLIMATOLOGY_SMOOTH_DAYS,
    ACTIVE_BREAK_MIN_RUN_DAYS,
    ACTIVE_BREAK_MONTHS,
    ACTIVE_BREAK_SD_THRESHOLD,
)
from src.features.engineering import compute_daily_climatology, day_of_year_no_leap

logger = logging.getLogger(__name__)

ACTIVE = "active"
BREAK = "break"
NORMAL = "normal"
NOT_APPLICABLE = "not_applicable"

PHASE_DESCRIPTIONS = {
    ACTIVE: "Active spell: rainfall is running well above normal for this time of year.",
    BREAK: "Break spell: rainfall has paused well below normal for this time of year.",
    NORMAL: "Neither an active nor a break spell; rainfall is within its usual range.",
    NOT_APPLICABLE: (
        "Outside the June-September monsoon season, where active and break spells are not "
        "defined and a standardised anomaly would be dominated by a near-zero climatological mean."
    ),
}

# The smallest climatological mean (mm/day, averaged over the accumulation
# window) for which a standardised anomaly is meaningful. Below this the
# denominator is so small that ordinary variability produces extreme
# z-scores. Districts that are arid even in JJAS are reported as
# not_applicable rather than handed a spurious phase.
MIN_CLIMATOLOGY_MEAN_MM = 0.5

# The column the anomaly is actually computed on.
ACCUMULATION_COLUMN = "rainfall_trailing_mean"


def standardised_anomaly(
    daily: pd.DataFrame, climatology: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Add `rainfall_anomaly_sd` to a district's daily frame.

    `climatology` is computed from `daily` itself when not supplied. That
    is correct for the descriptive use here — the question is "is today
    unusual relative to this place's normal seasonal cycle", which is a
    statement about the whole record, not a forecast. Anything that feeds
    a MODEL must pass a train-split-only climatology explicitly, the same
    rule RainfallFeaturePreprocessor follows.
    """
    frame = daily.copy()
    frame["date_of_record"] = pd.to_datetime(frame["date_of_record"])
    frame = frame.sort_values("date_of_record").reset_index(drop=True)
    # Smooth BEFORE standardising -- see the module docstring for why a
    # daily z-score cannot represent a break at all.
    frame[ACCUMULATION_COLUMN] = (
        frame["rainfall"]
        .rolling(ACTIVE_BREAK_ACCUMULATION_DAYS, min_periods=ACTIVE_BREAK_ACCUMULATION_DAYS)
        .mean()
    )
    if climatology is None:
        climatology = compute_daily_climatology(
            frame,
            smooth_days=ACTIVE_BREAK_CLIMATOLOGY_SMOOTH_DAYS,
            value_column=ACCUMULATION_COLUMN,
        )

    frame["doy"] = day_of_year_no_leap(frame["date_of_record"]).to_numpy()
    key = pd.MultiIndex.from_arrays([frame["district"], frame["doy"]])
    joined = climatology.reindex(key)

    mean = joined["climatology_daily_mean"].to_numpy()
    sd = joined["climatology_daily_sd"].to_numpy()
    frame["climatology_daily_mean"] = mean
    frame["climatology_daily_sd"] = sd

    # Guard the denominator explicitly instead of letting numpy emit inf.
    usable = (sd > 0) & np.isfinite(sd) & (mean >= MIN_CLIMATOLOGY_MEAN_MM)
    anomaly = np.full(len(frame), np.nan)
    values = frame[ACCUMULATION_COLUMN].to_numpy(dtype=float)
    # A row whose trailing window is incomplete has no value to standardise.
    usable = usable & np.isfinite(values)
    np.divide(values - mean, sd, out=anomaly, where=usable)
    frame["rainfall_anomaly_sd"] = anomaly
    return frame.drop(columns="doy")


def classify_spells(frame: pd.DataFrame) -> pd.DataFrame:
    """Label each day active / break / normal / not_applicable.

    A day belongs to a spell only if it sits inside a run of at least
    ACTIVE_BREAK_MIN_RUN_DAYS consecutive days past the threshold. The run
    requirement is the entire point: single-day excursions are weather,
    and calling them spells would make the signal useless.
    """
    frame = frame.copy()
    in_season = pd.to_datetime(frame["date_of_record"]).dt.month.isin(ACTIVE_BREAK_MONTHS).to_numpy()
    anomaly = frame["rainfall_anomaly_sd"].to_numpy(dtype=float)

    above = in_season & (anomaly >= ACTIVE_BREAK_SD_THRESHOLD)
    below = in_season & (anomaly <= -ACTIVE_BREAK_SD_THRESHOLD)

    phase = np.where(in_season, NORMAL, NOT_APPLICABLE).astype(object)
    # A day with no anomaly at all (missing rainfall, or a degenerate
    # climatology) cannot be classified, in season or not.
    phase[in_season & ~np.isfinite(anomaly)] = NOT_APPLICABLE

    for mask, label in ((above, ACTIVE), (below, BREAK)):
        for start, end in _runs(mask):
            if end - start >= ACTIVE_BREAK_MIN_RUN_DAYS:
                phase[start:end] = label

    frame["monsoon_phase"] = phase
    return frame


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs for each maximal run of True."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


def current_phase(
    daily: pd.DataFrame,
    district: str,
    as_of: pd.Timestamp | None = None,
    climatology: pd.DataFrame | None = None,
) -> dict:
    """The live active/break answer for one district."""
    frame = daily.copy()
    frame["date_of_record"] = pd.to_datetime(frame["date_of_record"])
    frame = frame.sort_values("date_of_record").reset_index(drop=True)
    if as_of is not None:
        frame = frame[frame["date_of_record"] <= pd.Timestamp(as_of)].reset_index(drop=True)

    if frame.empty:
        return {
            "district": district,
            "monsoon_phase": NOT_APPLICABLE,
            "phase_description": "No data available for this district.",
        }

    labelled = classify_spells(standardised_anomaly(frame, climatology))
    last = labelled.iloc[-1]
    phase = str(last["monsoon_phase"])

    # How long the current phase has been running, counted backwards.
    phases = labelled["monsoon_phase"].to_numpy()
    days_in_phase = 1
    for i in range(len(phases) - 2, -1, -1):
        if phases[i] != phase:
            break
        days_in_phase += 1

    recent = labelled.tail(30)
    return {
        "district": district,
        "as_of_date": str(pd.Timestamp(last["date_of_record"]).date()),
        "monsoon_phase": phase,
        "phase_description": PHASE_DESCRIPTIONS[phase],
        "days_in_current_phase": int(days_in_phase),
        "rainfall_anomaly_sd": _round(last["rainfall_anomaly_sd"]),
        "rainfall_mm": _round(last["rainfall"]),
        "trailing_7day_mean_mm_per_day": _round(last[ACCUMULATION_COLUMN]),
        "climatology_7day_mean_mm_per_day": _round(last["climatology_daily_mean"]),
        "climatology_7day_sd_mm_per_day": _round(last["climatology_daily_sd"]),
        "recent_30_days": [
            {
                "date": str(pd.Timestamp(row["date_of_record"]).date()),
                "rainfall_mm": _round(row["rainfall"]),
                "anomaly_sd": _round(row["rainfall_anomaly_sd"]),
                "phase": row["monsoon_phase"],
            }
            for _, row in recent.iterrows()
        ],
        "method": (
            f"Standardised anomaly of the {ACTIVE_BREAK_ACCUMULATION_DAYS}-day trailing mean "
            f"rainfall vs a +/-{ACTIVE_BREAK_CLIMATOLOGY_SMOOTH_DAYS}-day smoothed day-of-year "
            f"climatology; active/break require |anomaly| >= {ACTIVE_BREAK_SD_THRESHOLD} SD "
            f"sustained for >= {ACTIVE_BREAK_MIN_RUN_DAYS} days (after Rajeevan et al. 2010)."
        ),
        "caveats": (
            "Rajeevan et al. define spells over the monsoon core zone as a single region; this "
            "runs per district and averages over 7 days instead, because single-district daily "
            "rainfall is too skewed for a symmetric +/-1 SD rule to register a break at all. "
            "Spells are only defined for June-September."
        ),
    }


def spell_history(
    daily: pd.DataFrame, district: str, climatology: pd.DataFrame | None = None
) -> list[dict]:
    """Every active and break spell in the record, as dated intervals.

    Useful for checking the detector against a season people remember,
    which is the only external validation available for this one - unlike
    onset, there is no published per-district list of spells to score
    against.
    """
    frame = daily.copy()
    frame["date_of_record"] = pd.to_datetime(frame["date_of_record"])
    frame = frame.sort_values("date_of_record").reset_index(drop=True)
    labelled = classify_spells(standardised_anomaly(frame, climatology))

    spells = []
    for label in (ACTIVE, BREAK):
        for start, end in _runs((labelled["monsoon_phase"] == label).to_numpy()):
            window = labelled.iloc[start:end]
            spells.append(
                {
                    "district": district,
                    "phase": label,
                    "start_date": str(pd.Timestamp(window["date_of_record"].iloc[0]).date()),
                    "end_date": str(pd.Timestamp(window["date_of_record"].iloc[-1]).date()),
                    "days": int(end - start),
                    "mean_anomaly_sd": _round(window["rainfall_anomaly_sd"].mean()),
                    "total_rainfall_mm": _round(window["rainfall"].sum()),
                }
            )
    return sorted(spells, key=lambda s: s["start_date"])


def _round(value, digits: int = 2):
    """None for anything non-finite, so JSON never carries NaN."""
    if value is None:
        return None
    number = float(value)
    return None if not np.isfinite(number) else round(number, digits)
