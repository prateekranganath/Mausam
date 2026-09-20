"""Global climate indices: ENSO (ONI), IOD (DMI) and MJO (RMM).

These are the three large-scale drivers the problem statement asks for.
All three come from free, key-less public feeds, verified live on
2026-09-20 rather than taken from documentation:

  ONI  NOAA CPC, https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
       Fixed-width `SEAS YR TOTAL ANOM`, 3-month overlapping seasons,
       DJF 1950 onward. ANOM is the index.
  DMI  NOAA PSL, https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data
       Year + 12 monthly values per line, 1870 onward, -9999 for absent
       months, then trailing prose lines that must not be parsed as data.
  RMM  IRI Data Library's mirror of the Australian BoM RMM index.
       Daily amplitude, phase (1-8), RMM1, RMM2.

WHY THE IRI MIRROR FOR MJO, AND NOT BoM DIRECTLY: BoM's own
rmm.74toRealtime.txt now returns an anti-scraping block page ("The Bureau
of Meteorology website does not support web scraping") instead of data.

WHY NOT CPC'S PENTAD MJO INDEX INSTEAD: it has explicit dates and would
be simpler to parse, but measured on 2026-09-20 its last published pentad
was 2026-08-26 — a ~25-day lag, which is roughly half an MJO cycle and
therefore useless as a real-time feature. The IRI/BoM RMM series was 3
days behind on the same day.

THE ONE REAL TRAP HERE IS ROW-TO-DATE ALIGNMENT. IRI's table output does
not include the date column, and its two selection syntaxes disagree
about order: `T/last/N/RANGE` returns newest-first while an explicit
`T/(start)/(end)/RANGEEDGES` returns oldest-first. Zipping rows against a
locally-generated date range would therefore have silently reversed the
whole MJO series, shifting a model feature by weeks with no error raised.
So dates are never inferred alone: _fetch_rmm_range anchors on the range
start and then VERIFIES the final row against an independent single-day
query, raising if they disagree. See _verify_rmm_alignment.

PUBLICATION LAG IS TREATED AS A LEAKAGE CONCERN, NOT A FOOTNOTE. An index
value labelled "July 2026" was not knowable in July: ONI for a given
month is a 3-month running mean centred on it, so it cannot exist until
the following month's SST is in, and CPC publishes it a further few weeks
later. Joining on the label date would hand the model information that
did not exist at prediction time — the same class of mistake the trailing
windows in src/features/engineering.py shift(1) to avoid. Each index
therefore declares a PUBLICATION_LAG and the as-of join uses
`available_from = date + lag`, so training and serving see an index
become available at the same point in its life.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from src.config import CLIMATE_INDEX_CACHE_DIR, CLIMATE_INDEX_CACHE_TTL_SECONDS
from src.utils.http_cache import fetch_text_cached

logger = logging.getLogger(__name__)

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
DMI_URL = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
IRI_RMM_BASE = "https://iridl.ldeo.columbia.edu/SOURCES/.BoM/.MJO/.RMM"

REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

# Publication lags, in days. See the module docstring for why these exist.
#
# Each of these is rounded UP from a measured lag, never down. Overstating
# a lag only costs a little information; understating one leaks, and a leak
# here would be invisible in every metric because it inflates train, val
# and test alike.
#
# ONI: a 3-month running mean centred on month M needs M+1's SST, and CPC
# publishes it some weeks later still. Measured 2026-09-20: the newest
# season was JJA 2026, i.e. centred 2026-07-01 — an 81-day lag.
ONI_PUBLICATION_LAG_DAYS = 90
# DMI: nominally monthly, materially slower in practice. Measured
# 2026-09-20: newest real month was 2026-05-01 (a 112-day lag), and the
# file's own "Created" stamp read 2026-07-25.
DMI_PUBLICATION_LAG_DAYS = 120
# RMM: genuinely near-real-time, which is what makes MJO the only one of
# the three that could even be a candidate at a 7-day horizon (it was
# tested and still did not earn a place -- see MJO_FEATURE_COLUMNS).
# Measured 2026-09-20: newest day 2026-09-17.
MJO_PUBLICATION_LAG_DAYS = 3

# DMI's absent-month sentinel.
DMI_MISSING = -9999.0

# ONI's 3-month seasons, mapped to the calendar month each is centred on.
# DJF 1950 spans Dec 1949 - Feb 1950 and is centred on Jan 1950, so the
# file's YR column is already the centre month's year.
_ONI_SEASON_TO_CENTRE_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}

# ENSO / IOD phase thresholds. The +/-0.5C ONI threshold is NOAA's own
# operational definition of an El Nino / La Nina episode. The +/-0.4
# DMI cut is the value commonly used for a positive/negative IOD event.
ONI_EVENT_THRESHOLD = 0.5
DMI_EVENT_THRESHOLD = 0.4
# Below this, the MJO is conventionally described as weak/incoherent and
# its phase carries little meaning (Wheeler & Hendon 2004).
MJO_WEAK_AMPLITUDE = 1.0

# Candidate feature columns for the ablation.
#
# NONE OF THESE IS CURRENTLY A SHIPPED MODEL FEATURE. That was decided by
# scripts/run_climate_ablation.py, not assumed, and the measured outcome
# was not the one expected:
#
#   arm              val ROC   test PR
#   baseline (34)     0.8142    0.3792
#   + MJO             0.8070    0.3842
#   + MJO+ONI+DMI     0.7724    0.3965
#
# ONI and DMI fail structurally rather than marginally. Over the pooled
# model's 2021-01..2023-06 training window, only 7.9% of VALIDATION ONI
# values and 15.8% of DMI values fall inside the range the model ever saw
# in training (train ONI spans [-1.11, 0.19], validation spans
# [0.19, 1.99] -- the 2020-2023 La Nina against the El Nino that followed).
# A tree can only split on values it has seen, so there is nothing there to
# generalise from; the model extrapolates off the end of its own feature,
# and validation ROC drops 0.042. More training years would fix this; no
# amount of tuning will.
#
# MJO does not have that problem -- 100% of validation amplitudes fall
# inside the training range, as its 30-60 day cycle implies -- but it still
# did not improve validation ROC (-0.0073). Both arms score better on TEST,
# which is not a reason to ship either: selecting on the held-out split is
# exactly what would stop it being held out.
#
# So all three are served as CONTEXT (see current_snapshot and
# GET /climate/context) and none is a model input.
#
# `mjo_phase` is excluded from the feature list even as a candidate. It is
# a cyclic 1-8 octant label: phase 8 and phase 1 are adjacent, but as an
# integer they look maximally far apart, and a tree would learn spurious
# splits from the ordering. The sin/cos encoding preserves the adjacency.
MJO_FEATURE_COLUMNS = ["mjo_amplitude", "mjo_phase_sin", "mjo_phase_cos"]
SEASONAL_FEATURE_COLUMNS = ["oni", "dmi"]

CLIMATE_FEATURE_COLUMNS = MJO_FEATURE_COLUMNS + SEASONAL_FEATURE_COLUMNS


class ClimateIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndexSnapshot:
    """One index's current value, for the API context layer."""

    name: str
    value: float
    as_of: str
    phase: str
    publication_lag_days: int
    source: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(float(self.value), 3),
            "as_of": self.as_of,
            "phase": self.phase,
            "publication_lag_days": self.publication_lag_days,
            "source": self.source,
        }


def _get_text(url: str, cache_name: str, label: str, params: dict | None = None) -> str:
    return fetch_text_cached(
        url=url,
        params=params,
        cache_file=CLIMATE_INDEX_CACHE_DIR / cache_name,
        label=label,
        getter=lambda u, params, timeout: requests.get(u, params=params, timeout=timeout),
        sleep=lambda seconds: time.sleep(seconds),
        error_cls=ClimateIndexError,
        ttl_seconds=CLIMATE_INDEX_CACHE_TTL_SECONDS,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        backoff_seconds=RETRY_BACKOFF_SECONDS,
    )


# --------------------------------------------------------------------------
# ONI (ENSO)
# --------------------------------------------------------------------------

def parse_oni(text: str) -> pd.DataFrame:
    """Parse CPC's oni.ascii.txt into columns [date, oni].

    `date` is the first of the calendar month the 3-month season is
    centred on, so the series is one value per month with no overlap.
    """
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[0] not in _ONI_SEASON_TO_CENTRE_MONTH:
            continue  # header, blanks, anything unexpected
        season, year, _total, anomaly = parts
        try:
            rows.append((int(year), _ONI_SEASON_TO_CENTRE_MONTH[season], float(anomaly)))
        except ValueError:
            continue

    if not rows:
        raise ClimateIndexError("No ONI rows parsed; the CPC file format may have changed.")

    frame = pd.DataFrame(rows, columns=["year", "month", "oni"])
    frame["date"] = pd.to_datetime(dict(year=frame["year"], month=frame["month"], day=1))
    return frame[["date", "oni"]].sort_values("date").reset_index(drop=True)


def load_oni() -> pd.DataFrame:
    return parse_oni(_get_text(ONI_URL, "oni.txt", "NOAA CPC ONI"))


def oni_phase(value: float) -> str:
    """NOAA's operational +/-0.5C episode definition."""
    if value >= ONI_EVENT_THRESHOLD:
        return "el_nino"
    if value <= -ONI_EVENT_THRESHOLD:
        return "la_nina"
    return "neutral"


# --------------------------------------------------------------------------
# DMI (IOD)
# --------------------------------------------------------------------------

def parse_dmi(text: str) -> pd.DataFrame:
    """Parse NOAA PSL's dmi.had.long.data into columns [date, dmi].

    Layout: a ` <first_year> <last_year>` header, then one line per year
    holding the year plus 12 monthly values, then a bare sentinel line and
    several lines of prose. Only 13-field all-numeric lines are data;
    everything else is skipped rather than guessed at, so the trailing
    "Created ..." and "Preliminary." lines cannot become rows.
    """
    records = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 13:
            continue
        if not re.fullmatch(r"\d{4}", parts[0]):
            continue
        try:
            values = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        year = int(parts[0])
        for month, value in enumerate(values, start=1):
            records.append((year, month, value))

    if not records:
        raise ClimateIndexError("No DMI rows parsed; the NOAA PSL file format may have changed.")

    frame = pd.DataFrame(records, columns=["year", "month", "dmi"])
    frame["date"] = pd.to_datetime(dict(year=frame["year"], month=frame["month"], day=1))
    frame["dmi"] = frame["dmi"].replace(DMI_MISSING, np.nan)
    # Trailing months of the current year are present-but-absent; drop them
    # so `as_of` reports the last month that actually has a value.
    frame = frame.dropna(subset=["dmi"])
    return frame[["date", "dmi"]].sort_values("date").reset_index(drop=True)


def load_dmi() -> pd.DataFrame:
    return parse_dmi(_get_text(DMI_URL, "dmi.txt", "NOAA PSL IOD DMI"))


def dmi_phase(value: float) -> str:
    if value >= DMI_EVENT_THRESHOLD:
        return "positive_iod"
    if value <= -DMI_EVENT_THRESHOLD:
        return "negative_iod"
    return "neutral"


# --------------------------------------------------------------------------
# MJO (RMM)
# --------------------------------------------------------------------------

_RMM_TABLE_SUFFIX = "T+exch+table-+text+text+text+-table+.tsv"


def _rmm_range_url(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """IRI wants dates as `(1 Sep 2026)`, URL-encoded.

    RANGEEDGES (rather than `T/last/N/RANGE`) is used deliberately: it
    returns rows oldest-first, whereas `last/N/RANGE` returns them
    newest-first. Pinning the syntax pins the order.
    """
    fmt = "%d %b %Y"
    lo = requests.utils.quote(f"({start.strftime(fmt)})")
    hi = requests.utils.quote(f"({end.strftime(fmt)})")
    return f"{IRI_RMM_BASE}/T/{lo}/{hi}/RANGEEDGES/{_RMM_TABLE_SUFFIX}"


def _rmm_single_day_url(day: pd.Timestamp) -> str:
    value = requests.utils.quote(f"({day.strftime('%d %b %Y')})")
    return f"{IRI_RMM_BASE}/T/{value}/VALUES/{_RMM_TABLE_SUFFIX}"


def _parse_rmm_table(text: str) -> list[list[float]]:
    """Rows of [amplitude, phase, RMM1, RMM2] from IRI's tsv.

    The payload is a header line, a blank tab-only line, then the data.
    Unavailable days come back as a single `1000` field rather than four
    values, so any row that is not exactly four numbers is dropped.
    """
    rows: list[list[float]] = []
    for line in text.splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) != 4:
            continue
        try:
            rows.append([float(f) for f in fields])
        except ValueError:
            continue
    return rows


def _verify_rmm_alignment(frame: pd.DataFrame) -> None:
    """Confirm the inferred date of the LAST row by querying that day on
    its own.

    IRI does not return dates alongside values, so `frame`'s dates are
    inferred from the range start plus contiguity. Contiguity is the
    assumption worth checking: a single missing interior day would shift
    every subsequent date by one, silently, and an MJO feature off by
    even a few days is worse than no MJO feature. One cheap extra call
    (cached alongside the rest) turns the assumption into a check.
    """
    if frame.empty:
        return
    last_date = pd.Timestamp(frame["date"].iloc[-1])
    text = _get_text(
        _rmm_single_day_url(last_date),
        f"rmm_check_{last_date.date()}.tsv",
        f"IRI BoM RMM alignment check {last_date.date()}",
    )
    rows = _parse_rmm_table(text)
    if not rows:
        raise ClimateIndexError(
            f"MJO alignment check failed: IRI has no data for {last_date.date()}, "
            "which the range query implied was the last available day. The series "
            "may have an interior gap, so dates cannot be trusted."
        )
    expected = rows[-1]
    actual = [
        float(frame["mjo_amplitude"].iloc[-1]),
        float(frame["mjo_phase"].iloc[-1]),
        float(frame["mjo_rmm1"].iloc[-1]),
        float(frame["mjo_rmm2"].iloc[-1]),
    ]
    if not np.allclose(expected, actual, rtol=1e-6, atol=1e-6):
        raise ClimateIndexError(
            f"MJO date alignment check FAILED for {last_date.date()}: the range query's "
            f"last row is {actual} but a direct single-day query returns {expected}. "
            "Row order or contiguity has changed upstream; refusing to attach "
            "misaligned MJO values to a model feature."
        )
    logger.info("MJO date alignment verified at %s", last_date.date())


def load_mjo(start: str = "2014-12-01", end: str | None = None, verify: bool = True) -> pd.DataFrame:
    """Daily MJO RMM index as [date, mjo_rmm1, mjo_rmm2, mjo_amplitude,
    mjo_phase, mjo_phase_sin, mjo_phase_cos].

    `start` defaults to just before POWER_START_DATE so the as-of join has
    a value available for the very first day of the weather record.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now().normalize()

    text = _get_text(
        _rmm_range_url(start_ts, end_ts),
        f"rmm_{start_ts.date()}_{end_ts.date()}.tsv",
        f"IRI BoM RMM {start_ts.date()}..{end_ts.date()}",
    )
    rows = _parse_rmm_table(text)
    if not rows:
        raise ClimateIndexError(f"No MJO rows parsed for {start_ts.date()}..{end_ts.date()}.")

    # Oldest-first, contiguous daily, anchored at the requested start.
    # Fewer rows than requested days simply means the tail is unpublished.
    dates = pd.date_range(start_ts, periods=len(rows), freq="D")
    frame = pd.DataFrame(rows, columns=["mjo_amplitude", "mjo_phase", "mjo_rmm1", "mjo_rmm2"])
    frame.insert(0, "date", dates)

    frame = add_mjo_phase_encoding(frame)
    if verify:
        _verify_rmm_alignment(frame)
    return frame


def add_mjo_phase_encoding(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode the cyclic 1-8 MJO phase as sin/cos.

    Phases 8 and 1 are neighbours in the physical cycle but are as far
    apart as possible as integers, which is exactly the artefact a tree
    model would exploit. sin/cos keeps the wrap-around intact.
    """
    frame = frame.copy()
    angle = 2 * np.pi * (frame["mjo_phase"] - 1) / 8.0
    frame["mjo_phase_sin"] = np.sin(angle)
    frame["mjo_phase_cos"] = np.cos(angle)
    return frame


def mjo_phase_label(phase: float, amplitude: float) -> str:
    """A plain-language phase label for the API context layer.

    Phases 2-4 place MJO convection over the Indian Ocean, which is the
    configuration associated with enhanced Indian monsoon rainfall;
    phases 6-8 place it over the west Pacific, associated with suppressed
    rainfall. Below MJO_WEAK_AMPLITUDE the phase is not meaningful.
    """
    if not np.isfinite(amplitude) or amplitude < MJO_WEAK_AMPLITUDE:
        return "weak (phase not meaningful)"
    p = int(round(phase))
    if p in (2, 3, 4):
        return f"phase {p} (Indian Ocean; favours enhanced rainfall)"
    if p in (6, 7, 8):
        return f"phase {p} (west Pacific; favours suppressed rainfall)"
    return f"phase {p} (transition)"


# --------------------------------------------------------------------------
# Joining onto a weather frame
# --------------------------------------------------------------------------

def _with_availability(frame: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """Shift an index's join key forward by its publication lag, so day T
    can only see values that had actually been published by day T."""
    out = frame.copy()
    out["available_from"] = out["date"] + pd.Timedelta(days=lag_days)
    return out.sort_values("available_from").reset_index(drop=True)


def attach_climate_indices(
    df: pd.DataFrame,
    oni: pd.DataFrame | None = None,
    dmi: pd.DataFrame | None = None,
    mjo: pd.DataFrame | None = None,
    date_column: str = "date_of_record",
) -> pd.DataFrame:
    """Left-join ONI, DMI and MJO onto a daily weather frame.

    Strictly backward-looking: `pd.merge_asof(direction="backward")` on
    each index's `available_from`, so row T carries the most recent value
    that was PUBLISHED on or before T, and never a future one. Combined
    with the publication lags, this is what keeps the indices honest as
    model features — the same discipline the rest of the feature code
    applies to rolling windows.

    Rows earlier than any published index value get NaN, which the
    preprocessor's median imputer already handles.
    """
    oni = load_oni() if oni is None else oni
    dmi = load_dmi() if dmi is None else dmi
    mjo = load_mjo() if mjo is None else mjo

    out = df.copy()
    out["_join_key"] = pd.to_datetime(out[date_column])
    # merge_asof requires both sides sorted on the key; remember the
    # caller's row order and restore it afterwards, because a multi-district
    # frame is grouped by district and must not be silently re-sorted.
    out["_original_order"] = np.arange(len(out))
    out = out.sort_values("_join_key").reset_index(drop=True)

    for source, columns, lag in (
        (oni, ["oni"], ONI_PUBLICATION_LAG_DAYS),
        (dmi, ["dmi"], DMI_PUBLICATION_LAG_DAYS),
        (mjo, ["mjo_amplitude", "mjo_phase", "mjo_phase_sin", "mjo_phase_cos"], MJO_PUBLICATION_LAG_DAYS),
    ):
        available = _with_availability(source, lag)
        present = [c for c in columns if c in available.columns]
        out = pd.merge_asof(
            out,
            available[["available_from"] + present],
            left_on="_join_key",
            right_on="available_from",
            direction="backward",
        ).drop(columns="available_from")

    out = out.sort_values("_original_order").reset_index(drop=True)
    return out.drop(columns=["_join_key", "_original_order"])


def current_snapshot() -> dict[str, dict]:
    """The three indices' latest published values, for GET /climate/context.

    Each carries its own `as_of` date and publication lag, because these
    are NOT current-day readings: measured 2026-09-20, ONI was ~2 months
    behind and DMI ~4. Presenting them without that would imply a
    freshness they do not have.
    """
    snapshots: dict[str, dict] = {}

    oni = load_oni()
    last = oni.iloc[-1]
    snapshots["enso"] = IndexSnapshot(
        name="ONI (Oceanic Nino Index)",
        value=float(last["oni"]),
        as_of=str(pd.Timestamp(last["date"]).date()),
        phase=oni_phase(float(last["oni"])),
        publication_lag_days=ONI_PUBLICATION_LAG_DAYS,
        source=ONI_URL,
    ).to_dict()

    dmi = load_dmi()
    last = dmi.iloc[-1]
    snapshots["iod"] = IndexSnapshot(
        name="DMI (IOD Dipole Mode Index)",
        value=float(last["dmi"]),
        as_of=str(pd.Timestamp(last["date"]).date()),
        phase=dmi_phase(float(last["dmi"])),
        publication_lag_days=DMI_PUBLICATION_LAG_DAYS,
        source=DMI_URL,
    ).to_dict()

    mjo = load_mjo()
    last = mjo.iloc[-1]
    snapshot = IndexSnapshot(
        name="MJO RMM amplitude",
        value=float(last["mjo_amplitude"]),
        as_of=str(pd.Timestamp(last["date"]).date()),
        phase=mjo_phase_label(float(last["mjo_phase"]), float(last["mjo_amplitude"])),
        publication_lag_days=MJO_PUBLICATION_LAG_DAYS,
        source="IRI Data Library mirror of Australian BoM RMM (Wheeler & Hendon 2004)",
    ).to_dict()
    snapshot["mjo_phase"] = int(round(float(last["mjo_phase"])))
    snapshots["mjo"] = snapshot

    return snapshots
