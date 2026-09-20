"""NASA POWER daily point client (https://power.larc.nasa.gov/docs/services/api/).

Free, no API key, no registration. Verified against the live API
(2026-09-20) rather than assumed from the docs:
  - one call returns the WHOLE multi-year series for a point: 4,279 days
    (2015-01-01..2026-09-18) x 10 daily parameters in ~3s, ~600KB, with
    essentially no gaps (1 fill value, on the final day).
  - `header.fill_value` is -999.0 and is stated in every response; it is
    read from the payload rather than hardcoded, so a future change to
    the sentinel cannot silently become a real -999mm rainfall reading.
  - `PS` is kPa, NOT hPa. It is multiplied by 10 here so the column
    matches the `air_pressure` scale the model was trained on (~1000).
    Getting this wrong would feed pressure values two orders of magnitude
    off into a trained feature.
  - `WS10M` is m/s at 10m, which already matches the training data's
    `wind_speed` units — no conversion needed (unlike Open-Meteo, whose
    default is km/h).
  - `geometry.coordinates` carries POWER's own grid-cell elevation, which
    differs from our station elevation (86.25m vs 32.0m for
    Thiruvananthapuram). We keep the district registry's value:
    `elevation` is a trained model feature, and it must mean the same
    thing at serve time as it did at train time.

WHY THIS EXISTS ALONGSIDE src/forecasting/open_meteo.py — the two are not
redundant, they do different jobs:
  - Open-Meteo serves the LIVE forecast: finer resolution (~2-11km vs
    POWER's MERRA-2 ~50km) and it has forecast days, which POWER lacks.
  - POWER serves the BULK HISTORY. Open-Meteo's archive is rate-weighted
    at roughly (days/14) x (vars/10) weighted calls, so one 11-year
    10-variable district pull costs ~306 calls against a 10,000/day free
    cap; all 316 districts would be ~96,600 calls, about ten days of
    quota. POWER has no such weighting, so the same backfill is 316
    unweighted calls.

That split means history and live come from different products, which is
a real bias risk. It is measured, not assumed — see
scripts/compare_rainfall_products.py.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import POWER_CACHE_DIR, POWER_START_DATE, district_slug
from src.utils.dates import yesterday_ist
from src.utils.http_cache import fetch_json_cached, read_stale_cache

logger = logging.getLogger(__name__)

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# AG(riculture) community: the parameter set tuned for agroclimatology,
# which is what exposes the soil-wetness fields below.
COMMUNITY = "AG"

# POWER parameter -> our column name.
#
# The first six reproduce exactly the six weather columns the trained
# model already uses, so a POWER-derived frame is drop-in compatible with
# src.features.engineering.build_features. The last four are NEW
# capability the bundled Excel dataset never had, and which the crop
# advisory needs (soil wetness answers "is there water in the root zone
# already?", which rainfall alone does not).
POWER_PARAM_TO_COLUMN = {
    "PRECTOTCORR": "rainfall",
    "T2M": "avg_temp",
    "T2M_MIN": "min_temp",
    "T2M_MAX": "max_temp",
    "WS10M": "wind_speed",
    "PS": "air_pressure",
    "RH2M": "relative_humidity",
    "ALLSKY_SFC_SW_DWN": "solar_radiation",
    "GWETTOP": "soil_wetness_surface",
    "GWETROOT": "soil_wetness_root_zone",
}

POWER_PARAMS = list(POWER_PARAM_TO_COLUMN)

# The 13 columns src.data.district.get_all_districts_daily_series emits.
# Spelled out explicitly so a POWER frame can be asserted against it
# rather than trusting the mapping above stayed in sync.
SCHEMA_COLUMNS = [
    "date_of_record", "district", "state", "rainfall", "avg_temp",
    "min_temp", "max_temp", "wind_speed", "air_pressure",
    "latitude", "longitude", "elevation", "n_stations_reporting",
]

# Columns POWER adds beyond the trained schema.
EXTRA_COLUMNS = [
    "relative_humidity",
    "solar_radiation",
    "soil_wetness_surface",
    "soil_wetness_root_zone",
]

# kPa -> hPa. See the module docstring.
PRESSURE_KPA_TO_HPA = 10.0

# Reanalysis history is immutable once published, so a shard stays valid
# indefinitely; it is refetched only when the caller asks for dates the
# shard does not yet cover (see _cached_covers), never because a clock
# ticked over.
CACHE_TTL_SECONDS = 30 * 24 * 3600

REQUEST_TIMEOUT_SECONDS = 60  # one call returns 11 years; ~3s typical, but be generous
MAX_RETRIES = 4
# Longer than the Open-Meteo client's 1.5s. POWER lists HTTP 429 among its
# status codes but never states the actual rate limit, so the safe
# assumption is that a limit exists and is unknown: back off hard rather
# than hammer a free service whose ceiling we cannot predict.
RETRY_BACKOFF_SECONDS = 5.0


class PowerError(RuntimeError):
    pass


def _shard_path(district: str) -> Path:
    return POWER_CACHE_DIR / f"power_{district_slug(district)}.json"


def _payload_end_date(payload: dict) -> dt.date | None:
    """The last date a cached payload actually covers, per its own header."""
    end = payload.get("header", {}).get("end")
    if not end:
        return None
    try:
        return dt.datetime.strptime(str(end), "%Y%m%d").date()
    except ValueError:
        return None


def _cached_covers(shard: Path, end: dt.date) -> dict | None:
    """A cached shard, but only if it already reaches `end`.

    This is what makes the 316-district backfill resumable and cheap to
    re-run: an interrupted run picks up exactly where it stopped, and a
    run a week later refetches only because it genuinely wants newer
    days — not because a TTL happened to lapse.
    """
    payload = read_stale_cache(shard)
    if payload is None:
        return None
    covered = _payload_end_date(payload)
    if covered is None or covered < end:
        return None
    return payload


def fetch_power_daily(
    district: str,
    latitude: float,
    longitude: float,
    start: str = POWER_START_DATE,
    end: dt.date | None = None,
) -> dict:
    """Raw POWER payload for one district centroid, `start`..`end`
    inclusive (default end: yesterday IST, since today's daily aggregate
    is still partial).

    Cached as one JSON shard per district. Falls back to a stale shard if
    POWER is unreachable, and raises PowerError only when the call fails
    with nothing cached at all.
    """
    end = end or yesterday_ist()
    shard = _shard_path(district)

    cached = _cached_covers(shard, end)
    if cached is not None:
        logger.info("POWER shard for %s already covers %s", district, end)
        return cached

    params = {
        "parameters": ",".join(POWER_PARAMS),
        "community": COMMUNITY,
        "latitude": latitude,
        "longitude": longitude,
        "start": pd.Timestamp(start).strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    return fetch_json_cached(
        url=BASE_URL,
        params=params,
        cache_file=shard,
        label=f"NASA POWER {district} ({latitude:.4f}, {longitude:.4f}) {start}..{end}",
        getter=lambda u, params, timeout: requests.get(u, params=params, timeout=timeout),
        sleep=lambda seconds: time.sleep(seconds),
        error_cls=PowerError,
        # 0 forces a live call: _cached_covers has already decided the
        # shard on disk is insufficient, so an age-based TTL check here
        # could only wrongly re-approve it.
        ttl_seconds=0,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        backoff_seconds=RETRY_BACKOFF_SECONDS,
    )


def to_daily_frame(
    payload: dict,
    district: str,
    state: str,
    latitude: float,
    longitude: float,
    elevation: float,
) -> pd.DataFrame:
    """Normalise a POWER payload into the same 13-column schema
    src.data.district.get_all_districts_daily_series produces, plus the
    four EXTRA_COLUMNS.

    Fill values become NaN, pressure is converted kPa -> hPa, and
    lat/lon/elevation come from the DISTRICT REGISTRY rather than POWER's
    grid geometry (see the module docstring).
    """
    params = payload.get("properties", {}).get("parameter")
    if not params:
        raise PowerError(f"POWER payload for {district} has no properties.parameter block")

    absent = [p for p in POWER_PARAMS if p not in params]
    if absent:
        raise PowerError(f"POWER payload for {district} is missing parameters: {absent}")

    # Read the sentinel from the response instead of trusting a constant.
    fill_value = payload.get("header", {}).get("fill_value", -999.0)

    frame = pd.DataFrame({POWER_PARAM_TO_COLUMN[p]: pd.Series(params[p]) for p in POWER_PARAMS})
    frame.index.name = "yyyymmdd"
    frame = frame.reset_index()
    frame["date_of_record"] = pd.to_datetime(frame["yyyymmdd"], format="%Y%m%d")
    frame = frame.drop(columns="yyyymmdd")

    value_columns = list(POWER_PARAM_TO_COLUMN.values())
    frame[value_columns] = frame[value_columns].replace(fill_value, np.nan)
    # Convert only after the sentinel is gone — scaling -999 by 10 first
    # would turn an unmatchable -9990 into a plausible-looking number.
    frame["air_pressure"] = frame["air_pressure"] * PRESSURE_KPA_TO_HPA

    frame["district"] = district
    frame["state"] = state
    frame["latitude"] = latitude
    frame["longitude"] = longitude
    frame["elevation"] = elevation
    # POWER is a gridded reanalysis, not a station network. There is no
    # station count to report; this column exists only so the frame stays
    # schema-compatible with the station-derived one, and 1 means "one
    # grid cell contributed", not "one station reported".
    frame["n_stations_reporting"] = 1

    ordered = SCHEMA_COLUMNS + EXTRA_COLUMNS
    return frame.sort_values("date_of_record").reset_index(drop=True)[ordered]


def trim_trailing_gap(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop trailing rows that have no measurements at all.

    POWER runs ~2-3 days behind real time (measured 2026-09-20: a request
    through 2026-09-19 returned rows for the 18th and 19th with every
    parameter set to the fill value). Those rows are real dates with no
    data, so leaving them in would make `consecutive_dry_days` and the
    rolling sums treat "not published yet" as "no rain" — the exact class
    of mistake the Excel dataset's missingness step-change already caused
    once in this project.

    Interior gaps are deliberately left alone: they are genuine holes and
    the feature code already represents them as NaN.
    """
    if frame.empty:
        return frame
    measured = [c for c in POWER_PARAM_TO_COLUMN.values() if c in frame.columns]
    has_data = frame[measured].notna().any(axis=1)
    if not has_data.any():
        return frame.iloc[0:0]
    return frame.loc[: has_data[has_data].index[-1]].copy()


def data_end(frame: pd.DataFrame) -> pd.Timestamp | None:
    """The last date in `frame` that actually carries a measurement."""
    trimmed = trim_trailing_gap(frame)
    if trimmed.empty:
        return None
    return pd.Timestamp(trimmed["date_of_record"].iloc[-1])


def get_district_daily_series(
    district: str,
    state: str,
    latitude: float,
    longitude: float,
    elevation: float,
    start: str = POWER_START_DATE,
    end: dt.date | None = None,
    trim: bool = True,
) -> pd.DataFrame:
    """fetch + normalise, for one district.

    `trim` drops POWER's not-yet-published trailing days by default; pass
    False only if you specifically want to see the unpublished tail.
    """
    payload = fetch_power_daily(district, latitude, longitude, start=start, end=end)
    frame = to_daily_frame(payload, district, state, latitude, longitude, elevation)
    return trim_trailing_gap(frame) if trim else frame
