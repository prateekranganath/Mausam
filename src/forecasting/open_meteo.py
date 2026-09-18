"""Open-Meteo API client (https://open-meteo.com/en/docs) — free, no API
key required for non-commercial use.

Verified against the live API (2026-09-18) rather than assumed from docs
alone:
  - base URL: https://api.open-meteo.com/v1/forecast
  - past_days (0-92) + forecast_days (0-16) together cover both the
    trailing lookback our features need and the forward window
  - wind_speed_10m_mean with wind_speed_unit=ms matches our training
    data's wind_speed units/magnitude (Open-Meteo's default is km/h;
    only wind_speed_10m_max exists in km/h form, no _mean, so requesting
    ms explicitly avoids a silent unit mismatch)
  - pressure_msl_mean (hPa) matches our training data's air_pressure
    range directly (that field is already sea-level-reduced, not raw
    station pressure — consistent with typical hilly-station values in
    the source dataset)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import OPEN_METEO_CACHE_DIR

logger = logging.getLogger(__name__)

BASE_URL = "https://api.open-meteo.com/v1/forecast"
# Historical/reanalysis endpoint: same variables and units, arbitrary
# start_date/end_date. Used for /historical because the forecast endpoint's
# past_days only holds ~70 usable days (its oldest rows come back null).
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_mean",
    "pressure_msl_mean",
    "relative_humidity_2m_mean",
]

# precipitation_probability_max only exists for forecast days
HISTORICAL_DAILY_VARS = [v for v in DAILY_VARS if v != "precipitation_probability_max"]

# Requested windows are rounded UP to one of these so the on-disk cache holds
# at most len(buckets) files per district, however many distinct `days`
# values clients try. The caller still gets exactly the days it asked for.
HISTORICAL_WINDOW_BUCKETS = (30, 90, 180, 365, 730)
MAX_HISTORICAL_DAYS = HISTORICAL_WINDOW_BUCKETS[-1]

# All Indian districts are in one timezone, so "yesterday" is well defined
# without a tz database (zoneinfo needs the tzdata package on Windows).
_IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _today_ist() -> dt.date:
    return dt.datetime.now(_IST).date()


# Maps Open-Meteo's daily column names onto our trained feature schema's
# raw daily columns (src.data.district.get_district_daily_series output).
# precipitation_probability_max and relative_humidity_2m_mean are kept
# under their own names — the local model was never trained on them, so
# they are NOT silently merged into the feature-schema columns.
_OPEN_METEO_TO_SCHEMA = {
    "temperature_2m_mean": "avg_temp",
    "temperature_2m_min": "min_temp",
    "temperature_2m_max": "max_temp",
    "precipitation_sum": "rainfall",
    "wind_speed_10m_mean": "wind_speed",
    "pressure_msl_mean": "air_pressure",
}

CACHE_TTL_SECONDS = 3600  # weather changes slowly enough hourly is plenty for an MVP
REQUEST_TIMEOUT_SECONDS = 10
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5


class OpenMeteoError(RuntimeError):
    pass


def _cache_path(latitude: float, longitude: float, past_days: int, forecast_days: int) -> Path:
    key = f"{latitude:.4f}_{longitude:.4f}_{past_days}_{forecast_days}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return OPEN_METEO_CACHE_DIR / f"forecast_{digest}.json"


def _read_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > CACHE_TTL_SECONDS:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_stale_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fetch_daily_weather(
    latitude: float,
    longitude: float,
    forecast_days: int = 7,
    past_days: int = 30,
) -> pd.DataFrame:
    """Fetch daily weather for one point: `past_days` days of history
    (Open-Meteo's own recent-observation/reanalysis blend) through
    `forecast_days` days ahead. Cached for CACHE_TTL_SECONDS; falls back
    to a stale cache (with a logged warning) if the live call fails and
    no fresh cache exists, rather than raising immediately.

    Returns a DataFrame with columns:
      date, temperature_2m_max/min/mean, precipitation_sum,
      precipitation_probability_max, wind_speed_10m_mean,
      pressure_msl_mean, relative_humidity_2m_mean
    """
    cache_file = _cache_path(latitude, longitude, past_days, forecast_days)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(DAILY_VARS),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "auto",
        "wind_speed_unit": "ms",
    }
    label = f"({latitude:.4f}, {longitude:.4f}) past_days={past_days} forecast_days={forecast_days}"
    return pd.DataFrame(_get_payload(BASE_URL, params, cache_file, label)["daily"])


def fetch_historical_weather(latitude: float, longitude: float, days: int) -> pd.DataFrame:
    """The last `days` COMPLETE days (ending yesterday, IST) for one point,
    from Open-Meteo's archive API. Today is excluded because its daily value
    is still partly a forecast. The most recent days are preliminary
    reanalysis and may be revised.

    Same cache/retry/stale-fallback behaviour as fetch_daily_weather. Returns
    exactly `days` rows (oldest first) with columns time and the
    HISTORICAL_DAILY_VARS.
    """
    if not 1 <= days <= MAX_HISTORICAL_DAYS:
        raise ValueError(f"days must be between 1 and {MAX_HISTORICAL_DAYS}, got {days}")

    bucket = next(b for b in HISTORICAL_WINDOW_BUCKETS if b >= days)
    end = _today_ist() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=bucket - 1)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(HISTORICAL_DAILY_VARS),
        "timezone": "auto",
        "wind_speed_unit": "ms",
    }
    key = hashlib.sha1(f"{latitude:.4f}_{longitude:.4f}_{bucket}".encode()).hexdigest()[:16]
    label = f"({latitude:.4f}, {longitude:.4f}) history {start}..{end}"
    payload = _get_payload(ARCHIVE_URL, params, OPEN_METEO_CACHE_DIR / f"history_{key}.json", label)
    return pd.DataFrame(payload["daily"]).tail(days).reset_index(drop=True)


def _get_payload(url: str, params: dict, cache_file: Path, label: str) -> dict:
    """GET with a TTL disk cache, retries with backoff, and a stale-cache
    fallback (with a logged warning) if the live call keeps failing. Raises
    OpenMeteoError only when the call fails AND nothing is cached."""
    cached = _read_cache(cache_file)
    if cached is not None:
        logger.info("Open-Meteo cache hit for %s", label)
        return cached

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Open-Meteo request %s (attempt %d/%d)", label, attempt, MAX_RETRIES)
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            payload = resp.json()
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            logger.warning("Open-Meteo request failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    stale = _read_stale_cache(cache_file)
    if stale is not None:
        logger.warning(
            "Open-Meteo unreachable after %d attempts; serving stale cache from %s", MAX_RETRIES, cache_file
        )
        return stale

    raise OpenMeteoError(
        f"Open-Meteo request failed after {MAX_RETRIES} attempts and no cached "
        f"fallback exists for {label}: {last_error}"
    )


def to_district_daily_schema(
    df: pd.DataFrame, district: str, state: str, latitude: float, longitude: float, elevation: float
) -> pd.DataFrame:
    """Rename/select Open-Meteo's daily columns onto the same schema
    src.data.district.get_district_daily_series produces, so the output
    can be passed straight into src.features.engineering.build_features
    exactly as historical data is.
    """
    out = df.rename(columns=_OPEN_METEO_TO_SCHEMA).copy()
    out["date_of_record"] = pd.to_datetime(df["time"])
    out["district"] = district
    out["state"] = state
    out["latitude"] = latitude
    out["longitude"] = longitude
    out["elevation"] = elevation
    out["n_stations_reporting"] = 1  # not station-based; kept for schema compatibility only

    cols = [
        "date_of_record", "district", "state", "rainfall", "avg_temp",
        "min_temp", "max_temp", "wind_speed", "air_pressure",
        "latitude", "longitude", "elevation", "n_stations_reporting",
    ]
    return out[cols]


def get_native_forecast_summary(df: pd.DataFrame, forecast_days: int) -> dict:
    """Open-Meteo's OWN forward-looking forecast, kept separate from our
    model's output — a different quantity (raw NWP-derived precipitation
    forecast + per-day rain probability), not directly comparable to our
    model's "probability the 7-day sum falls below the monthly climatology
    tercile," so it is reported alongside, never silently blended in.
    """
    future = df[pd.to_datetime(df["time"]) > pd.Timestamp.now().normalize()].head(forecast_days)
    return {
        "source": "open-meteo.com (live forecast, not a historical observation)",
        "forecast_days": len(future),
        "total_precipitation_sum_mm": round(float(future["precipitation_sum"].sum()), 2),
        "mean_daily_precipitation_probability_percent": round(
            float(future["precipitation_probability_max"].mean()), 1
        ) if len(future) else None,
        "daily": future[["time", "precipitation_sum", "precipitation_probability_max"]].to_dict("records"),
    }
