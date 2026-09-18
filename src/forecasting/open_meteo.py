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

    cached = _read_cache(cache_file)
    if cached is not None:
        logger.info("Open-Meteo cache hit for (%.4f, %.4f)", latitude, longitude)
        return pd.DataFrame(cached["daily"])

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(DAILY_VARS),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "auto",
        "wind_speed_unit": "ms",
    }

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "Open-Meteo request (%.4f, %.4f) past_days=%d forecast_days=%d (attempt %d/%d)",
                latitude, longitude, past_days, forecast_days, attempt, MAX_RETRIES,
            )
            resp = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            payload = resp.json()
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            return pd.DataFrame(payload["daily"])
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            logger.warning("Open-Meteo request failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    stale = _read_stale_cache(cache_file)
    if stale is not None:
        logger.warning(
            "Open-Meteo unreachable after %d attempts; serving stale cache from %s",
            MAX_RETRIES, cache_file,
        )
        return pd.DataFrame(stale["daily"])

    raise OpenMeteoError(
        f"Open-Meteo request failed after {MAX_RETRIES} attempts and no cached "
        f"fallback exists for ({latitude}, {longitude}): {last_error}"
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
