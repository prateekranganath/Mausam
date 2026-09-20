"""The long daily history the monsoon detectors run on.

Onset and active/break both need a multi-year, gap-free daily rainfall
record that also reaches TODAY. Neither existing source does both:

  the bundled Excel   ends 2025-02-10, so it cannot see the current season
  Open-Meteo archive  reaches yesterday, but is rate-weighted such that an
                      11-year pull costs ~306 weighted calls per district
  NASA POWER          11 years in one unweighted call, but runs ~3 days
                      behind real time

So POWER supplies the history and Open-Meteo's archive supplies only the
final few days. That splice is a real methodological compromise and is
labelled rather than hidden: the returned frame carries a `source` column,
and every response built from it reports how many days came from where.

The two products genuinely differ (POWER is MERRA-2 at ~50km, Open-Meteo's
archive is ERA5 at ~25km), so the joint is a small discontinuity. It is
worth accepting only because it is short and confined to the very end of
the series, where the alternative is being three days blind to the
question the user is actually asking. Anything that needs a strictly
homogeneous series — fitting a climatology, calibrating a threshold —
should pass bridge=False and use POWER alone.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.data import power
from src.forecasting.district_registry import DistrictConfig
from src.forecasting.open_meteo import (
    MAX_HISTORICAL_DAYS,
    fetch_historical_weather,
    to_district_daily_schema,
)

logger = logging.getLogger(__name__)

POWER_SOURCE = "nasa-power"
OPEN_METEO_SOURCE = "open-meteo-archive"

# How many recent days to request from Open-Meteo when bridging. POWER's
# measured lag is ~3 days; 30 gives ample overlap so the splice point is
# chosen from data rather than from an assumption about the lag.
BRIDGE_LOOKBACK_DAYS = 30


def district_history(cfg: DistrictConfig, bridge: bool = True) -> pd.DataFrame:
    """One district's daily series from POWER_START_DATE to as close to
    today as the data allows.

    Adds a `source` column. If the Open-Meteo bridge fails for any reason
    the POWER series is returned unchanged — being three days behind is a
    far better outcome than failing the request outright.
    """
    frame = power.get_district_daily_series(
        cfg.district, cfg.state, cfg.latitude, cfg.longitude, cfg.elevation
    )
    frame["source"] = POWER_SOURCE
    if not bridge or frame.empty:
        return frame

    power_end = pd.Timestamp(frame["date_of_record"].iloc[-1])
    try:
        recent = fetch_historical_weather(
            cfg.latitude, cfg.longitude, min(BRIDGE_LOOKBACK_DAYS, MAX_HISTORICAL_DAYS)
        )
        bridged = to_district_daily_schema(
            recent, cfg.district, cfg.state, cfg.latitude, cfg.longitude, cfg.elevation
        )
    except Exception as exc:  # noqa: BLE001 - any bridge failure is non-fatal
        logger.warning("Could not bridge %s past %s: %s", cfg.district, power_end.date(), exc)
        return frame

    bridged = bridged[bridged["date_of_record"] > power_end]
    # Open-Meteo pads the current end of its archive with all-null rows;
    # a null rainfall must never be spliced in as if it were a dry day.
    bridged = bridged[bridged["rainfall"].notna()]
    if bridged.empty:
        return frame

    bridged["source"] = OPEN_METEO_SOURCE
    combined = pd.concat([frame, bridged], ignore_index=True)
    return combined.sort_values("date_of_record").reset_index(drop=True)


def history_provenance(frame: pd.DataFrame) -> dict:
    """Where the series came from, for the API to report verbatim."""
    if frame.empty:
        return {"data_start": None, "data_end": None, "n_days": 0, "sources": {}}

    counts = frame["source"].value_counts().to_dict() if "source" in frame.columns else {}
    bridged = frame[frame.get("source") == OPEN_METEO_SOURCE] if "source" in frame.columns else frame.iloc[0:0]
    return {
        "data_start": str(pd.Timestamp(frame["date_of_record"].iloc[0]).date()),
        "data_end": str(pd.Timestamp(frame["date_of_record"].iloc[-1]).date()),
        "n_days": int(len(frame)),
        "sources": {str(k): int(v) for k, v in counts.items()},
        "bridged_days": int(len(bridged)),
        "note": (
            "History is NASA POWER (MERRA-2, ~50km). The most recent days come from Open-Meteo's "
            "ERA5 archive (~25km) because POWER runs about three days behind real time. The two "
            "are different products, so there is a small discontinuity at the join."
        ),
    }


def recent_soil_and_dryness(frame: pd.DataFrame) -> dict:
    """Signals the crop advisory needs that the forecast does not carry.

    Soil wetness is POWER-only (Open-Meteo's bridge rows do not have it),
    so the most recent row that actually has a value is used rather than
    the last row outright.
    """
    from src.config import DRY_DAY_THRESHOLD_MM

    if frame.empty:
        return {}

    out: dict = {}
    for column in ("soil_wetness_root_zone", "soil_wetness_surface"):
        if column in frame.columns:
            values = frame[column].dropna()
            if not values.empty:
                out[column] = round(float(values.iloc[-1]), 3)

    rainfall = frame["rainfall"].to_numpy(dtype=float)
    streak = 0
    for value in reversed(rainfall):
        if pd.isna(value):
            break  # unknown is not dry; stop counting rather than assume
        if value >= DRY_DAY_THRESHOLD_MM:
            break
        streak += 1
    out["consecutive_dry_days"] = int(streak)
    return out
