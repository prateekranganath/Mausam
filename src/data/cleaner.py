"""Cleaning pass over the raw dataset.

Responsibilities (per the project spec):
  - parse dates correctly
  - remove obviously invalid records
  - sort chronologically
  - preserve geographical information

Deliberately NOT done here (left to feature engineering, fit-on-train-only):
  - imputing missing rainfall/weather values
  - any statistic learned from the data (climatology, scalers, thresholds)
Doing those here would risk leaking test-period statistics into training.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _build_station_id(df: pd.DataFrame) -> pd.Series:
    """station_name is not a reliable unique key.

    Inspection found cases (e.g. district=Thiruvananthapuram) where a single
    station_name string maps to two distinct physical locations with
    different lat/lon/elevation and independently varying readings. A
    station is therefore identified by (station_name, latitude, longitude)
    rounded to 3 decimals (~110m), not by name alone.
    """
    return (
        df["station_name"].astype(str)
        + "__"
        + df["latitude"].round(3).astype(str)
        + "__"
        + df["longitude"].round(3).astype(str)
    )


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Parse, validate, and sort the raw dataset. Returns a new DataFrame."""
    df = df.copy()
    n_start = len(df)

    # --- parse dates ---------------------------------------------------
    df["date_of_record"] = pd.to_datetime(df["date_of_record"], errors="coerce")
    n_bad_date = df["date_of_record"].isna().sum()
    if n_bad_date:
        logger.warning("Dropping %d rows with unparseable dates", n_bad_date)
    df = df[df["date_of_record"].notna()]

    # --- drop rows missing identity fields (can't place them anywhere) -
    identity_cols = ["district", "state", "station_name", "latitude", "longitude"]
    n_before = len(df)
    df = df.dropna(subset=identity_cols)
    if len(df) < n_before:
        logger.warning(
            "Dropped %d rows missing identity fields %s",
            n_before - len(df),
            identity_cols,
        )

    # --- remove obviously invalid physical values -----------------------
    # rainfall cannot be negative
    if "rainfall" in df.columns:
        invalid_rain = df["rainfall"] < 0
        if invalid_rain.any():
            logger.warning("Dropping %d rows with negative rainfall", invalid_rain.sum())
            df = df[~invalid_rain.fillna(False)]

    # min_temp should not exceed max_temp when both are present
    both_present = df["min_temp"].notna() & df["max_temp"].notna()
    inconsistent_temp = both_present & (df["min_temp"] > df["max_temp"])
    if inconsistent_temp.any():
        logger.warning(
            "Dropping %d rows where min_temp > max_temp", inconsistent_temp.sum()
        )
        df = df[~inconsistent_temp]

    # implausible latitude/longitude for India (broad bounding box, catches
    # obvious data-entry errors without being overly strict)
    in_india_bbox = df["latitude"].between(6, 38) & df["longitude"].between(68, 98)
    if (~in_india_bbox).any():
        logger.warning(
            "Dropping %d rows with lat/lon outside India bounding box",
            (~in_india_bbox).sum(),
        )
        df = df[in_india_bbox]

    # --- station identity + sort ----------------------------------------
    df["station_id"] = _build_station_id(df)
    df = df.sort_values(["date_of_record", "district", "station_id"]).reset_index(
        drop=True
    )

    logger.info("clean_dataset: %d -> %d rows", n_start, len(df))
    return df
