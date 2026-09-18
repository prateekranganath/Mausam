"""Loads the raw Indian rainfall/weather dataset.

The source file is a 64MB / ~970K-row .xlsx. Reading it with pandas'
openpyxl engine takes roughly a minute, so the first load is cached to a
Parquet file under Data/processed/. The raw .xlsx is never modified.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import RAW_CLEANED_CACHE, RAW_DATA_PATH

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "date_of_record",
    "month",
    "season",
    "station_name",
    "state",
    "district",
    "avg_temp",
    "min_temp",
    "max_temp",
    "wind_speed",
    "air_pressure",
    "elevation",
    "latitude",
    "longitude",
    "rainfall",
]


def load_raw_dataset(force_reload: bool = False) -> pd.DataFrame:
    """Load the full raw dataset, using a Parquet cache when available.

    Parameters
    ----------
    force_reload:
        If True, ignore any existing cache and re-read the source .xlsx.
    """
    if not force_reload and RAW_CLEANED_CACHE.exists():
        logger.info("Loading cached dataset from %s", RAW_CLEANED_CACHE)
        return pd.read_parquet(RAW_CLEANED_CACHE)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_DATA_PATH}. It must be present "
            "in the project's Data/ folder."
        )

    logger.info("Reading raw dataset from %s (first load is slow)", RAW_DATA_PATH)
    df = pd.read_excel(RAW_DATA_PATH, sheet_name="Sheet1", engine="openpyxl")

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing expected columns: {sorted(missing_cols)}. "
            "The schema may have changed — inspect the source file."
        )
    df = df[EXPECTED_COLUMNS]

    RAW_CLEANED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_CLEANED_CACHE, index=False)
    logger.info("Cached raw dataset to %s (%d rows)", RAW_CLEANED_CACHE, len(df))
    return df
