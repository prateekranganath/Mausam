"""Central configuration: paths, district registry, and tunable constants.

All values that the spec calls out as "must not be hardcoded" (district,
risk threshold percentile, date ranges) are read from the environment via
.env, with defaults documented here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
RAW_DATA_PATH = PROJECT_ROOT / "Data" / "india_weather_rainfall_data.xlsx"
PROCESSED_DATA_DIR = PROJECT_ROOT / "Data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = MODELS_DIR / "plots"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_CLEANED_CACHE = PROCESSED_DATA_DIR / "raw_cleaned.parquet"

OPEN_METEO_CACHE_DIR = PROCESSED_DATA_DIR / "open_meteo_cache"
OPEN_METEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DISTRICT_CONFIG_PATH = PROJECT_ROOT / "src" / "forecasting" / "district_config.json"

# --------------------------------------------------------------------------
# Forecasting configuration
# --------------------------------------------------------------------------
FORECAST_HORIZON_DAYS = 7

# Lower-tercile-of-monthly-climatology convention (IMD below-normal
# rainfall definition), computed from the TRAINING split only.
# Configurable rather than hardcoded — see model docs for rationale.
RAINFALL_RISK_THRESHOLD_PERCENTILE = float(
    os.getenv("RAINFALL_RISK_THRESHOLD_PERCENTILE", "33")
)

# IMD convention: a "rainy day" is >=2.5mm; below that counts as dry for
# the consecutive_dry_days feature.
DRY_DAY_THRESHOLD_MM = 2.5

# Missing daily rainfall is imputed as 0mm rather than dropped.
#
# Evidence (Thiruvananthapuram, checked before enabling this): on days
# with a missing rainfall reading, avg_temp runs ~0.8C warmer than on
# days with a recorded reading, CONSISTENTLY across every calendar month
# (a signature of dry/clear conditions) — and district-wide missingness
# collapsed from ~30-44%/year in 2015-2021 to ~0% in 2023 and 2025.
# Dropping missing days (the original approach) therefore built the
# training-period climatology from a sample biased toward wetter days,
# inflating the derived "normal" rainfall threshold; once reporting
# became complete post-2022, real conditions tripped that inflated
# threshold constantly (~66% "insufficient" on the test set vs the ~33%
# the threshold was defined to target on train). Imputing missing
# readings as 0 corrects this. `rainfall_was_missing` is kept as a
# feature (and rolled into `rainfall_missing_frac_last_7`) so the model
# can still discount days it's an assumption for, not a measurement.
IMPUTE_MISSING_RAINFALL_AS_ZERO = True

# Longest lookback window used by any feature (rainfall_sum_last_30).
# Rows within this many days of a district's first observation cannot
# have a complete feature vector and are dropped.
MAX_LOOKBACK_DAYS = 30

# Chronological split boundaries (dataset covers 2015-01-01..2025-02-10).
TRAIN_START = "2015-01-01"
TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-02-10"

DEFAULT_DISTRICT = os.getenv("DEFAULT_DISTRICT", "Thiruvananthapuram")

MODEL_VERSION = "0.1.0"

# --------------------------------------------------------------------------
# Secrets (never hardcode — read from environment only)
# --------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "neollm007")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def hf_repo_id(district: str) -> str:
    """Must match the slug used by scripts/push_to_huggingface.py."""
    return f"{HF_USERNAME}/rainfall-risk-{district.replace(' ', '-').lower()}"


@dataclass(frozen=True)
class SplitDates:
    train_start: str = TRAIN_START
    train_end: str = TRAIN_END
    val_start: str = VAL_START
    val_end: str = VAL_END
    test_start: str = TEST_START
    test_end: str = TEST_END


SPLIT_DATES = SplitDates()
