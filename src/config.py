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

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def district_slug(district: str) -> str:
    return district.replace(" ", "-").lower()


def local_model_dir(district: str) -> Path:
    """Each district's local artifacts live in their own subfolder, so
    training a second district never overwrites the first — matches the
    per-district repo naming already used on Hugging Face (hf_repo_id).
    """
    path = MODELS_DIR / district_slug(district)
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(parents=True, exist_ok=True)
    return path

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
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

API_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("API_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    if o.strip()
]


def hf_repo_id(district: str) -> str:
    return f"{HF_USERNAME}/rainfall-risk-{district_slug(district)}"


@dataclass(frozen=True)
class SplitDates:
    train_start: str = TRAIN_START
    train_end: str = TRAIN_END
    val_start: str = VAL_START
    val_end: str = VAL_END
    test_start: str = TEST_START
    test_end: str = TEST_END


SPLIT_DATES = SplitDates()

# --------------------------------------------------------------------------
# All-India pooled model — a SEPARATE, shorter split window from the
# single-district model above.
#
# Evidence (checked across the full dataset, all districts, before
# choosing this): rainfall-reporting completeness has a hard, dataset-wide
# step change exactly at 2021-01-01 (missingness ~87% in Dec 2020 to ~6.5%
# in Jan 2021, uniformly across every district simultaneously — a data-
# generation-process artifact, not a real station-rollout, which would be
# gradual and regionally staggered, not a single nationwide overnight
# jump). Missingness then falls further to ~0% by mid-2022 onward.
#
# Because of this, and because the Thiruvananthapuram-specific "missing
# correlates with dry/warm conditions" justification for zero-imputation
# did NOT replicate nationally (checked across 143 districts: only 66
# showed the same-direction effect, 77 showed the opposite — no reliable
# national signal), the all-India model trains ONLY on the reliably-
# reported era (2021 onward) and relies on dropping incomplete windows
# rather than imputing, avoiding an assumption the evidence doesn't
# support at national scale. See src/ml/train_all_india.py.
ALL_INDIA_TRAIN_START = "2021-01-01"
ALL_INDIA_TRAIN_END = "2023-06-30"
ALL_INDIA_VAL_START = "2023-07-01"
ALL_INDIA_VAL_END = "2024-06-30"
ALL_INDIA_TEST_START = "2024-07-01"
ALL_INDIA_TEST_END = "2025-02-10"

ALL_INDIA_SPLIT_DATES = SplitDates(
    train_start=ALL_INDIA_TRAIN_START,
    train_end=ALL_INDIA_TRAIN_END,
    val_start=ALL_INDIA_VAL_START,
    val_end=ALL_INDIA_VAL_END,
    test_start=ALL_INDIA_TEST_START,
    test_end=ALL_INDIA_TEST_END,
)

ALL_INDIA_MODEL_NAME = "Rainfall_Forecast_Mausam"
ALL_INDIA_PROCESSED_CACHE = PROCESSED_DATA_DIR / "all_districts_daily.parquet"
ALL_INDIA_MODEL_DIR = MODELS_DIR / "all-india"
ALL_INDIA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
(ALL_INDIA_MODEL_DIR / "plots").mkdir(parents=True, exist_ok=True)


def hf_repo_id_all_india() -> str:
    return f"{HF_USERNAME}/{ALL_INDIA_MODEL_NAME}"
