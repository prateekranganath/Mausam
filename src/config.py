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
# NASA POWER historical backfill (src/data/power.py)
#
# Why a second weather source at all: the bundled Excel dataset ends
# 2025-02-10 and has a hard reporting step-change at 2021-01-01, which is
# why the pooled model trains on only 2021-01..2023-06. Monsoon onset and
# active/break detection both need something the Excel data cannot give —
# a long, gap-free daily record (onset, to say what "normal onset" is for a
# district; active/break, to fit a stable day-of-year mean and SD) that
# also reaches TODAY, since detecting the CURRENT season is the whole point.
#
# POWER rather than Open-Meteo's archive purely on quota: Open-Meteo weights
# archive calls at roughly (days/14) x (vars/10), so 316 districts x 11 years
# x 10 variables is ~96,600 weighted calls against a 10,000/day free cap.
# POWER returns the same span in one unweighted call per district.
POWER_CACHE_DIR = PROCESSED_DATA_DIR / "power_cache"
POWER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
POWER_DISTRICTS_DAILY_CACHE = PROCESSED_DATA_DIR / "power_districts_daily.parquet"

# POWER's own record starts in 1981, but matching the Excel dataset's start
# keeps the two products directly comparable in the bias check, and 2015
# onward is already ample for a day-of-year climatology.
POWER_START_DATE = "2015-01-01"

# --------------------------------------------------------------------------
# Climate indices (src/climate/indices.py)
# --------------------------------------------------------------------------
CLIMATE_INDEX_CACHE_DIR = PROCESSED_DATA_DIR / "climate_index_cache"
CLIMATE_INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Indices are published on a monthly (ONI, DMI) or daily (MJO) cadence, so
# re-fetching more than once a day is pure waste. Six hours gives same-day
# freshness without hammering three free public feeds on every request.
CLIMATE_INDEX_CACHE_TTL_SECONDS = 6 * 3600

# --------------------------------------------------------------------------
# Monsoon onset detection (src/monsoon/onset.py)
#
# A LOCAL, rainfall-only onset rule. It is explicitly NOT IMD's operational
# Kerala criterion, which also requires outgoing-longwave-radiation fields
# and 925 hPa zonal winds that this project does not ingest. What it does
# reproduce is the shape every objective onset definition shares: a burst
# of rain that then SUSTAINS, as opposed to an isolated pre-monsoon
# thunderstorm week.
#
# The trigger window and the persistence window do different jobs and must
# not be collapsed into one longer window: the trigger says "rain arrived",
# the persistence check says "and it stayed", and a single 17-day total
# would let one torrential week mask a fortnight of drought.
# --------------------------------------------------------------------------

# Trigger: within a 7-day window starting on the candidate day, at least
# this many days must be rainy (>= DRY_DAY_THRESHOLD_MM) ...
ONSET_TRIGGER_WINDOW_DAYS = 7
ONSET_MIN_RAINY_DAYS = 5
# ... and the window must total more rain than a DISTRICT-RELATIVE
# threshold: this percentile of that district's own 7-day rolling rainfall
# totals.
#
# Relative, not absolute, because absolute cannot work nationally and this
# was measured rather than assumed. Sweeping a single fixed mm cutoff
# against IMD's published Kerala onset dates never beat ~15 days mean
# absolute error: any value low enough to fire in Jaisalmer fires weeks
# early in Kerala, and any value high enough for Kerala never fires in
# Jaisalmer at all. A percentile of the district's own distribution asks
# the right question — "is this week unusually wet FOR HERE?" — and is the
# same district-relative logic the rainfall-risk target already uses.
#
# 85 chosen by sweeping percentiles against IMD's normal monsoon ADVANCE
# dates across 9 districts from Kerala to Rajasthan (see
# scripts/validate_onset.py, which is the authority on this value):
#
#   percentile |  50   60   70   75   80   85   90
#   MAE (days) | 12.7 12.1 11.4 10.2  8.9  8.3  7.8
#   bias       | -9.8 -9.2 -8.6 -7.3 -5.6 -2.6 +3.3
#
# p90 edges out p85 on MAE but runs systematically late; p85 has the
# smallest bias at near-identical error. 8.3 days is comparable to the
# ~7-day standard deviation of IMD's own Kerala onset date, so the rule is
# about as precise as the thing it is estimating.
ONSET_TRIGGER_PERCENTILE = 85.0

# A floor under the relative threshold. In the most arid districts the 85th
# percentile of 7-day rainfall can itself be near zero, at which point the
# percentile rule would declare onset on a single drizzly week. 20mm over 7
# days is the minimum that can reasonably be called a rainfall onset
# anywhere.
ONSET_ABSOLUTE_FLOOR_MM = 20.0

# Persistence: across the 10 days AFTER the trigger window, the longest run
# of consecutive dry days must stay below this. 7 is the operative number
# because a full dry week immediately after onset is the signature of a
# false onset (a pre-monsoon system passing through), not of the monsoon
# having established itself.
ONSET_PERSISTENCE_WINDOW_DAYS = 10
ONSET_MAX_DRY_RUN_DAYS = 7

# Scan windows, as (month, day) starts and ends.
# SW monsoon: reaches Kerala around 1 June and the northwest by early
# July, so a scan from 1 May leaves room for the earliest arrivals and an
# end of 31 July for the latest.
SW_MONSOON_SCAN_START = (5, 1)
SW_MONSOON_SCAN_END = (7, 31)
# NE monsoon: Tamil Nadu, coastal Andhra, south interior Karnataka and
# Puducherry get the bulk of their rain in Oct-Dec, so a district whose
# rainfall peaks then would be misreported entirely by an SW-only rule.
NE_MONSOON_SCAN_START = (10, 1)
NE_MONSOON_SCAN_END = (12, 31)

# How long after a confirmed onset the status still reads "onset_confirmed"
# rather than "post_onset". Onset is news for a few weeks; after that the
# useful question is the active/break phase instead.
ONSET_RECENT_DAYS = 30

# A district needs at least this many complete historical seasons before a
# median onset date means anything. Below it, the anomaly is reported as
# unavailable rather than computed from two or three years.
ONSET_MIN_SEASONS_FOR_CLIMATOLOGY = 5

# --------------------------------------------------------------------------
# Active / break spell detection (src/monsoon/active_break.py)
#
# Follows Rajeevan et al. (2010): standardised daily rainfall anomaly,
# +/-1 SD sustained for >= 3 days. DEVIATION FROM THE PAPER: it defines
# spells over the monsoon core zone as a single region, whereas this runs
# per district, which is noisier. Documented rather than hidden.
# --------------------------------------------------------------------------
# Rainfall is averaged over this many trailing days BEFORE the anomaly is
# taken. Not cosmetic: single-district daily rainfall is so right-skewed
# that a completely rainless day only reaches about -0.65 SD, which makes a
# break spell arithmetically impossible to detect. Measured minimum daily
# z-scores over JJAS 2015-2026: Nagpur -0.68, Thiruvananthapuram -0.69,
# Bhopal -0.64, with 0.0% of days at or below -1 SD in all three. At a
# 7-day mean the skew drops from ~3.1-3.6 to ~1.6-1.7 and the two sides
# become roughly symmetric (Nagpur: 12.0% below -1 SD, 13.2% above).
# 7 also matches FORECAST_HORIZON_DAYS, so a spell and a forecast describe
# the same length of time.
ACTIVE_BREAK_ACCUMULATION_DAYS = 7

ACTIVE_BREAK_SD_THRESHOLD = 1.0
ACTIVE_BREAK_MIN_RUN_DAYS = 3
# Day-of-year climatology is smoothed over +/- this many days. Daily
# rainfall is far too noisy for a raw per-calendar-day mean to be stable
# with ~11 years of record; +/-7 days pools ~165 observations per point
# while staying well inside the seasonal cycle it needs to resolve.
ACTIVE_BREAK_CLIMATOLOGY_SMOOTH_DAYS = 7
# Active/break is a monsoon-season concept. Outside these months the
# standardised anomaly is dominated by near-zero climatological means and
# the labels stop meaning anything, so they are not reported at all.
ACTIVE_BREAK_MONTHS = (6, 7, 8, 9)

# --------------------------------------------------------------------------
# Point forecasts (src/api/routes.py)
#
# The model's climatology and risk threshold are DISTRICT-level. Serving a
# point far from any district centroid would apply one district's learned
# statistics to somewhere they were never fit, so the request is refused
# instead. ~150km is about the radius of a large Indian district.
# --------------------------------------------------------------------------
MAX_POINT_DISTANCE_KM = 150.0

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
