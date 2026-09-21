# Mausam — Indian District-Level Rainfall Risk Forecasting (MVP)

Predicts the probability of **insufficient rainfall over the next 7 days**
for an Indian district, plus an expected total-rainfall estimate (mm).
This is a hackathon MVP, not an operational forecasting system — see
[Limitations](#known-limitations).

## Contents

[Status](#status) · [Getting started](#getting-started) ·
[Usage reference](#usage-reference) · [Dataset](#dataset) ·
[The two workflows](#the-two-workflows-and-why-they-differ) ·
[Target](#target-definition) · [Features](#features-34) ·
[Leakage safety](#leakage-safety) · [Models](#models) ·
[Open-Meteo](#open-meteo-live-integration) ·
[Forecast pipeline](#forecast-pipeline-end-to-end) ·
[Backend API](#backend-api) · [Endpoint reference](#endpoint-reference) ·
[NASA POWER backfill](#nasa-power-the-long-daily-record) ·
[Climate indices](#climate-indices-enso-iod-mjo) ·
[Monsoon onset](#monsoon-onset-detection) ·
[Active/break spells](#activebreak-spell-detection) ·
[Point forecasts](#point-forecasts-and-the-granularity-question) ·
[Crop advisory](#crop-advisory-engine) ·
[Hugging Face](#hugging-face-packaging-scriptspush_to_huggingfacepy) ·
[Project structure](#project-structure) · [Testing](#testing) ·
[Limitations](#known-limitations)

## Status

**Two trained model workflows exist. The default is the pooled all-India
model.**

- **`Rainfall_Forecast_Mausam`** (pooled, all-India) — one model trained
  across the 316 district entries in the data at once, of which **313 are
  servable** (see [Which districts are served](#endpoint-reference)).
  `scripts/forecast.py` and the API use this by default: pull once from
  Hugging Face, forecast any servable district, no per-district training
  needed. Pushed to
  [`huggingface.co/neollm007/Rainfall_Forecast_Mausam`](https://huggingface.co/neollm007/Rainfall_Forecast_Mausam).
- **Per-district models** (`rainfall-risk-<district>`) — the original
  workflow, one model per district, e.g.
  [`rainfall-risk-thiruvananthapuram`](https://huggingface.co/neollm007/rainfall-risk-thiruvananthapuram).
  Kept available (`--per-district`) — see [Which one should I use](#which-one-should-i-use) for why.

Live forecasting is wired up to [Open-Meteo](https://open-meteo.com)
(free, no API key) for real recent weather. A **FastAPI backend** serves
the all-India model plus a rule-based forecast analysis with an optional OpenRouter plain-language note — see
[Backend API](#backend-api). A React dashboard in [`dashboard/`](dashboard/README.md) displays it.

**Beyond the rainfall model**, the service also answers the wider set of
questions the problem statement asks about, each measured against external
ground truth where any exists:

| Capability | Endpoint | Status |
|---|---|---|
| Monsoon onset detection | `/monsoon/onset/{district}` | Built; 8.4 days MAE vs IMD normal advance dates |
| Active/break spell detection | `/monsoon/phase/{district}` | Built; per-district adaptation of Rajeevan et al. (2010) |
| ENSO / IOD / MJO | `/climate/context` | Ingested and served as **context**; measured NOT to help as model features |
| Finer-than-district granularity | `/forecast/point?lat=&lon=` | Grid-downscaled point forecast, explicitly not validated below district level |
| Crop advisory | `/advisory/crop/{district}` | Built; deterministic rule engine, 9 kharif crops |
| Dashboard | `dashboard/` | Built (React + Vite); see [dashboard/README.md](dashboard/README.md) |
| Telegram alerts | `/alerts/telegram/preview`, `/alerts/telegram/send` | Built; preview works with no credentials, send needs a bot token |
| Risk maps, WhatsApp/SMS delivery | — | Not built |

All of it runs on free, key-less public data sources (NASA POWER, NOAA
CPC, NOAA PSL, the IRI Data Library and Open-Meteo).

## Getting started

Developed and tested on Python 3.13 (Windows). Run everything from the repo root.

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure `.env`

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

| Variable | Needed for | Default | Notes |
|---|---|---|---|
| `HF_TOKEN` | Pulling the model at API startup | — | The model repo was created **private** by `push_to_huggingface.py`, so a token is required. Without one the API falls back to `models/all-india/`, which exists only if you trained locally (the `.joblib` files are git-ignored). |
| `HF_USERNAME` | Hugging Face repo names | `neollm007` | |
| `OPENROUTER_API_KEY` | The optional AI note in `/advisory` | — | A key alone is enough: a default model chain is used. |
| `OPENROUTER_MODEL` | `/advisory` | first of the default chain | Tried first. |
| `OPENROUTER_FALLBACK_MODELS` | `/advisory` | laguna-xs, gemma-4-26b, nex-n2.5-mini (all `:free`) | Comma-separated, tried in order when the one before fails. Empty disables fallbacks. |
| `OPENROUTER_TIMEOUT_SECONDS` | `/advisory` | `25` | Per model attempt. |
| `OPENROUTER_TOTAL_BUDGET_SECONDS` | `/advisory` | `40` | For the whole chain: this is what bounds the wait. |
| `API_ALLOWED_ORIGINS` | CORS for a browser frontend | `http://localhost:3000,http://localhost:5173` | Comma-separated. |
| `RAINFALL_RISK_THRESHOLD_PERCENTILE` | Training | `33` | Defines "insufficient" (changing it requires retraining). |
| `DEFAULT_DISTRICT` | CLI scripts | `Thiruvananthapuram` | |

If there is no OpenRouter key, `/advisory` still returns HTTP 200 with the full
`analysis` and `ai_status: "unconfigured"`; every other endpoint works without it.

### 3. Run the API

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

- **Startup takes about 25s** (imports plus loading the model; the first run
  downloads ~1.3MB of artifacts, cached afterwards). It's ready when you see
  `Application startup complete`.
- **Check it:** `curl http://127.0.0.1:8000/health` should return
  `"status":"ok"` and `"n_districts":313`.
- **Interactive docs:** <http://127.0.0.1:8000/docs> (Swagger UI); the raw spec is at `/openapi.json`.
- Add `--reload` while developing. Stop with Ctrl+C.
- **Port already in use?** Use `--port 8001`, or find the process with `netstat -ano | findstr :8000`.

### 4. Or use the CLI (no server)

```bash
# Pulls the model from Hugging Face (cached after the first call) and recent
# weather from Open-Meteo. No local dataset or training needed.
python scripts/forecast.py --district "Thiruvananthapuram"
python scripts/forecast.py --district "Jaisalmer"
python scripts/forecast.py --district "Mumbai Suburban"
```

### What you need for what

| To... | You need |
|---|---|
| Run the API or CLI forecasts | `pip install` and `HF_TOKEN` (or local model artifacts). **No dataset.** |
| Use `/historical` | Nothing extra: it calls Open-Meteo live. |
| Retrain a model | The raw dataset at `Data/india_weather_rainfall_data.xlsx` (git-ignored, ~64MB). Training caches derived files under `Data/processed/`. The all-India run is slow (a grid search over ~270K rows); budget tens of minutes. |
| Run the tests | `pytest`. Tests are offline: all network calls are mocked. |

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Startup fails with `No model available` | Set `HF_TOKEN`, or train locally with `scripts/train_all_india_model.py`. |
| `404 District ... cannot be forecast` | The district is excluded (no or incomplete training data); see `/health` → `excluded_districts`. |
| `503 ... Open-Meteo ... unreachable` | No network and nothing cached. Retry. |
| `/advisory` has `ai_status: "unavailable"` | Every free model in the chain failed (`llm_error` lists why for each). The `analysis` is complete regardless. Retry in a minute, or check the key. |
| "cache-system uses symlinks" / "hf_xet not installed" warnings | Harmless on Windows. |

## Usage reference

```bash
# --- Data pipeline -------------------------------------------------
# Inspect/cache the dataset for one district (prints diagnostics)
python scripts/prepare_data.py --district "Thiruvananthapuram"
python scripts/prepare_data.py --list-districts   # see what's available

# --- Training --------------------------------------------------------
# Per-district model → models/<district-slug>/
python scripts/train_model.py --district "Thiruvananthapuram"

# Pooled all-India model (all 316 districts at once) → models/all-india/
python scripts/train_all_india_model.py

# --- Evaluation --------------------------------------------------------
# Re-evaluate a saved per-district model on its test split, no retraining
python scripts/evaluate_model.py --district "Thiruvananthapuram"

# Head-to-head: per-district vs all-India, on the IDENTICAL test window
python scripts/compare_models.py --districts "Thiruvananthapuram,New Delhi,Mumbai Suburban,Jaisalmer"

# --- Live forecasting (Open-Meteo + trained model) --------------------
python scripts/forecast.py --district "Jaisalmer"                            # all-India model, from HF (default)
python scripts/forecast.py --district "Jaisalmer" --local-model              # all-India model, local only
python scripts/forecast.py --district "Thiruvananthapuram" --per-district    # per-district model instead

# --- Hugging Face packaging -------------------------------------------
python scripts/push_to_huggingface.py --all-india --dry-run      # inspect only
python scripts/push_to_huggingface.py --all-india                # push Rainfall_Forecast_Mausam
python scripts/push_to_huggingface.py --district "Thiruvananthapuram"  # push a per-district model

# --- Monsoon / climate data -------------------------------------------
# Backfill the NASA POWER daily record (resumable; ~316 calls, ~20 min)
python scripts/build_power_dataset.py
python scripts/build_power_dataset.py --districts "Thiruvananthapuram,Jaisalmer" --dry-run

# Validate the onset rule against IMD's published dates (run after any
# change to the onset constants -- this script owns ONSET_TRIGGER_PERCENTILE)
python scripts/validate_onset.py
python scripts/validate_onset.py --check kerala   # the documented negative result
python scripts/validate_onset.py --sweep          # re-derive the percentile

# Is history (NASA POWER) consistent with live serving (Open-Meteo)?
python scripts/compare_rainfall_products.py

# Do ENSO/IOD/MJO earn a place in the model? (they do not -- see below)
python scripts/run_climate_ablation.py

# --- Tests ---------------------------------------------------------
python -m pytest tests/ -v
```

## Dataset

`Data/india_weather_rainfall_data.xlsx` — ~970K daily station-level rows,
316 districts (314 names — 2 collide across states, see below), 32
states, 2015-01-01 to 2025-02-10. Never modified in place;
`scripts/prepare_data.py` / `scripts/train_all_india_model.py` cache
cleaned Parquet copies under `Data/processed/` (gitignored).

**Data-quality findings** (investigated, not assumed):
- A `station_name` string is not a reliable unique key — some districts
  have two physically distinct stations sharing one name (different
  lat/lon). Station identity is `(station_name, lat, lon)`
  (`src/data/cleaner.py`). District-level daily values are the mean
  across all stations reporting that day, excluding missing readings
  from the average rather than treating them as zero
  (`src/data/district.py`).
- Two district *names* collide across states: Raipur (Chhattisgarh /
  Madhya Pradesh) and Cuddalore (Tamil Nadu / Puducherry). The
  single-district pipeline disambiguates via `--state`; the all-India
  pipeline renames them to `"Raipur (CT)"` / `"Raipur (MP)"` etc. so
  every groupby-by-district stays correct (`get_all_districts_daily_series`).
- **Rainfall-reporting completeness has a hard, dataset-wide step change
  exactly at 2021-01-01** — missingness ~87% in Dec 2020 to ~6.5% in Jan
  2021, uniformly across every district simultaneously, then ~0% from
  mid-2022 onward. That uniformity/abruptness is a data-generation-
  process artifact, not a real (gradual, regionally-staggered) station
  network rollout. This directly shaped how the two model workflows
  handle missing rainfall — see below.

## The two workflows, and why they differ

| | Per-district | All-India (pooled) |
|---|---|---|
| Training data | One district, 2015-2025 | All 316 districts, 2021-2025 only |
| Missing rainfall | Imputed as 0mm | Dropped (no imputation) |
| Climatology / threshold | Keyed by month | Keyed by (district, month) |
| Train rows (Thiruvananthapuram-equivalent) | ~2,557 | ~271,745 (pooled) |

**Why the date ranges differ:** the per-district model (built first, on
Thiruvananthapuram) uses the full 2015-2021 train window. The all-India
model deliberately restricts to 2021 onward — see the missingness
step-change above. Training a pooled model on 2015-2020 data would mean
the bulk of the training set is imputed rather than observed, at a scale
where the imputation assumption below doesn't hold.

**Why the imputation strategy differs:** the per-district model imputes
missing rainfall as 0mm, based on district-specific evidence for
Thiruvananthapuram (missing-rainfall days run ~0.8°C warmer than recorded
days, consistently across every month — a dry/clear-conditions
signature). Before applying the same assumption nationally, this was
checked across 143 districts: only 66 showed the same-direction effect,
77 showed the opposite. **No reliable national signal** — so the
all-India model does not impute; it relies on 2021+ already having near-
complete reporting, and drops the small remainder of incomplete rows.

## Target definition

- `rainfall_next_7_days`: sum of rainfall over T+1..T+7.
- `insufficient_rainfall_next_7_days`: 1 if that sum is below the
  **33rd percentile of that (district, calendar month)'s rainfall
  distribution**, computed from the training split only (IMD
  lower-tercile/below-normal convention — configurable via
  `RAINFALL_RISK_THRESHOLD_PERCENTILE` in `.env`). This adapts to season
  *and place* instead of one fixed mm cutoff that would be meaningless
  across monsoon vs. dry season, or Kerala vs. Rajasthan.

## Features (34)

All computed per district from daily data available up to day T
(`src/features/engineering.py`, ordered list in `FEATURE_COLUMNS`):

| Group | Features |
|---|---|
| Rainfall lags (3) | `rainfall_lag_1`, `_3`, `_7` |
| Trailing sums / means (7) | `rainfall_sum_last_3/7/14/30`, `rainfall_rolling_mean_7/14/30` (windows end on T-1) |
| Dry spell (1) | `consecutive_dry_days` (streak of days under 2.5mm ending T-1) |
| Local climatology (2) | this district-month's mean and median daily rainfall, learned in training |
| Same-day weather (6) | `avg_temp`, `min_temp`, `max_temp`, `air_pressure`, `wind_speed`, plus `avg_temp_lag_3` |
| Season (8) | sin/cos of month and of day-of-year; one-hot IMD season (winter, pre-monsoon, SW monsoon, post-monsoon) |
| Location (3) | `latitude`, `longitude`, `elevation` (these vary across districts, unlike in a single-district model) |
| Data-quality flags (4) | `rainfall_missing_frac_last_7`, `temp_/pressure_/wind_missing_flag` |

## Leakage safety

- Chronological split only, no shuffled CV. Per-district:
  train 2015-2021 / val 2022-2023 / test 2024-2025-02
  (`src/config.py::SPLIT_DATES`). All-India:
  train 2021-2023.5 / val 2023.5-2024.5 / test 2024.5-2025-02
  (`ALL_INDIA_SPLIT_DATES`).
- Climatology table, risk thresholds, and imputation medians are fit on
  the training split only (`RainfallFeaturePreprocessor` in
  `src/features/engineering.py`) and applied unchanged to val/test/live.
- Every feature uses only information available up to and including day
  T; the 7-day forward window is only ever used to build the target.
  For pooled data, feature engineering runs **per district** before
  concatenating (`build_features_multi_district`) so no rolling window
  ever crosses a district boundary. Covered by unit tests in
  `tests/test_engineering.py` and `tests/test_district.py`.

## Models

Random Forest and XGBoost classifiers/regressors are grid-searched
(hyperparameter grid selected by validation performance — smaller grid
for the all-India model, since ~100x more data), the better one is
calibrated (Platt scaling, fit on validation via `FrozenEstimator`) and
evaluated once on the held-out test set against a climatology-only
baseline. All-India model: XGBoost won both classifier (val AUC 0.814)
and regressor (val MAE 19.9) searches.

### Per-district vs. all-India — head-to-head (`scripts/compare_models.py`)

Same test window for both (2024-07-01 to 2025-02-10):

| District | Model | ROC-AUC | PR-AUC | Brier | Recall | Precision |
|---|---|---|---|---|---|---|
| Thiruvananthapuram | per-district | 0.560 | 0.143 | **0.126** | 0.000 | 0.000 |
| Thiruvananthapuram | all-india | 0.532 | **0.318** | 0.229 | **0.119** | **0.276** |
| New Delhi | per-district | 0.707 | 0.036 | **0.033** | 0.000 | 0.000 |
| New Delhi | all-india | **0.731** | **0.304** | 0.140 | **0.079** | **0.188** |
| Mumbai Suburban | per-district | 0.880 | 0.386 | 0.083 | 0.529 | 0.321 |
| Mumbai Suburban | all-india | **0.899** | **0.601** | 0.094 | **0.543** | **0.559** |
| Jaisalmer | per-district | **fails to train** — see below | | | | |
| Jaisalmer | all-india | 0.993 | 0.916 | 0.033 | 0.545 | 1.000 |

Full numbers: `models/comparison_results.json`.

### Which one should I use?

**All-India, by default** (`scripts/forecast.py` with no flags) —
it wins on PR-AUC/recall/precision in every district tested. Several
per-district models have recall/precision of exactly 0 at the default
threshold (they never fire an "insufficient" alert at all), which makes
them functionally useless as a risk-alert system regardless of how
well-calibrated their probabilities are (lower Brier score). All-India
is also the only one of the two that could train at all for Jaisalmer,
and it works out-of-the-box for any of its 313 usable districts, not just ones
individually trained.

**Per-district, if** you specifically need better-calibrated
probabilities (lower Brier) for one of the districts it was actually
trained on — Thiruvananthapuram and New Delhi both show this pattern.

## A real training failure, not papered over: Jaisalmer

Training the per-district model for Jaisalmer (a very dry Rajasthan
district) throws `DegenerateTargetError`: over its full 2015-2021 window,
the 33rd-percentile threshold legitimately computes to ~0mm for
most/all months (most weeks really do get ~0mm rain), so
`rainfall_next_7_days < threshold` never evaluates true — **zero
positive examples**, so no binary classifier can be fit at all. This is
a real limitation of a fixed-percentile threshold in a consistently arid
district, not a data bug — `src/ml/train.py` raises this explicitly
instead of crashing uninformatively deep inside sklearn. The all-India
model's shorter (2021-2023.5), coarser-grained per-(district,month)
statistics happened to avoid the same degeneracy and trained fine
(0.993 AUC) — see the code comment in `train_all_india.py` for why.

## Open-Meteo live integration

`src/forecasting/open_meteo.py` — free, no API key, verified against the
live API (not assumed from docs). `RainfallRiskPredictor.predict_live(district)`:

1. Looks up the district's centroid lat/lon (`district_config.json`,
   auto-registered by both training scripts — same coordinates used at
   training time, so live and historical data describe the same place).
2. Pulls the last ~35 days of Open-Meteo's **analysed** past weather (a
   reanalysis/model blend, not rain-gauge readings) for that point, mapped
   onto the exact same daily schema historical data uses
   (`to_district_daily_schema`) — including a verified unit fix
   (`wind_speed_unit=ms`, since Open-Meteo's default km/h didn't match the
   training data's magnitude). Forecast days after today are never fed to
   the model. One caveat: the rainfall features use completed days only,
   but five same-day features (temperature x3, pressure, wind) come from
   *today's* row, which is Open-Meteo's estimate and partly forecast, while
   training saw full observed days — a small train/serve mismatch.
3. Runs it through the identical feature-engineering/preprocessing code
   used in training, then the trained classifier + regressor.
4. Separately reports Open-Meteo's own forward-looking forecast
   (`total_precipitation_sum_mm`, `precipitation_probability_max`) for
   the same 7 days — **never merged into one number** with the model's
   output, since they answer different questions (ours: "unusually dry
   vs. local history?"; Open-Meteo's: "what does live NWP guidance say").

Two Open-Meteo endpoints are used, for different jobs: the **forecast**
endpoint (`past_days` + `forecast_days`) for the model's live inputs and the
comparison forecast, and the **archive** endpoint (explicit
`start_date`/`end_date`) for `/historical`. The forecast endpoint's
`past_days` is documented as up to 92, but measured on 2026-09-18 its oldest
~22 rows came back null for every location tried (data was non-null only
from 2026-07-10, i.e. ~70 usable days) — fine for the model's 35-day window,
but too short for a history chart, hence the archive endpoint.

Both share one client path with response caching (1hr TTL), retries with
backoff, and a stale-cache fallback if the live API is unreachable.

## Forecast pipeline, end to end

What happens from process start to the JSON a client receives. Values below
are from a real run for Kolkata on 2026-09-18.

```
 STARTUP (once)                          PER REQUEST  GET /forecast/Kolkata
 ──────────────                          ─────────────────────────────────
 Hugging Face ─► 4 artifacts             resolve district ─► lat/lon
 ─► RainfallRiskPredictor                       │
 ─► servable_districts (313)             ONE Open-Meteo call (35 past + 8 forecast days)
                                                │
                              rows ≤ today (36) ─┴─ rows > today (7)
                                   │                        │
                          build 34 features       Open-Meteo's own 7-day total
                          + climatology join                │
                                   │                        │
                       classifier + regressor               │
                                   └───────────► agreement ◄┘
```

### At startup (`src/api/main.py`, once per process)

1. `from_pretrained_all_india()` downloads four files from
   `neollm007/Rainfall_Forecast_Mausam` (reused from the Hugging Face cache
   afterwards): the Platt-calibrated XGBoost **classifier**, the XGBoost
   **regressor**, the **preprocessor**, and `model_metadata.json`. If the Hub
   is unreachable it loads `models/all-india/` instead; with neither, startup
   fails loudly.
2. The preprocessor carries what training learned: per-(district, month)
   climatology, per-(district, month) "insufficient" thresholds, median fill
   values and the ordered 34-feature list.
3. `servable_districts` compares the metadata's 316 entries with what the
   preprocessor actually learned, leaving 313.
4. `evaluation_results.json` is loaded for `/model/metrics`.

### Per request

| # | Stage | Code | Kolkata example |
|---|---|---|---|
| 1 | Resolve the district | `routes._resolve` | Case-insensitive; 404 if unknown or excluded → centroid (22.59, 88.39) |
| 2 | One Open-Meteo call | `open_meteo.fetch_daily_weather` | 43 rows, 2026-08-14 to 2026-09-25 (cached 1h) |
| 3 | Map to the training schema | `to_district_daily_schema` | Same columns and units the model was trained on (wind in m/s) |
| 4 | Split at today | `RainfallRiskPredictor.predict` | 36 rows ≤ today go to the model; 7 later rows are held back |
| 5 | Build 34 features | `engineering.build_features` | `rainfall_sum_last_7` = 47.9mm, `_last_30` = 324.4mm, `consecutive_dry_days` = 0 |
| 6 | Join climatology, fill gaps | `preprocessor.transform` | September climatology: mean 14.83, median 4.33 mm/day; 0 NaNs to fill |
| 7 | Score with both models | `predict` | P(unusually dry week) = **0.4959** → `MODERATE`; expected rainfall **99.16mm** |
| 8 | Compare with Open-Meteo | `compute_agreement` | Open-Meteo 95.7mm; Kolkata's September threshold 61.06mm → both above it, sources agree, gap 3.46mm |
| 9 | Shape the response | `routes._build_forecast` | `ml_model`, `open_meteo_forecast`, `agreement` |

`/advisory` adds two steps: a rule-based **analysis** of those numbers (instant,
always present), and an optional plain-language note from an LLM that rewrites
the analysis in simpler words. See [The forecast analysis](#the-forecast-analysis-advisory).

**Risk bands.** `risk_level` is a bucketing of the classifier's probability:
≥0.66 `HIGH`, ≥0.33 `MODERATE`, otherwise `LOW`. Those cutoffs are a
presentation choice, not calibrated to any outcome.

### What is, and isn't, an input to the model

Measured by changing parts of the Open-Meteo data and re-running the model
(baseline 0.4959 / 99.16mm):

| Change | Prediction moved? |
|---|---|
| Future days: rainfall set to 500mm, or temperature +15°C | **No** |
| Past 7 days: rainfall set to 0 | Yes (0.5429, 78.70mm) |
| Past 30 days: rainfall doubled | Yes (0.5341, 81.90mm) |
| Today's rainfall set to 500mm | **No** |
| Today's temperature +10°C | Yes (0.5908, 97.97mm) |

What follows from this:
- **The model uses Open-Meteo's recent weather, not its forecast.** Forecast
  days never enter the feature vector; they only feed the comparison.
- **Today is a partial exception.** Rainfall features use completed days only,
  but five same-day features (temperature ×3, pressure, wind) come from today's
  row, which is Open-Meteo's estimate and partly forecast. Training saw full
  observed days, so this is a small train/serve mismatch.
- **The two sources are not independent.** Both come from Open-Meteo (the
  model reads its recent weather; the comparison uses its forecast), so
  "sources agree" is weaker evidence than two independent forecasters agreeing.
- **The model isn't physically monotone.** Doubling recent rain *lowered* the
  expected rainfall. That was an extreme change and may be an out-of-range
  artefact, but it shows the model learned patterns, not weather physics.
- **Classifier and regressor can disagree.** Kolkata got `MODERATE` risk while
  the regressor and Open-Meteo both point to a wet week. They are separate models.

## Backend API

Run it with the steps in [Getting started](#3-run-the-api). It serves **only the pooled all-India model** (per-district models stay
CLI-only — see the comparison above). The model is pulled from Hugging Face
once at startup (falling back to `models/all-india/` if the Hub is
unreachable), then shared across requests.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model loaded, version, district count, whether the LLM is configured |
| `GET /districts?q=` | The 313 servable districts with coordinates (for a dropdown) |
| `GET /forecast/{district}` | ML risk + Open-Meteo forecast + agreement check. **No LLM** — fast and always available |
| `GET /historical/{district}?days=90` | Last N days (1-730) of weather fetched **live** from Open-Meteo's archive API, ending yesterday |
| `GET /model/metrics?district=` | Held-out test metrics vs. the climatology baseline |
| `GET /advisory/{district}` | Everything in `/forecast` plus a rule-based `analysis` (always present) and an optional LLM plain-language note |

### Endpoint reference

All endpoints are `GET` and **take no request body**: parameters go in the
path or query string. In Postman, import `http://127.0.0.1:8000/openapi.json`
(Import → Link) to generate every request. District names are case-insensitive
and must be URL-encoded (`Mumbai Suburban` → `Mumbai%20Suburban`; Postman does
this for you). Four names need their state suffix: `Raipur (CT)`,
`Raipur (MP)`, `Cuddalore (TN)`, `Cuddalore (PY)` (plain `Raipur` is a 404).

| Endpoint | Parameters | Returns |
|---|---|---|
| `GET /health` | — | `status`, `model_loaded`, `model_name`, `model_version`, `classifier`, `n_districts` (313), `excluded_districts` (name → reason), `openrouter_configured` |
| `GET /districts` | `q` (optional): substring of name or state | `count` and `districts[]` of `{name, state, latitude, longitude, elevation}` |
| `GET /forecast/{district}` | — | `ml_model`, `open_meteo_forecast`, `agreement` (below) |
| `GET /historical/{district}` | `days` 1–730 (default 90) | `source`, `requested_days`, `n_points`, `data_start`, `data_end`, `missing_rainfall_days`, `units`, `data[]` of `{date, rainfall, avg_temp, min_temp, max_temp, wind_speed, air_pressure, relative_humidity}` |
| `GET /model/metrics` | `district` (optional; only 6 sampled districts have per-district metrics: Bengaluru Urban, Jaisalmer, Kolkata, Mumbai Suburban, New Delhi, Thiruvananthapuram) | Test metrics for `classifier`, `baseline`, `regressor`, `baseline_regression`, date ranges, `district_metrics` |
| `GET /advisory/{district}` | `polish` (default `true`; `false` skips the LLM and returns instantly) | `forecast`, `analysis` (always present), `ai_status` (`ok`/`skipped`/`unconfigured`/`unavailable`), `ai_summary` `{text, model}`, `llm_error` |
| `GET /forecast/point` | `lat`, `lon` (required) | `resolved_district`, `distance_km`, `forecast`, `caveat`. 422 outside India or >150 km from any district centroid |
| `GET /climate/context` | — | `enso`, `iod`, `mjo`, each with `value`, `as_of`, `phase`, `publication_lag_days`, `source`. **Context, not a model input** |
| `GET /monsoon/onset/{district}` | `season` (optional): `southwest` or `northeast` | `status`, `onset_date`, `anomaly_days`, `climatology`, `rejected_false_onsets`, `data` (provenance), `not_imd_criterion` |
| `GET /monsoon/phase/{district}` | — | `monsoon_phase` (`active`/`break`/`normal`/`not_applicable`), `days_in_current_phase`, `rainfall_anomaly_sd`, `recent_30_days`, `caveats` |
| `GET /crops` | — | `count` and `crops[]` of `{key, display_name, season, duration_days, seasonal_water_mm}` |
| `GET /advisory/crop/{district}` | `crop` (required), `sowing_date` (optional `YYYY-MM-DD`) | `growth_stage`, `water_balance_mm`, `recommendations[]` with `rule_id` and `triggered_by`, `signals_unavailable`, `disclaimer` |
| `GET /alerts/telegram/preview` | `district` (required), `crop`, `sowing_date` | The exact message a send would deliver, `telegram_configured`, `sections_included`. **Needs no credentials** |
| `POST /alerts/telegram/send` | `district` (required), `crop`, `sowing_date` | `sent`, `message`, `message_id`, `error` (Telegram's own words). Recipient is fixed to `TELEGRAM_CHAT_ID`; message text is composed server-side and cannot be supplied; throttled; 503 if unconfigured |

**Route order note:** `/forecast/point` is declared *before*
`/forecast/{district}` in `routes.py`. Starlette matches in declaration
order, so flipping them makes `point` be read as a district name and the
endpoint 404s. There is a test pinning this.

**Read `rainfall_probability` carefully: despite the name it is the
probability of *insufficient* rainfall** (that the next 7 days are unusually
dry for this district and month). A high value means *drier*, not wetter.

Abridged `GET /forecast/Kolkata`:

```json
{
  "district": "Kolkata", "state": "WB", "as_of_date": "2026-09-18",
  "ml_model": {
    "rainfall_probability": 0.4959, "risk_level": "MODERATE",
    "predicted_rainfall_mm": 99.16, "forecast_horizon_days": 7,
    "model": "xgboost", "model_version": "0.1.0"
  },
  "open_meteo_forecast": {
    "total_precipitation_sum_mm": 95.7,
    "mean_daily_precipitation_probability_percent": 94.1,
    "forecast_days": 7,
    "daily": [{"time": "2026-09-24", "precipitation_sum": 16.8, "precipitation_probability_max": 100}]
  },
  "agreement": {
    "ml_predicted_mm": 99.16, "open_meteo_forecast_mm": 95.7, "difference_mm": 3.46,
    "magnitude_diverges": false, "threshold_mm": 61.06, "threshold_degenerate": false,
    "ml_implies_insufficient": false, "open_meteo_implies_insufficient": false,
    "sources_agree": true
  }
}
```

`GET /advisory/{district}` adds an `analysis` object, **always present** and derived
by rules from the numbers above:

```json
"analysis": {
  "source": "rules",
  "headline": "Moderate risk of an unusually dry week in Nagpur: about 54% chance, with 66.6 mm expected.",
  "risk_level": "MODERATE", "risk_meaningful": true,
  "confidence": "moderate", "confidence_reasons": ["The model and Open-Meteo agree.", "..."],
  "key_factors": ["..."], "model_disagreement": [], "actions": ["..."]
}
```

and, when an LLM answered, `"ai_summary": {"text": "...", "model": "..."}`.
The analysis takes ~0.2s; the AI note takes 2–12s and is cached in-process for an
hour (lost on restart). Call with `?polish=false` for the analysis alone.

| Status | When |
|---|---|
| `200` | Success. Also `/advisory` when the LLM fails: the full `analysis` plus `ai_status: "unavailable"` and `llm_error`. |
| `404` | Unknown district; an excluded district (the reason is given); `/model/metrics?district=` for a district without sampled metrics (lists the available ones). |
| `422` | Invalid parameter, e.g. `days` outside 1–730. |
| `503` | Open-Meteo unreachable with nothing cached; Open-Meteo returned no usable data; evaluation results unavailable. |

**Which districts are served.** Only districts the model learned local
statistics for in *every* calendar month: **313 of the 316 in the data**.
The other three are refused with a 404 giving the reason, and listed in
`/health` under `excluded_districts`: Raisen and Vidisha have no training
data (their records begin 2023-07-05, after the training window ends), and
Bathinda never saw July in training. Serving them would return a
confident-looking number with no local basis. The rule lives in
`servable_districts` (`src/api/state.py`) and also applies to
`scripts/forecast.py`.

**What `/historical` returns.** Daily rainfall, min/max/mean temperature,
wind, pressure and humidity for the district's centroid, straight from
Open-Meteo's archive API — no local dataset needed, so it works on a fresh
clone and reaches yesterday (the old dataset-backed version stopped at
2025-02-10). Today is excluded because its value is still partly a
forecast. The values are **reanalysis, not rain-gauge observations**, so they
can differ from the station dataset the model was trained on, and the most
recent days are preliminary and may be revised. Requested windows are rounded
up to 30/90/180/365/730 days for the on-disk cache (so its size stays bounded
however many `days` values clients try), but you always get exactly the days
you asked for. Trailing days Open-Meteo hasn't filled yet are trimmed so
`data_end` is honest; `n_points` can then be lower than `requested_days`.

**How a `/forecast` is built.** See [Forecast pipeline, end to end](#forecast-pipeline-end-to-end) for the full trace, including exactly which Open-Meteo data feeds the model and which is used only for comparison.

**"Agreement", not "validation".** Open-Meteo is itself a forecast, so it
can't validate our model — real validation needs observed rainfall for the
forecast window (i.e. storing predictions and waiting a week; not built). The
`agreement` block compares the two 7-day totals directly, and applies the
model's own learned (district, month) threshold to both so "insufficient"
means the same thing on each side. It flags `magnitude_diverges` when the two
totals differ substantially, even if both land on the same side of the
threshold.

### The forecast analysis (`/advisory`)

**What it used to be, and why it changed.** `/advisory` asked one free-tier LLM
for the whole answer as a strict six-field JSON object: a risk rating, a
confidence score, key factors, source disagreement and actions. Measured
against the live free tier on 2026-09-21 it was:

- **slow**: 11s to 49s;
- **flaky**: Bhopal failed outright because Nvidia returned `503 Service temporarily overloaded` and the client retried the *same* model twice, one second apart;
- **inconsistent**: it rated Nagpur `LOW` while the forecast model said `MODERATE`, and returned `confidence: 0.0` for Pune.

Asking one free model to do six jobs, several of them judgements the code
already knows the answer to, is what made it fragile.

**What it is now: substance by rules, wording by an optional model.**

| Part | Comes from | Reliability |
|---|---|---|
| `analysis`: headline, risk, confidence, key factors, where the sources differ, actions | `src/llm/analysis.py`: rules over the forecast numbers | Instant (~0.2s), reproducible, cannot fail |
| `ai_summary`: 2–3 plain sentences | A free-tier LLM **rewriting the analysis's own statements** in simpler words | 2–12s, sometimes unavailable; its absence never affects the analysis |

- **Every number in the analysis comes from the input by construction**, and a
  property test asserts it. The risk level is the forecast model's own, passed
  through: nothing re-rates it. Confidence is never `high` (the model is
  hackathon-grade) and drops to `low` when the model and Open-Meteo differ.
- **Where the sources differ, it says which to lean on, with evidence:** on
  held-out data the model's rainfall-amount estimate is not more accurate than
  a climatology baseline (MAE 23.65 mm against 23.40 mm), so the analysis says
  to lean on Open-Meteo for the amount.
- **A dry week that is normal is not reported as low risk.** Where the
  district-month threshold is degenerate, `risk_level` is `null` and the
  headline says a dry week is normal here.

**Measuring the free tier changed the LLM client more than any design choice.**

| Observed | Response |
|---|---|
| Almost every free model is a *reasoning* model. With a small token cap they spend it thinking and the answer is cut off mid-sentence: Nemotron returned `"In Nagpur the forecast indicates"`, laguna returned `"The"` | `reasoning` is switched off (laguna then answered completely in 3.4s), and a reply with `finish_reason: length` is **rejected**, not shown |
| `openrouter/free` returned its chain-of-thought as the answer (`"We need to produce 2 or 3 short plain sentences..."`) | Replies that read as leaked reasoning are rejected |
| `nex-n2.5-mini` answered `"please share the forecast details"` | A reply must mention the district or a supplied number |
| Any single model fails intermittently (503, 429), sometimes minutes after working | A **chain of models** (`OPENROUTER_MODEL`, then `OPENROUTER_FALLBACK_MODELS`); a model that just failed is skipped for a minute; a 40s total budget bounds the wait |
| Given raw numbers to *interpret*, a small model wrote fluent nonsense that no number-check can catch: *"not expected to be unusually dry, though the chance of an unusually dry week is 54%"*, and the **wettest** day called *"the lowest"* | It is given the analysis's already-correct statements to **rewrite**, never numbers to read. Re-measured on six districts afterwards, those contradictions were gone |

**A note that is not faithful is rejected, not shown with a warning.** Its only
value is rewriting the analysis, so an unfaithful one is worse than none (and the
analysis is always there without it). Two checks run on every reply, and either
one sends the request on to the next model:

- *a figure that is not in the input at all* (`999 mm`);
- *a figure attached to the wrong unit*: a live reply turned a **54% chance**
  into *"only about 54 mm of rain is expected against the typical 25.7 mm"*.
  The number 54 **is** in the input, as a percentage, so a numbers-only check
  passed it; the unit-aware check catches it.

What still cannot be caught is a fluent, wrong *interpretation* that uses only
correct figures with correct units. The rewrite-not-interpret design reduces
that rather than eliminating it, which is why the note is labelled as restating
the analysis and adding nothing to it.

If OpenRouter is unconfigured, slow or failing, `/advisory` still returns HTTP
200 with the complete `analysis`, so the numbers never depend on the narrative.
Only **free** models are used. Free models are sometimes overloaded (OpenRouter
returns HTTP 200 with an `error` body; the client treats that and a real error
status identically). To see which free models answer *today*, rank them by
running the same small task against each: availability changes by the hour, and
at the time of writing several (Gemma, GLM, Qwen) returned `429` even when
spaced out, while `nex-n2.5-mini` and `laguna-xs` answered.

Notes: the district registry is read at import time, so retraining (which
rewrites `district_config.json`) needs a server restart. Handlers are plain
`def` on purpose — forecasting does blocking I/O, and `async def` would
freeze the event loop. `API_ALLOWED_ORIGINS` controls CORS.

## NASA POWER: the long daily record

The bundled Excel dataset ends **2025-02-10** and has a hard
reporting step-change at 2021-01-01. That is survivable for a model
trained on 2021–2023, but it makes onset and active/break detection
impossible: both need a long, gap-free daily record that also reaches
**today**, since detecting the *current* season is the entire point.

So a second source was added. `src/data/power.py` pulls
[NASA POWER](https://power.larc.nasa.gov/) daily point data for each
district centroid — free, no API key, no registration.

**Why POWER and not Open-Meteo's archive**, which was already integrated:
Open-Meteo weights archive calls at roughly `(days/14) × (vars/10)`, so one
11-year 10-variable district pull costs ~306 weighted calls, and all 316
districts would be ~96,600 against a 10,000/day free cap — about ten days of
quota. POWER returns the same span in **one unweighted call**: measured
2026-09-20, 4,279 days × 10 variables in 2.6 s, ~570 KB, with one fill value.

```
python scripts/build_power_dataset.py        # ~316 calls, ~20 min, resumable
```

The run is resumable by design — one JSON shard per district, and a district
whose shard already reaches the requested end date is skipped without a call.

**Measured result: 0.000% missing rainfall** across 11.7 years, against the
Excel dataset's 87% → 6.5% step change. Two traps handled explicitly, both
with tests: `PS` is **kPa, not hPa** (×10, applied only *after* the fill
value is removed, so `-999` never becomes a plausible `-9990`), and POWER
runs **~3 days behind** real time, publishing those days as real dates with
every value filled — which would otherwise read as "no rain".

It also unlocks four variables the Excel data never had: relative humidity,
solar radiation, and surface and root-zone soil wetness. The crop advisory
uses soil wetness directly.

### Three products, one quantity — measured, not assumed

History now comes from POWER (MERRA-2, ~50 km) while live forecasts come
from Open-Meteo (ERA5 archive ~25 km). `scripts/compare_rainfall_products.py`
quantifies the difference over 20 districts spanning every Indian rainfall
regime, 730 days:

| | median |
|---|---|
| Bias | **−0.02 mm/day** |
| Correlation | **0.68** |
| Seasonal-total ratio | **0.98** |
| Rainy-day agreement | **87.2%** |

Good in aggregate, but **three districts disagree badly** and their onset and
active/break output should be trusted correspondingly less: **Kamrup**
(Open-Meteo reads 1.94× POWER), **Kozhikode** (1.41×) and **Ratnagiri**
(1.40×). The detectors use POWER for the multi-year climatology and
Open-Meteo only for the final ~3 days, so what matters is agreement on the
*shape* of a district's rainfall rather than matching mm for mm — but the
splice is a real discontinuity and every response reports how many days came
from where.

## Climate indices (ENSO, IOD, MJO)

All three are ingested from free, key-less feeds and served at
`GET /climate/context`:

| Index | Source | Cadence | Measured lag |
|---|---|---|---|
| ONI (ENSO) | NOAA CPC | 3-month seasons, 1950– | **81 days** |
| DMI (IOD) | NOAA PSL | monthly, 1870– | **112 days** |
| MJO RMM | IRI Data Library (mirror of Australian BoM) | daily, 1974– | **3 days** |

Two things worth recording about getting these:

- **BoM's own RMM file now returns an anti-scraping block page**, not data.
  The IRI Data Library mirror works.
- **IRI does not return dates alongside values**, and its two selection
  syntaxes disagree about order: `T/last/N/RANGE` is newest-first while
  `T/(start)/(end)/RANGEEDGES` is oldest-first. Zipping rows against a
  locally generated date range would have silently *reversed* the entire MJO
  series. `_verify_rmm_alignment` therefore re-queries the final day on its
  own and refuses to attach the series if the values disagree.

Publication lag is treated as a **leakage** concern rather than a footnote.
An ONI value labelled "July" could not be known in July; each index declares
a lag and the as-of join uses `available_from = date + lag`, so training and
serving see a value become available at the same point in its life.

### Do they improve the model? Measured: no.

`scripts/run_climate_ablation.py` trains three arms over identical splits,
identical preprocessing and identical hyperparameters, varying only the
feature set. The baseline arm reproduces the shipped model exactly
(val ROC 0.8142, test PR 0.3792), which is what makes the comparison
trustworthy.

| arm | features | val ROC-AUC | test PR-AUC |
|---|---|---|---|
| baseline | 34 | **0.8142** | 0.3792 |
| + MJO | 37 | 0.8070 | 0.3842 |
| + MJO + ONI + DMI | 39 | 0.7724 | 0.3965 |

**ONI and DMI fail structurally, not marginally.** Over the pooled model's
2021-01…2023-06 training window, only **7.9%** of validation ONI values and
**15.8%** of DMI values fall inside the range training ever covered — train
ONI spans `[−1.11, 0.19]`, validation spans `[0.19, 1.99]`, which is the
2020–23 La Niña against the El Niño that followed. A tree can only split on
values it has seen, so there is nothing to generalise from; the model
extrapolates off the end of its own feature and validation ROC drops 0.042.
**More training years would fix this. No amount of tuning will.**

**MJO does not have that problem** — 100% of validation amplitudes fall
inside the training range, as a 30–60 day cycle implies — but it still did
not improve validation ROC (−0.0073). This was contrary to expectation; the
prior written into the ablation script was that MJO *would* help.

Both arms score better on **test**, which is not a reason to ship either:
selecting on the held-out split is precisely what would stop it being held
out. So all three indices are served as context and **none is a model
input**.

## Monsoon onset detection

`GET /monsoon/onset/{district}` — has the monsoon actually arrived here yet?

The rule has two deliberately separate stages: a **trigger** (in a 7-day
window, ≥5 rainy days totalling more than this district's own 85th-percentile
7-day rainfall) and a **persistence** check (across the following 10 days, no
dry run of ≥7 days). Two stages rather than one long window, because a single
17-day total would let one torrential week hide a following fortnight of
drought — which is exactly the pre-monsoon false onset the second stage
exists to reject. A rejected candidate does not end the season's search; the
scan resumes after it, and the response lists what it rejected.

**The threshold is district-relative, and that was forced by measurement.**
No single absolute mm cutoff works nationally: sweeping fixed values never
beat ~15 days MAE, because any value low enough to fire in Jaisalmer fires
weeks early in Kerala, and any value high enough for Kerala never fires in
Jaisalmer at all.

### Status is not just a date

Persistence can only ever be evaluated in retrospect, so a live answer has
to distinguish **`onset_likely`** (rain has arrived, persistence not yet
verifiable) from **`onset_confirmed`**. Collapsing the two would turn an
honest "probably" into a false certainty in exactly the situation a farmer
would act on. The full set: `outside_season` → `pre_onset` → `onset_likely`
→ `onset_confirmed` → `post_onset`, plus `no_onset_detected`.

Both monsoons are handled. Which one applies is decided from the district's
own rainfall climatology, not a curated list of states, so the genuinely
mixed cases (south interior Karnataka, coastal Andhra) come out right.

### Validation: what it achieves, and what it cannot

Run `python scripts/validate_onset.py`. Against **IMD's normal monsoon
advance dates** across nine districts from Kerala to Rajasthan:

| district | detected | IMD normal | error |
|---|---|---|---|
| Thiruvananthapuram | 11 May | 01 Jun | −21 |
| Kozhikode | 21 May | 01 Jun | −11 |
| Mumbai Suburban | 17 Jun | 10 Jun | +7 |
| Kolkata | 28 May | 10 Jun | −13 |
| Nagpur | 21 Jun | 15 Jun | +6 |
| Bhopal | 17 Jun | 20 Jun | −3 |
| Lucknow | 26 Jun | 20 Jun | +6 |
| New Delhi | 28 Jun | 27 Jun | **+1** |
| Jaisalmer | 13 Jul | 05 Jul | +8 |

**MAE 8.4 days, bias −2.2 days.** For reference, the standard deviation of
IMD's own Kerala onset date is ~7 days, so the rule is about as precise as
the thing it is estimating. It is most accurate over central and northern
India and runs early on the pre-monsoon-heavy southern and eastern coasts.

**What it cannot do, stated plainly.** It is not IMD's operational onset
declaration and cannot be. Against IMD's *declared* Kerala onset dates
2015–2025, a rainfall-only rule fires **~20 days early every single year**
(mean −21, SD 5.2) — and that holds even for a faithful reconstruction of
IMD's own **multi-station** rainfall criterion over 10 proxy districts. This
is not a tuning failure; no threshold fixes it. IMD withholds the
declaration until the 925 hPa westerlies and OLR criteria are also met, and
**neither field is obtainable from the free sources here** (checked
2026-09-20: Open-Meteo's ERA5 archive accepts pressure-level variable names
but returns all-null, and NASA POWER has no pressure levels at all). So the
module reports *local rainfall onset* and every response says so.

## Active/break spell detection

`GET /monsoon/phase/{district}` — within the monsoon, is it raining or paused?

The monsoon does not rain steadily from June to September; it alternates
between **active** spells and **break** spells of near-drought that can last
a fortnight. A break during flowering does far more damage than the seasonal
total suggests. Method follows Rajeevan et al. (2010): standardised rainfall
anomaly against a smoothed day-of-year climatology, ±1 SD sustained for ≥3
days.

### The bug that looked like working code

The first implementation standardised **daily** rainfall and reported
**zero break spells in eleven years** while appearing to function
perfectly. Single-district daily rainfall is so right-skewed that the mean
sits well above the median, and a **completely rainless day only reaches
about −0.65 SD**:

| smoothing | skew | min reachable z | % of days ≤ −1 SD |
|---|---|---|---|
| 1 day | 3.08 | **−0.68** | **0.0%** |
| 3 days | 2.17 | −0.88 | 0.0% |
| 7 days | 1.57 | −1.15 | 12.0% |

(Nagpur, JJAS 2015–2026; Thiruvananthapuram −0.69 and Bhopal −0.64 at 1 day.)

A break was not rare, it was **arithmetically impossible**. The fix is to
standardise a **7-day trailing mean**, which cuts skew to ~1.6 and makes the
two sides roughly symmetric (12.0% below −1 SD against 13.2% above). Nagpur
now shows 28 active and 25 break spells over 11 years. The paper sidesteps
the same problem differently, by averaging over the whole core zone rather
than over time; a per-district product cannot do that, and the deviation is
documented in every response.

**JJAS only.** Outside the monsoon the climatological mean approaches zero
and a standardised anomaly explodes — 4 mm of December drizzle in Rajasthan
can score +8 SD. The detector returns `not_applicable` rather than a number
that is arithmetically valid and physically absurd.

## Point forecasts, and the granularity question

The problem statement asks for village/block/panchayat granularity. **That
cannot be validated here and is not claimed.** Training data is
district-mean; nothing below district level has been checked against
anything.

What *is* honest and useful: Open-Meteo is gridded at ~2–11 km, and latitude,
longitude and elevation are already model features. So
`GET /forecast/point?lat=&lon=` pulls weather at the exact coordinate and
runs the normal inference path — but the climatology and risk threshold are
the **resolved district's**, because that is the only level the model was
ever fit at. Every response carries that caveat, and points more than 150 km
from any district centroid are **refused** (422) rather than served by
stretching one district's learned statistics across half a state.

## Crop advisory engine

`GET /advisory/crop/{district}?crop=&sowing_date=` — deterministic,
rule-based sowing and irrigation advice for 9 kharif crops.

**The agronomy is data, not prose, and not the LLM's job.** It would have
been far less code to extend the existing OpenRouter prompt with "and give
crop advice". That is deliberately not what happens, because agronomic
recommendations are the part of this system a farmer would actually act on,
and they need three properties an LLM cannot provide: **reproducible** (same
inputs, same advice, every time), **attributable** (every recommendation
names the `rule_id` in `crop_rules.json` that produced it, and echoes the
signal values that triggered it), and **available** (advice still returns
when the free-tier model is down). The LLM's only remaining job is phrasing.
This is the same separation `routes.py` already applies to the forecast: the
numbers never depend on the narrative.

Rules combine the rainfall forecast with onset status, active/break phase,
root-zone soil wetness and the observed dry-day streak. Growth stage comes
from calendar days since sowing (thermal time would be better — documented
as a simplification, not an oversight).

Two design details that matter:

- **A missing signal never satisfies a condition**, for any operator
  including `<`. "We don't know the water balance" must not read as "the
  water balance is in deficit". Unavailable signals are listed explicitly
  in every response.
- **Dry-risk advice is suppressed where the risk threshold is degenerate.**
  In 265 of 314 districts in January, normal rainfall is already near zero,
  so the risk label carries no information; advice built on it would be
  confidently derived from a meaningless number.

**Sowing windows.** Each crop declares the indicative national months in which
sowing is sensible, and every "go ahead" sowing rule is gated on the window
being open. This exists because of a bug found by looking at the rendered
dashboard: on 20 September the engine told an unsown rice farmer *"Conditions
are suitable for sowing"*. The rules checked onset and forecast risk but never
the calendar, so they kept saying "go" long after the kharif window had
closed. Outside the window the engine now says the window has passed. The
windows are coarse on purpose (real ones vary by region and variety); they
exist to stop a plainly wrong answer, not to schedule a farm.

Crops: rice (transplanted and direct-seeded), maize, cotton, groundnut,
soybean, bajra, ragi and pigeonpea. Water requirements are indicative FAO-56
ETc figures cross-checked against ICAR seasonal totals — planning figures for
a rain-fed advisory, not irrigation prescriptions, and every response carries
that disclaimer plus a pointer to the local KVK.

## Telegram alerts

`GET /alerts/telegram/preview` composes the alert a farmer would receive from
the same data the panels show: the risk line, the onset and spell status, and
the crop advice. `POST /alerts/telegram/send` delivers it. Free, no approval
process, and a working demo needs only a token from @BotFather and a chat id.

```bash
python Messaging/send_telegram.py --district Bhopal --crop soybean --sowing-date 2026-06-28 --preview
python Messaging/send_telegram.py --district Bhopal --crop soybean --sowing-date 2026-06-28
```

Design decisions worth knowing:

- **Preview needs no credentials.** The feature demos fully on a fresh clone;
  only the final send needs `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
- **The client never supplies the text or the recipient.** The endpoint has no
  authentication, so one that accepted either would be an open relay for
  messaging anyone through the bot. Tests assert that caller-supplied
  `message`/`text`/`chat_id` parameters are ignored. Sends are throttled
  (default 5 per minute).
- **A send is never retried.** Every other client in the project retries; a send
  is not idempotent, and a retry after a timeout can deliver the same drought
  warning twice, which is worse than not delivering it.
- **Failures return Telegram's own words** (HTTP 200 with `sent: false`), because
  "Unauthorized", "chat not found" and "bot was blocked" are different problems.
- **Plain text, not Markdown.** Four district names contain parentheses
  (`Raipur (CT)`, `Cuddalore (PY)`), which MarkdownV2 would need escaped.
- **The honesty rules carry through.** Where the district-month risk threshold is
  degenerate the message says a dry week is normal rather than printing a
  meaningless "LOW"; an unconfirmed onset is never worded as settled.

## Hugging Face packaging (`scripts/push_to_huggingface.py`)

Uploads only: both `.joblib` models, the preprocessor, the climatology
baseline, feature schema, metadata, evaluation results, and a generated
model-card README — never raw data, `.env`, or source code. Requires
`HF_TOKEN`; `--dry-run` builds and inspects everything without calling
the API. `RainfallRiskPredictor.from_pretrained_all_india()` /
`.from_pretrained(district)` pull artifacts back down, cached locally by
`huggingface_hub` after the first call — no retraining needed in future
sessions.

## Project structure

```
Data/                         raw dataset (git-ignored) and Data/processed/ caches (git-ignored)
models/
  all-india/                  pooled model: metadata, eval JSON, model card tracked; .joblib git-ignored
  <district>/                 per-district models (same layout)
  comparison_results.json     output of scripts/compare_models.py
  ablation_climate_indices.json  output of scripts/run_climate_ablation.py
scripts/                      CLI entry points (train, evaluate, compare, forecast, push, prepare data,
                              build_power_dataset, validate_onset, compare_rainfall_products,
                              run_climate_ablation)
src/
  config.py                   paths, split dates, env vars, constants (every threshold carries its evidence)
  utils/                      http_cache.py (shared TTL/retry/stale-fallback fetcher), dates.py
  data/                       loader.py, cleaner.py, district.py (raw data → daily district series),
                              power.py (NASA POWER backfill), history.py (POWER + Open-Meteo splice)
  climate/indices.py          ONI, DMI, MJO RMM ingestion + leakage-safe as-of join
  monsoon/                    onset.py (arrival detection), active_break.py (spell detection)
  advisory/                   crop_rules.json (the agronomy, as data), engine.py (pure-function evaluator)
  features/engineering.py     features, target, leakage-safe preprocessor, daily climatology
  ml/                         train.py, train_all_india.py, baseline.py, evaluate.py, predict.py
  forecasting/                open_meteo.py, district_registry.py (+ district_config.json), agreement.py
  llm/                        analysis.py (rule-based forecast analysis), openrouter.py (model-chain client + number tripwire)
  api/                        main.py (app, startup), routes.py, schemas.py, state.py
  alerts/                     compose.py (pure message composer), telegram.py (sender; never retries)
Messaging/send_telegram.py    CLI over src/alerts (--district, --crop, --preview)
dashboard/                    React + Vite front end (see dashboard/README.md)
tests/                        417 tests, all offline
```

## Testing

```bash
python -m pytest tests/ -v
```

417 tests, all offline (Open-Meteo, OpenRouter and the model are stubbed), so
they need no network or keys and cost nothing to run:

| File | Tests | Covers |
|---|---|---|
| `test_engineering.py` | 9 | Target construction, no-leakage of lag/rolling features, per-district isolation |
| `test_district.py` | 6 | Station aggregation, gap days, ambiguous names |
| `test_agreement.py` | 11 | Cross-source comparison, divergence flag, degenerate thresholds |
| `test_open_meteo.py` | 19 | Cache, retries, stale fallback, archive windows and bucketing |
| `test_analysis.py` | 45 | The rule-based analysis; a property test that every number it writes comes from the input |
| `test_openrouter.py` | 61 | Model chain and cooldowns, rejecting truncated / empty / off-task / leaked-reasoning / unfaithful replies, the time budget, the unit-aware figure check |
| `test_api.py` | 39 | Every endpoint, error paths, excluded districts, NaN handling, `/advisory` with the LLM up, down and off |

## Known limitations

- In the driest months/districts, normal rainfall is already near 0mm,
  so the lower-tercile threshold collapses toward 0 and "insufficient"
  can structurally under-fire — most visible as Jaisalmer's per-district
  training failure above, but present to a lesser degree everywhere in
  dry-season months.
- Recall at the default 0.5 decision threshold is inconsistent across
  districts (some per-district models never fire at all); an operating
  threshold tuned to the intended use case (e.g. an advisory alert)
  would likely outperform 0.5 across the board.
- The all-India model's effective training window is short (~2.5 years)
  because of the 2021 data-completeness cutover — less historical depth
  than the per-district models, traded for far more districts (breadth)
  per split.
- Zero-imputation for the per-district models is a documented,
  district-specific assumption (validated for Thiruvananthapuram only),
  not a verified measurement for the imputed days themselves.
- Not validated against independent ground-truth rainfall records beyond
  the dataset's own test split. Not an operational forecast — does not
  claim panchayat-level accuracy.
- Not yet built: risk maps, WhatsApp/SMS delivery,
  stored prediction history (so no real backtest against observed rainfall
  yet), and an ensemble with external forecast sources (investigated — no
  second genuinely usable free/public source was found; see git history).
- **Onset is local rainfall onset, not an IMD declaration**, and runs ~20
  days ahead of IMD's Kerala announcement every year. The 925 hPa wind and
  OLR fields that would close the gap are not available from any free
  source checked. See [Monsoon onset](#monsoon-onset-detection).
- **ENSO and IOD are not model inputs**, because the 2.5-year training
  window does not contain the range of values validation and test require
  (7.9% and 15.8% overlap respectively). This is a data-quantity limit, not
  a modelling choice, and more training years would change it.
- **Active/break spells are per-district**, where the published method uses
  the monsoon core zone as one region, and use a 7-day trailing mean where
  the paper uses daily values. Both deviations make the signal noisier and
  slower-responding than the literature's.
- **History and live serving use different products** (NASA POWER
  MERRA-2 ~50 km vs Open-Meteo ERA5 ~25 km). Measured median bias is −0.02
  mm/day with 87% rainy-day agreement, but Kamrup, Kozhikode and Ratnagiri
  disagree by 1.4–1.9× on totals and their monsoon output is correspondingly
  less reliable.
- **`/forecast/point` is grid downscaling, not a village forecast.** The
  climatology and risk threshold remain the district's, and nothing below
  district level has been validated against anything.
- **Crop advisory growth stages are calendar days from sowing**, not thermal
  time, so a very early or late season will drift. Water requirements are
  indicative planning figures, not irrigation prescriptions.
- `/historical` is reanalysis from Open-Meteo, a different product from
  the station dataset the model was trained on; the two won't match exactly.
- **The API is not production-hardened**: no authentication, no rate
  limiting, and the advisory cache lives in process memory. It is a
  hackathon MVP meant to sit behind a frontend or gateway.
- **Seasonal coverage varies a lot.** The "insufficient rainfall" label only
  means something where a dry week is unusual. Districts (of the 314 with
  thresholds) where the threshold is under 1mm, so the label can't
  meaningfully fire, by month:

  | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 265 | 286 | 265 | 211 | 136 | 27 | 4 | 2 | 5 | 166 | 241 | 265 |

  So the rainfall estimate and Open-Meteo comparison are always available,
  but the risk label is informative for nearly every district in
  Jun–Sep and for only a small minority in winter. Each `/forecast` response
  flags this with `agreement.threshold_degenerate`.
- The agreement check compares two forecasts; it says nothing about which
  is right. In very dry districts the model's threshold collapses to ~0mm,
  so "insufficient" can never register there — visible as
  `threshold_mm: 0.0` in the response.
