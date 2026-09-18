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
the all-India model plus an optional OpenRouter reasoning layer — see
[Backend API](#backend-api). The dashboard has not been built yet.

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
| `OPENROUTER_API_KEY` | `/advisory` only | — | |
| `OPENROUTER_MODEL` | `/advisory` only | — | Any OpenRouter model id; `.env.example` has the recommended free one. |
| `OPENROUTER_TIMEOUT_SECONDS` | `/advisory` | `60` | Free models can take 20–90s. |
| `API_ALLOWED_ORIGINS` | CORS for a browser frontend | `http://localhost:3000,http://localhost:5173` | Comma-separated. |
| `RAINFALL_RISK_THRESHOLD_PERCENTILE` | Training | `33` | Defines "insufficient" (changing it requires retraining). |
| `DEFAULT_DISTRICT` | CLI scripts | `Thiruvananthapuram` | |

If the OpenRouter variables are unset, `/advisory` still returns HTTP 200
with `advisory: null` and an `llm_error`; every other endpoint works without them.

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
| `/advisory` is slow or returns `llm_error` | Free OpenRouter models are sometimes overloaded. Retry, or raise `OPENROUTER_TIMEOUT_SECONDS`. |
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

`/advisory` adds one step: it sends the numbers to OpenRouter, validates the
structured reply, and checks every number in the prose against the input.

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
| `GET /advisory/{district}` | Everything in `/forecast` plus an LLM-written structured advisory |

### Endpoint reference

All six endpoints are `GET` and **take no request body**: parameters go in the
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
| `GET /advisory/{district}` | — | `forecast` (same as `/forecast`), `advisory`, `unsupported_numbers`, `llm_model`, `llm_error` |

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

`GET /advisory/{district}` adds this structured object (all six fields always present):

```json
"advisory": {
  "forecast_summary": "...", "rainfall_risk": "LOW",
  "confidence": 0.5,
  "key_factors": ["..."], "model_disagreement": ["..."], "advisory": ["..."]
}
```

`/advisory` takes roughly 20–45s on the free model and is cached in-process
for an hour (lost on restart).

| Status | When |
|---|---|
| `200` | Success. Also `/advisory` when the LLM fails: `advisory: null` plus `llm_error`. |
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

**The LLM layer (`/advisory`)** is given the numbers and asked only to
summarise, explain disagreement, and give general guidance, as structured
JSON. It is **not a source of weather data**: the prompt forbids inventing
numbers, and every number in its narrative is checked against the input —
anything untraceable is returned in `unsupported_numbers` rather than hidden.
If OpenRouter is unconfigured or failing, `/advisory` still returns HTTP 200
with the full numeric forecast and `advisory: null` + `llm_error`, so the
numbers never depend on the narrative. Set `OPENROUTER_API_KEY` and
`OPENROUTER_MODEL`. Only **free** models are used. Measured through this
client with a realistic payload:

| Free model | Result |
|---|---|
| `nvidia/nemotron-3-super-120b-a12b:free` | **Recommended** — valid output in ~20s |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | Works, valid, but ~75s |
| `nvidia/nemotron-3.5-lightning:free` | Works but ~190s, and copied a probability into `confidence` |
| `thinkingmachines/inkling:free`, `inkling-small:free` | **Unusable** — HTTP 403, "only available on agentic harnesses" |

Free models are sometimes overloaded (OpenRouter returns HTTP 200 with an
`error` body; the client honours the embedded code, retrying transient
503/429 but not permanent 4xx or timeouts). Raise `OPENROUTER_TIMEOUT_SECONDS`
for slower models.

Notes: the district registry is read at import time, so retraining (which
rewrites `district_config.json`) needs a server restart. Handlers are plain
`def` on purpose — forecasting does blocking I/O, and `async def` would
freeze the event loop. `API_ALLOWED_ORIGINS` controls CORS.

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
scripts/                      CLI entry points (train, evaluate, compare, forecast, push, prepare data)
src/
  config.py                   paths, split dates, env vars, constants
  data/                       loader.py, cleaner.py, district.py (raw data → daily district series)
  features/engineering.py     features, target, leakage-safe preprocessor
  ml/                         train.py, train_all_india.py, baseline.py, evaluate.py, predict.py
  forecasting/                open_meteo.py, district_registry.py (+ district_config.json), agreement.py
  llm/openrouter.py           structured-output client + number tripwire
  api/                        main.py (app, startup), routes.py, schemas.py, state.py
tests/                        104 tests, all offline
```

## Testing

```bash
python -m pytest tests/ -v
```

104 tests, all offline (Open-Meteo, OpenRouter and the model are stubbed), so
they need no network or keys and cost nothing to run:

| File | Tests | Covers |
|---|---|---|
| `test_engineering.py` | 9 | Target construction, no-leakage of lag/rolling features, per-district isolation |
| `test_district.py` | 6 | Station aggregation, gap days, ambiguous names |
| `test_agreement.py` | 11 | Cross-source comparison, divergence flag, degenerate thresholds |
| `test_open_meteo.py` | 19 | Cache, retries, stale fallback, archive windows and bucketing |
| `test_openrouter.py` | 31 | Schema validation, retry rules, embedded HTTP-200 errors, number tripwire |
| `test_api.py` | 28 | Every endpoint, error paths, excluded districts, NaN handling |

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
- Not yet built: React dashboard, stored prediction history (so no real
  backtest against observed rainfall yet), and an ensemble with external
  forecast sources (investigated — no second genuinely usable free/public
  source was found; see git history for the writeup).
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
