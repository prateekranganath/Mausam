# Mausam — Indian District-Level Rainfall Risk Forecasting (MVP)

Predicts the probability of **insufficient rainfall over the next 7 days**
for an Indian district, plus an expected total-rainfall estimate (mm).
This is a hackathon MVP, not an operational forecasting system — see
[Limitations](#limitations).

## Status

**Two trained model workflows exist. The default is the pooled all-India
model.**

- **`Rainfall_Forecast_Mausam`** (pooled, all-India) — one model trained
  across 316 districts at once. `scripts/forecast.py` uses this by
  default: pull once from Hugging Face, forecast any trained district, no
  per-district training needed. Pushed to
  [`huggingface.co/neollm007/Rainfall_Forecast_Mausam`](https://huggingface.co/neollm007/Rainfall_Forecast_Mausam).
- **Per-district models** (`rainfall-risk-<district>`) — the original
  workflow, one model per district, e.g.
  [`rainfall-risk-thiruvananthapuram`](https://huggingface.co/neollm007/rainfall-risk-thiruvananthapuram).
  Kept available (`--per-district`) — see [Which one should I use](#which-one-should-i-use) for why.

Live forecasting is wired up to [Open-Meteo](https://open-meteo.com)
(free, no API key) for real recent weather. Backend/dashboard and the
OpenRouter reasoning layer have not been built yet — everything here runs
from the command line.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in HF_TOKEN / OPENROUTER_API_KEY if needed
```

## Quickstart

```bash
# Live forecast for any of the 316 trained districts — pulls the model
# from Hugging Face (cached locally after the first call), pulls recent
# observed weather from Open-Meteo, no local dataset or training needed.
python scripts/forecast.py --district "Thiruvananthapuram"
python scripts/forecast.py --district "Jaisalmer"
python scripts/forecast.py --district "Mumbai Suburban"
```

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
and it works out-of-the-box for any of its 316 districts, not just ones
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
2. Pulls the last ~35 days of Open-Meteo's **observed** weather (never
   forecast values) for that point, mapped onto the exact same daily
   schema historical data uses (`to_district_daily_schema`) — including
   a verified unit fix (`wind_speed_unit=ms`, since Open-Meteo's default
   km/h didn't match the training data's magnitude).
3. Runs it through the identical feature-engineering/preprocessing code
   used in training, then the trained classifier + regressor.
4. Separately reports Open-Meteo's own forward-looking forecast
   (`total_precipitation_sum_mm`, `precipitation_probability_max`) for
   the same 7 days — **never merged into one number** with the model's
   output, since they answer different questions (ours: "unusually dry
   vs. local history?"; Open-Meteo's: "what does live NWP guidance say").

Includes response caching (1hr TTL), retries with backoff, and a
stale-cache fallback if the live API is unreachable.

## Hugging Face packaging (`scripts/push_to_huggingface.py`)

Uploads only: both `.joblib` models, the preprocessor, the climatology
baseline, feature schema, metadata, evaluation results, and a generated
model-card README — never raw data, `.env`, or source code. Requires
`HF_TOKEN`; `--dry-run` builds and inspects everything without calling
the API. `RainfallRiskPredictor.from_pretrained_all_india()` /
`.from_pretrained(district)` pull artifacts back down, cached locally by
`huggingface_hub` after the first call — no retraining needed in future
sessions.

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
- Not yet built: FastAPI backend, React dashboard, OpenRouter reasoning
  layer, ensemble with external forecast sources (investigated — no
  second genuinely usable free/public source was found; see git history
  for the writeup).
