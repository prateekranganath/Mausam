# Mausam — Indian District-Level Rainfall Risk Forecasting (MVP)

Predicts the probability of **insufficient rainfall over the next 7 days**
for an Indian district, plus an expected total-rainfall estimate (mm).
This is a hackathon MVP, not an operational forecasting system — see
[Limitations](#limitations).

## Status

**Phase 1 (ML pipeline) complete** for the pilot district, Thiruvananthapuram,
Kerala. Backend/dashboard have not been built yet — everything here runs
from the command line.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in HF_TOKEN / OPENROUTER_API_KEY if needed
```

## Usage

```bash
# Inspect/cache the dataset for a district (prints diagnostics)
python scripts/prepare_data.py --district "Thiruvananthapuram"
python scripts/prepare_data.py --list-districts   # see what's available

# Train the classifier + regressor, save artifacts to models/
python scripts/train_model.py --district "Thiruvananthapuram"

# Re-evaluate the saved model on the test split without retraining
python scripts/evaluate_model.py --district "Thiruvananthapuram"

# Package + (optionally) push artifacts to Hugging Face
python scripts/push_to_huggingface.py --dry-run   # inspect only
python scripts/push_to_huggingface.py             # needs HF_TOKEN in .env

# Run tests
python -m pytest tests/ -v
```

To switch districts, pass `--district` (and `--state` if the name is
ambiguous — see `src/data/district.py`); nothing else needs to change.

## Dataset

`Data/india_weather_rainfall_data.xlsx` — ~970K daily station-level rows,
314 districts, 32 states, 2015-01-01 to 2025-02-10. Never modified in
place; `scripts/prepare_data.py` caches a cleaned Parquet copy under
`Data/processed/` (gitignored).

**Data-quality finding:** a `station_name` string is not a reliable unique
key — some districts have two physically distinct stations sharing one
name (different lat/lon). Station identity is `(station_name, lat, lon)`.
District-level daily values are the mean across all stations reporting
that day, excluding missing readings from the average rather than
treating them as zero.

**Missing rainfall (~27% of days) is imputed as 0mm, not dropped** — see
`src/config.py::IMPUTE_MISSING_RAINFALL_AS_ZERO` for the evidence this is
based on (missing days run measurably warmer than recorded days, and
missingness itself collapsed to ~0% after 2022, which was distorting the
training-period climatology). `rainfall_was_missing` is retained as a
feature so the model can still tell an imputed day from a measured one.

## Target definition

- `rainfall_next_7_days`: sum of rainfall over T+1..T+7.
- `insufficient_rainfall_next_7_days`: 1 if that sum is below the
  **33rd percentile of that calendar month's rainfall distribution**,
  computed from the training split only (IMD lower-tercile/below-normal
  convention — see `RAINFALL_RISK_THRESHOLD_PERCENTILE` in `.env`).
  This adapts to season instead of using one fixed mm cutoff that would
  be meaningless across monsoon vs. dry season.

## Leakage safety

- Chronological split only: train 2015-2021, validation 2022-2023, test
  2024-2025-02 (`src/config.py::SPLIT_DATES`). No shuffled CV.
- Climatology table, risk thresholds, and imputation medians are fit on
  the training split only (`RainfallFeaturePreprocessor` in
  `src/features/engineering.py`) and applied unchanged to val/test.
- Every feature uses only information available up to and including day
  T; the 7-day forward window is only ever used to build the target.
  Covered by unit tests in `tests/test_engineering.py`.

## Model

Random Forest and XGBoost classifiers/regressors are both grid-searched
(small hyperparameter grid, selected by validation performance), and the
better one is calibrated (Platt scaling, fit on validation) and evaluated
once on the held-out test set — see `models/evaluation_results.json` and
`models/README.md` (generated model card) for full numbers, including the
comparison against a climatology-only baseline.

## Known limitations

- Single-district model so far; the pipeline is designed to retarget via
  `--district`, not yet validated across multiple districts at once.
- In the driest months, normal rainfall is already near 0mm, so the
  lower-tercile threshold collapses to 0 and "insufficient" can
  structurally almost never fire there — the model is more informative
  for monsoon/transition months.
- Test-set recall at the default 0.5 threshold is low (positive class is
  only ~10% of test weeks); ROC-AUC/PR-AUC/Brier score show real skill
  over the baseline, but an operating threshold tuned to the intended use
  case would likely do better than 0.5.
- Not validated against independent ground-truth rainfall records.
