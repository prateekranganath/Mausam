---
license: mit
tags:
  - tabular-classification
  - rainfall-forecasting
  - india
  - random-forest
  - xgboost
---

# Rainfall Risk Model — Thiruvananthapuram, KL

Hackathon MVP. Predicts the probability of **insufficient rainfall over the
next 7 days** for Thiruvananthapuram,
KL, India, plus an expected total-rainfall estimate (mm).

**This is not an operational meteorological forecast.** It is a
district-level statistical model trained on ~10 years of historical daily
station data, validated with a chronological train/val/test split. It
does not claim panchayat-level accuracy and has not been validated by any
meteorological authority.

## Target definition

- `rainfall_next_7_days`: sum of observed rainfall (mm) over T+1..T+7.
- `insufficient_rainfall_next_7_days`: 1 if `rainfall_next_7_days` falls
  below the **33.0th percentile
  of that calendar month's rainfall distribution**, computed from the
  training period only (IMD-style below-normal/lower-tercile convention).
  This makes "insufficient" mean "notably drier than usual for this time
  of year," not a fixed mm cutoff that would be meaningless across
  monsoon vs. dry season.

Per-month threshold (mm), learned from training data only:
```json
{
  "1": 0.0,
  "2": 0.0,
  "3": 0.0,
  "4": 3.294000000000028,
  "5": 45.15400000000003,
  "6": 24.833500000000026,
  "7": 12.149999999999974,
  "8": 7.683999999999943,
  "9": 19.79850000000003,
  "10": 23.46800000000002,
  "11": 20.950000000000028,
  "12": 0.24999999999998934
}
```

## Data splits (chronological, no shuffling)

- Train: 2015-01-01 to 2021-12-31
- Validation: 2022-01-01 to 2023-12-31
- Test: 2024-01-01 to 2025-02-10

## Model

- Classifier: **random_forest** ({'max_depth': 4, 'min_samples_leaf': 5, 'n_estimators': 200}), calibrated (sigmoid (Platt), fit on validation split, cv='prefit')
- Regressor: **random_forest** ({'max_depth': 4, 'min_samples_leaf': 10, 'n_estimators': 200})
- Class imbalance handling: RandomForest: class_weight='balanced'. XGBoost: scale_pos_weight =3.078 (train neg/pos ratio). No resampling/SMOTE was used — the positive class is only mildly imbalanced (~25% by construction of the monthly tercile threshold), and synthetic resampling of lag/rolling features derived from a time series risks producing physically implausible feature combinations.

## Test-set performance vs. climatology baseline

Baseline predicts only the historical month-of-year base rate — no
information about recent/current weather. This is the bar the model must
clear.

| Metric | Climatology baseline | random_forest |
|---|---|---|
| Accuracy | 0.900 | 0.902 |
| Precision | 0.000 | 0.667 |
| Recall | 0.000 | 0.050 |
| F1 | 0.000 | 0.093 |
| ROC-AUC | 0.656 | 0.712 |
| PR-AUC | 0.143 | 0.191 |
| Brier score | 0.108 | 0.095 |

Regression (`rainfall_next_7_days`, mm):

| Metric | Climatology baseline | random_forest |
|---|---|---|
| MAE | 28.86 | 28.50 |
| RMSE | 48.19 | 42.73 |

Full metrics: see `evaluation_results.json` in this repo.

## Inputs / feature schema

See `feature_schema.json` for the exact feature list and order. All
features use only information available up to the prediction day (no
leakage from the T+1..T+7 forecast window) — see the source repository's
`src/features/engineering.py` for the full derivation.

## Files in this repo

- `rainfall_risk_classifier.joblib` — calibrated sklearn/XGBoost classifier
- `rainfall_amount_regressor.joblib` — regressor for expected 7-day rainfall
- `preprocessor.joblib` — climatology table, risk-threshold table, and
  median-imputation values, all fit on the training split only
- `climatology_baseline.joblib` — the baseline model shown above
- `feature_schema.json`, `model_metadata.json`, `evaluation_results.json`

## Limitations

- Single-district model (not yet generalized across India).
- ~27% of raw daily rainfall readings were missing in the source data.
  Missing days were imputed as 0mm rather than dropped, based on evidence
  that missing-rainfall days run systematically warmer (~0.8C, consistent
  across every calendar month) than recorded days — a signature of
  dry/clear conditions — and that missingness collapsed from ~30-44%/year
  pre-2022 to ~0% from 2023 onward. Dropping those rows instead (the
  original approach) built the training climatology from a sample biased
  toward wetter days and made the model unusable out-of-sample (test
  "insufficient" rate ballooned to 66% against a threshold defined to
  target ~33%). This zero-imputation is a documented assumption, not a
  verified measurement, for the imputed days specifically.
- In the driest months (Jan-Mar for this district), normal 7-day rainfall
  is already at or near 0mm, so the lower-tercile threshold itself
  collapses to 0mm — meaning "insufficient rainfall" can structurally
  almost never be flagged in those months by construction. The model is
  more informative for monsoon/transition months than for the already-dry
  season.
- At the default 0.5 decision threshold, test-set recall is low (the
  positive class is only ~10% of test weeks); ROC-AUC/PR-AUC/Brier score
  show the model ranks and calibrates better than climatology, but an
  operating threshold tuned for the intended use case (e.g. an advisory
  alert) would likely outperform the default 0.5 cutoff shown here.
- Not validated against independent ground-truth rainfall records beyond
  the dataset's own test split.
