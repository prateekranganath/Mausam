---
license: mit
tags:
  - tabular-classification
  - rainfall-forecasting
  - india
  - random-forest
  - xgboost
  - multi-district
---

# Rainfall_Forecast_Mausam — All-India Rainfall Risk Model

Hackathon MVP. A SINGLE pooled model, trained across **316 Indian districts**
simultaneously, predicting the probability of **insufficient rainfall over
the next 7 days** for any of them, plus
an expected total-rainfall estimate (mm). Replaces the earlier one-model-
per-district approach — pull this once, use it for any trained district.

**This is not an operational meteorological forecast.** It does not claim
panchayat-level accuracy and has not been validated by any meteorological
authority.

## Why a shorter (2021-2025) date range than the raw data

Restricted to 2021-01-01 onward: rainfall-reporting completeness has a hard, dataset-wide step change exactly at that date (missingness ~87% in Dec 2020 to ~6.5% in Jan 2021, uniformly across every district — a data-generation-process artifact, not a real gradual station-network rollout). Training only on the reliably-reported era avoids depending on an unverified imputation assumption for the bulk of the record.

## Why no zero-imputation for missing rainfall (unlike the single-district models)

impute_as_zero=False for this pooled model — rows with an incomplete 7-day target window or 30-day feature lookback are dropped, not imputed. The Thiruvananthapuram-specific zero-imputation justification (missing correlates with warmer/drier conditions) was checked across 143 districts and did NOT replicate nationally (66 districts same-direction, 77 opposite-direction) — so no blanket imputation assumption is applied at this scale.

## Target definition

- `rainfall_next_7_days`: sum of observed rainfall (mm) over T+1..T+7.
- `insufficient_rainfall_next_7_days`: 1 if that sum falls below the
  **33.0th percentile of that
  (district, calendar month)'s rainfall distribution**, computed from the
  training split only (IMD-style lower-tercile convention) — a district-
  and-season-specific threshold, never a single national cutoff.

## Data splits (chronological, no shuffling)

- Train: 2021-01-01 to 2023-06-30
- Validation: 2023-07-01 to 2024-06-30
- Test: 2024-07-01 to 2025-02-10

## Model

- Classifier: **xgboost** ({'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}), calibrated (sigmoid (Platt), fit on validation split, cv='prefit' (FrozenEstimator))
- Regressor: **xgboost** ({'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 200})
- Class imbalance handling: RandomForest: class_weight='balanced'. XGBoost: scale_pos_weight =4.190 (train neg/pos ratio). No resampling/SMOTE.

## Test-set performance vs. climatology baseline (pooled, all districts)

Baseline predicts only the historical (district, month-of-year) base rate
— no information about recent/current weather. This is the bar the model
must clear.

| Metric | Climatology baseline | xgboost |
|---|---|---|
| Accuracy | 0.790 | 0.754 |
| Precision | 0.000 | 0.394 |
| Recall | 0.000 | 0.318 |
| F1 | 0.000 | 0.352 |
| ROC-AUC | 0.692 | 0.754 |
| PR-AUC | 0.299 | 0.379 |
| Brier score | 0.141 | 0.152 |

Regression (`rainfall_next_7_days`, mm):

| Metric | Climatology baseline | xgboost |
|---|---|---|
| MAE | 23.40 | 23.65 |
| RMSE | 41.25 | 41.93 |

### Per-district sample (sanity check — the model isn't only good for high-row-count districts)

| District | Test rows | ROC-AUC | Regression MAE |
|---|---|---|---|
| Thiruvananthapuram | 211 | 0.494 | 28.45 |
| Mumbai Suburban | 210 | 0.895 | 38.88 |
| New Delhi | 210 | 0.678 | 26.77 |
| Kolkata | 209 | 0.838 | 24.21 |
| Jaisalmer | 200 | 0.995 | 13.78 |
| Bengaluru Urban | 198 | 0.667 | 18.35 |

Full metrics, including every sampled district: see `evaluation_results.json`.

## Inputs / feature schema

See `feature_schema.json`. All features use only information available up
to the prediction day; the 7-day forward window is only ever used to
build the target. Geographic features (latitude/longitude/elevation) have
real variance across training rows here (unlike a single-district model),
so this model can actually use location, not just season, to predict.

## Files in this repo

- `rainfall_risk_classifier.joblib`, `rainfall_amount_regressor.joblib`
- `preprocessor.joblib` — (district, month)-keyed climatology and risk-
  threshold tables, plus median-imputation values, all fit on the
  training split only
- `climatology_baseline.joblib` — the baseline model shown above
- `feature_schema.json`, `model_metadata.json`, `evaluation_results.json`

## Limitations

- Trained only on districts present in the source dataset; a district not
  in `model_metadata.json`'s `districts` list has no learned climatology/
  threshold and should not be queried.
- Two district names collide across states in the source data (Raipur:
  Chhattisgarh/Madhya Pradesh; Cuddalore: Tamil Nadu/Puducherry) — these
  are disambiguated as "Raipur (CT)"/"Raipur (MP)" etc.
- Effective training window is short (~2.5 years) because of the data-
  completeness cutover explained above — less historical depth than the
  single-district models, compensated for by far more districts (breadth)
  per split.
- Not validated against independent ground-truth rainfall records.
