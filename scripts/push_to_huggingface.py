"""Upload trained model artifacts to Hugging Face Hub (https://huggingface.co/neollm007).

What this uploads (nothing else):
    models/rainfall_risk_classifier.joblib   — calibrated classifier
    models/rainfall_amount_regressor.joblib  — regressor for expected mm
    models/preprocessor.joblib               — climatology/threshold/imputer, fit on train only
    models/climatology_baseline.joblib       — the baseline this model is compared against
    models/feature_schema.json               — feature column order/names
    models/model_metadata.json               — target definitions, thresholds, split dates, version
    models/evaluation_results.json           — test-set metrics (baseline vs model)
    A generated README.md model card summarizing all of the above.

Nothing else in the repo (raw data, .env, source code) is uploaded.

Requires HF_TOKEN in the environment (.env). Repo id defaults to
<HF_USERNAME>/rainfall-risk-<district-slug>, override with --repo-id.

Usage:
    python scripts/push_to_huggingface.py --dry-run          # inspect only, no auth needed
    python scripts/push_to_huggingface.py                    # actually push (needs HF_TOKEN)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import HF_TOKEN, MODELS_DIR, hf_repo_id

REQUIRED_FILES = [
    "rainfall_risk_classifier.joblib",
    "rainfall_amount_regressor.joblib",
    "preprocessor.joblib",
    "climatology_baseline.joblib",
    "feature_schema.json",
    "model_metadata.json",
    "evaluation_results.json",
]


def build_model_card(metadata: dict, results: dict) -> str:
    clf = results["classifier"]
    base = results["baseline"]
    reg = results["regressor"]
    base_reg = results["baseline_regression"]
    return f"""---
license: mit
tags:
  - tabular-classification
  - rainfall-forecasting
  - india
  - random-forest
  - xgboost
---

# Rainfall Risk Model — {metadata['district']}, {metadata['state']}

Hackathon MVP. Predicts the probability of **insufficient rainfall over the
next {metadata['forecast_horizon_days']} days** for {metadata['district']},
{metadata['state']}, India, plus an expected total-rainfall estimate (mm).

**This is not an operational meteorological forecast.** It is a
district-level statistical model trained on ~10 years of historical daily
station data, validated with a chronological train/val/test split. It
does not claim panchayat-level accuracy and has not been validated by any
meteorological authority.

## Target definition

- `rainfall_next_7_days`: sum of observed rainfall (mm) over T+1..T+7.
- `insufficient_rainfall_next_7_days`: 1 if `rainfall_next_7_days` falls
  below the **{metadata['rainfall_risk_threshold_percentile']}th percentile
  of that calendar month's rainfall distribution**, computed from the
  training period only (IMD-style below-normal/lower-tercile convention).
  This makes "insufficient" mean "notably drier than usual for this time
  of year," not a fixed mm cutoff that would be meaningless across
  monsoon vs. dry season.

Per-month threshold (mm), learned from training data only:
```json
{json.dumps(metadata['rainfall_risk_threshold_mm_by_month'], indent=2)}
```

## Data splits (chronological, no shuffling)

- Train: {metadata['train_date_range'][0]} to {metadata['train_date_range'][1]}
- Validation: {metadata['val_date_range'][0]} to {metadata['val_date_range'][1]}
- Test: {metadata['test_date_range'][0]} to {metadata['test_date_range'][1]}

## Model

- Classifier: **{metadata['classifier_model_name']}** ({metadata['classifier_params']}), calibrated ({metadata['classifier_calibration']})
- Regressor: **{metadata['regressor_model_name']}** ({metadata['regressor_params']})
- Class imbalance handling: {metadata['class_imbalance_handling']}

## Test-set performance vs. climatology baseline

Baseline predicts only the historical month-of-year base rate — no
information about recent/current weather. This is the bar the model must
clear.

| Metric | Climatology baseline | {metadata['classifier_model_name']} |
|---|---|---|
| Accuracy | {base['accuracy']:.3f} | {clf['accuracy']:.3f} |
| Precision | {base['precision']:.3f} | {clf['precision']:.3f} |
| Recall | {base['recall']:.3f} | {clf['recall']:.3f} |
| F1 | {base['f1']:.3f} | {clf['f1']:.3f} |
| ROC-AUC | {base['roc_auc']:.3f} | {clf['roc_auc']:.3f} |
| PR-AUC | {base['pr_auc']:.3f} | {clf['pr_auc']:.3f} |
| Brier score | {base['brier_score']:.3f} | {clf['brier_score']:.3f} |

Regression (`rainfall_next_7_days`, mm):

| Metric | Climatology baseline | {metadata['regressor_model_name']} |
|---|---|---|
| MAE | {base_reg['mae']:.2f} | {reg['mae']:.2f} |
| RMSE | {base_reg['rmse']:.2f} | {reg['rmse']:.2f} |

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
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Build everything but do not call the HF API")
    args = parser.parse_args()

    missing = [f for f in REQUIRED_FILES if not (MODELS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts {missing} in {MODELS_DIR} — run scripts/train_model.py first."
        )

    with open(MODELS_DIR / "model_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(MODELS_DIR / "evaluation_results.json", encoding="utf-8") as f:
        results = json.load(f)

    repo_id = args.repo_id or hf_repo_id(metadata["district"])

    card = build_model_card(metadata, results)
    card_path = MODELS_DIR / "README.md"
    card_path.write_text(card, encoding="utf-8")

    print(f"Target repo:    {repo_id}")
    print(f"Files to push:  {REQUIRED_FILES + ['README.md']}")
    print(f"Model card written to: {card_path}")

    if args.dry_run:
        print("\n--dry-run set: not calling the Hugging Face API.")
        return

    if not HF_TOKEN:
        print(
            "\nHF_TOKEN is not set in the environment (.env) — cannot push. "
            "The model card and file list above were still generated/validated."
        )
        return

    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)

    for filename in REQUIRED_FILES + ["README.md"]:
        api.upload_file(
            path_or_fileobj=str(MODELS_DIR / filename),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Uploaded {filename}")

    print(f"\nDone: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
