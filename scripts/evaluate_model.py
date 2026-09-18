"""Re-evaluate the already-trained, saved model on the chronological test
split without retraining. Useful for CI-style checks after packaging.

Usage:
    python scripts/evaluate_model.py --district "Thiruvananthapuram"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_DISTRICT, SPLIT_DATES
from src.data.cleaner import clean_dataset
from src.data.district import get_district_daily_series
from src.data.loader import load_raw_dataset
from src.features.engineering import build_features, compute_target
from src.ml import evaluate as ev
from src.ml.predict import RainfallRiskPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--district", default=DEFAULT_DISTRICT)
    parser.add_argument("--state", default=None)
    args = parser.parse_args()

    predictor = RainfallRiskPredictor.load_local(args.district)

    raw = load_raw_dataset()
    clean = clean_dataset(raw)
    daily = get_district_daily_series(clean, args.district, args.state)
    feats = compute_target(build_features(daily))

    d = feats["date_of_record"]
    test_df = feats[(d >= SPLIT_DATES.test_start) & (d <= SPLIT_DATES.test_end)]

    X, labeled = predictor.preprocessor.transform(test_df)
    valid = labeled["rainfall_next_7_days"].notna()
    X, labeled = X[valid], labeled[valid]

    y_class = labeled["insufficient_rainfall_next_7_days"].astype(int)
    y_reg = labeled["rainfall_next_7_days"]

    proba = predictor.classifier.predict_proba(X)[:, 1]
    pred_mm = predictor.regressor.predict(X)

    report = {
        "classifier": ev.classification_metrics(y_class, proba),
        "regressor": ev.regression_metrics(y_reg, pred_mm),
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
