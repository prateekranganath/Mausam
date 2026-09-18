"""Apples-to-apples comparison: per-district model vs. the pooled
all-India model ("Rainfall_Forecast_Mausam"), evaluated on the IDENTICAL
test-date window for each district (the all-India model's test split,
2024-07-01..2025-02-10 by default — a subset of the per-district model's
own, longer test window, so both are scored on the same held-out rows).

Each model uses its OWN preprocessor (own climatology/threshold/imputer,
each fit only on its own training split) — this compares the two trained
artifacts as shipped, not a re-fit of either.

Usage:
    python scripts/compare_models.py --districts "Thiruvananthapuram,New Delhi,Mumbai Suburban,Jaisalmer"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALL_INDIA_SPLIT_DATES
from src.data.cleaner import clean_dataset
from src.data.district import get_district_daily_series
from src.data.loader import load_raw_dataset
from src.features.engineering import build_features, compute_target
from src.ml import evaluate as ev
from src.ml.predict import RainfallRiskPredictor


def evaluate_on_window(predictor: RainfallRiskPredictor, district_daily, start: str, end: str) -> dict:
    feats = compute_target(build_features(district_daily))
    d = feats["date_of_record"]
    window = feats[(d >= start) & (d <= end)]

    X, labeled = predictor.preprocessor.transform(window)
    valid = labeled["rainfall_next_7_days"].notna()
    X, labeled = X[valid], labeled[valid]
    if len(X) == 0:
        return {"n_samples": 0}

    y_class = labeled["insufficient_rainfall_next_7_days"].astype(int)
    y_reg = labeled["rainfall_next_7_days"]
    proba = predictor.classifier.predict_proba(X)[:, 1]
    pred_mm = predictor.regressor.predict(X)

    return {
        "classifier": ev.classification_metrics(y_class, proba),
        "regressor": ev.regression_metrics(y_reg, pred_mm),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--districts", default="Thiruvananthapuram,New Delhi,Mumbai Suburban,Jaisalmer")
    parser.add_argument("--start", default=ALL_INDIA_SPLIT_DATES.test_start)
    parser.add_argument("--end", default=ALL_INDIA_SPLIT_DATES.test_end)
    args = parser.parse_args()
    districts = [d.strip() for d in args.districts.split(",")]

    raw = load_raw_dataset()
    clean = clean_dataset(raw)

    all_india = RainfallRiskPredictor.load_local_all_india()

    rows = []
    for district in districts:
        daily = get_district_daily_series(clean, district)

        try:
            per_district = RainfallRiskPredictor.load_local(district)
            pd_result = evaluate_on_window(per_district, daily, args.start, args.end)
        except FileNotFoundError:
            pd_result = None

        ai_result = evaluate_on_window(all_india, daily, args.start, args.end)

        rows.append({"district": district, "per_district": pd_result, "all_india": ai_result})

    print(f"\nTest window (identical for both): {args.start} to {args.end}\n")
    header = f"{'District':<22} {'Model':<14} {'n':>5} {'ROC-AUC':>8} {'PR-AUC':>7} {'Brier':>7} {'Recall':>7} {'Prec':>7} | {'MAE':>7} {'RMSE':>7} {'R2':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        for label, result in [("per-district", row["per_district"]), ("all-india", row["all_india"])]:
            if result is None:
                print(f"{row['district']:<22} {label:<14} {'(not trained locally)':>5}")
                continue
            if result.get("n_samples") == 0 or "classifier" not in result:
                print(f"{row['district']:<22} {label:<14} {'(no rows in window)':>5}")
                continue
            c, r = result["classifier"], result["regressor"]
            print(
                f"{row['district']:<22} {label:<14} {c['n_samples']:>5} "
                f"{c['roc_auc'] or float('nan'):>8.3f} {c['pr_auc'] or float('nan'):>7.3f} "
                f"{c['brier_score']:>7.3f} {c['recall']:>7.3f} {c['precision']:>7.3f} | "
                f"{r['mae']:>7.2f} {r['rmse']:>7.2f} {r['r2'] or float('nan'):>7.3f}"
            )
        print()

    out_path = Path(__file__).resolve().parent.parent / "models" / "comparison_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()
