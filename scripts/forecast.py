"""Live rainfall-risk forecast for a district: pulls recent observed
weather from Open-Meteo, runs the trained model, and shows Open-Meteo's
own forecast alongside it. No local dataset or retraining needed — model
artifacts are pulled from Hugging Face Hub.

DEFAULT is now the pooled all-India model ("Rainfall_Forecast_Mausam") —
one pull, works for any of its 316 trained districts, no per-district
training/pushing needed. See scripts/compare_models.py's results for why:
across Thiruvananthapuram/New Delhi/Mumbai Suburban (the districts with
both a per-district AND the all-India model, evaluated on the identical
test window) the all-India model matched or beat the per-district one on
PR-AUC/recall/precision in every case, and it's the only one of the two
that could be trained at all for Jaisalmer (the per-district workflow
hits a structural failure there — see train.py's DegenerateTargetError).
Per-district models are kept available (--per-district) since they're
occasionally better calibrated (lower Brier) for a specific district.

Usage:
    python scripts/forecast.py --district "Jaisalmer"                              # pooled all-India model from HF (default)
    python scripts/forecast.py --district "Jaisalmer" --local-model                # pooled model, local only
    python scripts/forecast.py --district "Thiruvananthapuram" --per-district      # per-district model from HF instead
    python scripts/forecast.py --district "Thiruvananthapuram" --per-district --local-model
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_DISTRICT
from src.ml.predict import RainfallRiskPredictor


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--district", default=DEFAULT_DISTRICT)
    parser.add_argument("--per-district", action="store_true", help="Use a single-district model instead of the pooled all-India default")
    parser.add_argument("--local-model", action="store_true", help="Load local artifacts instead of Hugging Face Hub")
    args = parser.parse_args()

    if args.per_district:
        predictor = RainfallRiskPredictor.load_local(args.district) if args.local_model else RainfallRiskPredictor.from_pretrained(args.district)
    else:
        predictor = RainfallRiskPredictor.load_local_all_india() if args.local_model else RainfallRiskPredictor.from_pretrained_all_india()

    if not args.per_district:
        # same usable-district rule as the API: refuse districts the pooled
        # model has no (complete) local statistics for, rather than print a
        # confident number with no local basis
        from src.api.state import servable_districts
        from src.forecasting.district_registry import get_district_config

        _, excluded = servable_districts(predictor)
        canonical = get_district_config(args.district).district
        if canonical in excluded:
            raise SystemExit(f"Cannot forecast '{canonical}': {excluded[canonical]}.")

    result = predictor.predict_live(args.district)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
