"""Live rainfall-risk forecast for a district: pulls recent observed
weather from Open-Meteo, runs the trained model, and shows Open-Meteo's
own forecast alongside it. No local dataset or retraining needed — model
artifacts are pulled from Hugging Face Hub.

Usage:
    python scripts/forecast.py --district "Thiruvananthapuram"
    python scripts/forecast.py --district "Thiruvananthapuram" --local-model  # use models/ instead of the Hub
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_DISTRICT, MODELS_DIR
from src.ml.predict import RainfallRiskPredictor


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--district", default=DEFAULT_DISTRICT)
    parser.add_argument("--local-model", action="store_true", help="Load from models/ instead of Hugging Face Hub")
    args = parser.parse_args()

    if args.local_model:
        predictor = RainfallRiskPredictor.load(MODELS_DIR)
    else:
        predictor = RainfallRiskPredictor.from_pretrained(args.district)

    result = predictor.predict_live(args.district)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
