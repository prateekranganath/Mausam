"""Train the pooled all-India rainfall-risk model ("Rainfall_Forecast_Mausam")
across every district in the dataset, instead of one model per district.

Usage:
    python scripts/train_all_india_model.py
    python scripts/train_all_india_model.py --force-reload-data   # rebuild the cached all-districts series
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.train_all_india import train_all_india


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-reload-data", action="store_true")
    args = parser.parse_args()

    results = train_all_india(force_reload_data=args.force_reload_data)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
