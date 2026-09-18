"""Run the raw-load -> clean -> district-aggregate pipeline and cache results.

Usage:
    python scripts/prepare_data.py --district "Thiruvananthapuram"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_DISTRICT, PROCESSED_DATA_DIR
from src.data.cleaner import clean_dataset
from src.data.district import get_district_daily_series, list_districts
from src.data.loader import load_raw_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--district", default=DEFAULT_DISTRICT)
    parser.add_argument("--state", default=None)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--list-districts", action="store_true")
    args = parser.parse_args()

    raw = load_raw_dataset(force_reload=args.force_reload)
    logger.info("Raw dataset: %d rows, %d columns", *raw.shape)

    clean = clean_dataset(raw)

    if args.list_districts:
        registry = list_districts(clean)
        print(registry.head(30).to_string(index=False))
        return

    daily = get_district_daily_series(clean, args.district, args.state)

    out_path = PROCESSED_DATA_DIR / f"{args.district.replace(' ', '_').lower()}_daily.parquet"
    daily.to_parquet(out_path, index=False)
    logger.info("Saved district daily series to %s", out_path)

    print("\n--- Summary ---")
    print(f"District: {daily['district'].iloc[0]} ({daily['state'].iloc[0]})")
    print(f"Date range: {daily['date_of_record'].min().date()} to {daily['date_of_record'].max().date()}")
    print(f"Calendar days: {len(daily)}")
    print(f"Missing rainfall days: {daily['rainfall'].isna().sum()} ({100*daily['rainfall'].isna().mean():.1f}%)")
    print(f"Missing avg_temp days: {daily['avg_temp'].isna().sum()} ({100*daily['avg_temp'].isna().mean():.1f}%)")
    print(daily.describe(include="all").T)


if __name__ == "__main__":
    main()
