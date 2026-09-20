"""Backfill the NASA POWER daily record for every district in the registry.

One unweighted API call per district returns that district's whole
2015-to-yesterday series, so a full national backfill is ~316 calls
(~15-25 min sequential). The run is RESUMABLE: each district is cached as
its own JSON shard under Data/processed/power_cache/, and a district whose
shard already reaches the requested end date is skipped without a call.
Interrupt it and re-run; it picks up where it stopped.

Usage:
    # everything (316 districts)
    python scripts/build_power_dataset.py

    # a few districts, for a quick check
    python scripts/build_power_dataset.py --districts "Thiruvananthapuram,Jaisalmer,Kolkata"

    # see what would be fetched without calling anything
    python scripts/build_power_dataset.py --dry-run

Output: Data/processed/power_districts_daily.parquet — the same 13-column
schema src.data.district.get_all_districts_daily_series produces, plus
relative_humidity, solar_radiation and the two soil-wetness columns.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import POWER_DISTRICTS_DAILY_CACHE, POWER_START_DATE
from src.data.power import (
    EXTRA_COLUMNS,
    SCHEMA_COLUMNS,
    PowerError,
    _shard_path,
    get_district_daily_series,
)
from src.forecasting.district_registry import list_district_configs
from src.utils.dates import yesterday_ist

logger = logging.getLogger("build_power_dataset")

# POWER never states its rate limit (its docs list HTTP 429 but no
# number), so we self-throttle rather than discover the ceiling the hard
# way. 0.5s between districts puts a full 316-district run at well under
# 2 requests/second and adds only ~3 minutes to the total.
SLEEP_BETWEEN_DISTRICTS_SECONDS = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--districts",
        default=None,
        help="Comma-separated district names. Default: every district in the registry.",
    )
    parser.add_argument("--start", default=POWER_START_DATE)
    parser.add_argument(
        "--end",
        default=None,
        help="YYYY-MM-DD. Default: yesterday IST (today's daily aggregate is still partial).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which districts are already cached and which would be fetched, then stop.",
    )
    parser.add_argument("--out", default=str(POWER_DISTRICTS_DAILY_CACHE))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # The per-request INFO lines from the shared fetcher are useful for one
    # district and unreadable for 316; the progress counter below replaces them.
    logging.getLogger("src.utils.http_cache").setLevel(logging.WARNING)

    end = pd.Timestamp(args.end).date() if args.end else yesterday_ist()

    configs = list_district_configs()
    if args.districts:
        wanted = {d.strip().casefold() for d in args.districts.split(",")}
        configs = [c for c in configs if c.district.casefold() in wanted]
        unknown = wanted - {c.district.casefold() for c in configs}
        if unknown:
            parser.error(f"Unknown district(s): {sorted(unknown)}. See src/forecasting/district_config.json.")
    configs.sort(key=lambda c: c.district)

    cached = [c for c in configs if _shard_path(c.district).exists()]
    logger.info(
        "%d districts in scope, %d already have a shard on disk; target end date %s",
        len(configs), len(cached), end,
    )
    if args.dry_run:
        for c in configs:
            state = "cached" if _shard_path(c.district).exists() else "would fetch"
            logger.info("  %-28s %s", c.district, state)
        return

    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    for i, cfg in enumerate(configs, start=1):
        try:
            frame = get_district_daily_series(
                cfg.district, cfg.state, cfg.latitude, cfg.longitude, cfg.elevation,
                start=args.start, end=end,
            )
        except PowerError as exc:
            # One unreachable district must not lose the other 315. Record
            # it and carry on; re-running the script retries only the
            # failures, since successes are already sharded.
            logger.error("[%d/%d] %s FAILED: %s", i, len(configs), cfg.district, exc)
            failures[cfg.district] = str(exc)
            continue

        frames.append(frame)
        span = (
            f"{frame['date_of_record'].iloc[0].date()}..{frame['date_of_record'].iloc[-1].date()}"
            if not frame.empty else "EMPTY"
        )
        logger.info(
            "[%d/%d] %-28s %5d rows  %s  rainfall nulls %d",
            i, len(configs), cfg.district, len(frame), span, int(frame["rainfall"].isna().sum()),
        )
        time.sleep(SLEEP_BETWEEN_DISTRICTS_SECONDS)

    if not frames:
        raise SystemExit("No district data was retrieved; nothing to write.")

    combined = pd.concat(frames, ignore_index=True)
    expected = SCHEMA_COLUMNS + EXTRA_COLUMNS
    assert list(combined.columns) == expected, f"schema drift: {list(combined.columns)}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False)

    logger.info(
        "Wrote %s - %d rows, %d districts, %s..%s",
        out, len(combined), combined["district"].nunique(),
        combined["date_of_record"].min().date(), combined["date_of_record"].max().date(),
    )
    missing_pct = 100 * combined["rainfall"].isna().mean()
    logger.info("Rainfall missingness across the whole table: %.3f%%", missing_pct)
    if failures:
        logger.warning(
            "%d district(s) failed and are NOT in the output: %s. Re-run to retry just these.",
            len(failures), sorted(failures),
        )


if __name__ == "__main__":
    main()
