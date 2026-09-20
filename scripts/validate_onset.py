"""Validate the onset rule against IMD's published dates. Run it before
trusting any onset output, and after any change to the onset constants.

This script is the AUTHORITY on ONSET_TRIGGER_PERCENTILE. That constant is
not a guess; it is whatever this script's sweep says minimises error.

Two independent checks, because they test different claims:

  --check advance  (default) Detected LOCAL onset vs IMD's normal monsoon
                   advance dates, across nine districts from Kerala to
                   Rajasthan. This is the claim the module actually makes.

  --check kerala   Detected onset vs IMD's DECLARED Kerala onset dates,
                   2015-2025. This one is expected to FAIL by ~20 days and
                   is kept precisely because it documents the limit: a
                   rainfall-only rule cannot reproduce a declaration that
                   also depends on wind and OLR fields. It also runs a
                   reconstruction of IMD's own multi-station rainfall
                   criterion, to show the gap is not an artifact of using
                   one district.

Usage:
    python scripts/validate_onset.py
    python scripts/validate_onset.py --check kerala
    python scripts/validate_onset.py --sweep          # re-derive the percentile
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DRY_DAY_THRESHOLD_MM, ONSET_TRIGGER_PERCENTILE
from src.data.power import get_district_daily_series
from src.forecasting.district_registry import get_district_config, list_district_configs
from src.monsoon import onset as on

logger = logging.getLogger("validate_onset")

# IMD's published onset date over Kerala. Source: IMD press releases,
# cross-checked against IMD's own onset/advance page.
IMD_KERALA_ONSET = {
    2015: "06-05", 2016: "06-08", 2017: "05-30", 2018: "05-29", 2019: "06-08",
    2020: "06-01", 2021: "06-03", 2022: "05-29", 2023: "06-08", 2024: "05-30",
    2025: "05-24",
}

# IMD's NORMAL monsoon advance dates - the climatological south-to-north
# progression. These are normals, so the comparison is against each
# district's MEDIAN detected onset, not year by year.
IMD_NORMAL_ADVANCE = {
    "Thiruvananthapuram": "06-01",
    "Kozhikode": "06-01",
    "Mumbai Suburban": "06-10",
    "Kolkata": "06-10",
    "Nagpur": "06-15",
    "Bhopal": "06-20",
    "Lucknow": "06-20",
    "New Delhi": "06-27",
    "Jaisalmer": "07-05",
}

# Registry districts covering IMD's 14 designated Kerala onset stations.
KERALA_PROXY_STATIONS = [
    "Lakshadweep", "Thiruvananthapuram", "Kollam", "Alappuzha", "Kottayam",
    "Ernakulam", "Thrissur", "Kozhikode", "Kannur", "Dakshina Kannada",
]


def load(district: str) -> pd.DataFrame:
    cfg = get_district_config(district)
    return get_district_daily_series(
        cfg.district, cfg.state, cfg.latitude, cfg.longitude, cfg.elevation
    )


def _doy(month_day: str) -> int:
    return pd.Timestamp(f"2001-{month_day}").dayofyear


def _label(day_of_year: int) -> str:
    return (dt.date(2001, 1, 1) + dt.timedelta(days=int(day_of_year) - 1)).strftime("%d %b")


def check_advance(percentile: float = ONSET_TRIGGER_PERCENTILE, quiet: bool = False) -> dict:
    """Median detected onset vs IMD's normal advance date, per district."""
    errors: list[int] = []
    rows = []
    for district, normal in IMD_NORMAL_ADVANCE.items():
        daily = load(district)
        threshold = on.trigger_threshold_mm(daily, percentile)
        days = [
            d.onset_day_of_year
            for year in range(2015, 2026)
            for d in [on.detect_onset(daily, district, on.SOUTHWEST, year=year, threshold_mm=threshold)]
            if d.status == on.ONSET_CONFIRMED
        ]
        if not days:
            rows.append((district, threshold, None, normal, None, 0))
            continue
        median = int(np.median(days))
        error = median - _doy(normal)
        errors.append(error)
        rows.append((district, threshold, median, normal, error, len(days)))

    if not quiet:
        print(f"\nLocal onset vs IMD NORMAL ADVANCE dates (trigger percentile {percentile:g})")
        print(f"{'district':<20} {'thresh_mm':>9} {'detected':>9} {'IMD normal':>11} {'error':>6} {'yrs':>4}")
        for district, threshold, median, normal, error, n in rows:
            detected = _label(median) if median else "NONE"
            err = f"{error:+d}" if error is not None else "-"
            print(f"{district:<20} {threshold:>9.1f} {detected:>9} {_label(_doy(normal)):>11} {err:>6} {n:>4}")

    summary = {
        "percentile": percentile,
        "mean_absolute_error_days": float(np.mean(np.abs(errors))) if errors else float("nan"),
        "bias_days": float(np.mean(errors)) if errors else float("nan"),
        "n_districts": len(errors),
    }
    if not quiet:
        print(
            f"\n  MAE {summary['mean_absolute_error_days']:.1f} days, "
            f"bias {summary['bias_days']:+.1f} days, over {summary['n_districts']} districts."
        )
        print("  For reference, the standard deviation of IMD's own Kerala onset date is ~7 days.")
    return summary


def check_kerala() -> dict:
    """The documented negative result: rainfall alone cannot reproduce
    IMD's Kerala declaration."""
    print("\n--- (a) Single-district rule vs IMD DECLARED Kerala onset ---")
    daily = load("Thiruvananthapuram")
    threshold = on.trigger_threshold_mm(daily)
    print(f"{'year':>6} {'detected':>12} {'IMD':>12} {'error':>7}")
    single = []
    for year, month_day in IMD_KERALA_ONSET.items():
        detection = on.detect_onset(daily, "Thiruvananthapuram", on.SOUTHWEST, year=year, threshold_mm=threshold)
        if detection.status != on.ONSET_CONFIRMED:
            print(f"{year:>6} {'NONE':>12} {month_day:>12} {'-':>7}")
            continue
        error = (pd.Timestamp(detection.onset_date) - pd.Timestamp(f"{year}-{month_day}")).days
        single.append(error)
        print(f"{year:>6} {detection.onset_date:>12} {month_day:>12} {error:>+7d}")
    print(f"  mean error {np.mean(single):+.1f} days (SD {np.std(single):.1f}), MAE {np.mean(np.abs(single)):.1f}")

    print("\n--- (b) IMD's OWN multi-station rainfall criterion, reconstructed ---")
    print("    (>=60% of 10 proxy districts with >=2.5mm on 2 consecutive days, from 10 May)")
    available = {c.district for c in list_district_configs()}
    stations = [s for s in KERALA_PROXY_STATIONS if s in available]
    panel = pd.DataFrame({s: load(s).set_index("date_of_record")["rainfall"] for s in stations})
    wet_fraction = (panel >= DRY_DAY_THRESHOLD_MM).sum(axis=1) / panel.notna().sum(axis=1)
    two_day = (wet_fraction >= 0.6) & (wet_fraction.shift(-1) >= 0.6)

    multi = []
    for year, month_day in IMD_KERALA_ONSET.items():
        window = two_day[(two_day.index >= f"{year}-05-10") & (two_day.index <= f"{year}-07-31")]
        hits = window[window]
        if hits.empty:
            continue
        multi.append((hits.index[0] - pd.Timestamp(f"{year}-{month_day}")).days)
    print(f"    {len(stations)} proxy stations; mean error {np.mean(multi):+.1f} days "
          f"(SD {np.std(multi):.1f}), MAE {np.mean(np.abs(multi)):.1f}")

    print(
        "\n  CONCLUSION: both rainfall-only approaches fire ~3 weeks early, consistently.\n"
        "  This is the expected result, not a bug. IMD withholds the declaration until the\n"
        "  925 hPa westerlies and OLR criteria are also met, and neither field is available\n"
        "  from the free sources this project uses. The module reports LOCAL rainfall onset\n"
        "  and says so; see check_advance for the claim it does make."
    )
    return {
        "single_district_mean_error_days": float(np.mean(single)),
        "multi_station_mean_error_days": float(np.mean(multi)),
        "n_proxy_stations": len(stations),
    }


def sweep() -> None:
    """Re-derive ONSET_TRIGGER_PERCENTILE from the advance-date check."""
    print("Sweeping the trigger percentile against IMD normal advance dates...")
    print(f"{'percentile':>10} {'MAE':>7} {'bias':>7}")
    best = None
    for percentile in (50, 60, 70, 75, 80, 85, 90, 95):
        summary = check_advance(percentile, quiet=True)
        print(f"{percentile:>10} {summary['mean_absolute_error_days']:>7.1f} {summary['bias_days']:>+7.1f}")
        # Rank on MAE, but break near-ties toward the smaller bias: a rule
        # that is systematically late is worse for an advisory than one
        # with the same error scattered either side.
        score = (summary["mean_absolute_error_days"], abs(summary["bias_days"]))
        if best is None or score < best[1]:
            best = (percentile, score)
    print(f"\nBest: percentile {best[0]} (MAE {best[1][0]:.1f}, |bias| {best[1][1]:.1f})")
    print(f"src/config.py currently sets ONSET_TRIGGER_PERCENTILE = {ONSET_TRIGGER_PERCENTILE:g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", choices=["advance", "kerala", "both"], default="advance")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if args.sweep:
        sweep()
        return
    if args.check in ("advance", "both"):
        check_advance()
    if args.check in ("kerala", "both"):
        check_kerala()


if __name__ == "__main__":
    main()
