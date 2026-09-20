"""Quantify the bias between the three rainfall products this project uses.

WHY THIS EXISTS. History now comes from NASA POWER (MERRA-2, ~50km) while
live forecasts come from Open-Meteo (ERA5 archive ~25km, plus NWP forecast
days), and the model itself was trained on the bundled station Excel. Three
different products describing the same quantity is a real risk, and the
README already documents one instance of it (/historical is reanalysis, the
model was trained on station data, "the two won't match exactly").

Rather than assume the difference is small, measure it. This is the same
discipline src/config.py applies to the zero-imputation decision: check
across districts, and be willing to find that the convenient assumption
does not hold.

Open-Meteo's archive is rate-weighted at roughly (days/14) x (vars/10), so
this deliberately samples ~20 districts over a 730-day window (~36 weighted
calls each, ~730 total) rather than sweeping all 316 over 11 years, which
would cost about ten days of free-tier quota.

Usage:
    python scripts/compare_rainfall_products.py
    python scripts/compare_rainfall_products.py --with-excel   # also load the 64MB xlsx
    python scripts/compare_rainfall_products.py --districts "Kolkata,Jaisalmer"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DRY_DAY_THRESHOLD_MM
from src.data.power import get_district_daily_series as power_series
from src.forecasting.district_registry import get_district_config
from src.forecasting.open_meteo import fetch_historical_weather, to_district_daily_schema

logger = logging.getLogger("compare_rainfall_products")

# Chosen to span the range of Indian rainfall regimes rather than to
# flatter the result: very wet coastal, arid northwest, northeast-monsoon
# dominated, Gangetic plain, peninsular interior and hill stations.
DEFAULT_SAMPLE = [
    "Thiruvananthapuram", "Kozhikode", "Dakshina Kannada", "Mumbai Suburban",
    "Ratnagiri", "Jaisalmer", "Bikaner", "Jodhpur", "New Delhi", "Lucknow",
    "Patna", "Kolkata", "Kamrup", "Bhopal", "Nagpur",
    "Hyderabad", "Bengaluru Urban", "Chennai", "Madurai", "Shimla",
]

COMPARISON_DAYS = 730


def metrics(a: pd.Series, b: pd.Series) -> dict:
    """How two daily rainfall series differ. `a` is the reference."""
    joined = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(joined) < 30:
        return {"n_days": len(joined)}
    diff = joined["b"] - joined["a"]
    rainy_a = joined["a"] >= DRY_DAY_THRESHOLD_MM
    rainy_b = joined["b"] >= DRY_DAY_THRESHOLD_MM
    return {
        "n_days": int(len(joined)),
        "mean_a_mm": float(joined["a"].mean()),
        "mean_b_mm": float(joined["b"].mean()),
        "bias_mm_per_day": float(diff.mean()),
        "mae_mm": float(diff.abs().mean()),
        "rmse_mm": float(np.sqrt((diff**2).mean())),
        "correlation": float(joined["a"].corr(joined["b"])),
        "total_ratio": float(joined["b"].sum() / joined["a"].sum()) if joined["a"].sum() else np.nan,
        "rainy_days_a": int(rainy_a.sum()),
        "rainy_days_b": int(rainy_b.sum()),
        # Do the two agree about whether it rained at all on a given day?
        "rainy_day_agreement": float((rainy_a == rainy_b).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--districts", default=None, help="Comma-separated; default is a 20-district sample")
    parser.add_argument("--days", type=int, default=COMPARISON_DAYS)
    parser.add_argument("--with-excel", action="store_true", help="Also compare against the bundled station dataset")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    names = [d.strip() for d in args.districts.split(",")] if args.districts else DEFAULT_SAMPLE

    excel = None
    if args.with_excel:
        from src.data.cleaner import clean_dataset
        from src.data.district import get_all_districts_daily_series
        from src.data.loader import load_raw_dataset

        print("Loading the bundled station dataset (this takes a minute)...")
        excel = get_all_districts_daily_series(clean_dataset(load_raw_dataset()))

    rows = []
    for name in names:
        try:
            cfg = get_district_config(name)
        except KeyError:
            logger.warning("Skipping unknown district %s", name)
            continue

        power = power_series(cfg.district, cfg.state, cfg.latitude, cfg.longitude, cfg.elevation)
        power_rain = power.set_index("date_of_record")["rainfall"]

        archive = fetch_historical_weather(cfg.latitude, cfg.longitude, args.days)
        om = to_district_daily_schema(
            archive, cfg.district, cfg.state, cfg.latitude, cfg.longitude, cfg.elevation
        )
        om_rain = om.set_index("date_of_record")["rainfall"]

        row = {"district": cfg.district, **metrics(power_rain, om_rain)}

        if excel is not None:
            subset = excel[excel["district"] == cfg.district]
            if not subset.empty:
                excel_rain = subset.set_index("date_of_record")["rainfall"]
                excel_metrics = metrics(power_rain, excel_rain)
                row["excel_bias_mm_per_day"] = excel_metrics.get("bias_mm_per_day")
                row["excel_correlation"] = excel_metrics.get("correlation")
                row["excel_n_days"] = excel_metrics.get("n_days")
        rows.append(row)

    table = pd.DataFrame(rows)
    usable = table[table["n_days"] >= 30]

    print(f"\nNASA POWER (reference) vs Open-Meteo ERA5 archive, last {args.days} days")
    print(f"{'district':<22} {'POWER':>7} {'OM':>7} {'bias':>7} {'MAE':>6} {'corr':>6} {'ratio':>6} {'wet-agree':>10}")
    for _, r in usable.iterrows():
        print(
            f"{r['district']:<22} {r['mean_a_mm']:>7.2f} {r['mean_b_mm']:>7.2f} "
            f"{r['bias_mm_per_day']:>+7.2f} {r['mae_mm']:>6.2f} {r['correlation']:>6.2f} "
            f"{r['total_ratio']:>6.2f} {r['rainy_day_agreement']:>9.1%}"
        )

    print(
        f"\n  Across {len(usable)} districts: "
        f"median bias {usable['bias_mm_per_day'].median():+.2f} mm/day, "
        f"median correlation {usable['correlation'].median():.2f}, "
        f"median seasonal-total ratio {usable['total_ratio'].median():.2f}, "
        f"median rainy-day agreement {usable['rainy_day_agreement'].median():.1%}"
    )

    # The number that actually matters for this project: the detectors use
    # POWER for climatology and Open-Meteo only for the last ~3 days, so a
    # high correlation and a near-1.0 total ratio mean the splice is benign.
    worst = usable.reindex(usable["total_ratio"].sub(1.0).abs().sort_values(ascending=False).index).head(3)
    print("\n  Largest total-rainfall disagreements (these districts' onset/phase output is least reliable):")
    for _, r in worst.iterrows():
        print(
            f"    {r['district']:<22} Open-Meteo is {r['total_ratio']:.2f}x POWER "
            f"(corr {r['correlation']:.2f})"
        )

    if excel is not None and "excel_correlation" in table:
        excel_rows = table.dropna(subset=["excel_correlation"])
        print(
            f"\n  Bundled station Excel vs POWER, {len(excel_rows)} districts: "
            f"median bias {excel_rows['excel_bias_mm_per_day'].median():+.2f} mm/day, "
            f"median correlation {excel_rows['excel_correlation'].median():.2f}"
        )

    print(
        "\n  Interpretation: POWER supplies the multi-year climatology and Open-Meteo only the final\n"
        "  ~3 days, so what matters is that the two agree on the SHAPE of a district's rainfall\n"
        "  (correlation, rainy-day agreement) rather than matching mm for mm. A district with a\n"
        "  poor correlation here should have its onset and active/break output treated with\n"
        "  correspondingly less confidence."
    )


if __name__ == "__main__":
    main()
