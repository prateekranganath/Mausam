"""Cross-source agreement between our ML model and Open-Meteo's own forecast.

This is an agreement check against an independent forecast, NOT validation
against ground truth: Open-Meteo is itself a forecast, so agreement only
says two forecasters point the same way, and disagreement is a flag worth
a human/LLM look, not proof either is wrong. Real validation needs observed
rainfall for the forecast window, i.e. waiting a week and storing predictions.

Both quantities compared are "total mm over T+1..T+7" for the same place, so
they are directly comparable. The class comparison applies THIS model's own
learned (district, month) threshold to both totals, so "insufficient" means
the same thing on both sides. The ML side uses the regressor's mm against
that threshold; the classifier's probability/risk level is reported
separately by the caller and is deliberately not collapsed to a hard label
here (the classifier's default 0.5 cutoff is a poor operating point at ~20%
base rate, see README limitations).

Pure functions, no I/O, so this is unit-testable without model or network.
"""
from __future__ import annotations

import pandas as pd

# Sources are flagged as diverging when their 7-day totals differ by more
# than both an absolute floor and a fraction of the larger total. The floor
# stops dry-region noise (7mm vs 0mm) from being called a big disagreement.
DIVERGENCE_FLOOR_MM = 15.0
DIVERGENCE_RELATIVE = 0.5

# Below this, the district-month's lower-tercile 7-day rainfall is ~0mm: a dry
# week is normal there, so "rainfall < threshold" can essentially never be
# true and the insufficient-rainfall label is uninformative (README:
# Limitations). Surfaced explicitly so nothing downstream reads "LOW risk" as
# "rain is expected".
DEGENERATE_THRESHOLD_MM = 1.0

AGREEMENT_NOTE = (
    "Agreement check against an independent forecast, NOT ground-truth "
    "validation - Open-Meteo is itself a forecast. Both totals are compared "
    "against this model's own (district, month) 'insufficient rainfall' "
    "threshold so the class label means the same thing on both sides."
)


def lookup_threshold(threshold_table: pd.Series, district: str, month: int) -> float | None:
    """(district, month) risk threshold in mm from the preprocessor's
    MultiIndex table, or None if that combination is absent/NaN."""
    try:
        value = threshold_table.loc[(district, month)]
    except KeyError:
        return None
    if pd.isna(value):
        return None
    return float(value)


def compute_agreement(
    ml_predicted_mm: float,
    open_meteo_mm: float,
    threshold_mm: float | None,
) -> dict:
    """Compare the two sources' 7-day rainfall totals.

    `threshold_mm` is the district-month 'insufficient rainfall' cutoff; if it
    is None the class-level fields are None and only magnitude is compared.
    """
    difference = abs(ml_predicted_mm - open_meteo_mm)
    larger = max(ml_predicted_mm, open_meteo_mm)
    diverges = difference > max(DIVERGENCE_FLOOR_MM, DIVERGENCE_RELATIVE * larger)

    result: dict = {
        "ml_predicted_mm": round(ml_predicted_mm, 2),
        "open_meteo_forecast_mm": round(open_meteo_mm, 2),
        "difference_mm": round(difference, 2),
        "magnitude_diverges": bool(diverges),
        "threshold_mm": None if threshold_mm is None else round(threshold_mm, 2),
        "threshold_degenerate": threshold_mm is not None and threshold_mm < DEGENERATE_THRESHOLD_MM,
        "ml_implies_insufficient": None,
        "open_meteo_implies_insufficient": None,
        "sources_agree": None,
        "note": AGREEMENT_NOTE,
    }

    if threshold_mm is not None:
        ml_low = ml_predicted_mm < threshold_mm
        om_low = open_meteo_mm < threshold_mm
        result["ml_implies_insufficient"] = bool(ml_low)
        result["open_meteo_implies_insufficient"] = bool(om_low)
        result["sources_agree"] = bool(ml_low == om_low)

    return result
