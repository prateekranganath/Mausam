"""Objects loaded once at startup and shared read-only across requests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.ml.predict import RainfallRiskPredictor


def servable_districts(predictor: RainfallRiskPredictor) -> tuple[frozenset[str], dict[str, str]]:
    """(servable, {excluded district: reason}).

    A district is servable only if the model learned local statistics for it
    in EVERY calendar month. The metadata's district list is everything in the
    data, but the preprocessor is fit on the training window only, so:
      - a district whose data starts after that window (Raisen, Vidisha: from
        2023-07-05) has no climatology or threshold at all;
      - a district missing some calendar month in training (Bathinda: July)
        would, for that month, silently fall back to district-wide averages.
    Either way the forecast would be a confident-looking number with no local
    basis, so those districts are not served.
    """
    listed = set(predictor.metadata.get("districts", []))
    with_threshold = set(predictor.preprocessor.risk_threshold_table_.index.get_level_values("district"))

    # climatology_table_ holds observed (district, month) rows only, with no fallback filling
    observed: dict[str, set[int]] = {}
    for district, month in predictor.preprocessor.climatology_table_.index:
        observed.setdefault(district, set()).add(int(month))

    servable: set[str] = set()
    excluded: dict[str, str] = {}
    for district in sorted(listed):
        months = observed.get(district, set())
        if district not in with_threshold or not months:
            excluded[district] = "no training data: its records begin after the model's training window"
        elif len(months) < 12:
            missing = sorted(set(range(1, 13)) - months)
            excluded[district] = f"incomplete training data: no observations for calendar month(s) {missing}"
        else:
            servable.add(district)
    return frozenset(servable), excluded


@dataclass
class ServiceState:
    predictor: RainfallRiskPredictor
    model_source: str
    trained_districts: frozenset[str]
    excluded_districts: dict[str, str] = field(default_factory=dict)
    eval_results: Optional[dict[str, Any]] = None
    # (district, as_of_date) -> (unix_ts, advisory, unsupported_numbers)
    advisory_cache: dict = field(default_factory=dict)
