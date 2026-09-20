"""Objects loaded once at startup and shared read-only across requests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.ml.predict import RainfallRiskPredictor

# A district's history changes at most once a day (POWER publishes daily,
# and the Open-Meteo bridge is hourly-cached upstream), so an hour is
# already conservative.
HISTORY_CACHE_TTL_SECONDS = 3600


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
    # district -> (unix_ts, daily history frame). The monsoon detectors need
    # 11 years of daily data per district; re-reading and re-splicing that on
    # every request would dominate the response time, and the underlying
    # POWER shard only gains a row once a day anyway.
    history_cache: dict = field(default_factory=dict)

    def district_history(self, cfg) -> Any:
        """Cached daily history for one district, loaded on first use.

        Deliberately lazy rather than loaded at startup: warming all 316
        districts would mean 316 upstream calls before the service could
        answer anything, and most deployments touch a handful.
        """
        import time

        from src.data.history import district_history

        cached = self.history_cache.get(cfg.district)
        if cached and time.time() - cached[0] < HISTORY_CACHE_TTL_SECONDS:
            return cached[1]
        frame = district_history(cfg)
        self.history_cache[cfg.district] = (time.time(), frame)
        return frame
