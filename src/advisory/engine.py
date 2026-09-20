"""Deterministic crop advisory engine.

WHY THIS IS NOT THE LLM'S JOB. The project already has an OpenRouter
advisory layer, and it would have been far less code to extend that
prompt with "and give crop advice". It is deliberately not done that way.
Agronomic recommendations are the part of this system a farmer would
actually act on, and they need three properties an LLM cannot provide:

  reproducible  the same inputs give the same advice, every time
  attributable  every recommendation names the rule_id that produced it,
                so it can be audited, corrected and version-controlled
  available     advice still returns when the free-tier LLM is down

So the agronomy lives in crop_rules.json as data, this module evaluates
it as pure functions, and the LLM's only remaining job is to phrase the
result. That is the same separation routes.py already applies to the
forecast itself: the numbers never depend on the narrative.

HOW A RULE FIRES. Each rule lists conditions over a flat context of
named signals (risk level, water balance, onset status, monsoon phase,
soil wetness, growth stage and so on). All of a rule's conditions must
hold. Nothing is inferred, and a signal that is MISSING never satisfies a
condition — an absent value means "we do not know", which must not be
allowed to read as "the threshold was met".
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

RULES_PATH = Path(__file__).resolve().parent / "crop_rules.json"

# Severity ordering, most urgent first, used to sort what the farmer sees.
SEVERITY_ORDER = {"high": 0, "medium": 1, "info": 2}


class CropAdvisoryError(ValueError):
    pass


@dataclass(frozen=True)
class Recommendation:
    rule_id: str
    category: str
    severity: str
    action: str
    rationale: str
    triggered_by: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def load_rules(path: str | None = None) -> dict:
    with open(path or RULES_PATH, encoding="utf-8") as f:
        return json.load(f)


def list_crops() -> list[dict]:
    rules = load_rules()
    return [
        {
            "key": key,
            "display_name": crop["display_name"],
            "season": crop["season"],
            "duration_days": crop["duration_days"],
            "seasonal_water_mm": crop["seasonal_water_mm"],
            "notes": crop.get("notes"),
        }
        for key, crop in sorted(rules["crops"].items())
    ]


def get_crop(crop: str) -> dict:
    crops = load_rules()["crops"]
    key = crop.strip().lower().replace("-", "_").replace(" ", "_")
    if key in crops:
        return {**crops[key], "key": key}
    # Accept a bare "rice" as the transplanted default, since that is what
    # most of India means by it, but only when it is unambiguous.
    aliases = {"rice": "rice_transplanted", "tur": "pigeonpea", "arhar": "pigeonpea",
               "pearl_millet": "bajra", "finger_millet": "ragi"}
    if key in aliases:
        return {**crops[aliases[key]], "key": aliases[key]}
    raise CropAdvisoryError(f"Unknown crop '{crop}'. Known crops: {sorted(crops)}.")


def growth_stage(crop: dict, days_since_sowing: int) -> dict | None:
    """The stage `days_since_sowing` falls in, or None if the crop is not
    yet sown or has passed maturity."""
    if days_since_sowing < 0:
        return None
    for stage in crop["stages"]:
        if stage["start_day"] <= days_since_sowing <= stage["end_day"]:
            return stage
    return None


def _compare(actual: Any, op: str, expected: Any) -> bool:
    """Evaluate one condition.

    A missing signal returns False for EVERY operator, including `lt`.
    That asymmetry is deliberate: `water_balance_mm < 0` must not fire
    when the water balance could not be computed, because "unknown" is
    not "in deficit".
    """
    if actual is None:
        return False
    try:
        if op == "in":
            return actual in expected
        if op == "eq":
            return actual == expected
        if op == "lt":
            return float(actual) < float(expected)
        if op == "gte":
            return float(actual) >= float(expected)
    except (TypeError, ValueError):
        return False
    raise CropAdvisoryError(f"Unknown condition operator '{op}' in crop_rules.json.")


def build_context(
    crop: dict,
    stage: dict | None,
    forecast: dict | None = None,
    onset: dict | None = None,
    phase: dict | None = None,
    soil: dict | None = None,
) -> dict[str, Any]:
    """Flatten every available signal into the namespace rules match on.

    Each source is optional; whatever is unavailable stays None and simply
    prevents the rules that depend on it from firing.
    """
    forecast = forecast or {}
    model = forecast.get("ml_model") or {}
    open_meteo = forecast.get("open_meteo_forecast") or {}
    agreement = forecast.get("agreement") or {}

    predicted_rainfall = model.get("predicted_rainfall_mm")
    # Prefer Open-Meteo's own forward forecast for the water balance: it is
    # a direct precipitation forecast, whereas the regressor is an estimate
    # of the same quantity that does not beat climatology pooled (see the
    # README's metrics). Fall back to the model only if Open-Meteo is absent.
    expected_rainfall = open_meteo.get("total_precipitation_sum_mm")
    if expected_rainfall is None:
        expected_rainfall = predicted_rainfall

    weekly_requirement = stage["weekly_water_mm"] if stage else None
    water_balance = (
        expected_rainfall - weekly_requirement
        if expected_rainfall is not None and weekly_requirement is not None
        else None
    )

    return {
        "crop_key": crop["key"],
        "crop_duration_days": crop["duration_days"],
        "stage_name": stage["name"] if stage else None,
        "stage_drought_sensitivity": stage["drought_sensitivity"] if stage else None,
        "stage_weekly_water_mm": weekly_requirement,
        "risk_level": model.get("risk_level"),
        "rainfall_probability": model.get("rainfall_probability"),
        "predicted_rainfall_mm": expected_rainfall,
        "water_balance_mm": water_balance,
        "threshold_degenerate": agreement.get("threshold_degenerate"),
        "onset_status": onset.get("status") if onset else None,
        "onset_anomaly_days": onset.get("anomaly_days") if onset else None,
        "monsoon_phase": phase.get("monsoon_phase") if phase else None,
        "days_in_current_phase": phase.get("days_in_current_phase") if phase else None,
        "consecutive_dry_days": (soil or {}).get("consecutive_dry_days"),
        "soil_wetness_root_zone": (soil or {}).get("soil_wetness_root_zone"),
        "soil_wetness_surface": (soil or {}).get("soil_wetness_surface"),
    }


def evaluate(context: dict[str, Any], is_sown: bool) -> list[Recommendation]:
    """Every rule whose conditions all hold, most severe first."""
    recommendations: list[Recommendation] = []
    for rule in load_rules()["rules"]:
        # Sowing rules are about a decision already taken once the crop is
        # in the ground; offering them afterwards would be noise.
        if rule.get("applies_when_unsown") and is_sown:
            continue
        conditions = rule["conditions"]
        if not all(_compare(context.get(c["field"]), c["op"], c["value"]) for c in conditions):
            continue
        recommendations.append(
            Recommendation(
                rule_id=rule["rule_id"],
                category=rule["category"],
                severity=rule["severity"],
                action=rule["action"],
                rationale=rule["rationale"],
                # Echo back the actual values that made this rule fire, so a
                # recommendation can be checked rather than merely believed.
                triggered_by={c["field"]: context.get(c["field"]) for c in conditions},
            )
        )
    recommendations.sort(key=lambda r: SEVERITY_ORDER.get(r.severity, 99))
    return recommendations


def _suppress_degenerate_dry_risk(
    recommendations: list[Recommendation], context: dict
) -> tuple[list[Recommendation], str | None]:
    """Drop dry-risk advice where the risk label cannot mean anything.

    In most districts outside the monsoon the lower-tercile threshold
    collapses toward 0mm, so "insufficient rainfall" can never fire and a
    LOW risk_level carries no information (the README's per-month table:
    265 of 314 districts in January). Advice premised on that label would
    be confidently derived from a number that means nothing.
    """
    if not context.get("threshold_degenerate"):
        return recommendations, None
    dry_risk_rules = {"SOW_DELAY_HIGH_DRY_RISK", "SOW_GO_AHEAD", "NUTRIENT_APPLY_BEFORE_RAIN"}
    kept = [r for r in recommendations if r.rule_id not in dry_risk_rules]
    if len(kept) == len(recommendations):
        return recommendations, None
    return kept, (
        "Rainfall-risk-based advice was withheld: this district and month has a degenerate "
        "risk threshold (normal rainfall is already near zero), so the risk label carries no "
        "information here."
    )


def _ensure_not_empty(
    recommendations: list[Recommendation],
    context: dict,
    stage: dict | None,
    past_maturity: bool,
) -> list[Recommendation]:
    """Never return an empty advisory.

    No rule firing is a real and common outcome — it means conditions are
    unremarkable — but an advisory endpoint that answers with an empty
    list reads as broken rather than as reassuring. So the "nothing needs
    doing" case is stated explicitly, and still carries a rule_id so it is
    as traceable as any other recommendation.
    """
    if recommendations:
        return recommendations

    if past_maturity:
        rule_id, action = (
            "STATUS_PAST_MATURITY",
            "This crop is past its expected duration from the sowing date given. If it has not "
            "been harvested, check whether the sowing date is correct.",
        )
    elif stage is None and context.get("onset_status") is None:
        rule_id, action = (
            "STATUS_INSUFFICIENT_SIGNALS",
            "Not enough information to give specific advice. Supply a sowing_date for "
            "stage-specific guidance, or retry when monsoon context is available.",
        )
    else:
        balance = context.get("water_balance_mm")
        # Only quantify a surplus worth mentioning. A balance of 0.4mm
        # rounds to "surplus 0mm", which reads like a bug.
        detail = ""
        if balance is not None and balance >= 1:
            detail = (
                f" Forecast rainfall covers the crop's requirement for this stage "
                f"(about {balance:.0f}mm more than it needs)."
            )
        elif balance is not None and balance >= 0:
            detail = " Forecast rainfall just covers the crop's requirement for this stage."
        rule_id, action = (
            "STATUS_NO_ACTION_NEEDED",
            "No specific action is indicated this week. Conditions are within the normal range "
            "for this crop and stage." + detail,
        )

    return [
        Recommendation(
            rule_id=rule_id,
            category="status",
            severity="info",
            action=action,
            rationale="No rule in crop_rules.json matched the current conditions.",
            triggered_by={
                k: context.get(k)
                for k in ("risk_level", "water_balance_mm", "monsoon_phase", "onset_status")
            },
        )
    ]


def advise(
    crop_name: str,
    sowing_date: str | None = None,
    as_of: str | None = None,
    forecast: dict | None = None,
    onset: dict | None = None,
    phase: dict | None = None,
    soil: dict | None = None,
) -> dict:
    """The full crop advisory for one district and crop.

    `sowing_date` may be None, meaning "not sown yet" — which is the case
    the sowing rules exist for, and the most decision-relevant moment in
    the season.
    """
    crop = get_crop(crop_name)
    today = pd.Timestamp(as_of) if as_of else pd.Timestamp(date.today())

    days_since_sowing = None
    stage = None
    if sowing_date:
        sown = pd.Timestamp(sowing_date)
        if sown > today:
            raise CropAdvisoryError(
                f"sowing_date {sown.date()} is in the future relative to as_of {today.date()}."
            )
        days_since_sowing = int((today - sown).days)
        stage = growth_stage(crop, days_since_sowing)

    context = build_context(crop, stage, forecast=forecast, onset=onset, phase=phase, soil=soil)
    recommendations = evaluate(context, is_sown=sowing_date is not None)
    recommendations, suppression_note = _suppress_degenerate_dry_risk(recommendations, context)
    recommendations = _ensure_not_empty(recommendations, context, stage, past_maturity=(
        days_since_sowing is not None and days_since_sowing > crop["duration_days"]
    ))

    rules = load_rules()
    past_maturity = (
        days_since_sowing is not None and days_since_sowing > crop["duration_days"]
    )

    return {
        "crop": crop["display_name"],
        "crop_key": crop["key"],
        "sowing_date": sowing_date,
        "as_of_date": str(today.date()),
        "days_since_sowing": days_since_sowing,
        "growth_stage": stage["name"] if stage else None,
        "stage_drought_sensitivity": stage["drought_sensitivity"] if stage else None,
        "stage_weekly_water_requirement_mm": stage["weekly_water_mm"] if stage else None,
        "expected_rainfall_next_7_days_mm": context["predicted_rainfall_mm"],
        "water_balance_mm": (
            round(context["water_balance_mm"], 1) if context["water_balance_mm"] is not None else None
        ),
        "past_maturity": past_maturity,
        "signals_used": {k: v for k, v in context.items() if v is not None},
        "signals_unavailable": sorted(k for k, v in context.items() if v is None),
        "recommendations": [r.to_dict() for r in recommendations],
        "suppression_note": suppression_note,
        "rules_version": rules["version"],
        "water_requirement_basis": rules["_about"]["water_requirement_basis"],
        "stage_basis": rules["_about"]["stage_basis"],
        "disclaimer": rules["_about"]["disclaimer"],
        "sources": rules["_about"]["sources"],
    }
