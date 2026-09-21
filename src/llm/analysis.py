"""The forecast analysis: everything the advisory says that is NOT prose.

WHY THIS EXISTS. /advisory used to ask a free-tier LLM for the whole answer:
a risk rating, a confidence score, key factors, source disagreement and
actions, as one strict six-field JSON object. Measured on 2026-09-21 that was
slow (11s to 49s), flaky (a single 503 from an overloaded provider returned
nothing at all) and inconsistent (it rated Nagpur LOW while the forecast
model said MODERATE, and returned confidence 0.0 for Pune). Asking one free
model to do six jobs, several of them judgements the code already knows the
answer to, is what made it fragile.

So the split is now:

  this module   risk, confidence, key factors, disagreement, actions - all
                derived from the forecast numbers by rules. Instant, always
                available, reproducible, and every number in it is a number
                from the input by construction, so it cannot fabricate one.
  the LLM       one optional 2-3 sentence plain-language paragraph over the
                result (src/llm/openrouter.py). If it is down, slow or wrong,
                the analysis is unaffected.

This is the same separation the crop advisory and the Telegram alert already
use: the substance is deterministic and attributable, and prose is garnish.

The risk level is the FORECAST MODEL'S, passed through. The LLM no longer
re-rates it, which is what let the two disagree.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

from src.config import DRY_DAY_THRESHOLD_MM

# IMD's lower bound for a "heavy rain" day. Used only to flag days on which
# drainage, rather than drought, is the concern.
HEAVY_RAIN_MM = 64.5

_RISK_TITLE = {"LOW": "Low", "MODERATE": "Moderate", "HIGH": "High"}


def _mm(value: float | None) -> str:
    """Rainfall for prose: one decimal, trailing zero dropped (61.4, 50.5, 25)."""
    if value is None:
        return "n/a"
    text = f"{float(value):.1f}"
    return text[:-2] if text.endswith(".0") else text


def _percent(fraction: float | None) -> str:
    return "n/a" if fraction is None else f"{round(float(fraction) * 100)}"


def _month_name(iso: str | None) -> str:
    try:
        return dt.date.fromisoformat(str(iso)[:10]).strftime("%B")
    except ValueError:
        return "this time of year"


def _day_label(iso: str) -> str:
    try:
        return dt.date.fromisoformat(str(iso)[:10]).strftime("%a %d %b")
    except ValueError:
        return str(iso)


def _daily(forecast: dict[str, Any]) -> list[dict[str, Any]]:
    days = (forecast.get("open_meteo_forecast") or {}).get("daily") or []
    return [d for d in days if d.get("precipitation_sum") is not None]


def _diverges(agreement: dict[str, Any]) -> bool:
    """The two sources disagree materially. `sources_agree` is None when it
    could not be determined, which is not a disagreement."""
    return agreement.get("magnitude_diverges") is True or agreement.get("sources_agree") is False


def build_analysis(forecast: dict[str, Any], reliability: dict[str, Any] | None = None) -> dict[str, Any]:
    """The analysis for one district's forecast.

    `forecast` is a ForecastResponse as a dict. `reliability` is optional
    held-out-test evidence: roc_auc, baseline_roc_auc, regressor_mae and
    baseline_mae. Anything missing simply omits the sentence that needed it.
    """
    model = forecast.get("ml_model") or {}
    open_meteo = forecast.get("open_meteo_forecast") or {}
    agreement = forecast.get("agreement") or {}
    reliability = reliability or {}

    district = forecast.get("district", "this district")
    month = _month_name(forecast.get("as_of_date"))

    predicted = model.get("predicted_rainfall_mm")
    om_total = open_meteo.get("total_precipitation_sum_mm")
    threshold = agreement.get("threshold_mm")
    probability = model.get("rainfall_probability")
    level = str(model.get("risk_level") or "").upper() or None
    if level not in _RISK_TITLE:
        level = None

    degenerate = bool(agreement.get("threshold_degenerate"))
    # Where the threshold is ~0 mm the "insufficient rainfall" label cannot
    # fire, so a LOW carries no information. It is withheld, not passed on.
    risk_meaningful = level is not None and not degenerate
    diverges = _diverges(agreement)
    gap = abs(float(agreement["difference_mm"])) if agreement.get("difference_mm") is not None else None

    days = _daily(forecast)
    wettest = max(days, key=lambda d: d["precipitation_sum"]) if days else None
    rainy = [d for d in days if d["precipitation_sum"] >= DRY_DAY_THRESHOLD_MM]
    heavy = [d for d in days if d["precipitation_sum"] >= HEAVY_RAIN_MM]

    # ---- headline ---------------------------------------------------------
    if risk_meaningful:
        headline = (
            f"{_RISK_TITLE[level]} risk of an unusually dry week in {district}: "
            f"about {_percent(probability)}% chance, with {_mm(predicted)} mm expected."
        )
    else:
        headline = (
            f"A dry week is normal in {district} in {month}: about {_mm(predicted)} mm "
            "expected over the next 7 days."
            if degenerate
            else f"About {_mm(predicted)} mm of rain is expected in {district} over the next 7 days."
        )

    # ---- confidence -------------------------------------------------------
    confidence_reasons: list[str] = []
    if diverges:
        confidence = "low"
        confidence_reasons.append(
            f"The model ({_mm(predicted)} mm) and Open-Meteo ({_mm(om_total)} mm) differ"
            + (f" by {_mm(gap)} mm." if gap is not None else ".")
        )
    else:
        # Never "high": the pooled model is hackathon-grade, and stating
        # otherwise would overstate what the held-out numbers support.
        confidence = "moderate"
        if agreement.get("sources_agree") is True:
            confidence_reasons.append("The model and Open-Meteo agree.")
    roc, base_roc = reliability.get("roc_auc"), reliability.get("baseline_roc_auc")
    if roc is not None and base_roc is not None:
        confidence_reasons.append(
            f"On held-out data the model's ROC-AUC is {roc:.2f} against {base_roc:.2f} for a "
            "climatology-only baseline: modest skill, and it varies by district."
        )
    if degenerate:
        confidence_reasons.append("The dry-risk score is not meaningful here at this time of year.")

    # ---- key factors ------------------------------------------------------
    key_factors = [
        f"The trained model expects {_mm(predicted)} mm over the next 7 days; Open-Meteo forecasts {_mm(om_total)} mm."
    ]
    if risk_meaningful:
        # "meaning under X mm of rain in total" rather than a bare "below X mm":
        # the terse form was repeatedly rewritten by the LLM as "less than X mm
        # expected", confusing the dry-week THRESHOLD with the expected rainfall
        # (Hyderabad, live: "less than 23 mm expected ... the model expects 54 mm").
        key_factors.append(
            f"There is a {_percent(probability)}% chance the week is unusually dry, meaning under "
            f"{_mm(threshold)} mm of rain in total for {district} in {month}."
        )
    elif degenerate:
        key_factors.append(
            f"A dry week is normal for {month} here (threshold {_mm(threshold)} mm), so the "
            "dry-risk score is not reported."
        )
    if days:
        if wettest["precipitation_sum"] > 0:
            key_factors.append(
                f"Wettest forecast day: {_day_label(wettest['time'])} with {_mm(wettest['precipitation_sum'])} mm."
            )
            key_factors.append(
                f"{len(rainy)} of {len(days)} days have at least {_mm(DRY_DAY_THRESHOLD_MM)} mm of rain forecast."
            )
        else:
            key_factors.append("No rain is forecast on any of the next 7 days.")
    if heavy:
        key_factors.append(
            f"Heavy rain ({_mm(HEAVY_RAIN_MM)} mm or more) is forecast on "
            + ", ".join(_day_label(d["time"]) for d in heavy)
            + "."
        )

    # ---- disagreement -----------------------------------------------------
    model_disagreement: list[str] = []
    if diverges:
        model_disagreement.append(
            f"The model says {_mm(predicted)} mm and Open-Meteo says {_mm(om_total)} mm"
            + (f", {_mm(gap)} mm apart." if gap is not None else ".")
        )
        if (
            agreement.get("ml_implies_insufficient") is not None
            and agreement.get("open_meteo_implies_insufficient") is not None
            and agreement["ml_implies_insufficient"] != agreement["open_meteo_implies_insufficient"]
        ):
            model_disagreement.append(
                f"They land on different sides of the {_mm(threshold)} mm dry-week threshold."
            )
        mae, base_mae = reliability.get("regressor_mae"), reliability.get("baseline_mae")
        if mae is not None and base_mae is not None and mae >= base_mae:
            # Evidence, not opinion: this is measured on the held-out test
            # split, and it is the reason to lean on the NWP forecast.
            model_disagreement.append(
                f"On held-out test data the model's rainfall-amount estimate is not more accurate than a "
                f"climatology baseline (average error {_mm(mae)} mm against {_mm(base_mae)} mm), so lean on "
                "Open-Meteo for the amount and treat both as uncertain."
            )

    # ---- actions ----------------------------------------------------------
    actions: list[str] = []
    if degenerate:
        actions.append(
            f"A dry week is normal here in {month}: plan around the forecast totals rather than the dry-risk score."
        )
    elif level == "HIGH":
        actions.append(
            "Plan for an unusually dry week: prioritise irrigation for water-critical fields and postpone "
            "non-essential sowing or top-dressing."
        )
    elif level == "MODERATE":
        actions.append("Keep irrigation options ready and check soil moisture before sowing or top-dressing this week.")
    elif level == "LOW":
        actions.append("No unusually dry week is expected; routine field monitoring is enough.")
    if heavy:
        actions.append("Check field drainage before the heavy-rain day.")
    if diverges:
        actions.append("Re-check the forecast in 2 to 3 days, since the two sources disagree.")
    actions.append("Confirm with your local Krishi Vigyan Kendra (KVK) before acting; this is an estimate, not an official forecast.")

    return {
        "source": "rules",
        "headline": headline,
        "risk_level": level if risk_meaningful else None,
        "risk_meaningful": risk_meaningful,
        "confidence": confidence,
        "confidence_reasons": confidence_reasons,
        "key_factors": key_factors,
        "model_disagreement": model_disagreement,
        "actions": actions,
    }


def summary_facts(forecast: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    """What the LLM is given: the analysis's own verified sentences, to REWRITE.

    The first version handed the model raw, rounded numbers and asked it to
    interpret them. Measured live on 2026-09-21 that produced fluent text that
    was sometimes plainly wrong, in ways the numbers-only tripwire cannot see:

      Nagpur  "The week is not expected to be unusually dry, though the chance
              of an unusually dry week is 54%."   (54% is more likely than not)
      Jaipur  "The lowest forecast day is Mon 28 Sep at 4.8 mm."
              (it was the WETTEST day)

    A small free model is unreliable at deciding what numbers mean. It is far
    more reliable at rephrasing a sentence that is already correct. So it is
    given the analysis's own statements as `points` and asked only to say them
    in simpler words. The meaning was decided by rules; the model contributes
    wording, and nothing else.

    Only the points that carry the message are passed: the headline, the top
    key factors, and the disagreement statements if the sources differ. The
    rest stay in the analysis for the reader.
    """
    points = [
        analysis["headline"],
        *analysis["key_factors"][:3],
        *analysis["model_disagreement"][:2],
    ]
    return {"district": forecast.get("district"), "points": points}
