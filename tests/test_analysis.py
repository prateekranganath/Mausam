"""The deterministic forecast analysis.

This is the part of /advisory that must always work, so it is tested
exhaustively and never touches the network.
"""
import json

import pytest

from src.llm import analysis as an
from src.llm.openrouter import find_unsupported_numbers

RELIABILITY = {"roc_auc": 0.754, "baseline_roc_auc": 0.692, "regressor_mae": 23.65, "baseline_mae": 23.40}


def days(values, start="2026-09-21"):
    import pandas as pd

    dates = pd.date_range(start, periods=len(values), freq="D")
    return [
        {"time": d.strftime("%Y-%m-%d"), "precipitation_sum": v, "precipitation_probability_max": 60.0}
        for d, v in zip(dates, values)
    ]


def forecast(
    district="Nagpur", risk="MODERATE", probability=0.5049, predicted=61.41, om_total=50.5,
    threshold=25.7, degenerate=False, sources_agree=True, diverges=False, diff=10.91,
    ml_insufficient=False, om_insufficient=False, daily=None, as_of="2026-09-20",
):
    return {
        "district": district, "state": "MH", "as_of_date": as_of,
        "ml_model": {"risk_level": risk, "rainfall_probability": probability, "predicted_rainfall_mm": predicted},
        "open_meteo_forecast": {
            "total_precipitation_sum_mm": om_total,
            "daily": daily if daily is not None else days([2.8, 4.5, 7.0, 17.0, 18.3, 0.8, 0.0]),
        },
        "agreement": {
            "threshold_mm": threshold, "threshold_degenerate": degenerate, "sources_agree": sources_agree,
            "magnitude_diverges": diverges, "difference_mm": diff,
            "ml_implies_insufficient": ml_insufficient, "open_meteo_implies_insufficient": om_insufficient,
        },
    }


def build(**kw):
    reliability = kw.pop("reliability", RELIABILITY)
    return an.build_analysis(forecast(**kw), reliability)


# --- risk: the model's, passed through, never re-rated --------------------------

@pytest.mark.parametrize("level", ["LOW", "MODERATE", "HIGH"])
def test_risk_level_is_the_forecast_models_own(level):
    """The LLM used to re-rate this and rated Nagpur LOW while the model said
    MODERATE. Nothing re-rates it now."""
    assert build(risk=level)["risk_level"] == level


def test_headline_leads_with_the_risk_and_the_numbers():
    headline = build(risk="MODERATE")["headline"]
    assert "Moderate risk" in headline and "Nagpur" in headline
    assert "50%" in headline and "61.4 mm" in headline


def test_a_degenerate_threshold_withholds_the_risk_level():
    """Where a dry week is normal, LOW carries no information and must not be
    passed on as reassurance."""
    result = build(risk="LOW", degenerate=True, threshold=0.0, predicted=5.0, om_total=0.0, diff=5.0)
    assert result["risk_level"] is None
    assert result["risk_meaningful"] is False
    assert "A dry week is normal" in result["headline"]
    assert "September" in result["headline"]
    assert not any("chance the week is unusually dry" in f for f in result["key_factors"])
    assert any("dry-risk score is not reported" in f for f in result["key_factors"])


def test_a_degenerate_district_is_told_to_plan_around_the_totals():
    actions = build(risk="LOW", degenerate=True)["actions"]
    assert any("plan around the forecast totals" in a for a in actions)
    assert not any("No unusually dry week is expected" in a for a in actions)


def test_an_unknown_risk_level_is_treated_as_not_meaningful():
    result = build(risk="WEIRD")
    assert result["risk_level"] is None and result["risk_meaningful"] is False


# --- confidence -----------------------------------------------------------------

def test_agreeing_sources_give_moderate_confidence_never_high():
    result = build(sources_agree=True, diverges=False)
    assert result["confidence"] == "moderate"
    assert any("agree" in r for r in result["confidence_reasons"])


def test_diverging_sources_give_low_confidence_and_quote_the_gap():
    result = build(sources_agree=False, diverges=True, diff=29.3, predicted=45.0, om_total=15.8)
    assert result["confidence"] == "low"
    assert any("29.3 mm" in r for r in result["confidence_reasons"])


def test_unknown_agreement_is_not_treated_as_disagreement():
    """`sources_agree` is None when it could not be determined."""
    result = build(sources_agree=None, diverges=False)
    assert result["confidence"] == "moderate"
    assert result["model_disagreement"] == []


def test_model_skill_is_stated_when_known_and_omitted_when_not():
    assert any("0.75" in r and "0.69" in r for r in build()["confidence_reasons"])
    assert not any("ROC-AUC" in r for r in build(reliability={})["confidence_reasons"])


def test_the_skill_statement_is_honest_about_being_modest():
    assert any("modest skill" in r for r in build()["confidence_reasons"])


# --- disagreement ---------------------------------------------------------------

def test_no_disagreement_section_when_the_sources_agree():
    assert build(sources_agree=True, diverges=False)["model_disagreement"] == []


def test_disagreement_states_both_totals_and_the_gap():
    result = build(sources_agree=False, diverges=True, diff=29.3, predicted=45.0, om_total=15.8)
    first = result["model_disagreement"][0]
    assert "45 mm" in first and "15.8 mm" in first and "29.3 mm apart" in first


def test_opposite_sides_of_the_threshold_are_called_out():
    result = build(diverges=True, sources_agree=False, ml_insufficient=False, om_insufficient=True)
    assert any("different sides" in line for line in result["model_disagreement"])


def test_same_side_of_the_threshold_is_not_called_out():
    result = build(diverges=True, sources_agree=False, ml_insufficient=True, om_insufficient=True)
    assert not any("different sides" in line for line in result["model_disagreement"])


def test_leans_on_open_meteo_when_the_regressor_does_not_beat_climatology():
    """This is measured on the held-out test split (MAE 23.65 vs 23.40), and it
    is the evidence-based reason to weight the NWP forecast for the amount."""
    result = build(diverges=True, sources_agree=False)
    assert any("lean on Open-Meteo" in line for line in result["model_disagreement"])


def test_does_not_lean_on_open_meteo_if_the_regressor_is_actually_better():
    better = {**RELIABILITY, "regressor_mae": 18.0, "baseline_mae": 23.4}
    result = build(diverges=True, sources_agree=False, reliability=better)
    assert not any("lean on Open-Meteo" in line for line in result["model_disagreement"])


# --- key factors ----------------------------------------------------------------

def test_key_factors_carry_both_forecasts():
    assert "61.4 mm" in build()["key_factors"][0] and "50.5 mm" in build()["key_factors"][0]


def test_wettest_day_and_rainy_day_count_are_derived_from_the_daily_data():
    factors = " ".join(build()["key_factors"])
    assert "Fri 25 Sep" in factors and "18.3 mm" in factors
    assert "5 of 7 days" in factors  # 2.8, 4.5, 7.0, 17.0 and 18.3 are all >= 2.5 mm


def test_a_dry_week_is_stated_as_such():
    result = build(daily=days([0.0] * 7), om_total=0.0)
    assert any("No rain is forecast on any of the next 7 days" in f for f in result["key_factors"])


def test_missing_daily_values_are_ignored_not_counted_as_dry():
    daily = days([5.0, 5.0, 5.0])
    daily[1]["precipitation_sum"] = None
    factors = " ".join(build(daily=daily)["key_factors"])
    assert "2 of 2 days" in factors


def test_heavy_rain_is_flagged_and_prompts_a_drainage_check():
    result = build(daily=days([2.0, 70.0, 5.0, 0.0, 0.0, 0.0, 0.0]))
    assert any("Heavy rain" in f and "64.5 mm" in f for f in result["key_factors"])
    assert any("drainage" in a for a in result["actions"])


def test_no_heavy_rain_no_drainage_advice():
    assert not any("drainage" in a for a in build()["actions"])


# --- actions --------------------------------------------------------------------

def test_high_risk_advises_prioritising_irrigation():
    assert any("irrigation" in a and "unusually dry" in a for a in build(risk="HIGH")["actions"])


def test_moderate_risk_advises_readiness():
    assert any("irrigation options ready" in a for a in build(risk="MODERATE")["actions"])


def test_low_risk_advises_routine_monitoring():
    assert any("routine field monitoring" in a for a in build(risk="LOW")["actions"])


def test_disagreeing_sources_prompt_a_recheck():
    assert any("Re-check the forecast" in a for a in build(diverges=True, sources_agree=False)["actions"])


def test_every_analysis_ends_with_the_kvk_pointer_and_not_official_caveat():
    for kw in ({}, {"risk": "HIGH"}, {"degenerate": True}, {"diverges": True, "sources_agree": False}):
        assert "KVK" in build(**kw)["actions"][-1]
        assert "not an official forecast" in build(**kw)["actions"][-1]


# --- the property that makes it safe --------------------------------------------

@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"risk": "HIGH", "probability": 0.81},
        {"degenerate": True, "threshold": 0.0, "predicted": 5.0, "om_total": 0.0, "diff": 5.0},
        {"diverges": True, "sources_agree": False, "predicted": 45.0, "om_total": 15.8, "diff": 29.2},
        {"daily": days([2.0, 70.0, 5.0, 0.0, 0.0, 0.0, 0.0])},
    ],
)
def test_every_number_in_the_analysis_comes_from_the_input(kw):
    """The analysis is written by rules, not a model, so it cannot fabricate a
    figure. This is the same tripwire applied to the LLM, and it must find
    nothing here."""
    reliability = kw.pop("reliability", RELIABILITY)
    data = forecast(**kw)
    result = an.build_analysis(data, reliability)
    prose = " ".join(
        [result["headline"], *result["confidence_reasons"], *result["key_factors"],
         *result["model_disagreement"], *result["actions"]]
    )
    source = {"forecast": data, "reliability": reliability, "constants": [an.HEAVY_RAIN_MM, 2.5]}
    assert find_unsupported_numbers(prose, source) == []


# --- robustness -----------------------------------------------------------------

def test_a_sparse_forecast_degrades_instead_of_raising():
    result = an.build_analysis({"district": "X"}, None)
    assert result["headline"] and result["actions"]
    assert result["risk_level"] is None and result["risk_meaningful"] is False


def test_the_analysis_is_json_serialisable():
    json.dumps(build(diverges=True, sources_agree=False))


def test_the_analysis_declares_its_source():
    assert build()["source"] == "rules"


# --- what the LLM is given ------------------------------------------------------

def facts_for(**kw):
    data = forecast(**kw)
    return an.summary_facts(data, an.build_analysis(data, RELIABILITY))


def test_the_llm_is_given_the_analysis_own_statements_to_rewrite():
    """It rewrites correct sentences; it does not interpret numbers. Interpreting
    is what produced 'not expected to be unusually dry, though the chance is 54%'."""
    data = forecast()
    analysis = an.build_analysis(data, RELIABILITY)
    facts = an.summary_facts(data, analysis)
    assert facts["district"] == "Nagpur"
    assert facts["points"][0] == analysis["headline"]
    assert all(point in analysis["key_factors"] for point in facts["points"][1:4])


def test_points_are_capped_so_the_prompt_stays_small():
    assert len(facts_for(diverges=True, sources_agree=False)["points"]) <= 6


def test_the_points_carry_the_numbers_the_reader_sees_elsewhere():
    joined = " ".join(facts_for()["points"])
    assert "61.4 mm" in joined and "50.5 mm" in joined and "50%" in joined


def test_disagreement_reaches_the_llm_only_when_the_sources_differ():
    assert not any("apart" in p for p in facts_for()["points"])
    assert any("apart" in p for p in facts_for(diverges=True, sources_agree=False)["points"])


def test_a_degenerate_district_passes_the_dry_week_is_normal_statement():
    points = facts_for(risk="LOW", degenerate=True, threshold=0.0, predicted=5.0, om_total=0.0, diff=5.0)["points"]
    assert "dry week is normal" in points[0].lower()
    assert not any("chance the week is unusually dry" in p for p in points)


def test_the_llm_is_never_shown_raw_field_names():
    """The very first prompt sent the whole forecast and the model then quoted
    `predicted_rainfall_mm` and `threshold_degenerate: false` to farmers."""
    blob = json.dumps(facts_for(diverges=True, sources_agree=False))
    for raw in ("predicted_rainfall_mm", "threshold_degenerate", "rainfall_probability", "magnitude_diverges", "roc_auc"):
        assert raw not in blob


def test_every_number_the_llm_may_quote_is_in_the_points():
    """So the unsupported-number tripwire has something exact to check against."""
    facts = facts_for()
    assert find_unsupported_numbers("Nagpur may see 61.4 mm and Open-Meteo 50.5 mm.", facts) == []
    assert find_unsupported_numbers("A 999 mm flood.", facts) == ["999"]


def test_the_facts_are_json_serialisable():
    json.dumps(facts_for())
