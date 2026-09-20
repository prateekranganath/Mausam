import json

import pytest

from src.advisory import engine


def forecast(risk="LOW", predicted=60.0, open_meteo=None, degenerate=False):
    return {
        "ml_model": {"risk_level": risk, "rainfall_probability": 0.2, "predicted_rainfall_mm": predicted},
        "open_meteo_forecast": {"total_precipitation_sum_mm": open_meteo},
        "agreement": {"threshold_degenerate": degenerate},
    }


def advise(**kw):
    kw.setdefault("crop_name", "rice_transplanted")
    kw.setdefault("as_of", "2026-08-01")
    return engine.advise(**kw)


def rule_ids(result):
    return {r["rule_id"] for r in result["recommendations"]}


# --- the rules file is data, and must stay valid -------------------------------

def test_rules_file_is_well_formed():
    rules = engine.load_rules()
    assert rules["version"]
    for rule in rules["rules"]:
        assert rule["rule_id"] and rule["action"] and rule["rationale"]
        assert rule["severity"] in engine.SEVERITY_ORDER
        for condition in rule["conditions"]:
            assert condition["op"] in {"in", "eq", "lt", "gte"}


def test_rule_ids_are_unique():
    ids = [r["rule_id"] for r in engine.load_rules()["rules"]]
    assert len(ids) == len(set(ids))


def test_every_crop_has_contiguous_stages_covering_its_duration():
    for key, crop in engine.load_rules()["crops"].items():
        stages = crop["stages"]
        assert stages[0]["start_day"] == 0, key
        assert stages[-1]["end_day"] == crop["duration_days"], key
        for earlier, later in zip(stages, stages[1:]):
            assert later["start_day"] == earlier["end_day"] + 1, f"{key}: gap or overlap"


def test_every_stage_declares_a_drought_sensitivity():
    valid = {"low", "medium", "high", "critical"}
    for key, crop in engine.load_rules()["crops"].items():
        for stage in crop["stages"]:
            assert stage["drought_sensitivity"] in valid, key
            assert stage["weekly_water_mm"] > 0, key


# --- crop and stage resolution -------------------------------------------------

def test_bare_rice_resolves_to_transplanted():
    assert engine.get_crop("rice")["key"] == "rice_transplanted"


def test_crop_lookup_is_forgiving_about_separators():
    assert engine.get_crop("Rice-Transplanted")["key"] == "rice_transplanted"
    assert engine.get_crop("pearl millet")["key"] == "bajra"


def test_unknown_crop_names_the_alternatives():
    with pytest.raises(engine.CropAdvisoryError, match="maize"):
        engine.get_crop("quinoa")


def test_growth_stage_tracks_days_since_sowing():
    crop = engine.get_crop("maize")
    assert engine.growth_stage(crop, 0)["name"] == "establishment"
    assert engine.growth_stage(crop, 55)["name"] == "tasselling_silking"
    assert engine.growth_stage(crop, 999) is None


def test_a_future_sowing_date_is_rejected():
    with pytest.raises(engine.CropAdvisoryError, match="future"):
        advise(sowing_date="2026-09-01", as_of="2026-08-01")


# --- the missing-signal asymmetry ----------------------------------------------

def test_a_missing_signal_never_satisfies_a_condition():
    """`water_balance_mm < 0` must not fire when the balance is unknown.
    "We don't know" is not "in deficit", and this is the difference between
    a cautious advisory and a confidently wrong one."""
    assert engine._compare(None, "lt", 0) is False
    assert engine._compare(None, "gte", 0) is False
    assert engine._compare(None, "in", ["x"]) is False
    assert engine._compare(None, "eq", "x") is False


def test_irrigation_advice_does_not_fire_without_a_water_balance():
    result = advise(sowing_date="2026-05-20")  # no forecast supplied at all
    assert "IRRIGATE_CRITICAL_STAGE_DEFICIT" not in rule_ids(result)


def test_unavailable_signals_are_reported_not_hidden():
    result = advise(sowing_date="2026-05-20")
    assert "monsoon_phase" in result["signals_unavailable"]
    assert "risk_level" in result["signals_unavailable"]


# --- rules firing --------------------------------------------------------------

def test_critical_stage_deficit_triggers_irrigation_advice():
    result = advise(
        crop_name="maize",
        sowing_date="2026-06-15",
        as_of="2026-08-05",  # day 51: tasselling/silking, critical
        forecast=forecast(risk="HIGH", open_meteo=5.0),
    )
    assert result["stage_drought_sensitivity"] == "critical"
    assert "IRRIGATE_CRITICAL_STAGE_DEFICIT" in rule_ids(result)


def test_a_surplus_does_not_trigger_irrigation_advice():
    result = advise(
        crop_name="maize", sowing_date="2026-06-15", as_of="2026-08-05",
        forecast=forecast(open_meteo=200.0),
    )
    assert "IRRIGATE_CRITICAL_STAGE_DEFICIT" not in rule_ids(result)


def test_break_spell_triggers_conservation_advice():
    result = advise(sowing_date="2026-06-20", phase={"monsoon_phase": "break", "days_in_current_phase": 8})
    assert "CONSERVE_BREAK_SPELL" in rule_ids(result)


def test_active_spell_triggers_drainage_advice():
    result = advise(sowing_date="2026-06-20", phase={"monsoon_phase": "active", "days_in_current_phase": 4})
    assert "DRAINAGE_ACTIVE_SPELL" in rule_ids(result)


def test_pre_onset_tells_an_unsown_farmer_to_wait():
    result = advise(crop_name="maize", onset={"status": "pre_onset"}, as_of="2026-05-20")
    assert "SOW_WAIT_FOR_ONSET" in rule_ids(result)


def test_onset_only_likely_gives_a_hedged_sowing_answer():
    """The onset_likely / onset_confirmed distinction has to survive all the
    way to the advice, or the honesty upstream is wasted."""
    result = advise(crop_name="maize", onset={"status": "onset_likely"}, as_of="2026-06-05")
    assert "SOW_ONSET_ONLY_LIKELY" in rule_ids(result)
    assert "SOW_GO_AHEAD" not in rule_ids(result)


def test_sowing_rules_are_suppressed_once_the_crop_is_sown():
    result = advise(sowing_date="2026-06-20", onset={"status": "pre_onset"})
    assert not any(r.startswith("SOW_") for r in rule_ids(result))


def test_a_very_late_onset_suggests_a_shorter_duration_crop():
    result = advise(
        crop_name="rice_transplanted",
        onset={"status": "onset_confirmed", "anomaly_days": 20},
        as_of="2026-07-01",
    )
    assert "SOW_SWITCH_TO_SHORT_DURATION" in rule_ids(result)


def test_a_short_duration_crop_is_not_told_to_switch():
    result = advise(
        crop_name="bajra",
        onset={"status": "onset_confirmed", "anomaly_days": 20},
        as_of="2026-07-01",
    )
    assert "SOW_SWITCH_TO_SHORT_DURATION" not in rule_ids(result)


def test_prolonged_dry_spell_is_flagged():
    result = advise(sowing_date="2026-06-20", soil={"consecutive_dry_days": 14})
    assert "DRY_SPELL_PROLONGED" in rule_ids(result)


def test_a_short_dry_spell_is_not_flagged():
    result = advise(sowing_date="2026-06-20", soil={"consecutive_dry_days": 4})
    assert "DRY_SPELL_PROLONGED" not in rule_ids(result)


def test_dry_root_zone_plus_deficit_triggers_irrigation():
    result = advise(
        sowing_date="2026-06-20",
        forecast=forecast(open_meteo=5.0),
        soil={"soil_wetness_root_zone": 0.2},
    )
    assert "IRRIGATE_DRY_ROOT_ZONE" in rule_ids(result)


def test_a_wet_root_zone_does_not_trigger_irrigation():
    result = advise(
        sowing_date="2026-06-20",
        forecast=forecast(open_meteo=5.0),
        soil={"soil_wetness_root_zone": 0.8},
    )
    assert "IRRIGATE_DRY_ROOT_ZONE" not in rule_ids(result)


def test_rain_at_maturity_triggers_harvest_advice():
    result = advise(
        crop_name="maize", sowing_date="2026-05-01", as_of="2026-08-14",  # day 105: maturity
        forecast=forecast(open_meteo=120.0),
    )
    assert result["growth_stage"] == "maturity"
    assert "HARVEST_WET_WEEK_AHEAD" in rule_ids(result)


# --- degenerate threshold suppression ------------------------------------------

def test_dry_risk_advice_is_withheld_where_the_threshold_is_degenerate():
    """In most districts outside the monsoon the lower-tercile threshold
    collapses to ~0mm, so the risk label carries no information. Advice built
    on it would be confidently derived from a meaningless number."""
    result = advise(
        crop_name="maize", onset={"status": "onset_confirmed"}, as_of="2026-06-10",
        forecast=forecast(risk="HIGH", degenerate=True),
    )
    assert "SOW_DELAY_HIGH_DRY_RISK" not in rule_ids(result)
    assert result["suppression_note"] is not None


def test_dry_risk_advice_is_kept_where_the_threshold_is_meaningful():
    result = advise(
        crop_name="maize", onset={"status": "onset_confirmed"}, as_of="2026-06-10",
        forecast=forecast(risk="HIGH", degenerate=False),
    )
    assert "SOW_DELAY_HIGH_DRY_RISK" in rule_ids(result)
    assert result["suppression_note"] is None


# --- output contract -----------------------------------------------------------

def test_an_advisory_is_never_empty():
    """No rule firing is a normal outcome, but an empty list reads as broken.

    Scenario chosen so nothing genuinely matches: a critical stage (so the
    nutrient rule is excluded) with a rainfall surplus (so the irrigation
    rules are excluded) and no monsoon-phase signal.
    """
    result = advise(
        sowing_date="2026-06-20", as_of="2026-08-29",  # day 70: panicle initiation, critical
        forecast=forecast(open_meteo=500.0),
    )
    assert result["stage_drought_sensitivity"] == "critical"
    assert result["recommendations"]
    assert rule_ids(result) == {"STATUS_NO_ACTION_NEEDED"}
    assert result["recommendations"][0]["severity"] == "info"


def test_recommendations_are_ordered_most_severe_first():
    result = advise(
        sowing_date="2026-06-20",
        forecast=forecast(risk="HIGH", open_meteo=2.0),
        phase={"monsoon_phase": "break"},
        soil={"consecutive_dry_days": 15, "soil_wetness_root_zone": 0.1},
    )
    severities = [engine.SEVERITY_ORDER[r["severity"]] for r in result["recommendations"]]
    assert severities == sorted(severities)


def test_every_recommendation_shows_what_triggered_it():
    """Attributability is the point of doing this deterministically."""
    result = advise(sowing_date="2026-06-20", phase={"monsoon_phase": "break"})
    for rec in result["recommendations"]:
        assert isinstance(rec["triggered_by"], dict)
        assert rec["rule_id"]


def test_open_meteo_rainfall_is_preferred_over_the_regressor():
    """The regressor does not beat climatology pooled; Open-Meteo is a direct
    precipitation forecast. When both exist, use the better one."""
    result = advise(sowing_date="2026-06-20", forecast=forecast(predicted=10.0, open_meteo=99.0))
    assert result["expected_rainfall_next_7_days_mm"] == 99.0


def test_the_model_regressor_is_used_when_open_meteo_is_absent():
    result = advise(sowing_date="2026-06-20", forecast=forecast(predicted=10.0, open_meteo=None))
    assert result["expected_rainfall_next_7_days_mm"] == 10.0


def test_past_maturity_is_reported():
    result = advise(crop_name="maize", sowing_date="2025-01-01", as_of="2026-08-01")
    assert result["past_maturity"] is True
    assert rule_ids(result) == {"STATUS_PAST_MATURITY"}


def test_the_whole_response_is_json_serialisable():
    result = advise(sowing_date="2026-06-20", forecast=forecast(open_meteo=5.0))
    json.dumps(result)
