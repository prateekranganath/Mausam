"""The sowing-window gate.

Regression for a bug found by looking at the rendered dashboard: on 20
September the engine told an unsown rice farmer "Conditions are suitable for
sowing". The sowing rules checked onset and risk but never the calendar, so
they kept saying "go" long after the kharif window had closed.
"""
import pandas as pd
import pytest

from src.advisory import engine

POST_ONSET = {"status": "post_onset", "anomaly_days": 3}
LOW_RISK = {
    "ml_model": {"risk_level": "LOW", "predicted_rainfall_mm": 61.0},
    "open_meteo_forecast": {"total_precipitation_sum_mm": 50.0},
    "agreement": {"threshold_degenerate": False},
}


def advise(crop="rice_transplanted", as_of="2026-06-20", onset=POST_ONSET, forecast=LOW_RISK, **kw):
    return engine.advise(crop, as_of=as_of, onset=onset, forecast=forecast, **kw)


def ids(result):
    return {r["rule_id"] for r in result["recommendations"]}


def test_sowing_is_called_suitable_inside_the_window():
    assert "SOW_GO_AHEAD" in ids(advise(as_of="2026-06-20"))


def test_sowing_is_not_called_suitable_after_the_window_closes():
    """The exact case from the screenshot: rice, unsown, 20 September."""
    result = advise(as_of="2026-09-20")
    assert "SOW_GO_AHEAD" not in ids(result)
    assert "SOW_WINDOW_CLOSED" in ids(result)


def test_the_closed_window_message_says_so_plainly():
    result = advise(as_of="2026-09-20")
    action = next(r["action"] for r in result["recommendations"] if r["rule_id"] == "SOW_WINDOW_CLOSED")
    assert "window" in action and "passed" in action
    assert "KVK" in action


@pytest.mark.parametrize(
    "crop,month,open_expected",
    [
        ("maize", 5, False),   # before it opens
        ("maize", 6, True),    # first month
        ("maize", 7, True),    # last month
        ("maize", 8, False),   # first month after closing
        ("cotton", 5, True),   # cotton opens a month earlier than the cereals
        ("rice_transplanted", 8, True),  # transplanted rice runs to August
    ],
)
def test_window_boundaries(crop, month, open_expected):
    result = advise(crop=crop, as_of=f"2026-{month:02d}-15")
    assert ("SOW_GO_AHEAD" in ids(result)) is open_expected
    assert ("SOW_WINDOW_CLOSED" in ids(result)) is (not open_expected)


def test_a_window_that_has_not_opened_yet_still_waits_for_onset_rather_than_saying_closed():
    """In May, before onset, the right advice is 'wait for the rains'. It must
    not be reworded as 'the window has passed'."""
    result = advise(crop="maize", as_of="2026-05-20", onset={"status": "pre_onset"})
    assert "SOW_WAIT_FOR_ONSET" in ids(result)
    assert "SOW_WINDOW_CLOSED" not in ids(result)


def test_every_sowing_go_rule_is_gated_on_the_window():
    """Otherwise a late farmer is told to delay sowing for a week, or switch
    crop, or that onset is only 'likely' - all advice premised on a sowing
    that should not happen at all."""
    gated = {"SOW_ONSET_ONLY_LIKELY", "SOW_GO_AHEAD", "SOW_DELAY_HIGH_DRY_RISK", "SOW_SWITCH_TO_SHORT_DURATION"}
    for rule in engine.load_rules()["rules"]:
        if rule["rule_id"] in gated:
            fields = {c["field"] for c in rule["conditions"]}
            assert "sowing_window_open" in fields, rule["rule_id"]


def test_high_dry_risk_after_the_window_says_closed_not_delay():
    high = {**LOW_RISK, "ml_model": {"risk_level": "HIGH", "predicted_rainfall_mm": 5.0}}
    result = advise(as_of="2026-09-20", forecast=high)
    assert "SOW_DELAY_HIGH_DRY_RISK" not in ids(result)
    assert "SOW_WINDOW_CLOSED" in ids(result)


def test_an_unknown_date_never_satisfies_the_window_condition():
    """If the window state cannot be determined, the sowing rules must stay
    silent rather than guess in either direction."""
    crop = engine.get_crop("maize")
    context = engine.build_context(crop, None, onset=POST_ONSET, forecast=LOW_RISK, today=None)
    assert context["sowing_window_open"] is None
    result = engine.evaluate(context, is_sown=False)
    assert not {r.rule_id for r in result} & {"SOW_GO_AHEAD", "SOW_WINDOW_CLOSED"}


def test_a_crop_without_a_declared_window_never_fires_the_window_rules():
    crop = {**engine.get_crop("maize")}
    del crop["sowing_window"]
    context = engine.build_context(crop, None, onset=POST_ONSET, forecast=LOW_RISK, today=pd.Timestamp("2026-06-15"))
    assert context["sowing_window_open"] is None


def test_the_window_rules_do_not_apply_once_the_crop_is_sown():
    """A sown crop is past the sowing decision; the window is irrelevant."""
    result = advise(as_of="2026-09-20", sowing_date="2026-06-25")
    assert not any(r.startswith("SOW_") for r in ids(result))


def test_every_crop_declares_a_valid_window():
    for key, crop in engine.load_rules()["crops"].items():
        window = crop["sowing_window"]
        assert 1 <= window["opens_month"] <= window["closes_month"] <= 12, key
