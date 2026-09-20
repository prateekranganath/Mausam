"""Alert composition and delivery.

Every test here mocks the network. Nothing in this file may send a real
Telegram message: a suite that messages a live chat on every run is its
own kind of bug.
"""
import pytest
import requests

from src.alerts import telegram
from src.alerts.compose import DISCLAIMER, MAX_TELEGRAM_CHARS, compose_alert


def forecast(district="Nagpur", state="MH", risk="MODERATE", probability=0.5,
             predicted=61.0, om_total=50.0, degenerate=False, agree=True):
    return {
        "district": district, "state": state, "as_of_date": "2026-09-20",
        "ml_model": {"risk_level": risk, "rainfall_probability": probability,
                     "predicted_rainfall_mm": predicted},
        "open_meteo_forecast": {"total_precipitation_sum_mm": om_total},
        "agreement": {"threshold_degenerate": degenerate, "sources_agree": agree,
                      "magnitude_diverges": not agree},
    }


# --- the honesty rules the composer must enforce -------------------------------

def test_degenerate_threshold_replaces_the_risk_line():
    """Where a dry week is normal the risk label carries no information, so
    printing 'LOW risk' would be technically true and useless."""
    message = compose_alert(forecast(risk="LOW", degenerate=True))
    assert "not meaningful here" in message
    assert "RISK: LOW" not in message


def test_a_meaningful_threshold_keeps_the_risk_line():
    message = compose_alert(forecast(risk="HIGH", probability=0.8, degenerate=False))
    assert "RISK: HIGH" in message
    assert "80% chance" in message


def test_onset_likely_is_never_worded_as_settled():
    """The difference between 'rain arrived' and 'rain arrived and stayed' is
    the whole reason onset has two states; it must survive into the message."""
    message = compose_alert(
        forecast(), onset={"status": "onset_likely", "onset_date": "2026-06-10"}
    )
    assert "NOT yet confirmed" in message
    assert "arrived 10 Jun" not in message


def test_confirmed_onset_reports_the_date_and_anomaly():
    message = compose_alert(
        forecast(),
        onset={"status": "onset_confirmed", "onset_date": "2026-06-25",
               "anomaly_label": "8 days late"},
    )
    assert "arrived 25 Jun 2026" in message
    assert "8 days late" in message


def test_pre_onset_says_the_monsoon_has_not_arrived():
    assert "has not arrived" in compose_alert(forecast(), onset={"status": "pre_onset"})


def test_break_spell_is_emphasised():
    message = compose_alert(forecast(), phase={"monsoon_phase": "break", "days_in_current_phase": 9})
    assert "BREAK SPELL" in message
    assert "9 days" in message


def test_out_of_season_phase_is_omitted_entirely():
    message = compose_alert(forecast(), phase={"monsoon_phase": "not_applicable"})
    assert "Current phase" not in message


def test_the_disclaimer_is_unconditional():
    assert DISCLAIMER in compose_alert(forecast())
    assert DISCLAIMER in compose_alert(forecast(degenerate=True), onset={"status": "pre_onset"})


def test_source_disagreement_is_flagged():
    assert "differs from our estimate" in compose_alert(forecast(agree=False))


# --- formatting robustness -----------------------------------------------------

def test_district_names_with_parentheses_survive_intact():
    """Four served districts contain parentheses. This is why the message is
    plain text rather than MarkdownV2, which would need them escaped."""
    for name in ["Raipur (CT)", "Raipur (MP)", "Cuddalore (TN)", "Cuddalore (PY)"]:
        assert name in compose_alert(forecast(district=name))


def test_no_markdown_control_characters_are_emitted():
    message = compose_alert(forecast(), crop={
        "crop": "Rice (transplanted)", "growth_stage": "panicle_initiation",
        "stage_drought_sensitivity": "critical", "water_balance_mm": -20.0,
        "stage_weekly_water_requirement_mm": 65,
        "recommendations": [{"severity": "high", "category": "irrigation", "action": "Irrigate."}],
    })
    assert "*" not in message
    assert "_" not in message


def test_a_malformed_date_does_not_lose_the_alert():
    data = forecast()
    data["as_of_date"] = "not-a-date"
    assert "Mausam alert" in compose_alert(data)


def test_an_overlong_message_is_truncated_but_keeps_the_disclaimer():
    crop = {
        "crop": "Rice", "growth_stage": "flowering", "stage_drought_sensitivity": "critical",
        "recommendations": [
            {"severity": "high", "category": "irrigation", "action": "x" * 3000} for _ in range(3)
        ],
    }
    message = compose_alert(forecast(), crop=crop)
    assert len(message) <= MAX_TELEGRAM_CHARS
    assert message.endswith(DISCLAIMER)


def test_missing_sections_degrade_rather_than_fail():
    """A partial upstream outage must still produce a usable alert."""
    message = compose_alert(forecast())
    assert "Mausam alert" in message
    assert "RISK:" in message


def test_status_only_advice_is_dropped_when_the_balance_line_says_the_same():
    crop = {
        "crop": "Cotton", "growth_stage": "squaring_flowering",
        "stage_drought_sensitivity": "critical",
        "water_balance_mm": 25.0, "stage_weekly_water_requirement_mm": 50,
        "recommendations": [{"severity": "info", "category": "status",
                             "action": "No specific action is indicated this week."}],
    }
    message = compose_alert(forecast(), crop=crop)
    assert "forecast covers it" in message
    assert "No specific action" not in message


def test_status_advice_is_kept_when_there_is_no_balance_line():
    crop = {
        "crop": "Maize",
        "recommendations": [{"severity": "info", "category": "status",
                             "action": "No specific action is indicated this week."}],
    }
    assert "No specific action" in compose_alert(forecast(), crop=crop)


def test_a_water_deficit_is_quantified():
    crop = {
        "crop": "Maize", "growth_stage": "tasselling_silking",
        "stage_drought_sensitivity": "critical",
        "water_balance_mm": -45.0, "stage_weekly_water_requirement_mm": 55,
        "recommendations": [],
    }
    assert "short by 45 mm" in compose_alert(forecast(), crop=crop)


# --- the sender ----------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(telegram, "TELEGRAM_BOT_TOKEN", "123:FAKE")
    monkeypatch.setattr(telegram, "TELEGRAM_CHAT_ID", "999")


def install_post(monkeypatch, payload=None, exc=None, status=200):
    calls = []

    def fake_post(url, data=None, timeout=None):
        calls.append({"url": url, "data": data})
        if exc is not None:
            raise exc
        return FakeResponse(payload, status)

    monkeypatch.setattr(telegram.requests, "post", fake_post)
    return calls


def test_send_reports_the_message_id(configured, monkeypatch):
    install_post(monkeypatch, {"ok": True, "result": {"message_id": 42}})
    result = telegram.send_message("hello")
    assert result.sent is True
    assert result.message_id == 42


def test_send_targets_the_configured_chat(configured, monkeypatch):
    calls = install_post(monkeypatch, {"ok": True, "result": {"message_id": 1}})
    telegram.send_message("hello")
    assert calls[0]["data"]["chat_id"] == "999"


def test_send_surfaces_telegrams_own_description(configured, monkeypatch):
    """Verified against the live API: Telegram answers HTTP 401 with a JSON
    body carrying a 'description'. 'Unauthorized' and 'chat not found' are
    different problems and the operator needs to know which one happened."""
    install_post(monkeypatch, {"ok": False, "error_code": 401, "description": "Unauthorized"}, status=401)
    result = telegram.send_message("hello")
    assert result.sent is False
    assert result.error == "Unauthorized"


def test_send_surfaces_a_bad_chat_id(configured, monkeypatch):
    install_post(monkeypatch, {"ok": False, "description": "Bad Request: chat not found"}, status=400)
    assert "chat not found" in telegram.send_message("hello").error


def test_a_send_is_never_retried(configured, monkeypatch):
    """sendMessage is not idempotent. Retrying after a timeout can deliver the
    same drought warning twice, which is worse than not delivering it."""
    calls = install_post(monkeypatch, exc=requests.Timeout("timed out"))
    result = telegram.send_message("hello")
    assert len(calls) == 1
    assert result.sent is False
    assert "may or may not have been delivered" in result.error


def test_a_non_json_reply_does_not_crash(configured, monkeypatch):
    install_post(monkeypatch, None, status=502)
    assert telegram.send_message("hello").sent is False


def test_sending_unconfigured_raises_rather_than_failing_silently(monkeypatch):
    monkeypatch.setattr(telegram, "TELEGRAM_BOT_TOKEN", None)
    with pytest.raises(telegram.TelegramNotConfigured):
        telegram.send_message("hello")


def test_a_token_without_a_chat_id_counts_as_unconfigured(monkeypatch):
    """Telegram's error for an empty chat id is unhelpful; refuse up front."""
    monkeypatch.setattr(telegram, "TELEGRAM_BOT_TOKEN", "123:FAKE")
    monkeypatch.setattr(telegram, "TELEGRAM_CHAT_ID", None)
    assert telegram.is_configured() is False
    assert "TELEGRAM_CHAT_ID" in telegram.configuration_hint()


def test_the_hint_names_whichever_half_is_missing(monkeypatch):
    monkeypatch.setattr(telegram, "TELEGRAM_BOT_TOKEN", None)
    monkeypatch.setattr(telegram, "TELEGRAM_CHAT_ID", "999")
    hint = telegram.configuration_hint()
    assert "BotFather" in hint
    assert "TELEGRAM_CHAT_ID" not in hint
