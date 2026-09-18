import json

import pytest
from pydantic import ValidationError

from src.llm import openrouter
from src.llm.openrouter import Advisory, OpenRouterError, find_unsupported_numbers, generate_advisory

SOURCE = {
    "as_of_date": "2026-09-18",
    "ml_model": {"rainfall_probability": 0.3198, "predicted_rainfall_mm": 36.49},
    "open_meteo_forecast": {
        "total_precipitation_sum_mm": 109.4,
        "daily": [{"time": "2026-09-19"}, {"time": "2026-09-25"}],
    },
    "agreement": {"threshold_mm": 10.49},
}


def _advisory(**overrides) -> Advisory:
    base = dict(
        forecast_summary="ok",
        rainfall_risk="LOW",
        confidence=0.6,
        key_factors=[],
        model_disagreement=[],
        advisory=[],
    )
    base.update(overrides)
    return Advisory(**base)


def _valid_json(**overrides) -> str:
    return _advisory(**overrides).model_dump_json()


# --- schema ---------------------------------------------------------------

def test_confidence_must_be_between_0_and_1():
    with pytest.raises(ValidationError):
        _advisory(confidence=1.4)
    with pytest.raises(ValidationError):
        _advisory(confidence=-0.1)


def test_extra_fields_are_rejected():
    payload = json.loads(_valid_json())
    payload["invented_field"] = "x"
    with pytest.raises(ValidationError):
        Advisory.model_validate(payload)


def test_invalid_risk_level_rejected():
    with pytest.raises(ValidationError):
        _advisory(rainfall_risk="SEVERE")


def test_json_schema_avoids_min_max_keywords_strict_modes_reject():
    schema = json.dumps(Advisory.model_json_schema())
    assert "minimum" not in schema and "maximum" not in schema
    assert '"additionalProperties": false' in schema


# --- number tripwire --------------------------------------------------------

def test_quoted_and_rounded_input_numbers_are_supported():
    adv = _advisory(forecast_summary="Open-Meteo expects 109.4 mm, roughly 109 mm, and the model 36.49 mm.")
    assert find_unsupported_numbers(adv, SOURCE) == []


def test_probability_restated_as_percent_is_supported():
    # Regression: real Nemotron output wrote 0.3198 as "31.98%" and was flagged.
    adv = _advisory(forecast_summary="There is a 31.98% chance of an unusually dry week.")
    assert find_unsupported_numbers(adv, SOURCE) == []


def test_fabricated_number_is_flagged():
    adv = _advisory(forecast_summary="Expect 250 mm of rain.")
    assert find_unsupported_numbers(adv, SOURCE) == ["250"]


def test_date_components_from_iso_dates_are_supported():
    # Regression: a signed regex read "-19" in 2026-09-19 as negative, so
    # a narrative saying "Sep 19-25" was wrongly flagged as fabricated.
    adv = _advisory(forecast_summary="Between Sep 19 and Sep 25, 2026 conditions stay wet.")
    assert find_unsupported_numbers(adv, SOURCE) == []


def test_small_integers_like_seven_day_are_ignored():
    adv = _advisory(forecast_summary="Over the next 7 days, 3 factors matter.")
    assert find_unsupported_numbers(adv, SOURCE) == []


def test_each_unsupported_number_is_reported_once():
    adv = _advisory(forecast_summary="About 777 mm", key_factors=["777 mm again"])
    assert find_unsupported_numbers(adv, SOURCE) == ["777"]


def test_numbers_in_all_prose_fields_are_checked():
    adv = _advisory(advisory=["Store 555 litres"], model_disagreement=["gap of 444 mm"], key_factors=["666"])
    assert set(find_unsupported_numbers(adv, SOURCE)) == {"555", "444", "666"}


# --- generate_advisory -----------------------------------------------------------

@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(openrouter, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openrouter, "OPENROUTER_MODEL", "test/model")
    monkeypatch.setattr(openrouter.time, "sleep", lambda _s: None)


def test_unconfigured_raises_clear_error(monkeypatch):
    monkeypatch.setattr(openrouter, "OPENROUTER_API_KEY", None)
    monkeypatch.setattr(openrouter, "OPENROUTER_MODEL", None)
    with pytest.raises(OpenRouterError, match="not configured"):
        generate_advisory(SOURCE)


def test_key_alone_is_not_enough_model_also_required(monkeypatch):
    monkeypatch.setattr(openrouter, "OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(openrouter, "OPENROUTER_MODEL", "")
    assert openrouter.is_configured() is False


def test_valid_response_returns_advisory(configured, monkeypatch):
    monkeypatch.setattr(openrouter, "_post", lambda messages: _valid_json(forecast_summary="Wet week, 109.4 mm."))
    adv, unsupported = generate_advisory(SOURCE)
    assert adv.forecast_summary.startswith("Wet week")
    assert unsupported == []


def test_markdown_fenced_json_is_accepted(configured, monkeypatch):
    monkeypatch.setattr(openrouter, "_post", lambda messages: f"```json\n{_valid_json()}\n```")
    adv, _ = generate_advisory(SOURCE)
    assert adv.rainfall_risk == "LOW"


def test_invalid_then_valid_retries_once(configured, monkeypatch):
    replies = iter(["this is not json", _valid_json()])
    calls = []

    def fake_post(messages):
        calls.append(1)
        return next(replies)

    monkeypatch.setattr(openrouter, "_post", fake_post)
    adv, _ = generate_advisory(SOURCE)
    assert len(calls) == 2
    assert adv.rainfall_risk == "LOW"


def test_repeated_invalid_output_raises(configured, monkeypatch):
    monkeypatch.setattr(openrouter, "_post", lambda messages: json.dumps({"forecast_summary": "missing fields"}))
    with pytest.raises(OpenRouterError, match="failed after"):
        generate_advisory(SOURCE)


def test_transport_error_is_retried_then_raised(configured, monkeypatch):
    import httpx

    calls = []

    def boom(messages):
        calls.append(1)
        raise httpx.ConnectError("connection reset")

    monkeypatch.setattr(openrouter, "_post", boom)
    with pytest.raises(OpenRouterError):
        generate_advisory(SOURCE)
    assert len(calls) == openrouter.MAX_ATTEMPTS


def _status_error(code: int, text: str = "err"):
    import httpx

    req = httpx.Request("POST", openrouter.API_URL)
    return httpx.HTTPStatusError("x", request=req, response=httpx.Response(code, text=text, request=req))


def _count_calls(monkeypatch, exc):
    calls = []

    def boom(messages):
        calls.append(1)
        raise exc

    monkeypatch.setattr(openrouter, "_post", boom)
    return calls


def test_timeout_is_not_retried(configured, monkeypatch):
    import httpx

    calls = _count_calls(monkeypatch, httpx.ReadTimeout("slow"))
    with pytest.raises(OpenRouterError, match="timed out"):
        generate_advisory(SOURCE)
    assert len(calls) == 1  # retrying a slow provider would only double the client's wait


def test_403_blocked_model_is_not_retried_and_message_is_surfaced(configured, monkeypatch):
    # what thinkingmachines/inkling:free actually returns to a plain API client
    calls = _count_calls(monkeypatch, _status_error(403, "only available on agentic harnesses"))
    with pytest.raises(OpenRouterError, match="agentic harnesses"):
        generate_advisory(SOURCE)
    assert len(calls) == 1


@pytest.mark.parametrize("code", [401, 404])
def test_other_4xx_are_permanent(configured, monkeypatch, code):
    calls = _count_calls(monkeypatch, _status_error(code))
    with pytest.raises(OpenRouterError):
        generate_advisory(SOURCE)
    assert len(calls) == 1


@pytest.mark.parametrize("code", [429, 500, 503])
def test_rate_limit_and_5xx_are_retried(configured, monkeypatch, code):
    calls = _count_calls(monkeypatch, _status_error(code))
    with pytest.raises(OpenRouterError, match="failed after"):
        generate_advisory(SOURCE)
    assert len(calls) == openrouter.MAX_ATTEMPTS


def _fake_post(monkeypatch, json_body, calls=None):
    import httpx

    def fake(url, **kwargs):
        if calls is not None:
            calls.append(1)
        return httpx.Response(200, json=json_body, request=httpx.Request("POST", url))

    monkeypatch.setattr(openrouter.httpx, "post", fake)


def test_http_200_with_embedded_overload_error_is_retryable(configured, monkeypatch):
    # Real response from nvidia/nemotron-3-super-120b-a12b:free: HTTP 200, no "choices".
    body = {"id": "gen-1", "error": {"message": "Upstream error from Nvidia: Service temporarily overloaded", "code": 503}}
    calls: list = []
    _fake_post(monkeypatch, body, calls)
    with pytest.raises(OpenRouterError, match="overloaded") as info:
        generate_advisory(SOURCE)
    assert len(calls) == openrouter.MAX_ATTEMPTS  # transient, so it was retried
    assert "failed after" in str(info.value)


def test_http_200_with_embedded_403_is_permanent_and_not_retried(configured, monkeypatch):
    body = {"error": {"message": "only available on agentic harnesses", "code": 403}}
    calls: list = []
    _fake_post(monkeypatch, body, calls)
    with pytest.raises(OpenRouterError, match="agentic harnesses"):
        generate_advisory(SOURCE)
    assert len(calls) == 1


def test_post_returns_content_from_choices(configured, monkeypatch):
    _fake_post(monkeypatch, {"choices": [{"message": {"content": "hello"}}]})
    assert openrouter._post([{"role": "user", "content": "x"}]) == "hello"


def test_system_prompt_states_exact_output_format():
    p = openrouter.SYSTEM_PROMPT
    for key in ("forecast_summary", "rainfall_risk", "confidence", "key_factors", "model_disagreement", "advisory"):
        assert f'"{key}"' in p
    assert "Do NOT echo" in p


def test_system_prompt_covers_degenerate_threshold():
    p = openrouter.SYSTEM_PROMPT
    assert "threshold_degenerate" in p
    assert "does NOT mean rain is expected" in p


def test_system_prompt_forbids_inventing_numbers_and_requires_reporting_divergence():
    p = openrouter.SYSTEM_PROMPT
    assert "Never invent" in p
    assert "magnitude_diverges" in p
    assert "EVEN IF both sources land on the same side" in p
