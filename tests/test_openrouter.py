"""The OpenRouter client.

Every test here mocks the network. Several are regressions for failure modes
observed on the live free tier on 2026-09-21, and say so.
"""
import json

import httpx
import pytest

from src.llm import openrouter
from src.llm.openrouter import (
    OpenRouterError,
    Summary,
    _Rejected,
    find_unit_mismatches,
    find_unsupported_numbers,
    generate_summary,
)

FACTS = {
    "district": "Nagpur",
    "points": [
        "Moderate risk of an unusually dry week in Nagpur: about 50% chance, with 61.4 mm expected.",
        "The trained model expects 61.4 mm over the next 7 days; Open-Meteo forecasts 50.5 mm.",
    ],
}

GOOD = "Nagpur can expect about 61.4 mm this week and Open-Meteo shows 50.5 mm, so the two sources agree."


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(openrouter, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openrouter, "OPENROUTER_MODEL", "m/primary:free")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "m/second:free,m/third:free")
    openrouter.reset_cooldowns()
    yield
    openrouter.reset_cooldowns()


def script(monkeypatch, outcomes):
    """Replace the network call. `outcomes` maps model -> (content, finish) or
    an exception to raise. Returns the list of models actually called."""
    called = []

    def fake_call(model, messages, timeout):
        called.append(model)
        outcome = outcomes[model]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(openrouter, "_call", fake_call)
    return called


# --- configuration --------------------------------------------------------------

def test_unconfigured_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr(openrouter, "OPENROUTER_API_KEY", None)
    with pytest.raises(OpenRouterError, match="not configured"):
        generate_summary(FACTS)


def test_a_key_alone_is_enough_because_a_default_chain_exists(monkeypatch):
    """The old client also demanded OPENROUTER_MODEL. The default fallbacks
    make that unnecessary."""
    monkeypatch.setattr(openrouter, "OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(openrouter, "OPENROUTER_MODEL", "")
    monkeypatch.delenv("OPENROUTER_FALLBACK_MODELS", raising=False)
    assert openrouter.is_configured() is True


def test_no_key_means_unconfigured_even_with_a_model(monkeypatch):
    monkeypatch.setattr(openrouter, "OPENROUTER_API_KEY", None)
    monkeypatch.setattr(openrouter, "OPENROUTER_MODEL", "m/x")
    assert openrouter.is_configured() is False


def test_no_models_at_all_is_unconfigured(monkeypatch):
    monkeypatch.setattr(openrouter, "OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(openrouter, "OPENROUTER_MODEL", "")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "")
    assert openrouter.is_configured() is False


def test_the_chain_is_primary_first_then_fallbacks_without_duplicates(monkeypatch):
    monkeypatch.setattr(openrouter, "OPENROUTER_MODEL", "m/a")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "m/b, m/a ,m/c")
    assert openrouter.model_chain() == ["m/a", "m/b", "m/c"]


def test_an_empty_fallback_setting_disables_fallbacks(monkeypatch):
    monkeypatch.setattr(openrouter, "OPENROUTER_MODEL", "m/a")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "")
    assert openrouter.model_chain() == ["m/a"]


def test_the_default_chain_is_used_when_the_setting_is_absent(monkeypatch):
    monkeypatch.setattr(openrouter, "OPENROUTER_MODEL", "m/a")
    monkeypatch.delenv("OPENROUTER_FALLBACK_MODELS", raising=False)
    assert openrouter.model_chain() == ["m/a", *openrouter.DEFAULT_FALLBACK_MODELS]


# --- the fallback chain ---------------------------------------------------------

def test_the_primary_model_answers_and_the_rest_are_not_called(configured, monkeypatch):
    called = script(monkeypatch, {"m/primary:free": (GOOD, "stop")})
    result = generate_summary(FACTS)
    assert isinstance(result, Summary)
    assert result.model == "m/primary:free" and result.text == GOOD
    assert called == ["m/primary:free"]


def test_an_overloaded_primary_falls_through_to_the_next_model(configured, monkeypatch):
    """Regression. The old client retried the SAME model twice, 1s apart, so a
    single Nvidia 503 meant no answer at all (Bhopal, 2026-09-21)."""
    called = script(monkeypatch, {
        "m/primary:free": _Rejected("HTTP 503: Service temporarily overloaded"),
        "m/second:free": (GOOD, "stop"),
    })
    result = generate_summary(FACTS)
    assert result.model == "m/second:free"
    assert called == ["m/primary:free", "m/second:free"]


def test_a_model_that_just_failed_is_skipped_on_the_next_request(configured, monkeypatch):
    """Otherwise every request pays the failing model's latency again."""
    called = script(monkeypatch, {
        "m/primary:free": _Rejected("HTTP 503"),
        "m/second:free": (GOOD, "stop"),
    })
    generate_summary(FACTS)
    called.clear()
    generate_summary(FACTS)
    assert called == ["m/second:free"]


def test_when_every_model_is_cooling_down_they_are_tried_anyway(configured, monkeypatch):
    """Being locked out for a minute because all of them failed once helps nobody."""
    script(monkeypatch, {m: _Rejected("HTTP 503") for m in ("m/primary:free", "m/second:free", "m/third:free")})
    with pytest.raises(OpenRouterError):
        generate_summary(FACTS)
    called = script(monkeypatch, {m: (GOOD, "stop") for m in ("m/primary:free", "m/second:free", "m/third:free")})
    assert generate_summary(FACTS).model == "m/primary:free"
    assert called == ["m/primary:free"]


def test_a_success_clears_that_models_cooldown(configured, monkeypatch):
    """A model that failed once and then works must not stay penalised."""
    far_future = 10**12
    for model in ("m/primary:free", "m/second:free", "m/third:free"):
        openrouter._COOLDOWN[model] = far_future
    script(monkeypatch, {"m/primary:free": (GOOD, "stop")})
    generate_summary(FACTS)  # every model is cooling, so all are tried; the primary answers
    assert "m/primary:free" not in openrouter._COOLDOWN
    assert "m/second:free" in openrouter._COOLDOWN  # never reached, so untouched


def test_every_model_failing_raises_with_each_reason_listed(configured, monkeypatch):
    script(monkeypatch, {
        "m/primary:free": _Rejected("HTTP 503: overloaded"),
        "m/second:free": _Rejected("HTTP 429: rate limited"),
        "m/third:free": _Rejected("timed out after 25s"),
    })
    with pytest.raises(OpenRouterError) as info:
        generate_summary(FACTS)
    message = str(info.value)
    assert "m/primary: HTTP 503" in message and "m/second: HTTP 429" in message and "timed out" in message
    assert ":free" not in message  # the suffix is noise in an error line


def test_the_total_time_budget_is_enforced(configured, monkeypatch):
    """The budget is what bounds the user's wait; without it a chain of slow
    models could add up to minutes."""
    monkeypatch.setattr(openrouter, "TOTAL_BUDGET_SECONDS", 0.0)
    called = script(monkeypatch, {"m/primary:free": (GOOD, "stop")})
    with pytest.raises(OpenRouterError, match="time budget"):
        generate_summary(FACTS)
    assert called == []


def test_each_attempt_is_capped_by_the_remaining_budget(configured, monkeypatch):
    seen = []

    def fake_call(model, messages, timeout):
        seen.append(timeout)
        return GOOD, "stop"

    monkeypatch.setattr(openrouter, "_call", fake_call)
    monkeypatch.setattr(openrouter, "TOTAL_BUDGET_SECONDS", 10.0)
    monkeypatch.setattr(openrouter, "REQUEST_TIMEOUT_SECONDS", 90.0)
    generate_summary(FACTS)
    assert seen[0] <= 10.0


# --- rejecting replies that look like success but are not -----------------------

def test_a_truncated_reply_is_rejected_not_shown(configured, monkeypatch):
    """Regression. Nemotron returned 'In Nagpur the forecast indicates' and
    laguna returned 'The': reasoning models spent the token budget thinking and
    the visible answer was cut off. finish_reason 'length' says so."""
    script(monkeypatch, {
        "m/primary:free": ("In Nagpur the forecast indicates", "length"),
        "m/second:free": (GOOD, "stop"),
    })
    assert generate_summary(FACTS).model == "m/second:free"


def test_an_empty_reply_is_rejected(configured, monkeypatch):
    script(monkeypatch, {"m/primary:free": ("", "stop"), "m/second:free": (GOOD, "stop")})
    assert generate_summary(FACTS).model == "m/second:free"


@pytest.mark.parametrize(
    "leak",
    [
        "We need to produce 2 or 3 short plain sentences explaining the rainfall forecast to a farmer.",
        "The user wants a short note about Nagpur. Let me draft it now with 61.4 mm.",
        "Okay, so the forecast for Nagpur is 61.4 mm and I should mention Open-Meteo.",
        "Let me think about how to phrase the 61.4 mm figure for Nagpur.",
    ],
)
def test_leaked_chain_of_thought_never_reaches_the_reader(configured, monkeypatch, leak):
    """Regression. `openrouter/free` returned its reasoning as the answer. Text
    like this must never be shown to a farmer, even when it mentions the right
    district and the right numbers."""
    script(monkeypatch, {"m/primary:free": (leak, "stop"), "m/second:free": (GOOD, "stop")})
    assert generate_summary(FACTS).text == GOOD


def test_an_off_task_reply_is_rejected(configured, monkeypatch):
    """Regression. nex-n2.5-mini answered 'please share the forecast details'."""
    off_task = "I can help with that. Please share the rainfall forecast details or a screenshot and I will explain them."
    script(monkeypatch, {"m/primary:free": (off_task, "stop"), "m/second:free": (GOOD, "stop")})
    assert generate_summary(FACTS).model == "m/second:free"


def test_a_reply_that_only_cites_a_supplied_number_counts_as_on_topic(configured, monkeypatch):
    text = "About 61.4 mm of rain is expected this week, close to the 50.5 mm shown by the other source."
    script(monkeypatch, {"m/primary:free": (text, "stop")})
    assert generate_summary(FACTS).text == text


def test_a_reply_that_is_too_short_is_rejected(configured, monkeypatch):
    script(monkeypatch, {"m/primary:free": ("Nagpur: rain.", "stop"), "m/second:free": (GOOD, "stop")})
    assert generate_summary(FACTS).model == "m/second:free"


def test_a_runaway_reply_is_rejected(configured, monkeypatch):
    script(monkeypatch, {"m/primary:free": ("Nagpur " + "rain " * 400, "stop"), "m/second:free": (GOOD, "stop")})
    assert generate_summary(FACTS).model == "m/second:free"


def test_a_rejected_reply_puts_that_model_on_cooldown(configured, monkeypatch):
    script(monkeypatch, {"m/primary:free": ("", "stop"), "m/second:free": (GOOD, "stop")})
    generate_summary(FACTS)
    assert "m/primary:free" in openrouter._COOLDOWN


# --- cleaning -------------------------------------------------------------------

def test_a_think_block_is_stripped_and_the_answer_kept(configured, monkeypatch):
    script(monkeypatch, {"m/primary:free": (f"<think>hmm let me see</think>{GOOD}", "stop")})
    assert generate_summary(FACTS).text == GOOD


def test_markdown_bullets_quotes_and_newlines_are_flattened(configured, monkeypatch):
    raw = '"**Nagpur** can expect\n- about 61.4 mm this week,\n- and Open-Meteo shows 50.5 mm."'
    script(monkeypatch, {"m/primary:free": (raw, "stop")})
    text = generate_summary(FACTS).text
    assert "*" not in text and "\n" not in text and not text.startswith('"')
    assert "61.4 mm" in text and "50.5 mm" in text


def test_non_breaking_hyphens_are_normalised(configured, monkeypatch):
    """Seen in live output; it crashed a Windows console and is pointless in prose."""
    script(monkeypatch, {"m/primary:free": ("Nagpur can expect 61.4‑mm of rain and Open‑Meteo shows 50.5 mm.", "stop")})
    assert "‑" not in generate_summary(FACTS).text


# --- faithfulness: an unfaithful note is rejected, not shown with a warning -----

def test_a_fabricated_number_gets_the_reply_rejected(configured, monkeypatch):
    """The note's only value is being faithful, so one that cites a figure that
    is not in the input is refused and the next model is tried."""
    fabricated = "Nagpur can expect 61.4 mm and a 777 mm flood is possible this week."
    script(monkeypatch, {"m/primary:free": (fabricated, "stop"), "m/second:free": (GOOD, "stop")})
    result = generate_summary(FACTS)
    assert result.model == "m/second:free" and result.text == GOOD


def test_the_failure_that_motivated_the_unit_check_is_rejected(configured, monkeypatch):
    """Regression, live 2026-09-21 (laguna, Nagpur): a 54% chance became 'only
    about 54 mm of rain is expected against the typical 25.7 mm'. The number 54
    IS in the input, as a percentage, so a number-only check passed it."""
    facts = {
        "district": "Nagpur",
        "points": [
            "Moderate risk of an unusually dry week in Nagpur: about 54% chance, with 66.6 mm expected.",
            "There is a 54% chance the week is unusually dry: below 25.7 mm for Nagpur in September.",
            "The trained model expects 66.6 mm over the next 7 days; Open-Meteo forecasts 67.8 mm.",
        ],
    }
    bad = ("Nagpur faces a moderate risk of an unusually dry week with a 54% chance, as only about 54 mm of rain is "
           "expected against the typical 25.7 mm for September.")
    good = "Nagpur has a 54% chance of an unusually dry week, with the model expecting 66.6 mm and Open-Meteo 67.8 mm."
    script(monkeypatch, {"m/primary:free": (bad, "stop"), "m/second:free": (good, "stop")})
    result = generate_summary(facts)
    assert result.model == "m/second:free"
    assert find_unsupported_numbers(bad, facts) == []           # the number-only check really does miss it
    assert find_unit_mismatches(bad, facts) == ["54 mm"]        # and the unit-aware one catches it


def test_the_reason_for_a_faithfulness_rejection_is_reported(configured, monkeypatch):
    script(monkeypatch, {m: ("Nagpur may see 999 mm of rain and 61.4 mm too.", "stop") for m in ("m/primary:free", "m/second:free", "m/third:free")})
    with pytest.raises(OpenRouterError, match="not in the input"):
        generate_summary(FACTS)


def test_supported_numbers_pass(configured, monkeypatch):
    script(monkeypatch, {"m/primary:free": (GOOD, "stop")})
    assert generate_summary(FACTS).text == GOOD


def test_a_probability_restated_as_a_percentage_is_supported():
    assert find_unsupported_numbers("A 32% chance.", {"p": 0.3198}) == []


def test_small_integers_like_seven_day_are_ignored():
    assert find_unsupported_numbers("A 7-day outlook, 3 days wet.", {"x": 1.0}) == []


def test_dates_in_the_source_support_date_numbers_in_the_text():
    assert find_unsupported_numbers("Around Sep 25.", {"wettest_day": "Fri 25 Sep"}) == []


def test_each_unsupported_number_is_reported_once():
    assert find_unsupported_numbers("999 mm, then 999 mm again.", {"x": 1.0}) == ["999"]


# --- unit-aware check -----------------------------------------------------------

POINTS = {"points": ["A 54% chance of a dry week, with 66.6 mm expected and a 25.7 mm threshold. Wettest Fri 25 Sep with 22.2 mm."]}


def test_a_percentage_used_as_millimetres_is_a_mismatch():
    assert find_unit_mismatches("about 54 mm of rain", POINTS) == ["54 mm"]


def test_millimetres_used_as_a_percentage_is_a_mismatch():
    assert find_unit_mismatches("a 66.6% chance", POINTS) == ["66.6%"]


def test_matching_units_are_fine():
    assert find_unit_mismatches("a 54% chance and 66.6 mm, above the 25.7 mm threshold", POINTS) == []


def test_a_figure_with_no_unit_is_not_second_guessed():
    assert find_unit_mismatches("about 54 this week", POINTS) == []


def test_a_value_that_also_appears_unitless_is_compatible_with_any_unit():
    """'Fri 25 Sep' contains a bare 25; '25 mm' must not be flagged on that basis alone."""
    assert find_unit_mismatches("about 25 mm on the wettest day", {"points": ["Wettest Fri 25 Sep."]}) == []


def test_the_unit_check_is_case_insensitive():
    assert find_unit_mismatches("about 54 MM", POINTS) == ["54 mm"]


def test_a_value_not_in_the_input_at_all_is_left_to_the_other_check():
    assert find_unit_mismatches("about 999 mm", POINTS) == []


# --- the wire -------------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload, status=200, text=""):
        self._payload, self.status_code, self.text = payload, status, text or json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def install_post(monkeypatch, response=None, exc=None):
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        sent.update(url=url, body=json, headers=headers, timeout=timeout)
        if exc is not None:
            raise exc
        return response

    monkeypatch.setattr(openrouter.httpx, "post", fake_post)
    return sent


def ok(content=GOOD, finish="stop"):
    return FakeResponse({"choices": [{"message": {"content": content}, "finish_reason": finish}]})


def test_the_request_turns_reasoning_off_and_asks_for_plain_text(configured, monkeypatch):
    """The fix for the truncation problem: with reasoning disabled laguna
    answered completely in 3.4s (finish=stop, 63 tokens). No JSON schema either:
    structured output was the most likely thing to fail and nothing needs it."""
    sent = install_post(monkeypatch, ok())
    openrouter._call("m/x:free", [{"role": "user", "content": "hi"}], 20)
    assert sent["body"]["reasoning"] == {"enabled": False}
    assert "response_format" not in sent["body"]
    assert sent["body"]["model"] == "m/x:free"
    assert sent["body"]["max_tokens"] == openrouter.MAX_TOKENS
    assert sent["headers"]["Authorization"] == "Bearer test-key"
    assert sent["timeout"] == 20


def test_call_returns_the_content_and_finish_reason(configured, monkeypatch):
    install_post(monkeypatch, ok("hello there friend", "stop"))
    assert openrouter._call("m", [], 5) == ("hello there friend", "stop")


def test_http_200_with_an_embedded_overload_error_is_a_failure(configured, monkeypatch):
    """OpenRouter can answer 200 with an error object instead of choices."""
    install_post(monkeypatch, FakeResponse({"error": {"code": 503, "message": "Upstream error from Nvidia: Service temporarily overloaded"}}))
    with pytest.raises(_Rejected, match="503"):
        openrouter._call("m", [], 5)


def test_a_rate_limit_status_is_a_failure(configured, monkeypatch):
    install_post(monkeypatch, FakeResponse({"error": {"code": 429, "message": "Provider returned error"}}, status=429))
    with pytest.raises(_Rejected, match="429"):
        openrouter._call("m", [], 5)


def test_a_non_json_error_page_is_a_failure(configured, monkeypatch):
    install_post(monkeypatch, FakeResponse(None, status=502, text="<html>bad gateway</html>"))
    with pytest.raises(_Rejected, match="non-JSON"):
        openrouter._call("m", [], 5)


def test_a_timeout_is_a_failure_that_names_the_limit(configured, monkeypatch):
    install_post(monkeypatch, exc=httpx.ReadTimeout("slow"))
    with pytest.raises(_Rejected, match="timed out after 5s"):
        openrouter._call("m", [], 5)


def test_a_network_error_is_a_failure(configured, monkeypatch):
    install_post(monkeypatch, exc=httpx.ConnectError("no route"))
    with pytest.raises(_Rejected, match="network error"):
        openrouter._call("m", [], 5)


def test_an_unexpected_reply_shape_is_a_failure(configured, monkeypatch):
    install_post(monkeypatch, FakeResponse({"choices": []}))
    with pytest.raises(_Rejected, match="shape"):
        openrouter._call("m", [], 5)


def test_the_api_key_never_appears_in_a_failure_message(configured, monkeypatch):
    install_post(monkeypatch, FakeResponse({"error": {"code": 401, "message": "bad key"}}, status=401))
    with pytest.raises(_Rejected) as info:
        openrouter._call("m", [], 5)
    assert "test-key" not in str(info.value)


# --- the prompt -----------------------------------------------------------------

def test_the_prompt_asks_for_plain_sentences_only():
    prompt = openrouter.SYSTEM_PROMPT
    assert "2 or 3 sentences" in prompt
    assert "no markdown" in prompt and "no lists" in prompt


def test_the_prompt_asks_for_a_rewrite_not_an_interpretation():
    """Regression. Given raw numbers, a small model wrote 'not expected to be
    unusually dry, though the chance of an unusually dry week is 54%' and called
    the wettest day 'the lowest'. It is now given correct statements to rephrase."""
    prompt = openrouter.SYSTEM_PROMPT
    assert "Every point is already correct" in prompt
    assert "Do not reverse, soften or reinterpret" in prompt
    assert "wettest day stays the wettest day" in prompt


def test_the_prompt_forbids_inventing_numbers():
    assert "Never estimate, compute, round or add any number" in openrouter.SYSTEM_PROMPT


def test_the_prompt_keeps_units_attached_to_their_numbers():
    """Regression: a 54% chance was rewritten as 54 mm of rain."""
    prompt = openrouter.SYSTEM_PROMPT
    assert "a percentage stays a percentage" in prompt and "Never turn a chance into an amount of rain" in prompt


def test_the_prompt_covers_the_degenerate_threshold_case():
    prompt = openrouter.SYSTEM_PROMPT
    assert "a dry week is normal here" in prompt and "do not describe the outlook as reassuring" in prompt


def test_the_prompt_asks_the_model_to_report_disagreement():
    assert "the two sources differ" in openrouter.SYSTEM_PROMPT


def test_the_prompt_forbids_adding_advice():
    assert "Do not add facts, advice" in openrouter.SYSTEM_PROMPT


def test_the_prompt_no_longer_asks_the_llm_to_rate_risk_or_confidence():
    """Those are derived by rules now; asking a model for them is what let it
    contradict the forecast model (LOW against MODERATE) and return 0.0."""
    prompt = openrouter.SYSTEM_PROMPT.lower()
    assert "rainfall_risk" not in prompt and "confidence" not in prompt and "json" not in prompt
