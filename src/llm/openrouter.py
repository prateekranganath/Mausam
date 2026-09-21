"""OpenRouter: one optional plain-language paragraph over the analysis.

The LLM is NOT a source of weather truth and no longer a source of anything
structured. src/llm/analysis.py derives the risk, confidence, key factors,
disagreement and actions by rules. This module asks a free-tier model for one
thing only: two or three plain sentences saying it in words. If that fails,
the caller still has the whole analysis.

WHAT MEASURING THE FREE TIER (2026-09-21) SHOWED, and what each choice below
is a response to:

  * Almost every free model is a REASONING model. With a small token cap they
    spend the whole budget on hidden thinking and the visible answer is cut
    off mid-sentence: Nemotron returned "In Nagpur the forecast indicates",
    laguna returned "The". With `reasoning` disabled, laguna answered
    completely in 3.4s (finish=stop, 63 tokens, no thinking). So reasoning is
    turned off, and a reply with finish_reason "length" is REJECTED rather
    than shown.
  * `openrouter/free` returned its chain-of-thought as the answer ("We need
    to produce 2 or 3 short plain sentences..."). Text like that must never
    reach a farmer, so replies that read as leaked reasoning are rejected.
  * A model can answer a different question entirely (nex-n2.5-mini replied
    "please share the forecast details"). A reply must mention the district
    or one of the supplied numbers.
  * Any single free model fails intermittently: 503 "overloaded" and 429 in
    the same minute that it had worked. The old client retried the SAME model
    twice, one second apart, so one overload meant no answer at all. It now
    walks a chain of models, and skips a model that just failed for a minute
    rather than paying its timeout again on the next request.
  * Given raw numbers to INTERPRET, a small model wrote fluent nonsense that
    no number-check can catch ("not expected to be unusually dry, though
    the chance of an unusually dry week is 54%"; the WETTEST day called
    "the lowest"). So it is given already-correct statements to REWRITE
    in simpler words (see analysis.summary_facts), never numbers to read.
  * The old strict six-field JSON schema is gone. Structured output was the
    thing most likely to fail, and nothing needs it: the structure is now
    produced by code.

Every figure the LLM cites must come from the facts it was given, with the
unit it had there. That is enforced by the prompt AND checked after the fact,
and a note that fails the check is REJECTED (the next model is tried), not
shown with a warning. Rewriting the analysis is the note's only value, so an
unfaithful one is worse than none, and the analysis is always there without
it. The check has to be unit-aware: a live reply turned a "54% chance" into
"only about 54 mm of rain is expected against the typical 25.7 mm". The
number 54 IS in the input, as a percentage, so a number-only check passed it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass

import httpx

from src.config import OPENROUTER_API_KEY, OPENROUTER_MODEL

logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Per-attempt ceiling and an overall budget for the whole chain. The budget is
# what bounds the user's wait: without it, a chain of slow models could add up
# to minutes. (.env used to set the timeout to 90s for a single model.)
REQUEST_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "25"))
TOTAL_BUDGET_SECONDS = float(os.getenv("OPENROUTER_TOTAL_BUDGET_SECONDS", "40"))
# Do not start an attempt with less budget than this; it could not finish.
MIN_ATTEMPT_SECONDS = 4.0
# Headroom for models that ignore `reasoning: disabled`. A short answer needs
# ~70 tokens; this is only to let a model that insists on thinking finish.
MAX_TOKENS = 500
COOLDOWN_SECONDS = 60.0

# Tried after OPENROUTER_MODEL, in order. Override with a comma-separated
# OPENROUTER_FALLBACK_MODELS (an empty value disables fallbacks).
DEFAULT_FALLBACK_MODELS = (
    "poolside/laguna-xs-2.1:free",
    "google/gemma-4-26b-a4b-it:free",
    "nex-agi/nex-n2.5-mini:free",
)

# A figure with its unit, when it has one. Only mm and % are checked: they are
# the only units the points use, and they are the ones that get confused.
_FIGURE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|%)?", re.IGNORECASE)

# Unsigned on purpose: a signed pattern reads the "-19" in an ISO date like
# 2026-09-19 as a negative number, so a narrative saying "Sep 19" could never
# match. Rainfall/probability inputs are non-negative, so magnitude is enough.
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# What a model that is "thinking out loud" tends to open with.
_THINKING_PREFIXES = (
    "we need", "we should", "we have to", "the user", "user wants", "let me", "let's",
    "okay", "ok,", "first,", "i need", "i should", "i'll", "thinking", "hmm", "alright",
)

# model -> monotonic time before which it is skipped
_COOLDOWN: dict[str, float] = {}


class OpenRouterError(RuntimeError):
    pass


class _Rejected(Exception):
    """One model produced nothing usable. The reason is shown to operators."""


@dataclass(frozen=True)
class Summary:
    text: str
    model: str


SYSTEM_PROMPT = """You rewrite verified statements about a 7-day rainfall forecast for one Indian district as a short plain-language note for farmers and local officials.

The input has a district and a list of "points". Every point is already correct.

Write 2 or 3 sentences of plain text that say the same things in simpler words.

Rules:
- Keep the meaning of every point you use. Do not reverse, soften or reinterpret any of them: a chance of a dry week stays a chance of a dry week, and the wettest day stays the wettest day.
- Use ONLY the numbers in the points, exactly as written. Never estimate, compute, round or add any number, date or percentage.
- Keep every number with the unit it has in the points: a percentage stays a percentage and millimetres stay millimetres. Never turn a chance into an amount of rain.
- Do not add facts, advice, crop or chemical recommendations, or claims about village-level accuracy.
- If a point says the two sources differ, say so. If a point says a dry week is normal here, say that and do not describe the outlook as reassuring.
- You may leave out the least important points.
- Reply with the sentences only: no preamble, no markdown, no lists, no headings. Do not repeat these instructions or describe what you are doing."""


def model_chain() -> list[str]:
    """OPENROUTER_MODEL first, then the fallbacks, without duplicates."""
    raw = os.getenv("OPENROUTER_FALLBACK_MODELS")
    fallbacks = list(DEFAULT_FALLBACK_MODELS) if raw is None else [m.strip() for m in raw.split(",") if m.strip()]
    chain: list[str] = []
    for model in [OPENROUTER_MODEL, *fallbacks]:
        if model and model not in chain:
            chain.append(model)
    return chain


def is_configured() -> bool:
    # A key is enough: with no OPENROUTER_MODEL the default chain is used.
    return bool(OPENROUTER_API_KEY) and bool(model_chain())


def reset_cooldowns() -> None:
    _COOLDOWN.clear()


def _numbers(text: str) -> list[float]:
    return [float(m) for m in _NUMBER_RE.findall(text)]


def find_unsupported_numbers(text: str, source_facts: dict) -> list[str]:
    """Numbers in `text` that cannot be traced to `source_facts`.

    A number counts as supported if it matches any number in the serialised
    facts within a small tolerance (so "100" is accepted for 100.3). This is a
    tripwire for fabrication, not a proof of correctness, and is reported to
    the caller rather than blocking the response. Bare small integers are
    ignored ("7-day", "1st") since forecast horizon words are not measurements.
    """
    source_numbers = _numbers(json.dumps(source_facts, default=str))

    def supported(n: float) -> bool:
        # A probability in [0, 1] restated as a percentage is the same figure.
        candidates = source_numbers + [x * 100 for x in source_numbers if 0 < x <= 1]
        return any(abs(n - x) <= max(0.5, 0.01 * abs(x)) for x in candidates)

    unsupported: list[str] = []
    for token in _NUMBER_RE.findall(text):
        value = float(token)
        if value.is_integer() and abs(value) <= 10:
            continue
        if not supported(value) and token not in unsupported:
            unsupported.append(token)
    return unsupported


def find_unit_mismatches(text: str, source_facts: dict) -> list[str]:
    """Figures in `text` attached to a unit the input never gave them.

    find_unsupported_numbers only asks whether a NUMBER appears in the input. It
    cannot tell "54%" from "54 mm", and a live reply turned a 54% chance into
    "only about 54 mm of rain is expected". This flags a figure whose value is in
    the input ONLY with a different unit. A value that also appears unitless in
    the input (a date such as "Fri 25 Sep") is compatible with any unit, so this
    stays quiet unless the mismatch is unambiguous.
    """
    source = [(float(v), u.lower()) for v, u in _FIGURE_RE.findall(json.dumps(source_facts, default=str))]
    mismatched: list[str] = []
    for token, unit in _FIGURE_RE.findall(text):
        if not unit:
            continue
        value = float(token)
        unit = unit.lower()
        units = {u for v, u in source if abs(value - v) <= max(0.5, 0.01 * abs(v))}
        if units and unit not in units and "" not in units:
            label = f"{token}{unit}" if unit == "%" else f"{token} {unit}"
            if label not in mismatched:
                mismatched.append(label)
    return mismatched


def _clean(text: str) -> str:
    """Normalise what a model returned into one plain paragraph."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    # Non-breaking hyphens appear in model output and break some consoles.
    text = text.replace("‑", "-").replace("‐", "-")
    text = re.sub(r"(\*\*|__|`)", "", text)
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip("\"“” ")


def _on_topic(text: str, facts: dict) -> bool:
    """A reply must mention the district or at least one supplied number."""
    lowered = text.lower()
    district = str(facts.get("district") or "").lower()
    if district and district in lowered:
        return True
    return any(n in text for n in _NUMBER_RE.findall(json.dumps(facts, default=str)) if len(n) >= 2)


def _reject_reason(text: str, finish_reason: str | None, facts: dict) -> str | None:
    if finish_reason == "length":
        return "truncated (hit the token limit)"
    if not text:
        return "empty reply"
    if text.lower().startswith(_THINKING_PREFIXES):
        return "looks like leaked reasoning"
    if len(text) < 30:
        return "too short to be an answer"
    if len(text) > 1000:
        return "too long"
    if not _on_topic(text, facts):
        return "did not address the forecast"
    untraceable = find_unsupported_numbers(text, facts)
    if untraceable:
        return "cited figures that are not in the input (" + ", ".join(untraceable) + ")"
    mismatched = find_unit_mismatches(text, facts)
    if mismatched:
        return "attached a figure to the wrong unit (" + ", ".join(mismatched) + ")"
    return None


def _call(model: str, messages: list[dict], timeout: float) -> tuple[str, str | None]:
    """One request to one model. Returns (content, finish_reason)."""
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
        # The fix for the truncation problem above: no hidden thinking to
        # spend the token budget on.
        "reasoning": {"enabled": False},
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "Mausam",
    }
    try:
        response = httpx.post(API_URL, json=body, headers=headers, timeout=timeout)
    except httpx.TimeoutException:
        raise _Rejected(f"timed out after {timeout:.0f}s") from None
    except httpx.HTTPError as exc:
        raise _Rejected(f"network error ({type(exc).__name__})") from None

    try:
        data = response.json()
    except ValueError:
        raise _Rejected(f"non-JSON reply (HTTP {response.status_code})") from None

    # OpenRouter can answer HTTP 200 with an "error" object instead of
    # "choices" (seen with free models: {"error": {"code": 503, "message":
    # "Upstream error from Nvidia: Service temporarily overloaded"}}), as well
    # as with a real error status. Both are the same failure.
    error = data.get("error") if isinstance(data, dict) else None
    if error or response.status_code >= 400:
        code = (error.get("code") if isinstance(error, dict) else None) or response.status_code
        message = (error.get("message") if isinstance(error, dict) else None) or response.text[:120]
        raise _Rejected(f"HTTP {code}: {str(message)[:90]}")

    try:
        choice = data["choices"][0]
        return choice["message"].get("content") or "", choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        raise _Rejected("unexpected reply shape") from None


def generate_summary(facts: dict) -> Summary:
    """A short plain-language paragraph over `facts`, from the first model in
    the chain that produces a usable one.

    Raises OpenRouterError if unconfigured or if every model fails; the
    message lists what happened to each so it is diagnosable. Never raises
    merely because a number could not be traced (that is reported instead).
    """
    if not is_configured():
        raise OpenRouterError("OpenRouter is not configured: set OPENROUTER_API_KEY.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(facts, default=str)},
    ]

    deadline = time.monotonic() + TOTAL_BUDGET_SECONDS
    chain = model_chain()
    now = time.monotonic()
    available = [m for m in chain if _COOLDOWN.get(m, 0.0) <= now]
    # If every model is cooling down, try them anyway: being locked out for a
    # minute because all of them failed once helps nobody.
    candidates = available or chain

    failures: list[str] = []
    for model in candidates:
        remaining = deadline - time.monotonic()
        if remaining < MIN_ATTEMPT_SECONDS:
            failures.append("out of time budget")
            break
        try:
            content, finish_reason = _call(model, messages, min(REQUEST_TIMEOUT_SECONDS, remaining))
            text = _clean(content)
            reason = _reject_reason(text, finish_reason, facts)
            if reason:
                raise _Rejected(reason)
        except _Rejected as rejected:
            _COOLDOWN[model] = time.monotonic() + COOLDOWN_SECONDS
            failures.append(f"{model.removesuffix(':free')}: {rejected}")
            logger.warning("OpenRouter model %s gave no usable answer: %s", model, rejected)
            continue

        _COOLDOWN.pop(model, None)
        return Summary(text=text, model=model)

    raise OpenRouterError("No model produced a usable answer (" + "; ".join(failures) + ").")
