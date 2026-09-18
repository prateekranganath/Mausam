"""OpenRouter reasoning layer.

The LLM is NOT a source of weather truth. It receives the numeric outputs
(ML model, Open-Meteo forecast, their agreement) and is asked only to
summarise, explain disagreement, and turn them into plain-language advice.
Every number it cites must come from the input; that is enforced by prompt
AND checked after the fact (find_unsupported_numbers), with any number that
cannot be traced back to the input surfaced to the caller rather than hidden.

API shape verified against the OpenRouter docs: POST
https://openrouter.ai/api/v1/chat/completions, Bearer auth, structured output
via response_format {"type": "json_schema", "json_schema": {...}}. Not every
model honours response_format, so the response is also validated server-side
with pydantic and retried once rather than trusted.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from src.config import OPENROUTER_API_KEY, OPENROUTER_MODEL

logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
# Free models are slow (measured 68-190s for the large Nemotron models under
# load; a small paid model takes ~10s), so the timeout is configurable. Past
# it, /advisory degrades to numbers-only (HTTP 200, advisory null).
REQUEST_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "60"))
MAX_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 1.0

# Unsigned on purpose: a signed pattern reads the "-19" in an ISO date like
# 2026-09-19 as a negative number, so a narrative saying "Sep 19" could never
# match. Rainfall/probability inputs are non-negative, so magnitude is enough.
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class Advisory(BaseModel):
    """Structured LLM output. extra='forbid' makes the JSON schema strict."""

    model_config = ConfigDict(extra="forbid")

    forecast_summary: str
    rainfall_risk: Literal["LOW", "MODERATE", "HIGH"]
    confidence: float
    key_factors: list[str]
    model_disagreement: list[str]
    advisory: list[str]

    # Range-checked here rather than via Field(ge/le) so the JSON schema sent
    # to the provider stays free of min/max keywords some strict modes reject.
    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v


SYSTEM_PROMPT = """You are a careful assistant that explains rainfall forecasts for Indian districts to farmers and local officials.

You are given structured numeric data from two independent sources:
- "ml_model": a statistical model's estimate of the probability that the next 7 days are unusually dry for this district and time of year, plus its expected 7-day rainfall in mm.
- "open_meteo_forecast": a live numerical weather forecast for the same 7 days.
- "agreement": a comparison of the two.
- "model_reliability": how well the ML model performed on held-out data.

STRICT RULES:
1. You are NOT a source of weather data. Never invent, estimate, adjust or round-trip any number, probability, date or measurement. Every number you write must appear in the input data exactly as given (you may quote it; do not compute new ones).
2. Report disagreement in "model_disagreement" whenever agreement.magnitude_diverges is true OR agreement.sources_agree is false. This applies EVEN IF both sources land on the same side of the threshold: a large gap between the two rainfall totals is a disagreement worth stating. Give both totals from the input and the gap, and say what it means for how much to trust each. Do not smooth it over or pick a winner without stating the uncertainty. Only when magnitude_diverges is false AND sources_agree is true (or null) may "model_disagreement" be an empty list. Your "forecast_summary" must also mention a divergence when one exists.
3. The ML model is a hackathon-grade statistical model, not an official forecast. "confidence" (0 to 1) is your judgement of how far to trust the overall outlook. It must not simply repeat any single number from the input (not a metric, and not a probability). Lower it when the sources diverge or disagree, and when model_reliability is weak.
4. "rainfall_risk" is your overall judgement of the risk of INSUFFICIENT rainfall: LOW, MODERATE or HIGH.
5. "advisory" must be general, practical agricultural/water-planning guidance that follows from the numbers. Do not give specific crop, chemical or dosage recommendations. Do not claim certainty.
6. Do not claim village- or panchayat-level accuracy. Data is district level.
7. If agreement.threshold_degenerate is true, this district normally gets little or no rain at this time of year, so the "insufficient rainfall" label cannot meaningfully trigger. In that case state plainly that a LOW risk rating does NOT mean rain is expected, describe the forecast totals in plain terms (for example that a dry week is normal here), and do not present staying above a threshold near zero as reassuring.

OUTPUT FORMAT. Respond with ONE JSON object and nothing else (no prose, no markdown fences), with exactly these six keys and no others. Do NOT echo or restate the input data:
{
  "forecast_summary": string,
  "rainfall_risk": "LOW" | "MODERATE" | "HIGH",
  "confidence": number between 0 and 1,
  "key_factors": [string, ...],
  "model_disagreement": [string, ...],
  "advisory": [string, ...]
}"""


def is_configured() -> bool:
    return bool(OPENROUTER_API_KEY and OPENROUTER_MODEL)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _numbers(text: str) -> list[float]:
    return [float(m) for m in _NUMBER_RE.findall(text)]


def find_unsupported_numbers(advisory: Advisory, source_payload: dict) -> list[str]:
    """Numbers appearing in the LLM's prose that cannot be traced to the input.

    A number counts as supported if it matches any number in the serialised
    input within a small tolerance (so "100" is accepted for 100.3 and "0.06"
    for 0.0639). This is a tripwire for fabrication, not a proof of
    correctness, and is reported to the caller rather than blocking the
    response. Bare small integers are ignored ("7-day", "1st") since forecast
    horizon words are not measurements.
    """
    source_numbers = _numbers(json.dumps(source_payload, default=str))

    def supported(n: float) -> bool:
        # A probability in [0, 1] restated as a percentage (0.3198 -> 31.98%)
        # is the same figure, not a new one.
        candidates = source_numbers + [x * 100 for x in source_numbers if 0 < x <= 1]
        return any(abs(n - x) <= max(0.5, 0.01 * abs(x)) for x in candidates)

    prose = " ".join(
        [
            advisory.forecast_summary,
            *advisory.key_factors,
            *advisory.model_disagreement,
            *advisory.advisory,
        ]
    )
    unsupported: list[str] = []
    for token in _NUMBER_RE.findall(prose):
        value = float(token)
        if value.is_integer() and abs(value) <= 10:
            continue
        if not supported(value) and token not in unsupported:
            unsupported.append(token)
    return unsupported


def _post(messages: list[dict]) -> str:
    body = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "rainfall_advisory",
                "strict": True,
                "schema": Advisory.model_json_schema(),
            },
        },
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "Mausam",
    }
    resp = httpx.post(API_URL, json=body, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    # OpenRouter can return HTTP 200 with an "error" object instead of
    # "choices" (seen with free models: {"error": {"code": 503, "message":
    # "Upstream error from Nvidia: Service temporarily overloaded"}}). Honour
    # the embedded code, or a transient overload looks like a malformed reply
    # and a permanent 403 gets pointlessly retried.
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        code = err.get("code") if isinstance(err, dict) else None
        message = err.get("message") if isinstance(err, dict) else str(err)
        retryable = code is None or code == 429 or (isinstance(code, int) and code >= 500)
        raise OpenRouterError(f"HTTP {code}: {str(message)[:200]}", retryable=retryable)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError(f"Unexpected OpenRouter response shape: {str(data)[:300]}") from exc


def generate_advisory(source_payload: dict) -> tuple[Advisory, list[str]]:
    """Ask the LLM for a structured advisory over `source_payload`.

    Returns (advisory, unsupported_numbers). Raises OpenRouterError if the
    service is unconfigured, unreachable, or repeatedly returns output that
    fails schema validation. Never raises for merely unsupported numbers.
    """
    if not is_configured():
        raise OpenRouterError(
            "OpenRouter is not configured: set OPENROUTER_API_KEY and OPENROUTER_MODEL."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(source_payload, default=str)},
    ]

    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            content = _post(messages)
            advisory = Advisory.model_validate(json.loads(_strip_fences(content)))
            return advisory, find_unsupported_numbers(advisory, source_payload)
        except (httpx.HTTPError, ValueError, OpenRouterError) as exc:
            # ValueError covers json.JSONDecodeError and pydantic.ValidationError
            last_error = _describe(exc)
            logger.warning("OpenRouter attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, last_error)
            if not _is_retryable(exc):
                raise OpenRouterError(f"OpenRouter request failed: {last_error}") from exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS)

    raise OpenRouterError(f"OpenRouter failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _is_retryable(exc: Exception) -> bool:
    """Retry what a second try can plausibly fix: dropped connections, rate
    limits, 5xx, and malformed model output. Do NOT retry a timeout (the
    provider is just slow, so retrying doubles the client's wait) or a 4xx
    such as 401/403/404 (permanent: bad key, blocked or unknown model)."""
    if isinstance(exc, OpenRouterError):
        return exc.retryable
    if isinstance(exc, httpx.TimeoutException):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return True


def _describe(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
    if isinstance(exc, httpx.TimeoutException):
        return f"timed out after {REQUEST_TIMEOUT_SECONDS:.0f}s"
    return str(exc)
