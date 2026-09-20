"""Shared TTL-disk-cache + retry + stale-fallback JSON fetcher.

Extracted from src/forecasting/open_meteo.py, which held the only copy.
Three clients now need byte-identical behaviour — Open-Meteo, NASA POWER
and the climate-index feeds — and all three sit on services that are
free, unauthenticated and therefore occasionally slow, throttled or
briefly unavailable. For every one of them "serve yesterday's answer with
a loud warning" beats "fail the request", so the policy belongs in one
place rather than being re-typed per client.

`getter` and `sleep` are passed in as callables instead of this module
importing `requests`/`time` and calling them directly. That is
deliberate, not indirection for its own sake: each client's tests
monkeypatch `<client>.requests.get` and `<client>.time.sleep` on their own
module object, and a late-bound callable supplied by the client keeps
those patches effective. A direct `requests.get` here would silently
escape them and start making real network calls during the test run.
"""
from __future__ import annotations

import json
import logging
import time as _time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import requests

logger = logging.getLogger(__name__)

# What counts as "the call failed, try again": a transport/HTTP-status
# problem (requests.RequestException covers ConnectionError, Timeout and
# HTTPError from raise_for_status, so a 429 or 5xx lands here) or a body
# that is not JSON at all (ValueError from .json()) — which these APIs do
# return occasionally, as an HTML error page with a 200 status.
DEFAULT_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (requests.RequestException, ValueError)


def read_cache(cache_file: Path, ttl_seconds: float) -> dict | None:
    """The cached payload if it exists and is younger than ttl_seconds."""
    if not cache_file.exists():
        return None
    if _time.time() - cache_file.stat().st_mtime > ttl_seconds:
        return None
    with open(cache_file, encoding="utf-8") as f:
        return json.load(f)


def read_stale_cache(cache_file: Path) -> dict | None:
    """The cached payload regardless of age, or None. Used only as a
    last resort after every retry has failed."""
    if not cache_file.exists():
        return None
    with open(cache_file, encoding="utf-8") as f:
        return json.load(f)


def fetch_json_cached(
    *,
    url: str,
    params: dict[str, Any],
    cache_file: Path,
    label: str,
    getter: Callable[..., Any],
    sleep: Callable[[float], None],
    error_cls: type[Exception],
    ttl_seconds: float,
    timeout_seconds: float,
    max_retries: int,
    backoff_seconds: float,
    retry_exceptions: Sequence[type[BaseException]] | Iterable[type[BaseException]] = DEFAULT_RETRY_EXCEPTIONS,
) -> dict:
    """GET `url` as JSON, through a TTL disk cache, with linear backoff
    retries and a stale-cache fallback.

    Order of preference: fresh cache, then a live call, then stale cache
    (with a warning), then `error_cls`. So this raises only when the call
    failed AND nothing was ever cached for `label`.

    `label` is a human-readable description of the request used in log
    lines; it must not contain secrets, since it is also embedded in the
    raised error message.
    """
    cached = read_cache(cache_file, ttl_seconds)
    if cached is not None:
        logger.info("Cache hit for %s", label)
        return cached

    retry_exceptions = tuple(retry_exceptions)
    last_error: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Request %s (attempt %d/%d)", label, attempt, max_retries)
            resp = getter(url, params=params, timeout=timeout_seconds)
            resp.raise_for_status()
            payload = resp.json()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            return payload
        except retry_exceptions as exc:
            last_error = exc
            logger.warning("Request failed (attempt %d/%d): %s", attempt, max_retries, exc)
            if attempt < max_retries:
                sleep(backoff_seconds * attempt)

    stale = read_stale_cache(cache_file)
    if stale is not None:
        logger.warning(
            "Upstream unreachable after %d attempts; serving stale cache from %s", max_retries, cache_file
        )
        return stale

    raise error_cls(
        f"Request failed after {max_retries} attempts and no cached "
        f"fallback exists for {label}: {last_error}"
    )


def fetch_text_cached(
    *,
    url: str,
    cache_file: Path,
    label: str,
    getter: Callable[..., Any],
    sleep: Callable[[float], None],
    error_cls: type[Exception],
    ttl_seconds: float,
    timeout_seconds: float,
    max_retries: int,
    backoff_seconds: float,
    params: dict[str, Any] | None = None,
    retry_exceptions: Sequence[type[BaseException]] | Iterable[type[BaseException]] = DEFAULT_RETRY_EXCEPTIONS,
) -> str:
    """Same policy as fetch_json_cached, for endpoints that serve plain
    text rather than JSON.

    The climate-index feeds (NOAA CPC, NOAA PSL, the IRI mirror) are all
    fixed-width or tab-separated text files, so there is no JSON body to
    parse and no `.json()` call that could fail. The cache file therefore
    holds the raw text exactly as served, which also makes it trivially
    inspectable when a parser needs debugging.
    """
    if cache_file.exists() and _time.time() - cache_file.stat().st_mtime <= ttl_seconds:
        logger.info("Cache hit for %s", label)
        return cache_file.read_text(encoding="utf-8")

    retry_exceptions = tuple(retry_exceptions)
    last_error: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Request %s (attempt %d/%d)", label, attempt, max_retries)
            resp = getter(url, params=params, timeout=timeout_seconds)
            resp.raise_for_status()
            text = resp.text
            if not text.strip():
                raise ValueError("upstream returned an empty body")
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(text, encoding="utf-8")
            return text
        except retry_exceptions as exc:
            last_error = exc
            logger.warning("Request failed (attempt %d/%d): %s", attempt, max_retries, exc)
            if attempt < max_retries:
                sleep(backoff_seconds * attempt)

    if cache_file.exists():
        logger.warning(
            "Upstream unreachable after %d attempts; serving stale cache from %s", max_retries, cache_file
        )
        return cache_file.read_text(encoding="utf-8")

    raise error_cls(
        f"Request failed after {max_retries} attempts and no cached "
        f"fallback exists for {label}: {last_error}"
    )
