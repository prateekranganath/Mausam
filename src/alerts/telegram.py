"""Telegram Bot API sender.

Free, no registration, no approval: a token from @BotFather and a chat id
is the whole setup. (This is why Telegram, and not WhatsApp, is the
delivery channel here -- the WhatsApp Business API needs an approved
business account, which is not something a demo can assume.)

DELIBERATELY NO RETRIES, which is the opposite of every other client in
this project. src/utils/http_cache.py retries three times and falls back
to a stale cache, and src/llm/openrouter.py retries dropped connections
and 5xx. Both are safe to repeat because both are READS. sendMessage is
not: if a request times out after Telegram has already accepted it, a
retry delivers the alert twice. A farmer receiving the same drought
warning twice is a worse failure than one that did not arrive and can be
re-sent by hand, so a failure is reported rather than retried.

VERIFIED ERROR SHAPE (checked against the live API on 2026-09-20 with an
invalid token): Telegram answers HTTP 401 with a JSON body
`{"ok": false, "error_code": 401, "description": "Unauthorized"}`. The
body is present even on an error status, so `description` is read and
surfaced instead of calling raise_for_status and throwing it away --
"Unauthorized" (bad token), "Bad Request: chat not found" (wrong or empty
chat id) and "Forbidden: bot was blocked by the user" are three very
different problems and the operator needs to know which one happened.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import requests

from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

BASE_URL = "https://api.telegram.org"
REQUEST_TIMEOUT_SECONDS = 15


class TelegramNotConfigured(RuntimeError):
    """Raised when a send is attempted with no token or no chat id."""


@dataclass(frozen=True)
class SendResult:
    sent: bool
    chat_id: str | None = None
    message_id: int | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def is_configured() -> bool:
    """Whether a send could be attempted at all.

    Both halves are required: a token with no chat id produces Telegram's
    unhelpful "chat_id is empty" rather than anything actionable, so it is
    better to report the service as unconfigured up front.
    """
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def configuration_hint() -> str:
    """Which half is missing, for an operator reading a 503."""
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN (get one from @BotFather)")
    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID (message the bot, then read message.chat.id from /getUpdates)")
    if not missing:
        return "Telegram is configured."
    return "Telegram is not configured. Set " + " and ".join(missing) + " in .env."


def send_message(text: str, chat_id: str | None = None) -> SendResult:
    """Send `text` to `chat_id`, defaulting to the configured chat.

    Never raises for an upstream failure -- a failed alert is a result,
    not an exception, because the caller has to report it either way.
    TelegramNotConfigured is the one exception, since that is a
    deployment mistake rather than a delivery failure.
    """
    if not is_configured():
        raise TelegramNotConfigured(configuration_hint())

    target = chat_id or TELEGRAM_CHAT_ID
    url = f"{BASE_URL}/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            data={"chat_id": target, "text": text},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        # No retry -- see the module docstring. The send may or may not
        # have landed, and guessing wrong duplicates the alert.
        logger.warning("Telegram send failed before a reply was received: %s", exc)
        return SendResult(
            sent=False,
            chat_id=target,
            error=f"Could not reach Telegram: {exc}. The message may or may not have been delivered.",
        )

    try:
        payload = response.json()
    except ValueError:
        return SendResult(
            sent=False,
            chat_id=target,
            error=f"Telegram returned a non-JSON response (HTTP {response.status_code}).",
        )

    if payload.get("ok"):
        message_id = (payload.get("result") or {}).get("message_id")
        logger.info("Telegram alert delivered to %s (message %s)", target, message_id)
        return SendResult(sent=True, chat_id=target, message_id=message_id)

    # Surface Telegram's own words: they distinguish a bad token from a
    # bad chat id from a blocked bot, and the operator needs that.
    description = payload.get("description") or f"HTTP {response.status_code}"
    logger.warning("Telegram rejected the send: %s", description)
    return SendResult(sent=False, chat_id=target, error=description)
