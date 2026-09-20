"""The /alerts/telegram endpoints.

Upstream Telegram is always mocked. These pin the HTTP contract and the
guards around an endpoint that sends outbound messages from a service
with no authentication in front of it.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.alerts import telegram
from src.api import routes
from src.api.main import app
from src.data import history as history_module

from tests.test_api import _install
from tests.test_api_monsoon import synthetic_history


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload


@pytest.fixture
def client(monkeypatch):
    svc = _install()
    svc.telegram_sends.clear()
    svc.history_cache.clear()
    monkeypatch.setattr(
        history_module, "district_history",
        lambda cfg, bridge=True: synthetic_history(cfg.district, cfg.state),
    )
    return TestClient(app)


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(telegram, "TELEGRAM_BOT_TOKEN", "123:FAKE")
    monkeypatch.setattr(telegram, "TELEGRAM_CHAT_ID", "999")


@pytest.fixture
def unconfigured(monkeypatch):
    monkeypatch.setattr(telegram, "TELEGRAM_BOT_TOKEN", None)
    monkeypatch.setattr(telegram, "TELEGRAM_CHAT_ID", None)


def install_post(monkeypatch, payload=None):
    calls = []

    def fake_post(url, data=None, timeout=None):
        calls.append({"url": url, "data": data})
        return FakeResponse(payload or {"ok": True, "result": {"message_id": 7}})

    monkeypatch.setattr(telegram.requests, "post", fake_post)
    return calls


# --- preview -------------------------------------------------------------------

def test_preview_works_with_no_credentials_at_all(client, unconfigured):
    """The whole point of a separate preview endpoint: the feature demos end
    to end on a fresh clone, and only the final send needs a token."""
    body = client.get("/alerts/telegram/preview", params={"district": "Kolkata"}).json()
    assert body["telegram_configured"] is False
    assert body["message"]
    assert "Mausam alert" in body["message"]
    assert "BotFather" in body["configuration_hint"]


def test_preview_reports_which_sections_made_it_in(client, configured):
    body = client.get(
        "/alerts/telegram/preview", params={"district": "Kolkata", "crop": "rice_transplanted"}
    ).json()
    assert "forecast" in body["sections_included"]
    assert "crop_advisory" in body["sections_included"]


def test_preview_never_sends(client, configured, monkeypatch):
    calls = install_post(monkeypatch)
    client.get("/alerts/telegram/preview", params={"district": "Kolkata"})
    assert calls == []


def test_preview_404s_for_an_unserved_district(client, configured):
    assert client.get("/alerts/telegram/preview", params={"district": "Atlantis"}).status_code == 404


def test_preview_422s_for_an_unknown_crop(client, configured):
    r = client.get("/alerts/telegram/preview", params={"district": "Kolkata", "crop": "quinoa"})
    assert r.status_code == 422


# --- send ----------------------------------------------------------------------

def test_send_delivers_and_returns_the_message_id(client, configured, monkeypatch):
    calls = install_post(monkeypatch)
    body = client.post("/alerts/telegram/send", params={"district": "Kolkata"}).json()
    assert body["sent"] is True
    assert body["message_id"] == 7
    assert len(calls) == 1


def test_send_composes_server_side_and_ignores_caller_supplied_text(client, configured, monkeypatch):
    """The client posts a district, never a message body. An endpoint that
    accepted arbitrary text would be a relay for sending anything through
    this bot."""
    calls = install_post(monkeypatch)
    client.post(
        "/alerts/telegram/send",
        params={"district": "Kolkata", "message": "ARBITRARY ATTACKER TEXT", "text": "ALSO THIS"},
    )
    assert "ARBITRARY ATTACKER TEXT" not in calls[0]["data"]["text"]
    assert "Mausam alert" in calls[0]["data"]["text"]


def test_send_ignores_a_caller_supplied_chat_id(client, configured, monkeypatch):
    """The recipient is TELEGRAM_CHAT_ID and cannot be overridden, or this
    becomes an open spam relay."""
    calls = install_post(monkeypatch)
    client.post("/alerts/telegram/send", params={"district": "Kolkata", "chat_id": "12345"})
    assert calls[0]["data"]["chat_id"] == "999"


def test_send_503s_when_unconfigured(client, unconfigured):
    r = client.post("/alerts/telegram/send", params={"district": "Kolkata"})
    assert r.status_code == 503
    assert "BotFather" in r.json()["detail"]


def test_a_delivery_failure_is_200_with_the_reason_not_a_5xx(client, configured, monkeypatch):
    """The caller needs to know WHICH failure it was to fix it."""
    install_post(monkeypatch, {"ok": False, "description": "Bad Request: chat not found"})
    r = client.post("/alerts/telegram/send", params={"district": "Kolkata"})
    assert r.status_code == 200
    body = r.json()
    assert body["sent"] is False
    assert "chat not found" in body["error"]


def test_send_404s_for_an_unserved_district(client, configured, monkeypatch):
    install_post(monkeypatch)
    assert client.post("/alerts/telegram/send", params={"district": "Atlantis"}).status_code == 404


# --- throttle ------------------------------------------------------------------

def test_sends_are_throttled(client, configured, monkeypatch):
    """No authentication sits in front of this endpoint, and an unthrottled
    outbound sender is a good way to get the bot banned."""
    install_post(monkeypatch)
    monkeypatch.setattr(routes, "TELEGRAM_MAX_SENDS_PER_WINDOW", 3)
    monkeypatch.setattr(routes, "TELEGRAM_THROTTLE_WINDOW_SECONDS", 60)

    codes = [
        client.post("/alerts/telegram/send", params={"district": "Kolkata"}).status_code
        for _ in range(5)
    ]
    assert codes[:3] == [200, 200, 200]
    assert codes[3:] == [429, 429]


def test_the_throttle_window_slides(client, configured, monkeypatch):
    """A fixed-reset counter would let a burst through on every boundary."""
    install_post(monkeypatch)
    monkeypatch.setattr(routes, "TELEGRAM_MAX_SENDS_PER_WINDOW", 2)
    monkeypatch.setattr(routes, "TELEGRAM_THROTTLE_WINDOW_SECONDS", 60)

    for _ in range(2):
        client.post("/alerts/telegram/send", params={"district": "Kolkata"})
    assert client.post("/alerts/telegram/send", params={"district": "Kolkata"}).status_code == 429

    # Age the recorded sends past the window.
    app.state.svc.telegram_sends = [t - 120 for t in app.state.svc.telegram_sends]
    assert client.post("/alerts/telegram/send", params={"district": "Kolkata"}).status_code == 200


def test_a_throttled_request_does_not_reach_telegram(client, configured, monkeypatch):
    calls = install_post(monkeypatch)
    monkeypatch.setattr(routes, "TELEGRAM_MAX_SENDS_PER_WINDOW", 1)
    client.post("/alerts/telegram/send", params={"district": "Kolkata"})
    client.post("/alerts/telegram/send", params={"district": "Kolkata"})
    assert len(calls) == 1


# --- CORS ----------------------------------------------------------------------

def test_cors_preflight_allows_the_send(client):
    """allow_methods was GET-only. Without POST the browser blocks the
    preflight and the UI reports an opaque CORS error, not a send failure."""
    r = client.options(
        "/alerts/telegram/send",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code == 200
    assert "POST" in r.headers.get("access-control-allow-methods", "")


def test_cors_still_allows_get(client):
    r = client.options(
        "/forecast/Kolkata",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code == 200
