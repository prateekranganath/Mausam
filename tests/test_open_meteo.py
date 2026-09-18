import datetime as dt
import os

import pandas as pd
import pytest
import requests

from src.forecasting import open_meteo as om


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(om, "OPEN_METEO_CACHE_DIR", tmp_path)
    monkeypatch.setattr(om.time, "sleep", lambda _s: None)
    monkeypatch.setattr(om, "_today_ist", lambda: dt.date(2026, 9, 18), raising=False)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def install_get(monkeypatch, payload=None, exc=None):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append({"url": url, "params": params})
        if exc is not None:
            raise exc
        return FakeResponse(payload)

    monkeypatch.setattr(om.requests, "get", fake_get)
    return calls


def daily_payload(n=30, start="2026-08-19", value=1.0):
    times = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, periods=n, freq="D")]
    return {"daily": {"time": times, "precipitation_sum": [value] * n, "temperature_2m_mean": [25.0] * n}}


def age_cache(tmp_path, seconds):
    old = os.path.getmtime(next(tmp_path.glob("*.json"))) - seconds
    for f in tmp_path.glob("*.json"):
        os.utime(f, (old, old))


# --- forecast fetch (existing behaviour, pinned before the refactor) ---------------------

def test_forecast_fetch_hits_forecast_endpoint_with_expected_params(monkeypatch):
    calls = install_get(monkeypatch, daily_payload())
    om.fetch_daily_weather(8.475, 76.95, forecast_days=8, past_days=35)
    assert calls[0]["url"] == om.BASE_URL
    p = calls[0]["params"]
    assert p["past_days"] == 35 and p["forecast_days"] == 8
    assert p["wind_speed_unit"] == "ms" and p["timezone"] == "auto"
    assert "precipitation_probability_max" in p["daily"]


def test_forecast_fetch_is_cached(monkeypatch):
    calls = install_get(monkeypatch, daily_payload())
    first = om.fetch_daily_weather(8.475, 76.95, 8, 35)
    second = om.fetch_daily_weather(8.475, 76.95, 8, 35)
    assert len(calls) == 1
    assert first.equals(second)


def test_forecast_fetch_serves_stale_cache_when_api_is_down(monkeypatch, tmp_path):
    install_get(monkeypatch, daily_payload())
    om.fetch_daily_weather(8.475, 76.95, 8, 35)
    age_cache(tmp_path, om.CACHE_TTL_SECONDS + 60)
    calls = install_get(monkeypatch, exc=requests.ConnectionError("down"))
    df = om.fetch_daily_weather(8.475, 76.95, 8, 35)
    assert len(df) == 30  # stale data, not an error
    assert len(calls) == om.MAX_RETRIES


def test_forecast_fetch_raises_when_down_and_nothing_cached(monkeypatch):
    install_get(monkeypatch, exc=requests.ConnectionError("down"))
    with pytest.raises(om.OpenMeteoError, match="no cached"):
        om.fetch_daily_weather(8.475, 76.95, 8, 35)


def test_http_error_status_is_retried_then_raised(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(1)
        return FakeResponse({}, status=500)

    monkeypatch.setattr(om.requests, "get", fake_get)
    with pytest.raises(om.OpenMeteoError):
        om.fetch_daily_weather(1.0, 2.0, 8, 35)
    assert len(calls) == om.MAX_RETRIES


# --- historical fetch (archive API) -----------------------------------------------------------

def test_historical_uses_archive_endpoint_and_ends_yesterday_ist(monkeypatch):
    calls = install_get(monkeypatch, daily_payload(30))
    om.fetch_historical_weather(22.5, 88.3, days=30)
    assert calls[0]["url"] == om.ARCHIVE_URL
    p = calls[0]["params"]
    assert p["end_date"] == "2026-09-17"  # yesterday: today's value is incomplete
    assert p["start_date"] == "2026-08-19"  # 30 days inclusive
    assert p["wind_speed_unit"] == "ms"
    assert "precipitation_probability_max" not in p["daily"]  # a forecast-only variable


@pytest.mark.parametrize(
    "days,bucket",
    [(1, 30), (30, 30), (31, 90), (90, 90), (91, 180), (181, 365), (366, 730), (730, 730)],
)
def test_historical_window_is_bucketed_so_cache_cannot_grow_without_bound(monkeypatch, days, bucket):
    calls = install_get(monkeypatch, daily_payload(bucket))
    df = om.fetch_historical_weather(22.5, 88.3, days=days)
    start = dt.date.fromisoformat(calls[0]["params"]["start_date"])
    end = dt.date.fromisoformat(calls[0]["params"]["end_date"])
    assert (end - start).days + 1 == bucket
    assert len(df) == days  # caller gets exactly what it asked for


def test_different_days_in_one_bucket_share_a_single_upstream_call(monkeypatch):
    calls = install_get(monkeypatch, daily_payload(30))
    om.fetch_historical_weather(22.5, 88.3, days=10)
    om.fetch_historical_weather(22.5, 88.3, days=25)
    assert len(calls) == 1


def test_historical_returns_the_most_recent_days(monkeypatch):
    install_get(monkeypatch, daily_payload(30, start="2026-08-19"))
    df = om.fetch_historical_weather(22.5, 88.3, days=3)
    assert list(df["time"]) == ["2026-09-15", "2026-09-16", "2026-09-17"]


def test_historical_rejects_windows_beyond_the_maximum(monkeypatch):
    install_get(monkeypatch, daily_payload(30))
    with pytest.raises(ValueError):
        om.fetch_historical_weather(22.5, 88.3, days=731)
    with pytest.raises(ValueError):
        om.fetch_historical_weather(22.5, 88.3, days=0)


def test_historical_falls_back_to_stale_cache_then_raises_without_one(monkeypatch, tmp_path):
    install_get(monkeypatch, exc=requests.ConnectionError("down"))
    with pytest.raises(om.OpenMeteoError):
        om.fetch_historical_weather(22.5, 88.3, days=10)

    install_get(monkeypatch, daily_payload(30))
    om.fetch_historical_weather(22.5, 88.3, days=10)
    age_cache(tmp_path, om.CACHE_TTL_SECONDS + 60)
    install_get(monkeypatch, exc=requests.ConnectionError("down"))
    assert len(om.fetch_historical_weather(22.5, 88.3, days=10)) == 10


def test_historical_and_forecast_caches_do_not_collide(monkeypatch):
    calls = install_get(monkeypatch, daily_payload(30))
    om.fetch_daily_weather(22.5, 88.3, forecast_days=8, past_days=30)
    om.fetch_historical_weather(22.5, 88.3, days=30)
    assert len(calls) == 2
    assert {c["url"] for c in calls} == {om.BASE_URL, om.ARCHIVE_URL}
