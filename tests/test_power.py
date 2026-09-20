import datetime as dt

import numpy as np
import pandas as pd
import pytest
import requests

from src.data import power as pw


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(pw, "POWER_CACHE_DIR", tmp_path)
    monkeypatch.setattr(pw.time, "sleep", lambda _s: None)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def install_get(monkeypatch, payload=None, exc=None, status=200):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append({"url": url, "params": params})
        if exc is not None:
            raise exc
        return FakeResponse(payload, status)

    monkeypatch.setattr(pw.requests, "get", fake_get)
    return calls


def power_payload(n=5, start="2024-01-01", end=None, overrides=None, fill_value=-999.0):
    """A POWER-shaped payload: parameters keyed by YYYYMMDD strings."""
    dates = [d.strftime("%Y%m%d") for d in pd.date_range(start, periods=n, freq="D")]
    defaults = {
        "PRECTOTCORR": 1.5, "T2M": 25.0, "T2M_MIN": 22.0, "T2M_MAX": 30.0,
        "WS10M": 3.0, "PS": 100.0, "RH2M": 80.0, "ALLSKY_SFC_SW_DWN": 20.0,
        "GWETTOP": 0.5, "GWETROOT": 0.4,
    }
    parameter = {k: {d: v for d in dates} for k, v in defaults.items()}
    for param, by_date in (overrides or {}).items():
        parameter[param].update(by_date)
    return {
        "header": {"fill_value": fill_value, "start": dates[0], "end": end or dates[-1]},
        "geometry": {"type": "Point", "coordinates": [76.95, 8.475, 86.25]},
        "properties": {"parameter": parameter},
    }


def to_frame(payload, **kw):
    args = {
        "district": "Thiruvananthapuram", "state": "KL",
        "latitude": 8.475, "longitude": 76.95, "elevation": 32.0,
    }
    args.update(kw)
    return pw.to_daily_frame(payload, **args)


# --- unit and sentinel handling: the two traps in this client -----------------

def test_pressure_is_converted_from_kpa_to_hpa():
    """POWER serves PS in kPa (~100); the trained air_pressure feature is
    hPa (~1000). Missing the x10 would feed values 2 orders of magnitude off."""
    frame = to_frame(power_payload(overrides={"PS": {"20240101": 99.97}}))
    assert frame["air_pressure"].iloc[0] == pytest.approx(999.7)


def test_fill_values_become_nan_and_are_not_scaled():
    """-999 must become NaN BEFORE the pressure conversion. Scaling first
    would turn an obvious sentinel into a plausible-looking -9990."""
    payload = power_payload(overrides={"PS": {"20240102": -999.0}, "PRECTOTCORR": {"20240102": -999.0}})
    frame = to_frame(payload)
    assert np.isnan(frame["air_pressure"].iloc[1])
    assert np.isnan(frame["rainfall"].iloc[1])
    assert -9990.0 not in set(frame["air_pressure"].dropna())


def test_fill_value_is_read_from_the_payload_not_hardcoded():
    payload = power_payload(fill_value=-777.0, overrides={"PRECTOTCORR": {"20240103": -777.0}})
    frame = to_frame(payload)
    assert np.isnan(frame["rainfall"].iloc[2])


def test_wind_speed_is_passed_through_unconverted():
    """WS10M is already m/s, matching the training data. Unlike Open-Meteo,
    no unit parameter is needed and no conversion should be applied."""
    frame = to_frame(power_payload(overrides={"WS10M": {"20240101": 3.83}}))
    assert frame["wind_speed"].iloc[0] == pytest.approx(3.83)


def test_registry_elevation_wins_over_power_grid_elevation():
    """elevation is a trained model feature; it must mean the same thing at
    serve time as at train time. POWER's geometry says 86.25 for this point."""
    frame = to_frame(power_payload(), elevation=32.0)
    assert set(frame["elevation"]) == {32.0}


# --- schema compatibility -----------------------------------------------------

def test_frame_matches_the_trained_schema_plus_extras():
    frame = to_frame(power_payload())
    assert list(frame.columns) == pw.SCHEMA_COLUMNS + pw.EXTRA_COLUMNS


def test_frame_is_sorted_ascending_by_date():
    payload = power_payload(n=4)
    reversed_params = {
        k: dict(reversed(list(v.items())))
        for k, v in payload["properties"]["parameter"].items()
    }
    payload["properties"]["parameter"] = reversed_params
    frame = to_frame(payload)
    assert frame["date_of_record"].is_monotonic_increasing


def test_n_stations_reporting_is_one_grid_cell_not_a_station_count():
    frame = to_frame(power_payload())
    assert set(frame["n_stations_reporting"]) == {1}


def test_missing_parameter_block_is_an_error_not_a_silent_empty_frame():
    with pytest.raises(pw.PowerError, match="properties.parameter"):
        to_frame({"header": {}})


def test_a_dropped_parameter_is_reported_by_name():
    payload = power_payload()
    del payload["properties"]["parameter"]["GWETROOT"]
    with pytest.raises(pw.PowerError, match="GWETROOT"):
        to_frame(payload)


# --- POWER's publication lag --------------------------------------------------

def test_trailing_unpublished_days_are_trimmed():
    """POWER runs ~2-3 days behind. Its not-yet-published rows arrive as
    real dates with every value filled, and must not read as zero rain."""
    payload = power_payload(
        n=5,
        overrides={p: {"20240104": -999.0, "20240105": -999.0} for p in pw.POWER_PARAMS},
    )
    frame = pw.trim_trailing_gap(to_frame(payload))
    assert len(frame) == 3
    assert frame["date_of_record"].iloc[-1] == pd.Timestamp("2024-01-03")


def test_interior_gaps_are_preserved_as_nan():
    """Only the trailing gap is POWER's lag; an interior hole is a genuine
    gap the feature code already represents as NaN."""
    payload = power_payload(n=5, overrides={p: {"20240103": -999.0} for p in pw.POWER_PARAMS})
    frame = pw.trim_trailing_gap(to_frame(payload))
    assert len(frame) == 5
    assert np.isnan(frame["rainfall"].iloc[2])


def test_data_end_reports_the_last_measured_day():
    payload = power_payload(n=4, overrides={p: {"20240104": -999.0} for p in pw.POWER_PARAMS})
    assert pw.data_end(to_frame(payload)) == pd.Timestamp("2024-01-03")


def test_data_end_is_none_when_nothing_was_published():
    payload = power_payload(n=3, overrides={p: {d: -999.0 for d in ("20240101", "20240102", "20240103")} for p in pw.POWER_PARAMS})
    assert pw.data_end(to_frame(payload)) is None


# --- caching / resumability ---------------------------------------------------

def test_request_targets_the_daily_point_endpoint_with_expected_params(monkeypatch):
    calls = install_get(monkeypatch, power_payload(end="20240105"))
    pw.fetch_power_daily("Kolkata", 22.57, 88.36, start="2024-01-01", end=dt.date(2024, 1, 5))
    assert calls[0]["url"] == pw.BASE_URL
    p = calls[0]["params"]
    assert p["community"] == "AG" and p["format"] == "JSON"
    assert p["start"] == "20240101" and p["end"] == "20240105"
    assert p["parameters"].split(",") == pw.POWER_PARAMS


def test_a_shard_that_already_covers_the_end_date_is_not_refetched(monkeypatch):
    calls = install_get(monkeypatch, power_payload(end="20240105"))
    for _ in range(3):
        pw.fetch_power_daily("Kolkata", 22.57, 88.36, start="2024-01-01", end=dt.date(2024, 1, 5))
    assert len(calls) == 1


def test_a_shard_that_stops_short_of_the_end_date_is_refetched(monkeypatch):
    """This is what makes the backfill resumable AND extendable: coverage,
    not cache age, decides whether a call happens."""
    install_get(monkeypatch, power_payload(end="20240105"))
    pw.fetch_power_daily("Kolkata", 22.57, 88.36, start="2024-01-01", end=dt.date(2024, 1, 5))
    calls = install_get(monkeypatch, power_payload(n=10, end="20240110"))
    pw.fetch_power_daily("Kolkata", 22.57, 88.36, start="2024-01-01", end=dt.date(2024, 1, 10))
    assert len(calls) == 1


def test_each_district_gets_its_own_shard(monkeypatch):
    calls = install_get(monkeypatch, power_payload(end="20240105"))
    pw.fetch_power_daily("Kolkata", 22.57, 88.36, start="2024-01-01", end=dt.date(2024, 1, 5))
    pw.fetch_power_daily("Jaisalmer", 26.91, 70.92, start="2024-01-01", end=dt.date(2024, 1, 5))
    assert len(calls) == 2
    assert pw._shard_path("Kolkata") != pw._shard_path("Jaisalmer")


def test_stale_shard_is_served_when_power_is_unreachable(monkeypatch):
    install_get(monkeypatch, power_payload(end="20240105"))
    pw.fetch_power_daily("Kolkata", 22.57, 88.36, start="2024-01-01", end=dt.date(2024, 1, 5))
    calls = install_get(monkeypatch, exc=requests.ConnectionError("down"))
    payload = pw.fetch_power_daily("Kolkata", 22.57, 88.36, start="2024-01-01", end=dt.date(2024, 1, 10))
    assert payload["header"]["end"] == "20240105"  # stale, but usable
    assert len(calls) == pw.MAX_RETRIES


def test_raises_when_unreachable_and_nothing_cached(monkeypatch):
    install_get(monkeypatch, exc=requests.ConnectionError("down"))
    with pytest.raises(pw.PowerError, match="no cached"):
        pw.fetch_power_daily("Kolkata", 22.57, 88.36, start="2024-01-01", end=dt.date(2024, 1, 5))


def test_rate_limit_status_is_retried(monkeypatch):
    """POWER documents 429 but never states its limit, so throttling must
    be treated as transient rather than fatal."""
    calls = install_get(monkeypatch, {}, status=429)
    with pytest.raises(pw.PowerError):
        pw.fetch_power_daily("Kolkata", 22.57, 88.36, start="2024-01-01", end=dt.date(2024, 1, 5))
    assert len(calls) == pw.MAX_RETRIES
