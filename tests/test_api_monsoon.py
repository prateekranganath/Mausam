"""API tests for the endpoints added for onset, active/break, climate
context, point forecasts and crop advice.

Everything upstream is stubbed: these pin the HTTP contract and the
routing, not the science (which tests/test_onset.py, test_active_break.py
and test_crop_advisory.py cover).
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import routes
from src.api.main import app
from src.climate import indices as climate
from src.data import history as history_module

from tests.test_api import LIVE, EVAL, StubPredictor, _install


def synthetic_history(district="Kolkata", state="WB", years=range(2015, 2027), seed=1):
    """A district record with a clear monsoon season and a June onset."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-09-19", freq="D")
    in_season = dates.month.isin((6, 7, 8, 9))
    rainfall = rng.gamma(2.0, np.where(in_season, 6.0, 0.3))
    frame = pd.DataFrame(
        {
            "date_of_record": dates,
            "district": district,
            "state": state,
            "rainfall": rainfall,
            "soil_wetness_root_zone": np.where(in_season, 0.55, 0.2),
            "soil_wetness_surface": np.where(in_season, 0.5, 0.15),
            "source": "nasa-power",
        }
    )
    return frame


SNAPSHOT = {
    "enso": {
        "name": "ONI (Oceanic Nino Index)", "value": 1.8, "as_of": "2026-07-01",
        "phase": "el_nino", "publication_lag_days": 90, "source": "cpc",
    },
    "iod": {
        "name": "DMI (IOD Dipole Mode Index)", "value": 0.146, "as_of": "2026-05-01",
        "phase": "neutral", "publication_lag_days": 120, "source": "psl",
    },
    "mjo": {
        "name": "MJO RMM amplitude", "value": 0.81, "as_of": "2026-09-17",
        "phase": "weak (phase not meaningful)", "publication_lag_days": 3,
        "source": "iri", "mjo_phase": 8,
    },
}


@pytest.fixture
def client(monkeypatch):
    svc = _install()
    monkeypatch.setattr(
        history_module, "district_history", lambda cfg, bridge=True: synthetic_history(cfg.district, cfg.state)
    )
    monkeypatch.setattr(climate, "current_snapshot", lambda: {k: dict(v) for k, v in SNAPSHOT.items()})
    svc.history_cache.clear()
    return TestClient(app)


# --- climate context ----------------------------------------------------------

def test_climate_context_returns_all_three_indices(client):
    body = client.get("/climate/context").json()
    assert {"enso", "iod", "mjo"} <= set(body)
    assert body["enso"]["phase"] == "el_nino"
    assert body["mjo_phase"] == 8


def test_climate_context_exposes_each_index_staleness(client):
    """ONI and DMI lag by months. Serving them without as_of would imply a
    freshness they do not have."""
    body = client.get("/climate/context").json()
    assert body["enso"]["as_of"] == "2026-07-01"
    assert body["iod"]["publication_lag_days"] == 120


def test_climate_context_says_it_is_not_a_model_input(client):
    body = client.get("/climate/context").json()
    assert "not inputs" in body["note"]


def test_climate_context_503_when_feeds_are_down(client, monkeypatch):
    def boom():
        raise climate.ClimateIndexError("feeds down")

    monkeypatch.setattr(climate, "current_snapshot", boom)
    assert client.get("/climate/context").status_code == 503


def test_a_broken_climate_feed_does_not_break_the_forecast(client, monkeypatch):
    """Context is decoration; it must never gate the numbers."""
    def boom():
        raise climate.ClimateIndexError("feeds down")

    monkeypatch.setattr(climate, "current_snapshot", boom)
    assert client.get("/forecast/Kolkata").status_code == 200


# --- onset --------------------------------------------------------------------

def test_onset_returns_a_status_and_provenance(client):
    body = client.get("/monsoon/onset/Kolkata").json()
    assert body["district"] == "Kolkata" and body["state"] == "WB"
    assert body["status"] in {
        "outside_season", "pre_onset", "onset_likely", "onset_confirmed",
        "post_onset", "no_onset_detected",
    }
    assert body["data"]["n_days"] > 3000
    assert body["data"]["data_start"] == "2015-01-01"


def test_onset_response_disclaims_imd_parity(client):
    """A user must not be able to read this as an IMD declaration."""
    body = client.get("/monsoon/onset/Kolkata").json()
    assert "NOT IMD" in body["not_imd_criterion"]
    assert "925 hPa" in body["not_imd_criterion"]


def test_onset_accepts_an_explicit_season(client):
    body = client.get("/monsoon/onset/Kolkata", params={"season": "northeast"}).json()
    assert body["season"] == "northeast"


def test_onset_rejects_an_unknown_season(client):
    r = client.get("/monsoon/onset/Kolkata", params={"season": "winter"})
    assert r.status_code == 422
    assert "southwest" in r.json()["detail"]


def test_onset_404s_for_an_unserved_district(client):
    assert client.get("/monsoon/onset/Atlantis").status_code == 404


def test_onset_does_not_leak_the_district_list_in_errors(client):
    """_resolve exists partly so a 404 never echoes all 316 names."""
    detail = client.get("/monsoon/onset/Atlantis").json()["detail"]
    assert "Thiruvananthapuram" not in detail


# --- active / break -----------------------------------------------------------

def test_phase_returns_a_valid_phase_and_recent_window(client):
    body = client.get("/monsoon/phase/Kolkata").json()
    assert body["monsoon_phase"] in {"active", "break", "normal", "not_applicable"}
    assert len(body["recent_30_days"]) == 30
    assert body["state"] == "WB"


def test_phase_states_its_deviation_from_the_published_method(client):
    body = client.get("/monsoon/phase/Kolkata").json()
    assert "core zone" in body["caveats"]
    assert "Rajeevan" in body["method"]


def test_phase_response_contains_no_nan(client):
    """NaN is not valid JSON and would break a strict client."""
    import json

    raw = client.get("/monsoon/phase/Kolkata").text
    assert "NaN" not in raw
    json.loads(raw)


def test_phase_404s_for_an_unserved_district(client):
    assert client.get("/monsoon/phase/Atlantis").status_code == 404


# --- point forecast -----------------------------------------------------------

def test_point_route_is_not_captured_by_the_district_route(client):
    """Regression guard: /forecast/{district} is declared after this one on
    purpose. If the order is ever flipped, 'point' is read as a district
    name and this 404s."""
    r = client.get("/forecast/point", params={"lat": 22.57, "lon": 88.36})
    assert r.status_code == 200
    assert r.json()["resolved_district"] == "Kolkata"


def test_point_forecast_reports_distance_and_caveat(client):
    body = client.get("/forecast/point", params={"lat": 22.6, "lon": 88.4}).json()
    assert body["distance_km"] < 20
    assert "Not validated below district level" in body["caveat"]
    assert "district" in body["caveat"]


def test_point_forecast_rejects_coordinates_outside_india(client):
    for lat, lon in [(48.0, 77.0), (20.0, 120.0), (-5.0, 77.0)]:
        r = client.get("/forecast/point", params={"lat": lat, "lon": lon})
        assert r.status_code == 422
        assert "outside India" in r.json()["detail"]


def test_point_forecast_refuses_points_far_from_any_district(client):
    """Better to refuse than to stretch one district's climatology across
    hundreds of kilometres."""
    r = client.get("/forecast/point", params={"lat": 15.0, "lon": 69.0})
    assert r.status_code == 422
    assert "beyond the" in r.json()["detail"]


def test_point_forecast_still_carries_the_full_forecast(client):
    body = client.get("/forecast/point", params={"lat": 22.57, "lon": 88.36}).json()
    assert body["forecast"]["ml_model"]["risk_level"] in {"LOW", "MODERATE", "HIGH"}
    assert body["forecast"]["agreement"]["threshold_mm"] is not None


# --- crop advisory ------------------------------------------------------------

def test_crops_endpoint_lists_the_calendar(client):
    body = client.get("/crops").json()
    assert body["count"] >= 8
    assert any(c["key"] == "rice_transplanted" for c in body["crops"])


def test_crop_advisory_returns_stage_and_recommendations(client):
    body = client.get(
        "/advisory/crop/Kolkata", params={"crop": "rice_transplanted", "sowing_date": "2026-06-20"}
    ).json()
    assert body["district"] == "Kolkata"
    assert body["growth_stage"] is not None
    assert body["recommendations"], "an advisory must never be empty"
    for rec in body["recommendations"]:
        assert rec["rule_id"] and rec["action"]


def test_crop_advisory_works_with_no_sowing_date(client):
    """The unsown case is the most decision-relevant moment in the season."""
    body = client.get("/advisory/crop/Kolkata", params={"crop": "maize"}).json()
    assert body["sowing_date"] is None
    assert body["days_since_sowing"] is None
    assert body["recommendations"]


def test_crop_advisory_rejects_an_unknown_crop(client):
    r = client.get("/advisory/crop/Kolkata", params={"crop": "quinoa"})
    assert r.status_code == 422
    assert "Unknown crop" in r.json()["detail"]


def test_crop_advisory_rejects_a_future_sowing_date(client):
    r = client.get("/advisory/crop/Kolkata", params={"crop": "maize", "sowing_date": "2099-06-01"})
    assert r.status_code == 422


def test_crop_advisory_always_carries_the_disclaimer(client):
    body = client.get("/advisory/crop/Kolkata", params={"crop": "maize"}).json()
    assert "Krishi Vigyan Kendra" in body["disclaimer"]
    assert body["rules_version"]
    assert body["sources"]


def test_crop_advisory_survives_missing_monsoon_context(client, monkeypatch):
    """Advice premised on the forecast alone is still worth returning."""
    def boom(cfg, bridge=True):
        raise RuntimeError("history source down")

    monkeypatch.setattr(history_module, "district_history", boom)
    app.state.svc.history_cache.clear()
    r = client.get("/advisory/crop/Kolkata", params={"crop": "maize"})
    assert r.status_code == 200
    assert r.json()["recommendations"]


def test_crop_advisory_404s_for_an_unserved_district(client):
    assert client.get("/advisory/crop/Atlantis", params={"crop": "maize"}).status_code == 404


# --- history caching ----------------------------------------------------------

def test_district_history_is_loaded_once_per_district(client, monkeypatch):
    calls = []

    def counting(cfg, bridge=True):
        calls.append(cfg.district)
        return synthetic_history(cfg.district, cfg.state)

    monkeypatch.setattr(history_module, "district_history", counting)
    app.state.svc.history_cache.clear()

    client.get("/monsoon/onset/Kolkata")
    client.get("/monsoon/phase/Kolkata")
    client.get("/monsoon/onset/Kolkata")
    assert calls == ["Kolkata"], "11 years of daily data must not be re-fetched per request"
