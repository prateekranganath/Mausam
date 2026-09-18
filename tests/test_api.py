import copy
import math
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import routes
from src.api.main import app
from src.api.state import ServiceState
from src.forecasting.open_meteo import OpenMeteoError
from src.llm.openrouter import Advisory, OpenRouterError

LIVE = {
    "district": "Kolkata",
    "state": "WB",
    "as_of_date": "2026-09-18",
    "local_model": {
        "district": "Kolkata",
        "forecast_horizon_days": 7,
        "rainfall_probability": 0.4959,
        "risk_level": "MODERATE",
        "predicted_rainfall_mm": 99.16,
        "as_of_date": "2026-09-18",
        "model": "xgboost",
        "model_version": "0.1.0",
    },
    "open_meteo_forecast": {
        "source": "open-meteo.com (live forecast, not a historical observation)",
        "forecast_days": 2,
        "total_precipitation_sum_mm": 95.7,
        "mean_daily_precipitation_probability_percent": 94.1,
        "daily": [
            {"time": "2026-09-19", "precipitation_sum": 50.0, "precipitation_probability_max": 90},
            {"time": "2026-09-20", "precipitation_sum": 45.7, "precipitation_probability_max": 98},
        ],
    },
    "note": "test note",
}

EVAL = {
    "classifier": {"roc_auc": 0.754, "pr_auc": 0.379, "positive_rate": 0.21},
    "baseline": {"roc_auc": 0.692, "pr_auc": 0.299},
    "regressor": {"mae": 23.6},
    "baseline_regression": {"mae": 23.4},
    "row_counts": {"train": 10, "val": 5, "test": 5},
    "per_district_sample": {"Kolkata": {"n_test_rows": 209, "classifier": {"roc_auc": 0.838}}},
}


class StubPredictor:
    def __init__(self, live=None, exc=None):
        table = pd.Series(
            [61.06], index=pd.MultiIndex.from_tuples([("Kolkata", 9)], names=["district", "month"])
        )
        self.metadata = {
            "model_version": "0.1.0",
            "classifier_model_name": "xgboost",
            "regressor_model_name": "xgboost",
            "scope": "all-india (pooled, multi-district)",
            "n_districts_trained": 2,
            "train_date_range": ["2021-01-01", "2023-06-30"],
            "val_date_range": ["2023-07-01", "2024-06-30"],
            "test_date_range": ["2024-07-01", "2025-02-10"],
        }
        self.preprocessor = SimpleNamespace(risk_threshold_table_=table)
        self._live, self._exc = live if live is not None else LIVE, exc
        self.calls = 0

    def predict_live(self, district):
        self.calls += 1
        if self._exc:
            raise self._exc
        return copy.deepcopy(self._live)


def _install(predictor=None, eval_results=EVAL, trained=("Kolkata", "Jaisalmer"), excluded=None):
    app.state.svc = ServiceState(
        predictor=predictor or StubPredictor(),
        model_source="stub",
        trained_districts=frozenset(trained),
        excluded_districts=dict(excluded or {}),
        eval_results=eval_results,
    )
    return app.state.svc


@pytest.fixture
def client():
    _install()
    return TestClient(app)


# --- health / districts ------------------------------------------------------------

def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["n_districts"] == 2


def test_districts_lists_only_trained_and_filters(client):
    body = client.get("/districts").json()
    assert body["count"] == 2
    assert [d["name"] for d in body["districts"]] == ["Jaisalmer", "Kolkata"]  # sorted
    assert client.get("/districts", params={"q": "wb"}).json()["count"] == 1  # state match, case-insensitive
    assert client.get("/districts", params={"q": "zzz"}).json()["count"] == 0


# --- districts without training statistics ---------------------------------------

def _predictor_with_coverage(months_by_district: dict[str, set[int]], all_listed: list[str]):
    clim_idx = pd.MultiIndex.from_tuples(
        [(d, m) for d, ms in months_by_district.items() for m in sorted(ms)], names=["district", "month"]
    )
    thr_idx = pd.MultiIndex.from_tuples(
        [(d, m) for d in months_by_district for m in range(1, 13)], names=["district", "month"]
    )
    return SimpleNamespace(
        metadata={"districts": all_listed},
        preprocessor=SimpleNamespace(
            climatology_table_=pd.DataFrame({"climatology_mean_rainfall_month": 1.0}, index=clim_idx),
            risk_threshold_table_=pd.Series(1.0, index=thr_idx),
        ),
    )


def test_servable_districts_requires_every_calendar_month_and_explains_exclusions():
    from src.api.state import servable_districts

    predictor = _predictor_with_coverage(
        {
            "Kolkata": set(range(1, 13)),
            "Jaisalmer": set(range(1, 13)),
            "Bathinda": set(range(1, 13)) - {7},  # trained on every month but July
        },
        all_listed=["Kolkata", "Jaisalmer", "Bathinda", "Raisen"],  # Raisen: no training rows at all
    )
    servable, excluded = servable_districts(predictor)
    assert servable == frozenset({"Kolkata", "Jaisalmer"})
    assert set(excluded) == {"Bathinda", "Raisen"}
    assert "[7]" in excluded["Bathinda"]  # says WHICH month is missing
    assert "begin after" in excluded["Raisen"]


def test_excluded_district_gets_a_specific_404_with_its_reason_and_is_not_listed():
    _install(trained=("Kolkata",), excluded={"Agra": "incomplete training data: no observations for calendar month(s) [7]"})
    c = TestClient(app)
    r = c.get("/forecast/Agra")
    assert r.status_code == 404
    assert "[7]" in r.json()["detail"]
    assert [d["name"] for d in c.get("/districts").json()["districts"]] == ["Kolkata"]
    assert list(c.get("/health").json()["excluded_districts"]) == ["Agra"]


# --- forecast -----------------------------------------------------------------

def test_forecast_shape_and_agreement(client):
    r = client.get("/forecast/Kolkata")
    assert r.status_code == 200
    body = r.json()
    assert body["ml_model"]["predicted_rainfall_mm"] == 99.16  # renamed from local_model at the boundary
    assert "local_model" not in body
    a = body["agreement"]
    assert a["threshold_mm"] == 61.06
    assert a["ml_implies_insufficient"] is False and a["open_meteo_implies_insufficient"] is False
    assert a["sources_agree"] is True
    assert a["difference_mm"] == pytest.approx(3.46)


def test_forecast_is_case_insensitive(client):
    assert client.get("/forecast/kolkata").status_code == 200


def test_unknown_district_404_without_leaking_district_list(client):
    r = client.get("/forecast/NotARealDistrict")
    assert r.status_code == 404
    assert len(r.json()["detail"]) < 200
    assert "Agra" not in r.json()["detail"]


def test_registered_but_untrained_district_is_404(client):
    # Agra exists in the registry but isn't in this model's trained set
    assert client.get("/forecast/Agra").status_code == 404


def test_nan_from_open_meteo_becomes_null_not_a_500():
    live = copy.deepcopy(LIVE)
    live["open_meteo_forecast"]["daily"][0]["precipitation_probability_max"] = math.nan
    _install(StubPredictor(live=live))
    r = TestClient(app).get("/forecast/Kolkata")
    assert r.status_code == 200
    assert r.json()["open_meteo_forecast"]["daily"][0]["precipitation_probability_max"] is None


def test_forecast_with_no_usable_precipitation_is_503_not_a_confident_zero():
    live = copy.deepcopy(LIVE)
    for d in live["open_meteo_forecast"]["daily"]:
        d["precipitation_sum"] = math.nan
    live["open_meteo_forecast"]["total_precipitation_sum_mm"] = 0.0  # what the summariser reports for all-NaN
    _install(StubPredictor(live=live))
    assert TestClient(app).get("/forecast/Kolkata").status_code == 503


def test_open_meteo_outage_maps_to_503_with_friendly_message():
    _install(StubPredictor(exc=OpenMeteoError("boom, internal detail")))
    r = TestClient(app).get("/forecast/Kolkata")
    assert r.status_code == 503
    assert "internal detail" not in r.text


def test_missing_threshold_still_returns_forecast():
    live = copy.deepcopy(LIVE)
    live["as_of_date"] = "2026-01-05"  # month 1: no threshold in the stub table
    _install(StubPredictor(live=live))
    body = TestClient(app).get("/forecast/Kolkata").json()
    assert body["agreement"]["threshold_mm"] is None
    assert body["agreement"]["sources_agree"] is None


# --- historical (live Open-Meteo archive) ---------------------------------------------------

def _weather(n=5, start="2026-09-10", trailing_null=0):
    times = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, periods=n, freq="D")]
    df = pd.DataFrame(
        {
            "time": times,
            "precipitation_sum": [float(i) for i in range(n)],
            "temperature_2m_mean": 25.0,
            "temperature_2m_min": 20.0,
            "temperature_2m_max": 30.0,
            "wind_speed_10m_mean": 2.0,
            "pressure_msl_mean": 1010.0,
            "relative_humidity_2m_mean": 70.0,
        }
    )
    if trailing_null:
        df.loc[df.index[-trailing_null:], df.columns.drop("time")] = math.nan
    return df


@pytest.fixture
def fetch_calls(monkeypatch):
    calls = []

    def install(df):
        def fake(latitude, longitude, days):
            calls.append((latitude, longitude, days))
            return df.tail(days).reset_index(drop=True)

        monkeypatch.setattr(routes, "fetch_historical_weather", fake)

    install.calls = calls
    return install


def test_historical_maps_open_meteo_columns_and_reports_provenance(client, fetch_calls):
    from src.forecasting.district_registry import get_district_config

    fetch_calls(_weather(5))
    body = client.get("/historical/Kolkata", params={"days": 5}).json()
    cfg = get_district_config("Kolkata")
    assert fetch_calls.calls == [(cfg.latitude, cfg.longitude, 5)]  # this district's own coordinates
    assert body["n_points"] == 5 and body["requested_days"] == 5
    assert [p["date"] for p in body["data"]] == ["2026-09-10", "2026-09-11", "2026-09-12", "2026-09-13", "2026-09-14"]
    assert body["data_end"] == "2026-09-14"
    last = body["data"][-1]
    assert last["rainfall"] == 4.0 and last["relative_humidity"] == 70.0 and last["air_pressure"] == 1010.0
    assert "archive" in body["source"] and "not rain-gauge" in body["source"]
    assert body["units"]["wind_speed"].startswith("m/s")
    assert "reanalysis" in body["note"]


def test_historical_trims_only_trailing_all_null_days_so_data_end_is_honest(client, fetch_calls):
    fetch_calls(_weather(5, trailing_null=2))
    body = client.get("/historical/Kolkata", params={"days": 5}).json()
    assert body["requested_days"] == 5 and body["n_points"] == 3
    assert body["data_end"] == "2026-09-12"


def test_historical_keeps_interior_nulls_visible(client, fetch_calls):
    df = _weather(5)
    df.loc[2, "precipitation_sum"] = math.nan
    fetch_calls(df)
    body = client.get("/historical/Kolkata", params={"days": 5}).json()
    assert body["n_points"] == 5  # not dropped: the series stays continuous
    assert body["data"][2]["rainfall"] is None
    assert body["missing_rainfall_days"] == 1


def test_historical_all_null_window_is_503_not_an_empty_success(client, fetch_calls):
    fetch_calls(_weather(3, trailing_null=3))
    assert client.get("/historical/Kolkata", params={"days": 3}).status_code == 503


def test_historical_validates_days_bounds(client, fetch_calls):
    fetch_calls(_weather(5))
    assert client.get("/historical/Kolkata", params={"days": 0}).status_code == 422
    assert client.get("/historical/Kolkata", params={"days": 731}).status_code == 422
    assert client.get("/historical/Kolkata", params={"days": 730}).status_code == 200


def test_historical_open_meteo_outage_is_503_without_leaking_internals(client, monkeypatch):
    def down(latitude, longitude, days):
        raise OpenMeteoError("internal detail")

    monkeypatch.setattr(routes, "fetch_historical_weather", down)
    r = client.get("/historical/Kolkata")
    assert r.status_code == 503 and "internal detail" not in r.text


def test_historical_no_longer_depends_on_any_local_dataset_file(client, fetch_calls):
    assert not hasattr(routes, "ALL_INDIA_PROCESSED_CACHE")
    fetch_calls(_weather(3))
    assert client.get("/historical/Jaisalmer", params={"days": 3}).status_code == 200


# --- metrics ---------------------------------------------------------------------------------

def test_metrics_pooled_and_district(client):
    body = client.get("/model/metrics").json()
    assert body["classifier"]["roc_auc"] == 0.754 and body["baseline"]["roc_auc"] == 0.692
    assert body["available_district_metrics"] == ["Kolkata"]
    d = client.get("/model/metrics", params={"district": "kolkata"}).json()
    assert d["district"] == "Kolkata" and d["district_metrics"]["classifier"]["roc_auc"] == 0.838


def test_metrics_for_unsampled_district_is_404_and_lists_available(client):
    r = client.get("/model/metrics", params={"district": "Jaisalmer"})
    assert r.status_code == 404
    assert "Kolkata" in r.json()["detail"]


def test_metrics_503_when_eval_results_missing():
    _install(eval_results=None)
    assert TestClient(app).get("/model/metrics").status_code == 503


# --- advisory --------------------------------------------------------------------------------

def _advisory(**kw):
    base = dict(
        forecast_summary="s", rainfall_risk="LOW", confidence=0.5,
        key_factors=[], model_disagreement=[], advisory=["a"],
    )
    base.update(kw)
    return Advisory(**base)


def test_advisory_degrades_to_numbers_only_when_llm_unconfigured(client, monkeypatch):
    monkeypatch.setattr(routes.openrouter, "is_configured", lambda: False)
    r = client.get("/advisory/Kolkata")
    assert r.status_code == 200
    body = r.json()
    assert body["advisory"] is None and "not configured" in body["llm_error"]
    assert body["forecast"]["ml_model"]["predicted_rainfall_mm"] == 99.16  # numbers still delivered


def test_advisory_success_passes_through_unsupported_numbers(client, monkeypatch):
    monkeypatch.setattr(routes.openrouter, "is_configured", lambda: True)
    monkeypatch.setattr(routes.openrouter, "generate_advisory", lambda payload: (_advisory(), ["777"]))
    body = client.get("/advisory/Kolkata").json()
    assert body["advisory"]["rainfall_risk"] == "LOW"
    assert body["unsupported_numbers"] == ["777"]
    assert body["llm_error"] is None


def test_advisory_llm_failure_is_200_with_numbers_not_5xx(client, monkeypatch):
    monkeypatch.setattr(routes.openrouter, "is_configured", lambda: True)

    def fail(payload):
        raise OpenRouterError("provider down")

    monkeypatch.setattr(routes.openrouter, "generate_advisory", fail)
    r = client.get("/advisory/Kolkata")
    assert r.status_code == 200
    assert r.json()["advisory"] is None and "provider down" in r.json()["llm_error"]


def test_advisory_is_cached_per_district_and_date(client, monkeypatch):
    monkeypatch.setattr(routes.openrouter, "is_configured", lambda: True)
    calls = []

    def gen(payload):
        calls.append(1)
        return _advisory(), []

    monkeypatch.setattr(routes.openrouter, "generate_advisory", gen)
    client.get("/advisory/Kolkata")
    client.get("/advisory/Kolkata")
    assert len(calls) == 1


def test_failed_advisory_is_not_cached(client, monkeypatch):
    monkeypatch.setattr(routes.openrouter, "is_configured", lambda: True)
    attempts = iter([OpenRouterError("first fails"), (_advisory(), [])])
    calls = []

    def gen(payload):
        calls.append(1)
        outcome = next(attempts)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(routes.openrouter, "generate_advisory", gen)
    assert client.get("/advisory/Kolkata").json()["advisory"] is None
    assert client.get("/advisory/Kolkata").json()["advisory"] is not None
    assert len(calls) == 2


def test_llm_payload_contains_numbers_but_not_secrets_or_notes(client, monkeypatch):
    monkeypatch.setattr(routes.openrouter, "is_configured", lambda: True)
    seen = {}

    def gen(payload):
        seen.update(payload)
        return _advisory(), []

    monkeypatch.setattr(routes.openrouter, "generate_advisory", gen)
    client.get("/advisory/Kolkata")
    assert seen["ml_model"]["predicted_rainfall_mm"] == 99.16
    assert seen["agreement"]["threshold_mm"] == 61.06
    assert "note" not in seen["agreement"]
    assert seen["model_reliability"]["model_roc_auc"] == 0.754
