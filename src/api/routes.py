"""HTTP endpoints.

Every handler that predicts or fetches is a plain `def`, deliberately not
`async def`: predict_live does blocking `requests` calls (with retry backoff,
worst case ~30s) plus CPU-bound inference, and FastAPI runs plain `def`
handlers in a threadpool. An `async def` here would freeze the event loop for
every other request while one district's forecast is in flight.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from src.api.schemas import (
    Agreement,
    AdvisoryResponse,
    District,
    DistrictsResponse,
    ForecastResponse,
    HealthResponse,
    HistoricalPoint,
    HistoricalResponse,
    MetricsResponse,
    MlModelOutput,
    OpenMeteoDaily,
    OpenMeteoForecast,
)
from src.api.state import ServiceState
from src.forecasting.agreement import compute_agreement, lookup_threshold
from src.forecasting.district_registry import DistrictConfig, get_district_config, list_district_configs
from src.forecasting.open_meteo import MAX_HISTORICAL_DAYS, fetch_historical_weather
from src.llm import openrouter

logger = logging.getLogger(__name__)
router = APIRouter()

ADVISORY_CACHE_TTL_SECONDS = 3600  # matches the Open-Meteo cache, so a cached advisory never outlives its data by much
HISTORICAL_SOURCE = "open-meteo.com archive API (reanalysis, not rain-gauge observations)"
HISTORICAL_UNITS = {
    "rainfall": "mm",
    "avg_temp": "degC",
    "min_temp": "degC",
    "max_temp": "degC",
    "wind_speed": "m/s (10m, daily mean)",
    "air_pressure": "hPa (mean sea level)",
    "relative_humidity": "% (2m, daily mean)",
}


def _svc(request: Request) -> ServiceState:
    return request.app.state.svc


def _clean(value: Any) -> Any:
    """NaN/inf -> None. Open-Meteo nulls arrive as NaN once they pass through
    pandas. Pydantic would serialise NaN as null anyway, but our own logic
    compares against None (counting missing days, and the "no usable
    precipitation" guard in _build_forecast), and NaN is not None."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _resolve(svc: ServiceState, name: str) -> DistrictConfig:
    # get_district_config's KeyError message embeds all 316 names; never
    # forward it to a client.
    try:
        cfg = get_district_config(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown district '{name}'. See GET /districts for valid names.") from None
    if cfg.district in svc.excluded_districts:
        raise HTTPException(
            status_code=404,
            detail=f"District '{cfg.district}' cannot be forecast ({svc.excluded_districts[cfg.district]}).",
        )
    if cfg.district not in svc.trained_districts:
        raise HTTPException(
            status_code=404,
            detail=f"District '{cfg.district}' is not covered by the served model. See GET /districts.",
        )
    return cfg


def _build_forecast(svc: ServiceState, cfg: DistrictConfig) -> ForecastResponse:
    raw = svc.predictor.predict_live(cfg.district)
    ml, om = raw["local_model"], raw["open_meteo_forecast"]

    daily = [
        OpenMeteoDaily(
            time=str(d["time"]),
            precipitation_sum=_clean(d.get("precipitation_sum")),
            precipitation_probability_max=_clean(d.get("precipitation_probability_max")),
        )
        for d in om["daily"]
    ]
    # predict_live sums with NaN skipped, so a forecast with no usable values
    # would read as a confident 0.0mm rather than as missing data.
    if not any(d.precipitation_sum is not None for d in daily):
        raise HTTPException(status_code=503, detail="Open-Meteo returned no usable precipitation forecast.")

    month = pd.Timestamp(raw["as_of_date"]).month
    threshold = lookup_threshold(svc.predictor.preprocessor.risk_threshold_table_, cfg.district, month)
    agreement = compute_agreement(ml["predicted_rainfall_mm"], om["total_precipitation_sum_mm"], threshold)

    return ForecastResponse(
        district=cfg.district,
        state=cfg.state,
        as_of_date=raw["as_of_date"],
        ml_model=MlModelOutput(**ml),
        open_meteo_forecast=OpenMeteoForecast(
            source=om["source"],
            forecast_days=om["forecast_days"],
            total_precipitation_sum_mm=om["total_precipitation_sum_mm"],
            mean_daily_precipitation_probability_percent=_clean(om["mean_daily_precipitation_probability_percent"]),
            daily=daily,
        ),
        agreement=Agreement(**agreement),
        note=raw["note"],
    )


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    svc = _svc(request)
    md = svc.predictor.metadata
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=svc.model_source,
        model_version=md.get("model_version", "unknown"),
        classifier=md.get("classifier_model_name", "unknown"),
        n_districts=len(svc.trained_districts),
        excluded_districts=svc.excluded_districts,
        openrouter_configured=openrouter.is_configured(),
    )


@router.get("/districts", response_model=DistrictsResponse, tags=["meta"])
def districts(request: Request, q: Optional[str] = Query(default=None, description="Case-insensitive substring of district or state")) -> DistrictsResponse:
    svc = _svc(request)
    needle = q.strip().casefold() if q else None
    items = [
        District(name=c.district, state=c.state, latitude=c.latitude, longitude=c.longitude, elevation=c.elevation)
        for c in list_district_configs()
        if c.district in svc.trained_districts
        and (needle is None or needle in c.district.casefold() or needle in c.state.casefold())
    ]
    items.sort(key=lambda d: d.name)
    return DistrictsResponse(count=len(items), districts=items)


@router.get("/forecast/{district}", response_model=ForecastResponse, tags=["forecast"])
def forecast(district: str, request: Request) -> ForecastResponse:
    """ML rainfall-risk estimate plus Open-Meteo's own forecast and how well
    they agree. No LLM involved, so this is the fast, always-available call."""
    svc = _svc(request)
    return _build_forecast(svc, _resolve(svc, district))


@router.get("/historical/{district}", response_model=HistoricalResponse, tags=["data"])
def historical(
    district: str,
    request: Request,
    days: int = Query(default=90, ge=1, le=MAX_HISTORICAL_DAYS),
) -> HistoricalResponse:
    """The last `days` complete days of weather for a district, fetched live
    from Open-Meteo's archive API (cached for an hour). Ends yesterday: today
    is excluded because its value is still partly a forecast."""
    svc = _svc(request)
    cfg = _resolve(svc, district)
    weather = fetch_historical_weather(cfg.latitude, cfg.longitude, days)

    points = [
        HistoricalPoint(
            date=str(r["time"]),
            rainfall=_clean(r.get("precipitation_sum")),
            avg_temp=_clean(r.get("temperature_2m_mean")),
            min_temp=_clean(r.get("temperature_2m_min")),
            max_temp=_clean(r.get("temperature_2m_max")),
            wind_speed=_clean(r.get("wind_speed_10m_mean")),
            air_pressure=_clean(r.get("pressure_msl_mean")),
            relative_humidity=_clean(r.get("relative_humidity_2m_mean")),
        )
        for r in weather.to_dict("records")
    ]
    # Days Open-Meteo has not filled yet come back all-null. Trim only the
    # trailing ones so data_end is honest; interior nulls stay visible as nulls.
    while points and all(v is None for k, v in points[-1].model_dump().items() if k != "date"):
        points.pop()
    if not points:
        raise HTTPException(status_code=503, detail="Open-Meteo returned no data for this window.")

    data_end = points[-1].date
    return HistoricalResponse(
        district=cfg.district,
        state=cfg.state,
        source=HISTORICAL_SOURCE,
        requested_days=days,
        n_points=len(points),
        data_start=points[0].date,
        data_end=data_end,
        missing_rainfall_days=sum(1 for p in points if p.rainfall is None),
        units=HISTORICAL_UNITS,
        note=(
            f"Fetched live from Open-Meteo; complete days through {data_end}. Values are reanalysis, not "
            "rain-gauge observations, so they can differ from the dataset the model was trained on. The most "
            "recent days are preliminary and may be revised. Null means no value was returned for that day."
        ),
        data=points,
    )


@router.get("/model/metrics", response_model=MetricsResponse, tags=["meta"])
def model_metrics(
    request: Request,
    district: Optional[str] = Query(default=None, description="Optional: per-district test metrics (only a sample of districts is available)"),
) -> MetricsResponse:
    """Held-out test performance of the served model against the climatology
    baseline it has to beat."""
    svc = _svc(request)
    ev = svc.eval_results
    if not ev:
        raise HTTPException(status_code=503, detail="Evaluation results are not available for the served model.")
    md = svc.predictor.metadata
    sample: dict[str, Any] = ev.get("per_district_sample", {})

    district_name: Optional[str] = None
    district_metrics: Optional[dict[str, Any]] = None
    if district:
        cfg = _resolve(svc, district)
        district_name = cfg.district
        district_metrics = sample.get(cfg.district)
        if district_metrics is None:
            raise HTTPException(
                status_code=404,
                detail=f"No per-district metrics for '{cfg.district}'. Available for: {sorted(sample)}.",
            )

    return MetricsResponse(
        model_version=md.get("model_version", "unknown"),
        scope=md.get("scope", "unknown"),
        n_districts_trained=md.get("n_districts_trained", len(svc.trained_districts)),
        classifier_model=md.get("classifier_model_name", "unknown"),
        regressor_model=md.get("regressor_model_name", "unknown"),
        train_date_range=md.get("train_date_range", []),
        val_date_range=md.get("val_date_range", []),
        test_date_range=md.get("test_date_range", []),
        test_row_counts=ev.get("row_counts", {}),
        classifier=ev["classifier"],
        baseline=ev["baseline"],
        regressor=ev["regressor"],
        baseline_regression=ev["baseline_regression"],
        district=district_name,
        district_metrics=district_metrics,
        available_district_metrics=sorted(sample),
        note=(
            "Pooled metrics are over all districts' held-out test weeks. Performance varies a lot by district "
            "(see district_metrics). The baseline uses only each district's historical month-of-year rate, so "
            "the model is only adding value where it beats it."
        ),
    )


def _reliability_context(svc: ServiceState) -> Optional[dict[str, Any]]:
    ev = svc.eval_results
    if not ev:
        return None
    c, b = ev["classifier"], ev["baseline"]
    return {
        "scope": f"pooled over {len(svc.trained_districts)} districts on held-out test data",
        "model_roc_auc": round(c["roc_auc"], 3),
        "climatology_baseline_roc_auc": round(b["roc_auc"], 3),
        "model_pr_auc": round(c["pr_auc"], 3),
        "climatology_baseline_pr_auc": round(b["pr_auc"], 3),
        "share_of_test_weeks_that_were_unusually_dry": round(c["positive_rate"], 3),
        "note": "A ROC-AUC of 0.5 is chance. Skill varies substantially between districts.",
    }


def _llm_payload(forecast: ForecastResponse, svc: ServiceState) -> dict[str, Any]:
    agreement = forecast.agreement.model_dump()
    agreement.pop("note", None)
    return {
        "district": forecast.district,
        "state": forecast.state,
        "as_of_date": forecast.as_of_date,
        "field_definitions": {
            "ml_model.rainfall_probability": "probability the next 7 days are unusually dry for this district and time of year",
            "ml_model.predicted_rainfall_mm": "the model's expected total rainfall over the next 7 days",
            "open_meteo_forecast.total_precipitation_sum_mm": "Open-Meteo's forecast total rainfall over the next 7 days",
            "agreement.threshold_mm": "7-day rainfall below this counts as insufficient for this district and month",
        },
        "ml_model": forecast.ml_model.model_dump(include={"rainfall_probability", "risk_level", "predicted_rainfall_mm"}),
        "open_meteo_forecast": forecast.open_meteo_forecast.model_dump(),
        "agreement": agreement,
        "model_reliability": _reliability_context(svc),
    }


@router.get("/advisory/{district}", response_model=AdvisoryResponse, tags=["forecast"])
def advisory(district: str, request: Request) -> AdvisoryResponse:
    """Everything in /forecast plus an LLM-written structured advisory. If the
    LLM is unavailable this still returns 200 with the full numeric forecast
    and `advisory: null`, so the numbers never depend on the narrative."""
    svc = _svc(request)
    cfg = _resolve(svc, district)
    forecast_data = _build_forecast(svc, cfg)

    if not openrouter.is_configured():
        return AdvisoryResponse(
            forecast=forecast_data,
            llm_error="OpenRouter is not configured (set OPENROUTER_API_KEY and OPENROUTER_MODEL).",
        )

    key = (cfg.district, forecast_data.as_of_date)
    cached = svc.advisory_cache.get(key)
    if cached and time.time() - cached[0] < ADVISORY_CACHE_TTL_SECONDS:
        return AdvisoryResponse(
            forecast=forecast_data, advisory=cached[1], unsupported_numbers=cached[2], llm_model=openrouter.OPENROUTER_MODEL
        )

    try:
        result, unsupported = openrouter.generate_advisory(_llm_payload(forecast_data, svc))
    except openrouter.OpenRouterError as exc:
        logger.warning("Advisory generation failed for %s: %s", cfg.district, exc)
        return AdvisoryResponse(forecast=forecast_data, llm_error=str(exc)[:300])

    svc.advisory_cache[key] = (time.time(), result, unsupported)
    return AdvisoryResponse(
        forecast=forecast_data, advisory=result, unsupported_numbers=unsupported, llm_model=openrouter.OPENROUTER_MODEL
    )
