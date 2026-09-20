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
    AlertPreviewResponse,
    AlertSendResponse,
    ClimateContextResponse,
    CropAdvisoryResponse,
    CropsResponse,
    District,
    DistrictCoverageResponse,
    DistrictsResponse,
    ForecastResponse,
    ForecastSeriesPoint,
    ForecastSeriesResponse,
    HealthResponse,
    HistoricalPoint,
    HistoricalResponse,
    MetricsResponse,
    MlModelOutput,
    MonsoonPhaseResponse,
    OnsetResponse,
    OpenMeteoDaily,
    OpenMeteoForecast,
    PointForecastResponse,
)
from src.api.state import ServiceState
from src.advisory import engine as crop_engine
from src.alerts import telegram
from src.alerts.compose import compose_alert
from src.climate import indices as climate
from src.config import (
    MAX_POINT_DISTANCE_KM,
    TELEGRAM_MAX_SENDS_PER_WINDOW,
    TELEGRAM_THROTTLE_WINDOW_SECONDS,
)
from src.data.history import history_provenance, recent_soil_and_dryness
from src.forecasting.agreement import compute_agreement, lookup_threshold
from src.forecasting.district_registry import DistrictConfig, get_district_config, list_district_configs
from src.forecasting.open_meteo import MAX_HISTORICAL_DAYS, fetch_historical_weather
from src.llm import openrouter
from src.monsoon import active_break, onset as onset_module

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


# --------------------------------------------------------------------------
# Point forecast
# --------------------------------------------------------------------------

# India's bounding box, reused from the dataset cleaner so the API and the
# training pipeline agree on what counts as in-country.
_INDIA_LAT = (6.0, 38.0)
_INDIA_LON = (68.0, 98.0)
_EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@router.get("/forecast/point", response_model=PointForecastResponse, tags=["forecast"])
def forecast_point(
    request: Request,
    lat: float = Query(description="Latitude, within India"),
    lon: float = Query(description="Longitude, within India"),
) -> PointForecastResponse:
    """A forecast at an arbitrary coordinate, for finer-than-district use.

    HOW FAR THIS ACTUALLY GOES, stated plainly: weather is pulled from
    Open-Meteo's grid at the exact point (~2-11km), and latitude, longitude
    and elevation are already model features, so the inference is genuinely
    local. But the climatology and the risk threshold are the resolved
    DISTRICT's, because district means are the only level the model was
    ever fit at, and nothing here has been validated below district level.
    It is a grid-downscaled estimate, not a village forecast, and the
    response says so.

    Points further than MAX_POINT_DISTANCE_KM from any district centroid
    are refused rather than served by stretching one district's learned
    statistics across half a state.
    """
    svc = _svc(request)
    if not (_INDIA_LAT[0] <= lat <= _INDIA_LAT[1] and _INDIA_LON[0] <= lon <= _INDIA_LON[1]):
        raise HTTPException(
            status_code=422,
            detail=(
                f"({lat}, {lon}) is outside India. Expected latitude in {_INDIA_LAT} and "
                f"longitude in {_INDIA_LON}."
            ),
        )

    servable = [c for c in list_district_configs() if c.district in svc.trained_districts]
    if not servable:
        raise HTTPException(status_code=503, detail="No servable districts are loaded.")

    nearest = min(servable, key=lambda c: _haversine_km(lat, lon, c.latitude, c.longitude))
    distance = _haversine_km(lat, lon, nearest.latitude, nearest.longitude)
    if distance > MAX_POINT_DISTANCE_KM:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Nearest servable district ({nearest.district}) is {distance:.0f}km away, beyond the "
                f"{MAX_POINT_DISTANCE_KM:.0f}km limit. Its climatology and risk threshold would not "
                "meaningfully apply to this point."
            ),
        )

    # Same district identity and learned statistics, but Open-Meteo pulled
    # at the requested coordinate rather than the district centroid.
    point_cfg = DistrictConfig(
        district=nearest.district,
        state=nearest.state,
        latitude=lat,
        longitude=lon,
        elevation=nearest.elevation,
    )
    forecast_data = _build_forecast(svc, point_cfg)

    return PointForecastResponse(
        latitude=lat,
        longitude=lon,
        resolved_district=nearest.district,
        state=nearest.state,
        distance_km=round(distance, 1),
        forecast=forecast_data,
        caveat=(
            "Downscaled by weather grid only. Live weather is from Open-Meteo at this exact "
            f"coordinate, but the climatology, risk threshold and elevation are {nearest.district} "
            "district's, and the model was trained on district-mean data. Not validated below "
            "district level; do not read this as a village- or panchayat-level forecast."
        ),
    )


@router.get("/districts/coverage", response_model=DistrictCoverageResponse, tags=["meta"])
def districts_coverage(request: Request) -> DistrictCoverageResponse:
    """Coverage metadata for selectors, maps, and extension dashboards."""
    svc = _svc(request)
    return DistrictCoverageResponse(
        trained_count=len(svc.trained_districts),
        trained_districts=sorted(svc.trained_districts),
        excluded_districts=svc.excluded_districts,
    )


# NOTE ON ORDER: every literal route above MUST stay above this one.
# Starlette matches in declaration order, so if this parameterised route
# comes first it captures "point" as a district name and /forecast/point
# 404s with 'Unknown district'. There is a test pinning that case
# (tests/test_api_monsoon.py::test_point_route_is_not_captured_by_the_district_route).
@router.get("/forecast/{district}", response_model=ForecastResponse, tags=["forecast"])
def forecast(district: str, request: Request) -> ForecastResponse:
    """ML rainfall-risk estimate plus Open-Meteo's own forecast and how well
    they agree. No LLM involved, so this is the fast, always-available call."""
    svc = _svc(request)
    return _build_forecast(svc, _resolve(svc, district))


@router.get("/forecast/{district}/series", response_model=ForecastSeriesResponse, tags=["forecast"])
def forecast_series(district: str, request: Request) -> ForecastSeriesResponse:
    """Chart-ready daily forecast values with explicit units."""
    svc = _svc(request)
    data = _build_forecast(svc, _resolve(svc, district))
    return ForecastSeriesResponse(
        district=data.district,
        state=data.state,
        as_of_date=data.as_of_date,
        source=data.open_meteo_forecast.source,
        data=[
            ForecastSeriesPoint(
                time=point.time,
                rainfall_mm=point.precipitation_sum,
                precipitation_probability_percent=point.precipitation_probability_max,
            )
            for point in data.open_meteo_forecast.daily
        ],
    )


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


def _climate_context_safe() -> Optional[dict[str, Any]]:
    """Climate indices, or None if the feeds are unreachable.

    Context must never be able to fail a forecast: these are three
    third-party feeds that exist to add colour, not to gate the numbers.
    """
    try:
        return climate.current_snapshot()
    except climate.ClimateIndexError as exc:
        logger.warning("Climate context unavailable: %s", exc)
        return None


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
        # Seasonal-scale CONTEXT ONLY. These are deliberately not model
        # inputs (see README: the pooled model's 2.5-year training window
        # cannot support them), so the prompt must not let them read as
        # part of the prediction. They go through the payload rather than
        # into the prompt text because find_unsupported_numbers flags any
        # figure the model states that is not traceable to this dict.
        "climate_context": _climate_context_safe(),
        "climate_context_caveat": (
            "ENSO and IOD describe the season as a whole and are NOT inputs to the "
            "7-day model. Treat them as background, never as the basis for a specific "
            "rainfall number. Note each index's as_of date: several are months old."
        ),
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


# --------------------------------------------------------------------------
# Climate context (ENSO / IOD / MJO)
# --------------------------------------------------------------------------

@router.get("/climate/context", response_model=ClimateContextResponse, tags=["climate"])
def climate_context() -> ClimateContextResponse:
    """Current ENSO, IOD and MJO state.

    CONTEXT, NOT PREDICTION. None of these is an input to the rainfall
    model, and that was measured rather than assumed -- see
    scripts/run_climate_ablation.py and models/ablation_climate_indices.json.

    ONI and DMI fail structurally: over the pooled model's 2021-01..2023-06
    training window only 7.9% of validation ONI values and 15.8% of DMI
    values fall inside the range training ever covered (the 2020-2023 La
    Nina against the El Nino that followed), so there is nothing for the
    model to generalise from and validation ROC-AUC drops 0.042. MJO has no
    such problem -- 100% overlap, as its 30-60 day cycle implies -- but it
    did not improve validation ROC either. All three are therefore reported
    here as background and none is a model input.

    Every value carries its own as_of date because these feeds lag badly:
    measured 2026-09-20, ONI was 81 days behind and DMI 112.
    """
    try:
        snapshot = climate.current_snapshot()
    except climate.ClimateIndexError as exc:
        raise HTTPException(status_code=503, detail=f"Climate index feeds unavailable: {exc}") from None

    mjo = dict(snapshot["mjo"])
    mjo_phase = mjo.pop("mjo_phase", None)
    return ClimateContextResponse(
        enso=snapshot["enso"],
        iod=snapshot["iod"],
        mjo=mjo,
        mjo_phase=mjo_phase,
        note=(
            "These indices are background context for the season, not inputs to the 7-day "
            "rainfall model. Check each as_of date: ONI and DMI are typically months old."
        ),
    )


# --------------------------------------------------------------------------
# Monsoon onset and active/break
# --------------------------------------------------------------------------

@router.get("/monsoon/onset/{district}", response_model=OnsetResponse, tags=["monsoon"])
def monsoon_onset(
    district: str,
    request: Request,
    season: Optional[str] = Query(
        default=None,
        description="'southwest' or 'northeast'. Default: whichever delivers more of this district's rain.",
    ),
) -> OnsetResponse:
    """Has the monsoon arrived in this district yet?

    Reports LOCAL rainfall onset, which is not the same thing as an IMD
    onset declaration and measurably precedes it -- see `not_imd_criterion`
    in the response. `status` distinguishes `onset_likely` (rain has
    arrived, persistence not yet verifiable) from `onset_confirmed`; that
    distinction is the honest part and should be shown to users, not
    collapsed.
    """
    svc = _svc(request)
    cfg = _resolve(svc, district)
    if season is not None and season not in onset_module.SEASONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown season '{season}'. Use one of: {sorted(onset_module.SEASONS)}.",
        )

    history = svc.district_history(cfg)
    if history.empty:
        raise HTTPException(status_code=503, detail=f"No daily history available for {cfg.district}.")

    result = onset_module.onset_status(history, cfg.district, season=season)
    result["state"] = cfg.state
    result["data"] = history_provenance(history)
    return OnsetResponse(**result)


@router.get("/monsoon/phase/{district}", response_model=MonsoonPhaseResponse, tags=["monsoon"])
def monsoon_phase(district: str, request: Request) -> MonsoonPhaseResponse:
    """Is the monsoon currently active or in a break here?

    Only defined for June-September; outside those months the response is
    `not_applicable` rather than a number, because a standardised anomaly
    against a near-zero climatological mean is arithmetically valid and
    physically meaningless.
    """
    svc = _svc(request)
    cfg = _resolve(svc, district)

    history = svc.district_history(cfg)
    if history.empty:
        raise HTTPException(status_code=503, detail=f"No daily history available for {cfg.district}.")

    result = active_break.current_phase(history, cfg.district)
    result["state"] = cfg.state
    result["data"] = history_provenance(history)
    return MonsoonPhaseResponse(**result)


# --------------------------------------------------------------------------
# Crop advisory
# --------------------------------------------------------------------------

@router.get("/crops", response_model=CropsResponse, tags=["advisory"])
def crops() -> CropsResponse:
    """Crops the advisory engine has calendars for."""
    items = crop_engine.list_crops()
    return CropsResponse(count=len(items), crops=items)


@router.get("/advisory/crop/{district}", response_model=CropAdvisoryResponse, tags=["advisory"])
def crop_advisory(
    district: str,
    request: Request,
    crop: str = Query(description="Crop key from GET /crops, e.g. 'rice_transplanted' or 'maize'"),
    sowing_date: Optional[str] = Query(
        default=None,
        description="YYYY-MM-DD. Omit if not sown yet -- that is what the sowing rules are for.",
    ),
) -> CropAdvisoryResponse:
    """Crop-specific sowing and irrigation advice.

    Fully deterministic: the rules live in src/advisory/crop_rules.json and
    every recommendation names the rule_id that produced it. No LLM is
    involved, so this still answers when the free-tier model is down, and
    the same inputs always give the same advice.

    Combines the rainfall forecast with onset status, active/break phase,
    soil wetness and the observed dry-day streak. Any signal that cannot be
    obtained is simply absent, and the rules depending on it do not fire --
    a missing signal never counts as a satisfied condition.
    """
    svc = _svc(request)
    cfg = _resolve(svc, district)

    forecast_data = _build_forecast(svc, cfg)

    # Monsoon context is best-effort: crop advice premised on the forecast
    # alone is still worth returning if the history source is down.
    onset_result = phase_result = None
    soil: dict[str, Any] = {}
    try:
        history = svc.district_history(cfg)
        if not history.empty:
            onset_result = onset_module.onset_status(history, cfg.district)
            phase_result = active_break.current_phase(history, cfg.district)
            soil = recent_soil_and_dryness(history)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Monsoon context unavailable for %s: %s", cfg.district, exc)

    try:
        result = crop_engine.advise(
            crop,
            sowing_date=sowing_date,
            forecast=forecast_data.model_dump(),
            onset=onset_result,
            phase=phase_result,
            soil=soil,
        )
    except crop_engine.CropAdvisoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    result["district"] = cfg.district
    result["state"] = cfg.state
    return CropAdvisoryResponse(**result)


# --------------------------------------------------------------------------
# Telegram alerts
#
# Two endpoints rather than one, because a send is irreversible and the
# operator should be able to read the exact text first. They share one
# composer so the preview cannot drift from what is delivered.
#
# THE CLIENT NEVER SUPPLIES THE MESSAGE TEXT. Both endpoints take a
# district (and optionally a crop) and compose server-side. Accepting a
# caller-supplied string would turn an unauthenticated endpoint into a
# relay for sending arbitrary content through the project's bot.
# --------------------------------------------------------------------------

def _alert_inputs(
    svc: ServiceState, cfg: DistrictConfig, crop: Optional[str], sowing_date: Optional[str]
) -> tuple[dict[str, Any], list[str]]:
    """Gather everything the composer needs, plus which parts were available.

    Only the forecast is required. Monsoon context and crop advice are
    best-effort: an alert carrying the rainfall numbers alone is far more
    useful than a 503 because one upstream source was slow.
    """
    forecast_data = _build_forecast(svc, cfg).model_dump()
    included = ["forecast"]

    onset_result = phase_result = crop_result = None
    soil: dict[str, Any] = {}
    try:
        history = svc.district_history(cfg)
        if not history.empty:
            onset_result = onset_module.onset_status(history, cfg.district)
            phase_result = active_break.current_phase(history, cfg.district)
            soil = recent_soil_and_dryness(history)
            included += ["onset", "monsoon_phase"]
    except Exception as exc:  # noqa: BLE001 - context is optional by design
        logger.warning("Monsoon context unavailable for %s alert: %s", cfg.district, exc)

    if crop:
        try:
            crop_result = crop_engine.advise(
                crop,
                sowing_date=sowing_date,
                forecast=forecast_data,
                onset=onset_result,
                phase=phase_result,
                soil=soil,
            )
            included.append("crop_advisory")
        except crop_engine.CropAdvisoryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    return (
        {
            "forecast": forecast_data,
            "onset": onset_result,
            "phase": phase_result,
            "crop": crop_result,
        },
        included,
    )


@router.get("/alerts/telegram/preview", response_model=AlertPreviewResponse, tags=["alerts"])
def alert_preview(
    request: Request,
    district: str = Query(description="District to compose an alert for"),
    crop: Optional[str] = Query(default=None, description="Optional crop key from GET /crops"),
    sowing_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD"),
) -> AlertPreviewResponse:
    """The exact message a send would deliver, without sending it.

    Works with NO Telegram credentials configured. That is deliberate: the
    whole feature can then be demonstrated on a fresh clone, and nobody
    sends an alert without first reading it.
    """
    svc = _svc(request)
    cfg = _resolve(svc, district)
    inputs, included = _alert_inputs(svc, cfg, crop, sowing_date)
    message = compose_alert(**inputs)

    return AlertPreviewResponse(
        district=cfg.district,
        state=cfg.state,
        crop=crop,
        message=message,
        characters=len(message),
        telegram_configured=telegram.is_configured(),
        configuration_hint=telegram.configuration_hint(),
        sections_included=included,
        note=(
            "Composed server-side from the same data the forecast endpoints return. "
            "Sending delivers this text verbatim to the configured chat."
        ),
    )


@router.post("/alerts/telegram/send", response_model=AlertSendResponse, tags=["alerts"])
def alert_send(
    request: Request,
    district: str = Query(description="District to compose and send an alert for"),
    crop: Optional[str] = Query(default=None, description="Optional crop key from GET /crops"),
    sowing_date: Optional[str] = Query(default=None, description="Optional YYYY-MM-DD"),
) -> AlertSendResponse:
    """Compose and send the alert to the CONFIGURED chat.

    The recipient is TELEGRAM_CHAT_ID and cannot be overridden through this
    API. This endpoint has no authentication in front of it (see the
    README's production-hardening note), and one that accepted an arbitrary
    chat id would be an open relay: anyone able to reach it could message
    any Telegram user through this bot.

    Returns 200 with `sent: false` and Telegram's own error description when
    delivery fails, rather than a 5xx -- the caller needs to know which
    failure it was (bad token, unknown chat, blocked bot) to fix it.
    """
    svc = _svc(request)
    cfg = _resolve(svc, district)

    if not telegram.is_configured():
        raise HTTPException(status_code=503, detail=telegram.configuration_hint())

    if not svc.telegram_send_allowed(TELEGRAM_MAX_SENDS_PER_WINDOW, TELEGRAM_THROTTLE_WINDOW_SECONDS):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Alert send throttled: at most {TELEGRAM_MAX_SENDS_PER_WINDOW} sends per "
                f"{TELEGRAM_THROTTLE_WINDOW_SECONDS}s. Try again shortly."
            ),
        )

    # Recomposed here rather than accepting the previewed text -- see the
    # section comment above.
    inputs, _ = _alert_inputs(svc, cfg, crop, sowing_date)
    message = compose_alert(**inputs)

    try:
        result = telegram.send_message(message)
    except telegram.TelegramNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None

    return AlertSendResponse(
        district=cfg.district,
        state=cfg.state,
        sent=result.sent,
        message=message,
        message_id=result.message_id,
        error=result.error,
        note=(
            "Delivered to the chat configured in TELEGRAM_CHAT_ID. The recipient cannot be "
            "set through this API."
        ),
    )
