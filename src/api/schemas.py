"""Pydantic response models. They double as the OpenAPI documentation, so
field descriptions here are what a frontend developer will read at /docs."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from src.llm.openrouter import Advisory


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_name: str = Field(description="Repo the model was pulled from")
    model_version: str
    classifier: str
    n_districts: int = Field(description="Districts the model can actually forecast")
    excluded_districts: dict[str, str] = Field(
        default_factory=dict,
        description="In the source data but not served, mapped to the reason (no training data, or a calendar month missing from it)",
    )
    openrouter_configured: bool = Field(
        description="Whether /advisory can call the LLM (key and model set)"
    )


class District(BaseModel):
    name: str
    state: str
    latitude: float
    longitude: float
    elevation: float


class DistrictsResponse(BaseModel):
    count: int
    districts: list[District]


class MlModelOutput(BaseModel):
    """Output of the trained all-India model. An estimate from a statistical
    model, not an official forecast."""

    district: str
    forecast_horizon_days: int
    rainfall_probability: float = Field(
        description="Probability that the next 7 days are unusually dry for this district and month (below its lower-tercile 7-day rainfall)"
    )
    risk_level: Literal["LOW", "MODERATE", "HIGH"]
    predicted_rainfall_mm: float = Field(description="Expected total rainfall over the next 7 days")
    as_of_date: str
    model: str
    model_version: str


class OpenMeteoDaily(BaseModel):
    time: str
    precipitation_sum: Optional[float] = None
    precipitation_probability_max: Optional[float] = None


class OpenMeteoForecast(BaseModel):
    """Open-Meteo's own independent forward forecast (a live NWP forecast,
    not an observation and not produced by our model)."""

    source: str
    forecast_days: int
    total_precipitation_sum_mm: float
    mean_daily_precipitation_probability_percent: Optional[float] = None
    daily: list[OpenMeteoDaily]


class Agreement(BaseModel):
    ml_predicted_mm: float
    open_meteo_forecast_mm: float
    difference_mm: float
    magnitude_diverges: bool
    threshold_mm: Optional[float] = Field(
        default=None, description="This district-month's 'insufficient rainfall' cutoff, learned in training"
    )
    threshold_degenerate: bool = Field(
        default=False,
        description=(
            "True when this district-month's threshold is ~0mm (a dry week is normal here), so the "
            "'insufficient rainfall' label cannot meaningfully trigger. A LOW risk then does NOT mean rain is expected."
        ),
    )
    ml_implies_insufficient: Optional[bool] = None
    open_meteo_implies_insufficient: Optional[bool] = None
    sources_agree: Optional[bool] = None
    note: str


class ForecastResponse(BaseModel):
    district: str
    state: str
    as_of_date: str
    ml_model: MlModelOutput
    open_meteo_forecast: OpenMeteoForecast
    agreement: Agreement
    note: str


class HistoricalPoint(BaseModel):
    date: str
    rainfall: Optional[float] = Field(default=None, description="mm; null where the source had no reading")
    avg_temp: Optional[float] = None
    min_temp: Optional[float] = None
    max_temp: Optional[float] = None
    wind_speed: Optional[float] = None
    air_pressure: Optional[float] = None
    relative_humidity: Optional[float] = None


class HistoricalResponse(BaseModel):
    district: str
    state: str
    source: str
    requested_days: int
    n_points: int = Field(description="Days actually returned; can be fewer than requested if Open-Meteo has not filled the latest days yet")
    data_start: str
    data_end: str
    missing_rainfall_days: int
    units: dict[str, str]
    note: str
    data: list[HistoricalPoint]


class MetricsResponse(BaseModel):
    model_version: str
    scope: str
    n_districts_trained: int
    classifier_model: str
    regressor_model: str
    train_date_range: list[str]
    val_date_range: list[str]
    test_date_range: list[str]
    test_row_counts: dict[str, int]
    classifier: dict[str, Any] = Field(description="Held-out test metrics for the trained classifier")
    baseline: dict[str, Any] = Field(description="Same metrics for the climatology-only baseline it must beat")
    regressor: dict[str, Any]
    baseline_regression: dict[str, Any]
    district: Optional[str] = None
    district_metrics: Optional[dict[str, Any]] = Field(
        default=None,
        description="Per-district test metrics, present only for the sampled districts",
    )
    available_district_metrics: list[str]
    note: str


class AdvisoryResponse(BaseModel):
    forecast: ForecastResponse
    advisory: Optional[Advisory] = Field(
        default=None, description="null if the LLM was unavailable; see llm_error"
    )
    unsupported_numbers: list[str] = Field(
        default_factory=list,
        description="Numbers in the narrative that could not be traced to the input data - treat with suspicion",
    )
    llm_model: Optional[str] = None
    llm_error: Optional[str] = None
    note: str = (
        "The advisory is LLM-generated commentary over the numeric forecast. "
        "It is not a source of weather data."
    )
