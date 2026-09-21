"""Pydantic response models. They double as the OpenAPI documentation, so
field descriptions here are what a frontend developer will read at /docs."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field



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


class DistrictCoverageResponse(BaseModel):
    trained_count: int
    trained_districts: list[str]
    excluded_districts: dict[str, str]


class ForecastSeriesPoint(BaseModel):
    time: str
    rainfall_mm: Optional[float] = None
    precipitation_probability_percent: Optional[float] = None


class ForecastSeriesResponse(BaseModel):
    district: str
    state: str
    as_of_date: str
    source: str
    data: list[ForecastSeriesPoint]


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


class AnalysisBlock(BaseModel):
    """The forecast analysis, derived by rules from the forecast numbers.

    Always present and never depends on an LLM: instant, reproducible, and
    every number in it is a number from the forecast. `risk_level` is the
    forecast MODEL's own level, passed through; nothing re-rates it.
    """

    source: Literal["rules"]
    headline: str
    risk_level: Optional[Literal["LOW", "MODERATE", "HIGH"]] = Field(
        default=None,
        description="The forecast model's level, or null where a dry week is normal for this district-month and the label carries no information (see risk_meaningful)",
    )
    risk_meaningful: bool
    confidence: Literal["low", "moderate"] = Field(
        description="Never 'high': the pooled model is hackathon-grade. 'low' when the model and Open-Meteo differ materially."
    )
    confidence_reasons: list[str]
    key_factors: list[str]
    model_disagreement: list[str] = Field(
        default_factory=list, description="Empty unless the model and Open-Meteo disagree"
    )
    actions: list[str]


class AiSummary(BaseModel):
    """An LLM-written plain-language rewrite of the analysis. A note that cites a figure not in the
    input, or attaches a number to the wrong unit, is rejected before it gets here."""

    text: str
    model: str = Field(description="Which model in the fallback chain actually answered")


class AdvisoryResponse(BaseModel):
    forecast: ForecastResponse
    analysis: AnalysisBlock
    ai_status: Literal["ok", "skipped", "unconfigured", "unavailable"] = Field(
        description="ok: ai_summary is present. skipped: the caller passed polish=false. unconfigured: no API key. unavailable: every model failed (see llm_error)."
    )
    ai_summary: Optional[AiSummary] = None
    llm_error: Optional[str] = None
    note: str = (
        "The analysis is derived by rules from the forecast numbers. The AI summary, when present, only "
        "restates it in plain language and is not a source of weather data."
    )


# --------------------------------------------------------------------------
# Climate context (ENSO / IOD / MJO)
# --------------------------------------------------------------------------

class ClimateIndex(BaseModel):
    """One global climate index's latest PUBLISHED value.

    `as_of` and `publication_lag_days` are not decoration. ONI ran ~81 days
    behind and DMI ~112 days behind when measured; presenting either as a
    current reading would overstate its freshness considerably.
    """

    name: str
    value: float
    as_of: str = Field(description="The month or day the value describes, not when it was fetched")
    phase: str
    publication_lag_days: int = Field(
        description="Typical delay between the period a value describes and its publication"
    )
    source: str


class ClimateContextResponse(BaseModel):
    enso: ClimateIndex
    iod: ClimateIndex
    mjo: ClimateIndex
    mjo_phase: Optional[int] = Field(default=None, description="MJO octant 1-8, if amplitude is meaningful")
    note: str


# --------------------------------------------------------------------------
# Monsoon onset and active/break
# --------------------------------------------------------------------------

class OnsetResponse(BaseModel):
    """Local rainfall onset. Explicitly NOT an IMD onset declaration -- see
    `not_imd_criterion`, and the README for the measured ~20-day difference."""

    district: str
    state: str
    season: str
    dominant_season: str
    as_of_date: str
    status: Literal[
        "outside_season", "pre_onset", "onset_likely", "onset_confirmed", "post_onset", "no_onset_detected"
    ]
    status_description: str
    onset_date: Optional[str] = None
    onset_day_of_year: Optional[int] = None
    anomaly_days: Optional[int] = Field(
        default=None, description="Days later (+) or earlier (-) than this district's median onset"
    )
    anomaly_label: Optional[str] = None
    trigger_7day_rainfall_mm: Optional[float] = None
    trigger_rainy_days: Optional[int] = None
    persistence_longest_dry_run_days: Optional[int] = None
    rejected_false_onsets: list[str] = Field(
        default_factory=list, description="Candidate onsets rejected because the rain did not persist"
    )
    climatology: dict[str, Any]
    data: dict[str, Any] = Field(description="Where the underlying daily record came from")
    method: str
    not_imd_criterion: str


class MonsoonPhaseDay(BaseModel):
    date: str
    rainfall_mm: Optional[float] = None
    anomaly_sd: Optional[float] = None
    phase: str


class MonsoonPhaseResponse(BaseModel):
    district: str
    state: str
    as_of_date: str
    monsoon_phase: Literal["active", "break", "normal", "not_applicable"]
    phase_description: str
    days_in_current_phase: Optional[int] = None
    rainfall_anomaly_sd: Optional[float] = None
    rainfall_mm: Optional[float] = None
    trailing_7day_mean_mm_per_day: Optional[float] = None
    climatology_7day_mean_mm_per_day: Optional[float] = None
    climatology_7day_sd_mm_per_day: Optional[float] = None
    recent_30_days: list[MonsoonPhaseDay] = Field(default_factory=list)
    data: dict[str, Any]
    method: str
    caveats: str


# --------------------------------------------------------------------------
# Point forecast
# --------------------------------------------------------------------------

class PointForecastResponse(BaseModel):
    """A forecast at an arbitrary coordinate.

    Weather comes from Open-Meteo at the exact point, but the climatology
    and risk threshold are the resolved DISTRICT's, because that is the only
    level the model was fit at. `caveat` says so, and must be surfaced to
    anyone reading the numbers."""

    latitude: float
    longitude: float
    resolved_district: str
    state: str
    distance_km: float = Field(description="From the requested point to the district's station centroid")
    forecast: ForecastResponse
    caveat: str


# --------------------------------------------------------------------------
# Crop advisory
# --------------------------------------------------------------------------

class Crop(BaseModel):
    key: str
    display_name: str
    season: str
    duration_days: int
    seasonal_water_mm: int
    notes: Optional[str] = None


class CropsResponse(BaseModel):
    count: int
    crops: list[Crop]


class CropRecommendation(BaseModel):
    rule_id: str = Field(description="Identifies the rule in crop_rules.json that produced this")
    category: str
    severity: Literal["high", "medium", "info"]
    action: str
    rationale: str
    triggered_by: dict[str, Any] = Field(
        description="The actual signal values that satisfied the rule, so it can be checked"
    )


class CropAdvisoryResponse(BaseModel):
    """Deterministic, rule-based crop advice. No LLM is involved in producing
    it; the /advisory endpoint may separately phrase it."""

    district: str
    state: str
    crop: str
    crop_key: str
    sowing_date: Optional[str] = None
    as_of_date: str
    days_since_sowing: Optional[int] = None
    growth_stage: Optional[str] = None
    stage_drought_sensitivity: Optional[str] = None
    stage_weekly_water_requirement_mm: Optional[float] = None
    expected_rainfall_next_7_days_mm: Optional[float] = None
    water_balance_mm: Optional[float] = None
    past_maturity: bool = False
    recommendations: list[CropRecommendation] = Field(default_factory=list)
    signals_used: dict[str, Any] = Field(default_factory=dict)
    signals_unavailable: list[str] = Field(default_factory=list)
    suppression_note: Optional[str] = None
    rules_version: str
    water_requirement_basis: str
    stage_basis: str
    disclaimer: str
    sources: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Telegram alerts
# --------------------------------------------------------------------------

class AlertPreviewResponse(BaseModel):
    """The exact message a send would deliver.

    Deliberately available with NO bot token configured, so the feature can
    be demonstrated end to end on a fresh clone -- only the final send needs
    credentials. `telegram_configured` is what the UI uses to decide whether
    to enable its Send button, and `configuration_hint` says what is missing.
    """

    district: str
    state: str
    crop: Optional[str] = None
    message: str = Field(description="Plain text, exactly as it would be sent")
    characters: int
    telegram_configured: bool
    configuration_hint: str
    sections_included: list[str] = Field(
        default_factory=list,
        description="Which data blocks made it into the message; a missing one means that source was unavailable",
    )
    note: str


class AlertSendResponse(BaseModel):
    district: str
    state: str
    sent: bool
    message: str = Field(description="The text that was sent, recomposed server-side")
    message_id: Optional[int] = Field(default=None, description="Telegram's id for the delivered message")
    error: Optional[str] = Field(
        default=None, description="Telegram's own description of the failure, verbatim"
    )
    note: str
