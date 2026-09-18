"""Clean, FastAPI-independent prediction interface.

Usage (local artifacts, e.g. right after training, never pushed to HF):
    predictor = RainfallRiskPredictor.load_local("Thiruvananthapuram")

Usage (no local training needed — pulls the already-trained artifacts
from Hugging Face Hub, cached locally by huggingface_hub after the first
call, so this is fast on subsequent runs too):
    predictor = RainfallRiskPredictor.from_pretrained("Thiruvananthapuram")

Either way:
    result = predictor.predict(district_daily_df_up_to_today)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import pandas as pd

from src.config import FORECAST_HORIZON_DAYS, HF_TOKEN, MAX_LOOKBACK_DAYS, hf_repo_id
from src.features.engineering import RainfallFeaturePreprocessor, build_features

logger = logging.getLogger(__name__)

HUB_ARTIFACT_FILES = [
    "rainfall_risk_classifier.joblib",
    "rainfall_amount_regressor.joblib",
    "preprocessor.joblib",
    "model_metadata.json",
]


@dataclass
class RainfallRiskResult:
    district: str
    forecast_horizon_days: int
    rainfall_probability: float
    risk_level: str
    predicted_rainfall_mm: float
    as_of_date: str
    model: str
    model_version: str

    def to_dict(self) -> dict:
        return asdict(self)


def _risk_level(probability: float) -> str:
    if probability >= 0.66:
        return "HIGH"
    if probability >= 0.33:
        return "MODERATE"
    return "LOW"


class RainfallRiskPredictor:
    """Loads all artifacts needed for inference and exposes predict()."""

    def __init__(
        self,
        classifier,
        regressor,
        preprocessor: RainfallFeaturePreprocessor,
        metadata: dict,
    ) -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.preprocessor = preprocessor
        self.metadata = metadata

    @classmethod
    def load(cls, models_dir: Path) -> "RainfallRiskPredictor":
        """Load artifacts from an exact local directory."""
        classifier = joblib.load(models_dir / "rainfall_risk_classifier.joblib")
        regressor = joblib.load(models_dir / "rainfall_amount_regressor.joblib")
        preprocessor = joblib.load(models_dir / "preprocessor.joblib")
        with open(models_dir / "model_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        return cls(classifier, regressor, preprocessor, metadata)

    @classmethod
    def load_local(cls, district: str) -> "RainfallRiskPredictor":
        """Load a district's locally trained artifacts (models/<district>/),
        never pushed to Hugging Face — e.g. right after
        scripts/train_model.py --district "..." with no push step."""
        from src.config import local_model_dir

        return cls.load(local_model_dir(district))

    @classmethod
    def load_local_all_india(cls) -> "RainfallRiskPredictor":
        """Load the pooled all-India model's local artifacts
        (models/all-india/), before/without pushing to Hugging Face."""
        from src.config import ALL_INDIA_MODEL_DIR

        return cls.load(ALL_INDIA_MODEL_DIR)

    @classmethod
    def _from_hub(cls, repo_id: str) -> "RainfallRiskPredictor":
        from huggingface_hub import hf_hub_download

        logger.info("Pulling model artifacts from %s", repo_id)
        # The push script creates repos as private, so a token is required
        # to pull — same HF_TOKEN used to push.
        paths = {
            name: hf_hub_download(repo_id=repo_id, filename=name, token=HF_TOKEN)
            for name in HUB_ARTIFACT_FILES
        }
        classifier = joblib.load(paths["rainfall_risk_classifier.joblib"])
        regressor = joblib.load(paths["rainfall_amount_regressor.joblib"])
        preprocessor = joblib.load(paths["preprocessor.joblib"])
        with open(paths["model_metadata.json"], encoding="utf-8") as f:
            metadata = json.load(f)
        return cls(classifier, regressor, preprocessor, metadata)

    @classmethod
    def from_pretrained(cls, district: str, repo_id: str | None = None) -> "RainfallRiskPredictor":
        """Load a single-district model straight from Hugging Face Hub —
        no local training needed. huggingface_hub caches downloads under
        ~/.cache/huggingface, so repeat calls (even across processes)
        don't re-download once the first pull has happened.
        """
        return cls._from_hub(repo_id or hf_repo_id(district))

    @classmethod
    def from_pretrained_all_india(cls) -> "RainfallRiskPredictor":
        """Load the pooled all-India model ("Rainfall_Forecast_Mausam")
        straight from Hugging Face Hub — works for any district it was
        trained on, no per-district training or repo needed."""
        from src.config import hf_repo_id_all_india

        return cls._from_hub(hf_repo_id_all_india())

    def predict(self, district_daily_df: pd.DataFrame, as_of_date: str | None = None) -> RainfallRiskResult:
        """`district_daily_df` must be one district's continuous daily
        series (see src.data.district.get_district_daily_series) covering
        at least MAX_LOOKBACK_DAYS days up to and including the prediction
        date. Only rows up to `as_of_date` (default: the last row) are used
        — nothing at or after that date is touched, by construction of
        build_features/lag/rolling logic operating on the trailing slice.
        """
        df = district_daily_df.sort_values("date_of_record").reset_index(drop=True)
        if as_of_date is not None:
            df = df[df["date_of_record"] <= pd.Timestamp(as_of_date)]
        if df.empty:
            raise ValueError("No data available up to the requested as_of_date.")

        feats = build_features(df)
        X, _ = self.preprocessor.transform(feats)
        row = X.iloc[[-1]]

        proba = float(self.classifier.predict_proba(row)[0, 1])
        predicted_mm = float(self.regressor.predict(row)[0])
        as_of = df["date_of_record"].iloc[-1]

        return RainfallRiskResult(
            district=str(df["district"].iloc[-1]),
            forecast_horizon_days=FORECAST_HORIZON_DAYS,
            rainfall_probability=round(proba, 4),
            risk_level=_risk_level(proba),
            predicted_rainfall_mm=round(predicted_mm, 2),
            as_of_date=str(as_of.date()),
            model=self.metadata.get("classifier_model_name", "unknown"),
            model_version=self.metadata.get("model_version", "unknown"),
        )

    def predict_live(self, district: str) -> dict:
        """Real-time prediction: pulls recent observed weather for the
        district from Open-Meteo (no local dataset needed), runs our
        trained model on it exactly as at train time, and separately
        reports Open-Meteo's own forward-looking forecast for comparison.

        The two are deliberately NOT combined into one number — they
        answer different questions (our model: "is the next 7 days
        unusually dry for this time of year, per 10 years of local
        history"; Open-Meteo: "what does live NWP guidance say will
        happen"). See src.forecasting.open_meteo for why they aren't
        naively averaged.
        """
        from src.forecasting.district_registry import get_district_config
        from src.forecasting.open_meteo import (
            fetch_daily_weather,
            get_native_forecast_summary,
            to_district_daily_schema,
        )

        cfg = get_district_config(district)
        # Open-Meteo's forecast_days block starts at TODAY (today..today+N-1),
        # but our target is strictly T+1..T+7 — request one extra day so
        # get_native_forecast_summary's "> today" filter still has 7 full
        # future days left after excluding today itself.
        weather = fetch_daily_weather(
            cfg.latitude, cfg.longitude,
            forecast_days=FORECAST_HORIZON_DAYS + 1,
            past_days=MAX_LOOKBACK_DAYS + 5,
        )
        schema_df = to_district_daily_schema(
            weather, cfg.district, cfg.state, cfg.latitude, cfg.longitude, cfg.elevation
        )

        today = pd.Timestamp.now().normalize()
        local_result = self.predict(schema_df, as_of_date=str(today.date()))
        native_forecast = get_native_forecast_summary(weather, FORECAST_HORIZON_DAYS)

        return {
            "district": cfg.district,
            "state": cfg.state,
            "as_of_date": local_result.as_of_date,
            "local_model": local_result.to_dict(),
            "open_meteo_forecast": native_forecast,
            "note": (
                "local_model is our trained model, run on Open-Meteo's analysed "
                "past weather up to as_of_date. Its rainfall features use only "
                "completed days; today's temperature, wind and pressure are "
                "Open-Meteo's estimate for today (partly forecast). Forecast "
                "days after as_of_date are never fed to the model. "
                "open_meteo_forecast is Open-Meteo's own independent forward "
                "forecast. These are different quantities and are reported "
                "separately, not combined into a single number."
            ),
        }
