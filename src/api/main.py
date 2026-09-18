"""FastAPI app. Run with:  uvicorn src.api.main:app --reload

The model is loaded ONCE at startup (lifespan) and shared read-only across
requests. It is pulled from Hugging Face (cached by huggingface_hub after the
first pull); if the Hub is unreachable it falls back to locally trained
artifacts in models/all-india/ so development works offline. Startup fails
loudly if neither is available, rather than serving without a model.

Note: the district registry is read from district_config.json at import time,
so retraining (which rewrites that file) needs a server restart to be seen.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.api.state import ServiceState, servable_districts
from src.config import ALL_INDIA_MODEL_DIR, API_ALLOWED_ORIGINS, HF_TOKEN, hf_repo_id_all_india
from src.forecasting.open_meteo import OpenMeteoError
from src.ml.predict import RainfallRiskPredictor

logger = logging.getLogger("mausam.api")


def _load_predictor() -> tuple[RainfallRiskPredictor, str]:
    try:
        return RainfallRiskPredictor.from_pretrained_all_india(), f"huggingface:{hf_repo_id_all_india()}"
    except Exception as exc:  # network, auth, missing repo: any of these should trigger the local fallback
        logger.warning("Could not pull model from Hugging Face (%s); trying local artifacts", exc)
    try:
        return RainfallRiskPredictor.load_local_all_india(), f"local:{ALL_INDIA_MODEL_DIR}"
    except Exception as exc:
        raise RuntimeError(
            "No model available: Hugging Face pull failed and no local artifacts in "
            f"{ALL_INDIA_MODEL_DIR}. Train one with scripts/train_all_india_model.py "
            "or check HF_TOKEN / network."
        ) from exc


def _load_eval_results() -> dict | None:
    """evaluation_results.json for the served model: from the Hub if reachable
    (so metrics always describe the model actually being served), else local."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=hf_repo_id_all_india(), filename="evaluation_results.json", token=HF_TOKEN)
    except Exception:
        path = ALL_INDIA_MODEL_DIR / "evaluation_results.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        logger.warning("evaluation_results.json unavailable; /model/metrics will return 503")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor, source = _load_predictor()
    trained, excluded = servable_districts(predictor)
    app.state.svc = ServiceState(
        predictor=predictor,
        model_source=source,
        trained_districts=trained,
        excluded_districts=excluded,
        eval_results=_load_eval_results(),
    )
    logger.info("Model loaded from %s (%d servable districts)", source, len(trained))
    if excluded:
        logger.warning("Not serving %d districts: %s", len(excluded), excluded)
    yield


app = FastAPI(
    title="Mausam Rainfall Forecast API",
    version="0.1.0",
    description=(
        "District-level 7-day rainfall-risk forecasts for India from a pooled all-India ML model, "
        "cross-checked against Open-Meteo's own forecast. A hackathon MVP, not an operational "
        "meteorological forecast; no panchayat-level accuracy is claimed."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.include_router(router)


@app.exception_handler(OpenMeteoError)
async def open_meteo_unavailable(request: Request, exc: OpenMeteoError) -> JSONResponse:
    logger.error("Open-Meteo unavailable for %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "Weather data provider (Open-Meteo) is currently unreachable and no cached data is available. Try again shortly."},
    )
