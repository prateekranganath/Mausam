"""Trains ONE pooled model across every district in the dataset —
"Rainfall_Forecast_Mausam" — instead of one model per district.

Key differences from src/ml/train.py (single-district):
  - Uses ALL_INDIA_SPLIT_DATES (2021-2025), not SPLIT_DATES (2015-2025).
    See src/config.py for the evidence: reporting completeness has a
    hard, dataset-wide step change exactly at 2021-01-01, and the
    zero-imputation assumption validated for Thiruvananthapuram did not
    replicate nationally. Training only on the reliably-reported era
    avoids leaning on either.
  - impute_as_zero=False — rows with incomplete history/target are
    dropped rather than imputed, for the same reason.
  - Climatology/threshold are (district, month)-keyed (engineering.py
    already supports this — see compute_monthly_climatology's docstring).
  - Feature engineering runs per-district before pooling
    (build_features_multi_district) so no rolling window crosses a
    district boundary.
  - Smaller hyperparameter grids — pooled training data is ~100x larger
    than a single district, so an exhaustive grid would be far slower for
    limited extra benefit; more data already regularizes better than
    aggressive hyperparameter tuning would on a single district.

Run via scripts/train_all_india_model.py, not directly.
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier, XGBRegressor

from src.config import (
    ALL_INDIA_MODEL_DIR,
    ALL_INDIA_PROCESSED_CACHE,
    ALL_INDIA_SPLIT_DATES,
    DISTRICT_CONFIG_PATH,
    MODEL_VERSION,
    RAINFALL_RISK_THRESHOLD_PERCENTILE,
    RAW_DATA_PATH,
)
from src.data.cleaner import clean_dataset
from src.data.district import get_all_districts_daily_series
from src.data.loader import load_raw_dataset
from src.features.engineering import (
    FEATURE_COLUMNS,
    RainfallFeaturePreprocessor,
    build_features_multi_district,
    compute_target_multi_district,
)
from src.ml import evaluate as ev
from src.ml.baseline import ClimatologyBaseline
from src.ml.train import _drop_invalid_targets, _search_classifier, _search_regressor

logger = logging.getLogger(__name__)

# Trimmed relative to the single-district grids — pooled data is ~100x
# larger, so a handful of configs already differentiates well, and larger
# min_samples_leaf / max_depth guard against memorizing individual
# districts' quirks rather than learning transferable patterns.
RF_CLASSIFIER_GRID = {"n_estimators": [200], "max_depth": [8, 14], "min_samples_leaf": [20, 50]}
XGB_CLASSIFIER_GRID = {"n_estimators": [200], "max_depth": [5, 7], "learning_rate": [0.05, 0.1]}
RF_REGRESSOR_GRID = {"n_estimators": [200], "max_depth": [8, 14], "min_samples_leaf": [20, 50]}
XGB_REGRESSOR_GRID = {"n_estimators": [200], "max_depth": [5, 7], "learning_rate": [0.05, 0.1]}


def _load_all_districts_daily(force_reload: bool = False) -> pd.DataFrame:
    if not force_reload and ALL_INDIA_PROCESSED_CACHE.exists():
        logger.info("Loading cached all-districts daily series from %s", ALL_INDIA_PROCESSED_CACHE)
        return pd.read_parquet(ALL_INDIA_PROCESSED_CACHE)
    raw = load_raw_dataset()
    clean = clean_dataset(raw)
    all_daily = get_all_districts_daily_series(clean)
    all_daily.to_parquet(ALL_INDIA_PROCESSED_CACHE, index=False)
    return all_daily


def _split_by_date(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = df["date_of_record"]
    sd = ALL_INDIA_SPLIT_DATES
    return {
        "train": df[(d >= sd.train_start) & (d <= sd.train_end)],
        "val": df[(d >= sd.val_start) & (d <= sd.val_end)],
        "test": df[(d >= sd.test_start) & (d <= sd.test_end)],
    }


def train_all_india(force_reload_data: bool = False) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    all_daily = _load_all_districts_daily(force_reload=force_reload_data)
    n_districts = all_daily["district"].nunique()
    logger.info("All-districts daily series: %d districts, %d rows", n_districts, len(all_daily))

    feats = build_features_multi_district(all_daily, impute_as_zero=False)
    feats = compute_target_multi_district(feats)

    splits_raw = _split_by_date(feats)
    logger.info(
        "Split sizes (district-days, %s to %s): train=%d val=%d test=%d",
        ALL_INDIA_SPLIT_DATES.train_start, ALL_INDIA_SPLIT_DATES.test_end,
        len(splits_raw["train"]), len(splits_raw["val"]), len(splits_raw["test"]),
    )

    preprocessor = RainfallFeaturePreprocessor(RAINFALL_RISK_THRESHOLD_PERCENTILE)
    preprocessor.fit(splits_raw["train"])

    X, labeled = {}, {}
    for name, part in splits_raw.items():
        X[name], labeled[name] = preprocessor.transform(part)
        X[name], labeled[name] = _drop_invalid_targets(X[name], labeled[name])
        logger.info(
            "%s: %d rows with a valid 7-day target (dropped %d with incomplete history/future)",
            name, len(X[name]), len(part) - len(X[name]),
        )

    y_class = {k: v["insufficient_rainfall_next_7_days"].astype(int) for k, v in labeled.items()}
    y_reg = {k: v["rainfall_next_7_days"] for k, v in labeled.items()}

    train_pos_rate = float(y_class["train"].mean())
    logger.info("Train positive (insufficient) rate: %.3f", train_pos_rate)
    scale_pos_weight = (1 - train_pos_rate) / train_pos_rate

    # --- baseline (district-aware climatology, see ClimatologyBaseline) --
    baseline = ClimatologyBaseline().fit(labeled["train"])
    baseline_test_proba = baseline.predict_proba(labeled["test"])
    baseline_test_reg = baseline.predict_rainfall_mm(labeled["test"])

    # --- classifiers -------------------------------------------------
    rf_clf_result = _search_classifier(
        RandomForestClassifier, RF_CLASSIFIER_GRID,
        X["train"], y_class["train"], X["val"], y_class["val"],
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    xgb_clf_result = _search_classifier(
        XGBClassifier, XGB_CLASSIFIER_GRID,
        X["train"], y_class["train"], X["val"], y_class["val"],
        scale_pos_weight=scale_pos_weight, random_state=42,
        eval_metric="logloss", n_jobs=-1,
    )
    logger.info("RF classifier best val AUC=%.4f params=%s", rf_clf_result["auc"], rf_clf_result["params"])
    logger.info("XGB classifier best val AUC=%.4f params=%s", xgb_clf_result["auc"], xgb_clf_result["params"])

    chosen_name = "random_forest" if rf_clf_result["auc"] >= xgb_clf_result["auc"] else "xgboost"
    chosen_clf_result = rf_clf_result if chosen_name == "random_forest" else xgb_clf_result

    calibrated_clf = CalibratedClassifierCV(FrozenEstimator(chosen_clf_result["model"]), method="sigmoid")
    calibrated_clf.fit(X["val"], y_class["val"])

    # --- regressors ----------------------------------------------------
    rf_reg_result = _search_regressor(
        RandomForestRegressor, RF_REGRESSOR_GRID,
        X["train"], y_reg["train"], X["val"], y_reg["val"],
        random_state=42, n_jobs=-1,
    )
    xgb_reg_result = _search_regressor(
        XGBRegressor, XGB_REGRESSOR_GRID,
        X["train"], y_reg["train"], X["val"], y_reg["val"],
        random_state=42, n_jobs=-1,
    )
    logger.info("RF regressor best val MAE=%.4f params=%s", rf_reg_result["mae"], rf_reg_result["params"])
    logger.info("XGB regressor best val MAE=%.4f params=%s", xgb_reg_result["mae"], xgb_reg_result["params"])

    chosen_reg_name = "random_forest" if rf_reg_result["mae"] <= xgb_reg_result["mae"] else "xgboost"
    chosen_reg_result = rf_reg_result if chosen_reg_name == "random_forest" else xgb_reg_result
    chosen_regressor = chosen_reg_result["model"]

    # --- test evaluation (overall + a few sample districts) -----------
    model_test_proba = calibrated_clf.predict_proba(X["test"])[:, 1]
    model_test_reg = chosen_regressor.predict(X["test"])

    results = {
        "baseline": ev.classification_metrics(y_class["test"], baseline_test_proba),
        "classifier": ev.classification_metrics(y_class["test"], model_test_proba),
        "baseline_regression": ev.regression_metrics(y_reg["test"], baseline_test_reg),
        "regressor": ev.regression_metrics(y_reg["test"], model_test_reg),
        "chosen_classifier_model": chosen_name,
        "chosen_classifier_params": chosen_clf_result["params"],
        "chosen_classifier_val_auc": chosen_clf_result["auc"],
        "chosen_regressor_model": chosen_reg_name,
        "chosen_regressor_params": chosen_reg_result["params"],
        "chosen_regressor_val_mae": chosen_reg_result["mae"],
        "train_positive_rate": train_pos_rate,
        "val_positive_rate": float(y_class["val"].mean()),
        "test_positive_rate": float(y_class["test"].mean()),
        "row_counts": {k: len(v) for k, v in X.items()},
        "n_districts_trained": n_districts,
    }

    # per-district breakdown on a handful of geographically spread sample
    # districts — sanity check that the pooled model isn't only good for
    # whichever districts dominate the training data by row count
    sample_districts = [d for d in [
        "Thiruvananthapuram", "Mumbai Suburban", "New Delhi", "Kolkata",
        "Jaisalmer", "Bengaluru Urban", "Guwahati",
    ] if d in labeled["test"]["district"].unique()]
    per_district = {}
    for d in sample_districts:
        mask = labeled["test"]["district"] == d
        if mask.sum() < 5:
            continue
        per_district[d] = {
            "n_test_rows": int(mask.sum()),
            "classifier": ev.classification_metrics(y_class["test"][mask.values], model_test_proba[mask.values]),
            "regressor": ev.regression_metrics(y_reg["test"][mask.values], model_test_reg[mask.values]),
        }
    results["per_district_sample"] = per_district

    logger.info("TEST classifier vs baseline (pooled, all districts):")
    logger.info("  baseline: %s", results["baseline"])
    logger.info("  model:    %s", results["classifier"])
    logger.info("TEST regressor vs baseline (pooled):")
    logger.info("  baseline: %s", results["baseline_regression"])
    logger.info("  model:    %s", results["regressor"])
    for d, r in per_district.items():
        logger.info(
            "  [%s] n=%d roc_auc=%.3f mae=%.2f",
            d, r["n_test_rows"], r["classifier"]["roc_auc"] or float("nan"), r["regressor"]["mae"],
        )

    # --- plots -----------------------------------------------------
    out_dir = ALL_INDIA_MODEL_DIR
    plots_dir = out_dir / "plots"
    ev.plot_roc_curves(
        y_class["test"],
        {"baseline (climatology)": baseline_test_proba, f"{chosen_name} (calibrated)": model_test_proba},
        plots_dir / "roc_curve.png",
    )
    ev.plot_calibration_curves(
        y_class["test"],
        {"baseline (climatology)": baseline_test_proba, f"{chosen_name} (calibrated)": model_test_proba},
        plots_dir / "calibration_curve.png",
    )
    ev.plot_confusion_matrix(
        np.array(results["classifier"]["confusion_matrix"]["matrix"]),
        plots_dir / "confusion_matrix.png",
        title=f"{chosen_name} — all-India test set",
    )
    base_model_for_importance = chosen_clf_result["model"]
    if hasattr(base_model_for_importance, "feature_importances_"):
        ev.plot_feature_importance(
            FEATURE_COLUMNS,
            base_model_for_importance.feature_importances_,
            plots_dir / "feature_importance.png",
        )

    # --- save artifacts ----------------------------------------------
    joblib.dump(calibrated_clf, out_dir / "rainfall_risk_classifier.joblib")
    joblib.dump(chosen_regressor, out_dir / "rainfall_amount_regressor.joblib")
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")
    joblib.dump(baseline, out_dir / "climatology_baseline.joblib")

    feature_schema = {
        "feature_columns": FEATURE_COLUMNS,
        "target_classification": "insufficient_rainfall_next_7_days",
        "target_regression": "rainfall_next_7_days",
    }
    with open(out_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2)

    metadata = {
        "model_version": MODEL_VERSION,
        "scope": "all-india (pooled, multi-district)",
        "n_districts_trained": n_districts,
        "districts": sorted(all_daily["district"].unique().tolist()),
        "source_file": RAW_DATA_PATH.name,
        "forecast_horizon_days": 7,
        "classifier_model_name": chosen_name,
        "classifier_params": chosen_clf_result["params"],
        "classifier_calibration": "sigmoid (Platt), fit on validation split, cv='prefit' (FrozenEstimator)",
        "regressor_model_name": chosen_reg_name,
        "regressor_params": chosen_reg_result["params"],
        "rainfall_risk_threshold_percentile": RAINFALL_RISK_THRESHOLD_PERCENTILE,
        "train_date_range": [ALL_INDIA_SPLIT_DATES.train_start, ALL_INDIA_SPLIT_DATES.train_end],
        "val_date_range": [ALL_INDIA_SPLIT_DATES.val_start, ALL_INDIA_SPLIT_DATES.val_end],
        "test_date_range": [ALL_INDIA_SPLIT_DATES.test_start, ALL_INDIA_SPLIT_DATES.test_end],
        "date_range_rationale": (
            "Restricted to 2021-01-01 onward: rainfall-reporting completeness "
            "has a hard, dataset-wide step change exactly at that date "
            "(missingness ~87% in Dec 2020 to ~6.5% in Jan 2021, uniformly "
            "across every district — a data-generation-process artifact, not "
            "a real gradual station-network rollout). Training only on the "
            "reliably-reported era avoids depending on an unverified "
            "imputation assumption for the bulk of the record."
        ),
        "class_imbalance_handling": (
            "RandomForest: class_weight='balanced'. XGBoost: scale_pos_weight "
            f"={scale_pos_weight:.3f} (train neg/pos ratio). No resampling/SMOTE."
        ),
        "missing_rainfall_handling": (
            "impute_as_zero=False for this pooled model — rows with an "
            "incomplete 7-day target window or 30-day feature lookback are "
            "dropped, not imputed. The Thiruvananthapuram-specific "
            "zero-imputation justification (missing correlates with warmer/"
            "drier conditions) was checked across 143 districts and did NOT "
            "replicate nationally (66 districts same-direction, 77 "
            "opposite-direction) — so no blanket imputation assumption is "
            "applied at this scale."
        ),
    }
    with open(out_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(out_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    _register_all_districts(all_daily)

    logger.info("Artifacts saved to %s", out_dir)
    return results


def _register_all_districts(all_daily: pd.DataFrame) -> None:
    """Register every trained district's coordinates for live Open-Meteo
    forecasting in one pass (see src/forecasting/district_registry.py)."""
    centroids = all_daily.groupby(["district", "state"])[["latitude", "longitude", "elevation"]].mean()
    with open(DISTRICT_CONFIG_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    for (district, state), row in centroids.iterrows():
        registry[district] = {
            "state": state,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "elevation": float(row["elevation"]),
        }
    with open(DISTRICT_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registered %d districts in %s for live forecasting", len(centroids), DISTRICT_CONFIG_PATH)
