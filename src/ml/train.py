"""End-to-end training: load -> clean -> feature-engineer -> chronological
split -> baseline -> RandomForest/XGBoost (small val-selected grid search)
-> calibration -> test evaluation -> save artifacts.

Run via scripts/train_model.py, not directly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier, XGBRegressor

from src.config import (
    DEFAULT_DISTRICT,
    MODEL_VERSION,
    RAINFALL_RISK_THRESHOLD_PERCENTILE,
    RAW_DATA_PATH,
    SPLIT_DATES,
    local_model_dir,
)
from src.data.cleaner import clean_dataset
from src.data.district import get_district_daily_series
from src.data.loader import load_raw_dataset
from src.features.engineering import (
    FEATURE_COLUMNS,
    RainfallFeaturePreprocessor,
    build_features,
    compute_target,
)
from src.ml import evaluate as ev
from src.ml.baseline import ClimatologyBaseline

logger = logging.getLogger(__name__)

RF_CLASSIFIER_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [4, 8, None],
    "min_samples_leaf": [5, 10],
}
XGB_CLASSIFIER_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
}
RF_REGRESSOR_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [4, 8, None],
    "min_samples_leaf": [5, 10],
}
XGB_REGRESSOR_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
}


def _split_by_date(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = df["date_of_record"]
    return {
        "train": df[(d >= SPLIT_DATES.train_start) & (d <= SPLIT_DATES.train_end)],
        "val": df[(d >= SPLIT_DATES.val_start) & (d <= SPLIT_DATES.val_end)],
        "test": df[(d >= SPLIT_DATES.test_start) & (d <= SPLIT_DATES.test_end)],
    }


def _drop_invalid_targets(X: pd.DataFrame, labeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = labeled["rainfall_next_7_days"].notna()
    return X[valid].reset_index(drop=True), labeled[valid].reset_index(drop=True)


def _search_classifier(model_cls, grid: dict, X_train, y_train, X_val, y_val, **fixed):
    best = None
    for params in ParameterGrid(grid):
        model = model_cls(**params, **fixed)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        auc = ev.roc_auc_score(y_val, proba) if len(set(y_val)) > 1 else 0.0
        if best is None or auc > best["auc"]:
            best = {"model": model, "params": params, "auc": auc}
    return best


def _search_regressor(model_cls, grid: dict, X_train, y_train, X_val, y_val, **fixed):
    best = None
    for params in ParameterGrid(grid):
        model = model_cls(**params, **fixed)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        mae = float(np.mean(np.abs(y_val - pred)))
        if best is None or mae < best["mae"]:
            best = {"model": model, "params": params, "mae": mae}
    return best


def train(district: str = DEFAULT_DISTRICT, state: str | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    raw = load_raw_dataset()
    clean = clean_dataset(raw)
    daily = get_district_daily_series(clean, district, state)

    feats = build_features(daily)
    feats = compute_target(feats)

    splits_raw = _split_by_date(feats)
    logger.info(
        "Split sizes (calendar days): train=%d val=%d test=%d",
        len(splits_raw["train"]), len(splits_raw["val"]), len(splits_raw["test"]),
    )

    preprocessor = RainfallFeaturePreprocessor(RAINFALL_RISK_THRESHOLD_PERCENTILE)
    preprocessor.fit(splits_raw["train"])

    X, labeled = {}, {}
    for name, part in splits_raw.items():
        X[name], labeled[name] = preprocessor.transform(part)
        X[name], labeled[name] = _drop_invalid_targets(X[name], labeled[name])
        logger.info(
            "%s: %d rows with a valid 7-day target (dropped %d with incomplete future rainfall)",
            name, len(X[name]), len(part) - len(X[name]),
        )

    y_class = {k: v["insufficient_rainfall_next_7_days"].astype(int) for k, v in labeled.items()}
    y_reg = {k: v["rainfall_next_7_days"] for k, v in labeled.items()}

    train_pos_rate = float(y_class["train"].mean())
    logger.info("Train positive (insufficient) rate: %.3f", train_pos_rate)
    scale_pos_weight = (1 - train_pos_rate) / train_pos_rate

    # --- baseline ---------------------------------------------------
    baseline = ClimatologyBaseline().fit(labeled["train"])
    baseline_val_proba = baseline.predict_proba(labeled["val"])
    baseline_test_proba = baseline.predict_proba(labeled["test"])
    baseline_test_reg = baseline.predict_rainfall_mm(labeled["test"])

    # --- classifiers: small val-selected grid search -----------------
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

    # calibrate the chosen (already-fit) classifier on the val split.
    # FrozenEstimator tells CalibratedClassifierCV not to refit the base
    # model — it only fits the calibration mapping, on val, never on train.
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

    # --- test evaluation -------------------------------------------
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
    }

    logger.info("TEST classifier vs baseline:")
    logger.info("  baseline: %s", results["baseline"])
    logger.info("  model:    %s", results["classifier"])
    logger.info("TEST regressor vs baseline:")
    logger.info("  baseline: %s", results["baseline_regression"])
    logger.info("  model:    %s", results["regressor"])

    # --- output dir (per-district, never overwrites another district) --
    out_dir = local_model_dir(district)
    plots_dir = out_dir / "plots"

    # --- plots -----------------------------------------------------
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
        title=f"{chosen_name} — test set",
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

    centroid = {
        "latitude": float(daily["latitude"].iloc[0]),
        "longitude": float(daily["longitude"].iloc[0]),
        "elevation": float(daily["elevation"].iloc[0]),
    }

    metadata = {
        "model_version": MODEL_VERSION,
        "district": daily["district"].iloc[0],
        "state": daily["state"].iloc[0],
        **centroid,
        "source_file": RAW_DATA_PATH.name,
        "forecast_horizon_days": 7,
        "classifier_model_name": chosen_name,
        "classifier_params": chosen_clf_result["params"],
        "classifier_calibration": "sigmoid (Platt), fit on validation split, cv='prefit'",
        "regressor_model_name": chosen_reg_name,
        "regressor_params": chosen_reg_result["params"],
        "rainfall_risk_threshold_percentile": RAINFALL_RISK_THRESHOLD_PERCENTILE,
        "rainfall_risk_threshold_mm_by_month": {
            int(m): (None if pd.isna(v) else float(v))
            for m, v in preprocessor.risk_threshold_table_.items()
        },
        "climatology_mean_rainfall_mm_by_month": {
            int(m): (None if pd.isna(row["climatology_mean_rainfall_month"]) else float(row["climatology_mean_rainfall_month"]))
            for m, row in preprocessor.climatology_table_.iterrows()
        },
        "train_date_range": [SPLIT_DATES.train_start, SPLIT_DATES.train_end],
        "val_date_range": [SPLIT_DATES.val_start, SPLIT_DATES.val_end],
        "test_date_range": [SPLIT_DATES.test_start, SPLIT_DATES.test_end],
        "class_imbalance_handling": (
            "RandomForest: class_weight='balanced'. XGBoost: scale_pos_weight "
            f"={scale_pos_weight:.3f} (train neg/pos ratio). No resampling/SMOTE "
            "was used — the positive class is only mildly imbalanced (~"
            f"{train_pos_rate:.0%} by construction of the monthly tercile "
            "threshold), and synthetic resampling of lag/rolling features "
            "derived from a time series risks producing physically "
            "implausible feature combinations."
        ),
    }
    with open(out_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(out_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    _register_district_config(metadata["district"], metadata["state"], centroid)

    logger.info("Artifacts saved to %s", out_dir)
    return results


def _register_district_config(district: str, state: str, centroid: dict) -> None:
    """Auto-register this district's coordinates for live Open-Meteo
    forecasting (src/forecasting/district_registry.py), so training a new
    district also makes scripts/forecast.py work for it immediately —
    no manual district_config.json editing needed.
    """
    from src.config import DISTRICT_CONFIG_PATH

    with open(DISTRICT_CONFIG_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    registry[district] = {"state": state, **centroid}
    with open(DISTRICT_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registered %s in %s for live forecasting", district, DISTRICT_CONFIG_PATH)
