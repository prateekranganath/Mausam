"""Does adding ENSO, IOD or MJO actually improve the rainfall-risk model?

The problem statement asks for these three indices as model inputs. This
script decides whether they earn a place, by measurement rather than by
assertion, and it is expected to return a partly NEGATIVE result.

THE PRIOR, stated up front so a null result is not quietly reinterpreted
afterwards. The pooled model trains on 2021-01-01..2023-06-30. ONI and DMI
are monthly, so that window contains roughly 30 distinct values of each,
and almost all of it sits inside the prolonged 2020-2023 La Nina while the
El Nino transition falls in the VALIDATION split. As features they can
therefore act as a proxy for "which year is it" -- which can look like a
validation gain and cannot generalise. MJO is different in kind: it is
daily and cycles every 30-60 days, so even 2.5 years holds ~20 independent
cycles, and its timescale actually matches a 7-day forecast horizon.

METHOD. Three arms over identical splits, identical preprocessing and
identical hyperparameters (the ones train_all_india already selected), so
the ONLY thing that varies is the feature set. Re-running the grid search
per arm would let hyperparameter luck contaminate the comparison.

  1. baseline          the shipped 34 features
  2. + MJO             amplitude and cyclic phase sin/cos
  3. + MJO + ONI + DMI everything the problem statement asks for

Selection is on VALIDATION ROC-AUC, matching _search_classifier. Test
metrics are reported for all arms but must not be used to choose, or the
test split stops being held out.

Usage:
    python scripts/run_climate_ablation.py
    python scripts/run_climate_ablation.py --sample-districts 60   # faster
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.climate.indices import (
    MJO_FEATURE_COLUMNS,
    SEASONAL_FEATURE_COLUMNS,
    attach_climate_indices,
    load_dmi,
    load_mjo,
    load_oni,
)
from src.config import (
    ALL_INDIA_PROCESSED_CACHE,
    ALL_INDIA_SPLIT_DATES,
    MODELS_DIR,
    RAINFALL_RISK_THRESHOLD_PERCENTILE,
)
from src.features.engineering import (
    FEATURE_COLUMNS,
    RainfallFeaturePreprocessor,
    build_features_multi_district,
    compute_target_multi_district,
)
from src.ml import evaluate as ev

logger = logging.getLogger("ablation")

OUTPUT_PATH = MODELS_DIR / "ablation_climate_indices.json"

# The configuration train_all_india selected for the shipped classifier.
# Held fixed across arms so the comparison isolates the features.
CLASSIFIER_PARAMS = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1}

ARMS = {
    "baseline": [],
    "mjo": list(MJO_FEATURE_COLUMNS),
    "mjo_enso_iod": list(MJO_FEATURE_COLUMNS) + list(SEASONAL_FEATURE_COLUMNS),
}


def split(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = frame["date_of_record"]
    return frame[(dates >= start) & (dates <= end)]


def train_arm(name: str, extra: list[str], data: dict) -> dict:
    from xgboost import XGBClassifier

    columns = list(FEATURE_COLUMNS) + extra
    X_train = data["train_X"][columns]
    X_val = data["val_X"][columns]
    X_test = data["test_X"][columns]

    positives = int(data["train_y"].sum())
    scale = (len(data["train_y"]) - positives) / max(positives, 1)

    started = time.time()
    model = XGBClassifier(
        **CLASSIFIER_PARAMS,
        scale_pos_weight=scale,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, data["train_y"])

    val = ev.classification_metrics(data["val_y"], model.predict_proba(X_val)[:, 1])
    test = ev.classification_metrics(data["test_y"], model.predict_proba(X_test)[:, 1])

    importances = dict(zip(columns, model.feature_importances_))
    return {
        "arm": name,
        "n_features": len(columns),
        "extra_features": extra,
        "val_roc_auc": val["roc_auc"],
        "val_pr_auc": val["pr_auc"],
        "test_roc_auc": test["roc_auc"],
        "test_pr_auc": test["pr_auc"],
        "test_brier": test["brier_score"],
        "train_seconds": round(time.time() - started, 1),
        "extra_feature_importance": {k: round(float(importances[k]), 5) for k in extra},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sample-districts",
        type=int,
        default=None,
        help="Use a random sample of districts for a faster run. Sampling is by district, never by row, so no district is split across arms.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("src.utils.http_cache").setLevel(logging.WARNING)

    if not ALL_INDIA_PROCESSED_CACHE.exists():
        raise SystemExit(
            f"{ALL_INDIA_PROCESSED_CACHE} not found. Run scripts/train_all_india_model.py first "
            "so the pooled daily table is cached."
        )

    logger.info("Loading pooled daily table...")
    daily = pd.read_parquet(ALL_INDIA_PROCESSED_CACHE)

    if args.sample_districts:
        rng = np.random.default_rng(args.seed)
        districts = daily["district"].unique()
        keep = rng.choice(districts, size=min(args.sample_districts, len(districts)), replace=False)
        daily = daily[daily["district"].isin(keep)]
        logger.info("Sampled %d districts (%d rows)", len(keep), len(daily))

    logger.info("Building features (impute_as_zero=False, matching the pooled model)...")
    features = build_features_multi_district(daily, impute_as_zero=False)
    features = compute_target_multi_district(features)

    logger.info("Fetching climate indices and attaching (backward as-of join)...")
    oni, dmi, mjo = load_oni(), load_dmi(), load_mjo()
    features = attach_climate_indices(features, oni=oni, dmi=dmi, mjo=mjo)

    s = ALL_INDIA_SPLIT_DATES
    train = split(features, s.train_start, s.train_end)
    val = split(features, s.val_start, s.val_end)
    test = split(features, s.test_start, s.test_end)

    # Fit the preprocessor on TRAIN ONLY, exactly as the shipped pipeline
    # does, then reuse the identical fitted object for every arm.
    preprocessor = RainfallFeaturePreprocessor(RAINFALL_RISK_THRESHOLD_PERCENTILE)
    preprocessor.fit(train)

    prepared = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        X, labelled = preprocessor.transform(part)
        valid = labelled["insufficient_rainfall_next_7_days"].notna()
        # The extra columns are not in preprocessor.feature_columns, so carry
        # them across from the labelled frame and median-impute the early rows
        # that predate the first published index value.
        extras = labelled.loc[valid, MJO_FEATURE_COLUMNS + SEASONAL_FEATURE_COLUMNS]
        X = pd.concat([X[valid], extras], axis=1)
        X = X.fillna(X.median(numeric_only=True))
        prepared[f"{name}_X"] = X
        prepared[f"{name}_y"] = labelled.loc[valid, "insufficient_rainfall_next_7_days"].astype(int)
        logger.info("%s: %d rows, positive rate %.3f", name, len(X), prepared[f"{name}_y"].mean())

    # How much do the seasonal indices actually vary in training? This is
    # the number the prior above rests on, so report it rather than assert it.
    index_variation = {}
    for column in SEASONAL_FEATURE_COLUMNS + ["mjo_amplitude"]:
        train_lo, train_hi = float(prepared["train_X"][column].min()), float(prepared["train_X"][column].max())
        val_lo, val_hi = float(prepared["val_X"][column].min()), float(prepared["val_X"][column].max())
        # The decisive diagnostic. A tree can only split on values it has
        # seen; if validation lies almost entirely outside the training
        # range, the feature is not being generalised from, it is being
        # extrapolated past, and every split learned on it is meaningless
        # there.
        overlap = max(0.0, min(train_hi, val_hi) - max(train_lo, val_lo))
        val_span = val_hi - val_lo
        index_variation[column] = {
            "train_unique_values": int(prepared["train_X"][column].nunique()),
            "train_range": [round(train_lo, 3), round(train_hi, 3)],
            "val_range": [round(val_lo, 3), round(val_hi, 3)],
            "val_share_inside_train_range": round(
                float((prepared["val_X"][column].between(train_lo, train_hi)).mean()), 3
            ),
            "range_overlap_fraction_of_val": round(overlap / val_span, 3) if val_span > 0 else 1.0,
        }

    results = []
    for name, extra in ARMS.items():
        logger.info("Training arm '%s' (%d extra features)...", name, len(extra))
        results.append(train_arm(name, extra, prepared))

    best = max(results, key=lambda r: r["val_roc_auc"])
    baseline = next(r for r in results if r["arm"] == "baseline")

    print("\n" + "=" * 78)
    print("CLIMATE INDEX ABLATION")
    print("=" * 78)
    print(f"{'arm':<16} {'feats':>6} {'val ROC':>9} {'val PR':>8} {'test ROC':>9} {'test PR':>8}")
    for r in results:
        print(
            f"{r['arm']:<16} {r['n_features']:>6} {r['val_roc_auc']:>9.4f} {r['val_pr_auc']:>8.4f} "
            f"{r['test_roc_auc']:>9.4f} {r['test_pr_auc']:>8.4f}"
        )

    print(f"\nSelected on validation ROC-AUC: '{best['arm']}'")
    print(
        f"  vs baseline: val ROC {best['val_roc_auc'] - baseline['val_roc_auc']:+.4f}, "
        f"test PR {best['test_pr_auc'] - baseline['test_pr_auc']:+.4f}"
    )

    print("\nWHY -- do the indices even overlap between train and validation?")
    print(f"  {'feature':<15} {'uniq':>5} {'train range':>18} {'val range':>18} {'val inside train':>17}")
    for column, stats in index_variation.items():
        print(
            f"  {column:<15} {stats['train_unique_values']:>5} "
            f"{str(stats['train_range']):>18} {str(stats['val_range']):>18} "
            f"{stats['val_share_inside_train_range']:>16.1%}"
        )
    starved = [c for c, s in index_variation.items() if s["val_share_inside_train_range"] < 0.5]
    if starved:
        print(
            f"\n  {', '.join(starved)}: most validation values fall OUTSIDE the range ever seen in\n"
            "  training. A tree can only split on values it has seen, so there is nothing to\n"
            "  generalise from -- the model is extrapolating past the end of its own feature.\n"
            "  This is the ~2.5-year training window showing up directly, not a tuning problem."
        )

    print("\nDoes each arm beat the baseline on the selection metric?")
    for r in results:
        if r["arm"] == "baseline":
            continue
        val_gain = r["val_roc_auc"] - baseline["val_roc_auc"]
        test_gain = r["test_pr_auc"] - baseline["test_pr_auc"]
        if val_gain > 0 and test_gain > 0:
            verdict = "gain holds on both"
        elif val_gain > 0:
            verdict = "val gain does NOT survive on test -- era proxy"
        elif test_gain > 0:
            verdict = "test-only gain; rejected (selecting on test would leak)"
        else:
            verdict = "no gain"
        print(f"  {r['arm']:<16} val ROC {val_gain:+.4f}   test PR {test_gain:+.4f}   {verdict}")

    payload = {
        "method": (
            "Three arms, identical splits, identical preprocessing, identical hyperparameters "
            "(the ones train_all_india selected). Only the feature set varies. Selection on "
            "validation ROC-AUC; test metrics reported but not used to choose."
        ),
        "classifier_params": CLASSIFIER_PARAMS,
        "train_date_range": [s.train_start, s.train_end],
        "val_date_range": [s.val_start, s.val_end],
        "test_date_range": [s.test_start, s.test_end],
        "row_counts": {k.replace("_y", ""): int(len(v)) for k, v in prepared.items() if k.endswith("_y")},
        "n_districts": int(daily["district"].nunique()),
        "seasonal_index_variation_in_training": index_variation,
        "arms": results,
        "selected_arm": best["arm"],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
