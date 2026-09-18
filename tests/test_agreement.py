import pandas as pd
import pytest

from src.forecasting.agreement import compute_agreement, lookup_threshold


def _table():
    idx = pd.MultiIndex.from_tuples(
        [("Kolkata", 9), ("Kolkata", 10), ("Jaisalmer", 9)], names=["district", "month"]
    )
    return pd.Series([30.0, float("nan"), 0.0], index=idx)


def test_lookup_threshold_hit_and_misses():
    t = _table()
    assert lookup_threshold(t, "Kolkata", 9) == 30.0
    assert lookup_threshold(t, "Kolkata", 10) is None  # NaN entry
    assert lookup_threshold(t, "Kolkata", 1) is None  # absent month
    assert lookup_threshold(t, "Nowhere", 9) is None  # absent district


def test_both_sources_sufficient_agree():
    out = compute_agreement(ml_predicted_mm=52.0, open_meteo_mm=100.0, threshold_mm=30.0)
    assert out["ml_implies_insufficient"] is False
    assert out["open_meteo_implies_insufficient"] is False
    assert out["sources_agree"] is True
    assert out["difference_mm"] == 48.0


def test_both_sources_insufficient_agree():
    out = compute_agreement(ml_predicted_mm=5.0, open_meteo_mm=8.0, threshold_mm=30.0)
    assert out["sources_agree"] is True
    assert out["ml_implies_insufficient"] is True
    assert out["open_meteo_implies_insufficient"] is True


def test_sources_disagree_on_class():
    out = compute_agreement(ml_predicted_mm=10.0, open_meteo_mm=80.0, threshold_mm=30.0)
    assert out["ml_implies_insufficient"] is True
    assert out["open_meteo_implies_insufficient"] is False
    assert out["sources_agree"] is False
    assert out["magnitude_diverges"] is True


def test_dry_region_small_gap_is_not_divergence():
    # 7.19mm vs 0mm is a tiny absolute gap; the floor must stop it counting.
    out = compute_agreement(ml_predicted_mm=7.19, open_meteo_mm=0.0, threshold_mm=0.0)
    assert out["magnitude_diverges"] is False


def test_large_gap_but_within_relative_band_is_not_divergence():
    # 48mm apart but under 50% of the larger total (100mm) -> not flagged.
    out = compute_agreement(ml_predicted_mm=52.0, open_meteo_mm=100.0, threshold_mm=None)
    assert out["magnitude_diverges"] is False


def test_missing_threshold_leaves_class_fields_none_but_keeps_magnitude():
    out = compute_agreement(ml_predicted_mm=20.0, open_meteo_mm=25.0, threshold_mm=None)
    assert out["threshold_mm"] is None
    assert out["ml_implies_insufficient"] is None
    assert out["open_meteo_implies_insufficient"] is None
    assert out["sources_agree"] is None
    assert out["difference_mm"] == 5.0


def test_near_zero_threshold_is_flagged_degenerate():
    # Jaisalmer-style: a dry week is normal, so the label can't meaningfully fire.
    out = compute_agreement(ml_predicted_mm=8.0, open_meteo_mm=0.0, threshold_mm=0.0)
    assert out["threshold_degenerate"] is True
    # ...and 0mm is not "< 0mm", which is exactly why the flag matters
    assert out["open_meteo_implies_insufficient"] is False


def test_normal_threshold_is_not_degenerate():
    assert compute_agreement(50.0, 60.0, 30.0)["threshold_degenerate"] is False


def test_missing_threshold_is_not_reported_degenerate():
    assert compute_agreement(50.0, 60.0, None)["threshold_degenerate"] is False


def test_note_states_this_is_not_ground_truth():
    out = compute_agreement(1.0, 2.0, 5.0)
    assert "NOT ground-truth" in out["note"]
