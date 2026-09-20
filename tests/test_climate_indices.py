import numpy as np
import pandas as pd
import pytest
import requests

from src.climate import indices as ci


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(ci, "CLIMATE_INDEX_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ci.time, "sleep", lambda _s: None)


ONI_SAMPLE = """ SEAS  YR   TOTAL   ANOM
  DJF 1950  25.01  -1.32
  JFM 1950  25.36  -1.20
  NDJ 2025  27.10   0.31
  AMJ 2026  28.74   0.95
  MJJ 2026  29.02   1.39
  JJA 2026  29.09   1.80
"""

DMI_SAMPLE = """ 1870 2026
1870    -0.438    -0.337     0.177    -0.048    -0.480    -0.548    -0.650    -0.522    -0.728    -0.636    -0.401    -0.376
2025    -0.196     0.017     0.058     0.149     0.107    -0.014    -0.072    -0.420    -0.467    -0.448    -0.256    -0.017
2026     0.123     0.529     0.285     0.279     0.146 -9999.000 -9999.000 -9999.000 -9999.000 -9999.000 -9999.000 -9999.000
-9999
DMI HadISST1.1
Created Sat Jul 25 13:56:30 MDT 2026
using SST anomaly 10S:10N,50E-70E minus 10S:0,90E-110E area averaged
Timeseries output created at NOAA PSL
https://psl.noaa.gov/gcos_wgsp/timeseries/DMI
Preliminary.
"""


# --- ONI parsing --------------------------------------------------------------

def test_oni_seasons_map_to_their_centre_month():
    frame = ci.parse_oni(ONI_SAMPLE)
    # DJF 1950 spans Dec 1949 - Feb 1950 and is centred on January 1950.
    assert frame["date"].iloc[0] == pd.Timestamp("1950-01-01")
    # JJA 2026 is centred on July.
    assert frame["date"].iloc[-1] == pd.Timestamp("2026-07-01")
    assert frame["oni"].iloc[-1] == pytest.approx(1.80)


def test_oni_ignores_the_header_row():
    frame = ci.parse_oni(ONI_SAMPLE)
    assert len(frame) == 6
    assert frame["oni"].notna().all()


def test_oni_is_sorted_ascending():
    frame = ci.parse_oni(ONI_SAMPLE)
    assert frame["date"].is_monotonic_increasing


def test_oni_raises_rather_than_returning_an_empty_frame():
    with pytest.raises(ci.ClimateIndexError, match="No ONI rows"):
        ci.parse_oni("garbage\nnot a data file\n")


@pytest.mark.parametrize(
    "value,expected",
    [(1.8, "el_nino"), (0.5, "el_nino"), (0.2, "neutral"), (-0.49, "neutral"), (-0.5, "la_nina"), (-2.0, "la_nina")],
)
def test_oni_phase_uses_noaas_half_degree_threshold(value, expected):
    assert ci.oni_phase(value) == expected


# --- DMI parsing --------------------------------------------------------------

def test_dmi_expands_twelve_monthly_columns_into_rows():
    frame = ci.parse_dmi(DMI_SAMPLE)
    jan_1870 = frame[frame["date"] == pd.Timestamp("1870-01-01")]
    assert jan_1870["dmi"].iloc[0] == pytest.approx(-0.438)
    dec_1870 = frame[frame["date"] == pd.Timestamp("1870-12-01")]
    assert dec_1870["dmi"].iloc[0] == pytest.approx(-0.376)


def test_dmi_drops_the_unpublished_tail_of_the_current_year():
    """2026 has real values through May and -9999 after; `as_of` must be May."""
    frame = ci.parse_dmi(DMI_SAMPLE)
    assert frame["date"].max() == pd.Timestamp("2026-05-01")
    assert frame["dmi"].max() < 1000  # no sentinel survived


def test_dmi_does_not_parse_the_trailing_prose_as_data():
    """The file ends with a bare -9999, a 'Created ...' line and several
    prose lines. Any of those becoming a row would corrupt the series."""
    frame = ci.parse_dmi(DMI_SAMPLE)
    assert len(frame) == 12 + 12 + 5  # 1870 full, 2025 full, 2026 Jan-May
    assert frame["date"].dt.year.isin({1870, 2025, 2026}).all()


def test_dmi_raises_rather_than_returning_an_empty_frame():
    with pytest.raises(ci.ClimateIndexError, match="No DMI rows"):
        ci.parse_dmi("Created Sat Jul 25\nPreliminary.\n")


@pytest.mark.parametrize(
    "value,expected",
    [(0.9, "positive_iod"), (0.4, "positive_iod"), (0.0, "neutral"), (-0.4, "negative_iod")],
)
def test_dmi_phase_thresholds(value, expected):
    assert ci.dmi_phase(value) == expected


# --- MJO parsing and the cyclic-phase encoding --------------------------------

def rmm_table(rows):
    header = "amplitude\tphase\tRMM1\tRMM2"
    blank = "\t\t\t"
    body = ["\t".join(str(v) for v in r) for r in rows]
    return "\n".join([header, blank, *body]) + "\n"


def test_mjo_table_rows_are_parsed_in_order():
    text = rmm_table([[1.1, 7.0, -0.5, 1.0], [1.2, 8.0, -0.9, 0.6]])
    assert ci._parse_rmm_table(text) == [[1.1, 7.0, -0.5, 1.0], [1.2, 8.0, -0.9, 0.6]]


def test_mjo_unavailable_days_are_dropped_not_read_as_values():
    """IRI returns a lone `1000` field for a day it has no data for."""
    text = rmm_table([[1.1, 7.0, -0.5, 1.0]]) + "1000\n"
    assert ci._parse_rmm_table(text) == [[1.1, 7.0, -0.5, 1.0]]


def test_mjo_phase_encoding_keeps_phase_8_adjacent_to_phase_1():
    """The whole point of sin/cos: as integers, 8 and 1 look maximally far
    apart, but they are neighbours in the physical cycle."""
    frame = ci.add_mjo_phase_encoding(pd.DataFrame({"mjo_phase": [1.0, 2.0, 8.0]}))
    p1 = np.array([frame["mjo_phase_sin"][0], frame["mjo_phase_cos"][0]])
    p2 = np.array([frame["mjo_phase_sin"][1], frame["mjo_phase_cos"][1]])
    p8 = np.array([frame["mjo_phase_sin"][2], frame["mjo_phase_cos"][2]])
    assert np.linalg.norm(p1 - p8) == pytest.approx(np.linalg.norm(p1 - p2))


def test_raw_integer_phase_is_not_offered_as_a_model_feature():
    assert "mjo_phase" not in ci.MJO_FEATURE_COLUMNS
    assert set(ci.MJO_FEATURE_COLUMNS) == {"mjo_amplitude", "mjo_phase_sin", "mjo_phase_cos"}


@pytest.mark.parametrize(
    "phase,amplitude,fragment",
    [
        (3.0, 1.5, "Indian Ocean"),
        (7.0, 1.5, "west Pacific"),
        (5.0, 1.5, "transition"),
        (3.0, 0.4, "weak"),
    ],
)
def test_mjo_phase_label(phase, amplitude, fragment):
    assert fragment in ci.mjo_phase_label(phase, amplitude)


def test_mjo_alignment_failure_raises_instead_of_shifting_the_series(monkeypatch):
    """The guard that matters most: if the verification day's values do not
    match the range query's last row, dates are untrustworthy and we refuse
    rather than attach a silently time-shifted feature."""
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-16", "2026-09-17"]),
        "mjo_amplitude": [0.58, 0.81],
        "mjo_phase": [8.0, 8.0],
        "mjo_rmm1": [-0.52, -0.59],
        "mjo_rmm2": [0.26, 0.56],
    })
    monkeypatch.setattr(ci, "_get_text", lambda *a, **k: rmm_table([[9.99, 2.0, 1.0, 1.0]]))
    with pytest.raises(ci.ClimateIndexError, match="alignment check FAILED"):
        ci._verify_rmm_alignment(frame)


def test_mjo_alignment_passes_when_values_match(monkeypatch):
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-17"]),
        "mjo_amplitude": [0.8125413],
        "mjo_phase": [8.0],
        "mjo_rmm1": [-0.5887212],
        "mjo_rmm2": [0.5600275],
    })
    monkeypatch.setattr(
        ci, "_get_text", lambda *a, **k: rmm_table([[0.8125413, 8.0, -0.5887212, 0.5600275]])
    )
    ci._verify_rmm_alignment(frame)  # must not raise


def test_mjo_missing_verification_day_raises(monkeypatch):
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-17"]),
        "mjo_amplitude": [0.81], "mjo_phase": [8.0], "mjo_rmm1": [-0.59], "mjo_rmm2": [0.56],
    })
    monkeypatch.setattr(ci, "_get_text", lambda *a, **k: "amplitude\tphase\tRMM1\tRMM2\n\t\t\t\n1000\n")
    with pytest.raises(ci.ClimateIndexError, match="interior gap|has no data"):
        ci._verify_rmm_alignment(frame)


def test_range_url_uses_rangeedges_so_row_order_is_pinned():
    url = ci._rmm_range_url(pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-18"))
    assert "RANGEEDGES" in url
    assert "last" not in url  # T/last/N/RANGE returns newest-first
    assert "01%20Sep%202026" in url


# --- the as-of join: the leakage surface --------------------------------------

def index_fixtures():
    oni = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-02-01"]), "oni": [1.0, 2.0]})
    dmi = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-02-01"]), "dmi": [0.1, 0.2]})
    mjo = ci.add_mjo_phase_encoding(pd.DataFrame({
        "date": pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-03"]),
        "mjo_amplitude": [1.0, 2.0, 3.0],
        "mjo_phase": [1.0, 2.0, 3.0],
    }))
    return oni, dmi, mjo


def test_join_is_strictly_backward_looking():
    """Day T must never carry an index value published after T. This is the
    same discipline the rolling windows in features/engineering.py apply.

    MJO values here are dated 06-01..06-03 with a 3-day publication lag, so
    they become available on 06-04..06-06 respectively.
    """
    oni, dmi, mjo = index_fixtures()
    out = ci.attach_climate_indices(
        pd.DataFrame({"date_of_record": pd.to_datetime(["2024-06-02", "2024-06-04", "2024-06-06"])}),
        oni=oni, dmi=dmi, mjo=mjo,
    )
    # 06-02: nothing published yet, so no value may be invented.
    assert np.isnan(out["mjo_amplitude"].iloc[0])
    # 06-04: 06-01's value has just become available -- and only that one.
    assert out["mjo_amplitude"].iloc[1] == pytest.approx(1.0)
    # 06-06: 06-03's value is now the newest available.
    assert out["mjo_amplitude"].iloc[2] == pytest.approx(3.0)


def test_join_never_reaches_a_value_published_the_very_next_day():
    """The off-by-one that would matter: day T must not see the value that
    becomes available on T+1."""
    oni, dmi, mjo = index_fixtures()
    out = ci.attach_climate_indices(
        pd.DataFrame({"date_of_record": pd.to_datetime(["2024-06-04", "2024-06-05"])}),
        oni=oni, dmi=dmi, mjo=mjo,
    )
    assert out["mjo_amplitude"].iloc[0] == pytest.approx(1.0)  # not 2.0, published 06-05
    assert out["mjo_amplitude"].iloc[1] == pytest.approx(2.0)


def test_join_respects_the_publication_lag_not_the_label_date():
    oni, dmi, mjo = index_fixtures()
    # ONI labelled 2024-02-01 becomes available 90 days later (2024-05-01).
    before = ci.attach_climate_indices(
        pd.DataFrame({"date_of_record": pd.to_datetime(["2024-04-30"])}), oni=oni, dmi=dmi, mjo=mjo
    )
    after = ci.attach_climate_indices(
        pd.DataFrame({"date_of_record": pd.to_datetime(["2024-05-02"])}), oni=oni, dmi=dmi, mjo=mjo
    )
    assert before["oni"].iloc[0] == pytest.approx(1.0)  # still January's
    assert after["oni"].iloc[0] == pytest.approx(2.0)   # February's, now published


def test_rows_before_any_published_value_get_nan_not_a_fabricated_number():
    oni, dmi, mjo = index_fixtures()
    out = ci.attach_climate_indices(
        pd.DataFrame({"date_of_record": pd.to_datetime(["2020-01-01"])}), oni=oni, dmi=dmi, mjo=mjo
    )
    assert np.isnan(out["oni"].iloc[0])
    assert np.isnan(out["mjo_amplitude"].iloc[0])


def test_join_preserves_the_callers_row_order():
    """A multi-district frame is grouped by district, not sorted by date.
    merge_asof needs date order internally, so the original order must be
    restored or every district's rows would be interleaved."""
    oni, dmi, mjo = index_fixtures()
    weather = pd.DataFrame({
        "date_of_record": pd.to_datetime(["2024-06-05", "2024-06-04", "2024-06-06"]),
        "district": ["B", "A", "C"],
    })
    out = ci.attach_climate_indices(weather, oni=oni, dmi=dmi, mjo=mjo)
    assert list(out["district"]) == ["B", "A", "C"]


def test_join_adds_no_helper_columns_to_the_output():
    oni, dmi, mjo = index_fixtures()
    weather = pd.DataFrame({"date_of_record": pd.to_datetime(["2024-06-05"])})
    out = ci.attach_climate_indices(weather, oni=oni, dmi=dmi, mjo=mjo)
    assert not [c for c in out.columns if c.startswith("_")]
    assert "available_from" not in out.columns


def test_join_does_not_change_row_count():
    oni, dmi, mjo = index_fixtures()
    weather = pd.DataFrame({"date_of_record": pd.date_range("2024-06-01", periods=20, freq="D")})
    out = ci.attach_climate_indices(weather, oni=oni, dmi=dmi, mjo=mjo)
    assert len(out) == 20
