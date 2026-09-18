"""District selection, aggregation, and the district config registry.

A district can have multiple physical stations (see cleaner.py for why
station identity is (name, lat, lon), not name alone). This module
aggregates station-level rows into one daily district-level time series by
averaging across stations that reported a non-missing value that day
(missing readings are excluded from the average, never treated as 0).

Two district names collide across states in this dataset (Raipur: CT/MP;
Cuddalore: TN/PY) so district lookup optionally takes a state code to
disambiguate.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

AGG_COLUMNS = [
    "avg_temp",
    "min_temp",
    "max_temp",
    "wind_speed",
    "air_pressure",
    "rainfall",
]
STATIC_COLUMNS = ["latitude", "longitude", "elevation"]


class DistrictNotFoundError(ValueError):
    pass


class AmbiguousDistrictError(ValueError):
    pass


def list_districts(df: pd.DataFrame) -> pd.DataFrame:
    """Registry of districts: row count, station count, state(s), date range."""
    grouped = df.groupby(["district", "state"]).agg(
        n_rows=("date_of_record", "size"),
        n_stations=("station_id", "nunique"),
        date_min=("date_of_record", "min"),
        date_max=("date_of_record", "max"),
    )
    return grouped.reset_index().sort_values("n_rows", ascending=False)


def resolve_district(
    df: pd.DataFrame, district: str, state: str | None = None
) -> tuple[str, str]:
    """Resolve a district name (case-insensitive) to its (district, state).

    Raises AmbiguousDistrictError if the name maps to multiple states and
    `state` was not given to disambiguate.
    """
    matches = df.loc[
        df["district"].str.casefold() == district.casefold(), ["district", "state"]
    ].drop_duplicates()

    if state is not None:
        matches = matches[matches["state"].str.casefold() == state.casefold()]

    if len(matches) == 0:
        raise DistrictNotFoundError(
            f"No district matching {district!r}"
            + (f" in state {state!r}" if state else "")
        )
    if len(matches) > 1:
        states = sorted(matches["state"].unique())
        raise AmbiguousDistrictError(
            f"District {district!r} exists in multiple states {states}; "
            "pass `state=` to disambiguate."
        )
    row = matches.iloc[0]
    return row["district"], row["state"]


def get_district_daily_series(
    df: pd.DataFrame, district: str, state: str | None = None
) -> pd.DataFrame:
    """Filter to one district and aggregate to a continuous daily series.

    Returns a DataFrame indexed by every calendar day between the
    district's first and last observation (missing days are present as
    rows of NaNs, not silently dropped) with columns:
      date_of_record, district, state, rainfall, avg_temp, min_temp,
      max_temp, wind_speed, air_pressure, latitude, longitude, elevation,
      n_stations_reporting
    """
    resolved_district, resolved_state = resolve_district(df, district, state)

    subset = df[
        (df["district"] == resolved_district) & (df["state"] == resolved_state)
    ]
    if subset.empty:
        raise DistrictNotFoundError(f"No rows for {resolved_district}, {resolved_state}")

    n_stations = subset["station_id"].nunique()
    logger.info(
        "District %s (%s): %d rows across %d stations, %s to %s",
        resolved_district,
        resolved_state,
        len(subset),
        n_stations,
        subset["date_of_record"].min().date(),
        subset["date_of_record"].max().date(),
    )

    daily = subset.groupby("date_of_record").agg(
        {**{c: "mean" for c in AGG_COLUMNS}, **{c: "mean" for c in STATIC_COLUMNS}}
    )
    n_reporting = subset.groupby("date_of_record")["rainfall"].apply(
        lambda s: s.notna().sum()
    )
    daily["n_stations_reporting"] = n_reporting

    full_index = pd.date_range(
        daily.index.min(), daily.index.max(), freq="D", name="date_of_record"
    )
    daily = daily.reindex(full_index)
    daily["n_stations_reporting"] = daily["n_stations_reporting"].fillna(0).astype(int)

    daily = daily.reset_index()
    daily["district"] = resolved_district
    daily["state"] = resolved_state

    n_calendar_days = len(daily)
    n_missing_rainfall = daily["rainfall"].isna().sum()
    logger.info(
        "Daily series: %d calendar days, %d (%.1f%%) with missing rainfall",
        n_calendar_days,
        n_missing_rainfall,
        100 * n_missing_rainfall / n_calendar_days,
    )

    cols = [
        "date_of_record",
        "district",
        "state",
        "rainfall",
        "avg_temp",
        "min_temp",
        "max_temp",
        "wind_speed",
        "air_pressure",
        "latitude",
        "longitude",
        "elevation",
        "n_stations_reporting",
    ]
    return daily[cols]
