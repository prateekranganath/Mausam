"""District -> (state, latitude, longitude, elevation) lookup for live
forecasting. Coordinates are the same station-centroid values used when
building each district's training data (see src/data/district.py), so a
live Open-Meteo pull and the historical training series describe the same
physical location.

Add a district by adding an entry to district_config.json — nothing else
in the forecasting code needs to change per-district.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from src.config import DISTRICT_CONFIG_PATH


@dataclass(frozen=True)
class DistrictConfig:
    district: str
    state: str
    latitude: float
    longitude: float
    elevation: float


def _load_registry() -> dict[str, DistrictConfig]:
    with open(DISTRICT_CONFIG_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        name: DistrictConfig(district=name, **fields) for name, fields in raw.items()
    }


_REGISTRY = _load_registry()


def list_district_configs() -> list[DistrictConfig]:
    return list(_REGISTRY.values())


def get_district_config(district: str) -> DistrictConfig:
    for name, cfg in _REGISTRY.items():
        if name.casefold() == district.casefold():
            return cfg
    raise KeyError(
        f"No live-forecasting config for district {district!r}. "
        f"Known districts: {sorted(_REGISTRY)}. Add an entry to "
        "src/forecasting/district_config.json (use the station-centroid "
        "lat/lon from src.data.district.get_district_daily_series output "
        "for consistency with the trained model)."
    )
