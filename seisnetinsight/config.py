"""Configuration helpers for SeisNetInsight."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import streamlit as st


@dataclass
class GridParameters:
    """User-configurable parameters for grid generation and weighting."""

    lons: Tuple[float, float] = (-102.9, -100.4)
    lats: Tuple[float, float] = (30.5, 34.0)
    grid_step: float = 0.01
    subject_primary_radius_km: float = 4.0
    subject_primary_min_stations: int = 1
    subject_primary_weight: float = 0.4
    subject_secondary_radius_km: float = 10.0
    subject_secondary_min_stations: int = 1
    subject_secondary_weight: float = 0.35
    gap_search_km: float = 30.0
    gap_target_angle_deg: float = 90.0
    weight_gap: float = 0.05
    swd_radius_km: float = 25.0
    context_aggregation: str = "sum"
    weight_swd: float = 0.2
    half_time_years: float = 5.0
    overwrite: bool = False

    @property
    def context_radius_km(self) -> float:
        return self.swd_radius_km

    @context_radius_km.setter
    def context_radius_km(self, value: float) -> None:
        self.swd_radius_km = float(value)

    @property
    def weight_context(self) -> float:
        return self.weight_swd

    @weight_context.setter
    def weight_context(self, value: float) -> None:
        self.weight_swd = float(value)

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            "longitude": self.lons,
            "latitude": self.lats,
        }

    def normalized_weights(self) -> Dict[str, float]:
        weights = {
            "subject_primary": max(self.subject_primary_weight, 0.0),
            "subject_secondary": max(self.subject_secondary_weight, 0.0),
            "gap": max(self.weight_gap, 0.0),
            "context": max(self.weight_context, 0.0),
        }
        total = sum(weights.values())
        if total == 0:
            normalized = {k: 0.0 for k in weights}
        else:
            normalized = {k: v / total for k, v in weights.items()}
        # Keep the legacy SWD key readable for older callers.
        normalized["swd"] = normalized["context"]
        return normalized


def default_parameters() -> GridParameters:
    return GridParameters()


def parse_bounds(value: str, fallback: Tuple[float, float]) -> Tuple[float, float]:
    try:
        parts = [float(part.strip()) for part in value.split(",")]
    except Exception:  # pragma: no cover - defensive
        return fallback
    if len(parts) != 2:
        return fallback
    lo, hi = min(parts), max(parts)
    if lo == hi:
        hi = lo + 1e-6
    return (lo, hi)


def warn_missing_parameter(name: str, default_value: object, logger: Optional[Callable[[str], None]] = None) -> None:
    message = f"Parameter '{name}' missing; using default value {default_value!r}."
    if logger is not None:
        logger(message)
    else:
        st.warning(message)


def parameter_from_inputs(inputs: Dict[str, object], logger: Optional[Callable[[str], None]] = None) -> GridParameters:
    params = default_parameters()

    def get(keys: Iterable[str] | str, default: object) -> object:
        if isinstance(keys, str):
            keys = (keys,)
        primary = next(iter(keys))
        for key in keys:
            if key in inputs and inputs[key] not in (None, ""):
                return inputs[key]
        warn_missing_parameter(primary, default, logger)
        return default

    params.lons = parse_bounds(str(get("LONS", ",".join(map(str, params.lons)))), params.lons)
    params.lats = parse_bounds(str(get("LATS", ",".join(map(str, params.lats)))), params.lats)
    params.grid_step = float(get("GRID_STEP", params.grid_step))
    params.subject_primary_radius_km = float(
        get(("SUBJECT_PRIMARY_RADIUS_KM", "DIST_THRESHOLD_SUB4"), params.subject_primary_radius_km)
    )
    params.subject_primary_min_stations = int(
        get(("SUBJECT_PRIMARY_MIN_STATIONS", "MIN_STA_SUB4"), params.subject_primary_min_stations)
    )
    params.subject_primary_weight = float(
        get(("SUBJECT_PRIMARY_WEIGHT", "WEIGHT_SUB4"), params.subject_primary_weight)
    )
    params.subject_secondary_radius_km = float(
        get(("SUBJECT_SECONDARY_RADIUS_KM", "DIST_THRESHOLD_SUB10"), params.subject_secondary_radius_km)
    )
    params.subject_secondary_min_stations = int(
        get(("SUBJECT_SECONDARY_MIN_STATIONS", "MIN_STA_SUB10"), params.subject_secondary_min_stations)
    )
    params.subject_secondary_weight = float(
        get(("SUBJECT_SECONDARY_WEIGHT", "WEIGHT_SUB10"), params.subject_secondary_weight)
    )
    params.gap_search_km = float(get("GAP_SEARCH_KM", params.gap_search_km))
    params.gap_target_angle_deg = float(get("GAP_TARGET_ANGLE", params.gap_target_angle_deg))
    params.weight_gap = float(get("WEIGHT_GAP", params.weight_gap))
    params.context_radius_km = float(
        get(("CONTEXT_RADIUS_KM", "SWD_RADIUS_KM"), params.context_radius_km)
    )
    params.context_aggregation = str(get("CONTEXT_AGGREGATION", params.context_aggregation)).strip().lower() or "sum"
    params.weight_context = float(get(("WEIGHT_CONTEXT", "WEIGHT_SWD"), params.weight_context))
    params.half_time_years = float(get("HALF_TIME_YEARS", params.half_time_years))
    params.overwrite = bool(get("OVERWRITE", params.overwrite))
    return params


def parameter_dict(params: GridParameters) -> Dict[str, object]:
    return {
        "LONS": f"{params.lons[0]},{params.lons[1]}",
        "LATS": f"{params.lats[0]},{params.lats[1]}",
        "GRID_STEP": params.grid_step,
        "SUBJECT_PRIMARY_RADIUS_KM": params.subject_primary_radius_km,
        "SUBJECT_PRIMARY_MIN_STATIONS": params.subject_primary_min_stations,
        "SUBJECT_PRIMARY_WEIGHT": params.subject_primary_weight,
        "SUBJECT_SECONDARY_RADIUS_KM": params.subject_secondary_radius_km,
        "SUBJECT_SECONDARY_MIN_STATIONS": params.subject_secondary_min_stations,
        "SUBJECT_SECONDARY_WEIGHT": params.subject_secondary_weight,
        "GAP_SEARCH_KM": params.gap_search_km,
        "GAP_TARGET_ANGLE": getattr(params, "gap_target_angle_deg", 90.0),
        "WEIGHT_GAP": params.weight_gap,
        "CONTEXT_RADIUS_KM": params.context_radius_km,
        "CONTEXT_AGGREGATION": params.context_aggregation,
        "WEIGHT_CONTEXT": params.weight_context,
        # Persist legacy keys too so older sessions remain easy to inspect and restore.
        "SWD_RADIUS_KM": params.swd_radius_km,
        "WEIGHT_SWD": params.weight_swd,
        "HALF_TIME_YEARS": params.half_time_years,
        "OVERWRITE": params.overwrite,
    }
