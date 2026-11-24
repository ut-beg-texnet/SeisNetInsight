"""Versioned configuration for the headless analysis workflow."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import GridParameters
from .grids import SUPPORTED_CONTEXT_AGGREGATIONS


class WorkflowError(ValueError):
    """An actionable failure with a stable identifier for CLI clients."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _object(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    unknown = sorted(value.keys() - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {label} option(s): {', '.join(unknown)}.")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _number(value, name, minimum, *, maximum=None, integer=False):
    accepted = (int,) if integer else (int, float)
    if type(value) not in accepted or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'}.")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"{name} must be {bounds}.")
    return value


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false.")
    return value


def _utc(value, name):
    value = _text(value, name)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ].+)?", value):
        raise ValueError(f"{name} must be an ISO 8601 date or timestamp.")
    try:
        date = pd.Timestamp(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO 8601 date or timestamp.") from exc
    return (date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")).isoformat()


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON option: {key}.")
        result[key] = value
    return result


def grid_settings(params: GridParameters) -> dict:
    settings = asdict(params)
    settings.pop("overwrite")
    settings["context_radius_km"] = settings.pop("swd_radius_km")
    settings["weight_context"] = settings.pop("weight_swd")
    return settings


def _grid(options, has_context, max_cells):
    defaults = grid_settings(GridParameters())
    settings = {**defaults, **_object(options, defaults, "grid")}
    for name, limit in (("lats", 90), ("lons", 180)):
        pair = settings[name]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"grid.{name} must contain two bounds.")
        lo, hi = [_number(x, f"grid.{name}", -limit, maximum=limit) for x in pair]
        if lo >= hi:
            raise ValueError(f"grid.{name} must be in increasing order.")
        settings[name] = [lo, hi]
    aggregation = settings["context_aggregation"]
    if not isinstance(aggregation, str) or aggregation not in SUPPORTED_CONTEXT_AGGREGATIONS:
        raise ValueError("grid.context_aggregation must be one of " + ", ".join(sorted(SUPPORTED_CONTEXT_AGGREGATIONS)) + ".")
    for name in ("grid_step", "subject_primary_radius_km", "subject_secondary_radius_km",
                 "gap_search_km", "context_radius_km", "half_time_years"):
        if _number(settings[name], f"grid.{name}", 0) == 0:
            raise ValueError(f"grid.{name} must be greater than zero.")
    for name in ("subject_primary_min_stations", "subject_secondary_min_stations"):
        _number(settings[name], f"grid.{name}", 1, integer=True)
    _number(settings["gap_target_angle_deg"], "grid.gap_target_angle_deg", 1, maximum=360)
    weights = ("subject_primary_weight", "subject_secondary_weight", "weight_gap", "weight_context")
    for name in weights:
        _number(settings[name], f"grid.{name}", 0)
    if not has_context:
        if options.get("weight_context", 0) > 0:
            raise ValueError("grid.weight_context requires inputs.context.")
        settings["weight_context"] = 0
    total = sum(settings[name] for name in weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("The sum of grid weights must be finite and positive.")

    # Mirror generate_grid's inclusive-step formula before allocating arrays.
    step = settings["grid_step"]
    lengths = [(hi + step - lo) / step for lo, hi in (settings["lats"], settings["lons"])]
    if any(not math.isfinite(n) or n > max_cells for n in lengths):
        raise ValueError(f"Grid exceeds max_grid_cells ({max_cells}). Increase grid.grid_step or the limit.")
    rows, cols = (math.ceil(n) for n in lengths)
    if min(rows, cols) < 2:
        raise ValueError("The grid needs at least two latitudes and longitudes. Reduce grid.grid_step.")
    if rows * cols > max_cells:
        raise ValueError(f"Grid needs {rows * cols} cells, more than max_grid_cells ({max_cells}).")
    return settings, rows * cols


def grid_parameters(settings):
    kwargs = dict(settings)
    kwargs["swd_radius_km"] = kwargs.pop("context_radius_km")
    kwargs["weight_swd"] = kwargs.pop("weight_context")
    return GridParameters(**kwargs)


def resolve_priority(options, cell_count):
    defaults = {"n_clusters": 4, "random_state": 42, "n_init": 20, "apply_feature_weights": True}
    priority = {**defaults, **_object(options, defaults, "priority")}
    _number(priority["n_clusters"], "priority.n_clusters", 2, maximum=cell_count, integer=True)
    _number(priority["n_init"], "priority.n_init", 1, integer=True)
    _number(priority["random_state"], "priority.random_state", 0, maximum=2**32 - 1, integer=True)
    _boolean(priority["apply_feature_weights"], "priority.apply_feature_weights")
    return priority


def resolve_config(raw: bytes, directory: Path) -> dict:
    """Read the public JSON schema and make defaults and input paths explicit."""
    try:
        config = _object(json.loads(raw, object_pairs_hook=_unique_keys), (
            "schema_version", "inputs", "grid", "event_window", "balltree", "priority", "maps", "max_grid_cells",
        ), "config")
        if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
            raise ValueError("schema_version must be 1.")
        sources = _object(config.get("inputs"), ("events", "stations", "context", "boundary"), "inputs")
        inputs = {}
        coordinates = ("latitude", "longitude")
        for name, columns in {
            "events": (*coordinates, "origin_time"),
            "stations": coordinates,
            "context": (*coordinates, "value"),
        }.items():
            if name == "context" and name not in sources:
                continue
            keys = ("path", "columns", "units") if name == "context" else ("path", "columns")
            source = _object(sources.get(name), keys, f"inputs.{name}")
            path = _text(source.get("path"), f"inputs.{name}.path")
            mapping = _object(source.get("columns", {}), columns, f"inputs.{name}.columns")
            actual = [_text(value, f"inputs.{name}.columns.{key}").strip().casefold() for key, value in mapping.items()]
            if len(actual) != len(set(actual)):
                raise ValueError(f"inputs.{name}.columns maps one input column more than once.")
            inputs[name] = {"path": str((directory / path).resolve()), "columns": mapping}
            if name == "context" and "units" in source:
                inputs[name]["units"] = _text(source["units"], "inputs.context.units")
        if "boundary" in sources:
            inputs["boundary"] = {"path": str((directory / _text(sources["boundary"], "inputs.boundary")).resolve())}

        window = _object(config.get("event_window", {}), ("start_inclusive", "end_exclusive"), "event_window")
        window = {key: _utc(value, f"event_window.{key}") for key, value in window.items()}
        if len(window) == 2 and pd.Timestamp(window["start_inclusive"]) >= pd.Timestamp(window["end_exclusive"]):
            raise ValueError("event_window.end_exclusive must be after start_inclusive.")
        reduction = None
        if "balltree" in config:
            reduction = _object(config["balltree"], ("distance_km", "keep"), "balltree")
            distance = _number(reduction.get("distance_km"), "balltree.distance_km", 0)
            keep = reduction.get("keep", "most_recent")
            if distance == 0 or keep not in ("first", "most_recent"):
                raise ValueError("BallTree requires a positive distance_km and keep of 'first' or 'most_recent'.")
            reduction = {"distance_km": distance, "keep": keep}

        max_cells = _number(config.get("max_grid_cells", 1_000_000), "max_grid_cells", 4, integer=True)
        grid, cell_count = _grid(config.get("grid", {}), "context" in inputs, max_cells)
        priority = resolve_priority(config.get("priority", {}), cell_count)
        maps = {"basemap": False, "dpi": 150}
        maps.update(_object(config.get("maps", {}), maps, "maps"))
        _boolean(maps["basemap"], "maps.basemap")
        _number(maps["dpi"], "maps.dpi", 30, maximum=600, integer=True)
        return {
            "schema_version": 1, "inputs": inputs, "event_window": window, "max_grid_cells": max_cells,
            "balltree": reduction, "grid": grid, "priority": priority, "maps": maps, "grid_cells": cell_count,
        }
    except (ValueError, OSError, OverflowError) as exc:
        raise WorkflowError("invalid_config", str(exc)) from exc
