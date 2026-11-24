"""Validate, run, and render a SeisNetInsight analysis from local files."""

from __future__ import annotations

import hashlib
import io
import json
import platform
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import data, grids
from .run_config import WorkflowError, grid_parameters, resolve_config

FEATURE_MAPS = {
    "subject_primary_weighted": "feature_primary.png",
    "subject_secondary_weighted": "feature_secondary.png",
    "delta_gap90_weighted": "feature_gap.png",
    "context_value": "feature_context.png",
    "composite_index": "composite_index.png",
}
GRID_FILE = "feature_composite_grid.csv"
PRIORITY_FILE = "priority_regions_grid.csv"
STATION_FILE = "map_stations.csv"
BOUNDARY_FILE = "boundary.bna"
MANIFEST_FILE = "manifest.json"


@dataclass
class PreparedRun:
    settings: dict
    source: dict
    counts: dict
    warnings: list[str]
    events: pd.DataFrame
    stations: pd.DataFrame
    context: pd.DataFrame | None
    boundary: bytes | None

    def summary(self):
        weights = grid_parameters(self.settings["grid"]).normalized_weights()
        weights.pop("swd")
        return {
            "config": self.settings, "source": self.source, "counts": self.counts,
            "normalized_weights": weights, "warnings": self.warnings,
        }


def _fingerprint(content: bytes) -> dict:
    return {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def _load_input(name, content, columns):
    loader = {"events": data.load_events, "stations": data.load_stations, "context": data.load_context_points}[name]
    kwargs = {"utc": True} if name == "events" else {}
    try:
        frame, missing = loader(content, column_map=columns, warn=False, strict_mapping=True, **kwargs)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}.")
        if frame.empty:
            raise ValueError("The input has no rows.")
        for column, limit in (("latitude", 90), ("longitude", 180)):
            values = pd.to_numeric(frame[column], errors="coerce")
            invalid = int((~values.between(-limit, limit)).sum())
            if invalid:
                raise ValueError(f"{invalid} row(s) have invalid {column}. Expected decimal degrees within ±{limit}.")
            frame[column] = values
        if name == "events" and frame["origin_time"].isna().any():
            raise ValueError(f"{int(frame['origin_time'].isna().sum())} row(s) have invalid origin_time.")
        if name == "context" and not np.isfinite(frame["value"]).all():
            raise ValueError("Context values must be finite numbers.")
        return frame
    except (ValueError, TypeError) as exc:
        raise WorkflowError("invalid_input", f"{name}: {exc}") from exc


def _prepare(config_path: Path) -> PreparedRun:
    config_path = config_path.resolve()
    try:
        raw = config_path.read_bytes()
    except OSError as exc:
        raise WorkflowError("invalid_config", f"Cannot read {config_path}: {exc}") from exc
    settings = resolve_config(raw, config_path.parent)
    source = {"config": {"path": str(config_path), **_fingerprint(raw)}, "inputs": {}}
    frames = {}
    boundary = None
    for name, spec in settings["inputs"].items():
        path = Path(spec["path"])
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise WorkflowError("invalid_input", f"{name}: cannot read {path}: {exc}") from exc
        source["inputs"][name] = {"path": str(path), **_fingerprint(content)}
        if name == "boundary":
            if not content.strip():
                raise WorkflowError("invalid_input", "boundary: the BNA file is empty.")
            boundary = content
        else:
            frames[name] = _load_input(name, content, spec["columns"])

    events, stations, context = frames["events"], frames["stations"], frames.get("context")
    counts = {"events_input": len(events), "stations_input": len(stations), "context_input": len(context) if context is not None else 0}
    window = settings["event_window"]
    for key, comparison in (("start_inclusive", "ge"), ("end_exclusive", "lt")):
        if key in window:
            events = events.loc[getattr(events["origin_time"], comparison)(pd.Timestamp(window[key]))]
    counts["events_after_window"] = len(events)
    if events.empty:
        raise WorkflowError("invalid_input", "No events remain in the configured UTC event window.")
    if settings["balltree"] is not None:
        options = settings["balltree"]
        events = data.balltree_reduce_events(events, distance_threshold_km=options["distance_km"], keep=options["keep"])
    counts["events_after_balltree"] = len(events)
    params = grid_parameters(settings["grid"])
    counts["gap_events_in_bounds"] = len(grids._clip_to_aoi(events, params))
    counts["gap_stations"] = len(grids._filter_stations(stations))
    counts["map_stations_in_bounds"] = len(grids._clip_to_aoi(stations, params))
    if not counts["gap_events_in_bounds"] or not counts["gap_stations"]:
        raise WorkflowError("invalid_input", "Gap analysis requires events inside the grid bounds and at least one non-AM station.")
    warnings = []
    if context is None:
        warnings.append("No context input supplied. The effective context weight is zero.")
    return PreparedRun(settings, source, counts, warnings, events, stations, context, boundary)


def validate(config_path: Path) -> dict:
    """Check local input files, report effective settings, and estimate grid size."""
    return _prepare(config_path).summary()


def _runtime() -> dict:
    modules = Path(__file__).parent.glob("*.py")
    source_catalog = {file.name: _fingerprint(file.read_bytes())["sha256"] for file in sorted(modules)}
    try:
        version = metadata.version("seisnetinsight")
    except metadata.PackageNotFoundError:
        version = None
    return {
        "seisnetinsight_version": version,
        "source_sha256": _fingerprint(json.dumps(source_catalog, sort_keys=True).encode())["sha256"],
        "python": platform.python_version(),
        "dependencies": {name: metadata.version(name) for name in (
            "numpy", "pandas", "scikit-learn", "pyproj", "matplotlib", "cartopy", "cmcrameri",
        )},
    }


def _artifact(path: Path) -> dict:
    return {"path": path.name, **_fingerprint(path.read_bytes())}


@contextmanager
def _destination(output_dir: Path):
    """Stage all artifacts next to the target and publish them together."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise WorkflowError("output_exists", f"{output_dir} already exists. Choose a new output directory.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".seisnetinsight-", dir=output_dir.parent) as workspace:
        stage = Path(workspace) / "result"
        stage.mkdir()
        yield stage
        if output_dir.exists():
            raise WorkflowError("output_exists", f"{output_dir} already exists. Choose a new output directory.")
        stage.rename(output_dir)


def _render_maps(stage, composite, priority, stations, boundary, settings, progress):
    from matplotlib import pyplot as plt

    from .legacy_maps import LegacyMapConfig, figure_png_bytes, render_legacy_contour, render_priority_clusters

    params = grid_parameters(settings["grid"])
    options = settings["maps"]
    config = LegacyMapConfig(show_roads=False, show_cities=False, show_states=options["basemap"], show_basemap=options["basemap"])
    artifacts = {}
    if progress:
        progress(0, "Rendering maps")
    def save(key, filename, fig):
        path = stage / filename
        try:
            path.write_bytes(figure_png_bytes(fig, dpi=options["dpi"]))
            artifacts[key] = _artifact(path)
        finally:
            plt.close(fig)

    for feature, filename in FEATURE_MAPS.items():
        if feature == "context_value" and "context" not in settings["inputs"]:
            continue
        fig = render_legacy_contour(composite, feature, params, config=config, bna_bytes=boundary)
        save(feature, filename, fig)
    fig = render_priority_clusters(priority, params, stations=stations, config=config, bna_bytes=boundary)
    save("priority_map", "priority_regions.png", fig)
    return artifacts


def _finish(stage, output_dir, kind, details, artifacts):
    manifest = {
        "schema_version": 1, "kind": kind, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime(), **details, "artifacts": artifacts,
    }
    (stage / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output_dir = output_dir.resolve()
    return {
        "manifest": str(output_dir / MANIFEST_FILE),
        "artifacts": {key: str(output_dir / value["path"]) for key, value in artifacts.items()},
        "counts": details["counts"], "warnings": details["warnings"],
    }


def run(config_path: Path, output_dir: Path, *, progress=None) -> dict:
    """Compute the feature, composite, and priority grids and publish a complete run."""
    from .maps import DEFAULT_PRIORITY_FEATURES, classify_priority_clusters

    with _destination(output_dir) as stage:
        prepared = _prepare(config_path)
        details = prepared.summary()
        params = grid_parameters(prepared.settings["grid"])
        grid = grids.generate_grid(params)
        details["counts"]["grid_cells"] = grid.size
        details["event_recency_reference_utc"] = prepared.events["origin_time"].max().isoformat()
        gap_events = grids._clip_to_aoi(prepared.events, params)
        details["gap_recency_reference_utc"] = gap_events["origin_time"].max().isoformat()
        subject = grids.compute_subject_grids(prepared.events, prepared.stations, grid, params, progress=progress)
        gap = grids.compute_gap_grid(prepared.events, prepared.stations, grid, params, progress=progress)
        context = grids.compute_context_grid(prepared.context, grid, params, progress=progress)
        composite = grids.compute_composite_index(grids.merge_grids(subject, gap, context), params)
        if not np.isfinite(composite.select_dtypes(include="number").to_numpy()).all():
            raise WorkflowError("analysis_failed", "The calculated grid has non-finite values. Check the input values and settings.")
        weights = details["normalized_weights"]
        feature_weights = dict(zip(DEFAULT_PRIORITY_FEATURES, (weights[k] for k in ("subject_primary", "subject_secondary", "gap", "context"))))
        details["clustering_feature_weights"] = feature_weights
        with threadpool_limits(limits=1):
            priority = classify_priority_clusters(composite, feature_weights=feature_weights, **prepared.settings["priority"])
        details["counts"]["priority_labels"] = priority["priority_label"].value_counts().to_dict()
        if len(details["counts"]["priority_labels"]) < prepared.settings["priority"]["n_clusters"]:
            details["warnings"].append("Clustering produced fewer distinct priority groups than requested.")

        stations = grids._clip_to_aoi(prepared.stations, params)[["latitude", "longitude"]]
        artifacts = {}
        for key, name, frame in (
            ("feature_composite_grid", GRID_FILE, composite), ("priority_grid", PRIORITY_FILE, priority),
            ("map_stations", STATION_FILE, stations),
        ):
            path = stage / name
            frame.to_csv(path, index=False)
            artifacts[key] = _artifact(path)
        if prepared.boundary is not None:
            path = stage / BOUNDARY_FILE
            path.write_bytes(prepared.boundary)
            artifacts["boundary"] = _artifact(path)
        artifacts.update(_render_maps(stage, composite, priority, stations, prepared.boundary, prepared.settings, progress))
        return _finish(stage, output_dir, "analysis", details, artifacts)


def render(manifest_path: Path, output_dir: Path, *, progress=None) -> dict:
    """Render the saved grids and overlays from an existing analysis manifest."""
    manifest_path = manifest_path.resolve()
    try:
        raw = manifest_path.read_bytes()
        source = json.loads(raw)
        if source["schema_version"] != 1 or source["kind"] != "analysis":
            raise ValueError("Expected a schema version 1 analysis manifest from the run command.")
        settings = source["config"]
        artifacts = source["artifacts"]

        def checked_artifact(key, expected_name):
            record = artifacts[key]
            if record["path"] != expected_name:
                raise ValueError(f"Unexpected artifact path for {key}.")
            path = manifest_path.parent / expected_name
            if not path.resolve().is_relative_to(manifest_path.parent):
                raise ValueError(f"Artifact is outside the analysis directory: {expected_name}.")
            content = path.read_bytes()
            if record["sha256"] != _fingerprint(content)["sha256"]:
                raise ValueError(f"Saved artifact changed: {expected_name}.")
            return content

        composite, priority, stations = (
            pd.read_csv(io.BytesIO(checked_artifact(key, name)), float_precision="round_trip")
            for key, name in (("feature_composite_grid", GRID_FILE), ("priority_grid", PRIORITY_FILE), ("map_stations", STATION_FILE))
        )
        boundary = checked_artifact("boundary", BOUNDARY_FILE) if "boundary" in artifacts else None
        details = {
            "source_run": {"path": str(manifest_path), **_fingerprint(raw)},
            "config": settings, "counts": source["counts"], "warnings": source["warnings"],
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise WorkflowError("invalid_manifest", f"Cannot render {manifest_path}: {exc}") from exc
    with _destination(output_dir) as stage:
        maps = _render_maps(stage, composite, priority, stations, boundary, settings, progress)
        return _finish(stage, output_dir, "render", details, maps)
