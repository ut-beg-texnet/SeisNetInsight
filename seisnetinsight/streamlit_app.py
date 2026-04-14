"""Streamlit application for SeisNetInsight."""

from __future__ import annotations

import datetime as dt
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import zipfile
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from cmcrameri import cm as cmc
from cartopy import config as cartopy_config
from cartopy.io import Downloader
import cartopy.feature as cfeature
from PIL import Image
import matplotlib as mpl

from seisnetinsight.config import (
    GridParameters,
    default_parameters,
    parameter_from_inputs,
)
from seisnetinsight.data import (
    EXPECTED_EVENT_COLUMNS,
    EXPECTED_STATION_COLUMNS,
    COLUMN_ALIASES,
    balltree_reduce_events,
    load_context_points,
    load_events,
    load_stations,
)
from seisnetinsight.grids import (
    GridDefinition,
    compute_composite_index,
    compute_context_grid,
    compute_gap_grid,
    compute_subject_grids,
    generate_grid,
    merge_grids,
)
from seisnetinsight.legacy_maps import (
    LegacyMapConfig,
    figure_png_bytes,
    render_legacy_contour,
    render_priority_clusters,
)
from seisnetinsight.maps import PRIORITY_COLORS, PRIORITY_LEVELS, classify_priority_clusters
from seisnetinsight.sessions import SessionFiles, SessionState, list_sessions


def _default_event_columns() -> Dict[str, str]:
    return {
        "latitude": "latitude",
        "longitude": "longitude",
        "origin_time": "origin_time",
    }


def _default_station_columns() -> Dict[str, str]:
    return {
        "latitude": "latitude",
        "longitude": "longitude",
    }


def _default_context_columns() -> Dict[str, str]:
    return {
        "latitude": "latitude",
        "longitude": "longitude",
        "value": "value",
    }


def _normalize_column_mapping(raw: Dict[str, str], defaults: Dict[str, str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key, fallback in defaults.items():
        value = raw.get(key, "")
        normalized[key] = value.strip() or fallback
    return normalized


def _extract_columns(uploaded_file) -> List[str]:
    if uploaded_file is None:
        return []
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    try:
        preview = pd.read_csv(uploaded_file, nrows=0)
    except Exception:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        return []
    columns = [str(col) for col in preview.columns]
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    return columns


def _match_column(canonical: str, available: List[str], current: str) -> str:
    if current:
        for candidate in available:
            if candidate == current:
                return candidate
        for candidate in available:
            if candidate.lower() == current.lower():
                return candidate
    for alias in COLUMN_ALIASES.get(canonical, []):
        for candidate in available:
            if candidate.lower() == alias.lower():
                return candidate
    for candidate in available:
        if candidate.lower() == canonical.lower():
            return candidate
    return ""


def _render_column_selectors(
    fields: List[tuple[str, str]],
    available_columns: List[str],
    current_mapping: Dict[str, str],
    defaults: Dict[str, str],
    key_prefix: str,
    field_help: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    if not available_columns:
        return current_mapping
    placeholder = "Select column…"
    columns_ui = st.columns(len(fields))
    selections: Dict[str, str] = {}
    for idx, (canonical, label) in enumerate(fields):
        default_choice = _match_column(canonical, available_columns, current_mapping.get(canonical, ""))
        index = 0
        if default_choice:
            try:
                index = available_columns.index(default_choice) + 1
            except ValueError:
                index = 0
        options = [placeholder] + available_columns
        selection = columns_ui[idx].selectbox(
            label,
            options,
            index=index,
            key=f"{key_prefix}_{canonical}",
            help=field_help.get(canonical) if field_help else None,
        )
        if selection == placeholder:
            selection = default_choice or current_mapping.get(canonical, "")
        selections[canonical] = selection
    return _normalize_column_mapping(selections, defaults)


@dataclass
class WorkingSession:
    name: str
    parameters: GridParameters = field(default_factory=default_parameters)
    events: Optional[pd.DataFrame] = None
    stations: Optional[pd.DataFrame] = None
    context: Optional[pd.DataFrame] = None
    grid: Optional[GridDefinition] = None
    grids: Dict[str, pd.DataFrame] = field(default_factory=dict)
    column_warnings: Dict[str, List[str]] = field(default_factory=dict)
    balltree_enabled: bool = False
    balltree_distance: float = 1.0
    storage: Optional[SessionState] = None
    bna_files: Dict[str, bytes] = field(default_factory=dict)
    events_columns: Dict[str, str] = field(default_factory=_default_event_columns)
    stations_columns: Dict[str, str] = field(default_factory=_default_station_columns)
    context_columns: Dict[str, str] = field(default_factory=_default_context_columns)
    context_label: str = "Context layer"

    @property
    def data_loaded(self) -> bool:
        return self.events is not None and self.stations is not None

    @property
    def has_context_input(self) -> bool:
        return self.context is not None and not self.context.empty

    @property
    def has_context_map_data(self) -> bool:
        return self.has_context_input

    def ensure_grid(self) -> GridDefinition:
        if self.grid is None:
            self.grid = generate_grid(self.parameters)
        return self.grid

    def merged(self) -> Optional[pd.DataFrame]:
        frames = []
        if "subject" in self.grids:
            frames.append(self.grids["subject"])
        if "gap" in self.grids:
            frames.append(self.grids["gap"])
        if "context" in self.grids:
            frames.append(self.grids["context"])
        if not frames:
            return None
        merged = merge_grids(*frames)
        if "composite" in self.grids:
            composite_columns = [
                "latitude",
                "longitude",
                "composite_index",
                "contrib_subject_primary",
                "contrib_subject_secondary",
                "contrib_gap",
                "contrib_context",
            ]
            if "context_value" not in merged.columns:
                composite_columns.insert(2, "context_value")
            merged = merged.merge(
                self.grids["composite"][composite_columns],
                on=["latitude", "longitude"],
                how="left",
            )
        return merged


SESSION_KEY = "seisnetinsight_session"
SESSION_FEEDBACK_KEY = "seisnetinsight_session_feedback"
FORM_SESSION_NAME_KEY = "form_session_name"
FORM_LONS_KEY = "form_lons"
FORM_LATS_KEY = "form_lats"
FORM_GRID_STEP_KEY = "form_grid_step"
FORM_PRIMARY_RADIUS_KEY = "form_primary_radius"
FORM_PRIMARY_MIN_KEY = "form_primary_min"
FORM_PRIMARY_WEIGHT_KEY = "form_primary_weight"
FORM_SECONDARY_RADIUS_KEY = "form_secondary_radius"
FORM_SECONDARY_MIN_KEY = "form_secondary_min"
FORM_SECONDARY_WEIGHT_KEY = "form_secondary_weight"
FORM_GAP_SEARCH_KEY = "form_gap_search"
FORM_GAP_TARGET_KEY = "form_gap_target"
FORM_WEIGHT_GAP_KEY = "form_weight_gap"
FORM_CONTEXT_LABEL_KEY = "form_context_label"
FORM_CONTEXT_RADIUS_KEY = "form_context_radius"
FORM_CONTEXT_AGGREGATION_KEY = "form_context_aggregation"
FORM_WEIGHT_CONTEXT_KEY = "form_weight_context"
FORM_HALF_TIME_KEY = "form_half_time"
FORM_BALLTREE_ENABLED_KEY = "form_balltree_enabled"
FORM_BALLTREE_DISTANCE_KEY = "form_balltree_distance"

EVENT_COLUMN_HELP = {
    "latitude": "Column containing event latitude in decimal degrees.",
    "longitude": "Column containing event longitude in decimal degrees.",
    "origin_time": "Column containing the event origin timestamp used for recency weighting.",
}

STATION_COLUMN_HELP = {
    "latitude": "Column containing station latitude in decimal degrees.",
    "longitude": "Column containing station longitude in decimal degrees.",
}

CONTEXT_COLUMN_HELP = {
    "latitude": "Column containing context-point latitude in decimal degrees.",
    "longitude": "Column containing context-point longitude in decimal degrees.",
    "value": "Numeric column aggregated within the selected context radius.",
}


def _session_form_defaults(session: WorkingSession) -> Dict[str, object]:
    return {
        FORM_SESSION_NAME_KEY: session.name,
        FORM_LONS_KEY: ",".join(map(str, session.parameters.lons)),
        FORM_LATS_KEY: ",".join(map(str, session.parameters.lats)),
        FORM_GRID_STEP_KEY: float(session.parameters.grid_step),
        FORM_PRIMARY_RADIUS_KEY: float(session.parameters.subject_primary_radius_km),
        FORM_PRIMARY_MIN_KEY: int(session.parameters.subject_primary_min_stations),
        FORM_PRIMARY_WEIGHT_KEY: float(session.parameters.subject_primary_weight),
        FORM_SECONDARY_RADIUS_KEY: float(session.parameters.subject_secondary_radius_km),
        FORM_SECONDARY_MIN_KEY: int(session.parameters.subject_secondary_min_stations),
        FORM_SECONDARY_WEIGHT_KEY: float(session.parameters.subject_secondary_weight),
        FORM_GAP_SEARCH_KEY: float(session.parameters.gap_search_km),
        FORM_GAP_TARGET_KEY: float(getattr(session.parameters, "gap_target_angle_deg", 90.0)),
        FORM_WEIGHT_GAP_KEY: float(session.parameters.weight_gap),
        FORM_CONTEXT_LABEL_KEY: session.context_label,
        FORM_CONTEXT_RADIUS_KEY: float(session.parameters.context_radius_km),
        FORM_CONTEXT_AGGREGATION_KEY: session.parameters.context_aggregation,
        FORM_WEIGHT_CONTEXT_KEY: float(session.parameters.weight_context),
        FORM_HALF_TIME_KEY: float(session.parameters.half_time_years),
        FORM_BALLTREE_ENABLED_KEY: bool(session.balltree_enabled),
        FORM_BALLTREE_DISTANCE_KEY: float(session.balltree_distance),
    }


def _populate_form_state_from_session(session: WorkingSession, *, overwrite: bool = False) -> None:
    for key, value in _session_form_defaults(session).items():
        if overwrite or key not in st.session_state:
            st.session_state[key] = value


def _pluralize(count: int, singular: str, plural: Optional[str] = None) -> str:
    if count == 1:
        return singular
    return plural or f"{singular}s"


def _session_feedback_details(session: WorkingSession, action: str) -> List[str]:
    details = [
        (
            f"{action} form values and grid settings for "
            f"{session.parameters.lons[0]:g},{session.parameters.lons[1]:g} / "
            f"{session.parameters.lats[0]:g},{session.parameters.lats[1]:g}."
        )
    ]
    if session.events is not None:
        details.append(f"{action} events catalog: {len(session.events)} {_pluralize(len(session.events), 'row')}.")
    if session.stations is not None:
        details.append(
            f"{action} stations catalog: {len(session.stations)} {_pluralize(len(session.stations), 'row')}."
        )
    if session.context is not None:
        details.append(
            f"{action} {session.context_label.lower()}: {len(session.context)} {_pluralize(len(session.context), 'row')}."
        )
    else:
        details.append(f"{action} no context layer.")
    bna_count = len(session.bna_files)
    if session.storage:
        bna_count = max(bna_count, len(session.storage.list_bna()))
    details.append(f"{action} {bna_count} {_pluralize(bna_count, 'BNA overlay')}.")
    grid_names = sorted(session.grids)
    if grid_names:
        details.append(f"{action} cached grids: {', '.join(grid_names)}.")
    else:
        details.append(f"{action} no cached grids yet.")
    return details


def _queue_session_feedback(kind: str, title: str, details: List[str]) -> None:
    st.session_state[SESSION_FEEDBACK_KEY] = {
        "kind": kind,
        "title": title,
        "details": details,
    }


def _render_session_feedback(*, consume: bool = True) -> None:
    payload = st.session_state.get(SESSION_FEEDBACK_KEY)
    if not payload:
        return
    kind = str(payload.get("kind", "info"))
    title = str(payload.get("title", ""))
    details = [str(item) for item in payload.get("details", [])]
    renderer = {
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
    }.get(kind, st.info)
    renderer(title)
    if details:
        with st.expander("Details", expanded=True):
            for item in details:
                st.markdown(f"- {item}")
    if consume:
        st.session_state.pop(SESSION_FEEDBACK_KEY, None)


def _required_cartopy_specs(params: GridParameters) -> List[Dict[str, str]]:
    extent = [params.lons[0], params.lons[1], params.lats[0], params.lats[1]]
    specs: List[Dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    feature_specs = [
        cfeature.LAND,
        cfeature.OCEAN,
        cfeature.LAKES,
        cfeature.RIVERS,
        cfeature.BORDERS,
        cfeature.STATES,
    ]
    for feature in feature_specs:
        resolution = feature.scaler.scale_from_extent(extent) if getattr(feature, "scaler", None) else feature.scale
        spec = (str(resolution), str(feature.category), str(feature.name))
        if spec in seen:
            continue
        seen.add(spec)
        specs.append(
            {
                "resolution": spec[0],
                "category": spec[1],
                "name": spec[2],
                "label": f"{spec[2]} ({spec[0]})",
            }
        )

    roads_spec = ("10m", "cultural", "roads")
    if roads_spec not in seen:
        specs.append(
            {
                "resolution": roads_spec[0],
                "category": roads_spec[1],
                "name": roads_spec[2],
                "label": "roads (10m)",
            }
        )
    return specs


def _cartopy_target_exists(spec: Dict[str, str]) -> bool:
    downloader = Downloader.from_config(
        ("shapefiles", "natural_earth", spec["resolution"], spec["category"], spec["name"])
    )
    format_dict = {
        "config": cartopy_config,
        "resolution": spec["resolution"],
        "category": spec["category"],
        "name": spec["name"],
    }
    pre_downloaded_path = downloader.pre_downloaded_path(format_dict)
    if pre_downloaded_path is not None and pre_downloaded_path.exists():
        return True
    return downloader.target_path(format_dict).exists()


def _missing_cartopy_specs(params: GridParameters) -> List[Dict[str, str]]:
    return [spec for spec in _required_cartopy_specs(params) if not _cartopy_target_exists(spec)]


def _download_cartopy_specs(specs: List[Dict[str, str]]) -> None:
    progress = st.progress(0.0, text="Preparing Cartopy map data…")
    try:
        total = max(len(specs), 1)
        for idx, spec in enumerate(specs, start=1):
            progress.progress(
                (idx - 1) / total,
                text=f"Downloading {spec['label']} ({idx}/{total})…",
            )
            downloader = Downloader.from_config(
                ("shapefiles", "natural_earth", spec["resolution"], spec["category"], spec["name"])
            )
            format_dict = {
                "config": cartopy_config,
                "resolution": spec["resolution"],
                "category": spec["category"],
                "name": spec["name"],
            }
            downloader.path(format_dict)
        progress.progress(1.0, text="Cartopy map data ready.")
    finally:
        progress.empty()


def _ensure_cartopy_map_data(params: GridParameters) -> bool:
    missing_specs = _missing_cartopy_specs(params)
    if not missing_specs:
        return True

    st.info(
        "Map previews use Cartopy's Natural Earth basemap files for land, borders, water bodies, "
        "state boundaries, and roads. On first use, these files must be downloaded and cached locally. "
        "This can take longer than the grid computations, but it only needs to happen once."
    )
    missing_labels = ", ".join(spec["label"] for spec in missing_specs)
    st.caption(f"Missing map datasets: {missing_labels}")

    if st.button(
        "Download map data and continue",
        key="download_cartopy_map_data",
        help="Download the missing Natural Earth basemap files required for static map rendering, then continue to the map previews.",
    ):
        try:
            with st.spinner("Downloading Cartopy map data…"):
                _download_cartopy_specs(missing_specs)
        except Exception as exc:
            st.error(f"Failed to download Cartopy map data: {exc}")
            return False
        st.success("Cartopy map data downloaded. Rendering maps…")
        st.rerun()

    st.warning("Maps will appear after the required Cartopy files are downloaded.")
    return False
STOP_PREFIX = "stop_"
RUN_PREFIX = "run_"

LEGACY_SUBJECT_COLUMN_MAP = {
    "subject4_within_4km": "subject_primary_count",
    "subject10_within_10km": "subject_secondary_count",
    "subject4_within_4km_weighted": "subject_primary_weighted",
    "subject10_within_10km_weighted": "subject_secondary_weighted",
}

LEGACY_COMPOSITE_COLUMN_MAP = {
    "contrib_subject4": "contrib_subject_primary",
    "contrib_subject10": "contrib_subject_secondary",
}

LEGACY_FEATURE_ALIAS = {
    "subject4_within_4km_weighted": "subject_primary_weighted",
    "subject10_within_10km_weighted": "subject_secondary_weighted",
    "delta_gap90_weighted": "delta_gap90_weighted",
    "context_value": "context_value",
    "composite_index": "composite_index",
}

OVERLAY_ALPHA = 0.7


LEGACY_FEATURE_ORDER = [
    ("subject4_within_4km_weighted", "Primary source-station distance (recency-weighted)"),
    ("subject10_within_10km_weighted", "Secondary source-station distance (recency-weighted)"),
    ("delta_gap90_weighted", "ΔGap (recency-weighted)"),
    ("context_value", "Context layer"),
    ("composite_index", "Composite index"),
]

def _pivot_grid(df: pd.DataFrame, value_col: str) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if value_col not in df.columns:
        return None
    pivot = (
        df.pivot_table(index="latitude", columns="longitude", values=value_col, aggfunc="mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if pivot.empty:
        return None
    return pivot.index.to_numpy(), pivot.columns.to_numpy(), pivot.to_numpy()


def _rgba_overlay_png(rgba: np.ndarray) -> Optional[bytes]:
    if rgba.size == 0:
        return None
    rgba_uint8 = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
    # Flip vertically so north is up when rendered.
    rgba_uint8 = rgba_uint8[::-1]
    buffer = io.BytesIO()
    try:
        Image.fromarray(rgba_uint8, mode="RGBA").save(buffer, format="PNG")
    except Exception:
        return None
    buffer.seek(0)
    return buffer.getvalue()


def _ground_overlay_kmz(png_bytes: bytes, name: str, *, north: float, south: float, east: float, west: float) -> Optional[bytes]:
    if not png_bytes:
        return None
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <GroundOverlay>
      <name>{name}</name>
      <Icon>
        <href>overlay.png</href>
      </Icon>
      <LatLonBox>
        <north>{north:.6f}</north>
        <south>{south:.6f}</south>
        <east>{east:.6f}</east>
        <west>{west:.6f}</west>
      </LatLonBox>
    </GroundOverlay>
  </Document>
</kml>
"""
    buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("doc.kml", kml)
            archive.writestr("overlay.png", png_bytes)
    except Exception:
        return None
    buffer.seek(0)
    return buffer.getvalue()


def _legacy_overlay_kmz(df: pd.DataFrame, feature: str, params: GridParameters) -> Optional[bytes]:
    grid = _pivot_grid(df, feature)
    if grid is None:
        return None
    lats, lons, values = grid
    valid = np.isfinite(values)
    if not valid.any():
        return None
    if feature == "composite_index":
        levels = np.linspace(0.0, 1.0, 11)
    else:
        zmin = float(np.nanmin(values))
        zmax = float(np.nanmax(values))
        if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
            levels = np.linspace(0.0, 1.0, 12)
        else:
            levels = np.linspace(zmin, zmax, 12)
    cmap = cmc.batlow
    norm = mpl.colors.Normalize(vmin=float(levels[0]), vmax=float(levels[-1]))
    rgba = cmap(norm(values))
    rgba[..., 3] = np.where(valid, OVERLAY_ALPHA, 0.0)
    png_bytes = _rgba_overlay_png(rgba)
    if png_bytes is None:
        return None
    north = float(params.lats[1])
    south = float(params.lats[0])
    east = float(params.lons[1])
    west = float(params.lons[0])
    return _ground_overlay_kmz(png_bytes, feature, north=north, south=south, east=east, west=west)


def _priority_overlay_kmz(df: pd.DataFrame, params: GridParameters) -> Optional[bytes]:
    if "priority_label" not in df.columns:
        return None
    pivot = (
        df.pivot_table(index="latitude", columns="longitude", values="priority_label", aggfunc="first")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if pivot.empty:
        return None
    labels = pivot.to_numpy(dtype=object)
    rgba = np.zeros(labels.shape + (4,), dtype=float)
    for label in np.unique(labels.astype(object)):
        if label is None or (isinstance(label, float) and np.isnan(label)):
            continue
        rgb = PRIORITY_COLORS.get(label, PRIORITY_COLORS[PRIORITY_LEVELS[-1]])
        mask = labels == label
        rgba[mask, 0] = rgb[0] / 255.0
        rgba[mask, 1] = rgb[1] / 255.0
        rgba[mask, 2] = rgb[2] / 255.0
        rgba[mask, 3] = OVERLAY_ALPHA
    png_bytes = _rgba_overlay_png(rgba)
    if png_bytes is None:
        return None
    north = float(params.lats[1])
    south = float(params.lats[0])
    east = float(params.lons[1])
    west = float(params.lons[0])
    return _ground_overlay_kmz(png_bytes, "priority_overlay", north=north, south=south, east=east, west=west)



def _default_session_name() -> str:
    return dt.datetime.now(dt.UTC).strftime("session-%Y%m%d-%H%M%S")


def get_working_session() -> WorkingSession:
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = WorkingSession(name=_default_session_name())
    return st.session_state[SESSION_KEY]


def set_working_session(session: WorkingSession) -> None:
    st.session_state[SESSION_KEY] = session
    _populate_form_state_from_session(session, overwrite=True)


def _stop_key(name: str) -> str:
    return f"{STOP_PREFIX}{name}"


def _run_key(name: str) -> str:
    return f"{RUN_PREFIX}{name}"


def _stop_requested(name: str) -> bool:
    return bool(st.session_state.get(_stop_key(name), False))


def _reset_stop(name: str) -> None:
    st.session_state[_stop_key(name)] = False


def _bna_bytes(session: WorkingSession) -> Optional[bytes]:
    if session.bna_files:
        name = sorted(session.bna_files)[0]
        return session.bna_files[name]
    if session.storage:
        for path in session.storage.list_bna():
            try:
                return path.read_bytes()
            except OSError:
                continue
    return None


def _show_column_warnings(warnings: Dict[str, List[str]]) -> None:
    for dataset, missing in warnings.items():
        if missing:
            st.warning(
                f"{dataset} is missing expected columns: {', '.join(missing)}."
            )


def _load_session_from_disk(name: str) -> WorkingSession:
    state = SessionState.load(name)
    events_cols = _normalize_column_mapping(
        state.column_mapping.get("events", {}),
        _default_event_columns(),
    )
    stations_cols = _normalize_column_mapping(
        state.column_mapping.get("stations", {}),
        _default_station_columns(),
    )
    context_cols = _normalize_column_mapping(
        state.column_mapping.get("context", state.column_mapping.get("swd", {})),
        _default_context_columns(),
    )
    ws = WorkingSession(
        name=name,
        parameters=state.parameters,
        storage=state,
        events_columns=events_cols,
        stations_columns=stations_cols,
        context_columns=context_cols,
        context_label=state.context_label,
    )
    for path in state.list_bna():
        try:
            ws.bna_files[path.name] = path.read_bytes()
        except OSError:
            continue
    ws.events = state.load_source("events")
    ws.stations = state.load_source("stations")
    ws.context = state.load_source("context")
    ws.grid = generate_grid(ws.parameters)
    warnings: Dict[str, List[str]] = {}
    warnings["Events"] = [] if ws.events is not None else EXPECTED_EVENT_COLUMNS
    warnings["Stations"] = [] if ws.stations is not None else EXPECTED_STATION_COLUMNS
    warnings[ws.context_label] = [] if ws.context is not None else []
    ws.column_warnings = warnings
    for key in ["subject", "gap", "context", "swd", "composite"]:
        if key in state.grids:
            frame = state.load_dataframe(key)
            if key == "subject":
                frame = frame.rename(columns=LEGACY_SUBJECT_COLUMN_MAP)
            if key == "composite":
                frame = frame.rename(columns=LEGACY_COMPOSITE_COLUMN_MAP)
                if "contrib_swd" in frame.columns and "contrib_context" not in frame.columns:
                    frame["contrib_context"] = frame["contrib_swd"]
                if "swd_volume_25km_bbl" in frame.columns and "context_value" not in frame.columns:
                    frame["context_value"] = frame["swd_volume_25km_bbl"]
            if key in {"context", "swd"}:
                if "swd_volume_25km_bbl" in frame.columns:
                    frame = frame.rename(columns={"swd_volume_25km_bbl": "context_value"})
                ws.grids["context"] = frame
                continue
            ws.grids[key] = frame
    return ws


def _save_sources(
    session: WorkingSession,
    events: pd.DataFrame,
    stations: pd.DataFrame,
    context: Optional[pd.DataFrame],
) -> None:
    if session.storage is None:
        session.storage = SessionState(session.name, session.parameters, SessionFiles(), column_mapping={})
    session.storage.column_mapping["events"] = dict(session.events_columns)
    session.storage.column_mapping["stations"] = dict(session.stations_columns)
    session.storage.column_mapping["context"] = dict(session.context_columns)
    session.storage.context_label = session.context_label
    session.storage.save_source(events, "events")
    session.storage.save_source(stations, "stations")
    if context is not None:
        session.storage.save_source(context, "context")
    else:
        session.storage.clear_source("context")
        session.storage.clear_source("swd")
        session.storage.clear_dataframe("context")
        session.storage.clear_dataframe("swd")
    session.storage.save_metadata()


def _save_bna_files(session: WorkingSession) -> None:
    if session.storage is None:
        return
    for name, data in session.bna_files.items():
        session.storage.save_bna(name, data)


def build_working_session(
    *,
    session_name: str,
    parameters: GridParameters,
    events_bytes: bytes,
    stations_bytes: bytes,
    context_bytes: Optional[bytes] = None,
    bna_data: Optional[Dict[str, bytes]] = None,
    events_columns: Optional[Dict[str, str]] = None,
    stations_columns: Optional[Dict[str, str]] = None,
    context_columns: Optional[Dict[str, str]] = None,
    context_label: str = "Context layer",
    balltree_enabled: bool = False,
    balltree_distance: float = 1.0,
) -> WorkingSession:
    if events_bytes is None or stations_bytes is None:
        raise ValueError("Events and stations files are required.")

    resolved_events_columns = events_columns or _default_event_columns()
    resolved_stations_columns = stations_columns or _default_station_columns()
    resolved_context_columns = context_columns or _default_context_columns()

    events_df, events_missing = load_events(io.BytesIO(events_bytes), column_map=resolved_events_columns)
    stations_df, stations_missing = load_stations(io.BytesIO(stations_bytes), column_map=resolved_stations_columns)
    context_df = None
    context_missing: List[str] = []
    if context_bytes is not None:
        context_df, context_missing = load_context_points(
            io.BytesIO(context_bytes),
            column_map=resolved_context_columns,
        )

    warnings = {
        "Events": events_missing,
        "Stations": stations_missing,
        context_label: context_missing,
    }
    if balltree_enabled:
        events_df = balltree_reduce_events(events_df, distance_threshold_km=balltree_distance)

    session = WorkingSession(
        name=session_name,
        parameters=parameters,
        events=events_df,
        stations=stations_df,
        context=context_df,
        balltree_enabled=balltree_enabled,
        balltree_distance=balltree_distance,
        column_warnings=warnings,
        events_columns=dict(resolved_events_columns),
        stations_columns=dict(resolved_stations_columns),
        context_columns=dict(resolved_context_columns),
        context_label=context_label,
    )
    session.grid = generate_grid(parameters)
    if bna_data:
        session.bna_files.update(bna_data)
    return session


def _run_subject_grid(session: WorkingSession) -> None:
    status = st.empty()
    progress = st.progress(0.0, text="Starting source-station distance grids…")

    def update(fraction: float, message: str) -> None:
        progress.progress(min(fraction, 1.0), text=message)

    try:
        grid = session.ensure_grid()
        result = compute_subject_grids(
            session.events,
            session.stations,
            grid,
            session.parameters,
            progress=update,
            should_stop=lambda: _stop_requested("subject"),
        )
        session.grids["subject"] = result
        if session.storage:
            session.storage.save_dataframe(result, "subject")
        status.success("Primary and secondary source-station distance grids computed.")
    except InterruptedError:
        status.warning("Source-station distance grid computation stopped by user.")
    except Exception as exc:
        status.error(f"Failed to compute source-station distance grids: {exc}")
    finally:
        progress.empty()
        _reset_stop("subject")
        st.session_state[_run_key("subject")] = False


def _run_gap_grid(session: WorkingSession) -> None:
    status = st.empty()
    progress = st.progress(0.0, text="Starting ΔGap grid…")

    def update(fraction: float, message: str) -> None:
        progress.progress(min(fraction, 1.0), text=message)

    try:
        grid = session.ensure_grid()
        result = compute_gap_grid(
            session.events,
            session.stations,
            grid,
            session.parameters,
            progress=update,
            should_stop=lambda: _stop_requested("gap"),
        )
        session.grids["gap"] = result
        if session.storage:
            session.storage.save_dataframe(result, "gap")
        status.success("ΔGap grid computed.")
    except InterruptedError:
        status.warning("ΔGap computation stopped by user.")
    except Exception as exc:
        status.error(f"Failed to compute ΔGap grid: {exc}")
    finally:
        progress.empty()
        _reset_stop("gap")
        st.session_state[_run_key("gap")] = False


def _run_context_grid(session: WorkingSession) -> None:
    if not session.has_context_input:
        session.grids.pop("context", None)
        if session.storage is not None and "context" in session.storage.grids:
            session.storage.grids.pop("context", None)
            session.storage.save_metadata()
        st.session_state[_run_key("context")] = False
        return

    status = st.empty()
    progress = st.progress(0.0, text="Starting context grid…")

    def update(fraction: float, message: str) -> None:
        progress.progress(min(fraction, 1.0), text=message)

    try:
        grid = session.ensure_grid()
        result = compute_context_grid(
            session.context,
            grid,
            session.parameters,
            aggregation=session.parameters.context_aggregation,
            radius_km=session.parameters.context_radius_km,
            progress=update,
            should_stop=lambda: _stop_requested("context"),
        )
        session.grids["context"] = result
        if session.storage:
            session.storage.save_dataframe(result, "context")
        status.success(f"{session.context_label} grid computed.")
    except InterruptedError:
        status.warning(f"{session.context_label} computation stopped by user.")
    except Exception as exc:
        status.error(f"Failed to compute {session.context_label} grid: {exc}")
    finally:
        progress.empty()
        _reset_stop("context")
        st.session_state[_run_key("context")] = False


def _run_composite_grid(session: WorkingSession) -> None:
    merged = session.merged()
    if merged is None:
        st.warning("Compute the source-station distance and ΔGap grids before the composite index.")
        return
    try:
        composite = compute_composite_index(merged, session.parameters)
        session.grids["composite"] = composite
        if session.storage:
            session.storage.save_dataframe(composite, "composite")
        st.success("Composite index grid computed.")
    except Exception as exc:
        st.error(f"Failed to compute composite index: {exc}")


def _render_data_loading(session: WorkingSession) -> None:
    st.header("1. Data Loading")
    _populate_form_state_from_session(session)
    _render_session_feedback()
    if session.storage is None:
        st.info(
            "Current state: draft session in memory only. "
            "A saved session is created when you click `Save session and load data` "
            "or `Save session, load data, and run all`."
        )
    else:
        st.info(
            f"Current state: saved session `{session.name}`. "
            "Editing the fields below updates the draft in memory; the saved session is updated "
            "the next time you click a save/load button."
        )
    existing_sessions = list_sessions()
    if existing_sessions:
        st.markdown("**Restore saved session**")
        restore_choice = st.checkbox(
            "Restore a saved session",
            key="restore_toggle",
            help="Enable this to load saved parameters, uploaded datasets, computed grids, and map-ready outputs from a previous session.",
        )
        selected_name = st.selectbox(
            "Available sessions",
            existing_sessions,
            disabled=not restore_choice,
            help="Choose which saved session to restore.",
        )
        if restore_choice and st.button(
            "Load session",
            help="Restore the selected session's saved settings, source data, computed grids, and BNA overlays.",
        ):
            try:
                new_session = _load_session_from_disk(selected_name)
                set_working_session(new_session)
                _queue_session_feedback(
                    "success",
                    f"Session '{selected_name}' restored.",
                    _session_feedback_details(new_session, "Restored"),
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load session: {exc}")
    st.markdown("---")
    st.subheader("Draft session setup")
    session_name = st.text_input(
        "Session name",
        key=FORM_SESSION_NAME_KEY,
        help="Name to use when this draft session is saved to disk.",
    )

    gap_target_angle_default = float(getattr(session.parameters, "gap_target_angle_deg", 90.0))
    if not hasattr(session.parameters, "gap_target_angle_deg"):
        session.parameters.gap_target_angle_deg = gap_target_angle_default

    bounds_cols = st.columns(3)
    with bounds_cols[0]:
        lons_input = st.text_input(
            "Longitude bounds (min,max)",
            key=FORM_LONS_KEY,
            help="West and east limits of the area of interest in decimal degrees, entered as min,max.",
        )
    with bounds_cols[1]:
        lats_input = st.text_input(
            "Latitude bounds (min,max)",
            key=FORM_LATS_KEY,
            help="South and north limits of the area of interest in decimal degrees, entered as min,max.",
        )
    with bounds_cols[2]:
        grid_step = st.number_input(
            "Grid step (degrees)",
            min_value=0.001,
            step=0.001,
            key=FORM_GRID_STEP_KEY,
            help="Grid spacing in decimal degrees. Smaller values produce finer maps but increase computation time.",
        )

    st.markdown("**Source-station distance weighting**")
    subject_primary_cols = st.columns(3)
    with subject_primary_cols[0]:
        primary_radius = st.number_input(
            "Primary source-station distance radius (km)",
            key=FORM_PRIMARY_RADIUS_KEY,
            help="Radius used for the primary source-station distance metric around each grid cell.",
        )
    with subject_primary_cols[1]:
        primary_min = st.number_input(
            "Minimum stations within primary distance",
            min_value=0,
            key=FORM_PRIMARY_MIN_KEY,
            help="Minimum number of stations expected inside the primary radius.",
        )
    with subject_primary_cols[2]:
        primary_weight = st.number_input(
            "Weight primary source-station distance",
            key=FORM_PRIMARY_WEIGHT_KEY,
            help="Contribution of the primary source-station distance metric to the composite index.",
        )

    subject_secondary_cols = st.columns(3)
    with subject_secondary_cols[0]:
        secondary_radius = st.number_input(
            "Secondary source-station distance radius (km)",
            key=FORM_SECONDARY_RADIUS_KEY,
            help="Radius used for the secondary source-station distance metric around each grid cell.",
        )
    with subject_secondary_cols[1]:
        secondary_min = st.number_input(
            "Minimum stations within secondary distance",
            min_value=0,
            key=FORM_SECONDARY_MIN_KEY,
            help="Minimum number of stations expected inside the secondary radius.",
        )
    with subject_secondary_cols[2]:
        secondary_weight = st.number_input(
            "Weight secondary source-station distance",
            key=FORM_SECONDARY_WEIGHT_KEY,
            help="Contribution of the secondary source-station distance metric to the composite index.",
        )

    gap_cols = st.columns(3)
    with gap_cols[0]:
        gap_search = st.number_input(
            "Gap search radius (km)",
            key=FORM_GAP_SEARCH_KEY,
            help="Stations within this radius are used to estimate azimuthal coverage.",
        )
    with gap_cols[1]:
        gap_target_angle = st.number_input(
            "ΔGap target angle (°)",
            min_value=0.0,
            max_value=360.0,
            step=1.0,
            key=FORM_GAP_TARGET_KEY,
            help="Reference gap angle used to score azimuthal coverage. Lower resulting gaps indicate better coverage.",
        )
    with gap_cols[2]:
        weight_gap = st.number_input(
            "Weight ΔGap",
            key=FORM_WEIGHT_GAP_KEY,
            help="Contribution of the ΔGap metric to the composite index.",
        )

    context_cols = st.columns(4)
    with context_cols[0]:
        context_label = st.text_input(
            "Context layer name",
            key=FORM_CONTEXT_LABEL_KEY,
            help="Display name used for the optional contextual layer in previews and maps.",
        )
    with context_cols[1]:
        context_radius = st.number_input(
            "Context radius (km)",
            key=FORM_CONTEXT_RADIUS_KEY,
            help="Search radius used to aggregate context-point values around each grid cell.",
        )
    with context_cols[2]:
        aggregation_options = ["sum", "average", "count", "min", "max"]
        aggregation_index = aggregation_options.index(
            st.session_state.get(FORM_CONTEXT_AGGREGATION_KEY)
            if st.session_state.get(FORM_CONTEXT_AGGREGATION_KEY) in aggregation_options
            else "sum"
        )
        context_aggregation = st.selectbox(
            "Context aggregation",
            aggregation_options,
            index=aggregation_index,
            key=FORM_CONTEXT_AGGREGATION_KEY,
            help="How context-point values are combined within the context radius.",
        )
    with context_cols[3]:
        weight_context = st.number_input(
            "Weight context",
            key=FORM_WEIGHT_CONTEXT_KEY,
            help="Contribution of the optional context grid to the composite index.",
        )

    final_cols = st.columns(2)
    with final_cols[0]:
        half_time = st.number_input(
            "Half-time (years)",
            key=FORM_HALF_TIME_KEY,
            help="Recency-weighting half-time for event-based metrics. Older events contribute less over time.",
        )

    balltree_cols = st.columns(2)
    with balltree_cols[0]:
        balltree_enabled = st.checkbox(
            "Apply BallTree data reduction",
            key=FORM_BALLTREE_ENABLED_KEY,
            help="Reduce dense event catalogs before gridding by collapsing nearby events within the selected distance threshold.",
        )
    with balltree_cols[1]:
        balltree_distance = st.number_input(
            "BallTree distance threshold (km)",
            min_value=0.1,
            step=0.1,
            key=FORM_BALLTREE_DISTANCE_KEY,
            help="Maximum separation between events considered neighbors during BallTree reduction.",
        )

    events_file = st.file_uploader(
        "Events file",
        type=["csv"],
        help="Upload the earthquake catalog used to compute the event-based grids.",
    )
    stations_file = st.file_uploader(
        "Stations file",
        type=["csv"],
        help="Upload the station catalog used to measure source-station distance and azimuthal coverage.",
    )
    context_file = st.file_uploader(
        "Context file (optional)",
        type=["csv"],
        key="context",
        help="Optional point dataset with latitude, longitude, and a numeric value for the contextual layer.",
    )
    bna_files = st.file_uploader(
        "BNA files (optional)",
        type=["bna"],
        accept_multiple_files=True,
        help="Optional polygon overlays drawn on the static maps.",
    )

    events_bytes = events_file.getvalue() if events_file is not None else None
    stations_bytes = stations_file.getvalue() if stations_file is not None else None
    context_bytes = context_file.getvalue() if context_file is not None else None

    if events_bytes is not None:
        event_columns = _extract_columns(io.BytesIO(events_bytes))
        if not event_columns:
            st.warning("Could not read column names from events file.")
        else:
            st.caption("Events column mapping")
            session.events_columns = _render_column_selectors(
                [
                    ("latitude", "Latitude"),
                    ("longitude", "Longitude"),
                    ("origin_time", "Origin time"),
                ],
                event_columns,
                session.events_columns,
                _default_event_columns(),
                "events_column",
                field_help=EVENT_COLUMN_HELP,
            )

    if stations_bytes is not None:
        station_columns = _extract_columns(io.BytesIO(stations_bytes))
        if not station_columns:
            st.warning("Could not read column names from stations file.")
        else:
            st.caption("Stations column mapping")
            session.stations_columns = _render_column_selectors(
                [
                    ("latitude", "Latitude"),
                    ("longitude", "Longitude"),
                ],
                station_columns,
                session.stations_columns,
                _default_station_columns(),
                "stations_column",
                field_help=STATION_COLUMN_HELP,
            )

    if context_bytes is not None:
        context_columns = _extract_columns(io.BytesIO(context_bytes))
        if not context_columns:
            st.warning("Could not read column names from context file.")
        else:
            st.caption("Context column mapping")
            session.context_columns = _render_column_selectors(
                [
                    ("latitude", "Latitude"),
                    ("longitude", "Longitude"),
                    ("value", "Value"),
                ],
                context_columns,
                session.context_columns,
                _default_context_columns(),
                "context_column",
                field_help=CONTEXT_COLUMN_HELP,
            )

    inputs = {
        "LONS": lons_input,
        "LATS": lats_input,
        "GRID_STEP": grid_step,
        "SUBJECT_PRIMARY_RADIUS_KM": primary_radius,
        "SUBJECT_PRIMARY_MIN_STATIONS": primary_min,
        "SUBJECT_PRIMARY_WEIGHT": primary_weight,
        "SUBJECT_SECONDARY_RADIUS_KM": secondary_radius,
        "SUBJECT_SECONDARY_MIN_STATIONS": secondary_min,
        "SUBJECT_SECONDARY_WEIGHT": secondary_weight,
        "GAP_SEARCH_KM": gap_search,
        "GAP_TARGET_ANGLE": gap_target_angle,
        "WEIGHT_GAP": weight_gap,
        "CONTEXT_RADIUS_KM": context_radius,
        "CONTEXT_AGGREGATION": context_aggregation,
        "WEIGHT_CONTEXT": weight_context,
        "HALF_TIME_YEARS": half_time,
    }
    parameters = parameter_from_inputs(inputs, logger=st.warning)
    session.context_label = context_label.strip() or "Context layer"

    previous_params = session.parameters

    def _tuple_changed(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        if len(a) != len(b):
            return True
        return any(not math.isclose(float(x), float(y), rel_tol=1e-9, abs_tol=1e-9) for x, y in zip(a, b))

    def _scalar_changed(a: float, b: float) -> bool:
        return not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)

    geometry_changed = (
        _tuple_changed(getattr(previous_params, "lons", ()), getattr(parameters, "lons", ()))
        or _tuple_changed(getattr(previous_params, "lats", ()), getattr(parameters, "lats", ()))
        or _scalar_changed(getattr(previous_params, "grid_step", 0.0), getattr(parameters, "grid_step", 0.0))
    )

    float_fields = [
        "subject_primary_radius_km",
        "subject_primary_weight",
        "subject_secondary_radius_km",
        "subject_secondary_weight",
        "gap_search_km",
        "gap_target_angle_deg",
        "weight_gap",
        "context_radius_km",
        "weight_context",
        "half_time_years",
    ]
    int_fields = [
        "subject_primary_min_stations",
        "subject_secondary_min_stations",
    ]

    params_changed = geometry_changed
    if not params_changed:
        for field in float_fields:
            old_value = getattr(previous_params, field, None)
            new_value = getattr(parameters, field, None)
            if old_value is None or new_value is None:
                if old_value != new_value:
                    params_changed = True
                    break
            elif _scalar_changed(old_value, new_value):
                params_changed = True
                break
    if not params_changed:
        for field in int_fields:
            if getattr(previous_params, field, None) != getattr(parameters, field, None):
                params_changed = True
                break
    if not params_changed:
        if getattr(previous_params, "context_aggregation", None) != getattr(parameters, "context_aggregation", None):
            params_changed = True
    session.parameters = parameters
    if session.storage:
        session.storage.parameters = session.parameters
    if geometry_changed:
        session.grid = None
        session.grids.clear()
    elif params_changed:
        session.grids.pop("composite", None)

    def process_inputs(run_all: bool) -> None:
        try:
            bna_data = None
            if bna_files:
                bna_data = {file.name: file.getvalue() for file in bna_files}
            new_session = build_working_session(
                session_name=session_name,
                parameters=parameters,
                events_bytes=events_bytes,
                stations_bytes=stations_bytes,
                context_bytes=context_bytes,
                bna_data=bna_data,
                events_columns=session.events_columns,
                stations_columns=session.stations_columns,
                context_columns=session.context_columns,
                context_label=session.context_label,
                balltree_enabled=balltree_enabled,
                balltree_distance=balltree_distance,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        set_working_session(new_session)
        _show_column_warnings(new_session.column_warnings)
        st.markdown("**Events preview**")
        st.dataframe(new_session.events.head(10), width="stretch")
        st.markdown("**Stations preview**")
        st.dataframe(new_session.stations.head(10), width="stretch")
        if new_session.context is not None:
            st.markdown(f"**{new_session.context_label} preview**")
            st.dataframe(new_session.context.head(10), width="stretch")
        _save_sources(new_session, new_session.events, new_session.stations, new_session.context)
        _save_bna_files(new_session)
        _queue_session_feedback(
            "success",
            f"Session '{new_session.name}' saved.",
            _session_feedback_details(new_session, "Saved"),
        )
        _render_session_feedback()
        if run_all:
            st.session_state[_run_key("subject")] = True
            st.session_state[_run_key("gap")] = True
            st.session_state[_run_key("context")] = bool(new_session.has_context_input)
            _run_subject_grid(new_session)
            _run_gap_grid(new_session)
            if new_session.has_context_input:
                _run_context_grid(new_session)
            _run_composite_grid(new_session)

    col1, col2 = st.columns(2)
    if col1.button(
        "Save session and load data",
        help="Create or update the saved session on disk, parse the uploaded files, and preview the data without computing grids.",
    ):
        process_inputs(False)
    if col2.button(
        "Save session, load data, and run all",
        help="Create or update the saved session on disk, parse the uploaded files, compute all available grids, and build the composite index in one step.",
    ):
        process_inputs(True)



def _render_grid_section(session: WorkingSession) -> None:
    st.header("2. Grids Computation")
    if not session.data_loaded:
        st.info("Load data to enable grid computations.")
        return

    recompute_help = {
        "subject": "Re-run the primary and secondary source-station distance grids using the current settings.",
        "gap": "Re-run the ΔGap grid using the current settings.",
        "context": "Re-run the contextual grid using the current context file, radius, and aggregation settings.",
    }

    def run_with_stop(name: str, label: str, runner) -> None:
        trigger_key = _run_key(name)
        if st.button(f"Re-compute {label}", help=recompute_help.get(name)):
            st.session_state[trigger_key] = True
        if st.session_state.get(trigger_key):
            stop_placeholder = st.empty()
            stop_placeholder.button(
                "Stop",
                key=f"stop_btn_{name}",
                help="Request an orderly stop after the current computation step completes.",
                on_click=lambda: st.session_state.update({_stop_key(name): True}),
            )
            runner(session)
            stop_placeholder.empty()

    run_with_stop("subject", "primary and secondary source-station distance grids", _run_subject_grid)
    run_with_stop("gap", "ΔGap grid", _run_gap_grid)
    if session.has_context_input:
        run_with_stop("context", f"{session.context_label} grid", _run_context_grid)

    if st.button(
        "Re-compute composite index",
        help="Rebuild the weighted composite index from the currently available grids.",
    ):
        _run_composite_grid(session)

    merged = session.merged()
    if merged is not None:
        st.success("Merged grid summary")
        st.dataframe(merged.describe(), width="stretch")



def _render_maps_section(session: WorkingSession) -> None:
    st.header("3. Maps")
    merged = session.merged()
    if merged is None:
        st.info("Compute grids to enable map previews.")
        return
    if not _ensure_cartopy_map_data(session.parameters):
        return

    legacy_df = merged.rename(
        columns={alias: legacy for legacy, alias in LEGACY_FEATURE_ALIAS.items()},
        errors="ignore",
    )

    bna_bytes = _bna_bytes(session)
    config = LegacyMapConfig()

    st.subheader("Legacy contour maps")
    for feature, label in LEGACY_FEATURE_ORDER:
        if feature not in legacy_df.columns:
            continue
        if feature == "context_value" and not session.has_context_map_data:
            continue
        if feature == "context_value":
            label = (
                f"{session.context_label} "
                f"({session.parameters.context_aggregation} within {session.parameters.context_radius_km:g} km)"
            )
        feature_help = f"Show or hide the {label} contour map preview."
        if feature == "context_value":
            feature_help = (
                "Show or hide the contextual contour map built from the uploaded context layer "
                "using the selected aggregation and radius."
            )
        if st.checkbox(f"Show {label}", value=True, key=f"legacy_feature_{feature}", help=feature_help):
            fig = render_legacy_contour(
                legacy_df,
                feature,
                session.parameters,
                config=config,
                bna_bytes=bna_bytes,
                colorbar_label=label,
            )
            png_bytes = figure_png_bytes(fig)
            st.image(png_bytes, caption=label)
            kmz_bytes = _legacy_overlay_kmz(legacy_df, feature, session.parameters)
            download_cols = st.columns(2)
            download_cols[0].download_button(
                label=f"Download {label} PNG",
                data=png_bytes,
                file_name=f"{feature}.png",
                mime="image/png",
                key=f"legacy_png_{feature}",
                help=f"Download the {label} contour map as a PNG image.",
            )
            if kmz_bytes:
                download_cols[1].download_button(
                    label=f"Download {label} overlay (KMZ)",
                    data=kmz_bytes,
                    file_name=f"{feature}_overlay.kmz",
                    mime="application/vnd.google-earth.kmz",
                    key=f"legacy_kmz_{feature}",
                    help=f"Download the {label} raster overlay for Google Earth.",
                )
            else:
                download_cols[1].caption("KMZ export unavailable")

    st.subheader("K-means priority map")
    if "composite_index" not in merged.columns:
        st.info("Compute the composite index to enable the priority map.")
        return

    k_col, init_col = st.columns(2)
    max_clusters = max(2, min(len(PRIORITY_LEVELS), 6))
    default_k = 4 if 4 <= max_clusters else max_clusters
    n_clusters = k_col.slider(
        "Number of clusters (k)",
        min_value=2,
        max_value=max_clusters,
        value=default_k,
        step=1,
        key="priority_kmeans_cluster_count",
        help="Number of K-means clusters used to group grid cells before ranking them into priority classes.",
    )
    n_init = init_col.slider(
        "K-means initializations",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        key="priority_kmeans_n_init",
        help="Number of K-means restarts. Higher values can improve cluster stability but take longer.",
    )
    normalized_weights = session.parameters.normalized_weights()
    clustering_feature_weights = {
        "subject_primary_weighted": normalized_weights.get("subject_primary", 0.0),
        "subject_secondary_weighted": normalized_weights.get("subject_secondary", 0.0),
        "delta_gap90_weighted": normalized_weights.get("gap", 0.0),
        "context_value": normalized_weights.get("context", 0.0),
    }
    apply_feature_weight_scaling = st.checkbox(
        "Emphasize priority weights during clustering",
        value=True,
        help="When enabled, each normalized feature is scaled by the square root of its priority weight before running K-means.",
        key="priority_apply_feature_weights",
    )

    try:
        prioritized = classify_priority_clusters(
            merged,
            n_clusters=n_clusters,
            n_init=int(n_init),
            apply_feature_weights=apply_feature_weight_scaling,
            feature_weights=clustering_feature_weights,
        )
    except Exception as exc:
        st.error(f"Priority clustering failed: {exc}")
        return

    with st.expander("Decoration options", expanded=False):
        scale_col1, scale_col2, scale_col3 = st.columns(3)
        arrow_col1, arrow_col2, arrow_col3 = st.columns(3)

        scale_bar_length = scale_col1.slider(
            "Scale length (km)",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="priority_scale_length",
            help="Length of the map scale bar in kilometers.",
        )
        scale_bar_x = scale_col2.slider(
            "Scale X",
            min_value=0.0,
            max_value=1.0,
            value=0.22,
            step=0.01,
            key="priority_scale_x",
            help="Horizontal placement of the scale bar as a fraction of the figure width.",
        )
        scale_bar_y = scale_col3.slider(
            "Scale Y",
            min_value=0.0,
            max_value=1.0,
            value=0.06,
            step=0.01,
            key="priority_scale_y",
            help="Vertical placement of the scale bar as a fraction of the figure height.",
        )

        north_arrow_x = arrow_col1.slider(
            "Arrow X",
            min_value=0.0,
            max_value=1.0,
            value=0.92,
            step=0.01,
            key="priority_arrow_x",
            help="Horizontal placement of the north arrow as a fraction of the figure width.",
        )
        north_arrow_y = arrow_col2.slider(
            "Arrow Y",
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.01,
            key="priority_arrow_y",
            help="Vertical placement of the north arrow as a fraction of the figure height.",
        )
        north_arrow_length = arrow_col3.slider(
            "Arrow length",
            min_value=0.02,
            max_value=0.15,
            value=0.03,
            step=0.01,
            key="priority_arrow_length",
            help="Length of the north arrow relative to the figure size.",
        )

    fig = render_priority_clusters(
        prioritized,
        session.parameters,
        stations=session.stations,
        bna_bytes=bna_bytes,
        title=f"Priority areas (K-means, k={n_clusters})",
        scale_bar_length_km=float(scale_bar_length),
        scale_bar_location=(float(scale_bar_x), float(scale_bar_y)),
        north_arrow_location=(float(north_arrow_x), float(north_arrow_y)),
        north_arrow_length=float(north_arrow_length),
    )
    st.pyplot(fig)
    png_bytes = figure_png_bytes(fig)
    shapefile_columns = ["priority_cluster", "priority_label"]
    if "composite_index" in prioritized.columns:
        shapefile_columns.append("composite_index")
    priority_overlay = _priority_overlay_kmz(prioritized, session.parameters)
    download_cols = st.columns(2)
    download_cols[0].download_button(
        label="Download priority map PNG",
        data=png_bytes,
        file_name=f"priority_map_k{n_clusters}.png",
        mime="image/png",
        key="priority_png_download",
        help="Download the priority clustering map as a PNG image.",
    )
    if priority_overlay:
        download_cols[1].download_button(
            label="Download priority overlay (KMZ)",
            data=priority_overlay,
            file_name=f"priority_map_k{n_clusters}_overlay.kmz",
            mime="application/vnd.google-earth.kmz",
            key="priority_overlay_download",
            help="Download the priority clustering raster overlay for Google Earth.",
        )
    else:
        download_cols[1].caption("Overlay export unavailable")

    summary = (
        prioritized.groupby("priority_label")
        .size()
        .rename("grid_cells")
        .reset_index()
        .sort_values("priority_label")
    )
    st.dataframe(summary, width="stretch")

    export_columns = ["latitude", "longitude", "priority_cluster", "priority_label"]
    if "composite_index" in prioritized.columns:
        export_columns.append("composite_index")
    csv_bytes = prioritized[export_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download priority labels CSV",
        data=csv_bytes,
        file_name=f"priority_labels_k{n_clusters}.csv",
        mime="text/csv",
        help="Download the per-grid-cell priority labels as a CSV table.",
    )


def main() -> None:
    st.set_page_config(page_title="SeisNetInsight", layout="wide")
    st.title("SeisNetInsight")
    st.caption("Insights to support data-informed station siting")
    session = get_working_session()
    _render_data_loading(session)
    session = get_working_session()
    _render_grid_section(session)
    session = get_working_session()
    _render_maps_section(session)


if __name__ == "__main__":
    main()
