"""Grid generation and metric computation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from pyproj import Geod
from sklearn.neighbors import BallTree

from .config import GridParameters

EARTH_RADIUS_KM = 6371.0
GEOD = Geod(ellps="WGS84")


@dataclass
class GridDefinition:
    latitudes: np.ndarray
    longitudes: np.ndarray
    coordinates: np.ndarray

    @property
    def size(self) -> int:
        return self.coordinates.shape[0]


def _clip_to_aoi(df: pd.DataFrame, params: GridParameters) -> pd.DataFrame:
    """Return only rows that fall within the configured AOI bounds."""
    if df is None or df.empty:
        return df
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df
    lat_min, lat_max = sorted(params.lats)
    lon_min, lon_max = sorted(params.lons)
    mask = (
        df["latitude"].between(lat_min, lat_max)
        & df["longitude"].between(lon_min, lon_max)
    )
    return df.loc[mask].copy()


def _filter_stations(stations: pd.DataFrame) -> pd.DataFrame:
    """Drop stations from filtered networks (e.g., AM) when that metadata is available."""
    if stations is None or stations.empty:
        return stations
    for col in ("Network Code", "network_code", "network"):
        if col in stations.columns:
            mask = stations[col].astype(str).str.upper() != "AM"
            return stations.loc[mask].copy()
    return stations


def generate_grid(params: GridParameters) -> GridDefinition:
    lats = np.arange(params.lats[0], params.lats[1] + params.grid_step, params.grid_step)
    lons = np.arange(params.lons[0], params.lons[1] + params.grid_step, params.grid_step)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    coords = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    return GridDefinition(latitudes=lats, longitudes=lons, coordinates=coords)


def _haversine(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    a = np.radians(points_a)
    b = np.radians(points_b)
    dlat = a[:, None, 0] - b[None, :, 0]
    dlon = a[:, None, 1] - b[None, :, 1]
    lat1 = a[:, None, 0]
    lat2 = b[None, :, 0]
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(h), np.sqrt(1 - h))


def _flat_distance_km(point_lat: float, point_lon: float, coords: np.ndarray) -> np.ndarray:
    """Planar distance approximation (deg -> km) to mirror legacy notebook behavior."""
    delta = coords - np.array([[point_lat, point_lon]])
    return np.hypot(delta[:, 0], delta[:, 1]) * 111.0


def _progress_wrapper(progress: Optional[Callable[[float, str], None]], fraction: float, message: str) -> None:
    if progress:
        progress(fraction, message)


def compute_subject_grids(
    events: pd.DataFrame,
    stations: pd.DataFrame,
    grid: GridDefinition,
    params: GridParameters,
    *,
    progress: Optional[Callable[[float, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> pd.DataFrame:
    if events.empty or stations.empty:
        raise ValueError("Events and stations data must be provided to compute subject grids.")

    event_coords = events[["latitude", "longitude"]].to_numpy()
    station_coords = stations[["latitude", "longitude"]].to_numpy()

    dist_events_stations = _haversine(event_coords, station_coords)
    within_primary = (
        (dist_events_stations <= params.subject_primary_radius_km).sum(axis=1)
        >= params.subject_primary_min_stations
    )
    within_secondary = (
        (dist_events_stations <= params.subject_secondary_radius_km).sum(axis=1)
        >= params.subject_secondary_min_stations
    )
    subject_primary_mask = ~within_primary
    subject_secondary_mask = ~within_secondary

    if "origin_time" in events.columns and events["origin_time"].notna().any():
        ref_time = events["origin_time"].max()
        age_years = (ref_time - events["origin_time"]).dt.days / 365.25
        weights = np.power(0.5, age_years / params.half_time_years)
    else:
        weights = np.ones(len(events), dtype=float)
    weights = np.asarray(weights, dtype=float)

    grid_coords = grid.coordinates
    n_grid = grid.size
    chunk = max(500, int(2000 * (0.01 / params.grid_step)))
    chunk = min(chunk, n_grid)

    primary_counts = np.zeros(n_grid, dtype=np.int32)
    secondary_counts = np.zeros(n_grid, dtype=np.int32)
    primary_weighted = np.zeros(n_grid, dtype=float)
    secondary_weighted = np.zeros(n_grid, dtype=float)

    for start in range(0, n_grid, chunk):
        if should_stop and should_stop():
            raise InterruptedError("Subject grid computation interrupted by user.")
        end = min(start + chunk, n_grid)
        distances = _haversine(grid_coords[start:end], event_coords)
        if subject_primary_mask.any():
            mask = distances[:, subject_primary_mask] <= params.subject_primary_radius_km
            primary_counts[start:end] = mask.sum(axis=1)
            primary_weighted[start:end] = (mask * weights[subject_primary_mask]).sum(axis=1)
        if subject_secondary_mask.any():
            mask_secondary = distances[:, subject_secondary_mask] <= params.subject_secondary_radius_km
            secondary_counts[start:end] = mask_secondary.sum(axis=1)
            secondary_weighted[start:end] = (mask_secondary * weights[subject_secondary_mask]).sum(axis=1)
        _progress_wrapper(progress, end / n_grid, "Subject grids")

    return pd.DataFrame(
        {
            "latitude": grid_coords[:, 0],
            "longitude": grid_coords[:, 1],
            "subject_primary_count": primary_counts,
            "subject_secondary_count": secondary_counts,
            "subject_primary_weighted": primary_weighted,
            "subject_secondary_weighted": secondary_weighted,
        }
    )


def _station_subset(stations: pd.DataFrame, event_lat: float, event_lon: float, max_distance_km: float) -> pd.DataFrame:
    coords = stations[["latitude", "longitude"]].to_numpy()
    d = _flat_distance_km(event_lat, event_lon, coords)
    mask = d <= max_distance_km
    if mask.sum() < 2:
        return stations.iloc[0:0]
    return stations.iloc[mask]


def _max_gap(event_lat: float, event_lon: float, stations: pd.DataFrame) -> float:
    if len(stations) < 2:
        return 360.0
    azimuths = []
    for lat, lon in stations[["latitude", "longitude"]].itertuples(index=False):
        _, az, _ = GEOD.inv(event_lon, event_lat, lon, lat)
        azimuths.append(az % 360)
    azimuths = np.sort(np.asarray(azimuths))
    diffs = np.diff(np.append(azimuths, azimuths[0] + 360.0))
    return float(diffs.max())


def compute_gap_grid(
    events: pd.DataFrame,
    stations: pd.DataFrame,
    grid: GridDefinition,
    params: GridParameters,
    *,
    progress: Optional[Callable[[float, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> pd.DataFrame:
    events = _clip_to_aoi(events, params)
    stations = _filter_stations(_clip_to_aoi(stations, params))
    if events.empty or stations.empty:
        raise ValueError("Events and stations data must be provided to compute gap grids.")

    critical_events = []
    threshold = float(getattr(params, "gap_target_angle_deg", 90.0))
    for idx, row in events.iterrows():
        subset = _station_subset(stations, row["latitude"], row["longitude"], params.gap_search_km)
        max_gap = _max_gap(row["latitude"], row["longitude"], subset)
        if max_gap > threshold:
            critical_events.append((idx, row, max_gap))
    if not critical_events:
        return pd.DataFrame(
            {
                "latitude": grid.coordinates[:, 0],
                "longitude": grid.coordinates[:, 1],
                "delta_gap90_weighted": np.zeros(grid.size, dtype=float),
            }
        )

    if "origin_time" in events.columns and events["origin_time"].notna().any():
        ref_time = events["origin_time"].max()
        weights = np.power(
            0.5,
            (ref_time - events.loc[[idx for idx, _, _ in critical_events]]["origin_time"]).dt.days / 365.25 / params.half_time_years,
        ).astype(float)
    else:
        weights = np.ones(len(critical_events), dtype=float)

    tree = BallTree(np.radians(grid.coordinates), metric="haversine")
    radius = params.gap_search_km / EARTH_RADIUS_KM
    improvements = np.zeros(grid.size, dtype=float)

    for i, (idx, row, _) in enumerate(critical_events):
        if should_stop and should_stop():
            raise InterruptedError("Gap grid computation interrupted by user.")
        event_coord = np.radians([[row["latitude"], row["longitude"]]])
        indices = tree.query_radius(event_coord, r=radius)[0]
        if indices.size == 0:
            _progress_wrapper(progress, (i + 1) / len(critical_events), "ΔGap90 grid")
            continue
        subset = _station_subset(stations, row["latitude"], row["longitude"], params.gap_search_km)
        base_azimuths = []
        for lat, lon in subset[["latitude", "longitude"]].itertuples(index=False):
            _, az, _ = GEOD.inv(row["longitude"], row["latitude"], lon, lat)
            base_azimuths.append(az % 360)
        base_azimuths = np.sort(np.asarray(base_azimuths))
        for gi in indices:
            grid_lat, grid_lon = grid.coordinates[gi]
            _, az_new, _ = GEOD.inv(row["longitude"], row["latitude"], grid_lon, grid_lat)
            az_new = az_new % 360
            insert_pos = np.searchsorted(base_azimuths, az_new)
            extended = np.insert(base_azimuths, insert_pos, az_new)
            diffs = np.diff(np.append(extended, extended[0] + 360.0))
            if diffs.max() <= threshold:
                improvements[gi] += weights.iloc[i]
        _progress_wrapper(progress, (i + 1) / len(critical_events), "ΔGap90 grid")

    return pd.DataFrame(
        {
            "latitude": grid.coordinates[:, 0],
            "longitude": grid.coordinates[:, 1],
            "delta_gap90_weighted": improvements,
        }
    )


def compute_swd_grid(
    swd: pd.DataFrame,
    grid: GridDefinition,
    params: GridParameters,
    *,
    progress: Optional[Callable[[float, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> pd.DataFrame:
    if swd is None or swd.empty:
        return pd.DataFrame(
            {
                "latitude": grid.coordinates[:, 0],
                "longitude": grid.coordinates[:, 1],
                "swd_volume_25km_bbl": np.zeros(grid.size, dtype=float),
            }
        )

    well_coords = swd[["latitude", "longitude"]].to_numpy()
    tree = BallTree(np.radians(well_coords), metric="haversine")
    radius = params.swd_radius_km / EARTH_RADIUS_KM
    totals = np.zeros(grid.size, dtype=float)

    chunk = 2000
    for start in range(0, grid.size, chunk):
        if should_stop and should_stop():
            raise InterruptedError("SWD grid computation interrupted by user.")
        end = min(start + chunk, grid.size)
        grid_chunk = np.radians(grid.coordinates[start:end])
        neighborhoods = tree.query_radius(grid_chunk, r=radius)
        for offset, neighbors in enumerate(neighborhoods):
            if neighbors.size:
                totals[start + offset] = swd.iloc[neighbors]["volume"].sum()
        _progress_wrapper(progress, end / grid.size, "SWD grid")

    return pd.DataFrame(
        {
            "latitude": grid.coordinates[:, 0],
            "longitude": grid.coordinates[:, 1],
            "swd_volume_25km_bbl": totals,
        }
    )


def merge_grids(*frames: pd.DataFrame) -> pd.DataFrame:
    base = None
    for frame in frames:
        if frame is None or frame.empty:
            continue
        if base is None:
            base = frame.copy()
        else:
            base = base.merge(frame, on=["latitude", "longitude"], how="inner")
    if base is None:
        raise ValueError("At least one grid frame must be provided.")
    base.sort_values(["latitude", "longitude"], inplace=True)
    base.reset_index(drop=True, inplace=True)
    return base


def compute_composite_index(df: pd.DataFrame, params: GridParameters) -> pd.DataFrame:
    required_columns = [
        "subject_primary_weighted",
        "subject_secondary_weighted",
        "delta_gap90_weighted",
        "swd_volume_25km_bbl",
    ]
    for column in required_columns:
        if column not in df.columns:
            df[column] = 0.0

    def minmax(values: pd.Series) -> pd.Series:
        vmin, vmax = values.min(), values.max()
        if not np.isfinite(vmin) or not np.isfinite(vmax) or math.isclose(vmin, vmax):
            return pd.Series(np.zeros(len(values)), index=values.index)
        return (values - vmin) / (vmax - vmin)

    weights = params.normalized_weights()

    scaled = {
        "subject_primary": minmax(df["subject_primary_weighted"]),
        "subject_secondary": minmax(df["subject_secondary_weighted"]),
        "gap": minmax(df["delta_gap90_weighted"]),
        "swd": minmax(df["swd_volume_25km_bbl"]),
    }

    composite = sum(weights[key] * scaled[key] for key in scaled)
    df = df.copy()
    df["composite_index"] = composite
    df["contrib_subject_primary"] = weights["subject_primary"] * scaled["subject_primary"]
    df["contrib_subject_secondary"] = weights["subject_secondary"] * scaled["subject_secondary"]
    df["contrib_gap"] = weights["gap"] * scaled["gap"]
    df["contrib_swd"] = weights["swd"] * scaled["swd"]
    return df
