"""Legacy static map rendering to match historical scripted outputs."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from cmcrameri import cm as cmc
from pyproj import Geod

from .config import GridParameters
from .maps import PRIORITY_COLORS, PRIORITY_LEVELS

REFERENCE_CITIES: Tuple[Tuple[str, float, float], ...] = (
    ("Midland, TX", -102.077408, 32.000507),
    ("Odessa, TX", -102.367645, 31.845682),
    ("Lubbock, TX", -101.855166, 33.577863),
    ("Big Spring, TX", -101.479073, 32.250286),
)


@dataclass
class LegacyMapConfig:
    """Configuration options for legacy contour maps."""

    show_roads: bool = True
    show_states: bool = True
    show_cities: bool = True
    bna_label: Optional[str] = "Area of interest"


def _auto_levels(values: np.ndarray, *, n: int = 12, force_min: Optional[float] = None, force_max: Optional[float] = None) -> np.ndarray:
    zmin = np.nanmin(values) if force_min is None else force_min
    zmax = np.nanmax(values) if force_max is None else force_max
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        return np.linspace(0.0, 1.0, n)
    return np.linspace(zmin, zmax, n)


def _pivot_grid(df: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = (
        df.pivot(index="latitude", columns="longitude", values=value_col)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    lats = pivot.index.to_numpy()
    lons = pivot.columns.to_numpy()
    data = pivot.to_numpy()
    return lats, lons, data


def _parse_bna_lines(bna_bytes: bytes) -> List[np.ndarray]:
    """Parse a minimal BNA polygon definition into coordinate arrays."""
    polygons: List[np.ndarray] = []
    buffer = io.StringIO(bna_bytes.decode("utf-8", errors="ignore"))
    while True:
        header = buffer.readline()
        if not header:
            break
        parts = [part.strip().strip('"') for part in header.split(",")]
        if len(parts) < 3:
            continue
        try:
            num_points = int(parts[2])
        except ValueError:
            continue
        coords: List[Tuple[float, float]] = []
        for _ in range(num_points):
            line = buffer.readline()
            if not line:
                break
            try:
                x_str, y_str = line.split(",")[:2]
                coords.append((float(x_str), float(y_str)))
            except ValueError:
                continue
        if coords:
            polygons.append(np.array(coords))
    return polygons


GEOD = Geod(ellps="WGS84")


def _cities_within_aoi(
    params: GridParameters,
    cities: Sequence[Tuple[str, float, float]] = REFERENCE_CITIES,
) -> List[Tuple[str, float, float]]:
    lon_min, lon_max = sorted(params.lons)
    lat_min, lat_max = sorted(params.lats)
    visible: List[Tuple[str, float, float]] = []
    for name, lon, lat in cities:
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            visible.append((name, lon, lat))
    return visible


def _add_scale_bar(
    ax: plt.Axes,
    *,
    length_km: float = 20.0,
    location: Tuple[float, float] = (0.12, 0.08),
) -> None:
    xmin, xmax, ymin, ymax = ax.get_extent(ccrs.PlateCarree())
    lon0 = xmin + (xmax - xmin) * location[0]
    lat0 = ymin + (ymax - ymin) * location[1]
    lon1, lat1, _ = GEOD.fwd(lon0, lat0, 90.0, length_km * 1000.0)
    ax.plot(
        [lon0, lon1],
        [lat0, lat0],
        color="black",
        linewidth=1.5,
        transform=ccrs.PlateCarree(),
        zorder=7,
    )
    tick_offset = (ymax - ymin) * 0.004
    for lon in (lon0, lon1):
        ax.plot(
            [lon, lon],
            [lat0 - tick_offset, lat0 + tick_offset],
            color="black",
            linewidth=1.5,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )
    ax.text(
        (lon0 + lon1) / 2.0,
        lat0 - tick_offset * 3,
        f"{int(length_km)} km",
        ha="center",
        va="top",
        fontsize=8,
        transform=ccrs.PlateCarree(),
        zorder=7,
    )


def _add_north_arrow(
    ax: plt.Axes,
    *,
    location: Tuple[float, float] = (0.92, 0.12),
    length: float = 0.05,
) -> None:
    x, y = location
    ax.annotate(
        "",
        xy=(x, y + length),
        xytext=(x, y),
        xycoords="axes fraction",
        arrowprops=dict(facecolor="black", edgecolor="black", width=2, headwidth=8),
        zorder=7,
    )
    ax.text(
        x,
        y + length + 0.01,
        "N",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        zorder=7,
    )


def render_legacy_contour(
    df: pd.DataFrame,
    value_col: str,
    params: GridParameters,
    *,
    cmap=cmc.batlow,
    config: Optional[LegacyMapConfig] = None,
    levels: Optional[Sequence[float]] = None,
    bna_bytes: Optional[bytes] = None,
    title: Optional[str] = None,
    colorbar_labelpad: Optional[float] = None,
    colorbar_label: Optional[str] = None,
) -> plt.Figure:
    """Create a contour map figure that mirrors the legacy scripted output."""
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not available for mapping.")
    config = config or LegacyMapConfig()

    lats, lons, Z = _pivot_grid(df, value_col)
    if levels is None:
        levels = (
            np.linspace(0.0, 1.0, 11)
            if value_col == "composite_index"
            else _auto_levels(Z, n=12)
        )

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([params.lons[0], params.lons[1], params.lats[0], params.lats[1]], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, color="lightgrey", zorder=0)
    if config.show_roads:
        roads_feat = cfeature.NaturalEarthFeature("cultural", "roads", "10m", linewidth=0.4, facecolor="none")
        ax.add_feature(roads_feat, edgecolor="gray", zorder=1)
    if config.show_states:
        ax.add_feature(cfeature.STATES, linestyle=":", zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=2)

    if config.show_cities:
        for name, lon, lat in _cities_within_aoi(params):
            ax.text(lon, lat, name, fontsize=5, ha="right", color="gray", transform=ccrs.PlateCarree(), zorder=3)

    cs = ax.contourf(lons, lats, Z, levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), alpha=0.9)
    cb = plt.colorbar(cs, ax=ax)
    label_text = value_col.replace("_", " ") if colorbar_label is None else colorbar_label
    if colorbar_labelpad is None:
        cb.set_label(label_text)
    else:
        cb.set_label(label_text, labelpad=colorbar_labelpad)
    # cb.set_label(value_col.replace("_", " "), labelpad=-20)
    if bna_bytes:
        for poly in _parse_bna_lines(bna_bytes):
            ax.plot(poly[:, 0], poly[:, 1], color="black", linewidth=1.0, transform=ccrs.PlateCarree(), label=config.bna_label)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title, fontsize=10)

    return fig


def figure_png_bytes(fig: plt.Figure, *, dpi: int = 300, pad_inches: float = 0.1) -> bytes:
    """Serialize a Matplotlib figure to PNG bytes."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def render_priority_clusters(
    df: pd.DataFrame,
    params: GridParameters,
    *,
    stations: Optional[pd.DataFrame] = None,
    priority_column: str = "priority_label",
    config: Optional[LegacyMapConfig] = None,
    bna_bytes: Optional[bytes] = None,
    title: str = "Priority areas (K-means)",
    scale_bar_length_km: float = 20.0,
    scale_bar_location: Tuple[float, float] = (0.22, 0.06),
    north_arrow_location: Tuple[float, float] = (0.92, 0.02),
    north_arrow_length: float = 0.03,
) -> plt.Figure:
    """Render a K-means based priority classification map."""
    if priority_column not in df.columns:
        raise KeyError(f"Column '{priority_column}' not available for mapping.")
    config = config or LegacyMapConfig()

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([params.lons[0], params.lons[1], params.lats[0], params.lats[1]], crs=ccrs.PlateCarree())

    handles: List[patches.Patch] = []
    for label in PRIORITY_LEVELS:
        subset = df[df[priority_column] == label]
        if subset.empty:
            continue
        rgb = PRIORITY_COLORS.get(label, [127, 127, 127])
        color = "#%02x%02x%02x" % tuple(rgb)
        ax.scatter(
            subset["longitude"],
            subset["latitude"],
            color=color,
            label=None,
            s=6,
            alpha=0.9,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
        handles.append(patches.Patch(facecolor=color, edgecolor="none", alpha=0.7, label=label))

    other_labels = [lbl for lbl in sorted(df[priority_column].unique()) if lbl not in PRIORITY_LEVELS]
    for label in other_labels:
        subset = df[df[priority_column] == label]
        if subset.empty:
            continue
        rgb = PRIORITY_COLORS.get(label, [127, 127, 127])
        color = "#%02x%02x%02x" % tuple(rgb)
        ax.scatter(
            subset["longitude"],
            subset["latitude"],
            color=color,
            label=None,
            s=6,
            alpha=0.9,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
        handles.append(patches.Patch(facecolor=color, edgecolor="none", alpha=0.7, label=label))

    ax.add_feature(cfeature.OCEAN, color="#a9cce3", zorder=5)
    ax.add_feature(cfeature.LAKES, edgecolor="#a9cce3", facecolor="#a9cce3", linewidth=0.5, zorder=5)
    ax.add_feature(cfeature.RIVERS, edgecolor="#a9cce3", linewidth=0.5, zorder=5)
    if config.show_roads:
        roads_feat = cfeature.NaturalEarthFeature("cultural", "roads", "10m", linewidth=0.4, facecolor="none")
        ax.add_feature(roads_feat, edgecolor="gray", zorder=5)
    if config.show_states:
        ax.add_feature(cfeature.STATES, linestyle=":", zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=5)

    if config.show_cities:
        for name, lon, lat in _cities_within_aoi(params):
            ax.text(lon, lat, name, fontsize=5, ha="right", color="gray", transform=ccrs.PlateCarree(), zorder=5)

    if stations is not None and not stations.empty:
        ax.scatter(
            stations["longitude"],
            stations["latitude"],
            marker="^",
            color="#2533D0",
            s=25,
            label="Stations",
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

    if bna_bytes:
        for poly in _parse_bna_lines(bna_bytes):
            ax.plot(
                poly[:, 0],
                poly[:, 1],
                color="black",
                linewidth=1.0,
                transform=ccrs.PlateCarree(),
                zorder=6,
            )

    if handles:
        legend = ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.7)
        legend.set_title("Priority")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=10)
    _add_scale_bar(ax, length_km=scale_bar_length_km, location=scale_bar_location)
    _add_north_arrow(ax, location=north_arrow_location, length=north_arrow_length)
    return fig
