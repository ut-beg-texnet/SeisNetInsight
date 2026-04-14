"""Data loading utilities."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

EXPECTED_EVENT_COLUMNS = ["latitude", "longitude", "origin_time"]
EXPECTED_STATION_COLUMNS = ["latitude", "longitude"]
EXPECTED_CONTEXT_COLUMNS = ["latitude", "longitude", "value"]
EXPECTED_SWD_COLUMNS = ["latitude", "longitude", "volume"]

COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "latitude": ["latitude", "lat", "Latitude", "Latitude (WGS84)"],
    "longitude": ["longitude", "lon", "Longitude", "Longitude (WGS84)", "Longtitude"],
    "magnitude": ["magnitude", "mag", "Magnitude"],
    "origin_time": ["origin_time", "time", "Time", "origin", "Origin Date", "event_time"],
    "value": [
        "value",
        "Value",
        "val",
        "volume",
        "vol",
        "SUM_injected_liquid_BBL",
        "total_volume",
        "PopMax",
        "pop_max",
        "population",
    ],
    "volume": ["volume", "vol", "SUM_injected_liquid_BBL", "total_volume", "value"],
}


def _read_pipe_dataframe(text: str) -> pd.DataFrame:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return pd.read_csv(io.StringIO(text))
    normalized_lines = lines.copy()
    if normalized_lines[0].startswith("#"):
        normalized_lines[0] = normalized_lines[0][1:]
    df = pd.read_csv(
        io.StringIO("\n".join(normalized_lines)),
        sep="|",
        engine="python",
        skipinitialspace=True,
    )
    df.columns = [str(col).strip() for col in df.columns]
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].astype(str).str.strip()
    return df


def _read_dataframe(source) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        content = Path(source).read_bytes()
    elif isinstance(source, bytes):
        content = source
    elif hasattr(source, "read"):
        content = source.read()
    else:
        raise TypeError(f"Unsupported source type: {type(source)!r}")

    if isinstance(content, bytes):
        text = content.decode("utf-8-sig", errors="replace")
    else:
        text = str(content)
    first_non_empty = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if "|" in first_non_empty:
        return _read_pipe_dataframe(text)
    if isinstance(content, bytes):
        return pd.read_csv(io.BytesIO(content))
    if hasattr(source, "read"):
        return pd.read_csv(io.StringIO(text))
    if isinstance(source, (str, Path)):
        return pd.read_csv(io.StringIO(text))
    raise TypeError(f"Unsupported source type: {type(source)!r}")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    lowered = {str(column).strip().lower(): column for column in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            candidate = lowered.get(str(alias).strip().lower())
            if candidate is not None:
                rename_map[candidate] = canonical
                break
    return df.rename(columns=rename_map)


def _apply_column_mapping(df: pd.DataFrame, column_map: Optional[Dict[str, str]]) -> pd.DataFrame:
    if not column_map:
        return df
    rename_map: Dict[str, str] = {}
    lower_lookup = {col.lower(): col for col in df.columns}
    for canonical, raw_actual in column_map.items():
        if not raw_actual:
            continue
        actual = str(raw_actual).strip()
        if actual in df.columns:
            rename_map[actual] = canonical
            continue
        candidate = lower_lookup.get(actual.lower())
        if candidate:
            rename_map[candidate] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def validate_required_columns(df: pd.DataFrame, required: Iterable[str]) -> List[str]:
    missing = [col for col in required if col not in df.columns]
    return missing


def load_events(
    source,
    *,
    column_map: Optional[Dict[str, str]] = None,
    warn: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    df = _read_dataframe(source)
    df = _apply_column_mapping(df, column_map)
    df = _standardize_columns(df)
    missing = validate_required_columns(df, EXPECTED_EVENT_COLUMNS)
    if "origin_time" in df.columns:
        origin_time = df["origin_time"].astype(str).str.strip()
        if "Origin Time" in df.columns:
            origin_clock = df["Origin Time"].astype(str).str.strip()
            combined = pd.to_datetime(origin_time + " " + origin_clock, errors="coerce")
            if combined.notna().any():
                df["origin_time"] = combined
            else:
                df["origin_time"] = pd.to_datetime(df["origin_time"], errors="coerce")
        else:
            df["origin_time"] = pd.to_datetime(df["origin_time"], errors="coerce")
    if warn and missing:
        for column in missing:
            pd.Series(dtype="object")  # touch pandas to avoid lint
    return df, missing


def load_stations(
    source,
    *,
    column_map: Optional[Dict[str, str]] = None,
    warn: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    df = _read_dataframe(source)
    df = _apply_column_mapping(df, column_map)
    df = _standardize_columns(df)
    missing = validate_required_columns(df, EXPECTED_STATION_COLUMNS)
    if warn and missing:
        for column in missing:
            pd.Series(dtype="object")
    return df, missing


def load_context_points(
    source,
    *,
    column_map: Optional[Dict[str, str]] = None,
    warn: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    df = _read_dataframe(source)
    df = _apply_column_mapping(df, column_map)
    df = _standardize_columns(df)
    if "volume" in df.columns and "value" not in df.columns:
        df = df.rename(columns={"volume": "value"})
    missing = validate_required_columns(df, EXPECTED_CONTEXT_COLUMNS)
    for column in ("latitude", "longitude", "value"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if warn and missing:
        for column in missing:
            pd.Series(dtype="object")
    return df, missing


def load_swd_wells(
    source,
    *,
    column_map: Optional[Dict[str, str]] = None,
    warn: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    df, missing = load_context_points(source, column_map=column_map, warn=warn)
    if "value" in df.columns:
        df = df.rename(columns={"value": "volume"})
    missing = ["volume" if column == "value" else column for column in missing]
    return df, missing


def balltree_reduce_events(
    df: pd.DataFrame,
    *,
    distance_threshold_km: float,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    keep: str = "first",
    leaf_size: int = 40,
) -> pd.DataFrame:
    from sklearn.neighbors import BallTree
    import numpy as np

    working = df.copy()
    if keep == "most_recent":
        sort_key = None
        if "origin_time" in working.columns:
            sort_key = pd.to_datetime(working["origin_time"], errors="coerce")
        elif {"Origin Date", "Origin Time"}.issubset(working.columns):
            sort_key = pd.to_datetime(
                working["Origin Date"].astype(str).str.strip()
                + " "
                + working["Origin Time"].astype(str).str.strip(),
                errors="coerce",
            )
        elif "Origin Date" in working.columns:
            sort_key = pd.to_datetime(working["Origin Date"], errors="coerce")
        if sort_key is not None:
            working["_balltree_sort_key"] = sort_key
            working.sort_values(
                ["_balltree_sort_key", lat_col, lon_col],
                ascending=[False, True, True],
                inplace=True,
                na_position="last",
            )
            working.drop(columns=["_balltree_sort_key"], inplace=True)
        working.reset_index(drop=True, inplace=True)
    elif keep != "first":
        raise ValueError("keep must be either 'first' or 'most_recent'.")

    coords = working[[lat_col, lon_col]].to_numpy()
    radians = np.radians(coords)
    tree = BallTree(radians, metric="haversine", leaf_size=leaf_size)
    mask = np.ones(len(working), dtype=bool)
    radius = distance_threshold_km / 6371.0
    for idx in range(len(working)):
        if not mask[idx]:
            continue
        neighbors = tree.query_radius([radians[idx]], r=radius)[0]
        mask[neighbors] = False
        mask[idx] = True
    return working.loc[mask].reset_index(drop=True)
