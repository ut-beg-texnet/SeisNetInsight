"""Session persistence helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import GridParameters, parameter_dict, parse_bounds

SESSION_ROOT = Path.home() / ".seisnetinsight" / "sessions"
SESSION_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class SessionFiles:
    events: Optional[str] = None
    stations: Optional[str] = None
    context: Optional[str] = None
    swd: Optional[str] = None
    bna_files: List[str] = field(default_factory=list)


@dataclass
class SessionState:
    name: str
    parameters: GridParameters
    files: SessionFiles
    grids: Dict[str, str] = field(default_factory=dict)
    column_mapping: Dict[str, Dict[str, str]] = field(default_factory=dict)
    context_label: str = "Context layer"

    def directory(self) -> Path:
        return SESSION_ROOT / self.name

    def save_metadata(self) -> None:
        directory = self.directory()
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "parameters": parameter_dict(self.parameters),
            "files": asdict(self.files),
            "grids": self.grids,
            "column_mapping": self.column_mapping,
            "context_label": self.context_label,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, name: str) -> "SessionState":
        directory = SESSION_ROOT / name
        if not directory.exists():
            raise FileNotFoundError(f"Session '{name}' does not exist")
        metadata = json.loads((directory / "metadata.json").read_text())
        raw_params = metadata.get("parameters", {})
        params = GridParameters()

        def _get(primary: str, legacy: Optional[str], cast, default):
            for key in filter(None, (primary, legacy)):
                if key in raw_params:
                    value = raw_params[key]
                    try:
                        return cast(value)
                    except Exception:
                        return default
            return default

        params.lons = parse_bounds(str(raw_params.get("LONS", ",".join(map(str, params.lons)))), params.lons)
        params.lats = parse_bounds(str(raw_params.get("LATS", ",".join(map(str, params.lats)))), params.lats)
        params.grid_step = _get("GRID_STEP", None, float, params.grid_step)
        params.subject_primary_radius_km = _get(
            "SUBJECT_PRIMARY_RADIUS_KM", "DIST_THRESHOLD_SUB4", float, params.subject_primary_radius_km
        )
        params.subject_primary_min_stations = _get(
            "SUBJECT_PRIMARY_MIN_STATIONS", "MIN_STA_SUB4", int, params.subject_primary_min_stations
        )
        params.subject_primary_weight = _get(
            "SUBJECT_PRIMARY_WEIGHT", "WEIGHT_SUB4", float, params.subject_primary_weight
        )
        params.subject_secondary_radius_km = _get(
            "SUBJECT_SECONDARY_RADIUS_KM", "DIST_THRESHOLD_SUB10", float, params.subject_secondary_radius_km
        )
        params.subject_secondary_min_stations = _get(
            "SUBJECT_SECONDARY_MIN_STATIONS", "MIN_STA_SUB10", int, params.subject_secondary_min_stations
        )
        params.subject_secondary_weight = _get(
            "SUBJECT_SECONDARY_WEIGHT", "WEIGHT_SUB10", float, params.subject_secondary_weight
        )
        params.gap_search_km = _get("GAP_SEARCH_KM", None, float, params.gap_search_km)
        params.gap_target_angle_deg = _get("GAP_TARGET_ANGLE", None, float, params.gap_target_angle_deg)
        params.weight_gap = _get("WEIGHT_GAP", None, float, params.weight_gap)
        params.context_radius_km = _get("CONTEXT_RADIUS_KM", "SWD_RADIUS_KM", float, params.context_radius_km)
        params.context_aggregation = str(
            raw_params.get("CONTEXT_AGGREGATION", getattr(params, "context_aggregation", "sum"))
        ).strip().lower() or "sum"
        params.weight_context = _get("WEIGHT_CONTEXT", "WEIGHT_SWD", float, params.weight_context)

        def _to_bool(value: object, default: bool) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in {"1", "true", "yes", "on"}
            return bool(value)

        params.half_time_years = _get("HALF_TIME_YEARS", None, float, params.half_time_years)
        params.overwrite = _to_bool(raw_params.get("OVERWRITE", params.overwrite), params.overwrite)
        files = SessionFiles(**metadata["files"])
        return cls(
            name=name,
            parameters=params,
            files=files,
            grids=metadata.get("grids", {}),
            column_mapping=metadata.get("column_mapping", {}),
            context_label=metadata.get("context_label", "Context layer"),
        )

    def save_dataframe(self, df: pd.DataFrame, key: str) -> None:
        directory = self.directory()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{key}.parquet"
        df.to_parquet(path, index=False)
        self.grids[key] = path.name
        self.save_metadata()

    def load_dataframe(self, key: str) -> pd.DataFrame:
        directory = self.directory()
        if key not in self.grids:
            raise KeyError(f"Grid '{key}' not available in session '{self.name}'")
        return pd.read_parquet(directory / self.grids[key])

    def save_source(self, df: pd.DataFrame, key: str) -> None:
        directory = self.directory()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{key}_source.parquet"
        df.to_parquet(path, index=False)
        setattr(self.files, key, path.name)
        self.save_metadata()

    def save_bna(self, filename: str, data: bytes) -> None:
        directory = self.directory()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_bytes(data)
        if filename not in self.files.bna_files:
            self.files.bna_files.append(filename)
        self.save_metadata()

    def load_source(self, key: str) -> Optional[pd.DataFrame]:
        filename = getattr(self.files, key, None)
        if key == "context" and not filename:
            filename = getattr(self.files, "swd", None)
        if not filename:
            return None
        path = self.directory() / filename
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def list_bna(self) -> List[Path]:
        directory = self.directory()
        return [directory / name for name in self.files.bna_files if (directory / name).exists()]


def list_sessions() -> List[str]:
    return sorted([p.name for p in SESSION_ROOT.glob("*") if p.is_dir() and (p / "metadata.json").exists()])
