"""Priority clustering helpers for map outputs."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


FEATURE_COLUMNS = [
    "subject_primary_weighted",
    "subject_secondary_weighted",
    "delta_gap90_weighted",
    "context_value",
    "composite_index",
]
DEFAULT_PRIORITY_FEATURES = [
    "subject_primary_weighted",
    "subject_secondary_weighted",
    "delta_gap90_weighted",
    "context_value",
]
PRIORITY_LEVELS = ["Very High", "High", "Medium", "Low"]


def _priority_color_lookup(levels: Iterable[str]) -> Dict[str, list[int]]:
    palette = mpl.colormaps["hot_r"](np.linspace(0.75, 0.35, len(PRIORITY_LEVELS)))
    colors: Dict[str, list[int]] = {}
    for label, rgba in zip(PRIORITY_LEVELS, palette):
        rgb = [int(channel * 255) for channel in rgba[:3]]
        colors[label] = rgb
    for label in levels:
        if label not in colors:
            colors[label] = list(colors[PRIORITY_LEVELS[min(len(PRIORITY_LEVELS) - 1, len(colors))]])
    return colors


PRIORITY_COLORS = _priority_color_lookup(PRIORITY_LEVELS)


def classify_priority_clusters(
    df: pd.DataFrame,
    *,
    features: Optional[Iterable[str]] = None,
    n_clusters: int = 4,
    random_state: int = 42,
    n_init: int = 20,
    apply_feature_weights: bool = True,
    feature_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    if features is None:
        feature_names = list(DEFAULT_PRIORITY_FEATURES)
    else:
        feature_names = list(features)
    if "context_value" not in df.columns and "swd_volume_25km_bbl" in df.columns:
        df = df.copy()
        df["context_value"] = df["swd_volume_25km_bbl"]
    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Cannot classify priorities on an empty DataFrame.")

    feature_frame = df[feature_names].fillna(0.0)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feature_frame.to_numpy(dtype=float))
    if apply_feature_weights and feature_weights:
        weight_vector = np.array(
            [max(float(feature_weights.get(name, 1.0)), 0.0) for name in feature_names],
            dtype=float,
        )
        if np.any(weight_vector > 0):
            scaled = scaled * np.sqrt(weight_vector)

    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
    )
    clusters = model.fit_predict(scaled)
    result = df.copy()
    result["priority_cluster"] = clusters

    def _label_mapping(series: pd.Series) -> Dict[int, str]:
        order = series.sort_values(ascending=False).index.tolist()
        labels = PRIORITY_LEVELS.copy()
        if len(order) > len(labels):
            labels.extend([f"Priority {idx + 1}" for idx in range(len(labels), len(order))])
        mapping: Dict[int, str] = {}
        for idx, cluster in enumerate(order):
            mapping[cluster] = labels[idx] if idx < len(labels) else labels[-1]
        return mapping

    if "composite_index" in result.columns:
        mean_scores = result.groupby("priority_cluster")["composite_index"].mean()
        mapping = _label_mapping(mean_scores)
    else:
        sums = result.groupby("priority_cluster")[feature_names].mean().sum(axis=1)
        mapping = _label_mapping(sums)
    result["priority_label"] = result["priority_cluster"].map(mapping)
    return result
