# SeisNetInsight

SeisNetInsight provides tooling for the seismic community to evaluate station coverage, compute monitoring priority grids, and produce interactive maps. The new Python package bundles the grid computations, session persistence, and a Streamlit web application for exploratory analysis.

## Package overview

The package exposes utilities for:

- Loading and validating seismic event, station, and SWD well catalogues.
- Generating subject proximity (primary/secondary), ΔGap90, and SWD influence grids on a configurable mesh.
- Computing a composite monitoring index with user-defined weights and half-life decay.
- Exporting map artefacts (PNG, KMZ) from the computed grids.
- Launching an interactive web UI to guide users through data loading, grid computation, and map visualisation.

## Installation

```bash
pip install .
```

This installs the `seisnetinsight` package along with the `seisnetinsight-app` console script for the Streamlit interface.

## Running the web app

After installation, launch the interface with:

```bash
seisnetinsight-app
```

The UI is divided into three sections:

1. **Data loading** – Restore a previous session or configure a new one by uploading events, stations, optional SWD well catalogues, and BNA polygons. The interface shows the first 10 rows of each dataset, warns about missing expected columns, and optionally applies BallTree catalogue reduction.
2. **Grids computation** – Compute subject, ΔGap90, SWD, and composite grids with progress indicators, stop controls, and automatic session caching.
3. **Maps** – Explore interactive pydeck-based maps for each feature and the composite priority map. Download PNG, KML, Shapefile, or a consolidated PDF for enabled maps.

## GUI Overview:
![seisnetinsight_data_loadding_short](https://github.com/user-attachments/assets/358e7c7b-75b3-4b87-8a9c-7f8350d09706)

## Programmatic usage
For more details see the examples in the notebook folder
```python
from seisnetinsight.config import GridParameters
from seisnetinsight.data import load_events, load_stations, load_swd_wells
from seisnetinsight.grids import (
    compute_composite_index,
    compute_gap_grid,
    compute_subject_grids,
    compute_swd_grid,
    generate_grid,
    merge_grids,
)

bna_path = Path("../sample_files/midland.bna")
bna_bytes = bna_path.read_bytes()

SWD_COLUMN_MAP = {
    "latitude": "latitude_wgs84",
    "longitude": "longitude_wgs84",
    "volume": "SUM_injected_liquid_BBL",
}

PARAMS = GridParameters()

# events_df, stations_df, swd_df are pandas dataframes with latitude and longitude columns (more details in the notebooks folder)
grid = generate_grid(params)

subject_df = load_or_compute("subject",
    lambda: compute_subject_grids(events_df, stations_df, grid, params), overwrite=False)

gap_df = load_or_compute(
    "gap", lambda: compute_gap_grid(events_df, stations_df, grid, params), overwrite=False)

swd_grid = compute_swd_grid(swd_df, grid, params)
swd_grid = load_or_compute(
    "swd", lambda: compute_swd_grid(swd_df, grid, params), overwrite=False)

merged = merge_grids(subject_df, gap_df, swd_grid)
composite_grid_df = load_or_compute(
    "composite",
    lambda: compute_composite_index(
        merged,
        params),
    overwrite=False)
```

# Output Examples
Panels (a–c) show how adding a new station would affect nearby events. Brighter colors mean that placing a station in that grid cell would impact more recent events. (a) shows how many recency-weighted events would fall within 4 km of a station. (b) shows the same for a 10 km radius. (c) shows how many events would have their maximum azimuthal gap reduced to ≤90° if a station were placed there. (d) shows the total salt-water-disposal volume within 25 km (in barrels). The black outline marks the Midland Basin:
<img width="520" height="500" alt="image" src="https://github.com/user-attachments/assets/3f67335f-5525-4fbe-b931-56f7d7fa2c0d" />

(a) Map of a composite impact index (ranging from 0 to 1) that combines four metrics—S₄, S₁₀, ΔGap ≤ 90°, and SWD after adjusting for scaling, and weighting. Brighter colors highlight areas where a new station would have the most combined benefit. (b) Priority areas determined by k-means clustering of these metrics, ranked by their average composite index (Very High, High, Medium, Low). Blue triangles show current stations; the black outline marks the Midland Basin.
<img width="900" height="521" alt="image" src="https://github.com/user-attachments/assets/53a6fec4-0de5-4c7e-8e01-c358656033da" />



