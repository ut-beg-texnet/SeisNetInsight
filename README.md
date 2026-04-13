# SeisNetInsight

SeisNetInsight is an open-source Python tool for identifying where additional seismic stations could provide the greatest monitoring benefit. It combines source-station proximity, azimuthal-gap coverage, and an optional generic contextual  layer into insightful contour maps and a priority map that can be reviewed in the app or exported as figures and KMZ overlays.

## What The App Does

- Loads earthquake catalogs and station catalogs from CSV.
- Accepts an optional contextual point layer with `latitude`, `longitude`, and `value`.
- Supports context aggregation with `sum`, `average`, `count`, `min`, or `max`.
- Supports optional BNA overlays and optional BallTree event reduction.
- Computes primary and secondary source-station distance, ΔGap, context, and composite-priority grids.
- Renders contour maps and a K-means priority map, then exports PNG and KMZ outputs.

## Installation

SeisNetInsight requires Python `3.9` or newer.

It is recommended to install it in a separate Python environment, for example with `venv` or `conda`.

```bash
git clone https://github.com/ut-beg-texnet/SeisNetInsight
cd SeisNetInsight
python -m pip install .
```

## Run The App

```bash
seisnetinsight-app
```

## UI Overview
<!-- <img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/4b028db1-1b23-41fe-969b-de0d26c31011" /> -->



![SeisNetInsight animated UI overview](https://github.com/user-attachments/assets/358e7c7b-75b3-4b87-8a9c-7f8350d09706)

### Export the Maps to Google Earth

<img width="1898" height="965" alt="earth_web" src="https://github.com/user-attachments/assets/59b3c58f-eb04-486a-8253-76148a22bbcb" />

The app is organized into three sections:

1. `Data Loading`
   Upload events, stations, optional context CSV, and optional BNA files.
   Map incoming columns before loading, set context aggregation and radius, and optionally apply BallTree event reduction.
2. `Grids Computation`
   Compute primary and secondary source-station distance, ΔGap, context, and composite grids with progress bars and stop controls.
3. `Maps`
   Preview contour outputs and the priority-clustering map, then download PNG or KMZ products.

## Method Figures

The core method remains grounded in source-to-station distance and azimuthal coverage:

<img width="5141" height="1535" alt="image" src="https://github.com/user-attachments/assets/5dbc6389-419b-4af7-8b1e-6284705f48fc" />

The end-to-end workflow is summarized here:
<img width="1969" height="829" alt="image" src="https://github.com/user-attachments/assets/8136ad2b-d4d3-43b8-9282-64bd9cb355ab" />

## Input Data

Event and station inputs can be provided either as CSV files with the columns described below or as FDSNWS event/station files in text format.

### Required event columns

- `latitude`
- `longitude`
- `origin_time`

### Required station columns

- `latitude`
- `longitude`

### Optional context columns

- `latitude`
- `longitude`
- `value`

Column names can be mapped in the UI, and the loaders also recognize common aliases. The main workflows rely on event location and origin time.

## Programmatic Usage

### Example: bundled sample workflow without context

```python
from pathlib import Path

from seisnetinsight import (
    GridParameters,
    compute_composite_index,
    compute_gap_grid,
    compute_subject_grids,
    generate_grid,
    load_events,
    load_stations,
    merge_grids,
)

root = Path("sample_files")
params = GridParameters()

events, events_missing = load_events(root / "events_sample.csv", warn=False)
stations, stations_missing = load_stations(root / "texnet_stations_2025.csv", warn=False)

if events_missing or stations_missing:
    raise RuntimeError((events_missing, stations_missing))

grid = generate_grid(params)
subject = compute_subject_grids(events, stations, grid, params)
gap = compute_gap_grid(events, stations, grid, params)
merged = merge_grids(subject, gap)
composite = compute_composite_index(merged, params)

print(composite[["latitude", "longitude", "composite_index"]].head())
```

### Example: bundled sample workflow with a generic context layer

```python
from pathlib import Path

from seisnetinsight import (
    GridParameters,
    compute_composite_index,
    compute_context_grid,
    compute_gap_grid,
    compute_subject_grids,
    generate_grid,
    load_context_points,
    load_events,
    load_stations,
    merge_grids,
)

root = Path("sample_files")
params = GridParameters()
params.context_radius_km = 25.0
params.context_aggregation = "sum"
params.weight_context = 0.2

context_column_map = {
    "latitude": "latitude_wgs84",
    "longitude": "longitude_wgs84",
    "value": "SUM_injected_liquid_BBL",
}

events, _ = load_events(root / "events_sample.csv", warn=False)
stations, _ = load_stations(root / "texnet_stations_2025.csv", warn=False)
context, context_missing = load_context_points(
    root / "swd_wells_sample.csv",
    column_map=context_column_map,
    warn=False,
)

if context_missing:
    raise RuntimeError(context_missing)

grid = generate_grid(params)
subject = compute_subject_grids(events, stations, grid, params)
gap = compute_gap_grid(events, stations, grid, params)
context_grid = compute_context_grid(context, grid, params)
merged = merge_grids(subject, gap, context_grid)
composite = compute_composite_index(merged, params)

print(composite[["latitude", "longitude", "context_value", "composite_index"]].head())
```

## Output Examples

Panels (a-c) show how adding a new station would affect nearby events. Brighter colors mean that placing a station in that grid cell would impact more recent events. (a) shows how many recency-weighted events would fall within 4 km of a station. (b) shows the same for a 10 km radius. (c) shows how many events would have their maximum azimuthal gap reduced to <=90 deg if a station were placed there. (d) shows the total salt-water-disposal volume within 25 km (in barrels). The black outline marks the Midland Basin:

<img width="520" height="500" alt="output example panels" src="https://github.com/user-attachments/assets/3f67335f-5525-4fbe-b931-56f7d7fa2c0d" />

(a) Map of a composite impact index (ranging from 0 to 1) that combines four metrics-S4, S10, ΔGap <= 90 deg, and SWD after adjusting for scaling, and weighting. Brighter colors highlight areas where a new station would have the most combined benefit. (b) Priority areas determined by k-means clustering of these metrics, ranked by their average composite index (Very High, High, Medium, Low). Blue triangles show current stations; the black outline marks the Midland Basin.

<img width="900" height="521" alt="composite and priority example" src="https://github.com/user-attachments/assets/53a6fec4-0de5-4c7e-8e01-c358656033da" />
