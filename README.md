# SeisNetInsight

## Published paper

Siervo, D., Breton, C., Huang, G.-c. D., and Savvaidis, A. (2026).
[Data-Driven, Automated Mapping of High-Impact Regions for Seismic Station Deployment](https://doi.org/10.1785/0120250291).
*Bulletin of the Seismological Society of America*. DOI: **10.1785/0120250291**.

## Map colors and event ordering

Feature and composite maps use `cmcrameri.batlowW_r`: low values are light and high values are dark, matching the published paper. PNG maps and KMZ overlays use the same palette. Categorical priority colors are unchanged.

When enabling BallTree reduction, select **Most recent events first** (the default for new app sessions) or **Input file order**. The selected ordering and radius are saved with the session. Existing sessions without ordering metadata retain input-order behavior. Programmatic callers can pass `keep="most_recent"` explicitly; the helper's legacy default remains `first`.

For weighted clustering, use `context_value` as the contextual feature weight key. Legacy `swd_volume_25km_bbl` keys are supported as aliases; conflicting aliases or unknown weight keys raise an error rather than silently changing the weighting.

SeisNetInsight is an open-source Python tool for identifying high-impact regions for seismic station deployment. It combines source-station proximity, azimuthal-gap coverage, and an optional generic contextual  layer into insightful contour maps and a priority map that can be reviewed in the app or exported as figures and KMZ overlays.

## What The App Does

- Loads earthquake catalogs and station catalogs from CSV.
- Accepts an optional contextual layer with `latitude`, `longitude`, and a `value`. A spatially referenced numeric variable indicating where denser monitoring is more relevant, such as SWD volume, fault proximity, GNSS deformation, geothermal indicators or activity, population density, and so on.
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

Event and station inputs can be provided as CSV files with the columns described below.

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

### Example: Sample workflow with a context layer

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

## Output Maps

Feature and composite maps use a light-to-dark `batlowW_r` palette: darker cells have higher values. The composite combines normalized features with the selected weights. The categorical priority map uses four colors to identify clusters ranked by their mean composite score; blue triangles represent stations.

The sample notebook demonstrates the generic API, with its own illustrative parameters. Input files, parameter values and implementation hashes are included in its cache keys so changes trigger recomputation. Its settings are not a paper preset.
