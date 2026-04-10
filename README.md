# SeisNetInsight

SeisNetInsight is an open-source Python package and Streamlit app for identifying where additional seismic stations would provide the greatest monitoring benefit. It combines seismic-event proximity, azimuthal-gap improvement, and an optional generic contextual point layer into gridded priority maps that can be reviewed in the app or exported as figures and KMZ overlays.

The launch surface in this repository is:

- the installed `seisnetinsight-app` Streamlit interface
- the public Python APIs in `seisnetinsight`
- saved-session restore and re-run workflows
- map export and clustering outputs

## What The App Does

- Loads earthquake catalogs and station catalogs from CSV.
- Accepts an optional contextual point layer with `latitude`, `longitude`, and `value`.
- Supports context aggregation with `sum`, `average`, `count`, `min`, or `max`.
- Supports optional BNA overlays and optional BallTree event reduction.
- Computes subject-primary, subject-secondary, Delta Gap90, context, and composite-priority grids.
- Renders contour maps and a K-means priority map, then exports PNG and KMZ outputs.
- Preserves backward compatibility with the legacy SWD-specific APIs:
  `load_swd_wells(...)` and `compute_swd_grid(...)`.

## Installation

Install the package and app entrypoint:

```bash
python -m pip install .
```

Install test tooling:

```bash
python -m pip install ".[test]"
```

## Run The App

```bash
seisnetinsight-app
```

The launcher now sets a writable `MPLCONFIGDIR` automatically before Streamlit imports Matplotlib, which avoids the common first-run cache warning in locked-down environments.

## UI Overview

![SeisNetInsight UI overview](figures/ui_overview.svg)

The app is organized into three sections:

1. `Data Loading`
   Upload events, stations, optional context CSV, and optional BNA files.
   Map incoming columns before loading, set context aggregation and radius, and optionally apply BallTree event reduction.
2. `Grids Computation`
   Compute subject, Delta Gap90, context, and composite grids with progress bars and stop controls.
3. `Maps`
   Preview contour outputs and the priority-clustering map, then download PNG or KMZ products.

## Method Figures

The core method remains grounded in source-to-station distance and azimuthal coverage:

![Azimuthal gap and distance illustration](figures/az_gap_and_dist_sta_illustration.png)

The end-to-end workflow is summarized here:

![Workflow figure](figures/Flowchart%20for%20Poster%20SSA.png)

## Input Data

### Required event columns

- `latitude`
- `longitude`
- `magnitude`
- `origin_time`

### Required station columns

- `latitude`
- `longitude`

### Optional context columns

- `latitude`
- `longitude`
- `value`

Column names can be mapped in the UI, and the loaders also recognize common aliases. The package now accepts official FDSN text responses directly in addition to CSV, which makes external-network validation easier.

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

### Legacy SWD compatibility

Existing notebooks or scripts that still use the SWD-specific names continue to work:

```python
from seisnetinsight import compute_swd_grid, load_swd_wells
```

Internally these compatibility wrappers now call the generic context pipeline.

## External Seismic-Network Validation

Bounded upstream samples for manual launch checks are versioned in [tests/fixtures/external](tests/fixtures/external). They come from official services:

- SCEDC FDSN event and station services
- NCEDC FDSN event and station services
- GeoNet FDSN event and station services
- GEOFON FDSN event and station services
- Natural Earth populated places for the optional context layer

### Refresh the external fixtures

```bash
python scripts/fetch_external_validation_data.py
```

### Run the bundled external validation workflows

```bash
python scripts/validate_external_workflows.py
```

That script runs bounded end-to-end workflows for SCEDC, NCEDC, and GeoNet with and without context. It also checks that the bundled GEOFON event and station fixtures still parse cleanly, since the GEOFON samples are intentionally sourced from different geographic slices for stretch validation.

### Fetch live FDSN samples manually

The loaders accept the official pipe-delimited FDSN text format directly. For example:

```bash
curl -L "https://service.scedc.caltech.edu/fdsnws/event/1/query?format=text&starttime=2024-01-01&endtime=2024-01-02&minlatitude=33&maxlatitude=35&minlongitude=-119&maxlongitude=-117&limit=3" -o scedc_events.txt
curl -L "https://service.scedc.caltech.edu/fdsnws/station/1/query?format=text&level=station&minlatitude=33.70&maxlatitude=33.90&minlongitude=-118.60&maxlongitude=-118.20" -o scedc_stations.txt
```

You can load those files directly:

```python
from seisnetinsight import load_events, load_stations

events, _ = load_events("scedc_events.txt", warn=False)
stations, _ = load_stations("scedc_stations.txt", warn=False)
```

## Tests And Launch Checks

Run the automated tests:

```bash
python -m pytest -q
```

Benchmark the bundled sample workflow against the current launch budgets:

```bash
python scripts/benchmark_sample_workflow.py
```

That script fails if:

- the full sample workflow exceeds `45 s`
- any one of `compute_subject_grids`, `compute_gap_grid`, or `compute_context_grid` exceeds `20 s`

Run a fresh-install smoke test in an isolated virtual environment:

```bash
python scripts/smoke_local_install.py
```

The smoke script installs the package into a temporary environment, launches `seisnetinsight-app`, waits for the Streamlit health endpoint, and fails if the old Matplotlib cache warnings appear.

## Repository Notes

- [sample_files](sample_files) contains the bundled example datasets.
- [figures](figures) contains the tracked documentation figures used in this README.
- [tests](tests) contains automated loader, grid, session, Streamlit, and external-validation tests.
- [scripts](scripts) contains manual launch-prep utilities for fixture refresh, validation, benchmarking, and smoke testing.

## Current Validation Coverage

The launch-readiness work in this branch includes:

- automated tests for loaders, aliasing, malformed timestamps, missing columns, duplicate rows, session persistence, backward compatibility, aggregation modes, interruption, and Streamlit shell/workflow behavior
- fixture-based validation for external seismic networks
- a generic context layer in the API and UI
- removal of the unused pydeck export path and other dead code from the older app flow
