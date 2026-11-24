# SeisNetInsight

SeisNetInsight is an open-source Python tool for identifying high-impact regions for seismic station deployment. It combines source-station proximity, azimuthal-gap coverage, and an optional context layer to produce feature, composite, and priority maps.

**Using an AI agent?** Copy this prompt into your agent. The [headless quickstart](#headless-command-line) has the commands it will use.

```text
Set up https://github.com/ut-beg-texnet/SeisNetInsight in this workspace. Reuse the checkout if it is already here. Install it in a project-local Python environment, then follow the README's headless quickstart to validate and run the bundled sample with JSON output. Report the run manifest and map paths.
```

## Published paper

Siervo, D., Breton, C., Huang, G.-c. D., and Savvaidis, A. (2026).
[Data-Driven, Automated Mapping of High-Impact Regions for Seismic Station Deployment](https://doi.org/10.1785/0120250291).
*Bulletin of the Seismological Society of America*. DOI: **10.1785/0120250291**.

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
   Map incoming columns before loading, set context aggregation and radius, and optionally apply BallTree event reduction. When enabled, the app processes the most recent events first by default. Select **Input file order** to use the catalog order.
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

The app and CLI share input validation, grid calculations, priority classification, and the analysis export. After computing the grids in the app, use **Prepare analysis bundle** in the priority map section, then **Download analysis bundle**. The ZIP contains the same manifest, grids, and canonical PNGs as the CLI with matching settings. The app's preview decorations and KMZ downloads are available separately. Run provenance identifies the source files and time of each run.

Saved app sessions retain the original uploads for reproducibility. For older sessions that only saved processed inputs, recompute the grids before preparing a bundle. The manifest identifies the use of the saved processed data.

### Headless command line

From a checkout installed with `python -m pip install .`, run the bundled sample:

```sh
seisnetinsight validate --config sample_files/analysis.json --json
seisnetinsight run --config sample_files/analysis.json --output-dir sample-run --json
seisnetinsight render --manifest sample-run/manifest.json --output-dir sample-redraw --json
```

`python -m seisnetinsight` is equivalent to the `seisnetinsight` executable. The sample uses a coarse grid and runs headless without downloading basemap data. Both output directories must be new. `render` uses the saved grids, station coordinates, and optional boundary from the analysis directory. It checks their hashes before rendering and can run after the original input files have moved.

With `--json`, stdout contains one JSON object. Successful responses contain `schema_version`, `ok: true`, and `command`. Validation returns resolved `config`, input `source`, row `counts`, `normalized_weights`, and `warnings`. Run and render return the absolute `manifest` path, an `artifacts` mapping from artifact name to absolute path, `counts`, and `warnings`. Errors return `ok: false` and `error` with a `code` and `message`. Read progress and library diagnostics from stderr. Exit statuses are 0 for success, 2 for usage, configuration, input, manifest, or existing-output errors, 1 for execution failures, and 130 for interruption.

The analysis directory contains the feature and composite grid CSV, priority grid CSV, and each map as a PNG. The analysis manifest records the resolved configuration, input paths and SHA-256 hashes, filter counts, recency reference times, effective weights, package and dependency versions, a source fingerprint, and SHA-256 hashes of outputs. Artifact paths **inside the manifest** are relative to its directory. A render manifest also identifies the source analysis manifest and its hash.

#### Analysis configuration, version 1

Copy [the sample config](sample_files/analysis.json) to start a new analysis. Required top-level fields are `schema_version: 1` and `inputs`. Paths are relative to the config file unless absolute.

| Config field | Supported settings |
| --- | --- |
| `inputs.events`, `inputs.stations` | Each requires `path`. An optional `columns` object maps canonical names to input headers. Use `latitude`, `longitude`, and, for events, `origin_time`. Coordinates are WGS84 decimal degrees. |
| `inputs.context` | Optional local point table with `path`, optional `columns` for `latitude`, `longitude`, and `value`, and optional `units` for provenance. Context values are aggregated as supplied. Map the already weighted column if the source contains precomputed weights. |
| `inputs.boundary` | Optional path to a BNA file. |
| `event_window` | Optional `start_inclusive` and `end_exclusive` as ISO dates or timestamps. Dates and timestamps without an offset are interpreted as UTC. The CLI uses the event catalog and station snapshot you supply. Apply any project-specific catalog quality or station activity filters before passing the files in. |
| `balltree` | Optional event reduction after the date filter. Set positive `distance_km` and `keep` to `most_recent` (default) or `first` (input order). |
| `grid` | `lats` and `lons` are increasing pairs. `grid_step` is in degrees. Other accepted keys are `subject_primary_radius_km`, `subject_primary_min_stations`, `subject_primary_weight`, `subject_secondary_radius_km`, `subject_secondary_min_stations`, `subject_secondary_weight`, `gap_search_km`, `gap_target_angle_deg`, `weight_gap`, `context_radius_km`, `weight_context`, `half_time_years`, and `context_aggregation` (`sum`, `average`, `count`, `min`, or `max`). Omitted settings use library defaults. Without context, the effective context weight is zero and the context PNG is omitted. |
| `max_grid_cells` | Limits the grid size before computation. Default 1,000,000. |
| `priority` | `n_clusters` (4), `random_state` (42), `n_init` (20), and `apply_feature_weights` (true). |
| `maps` | `dpi` (150) and `basemap` (false). Enable the basemap for Natural Earth layers. Cartopy may download map data on first use. |

The core gap calculation uses events inside the configured bounds and excludes AM stations when a network column is present. Coverage can use events and stations outside the bounds. The manifest records the corresponding inputs and counts.

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

When passing feature weights to `classify_priority_clusters`, use `context_value` as the key for the context layer.

## Output Maps

Feature and composite maps use a light-to-dark `batlowW_r` palette: darker cells have higher values. The composite combines normalized features with the selected weights. The categorical priority map uses four colors to identify clusters ranked by their mean composite score; blue triangles represent stations.

The sample notebook demonstrates the generic API, with its own illustrative parameters. Input files, parameter values and implementation hashes are included in its cache keys so changes trigger recomputation. Its settings are not a paper preset.
