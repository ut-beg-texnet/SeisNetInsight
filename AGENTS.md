# Working with SeisNetInsight

Use the headless CLI for analyses. The [README](README.md#headless-command-line) documents the version 1 config and JSON results, and [sample_files/analysis.json](sample_files/analysis.json) runs offline with the bundled data.

## Talk with the user about the analysis

Use plain language and standard seismology terms. Say earthquake catalog, origin time in UTC, station coverage, and azimuthal gap. Refer to the context layer by what it measures, such as saltwater disposal volume, and give its units. Explain a term briefly if the user is unfamiliar with it. Keep config keys and internal column names in commands or files unless they help the user make a choice.

Use the user's stated inputs and settings. For a reproducibility run, look first for an existing config or run manifest. Ask only about missing decisions that affect the result. Give two or three useful options, recommend one when the goal supports it, and explain the tradeoff in a sentence. If the user wants to try the tool, run the bundled sample. For a new study, ask for the study area and UTC event period if they are missing, and confirm the station snapshot date. For a finer grid, explain the tradeoff between spatial detail and computation time.

Group related questions. Discuss only the settings that matter to the task, such as minimum magnitude, catalog quality criteria, station coverage distances, the azimuthal gap threshold, event recency half-life, or the relative weight of the features. When offering a software default, give its value and units and explain what it does. For example, ask whether to use the default 4 km and 10 km coverage distances or distances used by the user's network. Treat a change in catalog, quality filter, spatial event reduction, weighting, or time window as a scientific choice. Ask before making such a change unless the user already authorized it.

## Offer sources for missing data

- **Earthquakes.** Offer the user's catalog, a download from the [USGS earthquake catalog service](https://earthquake.usgs.gov/fdsnws/event/1/), or a specified regional network catalog. Recommend the regional catalog when local detection, review status, or reproducibility depends on it. For a USGS download, use the agreed bounds, UTC time window and magnitude threshold if specified. Map CSV `time` to origin time and keep the service event IDs. Check the service count and split large requests if needed. Verify completeness and handle duplicate IDs at query boundaries. Catalogs may differ in event completeness and review criteria.
- **Stations.** Offer to use the supplied station list or retrieve an inventory from a network or service the user identifies. Confirm which station epochs were active on the requested date and retain network and station identifiers when available. Record which date and network selection the analysis uses.
- **Context.** Offer to use a supplied file, download the dataset from a URL or service the user provides, or run without a context layer. Establish what the values measure, the units, and the period covered. Match longitude, latitude, and the numeric value column to the config, and ask about the aggregation rule if it affects interpretation. Use supplied recency-weighted values as supplied. If conversion from a raster or polygon dataset requires choosing how to represent the values as points, agree on that conversion first.

For example, ask “Should I use your earthquake catalog or download the USGS catalog for the study area and dates?” or “Should the map include saltwater disposal volume from your link, a different context dataset, or station coverage and azimuthal gap alone?”

For downloads, keep the original response and record the source URL or query, retrieval time, coverage, units, and any conversion or filter. Check the data before using it. Report what was retained after date, magnitude, or quality filters and spatial event reduction. Catalog quality filtering and BallTree reduction are separate steps. Describe BallTree as reducing nearby events by distance and retaining one according to the chosen ordering. Keep input files and generated runs out of commits unless the user asks to publish them.

## Ask what to show on the maps

If the user has not specified map annotations, ask “Would you like any boundaries or city labels on the maps?” Offer useful overlays for the area, such as a study boundary, basin outline, or nearby cities, and let the user choose which to include.

- **Polygons.** Ask for a boundary file or a source URL. Use a BNA file through `inputs.boundary` in the CLI or the BNA upload in Streamlit. If the file is in another format, check its coordinate reference system before converting it. Explain that the BNA overlay draws an outline. Ask separately if the boundary should limit the analysis.
- **Cities.** Ask which places to label. The existing renderer includes a fixed set of West Texas city labels for the Streamlit previews. For other cities, verify their names and coordinates from the user or an appropriate source. The CLI has no city-label setting, so offer a separately rendered map with the requested labels if needed.

Record the overlay source and any conversion or separate rendering step when sharing the maps.

## Run and report

1. Run `seisnetinsight validate --config <config.json> --json` and inspect the resolved settings, row counts, and warnings.
2. Run `seisnetinsight run --config <config.json> --output-dir <new-dir> --json`.
3. Use the returned artifact paths and run manifest when reporting results. Use `seisnetinsight render --manifest <manifest.json> --output-dir <new-dir> --json` to redraw the saved grids.

Summarize the study area, earthquake catalog and UTC interval, station snapshot, context data and units, and the settings that affect the result. Link the maps and manifest. Describe priority regions as rankings for seismic monitoring under the chosen inputs and weights.

After sharing the PNGs, ask “Would you also like KMZ overlays so you can pan and zoom through the maps in Google Earth?” If the user already requested KMZ files, include them with the PNGs. Use the existing Streamlit KMZ downloads or its `_legacy_overlay_kmz` and `_priority_overlay_kmz` exporters with the saved grids and run settings. Share the resulting files without recomputing the analysis.

Reuse the functions in `grids.py`, `maps.py`, and `legacy_maps.py` for analysis and rendering. Run `python -m pytest -q` for code changes.
