# Working with SeisNetInsight

Use the headless CLI for analyses. The [README](README.md#headless-command-line) documents the version 1 config and JSON results, and [sample_files/analysis.json](sample_files/analysis.json) runs offline with the bundled data.

1. Run `seisnetinsight validate --config <config.json> --json` and inspect the resolved settings, row counts, and warnings.
2. Run `seisnetinsight run --config <config.json> --output-dir <new-dir> --json`.
3. Use the returned artifact paths and run manifest when reporting results. Use `seisnetinsight render --manifest <manifest.json> --output-dir <new-dir> --json` to redraw the saved grids.

Record the source data, UTC date bounds, context column and units, and any preprocessing. Reuse the functions in `grids.py`, `maps.py`, and `legacy_maps.py` for analysis and rendering. Run `python -m pytest -q` before submitting changes.
