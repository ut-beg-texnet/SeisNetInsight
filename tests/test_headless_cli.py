"""Exercise the documented sample and the machine-readable command contract."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from seisnetinsight.cli import main


ROOT = Path(__file__).resolve().parents[1]


def invoke(capsys, *args, status=0):
    assert main([*map(str, args), "--json"]) == status
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    result = json.loads(captured.out)
    assert result["schema_version"] == 1
    assert result["ok"] is (status == 0)
    return result


def test_sample_validate_run_render(tmp_path, monkeypatch, capsys):
    from cartopy.io import Downloader

    def unexpected_download(*args, **kwargs):
        raise AssertionError("The sample must run offline.")

    monkeypatch.setattr(Downloader, "acquire_resource", unexpected_download)
    sample = tmp_path / "sample"
    shutil.copytree(ROOT / "sample_files", sample)
    config = sample / "analysis.json"
    run_dir, redraw_dir = tmp_path / "run", tmp_path / "redraw"
    monkeypatch.chdir(tmp_path)

    validation = invoke(capsys, "validate", "--config", config)
    assert validation["config"]["grid_cells"] == 48
    assert validation["counts"]["events_after_balltree"] == validation["counts"]["events_input"]
    assert validation["counts"]["map_stations_in_bounds"] > 0

    run = invoke(capsys, "run", "--config", config, "--output-dir", run_dir)
    manifest_path = Path(run["manifest"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["kind"] == "analysis" and manifest["config"] == validation["config"]
    assert run["counts"]["grid_cells"] == 48
    assert sum(run["counts"]["priority_labels"].values()) == 48
    assert len(run["counts"]["priority_labels"]) == 4
    assert manifest["source"]["inputs"]["context"]["sha256"] == hashlib.sha256((sample / "swd_wells_sample.csv").read_bytes()).hexdigest()
    assert len(manifest["runtime"]["source_sha256"]) == 64

    composite = pd.read_csv(run["artifacts"]["feature_composite_grid"])
    priorities = pd.read_csv(run["artifacts"]["priority_grid"])
    pd.testing.assert_frame_equal(composite, priorities[composite.columns])
    assert composite["composite_index"].between(0, 1).all()
    assert composite["composite_index"].max() > 0.2

    # Independently verify the summed SWD context for a known grid point.
    well_data = pd.read_csv(sample / "swd_wells_sample.csv")
    lat, lon = 32.0, -102.0
    well_lats = np.radians(well_data["latitude_wgs84"])
    a = np.sin((well_lats - np.radians(lat)) / 2)**2
    a += np.cos(well_lats) * np.cos(np.radians(lat)) * np.sin(np.radians(well_data["longitude_wgs84"] - lon) / 2)**2
    distances = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    expected_context = well_data.loc[distances <= 25, "SUM_injected_liquid_BBL"].sum()
    assert expected_context > 0
    assert composite.loc[(composite["latitude"] == lat) & (composite["longitude"] == lon), "context_value"].item() == expected_context

    for key, artifact in manifest["artifacts"].items():
        output = Path(run["artifacts"][key])
        assert (manifest_path.parent / artifact["path"]) == output
        assert hashlib.sha256(output.read_bytes()).hexdigest() == artifact["sha256"]
        if output.suffix == ".png":
            assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(list(run_dir.glob("*.png"))) == 6
    assert invoke(capsys, "run", "--config", config, "--output-dir", run_dir, status=2)["error"]["code"] == "output_exists"

    # Redraw from the run directory alone and reproduce every PNG.
    sample.rename(tmp_path / "moved-inputs")
    redraw = invoke(capsys, "render", "--manifest", manifest_path, "--output-dir", redraw_dir)
    for key, path in redraw["artifacts"].items():
        assert Path(path).read_bytes() == Path(run["artifacts"][key]).read_bytes()
    render_manifest = json.loads(Path(redraw["manifest"]).read_text())
    assert render_manifest["source_run"]["sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    grid_path = Path(run["artifacts"]["feature_composite_grid"])
    with grid_path.open("ab") as grid:
        grid.write(b"\n")
    rejected = tmp_path / "bad-render"
    error = invoke(capsys, "render", "--manifest", manifest_path, "--output-dir", rejected, status=2)["error"]
    assert error["code"] == "invalid_manifest" and "Saved artifact changed" in error["message"]
    assert not rejected.exists()


def test_window_balltree_and_actionable_validation(tmp_path, capsys):
    config = tmp_path / "analysis.json"
    catalog = tmp_path / "events.csv"
    catalog.write_text(
        "latitude,longitude,origin_time\n"
        "32,-102,2025-01-01\n"
        "32,-102,2024-12-31T20:00:00-05:00\n"
        "32.1,-102,2025-01-02T00:00:00Z\n",
        encoding="utf-8",
    )
    (tmp_path / "stations.csv").write_text("latitude,longitude\n32.1,-102.1\n", encoding="utf-8")
    settings = {
        "schema_version": 1,
        "inputs": {"events": {"path": "events.csv"}, "stations": {"path": "stations.csv"}},
        "grid": {"grid_step": 0.5},
        "event_window": {"start_inclusive": "2025-01-01", "end_exclusive": "2025-01-02"},
        "balltree": {"distance_km": 0.5, "keep": "most_recent"},
    }
    config.write_text(json.dumps(settings))
    result = invoke(capsys, "validate", "--config", config)
    assert result["counts"]["events_after_window"] == 2
    assert result["counts"]["events_after_balltree"] == 1
    assert result["normalized_weights"]["context"] == 0
    assert result["warnings"]

    settings["inputs"]["events"]["columns"] = {"latitude": "misspelled"}
    config.write_text(json.dumps(settings))
    output = tmp_path / "rejected"
    error = invoke(capsys, "run", "--config", config, "--output-dir", output, status=2)["error"]
    assert error["code"] == "invalid_input" and "misspelled" in error["message"]
    assert not output.exists()

    settings["grid"]["grid_step"] = 0.00001
    config.write_text(json.dumps(settings))
    error = invoke(capsys, "validate", "--config", config, status=2)["error"]
    assert error["code"] == "invalid_config" and "max_grid_cells" in error["message"]
    assert invoke(capsys, "validate", status=2)["error"]["code"] == "usage_error"
