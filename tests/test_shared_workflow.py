"""Verify the Streamlit run uses the CLI's analysis and export contract."""

import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from streamlit.testing.v1 import AppTest

from seisnetinsight import sessions, streamlit_app as ui, workflow
from seisnetinsight.cli import main
from seisnetinsight.config import GridParameters
from seisnetinsight.legacy_maps import LegacyMapConfig, render_legacy_contour, render_priority_clusters


SAMPLE = Path(__file__).resolve().parents[1] / "sample_files"


def test_streamlit_run_and_bundle_match_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sessions, "SESSION_ROOT", tmp_path / "sessions")
    monkeypatch.setattr(ui, "_ensure_cartopy_map_data", lambda _: True)
    # Leave the canonical renderer untouched. Only the interactive preview
    # uses these replacements, so the test can run without basemap downloads.
    preview = LegacyMapConfig(show_basemap=False, show_roads=False, show_states=False, show_cities=False)

    def offline(renderer):
        def render(*args, **kwargs):
            return renderer(*args, **{**kwargs, "config": preview})
        return render

    monkeypatch.setattr(ui, "render_legacy_contour", offline(render_legacy_contour))
    monkeypatch.setattr(ui, "render_priority_clusters", offline(render_priority_clusters))
    config = json.loads((SAMPLE / "analysis.json").read_text())
    config["maps"].pop("dpi")  # Match the app's canonical export settings.
    config["inputs"]["context"].pop("units")  # This is optional CLI annotation.
    config["balltree"] = {"distance_km": 0.5, "keep": "most_recent"}
    config["priority"] = {"n_clusters": 3, "n_init": 10, "apply_feature_weights": False}
    uploader_labels = {"events": "Events file", "stations": "Stations file", "context": "Context file (optional)"}
    uploaded = {}
    for name, label in uploader_labels.items():
        item = config["inputs"][name]
        item["path"] = str(SAMPLE / item["path"])
        uploaded[label] = io.BytesIO(Path(item["path"]).read_bytes())
    boundary = Path(SAMPLE / config["inputs"]["boundary"])
    config["inputs"]["boundary"] = str(boundary)
    overlay = io.BytesIO(boundary.read_bytes())
    overlay.name = boundary.name
    uploaded["BNA files (optional)"] = [overlay]
    monkeypatch.setattr(ui.st, "file_uploader", lambda label, **kwargs: uploaded[label])

    config_path = tmp_path / "cli.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "cli"
    assert main(["run", "--config", str(config_path), "--output-dir", str(output_dir), "--json"]) == 0
    cli = json.loads(capsys.readouterr().out)

    at = AppTest.from_string("from seisnetinsight.streamlit_app import main\nmain()", default_timeout=60).run()
    assert not at.exception
    at.text_input(key=ui.FORM_SESSION_NAME_KEY).set_value("parity")
    at.text_input(key=ui.FORM_LONS_KEY).set_value("-103,-100.5")
    at.number_input(key=ui.FORM_GRID_STEP_KEY).set_value(0.5)
    at.checkbox(key=ui.FORM_BALLTREE_ENABLED_KEY).check()
    at.number_input(key=ui.FORM_BALLTREE_DISTANCE_KEY).set_value(0.5)
    for role, column in config["inputs"]["context"]["columns"].items():
        at.selectbox(key=f"context_column_{role}").select(column)

    def click(label):
        next(button for button in at.button if button.label == label).click().run()
        assert not at.exception
        assert not at.error

    click("Save session, load data, and run all")
    at.slider(key="priority_kmeans_cluster_count").set_value(3)
    at.slider(key="priority_kmeans_n_init").set_value(10)
    at.checkbox(key="priority_apply_feature_weights").uncheck()
    click("Prepare analysis bundle")
    session = at.session_state[ui.SESSION_KEY]
    with ZipFile(io.BytesIO(session.bundle[1])) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        baseline = json.loads(Path(cli["manifest"]).read_text())
        for key in baseline.keys() - {"created_at_utc", "config", "source"}:
            assert manifest[key] == baseline[key], key
        for key in baseline["config"].keys() - {"inputs"}:
            assert manifest["config"][key] == baseline["config"][key], key
        for key, source in baseline["source"]["inputs"].items():
            assert manifest["source"]["inputs"][key]["sha256"] == source["sha256"]
        for key, record in manifest["artifacts"].items():
            assert archive.read(record["path"]) == Path(cli["artifacts"][key]).read_bytes(), key

    restored = ui._load_session_from_disk("parity")
    assert restored.prepared.counts == session.prepared.counts
    assert len(restored.events) == len(session.events)
    assert set(restored.grids) == {"subject", "gap", "context", "composite"}
    assert not restored.legacy_grids

    # Changing a coverage setting invalidates the affected feature and
    # composite while the independently computed grids remain available.
    at.number_input(key=ui.FORM_PRIMARY_RADIUS_KEY).set_value(5.0).run()
    assert not at.exception
    changed = at.session_state[ui.SESSION_KEY]
    assert set(changed.grids) == {"gap", "context"}
    assert changed.bundle is None


def test_invalid_upload_and_cli_input_share_validation(tmp_path):
    events = b"latitude,longitude,origin_time\n320,-102,2025-01-01\n"
    stations = b"latitude,longitude\n32.1,-102.1\n"
    (tmp_path / "events.csv").write_bytes(events)
    (tmp_path / "stations.csv").write_bytes(stations)
    config = tmp_path / "analysis.json"
    config.write_text(json.dumps({
        "schema_version": 1,
        "inputs": {"events": {"path": "events.csv"}, "stations": {"path": "stations.csv"}},
    }))
    with pytest.raises(workflow.WorkflowError) as cli:
        workflow.validate(config)
    with pytest.raises(workflow.WorkflowError) as upload:
        ui.build_working_session(
            session_name="invalid", parameters=GridParameters(), events_bytes=events, stations_bytes=stations,
        )
    assert upload.value.code == cli.value.code == "invalid_input"
    assert str(upload.value) == str(cli.value)
