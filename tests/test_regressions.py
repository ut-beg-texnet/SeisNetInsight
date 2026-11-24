import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from cmcrameri import cm as cmc
from threadpoolctl import threadpool_limits

from seisnetinsight.config import GridParameters
from seisnetinsight.legacy_maps import LegacyMapConfig, render_legacy_contour, render_priority_clusters
from seisnetinsight.maps import DEFAULT_PRIORITY_FEATURES, classify_priority_clusters
from seisnetinsight import sessions
from seisnetinsight.streamlit_app import build_working_session, _save_sources, _load_session_from_disk, _legacy_overlay_kmz


def test_weight_aliases_and_conflicts():
    frame = pd.DataFrame(np.random.default_rng(12).random((100, 4)), columns=DEFAULT_PRIORITY_FEATURES)
    frame['composite_index'] = frame.mean(axis=1)
    frame['swd_volume_25km_bbl'] = frame.context_value
    current = dict(zip(DEFAULT_PRIORITY_FEATURES, [.3, .3, .2, .2]))
    legacy = dict(current)
    legacy['swd_volume_25km_bbl'] = legacy.pop('context_value')
    original = dict(legacy)
    with threadpool_limits(limits=1):
        a = classify_priority_clusters(frame, feature_weights=current)
        b = classify_priority_clusters(frame, feature_weights=legacy)
        old_features = [*DEFAULT_PRIORITY_FEATURES[:3], 'swd_volume_25km_bbl']
        c = classify_priority_clusters(frame, features=old_features, feature_weights=current)
    assert a.priority_label.equals(b.priority_label)
    assert a.priority_label.equals(c.priority_label)
    assert legacy == original
    with pytest.raises(ValueError, match='Conflicting'):
        classify_priority_clusters(frame, feature_weights={**current, 'swd_volume_25km_bbl': .9})
    with pytest.raises(ValueError, match='Unknown'):
        classify_priority_clusters(frame, feature_weights={'context_typo': .2})


def test_balltree_ordering_and_session_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, 'SESSION_ROOT', tmp_path)
    events = b'latitude,longitude,origin_time\n32,-102,2020-01-01\n32,-102,2026-01-01\n'
    stations = b'latitude,longitude\n32.1,-102.1\n'
    recent = build_working_session(session_name='recent', parameters=GridParameters(), events_bytes=events,
                                   stations_bytes=stations, balltree_enabled=True, balltree_distance=.5)
    assert recent.events.origin_time.iloc[0].year == 2026
    first = build_working_session(session_name='first', parameters=GridParameters(), events_bytes=events,
                                  stations_bytes=stations, balltree_enabled=True, balltree_distance=.5, balltree_keep='first')
    assert first.events.origin_time.iloc[0].year == 2020
    _save_sources(recent, recent.events, recent.stations, None)
    restored = _load_session_from_disk('recent')
    assert restored.balltree_keep == 'most_recent'
    assert restored.balltree_enabled and restored.balltree_distance == .5
    metadata_path = tmp_path / 'recent/metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.pop('upload_files')
    metadata_path.write_text(json.dumps(metadata))
    legacy = _load_session_from_disk('recent')
    assert len(legacy.events) == 1 and legacy.events.origin_time.iloc[0].year == 2026
    assert legacy.prepared.settings['balltree'] is None
    assert legacy.prepared.source['inputs']['events']['path'] is None
    metadata.pop('data_options')
    metadata_path.write_text(json.dumps(metadata))
    assert _load_session_from_disk('recent').balltree_keep == 'first'


def test_palette_and_rendering_compatibility(monkeypatch):
    from seisnetinsight import legacy_maps, streamlit_app
    monkeypatch.setattr(legacy_maps, '_us_county_records', lambda: ())
    params = GridParameters(lats=(31, 32), lons=(-102, -101))
    frame = pd.DataFrame({'latitude':[31,31,32,32], 'longitude':[-102,-101,-102,-101], 'composite_index':[0,.2,.7,1], 'priority_label':['Low','Medium','High','Very High']})
    config = LegacyMapConfig(show_roads=False, show_cities=False, show_counties=True)
    fig = render_legacy_contour(frame, 'composite_index', params, config=config)
    from matplotlib.contour import QuadContourSet
    contours = [item for item in fig.axes[0].collections if isinstance(item, QuadContourSet)]
    assert contours and contours[0].cmap.name == cmc.batlowW_r.name
    plt.close(fig)
    fig = render_priority_clusters(frame, params, config=config, coordinate_ticks=True)
    assert len(fig.axes[0].get_xticks()) > 0
    plt.close(fig)
    captured = {}
    def capture(rgba):
        captured['rgba'] = rgba
        return b'png'
    monkeypatch.setattr(streamlit_app, '_rgba_overlay_png', capture)
    assert _legacy_overlay_kmz(frame, 'composite_index', params)
    assert np.allclose(captured['rgba'][0,0,:3], cmc.batlowW_r(0.)[:3])


def test_notebook_all_cells_and_cache_invalidation(tmp_path, monkeypatch):
    import shutil
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / 'sample_files', tmp_path / 'sample_files')
    work = tmp_path / 'notebooks'
    work.mkdir()
    monkeypatch.chdir(work)
    notebook = json.loads((root / 'notebooks/test_seisnetinsight_api_sample_dataset.ipynb').read_text())
    namespace = {'display': lambda value: plt.close(value)}
    with threadpool_limits(limits=1):
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code': continue
            source = ''.join(line for line in cell['source'] if not line.lstrip().startswith('%'))
            if 'grid = generate_grid(params)' in source:
                source = source.replace('grid = generate_grid(params)', 'params.grid_step = 0.5\ngrid = generate_grid(params)')
            exec(compile(source, '<notebook>', 'exec'), namespace)
    before = namespace['_cache_path']('subject')
    namespace['PARAMS'].half_time_years = 2
    assert namespace['_cache_path']('subject') != before
    after_params = namespace['_cache_path']('subject')
    event_file = Path(namespace['EVENTS_CSV'])
    event_file.write_text(event_file.read_text() + '\n')
    assert namespace['_cache_path']('subject') != after_params
