"""
Unit tests for the CSV writer/provenance/metadata layer (ade_dls/gui/export/csv_export.py).

Per-plot-type table builders were removed in v3.4.0 (superseded by the
generic Figure-based extraction in plot_series_export.py, see
test_plot_series_export.py); this file now only covers the remaining
reusable data layer: _safe_filename, write_tables, register_outputs_in_provenance,
build_export_metadata, write_metadata.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make the repo root importable when running pytest from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ade_dls.gui.export import csv_export


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

def _tables():
    return {
        'demo__panel0__data': pd.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [3.0, 4.0, 5.0]}),
        'demo__panel0__fit': pd.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [3.1, 3.9, 5.1]}),
    }


class _FakeLaplaceAnalyzer:
    """Minimal stand-in mimicking LaplaceAnalyzer attributes used by build_export_metadata."""

    def __init__(self):
        self.df_basedata = pd.DataFrame({
            'filename': ['s1_30.asc', 's2_30.asc', 's3_90.asc'],
            'angle [°]': [30.0, 30.0, 90.0],
            'temperature [K]': [298.15, 298.20, 298.10],
            'viscosity [cp]': [1.002, 1.002, 1.002],
        })
        self.nnls_params = {'prominence': 0.05, 'distance': 10}

    def get_nnls_summary(self):
        return {'n_peaks': 2, 'mean_R2': 0.998}


# ---------------------------------------------------------------------------
# _safe_filename
# ---------------------------------------------------------------------------

def test_safe_filename_strips_invalid_characters():
    assert csv_export._safe_filename('C: sample <file>.ASC') == 'C__sample__file_.ASC'


def test_safe_filename_truncates_to_80_chars():
    assert len(csv_export._safe_filename('x' * 200)) == 80


# ---------------------------------------------------------------------------
# Writer + provenance
# ---------------------------------------------------------------------------

def test_write_tables_roundtrip(tmp_path):
    tables = _tables()
    written = csv_export.write_tables(tables, str(tmp_path), prefix="test_")

    assert len(written) == len(tables)
    for path in written:
        assert os.path.isfile(path)
        assert os.path.basename(path).startswith("test_")
        # comma-separated, dot decimal -> reloadable by pandas default
        reloaded = pd.read_csv(path)
        assert not reloaded.empty


def test_write_tables_raises_on_empty():
    with pytest.raises(ValueError):
        csv_export.write_tables({}, "/tmp/wherever")


def test_provenance_registration_records_hash_path_and_description(tmp_path):
    """register_outputs_in_provenance should add a hashed, path-carrying output entry."""
    from ade_dls.gui.core.provenance import ProvenanceRecord

    class _PanelStub:
        def __init__(self):
            self.record = ProvenanceRecord(version="test")

        def register_output(self, output_type, label, filepath=None, extra_fields=None):
            return self.record.add_output(output_type, label, filepath,
                                          extra_fields=extra_fields)

    tables = _tables()
    written = csv_export.write_tables(tables, str(tmp_path))

    panel = _PanelStub()
    description = {'plot': 'demo', 'method': 'Method C', 'panel': 0, 'series': 'data',
                   'kind': 'line', 'columns': ['x', 'y'], 'n_points': 3}
    for path in written:
        csv_export.register_outputs_in_provenance(
            panel, [path], extra_fields={'description': description})

    catalog = panel.record.to_dict()['output']['catalog']
    assert len(catalog) == len(written)
    for entry in catalog:
        assert entry['type'] == 'plot_data_csv'
        assert entry.get('sha256'), "expected a non-empty SHA-256 for each exported CSV"
        assert entry.get('path'), "expected an absolute path for each exported CSV"
        assert os.path.isabs(entry['path'])
        assert entry.get('description') == description

    # PROV-JSON round-trips the description as a JSON string
    prov = json.loads(panel.record.to_prov_json())
    out_entity = prov['entity']['jade:out-001']
    assert json.loads(out_entity['jade:description']) == description
    assert out_entity['jade:path'] == catalog[0]['path']


def test_provenance_registration_handles_none_panel(tmp_path):
    # Must be a no-op (not raise) when no provenance panel is available.
    csv_export.register_outputs_in_provenance(None, ['nonexistent.csv'])


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_build_export_metadata_from_laplace_analyzer(tmp_path):
    analyzer = _FakeLaplaceAnalyzer()
    # Write a real file so SHA-256 can be computed
    test_file = str(tmp_path / 'plot_data.csv')
    pd.DataFrame({'a': [1, 2]}).to_csv(test_file, index=False)

    meta = csv_export.build_export_metadata(
        'plot_csv', [test_file], analyzer=analyzer, method_tag='NNLS')

    # Top-level schema keys must all be present
    for key in ('$schema', 'export_id', 'created', 'session_record_id',
                'agent', 'export', 'data', 'analysis'):
        assert key in meta, f"Missing key: {key}"

    # Agent info
    import ade_dls
    assert meta['agent']['software'] == 'JADE-DLS'
    assert meta['agent']['version'] == ade_dls.__version__

    # Export type and files
    assert meta['export']['type'] == 'plot_csv'
    assert len(meta['export']['files']) == 1
    assert meta['export']['files'][0]['sha256'] is not None

    # Data section
    assert meta['data']['angles_deg'] == [30.0, 90.0]
    assert meta['data']['temperature_K'] is not None

    # Analysis section: method_tag='NNLS' -> NNLS branch
    assert meta['analysis']['method'].startswith('NNLS')
    assert meta['analysis']['summary'] == {'n_peaks': 2, 'mean_R2': 0.998}


def test_build_export_metadata_graceful_none():
    """analyzer=None and provenance_panel=None must not raise."""
    meta = csv_export.build_export_metadata('plot_csv', [], analyzer=None, provenance_panel=None)

    assert meta['session_record_id'] is None
    assert meta['data']['folder'] is None
    assert meta['data']['n_datasets'] is None
    assert meta['analysis']['method'] is None


def test_write_metadata_roundtrip(tmp_path):
    meta = csv_export.build_export_metadata('plot_csv', [], analyzer=None)
    path = csv_export.write_metadata(meta, str(tmp_path), prefix="test_")

    assert os.path.isfile(path)
    with open(path, encoding='utf-8') as fh:
        loaded = json.load(fh)

    assert loaded['$schema'] == 'https://jade-dls.de/schema/export-metadata/v1.0'
    assert 'export_id' in loaded
    assert 'created' in loaded
