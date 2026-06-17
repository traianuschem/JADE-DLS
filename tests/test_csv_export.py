"""
Unit tests for the CSV plot-data export data layer (ade_dls/gui/export/csv_export.py).

These exercise the pure-Python builders and the writer/provenance helpers with
synthetic data — no Qt and no real measurement files required.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make the repo root importable when running pytest from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ade_dls.gui.export import csv_export

_D_FACTOR = 1e-15


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

class _FakeCumulantAnalyzer:
    """Minimal stand-in mimicking the attributes build_diffusion_tables reads."""

    def __init__(self):
        q2 = np.array([1.0e-4, 2.0e-4, 3.0e-4, 4.0e-4])
        # Γ = slope·q² + intercept for a clean linear relation
        slope, intercept = 5.0e3, 2.0
        gamma1 = slope * q2 + intercept
        self.method_a_data = pd.DataFrame({
            'q^2': q2,
            '1st order frequency [1/ms]': gamma1,
            '2nd order frequency [1/ms]': gamma1 * 1.05,
        })
        self.method_a_regression_stats = {
            'gamma_cols': ['1st order frequency [1/ms]', '2nd order frequency [1/ms]'],
            'regression_results': [
                {'gamma_col': '1st order frequency [1/ms]', 'intercept': intercept,
                 'q^2_coef': slope, 'q^2_se': 1.0, 'R_squared': 1.0},
                {'gamma_col': '2nd order frequency [1/ms]', 'intercept': intercept * 1.05,
                 'q^2_coef': slope * 1.05, 'q^2_se': 1.2, 'R_squared': 0.999},
            ],
        }


def _fake_cluster_info():
    stats = pd.DataFrame([
        {'cluster_id': 0, 'n_files': 4, 'abundance': 1.0,
         'mean_gamma': 100.0, 'std_gamma': 5.0, 'mean_D': 1e-12, 'std_D': 1e-13},
        {'cluster_id': 1, 'n_files': 3, 'abundance': 0.75,
         'mean_gamma': 400.0, 'std_gamma': 8.0, 'mean_D': 4e-12, 'std_D': 2e-13},
    ])
    gammas = pd.DataFrame({
        'filename': ['a', 'a', 'b', 'b'],
        'original_col': ['gamma_1', 'gamma_2', 'gamma_1', 'gamma_2'],
        'q_squared': [1e-4, 1e-4, 2e-4, 2e-4],
        'gamma': [100.0, 400.0, 101.0, 402.0],
        'D': [1e-12, 4e-12, 1.01e-12, 4.02e-12],
        'log_clustering_val': [-12.0, -11.4, -12.0, -11.4],
        'cluster': [0, 1, 0, 1],
    })
    return {
        'n_populations': 2,
        'population_stats': stats,
        'gammas_df': gammas,
        'cluster_id_to_pop': {0: 1, 1: 2},
    }


def _fake_full_results():
    decay_times = np.logspace(-6, 0, 20)
    return decay_times, {
        'sample_30deg_1': {'decay_times': decay_times,
                           'distribution': np.linspace(0, 1, 20), 'peaks': [5]},
        'sample_30deg_2': {'decay_times': decay_times,
                           'distribution': np.linspace(0, 2, 20), 'peaks': [5]},
        'sample_90deg_1': {'decay_times': decay_times,
                           'distribution': np.linspace(0, 3, 20), 'peaks': [6]},
    }


def _fake_basedata():
    return pd.DataFrame({
        'filename': ['sample_30deg_1', 'sample_30deg_2', 'sample_90deg_1'],
        'angle [°]': [30.0, 30.0, 90.0],
    })


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------

def test_diffusion_tables_structure_and_fit_consistency():
    analyzer = _FakeCumulantAnalyzer()
    tables = csv_export.build_diffusion_tables(analyzer)

    assert set(tables) == {'diffusion_datapoints', 'diffusion_fitcurve',
                           'diffusion_fit_parameters'}

    points = tables['diffusion_datapoints']
    assert 'q^2 [nm^-2]' in points.columns
    assert 'gamma_1st_order [1/ms]' in points.columns
    assert 'D_1st_order [m^2/s]' in points.columns
    # D = Γ/q²·1e-15
    expected_D = (points['gamma_1st_order [1/ms]'] / points['q^2 [nm^-2]'] * _D_FACTOR)
    np.testing.assert_allclose(points['D_1st_order [m^2/s]'], expected_D)

    params = tables['diffusion_fit_parameters'].set_index('order')
    # D [m^2/s] == slope·1e-15
    np.testing.assert_allclose(
        params.loc['1st_order', 'D [m^2/s]'],
        params.loc['1st_order', 'slope [ms^-1 nm^2]'] * _D_FACTOR)

    fit = tables['diffusion_fitcurve']
    assert len(fit) == 100
    assert 'gamma_fit_1st_order [1/ms]' in fit.columns


def test_diffusion_tables_requires_method_a():
    class _Empty:
        method_a_data = None
    with pytest.raises(ValueError):
        csv_export.build_diffusion_tables(_Empty())


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def test_clustering_tables_populations_as_columns():
    tables = csv_export.build_clustering_tables(None, _fake_cluster_info())
    assert 'clustering_populations' in tables
    assert 'clustering_points' in tables

    pops = tables['clustering_populations']
    # Populations appear as columns (population_1, population_2)
    assert 'population_1' in pops.columns
    assert 'population_2' in pops.columns
    assert 'statistic' in pops.columns

    points = tables['clustering_points']
    assert 'population' in points.columns
    # cluster 0 -> pop 1, cluster 1 -> pop 2
    assert set(points['population'].unique()) == {1, 2}


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def test_distribution_all():
    decay_times, full = _fake_full_results()
    tables = csv_export.build_distribution_tables(full, mode='all')
    df = tables['distributions']
    assert 'decay_time [s]' in df.columns
    # one intensity column per distribution
    assert len([c for c in df.columns if c.endswith('_intensity')]) == 3
    np.testing.assert_allclose(df['decay_time [s]'], decay_times)


def test_distribution_random_reproducible():
    _, full = _fake_full_results()
    t1 = csv_export.build_distribution_tables(full, mode='random', n_random=2, rng_seed=42)
    t2 = csv_export.build_distribution_tables(full, mode='random', n_random=2, rng_seed=42)
    cols1 = [c for c in t1['distributions'].columns if c != 'decay_time [s]']
    cols2 = [c for c in t2['distributions'].columns if c != 'decay_time [s]']
    assert cols1 == cols2
    assert len(cols1) == 2


def test_distribution_average_per_angle():
    _, full = _fake_full_results()
    tables = csv_export.build_distribution_tables(
        full, df_basedata=_fake_basedata(), mode='average_per_angle')
    df = tables['distributions']
    angle_cols = [c for c in df.columns if c.startswith('angle_')]
    # two angles -> two averaged columns
    assert len(angle_cols) == 2
    # 30° column is the mean of its two member distributions
    col_30 = [c for c in angle_cols if c.startswith('angle_30')][0]
    expected = np.mean([np.linspace(0, 1, 20), np.linspace(0, 2, 20)], axis=0)
    np.testing.assert_allclose(df[col_30], expected)


def test_distribution_average_requires_basedata():
    _, full = _fake_full_results()
    with pytest.raises(ValueError):
        csv_export.build_distribution_tables(full, mode='average_per_angle')


def test_full_results_from_plots_adapter():
    decay_times = np.logspace(-6, 0, 10)
    plots = {
        'a': ('FIGURE', {'f_optimized': np.ones(10), 'peaks': [2], 'num_peaks': 1}),
        'b': ('FIGURE', {'f_optimized': np.zeros(10), 'peaks': [], 'num_peaks': 0}),
    }
    full = csv_export.full_results_from_plots(plots, decay_times)
    assert set(full) == {'a', 'b'}
    assert full['a']['distribution'].shape == decay_times.shape
    tables = csv_export.build_distribution_tables(full, mode='all')
    assert len([c for c in tables['distributions'].columns if c.endswith('_intensity')]) == 2


# ---------------------------------------------------------------------------
# Writer + provenance
# ---------------------------------------------------------------------------

def test_write_tables_roundtrip(tmp_path):
    analyzer = _FakeCumulantAnalyzer()
    tables = csv_export.build_diffusion_tables(analyzer)
    written = csv_export.write_tables(tables, str(tmp_path), prefix="test_")

    assert len(written) == len(tables)
    for path in written:
        assert os.path.isfile(path)
        assert os.path.basename(path).startswith("test_")
        # comma-separated, dot decimal -> reloadable by pandas default
        reloaded = pd.read_csv(path)
        assert not reloaded.empty


def test_provenance_registration_records_hash(tmp_path):
    """register_outputs_in_provenance should add a hashed output entry."""
    from ade_dls.gui.core.provenance import ProvenanceRecord

    class _PanelStub:
        def __init__(self):
            self.record = ProvenanceRecord(version="test")

        def register_output(self, output_type, label, filepath=None, extra_fields=None):
            return self.record.add_output(output_type, label, filepath,
                                          extra_fields=extra_fields)

    analyzer = _FakeCumulantAnalyzer()
    tables = csv_export.build_diffusion_tables(analyzer)
    written = csv_export.write_tables(tables, str(tmp_path))

    panel = _PanelStub()
    csv_export.register_outputs_in_provenance(panel, written)

    catalog = panel.record.to_dict()['output']['catalog']
    assert len(catalog) == len(written)
    for entry in catalog:
        assert entry['type'] == 'plot_data_csv'
        assert entry.get('sha256'), "expected a non-empty SHA-256 for each exported CSV"


def test_provenance_registration_handles_none_panel(tmp_path):
    # Must be a no-op (not raise) when no provenance panel is available.
    csv_export.register_outputs_in_provenance(None, ['nonexistent.csv'])


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

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
        self.nnls_cluster_info = _fake_cluster_info()
        self.nnls_data = _fake_cluster_info()['gammas_df']

    def get_nnls_summary(self):
        return {'n_peaks': 2, 'mean_R2': 0.998}


def test_build_export_metadata_from_laplace_analyzer(tmp_path):
    analyzer = _FakeLaplaceAnalyzer()
    # Write a real file so SHA-256 can be computed
    test_file = str(tmp_path / 'nnls_distributions.csv')
    pd.DataFrame({'a': [1, 2]}).to_csv(test_file, index=False)

    meta = csv_export.build_export_metadata(
        "distributions_csv", [test_file], analyzer=analyzer)

    # Top-level schema keys must all be present
    for key in ('$schema', 'export_id', 'created', 'session_record_id',
                'agent', 'export', 'data', 'analysis'):
        assert key in meta, f"Missing key: {key}"

    # Agent info
    import ade_dls
    assert meta['agent']['software'] == 'JADE-DLS'
    assert meta['agent']['version'] == ade_dls.__version__

    # Export type and files
    assert meta['export']['type'] == 'distributions_csv'
    assert len(meta['export']['files']) == 1
    assert meta['export']['files'][0]['sha256'] is not None

    # Data section
    assert meta['data']['angles_deg'] == [30.0, 90.0]
    assert meta['data']['temperature_K'] is not None

    # Analysis section
    assert meta['analysis']['method'] is not None


def test_build_export_metadata_graceful_none():
    """analyzer=None and provenance_panel=None must not raise."""
    meta = csv_export.build_export_metadata("clustering_csv", [], analyzer=None, provenance_panel=None)

    assert meta['session_record_id'] is None
    assert meta['data']['folder'] is None
    assert meta['data']['n_datasets'] is None
    assert meta['analysis']['method'] is None


def test_write_metadata_roundtrip(tmp_path):
    meta = csv_export.build_export_metadata("diffusion_csv", [], analyzer=None)
    path = csv_export.write_metadata(meta, str(tmp_path), prefix="test_")

    assert os.path.isfile(path)
    import json
    with open(path, encoding='utf-8') as fh:
        loaded = json.load(fh)

    assert loaded['$schema'] == 'https://jade-dls.de/schema/export-metadata/v1.0'
    assert 'export_id' in loaded
    assert 'created' in loaded


def test_clustering_tables_nnls_shape():
    """build_clustering_tables works with NNLS cluster_info (same structure as Method D)."""
    cluster_info = _fake_cluster_info()
    tables = csv_export.build_clustering_tables(None, cluster_info)

    assert 'clustering_populations' in tables
    assert 'clustering_points' in tables
    pops = tables['clustering_populations']
    assert 'population_1' in pops.columns
    assert 'population_2' in pops.columns
