"""
Unit tests for ade_dls/gui/export/plot_series_export.py — generic per-series
data extraction from a matplotlib Figure, used by the "Export current plot
as CSV" button.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ade_dls.gui.export import plot_series_export as pse
from ade_dls.gui.export.csv_export import write_tables


def _demo_figure():
    """3-panel figure covering line/scatter/axhline/fill_between/proxy-line/log-x
    (panel 0), errorbar+hist (panel 1), and an unlabelled Q-Q plot (panel 2)."""
    fig = Figure(figsize=(9, 3))
    ax1, ax2, ax3 = fig.subplots(1, 3)

    x = np.linspace(1e-6, 1e-2, 20)
    y = np.exp(-x * 500)
    ax1.plot(x, y, 'o', label='Data')
    ax1.plot(x, y * 0.99, '-', label='Fit (R²=0.9931)')
    ax1.axhline(0, color='r', linestyle='--')
    ax1.scatter(x[:5], y[:5], label='Population 1')
    ax1.fill_between(x, 0, y, alpha=0.3)
    ax1.plot([], [], label='proxy')  # empty legend-only line
    ax1.set_xscale('log')
    ax1.set_title('Data & Fit')
    ax1.set_xlabel('lag time τ [s]')
    ax1.set_ylabel('g2-1')

    ax2.errorbar([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], yerr=[0.1, 0.2, 0.3],
                fmt='o', capsize=3, label='I_pop1')
    ax2.hist(np.random.default_rng(0).normal(size=50), bins=5)

    residuals = y - y.mean()
    stats.probplot(residuals, dist='norm', plot=ax3)
    ax3.set_title('Q-Q Plot')

    return fig


def test_panel0_series_names_and_kinds():
    series = pse.extract_plot_series(_demo_figure(), 'C: sample file.ASC')
    panel0 = {s.name: s for s in series if s.panel_index == 0}

    assert set(panel0) == {'data', 'fit', 'population_1'}
    assert panel0['data'].kind == 'line'
    assert panel0['fit'].kind == 'line'
    assert panel0['fit'].label == 'Fit (R²=0.9931)'  # stats kept in label, stripped from slug
    assert panel0['population_1'].kind == 'scatter'
    assert panel0['population_1'].label == 'Population 1'
    assert panel0['data'].xscale == 'log'
    # axhline, fill_between and the empty proxy line must not appear as series
    assert all(not name.startswith('line_') for name in panel0)


def test_panel1_errorbar_and_histogram():
    series = pse.extract_plot_series(_demo_figure(), 'demo')
    panel1 = [s for s in series if s.panel_index == 1]

    errorbars = [s for s in panel1 if s.kind == 'errorbar']
    assert len(errorbars) == 1
    df = errorbars[0].data
    assert list(df.columns) == ['x', 'y', 'y_err_lo', 'y_err_hi']
    np.testing.assert_allclose(df['y_err_hi'], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(df['y_err_lo'], [0.1, 0.2, 0.3])

    bars = [s for s in panel1 if s.kind == 'bar']
    assert len(bars) == 1
    assert list(bars[0].data.columns) == ['bin_left', 'bin_right', 'height']
    assert len(bars[0].data) == 5  # 5 histogram bins

    # caplines/error-bar helper lines must not leak through as separate 'line' series
    assert not any(s.kind == 'line' for s in panel1)


def test_panel2_qq_fallback_names():
    series = pse.extract_plot_series(_demo_figure(), 'demo')
    panel2_names = {s.name for s in series if s.panel_index == 2}
    assert panel2_names == {'qq_points', 'qq_fit'}


def test_duplicate_labels_get_deduped_names():
    fig = Figure()
    ax = fig.subplots()
    ax.plot([1, 2], [1, 2], label='data')
    ax.plot([1, 2], [2, 3], label='data')
    series = pse.extract_plot_series(fig, 'dup')
    names = sorted(s.name for s in series)
    assert names == ['data', 'data_2']


def test_series_to_tables_keys_are_short_and_unique(tmp_path):
    series = pse.extract_plot_series(_demo_figure(), 'C: sample file.ASC')
    tables = pse.series_to_tables(series, 'C: sample file.ASC')

    assert len(tables) == len(series)
    for key in tables:
        assert len(key) < 80
        assert key.startswith('C__sample_file.ASC__panel')

    written = write_tables(tables, str(tmp_path))
    assert len(written) == len(tables)
    for path in written:
        reloaded = pd.read_csv(path)
        assert not reloaded.empty


def test_describe_series_has_expected_keys():
    series = pse.extract_plot_series(_demo_figure(), 'demo')
    desc = pse.describe_series(series[0], 'demo', 'Method C')
    expected_keys = {'plot', 'method', 'panel', 'panel_title', 'series', 'kind',
                     'x_label', 'y_label', 'x_scale', 'y_scale', 'columns', 'n_points'}
    assert set(desc) == expected_keys
    assert desc['method'] == 'Method C'
    assert desc['n_points'] == len(series[0].data)


def test_empty_figure_returns_no_series():
    fig = Figure()
    assert pse.extract_plot_series(fig, 'empty') == []


def test_empty_axes_returns_no_series():
    fig = Figure()
    fig.subplots()  # one empty Axes, no artists
    assert pse.extract_plot_series(fig, 'empty_axes') == []
