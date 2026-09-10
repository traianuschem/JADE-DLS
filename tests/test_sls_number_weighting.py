"""
Tests for the JADE-DLS v2.3.0 SLS number-weighting revision
(ade_dls/analysis/sls.py): the Rh^exponent correction is applied only to
the already-extrapolated Guinier I0 values, never to the raw per-angle
intensities feeding the fit.
"""
import numpy as np
import pandas as pd
import pytest

from ade_dls.analysis import sls


def test_compute_number_weighted_i0_basic():
    guinier_results = {
        1: {'I0 [kHz]': 12.5},
        2: {'I0 [kHz]': 3.2},
        3: {'I0 [kHz]': 0.8},
    }
    rh_values = {1: 20.0, 2: 80.0, 3: 150.0}

    nw = sls.compute_number_weighted_I0(guinier_results, rh_values, exponent=6)

    assert set(nw) == {1, 2, 3}
    # fractions sum to 100%
    total_frac = sum(v['N-fraction [%]'] for v in nw.values())
    assert total_frac == pytest.approx(100.0, rel=1e-9)
    # smaller Rh dominates the number-weighted fraction for equal-ish intensities
    assert nw[1]['N-fraction [%]'] > nw[2]['N-fraction [%]'] > nw[3]['N-fraction [%]']
    # I0_nw values sum to the total intensity-weighted I0
    I0_total = sum(r['I0 [kHz]'] for r in guinier_results.values())
    total_nw = sum(v['I0_nw [kHz]'] for v in nw.values())
    assert total_nw == pytest.approx(I0_total, rel=1e-9)


def test_compute_number_weighted_i0_skips_invalid_rh():
    guinier_results = {1: {'I0 [kHz]': 10.0}, 2: {'I0 [kHz]': 5.0}}
    rh_values = {1: 20.0, 2: np.nan}  # population 2 has no usable Rh

    nw = sls.compute_number_weighted_I0(guinier_results, rh_values, exponent=6)
    assert nw[1]['N-fraction [%]'] == pytest.approx(100.0)
    assert np.isnan(nw[2]['I0_nw [kHz]'])
    assert np.isnan(nw[2]['N-fraction [%]'])


def test_compute_guinier_extrapolation_has_no_nw_toggle():
    import inspect
    sig = inspect.signature(sls.compute_guinier_extrapolation)
    assert 'use_nw_columns' not in sig.parameters


def test_compute_sls_data_number_weighted_removed():
    assert not hasattr(sls, 'compute_sls_data_number_weighted')


def test_guinier_extrapolation_never_sees_nw_columns():
    """The Guinier fit must only ever read I_popN [kHz], regardless of what
    other columns happen to exist on sls_data (regression guard for the
    pre-v2.3.0 bug where a caller could opt the fit into corrupted data)."""
    angles = np.array([30.0, 60.0, 90.0, 120.0, 150.0])
    q2 = (4 * np.pi * 1.332 / 632.8 * np.sin(np.radians(angles) / 2)) ** 2
    I_true = 10.0 * np.exp(-q2 * 100.0 / 3.0)  # Rg^2/3 = 100 -> Rg=~17.3nm

    df = pd.DataFrame({
        'angle [°]': angles,
        'q^2': q2,
        'I_pop1 [kHz]': I_true,
        # a maliciously different 'number-weighted-looking' column that must
        # NOT be picked up (mirrors the removed I_pop1_nw [kHz] convention)
        'I_pop1_nw [kHz]': I_true * 0.1,
    })

    results = sls.compute_guinier_extrapolation(df, n_populations=1, q2_range=None)
    assert 1 in results
    # Recovered I0 must match the true (undistorted) amplitude, not the
    # 10x-scaled-down 'nw' column.
    assert results[1]['I0 [kHz]'] == pytest.approx(10.0, rel=1e-6)


def test_summarize_sls_combined_columns_and_total_row():
    guinier_results = {1: {'I0 [kHz]': 12.5, 'Rg [nm]': 20.0, 'qRg_max': 0.5, 'R2': 0.99}}
    total_result = {'I0 [kHz]': 12.5, 'Rg [nm]': 20.0, 'qRg_max': 0.5, 'R2': 0.99}
    sls_data = pd.DataFrame({
        'I_pop1 [kHz]': [10.0, 11.0],
        'MeanCR_corr [kHz]': [12.0, 13.0],
    })
    rh_values = {1: 20.0}

    df = sls.summarize_sls_combined(sls_data, guinier_results, total_result,
                                    n_populations=1, rh_values=rh_values, exponent=6)

    assert isinstance(df, pd.DataFrame)  # unstyled, unlike the JADE notebook's .style output
    expected_cols = {'Population', 'Rh [nm]', 'I-fraction [%]', 'I0 (intensity) [kHz]',
                     'N-fraction [%]', 'I0 (number) [kHz]', 'Rg [nm]', 'qRg_max', 'R_squared'}
    assert set(df.columns) == expected_cols
    assert df.iloc[-1]['Population'] == 'Total'
    assert df.iloc[-1]['N-fraction [%]'] == pytest.approx(100.0)
