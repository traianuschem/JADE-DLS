"""
Tests for ade_dls.gui.core.correlation_preprocessing (JADE-DLS v3.0):
the single place processed correlations, noise correction and noise
weights are computed once per (filtered) dataset.
"""
import numpy as np
import pandas as pd

from ade_dls.gui.core.correlation_preprocessing import preprocess_correlations


def _raw_correlations(n=50, seed=0):
    rng = np.random.default_rng(seed)
    t_ms = np.logspace(-4, 2, n)  # time [ms]
    corr = 0.8 * np.exp(-2.0 * (t_ms / 1000.0) / 1e-3) + rng.normal(0, 0.001, n)
    df = pd.DataFrame({
        'time [ms]': t_ms,
        'correlation 1': corr,
        'correlation 2': corr,
        'correlation 3': np.zeros(n),
        'correlation 4': np.zeros(n),
    })
    return {'file1.ASC': df}


def _basedata_with_weighting_cols(filenames, complete=True):
    rows = []
    for fn in filenames:
        rows.append({
            'filename': fn,
            'duration [s]': 10.0 if complete else np.nan,
            'meancr0 [kHz]': 200.0 if complete else np.nan,
            'meancr1 [kHz]': 210.0 if complete else np.nan,
        })
    return pd.DataFrame(rows)


def test_preprocess_correlations_produces_weights_when_metadata_complete():
    correlations = _raw_correlations()
    basedata = _basedata_with_weighting_cols(correlations.keys())

    processed, weights, reason = preprocess_correlations(correlations, basedata, None)

    assert set(processed.keys()) == {'file1.ASC'}
    assert {'t [s]', 'g(2)-1'} <= set(processed['file1.ASC'].columns)
    # No 'weight' column yet -- that is only attached on demand
    assert 'weight' not in processed['file1.ASC'].columns
    assert weights is not None
    assert reason is None
    assert len(weights['file1.ASC']) == len(processed['file1.ASC'])


def test_preprocess_correlations_no_basedata_gives_reason():
    correlations = _raw_correlations(seed=1)
    processed, weights, reason = preprocess_correlations(correlations, None, None)
    assert weights is None
    assert reason is not None
    assert 'processed_correlations' not in reason  # sanity: message is human text
    assert len(processed) == 1


def test_preprocess_correlations_missing_metadata_columns_gives_reason():
    correlations = _raw_correlations(seed=2)
    # basedata without the weighting columns at all (e.g. LS Instruments)
    basedata = pd.DataFrame({'filename': list(correlations.keys())})
    processed, weights, reason = preprocess_correlations(correlations, basedata, None)
    assert weights is None
    assert reason is not None
    assert len(processed) == 1


def test_preprocess_correlations_nan_metadata_for_one_file_gives_reason():
    correlations = _raw_correlations(seed=3)
    basedata = _basedata_with_weighting_cols(correlations.keys(), complete=False)
    processed, weights, reason = preprocess_correlations(correlations, basedata, None)
    assert weights is None
    assert reason is not None
    assert 'file1.ASC' in reason


def test_preprocess_correlations_applies_noise_correction():
    correlations = _raw_correlations(seed=4)
    basedata = _basedata_with_weighting_cols(correlations.keys())
    noise_params = {'baseline_correction': True, 'baseline_pct': 10,
                     'intercept_correction': False, 'intercept_pct': 5}

    processed_plain, _, _ = preprocess_correlations(correlations, basedata, None)
    processed_corrected, weights, reason = preprocess_correlations(correlations, basedata, noise_params)

    # Baseline correction shifts g(2)-1 -- the two runs must differ
    assert not np.allclose(
        processed_plain['file1.ASC']['g(2)-1'].to_numpy(),
        processed_corrected['file1.ASC']['g(2)-1'].to_numpy(),
    )
    # Weighting is still available after noise correction
    assert weights is not None
    assert reason is None
