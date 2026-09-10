"""
Tests for ade_dls.analysis.weighting (JADE-DLS v3.0 port).

Covers the module's own contract (mean-normalization, missing-metadata
guard, dict-level conveniences) -- numeric parity against the original
JADE-DLS notebook module is verified separately in a one-off comparison
script (see MEMORY / session notes), not re-checked here to avoid a hard
dependency on the sibling JADE_DLS_original checkout.
"""
import numpy as np
import pandas as pd
import pytest

from ade_dls.analysis.weighting import (
    estimate_weights,
    compute_weights_for_all,
    apply_weights_to_correlations,
    infer_sampling_times,
)


def _synthetic_curve(n=200, tau_c=1e-3, beta=0.85, seed=0):
    tau = np.logspace(-7, 0, n)
    rng = np.random.default_rng(seed)
    g2m1 = beta * np.exp(-2.0 * tau / tau_c) + rng.normal(0, 0.001, n)
    return tau, g2m1


def test_estimate_weights_mean_normalized():
    tau, g2m1 = _synthetic_curve()
    w = estimate_weights(tau, g2m1, cr_A_kHz=200.0, cr_B_kHz=210.0, T=30.0, verbose=False)
    assert w.shape == tau.shape
    assert np.all(w > 0)
    assert np.isclose(np.mean(w), 1.0)


def test_estimate_weights_raises_on_missing_metadata():
    tau, g2m1 = _synthetic_curve()
    with pytest.raises(ValueError):
        estimate_weights(tau, g2m1, cr_A_kHz=None, cr_B_kHz=210.0, T=30.0, verbose=False)
    with pytest.raises(ValueError):
        estimate_weights(tau, g2m1, cr_A_kHz=200.0, cr_B_kHz=210.0, T=float('nan'), verbose=False)


def test_infer_sampling_times_positive():
    tau = np.logspace(-7, 0, 50)
    dt = infer_sampling_times(tau)
    assert dt.shape == tau.shape
    assert np.all(dt > 0)


def test_compute_weights_for_all_and_apply():
    tau, g2m1 = _synthetic_curve(seed=1)
    df = pd.DataFrame({'t [s]': tau, 'g(2)-1': g2m1})
    processed = {'file1.ASC': df}
    basedata = pd.DataFrame({
        'filename': ['file1.ASC'],
        'duration [s]': [10.0],
        'meancr0 [kHz]': [200.0],
        'meancr1 [kHz]': [210.0],
    })

    weights_dict = compute_weights_for_all(processed, basedata)
    assert set(weights_dict.keys()) == {'file1.ASC'}
    assert len(weights_dict['file1.ASC']) == len(tau)

    weighted = apply_weights_to_correlations(processed, weights_dict)
    # Returns a NEW dict; original is untouched (no 'weight' column)
    assert 'weight' not in df.columns
    assert 'weight' in weighted['file1.ASC'].columns
    assert np.allclose(weighted['file1.ASC']['weight'].to_numpy(), weights_dict['file1.ASC'])


def test_compute_weights_for_all_raises_on_missing_file():
    tau, g2m1 = _synthetic_curve(seed=2)
    df = pd.DataFrame({'t [s]': tau, 'g(2)-1': g2m1})
    processed = {'missing.ASC': df}
    basedata = pd.DataFrame({
        'filename': ['other.ASC'],
        'duration [s]': [10.0],
        'meancr0 [kHz]': [200.0],
        'meancr1 [kHz]': [210.0],
    })
    with pytest.raises(ValueError):
        compute_weights_for_all(processed, basedata)


def test_apply_weights_to_correlations_raises_on_missing_weights():
    tau, g2m1 = _synthetic_curve(seed=3)
    df = pd.DataFrame({'t [s]': tau, 'g(2)-1': g2m1})
    with pytest.raises(ValueError):
        apply_weights_to_correlations({'a.ASC': df}, {})
