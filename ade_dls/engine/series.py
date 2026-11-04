"""Series-level preparation steps shared by the GUI and the headless API (Qt-free).

These are the numerical parts that used to sit inside ``MainWindow`` methods:
preprocessing of the correlation functions (noise correction + noise weights) and the
Stokes-Einstein constants ``c = kB*T/(6*pi*eta)`` / ``delta_c``.
"""
import numpy as np
import pandas as pd
from scipy.constants import k as boltzmann_constant

from ade_dls.engine.correlation_preprocessing import preprocess_correlations


def preprocess_series(data: dict, noise_params=None):
    """Compute the canonical processed correlations for ``data`` and store them on it.

    Sets ``processed_correlations``, ``weights``, ``weighting_available``,
    ``weighting_unavailable_reason`` and ``noise_params`` on *data* (exactly once per
    filtering pass; recomputed from the current ``correlations``/``basedata`` every call).

    Returns ``(processed, weights, reason)`` or ``None`` when *data* has no correlations.
    """
    correlations = data.get('correlations')
    if not correlations:
        return None
    processed, weights, reason = preprocess_correlations(
        correlations, data.get('basedata'), noise_params)
    data['processed_correlations'] = processed
    data['weights'] = weights
    data['weighting_available'] = weights is not None
    data['weighting_unavailable_reason'] = reason
    data['noise_params'] = noise_params
    return processed, weights, reason


def calculate_c_and_delta_c(data: dict) -> None:
    """Store ``c = kB*T/(6*pi*eta)`` and its propagated error ``delta_c`` (one-element Series)
    in ``data`` from ``data['df_basedata']`` (temperature [K], viscosity [cp])."""
    df_basedata = data['df_basedata']

    mean_temperature = df_basedata['temperature [K]'].mean()
    std_temperature = df_basedata['temperature [K]'].std()
    mean_viscosity = df_basedata['viscosity [cp]'].mean()
    std_viscosity = df_basedata['viscosity [cp]'].std()

    c = (boltzmann_constant * mean_temperature) / (6 * np.pi * mean_viscosity * 1e-3)
    fractional_error_c = np.sqrt(
        (std_temperature / mean_temperature) ** 2 + (std_viscosity / mean_viscosity) ** 2
    )
    delta_c = fractional_error_c * c

    data['c'] = pd.Series([c])
    data['delta_c'] = pd.Series([delta_c])
    print(f"[C calculation] c = {c:.4e} +/- {delta_c:.4e} (rel. error: {(delta_c/c):.4%})")
