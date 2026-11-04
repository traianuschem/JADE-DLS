"""Headless API: run the ADE-DLS analysis chain without the GUI (no PyQt5 needed).

Typical use::

    from ade_dls import headless
    data = headless.load_series("path/to/asc-folder")
    analyzer, a = headless.run_cumulant_a(data)
    analyzer, b = headless.run_cumulant_b(data, fit_limits=(1e-6, 1e-2))
    lap, nnls = headless.run_nnls(data)                       # Gold defaults, see nnls_defaults()
    lap, reg = headless.run_regularized(data, {"alpha": 0.5})  # overrides on top of the defaults

The functions call exactly the classes the GUI uses (ade_dls.engine.*); parameter defaults are the
JADE gold defaults that the GUI dialogs start with. ``data`` is the dictionary produced by
``load_series`` (the same structure as ``MainWindow.pipeline.data``).
"""
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ade_dls.engine.loading import DataLoadError, SeriesLoader  # noqa: F401  (DataLoadError re-exported)
from ade_dls.engine.series import calculate_c_and_delta_c, preprocess_series

__all__ = [
    "DataLoadError", "load_series", "preprocess_series", "cumulant_analyzer", "laplace_analyzer",
    "run_cumulant_a", "run_cumulant_b", "run_cumulant_c", "run_cumulant_d", "run_nnls",
    "run_regularized", "nnls_defaults", "regularized_defaults", "method_d_defaults",
]


# ----------------------------------------------------------------------------- defaults
def nnls_defaults() -> Dict[str, Any]:
    """JADE gold NNLS parameters (same as the GUI dialog): 100 grid points, tau 1e-8..10 s."""
    return {
        'decay_times': np.logspace(-8, 1, 100),
        'prominence': 0.01,
        'distance': 1,
        'distance_threshold': 2.0,
        'clustering_strategy': 'silhouette_refined',
        'use_clustering': True,
        'peak_method': 'maximum',
        'use_weighting': False,
    }


def regularized_defaults() -> Dict[str, Any]:
    """JADE gold Regularized parameters: alpha 0.2, prominence 0.005, 200 grid points, beta fitted."""
    return {
        'decay_times': np.logspace(-8, 1, 200),
        'alpha': 0.2,
        'prominence': 0.005,
        'distance': 1,
        'normalize': True,
        'sparsity_penalty': 0.0,
        'enforce_unimodality': False,
        'fit_beta': True,
        'distance_threshold': 2.0,
        'clustering_strategy': 'silhouette_refined',
        'use_clustering': True,
        'peak_method': 'maximum',
        'use_centroid': False,
        'use_weighting': False,
    }


def method_d_defaults() -> Dict[str, Any]:
    """JADE gold Method D parameters: gap_threshold 3.0, Ward cut 2.0, silhouette_refined."""
    return {
        'n_max': 25,
        'n_start': 1,
        'gap_threshold': 3.0,
        'method': 'hierarchical',
        'n_clusters': 'auto',
        'distance_threshold': 2.0,
        'min_abundance': 0.3,
        'clustering_strategy': 'silhouette_refined',
        'use_multiprocessing': False,
        'fit_through_origin': False,
    }


def _merged(defaults: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    params = defaults
    params.update(overrides or {})
    return params


# ----------------------------------------------------------------------------- loading
def load_series(folder: str, parser=None, *, noise_params: Optional[dict] = None,
                load_countrates: bool = True, load_correlations: bool = True,
                load_cumulants: bool = True, preprocess: bool = True,
                on_progress: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    """Load one measurement series (autodetecting the instrument format unless ``parser`` is given).

    With ``preprocess=True`` the processed correlations and noise weights are computed once (as the
    GUI does after loading/filtering). Raises DataLoadError on failure.
    """
    if parser is None:
        from ade_dls.core.parsers import detect_parser
        parser = detect_parser(folder)
        if parser is None:
            raise DataLoadError(f"No known instrument format detected in:\n{folder}")
    data = SeriesLoader(folder, parser, load_countrates, load_correlations, load_cumulants,
                        on_progress=on_progress).run()
    if preprocess and data and data.get('correlations'):
        preprocess_series(data, noise_params)
    return data


# ----------------------------------------------------------------------------- cumulants
def cumulant_analyzer(data: Dict[str, Any]):
    """CumulantAnalyzer for ``data`` with prepared basedata (as ``MainWindow._prepare_cumulant_analyzer``)."""
    from ade_dls.engine.cumulant_analyzer import CumulantAnalyzer
    if not data or 'correlations' not in data:
        raise ValueError("data has no correlations; call load_series first")
    if 'data_folder' not in data:
        raise ValueError("data has no data_folder")
    analyzer = CumulantAnalyzer(data, data['data_folder'])
    analyzer.prepare_basedata()
    return analyzer


def run_cumulant_a(data, q_range: Optional[Tuple[float, float]] = None,
                   fit_through_origin: bool = False):
    """Method A (instrument cumulants). Returns ``(analyzer, results_dataframe)``."""
    analyzer = cumulant_analyzer(data)
    return analyzer, analyzer.run_method_a(q_range=q_range, fit_through_origin=fit_through_origin)


def run_cumulant_b(data, fit_limits: Tuple[float, float], q_range=None,
                   fit_through_origin: bool = False, use_weighting: bool = False):
    """Method B (linearised g2 fit in ``fit_limits`` [s])."""
    analyzer = cumulant_analyzer(data)
    analyzer.use_weighting = use_weighting
    return analyzer, analyzer.run_method_b(fit_limits, q_range=q_range,
                                           fit_through_origin=fit_through_origin)


def run_cumulant_c(data, params: Dict[str, Any], q_range=None, use_weighting: bool = False):
    """Method C (iterative non-linear fit); ``params`` as built by the GUI dialog (fit_function,
    adaptive_initial_guesses, initial_parameters, fit_limits, optimizer, ...)."""
    analyzer = cumulant_analyzer(data)
    analyzer.use_weighting = use_weighting
    return analyzer, analyzer.run_method_c(params, q_range=q_range)


def run_cumulant_d(data, params: Optional[Dict[str, Any]] = None, q_range=None,
                   use_weighting: bool = False):
    """Method D (multi-exponential decomposition + cross-file clustering); gold defaults
    (see method_d_defaults) overridden by ``params``."""
    analyzer = cumulant_analyzer(data)
    analyzer.use_weighting = use_weighting
    return analyzer, analyzer.run_method_d(_merged(method_d_defaults(), params), q_range=q_range)


# ----------------------------------------------------------------------------- Laplace
def laplace_analyzer(data: Dict[str, Any]):
    """LaplaceAnalyzer for ``data`` (completes df_basedata, c, delta_c and processed correlations
    exactly like the GUI does before opening the NNLS/Regularized dialogs)."""
    from ade_dls.engine.laplace_analyzer import LaplaceAnalyzer
    if not data or 'correlations' not in data or 'basedata' not in data:
        raise ValueError("data needs 'correlations' and 'basedata'; call load_series first")
    if 'df_basedata' not in data:
        data['df_basedata'] = data['basedata'].copy()
    if 'c' not in data or 'delta_c' not in data:
        calculate_c_and_delta_c(data)
    if data.get('processed_correlations') is None:
        preprocess_series(data, data.get('noise_params'))
    processed = data.get('processed_correlations')
    return LaplaceAnalyzer(
        processed_correlations=processed,
        df_basedata=data['df_basedata'],
        c=data['c'],
        delta_c=data['delta_c'],
        raw_correlations=data.get('correlations') if processed is None else None,
        weights=data.get('weights'),
    )


def run_nnls(data, params: Optional[Dict[str, Any]] = None, use_multiprocessing: bool = False):
    """NNLS analysis with gold defaults overridden by ``params``.
    Returns ``(laplace_analyzer, params)``; results live on the analyzer (``nnls_data`` etc.)."""
    analyzer = laplace_analyzer(data)
    p = _merged(nnls_defaults(), params)
    analyzer.set_weighting(p.get('use_weighting', False))
    analyzer.run_nnls(p, use_multiprocessing=use_multiprocessing, show_plots=False)
    analyzer.calculate_nnls_diffusion_coefficients()
    return analyzer, p


def run_regularized(data, params: Optional[Dict[str, Any]] = None, use_multiprocessing: bool = False):
    """Regularized NNLS with gold defaults overridden by ``params``. Returns ``(analyzer, params)``."""
    analyzer = laplace_analyzer(data)
    p = _merged(regularized_defaults(), params)
    analyzer.set_weighting(p.get('use_weighting', False))
    analyzer.run_regularized(p, use_multiprocessing=use_multiprocessing, show_plots=False)
    analyzer.calculate_regularized_diffusion_coefficients(
        use_clustering=p.get('use_clustering', True),
        distance_threshold=p.get('distance_threshold', 2.0),
        clustering_strategy=p.get('clustering_strategy', 'silhouette_refined'),
    )
    return analyzer, p
