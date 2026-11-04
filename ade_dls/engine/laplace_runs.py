"""NNLS / Regularized run sequence (no Qt), shared by the GUI and ``ade_dls.headless``.

The numerics live in ``LaplaceAnalyzer``; the functions here fix the *order* of the calls and the
data preparation in front of them, so that the main window and the headless API cannot diverge:
``df_basedata`` -> ``c``/``delta_c`` -> processed correlations/weights -> analyzer ->
``set_weighting`` -> fit -> diffusion coefficients.
"""
from typing import Any, Callable, Dict, Optional

from ade_dls.engine.series import calculate_c_and_delta_c, preprocess_series


def build_laplace_analyzer(data: Dict[str, Any],
                           preprocess: Optional[Callable[[Dict[str, Any]], Any]] = None):
    """LaplaceAnalyzer for the series dictionary ``data`` (completed in place).

    Adds ``df_basedata``, ``c``/``delta_c`` and the processed correlations when missing.
    ``preprocess(data)`` computes the latter (default: ``preprocess_series`` with the noise
    parameters stored on ``data``); the GUI passes its recording variant.
    """
    from ade_dls.engine.laplace_analyzer import LaplaceAnalyzer
    if not data or 'correlations' not in data or 'basedata' not in data:
        raise ValueError("data needs 'correlations' and 'basedata'; call load_series first")
    if 'df_basedata' not in data:
        data['df_basedata'] = data['basedata'].copy()
    if 'c' not in data or 'delta_c' not in data:
        calculate_c_and_delta_c(data)
    if data.get('processed_correlations') is None:
        if preprocess is None:
            preprocess_series(data, data.get('noise_params'))
        else:
            preprocess(data)
    processed = data.get('processed_correlations')
    return LaplaceAnalyzer(
        processed_correlations=processed,
        df_basedata=data['df_basedata'],
        c=data['c'],
        delta_c=data['delta_c'],
        raw_correlations=data.get('correlations') if processed is None else None,
        weights=data.get('weights'),
    )


def execute_nnls(analyzer, params: Dict[str, Any], use_multiprocessing: Optional[bool] = None,
                 on_status: Optional[Callable[[str], None]] = None) -> None:
    """Weighting, NNLS fits and diffusion coefficients on ``analyzer`` (results stay on it).
    ``use_multiprocessing=None`` takes the value from ``params`` (as the GUI dialog does)."""
    status = on_status or (lambda _msg: None)
    if use_multiprocessing is None:
        use_multiprocessing = params.get('use_multiprocessing', False)
    analyzer.set_weighting(params.get('use_weighting', False))
    status("Performing NNLS fits...")
    analyzer.run_nnls(params, use_multiprocessing=use_multiprocessing, show_plots=False)
    status("Calculating diffusion coefficients...")
    analyzer.calculate_nnls_diffusion_coefficients()


def execute_regularized(analyzer, params: Dict[str, Any], use_multiprocessing: Optional[bool] = None,
                        on_status: Optional[Callable[[str], None]] = None) -> None:
    """Weighting, Regularized fits and diffusion coefficients (clustering settings from ``params``)."""
    status = on_status or (lambda _msg: None)
    if use_multiprocessing is None:
        use_multiprocessing = params.get('use_multiprocessing', False)
    analyzer.set_weighting(params.get('use_weighting', False))
    status("Performing regularized fits...")
    analyzer.run_regularized(params, use_multiprocessing=use_multiprocessing, show_plots=False)
    status("Calculating diffusion coefficients...")
    analyzer.calculate_regularized_diffusion_coefficients(
        use_clustering=params.get('use_clustering', True),
        distance_threshold=params.get('distance_threshold', 2.0),
        clustering_strategy=params.get('clustering_strategy', 'silhouette_refined'),
    )
