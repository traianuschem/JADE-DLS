"""Cumulant method run sequence (no Qt), shared by the main window and ``ade_dls.headless``.

``build_cumulant_analyzer`` creates the analyzer for a series dictionary; the ``execute_method_*``
functions fix the order *weighting flag -> method run*, so that the GUI and the headless API call
the analyzer identically. The numerics live in ``CumulantAnalyzer``.
"""
from typing import Any, Callable, Dict, Optional, Tuple


def build_cumulant_analyzer(data: Dict[str, Any],
                            on_status: Optional[Callable[[str], None]] = None):
    """CumulantAnalyzer for ``data`` with prepared basedata.

    ``data`` needs ``correlations`` and ``data_folder``; noise correction is *not* applied here (it
    happened once in ``preprocess_series`` and the analyzer reads the cached processed correlations).
    """
    from ade_dls.engine.cumulant_analyzer import CumulantAnalyzer
    if not data or 'correlations' not in data:
        raise ValueError("data has no correlations; call load_series first")
    if 'data_folder' not in data:
        raise ValueError("data has no data_folder")
    analyzer = CumulantAnalyzer(data, data['data_folder'])
    if on_status:
        on_status("Preparing basedata...")
    analyzer.prepare_basedata()
    return analyzer


def execute_method_a(analyzer, q_range: Optional[Tuple[float, float]] = None,
                     fit_through_origin: bool = False):
    """Method A (instrument cumulants)."""
    return analyzer.run_method_a(q_range=q_range, fit_through_origin=fit_through_origin)


def execute_method_b(analyzer, fit_limits: Tuple[float, float], q_range=None,
                     fit_through_origin: bool = False, use_weighting: bool = False):
    """Method B (linearised g2 fit in ``fit_limits`` [s])."""
    analyzer.use_weighting = use_weighting
    return analyzer.run_method_b(fit_limits, q_range=q_range, fit_through_origin=fit_through_origin)


def execute_method_c(analyzer, params: Dict[str, Any], q_range=None, use_weighting: bool = False):
    """Method C (iterative non-linear fit); ``params`` as built by the GUI dialog."""
    analyzer.use_weighting = use_weighting
    return analyzer.run_method_c(params, q_range=q_range)


def execute_method_d(analyzer, params: Dict[str, Any], q_range=None, use_weighting: bool = False):
    """Method D (multi-exponential decomposition + cross-file clustering)."""
    analyzer.use_weighting = use_weighting
    return analyzer.run_method_d(params, q_range=q_range)
