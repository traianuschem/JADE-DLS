"""Post-fit refinement helpers (no Qt): exclusion of fits and q-range restriction applied to an
already fitted ``CumulantAnalyzer``, shared by the GUI refinement dialogs and ``ade_dls.headless``.

The numerics live in the analyzer (``recompute_method_c_diffusion``, ``refine_method_b`` ...);
the functions here assemble the inputs and the bookkeeping that the result tables display.
"""
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

MIN_FILES_METHOD_C = 2      # fewer remaining files cannot be regressed


def refine_method_c(analyzer, q_range: Optional[Tuple[float, float]] = None,
                    excluded_fits: Optional[Sequence[str]] = None
                    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Re-compute the Method C diffusion regression with a q-range and/or excluded fits.

    Returns ``(result_df, refinement_stats)`` or ``None`` if there is no Method C fit, fewer than
    ``MIN_FILES_METHOD_C`` files remain, or the recomputation fails. ``result_df`` carries the
    refinement metadata columns (N files, q-range, original results); ``refinement_stats`` is the
    regression-stats dict extended by ``refinement_info``.
    """
    if not hasattr(analyzer, 'method_c_fit') or analyzer.method_c_fit is None:
        print("[ERROR] No Method C data available")
        return None

    # included files = complement of the excluded ones
    all_files = analyzer.method_c_fit['filename'].unique().tolist()
    included_files = [f for f in all_files if f not in (excluded_fits or [])]

    if len(included_files) < MIN_FILES_METHOD_C:
        print("[ERROR] Too few files remaining after exclusion")
        return None

    try:
        result = analyzer.recompute_method_c_diffusion(included_files, q_range)
    except Exception as e:
        print(f"[ERROR] recompute_method_c_diffusion failed: {e}")
        return None

    # actual data points and contributing files (set by recompute_method_c_diffusion)
    n_data_points = getattr(analyzer, 'method_c_recompute_n_points', None)
    n_files_in_range = getattr(analyzer, 'method_c_recompute_n_files_in_range', None)

    n_total = len(all_files)
    n_included = len(included_files)
    n_excluded = n_total - n_included
    result['N files (total)'] = [n_total]
    result['N files (included)'] = [n_included]
    result['N files (excluded)'] = [n_excluded]
    q_str = (f"{q_range[0]:.3e} – {q_range[1]:.3e} nm⁻²"
             if q_range is not None else "Full range")
    result['q-range applied'] = [q_str]

    # original results for comparison
    orig = getattr(analyzer, 'method_c_results', None)
    if orig is not None and not orig.empty:
        result['Original Rh [nm]'] = [orig['Rh [nm]'].iloc[0]
                                      if 'Rh [nm]' in orig.columns else np.nan]
        result['Original D [m²/s]'] = [orig['D [m²/s]'].iloc[0]
                                            if 'D [m²/s]' in orig.columns else np.nan]
        result['Original R²'] = [orig['R_squared'].iloc[0]
                                      if 'R_squared' in orig.columns else np.nan]

    refinement_stats = dict(getattr(analyzer, 'method_c_regression_stats', {}) or {})
    refinement_stats['refinement_info'] = {
        'n_total': n_total,
        'n_included': n_included,
        'n_excluded': n_excluded,
        'n_data_points': n_data_points,         # actual (q^2, Gamma) pairs in the OLS regression
        'n_files_in_range': n_files_in_range,   # unique files with data within the q-range
        'excluded_files': excluded_fits or [],
        'q_range': q_range,
        'q_range_str': q_str,
        'original_rh': float(orig['Rh [nm]'].iloc[0])
                       if orig is not None and 'Rh [nm]' in orig.columns else None,
        'original_d': float(orig['D [m²/s]'].iloc[0])
                      if orig is not None and 'D [m²/s]' in orig.columns else None,
        'original_r2': float(orig['R_squared'].iloc[0])
                       if orig is not None and 'R_squared' in orig.columns else None,
    }
    return result, refinement_stats


def exclude_files_from_method_d(analyzer, excluded_files: Optional[Sequence[str]]) -> int:
    """Drop the rows of the excluded files from ``analyzer.method_d_data`` (before re-clustering).

    Returns the number of rows removed (0 when nothing is excluded or Method D has not run)."""
    data = getattr(analyzer, 'method_d_data', None)
    if not excluded_files or data is None:
        return 0
    kept = data[~data['filename'].isin(excluded_files)].reset_index(drop=True)
    analyzer.method_d_data = kept
    return len(data) - len(kept)


# ----------------------------------------------------------------------------- Laplace (NNLS / Regularized)
_CLUSTERING_KEYS = {
    'NNLS': ('use_clustering', 'clustering_strategy', 'distance_threshold'),
    'Regularized': ('use_clustering', 'clustering_strategy', 'distance_threshold', 'min_abundance'),
}
POST_REFINED_SUFFIX = " (Post-Refined)"


def laplace_refinement_has_changes(q_range=None, excluded_files=None, clustering_params=None,
                                   per_pop_ranges=None) -> bool:
    """False if a refinement request changes nothing (no q-range, exclusion, clustering or
    per-population range): the GUI then skips the recomputation."""
    return bool(q_range is not None or excluded_files or clustering_params or per_pop_ranges)


def refine_laplace(analyzer, method: str, q_range: Optional[Tuple[float, float]] = None,
                   excluded_files: Optional[Sequence[str]] = None,
                   clustering_params: Optional[Dict[str, Any]] = None,
                   per_pop_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
                   fit_through_origin: bool = False) -> Optional[pd.DataFrame]:
    """Post-fit refinement of an NNLS / Regularized result (no Qt): restores the distribution table
    from the snapshot of the first fit (so repeated refinements always start from the original
    data), drops the distributions of ``excluded_files``, recomputes the diffusion coefficients with
    ``q_range`` / ``per_pop_ranges`` / the clustering settings and appends the final results with
    the label suffix " (Post-Refined)".

    ``method`` is ``'NNLS'`` or ``'Regularized'``; ``fit_through_origin`` applies to Regularized only
    (as in the GUI). Returns the analyzer's final-results table.
    """
    if method not in _CLUSTERING_KEYS:
        raise ValueError(f"Unknown Laplace method {method!r}; use 'NNLS' or 'Regularized'")
    is_nnls = method == 'NNLS'
    data_attr = 'nnls_data' if is_nnls else 'regularized_data'
    snapshot = getattr(analyzer, '_nnls_data_snapshot' if is_nnls else '_regularized_data_snapshot', None)
    clustering_params = clustering_params or {}
    excluded_files = list(excluded_files or [])

    # idempotent: always start from the original (unfiltered) distributions
    if snapshot is not None:
        setattr(analyzer, data_attr, snapshot.copy())

    if excluded_files and hasattr(analyzer, data_attr):
        data = getattr(analyzer, data_attr)
        kept = data[~data['filename'].isin(excluded_files)].reset_index(drop=True)
        kept.index = kept.index + 1
        setattr(analyzer, data_attr, kept)
        print(f"  Removed {len(excluded_files)} datasets, {len(kept)} remaining")

    kwargs = {k: v for k, v in clustering_params.items() if k in _CLUSTERING_KEYS[method]}
    if is_nnls:
        analyzer.calculate_nnls_diffusion_coefficients(
            x_range=q_range, per_pop_ranges=per_pop_ranges or None, **kwargs)
        analyzer._calculate_nnls_final_results(append_mode=True, label_suffix=POST_REFINED_SUFFIX)
        return getattr(analyzer, 'nnls_final_results', None)
    analyzer.calculate_regularized_diffusion_coefficients(
        x_range=q_range, per_pop_ranges=per_pop_ranges or None,
        fit_through_origin=fit_through_origin, **kwargs)
    analyzer._calculate_regularized_final_results(append_mode=True, label_suffix=POST_REFINED_SUFFIX)
    return getattr(analyzer, 'regularized_final_results', None)
