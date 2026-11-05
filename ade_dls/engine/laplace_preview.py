"""Analysis helpers behind the NNLS / Regularized parameter-dialog previews (no Qt, no plotting).

The dialogs fit a few randomly chosen measurements and show the distributions; the numbers on top
of that -- Γ per peak against q², a log-log estimate of D, the estimated number of modes -- are
computed here so that they can be tested and used without the GUI. ``results`` is the list of
per-measurement result dictionaries of the preview fits (keys ``tau_1``, ``tau_2`` ...), ``files``
the matching measurement labels (same order), ``df_basedata`` the analyzer's base table.
"""
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SLOPE_TOLERANCE = 0.2       # |slope - 1| below this: the log-log line is read as Γ = D·q²


def preview_gamma_table(results: Sequence[Dict[str, Any]], files: Sequence[str], df_basedata,
                        warn: bool = False) -> pd.DataFrame:
    """DataFrame of the preview results with a ``q^2`` column looked up from ``df_basedata`` by
    filename; rows without a q² value are dropped. ``results`` and ``files`` must be parallel
    (a length mismatch raises ``ValueError`` like any pandas column assignment)."""
    table = pd.DataFrame(list(results))
    q_squared = []
    for filename in files:
        rows = df_basedata[df_basedata['filename'] == filename]
        if len(rows) > 0:
            q_squared.append(rows.iloc[0]['q^2'])
        else:
            if warn:
                print(f"[Preview Clustering] Warning: {filename} not found in basedata")
            q_squared.append(None)
    table['q^2'] = q_squared
    return table.dropna(subset=['q^2'])


def tau_columns(table: pd.DataFrame) -> List[str]:
    return [c for c in table.columns if c.startswith('tau_')]


def with_decay_rates(table: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Add the Γ = 1/τ columns (``gamma_k`` for every ``tau_k``). Returns
    ``(table, tau_cols, gamma_cols)``; an empty ``tau_cols`` means there is nothing to convert."""
    from ade_dls.analysis.regularized_optimized import calculate_decay_rates
    tau_cols = tau_columns(table)
    if not tau_cols:
        return table, [], []
    table = calculate_decay_rates(table, tau_cols)
    return table, tau_cols, [c.replace('tau', 'gamma') for c in tau_cols]


def mode_count_color(n_modes: int) -> str:
    """Label colour of the 'Detected modes' display: red none, green one, blue 2-3, orange more."""
    if n_modes == 0:
        return "#f44336"
    if n_modes == 1:
        return "#4CAF50"
    if n_modes <= 3:
        return "#2196F3"
    return "#FF9800"


# ----------------------------------------------------------------------------- Regularized preview
class ModeEstimate(NamedTuple):
    status: str                 # 'ok' | 'disabled' | 'insufficient' | 'no_tau'
    n_modes: Optional[int]      # number of populations found (only for status 'ok')


def estimate_modes_by_clustering(results, files, df_basedata, use_clustering: bool = True,
                                 distance_threshold: float = 2.0,
                                 clustering_strategy: str = 'silhouette_refined') -> ModeEstimate:
    """Number of modes in the preview: Ward clustering of Γ/q² over the preview peaks, the same
    method as the full analysis (``cluster_all_gammas``). Needs at least two measurements with q²."""
    from ade_dls.analysis.clustering import cluster_all_gammas
    if not use_clustering:
        return ModeEstimate('disabled', None)
    table = preview_gamma_table(results, files, df_basedata, warn=True)
    if len(table) < 2:
        return ModeEstimate('insufficient', None)
    table, tau_cols, gamma_cols = with_decay_rates(table)
    if not tau_cols:
        return ModeEstimate('no_tau', None)
    _, cluster_info = cluster_all_gammas(
        table, gamma_cols=gamma_cols, q_squared_col='q^2', enable_clustering=True,
        normalize_by_q2=True, distance_threshold=distance_threshold,
        clustering_strategy=clustering_strategy, interactive=False, plot=False)
    return ModeEstimate('ok', cluster_info.get('n_populations', 0))


def regularized_preview_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of the dialog parameters that the Regularized preview fit uses."""
    return {
        'decay_times': params['decay_times'],
        'prominence': params['prominence'],
        'distance': params['distance'],
        'alpha': params['alpha'],
        'normalize': params['normalize'],
        'sparsity_penalty': params['sparsity_penalty'],
        'enforce_unimodality': params['enforce_unimodality'],
        'fit_beta': params.get('fit_beta', False),
    }


# ----------------------------------------------------------------------------- NNLS preview
def peak_count_summary(results: Sequence[Dict[str, Any]]) -> Optional[Tuple[int, int, int]]:
    """``(median, min, max)`` of the number of peaks per preview fit (``tau_*`` entries that are
    neither None nor NaN); ``None`` without results. The median is rounded to an integer."""
    if not results:
        return None
    counts = [sum(1 for k, v in r.items()
                  if k.startswith('tau_') and v is not None
                  and not (isinstance(v, float) and np.isnan(v)))
              for r in results]
    return int(round(np.median(counts))), int(min(counts)), int(max(counts))


def peak_count_text(summary: Tuple[int, int, int]) -> str:
    n_modes, low, high = summary
    if low == high:
        return f"Detected modes: {n_modes}"
    return f"Detected modes: {n_modes} (range {low}–{high})"


# ----------------------------------------------------------------------------- Γ vs q² log-log
class PeakSeries(NamedTuple):
    index: int                          # 0-based peak position (label "Peak index+1")
    q2: np.ndarray
    gamma: np.ndarray
    slope: Optional[float]              # log10-log10 fit (needs >= 2 points)
    intercept: Optional[float]
    q2_fit: Optional[np.ndarray]        # 50 points between min and max q² on the fitted line
    gamma_fit: Optional[np.ndarray]
    d_estimate: Optional[float]         # 10**intercept when the slope is within SLOPE_TOLERANCE of 1


def gamma_q2_series(results, files, df_basedata) -> Optional[Tuple[List[PeakSeries], int]]:
    """Per-peak Γ-vs-q² data of the preview with a straight-line fit in log-log space
    (Γ = D·q² has slope 1 and intercept log10 D). Returns ``(series, n_peak_columns)`` -- the column
    count lets a plot colour peak ``index`` consistently even if a trailing peak has no data --
    or ``None`` if the preview has no usable data."""
    table = preview_gamma_table(results, files, df_basedata)
    if len(table) < 1:
        return None
    table, tau_cols, gamma_cols = with_decay_rates(table)
    if not tau_cols:
        return None
    series = []
    for i, gamma_col in enumerate(gamma_cols):
        if gamma_col not in table.columns:
            continue
        valid = table[gamma_col].notna()
        q2 = table.loc[valid, 'q^2'].values
        gamma = table.loc[valid, gamma_col].values
        if len(q2) == 0:
            continue
        slope = intercept = q2_fit = gamma_fit = d_estimate = None
        if len(q2) >= 2:
            slope, intercept = np.polyfit(np.log10(q2), np.log10(gamma), 1)
            q2_fit = np.logspace(np.log10(q2.min()), np.log10(q2.max()), 50)
            gamma_fit = 10 ** (slope * np.log10(q2_fit) + intercept)
            d_estimate = 10 ** intercept if abs(slope - 1.0) < SLOPE_TOLERANCE else None
        series.append(PeakSeries(i, q2, gamma, slope, intercept, q2_fit, gamma_fit, d_estimate))
    return series, len(gamma_cols)
