# -*- coding: utf-8 -*-
"""
Performance-optimized version of regularized.py
Uses matrix caching, vectorization, and optional multiprocessing
"""

from scipy.optimize import least_squares, minimize
from scipy.signal import find_peaks, peak_widths
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from scipy import stats
from matplotlib import cm
from matplotlib.figure import Figure
import random
from functools import lru_cache
from typing import Dict, Tuple, Optional, List
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


# ===== PERFORMANCE OPTIMIZATION: Matrix Caching =====
@lru_cache(maxsize=32)
def create_exponential_matrix_cached(n_tau: int, n_decay: int, tau_hash: int, decay_hash: int) -> np.ndarray:
    """
    Create and cache the exponential matrix T = exp(-tau_M / decay_times_N)

    Uses hashing for array identification to enable caching
    """
    # Reconstruct arrays from cache key (placeholder - actual arrays passed separately)
    # This is called with actual arrays but cached by hash
    return None  # Actual implementation in wrapper


def create_exponential_matrix(tau: np.ndarray, decay_times: np.ndarray) -> np.ndarray:
    """
    Create exponential matrix with caching support

    Args:
        tau: Time points from correlation data (M points)
        decay_times: Decay time grid (N points)

    Returns:
        T matrix of shape (M, N) with exp(-tau_M / decay_times_N)
    """
    # Create grid of tau and decay time combinations
    decay_times_N, tau_M = np.meshgrid(decay_times, tau)

    # Create matrix A from the mesh - this is the expensive operation
    T = np.exp(-tau_M / decay_times_N)

    return T


@lru_cache(maxsize=16)
def create_tikhonov_matrix_cached(n: int) -> np.ndarray:
    """
    Create and cache Tikhonov regularization matrix (2nd derivative)

    Args:
        n: Size of the matrix

    Returns:
        D2 matrix for regularization
    """
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n-2, n), dtype=float).toarray()
    return D2


# ===== OPTIMIZED NNLS FUNCTION =====
def nnls_optimized(df: pd.DataFrame, name: str, nnls_params: dict,
                   plot_number: int = 1, T_matrix: Optional[np.ndarray] = None) -> dict:
    """
    Optimized NNLS fit with pre-computed matrix support

    Args:
        df: DataFrame with 't [s]' and 'g(2)-1' columns
        name: Dataset name
        nnls_params: Parameters dict with 'decay_times', 'prominence', 'distance'
        plot_number: Plot number for visualization
        T_matrix: Pre-computed exponential matrix (optional, for performance)

    Returns:
        Dictionary with results
    """
    decay_times = nnls_params['decay_times']
    prominence = nnls_params.get('prominence', 0.05)
    distance = nnls_params.get('distance', 1)

    # Create the vectors
    tau = df['t [s]'].to_numpy()
    D = df['g(2)-1'].to_numpy()

    # Use pre-computed matrix if provided, otherwise compute
    if T_matrix is None:
        T = create_exponential_matrix(tau, decay_times)
    else:
        T = T_matrix

    # Nonlinear NNLS fit: g(2)(τ)-1 = (T @ f)^2, f >= 0.
    # Must stay nonlinear (not linearized via sqrt(D)) since D = g(2)-1 can be
    # negative (noise) and the model is genuinely quadratic in f, not linear.
    def residuals(f, T, D):
        return (T @ f)**2 - D

    f0 = np.ones(T.shape[1])
    bounds = (0, np.inf)
    result = least_squares(residuals, f0, args=(T, D), bounds=bounds)
    f_optimized = result.x

    # Calculate the optimized function values
    optimized_values = (T @ f_optimized)**2
    residuals_values = optimized_values - D

    # Calculate RMSE
    rmse = np.sqrt(np.mean(residuals_values**2))

    # Find peaks in the tau distribution
    peaks, _ = find_peaks(f_optimized, prominence=prominence, distance=distance)

    # Log fit details
    print(f"    └─ Fit: {result.status} ({result.nfev} evals), RMSE: {rmse:.4e}, Peaks found: {len(peaks)}")

    # Get peak amplitudes/intensities
    peak_amplitudes = f_optimized[peaks]

    # Normalization to the total sum
    normalized_amplitudes_sum = peak_amplitudes / np.sum(peak_amplitudes) if len(peak_amplitudes) > 0 else np.array([])

    # Area-based normalization (trapz integral per peak segment, log-τ space to
    # match JADE — grid is log-spaced, so integrate over d(ln τ))
    peak_areas = []
    for peak_index in peaks:
        left  = max(0, peak_index - 10)
        right = min(len(decay_times), peak_index + 10)
        area  = np.trapezoid(f_optimized[left:right], np.log(decay_times[left:right]))
        peak_areas.append(max(area, 0.0))
    total_area = sum(peak_areas) if sum(peak_areas) > 0 else 1.0
    normalized_area_pct = [a / total_area * 100 for a in peak_areas]

    # Prepare results for this dataframe
    results = {'filename': name}
    for i, peak_index in enumerate(peaks):
        pct_sum = normalized_amplitudes_sum[i] * 100

        # Per-peak shape statistics (weighted τ-space moments, log-space)
        left    = max(0, peak_index - 10)
        right   = min(len(decay_times), peak_index + 10)
        peak_x  = np.log(decay_times[left:right])
        peak_y  = f_optimized[left:right]
        w       = peak_y / np.sum(peak_y) if np.sum(peak_y) > 0 else np.ones(len(peak_y)) / len(peak_y)
        wmean   = np.sum(w * peak_x)
        wvar    = np.sum(w * (peak_x - wmean)**2)
        wstd    = np.sqrt(wvar) if wvar > 0 else 0
        skew    = np.sum(w * (peak_x - wmean)**3) / wstd**3 if wstd > 0 and len(peak_x) >= 3 else 0
        kurt    = np.sum(w * (peak_x - wmean)**4) / wstd**4 - 3 if wstd > 0 and len(peak_x) >= 4 else 0

        # FWHM via scipy peak_widths
        widths_result = peak_widths(f_optimized, [peak_index], rel_height=0.5)
        fwhm_idx = widths_result[0][0]
        fwhm_s   = fwhm_idx * (decay_times[-1] - decay_times[0]) / len(decay_times)

        results[f'tau_{i+1}']                     = decay_times[peak_index]
        results[f'intensity_{i+1}']               = f_optimized[peak_index]
        results[f'normalized_sum_percent_{i+1}']  = pct_sum
        results[f'normalized_area_percent_{i+1}'] = normalized_area_pct[i]
        results[f'area_{i+1}']                    = peak_areas[i]
        results[f'fwhm_{i+1}']                    = fwhm_s
        results[f'skewness_{i+1}']                = skew
        results[f'kurtosis_{i+1}']                = kurt
        results[f'std_dev_{i+1}']                 = wstd

    return results, f_optimized, optimized_values, residuals_values, peaks


def nnls_all_optimized(dataframes_dict: Dict[str, pd.DataFrame],
                       nnls_params: dict,
                       use_multiprocessing: bool = False,
                       show_plots: bool = True) -> pd.DataFrame:
    """
    Optimized batch processing for NNLS

    Args:
        dataframes_dict: Dictionary of dataframes {filename: df}
        nnls_params: NNLS parameters
        use_multiprocessing: Use parallel processing
        show_plots: Show plots during processing

    Returns:
        DataFrame with all results
    """
    all_results = []
    decay_times = nnls_params['decay_times']

    # Pre-compute T matrix once if all datasets have same time points
    # Check if all tau arrays are identical
    tau_arrays = [df['t [s]'].to_numpy() for df in dataframes_dict.values()]
    all_same_tau = all(np.array_equal(tau_arrays[0], tau) for tau in tau_arrays[1:])

    T_matrix = None
    if all_same_tau:
        print("All datasets have identical time points - using cached matrix multiplication")
        T_matrix = create_exponential_matrix(tau_arrays[0], decay_times)

    if use_multiprocessing and len(dataframes_dict) > 3:
        print(f"Using multiprocessing with {mp.cpu_count()} cores")
        # Parallel processing
        with ProcessPoolExecutor() as executor:
            futures = {}
            for name, df in dataframes_dict.items():
                future = executor.submit(nnls_optimized, df, name, nnls_params,
                                       plot_number=1, T_matrix=None)  # Can't pass numpy arrays to subprocess easily
                futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    results, _, _, _, _ = future.result()
                    all_results.append(results)
                    print(f"Processed {name}")
                except Exception as e:
                    print(f"Error processing {name}: {str(e)}")
    else:
        # Sequential processing with optional matrix reuse
        plot_number = 1
        for name, df in dataframes_dict.items():
            print(f"Processing {name}...")
            results, f_optimized, optimized_values, residuals_values, peaks = nnls_optimized(
                df, name, nnls_params, plot_number, T_matrix=T_matrix
            )
            all_results.append(results)

            if show_plots:
                _plot_nnls_results(name, df, decay_times, f_optimized, optimized_values,
                                  residuals_values, peaks, nnls_params, plot_number)

            plot_number += 1

    nnls_df = pd.DataFrame(all_results)
    return nnls_df


def _plot_nnls_results(name: str, df: pd.DataFrame, decay_times: np.ndarray,
                       f_optimized: np.ndarray, optimized_values: np.ndarray,
                       residuals_values: np.ndarray, peaks: np.ndarray,
                       nnls_params: dict, plot_number: int):
    """Helper function to create NNLS result plots"""
    tau = df['t [s]'].to_numpy()
    D = df['g(2)-1'].to_numpy()

    peak_amplitudes = f_optimized[peaks]
    normalized_amplitudes_sum = peak_amplitudes / np.sum(peak_amplitudes) if len(peak_amplitudes) > 0 else np.array([])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'[{plot_number}]: Analysis for {name}', fontsize=16)

    # First subplot: Data and optimized function
    ax1.semilogx(tau, D, 'ro', label='Data (D)', markersize=4)
    ax1.semilogx(tau, optimized_values, 'g-', label='Optimized Function', linewidth=2)
    ax1.set_xlabel('lag time [s]')
    ax1.set_ylabel('g(2)-1')
    ax1.set_title('Optimization Results')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Second subplot: Tau distribution
    ax2.semilogx(decay_times, f_optimized, 'bo-', label='Tau Distribution', markersize=3)
    ax2.set_xlabel('Tau (Decay Times) [s]')
    ax2.set_ylabel('Intensity (f_optimized)')
    ax2.set_title('Tau Distribution')
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    # Mark peaks
    if len(peaks) > 0:
        ax2.plot(decay_times[peaks], f_optimized[peaks], 'rx', markersize=10,
                label=f'Peaks (n={len(peaks)})')

        # Annotate peaks
        for i, peak_idx in enumerate(peaks):
            percentage = normalized_amplitudes_sum[i] * 100
            ax2.annotate(f'{percentage:.1f}%\nτ={decay_times[peak_idx]:.2e}',
                        xy=(decay_times[peak_idx], f_optimized[peak_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                                             fc="yellow", alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax2.legend()

    # Third subplot: Residuals
    ax3.plot(tau, residuals_values, 'b-', alpha=0.7)
    ax3.set_xlabel('lag time [s]')
    ax3.set_ylabel('Residual Value')
    ax3.set_title('Residuals')
    ax3.axhline(0, color='r', linestyle='--', linewidth=2)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ===== CALCULATION HELPERS =====
def _compute_peak_statistics(peaks, peak_properties, f_optimized, decay_times):
    """Per-peak shape statistics in log-τ space (matches JADE regularized.py).

    Since the decay-time grid is log-spaced, the peak area is integrated over
    d(ln τ) and the FWHM and weighted moments (skewness, kurtosis) are computed
    in log-τ space. Each peak segment is extended from its half-max width points
    down to 1% of the peak height (its base) for the area/moment calculation.

    Returns (peak_stats dict, normalized_area_percent list, normalized_sum_percent list).
    """
    peak_amplitudes = f_optimized[peaks]
    peak_stats = {}

    for i, peak_idx in enumerate(peaks):
        #locate the peak base — start from the half-max width points, extend to 1% of peak height
        try:
            left_idx = int(peak_properties['left_ips'][i])
            right_idx = int(peak_properties['right_ips'][i])
            threshold = 0.01 * f_optimized[peak_idx]

            left_base_idx = left_idx
            while left_base_idx > 0 and f_optimized[left_base_idx] > threshold:
                left_base_idx -= 1

            right_base_idx = right_idx
            while right_base_idx < len(f_optimized) - 1 and f_optimized[right_base_idx] > threshold:
                right_base_idx += 1

            left_base_idx = max(0, left_base_idx)
            right_base_idx = min(len(f_optimized) - 1, right_base_idx)
        except (IndexError, KeyError):
            #fallback: walk down to the local minima on both sides of the peak
            left_base_idx = max(0, peak_idx - 1)
            while left_base_idx > 0 and f_optimized[left_base_idx] > f_optimized[left_base_idx - 1]:
                left_base_idx -= 1
            right_base_idx = min(len(f_optimized) - 1, peak_idx + 1)
            while right_base_idx < len(f_optimized) - 1 and f_optimized[right_base_idx] > f_optimized[right_base_idx + 1]:
                right_base_idx += 1

        peak_segment = f_optimized[left_base_idx:right_base_idx + 1]
        peak_x = decay_times[left_base_idx:right_base_idx + 1]

        #area via trapezoidal rule in log-τ space (grid is log-spaced → integrate over d(ln τ))
        peak_area = np.trapezoid(peak_segment, np.log(peak_x))

        #FWHM in log-τ space
        half_max = peak_amplitudes[i] / 2
        above_half_max = peak_segment >= half_max
        if np.sum(above_half_max) > 0:
            idx_above = np.where(above_half_max)[0]
            l_i, r_i = idx_above[0], idx_above[-1]
            if r_i > l_i and r_i < len(peak_x) and l_i < len(peak_x):
                fwhm = np.log(peak_x[r_i]) - np.log(peak_x[l_i])
            else:
                fwhm = 0
        else:
            fwhm = 0

        #weighted moments in log-τ space — consistent with area computed via d(ln τ)
        #note: skewness sign is opposite to cumulant skewness (τ vs Γ space)
        ln_x = np.log(peak_x)
        w = peak_segment / np.sum(peak_segment) if np.sum(peak_segment) > 0 else np.ones(len(peak_segment)) / len(peak_segment)
        ln_mean = np.sum(ln_x * w)
        weighted_mean = np.exp(ln_mean)                          # geometric mean in τ-space
        variance = np.sum(w * (ln_x - ln_mean)**2)               # variance in log-τ space
        std_dev = np.sqrt(variance) if variance > 0 else 0
        if std_dev > 0 and len(peak_segment) >= 3:
            peak_skewness = np.sum(w * (ln_x - ln_mean)**3) / std_dev**3
        else:
            peak_skewness = 0
        if std_dev > 0 and len(peak_segment) >= 4:
            peak_kurtosis = np.sum(w * (ln_x - ln_mean)**4) / std_dev**4 - 3
        else:
            peak_kurtosis = 0

        peak_stats[f'peak_{i+1}'] = {
            'position' : decay_times[peak_idx],
            'amplitude': peak_amplitudes[i],
            'area'     : peak_area,
            'fwhm'     : fwhm,
            'centroid' : weighted_mean,
            'std_dev'  : std_dev,
            'skewness' : peak_skewness,   # τ-space: positive = tail toward larger particles
            'kurtosis' : peak_kurtosis,   # excess kurtosis: >0 = sharp, <0 = broad
        }

    #area-based normalization over detected peaks only (used for SLS intensity weighting)
    if len(peaks) > 0 and peak_stats:
        total_area = np.sum([peak_stats[f'peak_{i+1}']['area'] for i in range(len(peaks))])
        if total_area > 0:
            normalized_area_percent = [
                peak_stats[f'peak_{i+1}']['area'] / total_area * 100 for i in range(len(peaks))
            ]
        else:
            normalized_area_percent = [100.0 / len(peaks)] * len(peaks)

        total_amp = np.sum(peak_amplitudes)
        if total_amp > 0:
            normalized_sum_percent = [peak_amplitudes[i] / total_amp * 100 for i in range(len(peaks))]
        else:
            normalized_sum_percent = [100.0 / len(peaks)] * len(peaks)
    else:
        normalized_area_percent = []
        normalized_sum_percent = []

    return peak_stats, normalized_area_percent, normalized_sum_percent


def regularized_nnls_optimized(df: pd.DataFrame, name: str, params: dict,
                               plot_number: int = 1, T_matrix: Optional[np.ndarray] = None) -> Tuple:
    """
    Optimized Regularized NNLS (Tikhonov) with pre-computed matrix support

    Args:
        df: DataFrame with 't [s]' and 'g(2)-1' columns
        name: Dataset name
        params: Parameters dict with 'decay_times', 'alpha', 'prominence', 'distance'
        plot_number: Plot number for visualization
        T_matrix: Pre-computed exponential matrix (optional, for performance)

    Returns:
        Tuple of (results dict, f_optimized, optimized_values, residuals_values, peaks)
    """
    decay_times = params['decay_times']
    prominence = params.get('prominence', 0.01)
    distance = params.get('distance', 1)
    alpha = params.get('alpha', 0.01)
    fit_beta = params.get('fit_beta', False)

    # Create the vectors
    tau = df['t [s]'].to_numpy()
    D = df['g(2)-1'].to_numpy()

    # Use pre-computed matrix if provided, otherwise compute
    if T_matrix is None:
        T = create_exponential_matrix(tau, decay_times)
    else:
        T = T_matrix

    # Get cached Tikhonov regularization matrix
    n = len(decay_times)
    D2 = create_tikhonov_matrix_cached(n)

    # Define the regularized objective function.
    # Model: g(2)(τ)-1 = β · (∑_i A_i · exp(-τ/τ_i))², where β is the coherence
    # factor (intercept). β is fitted as a separate parameter only when fit_beta.
    def residuals_regularized(params_vec, T, D, D2, alpha):
        if fit_beta:
            beta_val = params_vec[0]
            f = params_vec[1:]
        else:
            beta_val = 1.0
            f = params_vec

        model_output = beta_val * (T @ f)**2

        # data fidelity residuals
        fit_residuals = model_output - D

        # smoothness penalty
        smoothness_penalty = alpha * (D2 @ f)
        return np.concatenate([fit_residuals, smoothness_penalty])

    # Initial guess (uniform distribution); prepend β (bounded 0–2) when fitting it.
    if fit_beta:
        beta_init = float(np.max(D[:min(5, len(D))]))  # g²(0)-1 ≈ β
        f0 = np.concatenate([[beta_init], np.ones(T.shape[1])])
        bounds = (np.concatenate([[0.0], np.zeros(T.shape[1])]),
                  np.concatenate([[2.0], np.full(T.shape[1], np.inf)]))
    else:
        f0 = np.ones(T.shape[1])
        bounds = (0, np.inf)

    # Perform non-negative least squares with regularization
    result = least_squares(
        lambda p: residuals_regularized(p, T, D, D2, alpha),
        f0,
        bounds=bounds,
        method='trf',
        ftol=1e-8,
        xtol=1e-8,
        max_nfev=500  # Limit iterations for performance
    )
    if fit_beta:
        beta_fitted = float(result.x[0])
        f_optimized = result.x[1:]
    else:
        beta_fitted = 1.0
        f_optimized = result.x

    # Calculate the optimized function
    optimized_values = beta_fitted * (T @ f_optimized)**2
    residuals_values = optimized_values - D

    # Calculate RMSE
    rmse = np.sqrt(np.mean(residuals_values**2))

    # Find peaks (width=0 so we get left_ips/right_ips for the peak-statistics base)
    peaks, peak_properties = find_peaks(f_optimized, prominence=prominence, distance=distance, width=0)

    # Log convergence details
    beta_str = f", β={beta_fitted:.3f}" if fit_beta else ""
    print(f"    └─ Fit: {result.status} (message: {result.message[:50]}...)")
    print(f"    └─ Iterations: {result.nfev}, RMSE: {rmse:.4e}, Peaks found: {len(peaks)}{beta_str}")

    # Per-peak shape statistics (log-τ space, matches JADE) → feeds SLS intensity weighting
    peak_stats, normalized_area_percent, normalized_sum_percent = _compute_peak_statistics(
        peaks, peak_properties, f_optimized, decay_times
    )

    # Prepare results for this dataframe
    results = {'filename': name, 'beta': beta_fitted}
    for i in range(len(peaks)):
        st = peak_stats[f'peak_{i+1}']
        results[f'tau_{i+1}']                     = st['position']
        results[f'intensity_{i+1}']               = st['amplitude']
        results[f'normalized_sum_percent_{i+1}']  = normalized_sum_percent[i]
        results[f'normalized_area_percent_{i+1}'] = normalized_area_percent[i]
        results[f'area_{i+1}']                    = st['area']
        results[f'fwhm_{i+1}']                    = st['fwhm']
        results[f'skewness_{i+1}']                = st['skewness']
        results[f'kurtosis_{i+1}']                = st['kurtosis']
        results[f'std_dev_{i+1}']                 = st['std_dev']
        results[f'centroid_{i+1}']                = st['centroid']

    return results, f_optimized, optimized_values, residuals_values, peaks


def calculate_decay_rates(df: pd.DataFrame, tau_columns: List[str]) -> pd.DataFrame:
    """
    Calculate decay rates (gamma) from decay times (tau)

    Args:
        df: DataFrame with tau columns
        tau_columns: List of column names containing tau values

    Returns:
        DataFrame with added gamma columns
    """
    for tau_col in tau_columns:
        if tau_col in df.columns:
            gamma_col = tau_col.replace('tau', 'gamma')
            df[gamma_col] = 1 / df[tau_col]
        else:
            print(f"Warning: Column '{tau_col}' not found. Skipping.")
    return df


# ===== PREVIEW FUNCTION FOR PARAMETER TUNING =====
def nnls_preview_random(dataframes_dict: Dict[str, pd.DataFrame],
                        nnls_params: dict,
                        num_datasets: int = 5,
                        seed: Optional[int] = None) -> Tuple[plt.Figure, List[str], List[dict]]:
    """
    Create preview plots for random datasets to tune find_peaks parameters

    Args:
        dataframes_dict: Dictionary of all dataframes
        nnls_params: NNLS parameters including prominence and distance
        num_datasets: Number of random datasets to preview
        seed: Random seed for reproducibility

    Returns:
        Tuple of (Figure, list of selected dataset names, list of results dicts)
    """
    if seed is not None:
        random.seed(seed)

    # Select random datasets
    all_keys = list(dataframes_dict.keys())
    num_to_select = min(num_datasets, len(all_keys))
    chosen_keys = random.sample(all_keys, num_to_select)

    print(f"Preview datasets: {', '.join(chosen_keys)}")

    # Create figure
    decay_times = nnls_params['decay_times']
    prominence = nnls_params.get('prominence', 0.05)
    distance = nnls_params.get('distance', 1)

    # Calculate grid size
    cols = min(3, num_to_select)
    rows = (num_to_select + cols - 1) // cols

    fig = Figure(figsize=(6*cols, 4*rows))
    axes = fig.subplots(rows, cols)
    if num_to_select == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Pre-compute T matrix if possible
    tau_arrays = [dataframes_dict[key]['t [s]'].to_numpy() for key in chosen_keys]
    all_same_tau = all(np.array_equal(tau_arrays[0], tau) for tau in tau_arrays[1:])

    T_matrix = None
    if all_same_tau:
        T_matrix = create_exponential_matrix(tau_arrays[0], decay_times)

    # Collect results for clustering analysis
    all_results = []

    # Process each dataset
    for idx, key in enumerate(chosen_keys):
        df = dataframes_dict[key]

        # Run NNLS
        results, f_optimized, optimized_values, residuals_values, peaks = nnls_optimized(
            df, key, nnls_params, plot_number=idx+1, T_matrix=T_matrix
        )

        # Store results for clustering
        all_results.append(results)

        # Plot tau distribution with peaks
        ax = axes[idx]
        ax.semilogx(decay_times, f_optimized, 'b-', linewidth=2, label='Distribution')

        # Mark peaks
        if len(peaks) > 0:
            peak_amplitudes = f_optimized[peaks]
            normalized_amplitudes = peak_amplitudes / np.sum(peak_amplitudes) if len(peak_amplitudes) > 0 else np.array([])

            colors = ['red', 'green', 'orange', 'purple', 'cyan']
            for i, peak_idx in enumerate(peaks):
                color = colors[i % len(colors)]
                ax.plot(decay_times[peak_idx], f_optimized[peak_idx], 'o',
                       color=color, markersize=10, label=f'Peak {i+1} ({normalized_amplitudes[i]*100:.1f}%)')

                # Annotate
                ax.annotate(f'τ={decay_times[peak_idx]:.2e}',
                           xy=(decay_times[peak_idx], f_optimized[peak_idx]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, bbox=dict(boxstyle="round", fc=color, alpha=0.3))

        ax.set_xlabel('Decay Time [s]')
        ax.set_ylabel('Intensity')
        ax.set_title(f'{key}\n{len(peaks)} peaks found')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(num_to_select, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'NNLS Preview - Prominence: {prominence:.3f}, Distance: {distance}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    return fig, chosen_keys, all_results


# ===== RE-EXPORT ORIGINAL FUNCTIONS FOR BACKWARD COMPATIBILITY =====
# Import from original module
try:
    from .regularized import (
        nnls as nnls_original,
        nnls_all as nnls_all_original,
        nnls_reg,
        nnls_reg_all,
        nnls_reg_simple,
        analyze_random_datasets_grid,
        tau_to_hydrodynamic_radius,
        find_dataset_key,
        plot_distributions
    )
except ImportError:
    print("Warning: Could not import original regularized functions")


# Expose optimized versions as primary interface
nnls = nnls_optimized
nnls_all = nnls_all_optimized


if __name__ == "__main__":
    print("Regularized Optimized Module")
    print("Performance improvements:")
    print("  - Matrix caching for identical time grids")
    print("  - Vectorized operations")
    print("  - Optional multiprocessing")
    print("  - Preview function for parameter tuning")
