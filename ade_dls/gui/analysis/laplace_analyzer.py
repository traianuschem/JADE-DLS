"""
Laplace Transform Analysis (NNLS and Regularized Fits)
Handles inverse Laplace transform methods for DLS data
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from matplotlib.figure import Figure

# Lazy imports to avoid Windows multiprocessing issues
# These modules will be imported inside functions as needed


class LaplaceAnalyzer:
    """
    Analyzer for Inverse Laplace Transform methods (NNLS and Regularized)

    Handles:
        - NNLS (Non-Negative Least Squares)
        - Regularized NNLS with Tikhonov-Phillips regularization
        - Peak detection and analysis
        - Diffusion coefficient determination
        - Multi-population analysis
    """

    def __init__(self, processed_correlations: Dict[str, pd.DataFrame],
                 df_basedata: pd.DataFrame,
                 c: float, delta_c: float,
                 raw_correlations: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize the Laplace analyzer

        Args:
            processed_correlations: Dictionary of correlation data {filename: df}
                                   Each df must have 't (s)' and 'g(2)' columns
                                   If None and raw_correlations provided, will process them
            df_basedata: Base data with experimental parameters
            c: Pre-calculated constant for Rh calculation (kB*T / 6*pi*eta)
            delta_c: Error in c
            raw_correlations: Optional raw correlation data (will be processed if processed_correlations is None)
        """
        # If processed correlations not provided, process raw correlations
        if processed_correlations is None and raw_correlations is not None:
            print("Processing raw correlation data...")
            self.processed_correlations = self._process_raw_correlations(raw_correlations)
        else:
            self.processed_correlations = processed_correlations

        self.df_basedata = df_basedata
        self.c = c
        self.delta_c = delta_c

        # Results storage
        self.nnls_results = None
        self.nnls_data = None
        self.nnls_diff_results = None
        self.nnls_final_results = None

        self.regularized_results = None
        self.regularized_data = None
        self.regularized_diff_results = None
        self.regularized_final_results = None
        self.regularized_full_results = None  # For distribution plots

        # Parameters used
        self.nnls_params = None
        self.regularized_params = None

        # SLS analysis results
        self.df_intensity = None
        self.sls_data = None
        self.guinier_results = None
        self.guinier_total = None
        self.sls_summary = None

    def _process_raw_correlations(self, raw_correlations: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process raw correlation data to format needed for NNLS/Regularized fits

        Converts from ALV format (time [ms], correlation 1-4) to
        processed format (t (s), g(2))

        Args:
            raw_correlations: Dictionary with raw correlation data

        Returns:
            Dictionary with processed correlation data {filename: df with 't (s)' and 'g(2)'}
        """
        from ade_dls.core.preprocessing import process_correlation_data

        # Columns to drop (keep only time and mean correlation)
        columns_to_drop = ['time [ms]', 'correlation 1', 'correlation 2',
                          'correlation 3', 'correlation 4']

        processed = process_correlation_data(raw_correlations, columns_to_drop)

        print(f"Processed {len(processed)} correlation datasets")
        return processed

    # ===== NNLS METHODS =====

    def run_nnls(self, params: dict, use_multiprocessing: bool = False,
                 show_plots: bool = True) -> pd.DataFrame:
        """
        Run NNLS analysis on all datasets

        Args:
            params: Dictionary with NNLS parameters:
                - decay_times: np.ndarray of decay times (logspace)
                - prominence: Peak detection prominence threshold
                - distance: Minimum distance between peaks
            use_multiprocessing: Use parallel processing
            show_plots: Show plots during processing

        Returns:
            DataFrame with NNLS results (tau values, intensities, percentages)
        """
        self.nnls_params = params.copy()

        print("\n" + "="*60)
        print("NNLS (NON-NEGATIVE LEAST SQUARES) ANALYSIS")
        print("="*60)
        print(f"Number of datasets: {len(self.processed_correlations)}")
        print(f"Peak detection prominence: {params.get('prominence', 0.05)}")
        print(f"Peak detection distance: {params.get('distance', 1)}")
        print(f"Multiprocessing: {'Enabled' if use_multiprocessing else 'Disabled'}")
        print("="*60)

        # Run NNLS analysis and collect plots
        self.nnls_results, self.nnls_plots = self._run_nnls_with_plots(
            params,
            use_multiprocessing=use_multiprocessing,
            show_plots=show_plots
        )

        # Merge with basedata
        self.nnls_data = pd.merge(self.df_basedata, self.nnls_results,
                                 on='filename', how='outer')
        self.nnls_data = self.nnls_data.reset_index(drop=True)
        self.nnls_data.index = self.nnls_data.index + 1

        print(f"\nNNLS analysis complete. Processed {len(self.nnls_results)} datasets.")
        print(f"Generated {len(self.nnls_plots)} fit plots.")
        print("="*60 + "\n")

        return self.nnls_data

    def _run_nnls_with_plots(self, params: dict, use_multiprocessing: bool = False,
                             show_plots: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """
        Internal method to run NNLS and collect plots without showing them

        Args:
            params: NNLS parameters
            use_multiprocessing: Use parallel processing
            show_plots: Show plots during processing (for debugging)

        Returns:
            Tuple of (results DataFrame, plots dictionary)
        """
        import matplotlib.pyplot as plt
        from ade_dls.analysis.regularized_optimized import nnls_optimized, create_exponential_matrix

        all_results = []
        plots_dict = {}
        decay_times = params['decay_times']

        # Pre-compute T matrix once if all datasets have same time points
        tau_arrays = [df['t [s]'].to_numpy() for df in self.processed_correlations.values()]
        all_same_tau = all(np.array_equal(tau_arrays[0], tau) for tau in tau_arrays[1:])

        T_matrix = None
        if all_same_tau:
            print("[NNLS] All datasets have identical time points - using cached matrix")
            T_matrix = create_exponential_matrix(tau_arrays[0], decay_times)
            print("[NNLS] Matrix cache enabled - significant speedup expected")

        total = len(self.processed_correlations)

        # Use multiprocessing if requested and we have enough datasets
        if use_multiprocessing and total > 3:
            fit_results = {}  # Initialize outside try block
            multiprocessing_success = False

            try:
                from joblib import Parallel, delayed
                import multiprocessing as mp

                n_jobs = mp.cpu_count()
                print(f"[NNLS] Using parallel processing with {n_jobs} CPU cores (joblib backend)")
                print("[NNLS] Phase 1: Parallel fitting...")

                # Process in parallel using joblib (better Windows support than multiprocessing)
                # Use 'loky' backend which is more robust for scipy/numpy operations
                # Pass T_matrix to each worker to avoid recomputation
                results_list = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
                    delayed(nnls_optimized)(df, name, params, 1, T_matrix)
                    for name, df in self.processed_correlations.items()
                )

                # Collect results
                for idx, (name, df) in enumerate(self.processed_correlations.items()):
                    results, f_optimized, optimized_values, residuals_values, peaks = results_list[idx]
                    all_results.append(results)
                    fit_results[name] = (f_optimized, optimized_values, residuals_values, peaks)
                    print(f"[{idx+1}/{total}] Completed {name}")

                multiprocessing_success = True

            except ImportError as e:
                print(f"[NNLS] Warning: joblib not available ({e})")
                print("[NNLS] Install required packages: pip install joblib scikit-learn")
                print("[NNLS] Falling back to sequential processing")
                use_multiprocessing = False
            except Exception as e:
                print(f"[NNLS] Warning: Parallel processing failed: {type(e).__name__}: {str(e)}")
                print("[NNLS] Falling back to sequential processing")
                use_multiprocessing = False
                import traceback
                traceback.print_exc()

            # Only do Phase 2 if multiprocessing succeeded
            if multiprocessing_success:
                print("[NNLS] Phase 2: Generating plots sequentially...")
                # Generate plots sequentially (can't be done in parallel)
                plot_number = 1
                for name, df in self.processed_correlations.items():
                    if name in fit_results:
                        f_optimized, optimized_values, residuals_values, peaks = fit_results[name]

                        fig = self._create_nnls_plot(
                            name, df, decay_times, f_optimized, optimized_values,
                            residuals_values, peaks, params, plot_number
                        )

                        plots_dict[name] = (fig, {
                            'f_optimized': f_optimized,
                            'peaks': peaks,
                            'num_peaks': len(peaks)
                        })

                        if show_plots:
                            plt.show()
                        else:
                            plt.close(fig)

                        plot_number += 1

        if not use_multiprocessing or total <= 3:
            # Sequential processing
            if use_multiprocessing and total <= 3:
                print("[NNLS] Too few datasets for multiprocessing, using sequential processing")

            plot_number = 1
            for idx, (name, df) in enumerate(self.processed_correlations.items(), 1):
                print(f"[{idx}/{total}] Fitting {name}...")

                # Run NNLS fit
                results, f_optimized, optimized_values, residuals_values, peaks = nnls_optimized(
                    df, name, params, plot_number, T_matrix=T_matrix
                )
                all_results.append(results)

                # Create plot (don't show it)
                fig = self._create_nnls_plot(
                    name, df, decay_times, f_optimized, optimized_values,
                    residuals_values, peaks, params, plot_number
                )

                # Store plot
                plots_dict[name] = (fig, {
                    'f_optimized': f_optimized,
                    'peaks': peaks,
                    'num_peaks': len(peaks)
                })

                if show_plots:
                    plt.show()
                else:
                    plt.close(fig)

                plot_number += 1

        nnls_df = pd.DataFrame(all_results)
        return nnls_df, plots_dict

    def _create_nnls_plot(self, name: str, df: pd.DataFrame, decay_times: np.ndarray,
                         f_optimized: np.ndarray, optimized_values: np.ndarray,
                         residuals_values: np.ndarray, peaks: np.ndarray,
                         nnls_params: dict, plot_number: int) -> Figure:
        """
        Create NNLS result plot - shows only the distribution

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        peak_amplitudes = f_optimized[peaks]
        normalized_amplitudes_sum = peak_amplitudes / np.sum(peak_amplitudes) if len(peak_amplitudes) > 0 else np.array([])

        # Create single plot focused on distribution
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        fig.suptitle(f'NNLS Analysis - {name}', fontsize=14, fontweight='bold')

        # Plot tau distribution
        ax.semilogx(decay_times, f_optimized, 'b-', linewidth=2.5, label='Intensity Distribution')
        ax.fill_between(decay_times, 0, f_optimized, alpha=0.3, color='blue')
        ax.set_xlabel('Decay Time τ [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Intensity f(τ)', fontsize=12, fontweight='bold')
        ax.set_title(f'Decay Time Distribution ({len(peaks)} peaks detected)', fontsize=12)
        ax.grid(True, which="both", ls="--", alpha=0.3)

        # Mark peaks
        if len(peaks) > 0:
            ax.plot(decay_times[peaks], f_optimized[peaks], 'r*', markersize=15,
                    label=f'Detected Peaks', markeredgecolor='darkred', markeredgewidth=1.5)

            # Annotate peaks with percentage and tau value
            for i, peak_idx in enumerate(peaks):
                percentage = normalized_amplitudes_sum[i] * 100
                ax.annotate(f'{percentage:.1f}%\nτ={decay_times[peak_idx]:.2e}s',
                            xy=(decay_times[peak_idx], f_optimized[peak_idx]),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=9, bbox=dict(boxstyle="round,pad=0.4",
                                                 fc="yellow", ec="orange", alpha=0.8),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                          lw=1.5, color='darkred'))

        # Add quality metrics
        rmse = np.sqrt(np.mean(residuals_values**2))
        ax.text(0.98, 0.98, f'RMSE: {rmse:.4e}',
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='orange'),
                fontsize=10)

        ax.legend(loc='best', fontsize=10)
        fig.tight_layout()
        return fig

    def preview_nnls_parameters(self, params: dict, num_datasets: int = 5,
                               seed: Optional[int] = None) -> Tuple[any, List[str], List[dict]]:
        """
        Preview NNLS results on random datasets for parameter tuning

        Args:
            params: NNLS parameters to test
            num_datasets: Number of random datasets to show
            seed: Random seed for reproducibility

        Returns:
            Tuple of (matplotlib Figure, list of selected dataset names, list of results dicts)
        """
        from ade_dls.analysis.regularized_optimized import nnls_preview_random
        return nnls_preview_random(self.processed_correlations, params,
                                  num_datasets=num_datasets, seed=seed)

    def calculate_nnls_diffusion_coefficients(self, tau_columns: Optional[List[str]] = None,
                                              x_range: Optional[Tuple[float, float]] = None,
                                              per_pop_ranges: Optional[dict] = None,
                                              use_clustering: bool = True,
                                              distance_threshold: float = 2.0,
                                              clustering_strategy: str = 'silhouette_refined',
                                              use_robust_regression: bool = True,
                                              robust_method: str = 'ransac') -> pd.DataFrame:
        """
        Calculate diffusion coefficients from NNLS tau values using Ward hierarchical clustering.

        Args:
            tau_columns: List of tau column names (e.g., ['tau_1', 'tau_2'])
                        If None, auto-detects all tau columns
            x_range: Optional q² range for fitting (applied to all populations)
            per_pop_ranges: Optional dict {pop_num: (q_min, q_max)} for per-population ranges
            use_clustering: If True, cluster peaks across angles (Ward) before regression
            distance_threshold: Log-space distance threshold for Ward clustering
            clustering_strategy: 'simple', 'silhouette_refined', etc.
            use_robust_regression: If True, use robust regression (RANSAC, Theil-Sen, etc.)
            robust_method: 'ransac', 'theil-sen', or 'huber'

        Returns:
            DataFrame with diffusion coefficients for each peak
        """
        if self.nnls_data is None:
            raise ValueError("Must run NNLS analysis first")

        # Step 1: Auto-detect tau columns
        if tau_columns is None:
            tau_columns = [col for col in self.nnls_data.columns if col.startswith('tau_')]

        print(f"\n[NNLS] Calculating diffusion coefficients for {len(tau_columns)} peaks...")

        # Step 2: Compute gamma from tau (required as input to cluster_all_gammas)
        from ade_dls.analysis.regularized_optimized import calculate_decay_rates
        self.nnls_data = calculate_decay_rates(self.nnls_data, tau_columns)
        gamma_columns = [col.replace('tau', 'gamma') for col in tau_columns]

        # Step 3: Ward hierarchical clustering (replaces DBSCAN)
        if use_clustering:
            print(f"\n[NNLS] Performing Ward hierarchical clustering...")
            from ade_dls.analysis.clustering import cluster_all_gammas
            self.nnls_data, cluster_info = cluster_all_gammas(
                self.nnls_data,
                gamma_cols=gamma_columns,
                q_squared_col='q^2',
                enable_clustering=True,
                normalize_by_q2=True,
                distance_threshold=distance_threshold,
                clustering_strategy=clustering_strategy,
                interactive=False,
                plot=True,
                experiment_name='NNLS'
            )
            self.nnls_cluster_info = cluster_info
            self.nnls_clustering_plot = cluster_info.get('clustering_plot')
            # Use Ward-assigned population columns for regression
            pop_columns = sorted([c for c in self.nnls_data.columns if c.startswith('gamma_pop')])
            if not pop_columns:
                pop_columns = gamma_columns  # fallback
        else:
            pop_columns = gamma_columns

        # Step 4: Analyze diffusion coefficient via Γ vs q² regression
        # If per_pop_ranges is provided, fit each population individually with its own range
        if per_pop_ranges:
            all_results = []
            if use_robust_regression:
                from ade_dls.analysis.peak_clustering import analyze_diffusion_coefficient_robust
                for i, col in enumerate(pop_columns):
                    pop_range = per_pop_ranges.get(i + 1, x_range)
                    res = analyze_diffusion_coefficient_robust(
                        data_df=self.nnls_data,
                        q_squared_col='q^2',
                        gamma_cols=[col],
                        x_range=pop_range,
                        robust_method=robust_method,
                        show_plots=False
                    )
                    all_results.append(res)
            else:
                from ade_dls.analysis.cumulants import analyze_diffusion_coefficient
                for i, col in enumerate(pop_columns):
                    pop_range = per_pop_ranges.get(i + 1, x_range)
                    res = analyze_diffusion_coefficient(
                        data_df=self.nnls_data,
                        q_squared_col='q^2',
                        gamma_cols=[col],
                        x_range=pop_range,
                        show_plots=False
                    )
                    all_results.append(res)
            self.nnls_diff_results = pd.concat(all_results, ignore_index=True)
        elif use_robust_regression:
            from ade_dls.analysis.peak_clustering import analyze_diffusion_coefficient_robust
            print(f"[NNLS] Using robust regression: {robust_method.upper()}")
            self.nnls_diff_results = analyze_diffusion_coefficient_robust(
                data_df=self.nnls_data,
                q_squared_col='q^2',
                gamma_cols=pop_columns,
                x_range=x_range,
                robust_method=robust_method,
                show_plots=False
            )
        else:
            from ade_dls.analysis.cumulants import analyze_diffusion_coefficient
            self.nnls_diff_results = analyze_diffusion_coefficient(
                data_df=self.nnls_data,
                q_squared_col='q^2',
                gamma_cols=pop_columns,
                x_range=x_range,
                show_plots=False
            )

        # Step 5: Create summary plot (Gamma vs q²)
        print(f"[NNLS] Creating Gamma vs q² summary plot...")
        self._create_nnls_summary_plot()

        # Step 6: Calculate Rh for each peak
        self._calculate_nnls_final_results()

        print(f"[NNLS] Diffusion coefficient analysis complete.")
        return self.nnls_final_results

    def _create_nnls_summary_plot(self):
        """Create summary plot for NNLS diffusion analysis (Gamma vs q²)"""
        from ade_dls.gui.analysis.cumulant_plotting import create_summary_plot

        # Prefer Ward-clustered population columns; fall back to raw gamma columns
        gamma_cols = sorted([c for c in self.nnls_data.columns if c.startswith('gamma_pop')])
        if not gamma_cols:
            gamma_cols = sorted([c for c in self.nnls_data.columns
                                  if c.startswith('gamma_') and not c.startswith('gamma_pop')])

        # Create peak labels
        peak_labels = [f'NNLS Pop {i+1}' for i in range(len(gamma_cols))]

        # Create the summary plot
        self.nnls_summary_plot = create_summary_plot(
            self.nnls_data,
            'q^2',
            gamma_cols,
            peak_labels,
            '1/s'
        )

    def _calculate_nnls_final_results(self, append_mode=False, label_suffix=""):
        """
        Internal: Calculate final NNLS results with Rh values

        Args:
            append_mode: If True, append to existing results instead of replacing
            label_suffix: Suffix to add to result labels (e.g., " (Post-Refined)")
        """
        if self.nnls_diff_results is None:
            raise ValueError("Must calculate diffusion coefficients first")

        temp_results = []
        for i in range(len(self.nnls_diff_results)):
            # Skip if values are NaN
            if pd.isna(self.nnls_diff_results['q^2_coef'][i]):
                print(f"[NNLS] Skipping peak {i+1} - invalid diffusion coefficient")
                continue

            # Convert D from nm²/s to m²/s
            D_m2s = self.nnls_diff_results['q^2_coef'][i] * 1e-18
            D_err_m2s = self.nnls_diff_results['q^2_se'][i] * 1e-18

            # Calculate Rh
            Rh_nm = self.c.values[0] * (1 / D_m2s) * 1e9

            # Error propagation
            fractional_error = np.sqrt(
                (self.delta_c.values[0] / self.c.values[0])**2 +
                (D_err_m2s / D_m2s)**2
            )
            Rh_error_nm = fractional_error * Rh_nm

            # Create label with optional suffix
            fit_label = f'NNLS Peak {i+1}{label_suffix}'

            # Aggregate per-peak shape statistics from per-file data
            peak_num = i + 1
            def _col_mean(col):
                return self.nnls_data[col].mean(skipna=True) if col in self.nnls_data.columns else np.nan

            result = pd.DataFrame({
                'Rh [nm]':         [Rh_nm],
                'Rh error [nm]':   [Rh_error_nm],
                'D [m^2/s]':       [D_m2s],
                'D error [m^2/s]': [D_err_m2s],
                'R_squared':       [self.nnls_diff_results['R_squared'][i]],
                'Fit':             [fit_label],
                'Residuals':       [np.nan],
                'PDI':             [np.nan],
                'Skewness':        [_col_mean(f'skewness_{peak_num}')],
                'Kurtosis':        [_col_mean(f'kurtosis_{peak_num}')],
                'Abundance [%]':   [_col_mean(f'normalized_area_percent_{peak_num}')],
            })
            temp_results.append(result)

        # Check if we have valid results
        if len(temp_results) == 0:
            print("[NNLS] Warning: No valid results to finalize. Creating empty DataFrame.")
            new_results = pd.DataFrame(columns=[
                'Rh [nm]', 'Rh error [nm]', 'D [m^2/s]', 'D error [m^2/s]',
                'R_squared', 'Fit', 'Residuals', 'PDI', 'Skewness', 'Kurtosis', 'Abundance [%]'
            ])
        else:
            new_results = pd.concat(temp_results, ignore_index=True)

        # Append or replace based on mode
        if append_mode and self.nnls_final_results is not None and not self.nnls_final_results.empty:
            print(f"[NNLS] Appending {len(new_results)} new results to {len(self.nnls_final_results)} existing results")
            self.nnls_final_results = pd.concat([self.nnls_final_results, new_results], ignore_index=True)
        else:
            self.nnls_final_results = new_results

    def remove_nnls_outliers(self, indices_to_remove: List[int]):
        """
        Remove outlier datasets from NNLS data

        Args:
            indices_to_remove: List of row indices to remove
        """
        if self.nnls_data is None:
            raise ValueError("No NNLS data to filter")

        self.nnls_data = self.nnls_data.drop(indices_to_remove)
        self.nnls_data = self.nnls_data.reset_index(drop=True)
        self.nnls_data.index = self.nnls_data.index + 1

        print(f"Removed {len(indices_to_remove)} outliers. {len(self.nnls_data)} datasets remaining.")

    # ===== REGULARIZED FIT METHODS =====

    def run_regularized(self, params: dict, use_multiprocessing: bool = False,
                       show_plots: bool = False) -> Tuple[pd.DataFrame, dict]:
        """
        Run regularized NNLS analysis

        Args:
            params: Dictionary with regularized fit parameters:
                - decay_times: np.ndarray of decay times
                - prominence: Peak detection prominence
                - distance: Minimum distance between peaks
                - alpha: Regularization parameter
                - normalize: Whether to normalize distribution
                - sparsity_penalty: L1 penalty factor
                - enforce_unimodality: Force single peak
            use_multiprocessing: Use parallel processing
            show_plots: Show plots during processing

        Returns:
            Tuple of (results DataFrame, full_results dict with distributions)
        """
        self.regularized_params = params.copy()

        print("\n" + "="*60)
        print("REGULARIZED NNLS ANALYSIS (Tikhonov-Phillips)")
        print("="*60)
        print(f"Number of datasets: {len(self.processed_correlations)}")
        print(f"Alpha (regularization): {params.get('alpha', 1.0)}")
        print(f"Peak detection prominence: {params.get('prominence', 0.05)}")
        print(f"Peak detection distance: {params.get('distance', 1)}")
        print(f"Multiprocessing: {'Enabled' if use_multiprocessing else 'Disabled'}")
        print("="*60)

        # Run regularized analysis and collect plots
        self.regularized_results, self.regularized_full_results, self.regularized_plots = self._run_regularized_with_plots(
            params,
            use_multiprocessing=use_multiprocessing,
            show_plots=show_plots
        )

        # Merge with basedata
        self.regularized_data = pd.merge(self.df_basedata, self.regularized_results,
                                        on='filename', how='outer')
        self.regularized_data = self.regularized_data.reset_index(drop=True)
        self.regularized_data.index = self.regularized_data.index + 1

        print(f"\nRegularized analysis complete. Processed {len(self.regularized_results)} datasets.")
        print(f"Generated {len(self.regularized_plots)} fit plots.")
        print("="*60 + "\n")

        return self.regularized_data, self.regularized_full_results

    def _run_regularized_with_plots(self, params: dict, use_multiprocessing: bool = False,
                                     show_plots: bool = False) -> Tuple[pd.DataFrame, dict, Dict]:
        """
        Internal method to run regularized NNLS and collect plots

        Args:
            params: Regularized NNLS parameters
            use_multiprocessing: Use parallel processing
            show_plots: Show plots during processing

        Returns:
            Tuple of (results DataFrame, full_results dict, plots dictionary)
        """
        import matplotlib.pyplot as plt
        from ade_dls.analysis.regularized_optimized import regularized_nnls_optimized, create_exponential_matrix

        all_results = []
        plots_dict = {}
        full_results = {}
        decay_times = params['decay_times']

        # Pre-compute T matrix once if all datasets have same time points
        tau_arrays = [df['t [s]'].to_numpy() for df in self.processed_correlations.values()]
        all_same_tau = all(np.array_equal(tau_arrays[0], tau) for tau in tau_arrays[1:])

        T_matrix = None
        if all_same_tau:
            print("[Regularized] All datasets have identical time points - using cached matrix")
            T_matrix = create_exponential_matrix(tau_arrays[0], decay_times)
            print("[Regularized] Matrix cache enabled - significant speedup expected")

        total = len(self.processed_correlations)

        # Use multiprocessing if requested and we have enough datasets
        if use_multiprocessing and total > 3:
            fit_results = {}  # Initialize outside try block
            multiprocessing_success = False

            try:
                from joblib import Parallel, delayed
                import multiprocessing as mp

                n_jobs = mp.cpu_count()
                print(f"[Regularized] Using parallel processing with {n_jobs} CPU cores (joblib backend)")
                print("[Regularized] Phase 1: Parallel fitting...")

                # Process in parallel using joblib (better Windows support than multiprocessing)
                # Use 'loky' backend which is more robust for scipy/numpy operations
                # Pass T_matrix to each worker to avoid recomputation
                results_list = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
                    delayed(regularized_nnls_optimized)(df, name, params, 1, T_matrix)
                    for name, df in self.processed_correlations.items()
                )

                # Collect results
                for idx, (name, df) in enumerate(self.processed_correlations.items()):
                    results, f_optimized, optimized_values, residuals_values, peaks = results_list[idx]
                    all_results.append(results)
                    fit_results[name] = (f_optimized, optimized_values, residuals_values, peaks)

                    # Store distribution for later angle comparison
                    full_results[name] = {
                        'decay_times': decay_times,
                        'distribution': f_optimized,
                        'peaks': peaks
                    }

                    print(f"[{idx+1}/{total}] Completed {name} (Regularized)")

                multiprocessing_success = True

            except ImportError as e:
                print(f"[Regularized] Warning: joblib not available ({e})")
                print("[Regularized] Install required packages: pip install joblib scikit-learn")
                print("[Regularized] Falling back to sequential processing")
                use_multiprocessing = False
            except Exception as e:
                print(f"[Regularized] Warning: Parallel processing failed: {type(e).__name__}: {str(e)}")
                print("[Regularized] Falling back to sequential processing")
                use_multiprocessing = False
                import traceback
                traceback.print_exc()

            # Only do Phase 2 if multiprocessing succeeded
            if multiprocessing_success:
                print("[Regularized] Phase 2: Generating plots sequentially...")
                # Generate plots sequentially (can't be done in parallel)
                plot_number = 1
                for name, df in self.processed_correlations.items():
                    if name in fit_results:
                        f_optimized, optimized_values, residuals_values, peaks = fit_results[name]

                        fig = self._create_regularized_plot(
                            name, df, decay_times, f_optimized, optimized_values,
                            residuals_values, peaks, params, plot_number
                        )

                        plots_dict[name] = (fig, {
                            'f_optimized': f_optimized,
                            'peaks': peaks,
                            'num_peaks': len(peaks)
                        })

                        if show_plots:
                            plt.show()
                        else:
                            plt.close(fig)

                        plot_number += 1

        if not use_multiprocessing or total <= 3:
            # Sequential processing
            if use_multiprocessing and total <= 3:
                print("[Regularized] Too few datasets for multiprocessing, using sequential processing")

            plot_number = 1
            for idx, (name, df) in enumerate(self.processed_correlations.items(), 1):
                print(f"[{idx}/{total}] Fitting {name} (Regularized)...")

                # Run regularized NNLS fit with optimized function
                try:
                    results, f_optimized, optimized_values, residuals_values, peaks = regularized_nnls_optimized(
                        df, name, params, plot_number, T_matrix=T_matrix
                    )
                    all_results.append(results)

                    # Store distribution for later angle comparison
                    full_results[name] = {
                        'decay_times': decay_times,
                        'distribution': f_optimized,
                        'peaks': peaks
                    }

                    # Create plot (don't show it)
                    fig = self._create_regularized_plot(
                        name, df, decay_times, f_optimized, optimized_values,
                        residuals_values, peaks, params, plot_number
                    )

                    # Store plot
                    plots_dict[name] = (fig, {
                        'f_optimized': f_optimized,
                        'peaks': peaks,
                        'num_peaks': len(peaks)
                    })

                    if show_plots:
                        plt.show()
                    else:
                        plt.close(fig)

                except Exception as e:
                    print(f"  Error processing {name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                plot_number += 1

        regularized_df = pd.DataFrame(all_results)
        return regularized_df, full_results, plots_dict

    def _create_regularized_plot(self, name: str, df: pd.DataFrame, decay_times: np.ndarray,
                                f_optimized: np.ndarray, optimized_values: np.ndarray,
                                residuals_values: np.ndarray, peaks: np.ndarray,
                                params: dict, plot_number: int) -> Figure:
        """
        Create Regularized NNLS result plot - shows only the distribution
        (matching NNLS display style)

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        peak_amplitudes = f_optimized[peaks]
        normalized_amplitudes_sum = peak_amplitudes / np.sum(peak_amplitudes) if len(peak_amplitudes) > 0 else np.array([])

        # Create single plot focused on distribution (like NNLS)
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        alpha_val = params.get('alpha', 1.0)
        fig.suptitle(f'Regularized NNLS Analysis (α={alpha_val:.2f}) - {name}', fontsize=14, fontweight='bold')

        # Plot tau distribution
        ax.semilogx(decay_times, f_optimized, 'b-', linewidth=2.5, label='Intensity Distribution')
        ax.fill_between(decay_times, 0, f_optimized, alpha=0.3, color='blue')
        ax.set_xlabel('Decay Time τ [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Intensity f(τ)', fontsize=12, fontweight='bold')
        ax.set_title(f'Decay Time Distribution ({len(peaks)} peaks detected)', fontsize=12)
        ax.grid(True, which="both", ls="--", alpha=0.3)

        # Mark peaks
        if len(peaks) > 0:
            ax.plot(decay_times[peaks], f_optimized[peaks], 'r*', markersize=15,
                    label=f'Detected Peaks', markeredgecolor='darkred', markeredgewidth=1.5)

            # Annotate peaks with percentage and tau value
            for i, peak_idx in enumerate(peaks):
                percentage = normalized_amplitudes_sum[i] * 100
                ax.annotate(f'{percentage:.1f}%\nτ={decay_times[peak_idx]:.2e}s',
                            xy=(decay_times[peak_idx], f_optimized[peak_idx]),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=9, bbox=dict(boxstyle="round,pad=0.4",
                                                 fc="yellow", ec="orange", alpha=0.8),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                          lw=1.5, color='darkred'))

        # Add quality metrics
        rmse = np.sqrt(np.mean(residuals_values**2))
        ax.text(0.98, 0.98, f'RMSE: {rmse:.4e}\nα={alpha_val:.2f}',
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='orange'),
                fontsize=10)

        ax.legend(loc='best', fontsize=10)
        fig.tight_layout()
        return fig

    def calculate_regularized_diffusion_coefficients(self,
                                                    tau_columns: Optional[List[str]] = None,
                                                    x_range: Optional[Tuple[float, float]] = None,
                                                    per_pop_ranges: Optional[dict] = None,
                                                    use_clustering: bool = True,
                                                    distance_threshold: float = 2.0,
                                                    clustering_strategy: str = 'silhouette_refined',
                                                    use_robust_regression: bool = True,
                                                    robust_method: str = 'ransac') -> pd.DataFrame:
        """
        Calculate diffusion coefficients from regularized fit tau values using Ward hierarchical clustering.

        Args:
            tau_columns: List of tau column names
            x_range: Optional q² range for fitting (applied to all populations)
            per_pop_ranges: Optional dict {pop_num: (q_min, q_max)} for per-population ranges
            use_clustering: If True, cluster peaks across angles (Ward) before regression
            distance_threshold: Log-space distance threshold for Ward clustering
            clustering_strategy: 'simple', 'silhouette_refined', etc.
            use_robust_regression: If True, use robust regression (RANSAC, Theil-Sen, etc.)
            robust_method: 'ransac', 'theil-sen', or 'huber'

        Returns:
            DataFrame with diffusion coefficients
        """
        if self.regularized_data is None:
            raise ValueError("Must run regularized analysis first")

        # Step 1: Auto-detect tau columns
        if tau_columns is None:
            tau_columns = [col for col in self.regularized_data.columns if col.startswith('tau_')]

        print(f"\n[Regularized] Calculating diffusion coefficients for {len(tau_columns)} peaks...")

        # Step 2: Compute gamma from tau (required as input to cluster_all_gammas)
        from ade_dls.analysis.regularized_optimized import calculate_decay_rates
        self.regularized_data = calculate_decay_rates(self.regularized_data, tau_columns)
        gamma_columns = [col.replace('tau', 'gamma') for col in tau_columns]

        # Step 3: Ward hierarchical clustering (replaces DBSCAN)
        if use_clustering:
            print(f"\n[Regularized] Performing Ward hierarchical clustering...")
            from ade_dls.analysis.clustering import cluster_all_gammas
            self.regularized_data, cluster_info = cluster_all_gammas(
                self.regularized_data,
                gamma_cols=gamma_columns,
                q_squared_col='q^2',
                enable_clustering=True,
                normalize_by_q2=True,
                distance_threshold=distance_threshold,
                clustering_strategy=clustering_strategy,
                interactive=False,
                plot=True,
                experiment_name='Regularized'
            )
            self.regularized_cluster_info = cluster_info
            self.regularized_clustering_plot = cluster_info.get('clustering_plot')
            # Use Ward-assigned population columns for regression
            pop_columns = sorted([c for c in self.regularized_data.columns if c.startswith('gamma_pop')])
            if not pop_columns:
                pop_columns = gamma_columns  # fallback
        else:
            pop_columns = gamma_columns

        # Step 4: Analyze diffusion coefficient
        # If per_pop_ranges is provided, fit each population individually with its own range
        if per_pop_ranges:
            all_results = []
            if use_robust_regression:
                from ade_dls.analysis.peak_clustering import analyze_diffusion_coefficient_robust
                for i, col in enumerate(pop_columns):
                    pop_range = per_pop_ranges.get(i + 1, x_range)
                    res = analyze_diffusion_coefficient_robust(
                        data_df=self.regularized_data,
                        q_squared_col='q^2',
                        gamma_cols=[col],
                        x_range=pop_range,
                        robust_method=robust_method,
                        show_plots=False
                    )
                    all_results.append(res)
            else:
                from ade_dls.analysis.cumulants import analyze_diffusion_coefficient
                for i, col in enumerate(pop_columns):
                    pop_range = per_pop_ranges.get(i + 1, x_range)
                    res = analyze_diffusion_coefficient(
                        data_df=self.regularized_data,
                        q_squared_col='q^2',
                        gamma_cols=[col],
                        x_range=pop_range,
                        show_plots=False
                    )
                    all_results.append(res)
            self.regularized_diff_results = pd.concat(all_results, ignore_index=True)
        elif use_robust_regression:
            from ade_dls.analysis.peak_clustering import analyze_diffusion_coefficient_robust
            print(f"[Regularized] Using robust regression: {robust_method.upper()}")
            self.regularized_diff_results = analyze_diffusion_coefficient_robust(
                data_df=self.regularized_data,
                q_squared_col='q^2',
                gamma_cols=pop_columns,
                x_range=x_range,
                robust_method=robust_method,
                show_plots=False
            )
        else:
            from ade_dls.analysis.cumulants import analyze_diffusion_coefficient
            self.regularized_diff_results = analyze_diffusion_coefficient(
                data_df=self.regularized_data,
                q_squared_col='q^2',
                gamma_cols=pop_columns,
                x_range=x_range,
                show_plots=False
            )

        # Step 5: Create summary plot (Gamma vs q²)
        print(f"[Regularized] Creating Gamma vs q² summary plot...")
        self._create_regularized_summary_plot()

        # Step 6: Calculate Rh for each peak
        self._calculate_regularized_final_results()

        print(f"[Regularized] Diffusion coefficient analysis complete.")
        return self.regularized_final_results

    def _create_regularized_summary_plot(self):
        """Create summary plot for regularized diffusion analysis (Gamma vs q²)"""
        from ade_dls.gui.analysis.cumulant_plotting import create_summary_plot

        # Prefer Ward-clustered population columns; fall back to raw gamma columns
        gamma_cols = sorted([c for c in self.regularized_data.columns if c.startswith('gamma_pop')])
        if not gamma_cols:
            gamma_cols = sorted([c for c in self.regularized_data.columns
                                  if c.startswith('gamma_') and not c.startswith('gamma_pop')])

        # Create peak labels
        peak_labels = [f'Regularized Pop {i+1}' for i in range(len(gamma_cols))]

        # Create the summary plot
        self.regularized_summary_plot = create_summary_plot(
            self.regularized_data,
            'q^2',
            gamma_cols,
            peak_labels,
            '1/s'
        )

    def _calculate_regularized_final_results(self, append_mode=False, label_suffix=""):
        """
        Internal: Calculate final regularized results with Rh values

        Args:
            append_mode: If True, append to existing results instead of replacing
            label_suffix: Suffix to add to result labels (e.g., " (Post-Refined)")
        """
        if self.regularized_diff_results is None:
            raise ValueError("Must calculate diffusion coefficients first")

        temp_results = []
        for i in range(len(self.regularized_diff_results)):
            # Skip if values are NaN
            if pd.isna(self.regularized_diff_results['q^2_coef'][i]):
                print(f"[Regularized] Skipping peak {i+1} - invalid diffusion coefficient")
                continue

            # Convert D from nm²/s to m²/s
            D_m2s = self.regularized_diff_results['q^2_coef'][i] * 1e-18
            D_err_m2s = self.regularized_diff_results['q^2_se'][i] * 1e-18

            # Calculate Rh
            Rh_nm = self.c.values[0] * (1 / D_m2s) * 1e9

            # Error propagation
            fractional_error = np.sqrt(
                (self.delta_c.values[0] / self.c.values[0])**2 +
                (D_err_m2s / D_m2s)**2
            )
            Rh_error_nm = fractional_error * Rh_nm

            result = pd.DataFrame({
                'Rh [nm]': [Rh_nm],
                'Rh error [nm]': [Rh_error_nm],
                'D [m^2/s]': [D_m2s],
                'D error [m^2/s]': [D_err_m2s],
                'R_squared': [self.regularized_diff_results['R_squared'][i]],
                'Fit': [f'Regularized Peak {i+1}{label_suffix}'],
                'Residuals': [np.nan],
                'Alpha': [self.regularized_params.get('alpha', 0)]
            })
            temp_results.append(result)

        # Check if we have valid results
        if len(temp_results) == 0:
            print("[Regularized] Warning: No valid results to finalize. Creating empty DataFrame.")
            new_results = pd.DataFrame(columns=[
                'Rh [nm]', 'Rh error [nm]', 'D [m^2/s]', 'D error [m^2/s]',
                'R_squared', 'Fit', 'Residuals', 'Alpha'
            ])
        else:
            new_results = pd.concat(temp_results, ignore_index=True)

        # Append or replace based on mode
        if append_mode and self.regularized_final_results is not None and not self.regularized_final_results.empty:
            print(f"[Regularized] Appending {len(new_results)} new results to {len(self.regularized_final_results)} existing results")
            self.regularized_final_results = pd.concat([self.regularized_final_results, new_results], ignore_index=True)
        else:
            self.regularized_final_results = new_results

    def plot_angle_comparison(self, angles: Optional[List[float]] = None,
                             measurement_mode: str = 'average',
                             convert_to_radius: bool = True,
                             figsize: Tuple[int, int] = (12, 8),
                             title: str = "Distribution Comparison"):
        """
        Plot distribution comparison across angles (only for regularized fits)

        Args:
            angles: List of angles to compare (None = all)
            measurement_mode: 'average', 'first', or 'all'
            convert_to_radius: Convert tau to Rh
            figsize: Figure size
            title: Plot title

        Returns:
            Figure and axes objects
        """
        if self.regularized_full_results is None:
            raise ValueError("Must run regularized analysis first")

        return plot_distributions(
            self.regularized_full_results,
            self.regularized_params,
            self.regularized_data,
            angles=angles,
            measurement_mode=measurement_mode,
            convert_to_radius=convert_to_radius,
            figsize=figsize,
            title=title
        )

    def remove_regularized_outliers(self, indices_to_remove: List[int]):
        """
        Remove outlier datasets from regularized data

        Args:
            indices_to_remove: List of row indices to remove
        """
        if self.regularized_data is None:
            raise ValueError("No regularized data to filter")

        self.regularized_data = self.regularized_data.drop(indices_to_remove)
        self.regularized_data = self.regularized_data.reset_index(drop=True)
        self.regularized_data.index = self.regularized_data.index + 1

        print(f"Removed {len(indices_to_remove)} outliers. {len(self.regularized_data)} datasets remaining.")

    # ===== EXPORT METHODS =====

    def get_nnls_summary(self) -> dict:
        """
        Get summary of NNLS results

        Returns:
            Dictionary with summary statistics
        """
        if self.nnls_final_results is None:
            return {"status": "No NNLS results available"}

        summary = {
            "method": "NNLS",
            "num_datasets": len(self.nnls_data) if self.nnls_data is not None else 0,
            "num_peaks": len(self.nnls_final_results),
            "parameters": self.nnls_params,
            "results": self.nnls_final_results.to_dict('records')
        }

        return summary

    def get_regularized_summary(self) -> dict:
        """
        Get summary of regularized results

        Returns:
            Dictionary with summary statistics
        """
        if self.regularized_final_results is None:
            return {"status": "No regularized results available"}

        summary = {
            "method": "Regularized NNLS",
            "num_datasets": len(self.regularized_data) if self.regularized_data is not None else 0,
            "num_peaks": len(self.regularized_final_results),
            "parameters": self.regularized_params,
            "results": self.regularized_final_results.to_dict('records')
        }

        return summary

    def export_results(self, filename: str, method: str = 'both'):
        """
        Export results to file

        Args:
            filename: Output filename (extension determines format: .xlsx, .csv, .txt)
            method: 'nnls', 'regularized', or 'both'
        """
        if filename.endswith('.xlsx'):
            self._export_excel(filename, method)
        elif filename.endswith('.csv'):
            self._export_csv(filename, method)
        else:
            self._export_txt(filename, method)

    def _export_excel(self, filename: str, method: str):
        """Export to Excel with multiple sheets"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            if method in ['nnls', 'both'] and self.nnls_final_results is not None:
                self.nnls_final_results.to_excel(writer, sheet_name='NNLS_Results', index=False)
                if self.nnls_data is not None:
                    self.nnls_data.to_excel(writer, sheet_name='NNLS_Data', index=False)

            if method in ['regularized', 'both'] and self.regularized_final_results is not None:
                self.regularized_final_results.to_excel(writer, sheet_name='Regularized_Results', index=False)
                if self.regularized_data is not None:
                    self.regularized_data.to_excel(writer, sheet_name='Regularized_Data', index=False)

        print(f"Results exported to {filename}")

    def _export_csv(self, filename: str, method: str):
        """Export to CSV"""
        if method == 'nnls' and self.nnls_final_results is not None:
            self.nnls_final_results.to_csv(filename, sep='\t', index=False)
        elif method == 'regularized' and self.regularized_final_results is not None:
            self.regularized_final_results.to_csv(filename, sep='\t', index=False)
        elif method == 'both':
            # Combine results
            combined = pd.concat([
                self.nnls_final_results if self.nnls_final_results is not None else pd.DataFrame(),
                self.regularized_final_results if self.regularized_final_results is not None else pd.DataFrame()
            ], ignore_index=True)
            combined.to_csv(filename, sep='\t', index=False)

        print(f"Results exported to {filename}")

    def _export_txt(self, filename: str, method: str):
        """Export to formatted text file"""
        with open(filename, 'w') as f:
            f.write("JADE-DLS Laplace Transform Analysis Results\n")
            f.write("=" * 60 + "\n\n")

            if method in ['nnls', 'both'] and self.nnls_final_results is not None:
                f.write("NNLS Results:\n")
                f.write("-" * 60 + "\n")
                f.write(self.nnls_final_results.to_string())
                f.write("\n\n")

            if method in ['regularized', 'both'] and self.regularized_final_results is not None:
                f.write("Regularized NNLS Results:\n")
                f.write("-" * 60 + "\n")
                f.write(self.regularized_final_results.to_string())
                f.write("\n\n")

        print(f"Results exported to {filename}")

    # ------------------------------------------------------------------
    # SLS Analysis
    # ------------------------------------------------------------------

    def load_intensity_data(self, file_paths: list) -> bool:
        """
        Load and monitor-correct SLS intensity data from ALV .ASC files.

        Calls build_intensity_dataframe() for each file path and stores
        the result in self.df_intensity.

        Parameters
        ----------
        file_paths : list of str
            Full paths to the ALV .ASC files.

        Returns
        -------
        bool  True if data was loaded successfully, False otherwise.
        """
        from ade_dls.utils.intensity import build_intensity_dataframe
        df = build_intensity_dataframe(file_paths)
        if df is None or df.empty:
            print("load_intensity_data: no data loaded.")
            return False
        self.df_intensity = df
        print(f"load_intensity_data: loaded {len(df)} intensity records "
              f"from {df['angle [°]'].nunique()} angles.")
        return True

    def run_sls_analysis(self, n_populations: int, q2_range=None,
                         exponent: int = 6, use_nw: bool = True) -> 'pd.DataFrame':
        """
        Run population-resolved SLS analysis on the regularized NNLS results.

        Requires:
            - self.regularized_data  (from run_regularized_nnls)
            - self.regularized_final_results
            - self.df_intensity  (from load_intensity_data)

        Parameters
        ----------
        n_populations : int
            Number of populations to analyse (1 – 4).
        q2_range : tuple (q2_min, q2_max), dict {pop: (min, max)}, or None
            q² range for Guinier fit. None = all angles.
        exponent : int  (default 6)
            Rh exponent for number-weighting correction.
            6 = Rayleigh (compact spheres), 5 = Daoud-Cotton (star polymers).
        use_nw : bool  (default True)
            If True, also compute number-weighted intensities.

        Returns
        -------
        pd.DataFrame  summary table (from summarize_sls_combined)
        """
        from ade_dls.analysis.sls import (
            compute_sls_data,
            compute_sls_data_number_weighted,
            compute_guinier_total,
            compute_guinier_extrapolation,
            summarize_sls_combined,
        )

        if self.regularized_data is None or self.regularized_final_results is None:
            raise RuntimeError("No regularized results available. "
                               "Run Regularized NNLS first.")
        if self.df_intensity is None:
            raise RuntimeError("No intensity data loaded. "
                               "Call load_intensity_data() first.")

        # Build Rh lookup {1: Rh1, 2: Rh2, ...}
        rh_values = {}
        for _, row in self.regularized_final_results.iterrows():
            fit_label = str(row.get('Fit', ''))
            # Labels like 'Peak 1', 'pop1', 'Population 1' — extract trailing digit
            digits = ''.join(c for c in fit_label if c.isdigit())
            idx = int(digits) if digits else None
            if idx is not None and 1 <= idx <= n_populations:
                rh_values[idx] = float(row['Rh [nm]'])

        # Fallback: assign in order if label parsing failed
        if not rh_values:
            for i, (_, row) in enumerate(self.regularized_final_results.iterrows(), 1):
                if i > n_populations:
                    break
                rh_values[i] = float(row['Rh [nm]'])

        sls = compute_sls_data(self.regularized_data, self.df_intensity, n_populations)
        if use_nw:
            sls = compute_sls_data_number_weighted(sls, n_populations, rh_values,
                                                    exponent)
        self.sls_data = sls
        self.guinier_total = compute_guinier_total(sls, q2_range)
        self.guinier_results = compute_guinier_extrapolation(sls, n_populations,
                                                              q2_range)
        self.sls_summary = summarize_sls_combined(
            sls, self.guinier_results, self.guinier_total,
            n_populations, rh_values, exponent,
        )
        return self.sls_summary
