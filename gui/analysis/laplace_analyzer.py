"""
Laplace Transform Analysis (NNLS and Regularized Fits)
Handles inverse Laplace transform methods for DLS data
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from regularized_optimized import (
    nnls_all_optimized,
    nnls_preview_random,
    calculate_decay_rates
)
from regularized import (
    nnls_reg_all,
    plot_distributions,
    tau_to_hydrodynamic_radius
)
from cumulants import analyze_diffusion_coefficient


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
        from preprocessing import process_correlation_data

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

        print("Running NNLS analysis...")
        self.nnls_results = nnls_all_optimized(
            self.processed_correlations,
            params,
            use_multiprocessing=use_multiprocessing,
            show_plots=show_plots
        )

        # Merge with basedata
        self.nnls_data = pd.merge(self.df_basedata, self.nnls_results,
                                 on='filename', how='outer')
        self.nnls_data = self.nnls_data.reset_index(drop=True)
        self.nnls_data.index = self.nnls_data.index + 1

        print(f"NNLS analysis complete. Processed {len(self.nnls_results)} datasets.")

        return self.nnls_data

    def preview_nnls_parameters(self, params: dict, num_datasets: int = 5,
                               seed: Optional[int] = None) -> Tuple[any, List[str]]:
        """
        Preview NNLS results on random datasets for parameter tuning

        Args:
            params: NNLS parameters to test
            num_datasets: Number of random datasets to show
            seed: Random seed for reproducibility

        Returns:
            Tuple of (matplotlib Figure, list of selected dataset names)
        """
        return nnls_preview_random(self.processed_correlations, params,
                                  num_datasets=num_datasets, seed=seed)

    def calculate_nnls_diffusion_coefficients(self, tau_columns: Optional[List[str]] = None,
                                              x_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Calculate diffusion coefficients from NNLS tau values

        Args:
            tau_columns: List of tau column names (e.g., ['tau_1', 'tau_2'])
                        If None, auto-detects all tau columns
            x_range: Optional q² range for fitting (min, max)

        Returns:
            DataFrame with diffusion coefficients for each peak
        """
        if self.nnls_data is None:
            raise ValueError("Must run NNLS analysis first")

        # Auto-detect tau columns if not provided
        if tau_columns is None:
            tau_columns = [col for col in self.nnls_data.columns if col.startswith('tau_')]

        print(f"Calculating diffusion coefficients for {len(tau_columns)} peaks...")

        # Calculate gamma from tau
        self.nnls_data = calculate_decay_rates(self.nnls_data, tau_columns)

        # Get corresponding gamma columns
        gamma_columns = [col.replace('tau', 'gamma') for col in tau_columns]

        # Analyze diffusion coefficient via Γ vs q² regression
        self.nnls_diff_results = analyze_diffusion_coefficient(
            data_df=self.nnls_data,
            q_squared_col='q^2',
            gamma_cols=gamma_columns,
            x_range=x_range
        )

        # Calculate Rh for each peak
        self._calculate_nnls_final_results()

        return self.nnls_final_results

    def _calculate_nnls_final_results(self):
        """Internal: Calculate final NNLS results with Rh values"""
        if self.nnls_diff_results is None:
            raise ValueError("Must calculate diffusion coefficients first")

        temp_results = []
        for i in range(len(self.nnls_diff_results)):
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

            result = pd.DataFrame({
                'Rh [nm]': [Rh_nm],
                'Rh error [nm]': [Rh_error_nm],
                'D [m^2/s]': [D_m2s],
                'D error [m^2/s]': [D_err_m2s],
                'R_squared': [self.nnls_diff_results['R_squared'][i]],
                'Fit': [f'NNLS Peak {i+1}'],
                'Residuals': [self.nnls_diff_results.get('Normality', [0])[i]]
            })
            temp_results.append(result)

        self.nnls_final_results = pd.concat(temp_results, ignore_index=True)

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

    def run_regularized(self, params: dict, show_plots: bool = True) -> Tuple[pd.DataFrame, dict]:
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
            show_plots: Show plots during processing

        Returns:
            Tuple of (results DataFrame, full_results dict with distributions)
        """
        self.regularized_params = params.copy()

        print("Running regularized NNLS analysis...")
        self.regularized_results, self.regularized_full_results = nnls_reg_all(
            self.processed_correlations,
            params
        )

        # Merge with basedata
        self.regularized_data = pd.merge(self.df_basedata, self.regularized_results,
                                        on='filename', how='outer')
        self.regularized_data = self.regularized_data.reset_index(drop=True)
        self.regularized_data.index = self.regularized_data.index + 1

        print(f"Regularized analysis complete. Processed {len(self.regularized_results)} datasets.")

        return self.regularized_data, self.regularized_full_results

    def calculate_regularized_diffusion_coefficients(self,
                                                    tau_columns: Optional[List[str]] = None,
                                                    x_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Calculate diffusion coefficients from regularized fit tau values

        Args:
            tau_columns: List of tau column names
            x_range: Optional q² range for fitting

        Returns:
            DataFrame with diffusion coefficients
        """
        if self.regularized_data is None:
            raise ValueError("Must run regularized analysis first")

        # Auto-detect tau columns if not provided
        if tau_columns is None:
            tau_columns = [col for col in self.regularized_data.columns if col.startswith('tau_')]

        print(f"Calculating diffusion coefficients for {len(tau_columns)} peaks...")

        # Calculate gamma from tau
        self.regularized_data = calculate_decay_rates(self.regularized_data, tau_columns)

        # Get corresponding gamma columns
        gamma_columns = [col.replace('tau', 'gamma') for col in tau_columns]

        # Analyze diffusion coefficient
        self.regularized_diff_results = analyze_diffusion_coefficient(
            data_df=self.regularized_data,
            q_squared_col='q^2',
            gamma_cols=gamma_columns,
            x_range=x_range
        )

        # Calculate Rh for each peak
        self._calculate_regularized_final_results()

        return self.regularized_final_results

    def _calculate_regularized_final_results(self):
        """Internal: Calculate final regularized results with Rh values"""
        if self.regularized_diff_results is None:
            raise ValueError("Must calculate diffusion coefficients first")

        temp_results = []
        for i in range(len(self.regularized_diff_results)):
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
                'Fit': [f'Regularized Peak {i+1}'],
                'Residuals': [self.regularized_diff_results.get('Normality', [0])[i]],
                'Alpha': [self.regularized_params.get('alpha', 0)]
            })
            temp_results.append(result)

        self.regularized_final_results = pd.concat(temp_results, ignore_index=True)

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
