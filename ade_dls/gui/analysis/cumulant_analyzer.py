"""
Cumulant Analyzer
Executes the three cumulant analysis methods and manages results
"""

import pandas as pd
import numpy as np
from scipy.constants import k as boltzmann_constant
from scipy.stats import jarque_bera, normaltest, skew, kurtosis
from typing import Dict, List, Tuple, Any
import os


def _normality_status(resid) -> str:
    """
    Assess normality of OLS residuals using Jarque-Bera and D'Agostino tests.
    Matches JADE 2.0 logic exactly.
    """
    try:
        jb_p = jarque_bera(resid).pvalue
        omnibus_p = normaltest(resid).pvalue
        sk = skew(resid)
        ku = kurtosis(resid)
        conditions_met = [jb_p >= 0.05, omnibus_p >= 0.05, sk <= 1, ku <= 0]
        conditions_not_met = [jb_p < 0.05, omnibus_p < 0.05, sk > 1, ku > 0]
        if all(conditions_met):
            return "Normally distributed"
        elif all(conditions_not_met):
            return "Not normally distributed"
        else:
            return "Possibly not normally distributed"
    except Exception:
        return "N/A"


def _fit_single_method_d(name, x_data, y_data, n_max, n_start, gap_threshold):
    """
    Modul-Level-Wrapper für joblib-Parallelisierung von Method D.
    Führt Fitting, Clustering und Momentenberechnung durch – kein Plotting.
    Gibt (name, result_dict, error_str) zurück.
    """
    from ade_dls.analysis.cumulants_D import (
        fit_cumulant_D, calculate_moments_from_gammas, cluster_gammas
    )
    import numpy as np

    mask = y_data > 0
    x_fit = x_data[mask]
    y_fit = y_data[mask]

    if len(x_fit) < 5:
        return name, None, "Too few positive data points"

    try:
        result = fit_cumulant_D(x_fit, y_fit, n_max=n_max, n_start=n_start)
        clusters, representatives, cluster_info = cluster_gammas(
            result['gammas'], gap_threshold=gap_threshold
        )
        moments = calculate_moments_from_gammas(result['gammas'])

        ss_res = result['residual_ss']
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return name, {
            'result': result,
            'representatives': representatives,
            'cluster_info': cluster_info,
            'moments': moments,
            'r_squared': r_squared,
            'x_fit': x_fit,
            'y_fit': y_fit,
        }, None
    except Exception as e:
        return name, None, str(e)


class CumulantAnalyzer:
    """
    Performs cumulant analysis on DLS data using multiple methods

    Methods:
    - Method A: Extract cumulants from ALV software
    - Method B: Linear fit method
    - Method C: Iterative non-linear fit
    """

    def __init__(self, loaded_data: Dict[str, Any], data_folder: str):
        """
        Initialize the analyzer

        Args:
            loaded_data: Dictionary containing filtered data
                - 'correlations': dict of correlation DataFrames
                - 'countrates': dict of countrate DataFrames
                - 'num_files': number of files
            data_folder: Path to folder with .asc files
        """
        self.loaded_data = loaded_data
        self.data_folder = data_folder

        # Results storage
        self.method_a_results = None
        self.method_b_results = None
        self.method_c_results = None
        self.method_d_results = None

        # Per-order storage for Method C (enables per-order post-refinement)
        self.method_c_fits_by_order = {}     # {'2nd order': df, '3rd order': df, ...}
        self.method_c_results_by_order = {}  # {'2nd order': results_df, ...}
        self.method_c_stats_by_order = {}    # {'2nd order': stats_dict, ...}
        self.method_c_plots_by_order = {}    # {'2nd order': plots_dict, ...}

        # Intermediate data
        self.df_basedata = None
        self.processed_correlations = None
        self.c_value = None
        self.delta_c = None

    def prepare_basedata(self):
        """
        Prepare base data (temperature, viscosity, q, q^2)

        This is required for all cumulant methods
        """
        from ade_dls.core.preprocessing import extract_data

        # Get list of files from correlations (these are the filtered ones)
        filtered_files = list(self.loaded_data['correlations'].keys())

        # Create mapping from filename to full path (case-insensitive for Linux)
        import glob
        datafiles = []
        datafiles.extend(glob.glob(os.path.join(self.data_folder, "*.asc")))
        datafiles.extend(glob.glob(os.path.join(self.data_folder, "*.ASC")))
        file_to_path = {os.path.basename(f): f for f in datafiles}

        # Extract base data for filtered files only
        all_data = []
        for filename in filtered_files:
            if filename in file_to_path:
                file_path = file_to_path[filename]
                extracted_data = extract_data(file_path)
                if extracted_data is not None:
                    extracted_data['filename'] = filename
                    all_data.append(extracted_data)

        if all_data:
            self.df_basedata = pd.concat(all_data, ignore_index=True)
            self.df_basedata.index = self.df_basedata.index + 1
        else:
            raise ValueError("No basedata extracted!")

        # Calculate q and q^2
        self.df_basedata['q'] = abs(
            ((4 * np.pi * self.df_basedata['refractive_index']) /
             (self.df_basedata['wavelength [nm]'])) *
            np.sin(np.radians(self.df_basedata['angle [°]']) / 2)
        )
        self.df_basedata['q^2'] = self.df_basedata['q'] ** 2

        # Calculate mean temperature and viscosity
        mean_temperature = self.df_basedata['temperature [K]'].mean()
        std_temperature = self.df_basedata['temperature [K]'].std()

        mean_viscosity = self.df_basedata['viscosity [cp]'].mean()
        std_viscosity = self.df_basedata['viscosity [cp]'].std()

        # Calculate c = kb*T / (6*pi*eta) for Rh determination
        self.c_value = (
            (boltzmann_constant * mean_temperature) /
            (6 * np.pi * mean_viscosity * 10**(-3))
        )

        # Calculate error in c
        fractional_error_c = np.sqrt(
            (std_temperature / mean_temperature)**2 +
            (std_viscosity / mean_viscosity)**2
        )
        self.delta_c = fractional_error_c * self.c_value

        return self.df_basedata

    def prepare_processed_correlations(self):
        """
        Prepare processed correlation data

        Creates a dictionary with:
        - 't (s)': time in seconds
        - 'g(2)': mean of correlation detectors
        """
        from ade_dls.core.preprocessing import process_correlation_data

        columns_to_drop = ['time [ms]', 'correlation 1', 'correlation 2',
                          'correlation 3', 'correlation 4']

        self.processed_correlations = process_correlation_data(
            self.loaded_data['correlations'],
            columns_to_drop
        )

        # Apply noise correction if parameters were set by the filter dialog
        if getattr(self, 'noise_params', None):
            from ade_dls.analysis.noise import apply_noise_corrections
            self.processed_correlations = apply_noise_corrections(
                self.processed_correlations,
                **self.noise_params
            )

        return self.processed_correlations

    def run_method_a(self, q_range=None) -> pd.DataFrame:
        """
        Run Cumulant Method A

        Extracts cumulant fit data from ALV software output

        Args:
            q_range: Optional tuple (min_q, max_q) to restrict Diffusion Analysis

        Returns:
            DataFrame with results (Rh, errors, R^2, PDI)
        """
        from ade_dls.analysis.cumulants import extract_cumulants
        from ade_dls.gui.analysis.cumulant_plotting import create_summary_plot
        import statsmodels.api as sm
        import matplotlib.pyplot as plt
        from scipy import stats as scipy_stats

        print("\n" + "="*60)
        print("CUMULANT METHOD A - ALV SOFTWARE CUMULANT DATA")
        print("="*60)
        if q_range:
            print(f"q² range restriction: {q_range[0]:.4e} - {q_range[1]:.4e} nm⁻²")

        # Ensure basedata is prepared
        if self.df_basedata is None:
            print("[Step 1/4] Preparing basedata...")
            self.prepare_basedata()

        # Get mapping from filename to full path (case-insensitive for Linux)
        print("[Step 2/4] Extracting cumulant data from ALV files...")
        import glob
        datafiles = []
        datafiles.extend(glob.glob(os.path.join(self.data_folder, "*.asc")))
        datafiles.extend(glob.glob(os.path.join(self.data_folder, "*.ASC")))
        file_to_path = {os.path.basename(f): f for f in datafiles}

        # Extract cumulant data
        all_cumulant_data = []
        for filename in self.loaded_data['correlations'].keys():
            if filename in file_to_path:
                file_path = file_to_path[filename]
                extracted_cumulants = extract_cumulants(file_path)
                if extracted_cumulants is not None:
                    extracted_cumulants['filename'] = filename
                    all_cumulant_data.append(extracted_cumulants)

        if not all_cumulant_data:
            raise ValueError("No cumulant data could be extracted from files!")

        print(f"            Extracted cumulant data from {len(all_cumulant_data)} files")

        df_extracted_cumulants = pd.concat(all_cumulant_data, ignore_index=True)
        df_extracted_cumulants.index = df_extracted_cumulants.index + 1

        # Merge with basedata
        cumulant_method_A_data = pd.merge(
            self.df_basedata,
            df_extracted_cumulants,
            on='filename',
            how='outer'
        )
        cumulant_method_A_data = cumulant_method_A_data.reset_index(drop=True)
        cumulant_method_A_data.index = cumulant_method_A_data.index + 1

        # Store unfiltered data for post-fit refinement
        self.method_a_data = cumulant_method_A_data.copy()

        # Apply q-range filter if specified
        if q_range is not None:
            min_q, max_q = q_range
            mask = (cumulant_method_A_data['q^2'] >= min_q) & (cumulant_method_A_data['q^2'] <= max_q)
            cumulant_method_A_data = cumulant_method_A_data[mask].reset_index(drop=True)
            cumulant_method_A_data.index = cumulant_method_A_data.index + 1
            print(f"            Applied q² range filter: {min_q:.4e} - {max_q:.4e} nm⁻²")
            print(f"            {len(cumulant_method_A_data)} data points remaining")

        # Perform linear regression for each gamma column (Jarque-Bera normality test)
        print("[Step 3/5] Performing linear regression for each order...")
        gamma_cols = ['1st order frequency [1/ms]',
                     '2nd order frequency [1/ms]',
                     '3rd order frequency [1/ms]']

        results_list = []
        models_list  = []  # keep models for individual plots
        for i, gamma_col in enumerate(gamma_cols, 1):
            print(f"            Order {i}: {gamma_col}")
            if gamma_col in cumulant_method_A_data.columns:
                X = cumulant_method_A_data['q^2']
                Y = cumulant_method_A_data[gamma_col]
                X_with_const = sm.add_constant(X)
                model = sm.OLS(Y, X_with_const).fit()
                models_list.append(model)

                # Full normality test – identical to JADE 2.0 / analyze_diffusion_coefficient()
                residuals    = model.resid
                resid_skew   = scipy_stats.skew(residuals)
                resid_kurt   = scipy_stats.kurtosis(residuals)
                try:
                    jb_p_value      = float(model.summary().tables[2].data[2][3])
                    omnibus_p_value = float(model.summary().tables[2].data[1][1])
                except Exception:
                    jb_p_value, omnibus_p_value = 1.0, 1.0  # fallback

                conditions_met = [
                    jb_p_value >= 0.05,
                    omnibus_p_value >= 0.05,
                    resid_skew <= 1,
                    resid_kurt <= 0,
                ]
                conditions_not_met = [
                    jb_p_value < 0.05,
                    omnibus_p_value < 0.05,
                    resid_skew > 1,
                    resid_kurt > 0,
                ]
                if all(conditions_met):
                    normality_status = "Normally distributed"
                elif all(conditions_not_met):
                    normality_status = "Not normally distributed"
                else:
                    normality_status = "Possibly not normally distributed"

                results_list.append({
                    'gamma_col':         gamma_col,
                    'intercept':         model.params.iloc[0],
                    'q^2_coef':          model.params.iloc[1],
                    'q^2_se':            model.bse.iloc[1],
                    'R_squared':         model.rsquared,
                    'Normality':         normality_status,
                    'JB_p_value':        jb_p_value,
                    'Omnibus_p_value':   omnibus_p_value,
                    'Skewness_resid':    resid_skew,
                    'Kurtosis_resid':    resid_kurt,
                })

        cumulant_method_A_diff = pd.DataFrame(results_list)

        # Create summary plot (all 3 orders overlaid)
        self.method_a_summary_plot = create_summary_plot(
            cumulant_method_A_data,
            'q^2',
            gamma_cols,
            ['1st order', '2nd order', '3rd order'],
            '1/ms'
        )

        # Create 3 individual order plots (one per Cumulant order)
        print("[Step 4/5] Creating individual order plots...")
        order_labels = ['1st Order', '2nd Order', '3rd Order']
        self.method_a_order_plots = {}
        for idx, (gamma_col, order_label, reg) in enumerate(
                zip(gamma_cols, order_labels, results_list)):
            if gamma_col not in cumulant_method_A_data.columns:
                continue
            X = cumulant_method_A_data['q^2']
            Y = cumulant_method_A_data[gamma_col]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(X, Y, alpha=0.7, color='steelblue', label='Data points', zorder=3)

            X_line = np.linspace(float(X.min()), float(X.max()), 100)
            Y_line = reg['intercept'] + reg['q^2_coef'] * X_line
            ax.plot(X_line, Y_line, color='crimson', linewidth=2,
                    label=f"Linear fit  R² = {reg['R_squared']:.4f}")

            ax.set_xlabel('q² [nm⁻²]')
            ax.set_ylabel('Γ [ms⁻¹]')
            ax.set_title(
                f'Method A – {order_label}: Γ vs q²\n'
                f'D = {reg["q^2_coef"] * 1e-15:.4e} m²/s  |  '
                f'{reg["Normality"]}'
            )
            ax.legend()
            ax.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.close(fig)
            self.method_a_order_plots[f'Method A – {order_label}'] = (fig, {})

        # Create DataFrame with diffusion coefficients
        A_diff = pd.DataFrame()
        A_diff['D [m^2/s]']         = cumulant_method_A_diff['q^2_coef'] * 10**(-15)
        A_diff['std err D [m^2/s]'] = cumulant_method_A_diff['q^2_se']   * 10**(-15)

        # Polydispersity indices (PDI = µ₂ / Γ²)
        cumulant_method_A_data['polydispersity_2nd_order'] = (
            cumulant_method_A_data['2nd order frequency exp param [ms^2]'] /
            (cumulant_method_A_data['2nd order frequency [1/ms]'])**2
        )
        polydispersity_method_A_2 = cumulant_method_A_data['polydispersity_2nd_order'].mean()

        cumulant_method_A_data['polydispersity_3rd_order'] = (
            cumulant_method_A_data['3rd order frequency exp param [ms^2]'] /
            (cumulant_method_A_data['3rd order frequency [1/ms]'])**2
        )
        polydispersity_method_A_3 = cumulant_method_A_data['polydispersity_3rd_order'].mean()

        # Skewness from 3rd order expansion parameter (µ₃ / Γ³) – same as JADE 2.0
        cumulant_method_A_data['Skewness_3rd'] = (
            cumulant_method_A_data['3rd order frequency exp param [ms^2]'] /
            (cumulant_method_A_data['3rd order frequency [1/ms]'])**3
        )
        skewness_method_A_3 = cumulant_method_A_data['Skewness_3rd'].mean()

        # Stokes-Einstein: Rh = (kT / 6πη) / D  [nm]
        print("[Step 5/5] Calculating hydrodynamic radii (Stokes-Einstein)...")
        rh_values = []
        rh_errors = []
        for i in range(len(A_diff)):
            D_val = A_diff['D [m^2/s]'].iloc[i]
            D_err = A_diff['std err D [m^2/s]'].iloc[i]
            if D_val > 0:
                rh = self.c_value / D_val * 1e9
                rh_err = np.sqrt(
                    (self.delta_c / self.c_value)**2 +
                    (D_err / D_val)**2
                ) * rh
            else:
                rh, rh_err = 0.0, 0.0
            rh_values.append(rh)
            rh_errors.append(rh_err)
            print(f"            Order {i+1}: Rh = {rh:.2f} ± {rh_err:.2f} nm  "
                  f"(D = {D_val:.4e} m²/s)")

        # Build results DataFrame
        fit_names = [
            'Rh from 1st order cumulant fit',
            'Rh from 2nd order cumulant fit',
            'Rh from 3rd order cumulant fit',
        ]
        pdi_values      = [np.nan, polydispersity_method_A_2, polydispersity_method_A_3]
        skewness_values = [np.nan, np.nan, skewness_method_A_3]

        result_rows = []
        for i in range(len(cumulant_method_A_diff)):
            row = {
                'Fit':               fit_names[i] if i < len(fit_names) else f'Order {i+1}',
                'Rh [nm]':           rh_values[i],
                'Rh error [nm]':     rh_errors[i],
                'D [m^2/s]':         A_diff['D [m^2/s]'].iloc[i],
                'std err D [m^2/s]': A_diff['std err D [m^2/s]'].iloc[i],
                'R_squared':         cumulant_method_A_diff['R_squared'].iloc[i],
                'Residuals':         cumulant_method_A_diff['Normality'].iloc[i],
                'PDI':               pdi_values[i]      if i < len(pdi_values)      else np.nan,
                'Skewness':          skewness_values[i] if i < len(skewness_values) else np.nan,
            }
            result_rows.append(row)
        self.method_a_results = pd.DataFrame(result_rows)

        # Store regression statistics for detailed output
        self.method_a_regression_stats = {
            'gamma_cols':         gamma_cols,
            'regression_results': results_list,
            'diffusion_data':     A_diff,
        }

        return self.method_a_results

    def run_method_b(self, fit_limits: Tuple[float, float], q_range=None) -> pd.DataFrame:
        """
        Run Cumulant Method B

        Uses linear fit method

        Args:
            fit_limits: Tuple of (min_time, max_time) for fitting
            q_range: Optional tuple (min_q, max_q) to restrict Diffusion Analysis

        Returns:
            DataFrame with results (Rh, errors, R^2, PDI)
        """
        from ade_dls.analysis.cumulants import calculate_g2_B, analyze_diffusion_coefficient
        from ade_dls.gui.analysis.cumulant_plotting import plot_processed_correlations_no_show, create_summary_plot

        # Store fit_limits for post-fit refinement
        self.method_b_fit_limits = fit_limits

        print("\n" + "="*60)
        print("CUMULANT METHOD B - LINEAR FIT")
        print("="*60)
        print(f"Fit limits: {fit_limits[0]:.6f} - {fit_limits[1]:.6f} s")
        if q_range:
            print(f"q² range restriction: {q_range[0]:.4e} - {q_range[1]:.4e} nm⁻²")

        # Ensure data is prepared
        if self.df_basedata is None:
            print("[Step 1/5] Preparing basedata...")
            self.prepare_basedata()

        if self.processed_correlations is None:
            print("[Step 2/5] Preparing processed correlations...")
            self.prepare_processed_correlations()

        # Calculate sqrt(g2)
        print("[Step 3/5] Calculating sqrt(g2) and dropping negative values...")
        processed_correlations = calculate_g2_B(self.processed_correlations)

        # Define fit function (up to 1st moment extension)
        def fit_function(x, a, b, c):
            return 0.5 * np.log(a) - b * x + 0.5 * c * x**2

        # Plot and fit (without showing)
        print(f"[Step 4/5] Fitting {len(processed_correlations)} correlation functions...")
        cumulant_method_B_fit, self.method_b_plots = plot_processed_correlations_no_show(
            processed_correlations,
            fit_function,
            fit_limits
        )
        print(f"            Successfully fitted {len(cumulant_method_B_fit)} datasets")

        # Merge with basedata
        cumulant_method_B_data = pd.merge(
            self.df_basedata,
            cumulant_method_B_fit,
            on='filename',
            how='outer'
        )
        cumulant_method_B_data = cumulant_method_B_data.reset_index(drop=True)
        cumulant_method_B_data.index = cumulant_method_B_data.index + 1

        # Store unfiltered data for post-fit refinement
        self.method_b_data = cumulant_method_B_data.copy()

        # Apply q-range filter if specified
        if q_range is not None:
            min_q, max_q = q_range
            mask = (cumulant_method_B_data['q^2'] >= min_q) & (cumulant_method_B_data['q^2'] <= max_q)
            cumulant_method_B_data = cumulant_method_B_data[mask].reset_index(drop=True)
            cumulant_method_B_data.index = cumulant_method_B_data.index + 1
            print(f"            Applied q² range filter: {min_q:.4e} - {max_q:.4e} nm⁻²")
            print(f"            {len(cumulant_method_B_data)} data points remaining")

        # Analyze diffusion coefficient (without plotting)
        # We'll create our own summary plot
        print(f"[Step 5/5] Analyzing diffusion coefficient (Γ vs q²)...")
        import statsmodels.api as sm

        # Linear regression
        X = cumulant_method_B_data['q^2']
        Y = cumulant_method_B_data['b']
        X_with_const = sm.add_constant(X)
        model = sm.OLS(Y, X_with_const).fit()

        print(f"            Slope (D): {model.params.iloc[1]:.4e} s⁻¹·nm²")
        print(f"            R²: {model.rsquared:.6f}")

        # Create summary plot
        self.method_b_summary_plot = create_summary_plot(
            cumulant_method_B_data,
            'q^2',
            ['b'],
            ['Method B'],
            '1/s'
        )

        # Extract fit quality metrics
        self.method_b_fit_quality = {}
        for filename in cumulant_method_B_fit['filename']:
            row = cumulant_method_B_fit[cumulant_method_B_fit['filename'] == filename]
            if not row.empty:
                self.method_b_fit_quality[filename] = {
                    'R2': row['R-squared'].values[0] if 'R-squared' in row else 0,
                    'residuals': 'Normal'
                }

        # Calculate diffusion coefficients
        B_diff = pd.DataFrame()
        B_diff['D [m^2/s]'] = [model.params.iloc[1] * 10**(-18)]
        B_diff['std err D [m^2/s]'] = [model.bse.iloc[1] * 10**(-18)]

        # Calculate polydispersity
        cumulant_method_B_data['polydispersity'] = (
            cumulant_method_B_data['c'] / (cumulant_method_B_data['b'])**2
        )
        polydispersity_method_B = cumulant_method_B_data['polydispersity'].mean()

        # Calculate final results - use lists to ensure DataFrame rows are created
        print("\nCalculating hydrodynamic radius...")
        print(f"  D = {B_diff['D [m^2/s]'][0]:.4e} ± {B_diff['std err D [m^2/s]'][0]:.4e} m²/s")

        rh_value = self.c_value * (1 / B_diff['D [m^2/s]'][0]) * 10**9

        fractional_error_Rh_B = np.sqrt(
            (self.delta_c / self.c_value)**2 +
            (B_diff['std err D [m^2/s]'][0] / B_diff['D [m^2/s]'][0])**2
        )
        rh_error_value = fractional_error_Rh_B * rh_value

        # Normality test on Γ vs q² regression residuals (matches JADE 2.0)
        normality_b = _normality_status(model.resid)

        # Create DataFrame with lists to ensure rows are created
        self.method_b_results = pd.DataFrame({
            'Rh [nm]':        [rh_value],
            'Rh error [nm]':  [rh_error_value],
            'D [m²/s]':       [B_diff['D [m^2/s]'][0]],
            'D error [m²/s]': [B_diff['std err D [m^2/s]'][0]],
            'R_squared':      [model.rsquared],
            'Fit':            ['Rh from linear cumulant fit'],
            'Residuals':      [normality_b],
            'PDI':            [polydispersity_method_B]
        })

        print(f"\nFINAL RESULTS:")
        print(f"  Rh = {rh_value:.2f} ± {rh_error_value:.2f} nm")
        print(f"  PDI = {polydispersity_method_B:.4f}")
        print(f"  R² = {model.rsquared:.6f}")
        print("="*60 + "\n")

        # Store regression statistics as strings/dicts (not model object)
        self.method_b_regression_stats = {
            'summary': str(model.summary()),
            'params': model.params.to_dict(),
            'stderr_intercept': float(model.bse.iloc[0]),
            'stderr_slope': float(model.bse.iloc[1]),
            'rsquared': float(model.rsquared),
            'rsquared_adj': float(model.rsquared_adj),
            'fvalue': float(model.fvalue),
            'f_pvalue': float(model.f_pvalue),
            'aic': float(model.aic),
            'bic': float(model.bic)
        }

        return self.method_b_results

    def run_method_c(self, params: Dict[str, Any], q_range=None) -> pd.DataFrame:
        """
        Run Cumulant Method C

        Uses iterative non-linear fit method

        Args:
            params: Dictionary with parameters:
                - fit_limits: tuple of (min, max) times
                - fit_function: str ('fit_function2', 'fit_function3', 'fit_function4')
                - adaptive_initial_guesses: bool
                - adaptation_strategy: str ('individual', 'global', 'representative')
                - optimizer: str ('lm', 'trf', 'dogbox')
                - initial_parameters: list of initial parameter guesses
            q_range: Optional tuple (min_q, max_q) to restrict Diffusion Analysis

        Returns:
            DataFrame with results (Rh, errors, R^2, PDI)
        """
        from ade_dls.analysis.cumulants_C import get_adaptive_initial_parameters, get_meaningful_parameters
        from ade_dls.gui.analysis.cumulant_plotting import plot_processed_correlations_iterative_no_show, create_summary_plot

        # Ensure data is prepared
        if self.df_basedata is None:
            self.prepare_basedata()

        if self.processed_correlations is None:
            self.prepare_processed_correlations()

        # Define fit functions
        def fit_function2(x, a, b, c, f):
            inner_term = 1 + 0.5 * c * x**2
            term = f + a * (np.exp(-b * x) * inner_term)**2
            return term

        def fit_function3(x, a, b, c, d, f):
            inner_term = 1 + 0.5 * c * x**2 - (d * x**3) / 6
            term = f + a * (np.exp(-b * x) * inner_term)**2
            return term

        def fit_function4(x, a, b, c, d, e, f):
            inner_term = 1 + 0.5 * c * x**2 - (d * x**3) / 6 + ((e - 3 * c**2) * x**4) / 24
            term = f + a * (np.exp(-b * x) * inner_term)**2
            return term

        # Select fit function
        fit_func_map = {
            'fit_function2': fit_function2,
            'fit_function3': fit_function3,
            'fit_function4': fit_function4
        }
        chosen_fit_function = fit_func_map[params['fit_function']]

        # Get initial parameters
        if params['adaptive_initial_guesses']:
            initial_parameters = get_adaptive_initial_parameters(
                self.processed_correlations,
                chosen_fit_function,
                params['initial_parameters'],
                strategy=params['adaptation_strategy'],
                verbose=False
            )
        else:
            initial_parameters = get_meaningful_parameters(
                chosen_fit_function,
                params['initial_parameters']
            )

        # Run fitting (without showing plots)
        use_multiprocessing = params.get('use_multiprocessing', False)
        print(f"[Method C] Multiprocessing: {'Enabled' if use_multiprocessing else 'Disabled'}")

        self.method_c_fit, self.method_c_plots = plot_processed_correlations_iterative_no_show(
            self.processed_correlations,
            chosen_fit_function,
            params['fit_limits'],
            initial_parameters,
            method=params['optimizer'],
            use_multiprocessing=use_multiprocessing
        )

        # Merge with basedata
        cumulant_method_C_data = pd.merge(
            self.df_basedata,
            self.method_c_fit,
            on='filename',
            how='outer'
        )
        cumulant_method_C_data = cumulant_method_C_data.reset_index(drop=True)
        cumulant_method_C_data.index = cumulant_method_C_data.index + 1

        # Store unfiltered data for post-fit refinement
        self.method_c_data = cumulant_method_C_data.copy()

        # Store q_range for potential recomputation
        self.last_q_range = q_range

        # Apply q-range filter if specified
        if q_range is not None:
            min_q, max_q = q_range
            mask = (cumulant_method_C_data['q^2'] >= min_q) & (cumulant_method_C_data['q^2'] <= max_q)
            cumulant_method_C_data = cumulant_method_C_data[mask].reset_index(drop=True)
            cumulant_method_C_data.index = cumulant_method_C_data.index + 1
            print(f"[METHOD C] Applied q² range filter: {min_q} - {max_q} nm⁻², {len(cumulant_method_C_data)} points remaining")

        # Create summary plot
        self.method_c_summary_plot = create_summary_plot(
            cumulant_method_C_data,
            'q^2',
            ['best_b'],
            ['Method C'],
            '1/s'
        )

        # Extract fit quality metrics
        self.method_c_fit_quality = {}
        for filename in self.method_c_fit['filename']:
            row = self.method_c_fit[self.method_c_fit['filename'] == filename]
            if not row.empty:
                self.method_c_fit_quality[filename] = {
                    'R2': row['R_squared'].values[0] if 'R_squared' in row else 0,
                    'residuals': row['RMSE'].values[0] if 'RMSE' in row else 0
                }

        # Linear regression for diffusion coefficient
        import statsmodels.api as sm

        # Debug: Check data before regression
        print(f"[CUMULANT METHOD C DEBUG] Data for regression:")
        print(f"  q^2 shape: {cumulant_method_C_data['q^2'].shape}")
        print(f"  best_b shape: {cumulant_method_C_data['best_b'].shape}")
        print(f"  q^2 sample: {cumulant_method_C_data['q^2'].head()}")
        print(f"  best_b sample: {cumulant_method_C_data['best_b'].head()}")
        print(f"  q^2 has NaN: {cumulant_method_C_data['q^2'].isna().any()}")
        print(f"  best_b has NaN: {cumulant_method_C_data['best_b'].isna().any()}")

        X = cumulant_method_C_data['q^2']
        Y = cumulant_method_C_data['best_b']
        X_with_const = sm.add_constant(X)
        model = sm.OLS(Y, X_with_const).fit()

        print(f"[CUMULANT METHOD C DEBUG] Regression results:")
        print(f"  params: {model.params}")
        print(f"  R²: {model.rsquared}")
        print(f"  slope (params[1]): {model.params.iloc[1]}")

        # Create DataFrame with diffusion coefficients
        C_diff = pd.DataFrame()
        C_diff['D [m^2/s]'] = [model.params.iloc[1] * 10**(-18)]
        C_diff['std err D [m^2/s]'] = [model.bse.iloc[1] * 10**(-18)]

        print(f"[CUMULANT METHOD C DEBUG] Diffusion coefficient:")
        print(f"  D [m^2/s]: {C_diff['D [m^2/s]'][0]}")
        print(f"  c_value: {self.c_value}")
        print(f"  Rh calculation: {self.c_value} * (1 / {C_diff['D [m^2/s]'][0]}) * 10^9")

        # Calculate polydispersity
        cumulant_method_C_data['polydispersity'] = (
            cumulant_method_C_data['best_c'] / (cumulant_method_C_data['best_b'])**2
        )
        polydispersity_method_C = cumulant_method_C_data['polydispersity'].mean()

        # Calculate final results
        self.method_c_results = pd.DataFrame()

        # Calculate Rh step by step for debugging
        rh_value = self.c_value * (1 / C_diff['D [m^2/s]'][0]) * 10**9
        print(f"[CUMULANT METHOD C DEBUG] Rh calculation result:")
        print(f"  Calculated Rh: {rh_value}")
        print(f"  Type: {type(rh_value)}")

        self.method_c_results['Rh [nm]'] = [rh_value]

        fractional_error_Rh_C = np.sqrt(
            (self.delta_c / self.c_value)**2 +
            (C_diff['std err D [m^2/s]'][0] / C_diff['D [m^2/s]'][0])**2
        )
        rh_error = fractional_error_Rh_C * rh_value
        print(f"  Calculated Rh error: {rh_error}")

        self.method_c_results['Rh error [nm]'] = [rh_error]
        self.method_c_results['D [m²/s]'] = [C_diff['D [m^2/s]'][0]]
        self.method_c_results['D error [m²/s]'] = [C_diff['std err D [m^2/s]'][0]]
        self.method_c_results['R_squared'] = [model.rsquared]
        fit_func_name = params['fit_function']
        _order_label = {'fit_function2': '2nd order', 'fit_function3': '3rd order',
                        'fit_function4': '4th order'}.get(fit_func_name, fit_func_name)
        self.method_c_fit_label = _order_label  # store for recompute
        self.method_c_results['Fit'] = [f'Rh from iterative non-linear cumulant fit ({_order_label})']
        normality_c = _normality_status(model.resid)
        self.method_c_results['Residuals'] = [normality_c]
        self.method_c_results['PDI'] = [polydispersity_method_C]
        if fit_func_name in ('fit_function3', 'fit_function4'):
            skewness_c = (cumulant_method_C_data['best_d'] /
                          cumulant_method_C_data['best_c']**(3/2)).mean()
        else:
            skewness_c = np.nan
        self.method_c_results['Skewness'] = [skewness_c]

        if fit_func_name == 'fit_function4' and 'best_e' in cumulant_method_C_data.columns:
            kurtosis_c = (cumulant_method_C_data['best_e'] /
                          cumulant_method_C_data['best_c']**2).mean()
        else:
            kurtosis_c = np.nan
        self.method_c_results['Kurtosis'] = [kurtosis_c]

        print(f"[CUMULANT METHOD C DEBUG] Final DataFrame:")
        print(self.method_c_results)

        # Store regression statistics as strings/dicts (not model object)
        self.method_c_regression_stats = {
            'summary': str(model.summary()),
            'params': model.params.to_dict(),
            'stderr_intercept': float(model.bse.iloc[0]),
            'stderr_slope': float(model.bse.iloc[1]),
            'rsquared': float(model.rsquared),
            'rsquared_adj': float(model.rsquared_adj),
            'fvalue': float(model.fvalue),
            'f_pvalue': float(model.f_pvalue),
            'aic': float(model.aic),
            'bic': float(model.bic)
        }

        # Store per-order data for post-refinement linkage
        self.method_c_fits_by_order[_order_label] = self.method_c_fit.copy()
        self.method_c_results_by_order[_order_label] = self.method_c_results.copy()
        self.method_c_stats_by_order[_order_label] = dict(self.method_c_regression_stats)
        self.method_c_plots_by_order[_order_label] = self.method_c_plots

        return self.method_c_results

    def recompute_method_c_diffusion(self, included_files: list, q_range=None) -> pd.DataFrame:
        """
        Recompute Method C Diffusion Analysis with only selected files

        Used for post-fit filtering - recompute with only good fits

        Args:
            included_files: List of filenames to include
            q_range: Optional tuple (min_q, max_q) to restrict Diffusion Analysis

        Returns:
            DataFrame with recomputed results (Rh, errors, R^2, PDI)
        """
        import statsmodels.api as sm
        from ade_dls.gui.analysis.cumulant_plotting import create_summary_plot

        # Check if Method C data exists
        if not hasattr(self, 'method_c_fit') or self.method_c_fit is None:
            raise ValueError("Method C must be run first before recomputing!")

        # Filter the fit data to only include selected files
        cumulant_method_C_fit_filtered = self.method_c_fit[
            self.method_c_fit['filename'].isin(included_files)
        ].copy()

        if len(cumulant_method_C_fit_filtered) == 0:
            raise ValueError("No files selected for recomputation!")

        # Merge with basedata
        cumulant_method_C_data = pd.merge(
            self.df_basedata,
            cumulant_method_C_fit_filtered,
            on='filename',
            how='inner'  # Only include files that exist in both
        )
        cumulant_method_C_data = cumulant_method_C_data.reset_index(drop=True)
        cumulant_method_C_data.index = cumulant_method_C_data.index + 1

        # Apply q-range filter if specified
        if q_range is not None:
            min_q, max_q = q_range
            mask = (cumulant_method_C_data['q^2'] >= min_q) & (cumulant_method_C_data['q^2'] <= max_q)
            cumulant_method_C_data = cumulant_method_C_data[mask].reset_index(drop=True)
            cumulant_method_C_data.index = cumulant_method_C_data.index + 1
            print(f"[METHOD C RECOMPUTE] Applied q² range filter: {min_q} - {max_q} nm⁻², {len(cumulant_method_C_data)} points remaining")

        # Count actual data points and contributing files entering the regression (after all filtering)
        n_data_points = len(cumulant_method_C_data)
        n_files_in_range = cumulant_method_C_data['filename'].nunique() if 'filename' in cumulant_method_C_data.columns else n_data_points
        self.method_c_recompute_n_points = n_data_points        # expose to view layer
        self.method_c_recompute_n_files_in_range = n_files_in_range
        print(f"[METHOD C RECOMPUTE] Recomputing with {n_data_points} data points from {n_files_in_range} files")

        # Linear regression for diffusion coefficient
        X = cumulant_method_C_data['q^2']
        Y = cumulant_method_C_data['best_b']
        X_with_const = sm.add_constant(X)
        model = sm.OLS(Y, X_with_const).fit()

        # Create DataFrame with diffusion coefficients
        C_diff = pd.DataFrame()
        C_diff['D [m^2/s]'] = [model.params.iloc[1] * 10**(-18)]
        C_diff['std err D [m^2/s]'] = [model.bse.iloc[1] * 10**(-18)]

        # Calculate polydispersity
        cumulant_method_C_data['polydispersity'] = (
            cumulant_method_C_data['best_c'] / (cumulant_method_C_data['best_b'])**2
        )
        polydispersity_method_C = cumulant_method_C_data['polydispersity'].mean()

        # Calculate final results
        recomputed_results = pd.DataFrame()
        rh_value = self.c_value * (1 / C_diff['D [m^2/s]'][0]) * 10**9
        recomputed_results['Rh [nm]'] = [rh_value]

        fractional_error_Rh_C = np.sqrt(
            (self.delta_c / self.c_value)**2 +
            (C_diff['std err D [m^2/s]'][0] / C_diff['D [m^2/s]'][0])**2
        )
        rh_error = fractional_error_Rh_C * rh_value
        recomputed_results['Rh error [nm]'] = [rh_error]
        recomputed_results['D [m²/s]'] = [C_diff['D [m^2/s]'][0]]
        recomputed_results['D error [m²/s]'] = [C_diff['std err D [m^2/s]'][0]]
        recomputed_results['R_squared'] = [model.rsquared]
        _rc_label = getattr(self, 'method_c_fit_label', '')
        _rc_suffix = f', {_rc_label}' if _rc_label else ''
        recomputed_results['Fit'] = [f'Method C filtered (N={n_data_points} pts{_rc_suffix})']
        normality_recompute = _normality_status(model.resid)
        recomputed_results['Residuals'] = [normality_recompute]
        recomputed_results['PDI'] = [polydispersity_method_C]
        if 'best_d' in cumulant_method_C_data.columns:
            skewness_rc = (cumulant_method_C_data['best_d'] /
                           cumulant_method_C_data['best_c']**(3/2)).mean()
        else:
            skewness_rc = np.nan
        recomputed_results['Skewness'] = [skewness_rc]

        if 'best_e' in cumulant_method_C_data.columns:
            kurtosis_rc = (cumulant_method_C_data['best_e'] /
                           cumulant_method_C_data['best_c']**2).mean()
        else:
            kurtosis_rc = np.nan
        recomputed_results['Kurtosis'] = [kurtosis_rc]

        print(f"[METHOD C RECOMPUTE] Recomputed Rh: {rh_value:.2f} ± {rh_error:.2f} nm")
        print(f"[METHOD C RECOMPUTE] R²: {model.rsquared:.4f}")

        return recomputed_results

    def activate_method_c_order(self, order_label: str) -> bool:
        """
        Load stored per-order data for a specific fit order as the active data.

        Used before opening post-refinement dialog so that the dialog operates
        on the data of the selected order, not the last run order.

        Args:
            order_label: e.g. '2nd order', '3rd order', '4th order'

        Returns:
            True if the order was found and activated, False otherwise.
        """
        if order_label not in self.method_c_fits_by_order:
            return False
        self.method_c_fit = self.method_c_fits_by_order[order_label]
        self.method_c_results = self.method_c_results_by_order[order_label]
        self.method_c_regression_stats = self.method_c_stats_by_order[order_label]
        self.method_c_fit_label = order_label
        return True

    def run_method_d(self, params: Dict[str, Any], q_range=None) -> pd.DataFrame:
        """
        Run Cumulant Method D

        Multi-exponential Dirac-delta decomposition of g²(τ). Iteratively fits
        increasing numbers of exponential modes, selects optimal model order via
        convergence checks, clusters fitted decay rates into populations, and
        computes distribution moments (PDI, skewness, kurtosis).

        Args:
            params: Dictionary with parameters:
                - n_max: int, maximum number of modes (default 25)
                - n_start: int, starting number of modes (default 1)
                - gap_threshold: float, clustering gap ratio threshold (default 1.5)
            q_range: Optional tuple (min_q, max_q) to restrict Diffusion Analysis

        Returns:
            DataFrame with results (Rh, D, PDI, Skewness, Kurtosis, R²)
        """
        from ade_dls.gui.analysis.cumulant_plotting import create_summary_plot
        import matplotlib.pyplot as plt
        import statsmodels.api as sm
        import scipy.stats as scipy_stats

        n_max = params.get('n_max', 25)
        n_start = params.get('n_start', 1)
        gap_threshold = params.get('gap_threshold', 1.5)

        print("\n" + "="*60)
        print("CUMULANT METHOD D - MULTI-EXPONENTIAL DECOMPOSITION")
        print("="*60)
        print(f"Parameters: n_max={n_max}, n_start={n_start}, gap_threshold={gap_threshold}")
        if q_range:
            print(f"q² range restriction: {q_range[0]:.4e} - {q_range[1]:.4e} nm⁻²")

        # Ensure data is prepared
        if self.df_basedata is None:
            print("[Step 1/6] Preparing basedata...")
            self.prepare_basedata()

        if self.processed_correlations is None:
            print("[Step 2/6] Preparing processed correlations...")
            self.prepare_processed_correlations()

        total = len(self.processed_correlations)
        use_multiprocessing = params.get('use_multiprocessing', False)

        print(f"[Step 3/6] Fitting {total} correlation functions...")
        all_fit_results = []
        method_d_plots = {}
        method_d_fit_quality = {}

        # --- Phase 1: Fitting (parallel or sequential) ---
        multiprocessing_success = False
        raw_results = []

        if use_multiprocessing and total > 3:
            try:
                from joblib import Parallel, delayed
                import multiprocessing as mp
                n_jobs = mp.cpu_count()
                print(f"[Method D] Phase 1: Paralleles Fitting mit {n_jobs} CPU-Kernen (joblib loky)...")
                raw_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
                    delayed(_fit_single_method_d)(
                        name,
                        df['t [s]'].values,
                        df['g(2)-1'].values,
                        n_max, n_start, gap_threshold
                    )
                    for name, df in self.processed_correlations.items()
                )
                multiprocessing_success = True
            except Exception as e:
                print(f"[Method D] Warnung: Parallelverarbeitung fehlgeschlagen ({e}), Fallback auf sequenziell.")

        if not multiprocessing_success:
            if use_multiprocessing and total > 3:
                print("[Method D] Phase 1: Sequenzielles Fitting (Fallback)...")
            else:
                print("[Method D] Phase 1: Sequenzielles Fitting...")
            for name, df in self.processed_correlations.items():
                raw_results.append(
                    _fit_single_method_d(
                        name,
                        df['t [s]'].values,
                        df['g(2)-1'].values,
                        n_max, n_start, gap_threshold
                    )
                )

        # --- Phase 2: Plot generation (always sequential) ---
        if multiprocessing_success:
            print("[Method D] Phase 2: Sequentielle Ploterstellung...")

        for name, data, err in raw_results:
            if data is None:
                print(f"    Warning [{name}]: {err}")
                continue

            result = data['result']
            representatives = data['representatives']
            cluster_info = data['cluster_info']
            moments = data['moments']
            r_squared = data['r_squared']
            x_fit = data['x_fit']
            y_fit = data['y_fit']

            fit_result = {
                'filename': name,
                'n_modes': result['n_modes'],
                'beta': result['beta'],
                'residual_ss': result['residual_ss'],
                'R-squared': r_squared,
                'n_populations': cluster_info['n_clusters'],
                'gamma_mean': moments['gamma_mean'],
                'pdi': moments['pdi'],
                'skewness': moments['skewness'],
                'kurtosis': moments['kurtosis'],
            }
            for i, rep_gamma in enumerate(representatives):
                fit_result[f'gamma_{i+1}'] = rep_gamma

            all_fit_results.append(fit_result)

            # Create 3-panel diagnostic figure (use Figure directly, no pyplot event loop)
            from matplotlib.figure import Figure as _MplFigure
            residuals = y_fit - result['g2_fit']
            fig = _MplFigure(figsize=(15, 4))
            ax1 = fig.add_subplot(1, 3, 1)
            ax2 = fig.add_subplot(1, 3, 2)
            ax3 = fig.add_subplot(1, 3, 3)
            fig.suptitle(f'Method D — {name}', fontsize=11)

            ax1.plot(x_fit, y_fit, 'o', alpha=0.6, markersize=3, label='Data')
            ax1.plot(x_fit, result['g2_fit'], 'r-', linewidth=2,
                     label=f'Fit (n={result["n_modes"]})')
            ax1.set_xscale('log')
            ax1.set_xlabel('lag time τ [s]')
            ax1.set_ylabel('g²(τ) − 1')
            ax1.set_title('Data & Fit')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            param_text = (f"n modes: {result['n_modes']}  |  "
                          f"n populations: {cluster_info['n_clusters']}\n"
                          f"R² = {r_squared:.4f}\n"
                          f"⟨Γ⟩ = {moments['gamma_mean']:.3e} s⁻¹")
            ax1.text(0.97, 0.97, param_text, transform=ax1.transAxes,
                     va='top', ha='right', fontsize=7,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            ax2.plot(residuals, 'o', markersize=3, alpha=0.6)
            ax2.axhline(0, color='r', linestyle='--', linewidth=1)
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals')
            ax2.grid(True, alpha=0.3)

            scipy_stats.probplot(residuals, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot')
            ax3.grid(True, alpha=0.3)

            fig.tight_layout()
            method_d_plots[f"D: {name}"] = (fig, {})
            method_d_fit_quality[f"D: {name}"] = {'R2': r_squared, 'residuals': 'Normal'}

        if not all_fit_results:
            raise ValueError("Method D: no fits succeeded. Check input data quality.")

        print(f"[Step 4/7] Fitted {len(all_fit_results)} datasets successfully.")
        self.method_d_plots = method_d_plots
        self.method_d_fit_quality = method_d_fit_quality

        method_d_fit = pd.DataFrame(all_fit_results)
        # Store raw per-file fit for use by refine_method_d()
        self.method_d_fit = method_d_fit.copy()

        # Merge with basedata to get q² values
        method_d_data = pd.merge(
            self.df_basedata,
            method_d_fit,
            on='filename',
            how='outer'
        )
        method_d_data = method_d_data.reset_index(drop=True)
        method_d_data.index = method_d_data.index + 1

        # Store unfiltered for potential post-refinement
        self.method_d_data = method_d_data.copy()

        # Apply q-range filter if specified
        if q_range is not None:
            min_q, max_q = q_range
            mask = (method_d_data['q^2'] >= min_q) & (method_d_data['q^2'] <= max_q)
            method_d_data = method_d_data[mask].reset_index(drop=True)
            method_d_data.index = method_d_data.index + 1
            print(f"            Applied q² range filter: {min_q:.4e} - {max_q:.4e} nm⁻²")
            print(f"            {len(method_d_data)} data points remaining")
            # Update stored data to filtered version for consistent post-refinement
            self.method_d_data = method_d_data.copy()

        # ------------------------------------------------------------------
        # Step 5: Cross-file population clustering
        # ------------------------------------------------------------------
        from ade_dls.analysis.clustering import cluster_all_gammas, get_reliable_gamma_cols

        gamma_pop_cols = sorted([c for c in method_d_fit.columns
                                 if c.startswith('gamma_') and c != 'gamma_mean'])

        if gamma_pop_cols:
            print(f"[Step 5/7] Cross-file clustering on {gamma_pop_cols}...")
            n_clusters_param = params.get('n_clusters', 'auto')
            clustered_df, cluster_info = cluster_all_gammas(
                method_d_data,
                gamma_cols=gamma_pop_cols,
                q_squared_col='q^2',
                method=params.get('method', 'hierarchical'),
                n_clusters=n_clusters_param,
                distance_threshold=params.get('distance_threshold', 0.3),
                normalize_by_q2=True,  # always cluster on D = Γ/q² (angle-independent)
                min_abundance=params.get('min_abundance', 0.3),
                clustering_strategy=params.get('clustering_strategy', 'simple'),
                uncertainty_flags=False,
                plot=False,
                interactive=False,
            )
            self.method_d_clustered_df = clustered_df
            self.method_d_cluster_info = cluster_info
            reliable_cols = get_reliable_gamma_cols(cluster_info)
            n_pops = cluster_info.get('n_populations', 0)
            print(f"            Found {n_pops} populations, {len(reliable_cols)} reliable.")
        else:
            print("[Step 5/7] No per-file gamma columns found; skipping cross-file clustering.")
            self.method_d_clustered_df = method_d_data
            self.method_d_cluster_info = {'n_populations': 0, 'reliable_populations': []}
            reliable_cols = []

        # Clustering overview figure (D vs q² + log(D) histogram)
        from ade_dls.gui.analysis.cumulant_plotting import (
            create_clustering_overview_figure, create_population_ols_figure
        )
        self.method_d_clustering_plot = create_clustering_overview_figure(
            self.method_d_clustered_df, self.method_d_cluster_info, q_squared_col='q^2'
        )

        # ------------------------------------------------------------------
        # Step 6: Summary plot (⟨Γ⟩ vs q²)
        # ------------------------------------------------------------------
        print(f"[Step 6/7] Creating summary plot (⟨Γ⟩ vs q²)...")
        self.method_d_summary_plot = create_summary_plot(
            method_d_data,
            'q^2',
            ['gamma_mean'],
            ['Method D'],
            '1/s'
        )

        # ------------------------------------------------------------------
        # Step 7: Per-population OLS + combined Rh
        # ------------------------------------------------------------------
        print(f"[Step 7/7] Computing Rh per population and combined (⟨Γ⟩ vs q²)...")
        self.method_d_results = self._compute_method_d_results(
            self.method_d_clustered_df, reliable_cols, method_d_data,
            mode_params=None, combined_q_range=None
        )

        # Per-population Γ vs q² + OLS figure
        self.method_d_population_plot = create_population_ols_figure(
            self.method_d_clustered_df, self.method_d_cluster_info, q_squared_col='q^2'
        )

        print("\nFINAL RESULTS:")
        for _, row in self.method_d_results.iterrows():
            print(f"  {row['Fit']}: Rh = {row['Rh [nm]']:.2f} nm, "
                  f"D = {row['D [m²/s]']:.4e} m²/s, R² = {row['R_squared']:.4f}")
        print("="*60 + "\n")

        return self.method_d_results

    # ------------------------------------------------------------------
    # Shared helper: compute per-population OLS rows + combined row
    # ------------------------------------------------------------------

    def _compute_method_d_results(self, clustered_df, reliable_cols, method_d_data,
                                   mode_params=None, combined_q_range=None) -> pd.DataFrame:
        """
        Compute OLS regression for each reliable population and the combined row.

        Args:
            clustered_df: DataFrame with gamma_pop1, gamma_pop2, ... columns
            reliable_cols: list of column names ['gamma_pop1', ...]
            method_d_data: DataFrame with gamma_mean and q^2
            mode_params: optional dict {pop_num: {'q_min', 'q_max', 'outlier_sigma', 'min_points'}}
            combined_q_range: optional (min_q, max_q) for the gamma_mean regression

        Returns:
            DataFrame with one row per reliable population + one combined row
        """
        import statsmodels.api as sm

        result_rows = []

        # --- Per-population rows ---
        for col in reliable_cols:
            pop_num = int(col.replace('gamma_pop', ''))
            mparams = (mode_params or {}).get(pop_num, {})

            X = clustered_df['q^2']
            Y = clustered_df[col]
            valid = Y.notna() & X.notna()

            q_min = mparams.get('q_min')
            q_max = mparams.get('q_max')
            if q_min is not None:
                valid = valid & (X >= q_min)
            if q_max is not None:
                valid = valid & (X <= q_max)

            min_pts = mparams.get('min_points', 2)
            if valid.sum() < min_pts:
                print(f"  Population {pop_num}: only {valid.sum()} points, skipping.")
                continue

            model = sm.OLS(Y[valid], sm.add_constant(X[valid])).fit()

            # Optional outlier removal and re-fit
            outlier_sigma = mparams.get('outlier_sigma')
            if outlier_sigma:
                resid_std = model.resid.std()
                inliers = valid.copy()
                inliers[valid] = np.abs(model.resid) <= outlier_sigma * resid_std
                if inliers.sum() >= min_pts:
                    model = sm.OLS(Y[inliers], sm.add_constant(X[inliers])).fit()

            D_val = model.params.iloc[1] * 1e-18
            D_err = model.bse.iloc[1] * 1e-18
            Rh = self.c_value / D_val * 1e9
            Rh_err = np.sqrt((self.delta_c / self.c_value)**2 + (D_err / D_val)**2) * Rh

            result_rows.append({
                'Rh [nm]':        Rh,
                'Rh error [nm]':  Rh_err,
                'D [m²/s]':       D_val,
                'D error [m²/s]': D_err,
                'R_squared':      model.rsquared,
                'Fit':            f'Rh from Method D (Population {pop_num})',
                'Residuals':      _normality_status(model.resid),
                'PDI':            np.nan,
                'Skewness':       np.nan,
                'Kurtosis':       np.nan,
                'n_modes_mean':   np.nan,
                'n_populations_mean': len(reliable_cols),
            })

        # --- Combined row (gamma_mean → OLS) ---
        X_comb = method_d_data['q^2']
        Y_comb = method_d_data['gamma_mean']
        valid_comb = Y_comb.notna() & X_comb.notna()
        if combined_q_range is not None:
            valid_comb = valid_comb & X_comb.between(*combined_q_range)

        common = X_comb[valid_comb].index
        model_total = sm.OLS(Y_comb.loc[common], sm.add_constant(X_comb.loc[common])).fit()

        D_total = model_total.params.iloc[1] * 1e-18
        D_total_err = model_total.bse.iloc[1] * 1e-18
        Rh_total = self.c_value / D_total * 1e9
        Rh_total_err = np.sqrt(
            (self.delta_c / self.c_value)**2 + (D_total_err / D_total)**2
        ) * Rh_total

        mean_pdi = method_d_data['pdi'].mean() if 'pdi' in method_d_data.columns else np.nan
        mean_skewness = method_d_data['skewness'].mean() if 'skewness' in method_d_data.columns else np.nan
        mean_kurtosis = method_d_data['kurtosis'].mean() if 'kurtosis' in method_d_data.columns else np.nan
        mean_n_modes = method_d_data['n_modes'].mean() if 'n_modes' in method_d_data.columns else np.nan

        result_rows.append({
            'Rh [nm]':        Rh_total,
            'Rh error [nm]':  Rh_total_err,
            'D [m²/s]':       D_total,
            'D error [m²/s]': D_total_err,
            'R_squared':      model_total.rsquared,
            'Fit':            f'Rh from Method D (combined, {len(reliable_cols)} populations)',
            'Residuals':      _normality_status(model_total.resid),
            'PDI':            mean_pdi,
            'Skewness':       mean_skewness,
            'Kurtosis':       mean_kurtosis,
            'n_modes_mean':   mean_n_modes,
            'n_populations_mean': len(reliable_cols),
        })

        self.method_d_regression_stats = {
            'summary':           str(model_total.summary()),
            'params':            model_total.params.to_dict(),
            'stderr_intercept':  float(model_total.bse.iloc[0]),
            'stderr_slope':      float(model_total.bse.iloc[1]),
            'rsquared':          float(model_total.rsquared),
            'rsquared_adj':      float(model_total.rsquared_adj),
            'fvalue':            float(model_total.fvalue),
            'f_pvalue':          float(model_total.f_pvalue),
            'aic':               float(model_total.aic),
            'bic':               float(model_total.bic),
        }

        return pd.DataFrame(result_rows)

    def refine_method_d(self, clustering_params=None, mode_params=None,
                        combined_q_range=None) -> pd.DataFrame:
        """
        Re-run Steps 5 and 7 of Method D with updated parameters.

        Requires run_method_d() to have been called first (method_d_fit,
        method_d_data, method_d_clustered_df, method_d_cluster_info must exist).

        Args:
            clustering_params: dict with keys matching cluster_all_gammas kwargs,
                               or None to skip re-clustering.
            mode_params: dict {pop_num: {'q_min', 'q_max', 'outlier_sigma', 'min_points'}}
            combined_q_range: (min_q, max_q) for the combined gamma_mean regression

        Returns:
            Updated method_d_results DataFrame
        """
        from ade_dls.analysis.clustering import cluster_all_gammas, get_reliable_gamma_cols

        if clustering_params:
            print("\n[Method D Refinement – Step 5] Re-running cross-file clustering...")
            gamma_pop_cols = sorted([c for c in self.method_d_fit.columns
                                     if c.startswith('gamma_') and c != 'gamma_mean'])
            if gamma_pop_cols:
                # Build kwargs: map our param keys to cluster_all_gammas args
                cparams = dict(clustering_params)  # copy
                cparams.pop('q_squared_col', None)   # we supply this ourselves
                cparams['normalize_by_q2'] = True    # always cluster on D = Γ/q²
                self.method_d_clustered_df, self.method_d_cluster_info = cluster_all_gammas(
                    self.method_d_data,
                    gamma_cols=gamma_pop_cols,
                    q_squared_col='q^2',
                    uncertainty_flags=False,
                    plot=False,
                    interactive=False,
                    **cparams,
                )
                n_pops = self.method_d_cluster_info.get('n_populations', 0)
                print(f"  Found {n_pops} populations after re-clustering.")
            else:
                print("  No per-file gamma columns found; cannot re-cluster.")

        reliable_cols = get_reliable_gamma_cols(self.method_d_cluster_info)
        print(f"[Method D Refinement – Step 7] Re-computing OLS for {len(reliable_cols)} populations...")

        self.method_d_results = self._compute_method_d_results(
            self.method_d_clustered_df, reliable_cols, self.method_d_data,
            mode_params=mode_params, combined_q_range=combined_q_range,
        )

        print("Refinement complete.")
        for _, row in self.method_d_results.iterrows():
            print(f"  {row['Fit']}: Rh = {row['Rh [nm]']:.2f} nm, R² = {row['R_squared']:.4f}")

        return self.method_d_results

    def get_combined_results(self) -> pd.DataFrame:
        """
        Get combined results from all methods

        Returns:
            DataFrame with all results concatenated
        """
        results = []

        if self.method_a_results is not None:
            results.append(self.method_a_results)

        if self.method_b_results is not None:
            results.append(self.method_b_results)

        if self.method_c_results is not None:
            results.append(self.method_c_results)

        if self.method_d_results is not None:
            results.append(self.method_d_results)

        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
