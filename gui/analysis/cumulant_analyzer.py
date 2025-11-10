"""
Cumulant Analyzer
Executes the three cumulant analysis methods and manages results
"""

import pandas as pd
import numpy as np
from scipy.constants import k as boltzmann_constant
from typing import Dict, List, Tuple, Any
import os


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
        from preprocessing import extract_data

        # Get list of files from correlations (these are the filtered ones)
        filtered_files = list(self.loaded_data['correlations'].keys())

        # Create mapping from filename to full path
        import glob
        datafiles = glob.glob(os.path.join(self.data_folder, "*.asc"))
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
        from preprocessing import process_correlation_data

        columns_to_drop = ['time [ms]', 'correlation 1', 'correlation 2',
                          'correlation 3', 'correlation 4']

        self.processed_correlations = process_correlation_data(
            self.loaded_data['correlations'],
            columns_to_drop
        )

        return self.processed_correlations

    def run_method_a(self) -> pd.DataFrame:
        """
        Run Cumulant Method A

        Extracts cumulant fit data from ALV software output

        Returns:
            DataFrame with results (Rh, errors, R^2, PDI)
        """
        from cumulants import extract_cumulants, calculate_cumulant_results_A
        from gui.analysis.cumulant_plotting import create_summary_plot
        import statsmodels.api as sm

        # Ensure basedata is prepared
        if self.df_basedata is None:
            self.prepare_basedata()

        # Get mapping from filename to full path
        import glob
        datafiles = glob.glob(os.path.join(self.data_folder, "*.asc"))
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

        # Perform linear regression for each gamma column without showing plots
        gamma_cols = ['1st order frequency [1/ms]',
                     '2nd order frequency [1/ms]',
                     '3rd order frequency [1/ms]']

        results_list = []
        for gamma_col in gamma_cols:
            if gamma_col in cumulant_method_A_data.columns:
                X = cumulant_method_A_data['q^2']
                Y = cumulant_method_A_data[gamma_col]
                X_with_const = sm.add_constant(X)
                model = sm.OLS(Y, X_with_const).fit()

                results_list.append({
                    'gamma_col': gamma_col,
                    'q^2_coef': model.params.iloc[1],
                    'q^2_se': model.bse.iloc[1],
                    'R_squared': model.rsquared,
                    'Normality': 'Normal' if model.rsquared > 0.9 else 'Check'
                })

        cumulant_method_A_diff = pd.DataFrame(results_list)

        # Create summary plot (without showing)
        self.method_a_summary_plot = create_summary_plot(
            cumulant_method_A_data,
            'q^2',
            gamma_cols,
            ['1st order', '2nd order', '3rd order'],
            '1/ms'
        )

        # Create DataFrame with diffusion coefficients
        A_diff = pd.DataFrame()
        A_diff['D [m^2/s]'] = cumulant_method_A_diff['q^2_coef'] * 10**(-15)
        A_diff['std err D [m^2/s]'] = cumulant_method_A_diff['q^2_se'] * 10**(-15)

        # Calculate polydispersity indices
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

        # Calculate final results
        self.method_a_results = calculate_cumulant_results_A(
            A_diff,
            cumulant_method_A_diff,
            polydispersity_method_A_2,
            polydispersity_method_A_3,
            self.c_value,
            self.delta_c
        )

        return self.method_a_results

    def run_method_b(self, fit_limits: Tuple[float, float]) -> pd.DataFrame:
        """
        Run Cumulant Method B

        Uses linear fit method

        Args:
            fit_limits: Tuple of (min_time, max_time) for fitting

        Returns:
            DataFrame with results (Rh, errors, R^2, PDI)
        """
        from cumulants import calculate_g2_B, analyze_diffusion_coefficient
        from gui.analysis.cumulant_plotting import plot_processed_correlations_no_show, create_summary_plot

        # Ensure data is prepared
        if self.df_basedata is None:
            self.prepare_basedata()

        if self.processed_correlations is None:
            self.prepare_processed_correlations()

        # Calculate sqrt(g2)
        processed_correlations = calculate_g2_B(self.processed_correlations)

        # Define fit function (up to 1st moment extension)
        def fit_function(x, a, b, c):
            return 0.5 * np.log(a) - b * x + 0.5 * c * x**2

        # Plot and fit (without showing)
        cumulant_method_B_fit, self.method_b_plots = plot_processed_correlations_no_show(
            processed_correlations,
            fit_function,
            fit_limits
        )

        # Merge with basedata
        cumulant_method_B_data = pd.merge(
            self.df_basedata,
            cumulant_method_B_fit,
            on='filename',
            how='outer'
        )
        cumulant_method_B_data = cumulant_method_B_data.reset_index(drop=True)
        cumulant_method_B_data.index = cumulant_method_B_data.index + 1

        # Analyze diffusion coefficient (without plotting)
        # We'll create our own summary plot
        import statsmodels.api as sm

        # Linear regression
        X = cumulant_method_B_data['q^2']
        Y = cumulant_method_B_data['b']
        X_with_const = sm.add_constant(X)
        model = sm.OLS(Y, X_with_const).fit()

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

        # Calculate final results
        self.method_b_results = pd.DataFrame()
        self.method_b_results['Rh [nm]'] = self.c_value * (1 / B_diff['D [m^2/s]'][0]) * 10**9

        fractional_error_Rh_B = np.sqrt(
            (self.delta_c / self.c_value)**2 +
            (B_diff['std err D [m^2/s]'][0] / B_diff['D [m^2/s]'][0])**2
        )
        self.method_b_results['Rh error [nm]'] = fractional_error_Rh_B * self.method_b_results['Rh [nm]']
        self.method_b_results['R_squared'] = [model.rsquared]
        self.method_b_results['Fit'] = 'Rh from linear cumulant fit'
        self.method_b_results['Residuals'] = 'N/A'
        self.method_b_results['PDI'] = polydispersity_method_B

        return self.method_b_results

    def run_method_c(self, params: Dict[str, Any]) -> pd.DataFrame:
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

        Returns:
            DataFrame with results (Rh, errors, R^2, PDI)
        """
        from cumulants_C import get_adaptive_initial_parameters, get_meaningful_parameters
        from gui.analysis.cumulant_plotting import plot_processed_correlations_iterative_no_show, create_summary_plot

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
        cumulant_method_C_fit, self.method_c_plots = plot_processed_correlations_iterative_no_show(
            self.processed_correlations,
            chosen_fit_function,
            params['fit_limits'],
            initial_parameters,
            method=params['optimizer']
        )

        # Merge with basedata
        cumulant_method_C_data = pd.merge(
            self.df_basedata,
            cumulant_method_C_fit,
            on='filename',
            how='outer'
        )
        cumulant_method_C_data = cumulant_method_C_data.reset_index(drop=True)
        cumulant_method_C_data.index = cumulant_method_C_data.index + 1

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
        for filename in cumulant_method_C_fit['filename']:
            row = cumulant_method_C_fit[cumulant_method_C_fit['filename'] == filename]
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
        self.method_c_results['R_squared'] = [model.rsquared]
        self.method_c_results['Fit'] = ['Rh from iterative non-linear cumulant fit']
        self.method_c_results['Residuals'] = ['N/A']
        self.method_c_results['PDI'] = [polydispersity_method_C]

        print(f"[CUMULANT METHOD C DEBUG] Final DataFrame:")
        print(self.method_c_results)

        return self.method_c_results

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

        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
