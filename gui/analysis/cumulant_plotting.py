"""
Cumulant Plotting Functions (No Show)
Modified versions of cumulant plotting functions that return figures instead of showing them
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import inspect


def plot_processed_correlations_no_show(dataframes_dict, fit_function, fit_x_limits):
    """
    Plot and fit correlation data without showing plots

    Returns:
        tuple: (results_df, plots_dict)
            - results_df: DataFrame with fit parameters
            - plots_dict: Dictionary {filename: (fig, fit_data)}
    """
    all_fit_results = []
    plots_dict = {}
    plot_number = 1

    for name, df in dataframes_dict.items():
        try:
            fit_result = {'filename': name}

            # Main processing
            x_data = df['t (s)']
            y_data = df['g(2)_mod']
            x_fit = x_data[(x_data >= fit_x_limits[0]) & (x_data <= fit_x_limits[1])]
            y_fit = y_data[(x_data >= fit_x_limits[0]) & (x_data <= fit_x_limits[1])]

            # Perform the fit
            popt, pcov = curve_fit(fit_function, x_fit, y_fit, method='lm', maxfev=50000)

            # Calculate parameter errors
            perr = np.sqrt(np.diag(pcov))

            # Generate fit curve
            y_fit_values = fit_function(x_data, *popt)

            # Calculate residuals
            residuals = y_fit - fit_function(x_fit, *popt)

            # Extract parameter names and store fit parameters
            param_names = inspect.getfullargspec(fit_function).args[1:]
            for i, param_name in enumerate(param_names):
                fit_result[param_name] = popt[i]
                fit_result[f'{param_name}_error'] = perr[i]
                fit_result[f'{param_name}_relative_error'] = (perr[i] / abs(popt[i])) * 100 if popt[i] != 0 else np.inf

            # Calculate fit quality metrics
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
            r_squared = 1 - (ss_res / ss_tot)
            fit_result['R-squared'] = r_squared

            # Create figure (but don't show it)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Left plot: original data and fit
            ax1.plot(x_data, y_data, marker='.', linestyle='', label='Data')
            ax1.plot(x_data, y_fit_values, 'r-', label=f'Fit')
            ax1.set_xlabel('lag time (s)')
            ax1.set_ylabel(r"$\sqrt{g(2)-1}$")
            ax1.set_title(f'[{plot_number}]: g(2)-1 vs. lag time for {name}')
            ax1.grid(True)
            ax1.set_yscale('log')
            ax1.set_xlim(0, 0.002)
            ax1.legend()

            # Right plot: Q-Q plot
            stats.probplot(residuals, dist="norm", plot=ax2)
            ax2.set_title(f'[{plot_number}]: Q-Q Plot of Residuals')
            ax2.grid(True)

            # Add R^2 as text
            param_text = f"R² = {r_squared:.4f}"
            ax1.text(0.95, 0.95, param_text, transform=ax1.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            plt.tight_layout()

            # Store figure instead of showing it
            plots_dict[name] = (fig, {
                'popt': popt,
                'perr': perr,
                'r_squared': r_squared,
                'residuals': residuals
            })

            plot_number += 1

            # Store all results
            all_fit_results.append(fit_result)

        except (KeyError, TypeError) as e:
            print(f"Error processing DataFrame '{name}': {e}")
            fit_result['Error'] = str(e)
            all_fit_results.append(fit_result)
        except RuntimeError as e:
            print(f"Fit error for DataFrame '{name}': {e}")
            fit_result['Error'] = str(e)
            all_fit_results.append(fit_result)

    final_results_df = pd.DataFrame(all_fit_results)
    return final_results_df, plots_dict


def plot_processed_correlations_iterative_no_show(dataframes_dict, fit_function, fit_limits,
                                                   initial_parameters, method='lm'):
    """
    Iterative non-linear fit without showing plots

    Returns:
        tuple: (results_df, plots_dict)
    """
    all_fit_results = []
    plots_dict = {}
    plot_number = 1

    for name, df in dataframes_dict.items():
        try:
            fit_result = {'filename': name}

            # Get data
            x_data = df['t (s)'].values
            y_data = df['g(2)'].values

            # Filter data by fit limits
            mask = (x_data >= fit_limits[0]) & (x_data <= fit_limits[1])
            x_fit = x_data[mask]
            y_fit = y_data[mask]

            # Determine which parameters to use based on fit_function
            param_names = inspect.getfullargspec(fit_function).args[1:]
            num_params = len(param_names)

            # Get initial parameters for this function
            if isinstance(initial_parameters, dict):
                # If it's a dict, use parameters for this specific dataset
                init_params = initial_parameters.get(name, initial_parameters.get('default', []))[:num_params]
            elif isinstance(initial_parameters, list):
                init_params = initial_parameters[:num_params]
            else:
                init_params = initial_parameters

            # Perform fit
            popt, pcov = curve_fit(fit_function, x_fit, y_fit,
                                  p0=init_params, method=method, maxfev=50000)

            # Calculate errors
            perr = np.sqrt(np.diag(pcov))

            # Generate fit curve
            x_plot = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 1000)
            y_fit_values = fit_function(x_plot, *popt)

            # Calculate residuals
            residuals = y_fit - fit_function(x_fit, *popt)

            # Calculate R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
            r_squared = 1 - (ss_res / ss_tot)

            # RMSE
            rmse = np.sqrt(np.mean(residuals**2))

            # AIC (Akaike Information Criterion)
            n = len(x_fit)
            aic = n * np.log(ss_res / n) + 2 * num_params

            # Store results with 'best_' prefix for parameter b
            for i, param_name in enumerate(param_names):
                fit_result[f'best_{param_name}'] = popt[i]
                fit_result[f'best_{param_name}_error'] = perr[i]

            fit_result['R_squared'] = r_squared
            fit_result['RMSE'] = rmse
            fit_result['AIC'] = aic

            # Create figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

            # Plot 1: Data and fit
            ax1.plot(x_data, y_data, 'b.', label='Data', markersize=3)
            ax1.plot(x_plot, y_fit_values, 'r-', label='Fit', linewidth=2)
            ax1.set_xscale('log')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('g(2)')
            ax1.set_title(f'[{plot_number}]: Fit for {name}')
            ax1.legend()
            ax1.grid(True)

            # Add fit info
            info_text = f"R² = {r_squared:.4f}\nRMSE = {rmse:.4e}"
            ax1.text(0.05, 0.05, info_text, transform=ax1.transAxes,
                    verticalalignment='bottom', bbox=dict(boxstyle='round',
                    facecolor='white', alpha=0.7))

            # Plot 2: Residuals
            ax2.plot(x_fit, residuals, 'k.', markersize=3)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xscale('log')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals')
            ax2.grid(True)

            # Plot 3: Q-Q plot
            stats.probplot(residuals, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot')
            ax3.grid(True)

            # Plot 4: Parameter values with errors
            param_values = [popt[i] for i in range(len(param_names))]
            param_errors = [perr[i] for i in range(len(param_names))]

            ax4.barh(param_names, param_values, xerr=param_errors, capsize=5)
            ax4.set_xlabel('Parameter Value')
            ax4.set_title('Fit Parameters')
            ax4.grid(True, axis='x')

            plt.tight_layout()

            # Store figure
            plots_dict[name] = (fig, {
                'popt': popt,
                'perr': perr,
                'r_squared': r_squared,
                'rmse': rmse,
                'aic': aic,
                'residuals': residuals
            })

            plot_number += 1
            all_fit_results.append(fit_result)

        except Exception as e:
            print(f"Error fitting '{name}': {e}")
            fit_result['Error'] = str(e)
            all_fit_results.append(fit_result)

    final_results_df = pd.DataFrame(all_fit_results)
    return final_results_df, plots_dict


def create_summary_plot(data_df, q_squared_col, gamma_cols, method_names=None, gamma_unit='1/s'):
    """
    Create summary plot (Γ vs q²) and return the figure

    Returns:
        matplotlib.figure.Figure: The summary plot figure
    """
    import statsmodels.api as sm

    if not isinstance(gamma_cols, list):
        gamma_cols = [gamma_cols]

    if method_names is None:
        method_names = gamma_cols
    elif not isinstance(method_names, list):
        method_names = [method_names]

    # Create figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    X_full = data_df[q_squared_col]

    colors = ['blue', 'green', 'red', 'orange', 'purple']

    for i, gamma_col in enumerate(gamma_cols):
        if gamma_col not in data_df.columns:
            continue

        method_name = method_names[i] if i < len(method_names) else ''
        Y_full = data_df[gamma_col]

        # Scatter plot
        ax.scatter(X_full, Y_full, alpha=0.6, label=f'{method_name} Data',
                  color=colors[i % len(colors)])

        # Linear regression
        X_fit_with_constant = sm.add_constant(X_full)
        model = sm.OLS(Y_full, X_fit_with_constant).fit()

        # Plot regression line
        X_line = np.linspace(X_full.min(), X_full.max(), 100)
        X_line_with_constant = sm.add_constant(X_line)
        y_predicted_line = model.predict(X_line_with_constant)
        ax.plot(X_line, y_predicted_line, '-',
               label=f'{method_name} Fit (R²={model.rsquared:.4f})',
               color=colors[i % len(colors)])

    ax.set_xlabel(f'q² [nm⁻²]')
    ax.set_ylabel(f'Γ [{gamma_unit}]')
    ax.set_title('Diffusion Coefficient Analysis')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    return fig
