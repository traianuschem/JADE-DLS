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
                                                   initial_parameters, method='lm',
                                                   max_iterations=10, tolerance=1e-4):
    """
    Iterative non-linear fit without showing plots (matches original notebook implementation)

    Args:
        dataframes_dict: Dictionary of dataframes with 't (s)' and 'g(2)' columns
        fit_function: The fit function to use
        fit_limits: Tuple of (min, max) time limits
        initial_parameters: Initial parameter guesses (dict or list)
        method: Optimization method ('lm', 'trf', 'dogbox')
        max_iterations: Maximum number of iterations (default: 10)
        tolerance: Convergence tolerance for R-squared (default: 1e-4)

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

            if len(x_fit) < 2:
                print(f"Not enough data points in the specified range for fitting {name}. Skipping Fit.")
                all_fit_results.append(fit_result)
                continue

            # Determine which parameters to use based on fit_function
            param_names = inspect.getfullargspec(fit_function).args[1:]
            num_params = len(param_names)

            # Get initial parameters for this function
            if isinstance(initial_parameters, dict):
                # If it's a dict, use parameters for this specific dataset (adaptive mode)
                current_guess = initial_parameters.get(name, initial_parameters.get('default', []))[:num_params].copy()
            elif isinstance(initial_parameters, list):
                current_guess = initial_parameters[:num_params].copy()
            else:
                current_guess = initial_parameters

            # Store iterations for analysis
            all_y_fits = []
            all_r_squared = []
            all_rmse = []
            all_aic = []
            all_popt = []
            all_pcov = []

            # Iterative fitting loop (key difference from simple fit!)
            for i in range(max_iterations):
                try:
                    # Perform fit with current guess
                    if method == 'lm':
                        popt, pcov = curve_fit(fit_function, x_fit, y_fit,
                                              p0=current_guess, maxfev=50000)
                    else:
                        # For 'trf' and 'dogbox', use wide bounds
                        bounds = ([-np.inf] * len(current_guess), [np.inf] * len(current_guess))
                        popt, pcov = curve_fit(fit_function, x_fit, y_fit,
                                              p0=current_guess, method=method,
                                              bounds=bounds, maxfev=50000)

                    # Generate fit curve
                    y_fit_values = fit_function(x_data, *popt)
                    all_y_fits.append(y_fit_values)
                    all_popt.append(popt)
                    all_pcov.append(pcov)

                    # Use fitted parameters as next initial guess
                    current_guess = popt.copy()

                    # Calculate residuals (only in fitting range)
                    y_fit_current = fit_function(x_fit, *popt)
                    residuals = y_fit - y_fit_current

                    # Calculate R-squared
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    all_r_squared.append(r_squared)

                    # Calculate RMSE
                    n = len(y_fit)
                    rmse = np.sqrt(ss_res / n)
                    all_rmse.append(rmse)

                    # Calculate AIC (Akaike Information Criterion)
                    k = len(popt) + 1
                    aic = n * np.log(ss_res / n) + 2 * k
                    all_aic.append(aic)

                    # Store metrics for this iteration
                    fit_result[f'R-squared_iter{i+1}'] = r_squared
                    fit_result[f'RMSE_iter{i+1}'] = rmse
                    fit_result[f'AIC_iter{i+1}'] = aic

                    # Store parameters for this iteration
                    for j, param_name in enumerate(param_names):
                        fit_result[f'{param_name}_iter{i+1}'] = popt[j]

                    # Check for convergence
                    if i > 0:
                        delta_r_squared = abs(all_r_squared[i] - all_r_squared[i-1])
                        if delta_r_squared < tolerance:
                            print(f"Converged after {i+1} iterations for {name}.")
                            break

                except RuntimeError as e:
                    print(f"Fit error in iteration {i+1} for DataFrame '{name}': {e}")
                    break

            # Find the best iteration (using R-squared)
            if len(all_r_squared) > 0:
                best_iteration_index = np.argmax(all_r_squared)
                best_y_fit = all_y_fits[best_iteration_index]
                best_popt = all_popt[best_iteration_index]
                best_pcov = all_pcov[best_iteration_index]
                best_iteration = best_iteration_index + 1

                # Calculate errors from best fit
                perr = np.sqrt(np.diag(best_pcov))

                # Calculate best residuals
                best_y_fit_in_range = fit_function(x_fit, *best_popt)
                residuals = y_fit - best_y_fit_in_range

                # Generate smooth fit curve for plotting
                x_plot = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 1000)
                y_fit_plot = fit_function(x_plot, *best_popt)

                # Store best fit results
                r_squared = all_r_squared[best_iteration_index]
                rmse = all_rmse[best_iteration_index]
                aic = all_aic[best_iteration_index]

                # Store results with 'best_' prefix
                for i, param_name in enumerate(param_names):
                    fit_result[f'best_{param_name}'] = best_popt[i]
                    fit_result[f'best_{param_name}_error'] = perr[i]

                fit_result['R_squared'] = r_squared
                fit_result['RMSE'] = rmse
                fit_result['AIC'] = aic
            else:
                # No successful iterations
                print(f"No successful fits for {name}")
                all_fit_results.append(fit_result)
                continue

            # Create figure (2x1 layout like original notebook)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Left plot: Data and all fit iterations
            ax1.plot(x_data, y_data, marker='.', linestyle='', label='Data')

            # Plot all iterations with increasing opacity
            for i, y_fit_iter in enumerate(all_y_fits):
                if i == best_iteration_index:
                    # Highlight the best fit
                    ax1.plot(x_data, y_fit_iter, 'r-', linewidth=2,
                            label=f'Best Fit (Iter {i+1})')
                else:
                    # Plot other iterations with lower opacity
                    opacity = 0.3 + 0.5 * (i / len(all_y_fits))
                    ax1.plot(x_data, y_fit_iter, '-', alpha=opacity,
                            color='gray', linewidth=1)

            ax1.set_xlabel('lag time [s]')
            ax1.set_ylabel('g(2)-1')
            ax1.set_title(f'[{plot_number}]: g(2)-1 vs. lag time for {name}')
            ax1.grid(True)
            ax1.set_xscale('log')
            ax1.set_xlim(left=x_data.min()*0.9, right=x_data.max()*1.1)
            ax1.legend()

            # Add parameters of best fit as text
            param_text = f"Best Fit (Iteration {best_iteration}):\n"
            param_text += f"R² = {r_squared:.4f}\n"
            param_text += f"RMSE = {rmse:.4e}\n"
            param_text += f"AIC = {aic:.2f}"
            ax1.text(0.95, 0.95, param_text, transform=ax1.transAxes,
                    verticalalignment='top', horizontalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Right plot: Q-Q plot for the best fit
            stats.probplot(residuals, dist="norm", plot=ax2)
            ax2.set_title(f'[{plot_number}]: Q-Q Plot of Residuals (Best Fit: Iter {best_iteration})')
            ax2.grid(True)

            plt.tight_layout()

            # Store figure
            plots_dict[name] = (fig, {
                'popt': best_popt,
                'perr': perr,
                'r_squared': r_squared,
                'rmse': rmse,
                'aic': aic,
                'residuals': residuals,
                'num_iterations': len(all_y_fits),
                'best_iteration': best_iteration
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
