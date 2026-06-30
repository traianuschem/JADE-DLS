"""
Cumulant Plotting Functions (No Show)
Modified versions of cumulant plotting functions that return figures instead of showing them
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import linregress
from scipy.optimize import curve_fit
import inspect


def plot_processed_correlations_no_show(dataframes_dict, fit_x_limits, xlim=None, ylim=None):
    """
    Linear OLS fit of ln√(g²−1) vs lag time for Method B (JADE 2.2).
    Uses scipy.stats.linregress instead of curve_fit.

    Returns:
        tuple: (results_df, plots_dict)
            - results_df: DataFrame with Gamma, Gamma_error, a, R_squared
            - plots_dict: Dictionary {filename: (fig, fit_data)}
    """
    from matplotlib.figure import Figure

    all_fit_results = []
    plots_dict = {}
    plot_number = 1

    for name, df in dataframes_dict.items():
        if df is None:
            print(f"Skipping '{name}': no data (None).")
            continue
        try:
            fit_result = {'filename': name}

            x_data = df['t [s]']
            y_data = df['g(2)_mod']
            x_fit = x_data[(x_data >= fit_x_limits[0]) & (x_data <= fit_x_limits[1])]
            y_fit = y_data[(x_data >= fit_x_limits[0]) & (x_data <= fit_x_limits[1])]

            slope, intercept, r, p, se = linregress(x_fit, y_fit)

            # Γ = −slope,  a = exp(2·intercept)
            Gamma = -slope
            a = np.exp(2 * intercept)

            fit_result['Gamma'] = Gamma
            fit_result['Gamma_error'] = se
            fit_result['Gamma_relative_error'] = (se / abs(Gamma)) * 100 if Gamma != 0 else np.inf
            fit_result['a'] = a

            # Fit line over full x range (JADE 2.2 style)
            y_fit_values = slope * x_data + intercept

            # Residuals (in fit window only)
            residuals = y_fit - (slope * x_fit + intercept)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
            r_squared = 1 - (ss_res / ss_tot)
            fit_result['R_squared'] = r_squared

            # 3-panel figure (no plt.show — uses Figure API)
            fig = Figure(figsize=(20, 5))
            ax1, ax2, ax3 = fig.subplots(1, 3)
            fig.suptitle(f'[{plot_number}]: Method B — {name}', fontsize=12)

            # JADE 2.2 approach: default xlim = fit window so the linear region is visible
            # (the user can zoom/pan from there if needed)
            effective_xlim = xlim if xlim is not None else fit_x_limits

            # Panel 1: data + fit (zoomed to fit window by default, like JADE 2.2)
            ax1.plot(x_data, y_data, 'o', alpha=0.6, markersize=4, label='Data', zorder=2)
            ax1.plot(x_data, y_fit_values, 'r-', linewidth=2, label='Linear fit', zorder=3)
            ax1.set_xlabel(r'lag time τ [s]')
            ax1.set_ylabel(r'ln$\sqrt{g^{(2)}(\tau)-1}$')
            ax1.set_title('Data & Fit')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(*effective_xlim)
            # Compute proper ylim from the data AND fit within the visible x window
            x_arr = np.asarray(x_data)
            y_arr = np.asarray(y_data)
            mask_vis = (x_arr >= effective_xlim[0]) & (x_arr <= effective_xlim[1])
            y_vis = y_arr[mask_vis]
            yfit_vis = np.asarray(y_fit_values)[mask_vis]
            all_y_vis = np.concatenate([y_vis, yfit_vis])
            all_y_vis = all_y_vis[~np.isnan(all_y_vis)]
            if len(all_y_vis) > 0:
                y_lo = float(np.min(all_y_vis))
                y_hi = float(np.max(all_y_vis))
                y_margin = max(abs(y_hi - y_lo) * 0.20, 0.1)
                ax1.set_ylim(y_lo - y_margin, y_hi + y_margin)
            if ylim is not None:
                ax1.set_ylim(*ylim)
            ax1.legend(fontsize=8)
            ax1.text(0.95, 0.95,
                     f"R² = {r_squared:.4f}\n⟨Γ⟩ = {Gamma:.3e} s⁻¹",
                     transform=ax1.transAxes, va='top', ha='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Panel 2: residuals
            ax2.plot(residuals.values if hasattr(residuals, 'values') else residuals)
            ax2.axhline(0, color='r', linestyle='--', linewidth=1)
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals')
            ax2.grid(True, alpha=0.3)

            # Panel 3: Q-Q
            stats.probplot(residuals, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot')
            ax3.grid(True, alpha=0.3)

            fig.tight_layout()

            plots_dict[name] = (fig, {
                'Gamma': Gamma,
                'Gamma_error': se,
                'r_squared': r_squared,
                'residuals': np.array(residuals),
            })

            plot_number += 1
            all_fit_results.append(fit_result)

        except (KeyError, TypeError) as e:
            print(f"Error processing DataFrame '{name}': {e}")
            fit_result['Error'] = str(e)
            all_fit_results.append(fit_result)

    final_results_df = pd.DataFrame(all_fit_results)
    return final_results_df, plots_dict


def _fit_single_dataset_method_c(name, df, fit_function, fit_limits, initial_parameters,
                                  method, max_iterations, tolerance, plot_number):
    """
    Worker for parallel Method C fitting (joblib). Implements JADE 2.2 zoom-grid search.

    Returns:
        tuple: (fit_result dict, plot_data tuple)
    """
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy import stats
    import inspect
    from matplotlib.figure import Figure
    from ade_dls.analysis.cumulants_C import (
        _BOUNDS_LOWER, _BOUNDS_UPPER, get_meaningful_parameters
    )

    fit_result = {'filename': name}
    param_names = inspect.getfullargspec(fit_function).args[1:]

    _ZOOM_ROUNDS = [
        dict(half_decades=1.5, n_points=5, maxfev=1_000),
        dict(half_decades=0.5, n_points=5, maxfev=5_000),
        dict(half_decades=0.15, n_points=5, maxfev=50_000),
    ]

    try:
        x_data = df['t [s]'].values
        y_data = df['g(2)-1'].values
        mask = (x_data >= fit_limits[0]) & (x_data <= fit_limits[1])
        x_fit = x_data[mask]
        y_fit = y_data[mask]

        if len(x_fit) < 2:
            print(f"[Worker] Not enough data points in range for {name}. Skipping.")
            return (fit_result, None)

        def _metrics(popt):
            yc = fit_function(x_fit, *popt)
            res = y_fit - yc
            ss_res = np.sum(res**2)
            ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            n = len(y_fit)
            k = len(popt) + 1
            return r2, float(np.sqrt(ss_res / n)), float(n * np.log(ss_res / n) + 2 * k)

        def _do_fit(p0, maxfev_override=None):
            mfev = maxfev_override if maxfev_override is not None else 50000
            if method == 'lm':
                return curve_fit(fit_function, x_fit, y_fit, p0=p0, maxfev=mfev)
            bounds = ([_BOUNDS_LOWER[pn] for pn in param_names],
                      [_BOUNDS_UPPER[pn] for pn in param_names])
            return curve_fit(fit_function, x_fit, y_fit, p0=p0,
                             method=method, bounds=bounds, maxfev=mfev)

        all_y_fits = []
        all_r_squared = []
        best_popt = None
        best_pcov = None
        pcov = None
        _is_adaptive = isinstance(initial_parameters, dict)

        if _is_adaptive:
            guess = initial_parameters.get(name, list(initial_parameters.values())[0]).copy()
            b_center = float(guess[1])
            a_base   = float(guess[0])
            f_base   = float(guess[-1])
            c_base   = float(guess[param_names.index('c')]) if 'c' in param_names else None
            d_base   = float(guess[param_names.index('d')]) if 'd' in param_names else None
            e_base   = float(guess[param_names.index('e')]) if 'e' in param_names else None

            best_r2 = -np.inf
            prev_r2 = -np.inf

            for rnd in _ZOOM_ROUNDS:
                b_grid = b_center * np.logspace(-rnd['half_decades'], rnd['half_decades'], rnd['n_points'])
                rnd_best_popt = None
                rnd_best_pcov = None
                rnd_best_r2   = -np.inf

                for b_i in b_grid:
                    scale = b_i / b_center
                    p0 = []
                    for pn in param_names:
                        if   pn == 'a': p0.append(a_base)
                        elif pn == 'b': p0.append(float(b_i))
                        elif pn == 'c': p0.append(c_base * scale**2)
                        elif pn == 'd': p0.append(d_base * scale**3)
                        elif pn == 'e': p0.append(e_base * scale**4)
                        elif pn == 'f': p0.append(f_base)
                    try:
                        popt_i, pcov_i = _do_fit(p0, rnd['maxfev'])
                        r2_i, _, _     = _metrics(popt_i)
                        all_y_fits.append((popt_i, fit_function(x_data, *popt_i)))
                        all_r_squared.append(r2_i)
                        if r2_i > rnd_best_r2:
                            rnd_best_r2   = r2_i
                            rnd_best_popt = popt_i
                            rnd_best_pcov = pcov_i
                    except RuntimeError:
                        continue

                if rnd_best_popt is None:
                    continue
                if rnd_best_r2 > best_r2:
                    best_r2   = rnd_best_r2
                    best_popt = rnd_best_popt
                    best_pcov = rnd_best_pcov

                b_center = float(best_popt[1])
                a_base   = float(best_popt[0])
                f_base   = float(best_popt[-1])
                if 'c' in param_names: c_base = float(best_popt[param_names.index('c')])
                if 'd' in param_names: d_base = float(best_popt[param_names.index('d')])
                if 'e' in param_names: e_base = float(best_popt[param_names.index('e')])

                if (best_r2 - prev_r2) < tolerance:
                    break
                prev_r2 = best_r2

            if best_popt is None:
                raise RuntimeError("All zoom grid points failed to converge.")
            pcov = best_pcov

        else:
            current_guess = get_meaningful_parameters(fit_function, initial_parameters)
            for i in range(max_iterations):
                try:
                    popt, pcov = _do_fit(current_guess)
                except RuntimeError as e:
                    print(f"[Worker] Fit error in iteration {i+1} for {name}: {e}")
                    break
                all_y_fits.append((popt, fit_function(x_data, *popt)))
                current_guess = popt
                r2, _, _ = _metrics(popt)
                all_r_squared.append(r2)
                if i > 0 and abs(all_r_squared[-1] - all_r_squared[-2]) < tolerance:
                    break

        if not all_r_squared:
            raise RuntimeError("No successful fit attempts.")

        best_idx              = int(np.argmax(all_r_squared))
        best_popt_final, best_y_fit = all_y_fits[best_idx]
        best_r2_val           = all_r_squared[best_idx]

        yc_best       = fit_function(x_fit, *best_popt_final)
        best_residuals = y_fit - yc_best
        ss_res_best   = np.sum(best_residuals**2)
        n             = len(y_fit)
        k             = len(best_popt_final) + 1

        fit_result['best_R-squared'] = best_r2_val
        fit_result['best_RMSE']      = float(np.sqrt(ss_res_best / n))
        fit_result['best_AIC']       = float(n * np.log(ss_res_best / n) + 2 * k)

        for j, pn in enumerate(param_names):
            pv = best_popt_final[j]
            fit_result['best_' + pn] = float(pv.item() if isinstance(pv, np.ndarray) and pv.size == 1 else pv)

        if pcov is not None:
            perr = np.sqrt(np.diag(pcov))
            for j, pn in enumerate(param_names):
                if j < len(perr):
                    ev = perr[j]
                    fit_result['err_' + pn] = float(ev.item() if isinstance(ev, np.ndarray) and ev.size == 1 else ev)

        if fit_function.__name__ == 'fit_function4':
            c_p = best_popt_final[2]
            e_p = best_popt_final[4]
            fit_result['kurtosis'] = float(e_p / c_p**2) if c_p != 0 else float('nan')

        # Create figure
        fig = Figure(figsize=(20, 5))
        ax1, ax2, ax3 = fig.subplots(1, 3)
        fig.suptitle(f'[{plot_number}]: Method C ({fit_function.__name__}) — {name}', fontsize=12)

        ax1.plot(x_data, y_data, 'o', alpha=0.6, markersize=4, label='Data')
        ax1.plot(x_data, best_y_fit, 'r-', linewidth=2, label='Best fit')
        ax1.set_xlabel(r'lag time τ [s]')
        ax1.set_ylabel(r'$g^{(2)}(\tau) - 1$')
        ax1.set_title('Data & Fit')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_xlim(1e-6, 10)
        ax1.legend()
        ax1.text(0.95, 0.95,
                 f"R² = {best_r2_val:.4f}\n⟨Γ⟩ = {best_popt_final[1]:.3e} s⁻¹",
                 transform=ax1.transAxes, va='top', ha='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax2.plot(best_residuals)
        ax2.axhline(0, color='r', linestyle='--', linewidth=1)
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals')
        ax2.grid(True, alpha=0.3)

        stats.probplot(best_residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot')
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()

        return (fit_result, (fig, {
            'popt': best_popt_final,
            'r_squared': best_r2_val,
            'residuals': best_residuals,
        }))

    except RuntimeError as e:
        print(f"[WARNING] Worker '{name}': fit failed — {e}. Filling with NaN.")
        fit_result['Error'] = str(e)
        for col in ['best_R-squared', 'best_RMSE', 'best_AIC']:
            fit_result.setdefault(col, np.nan)
        for pn in param_names:
            fit_result.setdefault('best_' + pn, np.nan)
            fit_result.setdefault('err_'  + pn, np.nan)
        return (fit_result, None)

    except Exception as e:
        print(f"[Worker] Error processing '{name}': {e}")
        import traceback
        traceback.print_exc()
        fit_result['Error'] = str(e)
        return (fit_result, None)


def plot_processed_correlations_iterative_no_show(dataframes_dict, fit_function, fit_limits,
                                                   initial_parameters, method='lm',
                                                   max_iterations=10, tolerance=1e-4,
                                                   use_multiprocessing=False):
    """
    Adaptive zoom-grid search for Method C (JADE 2.2), GUI-optimised.
    Delegates fitting to cumulants_C.plot_processed_correlations_iterative,
    then builds matplotlib.figure.Figure objects (no plt.show) per dataset.

    Returns:
        tuple: (results_df, plots_dict)
            - results_df: DataFrame with best_b, best_c, best_R-squared, best_RMSE, …
            - plots_dict: {filename: (fig, {'popt', 'r_squared', 'residuals'})}
    """
    from matplotlib.figure import Figure
    from ade_dls.analysis.cumulants_C import plot_processed_correlations_iterative

    total = len(dataframes_dict)
    print(f"[Method C] Fitting {total} datasets (JADE 2.2 adaptive zoom-grid search)...")

    # Run the fitting without plots — uses adaptive/expert mode, zoom-grid, correct bounds
    results_df = plot_processed_correlations_iterative(
        dataframes_dict, fit_function, fit_limits, initial_parameters,
        max_iterations=max_iterations, tolerance=tolerance,
        method=method, show_plots=False
    )

    # Build GUI figures from fit results
    param_names = inspect.getfullargspec(fit_function).args[1:]
    plots_dict  = {}
    plot_number = 1

    for name, df in dataframes_dict.items():
        rows = results_df[results_df['filename'] == name]
        if rows.empty:
            continue
        row = rows.iloc[0]

        # Skip datasets where the fit failed
        if 'Error' in results_df.columns and pd.notna(row.get('Error', np.nan)):
            continue

        try:
            x_data = df['t [s]'].values
            y_data = df['g(2)-1'].values
            mask   = (x_data >= fit_limits[0]) & (x_data <= fit_limits[1])
            x_fit  = x_data[mask]
            y_fit  = y_data[mask]

            # Reconstruct best popt from result columns
            best_popt      = np.array([float(row[f'best_{pn}']) for pn in param_names])
            best_y_fit     = fit_function(x_data, *best_popt)
            best_residuals = y_fit - fit_function(x_fit, *best_popt)
            r_squared      = float(row['best_R-squared'])

            fig = Figure(figsize=(20, 5))
            ax1, ax2, ax3 = fig.subplots(1, 3)
            fig.suptitle(f'[{plot_number}]: Method C ({fit_function.__name__}) — {name}',
                         fontsize=12)

            ax1.plot(x_data, y_data, 'o', alpha=0.6, markersize=4, label='Data')
            ax1.plot(x_data, best_y_fit, 'r-', linewidth=2, label='Best fit')
            ax1.set_xlabel(r'lag time τ [s]')
            ax1.set_ylabel(r'$g^{(2)}(\tau) - 1$')
            ax1.set_title('Data & Fit')
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log')
            ax1.set_xlim(1e-6, 10)
            ax1.legend()
            ax1.text(0.95, 0.95,
                     f"R² = {r_squared:.4f}\n⟨Γ⟩ = {best_popt[1]:.3e} s⁻¹",
                     transform=ax1.transAxes, va='top', ha='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            ax2.plot(best_residuals)
            ax2.axhline(0, color='r', linestyle='--', linewidth=1)
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals')
            ax2.grid(True, alpha=0.3)

            stats.probplot(best_residuals, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot')
            ax3.grid(True, alpha=0.3)

            fig.tight_layout()

            plots_dict[name] = (fig, {
                'popt':       best_popt,
                'r_squared':  r_squared,
                'residuals':  best_residuals,
            })

            plot_number += 1

        except Exception as e:
            print(f"[Method C] Figure creation failed for '{name}': {e}")

    print(f"[Method C] Generated {len(plots_dict)} plots from {total} datasets")
    return results_df, plots_dict


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

    # Create figure (use Figure() to avoid memory leak)
    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    X_full = data_df[q_squared_col]

    colors = ['blue', 'green', 'red', 'orange', 'purple']

    for i, gamma_col in enumerate(gamma_cols):
        if gamma_col not in data_df.columns:
            continue

        method_name = method_names[i] if i < len(method_names) else ''
        Y_full = data_df[gamma_col]

        # Scatter plot with larger, more visible markers
        ax.scatter(X_full, Y_full, alpha=0.7, s=20,
                  label=f'{method_name} Data',
                  color=colors[i % len(colors)],
                  marker='o', edgecolors='black', linewidths=0.5)

        # Linear regression
        X_fit_with_constant = sm.add_constant(X_full)
        model = sm.OLS(Y_full, X_fit_with_constant).fit()

        # Plot regression line with thinner line to emphasize data points
        X_line = np.linspace(X_full.min(), X_full.max(), 100)
        X_line_with_constant = sm.add_constant(X_line)
        y_predicted_line = model.predict(X_line_with_constant)
        ax.plot(X_line, y_predicted_line, '-', linewidth=2,
               label=f'{method_name} Fit (R²={model.rsquared:.4f})',
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel(f'q² [nm⁻²]')
    ax.set_ylabel(f'Γ [{gamma_unit}]')
    ax.set_title('Diffusion Coefficient Analysis')
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    return fig


def create_clustering_overview_figure(clustered_df, cluster_info, q_squared_col='q^2'):
    """
    Create a 2-panel clustering overview figure:
      Left:  D = Γ/q² vs q² scatter coloured by population
      Right: log₁₀(D) histogram coloured by population

    Returns a matplotlib.figure.Figure (safe for Qt embedding — no plt.show()).
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    COLORS = plt.cm.tab10.colors
    reliable_pops = cluster_info.get('reliable_populations', [])
    q2 = clustered_df[q_squared_col].values

    for i, pop_num in enumerate(reliable_pops):
        col = f'gamma_pop{pop_num}'
        if col not in clustered_df.columns:
            continue
        gamma = clustered_df[col].values
        mask  = ~np.isnan(gamma) & (q2 > 0)
        if not mask.any():
            continue
        D_pm2  = (gamma[mask] / q2[mask]) * 1e12   # 10⁻¹² m²/s
        log_D  = np.log10(gamma[mask] / q2[mask])
        color  = COLORS[i % len(COLORS)]
        label  = f'Population {pop_num}'
        ax1.scatter(q2[mask], D_pm2, color=color, label=label, s=30, alpha=0.75)
        ax2.hist(log_D, bins=15, color=color, alpha=0.6, label=label)

    ax1.set_xlabel('q² [nm⁻²]')
    ax1.set_ylabel('D [10⁻¹² m²/s]')
    ax1.set_title('D vs q² by Population')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('log₁₀(D [m²/s])')
    ax2.set_ylabel('Count')
    ax2.set_title('D Distribution by Population')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return fig


def create_population_ols_figure(clustered_df, cluster_info, q_squared_col='q^2'):
    """
    Create a Γ vs q² figure with one scatter+OLS line per reliable population.

    Each population's NaN rows are excluded individually so populations with
    sparse coverage don't affect each other's regression.

    Returns a matplotlib.figure.Figure (safe for Qt embedding — no plt.show()).
    """
    import statsmodels.api as sm
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 5))
    ax  = fig.add_subplot(111)

    COLORS = plt.cm.tab10.colors
    reliable_pops = cluster_info.get('reliable_populations', [])
    q2 = clustered_df[q_squared_col].values

    for i, pop_num in enumerate(reliable_pops):
        col = f'gamma_pop{pop_num}'
        if col not in clustered_df.columns:
            continue
        gamma = clustered_df[col].values
        mask  = ~np.isnan(gamma) & (q2 > 0)
        if mask.sum() < 2:
            continue
        color = COLORS[i % len(COLORS)]
        ax.scatter(q2[mask], gamma[mask], color=color,
                   label=f'Pop {pop_num}', s=30, alpha=0.75)
        # OLS line
        X = sm.add_constant(q2[mask])
        model = sm.OLS(gamma[mask], X).fit()
        q2_line    = np.linspace(q2[mask].min(), q2[mask].max(), 100)
        gamma_line = model.params[0] + model.params[1] * q2_line
        ax.plot(q2_line, gamma_line, color=color, linewidth=1.5,
                label=f'Pop {pop_num} OLS (R²={model.rsquared:.3f})')

    ax.set_xlabel('q² [nm⁻²]')
    ax.set_ylabel('Γ [1/s]')
    ax.set_title('Γ vs q² per Population')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_clustering_heatmap(summary_df, scatter_df=None, scatter_panels=None):
    """
    Visualise clustering parameter sweep results as heatmaps.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of clustering_parameter_sweep() — one row per (dt, ma).
    scatter_df : pd.DataFrame or None
        Per-point D_t data from clustering_parameter_sweep(). When provided
        and scatter_panels is not None, scatter sub-panels are added.
    scatter_panels : list of (dt, ma, label) or None
        Up to 4 parameter combinations shown as scatter sub-panels.
        When None, only the two heatmaps are drawn.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.gridspec as gridspec

    show_scatter = (scatter_df is not None and scatter_panels is not None
                    and len(scatter_panels) > 0)
    n_scatter = min(len(scatter_panels), 4) if show_scatter else 0

    if show_scatter:
        fig = plt.figure(figsize=(14, 9))
        gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.40)
        ax_heat = fig.add_subplot(gs[0, :2])
        ax_sil  = fig.add_subplot(gs[0, 2:])
        scatter_axes = [fig.add_subplot(gs[1, i]) for i in range(n_scatter)]
    else:
        fig, (ax_heat, ax_sil) = plt.subplots(1, 2, figsize=(12, 5))
        scatter_axes = []

    dts = sorted(summary_df['distance_threshold'].unique())
    mas = sorted(summary_df['min_abundance'].unique())

    # ── (a) heatmap: n_populations ────────────────────────────────────
    pivot_n = summary_df.pivot(index='distance_threshold',
                               columns='min_abundance',
                               values='n_populations')
    pivot_n = pivot_n.reindex(index=dts, columns=mas)
    im1 = ax_heat.imshow(pivot_n.values, aspect='auto',
                         cmap='YlOrRd_r', origin='lower')
    ax_heat.set_xticks(range(len(mas)))
    ax_heat.set_xticklabels([f'{v:.2g}' for v in mas], fontsize=9)
    ax_heat.set_yticks(range(len(dts)))
    ax_heat.set_yticklabels([f'{v:.2g}' for v in dts], fontsize=9)
    ax_heat.set_xlabel('Min abundance', fontsize=11)
    ax_heat.set_ylabel('Distance threshold', fontsize=11)
    ax_heat.set_title('(a) No. of populations', fontsize=11,
                      fontweight='bold', loc='left')
    for i in range(len(dts)):
        for j in range(len(mas)):
            val = pivot_n.values[i, j]
            if not np.isnan(val):
                ax_heat.text(j, i, str(int(val)), ha='center', va='center',
                             fontsize=11, color='black')
    plt.colorbar(im1, ax=ax_heat, label='n populations')

    # ── (b) heatmap: silhouette score ──────────────────────────────────
    pivot_s = summary_df.pivot(index='distance_threshold',
                               columns='min_abundance',
                               values='silhouette')
    pivot_s = pivot_s.reindex(index=dts, columns=mas)
    vmin = max(0.0, float(np.nanmin(pivot_s.values)) - 0.05) if not pivot_s.isnull().all().all() else 0.0
    im2 = ax_sil.imshow(pivot_s.values, aspect='auto',
                        cmap='YlGn', origin='lower',
                        vmin=vmin, vmax=1.0)
    ax_sil.set_xticks(range(len(mas)))
    ax_sil.set_xticklabels([f'{v:.2g}' for v in mas], fontsize=9)
    ax_sil.set_yticks(range(len(dts)))
    ax_sil.set_yticklabels([f'{v:.2g}' for v in dts], fontsize=9)
    ax_sil.set_xlabel('Min abundance', fontsize=11)
    ax_sil.set_ylabel('Distance threshold', fontsize=11)
    ax_sil.set_title('(b) Silhouette score', fontsize=11,
                     fontweight='bold', loc='left')
    for i in range(len(dts)):
        for j in range(len(mas)):
            val = pivot_s.values[i, j]
            if not np.isnan(val):
                ax_sil.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=10, color='black')
    plt.colorbar(im2, ax=ax_sil, label='Silhouette score')

    # ── scatter sub-panels ─────────────────────────────────────────────
    pop_colors = {
        'gamma_pop1': 'royalblue',
        'gamma_pop2': 'darkorange',
        'gamma_pop3': 'forestgreen',
        'gamma_pop4': 'crimson',
    }

    for (dt, ma, plbl), ax in zip(scatter_panels[:4], scatter_axes):
        sub = scatter_df[
            (scatter_df['distance_threshold'] == dt) &
            (scatter_df['min_abundance']       == ma)
        ]

        excl = sub[sub['excluded']]
        if not excl.empty:
            ax.scatter(excl['q^2'], excl['D_t'],
                       color='lightgrey', s=10, alpha=0.5,
                       zorder=1, label='Excluded')

        incl = sub[~sub['excluded']]
        seen = set()
        for pop in incl['population'].unique():
            pts = incl[incl['population'] == pop]
            col = pop_colors.get(pop, 'grey')
            lbl = pop.replace('gamma_', 'Pop ') if pop not in seen else ''
            ax.scatter(pts['q^2'], pts['D_t'],
                       color=col, s=14, alpha=0.8,
                       zorder=3, label=lbl)
            seen.add(pop)

        match = summary_df[
            (summary_df['distance_threshold'] == dt) &
            (summary_df['min_abundance']       == ma)
        ]
        n_str = f'n={int(match["n_populations"].values[0])}' if len(match) else ''

        ax.set_xlabel(r'$q^2$ [nm$^{-2}$]', fontsize=10)
        ax.set_ylabel(r'$D_t$ [m² s⁻¹]', fontsize=10)
        ax.set_title(
            f'{plbl} $d_{{thr}}={dt}$, $f_{{min}}={ma}$ ({n_str})',
            fontsize=10, fontweight='bold', loc='left',
        )
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc='upper right')

    fig.tight_layout()
    return fig
