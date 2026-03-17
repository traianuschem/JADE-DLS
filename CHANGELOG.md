# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1dev] - 2026-03-17

### Added

#### Regularized NNLS – `RegularizedResultsDialog`

- New dedicated results dialog (`regularized_results_dialog.py`) with three tabs:
  - *Summary*: per-peak table with Rh, Error, R², D, Skewness, Kurtosis, Alpha; R² cells color-coded green (> 0.99) / orange (> 0.95) / red (≤ 0.95); automatic sample interpretation (monodisperse / bidisperse / polydisperse)
  - *Detailed Data*: full per-file table with tau, gamma, intensity, area, and FWHM columns
  - *Statistics*: HTML formatted per-peak statistics block with overall mean/min R²
- Export (xlsx/csv/txt) and "Refine Results" buttons

#### Regularized NNLS – Extended per-dataset plots

- 4-panel per-dataset plot layout (was 2-panel): *Data & Fit* | *τ Distribution* | *Residuals* | *Q-Q plot*; figure size 24 × 6 in
- R² annotated directly in the Data & Fit panel

#### Post-fit refinement – Clustering & per-population tabs

- New **"3. Clustering"** tab in `LaplacePostFitRefinementDialog` (applies to both NNLS and Regularized):
  - Live 2-panel clustering preview (D vs q² scatter colored by population + log₁₀(D) histogram); refreshes on demand without closing the dialog
  - Settings: clustering method (Hierarchical – Ward / Simple – gap-based), distance threshold, silhouette-based refinement toggle
- **Per-population tabs** (one per detected population): individual q² range filter per population
- Clustering parameters and per-population q² ranges forwarded through the full refinement flow

#### Clustering – non-interactive mode

- Added `interactive` flag (default `True`) to `cluster_all_gammas()`; when `False`, outliers are flagged but retained automatically without `input()` prompts — required for safe GUI integration

#### Clustering – per-peak → per-population column remapping

- After clustering, peak-indexed columns (`tau_1`, `intensity_1`, `area_1`, …) are remapped to population-indexed equivalents (`tau_pop1`, `intensity_pop1`, `area_pop1`, …) based on the cluster assignment
- Remapped prefixes: `tau`, `intensity`, `normalized_area_percent`, `normalized_sum_percent`, `area`, `fwhm`, `centroid`, `std_dev`, `skewness`, `kurtosis`

#### Method-specific analyzer instances

- `AnalysisView` now maintains separate `laplace_analyzer_nnls` and `laplace_analyzer_regularized` attributes; post-fit refinement dialogs and recalculation calls use the method-specific instance, preventing result cross-contamination when both NNLS and Regularized are run in the same session

#### Startup splash screen

- Custom PyQt5 splash screen in `run.py`: navy background (#1e3a5f), bold "ADE-DLS" title, "Loading, please wait…" subtitle
- Heavy imports (`main_window`) deferred until after the splash is rendered, visibly reducing startup time

#### Plot method filter

- Added "Regularized NNLS" entry to the method-filter dropdown in the Plots tab

### Fixed

- **`'NNLS Analysis'` refinement button grayed out**: `'NNLS Analysis'` was missing from the `_REFINABLE` set in `analysis_view.py`
- **NNLS / Regularized cross-contamination in post-fit refinement**: Refinement dialog previously always operated on `self.laplace_analyzer`; it now selects the method-specific analyzer
- **Clustering dendrogram incompatible with GUI**: Dendrogram plot switched from `plt.figure()` (interactive pyplot) to `matplotlib.figure.Figure` (non-interactive), enabling embedding in Qt dialogs

### Changed

- **Lazy imports in `__init__.py` files**: Removed eager top-level imports from `ade_dls/__init__.py`, `ade_dls/analysis/__init__.py`, `ade_dls/core/__init__.py`, `ade_dls/utils/__init__.py`; modules are now imported on demand for faster startup and to eliminate circular import risks
- **`main()` signature**: `run.py` constructs `QApplication` and the splash before calling `main(app=app, splash=splash)`
- **Regularized clustering parameters**: `calculate_regularized_diffusion_coefficients()` now receives `use_clustering`, `distance_threshold`, and `clustering_strategy` from `RegularizedDialog` parameters

---

## [2.0.3dev] - 2026-03-10

### Added

#### Analysis – Method D (Multi-Exponential / Multi-Population)

- **Cross-file population clustering**: After the per-file multi-exponential decomposition, `cluster_all_gammas()` groups the fitted decay rates across all measurement angles/files into consistent populations. Uses hierarchical clustering (Ward linkage, default) or gap-based clustering. Clustering unconditionally operates on D = Γ/q² (angle-independent diffusion coefficient) — the only physically correct basis for angle-series DLS data.
- **Multi-row results table**: The Results Overview now shows N+1 rows after a Method D run:
  - One row per reliable population: `Rh from Method D (Population N)` — Rh, D, R², Residuals
  - One combined row: `Rh from Method D (combined, N populations)` — Z-average equivalent from `⟨Γ⟩ → OLS → Rh`; also reports PDI, Skewness, Kurtosis from the per-file cumulant fits
- **`MethodDPostFitDialog`** — new two-stage post-fit refinement dialog (opened via the existing refinement button):
  - *Clustering tab*: Re-configure clustering method, number of populations, distance threshold, minimum population abundance, silhouette-based refinement
  - *Population N tabs*: Per-population q² range filter, outlier σ-threshold (residual-based), minimum OLS points
  - *Combined tab*: q² range for the `⟨Γ⟩ → Rh` regression
- **Interactive clustering preview** in the Clustering tab: An embedded 2-panel plot (D vs q² scatter colored by population + log₁₀(D) histogram) is drawn immediately on dialog open and refreshes in-place when "↺ Refresh Clustering Preview" is clicked. A stats line below shows the number of populations found, silhouette score (if available), and per-population abundance.

#### Plots Panel – Method D

Two additional plots are now generated automatically after every Method D run and listed in the Plots panel alongside the existing per-file diagnostic plots:

- **`Method D: Clustering Overview`** — D vs q² scatter (all files/angles, colored by population) + log₁₀(D) histogram. Horizontal bands in the scatter indicate angle-independent, well-separated populations.
- **`Method D: Population OLS`** — Γ vs q² scatter per reliable population with OLS regression line and R² in the legend. Directly comparable to the summary plot but separated by population.

### Fixed

- **Refinement button grayed out after Method D**: `"Method D"` was missing from the `_REFINABLE` set in `analysis_view.py`; the post-fit refinement button is now correctly enabled after a Method D run.
- **Detailed Results not population-specific**: All Method D rows previously showed the same detail HTML (the combined regression stats). Population rows now show a compact view (Rh ± error, D ± error, R², Residuals); the combined row retains the full regression statistics block (Model Statistics, Coefficients, Fit Quality Assessment).
- **Clustering grouped by angle, not by population**: The `normalize_by_q2` option defaulted to `False`, causing the clustering algorithm to operate on raw Γ values. Because Γ = D·q² varies with angle, this grouped measurements by angle range rather than by particle population. The option has been removed; D = Γ/q² is now used unconditionally.

### Changed

- **`CumulantDDialog`**: Removed "Cluster on D = Γ/q² (angle-independent)" checkbox — D-based clustering is now the only supported mode and requires no user action.
- **`MethodDPostFitDialog`** Clustering tab: Removed the same redundant checkbox.

---

## [2.1.0] - 2026-03-05

### Added

#### Analysis – Cumulant Methods
- **Method B & C – Extended output**: Results now include `D [m²/s]`, `D error [m²/s]`, `Residuals` (normality assessment) and `Skewness` (3rd/4th order fits only), matching JADE 2.0 output
- **Normality assessment** (`_normality_status`): New module-level helper using Jarque–Bera and D'Agostino–Omnibus tests plus skewness/kurtosis thresholds; classifies residuals as *Normally distributed*, *Possibly not normally distributed* or *Not normally distributed*
- **Method A – Individual order plots**: Three separate Γ vs q² scatter plots (1st / 2nd / 3rd order) with linear fit, R², D and normality status in the title; generated alongside the existing multi-order summary plot
- **Method C – Order label**: The cumulant fit order (2nd / 3rd / 4th order) is now shown in the `Fit` column of every Method C result row (fresh run and post-refinement)

#### Plots Tab
- **Method filter dropdown**: A second filter combo-box ("All Methods / Method A / Method B / Method C / NNLS") has been added next to the existing plot-type filter; both filters are combined when rebuilding the plot list
- **3-panel per-dataset plots** (Methods B & C): Each dataset plot now shows three panels — *Correlation fit*, *Residuals time-series*, *Q-Q plot* — at 20 × 5 in figure size (previously 2-panel 14 × 5)
- **Method-tagged plot keys**: Plots are keyed as `"B: <filename>"` / `"C: <filename>"` to prevent collision when both methods are active simultaneously

#### Results Tab (UI)
- **Results Overview**: The results panel is renamed from "Current Results" to "Results Overview" and shows a compact summary table
- **Row-click → Detailed Results**: Clicking any row in Results Overview loads the full HTML analysis detail for that run into the Detailed Results panel (stored per-row using `Qt.UserRole`)
- **Per-method replace / append logic**: Methods A and B replace their own result row on re-run; Method C always appends a new row (enabling comparison of different fit orders)
- **Method tag per row**: Each row stores its method name in `Qt.UserRole+1` for reliable identification during replacement and refinement

#### Post-fit Refinement
- **Full refinement history chain**: After refinement the Detailed Results panel of the refined row shows the *original* analysis HTML verbatim, followed by a styled divider ("▶ Post-fit Refinement") and the new refined results — applies to Methods A, B and C
- **Post-fit Refinement Details block** (Method C): The refined row's Detailed Results includes a summary table with *files included / excluded*, *q-range applied*, a list of excluded files and an *Original vs. Refined* comparison table (Rh, D, R²)
- **"Open Refinement" button driven by row selection**: The method-selection dropdown has been removed; instead the button is enabled only when a row is selected in Results Overview and is scoped to that row's method automatically

### Changed
- **Cumulant Analysis menu**: Split the single "Cumulant Analysis" menu entry into a sub-menu with separate actions for *Method A – ALV Cumulant Data*, *Method B – Linear Fit* and *Method C – Iterative Non-Linear Fit*
- **`run_cumulant_analysis()`**: Refactored into three focused methods (`run_cumulant_a`, `run_cumulant_b`, `run_cumulant_c`) with a shared `_prepare_cumulant_analyzer()` helper
- **`_recompute_method_c()`**: Now delegates entirely to `analyzer.recompute_method_c_diffusion()` instead of performing a manual OLS regression; returns a `(result_df, refinement_stats)` tuple
- **Plot list rebuild no longer clears on re-run**: `clear_results()` is no longer called before `_display_cumulant_results()`; plots and results from previous methods are preserved when a new method runs

### Fixed
- **3-panel plot display**: `_show_plot_by_item()` now correctly handles `n == 3` axes (previously fell through to a single subplot, stacking all three axes on top of each other)
- **Method B per-dataset plots missing**: Broadened the exception handler in `plot_processed_correlations_no_show()` from `(KeyError, TypeError, RuntimeError)` to `Exception` so that `ValueError` from `scipy.stats.probplot` no longer aborts the entire plot loop
- **Import paths**: Fixed stale module references in `cumulant_analyzer.py` (`preprocessing`, `cumulants`) to use fully-qualified `ade_dls.*` package paths
- **Column name alignment**: `cumulant_plotting.py` now reads `t [s]` and `g(2)-1` to match the column names produced by the preprocessing pipeline (previously `t (s)` / `g(2)`)
- **Fit curve not clipped to fit window**: Method B fit curve is now plotted only within `fit_x_limits` (restricts the log-quadratic fit line to the valid region)
- **Post-fit Refinement Details safe for Methods A/B**: The "Files included / excluded" rows are suppressed when `n_total` is `None` (Methods A and B do not have per-file exclusion)

---

## [2.0.0] - 2025-01-16

### Changed
- **BREAKING**: Renamed project from JADE-DLS to ADE-DLS (Angular Dependent Evaluator for Dynamic Light Scattering)
  - Removed "Jupyter-based" from name as project now features a full PyQt5 GUI
- **BREAKING**: Complete package restructuring to follow Python best practices
  - Reorganized as proper installable package `ade_dls`
  - Added `pyproject.toml` for modern Python packaging
  - Moved all modules into structured package hierarchy
- **BREAKING**: Changed license from CC-BY-NC-SA 4.0 to GPL-3.0+ (required by PyQt5)

### Added
- Proper package structure with `ade_dls` namespace
  - `ade_dls.core`: Data loading and preprocessing
  - `ade_dls.analysis`: Analysis algorithms (cumulants, regularized methods, peak clustering)
  - `ade_dls.gui`: PyQt5 GUI application
  - `ade_dls.utils`: Utility functions
- Installation support via pip (`pip install ade-dls`)
- Command-line entry point: `ade-dls` launches GUI
- Enhanced documentation structure
- CONTRIBUTING.md for contributor guidelines
- Optional dependency groups: `[gui]`, `[export]`, `[dev]`, `[all]`

### Improved
- Consolidated requirements into `pyproject.toml`
- Better import organization
- Preparation for test suite and CI/CD

### Fixed
- Import paths now follow standard Python package conventions

## [1.0.0] - 2024

### Added
- Initial release as JADE-DLS
- Jupyter notebook-based analysis
- PyQt5 GUI application
- Cumulant analysis methods (A, B, C)
- NNLS and regularized NNLS methods
- Peak clustering algorithms
- ALV .ASC file support
- Data filtering and preprocessing
- Export to Jupyter notebooks and Python scripts
- Batch processing capabilities
- Comprehensive visualization

### Analysis Methods
- Method A: ALV software pre-calculated fits
- Method B: Linear fit of ln[g(τ)^0.5] vs τ
- Method C: Iterative non-linear least squares
- NNLS (Non-Negative Least Squares)
- Regularized NNLS with Tikhonov-Phillips regularization
- Advanced constraints: normalization, sparsity, unimodality

---

## Migration Guide: v1.0 → v2.0

### Import Changes

**Old (v1.0):**
```python
import preprocessing
import cumulants
import regularized
from gui.main_window import MainWindow
```

**New (v2.0):**
```python
from ade_dls.core import preprocessing
from ade_dls.analysis import cumulants, regularized
from ade_dls.gui.main_window import MainWindow
```

### Installation Changes

**Old (v1.0):**
```bash
git clone repo
python jade-dls-gui.py
```

**New (v2.0):**
```bash
pip install ade-dls[all]
ade-dls  # or: python -m ade_dls.gui.main_window
```

### Configuration

All functionality remains the same; only import paths have changed.

---

[2.1dev]: https://github.com/traianuschem/JADE-DLS/compare/v2.0.3dev...v2.1dev
[2.0.3dev]: https://github.com/traianuschem/JADE-DLS/compare/v2.0.2dev...v2.0.3dev
[2.1.0]: https://github.com/traianuschem/JADE-DLS/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/traianuschem/JADE-DLS/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/traianuschem/JADE-DLS/releases/tag/v1.0.0
