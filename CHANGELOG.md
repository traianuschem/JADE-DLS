# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[2.1.0]: https://github.com/traianuschem/JADE-DLS/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/traianuschem/JADE-DLS/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/traianuschem/JADE-DLS/releases/tag/v1.0.0
