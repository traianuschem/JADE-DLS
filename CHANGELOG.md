# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.2.3] - 2026-06-30

### Added

- **Clustering parameter sweep** (`ade_dls/analysis/clustering.py`): new function `clustering_parameter_sweep()` runs `cluster_all_gammas()` over a full grid of `distance_threshold × min_abundance` combinations (default: 5 × 5 = 25 runs) and returns two DataFrames — `summary_df` (one row per combination: n_populations, silhouette score, per-population D_mean/D_std/abundance) and `scatter_df` (per data-point D_t values flagged as included or excluded). Plotting is suppressed during the sweep; figure and Qt objects are cleaned up between runs.

- **Clustering heatmap plot** (`ade_dls/gui/analysis/cumulant_plotting.py`): new function `plot_clustering_heatmap()` visualises the sweep results:
  - *Panel (a)* — heatmap of number of populations (`YlOrRd_r` colormap, cell values annotated).
  - *Panel (b)* — heatmap of silhouette score (`YlGn` colormap, cell values annotated).
  - Optional scatter sub-panels (c)–(f): D_t vs q² scatter for up to four selected (dt, ma) combinations, with populations colour-coded and excluded points shown in grey. Layout auto-adapts between a 2-panel (heatmaps only) and a 2×4 gridspec (heatmaps + scatter) figure.

- **"⊞ Parameter Sweep Heatmap…" button in `LaplacePostFitRefinementDialog`** (`ade_dls/gui/dialogs/laplace_postfit_dialog.py`): placed next to the existing "↺ Refresh Clustering Preview" button in a shared `QHBoxLayout`. Triggers `_show_parameter_sweep_heatmap()`, which derives gamma columns from the current data, honours the active clustering strategy/method settings, runs `clustering_parameter_sweep()`, and opens the heatmap figure in a resizable `QDialog` (1 100 × 700 px) with a `NavigationToolbar2QT`.

- **"⊞ Parameter Sweep Heatmap…" button in `MethodDPostFitDialog`** (`ade_dls/gui/dialogs/method_d_postfit_dialog.py`): identical behaviour for the Method D post-fit clustering tab. Gamma columns are derived from `analyzer.method_d_fit`; the sweep uses `method_d_data` and the active silhouette/method settings.

- **"⊞ Clustering Heatmap…" button in `NNLSDialog`** (`ade_dls/gui/dialogs/nnls_dialog.py`): added to the NNLS preview controls panel (initially disabled, enabled after a successful "Generate Preview" run). `_show_clustering_heatmap_from_preview()` reconstructs a gamma DataFrame from the last preview results, attaches q² values from `df_basedata`, computes decay rates via `calculate_decay_rates()`, then runs `clustering_parameter_sweep()` and shows the heatmap dialog.

- **"Min. population abundance" spinbox in `NNLSDialog`** (`ade_dls/gui/dialogs/nnls_dialog.py`): new `QDoubleSpinBox` (range 0.0–1.0, step 0.05, default 0.3) added to the NNLS clustering settings group. Live-updates `params['min_abundance']` via `valueChanged`; tooltip explains the threshold semantics.

- **`clustering-help.txt`**: new reference document (272 lines) describing the clustering algorithm, parameter meanings, and recommended sweep workflows.

---

## [3.2.2] - 2026-06-25

### Added

- **`fit_beta` parameter in regularized NNLS** (`ade_dls/analysis/regularized.py`): new optional parameter `fit_beta` (default `False`) in `nnls_reg()`, `nnls_reg_all()` and `nnls_reg_simple()`. When enabled, the instrument coherence factor β is optimized jointly with the decay-time distribution instead of being fixed at 1.0. The initial β estimate is derived from the first few data points (g²(0)−1 ≈ β). Both the normalized (`SLSQP`) and non-normalized (`least_squares`/`trf`) optimization branches support β; bounds are [0, 2]. The fitted β value is reported in the results dict (`'beta'` key), appears in exported CSVs, and is shown in the diagnostic plot title. Without `fit_beta: True` the behaviour is fully backward-compatible.
- **`fit_beta` in generated pipeline script** (`ade_dls/gui/main_window.py`): `_add_regularized_step_to_pipeline()` now includes `fit_beta` in the `reg_params` dict of the auto-generated reproducibility script so provenance records are fully reproducible.

---

## [3.2.1] - 2026-06-16

### Added

- **W3C PROV-JSON serialisation** (`ade_dls/gui/core/provenance.py`): new method `to_prov_json()` on `ProvenanceRecord` produces a second, standards-compliant serialisation alongside the existing `to_json()` / `to_dict()` (which remain unchanged and continue to drive the GUI panel). The PROV-JSON output follows the [W3C PROV-JSON specification](https://www.w3.org/TR/prov-json/) and is compatible with the `prov` Python library and W3C PROV-Toolbox. Top-level keys: `prefix`, `entity`, `activity`, `agent`, `wasGeneratedBy`, `wasDerivedFrom`, `wasInformedBy`, `wasAssociatedWith`, plus a `jade:sessionMetadata` extension block. Companion method `export_prov_json_to_file()` writes the file and registers it in the output catalog.
- **"Export PROV-JSON" button** (`ade_dls/gui/widgets/provenance_panel.py`): second export button in the provenance panel toolbar opens a save-file dialog and calls `export_prov_json_to_file()`.

### Fixed

- **PROV-DM relation correction in `add_output`** (`ade_dls/gui/core/provenance.py`): the output catalog entry previously stored `wasDerivedFrom` pointing at activity IDs, which violates the W3C PROV-DM specification (`wasDerivedFrom` is an Entity→Entity relation). The entry now correctly carries:
  - `wasGeneratedBy`: ID of the last activity (the one that triggered the export) — Entity←Activity.
  - `wasDerivedFrom`: IDs of the SHA-256 input entities — Entity→Entity.

---

## [3.2.0] - 2026-06-12

### Added

- **CSV plot-data export** (`ade_dls/gui/export/csv_export.py`): export the raw data behind plots as CSV tables so users can assemble their own comparison tables across measurement series. International CSV format (comma separator, dot decimal); each export writes one CSV per table into a user-chosen folder. Three plot types are supported:
  - **Diffusion coefficient (Γ vs q²)** — three tables: `diffusion_datapoints.csv` (q², Γ and D = Γ/q²·1e-15 per cumulant order), `diffusion_fitcurve.csv` (densely sampled linear-fit line per order) and `diffusion_fit_parameters.csv` (slope, intercept, D, standard errors, R² and fitted q² range — enough to re-plot the line via a function tool). Exposed via the "📤 Diffusionsdaten als CSV" button in the analysis view (Cumulants Method A); reuses `cumulant_analyzer.method_a_data` / `method_a_regression_stats`.
  - **Clustering (multimodal Method D)** — `clustering_populations.csv` (populations as columns, statistic fields as rows) and `clustering_points.csv` (per-point raw clustering data). Exposed via the "📤 Clusterdaten als CSV" button; reuses `method_d_clustered_df` / `method_d_cluster_info`.
  - **Distributions (multimodal NNLS / Regularized)** — one intensity column per distribution, with presets chosen via a new `DistributionExportDialog`: *all*, *random X* (reproducible via seed) or *one averaged distribution per scattering angle* (mirrors `plot_distributions` average mode). Exposed via the "📤 Verteilungen als CSV" button in the NNLS and Regularized result dialogs.
- **Provenance for exported CSVs**: every written CSV is registered in the FAIR provenance record (`output_type="plot_data_csv"`) so its SHA-256 hash is captured, via `register_outputs_in_provenance()` → `ProvenanceRecord.add_output()`.

### Changed

- The ScatterForge export buttons in the analysis view (Γ/D → ScatterForge) are replaced by the CSV plot-data export buttons; the ScatterForge integration is deferred in favour of the CSV intermediate step. The `scatterforge_bridge` module remains in the tree but is no longer wired into the GUI.

---

## [3.1.1] - 2026-06-12

### Added

- **Centralised cumulant store in the data loader** (`ade_dls/gui/core/data_loader.py`): the loader now extracts instrument-software cumulant fit results alongside basedata/correlations/countrates and exposes them as `all_data['cumulants']` (dict keyed by measurement label) plus `all_data['cumulant_mode']`. Cumulant Method A is no longer the only consumer that re-reads raw files — the parser is now the single source of truth for the cumulant format. A new `load_cumulants` flag (default `True`) mirrors the existing `load_correlations`/`load_countrates` switches; extraction is non-critical (failures are logged, not fatal).
- **Parser cumulant contract** (`ade_dls/core/parsers/base_parser.py`): new `extract_cumulants()` method (default `None`) and `CUMULANT_MODE` class attribute (`"frequency"` or `"radius"`) declaring the column schema each instrument returns.
  - `ALVParser` (`CUMULANT_MODE = "frequency"`): wraps the existing `ade_dls.analysis.cumulants.extract_cumulants` (Γ orders 1–3 + µ₂/µ₃), unchanged numerics.
  - `LSInstrumentsParser` (`CUMULANT_MODE = "radius"`): reads the per-order hydrodynamic radius (value, CoV, std) from `Cumulants Results.csv`.
- **Cumulant schema test** (`tests/test_parser_contract.py`): the parser contract test now verifies `extract_cumulants()` returns the columns declared by `CUMULANT_MODE` for both ALV and LS Instruments.

### Fixed

- **Cumulant Method A for LS Instruments** (`ade_dls/gui/analysis/cumulant_analyzer.py`): `run_method_a()` was hard-wired to ALV — it globbed `*.asc`/`*.ASC` in the data folder and called the ALV text parser, so LS Instruments folders (no `.asc` files) raised `ValueError: No cumulant data could be extracted from files!`. Method A now reads from the loader's `cumulants` store and branches on `cumulant_mode`:
  - `"frequency"` (ALV): unchanged Γ-vs-q² OLS regression path.
  - `"radius"` (LS Instruments): new `_method_a_radius()` aggregates the software-reported Rh per cumulant order across all measurements (mean ± standard error of the mean), back-calculates D via Stokes–Einstein, and uses the LS coefficient of variation (CoV²) as the polydispersity proxy. Produces the same results schema as the ALV path, so the GUI, report and export layers stay instrument-agnostic.
- **Method A plot handling** (`ade_dls/gui/main_window.py`): the "Method A Summary" plot is now skipped when `None`, so the radius path (which produces no Γ-vs-q² plots) no longer pushes an empty figure into the results view.

---

## [3.1.0] - 2026-06-11

### Fixed

- **Single-channel correlation support** (`ade_dls/core/preprocessing.py`): `process_correlation_data()` now handles single-channel correlation DataFrames (e.g. from LS Instruments hardware, which exports one correlation column). Previously the function unconditionally averaged `correlation 1` and `correlation 2`, raising `KeyError: 'correlation 2'` for single-channel data and silently dropping all datasets (0 fits). Multi-channel behaviour (ALV: mean of channels 1 and 2) is unchanged.
- **Filter/noise preview for single-channel data** (`ade_dls/gui/dialogs/filtering_dialogs.py`): `_get_raw_g2()` no longer returns `None, None` when `correlation 2` is absent. It now falls back to `correlation 1` directly, so the preview curve is displayed correctly for LS Instruments datasets.

---

## [3.0.0] - 2026-06-09

### Added

#### FAIR Provenance Tracking – new module

- New module `ade_dls/gui/core/provenance.py` — PROV-inspired JSON provenance record:
  - `ProvenanceRecord` class with UUID4 record ID, schema URI, software/platform metadata
  - SHA-256 hashing of all input files at load time for unambiguous data traceability
  - Activity DAG: every analysis step (load, filter, cumulant fit, regularized NNLS, refinement, …) is appended as a timestamped activity node with inferred semantic type (`data_loading` / `filter` / `analysis` / `refinement`)
  - JSON export embeds the record ID in every exported artifact so reports can be traced back to their provenance record
- New widget `ade_dls/gui/widgets/provenance_panel.py` — live JSON viewer:
  - JSON syntax highlighting (keys, strings, numbers, booleans) with light/dark-mode support
  - "Copy JSON" and "Export JSON…" toolbar buttons
  - Updates live as analysis steps are added
- **"Export Provenance JSON…"** menu action in the File menu (shortcut: `Ctrl+Shift+P`)
- `MainWindow` initialises the provenance record immediately after data load and emits `step_added` for the load step so it is captured as the first activity

#### Report – LaTeX/PDF export

- All `ReportBlock` sub-classes now implement `to_latex()` for direct LaTeX rendering:
  - `MetadataBlock.to_latex()` — two-column `tblr` key/value table
  - `PreprocessingBlock.to_latex()` — four-column `tblr` steps table (no., time, step, parameters)
  - `ResultSummaryBlock.to_latex()` — header + value row `tblr` with bold column names
  - `ResultDetailBlock.to_latex()` — parses HTML detail tables and re-renders them as `tblr`
- `_tex_escape()` — comprehensive LaTeX escaping: the 10 LaTeX special characters plus a Unicode→LaTeX mapping table covering:
  - Super/subscript digits (`²` → `$^{2}$`, `₂` → `$_{2}$`, …)
  - Math and unit symbols (`±`, `µ`, `°`, `×`, `≈`, `≤`, `≥`, `≠`, `∞`)
  - DLS-relevant Greek letters (`α β γ δ ε η κ λ ν π σ τ φ ω Γ Λ Σ`)
  - Punctuation/dashes (`…`, `–`, `—`, `→`, `←`)
- `_table_to_latex()` — renders parsed HTML tables as full-width `tblr` environments with bold header row and `\hline`

#### Method B – post-fit refinement

- `CumulantAnalyzer.refine_method_b()` — re-runs only the Γ vs q² OLS regression on the stored `method_b_data` snapshot, applying optional file exclusion and q² range filter; returns a results DataFrame with the same structure as `run_method_b()`
- `PostfitRefinementDialog` extended for Method B:
  - **q² range** selection with interactive plot and drag-to-select (mirrors Method C)
  - **"Inspect & Exclude Fits…"** button — opens a per-file fit inspector to remove individual correlation-function fits from the regression
  - **Γ vs q² preview plot** in the dialog (NaN-safe, sorted, with linear fit overlay)

#### NNLS / Regularized post-fit refinement – per-population improvements

- `LaplacePostFitDialog` population tabs now include:
  - **Outlier threshold (k×σ)** spinbox — points with `|residual| > k·σ` are excluded (0 = disabled, matches Method D behaviour)
  - **Minimum data points for regression** spinbox

### Changed

- **BREAKING – Method B column name**: `R-squared` renamed to `R_squared` in all DataFrames and plots; code that reads the old column name must be updated
- **BREAKING – Method B fit engine**: `curve_fit` (non-linear least squares) replaced by `scipy.stats.linregress` (linear OLS) — fit parameters `popt`/`perr` no longer exist; use `Gamma`, `Gamma_error`, `R_squared` instead
- **Inspector panel**: Python Code tab removed and replaced by the Provenance tab; the syntax highlighter is now JSON-focused
- **View menu**: "Show Code" renamed to "Show Provenance"
- **Method B plots**: default x-axis limits now match the fit window (linear region visible by default); annotation box shows R² and Γ directly in the Data & Fit panel; plot style updated (`alpha=0.6`, `markersize=4`, `grid alpha=0.3`)
- **`LaplacePostFitDialog` clustering/population tabs**: per-population outlier threshold and minimum-points fields added (consistent with `MethodDPostFitDialog`)
- **Jupyter notebook removed**: `JADE-DLS_vers1-0.ipynb` (v1.0 legacy notebook) deleted from the repository

### Fixed

- **`laplace_postfit_dialog.py` pyplot conflict**: Matplotlib figure construction switched from `plt.figure()` to `matplotlib.figure.Figure()` for both the q² range selector and the clustering preview — eliminates Qt event-loop interference
- **Method B `None`-data guard**: `plot_processed_correlations` and `plot_processed_correlations_no_show` now skip `None` DataFrames with a console message instead of raising `AttributeError`
- **Method B residuals plot**: residuals are now plotted by sample index (sequential) rather than by log-scale lag time, making normality visually assessable

---

## [2.1.1dev] - 2026-03-18

### Added

#### Static Light Scattering (SLS) – new analysis module

- New module `ade_dls/analysis/sls.py` with population-resolved SLS analysis:
  - `compute_sls_data()` — intensity decomposition per population using area-fraction weights from regularized NNLS (`I_pop = I_total × normalized_area_percent / 100`)
  - `compute_sls_data_number_weighted()` — number-weighting correction (`I_pop / Rh^exponent`) to remove the intensity-vs-size bias; exponent configurable (6 = Rayleigh, 5 = Daoud-Cotton)
  - `compute_guinier_total()` — Guinier fit on total intensity (reference baseline)
  - `compute_guinier_extrapolation()` — per-population Guinier analysis (`ln(I_pop)` vs q²), yields I₀ and Rg per population
  - `plot_sls_intensity()` / `plot_guinier()` — embeddable matplotlib plots (pass `ax=` for GUI, `ax=None` for standalone)
  - `summarize_sls()` / `summarize_sls_combined()` — summary DataFrames with I₀, Rg, qRg_max, R² per population
- New utility `ade_dls/utils/intensity.py`:
  - `build_intensity_dataframe()` — reads ALV .ASC files and computes monitor-corrected, geometry-corrected mean count rates: `MeanCR_corr = (CR0 + CR1) / 2 / (monitordiode × 1e−3) × sin(θ)`; falls back to plain average × sin(θ) when monitor diode is unavailable
- `LaplaceAnalyzer.load_intensity_data()` — loads and stores intensity data from ALV .ASC file list
- `LaplaceAnalyzer.run_sls_analysis()` — orchestrates full SLS pipeline; stores `sls_data`, `guinier_results`, `guinier_total`, `sls_summary` on the analyzer instance

#### SLS Analysis tab in results dialogs

- **`RegularizedResultsDialog`** (Regularized NNLS): new **"📡 SLS Analysis"** tab (4th tab):
  - Configuration: number of populations (auto-detected), q² Guinier fit range, Rh exponent, number-weighting toggle
  - "Load Intensity Data" button opens folder browser for ALV .ASC files
  - "▶ Run SLS Analysis" triggers `LaplaceAnalyzer.run_sls_analysis()`; Guinier plot and summary table rendered inline
- **`LaplacePostFitRefinementDialog`** (Regularized only): equivalent SLS Analysis tab added

#### Method D – parallel fitting

- `_fit_single_method_d()` — new module-level function (required by joblib) that runs fitting, clustering, and moment calculation for a single file in isolation; returns `(name, result_dict, error_str)`
- Method D fitting loop in `analyze_method_d()` now supports `use_multiprocessing` parameter to switch between sequential and parallel (joblib) execution

#### Report panel integration

- **`InspectorPanel`**: Report tab (`ReportPanel`) promoted to **Tab 0** (first tab, most prominent)
- **Export Report button** (toolbar-style `QToolButton` with drop-down menu): TXT / Markdown / PDF (portrait) / PDF (landscape)
- **`AnalysisView`**: new `send_to_report` PyQt5 signal; **"📤 Send Plot to Report"** button in the Plots panel navigation bar; right-click context menu on Results table rows adds **"Send to Report (Summary)"** and **"Send to Report (Details)"** actions
- **`MainWindow`**: connects `analysis_view.send_to_report` to `inspector_panel.report_panel.add_block_from_payload()` and automatically switches to the Report tab on receipt
- "Compare Results" menu action replaced by an informational dialog pointing users to the Report tab workflow

#### Count-rate inspection – FFT panel

- Filtering dialog count-rate preview expanded to **2-panel layout** (figure 10 × 9 in):
  - Top panel: count rate vs time (one line per detector slot)
  - Bottom panel: frequency spectrum (`semilogy` FFT magnitude vs Hz, DC component excluded)
- Applies to both `FilteringDialog` and `MethodDPostFitDialog` raw-data viewers

#### Pipeline – noise correction tracking

- `AnalysisStep.make_filter_step()` now accepts an optional `noise_params` argument; active noise-reduction settings (`baseline_correction`, `baseline_pct`, `intercept_correction`, `intercept_pct`) are included in the pipeline step parameters when at least one correction is enabled — enables accurate code export

### Fixed

- **FutureWarning (pandas dtype)**: `outlier_pop{N}` columns in `cluster_all_gammas()` were initialized with `np.nan` (float64), then assigned `True`/`False` (bool). Columns are now initialized as `pd.array([pd.NA] * n, dtype='boolean')` — full three-state logic (NA / False / True) retained without dtype mismatch.
- **ValueError in NNLS clustering re-run**: Auto-detection of tau columns (`col.startswith('tau_')`) included population-indexed columns (`tau_pop1`, `tau_pop2`, …) produced by a prior clustering run. On re-run this led to `original_col = 'gamma_pop1'` and `int('pop1')` → `ValueError`. Detection now explicitly excludes `tau_pop*` columns.
- **UserWarning from `tight_layout()`**: `LaplacePostFitDialog._show_plot()` called `tight_layout()` unconditionally; fails with a UserWarning when many subplots with decorated text annotations are present. Wrapped in `try/except` (best-effort layout).
- **Postfit refinement dialog – pyplot event loop conflict**: Figure construction in `PostfitRefinementDialog` switched from `plt.figure()` to `matplotlib.figure.Figure()` to avoid interference with the Qt event loop.

### Changed

- **Comparison tab removed** from `AnalysisView`: the dedicated comparison tab is replaced by the Report tab workflow (right-click → "Send to Report")
- **`MethodDPostFitDialog` clustering calls**: `cluster_all_gammas()` is now called with `interactive=False` to prevent blocking `input()` prompts inside the GUI

---

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

[3.0.0]: https://github.com/traianuschem/JADE-DLS/compare/v2.1.1dev...v3.0.0
[2.1.1dev]: https://github.com/traianuschem/JADE-DLS/compare/v2.1.0dev...v2.1.1dev
[2.1dev]: https://github.com/traianuschem/JADE-DLS/compare/v2.0.3dev...v2.1dev
[2.0.3dev]: https://github.com/traianuschem/JADE-DLS/compare/v2.0.2dev...v2.0.3dev
[2.1.0]: https://github.com/traianuschem/JADE-DLS/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/traianuschem/JADE-DLS/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/traianuschem/JADE-DLS/releases/tag/v1.0.0
