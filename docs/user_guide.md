# User Guide

## Overview

ADE-DLS provides a PyQt5 GUI for analyzing Dynamic Light Scattering (DLS) data from multi-angle ALV correlators. The interface is organized into three panels and follows a linear workflow: load → preprocess → analyze → export.

---

## The Three-Panel Layout

### Left: Workflow Panel

Contains categorized buttons for each processing step. Steps are ordered top to bottom. Click a step to select it, then click **Run** to execute. Steps that require a prerequisite (e.g., analysis before export) are disabled until that prerequisite is met.

Categories:
- **Load** — data loading and instrument file selection
- **Preprocess** — filtering dialogs for count rates and correlation functions
- **Analysis** — all fitting methods (Cumulant B/C/D, NNLS, Regularized)
- **Export** — CSV, Excel, Jupyter, Python script

### Center: Analysis View

A `QTabWidget` with four tabs:

| Tab | Content |
|-----|---------|
| **Data Overview** | Table of loaded files: path, angle, count rate, intercept, measurement duration |
| **Plots** | Per-dataset diagnostic plots (correlation function, fit, residuals, Q-Q, τ distribution) |
| **Results** | Results table; click a row to expand the full fit report |
| **Comparison** | Side-by-side comparison of all methods for each dataset |

### Right: Inspector Panel

A `QTabWidget` with four tabs:

| Tab | Content |
|-----|---------|
| **Report** | Modular report builder — add summary blocks, detail blocks, and plots from any result; export as TXT, Markdown, or PDF |
| **Code** | Auto-generated Python code for the current pipeline state; copy or export |
| **Parameters** | All active parameters (wavelength, temperature, viscosity, refractive index, fit ranges, regularization, clustering settings) |
| **Provenance** | W3C PROV-compliant provenance record for the current session |

---

## Loading Data

**File > Load Data** opens the load dialog. Select the folder with `.ASC` files. The dialog detects instrument format (ALV or LS Instruments) automatically.

Required physical parameters:
- **Wavelength** (nm) — laser wavelength, e.g. 632.8 nm for He-Ne
- **Temperature** (K) — sample temperature, e.g. 298.15 K
- **Viscosity** (cP) — solvent dynamic viscosity
- **Refractive index** — solvent refractive index (used to compute q²)

After loading, two sequential filtering dialogs open:

1. **Count Rate Filtering** — inspect detector count rates over time; exclude files with instabilities
2. **Correlation Filtering** — inspect g₂(τ) curves; exclude curves with artifacts

Both dialogs offer checkboxes per file and a preview plot. Excluded files are marked but kept in memory; they can be re-included by re-running the filtering step.

---

## Preprocessing

After filtering, **Preprocess** extracts g₁(τ) from the raw g₂(τ) via the Siegert relation:

```
g₁(τ) = sqrt((g₂(τ) − 1) / β)
```

where β is the intercept (coherence factor). Baseline correction and noise floor subtraction are applied if enabled in Parameters.

---

## Analysis Methods

See [analysis_methods.md](analysis_methods.md) for mathematical details.

### Cumulant B

Linear fit of ln[g₁(τ)] vs. τ over a user-defined lag-time range. Returns Γ (decay rate), D, and Rh. Simple and fast, suitable as a sanity check.

### Cumulant C

Iterative non-linear least squares fit of g₁(τ) to a cumulant expansion (2nd, 3rd, or 4th order). Adaptive parameter bounds. Most accurate single-population method.

### Cumulant D

Multi-exponential fit decomposing g₁(τ) into two or more populations. Followed by Ward hierarchical clustering across angles to track populations consistently.

### NNLS

Non-negative least squares inverse Laplace transform. Produces a discrete relaxation time distribution τ→A(τ) on a logarithmic grid. No regularization; sparse solutions.

The NNLS dialog includes a **clustering settings** group with:
- **Distance threshold** — Ward linkage cut-off for grouping populations across angles.
- **Min. population abundance** (0–1, default 0.3) — minimum fraction of datasets a population must appear in to be considered reliable.

After running a preview, the **"⊞ Clustering Heatmap…"** button becomes available and shows the parameter sweep heatmap (see [Clustering Parameter Sweep Heatmap](#clustering-parameter-sweep-heatmap) below).

### Regularized NNLS

Tikhonov-Phillips regularization applied to the NNLS problem. The regularization parameter α controls smoothness vs. data fidelity. Use the **alpha analysis dialog** (accessible via the post-fit refinement button) to select α from an L-curve or GCV criterion.

### Static Light Scattering (SLS)

Available after running Regularized NNLS. Decomposes the total static intensity into per-population contributions using monitor-corrected count rates. Performs Guinier analysis (I₀, Rg, qRg_max, R²) per population and for the total.

---

## Post-Fit Refinement

All methods expose a **post-fit refinement dialog** (button next to the result row, or via right-click). Refinement lets you:
- Adjust the fit range (τ_min, τ_max)
- Change the cumulant order (Method C)
- Re-run clustering with different distance thresholds and minimum population abundance (Methods D, NNLS, Regularized)
- Change α and re-run regularization
- Exclude individual angles from the result

Refinement re-runs only the selected method on the stored preprocessed data — no reload required.

### Clustering Parameter Sweep Heatmap

The **Clustering** tab in the post-fit refinement dialogs (Methods D, NNLS, Regularized) includes a **"⊞ Parameter Sweep Heatmap…"** button. It runs `cluster_all_gammas()` over a 5 × 5 grid of `distance_threshold × min_abundance` values (25 combinations) and displays two heatmaps:

- **(a) Number of populations** — how many reliable populations are found at each parameter combination.
- **(b) Silhouette score** — clustering quality (0–1; higher is better).

Up to four representative scatter sub-panels (D_t vs q²) are shown below the heatmaps to visualise how the population assignment changes across the parameter grid.

Use the heatmap to find a stable region where both the population count and silhouette score are consistent, then set `distance_threshold` and `min_abundance` accordingly before applying refinement.

---

## Export

### CSV

**File > Export CSV** writes one `.csv` per method with all results. A separate clustering file is written for Methods D and NNLS.

### Excel

**File > Export Excel** writes an `.xlsx` with one sheet per method, plus a metadata sheet.

### Jupyter Notebook

**File > Export Jupyter Notebook** (`Ctrl+J`) generates a self-contained `.ipynb` that reproduces the entire analysis from raw data. All parameters are embedded as code cells.

### Python Script

**File > Export Python Script** (`Ctrl+P`) generates a standalone `.py` equivalent of the notebook.

### Report Panel

The **Report** tab (right panel) lets you compose a custom document:
1. Click **Add Summary** or **Add Detail** on any result row to insert a block
2. Drag blocks to reorder
3. Click **Export** to save as TXT, Markdown, or PDF (portrait or landscape)

---

## Provenance

ADE-DLS records a W3C PROV-compliant provenance graph for the current session. Each loading, filtering, preprocessing, and analysis step is recorded as an `Activity` with input `Entity` (raw data) and output `Entity` (result), linked by `wasDerivedFrom` and `wasGeneratedBy` relations.

The **Provenance** tab (right panel) shows the current graph. It can be exported as JSON-LD.

---

## Transparency Features

- **Code tab**: Every action in the GUI corresponds to a Python code snippet shown in real time. The code is self-contained and can be pasted into a script.
- **Parameters tab**: All active parameter values are listed with their current settings and any per-step overrides.
- **Jupyter export**: The exported notebook is executable without the GUI.

---

## Tips

- **Multi-angle data**: ADE-DLS treats each `.ASC` file as one angle. For a series at different angles, load the entire folder — the software groups files by sample automatically.
- **R² check**: Cumulant fits with R² < 0.99 usually indicate a poor fit range or a contaminated sample. Narrow the fit range in the post-fit dialog.
- **PDI > 0.2**: High PDI with Cumulant C suggests polydispersity. Switch to Regularized NNLS for a size distribution.
- **Regularization α**: Start with the GCV suggestion, then check the L-curve. Over-regularized solutions (too large α) are too smooth; under-regularized solutions are noisy and unstable.
