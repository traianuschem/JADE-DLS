# Quick Start

## Launch the GUI

```bash
ade-dls
```

Or alternatively:

```bash
python -m ade_dls.gui.main_window
```

## Step-by-Step: First Analysis

### 1. Load Data

Go to **File > Load Data** (or `Ctrl+O`). Select the folder containing your ALV `.ASC` files.

A dialog lets you preview detected files and choose the laser wavelength, temperature, viscosity, and refractive index before loading. These physical parameters determine the hydrodynamic radius via the Stokes-Einstein equation.

The status bar shows how many files were loaded.

### 2. Inspect the Data Overview

The **Data Overview** tab (center panel) lists all loaded files with their scattering angles, count rates, and intercepts. Use the correlation and count-rate filtering dialogs that open automatically after loading to exclude bad measurements.

### 3. Run an Analysis Method

Select a method from the **Workflow Panel** (left):

| Button | Method |
|--------|--------|
| Cumulant B | Linear cumulant fit |
| Cumulant C | Iterative non-linear least squares |
| Cumulant D | Multi-exponential decomposition |
| NNLS | Non-negative least squares inverse Laplace |
| Regularized | Tikhonov-Phillips regularized NNLS |

Click **Run** to execute. Results appear in the **Results** tab.

### 4. View Results

Switch to the **Results** tab (center panel). Click any row to expand the detailed fit report. Columns include Γ, D, Rh, PDI, and R².

### 5. Export

- **File > Export CSV** — structured CSV with all results
- **File > Export Excel** — `.xlsx` with separate sheets per method
- **File > Export Jupyter Notebook** — fully reproducible `.ipynb`
- **File > Export Python Script** — standalone `.py` file
- Use the **Report Panel** (right panel) to build a custom PDF report

## Layout at a Glance

```
┌────────────────┬──────────────────────────────┬─────────────────────┐
│  WORKFLOW      │  ANALYSIS VIEW               │  INSPECTOR          │
│                │                              │                     │
│  Load Data     │  Tabs:                       │  Tabs:              │
│  Preprocess    │  • Data Overview             │  • Report           │
│  Cumulant B    │  • Plots                     │  • Code             │
│  Cumulant C    │  • Results                   │  • Parameters       │
│  Cumulant D    │                              │  • Provenance       │
│  NNLS          │                              │                     │
│  Regularized   │                              │                     │
└────────────────┴──────────────────────────────┴─────────────────────┘
```

## Common Workflows

### Monodisperse sample (quick)

1. Load data → accept defaults
2. Run **Cumulant C**
3. Check R² > 0.99 and PDI < 0.1
4. Export CSV

### Polydisperse sample

1. Load data
2. Run **Regularized** NNLS
3. Open the post-fit dialog to tune the regularization parameter α
4. Run **Cumulant D** for multi-exponential decomposition
5. Compare methods via the **Report** tab (right panel)

### Multi-angle measurement

All methods compute D and Rh per scattering angle. After running Cumulant C or NNLS, the Results tab shows one row per angle.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Load data |
| `Ctrl+J` | Export Jupyter Notebook |
| `Ctrl+P` | Export Python Script |
| `Ctrl+Q` | Quit |
