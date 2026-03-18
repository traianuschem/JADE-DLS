# ADE-DLS: Angular Dependent Evaluator for Dynamic Light Scattering

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-2.1.1dev-orange.svg)](CHANGELOG.md)

ADE-DLS is a comprehensive Python package for analyzing Dynamic Light Scattering (DLS) — and optionally Static Light Scattering (SLS) — data from multi-angle ALV instruments. It determines hydrodynamic radii, diffusion coefficients, and particle size distributions via a full-featured PyQt5 GUI or a Python API.

## Features

### Analysis Methods

- **Cumulant Analysis** (3 methods)
  - Method A: ALV software pre-calculated fits (reads Γ, PDI directly)
  - Method B: Linear fit of ln[g(τ)^0.5] vs τ
  - Method C: Iterative non-linear least squares (2nd / 3rd / 4th order)
  - Method D: Multi-exponential decomposition with cross-file population clustering

- **Inverse Laplace Transform Methods**
  - NNLS (Non-Negative Least Squares) with peak detection
  - Regularized NNLS (Tikhonov-Phillips regularization)
  - Advanced constraints: normalization, sparsity, unimodality
  - Ward hierarchical clustering of diffusion coefficients across angles

- **Static Light Scattering (SLS)**
  - Population-resolved intensity decomposition from regularized NNLS results
  - Monitor-corrected, geometry-corrected count rates from ALV .ASC files
  - Guinier analysis per population (I₀, Rg, qRg_max, R²) and for total intensity
  - Number-weighting correction (configurable Rh exponent)

- **Data Processing**
  - Automated file loading (ALV .ASC correlator files)
  - Preprocessing, filtering, baseline and intercept noise correction
  - Peak clustering for multi-modal distributions (Ward / gap-based)
  - Diffusion coefficient calculation (D = Γ / q²)
  - Hydrodynamic radius determination (Stokes-Einstein)

### GUI Application

- **Interactive PyQt5 interface** with splash screen
- **Transparent analysis pipeline** — generates reproducible Jupyter notebooks and Python scripts; pipeline steps record all parameters including active noise corrections
- **Comprehensive visualization** — per-dataset 3- or 4-panel diagnostic plots (correlation fit, residuals, Q-Q plot, τ distribution); clustering overview (D vs q² scatter + log₁₀(D) histogram)
- **Batch processing** — analyze multiple files across different scattering angles
- **Post-fit refinement dialogs** for all methods — re-run with adjusted q² ranges, outlier thresholds, and clustering parameters without re-running the full analysis
- **Report panel** — collect result summaries, detail blocks, and plots from any method; export as TXT, Markdown, or PDF (portrait / landscape)
- **Export capabilities** — Excel, CSV, Jupyter notebooks, Python scripts

## Installation

### From Source

```bash
git clone https://github.com/traianuschem/JADE-DLS.git
cd JADE-DLS
pip install -e .
```

### With GUI Support

```bash
pip install -e .[gui]
```

### Full Installation (all features)

```bash
pip install -e .[all]
```

## Quick Start

### GUI Application

```bash
ade-dls
```

Or with Python:

```python
python -m ade_dls.gui.main_window
```

### Python API

```python
from ade_dls.core import preprocessing
from ade_dls.analysis import cumulants

# Load .ASC files
dataframes = preprocessing.load_and_preprocess_data(
    folder_path="path/to/data",
    file_extension=".ASC"
)

# Cumulant Method C
params = {
    'method': 'C',
    'wavelength': 632.8,    # nm
    'temperature': 298.15,  # K
    'viscosity': 0.89,      # cP
    'refractive_index': 1.33
}
results = cumulants.cumulant_method_c_all(dataframes, params)
print(results)
```

## Supported Data Formats

- **ALV correlator** (.ASC files) — correlation functions, count rates, monitor diode

## Requirements

- Python 3.8+
- NumPy ≥ 1.22
- SciPy ≥ 1.8
- Pandas ≥ 1.5
- Matplotlib ≥ 3.5
- scikit-learn ≥ 1.0
- statsmodels ≥ 0.13

**For GUI:**
- PyQt5 ≥ 5.15

## Documentation

- [Changelog](CHANGELOG.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [GitHub Issues](https://github.com/traianuschem/JADE-DLS/issues)

## Citation

If you use ADE-DLS in your research, please cite:

```bibtex
@software{ade_dls,
  title  = {ADE-DLS: Angular Dependent Evaluator for Dynamic Light Scattering},
  author = {Vincent Schildknecht und Richard Neubert},
  year   = {2026},
  url    = {https://github.com/traianuschem/JADE-DLS}
}
```

## License

GNU General Public License v3.0 or later — see [LICENSE](LICENSE).

## Acknowledgments

Originally developed as JADE-DLS (Jupyter-based Angular Dependent Evaluator for DLS); evolved into a full-featured PyQt5 GUI application with a modular Python analysis library.
