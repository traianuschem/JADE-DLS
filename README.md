# ADE-DLS: Angular Dependent Evaluator for Dynamic Light Scattering

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-3.2.1-blue.svg)](CHANGELOG.md)

ADE-DLS is a Python package for analyzing Dynamic Light Scattering (DLS) — and optionally Static Light Scattering (SLS) — data from multi-angle instruments. It determines hydrodynamic radii, diffusion coefficients, and particle size distributions via a full-featured PyQt5 GUI or a Python API.

## Features

### Analysis Methods

- **Cumulant Analysis** (4 methods)
  - Method A: Instrument-software cumulant fits (reads Γ / Rh directly from ALV and LS Instruments files)
  - Method B: Linear fit of ln[g₁(τ)] vs. τ
  - Method C: Iterative non-linear least squares (2nd / 3rd / 4th order)
  - Method D: Multi-exponential decomposition with cross-angle population clustering

- **Inverse Laplace Transform**
  - NNLS (Non-Negative Least Squares) with peak detection
  - Regularized NNLS (Tikhonov-Phillips) with L-curve and GCV α selection
  - Optional constraints: normalization, sparsity, unimodality
  - Ward hierarchical clustering of diffusion coefficients across angles

- **Static Light Scattering (SLS)**
  - Population-resolved intensity decomposition from Regularized NNLS results
  - Monitor-corrected, geometry-corrected count rates
  - Guinier analysis per population (I₀, Rg, qRg_max, R²) and for total intensity
  - Number-weighting correction (configurable Rh exponent)

- **Data Processing**
  - Multi-instrument file loading (ALV .ASC, LS Instruments directory format)
  - Interactive count-rate and correlation filtering dialogs
  - Baseline and intercept noise correction
  - Diffusion coefficient calculation (D = Γ / q²) and hydrodynamic radius via Stokes-Einstein

### GUI Application

- **Interactive PyQt5 interface** with three-panel layout (Workflow · Analysis View · Inspector)
- **FAIR provenance tracking** — full W3C PROV-compliant session record; export as JSON or PROV-JSON
- **Transparent analysis pipeline** — auto-generates reproducible Jupyter notebooks and Python scripts
- **Comprehensive visualization** — per-dataset diagnostic plots (correlation fit, residuals, Q-Q, τ distribution); D vs. q² scatter; log₁₀(D) histogram
- **Post-fit refinement dialogs** — adjust fit ranges, regularization, clustering parameters without reloading
- **Report panel** — compose custom reports from result blocks and plots; export as TXT, Markdown, or PDF
- **Export** — CSV (plot data), Excel, Jupyter notebooks, Python scripts, PROV-JSON

## Installation

### From Source

```bash
git clone https://github.com/traianuschem/JADE-DLS.git
cd JADE-DLS
pip install -e .[gui]
```

For the full installation including Excel/CSV export:

```bash
pip install -e .[all]
```

See [docs/installation.md](docs/installation.md) for platform-specific notes and troubleshooting.

## Quick Start

### GUI

```bash
ade-dls
```

Or:

```bash
python -m ade_dls.gui.main_window
```

See [docs/quickstart.md](docs/quickstart.md) for a step-by-step walkthrough.

### Python API

```python
from ade_dls.core import preprocessing
from ade_dls.analysis import cumulants

# Load .ASC files
data = preprocessing.load_and_preprocess_data(
    folder_path="path/to/data",
    file_extension=".ASC"
)

# Cumulant Method C
params = {
    'method': 'C',
    'wavelength': 632.8,      # nm
    'temperature': 298.15,    # K
    'viscosity': 0.89,        # cP
    'refractive_index': 1.33
}
results = cumulants.cumulant_method_c_all(data, params)
print(results)
```

## Supported Instruments

| Instrument | Format | Extension |
|------------|--------|-----------|
| ALV correlators (ALV-5000, ALV-7004, …) | ASCII header + data blocks | `.ASC` |
| LS Instruments (NanoLab, …) | Directory per measurement | directory |

See [docs/data_formats.md](docs/data_formats.md) for format details and instructions on adding a new instrument parser.

## Requirements

- Python 3.8+
- NumPy ≥ 1.22, SciPy ≥ 1.8, Pandas ≥ 1.5, Matplotlib ≥ 3.5
- scikit-learn ≥ 1.0, statsmodels ≥ 0.13, joblib ≥ 1.2

**For GUI:** PyQt5 ≥ 5.15

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](docs/installation.md) | Setup, dependencies, platform notes |
| [Quick Start](docs/quickstart.md) | First analysis in minutes |
| [User Guide](docs/user_guide.md) | Full GUI reference |
| [Analysis Methods](docs/analysis_methods.md) | Mathematical background of all methods |
| [Data Formats](docs/data_formats.md) | Supported instruments and parser extension |
| [Filtering Guide](docs/FILTERING_GUIDE.md) | Count-rate and correlation filtering |
| [Changelog](CHANGELOG.md) | Version history |
| [Contributing](CONTRIBUTING.md) | Development guidelines |

## Citation

If you use ADE-DLS in your research, please cite:

```bibtex
@software{ade_dls,
  title  = {ADE-DLS: Angular Dependent Evaluator for Dynamic Light Scattering},
  author = {Richard Neubert and Vincent Schildknecht},
  year   = {2026},
  url    = {https://github.com/traianuschem/JADE-DLS}
}
```

## License

GNU General Public License v3.0 or later — see [LICENSE](LICENSE).

## Acknowledgments

Originally developed as JADE-DLS (Jupyter-based Angular Dependent Evaluator for DLS); evolved into a full-featured PyQt5 GUI application with a modular Python analysis library.
