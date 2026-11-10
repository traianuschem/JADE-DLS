# ADE-DLS: Angular Dependent Evaluator for Dynamic Light Scattering

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-4.0.0-blue.svg)](CHANGELOG.md)

ADE-DLS analyses Dynamic Light Scattering (DLS) data — and, optionally, Static Light Scattering (SLS) data — from multi-angle instruments. From a series of measurements at different scattering angles it determines diffusion coefficients, hydrodynamic radii and particle size distributions. It is used through a PyQt5 GUI or a Qt-free Python API; both run the same computation code.

ADE-DLS is the successor of the Jupyter-based JADE-DLS. Defaults and formulas follow the JADE-DLS reference implementation, and results are checked against it.

## Features

### Analysis methods

- **Cumulant analysis** (four methods)
  - **Method A** — cumulant fits of the instrument software (Γ read from the ALV / LS Instruments files)
  - **Method B** — linear fit of ln[g₁(τ)] vs. τ
  - **Method C** — iterative non-linear least squares (1st to 4th order)
  - **Method D** — multi-exponential decomposition with cross-angle population clustering
- **Inverse Laplace transform**
  - **NNLS** with peak detection
  - **Regularized NNLS** (Tikhonov–Phillips, fixed regularization parameter α) with optional normalization, sparsity and unimodality constraints
  - Ward hierarchical clustering of the diffusion coefficients across angles to identify populations
- **Static Light Scattering**
  - Population-resolved intensity decomposition from Regularized NNLS results
  - Guinier analysis per population and for the total intensity
  - Optional number-weighting correction of the extrapolated I₀ (configurable Rh exponent)

### Data processing

- Multi-instrument loading (ALV `.ASC`, LS Instruments directories)
- Count-rate and correlation filtering
- Baseline and intercept noise correction
- Optional heteroscedastic noise weighting (Biganzoli & Ferri, *Opt. Express* **26**, 29375, 2018)
- D = Γ / q² and hydrodynamic radius via the Stokes–Einstein relation

### GUI

- Three-panel layout: Workflow · Analysis View · Inspector
- Post-fit refinement dialogs (fit ranges, regularization, clustering parameters) without reloading the data, including a clustering-parameter sweep heatmap
- Report panel: compose reports from result blocks and plots, export as TXT, Markdown or PDF
- Provenance tracking (W3C PROV, exportable as JSON / PROV-JSON) and automatic generation of reproducible Jupyter notebooks and Python scripts
- Export of results and plot data as CSV and Excel

## Installation

```bash
git clone https://github.com/traianuschem/JADE-DLS.git
cd JADE-DLS
pip install -e .[gui]
```

`pip install -e .[all]` additionally installs the Excel export dependencies. The Python API alone needs only the base installation (`pip install -e .`), without PyQt5.

See [docs/installation.md](docs/installation.md) for platform notes and troubleshooting.

## Quick start

### GUI

```bash
ade-dls
```

A step-by-step walkthrough is in [docs/quickstart.md](docs/quickstart.md).

### Python API (no GUI required)

```python
from ade_dls import headless

# Load a measurement series; the instrument format is detected automatically
data = headless.load_series("path/to/data")

# Cumulant methods return (analyzer, results DataFrame)
analyzer, result_a = headless.run_cumulant_a(data)
analyzer, result_d = headless.run_cumulant_d(data, {"gap_threshold": 2.0})

# Inverse Laplace methods: defaults can be overridden per call;
# results are stored on the analyzer
laplace, params = headless.run_nnls(data)
print(laplace.nnls_final_results)

laplace, params = headless.run_regularized(data, {"alpha": 0.5})
print(laplace.regularized_final_results)
```

`headless.nnls_defaults()`, `headless.regularized_defaults()` and `headless.method_d_defaults()` return the default parameters. Post-fit refinement (`refine_*`) and the SLS analysis (`run_sls`) are available as well. See the docstrings in [`ade_dls/headless.py`](ade_dls/headless.py).

## Package layout

| Package | Content | Qt required |
|---------|---------|:-----------:|
| `ade_dls.analysis`, `ade_dls.core`, `ade_dls.utils` | Numerical kernels, parsers, preprocessing | no |
| `ade_dls.engine` | Analysis workflows shared by GUI and API | no |
| `ade_dls.headless` | Public Python API | no |
| `ade_dls.benchmark` | Simulator for synthetic multi-angle DLS data with known ground truth, evaluation rule, runner | no |
| `ade_dls.gui` | PyQt5 application (presentation and orchestration) | yes |

The dependency direction is kernels → engine → GUI. It is enforced by `tests/test_architecture.py`.

## Supported instruments

| Instrument | Format | Extension |
|------------|--------|-----------|
| ALV correlators (ALV-5000, ALV-7004, …) | ASCII header + data blocks | `.ASC` |
| LS Instruments (NanoLab, …) | Directory per measurement | directory |

Format details and how to add a parser: [docs/data_formats.md](docs/data_formats.md).

## Requirements

- Python 3.10+
- NumPy ≥ 2.0, SciPy 1.15–1.17, pandas ≥ 2.2, Matplotlib ≥ 3.9
- scikit-learn ≥ 1.5, statsmodels ≥ 0.14.4, joblib ≥ 1.4
- GUI: PyQt5 ≥ 5.15

## Known issues

- **Newer SciPy versions (≥ 1.18) are not supported yet.** The per-file selection of the number of modes in Method D is numerically unstable; with SciPy 1.18.1 it changed for individual files of a bimodal test series, which shifted the Rh of the larger population by about 5 %. SciPy is therefore limited to 1.15–1.17 in the installation requirements. A robust mode selection is planned for the next release.
- Method D can also deviate between platforms with identical library versions: on one CI runner the Rh of a simulated monodisperse series differed by 1.4 % from the stored reference. The test suite records these cases as expected failures.
- Results of the numerically sensitive fits (Method C, NNLS, Regularized NNLS, Method D) can differ slightly between platforms and library versions; the test suite compares them with tolerances and documents known deviations as expected failures.

## Development

```bash
pip install -e .[dev]
python -m pytest -q                 # full test suite
python -m pytest -q -m "not slow"   # without the slow optimizer tests
```

The synthetic benchmark (`ade_dls.benchmark`, simulator under GPL-3.0-or-later; simulated datasets and the golden reference samples in `tests/data/synthetic_golden` under CC BY-SA 4.0) checks ADE-DLS against stored JADE-DLS results and against the simulated ground truth.

Reference datasets of the paper (ALV and LS Instruments DLS raw data of the gold-nanoparticle samples, SAXS data) are in `Test+ReferenceData/` under CC BY-SA 4.0, see `Test+ReferenceData/README.md` and `Test+ReferenceData/LICENSE`.

Contribution guidelines: [CONTRIBUTING.md](CONTRIBUTING.md).

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
