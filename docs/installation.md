# Installation

## Requirements

- Python 3.8 or later
- Core dependencies (installed automatically): NumPy ≥ 1.22, SciPy ≥ 1.8, Pandas ≥ 1.5, Matplotlib ≥ 3.5, scikit-learn ≥ 1.0, statsmodels ≥ 0.13, joblib ≥ 1.2
- GUI dependencies (optional): PyQt5 ≥ 5.15, nbformat ≥ 5.1, jupyter, ipywidgets

## Installing from Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/traianuschem/JADE-DLS.git
cd JADE-DLS
```

### Core library only (no GUI)

```bash
pip install -e .
```

### With GUI support

```bash
pip install -e .[gui]
```

### Full installation (GUI + Excel/CSV export)

```bash
pip install -e .[all]
```

### Development setup

```bash
pip install -e .[all,dev]
```

This additionally installs pytest, black, isort, flake8, mypy, and pre-commit hooks.

## Verifying the Installation

```bash
python -c "import ade_dls; print(ade_dls.__version__)"
```

Launch the GUI:

```bash
ade-dls
```

## Platform Notes

### Windows

The `ade-dls` command is available after installation via the Scripts folder in your Python environment. If the command is not found, ensure your Python Scripts directory is on `PATH`, or run:

```bash
python -m ade_dls.gui.main_window
```

### Linux / macOS

Same as above. If you use a virtual environment, activate it before running `ade-dls`.

## Troubleshooting

**`ModuleNotFoundError: No module named 'PyQt5'`**
Install the GUI extras: `pip install -e .[gui]`

**`ade-dls` command not found**
Run `python -m ade_dls.gui.main_window` as a fallback, or check that your Python Scripts/bin directory is on `PATH`.

**Matplotlib plots not rendering in the GUI**
Ensure PyQt5 is installed. Matplotlib selects the Qt5Agg backend automatically when PyQt5 is present.
