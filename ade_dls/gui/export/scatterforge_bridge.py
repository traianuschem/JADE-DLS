"""
ScatterForge Bridge
===================
Transfers ADE-DLS analysis results to ScatterForge-Plot.

Since ADE-DLS uses PyQt5 and ScatterForge uses PySide6, they cannot share the
same Qt process.  The bridge therefore:

  1. Writes the data as tab-delimited .dat files to a temporary directory.
  2. Generates a small Python launcher script that imports ScatterForge,
     loads the data files into the tree model, and shows the window.
  3. Launches the script as an independent subprocess (separate Qt process).

The ScatterForge window runs completely independently of ADE-DLS and can
be used for styling and export while ADE-DLS continues to run normally.

Data format
-----------
Converter functions (``from_cumulant_method_a``, ``from_diffusion_vs_q2``)
return a list of *groups*::

    [
        ('Group name', [dataset_dict, dataset_dict, ...]),
        ('Group name', [...]),
        ...
    ]

Each dataset dict has keys: 'x', 'y', 'y_err' (may be None), 'label', 'style'.
One group becomes one DataGroup in ScatterForge's tree.
"""

import os
import re
import sys
import subprocess
import tempfile
import numpy as np

# Absolute path to the ScatterForge directory (co-located in the repository)
_SF_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ScatteringPlot')
)


def is_available() -> bool:
    """Return True if ScatterForge is present and importable."""
    if not os.path.isdir(_SF_DIR):
        return False
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'scatter_plot',
            os.path.join(_SF_DIR, 'scatter_plot.py')
        )
        return spec is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Data conversion helpers
# ---------------------------------------------------------------------------

def from_cumulant_method_a(analyzer) -> list:
    """
    Extract Γ vs q² datasets from CumulantAnalyzer (Method A).

    Returns a list of (group_name, datasets) tuples — one per cumulant order.
    Each group contains measurement points ('Messung') and a linear fit
    line ('Fit').

    Example return value::

        [
            ('Method A – 1st Order', [data_dict, fit_dict]),
            ('Method A – 2nd Order', [data_dict, fit_dict]),
            ('Method A – 3rd Order', [data_dict, fit_dict]),
        ]
    """
    if not hasattr(analyzer, 'method_a_data') or analyzer.method_a_data is None:
        raise ValueError("Cumulants Method A has not been run yet.")

    df = analyzer.method_a_data
    gamma_cols = [
        ('1st order frequency [1/ms]', '1st Order'),
        ('2nd order frequency [1/ms]', '2nd Order'),
        ('3rd order frequency [1/ms]', '3rd Order'),
    ]

    reg_results = []
    if hasattr(analyzer, 'method_a_regression_stats'):
        reg_results = analyzer.method_a_regression_stats.get('regression_results', [])

    x_q2 = df['q^2'].values
    x_fit = np.linspace(float(x_q2.min()), float(x_q2.max()), 100)

    groups = []
    for i, (col, order_label) in enumerate(gamma_cols):
        if col not in df.columns:
            continue

        y = df[col].values
        order_datasets = [
            {
                'x': x_q2,
                'y': y,
                'y_err': None,
                'label': f'Data points',
                'style': 'Messung',
            }
        ]

        # Linear fit line
        if i < len(reg_results):
            reg = reg_results[i]
            y_fit = reg['intercept'] + reg['q^2_coef'] * x_fit
            order_datasets.append({
                'x': x_fit,
                'y': y_fit,
                'y_err': None,
                'label': f'Linear fit',
                'style': 'Fit',
            })

        groups.append((f'Method A \u2013 {order_label}', order_datasets))

    return groups


def from_diffusion_vs_q2(analyzer) -> list:
    """
    Compute D = Γ / q² per measurement angle from Method A data.

    Unit conversion: Γ [ms⁻¹] / q² [nm⁻²] × 1e-15 = D [m²/s]

    Returns a list of (group_name, datasets) tuples — one per cumulant order.
    """
    if not hasattr(analyzer, 'method_a_data') or analyzer.method_a_data is None:
        raise ValueError("Cumulants Method A has not been run yet.")

    df = analyzer.method_a_data
    gamma_cols = [
        ('1st order frequency [1/ms]', '1st Order'),
        ('2nd order frequency [1/ms]', '2nd Order'),
        ('3rd order frequency [1/ms]', '3rd Order'),
    ]

    x_q2 = df['q^2'].values
    groups = []
    for col, order_label in gamma_cols:
        if col not in df.columns:
            continue
        # D [m²/s] = Γ [ms⁻¹] / q² [nm⁻²] × 1e-15
        D = df[col].values / x_q2 * 1e-15
        groups.append((
            f'Method A \u2013 {order_label}',
            [{
                'x': x_q2,
                'y': D,
                'y_err': None,
                'label': 'D vs q\u00b2',
                'style': 'Messung',
            }]
        ))

    return groups


# ---------------------------------------------------------------------------
# Export / launch
# ---------------------------------------------------------------------------

def _safe_filename(s: str) -> str:
    """Strip characters that are invalid in file names."""
    return re.sub(r'[<>:"/\\|?*\s]', '_', s)[:60]


def send_to_scatterforge(
    groups: list,
    plot_type: str = 'DLS - \u0393 vs q\u00b2',
) -> subprocess.Popen:
    """
    Send grouped datasets to ScatterForge by writing temp files and launching
    a subprocess.

    Args:
        groups:     List of (group_name, datasets) tuples as returned by
                    ``from_cumulant_method_a()`` / ``from_diffusion_vs_q2()``.
                    Each tuple becomes one DataGroup in ScatterForge's tree.
        plot_type:  Plot type string to activate in ScatterForge
                    (must match a key in ScatterForge's PLOT_TYPES).

    Returns:
        The subprocess.Popen object for the launched ScatterForge process.
    """
    if not groups:
        raise ValueError("No datasets to export.")

    # 1. Write data to a temporary directory
    tmpdir = tempfile.mkdtemp(prefix='ade_dls_sf_')

    # Build a structure that the launcher can reconstruct:
    # [ (group_name, [(filepath, label, style), ...]), ... ]
    launcher_groups = []
    file_counter = 0

    for group_name, datasets in groups:
        file_entries = []
        for d in datasets:
            safe_label = _safe_filename(d['label'])
            fpath = os.path.join(tmpdir, f'{file_counter:03d}_{safe_label}.dat')
            file_counter += 1

            x = np.asarray(d['x'], dtype=float)
            y = np.asarray(d['y'], dtype=float)

            if d.get('y_err') is not None:
                data = np.column_stack([x, y, np.asarray(d['y_err'], dtype=float)])
            else:
                data = np.column_stack([x, y])

            # ScatterForge's data_loader skips lines starting with '#'
            np.savetxt(fpath, data, delimiter='\t', comments='# ',
                       header=d['label'])

            file_entries.append((fpath, d['label'], d.get('style', 'Messung')))

        launcher_groups.append((group_name, file_entries))

    # 2. Generate a self-contained launcher script
    launcher_code = _build_launcher(launcher_groups, plot_type)
    launcher_path = os.path.join(tmpdir, 'launch_scatterforge.py')
    with open(launcher_path, 'w', encoding='utf-8') as f:
        f.write(launcher_code)

    # 3. Launch as independent subprocess
    proc = subprocess.Popen([sys.executable, launcher_path])
    return proc


def _build_launcher(launcher_groups: list, plot_type: str) -> str:
    """Build the Python launcher script for the ScatterForge subprocess."""
    groups_repr = repr(launcher_groups)
    sf_dir_repr = repr(_SF_DIR)
    plot_type_repr = repr(plot_type)

    return f"""\
import sys
import os

# Add ScatterForge to the Python path
sys.path.insert(0, {sf_dir_repr})

from PySide6.QtWidgets import QApplication
from scatter_plot import ScatterPlotApp
from core.models import DataSet, DataGroup

app = QApplication(sys.argv)
window = ScatterPlotApp()

# Each entry is (group_name, [(filepath, label, style), ...])
launcher_groups = {groups_repr}

for group_name, file_entries in launcher_groups:
    group = DataGroup(group_name)
    for filepath, label, style in file_entries:
        try:
            ds = DataSet(filepath, name=label, apply_auto_style=False)
            ds.display_label = label
            ds.apply_style_preset(style)
            group.add_dataset(ds)
        except Exception as exc:
            print(f"[ScatterForge Bridge] Could not load file: {{exc}}")
    window.groups.append(group)

# Switch to the requested DLS plot type
idx = window.plot_type_combo.findText({plot_type_repr})
if idx >= 0:
    window.plot_type_combo.setCurrentIndex(idx)

window.rebuild_tree()
window.update_plot()
window.show()
sys.exit(app.exec())
"""
