"""SLS (static light scattering) run helpers (no Qt), shared by the two result dialogs and
``ade_dls.headless``: locating the intensity files, the q^2 range from the dialog limits and the
cell formatting of the summary table. The numerics live in ``ade_dls.analysis.sls`` and
``LaplaceAnalyzer.run_sls_analysis``.
"""
import os
from typing import List, Optional, Tuple

import pandas as pd

QRG_MAX_WARNING = 1.3     # Guinier validity: q*Rg above this is flagged in the summary table


def q2_range_from_limits(q2_min: float, q2_max: float) -> Optional[Tuple[float, float]]:
    """Guinier-fit q^2 range from the two dialog fields (0 = unbounded; both 0 = all angles -> None)."""
    if q2_min > 0 or q2_max > 0:
        return (q2_min if q2_min > 0 else 0.0,
                q2_max if q2_max > 0 else float('inf'))
    return None


def asc_files_in_folder(folder: str) -> List[str]:
    """Full paths of the ``.asc`` files (any case) directly inside ``folder``."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.asc')]


def intensity_file_paths(df_basedata) -> List[str]:
    """Existing measurement files named in ``df_basedata`` (``folder`` + ``filename`` columns,
    one path per distinct filename); empty if the table has no ``folder`` column."""
    if df_basedata is None or df_basedata.empty or 'folder' not in df_basedata.columns:
        return []
    paths = []
    for _, row in df_basedata.drop_duplicates(subset=['filename']).iterrows():
        folder = str(row.get('folder', '') or '')
        fname = str(row.get('filename', '') or '')
        if folder and fname:
            fp = os.path.join(folder, fname)
            if os.path.isfile(fp):
                paths.append(fp)
    return paths


def format_sls_cell(val) -> str:
    """Text of one summary-table cell: '—' for missing, 4 decimals, scientific outside 0.01..1e4."""
    if pd.isna(val):
        return '—'
    try:
        f = float(val)
        if abs(f) < 0.01 or abs(f) > 1e4:
            return f'{f:.4e}'
        return f'{f:.4f}'
    except (TypeError, ValueError):
        return str(val)


def is_qrg_warning(column: str, val) -> bool:
    """True for a ``qRg_max`` cell above ``QRG_MAX_WARNING`` (the dialogs colour it)."""
    if column != 'qRg_max':
        return False
    try:
        return float(val) > QRG_MAX_WARNING
    except (TypeError, ValueError):
        return False


def load_intensity(analyzer, folder: Optional[str] = None) -> bool:
    """Make ``analyzer.df_intensity`` available (no-op if already loaded). Files come from
    ``analyzer.df_basedata``; if it names none, from the ``.asc`` files in ``folder``.
    Returns True when intensity data is available afterwards."""
    if getattr(analyzer, 'df_intensity', None) is not None:
        return True
    paths = intensity_file_paths(getattr(analyzer, 'df_basedata', None))
    if not paths and folder:
        paths = asc_files_in_folder(folder)
    if not paths:
        return False
    return bool(analyzer.load_intensity_data(paths))


def run_sls(analyzer, n_populations: int, q2_min: float = 0.0, q2_max: float = 0.0,
            exponent: int = 6, solvent_folder: Optional[str] = None) -> pd.DataFrame:
    """Population-resolved SLS analysis on the Regularized results of ``analyzer`` (needs
    ``load_intensity`` first). With ``solvent_folder`` the mean solvent intensity per angle is
    subtracted first (``sls.subtract_solvent``). Returns the summary table (also
    ``analyzer.sls_summary``)."""
    kwargs = {'solvent_folder': solvent_folder} if solvent_folder else {}
    return analyzer.run_sls_analysis(n_populations=n_populations,
                                     q2_range=q2_range_from_limits(q2_min, q2_max),
                                     exponent=exponent, **kwargs)
