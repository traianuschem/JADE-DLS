"""
CSV plot-data export
=====================
Export the raw data behind JADE-DLS plots as CSV tables, so users can assemble
their own comparison tables across measurement series (Excel, Origin, Python,
ScatterForge, ...).

This module is the **data layer only** — pure Python/pandas, no Qt. The GUI
handler (``analysis_view.py``'s "Export current plot as CSV" button) builds
the per-series tables in ``plot_series_export.py`` (generic extraction from
the currently displayed matplotlib Figure), writes them to a user-chosen
folder via :func:`write_tables`, and registers every written file in the
FAIR provenance record via :func:`register_outputs_in_provenance`.

One export type (v3.4.0): ``plot_csv`` — one CSV per data series (measured
points, fit curve, histogram bars, error bars, ...) of the plot currently
shown in the Analysis View, one file per panel/series. See
``plot_series_export.py`` for the extraction logic.

CSV format is international (comma separator, dot decimal) — the pandas default —
chosen for robust downstream processing.

Each export also writes an ``export_metadata.json`` alongside the CSVs so the
data can be traced back to its session, measurement series and analysis parameters
(see :func:`build_export_metadata` / :func:`write_metadata`).
"""

import json
import os
import platform
import re
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_filename(s: str) -> str:
    """Strip characters that are invalid in file names (mirrors scatterforge_bridge)."""
    return re.sub(r'[<>:"/\\|?*\s]', '_', str(s))[:80]


# ---------------------------------------------------------------------------
# 4. Export metadata
# ---------------------------------------------------------------------------

def _safe_json(value: Any) -> Any:
    """Convert a value to a JSON-serialisable form (best-effort)."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    try:
        import pandas as pd
        if isinstance(value, pd.DataFrame):
            return None
    except ImportError:
        pass
    try:
        return str(value)
    except Exception:
        return None


def _extract_data_section(analyzer) -> Dict[str, Any]:
    """Pull measurement-series metadata from a CumulantAnalyzer or LaplaceAnalyzer."""
    data: Dict[str, Any] = {
        'folder': None,
        'n_datasets': None,
        'angles_deg': None,
        'temperature_K': None,
        'viscosity_cp': None,
    }
    if analyzer is None:
        return data

    # Data folder — CumulantAnalyzer stores it, LaplaceAnalyzer may not
    folder = getattr(analyzer, 'data_folder', None)
    if folder is None:
        # LaplaceAnalyzer: derive from basedata filenames if possible
        bd = getattr(analyzer, 'df_basedata', None)
        if bd is not None and 'folder' in bd.columns:
            folder = bd['folder'].iloc[0] if len(bd) > 0 else None
    data['folder'] = str(folder) if folder else None

    # Number of datasets
    corr = getattr(analyzer, 'processed_correlations', None)
    if corr is not None:
        data['n_datasets'] = len(corr)
    else:
        ma_data = getattr(analyzer, 'method_a_data', None)
        if ma_data is not None:
            data['n_datasets'] = len(ma_data)

    # Basedata angles / temperature / viscosity
    bd = getattr(analyzer, 'df_basedata', None)
    if bd is not None:
        angle_col = next((c for c in ['angle [°]', 'angle'] if c in bd.columns), None)
        if angle_col:
            angles = sorted(bd[angle_col].dropna().unique().tolist())
            data['angles_deg'] = [float(a) for a in angles]
        temp_col = next((c for c in ['temperature [K]', 'temperature_K'] if c in bd.columns), None)
        if temp_col:
            t = pd.to_numeric(bd[temp_col], errors='coerce').dropna()
            if len(t) > 0:
                data['temperature_K'] = {'mean': float(t.mean()), 'std': float(t.std(ddof=0))}
        visc_col = next((c for c in ['viscosity [cp]', 'viscosity_mPas'] if c in bd.columns), None)
        if visc_col:
            v = pd.to_numeric(bd[visc_col], errors='coerce').dropna()
            if len(v) > 0:
                data['viscosity_cp'] = {'mean': float(v.mean()), 'std': float(v.std(ddof=0))}

    return data


def _extract_analysis_section(export_type: str, analyzer, method_tag: str = None) -> Dict[str, Any]:
    """
    Pull analysis method, parameters and summary from the analyzer.

    *method_tag* is the plot-registry method tag (``all_plot_methods[name]``,
    e.g. ``'Method A'``/``'Method B'``/``'Method C'``/``'Method D'``/``'NNLS'``/
    ``'Regularized'``) that identifies which analysis produced the exported
    plot -- it replaces the old substring-matching on *export_type*, which
    only ever distinguished 'diffusion'/'clustering'/'distributions'.
    """
    section: Dict[str, Any] = {'method': None, 'parameters': None, 'summary': None}
    if analyzer is None:
        return section

    tag = method_tag or ''

    if 'Regularized' in tag:
        section['method'] = 'Regularized NNLS (Tikhonov-Phillips)'
        section['parameters'] = _safe_json(getattr(analyzer, 'regularized_params', None))
        try:
            section['summary'] = analyzer.get_regularized_summary()
        except Exception:
            pass

    elif 'NNLS' in tag:
        section['method'] = 'NNLS (Non-Negative Least Squares)'
        section['parameters'] = _safe_json(getattr(analyzer, 'nnls_params', None))
        try:
            section['summary'] = analyzer.get_nnls_summary()
        except Exception:
            pass

    elif 'Method A' in tag:
        section['method'] = 'Cumulants Method A'
        stats = getattr(analyzer, 'method_a_regression_stats', None) or {}
        reg = stats.get('regression_results')
        if reg:
            section['parameters'] = _safe_json(
                [{k: v for k, v in r.items() if k != 'gamma_col'} for r in reg])
        results = getattr(analyzer, 'method_a_results', None)
        if results is not None:
            try:
                section['summary'] = _safe_json(results.to_dict(orient='records'))
            except Exception:
                pass

    elif 'Method B' in tag:
        section['method'] = 'Cumulants Method B (linear cumulant fit)'
        section['parameters'] = _safe_json(
            {'fit_limits': getattr(analyzer, 'method_b_fit_limits', None)})

    elif 'Method C' in tag:
        section['method'] = 'Cumulants Method C (iterative non-linear fit)'
        section['parameters'] = _safe_json(getattr(analyzer, 'method_c_params', None))

    elif 'Method D' in tag:
        section['method'] = 'Cumulants Method D (multi-exponential decomposition)'
        section['parameters'] = _safe_json(getattr(analyzer, 'method_d_params', None))
        ci = getattr(analyzer, 'method_d_cluster_info', None) or {}
        section['summary'] = _safe_json({
            k: v for k, v in ci.items()
            if k not in ('population_stats', 'gammas_df', 'clustering_plot',
                         'cluster_id_to_pop')
        })

    return section


def build_export_metadata(
    export_type: str,
    written_files: List[str],
    analyzer=None,
    provenance_panel=None,
    method_tag: str = None,
) -> Dict[str, Any]:
    """
    Build the export metadata dict that is written alongside every CSV export.

    The schema mirrors the FAIR provenance record produced by
    ``ProvenanceRecord.to_dict()`` (see ``ade_dls/gui/core/provenance.py``) but
    is lighter-weight and focused on a single export event.

    Schema::

        {
          "$schema":          "https://jade-dls.de/schema/export-metadata/v1.0",
          "export_id":        "<uuid4>",
          "created":          "<ISO8601 UTC>",
          "session_record_id":"<provenance record_id | null>",
          "agent": {
            "software":       "JADE-DLS",
            "version":        "<__version__>",
            "platform":       "<platform string>",
            "python_version": "<sys.version>"
          },
          "export": {
            "type": "<plot_csv>",
            "files": [{"filename": "...", "sha256": "..."}]
          },
          "data": {
            "folder": "<path | null>",
            "n_datasets": <int | null>,
            "angles_deg": [<float>],
            "temperature_K": {"mean": <float>, "std": <float>} | null,
            "viscosity_cp":  {"mean": <float>, "std": <float>} | null
          },
          "analysis": {
            "method":     "<string | null>",
            "parameters": <dict | null>,
            "summary":    <list | dict | null>
          }
        }

    All fields that cannot be determined are ``null`` rather than absent, so
    consumers can rely on the schema structure regardless of which analysis path
    produced the export. *method_tag* (e.g. ``'NNLS'``, ``'Method C'``) selects
    which analyzer attributes feed the ``analysis`` section — see
    :func:`_extract_analysis_section`.
    """
    import ade_dls
    from ade_dls.gui.core.provenance import compute_sha256

    # session_record_id from the live provenance panel
    session_record_id = None
    if provenance_panel is not None:
        try:
            session_record_id = provenance_panel.get_record_id()
        except Exception:
            pass

    # data folder fallback via provenance record
    if analyzer is None and provenance_panel is not None:
        try:
            record = provenance_panel.get_record()
            folder = getattr(record, '_input_folder', None)
        except Exception:
            folder = None
    else:
        folder = None

    data_section = _extract_data_section(analyzer)
    if data_section['folder'] is None and folder:
        data_section['folder'] = str(folder)

    analysis_section = _extract_analysis_section(export_type, analyzer, method_tag)

    # File list with SHA-256 hashes
    file_entries = []
    for path in written_files:
        sha = compute_sha256(path)
        file_entries.append({
            'filename': os.path.basename(path),
            'sha256': sha or None,
        })

    return {
        '$schema': 'https://jade-dls.de/schema/export-metadata/v1.0',
        'export_id': str(uuid.uuid4()),
        'created': datetime.now(tz=timezone.utc).isoformat(),
        'session_record_id': session_record_id,
        'agent': {
            'software': 'JADE-DLS',
            'version': ade_dls.__version__,
            'platform': platform.platform(),
            'python_version': sys.version,
        },
        'export': {
            'type': export_type,
            'files': file_entries,
        },
        'data': data_section,
        'analysis': analysis_section,
    }


def write_metadata(
    metadata: Dict[str, Any],
    target_dir: str,
    prefix: str = "",
) -> str:
    """
    Write *metadata* as ``<prefix>export_metadata.json`` into *target_dir*.

    Returns the written file path.
    """
    os.makedirs(target_dir, exist_ok=True)
    name = _safe_filename(f'{prefix}export_metadata') + '.json'
    path = os.path.join(target_dir, name)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Writer + provenance
# ---------------------------------------------------------------------------

def write_tables(
    tables: Dict[str, pd.DataFrame],
    target_dir: str,
    prefix: str = "",
) -> List[str]:
    """
    Write each table as a comma-separated CSV (international format) into
    *target_dir*. File name is ``<prefix><table_key>.csv``.

    Returns the list of written file paths.
    """
    if not tables:
        raise ValueError("No tables to write.")
    os.makedirs(target_dir, exist_ok=True)
    written: List[str] = []
    for key, df in tables.items():
        name = _safe_filename(f'{prefix}{key}') + '.csv'
        path = os.path.join(target_dir, name)
        df.to_csv(path, index=False)
        written.append(path)
    return written


def register_outputs_in_provenance(
    provenance_panel,
    filepaths: List[str],
    output_type: str = "plot_data_csv",
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Register each written file in the FAIR provenance record so its SHA-256 hash
    is captured. *provenance_panel* may be ``None`` (no-op) — exports must still
    succeed when provenance tracking is unavailable.

    *extra_fields* is forwarded to every catalog entry (e.g.
    ``{"export_id": "<uuid>"}`` to cross-link an export-metadata JSON).
    """
    if provenance_panel is None:
        return
    for path in filepaths:
        try:
            provenance_panel.register_output(
                output_type=output_type,
                label=os.path.basename(path),
                filepath=path,
                extra_fields=extra_fields,
            )
        except Exception:
            # Provenance must never break a successful data export.
            pass
