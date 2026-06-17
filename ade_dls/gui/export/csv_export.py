"""
CSV plot-data export
=====================
Export the raw data behind JADE-DLS plots as CSV tables, so users can assemble
their own comparison tables across measurement series (Excel, Origin, Python,
ScatterForge, ...).

This module is the **data layer only** — pure Python/pandas, no Qt. The GUI
handlers (``analysis_view.py``, the regularized/NNLS result dialogs) build the
tables here, write them to a user-chosen folder and register every written file
in the FAIR provenance record.

Three plot types are supported (v3.2.0):

1. Diffusion coefficient (Γ vs q²): data points, fit curve and the linear-fit
   parameters as three separate tables.
2. Clustering (multimodal): populations as columns plus the per-point raw data.
3. Distributions (multimodal): one column per distribution, with presets
   ``all`` / ``random`` / ``average_per_angle``.

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

# Γ [ms⁻¹] / q² [nm⁻²] × this factor = D [m²/s]  (see scatterforge_bridge / cumulant_analyzer)
_GAMMA_MS_TO_D = 1e-15

# Cumulant Method A gamma columns mapped to clean order tokens
_DIFFUSION_ORDERS = [
    ('1st order frequency [1/ms]', '1st_order'),
    ('2nd order frequency [1/ms]', '2nd_order'),
    ('3rd order frequency [1/ms]', '3rd_order'),
]


def _safe_filename(s: str) -> str:
    """Strip characters that are invalid in file names (mirrors scatterforge_bridge)."""
    return re.sub(r'[<>:"/\\|?*\s]', '_', str(s))[:80]


# ---------------------------------------------------------------------------
# 1. Diffusion coefficient tables (Γ vs q²)
# ---------------------------------------------------------------------------

def build_diffusion_tables(analyzer) -> Dict[str, pd.DataFrame]:
    """
    Build CSV tables for the Cumulant Method A diffusion plots (Γ vs q²).

    Reuses ``analyzer.method_a_data`` (measured points) and
    ``analyzer.method_a_regression_stats['regression_results']`` (linear OLS fit
    per cumulant order, with keys ``intercept``, ``q^2_coef``, ``q^2_se``,
    ``R_squared``).

    Returns a dict with up to three tables:
      - ``diffusion_datapoints``: q² and, per order, Γ and D = Γ/q²·1e-15.
      - ``diffusion_fitcurve``: densely sampled fit line(s) Γ_fit = intercept + slope·q².
      - ``diffusion_fit_parameters``: one row per order with slope, intercept,
        D, standard errors, R² and the fitted q² range — enough to re-plot the
        line directly via a function tool.
    """
    data = getattr(analyzer, 'method_a_data', None)
    if data is None or 'q^2' not in getattr(data, 'columns', []):
        raise ValueError("Cumulants Method A has not been run yet (no method_a_data).")

    stats = getattr(analyzer, 'method_a_regression_stats', None) or {}
    reg_results = stats.get('regression_results', []) or []
    # Index regression results by the gamma column they belong to.
    reg_by_col = {r.get('gamma_col'): r for r in reg_results}

    q2 = pd.to_numeric(data['q^2'], errors='coerce')

    # --- data points ---
    points = pd.DataFrame({'q^2 [nm^-2]': q2.values})
    available = []
    for col, token in _DIFFUSION_ORDERS:
        if col not in data.columns:
            continue
        available.append((col, token))
        gamma = pd.to_numeric(data[col], errors='coerce')
        points[f'gamma_{token} [1/ms]'] = gamma.values
        # D = Γ/q²·1e-15, guarding against division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            d = np.where(q2.values != 0, gamma.values / q2.values * _GAMMA_MS_TO_D, np.nan)
        points[f'D_{token} [m^2/s]'] = d

    if not available:
        raise ValueError("No cumulant gamma columns found in method_a_data.")

    tables: Dict[str, pd.DataFrame] = {'diffusion_datapoints': points}

    # --- fit curve + parameters ---
    q2_valid = q2.dropna()
    fit_curve = None
    param_rows = []
    if not q2_valid.empty:
        q2_min, q2_max = float(q2_valid.min()), float(q2_valid.max())
        x_line = np.linspace(q2_min, q2_max, 100)
        fit_curve = pd.DataFrame({'q^2 [nm^-2]': x_line})
        for col, token in available:
            reg = reg_by_col.get(col)
            if not reg:
                continue
            slope = float(reg.get('q^2_coef', np.nan))
            intercept = float(reg.get('intercept', np.nan))
            slope_se = float(reg.get('q^2_se', np.nan))
            r_squared = float(reg.get('R_squared', np.nan))
            fit_curve[f'gamma_fit_{token} [1/ms]'] = intercept + slope * x_line
            param_rows.append({
                'order': token,
                'slope [ms^-1 nm^2]': slope,
                'intercept [1/ms]': intercept,
                'D [m^2/s]': slope * _GAMMA_MS_TO_D,
                'slope_se [ms^-1 nm^2]': slope_se,
                'D_se [m^2/s]': slope_se * _GAMMA_MS_TO_D,
                'R_squared': r_squared,
                'q2_min [nm^-2]': q2_min,
                'q2_max [nm^-2]': q2_max,
            })

    if fit_curve is not None and len(fit_curve.columns) > 1:
        tables['diffusion_fitcurve'] = fit_curve
    if param_rows:
        tables['diffusion_fit_parameters'] = pd.DataFrame(param_rows)

    return tables


# ---------------------------------------------------------------------------
# 2. Clustering tables (multimodal)
# ---------------------------------------------------------------------------

def build_clustering_tables(clustered_df, cluster_info) -> Dict[str, pd.DataFrame]:
    """
    Build CSV tables for a multimodal clustering result.

    ``cluster_info`` is the dict returned by ``cluster_all_gammas`` (see
    ``ade_dls/analysis/clustering.py``). It carries:
      - ``population_stats``: per-population statistics DataFrame
        (cluster_id, n_files, abundance, mean_gamma, std_gamma, mean_D, std_D, ...).
      - ``gammas_df``: per-point raw data (filename, q_squared, gamma, D, cluster).

    Returns up to two tables:
      - ``clustering_populations``: populations as **columns**, statistic fields
        as rows (population_1 ... population_N).
      - ``clustering_points``: the per-point raw clustering data.
    """
    if not isinstance(cluster_info, dict):
        raise ValueError("cluster_info must be a dict (run a multimodal analysis first).")

    tables: Dict[str, pd.DataFrame] = {}

    # --- populations as columns ---
    stats_df = cluster_info.get('population_stats')
    cluster_id_to_pop = cluster_info.get('cluster_id_to_pop', {}) or {}
    if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
        wide = {}
        for _, row in stats_df.iterrows():
            cid = row.get('cluster_id')
            pop_num = cluster_id_to_pop.get(cid, cid)
            col_name = f'population_{pop_num}'
            # Statistic fields per population (drop the cluster_id row itself)
            wide[col_name] = row.drop(labels=['cluster_id'], errors='ignore')
        if wide:
            pops = pd.DataFrame(wide)
            # Order columns naturally (population_1, population_2, ...)
            pops = pops[sorted(pops.columns, key=lambda c: str(c))]
            pops.insert(0, 'statistic', pops.index)
            tables['clustering_populations'] = pops.reset_index(drop=True)

    # --- per-point raw data ---
    gammas_df = cluster_info.get('gammas_df')
    if isinstance(gammas_df, pd.DataFrame) and not gammas_df.empty:
        keep = [c for c in ['filename', 'original_col', 'q_squared', 'gamma', 'D',
                            'log_clustering_val', 'cluster', 'silhouette']
                if c in gammas_df.columns]
        points = gammas_df[keep].copy()
        if 'cluster' in points.columns and cluster_id_to_pop:
            points['population'] = points['cluster'].map(
                lambda c: cluster_id_to_pop.get(c, np.nan))
        tables['clustering_points'] = points.reset_index(drop=True)

    if not tables:
        raise ValueError("No clustering data available to export.")

    return tables


# ---------------------------------------------------------------------------
# 3. Distribution tables (multimodal)
# ---------------------------------------------------------------------------

def _get_distribution(entry) -> Optional[np.ndarray]:
    """Return the distribution array from a full_results entry (handles both keys)."""
    if not isinstance(entry, dict):
        return None
    dist = entry.get('distribution')
    if dist is None:
        dist = entry.get('f_optimized')
    return None if dist is None else np.asarray(dist, dtype=float)


def _get_decay_times(full_results) -> Optional[np.ndarray]:
    """Return the shared decay-time axis from the first usable entry."""
    for entry in full_results.values():
        if isinstance(entry, dict) and entry.get('decay_times') is not None:
            return np.asarray(entry['decay_times'], dtype=float)
    return None


def full_results_from_plots(plots_dict: dict, decay_times) -> dict:
    """
    Adapt a ``{name: (figure, plot_data)}`` plots dict (as stored in
    ``LaplaceAnalyzer.nnls_plots``) into the ``full_results`` shape expected by
    :func:`build_distribution_tables`.

    The NNLS path stores the distribution under ``plot_data['f_optimized']`` and
    does not carry a per-entry decay-time axis, so the shared *decay_times* is
    injected here.
    """
    full_results = {}
    if not plots_dict or decay_times is None:
        return full_results
    for name, entry in plots_dict.items():
        plot_data = entry[1] if isinstance(entry, (tuple, list)) and len(entry) > 1 else {}
        dist = plot_data.get('f_optimized') if isinstance(plot_data, dict) else None
        if dist is not None:
            full_results[name] = {
                'decay_times': decay_times,
                'distribution': dist,
                'peaks': plot_data.get('peaks'),
            }
    return full_results


def build_distribution_tables(
    full_results: dict,
    df_basedata=None,
    mode: str = 'all',
    n_random: Optional[int] = None,
    rng_seed: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build a CSV table of multimodal distributions.

    ``full_results`` maps a dataset key to ``{'decay_times', 'distribution', 'peaks'}``
    (see ``LaplaceAnalyzer.regularized_full_results`` / ``nnls_full_results``).

    Modes:
      - ``'all'``: one intensity column per distribution.
      - ``'random'``: ``n_random`` randomly chosen distributions (reproducible
        via ``rng_seed``).
      - ``'average_per_angle'``: one column per scattering angle, averaged over
        all measurements at that angle (mirrors ``plot_distributions`` mode
        ``'average'``). Requires ``df_basedata`` with an ``'angle [°]'`` column.

    Returns ``{'distributions': DataFrame}`` with a leading ``decay_time [s]``
    column followed by one intensity column per distribution/angle.
    """
    if not full_results:
        raise ValueError("No distribution results available (run a multimodal analysis first).")

    decay_times = _get_decay_times(full_results)
    if decay_times is None:
        raise ValueError("No decay-time axis found in distribution results.")

    out = pd.DataFrame({'decay_time [s]': decay_times})

    if mode == 'average_per_angle':
        if df_basedata is None or 'angle [°]' not in getattr(df_basedata, 'columns', []):
            raise ValueError("average_per_angle mode requires df_basedata with an 'angle [°]' column.")
        from ade_dls.analysis.regularized import find_dataset_key
        for angle in sorted(df_basedata['angle [°]'].dropna().unique()):
            angle_rows = df_basedata[df_basedata['angle [°]'] == angle]
            dists = []
            for _, row in angle_rows.iterrows():
                key = find_dataset_key(row['filename'], full_results)
                dist = _get_distribution(full_results.get(key)) if key else None
                if dist is not None and dist.shape == decay_times.shape:
                    dists.append(dist)
            if dists:
                mean_dist = np.mean(dists, axis=0)
                out[f'angle_{angle}deg_mean_intensity (n={len(dists)})'] = mean_dist
        if len(out.columns) <= 1:
            raise ValueError("No distributions could be matched to angles.")
        return {'distributions': out}

    # mode 'all' or 'random': select dataset keys
    keys = [k for k, v in full_results.items() if _get_distribution(v) is not None]
    if not keys:
        raise ValueError("No usable distributions found in results.")

    if mode == 'random':
        if not n_random or n_random < 1:
            raise ValueError("n_random must be a positive integer for mode='random'.")
        rng = np.random.default_rng(rng_seed)
        n_pick = min(int(n_random), len(keys))
        chosen_idx = rng.choice(len(keys), size=n_pick, replace=False)
        keys = [keys[i] for i in sorted(chosen_idx)]
    elif mode != 'all':
        raise ValueError(f"Unknown mode '{mode}'. Use 'all', 'random' or 'average_per_angle'.")

    for key in keys:
        dist = _get_distribution(full_results[key])
        if dist is not None and dist.shape == decay_times.shape:
            out[f'{key}_intensity'] = dist

    if len(out.columns) <= 1:
        raise ValueError("No distributions matched the shared decay-time axis.")

    return {'distributions': out}


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


def _extract_analysis_section(export_type: str, analyzer) -> Dict[str, Any]:
    """Pull analysis method, parameters and summary from the analyzer."""
    section: Dict[str, Any] = {'method': None, 'parameters': None, 'summary': None}
    if analyzer is None:
        return section

    if 'diffusion' in export_type:
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

    elif 'clustering' in export_type:
        # Determine which clustering method provided the data
        if hasattr(analyzer, 'method_d_cluster_info'):
            section['method'] = 'Cumulants Method D – cross-file clustering'
            ci = getattr(analyzer, 'method_d_cluster_info', None) or {}
        elif hasattr(analyzer, 'nnls_cluster_info'):
            section['method'] = 'NNLS peak clustering'
            ci = getattr(analyzer, 'nnls_cluster_info', None) or {}
        elif hasattr(analyzer, 'regularized_cluster_info'):
            section['method'] = 'Regularized NNLS peak clustering'
            ci = getattr(analyzer, 'regularized_cluster_info', None) or {}
        else:
            ci = {}
        section['parameters'] = _safe_json({
            k: v for k, v in ci.items()
            if k not in ('population_stats', 'gammas_df', 'clustering_plot',
                         'cluster_id_to_pop')
        })

    elif 'distributions' in export_type:
        if getattr(analyzer, 'regularized_params', None) is not None:
            section['method'] = 'Regularized NNLS (Tikhonov-Phillips)'
            section['parameters'] = _safe_json(getattr(analyzer, 'regularized_params', None))
            try:
                section['summary'] = analyzer.get_regularized_summary()
            except Exception:
                pass
        elif getattr(analyzer, 'nnls_params', None) is not None:
            section['method'] = 'NNLS (Non-Negative Least Squares)'
            section['parameters'] = _safe_json(getattr(analyzer, 'nnls_params', None))
            try:
                section['summary'] = analyzer.get_nnls_summary()
            except Exception:
                pass

    return section


def build_export_metadata(
    export_type: str,
    written_files: List[str],
    analyzer=None,
    provenance_panel=None,
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
            "type": "<diffusion_csv | clustering_csv | distributions_csv>",
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
    produced the export.
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

    analysis_section = _extract_analysis_section(export_type, analyzer)

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
