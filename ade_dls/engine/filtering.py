"""Data-set filtering (no Qt): the data operations behind the count-rate / correlation filter
dialogs, so that the GUI and ``ade_dls.headless`` exclude measurements in exactly the same way.

``data`` is the series dictionary produced by the loader (keys ``countrates``, ``correlations``,
``basedata``, ``num_files`` ...). The dialogs decide *which* files are kept; the functions here
apply that decision to the dictionary. Processed correlations / noise weights are not touched:
call ``ade_dls.engine.series.preprocess_series`` afterwards (as the GUI does).
"""
from typing import Any, Dict, Iterable, List, Optional


def excluded_files(original: Iterable[str], kept: Iterable[str]) -> List[str]:
    """Labels of ``original`` that are not in ``kept`` (order of ``original``)."""
    kept = set(kept)
    return [f for f in original if f not in kept]


def restrict_basedata(data: Dict[str, Any], remaining_files: Iterable[str]) -> None:
    """Keep only the ``basedata`` rows whose ``filename`` is in ``remaining_files`` and update
    ``num_files`` (in place)."""
    remaining_files = list(remaining_files)
    basedata = data['basedata']
    data['basedata'] = basedata[basedata['filename'].isin(remaining_files)]
    data['num_files'] = len(remaining_files)


def sync_basedata_to_correlations(data: Dict[str, Any]) -> None:
    """Make ``basedata`` / ``num_files`` match the files that still have correlations
    (no-op unless both entries exist)."""
    if 'basedata' in data and 'correlations' in data:
        restrict_basedata(data, data['correlations'].keys())


def correlations_to_filter(data: Dict[str, Any], filtered_data: Dict[str, Any]) -> Dict[str, Any]:
    """Correlations of ``data`` that survived the count-rate step recorded in ``filtered_data``
    (all of them if ``filtered_data`` has no count rates)."""
    surviving = filtered_data.get('countrates', data['correlations']).keys()
    return {k: v for k, v in data['correlations'].items() if k in surviving}


def apply_countrate_filter(data: Dict[str, Any], filtered_countrates: Dict[str, Any]) -> List[str]:
    """Reduce ``data`` (in place) to the files in ``filtered_countrates``: count rates,
    correlations and basedata. Returns the excluded labels."""
    excluded = excluded_files(data['countrates'], filtered_countrates)
    data['countrates'] = filtered_countrates
    if 'correlations' in data:
        data['correlations'] = {k: v for k, v in data['correlations'].items()
                                if k in filtered_countrates}
    if 'basedata' in data:
        restrict_basedata(data, filtered_countrates.keys())
    return excluded


def apply_correlation_filter(data: Dict[str, Any], filtered_correlations: Dict[str, Any]) -> List[str]:
    """Reduce ``data`` (in place) to the files in ``filtered_correlations``: correlations and
    basedata (count rates stay untouched, as in the GUI). Returns the excluded labels."""
    excluded = excluded_files(data['correlations'], filtered_correlations)
    data['correlations'] = filtered_correlations
    if 'basedata' in data:
        restrict_basedata(data, filtered_correlations.keys())
    return excluded


def exclude_files(data: Dict[str, Any], files: Iterable[str]) -> List[str]:
    """Remove the measurements ``files`` (labels) from count rates, correlations and basedata
    (in place). Returns the labels that were actually present."""
    files = set(files)
    present = [f for f in data.get('correlations', {}) if f in files]
    present += [f for f in data.get('countrates', {}) if f in files and f not in present]
    for key in ('countrates', 'correlations'):
        if key in data:
            data[key] = {k: v for k, v in data[key].items() if k not in files}
    if 'basedata' in data:
        basedata = data['basedata']
        data['basedata'] = basedata[~basedata['filename'].isin(files)]
        data['num_files'] = len(data['basedata'])
    return present


def noise_correction_lines(noise_params: Optional[Dict[str, Any]]) -> List[str]:
    """Bullet lines describing the active noise corrections (empty if none selected)."""
    p = noise_params or {}
    lines = []
    if p.get('baseline_correction'):
        lines.append(f"  • Baseline correction: last {p.get('baseline_pct', 5)} %")
    if p.get('intercept_correction'):
        lines.append(f"  • Intercept correction: first {p.get('intercept_pct', 2)} %")
    return lines


def filtering_summary(original_count: int, filtered_data: Dict[str, Any],
                      noise_params: Optional[Dict[str, Any]]) -> str:
    """Text of the 'Filtering complete' message (``noise_params`` falsy = not applied)."""
    final_count = filtered_data.get('num_files', 0)
    msg = "Filtering complete!\n\n"
    msg += f"Original files: {original_count}\n"
    msg += f"Excluded: {original_count - final_count}\n"
    msg += f"Final dataset: {final_count} files\n\n"
    msg += f"Base data: {len(filtered_data.get('basedata', []))} entries\n"
    msg += f"Correlations: {len(filtered_data.get('correlations', {}))} datasets\n"
    msg += f"Count rates: {len(filtered_data.get('countrates', {}))} datasets\n\n"
    if noise_params:
        lines = noise_correction_lines(noise_params)
        if lines:
            msg += "Noise correction active:\n" + "\n".join(lines)
        else:
            msg += "Noise correction: active (no corrections selected)"
    else:
        msg += "Noise correction: not applied"
    return msg
