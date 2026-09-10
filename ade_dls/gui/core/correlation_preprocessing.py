# -*- coding: utf-8 -*-
"""
correlation_preprocessing.py
=============================
Single, Qt-free place where the *unweighted* processed correlations, the
optional noise correction, and the optional Biganzoli-Ferri noise weights
(JADE-DLS v3.0, see ``ade_dls.analysis.weighting``) are computed exactly
ONCE per filtered dataset.

Rationale
---------
Before this module existed, `process_correlation_data()` + noise
correction were re-run lazily and independently by
``CumulantAnalyzer.prepare_processed_correlations()``,
``MainWindow.run_nnls_analysis()`` and ``MainWindow.run_regularized_analysis()``
every time an analysis was launched, each keeping its own
``processed_correlations`` copy. Re-running NNLS/Regularized a second time
after a re-filter could apply noise correction twice to the same cached
dict (a double-correction bug). This module fixes that by computing the
canonical *unweighted* ``processed_correlations`` once, right after
filtering, and storing it (plus the weight vectors) on the pipeline data
dict. Analyzers then read it instead of recomputing it; whether an
individual analysis run actually USES the weights is a separate, later
choice (see ``LaplaceAnalyzer.set_weighting`` /
``CumulantAnalyzer.use_weighting``).
"""
from typing import Dict, Optional, Tuple

import pandas as pd

# Columns dropped from the raw per-channel correlation DataFrame once the
# mean g(2)-1 has been computed (mirrors the notebook / existing call sites).
COLUMNS_TO_DROP = ['time [ms]', 'correlation 1', 'correlation 2',
                    'correlation 3', 'correlation 4']

# df_basedata columns required to estimate per-file noise weights (JADE v3.0
# preprocessing.extract_data(); see ade_dls.core.preprocessing).
REQUIRED_WEIGHT_COLS = ['duration [s]', 'meancr0 [kHz]', 'meancr1 [kHz]']


def preprocess_correlations(
    correlations: Dict[str, pd.DataFrame],
    df_basedata: Optional[pd.DataFrame],
    noise_params: Optional[dict] = None,
) -> Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, "object"]], Optional[str]]:
    """
    Build the canonical unweighted processed-correlations dict for one
    filtered dataset, apply optional noise correction, and attempt to
    estimate per-file Biganzoli-Ferri noise weights.

    Args:
        correlations: raw per-file correlation DataFrames (as stored in
            ``pipeline.data['correlations']`` after filtering).
        df_basedata: the matching (filtered) base-data DataFrame. May be
            ``None`` or lack the optional weighting columns (e.g. LS
            Instruments data) -- weighting is then unavailable, not an
            error.
        noise_params: optional dict forwarded to
            ``ade_dls.analysis.noise.apply_noise_corrections`` (baseline /
            intercept correction), or ``None``/falsy to skip it.

    Returns:
        (processed_correlations, weights_dict, reason)
        - processed_correlations: dict of DataFrames with 't [s]'/'g(2)-1'
          (and noise-corrected if requested). Never carries a 'weight'
          column -- that is added later, on demand, via
          ``apply_weights_to_correlations``.
        - weights_dict: {filename: weight_array} if weighting could be
          estimated for every file, else ``None``.
        - reason: human-readable explanation when weights_dict is ``None``,
          else ``None``.
    """
    from ade_dls.core.preprocessing import process_correlation_data

    processed = process_correlation_data(correlations, list(COLUMNS_TO_DROP))

    if noise_params:
        from ade_dls.analysis.noise import apply_noise_corrections
        processed = apply_noise_corrections(processed, **noise_params)

    weights: Optional[Dict[str, "object"]] = None
    reason: Optional[str] = None

    if df_basedata is None or not all(c in df_basedata.columns for c in REQUIRED_WEIGHT_COLS):
        reason = ("basedata carries no Duration/MeanCR0/MeanCR1 columns "
                   "(e.g. non-ALV instrument) -- noise weighting unavailable.")
    else:
        sub = df_basedata[df_basedata['filename'].isin(list(processed.keys()))]
        bad = sub.loc[sub[REQUIRED_WEIGHT_COLS].isna().any(axis=1), 'filename'].tolist()
        if bad:
            preview = ', '.join(bad[:3]) + (', ...' if len(bad) > 3 else '')
            reason = (f"{len(bad)} file(s) are missing Duration/MeanCR0/MeanCR1 "
                      f"metadata (e.g. {preview}) -- noise weighting unavailable.")
        else:
            try:
                from ade_dls.analysis.weighting import compute_weights_for_all
                weights = compute_weights_for_all(processed, df_basedata)
            except Exception as exc:
                # compute_weights_for_all() raises by design on bad/missing
                # metadata (see weighting.py docstring) -- the GUI must
                # degrade gracefully instead of crashing the whole analysis.
                reason = f"weight estimation failed: {exc}"

    return processed, weights, reason
