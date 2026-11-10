"""
Schema contract tests for the instrument parser plugin system.

These verify that every parser honours the data contract the analysis/GUI layers
depend on. The LS tests are skipped when the external dataset is not mounted.
"""

import os
import sys

import pytest

# Make the repo root importable when running pytest from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALV_FOLDER = os.path.join(_REPO, "example_files_monomodal", "example_files_monomodal")
if not os.path.isdir(ALV_FOLDER):
    # Fallback: small committed extract (see tests/data/README.md)
    ALV_FOLDER = os.path.join(_REPO, "tests", "data", "alv_mono")
# Primary: repo-local copy; fallback: Nextcloud mount
_LS_LOCAL = os.path.join(_REPO, "example_files_monomodal", "LS Instruments", "2D", "csv")
_LS_NC = (r"C:\Users\traja\TUBAF_Nextcloud\Zentrales Home\Messdaten"
          r"\FileDrop LS Datasets\LS Instruments\2D\csv")
LS_FOLDER = _LS_LOCAL if os.path.isdir(_LS_LOCAL) else _LS_NC

REQUIRED_BASE = {'angle [°]', 'temperature [K]', 'wavelength [nm]',
                 'refractive_index', 'viscosity [cp]', 'filename', 'folder'}
# JADE v3.0: optional metadata used by analysis.weighting's noise model and
# by SLS intensity normalization. Every parser must expose these columns
# (NaN where the instrument/format doesn't provide them) for schema parity.
OPTIONAL_BASE = {'duration [s]', 'meancr0 [kHz]', 'meancr1 [kHz]', 'monitordiode [cps]'}


def _check_contract(parser, folder):
    files = parser.get_file_list(folder)
    assert files, "No measurement units found"

    bd = parser.extract_basedata(files[0])
    assert bd is not None, "extract_basedata returned None"
    assert REQUIRED_BASE <= set(bd.columns), \
        f"Missing basedata columns: {REQUIRED_BASE - set(bd.columns)}"
    assert OPTIONAL_BASE <= set(bd.columns), \
        f"Missing optional (weighting/SLS) basedata columns: {OPTIONAL_BASE - set(bd.columns)}"

    corr = parser.extract_correlations(files[0])
    assert corr is not None, "extract_correlations returned None"
    assert {'time [ms]', 'correlation 1'} <= set(corr.columns)

    cr = parser.extract_countrates(files[0])
    assert cr is not None, "extract_countrates returned None"
    assert {'time [s]', 'detectorslot 1'} <= set(cr.columns)

    # get_label must be unique across all units (dict-key safety).
    labels = [parser.get_label(f) for f in files]
    assert len(labels) == len(set(labels)), "get_label is not unique"

    # Cumulant export (Method A): columns must match the declared CUMULANT_MODE.
    cum = parser.extract_cumulants(files[0])
    assert cum is not None, "extract_cumulants returned None"
    assert 'filename' in cum.columns
    if parser.CUMULANT_MODE == "frequency":
        assert {'1st order frequency [1/ms]', '2nd order frequency [1/ms]',
                '3rd order frequency [1/ms]'} <= set(cum.columns)
    elif parser.CUMULANT_MODE == "radius":
        assert {'1st order Rh [nm]', '2nd order Rh [nm]',
                '3rd order Rh [nm]'} <= set(cum.columns)


@pytest.mark.skipif(not os.path.isdir(ALV_FOLDER), reason="ALV example dataset not available")
def test_alv_contract():
    from ade_dls.core.parsers.alv_parser import ALVParser
    _check_contract(ALVParser(), ALV_FOLDER)


@pytest.mark.skipif(not os.path.isdir(LS_FOLDER), reason="LS dataset not available")
def test_ls_contract():
    from ade_dls.core.parsers.ls_instruments_parser import LSInstrumentsParser
    _check_contract(LSInstrumentsParser(), LS_FOLDER)


@pytest.mark.skipif(not os.path.isdir(LS_FOLDER), reason="LS dataset not available")
def test_ls_wavelength_back_calculation():
    """The Cobolt laser is 561 nm; back-calculation from q must recover it."""
    from ade_dls.core.parsers.ls_instruments_parser import LSInstrumentsParser
    p = LSInstrumentsParser()
    files = p.get_file_list(LS_FOLDER)
    bd = p.extract_basedata(files[0])
    wl = float(bd['wavelength [nm]'].iloc[0])
    assert 555.0 < wl < 567.0, f"Unexpected wavelength {wl} nm (expected ~561)"


def test_detection():
    from ade_dls.core.parsers import detect_parser
    if os.path.isdir(ALV_FOLDER):
        assert detect_parser(ALV_FOLDER).INSTRUMENT_NAME == "ALV"
    if os.path.isdir(LS_FOLDER):
        assert detect_parser(LS_FOLDER).INSTRUMENT_NAME == "LSInstruments"


def test_detection_unknown_returns_none(tmp_path):
    from ade_dls.core.parsers import detect_parser
    assert detect_parser(str(tmp_path)) is None


@pytest.mark.skipif(not os.path.isdir(LS_FOLDER), reason="LS dataset not available")
def test_ls_prepare_basedata():
    """CumulantAnalyzer.prepare_basedata must work for LS data (no .asc re-extraction)."""
    import pandas as pd
    from ade_dls.core.parsers.ls_instruments_parser import LSInstrumentsParser
    from ade_dls.gui.analysis.cumulant_analyzer import CumulantAnalyzer

    parser = LSInstrumentsParser()
    files = parser.get_file_list(LS_FOLDER)

    # Build the same loaded_data dict that DataLoadWorker emits
    basedata_rows = [parser.extract_basedata(f) for f in files]
    basedata = pd.concat(basedata_rows, ignore_index=True)
    correlations = {parser.get_label(f): parser.extract_correlations(f) for f in files}

    loaded_data = {
        'data_folder': LS_FOLDER,
        'instrument': parser.INSTRUMENT_NAME,
        'files': files,
        'num_files': len(files),
        'basedata': basedata,
        'correlations': correlations,
        'countrates': {},
        'errors': [],
        'total_files': len(files),
        'successful_files': len(files),
    }

    analyzer = CumulantAnalyzer(loaded_data, LS_FOLDER)
    df = analyzer.prepare_basedata()
    assert df is not None and not df.empty
    assert 'q' in df.columns and 'q^2' in df.columns
    assert set(df['filename']) == set(correlations.keys())


# ---------------------------------------------------------------------------
# LS Instruments: duration / mean count rates must come from Summary.csv
# (regression tests; the parser used to hard-code NaN for these columns).
# ---------------------------------------------------------------------------
LS_AU2 = os.path.join(_REPO, "_dev", "LS-ComparisonData_FlorianTrummer",
                      "TU Freiberg", "Au2")
_skip_au2 = pytest.mark.skipif(not os.path.isdir(LS_AU2),
                               reason="LS Au2 comparison dataset not available")


def _au2_units():
    from ade_dls.core.parsers.ls_instruments_parser import LSInstrumentsParser
    p = LSInstrumentsParser()
    return p, p.get_file_list(LS_AU2)


def _au2_sample(files):
    """First unit plus a deterministic sample spread over all units."""
    idx = sorted({0, len(files) - 1, *range(0, len(files), 7)})
    return [files[i] for i in idx]


@_skip_au2
def test_ls_basedata_duration_and_countrates_from_summary():
    import numpy as np
    import pandas as pd
    p, files = _au2_units()
    assert files
    for unit in _au2_sample(files):
        s = pd.read_csv(os.path.join(unit, "Summary.csv"), encoding="utf-8-sig").iloc[0]
        bd = p.extract_basedata(unit).iloc[0]
        dur = float(s['Duration [s]'])
        a = float(s['Average count rate A [kHz]'])
        b = float(s['Average count rate B [kHz]'])
        assert dur == 12
        assert bd['duration [s]'] == pytest.approx(dur, rel=1e-12), unit
        assert bd['meancr0 [kHz]'] == pytest.approx(a, rel=1e-12), unit
        assert bd['meancr1 [kHz]'] == pytest.approx(b, rel=1e-12), unit
        for col in ('duration [s]', 'meancr0 [kHz]', 'meancr1 [kHz]'):
            assert np.isfinite(bd[col]) and bd[col] > 0, (unit, col)
        # LS provides no monitor diode: stays NaN (explicitly intended).
        assert np.isnan(bd['monitordiode [cps]'])


@_skip_au2
def test_ls_basedata_countrates_consistent_with_count_trace():
    p, files = _au2_units()
    for unit in _au2_sample(files):
        bd = p.extract_basedata(unit).iloc[0]
        cr = p.extract_countrates(unit)
        assert bd['meancr0 [kHz]'] == pytest.approx(cr['detectorslot 1'].mean(), rel=1e-9), unit
        assert bd['meancr1 [kHz]'] == pytest.approx(cr['detectorslot 2'].mean(), rel=1e-9), unit


@_skip_au2
def test_ls_end_to_end_weighting_available():
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from ade_dls.headless import load_series
    from ade_dls.engine.series import preprocess_series

    data = load_series(LS_AU2, load_countrates=False, load_cumulants=False, preprocess=False)
    preprocess_series(data)
    assert data['weighting_available'] is True, data.get('weighting_unavailable_reason')
    weights = data['weights']
    assert set(weights) == set(data['correlations'])
    for name, w in weights.items():
        assert np.mean(np.asarray(w, dtype=float)) == pytest.approx(1.0, rel=1e-6), name
