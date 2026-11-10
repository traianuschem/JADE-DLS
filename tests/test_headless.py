"""Tests for ade_dls.headless (GUI-free API over the same engine classes the GUI uses).

Real data: tests/data/alv_mono (6 measurement units: angles 30/60/90 deg, 2 repeats each).
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")      # without it matplotlib may bind PySide6 (installed here) when a later test builds Qt canvases
import numpy as np
import pytest

from ade_dls import headless

ALV = Path(__file__).resolve().parent / "data" / "alv_mono"

pytestmark = pytest.mark.skipif(not ALV.exists(), reason="tests/data/alv_mono missing")

SMALL_GRID = np.logspace(-8, 1, 26)       # keeps NNLS/Regularized fits short (full grids: 100/200 points)
GAMMA_COL = "1st order frequency [1/ms]"


@pytest.fixture(scope="module")
def series():
    return headless.load_series(str(ALV))


@pytest.fixture
def data():
    return headless.load_series(str(ALV))


# ----------------------------------------------------------------------------- loading
def test_load_series_returns_preprocessed_data_for_all_files(series):
    assert series["num_files"] == 6 and series["instrument"] == "ALV"
    assert {"basedata", "correlations", "cumulants", "processed_correlations", "weights"} <= set(series)
    assert len(series["processed_correlations"]) == 6


def test_load_series_without_preprocessing_skips_processed_correlations():
    d = headless.load_series(str(ALV), preprocess=False)
    assert d.get("processed_correlations") is None and "correlations" in d


def test_load_series_reports_progress():
    calls = []
    headless.load_series(str(ALV), preprocess=False, on_progress=lambda *a: calls.append(a))
    assert calls and calls[-1][0] == calls[-1][1]


def test_load_series_unknown_format_raises_data_load_error(tmp_path):
    with pytest.raises(headless.DataLoadError):
        headless.load_series(str(tmp_path))


# ----------------------------------------------------------------------------- Method A oracle
def test_run_cumulant_a_matches_hand_computed_ols(series):
    """D = slope of Gamma [1/s] over q^2 [1/m^2]; Gamma from the instrument cumulants in 1/ms,
    q^2 from basedata in 1/nm^2 -> hand OLS on the raw table with the unit factors 1e3 and 1e18."""
    analyzer, res = headless.run_cumulant_a(series)
    labels = list(series["basedata"]["filename"])
    gamma = np.array([series["cumulants"][l][GAMMA_COL] for l in labels], float)
    q2 = series["basedata"]["q^2"].to_numpy(float)
    slope = np.polyfit(q2, gamma, 1)[0]
    expected_d = slope * 1e-15                        # (1/ms)/(1/nm^2) -> m^2/s
    row = res.iloc[0]
    assert row["D [m²/s]"] == pytest.approx(expected_d, rel=1e-6)
    assert row["D [m²/s]"] == pytest.approx(2.9114e-12, rel=1e-4)       # literal value of the fixture
    assert 0.9 < row["R_squared"] <= 1.0


def test_run_cumulant_a_equals_direct_analyzer_call(series):
    _, res = headless.run_cumulant_a(series)
    direct = headless.cumulant_analyzer(series).run_method_a()
    np.testing.assert_allclose(res["D [m²/s]"].to_numpy(float), direct["D [m²/s]"].to_numpy(float),
                               rtol=1e-12)


def test_run_cumulant_a_through_origin_differs_from_free_intercept(series):
    _, free = headless.run_cumulant_a(series)
    _, origin = headless.run_cumulant_a(series, fit_through_origin=True)
    assert origin.iloc[0]["D [m²/s]"] != pytest.approx(free.iloc[0]["D [m²/s]"], rel=1e-3, abs=0)


def test_run_cumulant_a_q_range_restricts_to_selected_angles(series):
    q = series["basedata"]["q"].to_numpy(float)
    _, res = headless.run_cumulant_a(series, q_range=(0.0, float(q[2]) * 1.0001))
    assert len(res) >= 1
    assert res.iloc[0]["D [m²/s]"] > 0


def test_run_cumulant_b_gives_plausible_diffusion_coefficient(series):
    _, res = headless.run_cumulant_b(series, (1e-6, 1e-2))
    assert 1e-13 < res.iloc[0]["D [m²/s]"] < 1e-10
    assert res.iloc[0]["Rh [nm]"] > 0


def test_run_cumulant_b_weighting_flag_is_forwarded(series):
    analyzer, _ = headless.run_cumulant_b(series, (1e-6, 1e-2), use_weighting=True)
    assert analyzer.use_weighting is True


# ----------------------------------------------------------------------------- guards
def test_cumulant_analyzer_requires_loaded_data():
    with pytest.raises(ValueError):
        headless.cumulant_analyzer({})


def test_laplace_analyzer_requires_basedata():
    with pytest.raises(ValueError):
        headless.laplace_analyzer({"correlations": {}})


def test_laplace_analyzer_completes_missing_derived_tables(data):
    data.pop("c", None)
    data.pop("delta_c", None)
    data.pop("df_basedata", None)
    headless.cumulant_analyzer(data)                  # prepares basedata
    lap = headless.laplace_analyzer(data)
    assert {"c", "delta_c", "df_basedata"} <= set(data) and lap is not None


# ----------------------------------------------------------------------------- defaults
def test_defaults_are_jade_gold_defaults():
    n = headless.nnls_defaults()
    assert (len(n["decay_times"]), n["prominence"]) == (100, 0.01)
    assert n["decay_times"][0] == pytest.approx(1e-8) and n["decay_times"][-1] == pytest.approx(10.0)
    r = headless.regularized_defaults()
    assert (len(r["decay_times"]), r["alpha"], r["prominence"]) == (200, 0.2, 0.005)
    assert r["fit_beta"] is True and r["normalize"] is True
    d = headless.method_d_defaults()
    assert (d["gap_threshold"], d["distance_threshold"], d["clustering_strategy"]) == (
        2.0, 2.0, "silhouette_refined")  # gap_threshold 2.0 = JADE reference runs (user decision 2026-11-06)


def test_defaults_return_fresh_objects_each_call():
    a, b = headless.nnls_defaults(), headless.nnls_defaults()
    a["prominence"] = 0.5
    a["decay_times"][0] = 123.0
    assert b["prominence"] == 0.01 and b["decay_times"][0] == pytest.approx(1e-8)


# ----------------------------------------------------------------------------- Laplace (slow)
@pytest.mark.slow
def test_run_nnls_populates_results_on_analyzer(series):
    lap, params = headless.run_nnls(series, {"decay_times": SMALL_GRID})
    assert params["prominence"] == 0.01 and len(params["decay_times"]) == 26
    assert lap.nnls_final_results is not None and len(lap.nnls_final_results) >= 1
    assert np.isfinite(lap.nnls_final_results["D [m^2/s]"].to_numpy(float)).all()   # coarse grid: sign not asserted


@pytest.mark.slow
def test_run_regularized_populates_results_and_applies_overrides(series):
    lap, params = headless.run_regularized(series, {"decay_times": SMALL_GRID, "alpha": 0.5})
    assert params["alpha"] == 0.5 and params["fit_beta"] is True
    assert lap.regularized_final_results is not None and len(lap.regularized_final_results) >= 1


# ----------------------------------------------------------------------------- GUI parity
def _dialog_params(cls, analyzer):
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])      # noqa: F841  (keep alive)
    return cls(analyzer).params


@pytest.mark.parametrize("kind", ["nnls", "regularized"])
def test_headless_defaults_equal_gui_dialog_defaults(series, kind):
    """Single source of truth until N-01 centralises the defaults: every key both sides define
    must carry the same value."""
    lap = headless.laplace_analyzer(series)
    pytest.importorskip("PyQt5.QtWidgets")
    if kind == "nnls":
        from ade_dls.gui.dialogs.nnls_dialog import NNLSDialog as Dlg
        mine = headless.nnls_defaults()
    else:
        from ade_dls.gui.dialogs.regularized_dialog import RegularizedDialog as Dlg
        mine = headless.regularized_defaults()
    gui = _dialog_params(Dlg, lap)
    shared = set(mine) & set(gui)
    assert {"decay_times", "prominence", "distance_threshold", "clustering_strategy"} <= shared
    for key in shared:
        np.testing.assert_array_equal(np.asarray(mine[key]), np.asarray(gui[key]), err_msg=key)
