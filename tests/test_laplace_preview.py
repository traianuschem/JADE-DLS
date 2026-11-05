"""Tests for ade_dls.engine.laplace_preview (analysis behind the NNLS / Regularized preview tabs)
and the dialogs' use of it. Inputs are synthetic: Γ = D·q² with known D, tau = 1/Γ."""
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from ade_dls.engine import laplace_preview as lp

D1, D2 = 5e5, 5e1                                  # nm^2/s: two populations 4 decades apart
Q2 = np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4])
FILES = [f"f{i}" for i in range(6)]
BASE = pd.DataFrame({"filename": FILES, "q^2": Q2})
DIALOGS = Path(__file__).resolve().parents[1] / "ade_dls" / "gui" / "dialogs"


def _results(*diffusion, files=FILES, q2=Q2):
    """One result dict per file with tau_k = 1 / (D_k q^2) for every population D_k."""
    return [{"filename": f, **{f"tau_{k + 1}": 1.0 / (d * q) for k, d in enumerate(diffusion)}}
            for f, q in zip(files, q2)]


# ----------------------------------------------------------------------------- table building
def test_preview_gamma_table_looks_up_q2_by_filename_and_drops_unknown_files(capsys):
    results = _results(D1)
    table = lp.preview_gamma_table(results, ["f1", "nope", "f3", "f0", "f2", "f4"], BASE, warn=True)
    assert list(table["q^2"]) == [2e-4, 4e-4, 1e-4, 3e-4, 5e-4]       # 'nope' dropped
    assert "nope not found in basedata" in capsys.readouterr().out
    assert len(table) == 5


def test_preview_gamma_table_length_mismatch_raises():
    with pytest.raises(ValueError):
        lp.preview_gamma_table(_results(D1)[:5], FILES, BASE)


def test_with_decay_rates_adds_gamma_as_inverse_tau():
    table, tau, gamma = lp.with_decay_rates(pd.DataFrame({"tau_1": [0.5, 0.25], "tau_2": [2.0, 4.0], "x": [1, 2]}))
    assert (tau, gamma) == (["tau_1", "tau_2"], ["gamma_1", "gamma_2"])
    assert list(table["gamma_1"]) == [2.0, 4.0] and list(table["gamma_2"]) == [0.5, 0.25]


def test_with_decay_rates_without_tau_columns_is_empty():
    table, tau, gamma = lp.with_decay_rates(pd.DataFrame({"x": [1]}))
    assert tau == [] and gamma == [] and list(table.columns) == ["x"]


@pytest.mark.parametrize("n,color", [(0, "#f44336"), (1, "#4CAF50"), (2, "#2196F3"), (3, "#2196F3"),
                                     (4, "#FF9800"), (9, "#FF9800")])
def test_mode_count_color(n, color):
    assert lp.mode_count_color(n) == color


# ----------------------------------------------------------------------------- NNLS peak counts
def test_peak_count_summary_ignores_none_and_nan_peaks():
    results = [{"tau_1": 1.0, "tau_2": 2.0, "x": 5}, {"tau_1": 1.0, "tau_2": None},
               {"tau_1": 1.0, "tau_2": float("nan"), "tau_3": float("nan")}]
    assert lp.peak_count_summary(results) == (1, 1, 2)
    assert lp.peak_count_summary([]) is None


def test_peak_count_summary_rounds_half_to_even_documents_current_behavior():
    """np.median([2, 3]) = 2.5 and Python's round() gives 2; [1, 2] gives 2."""
    assert lp.peak_count_summary([{"tau_1": 1, "tau_2": 1}, {"tau_1": 1, "tau_2": 1, "tau_3": 1}])[0] == 2
    assert lp.peak_count_summary([{"tau_1": 1}, {"tau_1": 1, "tau_2": 1}])[0] == 2


@pytest.mark.parametrize("summary,text", [((2, 2, 2), "Detected modes: 2"),
                                          ((2, 1, 3), "Detected modes: 2 (range 1–3)")])
def test_peak_count_text(summary, text):
    assert lp.peak_count_text(summary) == text


# ----------------------------------------------------------------------------- Regularized clustering
def test_estimate_modes_by_clustering_finds_one_and_two_populations():
    assert lp.estimate_modes_by_clustering(_results(D1), FILES, BASE) == lp.ModeEstimate('ok', 1)
    assert lp.estimate_modes_by_clustering(_results(D1, D2), FILES, BASE) == lp.ModeEstimate('ok', 2)


def test_estimate_modes_by_clustering_status_values():
    assert lp.estimate_modes_by_clustering(_results(D1), FILES, BASE, use_clustering=False) == (
        lp.ModeEstimate('disabled', None))
    assert lp.estimate_modes_by_clustering(_results(D1)[:1], FILES[:1], BASE) == (
        lp.ModeEstimate('insufficient', None))
    no_tau = [{"filename": f, "x": 1.0} for f in FILES]
    assert lp.estimate_modes_by_clustering(no_tau, FILES, BASE) == lp.ModeEstimate('no_tau', None)


def test_regularized_preview_params_keeps_only_the_fit_parameters():
    params = {"decay_times": np.logspace(-8, 1, 5), "prominence": 0.005, "distance": 1, "alpha": 0.2,
              "normalize": True, "sparsity_penalty": 0.0, "enforce_unimodality": False,
              "use_clustering": True, "distance_threshold": 2.0}
    out = lp.regularized_preview_params(params)
    assert set(out) == {"decay_times", "prominence", "distance", "alpha", "normalize",
                        "sparsity_penalty", "enforce_unimodality", "fit_beta"}
    assert out["fit_beta"] is False and lp.regularized_preview_params({**params, "fit_beta": True})["fit_beta"]
    with pytest.raises(KeyError):
        lp.regularized_preview_params({"alpha": 0.2})


# ----------------------------------------------------------------------------- Γ vs q²
def test_gamma_q2_series_fits_slope_one_and_estimates_D():
    series, n_columns = lp.gamma_q2_series(_results(D1, D2), FILES, BASE)
    assert n_columns == 2 and [s.index for s in series] == [0, 1]
    for s, d in zip(series, (D1, D2)):
        assert s.slope == pytest.approx(1.0, rel=1e-6)
        assert s.intercept == pytest.approx(np.log10(d), rel=1e-6)
        assert s.d_estimate == pytest.approx(d, rel=1e-6)
        assert len(s.q2_fit) == len(s.gamma_fit) == 50
        assert s.q2_fit[0] == pytest.approx(Q2.min()) and s.q2_fit[-1] == pytest.approx(Q2.max())
        np.testing.assert_allclose(s.gamma_fit, d * s.q2_fit, rtol=1e-6)
        np.testing.assert_allclose(s.gamma, d * Q2, rtol=1e-12)


def test_gamma_q2_series_slope_far_from_one_gives_no_D_estimate():
    results = [{"tau_1": 1.0 / q ** 2} for q in Q2]                     # Γ ∝ q^4: slope 2
    series, _ = lp.gamma_q2_series(results, FILES, BASE)
    assert series[0].slope == pytest.approx(2.0, rel=1e-6) and series[0].d_estimate is None
    assert series[0].q2_fit is not None


def test_gamma_q2_series_peak_with_single_point_has_no_fit_and_empty_peaks_are_skipped():
    results = _results(D1)
    for r in results[1:]:
        r["tau_2"] = np.nan
    results[0]["tau_2"] = 0.01
    results[3]["tau_3"] = np.nan                                         # peak 3 never has data
    series, n_columns = lp.gamma_q2_series(results, FILES, BASE)
    assert n_columns == 3 and [s.index for s in series] == [0, 1]       # n_columns counts the empty one
    assert series[1].q2_fit is None and series[1].slope is None and len(series[1].q2) == 1


@pytest.mark.parametrize("results,files", [([], []), ([{"x": 1.0}] * 6, FILES)])
def test_gamma_q2_series_without_usable_data_is_none(results, files):
    assert lp.gamma_q2_series(results, files, BASE) is None


# ----------------------------------------------------------------------------- dialogs
class _Label:
    def __init__(self):
        self.text, self.style = None, None

    def setText(self, text):
        self.text = text

    def setStyleSheet(self, style):
        self.style = style


@pytest.fixture(scope="module")
def dialogs():
    pytest.importorskip("PyQt5.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from ade_dls.gui.dialogs.nnls_dialog import NNLSDialog
    from ade_dls.gui.dialogs.regularized_dialog import RegularizedDialog
    return NNLSDialog, RegularizedDialog


def _fake(params=None):
    return SimpleNamespace(detected_modes_label=_Label(), params=params or {},
                           laplace_analyzer=SimpleNamespace(df_basedata=BASE))


def test_regularized_dialog_mode_label_per_status(dialogs):
    _, Dlg = dialogs
    fake = _fake({"use_clustering": True, "distance_threshold": 2.0, "clustering_strategy": "silhouette_refined"})
    Dlg._estimate_mode_count_from_preview(fake, _results(D1, D2), FILES)
    assert fake.detected_modes_label.text == "Detected modes: 2 modes"
    assert "#2196F3" in fake.detected_modes_label.style
    Dlg._estimate_mode_count_from_preview(fake, _results(D1), FILES)
    assert fake.detected_modes_label.text == "Detected modes: 1 mode"
    assert "#4CAF50" in fake.detected_modes_label.style
    fake.params["use_clustering"] = False
    Dlg._estimate_mode_count_from_preview(fake, _results(D1), FILES)
    assert fake.detected_modes_label.text == "Detected modes: N/A (clustering disabled)"
    assert "gray" in fake.detected_modes_label.style
    fake = _fake()
    Dlg._estimate_mode_count_from_preview(fake, _results(D1)[:1], FILES[:1])
    assert fake.detected_modes_label.text == "Detected modes: N/A (insufficient data)"


def test_nnls_dialog_mode_label_shows_peak_counts(dialogs):
    Dlg, _ = dialogs
    fake = _fake()
    Dlg._estimate_mode_count_from_preview(fake, [{"tau_1": 1.0, "tau_2": 2.0}, {"tau_1": 1.0}], FILES[:2])
    assert fake.detected_modes_label.text == "Detected modes: 2 (range 1–2)"
    Dlg._estimate_mode_count_from_preview(fake, [], [])
    assert fake.detected_modes_label.text == "Detected modes: N/A" and "orange" in fake.detected_modes_label.style


@pytest.mark.parametrize("which", [0, 1])
def test_dialog_gamma_q2_logplot_draws_points_fits_and_D_labels(dialogs, which):
    Dlg = dialogs[which]
    fake = _fake()
    fig = Dlg._create_gamma_q2_logplot(fake, _results(D1, D2), FILES)
    assert fig is not None and fake.gamma_q2_figure is fig
    ax = fig.axes[0]
    assert len(ax.lines) == 2 * 2 + 1                                   # points + fit per peak, + reference line
    assert sorted(t.get_text() for t in ax.texts) == sorted([f"D≈{D1:.2e}", f"D≈{D2:.2e}"])
    assert Dlg._create_gamma_q2_logplot(_fake(), [], []) is None


def test_dialogs_no_longer_build_the_preview_tables_themselves():
    for name in ("nnls_dialog.py", "regularized_dialog.py"):
        src = (DIALOGS / name).read_text(encoding="utf-8")
        assert "np.polyfit" not in src and "matching_rows" not in src and "calculate_decay_rates" not in src
        assert "laplace_preview" in src
