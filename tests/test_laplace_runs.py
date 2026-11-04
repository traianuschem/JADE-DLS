"""Tests for ade_dls.engine.laplace_runs (data preparation + call order of the NNLS / Regularized
runs shared by the GUI and the headless API). A recording stub isolates the call sequence; the
analyzer numerics are tested in test_regularized_optimized.py."""
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ade_dls.engine import laplace_runs

ALV = Path(__file__).resolve().parent / "data" / "alv_mono"
MAIN_WINDOW = Path(__file__).resolve().parents[1] / "ade_dls" / "gui" / "main_window.py"


class _Stub:
    def __init__(self):
        self.calls = []

    def set_weighting(self, flag):
        self.calls.append(("set_weighting", flag))

    def run_nnls(self, params, use_multiprocessing, show_plots):
        self.calls.append(("run_nnls", use_multiprocessing, show_plots))

    def run_regularized(self, params, use_multiprocessing, show_plots):
        self.calls.append(("run_regularized", use_multiprocessing, show_plots))

    def calculate_nnls_diffusion_coefficients(self):
        self.calls.append(("nnls_diffusion",))

    def calculate_regularized_diffusion_coefficients(self, use_clustering, distance_threshold,
                                                     clustering_strategy):
        self.calls.append(("reg_diffusion", use_clustering, distance_threshold, clustering_strategy))


# ----------------------------------------------------------------------------- execute_*
def test_execute_nnls_call_order_and_status_messages():
    an, msgs = _Stub(), []
    laplace_runs.execute_nnls(an, {"use_weighting": True}, False, on_status=msgs.append)
    assert an.calls == [("set_weighting", True), ("run_nnls", False, False), ("nnls_diffusion",)]
    assert msgs == ["Performing NNLS fits...", "Calculating diffusion coefficients..."]


def test_execute_nnls_takes_multiprocessing_from_params_when_not_given():
    an = _Stub()
    laplace_runs.execute_nnls(an, {"use_multiprocessing": True})
    assert ("run_nnls", True, False) in an.calls
    an = _Stub()
    laplace_runs.execute_nnls(an, {"use_multiprocessing": True}, False)      # explicit argument wins
    assert ("run_nnls", False, False) in an.calls


def test_execute_regularized_passes_clustering_settings_with_gold_fallbacks():
    an = _Stub()
    laplace_runs.execute_regularized(an, {"use_clustering": False, "distance_threshold": 3.5,
                                          "clustering_strategy": "x"}, False)
    assert an.calls == [("set_weighting", False), ("run_regularized", False, False),
                        ("reg_diffusion", False, 3.5, "x")]
    an = _Stub()
    laplace_runs.execute_regularized(an, {}, False)
    assert an.calls[-1] == ("reg_diffusion", True, 2.0, "silhouette_refined")


# ----------------------------------------------------------------------------- build_laplace_analyzer
def _series():
    base = pd.DataFrame({"filename": ["a", "b"], "temperature [K]": [298.0, 300.0],
                         "viscosity [cp]": [0.89, 0.91]})
    corr = {k: pd.DataFrame({"t": [1e-6, 1e-5], "g2": [1.5, 1.2]}) for k in ("a", "b")}
    return {"basedata": base, "correlations": corr}


@pytest.mark.parametrize("data", [None, {}, {"correlations": {"a": 1}}, {"basedata": 1}])
def test_build_laplace_analyzer_rejects_incomplete_data(data):
    with pytest.raises(ValueError):
        laplace_runs.build_laplace_analyzer(data)


def test_build_laplace_analyzer_completes_derived_tables_and_uses_preprocess_callback():
    data, seen = _series(), []

    def pre(d):
        seen.append(d is data)
        d["processed_correlations"] = {"a": "pa", "b": "pb"}
        d["weights"] = None

    an = laplace_runs.build_laplace_analyzer(data, preprocess=pre)
    assert seen == [True]
    assert data["df_basedata"] is not data["basedata"]
    pd.testing.assert_frame_equal(data["df_basedata"], data["basedata"])
    assert float(data["c"].iloc[0]) > 0 and float(data["delta_c"].iloc[0]) > 0
    assert an.processed_correlations == {"a": "pa", "b": "pb"}


def test_build_laplace_analyzer_does_not_preprocess_twice():
    data = _series()
    data["processed_correlations"] = {"a": 1, "b": 2}
    called = []
    laplace_runs.build_laplace_analyzer(data, preprocess=lambda d: called.append(1))
    assert called == []


# ----------------------------------------------------------------------------- headless / GUI use
@pytest.mark.skipif(not ALV.exists(), reason="tests/data/alv_mono missing")
def test_headless_laplace_analyzer_equals_engine_builder():
    from ade_dls import headless
    d1, d2 = headless.load_series(str(ALV)), headless.load_series(str(ALV))
    a1 = headless.laplace_analyzer(d1)
    a2 = laplace_runs.build_laplace_analyzer(d2)
    assert set(a1.processed_correlations) == set(a2.processed_correlations)
    assert float(d1["c"].iloc[0]) == float(d2["c"].iloc[0])


def test_main_window_uses_the_shared_run_sequence():
    """Structure check (no Qt needed): the window must not re-implement the call order."""
    src = MAIN_WINDOW.read_text(encoding="utf-8")
    assert "build_laplace_analyzer(" in src and src.count("build_laplace_analyzer(") == 2
    assert "execute_nnls(self.laplace_analyzer" in src
    assert "execute_regularized(self.laplace_analyzer" in src
    assert not re.search(r"self\.laplace_analyzer\.(run_nnls|run_regularized)\(", src)
    assert "LaplaceAnalyzer(" not in src
