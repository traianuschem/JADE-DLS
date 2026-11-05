"""Tests for ade_dls.engine.cumulant_runs (analyzer construction and run order shared by the main
window and the headless API). A recording stub isolates the call sequence; the Method A-D numerics
are tested in test_cumulants*.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import pytest

from ade_dls.engine import cumulant_runs

ALV = Path(__file__).resolve().parent / "data" / "alv_mono"
MAIN_WINDOW = Path(__file__).resolve().parents[1] / "ade_dls" / "gui" / "main_window.py"


class _Stub:
    use_weighting = "unset"

    def __init__(self):
        self.calls = []

    def run_method_a(self, q_range=None, fit_through_origin=False):
        self.calls.append(("a", q_range, fit_through_origin, self.use_weighting))
        return "A"

    def run_method_b(self, fit_limits, q_range=None, fit_through_origin=False):
        self.calls.append(("b", fit_limits, q_range, fit_through_origin, self.use_weighting))
        return "B"

    def run_method_c(self, params, q_range=None):
        self.calls.append(("c", params, q_range, self.use_weighting))
        return "C"

    def run_method_d(self, params, q_range=None):
        self.calls.append(("d", params, q_range, self.use_weighting))
        return "D"


def test_execute_method_a_passes_arguments_and_leaves_weighting_alone():
    an = _Stub()
    assert cumulant_runs.execute_method_a(an, (1.0, 2.0), True) == "A"
    assert an.calls == [("a", (1.0, 2.0), True, "unset")]


def test_execute_method_b_sets_weighting_before_the_run():
    an = _Stub()
    assert cumulant_runs.execute_method_b(an, (1e-6, 1e-2), (1.0, 2.0), True, use_weighting=True) == "B"
    assert an.calls == [("b", (1e-6, 1e-2), (1.0, 2.0), True, True)]
    cumulant_runs.execute_method_b(an, (1e-6, 1e-2))
    assert an.calls[-1] == ("b", (1e-6, 1e-2), None, False, False)


@pytest.mark.parametrize("name,tag", [("execute_method_c", "c"), ("execute_method_d", "d")])
def test_execute_method_c_and_d_set_weighting_before_the_run(name, tag):
    an = _Stub()
    out = getattr(cumulant_runs, name)(an, {"p": 1}, (1.0, 2.0), use_weighting=True)
    assert out == tag.upper() and an.calls == [(tag, {"p": 1}, (1.0, 2.0), True)]


@pytest.mark.parametrize("data", [None, {}, {"basedata": 1}])
def test_build_cumulant_analyzer_requires_correlations(data):
    with pytest.raises(ValueError, match="correlations"):
        cumulant_runs.build_cumulant_analyzer(data)


def test_build_cumulant_analyzer_requires_data_folder():
    with pytest.raises(ValueError, match="data_folder"):
        cumulant_runs.build_cumulant_analyzer({"correlations": {"a": 1}})


@pytest.mark.skipif(not ALV.exists(), reason="tests/data/alv_mono missing")
def test_build_cumulant_analyzer_reports_status_and_prepares_basedata():
    from ade_dls import headless
    data = headless.load_series(str(ALV))
    messages = []
    an = cumulant_runs.build_cumulant_analyzer(data, on_status=messages.append)
    assert messages == ["Preparing basedata..."]
    assert an.df_basedata is not None and len(an.df_basedata) == 6


@pytest.mark.skipif(not ALV.exists(), reason="tests/data/alv_mono missing")
def test_headless_run_cumulant_b_equals_the_shared_sequence():
    from ade_dls import headless
    data = headless.load_series(str(ALV))
    _, via_headless = headless.run_cumulant_b(data, (1e-6, 1e-2), use_weighting=False)
    direct = cumulant_runs.execute_method_b(cumulant_runs.build_cumulant_analyzer(data), (1e-6, 1e-2))
    pd.testing.assert_frame_equal(via_headless, direct)


def test_main_window_uses_the_shared_cumulant_sequence():
    """Structure check (no Qt needed): the window must not re-implement analyzer set-up / run order."""
    src = MAIN_WINDOW.read_text(encoding="utf-8")
    assert "build_cumulant_analyzer(working_data" in src
    for letter in "abcd":
        assert f"execute_method_{letter}(analyzer" in src
    assert "analyzer.run_method_" not in src and "CumulantAnalyzer(" not in src
    assert "analyzer.use_weighting" not in src
