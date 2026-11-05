"""Tests for ade_dls.engine.quality_assessment: intercept assessment and the thresholds of the
'Fit Quality Assessment' tables shown by the analysis view. Boundaries are hand-written from the
documented ranges (the table text: > 0.95 / 0.90-0.95 / < 0.90 etc.)."""
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ade_dls.engine import quality_assessment as qa

_APP = None
VIEW = Path(__file__).resolve().parents[1] / "ade_dls" / "gui" / "widgets" / "analysis_view.py"
EXC, ACC, POOR = qa.EXCELLENT, qa.ACCEPTABLE, qa.POOR


# ----------------------------------------------------------------------------- intercept
@pytest.mark.parametrize("icpt,se,text,color", [
    (None, 1.0, "N/A", None), (1.0, None, "N/A", None), ("x", 1.0, "N/A", None),
    (float("nan"), 1.0, "N/A", None), (1.0, 0.0, "N/A (no SE)", None),
    (1.9, 1.0, "Intercept ≈ 0 (fit through origin plausible)", "#90EE90"),
    (-2.0, 1.0, "Intercept significantly ≠ 0 (keep free intercept)", "#FFB6C1"),
])
def test_intercept_assessment(icpt, se, text, color):
    assert qa.intercept_assessment(icpt, se) == (text, color)


# ----------------------------------------------------------------------------- assess_metric
@pytest.mark.parametrize("value,expected", [
    (0.96, EXC), (0.95, EXC), (0.9499, ACC), (0.90, ACC), (0.8999, POOR), (float("nan"), POOR),
])
def test_assess_r2_boundaries_are_inclusive_at_the_lower_limit(value, expected):
    assert qa.assess_r2(value) == expected


@pytest.mark.parametrize("value,expected", [(100, EXC), (99.9, ACC), (10, ACC), (9.99, POOR)])
def test_assess_f_statistic_boundaries(value, expected):
    assert qa.assess_f_statistic(value) == expected


@pytest.mark.parametrize("p,expected", [(0.0, EXC), (0.00099, EXC), (0.001, ACC), (0.0499, ACC),
                                        (0.05, POOR), (0.5, POOR)])
def test_assess_p_value_boundaries_are_exclusive_upper_limits(p, expected):
    assert qa.assess_p_value(p) == expected


def test_assess_metric_with_upper_bounds_and_closed_ranges():
    assert qa.assess_metric(1, good_max=2, acceptable_max=5) == EXC
    assert qa.assess_metric(3, good_max=2, acceptable_max=5) == ACC
    assert qa.assess_metric(6, good_max=2, acceptable_max=5) == POOR
    # both bounds given: inside = excellent; outside falls through to the acceptable range
    assert qa.assess_metric(1.5, good_min=1, good_max=2, acceptable_min=0, acceptable_max=3) == EXC
    assert qa.assess_metric(2.5, good_min=1, good_max=2, acceptable_min=0, acceptable_max=3) == ACC
    assert qa.assess_metric(4, good_min=1, good_max=2, acceptable_min=0, acceptable_max=3) == POOR
    assert qa.assess_metric(1) == POOR                        # no ranges at all


@pytest.mark.parametrize("r2,expected", [
    (0.995, ("✓ Excellent", "green")), (0.99, ("○ Good", "orange")), (0.96, ("○ Good", "orange")),
    (0.95, ("✗ Check", "red")), (0.5, ("✗ Check", "red")),
])
def test_assess_order_r2_uses_strict_thresholds(r2, expected):
    assert qa.assess_order_r2(r2) == expected


# ----------------------------------------------------------------------------- overall
@pytest.mark.parametrize("args,expected", [
    ((0.96, 0.96, 101, 0.0001), ("EXCELLENT", 4)),
    ((0.96, 0.96, 101, 0.01), ("EXCELLENT", 3)),
    ((0.96, 0.96, 50, 0.01), ("ACCEPTABLE", 2)),
    ((0.96, 0.90, 50, 0.01), ("POOR", 1)),
    ((0.95, 0.95, 100, 0.001), ("POOR", 0)),             # every criterion is strict
    ((float("nan"),) * 4, ("POOR", 0)),
])
def test_overall_fit_quality_counts_met_criteria(args, expected):
    assert qa.overall_fit_quality(*args) == expected


def test_threshold_constants_match_the_texts_shown_in_the_view():
    src = VIEW.read_text(encoding="utf-8")
    assert (qa.R2_GOOD, qa.R2_ACCEPTABLE, qa.F_GOOD, qa.F_ACCEPTABLE, qa.P_GOOD, qa.P_ACCEPTABLE) == (
        0.95, 0.90, 100, 10, 0.001, 0.05)
    for text in ("> 0.95 (Excellent)<br>0.90-0.95 (Acceptable)<br>< 0.90 (Poor)",
                 "> 100 (Excellent)<br>10-100 (Acceptable)<br>< 10 (Poor)",
                 "< 0.001 (Excellent)<br>0.001-0.05 (Acceptable)<br>> 0.05 (Poor)"):
        assert text in src


def test_analysis_view_does_not_keep_its_own_threshold_logic():
    src = VIEW.read_text(encoding="utf-8")
    assert "def assess_metric" not in src and "overall_good" not in src
    assert "assess_order_r2(r2)" in src and "overall_fit_quality(" in src


# ----------------------------------------------------------------------------- rendered detail (Qt)
@pytest.mark.parametrize("stats,level,r2_label", [
    (dict(rsquared=0.99, rsquared_adj=0.98, fvalue=500.0, f_pvalue=1e-5), "EXCELLENT", "✓ Excellent"),
    (dict(rsquared=0.92, rsquared_adj=0.91, fvalue=50.0, f_pvalue=0.01), "POOR", "○ Acceptable"),
])
def test_analysis_view_renders_the_assessment_from_the_engine(stats, level, r2_label):
    pytest.importorskip("PyQt5.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    global _APP
    _APP = QApplication.instance() or QApplication([])      # module-level: widgets crash without a live app
    from ade_dls.gui.widgets.analysis_view import AnalysisView
    view = AnalysisView(SimpleNamespace())
    results = pd.DataFrame({"Rh [nm]": [50.0], "Rh error [nm]": [1.0], "D [m²/s]": [1e-11],
                            "D error [m²/s]": [1e-13], "R_squared": [stats["rsquared"]],
                            "Fit": ["x"], "Residuals": ["ok"]})
    reg = dict(stats, summary="s", params={"const": 1.0, "x1": 2.0}, aic=1.0, bic=2.0)
    view.display_cumulant_results("Method B", results, regression_stats=reg, switch_tab=False)
    html = view.results_table.item(0, 0).data(Qt.UserRole)
    assert f"Overall Fit Quality: {level}" in html
    assert r2_label in html
