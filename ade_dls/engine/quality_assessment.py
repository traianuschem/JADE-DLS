"""
Small shared quality-assessment helpers for analysis result displays.

Currently covers the Γ-vs-q² intercept: whether it is statistically
indistinguishable from zero (supporting a fit-through-origin re-run) or
significantly non-zero (a free intercept should be kept).
"""

import math
from typing import Optional, Tuple

# z-score threshold for "not significantly different from zero" (~95% CI),
# consistent with the p<0.05 convention already used elsewhere (e.g. Jarque-Bera/
# Omnibus normality checks in cumulant_analyzer.py).
_INTERCEPT_Z_THRESHOLD = 2.0


def intercept_assessment(intercept: Optional[float],
                         intercept_se: Optional[float]) -> Tuple[str, Optional[str]]:
    """
    Assess whether a Γ-vs-q² regression intercept is significantly different from zero.

    Returns (status_text, color_hex) where color_hex is a light background color
    suitable for QTableWidgetItem.setBackground() / QColor(...), or None when no
    assessment can be made.
    """
    if intercept is None or intercept_se is None:
        return "N/A", None
    try:
        intercept = float(intercept)
        intercept_se = float(intercept_se)
    except (TypeError, ValueError):
        return "N/A", None

    if math.isnan(intercept) or math.isnan(intercept_se):
        return "N/A", None
    if intercept_se <= 0:
        return "N/A (no SE)", None

    z = abs(intercept) / intercept_se
    if z < _INTERCEPT_Z_THRESHOLD:
        return "Intercept ≈ 0 (fit through origin plausible)", "#90EE90"  # light green
    return "Intercept significantly ≠ 0 (keep free intercept)", "#FFB6C1"  # light red


# ----------------------------------------------------------------------------- regression fit quality
# Thresholds of the "Fit Quality Assessment" tables in the analysis view (Γ vs q² regression).
R2_GOOD = 0.95                  # R^2 / adjusted R^2: excellent above, acceptable from R2_ACCEPTABLE
R2_ACCEPTABLE = 0.90
F_GOOD = 100                    # F statistic
F_ACCEPTABLE = 10
P_GOOD = 0.001                  # p-value of the F test (lower is better)
P_ACCEPTABLE = 0.05
ORDER_R2_EXCELLENT = 0.99       # per-order table of Method A (1st/2nd/3rd order cumulant)
ORDER_R2_GOOD = 0.95

EXCELLENT = ("✓ Excellent", "green")
ACCEPTABLE = ("○ Acceptable", "orange")
POOR = ("✗ Poor", "red")


def assess_metric(value, good_min=None, good_max=None, acceptable_min=None,
                  acceptable_max=None) -> Tuple[str, str]:
    """Assess ``value`` against optimal ranges; returns ``(label, color)``.

    "Excellent" if it lies in the good range (bounds inclusive; an open side is ignored),
    else "Acceptable" if it lies in the acceptable range, else "Poor" (also for NaN)."""
    if good_min is not None and good_max is not None:
        if good_min <= value <= good_max:
            return EXCELLENT
    elif good_min is not None:
        if value >= good_min:
            return EXCELLENT
    elif good_max is not None:
        if value <= good_max:
            return EXCELLENT

    if acceptable_min is not None and acceptable_max is not None:
        if acceptable_min <= value <= acceptable_max:
            return ACCEPTABLE
    elif acceptable_min is not None:
        if value >= acceptable_min:
            return ACCEPTABLE
    elif acceptable_max is not None:
        if value <= acceptable_max:
            return ACCEPTABLE

    return POOR


def assess_r2(r2) -> Tuple[str, str]:
    """R^2 or adjusted R^2 of the Γ-vs-q² regression (Methods B/C)."""
    return assess_metric(r2, good_min=R2_GOOD, acceptable_min=R2_ACCEPTABLE)


def assess_f_statistic(fvalue) -> Tuple[str, str]:
    return assess_metric(fvalue, good_min=F_GOOD, acceptable_min=F_ACCEPTABLE)


def assess_p_value(p) -> Tuple[str, str]:
    """p-value of the F test: below ``P_GOOD`` excellent, below ``P_ACCEPTABLE`` acceptable."""
    if p < P_GOOD:
        return EXCELLENT
    if p < P_ACCEPTABLE:
        return ACCEPTABLE
    return POOR


def assess_order_r2(r2) -> Tuple[str, str]:
    """R^2 of one cumulant order in the Method A table: > 0.99 Excellent, > 0.95 Good, else Check."""
    if r2 > ORDER_R2_EXCELLENT:
        return "✓ Excellent", "green"
    if r2 > ORDER_R2_GOOD:
        return "○ Good", "orange"
    return "✗ Check", "red"


def overall_fit_quality(r2, r2_adj, fvalue, f_pvalue) -> Tuple[str, int]:
    """Overall verdict of the regression: ``(level, n_good)`` with level 'EXCELLENT' (>= 3 of the 4
    criteria R^2 > 0.95, adj. R^2 > 0.95, F > 100, p < 0.001 met), 'ACCEPTABLE' (2) or 'POOR'."""
    n_good = sum([r2 > R2_GOOD, r2_adj > R2_GOOD, fvalue > F_GOOD, f_pvalue < P_GOOD])
    if n_good >= 3:
        return 'EXCELLENT', n_good
    if n_good >= 2:
        return 'ACCEPTABLE', n_good
    return 'POOR', n_good
