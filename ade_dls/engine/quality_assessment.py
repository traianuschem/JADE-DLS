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
