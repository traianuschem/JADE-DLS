"""Compatibility helpers for supported numpy versions (numpy >= 1.22, see pyproject.toml)."""
import numpy as np


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Trapezoidal rule: ``np.trapezoid`` (numpy >= 2.0) or ``np.trapz`` (numpy 1.x).

    Resolved at call time. Identical numbers in both versions (``np.trapz`` was renamed, not changed).
    """
    func = getattr(np, "trapezoid", None)
    if func is None:
        func = np.trapz
    return func(y, x, dx, axis)
