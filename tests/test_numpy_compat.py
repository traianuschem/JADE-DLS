"""numpy 1.x compatibility: np.trapezoid exists only from numpy 2.0 (pyproject declares numpy>=1.22).

A user on numpy 1.x got "module 'numpy' has no attribute 'trapezoid'" in the NNLS / Regularized analysis.
The tests simulate numpy 1.x (no np.trapezoid, np.trapz present) and check that the area integration still works
and gives the same numbers.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ade_dls import headless
from ade_dls.analysis import clustering  # noqa: F401  (import sklearn now: libraries probe np.trapezoid while importing)
from ade_dls.analysis import regularized_optimized as ro
from ade_dls.utils.compat import trapezoid

LAG = np.logspace(-7, 0, 71)
DT = np.logspace(-6, -1, 26)


@pytest.fixture
def numpy1(monkeypatch):
    real = getattr(np, "trapezoid", None) or np.trapz      # on a real numpy 1.x this simply stays as it is
    monkeypatch.delattr(np, "trapezoid", raising=False)
    monkeypatch.setattr(np, "trapz", real, raising=False)
    yield real
    plt.close("all")
    ro.create_tikhonov_matrix_cached.cache_clear()


def test_trapezoid_helper_matches_analytic_integral_without_np_trapezoid(numpy1):
    x = np.linspace(0.0, 2.0, 2001)
    assert trapezoid(x ** 2, x) == pytest.approx(8.0 / 3.0, rel=1e-6)


@pytest.mark.skipif(not hasattr(np, "trapezoid"), reason="numpy 1.x has no np.trapezoid")
def test_trapezoid_helper_equals_numpy_result_with_numpy2():
    x = np.log(np.logspace(-6, -1, 26))
    y = np.exp(-0.5 * ((x - x.mean()) / 1.2) ** 2)
    assert trapezoid(y, x) == np.trapezoid(y, x)


def test_nnls_optimized_runs_without_np_trapezoid(numpy1):
    import pandas as pd
    f = np.zeros(DT.size)
    f[10] = 1.0
    y = (ro.create_exponential_matrix(LAG, DT) @ f) ** 2
    df = pd.DataFrame({"t [s]": LAG, "g(2)-1": y})
    res = ro.nnls_optimized(df, "x", {"decay_times": DT, "prominence": 0.01, "distance": 1})
    assert res is not None


def test_peak_statistics_run_without_np_trapezoid(numpy1):
    from scipy.signal import find_peaks
    x = np.log(DT)
    f = np.exp(-0.5 * ((x - x[12]) / 0.6) ** 2)
    peaks, props = find_peaks(f, prominence=0.01)
    stats, area_pct, sum_pct = ro._compute_peak_statistics(peaks, props, f, DT)
    assert len(area_pct) == len(peaks) == 1 and area_pct[0] == pytest.approx(100.0)


# ----------------------------------------------------------------------------- headless end to end
@pytest.fixture(scope="module")
def two_files(tmp_path_factory):
    from ade_dls.benchmark import runner
    data, _ = runner.simulate_series("mono_pdi0.1", 1.0, "correlated", 0,
                                     str(tmp_path_factory.mktemp("two")), angles=(30, 90), runs=1)
    return data


SMALL_GRID = {"decay_times": np.logspace(-6, -1, 26), "use_weighting": False}


def test_regularized_runs_end_to_end_without_np_trapezoid(numpy1, two_files):
    analyzer, _ = headless.run_regularized(two_files, SMALL_GRID)
    assert len(analyzer.regularized_results) == 2 and "filename" in analyzer.regularized_results


def test_regularized_failing_for_every_file_reports_the_cause_not_a_filename_keyerror(monkeypatch, two_files):
    """Per-file errors are caught in the loop; if all files fail the user saw only "'filename'" (KeyError of the merge)."""

    def boom(*a, **k):
        raise AttributeError("module 'numpy' has no attribute 'trapezoid'")
    monkeypatch.setattr(ro, "trapezoid", boom)
    with pytest.raises(RuntimeError, match="trapezoid"):
        headless.run_regularized(two_files, SMALL_GRID)
