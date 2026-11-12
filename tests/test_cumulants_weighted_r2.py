# -*- coding: utf-8 -*-
"""JADE 3.1 adoption: weighted R^2 helper, Method B weighted standard error / R^2,
Method C weighted R^2 and weighted plot panels.

Written test-first: tests that need the new behavior fail until the production
code is updated.
"""
import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import linregress  # noqa: E402

from ade_dls.analysis import cumulants as cu  # noqa: E402
from ade_dls.analysis import cumulants_C as cc  # noqa: E402
from ade_dls.analysis import weighting as wt  # noqa: E402
from ade_dls.engine import cumulant_plotting as cp  # noqa: E402

TOL = dict(rel=1e-6, abs=1e-12)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# ---------------------------------------------------------------- fit_r_squared
Y = np.array([1.0, 2.0, 4.0, 3.0, 7.0])
YF = np.array([1.2, 2.5, 3.1, 3.9, 6.0])
WW = np.array([1.0, 2.0, 0.5, 3.0, 1.5])


def test_oracle_fit_r_squared_unweighted_equals_one_minus_ssres_over_sstot():
    r2 = wt.fit_r_squared(Y, YF)
    assert r2 == pytest.approx(1 - np.sum((Y - YF) ** 2) / np.sum((Y - Y.mean()) ** 2), **TOL)
    assert wt.fit_r_squared(Y, YF, np.ones(5)) == pytest.approx(r2, **TOL)


def test_oracle_fit_r_squared_weighted_hand_calculation():
    ym = np.sum(WW * Y) / np.sum(WW)
    expected = 1 - np.sum(WW * (Y - YF) ** 2) / np.sum(WW * (Y - ym) ** 2)
    assert wt.fit_r_squared(Y, YF, WW) == pytest.approx(expected, **TOL)
    # y=[0,1,2], yfit=[0,0,2], w=[1,2,1]: ybar_w=1, sstot=2, ssres=2 -> 0
    assert wt.fit_r_squared([0.0, 1.0, 2.0], [0.0, 0.0, 2.0], [1.0, 2.0, 1.0]) == pytest.approx(0.0, abs=1e-12)
    # y=[0,1,3], yfit=[0,1,2], w=[1,1,2]: ybar_w=7/4, sstot=6.75, ssres=2
    assert wt.fit_r_squared([0.0, 1.0, 3.0], [0.0, 1.0, 2.0], [1.0, 1.0, 2.0]) == pytest.approx(1 - 2 / 6.75, **TOL)


def test_meta_fit_r_squared_scale_invariant_in_weights():
    a = wt.fit_r_squared(Y, YF, WW)
    assert wt.fit_r_squared(Y, YF, 7.3 * WW) == pytest.approx(a, **TOL)


def test_edge_fit_r_squared_constant_y_returns_zero():
    assert wt.fit_r_squared([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) == 0.0
    assert wt.fit_r_squared([2.0, 2.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_edge_fit_r_squared_returns_python_float():
    assert type(wt.fit_r_squared(Y, YF)) is float
    assert type(wt.fit_r_squared(Y, YF, WW)) is float
    assert type(wt.fit_r_squared([2.0, 2.0], [1.0, 2.0])) is float


# ---------------------------------------------------------------- Method B
def _wdf(weights=True, n=8, seed=20261112):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 7e-4, n)
    g = 0.8 * np.exp(-2 * 1500.0 * t) + rng.normal(0, 3e-3, n)
    d = pd.DataFrame({"t [s]": t, "g(2)-1": g})
    d["g(2)_mod"] = 0.5 * np.log(g)
    if weights:
        d["weight"] = rng.uniform(0.5, 3.0, n)
    return d


def _wls_reference(d, lim):
    t = d["t [s]"].to_numpy()
    y = d["g(2)_mod"].to_numpy()
    w = d["weight"].to_numpy() * (2 * d["g(2)-1"].to_numpy()) ** 2
    w = w / w.mean()
    m = (t >= lim[0]) & (t <= lim[1])
    x, yy, ww = t[m], y[m], w[m]
    X = np.column_stack([x, np.ones_like(x)])
    A = X.T @ (ww[:, None] * X)
    slope, icpt = np.linalg.solve(A, X.T @ (ww * yy))
    res = yy - (slope * x + icpt)
    s2 = np.sum(ww * res ** 2) / (len(x) - 2)
    se = np.sqrt(s2 * np.linalg.inv(A)[0, 0])
    ym = np.average(yy, weights=ww)
    r2 = 1 - np.sum(ww * res ** 2) / np.sum(ww * (yy - ym) ** 2)
    return slope, se, r2, np.sqrt(ww) * res


FUNCS = {
    "analysis": lambda d, lim: cu.plot_processed_correlations(d, lim),
    "engine": lambda d, lim: cp.plot_processed_correlations_no_show(d, lim)[0],
}


@pytest.mark.parametrize("which", ["analysis", "engine"])
def test_oracle_method_B_weighted_gamma_error_and_r2(which):
    """JADE 3.1: weighted Gamma_error = sqrt(s2*Sw/denom), finite relative error, weighted R^2."""
    d = _wdf()
    lim = (0.0, 6e-4)
    r = FUNCS[which]({"w": d}, lim).iloc[0]
    slope, se, r2, _ = _wls_reference(d, lim)
    assert r["Gamma"] == pytest.approx(-slope, **TOL)
    assert r["Gamma_error"] == pytest.approx(se, **TOL)
    assert r["Gamma_relative_error"] == pytest.approx(se / abs(slope) * 100, **TOL)
    assert np.isfinite(r["Gamma_relative_error"])
    assert r["R_squared"] == pytest.approx(r2, **TOL)


@pytest.mark.parametrize("which", ["analysis", "engine"])
def test_meta_method_B_constant_effective_weights_gamma_error_equals_linregress(which):
    d = _wdf(weights=False)
    dw = d.copy()
    dw["weight"] = 1.0 / d["g(2)-1"] ** 2   # w * (2 g)^2 = const
    lim = (0.0, 6e-4)
    a = FUNCS[which]({"w": d}, lim).iloc[0]
    b = FUNCS[which]({"w": dw}, lim).iloc[0]
    m = (d["t [s]"] <= lim[1]).to_numpy()
    lr = linregress(d["t [s]"][m], d["g(2)_mod"][m])
    assert b["Gamma_error"] == pytest.approx(lr.stderr, rel=1e-6)
    assert b["Gamma_error"] == pytest.approx(a["Gamma_error"], rel=1e-6)
    assert b["R_squared"] == pytest.approx(a["R_squared"], rel=1e-6)


@pytest.mark.parametrize("which", ["analysis", "engine"])
def test_meta_method_B_weighted_error_invariant_under_weight_scale(which):
    d = _wdf()
    d7 = d.copy()
    d7["weight"] *= 7.0
    a = FUNCS[which]({"w": d}, (0, 6e-4)).iloc[0]
    b = FUNCS[which]({"w": d7}, (0, 6e-4)).iloc[0]
    assert b["Gamma_error"] == pytest.approx(a["Gamma_error"], rel=1e-6)


@pytest.mark.parametrize("which", ["analysis", "engine"])
def test_char_method_B_unweighted_r2_and_error_documents_current_behavior(which):
    """Documents current behavior: unweighted path is plain linregress."""
    d = _wdf(weights=False)
    lim = (0.0, 6e-4)
    r = FUNCS[which]({"u": d}, lim).iloc[0]
    m = (d["t [s]"] <= lim[1]).to_numpy()
    lr = linregress(d["t [s]"][m], d["g(2)_mod"][m])
    assert r["Gamma_error"] == pytest.approx(lr.stderr, rel=1e-6)
    assert r["R_squared"] == pytest.approx(lr.rvalue ** 2, rel=1e-6)


# ---------------------------------------------------------------- Method C
TAU = np.logspace(-6, 0, 61)
_RNG = np.random.default_rng(20261112)
YC = 1e-3 + 0.8 * np.exp(-2 * 1e3 * TAU) + _RNG.normal(0, 1e-3, TAU.size)
WC = np.linspace(1.0, 3.0, TAU.size)
GUESS = [0.7, 1300.0, 0, 0, 0, 0.0]


def fit_function1(x, a, b, f):
    return f + a * np.exp(-2 * b * x)


def _dfC(weighted=True):
    d = pd.DataFrame({"t [s]": TAU, "g(2)-1": YC})
    if weighted:
        d["weight"] = WC
    return d


def test_char_method_C_weighted_r2_is_fit_r_squared_documents_current_behavior():
    """Documents current behavior (weighted R2 already in ADE 3.0): equals fit_r_squared(y, fit, sigma**-2)."""
    res = cc.plot_processed_correlations_iterative(
        {"c": _dfC(True)}, fit_function1, (1e-9, 10.0), GUESS, show_plots=False).iloc[0]
    fit = fit_function1(TAU, res["best_a"], res["best_b"], res["best_f"])
    sigma = 1.0 / np.sqrt(WC / WC.mean())
    expected = wt.fit_r_squared(YC, fit, sigma ** -2)
    assert res["best_R-squared"] == pytest.approx(expected, **TOL)
    assert "best_RMSE" in res.index and "best_AIC" in res.index   # ADE keeps RMSE/AIC


# ---------------------------------------------------------------- plot panels
def _labels(fig):
    _, ax2, ax3 = fig.axes
    return ax2.get_xlabel(), ax2.get_ylabel(), ax3.get_xlabel(), ax3.get_ylabel()


@pytest.mark.parametrize("weighted,ylab", [(True, "weighted residuals"), (False, "residuals")])
def test_oracle_engine_method_B_panel_labels(weighted, ylab):
    _, plots = cp.plot_processed_correlations_no_show({"w": _wdf(weights=weighted)}, (0.0, 6e-4))
    fig, _ = plots["w"]
    assert _labels(fig) == ("lag-time index", ylab, "theoretical quantiles", "sample quantiles")


def test_oracle_engine_method_B_weighted_residual_panel_is_sqrt_w_times_raw():
    d = _wdf()
    _, plots = cp.plot_processed_correlations_no_show({"w": d}, (0.0, 6e-4))
    fig, _ = plots["w"]
    ydata = np.asarray(fig.axes[1].lines[0].get_ydata(), float)
    expected = _wls_reference(d, (0.0, 6e-4))[3]
    np.testing.assert_allclose(ydata, expected, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("weighted,ylab", [(True, "weighted residuals"), (False, "residuals")])
def test_oracle_engine_method_C_panel_labels(weighted, ylab):
    _, plots = cp.plot_processed_correlations_iterative_no_show(
        {"c": _dfC(weighted)}, fit_function1, (1e-9, 10.0), GUESS)
    fig, _ = plots["c"]
    assert _labels(fig) == ("lag-time index", ylab, "theoretical quantiles", "sample quantiles")
