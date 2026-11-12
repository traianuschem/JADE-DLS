# -*- coding: utf-8 -*-
"""Tests for ade_dls/analysis/cumulants_D.py (spec: _dev/notes/test_spec_cumulants_D.md).

Categories: test_oracle_* (closed-form expectations), test_meta_* (metamorphic),
test_edge_* (edge cases), test_char_* (characterization: documents current
behavior, not desired behavior), test_finding_* (xfail strict, known findings).

[MOCK] tests replace ``cd.curve_fit`` (ScriptedCurveFit) or ``cd.fit_cumulant_D``
(FakeFit) via monkeypatch to steer the control flow deterministically;
[REAL] tests use the real optimizer (rel 1e-3) or pure functions.
plt.show / input are never called; every call uses plot=False.
"""
import inspect

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import scipy.optimize  # noqa: E402

from ade_dls.analysis import cumulants_D as cd  # noqa: E402
from ade_dls.gui.analysis.cumulant_analyzer import _fit_single_method_d  # noqa: E402

DET = dict(rel=1e-6, abs=1e-12)
OPT = dict(rel=1e-3)
STAT = dict(rel=5e-2)

IGNORE_RT = pytest.mark.filterwarnings("ignore::RuntimeWarning")
SEED = 20261103


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# ----------------------------------------------------------------------------
# helpers (spec 0.3)
# ----------------------------------------------------------------------------
class ScriptedCurveFit:
    """H1: replaces cd.curve_fit; script maps n -> popt list | Exception | 'echo'."""

    def __init__(self, script, strict=True):
        self.script = script
        self.strict = strict
        self.calls = []

    def __call__(self, f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False,
                 bounds=(-np.inf, np.inf), method=None, maxfev=None, **kw):
        n = len(bounds[0]) - 1
        self.calls.append(dict(
            n=n, p0=[float(v) for v in p0], lb=list(bounds[0]), ub=list(bounds[1]),
            sigma=None if sigma is None else np.array(sigma, float),
            absolute_sigma=absolute_sigma, method=method, maxfev=maxfev, f=f))
        if self.strict and len(p0) != len(bounds[0]):
            raise ValueError("Inconsistent shapes between bounds and `x0`.")
        item = self.script.get(n, "echo")
        if isinstance(item, Exception):
            raise item
        popt = np.array(p0 if (isinstance(item, str) and item == "echo") else item, float)
        return popt, np.full((popt.size, popt.size), np.nan)

    @property
    def p0s(self):
        return [c["p0"] for c in self.calls]


def scripted(monkeypatch, script, strict=True):
    s = ScriptedCurveFit(script, strict=strict)
    monkeypatch.setattr(cd, "curve_fit", s)
    return s


class SpyCurveFit:
    """H3: counts calls and delegates to the real scipy curve_fit."""

    def __init__(self):
        self.ns = []

    def __call__(self, f, xdata, ydata, *a, **kw):
        self.ns.append(len(kw["bounds"][0]) - 1)
        return scipy.optimize.curve_fit(f, xdata, ydata, *a, **kw)


def spied(monkeypatch):
    s = SpyCurveFit()
    monkeypatch.setattr(cd, "curve_fit", s)
    return s


def res(gammas, beta=1.0, rss=0.01, n_modes=None, sel=None):
    g = np.sort(np.asarray(gammas, float))
    return {"n_modes": len(g) if n_modes is None else n_modes, "beta": beta, "gammas": g,
            "residual_ss": rss, "selection_ss": rss if sel is None else sel,
            "fit_ss": rss if sel is None else sel,
            "g1_fit": np.ones(4), "g2_fit": np.ones(4) * beta, "convergence": "success"}


class FakeFit:
    """H4: replaces cd.fit_cumulant_D; returns results in call order."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, x_data, y_data, n_max=25, beta_initial=1.0, gamma_initial=None,
                 n_start=1, weights=None):
        self.calls.append(dict(x=np.array(x_data), y=np.array(y_data), n_max=n_max,
                               n_start=n_start,
                               weights=None if weights is None else np.array(weights)))
        r = self.results[len(self.calls) - 1]
        if isinstance(r, Exception):
            raise r
        return r


def faked(monkeypatch, results):
    f = FakeFit(results)
    monkeypatch.setattr(cd, "fit_cumulant_D", f)
    return f


def spread(gammas):
    lg = np.log10(np.asarray(gammas, float))
    return (lg.max() - lg.min()) / abs(lg.mean())


XR = np.logspace(-7, 0, 80)
X0 = np.zeros(4)
Y0 = np.zeros(4)
Y1 = np.ones(4)
YW = np.array([1.0, 0.0, 0.0, 0.0])
W = np.array([4.0, 1.0, 1.0, 1.0])

R1 = cd.dirac_sum_g2(XR, 0.9, 1000.0)
R2 = cd.dirac_sum_g2(XR, 0.85, 500.0, 20000.0)
R3 = cd.dirac_sum_g2(XR, 0.8, 100.0, 3000.0, 1e5)
R21 = cd.dirac_sum_g2(XR, 0.9, 1000.0, 1000.0, 30000.0)
R2N = R2 + np.random.default_rng(SEED).normal(0, 1e-3, 80)
RB = cd.dirac_sum_g2(XR, 0.3, 1000.0)
GAMMA0_R2 = 1.0 / XR[33]

CG01 = [2000, 2100, 2150, 5000, 5200, 12000]
T3 = np.array([0.0, 1e-3, 2e-3])

# ----------------------------------------------------------------------------
# 3. dirac_sum_g1 / dirac_sum_g2
# ----------------------------------------------------------------------------


def test_oracle_dirac_sum_g1_single_mode():
    """G-01"""
    out = cd.dirac_sum_g1(T3, 1000.0)
    assert out == pytest.approx([1, np.exp(-1), np.exp(-2)], **DET)


def test_oracle_dirac_sum_g1_two_modes():
    """G-02"""
    out = cd.dirac_sum_g1(T3, 1000.0, 3000.0)
    exp = [1, (np.exp(-1) + np.exp(-3)) / 2, (np.exp(-2) + np.exp(-6)) / 2]
    assert out == pytest.approx(exp, **DET)
    assert out[1] == pytest.approx(0.208833, rel=1e-5)


def test_oracle_dirac_sum_g2_single_mode():
    """G-03"""
    out = cd.dirac_sum_g2(T3, 0.8, 1000.0)
    assert out == pytest.approx(0.8 * np.array([1, np.exp(-2), np.exp(-4)]), **DET)
    assert out[1] == pytest.approx(0.108268, rel=1e-5)


def test_oracle_dirac_sum_g2_two_modes():
    """G-04"""
    out = cd.dirac_sum_g2(T3, 1.0, 1000.0, 3000.0)
    exp = ((np.exp(-1000 * T3) + np.exp(-3000 * T3)) / 2) ** 2
    assert out == pytest.approx(exp, **DET)
    assert out[1] == pytest.approx(0.0436113, rel=1e-5)


@pytest.mark.parametrize("gammas", [(1000.0,), (3.0, 77.0), (1.0, 2.0, 1e5)])
def test_oracle_dirac_sum_zero_lag_is_exact(gammas):
    """G-05: g1(0) == 1.0 exactly, g2(0) == beta."""
    t = np.array([0.0])
    assert cd.dirac_sum_g1(t, *gammas)[0] == 1.0
    assert cd.dirac_sum_g2(t, 0.77, *gammas)[0] == 0.77 * 1.0


def test_meta_dirac_sum_gamma_permutation():
    """G-M1"""
    t = np.logspace(-5, -2, 20)
    a = cd.dirac_sum_g2(t, 0.9, 100.0, 2000.0, 30000.0)
    b = cd.dirac_sum_g2(t, 0.9, 30000.0, 100.0, 2000.0)
    assert b == pytest.approx(a, **DET)


def test_meta_dirac_sum_time_scaling():
    """G-M2"""
    s = 7.3
    t = np.logspace(-5, -2, 20)
    a = cd.dirac_sum_g2(t, 0.9, 100.0, 2000.0)
    b = cd.dirac_sum_g2(s * t, 0.9, 100.0 / s, 2000.0 / s)
    assert b == pytest.approx(a, **DET)


def test_meta_dirac_sum_duplicate_modes():
    """G-M3"""
    t = np.logspace(-5, -2, 20)
    g = (100.0, 2000.0)
    assert cd.dirac_sum_g1(t, *g, *g) == pytest.approx(cd.dirac_sum_g1(t, *g), **DET)


def test_meta_dirac_sum_g2_linear_in_beta():
    """G-M4"""
    t = np.logspace(-5, -2, 20)
    a = cd.dirac_sum_g2(t, 0.9, 100.0, 2000.0)
    b = cd.dirac_sum_g2(t, 0.9 * 0.37, 100.0, 2000.0)
    assert b == pytest.approx(0.37 * a, **DET)


@pytest.mark.parametrize("gammas", [(1.0,), (100.0,), (1000.0,), (1.0, 1000.0)])
def test_meta_dirac_sum_g1_monotone_decreasing(gammas):
    """G-M5"""
    g1 = cd.dirac_sum_g1(np.linspace(0, 0.5, 51), *gammas)
    assert np.all(np.diff(g1) < 0)
    assert np.all(g1 > 0)
    assert np.all(g1 <= 1)


@IGNORE_RT
def test_char_dirac_sum_g1_no_gammas_is_nan_documents_current_behavior():
    """G-E1: documents current behavior, not desired behavior (0/0)."""
    out = cd.dirac_sum_g1(np.array([0.0, 1.0]))
    assert np.all(np.isnan(out))


def test_edge_dirac_sum_g1_zero_gamma_constant_one():
    """G-E2"""
    assert np.array_equal(cd.dirac_sum_g1(T3, 0.0), np.ones(3))


def test_char_dirac_sum_g1_negative_gamma_documents_current_behavior():
    """G-E3: documents current behavior, not desired behavior (no validation)."""
    out = cd.dirac_sum_g1(np.array([1e-3]), -1000.0)
    assert out[0] == pytest.approx(np.e, **DET)
    assert out[0] > 1


def test_char_dirac_sum_g1_list_input_raises_documents_current_behavior():
    """G-E4: documents current behavior, not desired behavior (list * float)."""
    with pytest.raises(TypeError):
        cd.dirac_sum_g1([0.0, 1e-3], 1000.0)


# ----------------------------------------------------------------------------
# 4.2 fit_cumulant_D with scripted curve_fit [MOCK]
# ----------------------------------------------------------------------------


def test_oracle_fit_cumulant_D_call_protocol_and_result(monkeypatch):
    """SF-01 [MOCK]"""
    s = scripted(monkeypatch, {1: [0.9, 123.0]})
    r = cd.fit_cumulant_D(X0, Y0, n_start=1, n_max=1)
    assert len(s.calls) == 1
    c = s.calls[0]
    assert c["p0"] == pytest.approx([1.0, 100.0], **DET)
    assert c["lb"] == [0.5, 1e-6]
    assert c["ub"] == [1.5, 1e8]
    assert c["method"] == "trf"
    assert c["maxfev"] == 5000
    assert c["sigma"] is None
    assert c["absolute_sigma"] is False
    # JADE 3.1: new key fit_ss = sum(w_rel*residuals**2) (== residual_ss when unweighted)
    assert {"n_modes", "beta", "gammas", "residual_ss", "fit_ss",
            "g1_fit", "g2_fit", "convergence"} <= set(r)
    assert r["n_modes"] == 1
    assert r["beta"] == pytest.approx(0.9, **DET)
    assert r["gammas"] == pytest.approx([123.0], **DET)
    assert r["residual_ss"] == pytest.approx(4 * 0.81, **DET)
    assert r["fit_ss"] == pytest.approx(r["residual_ss"], **DET)
    assert np.array_equal(r["g1_fit"], np.ones(4))
    assert r["g2_fit"] == pytest.approx(0.9 * np.ones(4), **DET)
    assert r["convergence"] == "success"


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_oracle_fit_cumulant_D_start_grid(monkeypatch, k):
    """SF-02 [MOCK]: cold-start grid logspace(0.1*G0, 10*G0, k)."""
    s = scripted(monkeypatch, {})
    cd.fit_cumulant_D(X0, Y0, gamma_initial=1000.0, beta_initial=0.8, n_start=k, n_max=k)
    p0 = s.calls[0]["p0"]
    assert p0[0] == 0.8
    exp = [100.0] if k == 1 else [10 ** (2 + 2 * j / (k - 1)) for j in range(k)]
    assert p0[1:] == pytest.approx(exp, **DET)


@pytest.mark.parametrize("x,y,kw,expected", [
    ([1e-6, 1e-5, 1e-4, 1e-3], [1.0, 0.8, 0.3, 0.0], {}, 0.1 / 1e-4),
    ([1., 2., 3., 4.], [1.0, 0.37, 0.2, 0.0], {}, 0.1 / 2),
    ([1., 2., 3., 4.], [1.0, 0.5, -0.2, 0.1], {}, 0.1 / 3),
    ([0., 1., 2., 3.], [0.2, 1.0, 0.5, 0.0], {}, 100.0),
    ([1., 2., 3., 4.], [1.0, np.nan, 0.5, 0.0], {}, 100.0),
    ([1e-6, 1e-5, 1e-4, 1e-3], [1.0, 0.8, 0.3, 0.0], {"gamma_initial": 50.0}, 5.0),
], ids=["a", "b-equality", "c-negative-y", "d-x-zero", "e-nan", "f-explicit"])
def test_oracle_fit_cumulant_D_gamma0_estimate(monkeypatch, x, y, kw, expected):
    """SF-03 [MOCK]"""
    s = scripted(monkeypatch, {})
    if np.isnan(np.asarray(y)).any():
        # case e: residual NaN never beats inf -> "All fits failed"; the start
        # value is still recorded by the mocked curve_fit.
        with pytest.raises(RuntimeError, match="All fits failed"):
            cd.fit_cumulant_D(np.array(x), np.array(y), n_start=1, n_max=1, **kw)
    else:
        cd.fit_cumulant_D(np.array(x), np.array(y), n_start=1, n_max=1, **kw)
    assert s.calls[0]["p0"][1] == pytest.approx(expected, **DET)


SC_SF04 = {1: [1.0, 50.0], 2: [0.9, 1000.0, 10.0], 3: [0.8, 10.0, 100.0, 1e4],
           4: [0.7, 10.0, 100.0, 1000.0, 1e4]}


def test_oracle_fit_cumulant_D_warm_start(monkeypatch):
    """SF-04 [MOCK]"""
    s = scripted(monkeypatch, SC_SF04)
    r = cd.fit_cumulant_D(X0, Y0, n_max=4)
    assert len(s.calls) == 4
    exp = [[1.0, 100.0], [1.0, 50.0, 100.0], [0.9, 10.0, 100.0, 1000.0],
           [0.8, 10.0, 100.0, 1000.0, 1e4]]
    for got, e in zip(s.p0s, exp):
        assert got == pytest.approx(e, **DET)
    assert r["n_modes"] == 4
    assert r["gammas"] == pytest.approx([10, 100, 1000, 1e4], **DET)
    assert r["beta"] == pytest.approx(0.7, **DET)
    assert r["residual_ss"] == pytest.approx(4 * 0.49, **DET)


SC_SF05 = {1: [1.0, 100.0], 2: [0.5, 10.0, 1000.0], 3: [0.499, 10.0, 100.0, 1000.0]}


def test_oracle_fit_cumulant_D_stops_on_small_improvement(monkeypatch, capsys):
    """SF-05 [MOCK]: C2 abort; the aborting n is still selected."""
    s = scripted(monkeypatch, SC_SF05)
    r = cd.fit_cumulant_D(X0, Y0)
    assert len(s.calls) == 3
    assert "improvement < 1%" in capsys.readouterr().out
    assert r["n_modes"] == 3
    assert r["beta"] == pytest.approx(0.499, **DET)
    assert r["residual_ss"] == pytest.approx(4 * 0.499 ** 2, **DET)


def test_oracle_fit_cumulant_D_stops_on_consistently_small_improvements(monkeypatch, capsys):
    """SF-06 [MOCK]: C3 abort."""
    s = scripted(monkeypatch, {1: [1.0, 100.0], 2: [0.9925, 10.0, 1000.0],
                               3: [0.985, 10.0, 100.0, 1000.0]})
    r = cd.fit_cumulant_D(X0, Y0)
    assert len(s.calls) == 3
    assert "consistently small improvements" in capsys.readouterr().out
    assert r["n_modes"] == 3
    assert r["residual_ss"] == pytest.approx(4 * 0.985 ** 2, **DET)


def test_oracle_fit_cumulant_D_c3_does_not_trigger_tie_keeps_smaller_n(monkeypatch):
    """SF-06b [MOCK]: improvement 0.02503 >= 0.02; n=4 ties n=3 -> n=3 (strict <)."""
    s = scripted(monkeypatch, {1: [1.0, 100.0], 2: [0.9925, 10.0, 1000.0],
                               3: [0.98, 10.0, 100.0, 1000.0],
                               4: [0.98, 10.0, 100.0, 1000.0, 1e4]})
    r = cd.fit_cumulant_D(X0, Y0)
    assert len(s.calls) == 4
    assert r["n_modes"] == 3
    assert r["residual_ss"] == pytest.approx(4 * 0.98 ** 2, **DET)


def test_oracle_fit_cumulant_D_stops_on_mode_collapse(monkeypatch, capsys):
    """SF-07 [MOCK]: C1 abort; the collapsed n=2 fit is returned."""
    s = scripted(monkeypatch, {1: [1.0, 1000.0], 2: [0.9, 1000.0, 1100.0]})
    r = cd.fit_cumulant_D(X0, Y0)
    assert len(s.calls) == 2
    assert spread([1000.0, 1100.0]) == pytest.approx(0.013703, rel=1e-3)
    assert "modes collapsed" in capsys.readouterr().out
    assert r["n_modes"] == 2
    assert r["gammas"] == pytest.approx([1000.0, 1100.0], **DET)


def test_oracle_fit_cumulant_D_rss_increase_keeps_previous(monkeypatch):
    """SF-08 [MOCK]"""
    s = scripted(monkeypatch, {1: [0.5, 100.0], 2: [0.6, 10.0, 1000.0]})
    r = cd.fit_cumulant_D(X0, Y0)
    assert len(s.calls) == 2
    assert r["n_modes"] == 1
    assert r["gammas"] == pytest.approx([100.0], **DET)
    assert r["beta"] == pytest.approx(0.5, **DET)


SC_SF09 = {1: [0.6, 50.0], 2: [0.4, 10.0, 1000.0], 3: [0.4, 10.0, 100.0, 1000.0]}


def test_oracle_fit_cumulant_D_weighted_selection(monkeypatch):
    """SF-09 [MOCK]: weighted selection picks n=1; warm start uses best beta; RSS unweighted."""
    s = scripted(monkeypatch, SC_SF09)
    r = cd.fit_cumulant_D(X0, YW, weights=W)
    assert s.calls[0]["sigma"] == pytest.approx(1 / np.sqrt(W / W.mean()), **DET)
    assert len(s.calls) == 3
    assert s.p0s[1] == pytest.approx([0.6, 50.0, 100.0], **DET)
    assert s.p0s[2] == pytest.approx([0.6, 10.0, 100.0, 1000.0], **DET)
    assert r["n_modes"] == 1
    assert r["beta"] == pytest.approx(0.6, **DET)
    assert r["residual_ss"] == pytest.approx(1.24, **DET)
    assert r["fit_ss"] == pytest.approx(6.88 / 7, **DET)  # JADE 3.1: fit_ss = sum(w_rel*res^2)


def test_oracle_fit_cumulant_D_unweighted_selection_same_script(monkeypatch):
    """SF-10 [MOCK]"""
    scripted(monkeypatch, SC_SF09)
    r = cd.fit_cumulant_D(X0, YW)
    assert r["n_modes"] == 2
    assert r["beta"] == pytest.approx(0.4, **DET)
    assert r["residual_ss"] == pytest.approx(0.84, **DET)
    assert r["fit_ss"] == pytest.approx(r["residual_ss"], **DET)


def test_oracle_fit_cumulant_D_failure_at_n1_then_cold_start(monkeypatch):
    """SF-11 [MOCK]"""
    s = scripted(monkeypatch, {1: RuntimeError("x"), 2: [0.5, 10.0, 1000.0]})
    r = cd.fit_cumulant_D(X0, Y0, n_max=2)
    assert len(s.calls) == 2
    assert s.p0s[1] == pytest.approx([1.0, 100.0, 1e4], **DET)
    assert r["n_modes"] == 2


@pytest.mark.parametrize("n_max,calls,n_sel,beta", [(2, 2, 2, 0.5), (3, 3, 3, 0.499), (25, 3, 3, 0.499)])
def test_oracle_fit_cumulant_D_n_max_caps_orders(monkeypatch, n_max, calls, n_sel, beta):
    """SF-12 [MOCK]"""
    s = scripted(monkeypatch, SC_SF05)
    r = cd.fit_cumulant_D(X0, Y0, n_max=n_max)
    assert len(s.calls) == calls
    assert r["n_modes"] == n_sel
    assert r["beta"] == pytest.approx(beta, **DET)


def test_oracle_fit_cumulant_D_n_start_two_is_cold_start(monkeypatch):
    """SF-13 [MOCK]"""
    s = scripted(monkeypatch, {})
    r = cd.fit_cumulant_D(X0, Y0, n_start=2, n_max=2, gamma_initial=1000.0)
    assert len(s.calls) == 1
    assert s.p0s[0] == pytest.approx([1.0, 100.0, 1e4], **DET)
    assert spread([100.0, 1e4]) == pytest.approx(2 / 3, rel=1e-6)
    assert r["n_modes"] == 2


# ----------------------------------------------------------------------------
# 4.3 fit_cumulant_D real optimizer [REAL]
# ----------------------------------------------------------------------------


def test_oracle_fit_cumulant_D_real_monomodal():
    """RO-01 [REAL]"""
    r = cd.fit_cumulant_D(XR, R1, n_start=1, n_max=1)
    assert r["gammas"] == pytest.approx([1000.0], **OPT)
    assert r["beta"] == pytest.approx(0.9, **OPT)
    assert r["residual_ss"] < 1e-10


def test_oracle_fit_cumulant_D_real_bimodal_fixed_n():
    """RO-02 [REAL]"""
    r = cd.fit_cumulant_D(XR, R2, n_start=2, n_max=2)
    assert r["gammas"] == pytest.approx([500.0, 20000.0], **OPT)
    assert r["beta"] == pytest.approx(0.85, **OPT)
    assert r["residual_ss"] < 1e-10


def test_oracle_fit_cumulant_D_real_trimodal_fixed_n():
    """RO-03 [REAL]"""
    r = cd.fit_cumulant_D(XR, R3, n_start=3, n_max=3, gamma_initial=3000.0)
    assert r["gammas"] == pytest.approx([100.0, 3000.0, 1e5], **OPT)
    assert r["beta"] == pytest.approx(0.8, **OPT)


def test_oracle_fit_cumulant_D_real_bimodal_automatic(monkeypatch):
    """RO-04 [REAL]"""
    s = spied(monkeypatch)
    r = cd.fit_cumulant_D(XR, R2)
    assert r["n_modes"] == 2
    assert r["gammas"] == pytest.approx([500.0, 20000.0], **OPT)
    assert len(s.ns) == 3


def test_oracle_fit_cumulant_D_real_monomodal_automatic(monkeypatch):
    """RO-05 [REAL]"""
    s = spied(monkeypatch)
    r = cd.fit_cumulant_D(XR, R1)
    assert len(s.ns) == 2
    # spec RO-05 fixes no n_modes: n=2 may collapse (C1) or raise RSS (C2); observed n=1.
    assert r["n_modes"] in (1, 2)
    assert np.all((r["gammas"] >= 500) & (r["gammas"] <= 2000))
    assert np.mean(r["gammas"]) == pytest.approx(1000.0, rel=1e-2)


def test_oracle_fit_cumulant_D_real_noisy_bimodal():
    """RO-07 [REAL]: STAT tolerance."""
    r = cd.fit_cumulant_D(XR, R2N, n_start=2, n_max=2)
    assert r["gammas"] == pytest.approx([500.0, 20000.0], **STAT)


def test_char_fit_cumulant_D_beta_below_bound_documents_current_behavior():
    """RO-08 [REAL]: documents current behavior (F-05): beta clamped at 0.5, gamma biased."""
    r = cd.fit_cumulant_D(XR, RB, n_start=1, n_max=1)
    assert r["beta"] == pytest.approx(0.5, abs=1e-6)
    assert abs(r["gammas"][0] / 1000.0 - 1) > 0.05


# ----------------------------------------------------------------------------
# 4.4 metamorphic
# ----------------------------------------------------------------------------


def test_meta_fit_cumulant_D_repeat_is_bit_identical():
    """FM-01 [REAL]"""
    for y, kw in [(R2N, dict(n_start=2, n_max=2)), (R2, {})]:
        a = cd.fit_cumulant_D(XR, y, **kw)
        b = cd.fit_cumulant_D(XR, y, **kw)
        assert np.array_equal(a["gammas"], b["gammas"])
        assert a["beta"] == b["beta"]
        assert a["n_modes"] == b["n_modes"]


@pytest.mark.parametrize("s", [10.0, 0.1])
def test_meta_fit_cumulant_D_time_scaling(s):
    """FM-02 [REAL]"""
    a = cd.fit_cumulant_D(XR, R2, n_start=2, n_max=2)
    b = cd.fit_cumulant_D(XR * s, R2, n_start=2, n_max=2)
    assert b["gammas"] == pytest.approx(a["gammas"] / s, **OPT)
    assert b["beta"] == pytest.approx(a["beta"], **OPT)


def test_meta_fit_cumulant_D_amplitude_scaling():
    """FM-03 [REAL]"""
    c = 0.8
    for kw in (dict(n_start=2, n_max=2), {}):
        a = cd.fit_cumulant_D(XR, R2, **kw)
        b = cd.fit_cumulant_D(XR, c * R2, **kw)
        assert b["n_modes"] == a["n_modes"]
        assert b["gammas"] == pytest.approx(a["gammas"], **OPT)
        assert b["beta"] == pytest.approx(c * a["beta"], **OPT)


def test_meta_fit_cumulant_D_point_permutation_fixed_gamma0():
    """FM-04 [REAL]"""
    perm = np.random.default_rng(7).permutation(80)
    a = cd.fit_cumulant_D(XR, R2, n_start=2, n_max=2, gamma_initial=1.19e4)
    b = cd.fit_cumulant_D(XR[perm], R2[perm], n_start=2, n_max=2, gamma_initial=1.19e4)
    assert b["gammas"] == pytest.approx(a["gammas"], **OPT)
    assert b["beta"] == pytest.approx(a["beta"], **OPT)


def test_meta_fit_cumulant_D_unit_weights_equal_unweighted():
    """FM-05 [REAL]"""
    a = cd.fit_cumulant_D(XR, R2)
    b = cd.fit_cumulant_D(XR, R2, weights=np.full(80, 2.0))
    assert b["fit_ss"] == pytest.approx(b["residual_ss"], **DET)
    assert b["gammas"] == pytest.approx(a["gammas"], **DET)
    assert b["n_modes"] == a["n_modes"]


def test_meta_fit_cumulant_D_weight_scale_invariance():
    """FM-06 [REAL]"""
    w = np.linspace(0.5, 2.0, 80)
    a = cd.fit_cumulant_D(XR, R2N, n_start=2, n_max=2, weights=w)
    b = cd.fit_cumulant_D(XR, R2N, n_start=2, n_max=2, weights=3 * w)
    assert b["gammas"] == pytest.approx(a["gammas"], **OPT)
    assert b["beta"] == pytest.approx(a["beta"], **OPT)
    assert b["fit_ss"] == pytest.approx(a["fit_ss"], **OPT)


def test_meta_fit_cumulant_D_noise_free_weights_irrelevant():
    """FM-07 [REAL]"""
    w = np.linspace(0.5, 2.0, 80)
    a = cd.fit_cumulant_D(XR, R2, n_start=2, n_max=2, weights=w)
    b = cd.fit_cumulant_D(XR, R2, n_start=2, n_max=2)
    assert a["gammas"] == pytest.approx(b["gammas"], **OPT)


def test_meta_fit_cumulant_D_larger_n_max_not_worse(monkeypatch):
    """FM-08 [MOCK]+[REAL]"""
    scripted(monkeypatch, SC_SF05)
    k = cd.fit_cumulant_D(X0, Y0, n_max=2)
    K = cd.fit_cumulant_D(X0, Y0, n_max=3)
    assert K["fit_ss"] <= k["fit_ss"]
    monkeypatch.undo()
    k = cd.fit_cumulant_D(XR, R2, n_max=2)
    K = cd.fit_cumulant_D(XR, R2, n_max=25)
    assert K["fit_ss"] <= k["fit_ss"] + 1e-15


@IGNORE_RT
def test_meta_fit_cumulant_D_n_start_zero_equals_one():
    """FM-09 [REAL]: n=0 fit fails (scipy ValueError) and is skipped."""
    a = cd.fit_cumulant_D(XR, R1, n_start=0, n_max=1)
    b = cd.fit_cumulant_D(XR, R1, n_start=1, n_max=1)
    assert a["n_modes"] == b["n_modes"] == 1
    assert np.array_equal(a["gammas"], b["gammas"])
    assert a["beta"] == b["beta"]


@pytest.mark.parametrize("sc", [SC_SF04, SC_SF05, SC_SF09, {1: [0.5, 100.0], 2: [0.6, 10.0, 1000.0]}],
                         ids=["sf04", "sf05", "sf09", "sf08"])
def test_meta_fit_cumulant_D_scripted_gamma_scaling(monkeypatch, sc):
    """FM-10 [MOCK]"""
    c = 7.3
    y = YW if sc is SC_SF09 else Y0
    s1 = scripted(monkeypatch, sc)
    a = cd.fit_cumulant_D(X0, y)
    scaled = {n: [v[0]] + [g * c for g in v[1:]] for n, v in sc.items()}
    s2 = scripted(monkeypatch, scaled)
    b = cd.fit_cumulant_D(X0, y)
    assert len(s2.calls) == len(s1.calls)
    assert b["n_modes"] == a["n_modes"]
    assert b["beta"] == pytest.approx(a["beta"], **DET)
    assert b["gammas"] == pytest.approx(a["gammas"] * c, **DET)


# ----------------------------------------------------------------------------
# 4.6 edge cases
# ----------------------------------------------------------------------------


def test_edge_fit_cumulant_D_n_start_above_n_max(monkeypatch):
    """FE-01 [MOCK]"""
    s = scripted(monkeypatch, {})
    with pytest.raises(RuntimeError, match="All fits failed"):
        cd.fit_cumulant_D(X0, Y0, n_start=3, n_max=2)
    assert s.calls == []


def test_edge_fit_cumulant_D_all_fits_fail(monkeypatch):
    """FE-02 [MOCK]"""
    s = scripted(monkeypatch, {n: RuntimeError() for n in (1, 2, 3)})
    with pytest.raises(RuntimeError, match="All fits failed"):
        cd.fit_cumulant_D(X0, Y0, n_max=3)
    assert len(s.calls) == 3
    exp = [[1.0, 100.0], [1.0, 100.0, 1e4], [1.0, 100.0, 1000.0, 1e4]]
    for got, e in zip(s.p0s, exp):
        assert got == pytest.approx(e, **DET)


def test_edge_fit_cumulant_D_beta_initial_outside_bounds():
    """FE-03 [REAL]"""
    with pytest.raises(RuntimeError, match="All fits failed"):
        cd.fit_cumulant_D(XR, R1, beta_initial=2.0, n_max=3)


def test_edge_fit_cumulant_D_nan_in_y():
    """FE-04 [REAL]"""
    y = R1.copy()
    y[5] = np.nan
    with pytest.raises(RuntimeError, match="All fits failed"):
        cd.fit_cumulant_D(XR, y, n_max=2)


def test_edge_fit_cumulant_D_empty_input_raises_value_error():
    """FE-05 [REAL]"""
    with pytest.raises(ValueError):
        cd.fit_cumulant_D(np.array([]), np.array([]))


def test_edge_fit_cumulant_D_weights_wrong_length():
    """FE-06 [REAL]"""
    with pytest.raises(RuntimeError, match="All fits failed"):
        cd.fit_cumulant_D(XR, R1, n_max=2, weights=np.ones(79))


@IGNORE_RT
@pytest.mark.parametrize("g0", [0.0, -5.0])
def test_edge_fit_cumulant_D_invalid_gamma_initial(g0):
    """FE-07 [REAL]"""
    with pytest.raises(RuntimeError, match="All fits failed"):
        cd.fit_cumulant_D(XR, R1, gamma_initial=g0, n_max=2)


def test_edge_fit_cumulant_D_single_mode_only(monkeypatch):
    """FE-08 [MOCK]"""
    s = scripted(monkeypatch, {})
    r = cd.fit_cumulant_D(X0, Y0, n_max=1)
    assert len(s.calls) == 1
    assert r["n_modes"] == 1


@IGNORE_RT
def test_char_fit_cumulant_D_zero_rss_nan_improvement_documents_current_behavior(monkeypatch):
    """FE-09 [MOCK]: documents current behavior (F-08): 0/0 = NaN does not abort; -inf aborts."""
    s = scripted(monkeypatch, {1: [1.0, 100.0], 2: [1.0, 10.0, 1000.0],
                               3: [0.5, 10.0, 100.0, 1000.0]})
    r = cd.fit_cumulant_D(X0, Y1)
    assert len(s.calls) == 3
    assert r["n_modes"] == 1
    assert r["residual_ss"] == 0.0


# ----------------------------------------------------------------------------
# 4.7 characterization
# ----------------------------------------------------------------------------


def _fc_c1_run(monkeypatch):
    s = scripted(monkeypatch, {1: [1.0, 50.0], 2: RuntimeError("scripted")}, strict=True)
    r = cd.fit_cumulant_D(X0, Y0, n_max=5)
    return s, r


def test_char_fit_cumulant_D_failure_propagates_documents_current_behavior(monkeypatch):
    """FC-C1 [MOCK]: documents current behavior, not desired behavior (F-02)."""
    s, r = _fc_c1_run(monkeypatch)
    assert len(s.calls) == 5
    assert [len(c["p0"]) for c in s.calls] == [2, 3, 3, 3, 3]
    assert [len(c["lb"]) for c in s.calls] == [2, 3, 4, 5, 6]
    assert r["n_modes"] == 1
    assert r["beta"] == 1.0


@pytest.mark.parametrize("g2,calls,collapse", [([100.0, 158.0], 2, True), ([100.0, 170.0], 3, False),
                                               ([1e4, 2.5e4], 2, True)],
                         ids=["100-158", "100-170", "1e4-2.5e4"])
def test_char_fit_cumulant_D_collapse_threshold_depends_on_level_documents_current_behavior(
        monkeypatch, g2, calls, collapse):
    """FC-C2 [MOCK]: documents current behavior (F-04): collapse limit depends on absolute gamma."""
    n3 = [0.9, g2[0], np.sqrt(g2[0] * g2[1]), g2[1]]
    s = scripted(monkeypatch, {1: [1.0, 100.0], 2: [0.9] + g2, 3: n3})
    cd.fit_cumulant_D(X0, Y0)
    assert len(s.calls) == calls
    assert bool(spread(g2) < 0.1) is collapse


def _fc_c3_calls(monkeypatch):
    a = scripted(monkeypatch, {1: [1.0, 1000.0], 2: [0.9, 1000.0, 1100.0]})
    cd.fit_cumulant_D(X0, Y0)
    b = scripted(monkeypatch, {1: [1.0, 1.0], 2: [0.9, 1.0, 1.1], 3: [0.9, 1.0, 1.05, 1.1]})
    rb = cd.fit_cumulant_D(X0, Y0)
    return len(a.calls), len(b.calls), rb


def test_char_fit_cumulant_D_unit_dependence_documents_current_behavior(monkeypatch):
    """FC-C3 [MOCK]: documents current behavior (F-04): same ratio, different units."""
    na, nb, rb = _fc_c3_calls(monkeypatch)
    assert na == 2
    assert nb == 3
    assert rb["n_modes"] == 2
    assert spread([1.0, 1.1]) == pytest.approx(2.0, rel=1e-6)


@IGNORE_RT
def test_char_fit_cumulant_D_zero_mean_log_gamma_documents_current_behavior(monkeypatch):
    """FC-C4 [MOCK]: documents current behavior: mean log10 = 0 -> inf spread, no collapse."""
    s = scripted(monkeypatch, {1: [1.0, 1.0], 2: [0.9, 0.5, 2.0], 3: [0.9, 0.5, 1.0, 2.0]})
    cd.fit_cumulant_D(X0, Y0)
    assert len(s.calls) == 3


def test_char_fit_cumulant_D_single_mode_starts_at_tenth_documents_current_behavior(monkeypatch):
    """FC-C5 [MOCK]: documents current behavior (F-07): n=1 starts at 0.1*Gamma0."""
    s = scripted(monkeypatch, {})
    cd.fit_cumulant_D(X0, Y0, gamma_initial=1000.0, n_start=1, n_max=1)
    assert s.p0s[0][1] == pytest.approx(100.0, **DET)


def test_char_signature_defaults_documents_current_behavior():
    """FC-C6 [REAL]: documents current behavior (F-10)."""
    def defaults(f):
        return {k: v.default for k, v in inspect.signature(f).parameters.items()
                if v.default is not inspect.Parameter.empty}
    assert defaults(cd.fit_cumulant_D) == dict(n_max=25, beta_initial=1.0, gamma_initial=None,
                                               n_start=1, weights=None)
    assert defaults(cd.cluster_gammas) == dict(gap_threshold=1.5)
    assert defaults(cd.fit_correlations_method_D) == dict(
        x_col="t [s]", y_col="g(2)-1", n_max=25, n_start=1, gap_threshold=1.5, plot=True)


def test_char_fit_cumulant_D_model_wrapper_sorts_gammas_documents_current_behavior(monkeypatch):
    """FC-C7 [MOCK]: documents current behavior: model wrapper sorts the gammas."""
    s = scripted(monkeypatch, {1: [0.9, 123.0]})
    cd.fit_cumulant_D(X0, Y0, n_max=1)
    f = s.calls[0]["f"]
    t = np.array([0.0, 1e-3])
    assert np.array_equal(f(t, 0.9, 3000.0, 1000.0), cd.dirac_sum_g2(t, 0.9, 1000.0, 3000.0))


# RO-06 values frozen at first run (2026-11-03, numpy 2.4.6, scipy 1.17.1, Windows):
# the optimizer converged to the exact truth (RSS 7e-26); doublet split is path dependent.
FROZEN_RO06 = {"gammas": [999.9985726, 1000.0014274, 30000.0], "mean": 10666.666666666491}


def test_char_fit_cumulant_D_real_amplitude_ratio_documents_current_behavior():
    """RO-06 [REAL]: documents current behavior; values frozen at first run."""
    r = cd.fit_cumulant_D(XR, R21, n_start=3, n_max=3, gamma_initial=3000.0)
    g = r["gammas"]
    assert r["n_modes"] == 3
    assert np.all(np.diff(g) >= 0)
    assert g[2] == pytest.approx(30000.0, rel=1e-2)
    assert 500 <= g[0] <= 2000 and 500 <= g[1] <= 2000
    assert g == pytest.approx(FROZEN_RO06["gammas"], **OPT)
    assert np.mean(g) == pytest.approx(FROZEN_RO06["mean"], **OPT)


def test_char_scipy_curve_fit_inconsistent_p0_bounds_raises_documents_current_behavior():
    """RO-15 [REAL]: documents scipy behavior that H1(strict) emulates."""
    with pytest.raises(ValueError):
        scipy.optimize.curve_fit(lambda t, a, b: a * np.exp(-b * t), XR, np.exp(-XR),
                                 p0=[1.0, 1.0], bounds=([0, 0, 0], [2, 2, 2]), method="trf")


# ----------------------------------------------------------------------------
# 5. calculate_moments_from_gammas
# ----------------------------------------------------------------------------
MKEYS = ["gamma_mean", "gamma_2", "gamma_3", "pdi", "skewness", "gamma_4", "kurtosis"]


def mom(g):
    return cd.calculate_moments_from_gammas(np.asarray(g, float))


def test_oracle_moments_two_point():
    """MO-01"""
    m = mom([1.0, 3.0])
    assert list(m) == MKEYS
    assert m["gamma_mean"] == pytest.approx(2, **DET)
    assert m["gamma_2"] == pytest.approx(5, **DET)
    assert m["gamma_3"] == pytest.approx(14, **DET)
    assert m["gamma_4"] == pytest.approx(41, **DET)
    assert m["pdi"] == pytest.approx(0.25, **DET)
    assert m["skewness"] == pytest.approx(0, abs=1e-12)
    assert m["kurtosis"] == pytest.approx(1.0, **DET)  # JADE 3.1: mu4/mu2^2, Gaussian = 3


def test_oracle_moments_asymmetric_three_point():
    """MO-02"""
    m = mom([1.0, 1.0, 4.0])
    assert (m["gamma_mean"], m["gamma_2"], m["gamma_3"], m["gamma_4"]) == pytest.approx((2, 6, 22, 86), **DET)
    assert m["pdi"] == pytest.approx(0.5, **DET)
    assert m["skewness"] == pytest.approx(1 / np.sqrt(2), **DET)
    assert m["kurtosis"] == pytest.approx(1.5, **DET)  # JADE 3.1


def test_oracle_moments_single_mode_guard():
    """MO-03"""
    m = mom([1000.0])
    assert (m["gamma_mean"], m["gamma_2"], m["gamma_3"], m["gamma_4"]) == pytest.approx(
        (1000, 1e6, 1e9, 1e12), **DET)
    assert m["pdi"] == 0.0
    assert np.isnan(m["skewness"])  # JADE 3.1: NaN when variance is 0
    assert np.isnan(m["kurtosis"])


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_oracle_moments_two_point_family(k):
    """MO-04"""
    a, b, n = 100.0, 900.0, 5
    p = (n - k) / n
    m = mom([a] * k + [b] * (n - k))
    mu = (1 - p) * a + p * b
    assert m["gamma_mean"] == pytest.approx(mu, **DET)
    assert m["skewness"] == pytest.approx((1 - 2 * p) / np.sqrt(p * (1 - p)), **DET)
    assert m["kurtosis"] == pytest.approx((1 - 3 * p * (1 - p)) / (p * (1 - p)), **DET)  # JADE 3.1: non-excess
    assert m["pdi"] == pytest.approx(p * (1 - p) * (b - a) ** 2 / mu ** 2, **DET)


def test_oracle_moments_three_point_central_formula():
    """MO-05"""
    g = np.array([100.0, 110.0, 1000.0])
    m = mom(g)
    mu = g.mean()
    assert m["gamma_mean"] == pytest.approx(1210 / 3, **DET)
    assert m["gamma_2"] == pytest.approx(340700, **DET)
    assert m["pdi"] == pytest.approx(3066300 / 1464100 - 1, **DET)
    assert m["pdi"] == pytest.approx(1.094324, rel=1e-6)
    assert m["skewness"] == pytest.approx(np.mean((g - mu) ** 3) / np.var(g) ** 1.5, **DET)
    assert m["kurtosis"] == pytest.approx(np.mean((g - mu) ** 4) / np.var(g) ** 2, **DET)  # JADE 3.1: non-excess


G_MM = np.array([1.0, 2.0, 5.0, 11.0])
SK0 = dict(rel=1e-6, abs=1e-9)


def test_meta_moments_scaling():
    """MM-01"""
    c = 7.3
    a, b = mom(G_MM), mom(c * G_MM)
    for key, p in (("gamma_mean", 1), ("gamma_2", 2), ("gamma_3", 3), ("gamma_4", 4)):
        assert b[key] == pytest.approx(a[key] * c ** p, **DET)
    for key in ("pdi", "skewness", "kurtosis"):
        assert b[key] == pytest.approx(a[key], **SK0)


def test_meta_moments_permutation():
    """MM-02"""
    a = mom(G_MM)
    b = mom(G_MM[np.random.default_rng(7).permutation(4)])
    for k in MKEYS:
        assert b[k] == pytest.approx(a[k], **DET)


def test_meta_moments_duplication():
    """MM-03"""
    a, b = mom(G_MM), mom(np.tile(G_MM, 2))
    for k in MKEYS:
        assert b[k] == pytest.approx(a[k], **DET)


def test_meta_moments_append_mean_scales_pdi():
    """MM-04"""
    g = np.array([1.0, 3.0])
    a, b = mom(g), mom(np.append(g, g.mean()))
    assert b["gamma_mean"] == pytest.approx(a["gamma_mean"], **DET)
    assert b["pdi"] == pytest.approx(a["pdi"] * 2 / 3, **DET)
    assert b["pdi"] == pytest.approx(1 / 6, **DET)


def test_meta_moments_shift():
    """MM-05"""
    mu = G_MM.mean()
    c = 0.5 * mu
    a, b = mom(G_MM), mom(G_MM + c)
    assert b["skewness"] == pytest.approx(a["skewness"], **SK0)
    assert b["kurtosis"] == pytest.approx(a["kurtosis"], **DET)
    assert b["pdi"] == pytest.approx(np.var(G_MM) / (mu + c) ** 2, **DET)


def test_meta_moments_negation():
    """MM-06"""
    a, b = mom(G_MM), mom(-G_MM)
    assert b["pdi"] == pytest.approx(a["pdi"], **DET)
    assert b["kurtosis"] == pytest.approx(a["kurtosis"], **DET)
    assert b["skewness"] == pytest.approx(-a["skewness"], **DET)
    assert b["gamma_mean"] == pytest.approx(-a["gamma_mean"], **DET)
    assert b["gamma_3"] == pytest.approx(-a["gamma_3"], **DET)


def test_edge_moments_zero_mean_pdi_is_nan():
    """ME-01 (regression CHANGELOG 3.4.0)"""
    m = mom([-1.0, 1.0])
    assert np.isnan(m["pdi"])
    assert m["skewness"] == pytest.approx(0.0, abs=1e-12)
    assert m["kurtosis"] == pytest.approx(1.0, **DET)  # JADE 3.1


def test_edge_moments_all_zero():
    """ME-02"""
    m = mom([0.0, 0.0])
    assert np.isnan(m["pdi"])
    assert np.isnan(m["skewness"])  # JADE 3.1
    assert np.isnan(m["kurtosis"])


@IGNORE_RT
def test_char_moments_empty_is_all_nan_documents_current_behavior():
    """ME-03: documents current behavior, not desired behavior."""
    m = mom([])
    assert list(m) == MKEYS
    assert all(np.isnan(v) for v in m.values())


def test_char_moments_identical_modes_guard_documents_current_behavior():
    """ME-04: documents current behavior, not desired behavior."""
    m = mom([5.0, 5.0, 5.0])
    assert m["pdi"] == 0.0
    assert np.isnan(m["skewness"])  # JADE 3.1
    assert np.isnan(m["kurtosis"])


def test_char_moments_list_input_raises_documents_current_behavior():
    """ME-05: documents current behavior, not desired behavior."""
    with pytest.raises(TypeError):
        cd.calculate_moments_from_gammas([1.0, 3.0])


def test_char_moments_cancellation_documents_current_behavior():
    """ME-06: documents current behavior (F-09): raw-moment cancellation ruins kurtosis."""
    m = mom([1000.0, 1000.0 * (1 + 1e-9)])
    # observed (2026-11-03): Var computes to exactly 0 -> guard returns int 0 (true: -2)
    assert np.isnan(m["skewness"])  # JADE 3.1: NaN instead of 0
    assert np.isnan(m["kurtosis"])


def test_edge_moments_negative_gammas_not_validated():
    """ME-07"""
    m = mom([-2.0, -1.0])
    assert m["pdi"] == pytest.approx(2.5 / 2.25 - 1, **DET)
    assert m["skewness"] == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------------
# 6. cluster_gammas
# ----------------------------------------------------------------------------


def test_oracle_cluster_gammas_docstring_example_geometric_means():
    """CG-01 (docstring lists arithmetic means 2083/5100, F-11)"""
    cl, reps, info = cd.cluster_gammas(CG01, 1.5)
    assert [list(map(float, c)) for c in cl] == [[2000, 2100, 2150], [5000, 5200], [12000]]
    assert info["n_clusters"] == 3
    assert info["cluster_sizes"] == [3, 2, 1]
    g = np.array(CG01, float)
    assert info["gaps"] == pytest.approx(list(g[1:] / g[:-1]), **DET)
    assert reps == pytest.approx([(2000 * 2100 * 2150) ** (1 / 3), np.sqrt(5000 * 5200), 12000], **DET)
    assert reps[0] == pytest.approx(2082.39, rel=1e-5)
    assert reps[1] == pytest.approx(5099.02, rel=1e-5)


def test_oracle_cluster_gammas_gold_threshold_single_cluster():
    """CG-02"""
    cl, reps, info = cd.cluster_gammas(CG01, 3.0)
    assert info["n_clusters"] == 1
    assert reps[0] == pytest.approx(np.prod(np.array(CG01, float)) ** (1 / 6), **DET)


def test_oracle_cluster_gammas_threshold_is_strict():
    """CG-03"""
    assert cd.cluster_gammas([100.0, 300.0], 3.0)[2]["n_clusters"] == 1
    assert cd.cluster_gammas([100.0, 300.5], 3.0)[2]["n_clusters"] == 2
    assert cd.cluster_gammas([2.0, 3.0], 1.5)[2]["n_clusters"] == 1


def test_oracle_cluster_gammas_chaining():
    """CG-04: single-linkage chain exceeds the threshold in span."""
    g = [1, 1.25, 1.5625, 1.953125, 2.44140625]
    cl, _, info = cd.cluster_gammas(g, 1.5)
    assert info["n_clusters"] == 1
    assert max(g) / min(g) > 1.5


def test_oracle_cluster_gammas_unsorted_input():
    """CG-05"""
    cl, reps, _ = cd.cluster_gammas([1000.0, 10.0, 100.0], 1.5)
    assert [list(map(float, c)) for c in cl] == [[10.0], [100.0], [1000.0]]
    assert list(reps) == sorted(reps)


def test_oracle_cluster_gammas_single_value():
    """CG-06"""
    cl, reps, info = cd.cluster_gammas([500.0])
    assert [list(map(float, c)) for c in cl] == [[500.0]]
    assert reps == pytest.approx([500.0], **DET)
    assert info == {"n_clusters": 1, "cluster_sizes": [1], "gaps": []}


def test_oracle_cluster_gammas_threshold_below_one():
    """CG-07"""
    assert cd.cluster_gammas([100.0, 100.0, 200.0], 0.5)[2]["n_clusters"] == 3


def test_oracle_cluster_gammas_threshold_one():
    """CG-08"""
    cl, _, _ = cd.cluster_gammas([100.0, 100.0, 200.0], 1.0)
    assert [list(map(float, c)) for c in cl] == [[100.0, 100.0], [200.0]]


def test_meta_cluster_gammas_permutation_exact():
    """CM-01"""
    g = np.array(CG01, float)
    a = cd.cluster_gammas(g, 1.5)
    b = cd.cluster_gammas(g[np.random.default_rng(7).permutation(6)], 1.5)
    assert a[0] == b[0]
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_meta_cluster_gammas_scaling():
    """CM-02"""
    c = 7.3
    g = np.array(CG01, float)
    a, b = cd.cluster_gammas(g, 1.5), cd.cluster_gammas(g * c, 1.5)
    assert b[2]["cluster_sizes"] == a[2]["cluster_sizes"]
    assert [np.array(x) / c for x in b[0]] == [pytest.approx(x, **DET) for x in a[0]]
    assert b[1] == pytest.approx(a[1] * c, **DET)
    assert b[2]["gaps"] == pytest.approx(a[2]["gaps"], **DET)


def test_meta_cluster_gammas_reciprocals():
    """CM-03"""
    g = np.array(CG01, float)
    a, b = cd.cluster_gammas(g, 1.5), cd.cluster_gammas(1 / g, 1.5)
    assert b[2]["cluster_sizes"] == a[2]["cluster_sizes"][::-1]
    assert b[2]["gaps"] == pytest.approx(a[2]["gaps"][::-1], **DET)
    assert b[1] == pytest.approx(1 / a[1][::-1], **DET)


def test_meta_cluster_gammas_duplication():
    """CM-04"""
    g = np.array(CG01, float)
    a, b = cd.cluster_gammas(g, 1.5), cd.cluster_gammas(np.tile(g, 2), 1.5)
    assert b[2]["n_clusters"] == a[2]["n_clusters"]
    assert b[1] == pytest.approx(a[1], **DET)
    assert b[2]["cluster_sizes"] == [2 * s for s in a[2]["cluster_sizes"]]


def test_meta_cluster_gammas_threshold_monotone_nested():
    """CM-05"""
    ths = [1.01, 1.5, 3.0, 10.0]
    outs = [cd.cluster_gammas(CG01, t)[0] for t in ths]
    assert [len(o) for o in outs] == [6, 3, 1, 1]
    for lo, hi in zip(outs, outs[1:]):
        for c in lo:
            assert any(set(c) <= set(h) for h in hi)


def test_edge_cluster_gammas_empty_raises_index_error_documents_current_behavior():
    """CE-01: documents current behavior, not desired behavior."""
    with pytest.raises(IndexError):
        cd.cluster_gammas([])


@IGNORE_RT
def test_edge_cluster_gammas_zero_gamma():
    """CE-02"""
    cl, reps, info = cd.cluster_gammas([0.0, 5.0], 1.5)
    assert info["n_clusters"] == 2
    assert reps[0] == 0.0


@IGNORE_RT
def test_char_cluster_gammas_negative_gamma_documents_current_behavior():
    """CE-03: documents current behavior, not desired behavior."""
    cl, reps, info = cd.cluster_gammas([-2.0, 3.0], 1.5)
    assert info["n_clusters"] == 1
    assert np.isnan(reps[0])


def test_edge_cluster_gammas_integer_input():
    """CE-04"""
    cl, reps, info = cd.cluster_gammas([100, 400], 1.5)
    assert info["n_clusters"] == 2
    assert reps.dtype.kind == "f"
    assert reps == pytest.approx([100.0, 400.0], **DET)


# ----------------------------------------------------------------------------
# 7. fit_correlations_method_D [MOCK]
# ----------------------------------------------------------------------------
Y_A = [1.0, 0.5, 0.0, 0.0]
Y_B = [0.9, 0.6, 0.3, 0.0]
T4 = [1e-6, 1e-5, 1e-4, 1e-3]
COLS_K2 = ["filename", "n_modes", "beta", "residual_ss", "weighted", "n_populations",
           "gamma_1", "gamma_2", "gamma_mean", "moment_2", "moment_3", "pdi", "skewness",
           "moment_4", "kurtosis", "R-squared"]


def df_ab(x_col="t [s]", y_col="g(2)-1"):
    return {"a": pd.DataFrame({x_col: T4, y_col: Y_A}), "b": pd.DataFrame({x_col: T4, y_col: Y_B})}


def res_ab():
    return [res([100, 110, 1000], 0.9, 0.01), res([500.0], 1.0, 0.009)]


def run_D(dfs, **kw):
    return cd.fit_correlations_method_D(dfs, plot=False, **kw)


def test_oracle_fit_correlations_D_rows_and_columns(monkeypatch):
    """FC-01 [MOCK]"""
    faked(monkeypatch, res_ab())
    out = run_D(df_ab(), gap_threshold=1.5)
    assert list(out.columns) == COLS_K2
    a = out.set_index("filename").loc["a"]
    g = np.array([100.0, 110.0, 1000.0])
    mu = g.mean()
    assert a["n_populations"] == 2
    assert a["gamma_1"] == pytest.approx(104.880885, rel=1e-6)
    assert a["gamma_1"] == pytest.approx(np.sqrt(100 * 110), **DET)
    assert a["gamma_2"] == pytest.approx(1000.0, **DET)
    assert a["gamma_mean"] == pytest.approx(1210 / 3, **DET)
    assert a["moment_2"] == pytest.approx(340700, **DET)
    assert a["moment_3"] == pytest.approx((1e6 + 1.331e6 + 1e9) / 3, **DET)
    assert a["moment_4"] == pytest.approx(np.mean(g ** 4), **DET)
    assert a["pdi"] == pytest.approx(3066300 / 1464100 - 1, **DET)
    assert a["skewness"] == pytest.approx(np.mean((g - mu) ** 3) / np.var(g) ** 1.5, **DET)
    assert a["kurtosis"] == pytest.approx(np.mean((g - mu) ** 4) / np.var(g) ** 2, **DET)  # JADE 3.1
    # JADE 3.1: R2 = fit_r_squared(y, g2_fit, w_rel) from g2_fit (mock: 0.9 everywhere), ordinary R2
    y = np.array(Y_A)
    assert a["R-squared"] == pytest.approx(1 - np.sum((y - 0.9) ** 2) / np.sum((y - y.mean()) ** 2), **DET)
    assert bool(a["weighted"]) is False
    b = out.set_index("filename").loc["b"]
    assert b["n_populations"] == 1
    assert b["gamma_1"] == pytest.approx(500.0, **DET)
    assert np.isnan(b["gamma_2"])
    assert b["pdi"] == 0.0
    assert np.isnan(b["skewness"])  # JADE 3.1
    assert np.isnan(b["kurtosis"])
    y = np.array(Y_B)
    assert b["R-squared"] == pytest.approx(1 - np.sum((y - 1.0) ** 2) / np.sum((y - y.mean()) ** 2), **DET)


def test_oracle_fit_correlations_D_forwards_arguments_and_weights(monkeypatch):
    """FC-02 [MOCK]: no filtering of negative y; weights column forwarded."""
    f = faked(monkeypatch, [res([100.0]), res([100.0])])
    y = [1.0, 0.5, -0.1, -0.2]
    dfs = {"w": pd.DataFrame({"t [s]": T4, "g(2)-1": y, "weight": [1.0, 2.0, 3.0, 4.0]}),
           "u": pd.DataFrame({"t [s]": T4, "g(2)-1": y})}
    out = run_D(dfs, n_max=7, n_start=2).set_index("filename")
    for c in f.calls:
        assert c["n_max"] == 7 and c["n_start"] == 2
        assert np.array_equal(c["x"], T4) and np.array_equal(c["y"], y)
    assert np.array_equal(f.calls[0]["weights"], [1.0, 2.0, 3.0, 4.0])
    assert f.calls[1]["weights"] is None
    assert bool(out.loc["w", "weighted"]) is True
    assert bool(out.loc["u", "weighted"]) is False


def test_oracle_fit_correlations_D_gap_threshold(monkeypatch):
    """FC-03 [MOCK]"""
    faked(monkeypatch, [res([100.0, 250.0])])
    o1 = run_D({"a": df_ab()["a"]}, gap_threshold=1.5).iloc[0]
    assert o1["n_populations"] == 2
    assert o1["gamma_1"] == pytest.approx(100.0, **DET) and o1["gamma_2"] == pytest.approx(250.0, **DET)
    faked(monkeypatch, [res([100.0, 250.0])])
    o3 = run_D({"a": df_ab()["a"]}, gap_threshold=3.0)
    assert o3.iloc[0]["n_populations"] == 1
    assert o3.iloc[0]["gamma_1"] == pytest.approx(np.sqrt(25000), **DET)
    assert o3.iloc[0]["gamma_1"] == pytest.approx(158.113883, rel=1e-6)
    assert "gamma_2" not in o3.columns


def test_oracle_fit_correlations_D_custom_column_names(monkeypatch):
    """FC-04 [MOCK]"""
    faked(monkeypatch, res_ab())
    out = run_D(df_ab("tau", "g2m1"), x_col="tau", y_col="g2m1", gap_threshold=1.5)
    assert list(out.columns) == COLS_K2
    assert out.iloc[0]["gamma_1"] == pytest.approx(np.sqrt(100 * 110), **DET)


def _rows(out):
    return out.set_index("filename").sort_index()


def test_meta_fit_correlations_D_dict_order(monkeypatch):
    """FCM-01 [MOCK]"""
    faked(monkeypatch, res_ab())
    a = run_D(df_ab(), gap_threshold=1.5)
    faked(monkeypatch, res_ab()[::-1])
    d = df_ab()
    b = run_D({"b": d["b"], "a": d["a"]}, gap_threshold=1.5)
    assert set(a.columns) == set(b.columns)
    pd.testing.assert_frame_equal(_rows(a), _rows(b)[_rows(a).columns], check_dtype=False)


def test_meta_fit_correlations_D_duplicate_file(monkeypatch):
    """FCM-02 [MOCK]"""
    faked(monkeypatch, [res([100, 110, 1000]), res([100, 110, 1000])])
    d = df_ab()["a"]
    out = _rows(run_D({"a": d, "a2": d.copy()}, gap_threshold=1.5))
    pd.testing.assert_series_equal(out.loc["a"], out.loc["a2"], check_names=False)


def test_oracle_fit_correlations_D_weighted_r_squared_uses_relative_weights(monkeypatch):
    """JADE 3.1 [MOCK]: R-squared = fit_r_squared(y, g2_fit, w/mean(w)); scale-invariant in w."""
    y = np.array(Y_B)
    w = np.array([4.0, 1.0, 1.0, 2.0])
    wr = w / w.mean()
    ym = np.average(y, weights=wr)
    expected = 1 - np.sum(wr * (y - 1.0) ** 2) / np.sum(wr * (y - ym) ** 2)
    for scale in (1.0, 5.0):
        faked(monkeypatch, [res([500.0], 1.0, 0.009)])
        d = pd.DataFrame({"t [s]": T4, "g(2)-1": y, "weight": scale * w})
        out = run_D({"b": d})
        assert out.iloc[0]["R-squared"] == pytest.approx(expected, **DET)


def test_oracle_fit_cumulant_D_weighted_fit_ss_real_relative_weights():
    """JADE 3.1 [REAL]: fit_ss = sum(w_rel*residuals**2), residual_ss stays unweighted."""
    w = np.linspace(0.5, 2.0, 80)
    r = cd.fit_cumulant_D(XR, R2N, n_start=2, n_max=2, weights=w)
    resid = R2N - r["g2_fit"]
    assert r["fit_ss"] == pytest.approx(np.sum(w / w.mean() * resid ** 2), rel=1e-6)
    assert r["residual_ss"] == pytest.approx(np.sum(resid ** 2), rel=1e-6)


def test_meta_fit_correlations_D_r_squared_scale_invariant(monkeypatch):
    """FCM-03 [MOCK]"""
    c = 0.8
    faked(monkeypatch, [res([100.0], rss=0.01)])
    a = run_D({"a": df_ab()["a"]}).iloc[0]["R-squared"]
    r_c = res([100.0], rss=0.01 * c ** 2)
    r_c["g2_fit"] = c * r_c["g2_fit"]  # JADE 3.1: R2 is computed from g2_fit
    faked(monkeypatch, [r_c])
    d = pd.DataFrame({"t [s]": T4, "g(2)-1": c * np.array(Y_A)})
    b = run_D({"a": d}).iloc[0]["R-squared"]
    assert b == pytest.approx(a, **DET)


def test_meta_fit_correlations_D_gamma_scaling(monkeypatch):
    """FCM-04 [MOCK]"""
    c = 7.3
    g = np.array([100.0, 110.0, 1000.0])
    faked(monkeypatch, [res(g)])
    a = run_D({"a": df_ab()["a"]}, gap_threshold=1.5).iloc[0]
    faked(monkeypatch, [res(g * c)])
    b = run_D({"a": df_ab()["a"]}, gap_threshold=1.5).iloc[0]
    for k in ("gamma_1", "gamma_2", "gamma_mean"):
        assert b[k] == pytest.approx(a[k] * c, **DET)
    for j, k in ((2, "moment_2"), (3, "moment_3"), (4, "moment_4")):
        assert b[k] == pytest.approx(a[k] * c ** j, **DET)
    for k in ("pdi", "skewness", "kurtosis", "n_populations"):
        assert b[k] == pytest.approx(a[k], **DET)


def test_edge_fit_correlations_D_empty_dict():
    """FCE-01 [REAL]"""
    out = cd.fit_correlations_method_D({}, plot=False)
    assert len(out) == 0
    assert len(out.columns) == 0


def _fce02(monkeypatch):
    f = faked(monkeypatch, [res([100.0]), RuntimeError("All fits failed. Check input data quality.")])
    ok = pd.DataFrame({"t [s]": T4, "g(2)-1": Y_A})
    bad = pd.DataFrame({"other": T4})
    return run_D({"ok": ok, "bad": bad, "fail": ok.copy()}), f


def test_char_fit_correlations_D_error_isolation_documents_current_behavior(monkeypatch):
    """FCE-02 [MOCK]: documents current behavior, not desired behavior (F-12)."""
    out, f = _fce02(monkeypatch)
    assert list(out["filename"]) == ["ok", "bad", "fail"]
    r = out.set_index("filename")
    assert r.loc["bad", "Error"] == "'t [s]'"
    assert pd.isna(r.loc["bad", "n_modes"]) and pd.isna(r.loc["bad", "R-squared"])
    assert r.loc["fail", "Error"] == "All fits failed. Check input data quality."
    assert pd.isna(r.loc["fail", "weighted"])
    assert pd.isna(r.loc["ok", "Error"])
    assert len(f.calls) == 2


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="FINDING F-12, siehe _dev/notes/test_spec_cumulants_D.md")
def test_finding_fit_correlations_D_failed_row_has_weighted_false(monkeypatch):
    """X-03 [MOCK]: JADE parity: failed row carries weighted == False."""
    out, _ = _fce02(monkeypatch)
    r = out.set_index("filename")
    assert r.loc["fail", "weighted"] == False  # noqa: E712


def test_char_fit_correlations_D_constant_y_r_squared_zero_documents_current_behavior(monkeypatch):
    """FCE-03 [MOCK]: documents current behavior, not desired behavior."""
    faked(monkeypatch, [res([100.0], rss=0.0)])
    d = pd.DataFrame({"t [s]": T4, "g(2)-1": [0.5] * 4})
    out = run_D({"c": d})
    assert out.iloc[0]["R-squared"] == 0


def test_char_fit_correlations_D_late_gamma_column_appended_documents_current_behavior(monkeypatch):
    """FCE-04 [MOCK]: documents current behavior, not desired behavior."""
    faked(monkeypatch, [res([100.0]), res([100.0, 1000.0])])
    out = run_D(df_ab(), gap_threshold=1.5)
    assert out.columns[-1] == "gamma_2"
    assert out.columns[out.columns.get_loc("R-squared")] == "R-squared"
    assert list(out.columns).index("gamma_2") > list(out.columns).index("R-squared")


def test_char_fit_correlations_D_loses_multiplicity_documents_current_behavior(monkeypatch):
    """FCE-05 [MOCK]: documents current behavior (F-13): amplitude shares are not reported."""
    faked(monkeypatch, [res([100, 100, 100, 1000]), res([100, 1000, 1000, 1000])])
    out = run_D(df_ab(), gap_threshold=1.5)
    a, b = out.iloc[0], out.iloc[1]
    assert a["gamma_1"] == pytest.approx(b["gamma_1"], **DET)
    assert a["gamma_2"] == pytest.approx(b["gamma_2"], **DET)
    assert a["n_populations"] == b["n_populations"] == 2
    assert a["gamma_mean"] != pytest.approx(b["gamma_mean"], rel=1e-3)


# ----------------------------------------------------------------------------
# 7.5 integration with cluster_all_gammas
# ----------------------------------------------------------------------------


def _fc09_data(monkeypatch):
    from ade_dls.analysis import clustering as cl  # noqa: F401
    q2 = [1.0, 2.0, 3.0, 4.0]
    names = [f"f{i}" for i in range(4)]
    faked(monkeypatch, [res([10.0 * q, 1000.0 * q]) for q in q2])
    dfs = {n: pd.DataFrame({"t [s]": T4, "g(2)-1": Y_A}) for n in names}
    fit = run_D(dfs, gap_threshold=3.0)
    base = pd.DataFrame({"filename": names, "q^2": q2})
    data = pd.merge(base, fit, on="filename", how="outer").reset_index(drop=True)
    data.index = data.index + 1
    return data


def _gamma_cols(data):
    return [c for c in data.columns if c.startswith("gamma_") and c != "gamma_mean"]


def _cluster(data, cols):
    from ade_dls.analysis import clustering as cl
    return cl.cluster_all_gammas(data, cols, q_squared_col="q^2", normalize_by_q2=True,
                                 distance_threshold=2.0, min_abundance=0.3,
                                 clustering_strategy="silhouette_refined", plot=False,
                                 interactive=False)


def test_oracle_fit_to_cluster_all_gammas_two_populations(monkeypatch):
    """FC-09 [MOCK]: fit mocked, clustering real."""
    data = _fc09_data(monkeypatch)
    cols = _gamma_cols(data)
    assert cols == ["gamma_1", "gamma_2"]
    clustered, info = _cluster(data, cols)
    assert len(info["population_D_means"]) == 2
    assert sorted(info["population_D_means"]) == pytest.approx([10.0, 1000.0], **DET)
    # JADE 3.1: populations numbered DESCENDING by mean log10 D (pop1 = fastest = smallest particle)
    assert info["population_D_means"] == pytest.approx([1000.0, 10.0], **DET)
    assert list(info["population_abundances"]) == pytest.approx([1.0, 1.0], **DET)
    assert clustered["gamma_pop1"].to_numpy(float) == pytest.approx(data["gamma_2"].to_numpy(float), **DET)
    assert clustered["gamma_pop2"].to_numpy(float) == pytest.approx(data["gamma_1"].to_numpy(float), **DET)


def test_char_fit_to_cluster_all_gammas_gamma_mean_column_raises_documents_current_behavior(monkeypatch):
    """FC-09b [MOCK]: documents current behavior (naming convention gamma_<int>)."""
    data = _fc09_data(monkeypatch)
    cols = [c for c in data.columns if c.startswith("gamma_")]
    assert "gamma_mean" in cols
    with pytest.raises(ValueError):
        _cluster(data, cols)


def test_oracle_fit_correlations_D_end_to_end_real():
    """FC-10 [REAL]"""
    dfs = {"r1": pd.DataFrame({"t [s]": XR, "g(2)-1": R1}),
           "r2": pd.DataFrame({"t [s]": XR, "g(2)-1": R2})}
    out = run_D(dfs, n_max=3, gap_threshold=3.0).set_index("filename")
    assert out.loc["r1", "n_modes"] in (1, 2)
    assert out.loc["r1", "n_populations"] == 1
    assert out.loc["r2", "n_modes"] == 2
    assert out.loc["r2", "n_populations"] == 2
    assert out.loc["r2", "gamma_1"] == pytest.approx(500.0, **OPT)
    assert out.loc["r2", "gamma_2"] == pytest.approx(20000.0, **OPT)
    assert (out["R-squared"] > 0.999999).all()


# ----------------------------------------------------------------------------
# findings (xfail strict)
# ----------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="FINDING F-02, siehe _dev/notes/test_spec_cumulants_D.md")
def test_finding_fit_cumulant_D_p0_matches_bounds_after_failed_fit(monkeypatch):
    """X-01 [MOCK]: after one failed fit, p0 and bounds must still agree in length."""
    s, _ = _fc_c1_run(monkeypatch)
    for c in s.calls:
        assert len(c["p0"]) == len(c["lb"])


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="FINDING F-04, Soll offen, siehe _dev/notes/open_discussion_points.md")
def test_finding_fit_cumulant_D_model_order_independent_of_units(monkeypatch):
    """X-02 [MOCK]: unit change must not change the number of fits (target open)."""
    na, nb, _ = _fc_c3_calls(monkeypatch)
    assert na == nb


# ----------------------------------------------------------------------------
# 8. GUI wrapper _fit_single_method_d [MOCK]
# ----------------------------------------------------------------------------


def test_oracle_gui_wrapper_keeps_negative_tail(monkeypatch):
    """GUI-01 [MOCK] (regression CHANGELOG 3.3.2)"""
    x = np.logspace(-6, -1, 10)
    y = np.array([1, .8, .5, .3, .1, .02, -.01, -.02, .005, -.003])
    f = faked(monkeypatch, [res([100.0, 1000.0], rss=0.01)])
    name, data, err = _fit_single_method_d("n", x, y, 5, 1, 3.0)
    assert err is None and name == "n"
    assert len(f.calls[0]["y"]) == 10
    assert np.array_equal(f.calls[0]["x"], x) and np.array_equal(f.calls[0]["y"], y)
    assert np.array_equal(data["x_fit"], x) and np.array_equal(data["y_fit"], y)


def test_edge_gui_wrapper_too_few_points(monkeypatch):
    """GUI-02 [MOCK]"""
    f = faked(monkeypatch, [])
    out = _fit_single_method_d("n", np.arange(1.0, 5.0), np.ones(4), 5, 1, 3.0)
    assert out == ("n", None, "Too few data points")
    assert f.calls == []


def test_edge_gui_wrapper_fit_error_is_returned(monkeypatch):
    """GUI-03 [MOCK]"""
    faked(monkeypatch, [RuntimeError("boom")])
    out = _fit_single_method_d("n", np.arange(1.0, 7.0), np.ones(6), 5, 1, 3.0)
    assert out == ("n", None, "boom")


@pytest.mark.parametrize("weights", [np.array([1.0, 2, 3, 4, 5, 6]), None])
def test_oracle_gui_wrapper_weights_forwarded(monkeypatch, weights):
    """GUI-04 [MOCK]: weights forwarded; constant y gives r_squared == 0.0."""
    f = faked(monkeypatch, [res([100.0], rss=0.0)])
    name, data, err = _fit_single_method_d("n", np.arange(1.0, 7.0), np.full(6, 0.5), 5, 1, 3.0,
                                           weights=weights)
    assert err is None
    if weights is None:
        assert f.calls[0]["weights"] is None
    else:
        assert np.array_equal(f.calls[0]["weights"], weights)
    assert data["weighted"] is (weights is not None)
    assert data["r_squared"] == 0.0
