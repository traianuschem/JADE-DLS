# -*- coding: utf-8 -*-
"""Tests for ade_dls/analysis/cumulants_C.py (spec: _dev/notes/test_spec_cumulants_C.md).

Categories: test_oracle_* (closed-form expectations), test_meta_* (metamorphic),
test_edge_* (edge cases), test_char_* (characterization: documents current
behavior, not desired behavior), test_finding_* (xfail strict, known findings).

Case markers: [REAL] real small fits / pure functions, [WRAP] passive recording
wrapper around cc.curve_fit (delegates unchanged), [MOCK] patched behavior.
"""
import functools
import inspect

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import scipy.optimize  # noqa: E402

from ade_dls.analysis import cumulants_C as cc  # noqa: E402

_ORIG_CURVE_FIT = scipy.optimize.curve_fit


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# ----------------------------------------------------------------------------
# model functions (verbatim from gui/analysis/cumulant_analyzer.py:922-938)
# ----------------------------------------------------------------------------
def fit_function1(x, a, b, f):
    return f + a * np.exp(-2 * b * x)


def fit_function2(x, a, b, c, f):
    inner_term = 1 + 0.5 * c * x**2
    return f + a * (np.exp(-b * x) * inner_term)**2


def fit_function3(x, a, b, c, d, f):
    inner_term = 1 + 0.5 * c * x**2 - (d * x**3) / 6
    return f + a * (np.exp(-b * x) * inner_term)**2


def fit_function4(x, a, b, c, d, e, f):
    inner_term = 1 + 0.5 * c * x**2 - (d * x**3) / 6 + (e * x**4) / 24
    return f + a * (np.exp(-b * x) * inner_term)**2


FF = {1: fit_function1, 2: fit_function2, 3: fit_function3, 4: fit_function4}
PN = {n: inspect.getfullargspec(FF[n]).args[1:] for n in FF}

TAU = np.logspace(-6, 0, 121)
FIT_ALL = (1e-9, 10.0)
SEED = 20261103
NOISE = np.random.default_rng(SEED).normal(0.0, 1e-4, TAU.size)
BASE = [0.8, 10000, 0, 0, 0, 0]
TRUE = {1: dict(a=0.8, b=1000.0, f=1e-3),
        2: dict(a=0.8, b=1000.0, c=1e5, f=1e-3),
        3: dict(a=0.8, b=1000.0, c=1e5, d=1.5e7, f=1e-3),
        4: dict(a=0.8, b=1000.0, c=1e5, d=1.5e7, e=3e10, f=1e-3)}
GUESS = {1: [0.7, 1300.0, 0, 0, 0, 0.0],
         2: [0.7, 1300.0, 5e4, 0, 0, 0.0],
         3: [0.7, 1300.0, 5e4, 0.0, 0, 0.0],
         4: [0.7, 1300.0, 5e4, 0.0, 1e10, 0.0]}
W = np.linspace(1.0, 3.0, TAU.size)
SIG_W = 1.0 / np.sqrt(W / np.mean(W))

REL3 = dict(rel=1e-3)
TOL = dict(rel=1e-6, abs=1e-12)


def ptrue(n):
    return [TRUE[n][p] for p in PN[n]]


def y_model(n, t=TAU):
    return FF[n](t, *ptrue(n))


def make_df(t, y, w=None):
    d = {"t [s]": np.asarray(t, float), "g(2)-1": np.asarray(y, float)}
    if w is not None:
        d["weight"] = np.asarray(w, float)
    return pd.DataFrame(d)


def run(dfs, n, guesses, **kw):
    kw.setdefault("show_plots", False)
    return cc.plot_processed_correlations_iterative(dfs, FF[n], kw.pop("limits", FIT_ALL), guesses, **kw)


def adaptive(dfs, n, base=BASE):
    return cc.get_adaptive_initial_parameters(dfs, FF[n], list(base), verbose=False)


def metrics(n, x, y, p, w=None):
    yc = FF[n](x, *p)
    res = y - yc
    if w is None:
        ss_res = np.sum(res**2)
        ss_tot = np.sum((y - np.mean(y))**2)
    else:
        sig = 1.0 / np.sqrt(w / np.mean(w))
        W_ = sig**-2
        ss_res = np.sum(W_ * res**2)
        ss_tot = np.sum(W_ * (y - np.average(y, weights=W_))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    uss = np.sum(res**2)
    m = len(y)
    return r2, np.sqrt(uss / m), m * np.log(uss / m) + 2 * (len(p) + 1)


# datasets
K = {n: make_df(TAU, y_model(n)) for n in FF}
KN = {n: make_df(TAU, y_model(n) + NOISE) for n in FF}
KW = make_df(TAU, y_model(2) + NOISE, W)

FB1x = np.array([1e-4, 1e-3, 1e-2, 1e-1])
FB1y = np.array([2.0, 1.0, 0.5, 0.2])
FB2y = np.array([0.8, 0.5, np.nan, 0.05])
FB3x = np.logspace(-9, -5, 5)
FB3y = np.array([0.8, 0.79, 0.7, 0.3, 0.0])
dfFB1 = make_df(FB1x, FB1y)
K1x15 = make_df(TAU, 1.5 * y_model(1))
G20k = make_df(TAU, 1e-3 + 0.8 * np.exp(-2 * 2e4 * TAU))
G15k = make_df(TAU, 1e-3 + 0.8 * np.exp(-2 * 1.5e4 * TAU))
_k, _th = 20, 50.0
SCH = make_df(TAU, 1e-3 + 0.8 * ((1 + _th * TAU)**(-_k))**2)


def get(res, name):
    return res[res["filename"] == name].iloc[0]


def row_equal(a, b):
    pd.testing.assert_series_equal(a, b, check_exact=True, check_names=False)


def assert_params(r, n, ref, prefix="best_", rel=1e-3):
    for p in PN[n]:
        if p == "f":
            assert r[prefix + p] == pytest.approx(ref[p], rel=rel, abs=1e-6), p
        else:
            assert r[prefix + p] == pytest.approx(ref[p], rel=rel), p


def row_params(r, n, prefix="best_"):
    return [r[prefix + p] for p in PN[n]]


def install_rec(monkeypatch, calls, fake=None):
    """Passive recorder around cc.curve_fit. fake(entry, model_idx) may return an
    override result (no delegation) or raise; None delegates unchanged."""
    def rec(f, xdata, ydata, **kw):
        entry = dict(f=f, p0=list(kw["p0"]), maxfev=kw.get("maxfev"), method=kw.get("method"),
                     bounds=kw.get("bounds"), sigma=kw.get("sigma"), out=None)
        calls.append(entry)
        if fake is not None and f is not cc._simple_exp:
            idx = sum(1 for c in calls if c["f"] is not cc._simple_exp)
            r = fake(entry, idx)
            if r is not None:
                entry["out"] = r
                return r
        try:
            out = _ORIG_CURVE_FIT(f, xdata, ydata, **kw)
        except BaseException:
            entry["out"] = None
            raise
        entry["out"] = out
        return out
    monkeypatch.setattr(cc, "curve_fit", rec)
    return calls


def models(calls):
    return [c for c in calls if c["f"] is not cc._simple_exp]


# ============================================================================
# 3. get_meaningful_parameters
# ============================================================================
def test_oracle_get_meaningful_parameters_subsets():
    L = [1, 2, 3, 4, 5, 6]
    exp = {1: [1, 2, 6], 2: [1, 2, 3, 6], 3: [1, 2, 3, 4, 6], 4: [1, 2, 3, 4, 5, 6]}
    for n in FF:
        r = cc.get_meaningful_parameters(FF[n], L)
        assert r == exp[n]
        assert type(r) is list


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_meta_get_meaningful_parameters_consistent_with_param_names(n):
    L = [11., 12., 13., 14., 15., 16.]
    assert cc.get_meaningful_parameters(FF[n], L) == [L["abcdef".index(p)] for p in PN[n]]


@pytest.mark.parametrize("conv", [tuple, np.array])
def test_meta_get_meaningful_parameters_accepts_tuple_and_array(conv):
    L = [11., 12., 13., 14., 15., 16.]
    for n in FF:
        r = cc.get_meaningful_parameters(FF[n], conv(L))
        assert type(r) is list
        assert [float(v) for v in r] == [L["abcdef".index(p)] for p in PN[n]]


@pytest.mark.parametrize("length", [5, 7])
def test_edge_get_meaningful_parameters_wrong_length_raises(length):
    with pytest.raises(ValueError, match="unpack"):
        cc.get_meaningful_parameters(fit_function2, list(range(length)))


def test_edge_get_meaningful_parameters_unknown_function_raises():
    def fit_function5(x, a, b, f):
        return 0

    with pytest.raises(ValueError, match="Unknown fit function: fit_function5"):
        cc.get_meaningful_parameters(fit_function5, [1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="Unknown fit function: <lambda>"):
        cc.get_meaningful_parameters(lambda x, a, b, f: 0, [1, 2, 3, 4, 5, 6])


def test_edge_get_meaningful_parameters_partial_has_no_name():
    with pytest.raises(AttributeError):
        cc.get_meaningful_parameters(functools.partial(fit_function2), [1, 2, 3, 4, 5, 6])


def test_char_get_meaningful_parameters_dispatch_by_name_documents_current_behavior():
    """Documents current behavior: dispatch uses __name__ only, signature is ignored."""
    def fit_function2(x, a, b, c, d, e, f):  # noqa: F811 (6 params, name "2")
        return 0

    assert cc.get_meaningful_parameters(fit_function2, [1, 2, 3, 4, 5, 6]) == [1, 2, 3, 6]


# ============================================================================
# 4. _simple_exp / _simple_exp_prefit
# ============================================================================
def test_oracle_simple_exp_formula():
    assert cc._simple_exp(0.0, 0.8, 2000.0, 0.001) == pytest.approx(0.801, **TOL)
    assert cc._simple_exp(5e-4, 0.8, 2000.0, 0.001) == pytest.approx(0.8 * np.exp(-1) + 0.001, **TOL)
    assert cc._simple_exp(5e-4, 0.8, 2000.0, 0.001) == pytest.approx(0.2953036, rel=1e-6)
    np.testing.assert_allclose(cc._simple_exp(TAU, 0.8, 2000.0, 0.001),
                               0.8 * np.exp(-2000.0 * TAU) + 0.001, rtol=1e-6, atol=1e-12)


def test_oracle_prefit_k1_recovers_double_gamma():
    a, b, f = cc._simple_exp_prefit(TAU, y_model(1))
    assert a == pytest.approx(0.8, **REL3)
    assert b == pytest.approx(2000.0, **REL3)
    assert f == pytest.approx(0.001, abs=1e-6)


def test_oracle_prefit_fallback_fb1():
    assert cc._simple_exp_prefit(FB1x, FB1y) == pytest.approx((1.0, 100.0, 0.1), **TOL)


def test_oracle_prefit_fallback_fb2_nan_in_y():
    """[REAL] ZV: NaN in y makes curve_fit raise ValueError -> fallback."""
    assert cc._simple_exp_prefit(FB1x, FB2y) == pytest.approx((0.75, 100.0, 0.05), **TOL)


def test_oracle_prefit_fallback_fb3_b_above_bound():
    assert cc._simple_exp_prefit(FB3x, FB3y) == pytest.approx((0.8, 1e6, 0.0), **TOL)


@pytest.mark.parametrize("s", [10.0, 0.1])
def test_meta_prefit_lag_time_scaling(s):
    a, b, f = cc._simple_exp_prefit(TAU * s, y_model(1))
    assert a == pytest.approx(0.8, **REL3)
    assert b == pytest.approx(2000.0 / s, **REL3)
    assert f == pytest.approx(0.001, abs=1e-6)


def test_meta_prefit_amplitude_scaling():
    a, b, f = cc._simple_exp_prefit(TAU, 0.5 * y_model(1))
    assert a == pytest.approx(0.4, **REL3)
    assert b == pytest.approx(2000.0, **REL3)
    assert f == pytest.approx(0.0005, rel=1e-3, abs=1e-6)


def test_meta_prefit_baseline_shift():
    a, b, f = cc._simple_exp_prefit(TAU, y_model(1) + 0.02)
    assert a == pytest.approx(0.8, **REL3)
    assert b == pytest.approx(2000.0, **REL3)
    assert f == pytest.approx(0.021, rel=1e-3, abs=1e-6)


def test_meta_prefit_returns_tuple_of_python_floats():
    for x, y in [(TAU, y_model(1)), (FB1x, FB1y), (FB1x, FB2y), (FB3x, FB3y)]:
        r = cc._simple_exp_prefit(x, y)
        assert type(r) is tuple and len(r) == 3
        assert all(type(v) is float for v in r)


def test_edge_prefit_empty_arrays_raise_indexerror():
    with pytest.raises(IndexError):
        cc._simple_exp_prefit(np.array([]), np.array([]))


def test_edge_prefit_single_point_fallback():
    r = cc._simple_exp_prefit(np.array([1e-3]), np.array([0.5]))
    assert r == pytest.approx((0.01, 1000.0, 0.1), **TOL)


def test_edge_prefit_zero_lag_time_gives_b_upper_bound():
    x = FB1x.copy()
    x[2] = 0.0
    r = cc._simple_exp_prefit(x, FB1y)
    assert r == pytest.approx((1.0, 1e6, 0.1), **TOL)


def test_edge_prefit_nan_last_value_gives_nan_a_f():
    y = FB1y.copy()
    y[-1] = np.nan
    a, b, f = cc._simple_exp_prefit(FB1x, y)
    assert np.isnan(a) and np.isnan(f)
    assert b == pytest.approx(100.0, **TOL)


def test_char_prefit_depends_on_point_order_documents_current_behavior():
    """Documents current behavior, not desired behavior: reversed points give another fallback."""
    r = cc._simple_exp_prefit(FB1x[::-1].copy(), FB1y[::-1].copy())
    assert r == pytest.approx((0.01, 1000.0, 0.1), **TOL)
    assert r != pytest.approx((1.0, 100.0, 0.1), **TOL)


def test_char_prefit_silent_fallback_for_amplitude_above_one_documents_current_behavior():
    """Documents current behavior (F-09): a > 1 -> silent fallback b = 1/x_mid."""
    r = cc._simple_exp_prefit(TAU, 1.5 * y_model(1))
    assert r == pytest.approx((1.0, 1000.0, 0.0015), **TOL)


def test_char_prefit_factor_two_documents_current_behavior():
    """Documents current behavior (F-01, JADE-identical): b_pre = 2*Gamma."""
    _, b, _ = cc._simple_exp_prefit(TAU, y_model(1))
    assert b / 1000.0 == pytest.approx(2.0, **REL3)


def test_oracle_prefit_call_constants_wrapper(monkeypatch):
    """[WRAP] constants of the pre-fit curve_fit call."""
    calls = install_rec(monkeypatch, [])
    cc.estimate_parameters_from_data(TAU, y_model(1), BASE)
    assert len(calls) == 1
    c = calls[0]
    assert c["f"] is cc._simple_exp
    assert c["method"] == "trf"
    assert c["bounds"] == ([0.0, 0.0, -0.1], [1.0, 1e6, 0.1])
    assert c["maxfev"] == 5000


# ============================================================================
# 5. estimate_parameters_from_data
# ============================================================================
def _approx_dict(d, ref):
    assert list(d) == list(ref)
    for k in ref:
        assert d[k] == pytest.approx(ref[k], **TOL), k


def test_oracle_estimate_parameters_fb1():
    r = cc.estimate_parameters_from_data(FB1x, FB1y, BASE)
    _approx_dict(r, dict(a=1.0, b=100.0, c=500.0, d=1e4, e=7.5e5, f=0.1))


def test_oracle_estimate_parameters_fb3():
    r = cc.estimate_parameters_from_data(FB3x, FB3y, BASE)
    _approx_dict(r, dict(a=0.8, b=1e6, c=5e10, d=1e16, e=7.5e21, f=0.0))


def test_oracle_estimate_parameters_k1_relations_exact_from_b():
    r = cc.estimate_parameters_from_data(TAU, y_model(1), BASE)
    assert r["a"] == pytest.approx(0.8, **REL3)
    assert r["b"] == pytest.approx(2000.0, **REL3)
    assert r["f"] == pytest.approx(0.001, abs=1e-6)
    assert r["c"] == pytest.approx(0.05 * r["b"]**2, **TOL)
    assert r["d"] == pytest.approx(0.01 * r["b"]**3, **TOL)
    assert r["e"] == pytest.approx(3 * r["c"]**2, **TOL)
    assert r["c"] == pytest.approx(0.2 * 1000.0**2, rel=2e-3)


def test_meta_estimate_parameters_lag_time_scaling():
    r = cc.estimate_parameters_from_data(TAU * 10, y_model(1), BASE)
    assert r["b"] == pytest.approx(200.0, **REL3)
    assert r["c"] == pytest.approx(0.05 * r["b"]**2, **TOL)
    assert r["d"] == pytest.approx(0.01 * r["b"]**3, **TOL)
    assert r["e"] == pytest.approx(3 * r["c"]**2, **TOL)


def test_meta_estimate_parameters_key_order_and_float_values():
    for x, y in [(FB1x, FB1y), (TAU, y_model(1))]:
        r = cc.estimate_parameters_from_data(x, y, BASE)
        assert list(r) == ["a", "b", "c", "d", "e", "f"]
        assert all(isinstance(v, float) for v in r.values())


def test_edge_estimate_parameters_empty_arrays_use_base(capsys):
    e = np.array([])
    r = cc.estimate_parameters_from_data(e, e, [])
    _approx_dict(r, dict(a=1.0, b=1000.0, c=100.0, d=1.0, e=30000.0, f=0.0))
    r = cc.estimate_parameters_from_data(e, e, [0.5, 200.0])
    _approx_dict(r, dict(a=0.5, b=200.0, c=100.0, d=1.0, e=30000.0, f=0.0))
    r = cc.estimate_parameters_from_data(e, e, BASE)
    _approx_dict(r, dict(a=0.8, b=10000.0, c=0.0, d=0.0, e=0.0, f=0.0))
    assert "Parameter estimation failed" in capsys.readouterr().out


def test_edge_estimate_parameters_prefit_exception_uses_base(monkeypatch, capsys):
    def boom(x, y):
        raise RuntimeError("boom")
    monkeypatch.setattr(cc, "_simple_exp_prefit", boom)
    r = cc.estimate_parameters_from_data(TAU, y_model(1), BASE)
    _approx_dict(r, dict(a=0.8, b=10000.0, c=0.0, d=0.0, e=0.0, f=0.0))
    assert "(boom)" in capsys.readouterr().out


def _fake_prefit(monkeypatch, val):
    monkeypatch.setattr(cc, "_simple_exp_prefit", lambda x, y: val)


def test_char_estimate_parameters_priors_from_unclipped_b_documents_current_behavior(monkeypatch):
    """Documents current behavior (F-10): b clipped to 1 but c, d, e derived from b=0.5. [MOCK]"""
    _fake_prefit(monkeypatch, (0.5, 0.5, 0.0))
    r = cc.estimate_parameters_from_data(TAU, y_model(1), BASE)
    _approx_dict(r, dict(a=0.5, b=1.0, c=0.0125, d=0.00125, e=4.6875e-4, f=0.0))


def test_char_estimate_parameters_amplitude_lower_clip_documents_current_behavior(monkeypatch):
    """Documents current behavior (F-10). [MOCK]"""
    _fake_prefit(monkeypatch, (0.001, 100.0, 0.0))
    r = cc.estimate_parameters_from_data(TAU, y_model(1), BASE)
    _approx_dict(r, dict(a=0.01, b=100.0, c=500.0, d=1e4, e=7.5e5, f=0.0))


def test_char_estimate_parameters_upper_clips_documents_current_behavior(monkeypatch):
    """Documents current behavior (F-10). [MOCK]"""
    _fake_prefit(monkeypatch, (0.5, 2e6, 0.0))
    r = cc.estimate_parameters_from_data(TAU, y_model(1), BASE)
    _approx_dict(r, dict(a=0.5, b=1e6, c=2e11, d=8e16, e=1.2e23, f=0.0))


def test_char_estimate_parameters_wider_clip_than_prefit_bounds_documents_current_behavior(monkeypatch):
    """Documents current behavior (F-10). [MOCK]"""
    _fake_prefit(monkeypatch, (20.0, 100.0, 0.5))
    r = cc.estimate_parameters_from_data(TAU, y_model(1), BASE)
    assert r["a"] == pytest.approx(10.0, **TOL)
    assert r["f"] == pytest.approx(0.5, **TOL)
    assert r["c"] == pytest.approx(500.0, **TOL)


# ============================================================================
# 6. get_adaptive_initial_parameters
# ============================================================================
def test_oracle_adaptive_parameters_fb1_all_orders():
    exp = {1: [1.0, 100.0, 0.1], 2: [1.0, 100.0, 500.0, 0.1],
           3: [1.0, 100.0, 500.0, 1e4, 0.1], 4: [1.0, 100.0, 500.0, 1e4, 7.5e5, 0.1]}
    for n in FF:
        r = adaptive({"A": dfFB1}, n)
        assert r["A"] == pytest.approx(exp[n], **TOL)


def test_oracle_adaptive_parameters_k1_ff2():
    r = adaptive({"K1": K[1]}, 2)["K1"]
    assert r[0] == pytest.approx(0.8, **REL3)
    assert r[1] == pytest.approx(2000.0, **REL3)
    assert r[2] == pytest.approx(0.05 * r[1]**2, **TOL)
    assert r[3] == pytest.approx(0.001, abs=1e-6)


def test_meta_adaptive_parameters_order_and_single_vs_joint():
    both = adaptive({"K1": K[1], "K2N": KN[2]}, 2)
    rev = adaptive({"K2N": KN[2], "K1": K[1]}, 2)
    assert list(both) == ["K1", "K2N"] and list(rev) == ["K2N", "K1"]
    assert both["K1"] == adaptive({"K1": K[1]}, 2)["K1"] == rev["K1"]
    assert both["K2N"] == adaptive({"K2N": KN[2]}, 2)["K2N"] == rev["K2N"]


def test_meta_adaptive_parameters_strategy_ignored(capsys):
    ref = adaptive({"K1": K[1]}, 2)
    capsys.readouterr()
    for strat in ["individual", None, "global", "representative"]:
        r = cc.get_adaptive_initial_parameters({"K1": K[1]}, FF[2], list(BASE), strategy=strat, verbose=False)
        out = capsys.readouterr().out
        assert r == ref
        assert ("deprecated" in out) == (strat in ("global", "representative"))


def test_meta_adaptive_parameters_does_not_mutate_base():
    base = list(BASE)
    r = cc.get_adaptive_initial_parameters({"K1": K[1]}, FF[2], base, verbose=False)
    assert base == BASE
    assert r["K1"] is not base


def test_meta_adaptive_parameters_custom_columns():
    df = K[1].rename(columns={"t [s]": "tau", "g(2)-1": "g2m1"})
    r = cc.get_adaptive_initial_parameters({"K1": df}, FF[2], list(BASE), x_col="tau", y_col="g2m1",
                                           verbose=False)
    assert r == adaptive({"K1": K[1]}, 2)


def test_edge_adaptive_parameters_empty_dict():
    assert adaptive({}, 2) == {}


def test_edge_adaptive_parameters_missing_column_falls_back(capsys):
    r = adaptive({"X": K[1].drop(columns="t [s]")}, 2)
    assert r["X"] == cc.get_meaningful_parameters(FF[2], BASE)
    assert "Failed to adapt parameters" in capsys.readouterr().out


def test_edge_adaptive_parameters_zero_rows_falls_back(capsys):
    r = adaptive({"X": K[1].iloc[0:0]}, 2)
    assert r["X"] == cc.get_meaningful_parameters(FF[2], BASE)
    assert "Parameter estimation failed" in capsys.readouterr().out


def test_edge_adaptive_parameters_base_too_short_raises():
    with pytest.raises(ValueError, match="unpack"):
        cc.get_adaptive_initial_parameters({"K1": K[1]}, FF[2], BASE[:5], verbose=False)


def test_edge_adaptive_parameters_unknown_function_raises():
    def fit_function5(x, a, b, c, f):
        return 0
    with pytest.raises(ValueError, match="Unknown fit function"):
        cc.get_adaptive_initial_parameters({"K1": K[1]}, fit_function5, list(BASE), verbose=False)


def test_edge_adaptive_parameters_numpy_base_equals_list_base():
    a = cc.get_adaptive_initial_parameters({"K1": K[1]}, FF[2], np.array(BASE, dtype=float), verbose=False)
    b = adaptive({"K1": K[1]}, 2)
    assert a["K1"] == pytest.approx(b["K1"], **TOL)


def test_char_adaptive_parameters_tuple_base_silently_unchanged_documents_current_behavior():
    """Documents current behavior (F-11): tuple base has no .copy() -> silent base fallback."""
    r = cc.get_adaptive_initial_parameters({"K1": K[1]}, FF[2], tuple(BASE), verbose=False)
    assert r["K1"] == cc.get_meaningful_parameters(FF[2], BASE)


def test_char_adaptive_parameters_int_array_truncates_documents_current_behavior():
    """Documents current behavior (F-11, ZV): int base array truncates f=0.1 to 0."""
    r = cc.get_adaptive_initial_parameters({"A": dfFB1}, FF[4], np.array([1, 10000, 0, 0, 0, 0]),
                                           verbose=False)
    assert list(r["A"]) == pytest.approx([1, 100, 500, 10000, 750000, 0], **TOL)


def test_char_adaptive_parameters_verbose_output_documents_current_behavior(capsys):
    """Documents current behavior."""
    cc.get_adaptive_initial_parameters({"K1": K[1]}, FF[2], list(BASE), verbose=True)
    out = capsys.readouterr().out
    assert "Estimating adaptive parameters for fit_function2" in out
    assert "Generated adaptive parameters for 1 datasets" in out
    cc.get_adaptive_initial_parameters({"K1": K[1]}, FF[2], list(BASE), verbose=False)
    assert capsys.readouterr().out == ""


def test_char_adaptive_parameters_use_whole_curve_documents_current_behavior():
    """Documents current behavior (F-08): start values ignore the fit window."""
    tx = np.logspace(0.2, 2, 20)
    ext = pd.concat([K[1], make_df(tx, np.full(20, 0.05))], ignore_index=True)
    a = adaptive({"K1": K[1]}, 2)["K1"]
    b = adaptive({"K1": ext}, 2)["K1"]
    assert a != b


# ============================================================================
# 7.1 Oracles
# ============================================================================
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_oracle_adaptive_noise_free_recovers_true_parameters(n):
    d = {"Kn": K[n]}
    r = get(run(d, n, adaptive(d, n)), "Kn")
    assert_params(r, n, TRUE[n])
    assert r["best_R-squared"] >= 1 - 1e-10
    assert r["weighted"] is False or r["weighted"] == False  # noqa: E712
    if n == 4:
        assert r["kurtosis"] == pytest.approx(3.0, rel=5e-3)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_oracle_expert_noise_free_recovers_true_parameters(n):
    r = get(run({"Kn": K[n]}, n, GUESS[n]), "Kn")
    assert_params(r, n, TRUE[n])
    assert r["best_R-squared"] >= 1 - 1e-10
    if n == 4:
        assert r["kurtosis"] == pytest.approx(3.0, rel=5e-3)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_oracle_adaptive_noisy_recovers_within_five_sigma(n):
    d = {"Kn": KN[n]}
    r = get(run(d, n, adaptive(d, n)), "Kn")
    for p in PN[n]:
        err = r["err_" + p]
        assert np.isfinite(err) and err > 0, p
        assert abs(r["best_" + p] - TRUE[n][p]) <= 5 * err, p
    assert r["best_b"] == pytest.approx(1000.0, rel=1e-2)


@pytest.mark.parametrize("n", [2, 4])
def test_oracle_metrics_recomputed_from_stored_parameters(n):
    d = {"Kn": KN[n]}
    r = get(run(d, n, adaptive(d, n)), "Kn")
    r2, rmse, aic = metrics(n, TAU, KN[n]["g(2)-1"].to_numpy(), row_params(r, n))
    assert r["best_R-squared"] == pytest.approx(r2, **TOL)
    assert r["best_RMSE"] == pytest.approx(rmse, **TOL)
    assert r["best_AIC"] == pytest.approx(aic, **TOL)
    if n == 4:
        assert r["kurtosis"] == pytest.approx(r["best_e"] / r["best_c"]**2, **TOL)


def test_oracle_expert_iteration_metrics_and_best_selection():
    res = run({"K2N": KN[2]}, 2, GUESS[2], tolerance=0, max_iterations=3)
    r = res.iloc[0]
    y = KN[2]["g(2)-1"].to_numpy()
    r2s = []
    for i in (1, 2, 3):
        p = [r[f"{pn}_iter{i}"] for pn in PN[2]]
        m = metrics(2, TAU, y, p)
        assert r[f"R-squared_iter{i}"] == pytest.approx(m[0], **TOL)
        assert r[f"RMSE_iter{i}"] == pytest.approx(m[1], **TOL)
        assert r[f"AIC_iter{i}"] == pytest.approx(m[2], **TOL)
        r2s.append(r[f"R-squared_iter{i}"])
    k = int(np.argmax(r2s)) + 1
    assert r["best_R-squared"] == max(r2s)
    for pn in PN[2]:
        assert r["best_" + pn] == r[f"{pn}_iter{k}"]


def _direct(n, y, p0, w=None, method="lm"):
    s = None if w is None else 1.0 / np.sqrt(w / np.mean(w))
    kw = dict(p0=p0, maxfev=50000, sigma=s, absolute_sigma=False)
    if method != "lm":
        kw.update(method=method, bounds=([0, 0, 0, -0.1], [1, 1e6, np.inf, 0.1]))
    return _ORIG_CURVE_FIT(FF[n], TAU, y, **kw)


@pytest.mark.parametrize("variant", ["lm", "trf", "weighted"])
def test_oracle_expert_single_iteration_equals_direct_curve_fit(variant):
    df = KW if variant == "weighted" else KN[2]
    y = df["g(2)-1"].to_numpy()
    w = W if variant == "weighted" else None
    method = "trf" if variant == "trf" else "lm"
    popt, pcov = _direct(2, y, [0.7, 1300.0, 5e4, 0.0], w, method)
    r = run({"X": df}, 2, GUESS[2], max_iterations=1, method=method).iloc[0]
    perr = np.sqrt(np.diag(pcov))
    for j, p in enumerate(PN[2]):
        assert r["best_" + p] == popt[j]
        assert r["err_" + p] == perr[j]
    assert r["R-squared_iter1"] == r["best_R-squared"]


def test_oracle_higher_order_fit_of_lower_order_data():
    for n in (2, 3, 4):
        d = {"K1": K[1]}
        r = get(run(d, n, adaptive(d, n)), "K1")
        assert r["best_b"] == pytest.approx(1000.0, **REL3)
        assert abs(r["best_c"]) / r["best_b"]**2 < 1e-5
        if n >= 3:
            assert abs(r["best_d"]) / r["best_b"]**3 < 1e-4
        if n == 4:
            assert abs(r["best_e"]) / r["best_b"]**4 < 1e-3


def test_oracle_order_one_cannot_represent_order_two_data():
    d = {"K2": K[2]}
    r1 = get(run(d, 1, adaptive(d, 1)), "K2")
    r2 = get(run(d, 2, adaptive(d, 2)), "K2")
    assert r1["best_R-squared"] < r2["best_R-squared"]


def test_oracle_column_lists():
    d = {"K2N": KN[2]}
    c2 = list(run(d, 2, adaptive(d, 2)).columns)
    base2 = ["filename", "weighted", "best_R-squared", "best_RMSE", "best_AIC",
             "best_a", "best_b", "best_c", "best_f", "err_a", "err_b", "err_c", "err_f"]
    assert c2 == base2
    d4 = {"K4N": KN[4]}
    c4 = list(run(d4, 4, adaptive(d4, 4)).columns)
    assert len(c4) == 18 and c4[-1] == "kurtosis"
    d3 = {"K3N": KN[3]}
    assert "kurtosis" not in run(d3, 3, adaptive(d3, 3)).columns
    ce = list(run(d, 2, GUESS[2], tolerance=0, max_iterations=2).columns)
    exp = ["filename", "weighted"]
    for i in (1, 2):
        exp += [f"R-squared_iter{i}", f"RMSE_iter{i}", f"AIC_iter{i}"] + [f"{p}_iter{i}" for p in PN[2]]
    exp += base2[2:]
    assert ce == exp


def test_oracle_expert_convergence_and_iteration_control(capsys):
    d = {"K2N": KN[2]}
    res = run(d, 2, GUESS[2])
    out = capsys.readouterr().out
    cols = list(res.columns)
    assert "R-squared_iter1" in cols and "R-squared_iter2" in cols and "R-squared_iter3" not in cols
    assert "Converged after 2 iterations for K2N." in out
    res = run(d, 2, GUESS[2], tolerance=0, max_iterations=3)
    out = capsys.readouterr().out
    assert "R-squared_iter3" in res.columns and "Converged" not in out
    res = run(d, 2, GUESS[2], max_iterations=1)
    assert "R-squared_iter1" in res.columns and "R-squared_iter2" not in res.columns


# ============================================================================
# 7.2 Metamorphic
# ============================================================================
def _cmp(new, ref, n, mult=None, shift=None, err=False):
    """Compare parameter (and optionally error) columns: new = ref*mult + shift."""
    mult = mult or {}
    shift = shift or {}
    for p in PN[n]:
        exp = ref["best_" + p] * mult.get(p, 1.0) + shift.get(p, 0.0)
        assert new["best_" + p] == pytest.approx(exp, rel=1e-3, abs=1e-6 if p == "f" else 0), p
        if err:
            assert new["err_" + p] == pytest.approx(ref["err_" + p] * mult.get(p, 1.0), rel=1e-3), p


@pytest.mark.parametrize("s", [10.0, 0.01])
@pytest.mark.parametrize("mode,n", [("expert", 2), ("pipeline", 2), ("pipeline", 4)])
def test_meta_lag_time_scaling(n, s, mode):
    ref_df = KN[n]
    new_df = make_df(TAU * s, ref_df["g(2)-1"])
    lim = (1e-9 * s, 10 * s)
    pw = {"a": 1, "b": 1 / s, "c": 1 / s**2, "d": 1 / s**3, "e": 1 / s**4, "f": 1}
    if mode == "expert":
        g = GUESS[n]
        g2 = [g[0], g[1] / s, g[2] / s**2, g[3] / s**3, g[4] / s**4, g[5]]
        ref = run({"X": ref_df}, n, g).iloc[0]
        new = run({"X": new_df}, n, g2, limits=lim).iloc[0]
    else:
        ref = run({"X": ref_df}, n, adaptive({"X": ref_df}, n)).iloc[0]
        new = run({"X": new_df}, n, adaptive({"X": new_df}, n), limits=lim).iloc[0]
    _cmp(new, ref, n, mult=pw, err=True)
    for c in ("best_R-squared", "best_RMSE", "best_AIC"):
        assert new[c] == pytest.approx(ref[c], rel=1e-6)


@pytest.mark.parametrize("k", [0.5, 1.2])
@pytest.mark.parametrize("mode,n", [("expert", 2), ("pipeline", 2), ("pipeline", 4)])
def test_meta_amplitude_scaling(n, k, mode):
    ref_df = KN[n]
    new_df = make_df(TAU, ref_df["g(2)-1"].to_numpy() * k)
    if mode == "expert":
        g = GUESS[n]
        g2 = list(g)
        g2[0] = g[0] * k
        g2[5] = g[5] * k
        ref = run({"X": ref_df}, n, g).iloc[0]
        new = run({"X": new_df}, n, g2).iloc[0]
    else:
        ref = run({"X": ref_df}, n, adaptive({"X": ref_df}, n)).iloc[0]
        new = run({"X": new_df}, n, adaptive({"X": new_df}, n)).iloc[0]
    _cmp(new, ref, n, mult={"a": k, "f": k})
    assert new["best_R-squared"] == pytest.approx(ref["best_R-squared"], rel=1e-6)
    assert new["best_RMSE"] == pytest.approx(ref["best_RMSE"] * k, rel=1e-6)
    assert new["best_AIC"] == pytest.approx(ref["best_AIC"] + 2 * TAU.size * np.log(k), abs=1e-3)


@pytest.mark.parametrize("mode,n", [("expert", 2), ("pipeline", 2), ("pipeline", 4)])
def test_meta_baseline_shift(n, mode):
    delta = 0.02
    ref_df = KN[n]
    new_df = make_df(TAU, ref_df["g(2)-1"].to_numpy() + delta)
    if mode == "expert":
        g = GUESS[n]
        g2 = list(g)
        g2[5] = g[5] + delta
        ref = run({"X": ref_df}, n, g).iloc[0]
        new = run({"X": new_df}, n, g2).iloc[0]
    else:
        ref = run({"X": ref_df}, n, adaptive({"X": ref_df}, n)).iloc[0]
        new = run({"X": new_df}, n, adaptive({"X": new_df}, n)).iloc[0]
    _cmp(new, ref, n, shift={"f": delta})
    assert new["best_R-squared"] == pytest.approx(ref["best_R-squared"], rel=1e-6)
    assert new["best_RMSE"] == pytest.approx(ref["best_RMSE"], rel=1e-6)


def _fixed(dfs, n):
    return adaptive(dfs, n)


def test_meta_file_order_additional_and_duplicate_files_independent():
    dfs = {"A": KN[2], "B": K[2], "C": make_df(TAU, KN[2]["g(2)-1"] * 1.2)}
    g = _fixed(dfs, 2)
    base = run(dfs, 2, g)
    rev = {k: dfs[k] for k in ["C", "A", "B"]}
    perm = run(rev, 2, {k: g[k] for k in rev})
    for k in dfs:
        row_equal(get(base, k), get(perm, k))
    sub = {"A": dfs["A"]}
    only = run(sub, 2, {"A": g["A"]})
    row_equal(get(base, "A"), get(only, "A"))
    dup = dict(dfs)
    dup["A_copy"] = dfs["A"]
    dup_g = dict(g)
    dup_g["A_copy"] = g["A"]
    d = run(dup, 2, dup_g)
    for k in dfs:
        row_equal(get(base, k), get(d, k))
    a, c = get(d, "A").drop("filename"), get(d, "A_copy").drop("filename")
    row_equal(a, c)


@pytest.mark.parametrize("method", ["lm", "trf"])
@pytest.mark.parametrize("mode", ["expert", "adaptive"])
def test_meta_repeated_call_is_deterministic(method, mode):
    g = GUESS[2] if mode == "expert" else adaptive({"K2N": KN[2]}, 2)
    r1 = run({"K2N": KN[2]}, 2, g, method=method)
    r2 = run({"K2N": KN[2]}, 2, g, method=method)
    pd.testing.assert_frame_equal(r1, r2, check_exact=True)


@pytest.mark.parametrize("mode", ["expert", "adaptive"])
def test_meta_points_outside_window_irrelevant_with_fixed_start(mode):
    tx = np.logspace(0.2, 2, 10)
    ext = pd.concat([KN[2], make_df(tx, np.full(10, 1e-3))], ignore_index=True)
    g = GUESS[2] if mode == "expert" else adaptive({"K2N": KN[2]}, 2)
    lim = (1e-9, 1.0)
    a = run({"K2N": KN[2]}, 2, g, limits=lim)
    b = run({"K2N": ext}, 2, g, limits=lim)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


@pytest.mark.parametrize("mode", ["expert", "adaptive"])
def test_meta_nan_lag_time_rows_ignored_with_fixed_start(mode):
    nan_rows = make_df(np.full(3, np.nan), np.array([0.1, 0.2, 0.3]))
    ext = pd.concat([KN[2].iloc[:30], nan_rows, KN[2].iloc[30:]], ignore_index=True)
    g = GUESS[2] if mode == "expert" else adaptive({"K2N": KN[2]}, 2)
    a = run({"K2N": KN[2]}, 2, g)
    b = run({"K2N": ext}, 2, g)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def _numeric_close(a, b, rel=1e-6):
    for c in a.index:
        if c in ("filename", "weighted"):
            continue
        assert a[c] == pytest.approx(b[c], rel=rel, abs=1e-12), c


def test_meta_constant_weights_equal_unweighted():
    g = adaptive({"X": KN[2]}, 2)
    ref = run({"X": KN[2]}, 2, g).iloc[0]
    for const in (1.0, 0.37):
        w = run({"X": make_df(TAU, KN[2]["g(2)-1"], np.full(TAU.size, const))}, 2, g).iloc[0]
        assert w["weighted"] == True  # noqa: E712
        _numeric_close(w, ref)


def test_meta_weight_scale_invariance():
    g = adaptive({"X": KW}, 2)
    a = run({"X": KW}, 2, g).iloc[0]
    b = run({"X": make_df(TAU, KW["g(2)-1"], 7.3 * W)}, 2, g).iloc[0]
    _numeric_close(a, b)


@pytest.mark.parametrize("mode", ["expert", "adaptive"])
def test_meta_weights_irrelevant_for_exact_model(mode):
    kw = make_df(TAU, y_model(2), W)
    g = GUESS[2] if mode == "expert" else adaptive({"X": K[2]}, 2)
    a = run({"X": K[2]}, 2, g).iloc[0]
    b = run({"X": kw}, 2, g).iloc[0]
    assert_params(a, 2, TRUE[2])
    assert_params(b, 2, TRUE[2])


def test_meta_fit_method_invariance_interior_optimum():
    ref = run({"X": KN[2]}, 2, GUESS[2], method="lm").iloc[0]
    for m in ("trf", "dogbox"):
        r = run({"X": KN[2]}, 2, GUESS[2], method=m).iloc[0]
        _cmp(r, ref, 2)
        assert r["best_R-squared"] == pytest.approx(ref["best_R-squared"], rel=1e-6)


def test_meta_expert_equals_pipeline():
    e = run({"X": KN[2]}, 2, GUESS[2]).iloc[0]
    p = run({"X": KN[2]}, 2, adaptive({"X": KN[2]}, 2)).iloc[0]
    _cmp(p, e, 2)


def test_meta_row_permutation_within_file():
    shuf = KN[2].sample(frac=1, random_state=7).reset_index(drop=True)
    a = run({"X": KN[2]}, 2, GUESS[2]).iloc[0]
    b = run({"X": shuf}, 2, GUESS[2]).iloc[0]
    _cmp(b, a, 2)
    assert b["best_R-squared"] == pytest.approx(a["best_R-squared"], rel=1e-6)


# ============================================================================
# 7.4 Edge cases
# ============================================================================
@pytest.mark.parametrize("lim", [(TAU[60], TAU[60]), (1.0, 1e-3)])
def test_edge_window_with_fewer_than_two_points(lim, capsys):
    res = run({"K1N": KN[1]}, 1, GUESS[1], limits=lim)
    assert list(res.columns) == ["filename"] and len(res) == 1
    assert "Not enough data points" in capsys.readouterr().out


def test_edge_window_with_exactly_two_points_lm_type_error():
    lim = (TAU[60], TAU[61])
    for g in (GUESS[1], {"K1": [0.8, 1000.0, 0.001]}):
        dfs = {"K1": K[1]}
        res = run(dfs, 1, g, limits=lim, method="lm")
        r = res.iloc[0]
        assert "must not exceed" in r["Error"]
        assert not any(c.startswith("best_") for c in res.columns)


@pytest.mark.slow
def test_edge_window_with_exactly_two_points_trf_documents_outcome():
    """[REAL] ZV: trf has no TypeError; either inf errors or a RuntimeError row."""
    lim = (TAU[60], TAU[61])
    for g in (GUESS[1], {"K1": [0.8, 1000.0, 0.001]}):
        r = run({"K1": K[1]}, 1, g, limits=lim, method="trf").iloc[0]
        if "Error" in r.index and isinstance(r["Error"], str):
            assert "must not exceed" not in r["Error"]
        else:
            assert all(np.isinf(r["err_" + p]) for p in PN[1])


def test_edge_empty_dict_returns_empty_frame():
    res = run({}, 2, GUESS[2])
    assert isinstance(res, pd.DataFrame) and res.empty and list(res.columns) == []


def test_edge_bad_dataframes_give_error_rows():
    dfs = {"noT": KN[1].drop(columns="t [s]"), "noG": KN[1].drop(columns="g(2)-1"),
           "none": None, "K1N": KN[1]}
    res = run(dfs, 1, GUESS[1])
    assert get(res, "noT")["Error"] == "'t [s]'"
    assert get(res, "noG")["Error"] == "'g(2)-1'"
    assert "not subscriptable" in get(res, "none")["Error"]
    ok = get(res, "K1N")
    assert pd.isna(ok["Error"])
    assert_params(ok, 1, TRUE[1], rel=2e-2)


def test_edge_adaptive_missing_start_key_gives_error_row():
    res = run({"X": KN[1]}, 1, {"other": [0.8, 1000.0, 0.001]})
    assert res.iloc[0]["Error"] == "'X'"


def test_edge_expert_zero_iterations():
    r = run({"K2N": KN[2]}, 2, GUESS[2], max_iterations=0).iloc[0]
    assert r["Error"] == "No successful fit attempts."
    assert r["weighted"] == False  # noqa: E712
    for p in PN[2]:
        assert np.isnan(r["best_" + p]) and np.isnan(r["err_" + p])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("mode", ["expert", "adaptive"])
def test_edge_nonfinite_y_in_window_aborts_whole_call(bad, mode):
    y = KN[1]["g(2)-1"].to_numpy().copy()
    y[50] = bad
    dfs = {"bad": make_df(TAU, y), "ok": KN[1]}
    g = GUESS[1] if mode == "expert" else {"bad": [0.8, 1000.0, 0.001], "ok": [0.8, 1000.0, 0.001]}
    with pytest.raises(ValueError):
        run(dfs, 1, g)
    with pytest.raises(ValueError):
        run({"bad": dfs["bad"]}, 1, g)


@pytest.mark.parametrize("mode", ["expert", "adaptive"])
def test_edge_nan_outside_window_irrelevant(mode):
    y = KN[1]["g(2)-1"].to_numpy().copy()
    y_nan = y.copy()
    y_nan[-1] = np.nan
    g = GUESS[1] if mode == "expert" else {"X": [0.8, 1000.0, 0.001]}
    a = run({"X": make_df(TAU, y)}, 1, g, limits=(1e-9, 0.5))
    b = run({"X": make_df(TAU, y_nan)}, 1, g, limits=(1e-9, 0.5))
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_edge_constant_y_binary_exact():
    df = make_df(TAU, np.full(TAU.size, 0.25))
    res = run({"C": df}, 1, {"C": [0.5, 1000.0, 0.0]})
    r = res.iloc[0]
    if "Error" in r.index and isinstance(r["Error"], str):
        assert np.isnan(r["best_R-squared"])
    else:
        assert r["best_R-squared"] == 0


def test_edge_parameter_count_equals_point_count():
    lim = (TAU[60], TAU[62])
    res = run({"K1": K[1]}, 1, {"K1": [0.8, 1000.0, 0.001]}, limits=lim)
    r = res.iloc[0]
    if "Error" in res.columns and isinstance(r["Error"], str):
        assert r["Error"] == "All zoom grid points failed to converge."
    else:
        assert all(np.isinf(r["err_" + p]) for p in PN[1])


def test_edge_expert_start_as_tuple_or_array_equals_list():
    ref = run({"K2N": KN[2]}, 2, GUESS[2])
    for conv in (tuple, np.array):
        pd.testing.assert_frame_equal(run({"K2N": KN[2]}, 2, conv(GUESS[2])), ref, check_exact=True)


def test_edge_unknown_function_name_expert_raises_adaptive_runs():
    def fit_function5(x, a, b, c, f):
        return f + a * ((np.exp(-b * x)) * (1 + 0.5 * c * x**2))**2

    with pytest.raises(ValueError, match="Unknown fit function"):
        cc.plot_processed_correlations_iterative({"K2N": KN[2]}, fit_function5, FIT_ALL, GUESS[2],
                                                 show_plots=False)
    res = cc.plot_processed_correlations_iterative({"K2N": KN[2]}, fit_function5, FIT_ALL,
                                                   {"K2N": [0.8, 2000.0, 2e5, 0.001]}, show_plots=False)
    assert "Error" not in res.columns
    assert res.iloc[0]["best_b"] == pytest.approx(1000.0, rel=1e-2)


def test_edge_adaptive_start_as_tuple_raises_attribute_error():
    with pytest.raises(AttributeError):
        run({"K2N": KN[2]}, 2, {"K2N": (0.8, 2000.0, 2e5, 0.001)})


def test_edge_weighted_adaptive_passes_sigma_wrapper(monkeypatch):
    """[WRAP]"""
    g = adaptive({"KW": KW}, 2)
    calls = install_rec(monkeypatch, [])
    res = run({"KW": KW}, 2, g)
    assert res.iloc[0]["weighted"] == True  # noqa: E712
    for c in models(calls):
        np.testing.assert_allclose(c["sigma"], SIG_W, rtol=1e-6, atol=1e-12)


# ============================================================================
# 7.5 Characterization
# ============================================================================
def test_char_expert_b_warning_threshold_documents_current_behavior(capsys):
    """Documents current behavior (F-01): asymmetric warning 0.2*Gamma ... 20*Gamma."""
    hits = {}
    for b in (150.0, 250.0, 15000.0, 25000.0):
        g = list(GUESS[1])
        g[1] = b
        run({"K1": K[1]}, 1, g, max_iterations=1)
        hits[b] = "[INFO] 'K1': provided b=" in capsys.readouterr().out
    assert hits == {150.0: True, 250.0: False, 15000.0: False, 25000.0: True}


def test_char_weighted_metrics_mixed_documents_current_behavior():
    """Documents current behavior (F-17): R2 weighted, RMSE/AIC unweighted."""
    g = adaptive({"KW": KW}, 2)
    r = run({"KW": KW}, 2, g).iloc[0]
    y = KW["g(2)-1"].to_numpy()
    p = row_params(r, 2)
    rw, _, _ = metrics(2, TAU, y, p, W)
    ru, rmse, aic = metrics(2, TAU, y, p)
    assert r["best_R-squared"] == pytest.approx(rw, **TOL)
    assert r["best_R-squared"] != pytest.approx(ru, rel=1e-12, abs=0)
    assert r["best_RMSE"] == pytest.approx(rmse, **TOL)
    assert r["best_AIC"] == pytest.approx(aic, **TOL)


def test_char_short_file_row_misses_columns_documents_current_behavior():
    """Documents current behavior (F-13)."""
    short = make_df([1e-3], [0.5])
    res = run({"short": short, "K1N": KN[1]}, 1, GUESS[1])
    r = get(res, "short")
    assert pd.isna(r["weighted"]) and pd.isna(r["best_b"])
    assert "Error" not in res.columns  # no row has an Error entry -> column does not exist


def test_char_error_row_shapes_documents_current_behavior():
    """Documents current behavior (F-15)."""
    t = run({"K1": K[1]}, 1, GUESS[1], limits=(TAU[60], TAU[61]), method="lm")
    assert set(t.columns) == {"filename", "weighted", "Error"}
    r = run({"K2N": KN[2]}, 2, GUESS[2], max_iterations=0)
    assert {"Error", "best_R-squared", "best_RMSE", "best_AIC"} <= set(r.columns)
    for c in [c for c in r.columns if c.startswith(("best_", "err_"))]:
        assert r[c].isna().all()


def test_char_trf_infeasible_start_aborts_documents_current_behavior():
    """Documents current behavior (F-03)."""
    g = adaptive({"G": G20k}, 1)
    assert g["G"][1] == pytest.approx(4e4, rel=1e-3)
    with pytest.raises(ValueError, match="infeasible|outside of provided bounds"):
        run({"G": G20k}, 1, g, method="trf")
    r = run({"G": G20k}, 1, g, method="lm").iloc[0]
    assert r["best_b"] == pytest.approx(2e4, **REL3)
    g15 = adaptive({"G": G15k}, 1)
    r = run({"G": G15k}, 1, g15, method="trf").iloc[0]
    assert r["best_b"] == pytest.approx(1.5e4, **REL3)
    with pytest.raises(ValueError, match="infeasible|outside of provided bounds"):
        run({"X": G20k}, 1, {"X": [0.8, 5e4, 0.001]}, method="trf")
    run({"X": G20k}, 1, {"X": [0.8, 3e4, 0.001]}, method="trf")


def test_char_expert_double_reduction_documents_current_behavior():
    """Documents current behavior (F-02, JADE-identical): reduced list is reduced again."""
    for n in (1, 2, 3):
        red = cc.get_meaningful_parameters(FF[n], BASE)
        with pytest.raises(ValueError, match="unpack"):
            run({"K": K[n]}, n, red)
    red = cc.get_meaningful_parameters(FF[4], BASE)
    a = run({"K": K[4]}, 4, red)
    b = run({"K": K[4]}, 4, BASE)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


# ============================================================================
# 7.6 Validation with model error (ZV)
# ============================================================================
@pytest.mark.parametrize("n", [2, 3, 4])
def test_oracle_schulz_validation_with_model_error(n):
    """ZV: no exact oracle (moment expansion truncated); spec tolerances T-val."""
    d = {"S": SCH}
    r = get(run(d, n, adaptive(d, n)), "S")
    assert r["best_b"] == pytest.approx(1000.0, rel=1e-2)
    assert r["best_c"] / r["best_b"]**2 == pytest.approx(0.05, abs=0.02)


# ============================================================================
# 7.7 / 7.8 Algorithm oracles with recording wrapper
# ============================================================================
G0 = adaptive({"K2N": KN[2]}, 2)["K2N"]
Y2N = KN[2]["g(2)-1"].to_numpy()


def _r2(c):
    return metrics(2, TAU, Y2N, c["out"][0])[0]


def _p0_round(center, base, h):
    """expected p0 list per grid point for FF2 given center popt/base list."""
    fac = np.logspace(-h, h, 5)
    out = []
    for fct in fac:
        b_i = center[1] * fct
        sc = b_i / center[1]
        out.append([center[0], b_i, center[2] * sc**2, center[3]])
    return out


def test_oracle_zoom_default_stops_after_round_two(monkeypatch):
    """[WRAP] F-07."""
    calls = install_rec(monkeypatch, [])
    res = run({"K2N": KN[2]}, 2, {"K2N": list(G0)})
    m = models(calls)
    assert len(m) == 10
    assert [c["maxfev"] for c in m] == [1000] * 5 + [5000] * 5
    for i, fct in enumerate(np.logspace(-1.5, 1.5, 5)):
        b_i = G0[1] * fct
        assert m[i]["p0"] == pytest.approx([G0[0], b_i, G0[2] * (b_i / G0[1])**2, G0[3]], **TOL)
    r1 = [_r2(c) for c in m[:5]]
    ctr = m[int(np.argmax(r1))]["out"][0]
    for i, exp in enumerate(_p0_round(ctr, None, 0.5)):
        assert m[5 + i]["p0"] == pytest.approx(exp, **TOL)
    # selection (ZG-03)
    r2s = [_r2(c) for c in m]
    best = m[int(np.argmax(r2s))]["out"]
    r = res.iloc[0]
    perr = np.sqrt(np.diag(best[1]))
    for j, p in enumerate(PN[2]):
        assert r["best_" + p] == best[0][j]
        assert r["err_" + p] == perr[j]


def test_oracle_zoom_tolerance_zero_runs_three_rounds(monkeypatch):
    """[WRAP]"""
    calls = install_rec(monkeypatch, [])
    run({"K2N": KN[2]}, 2, {"K2N": list(G0)}, tolerance=0)
    m = models(calls)
    assert len(m) == 15
    assert [c["maxfev"] for c in m[10:]] == [50000] * 5
    r2s = [_r2(c) for c in m[:10]]
    ctr = m[int(np.argmax(r2s))]["out"][0]
    for i, exp in enumerate(_p0_round(ctr, None, 0.15)):
        assert m[10 + i]["p0"] == pytest.approx(exp, **TOL)


def test_oracle_zoom_failed_first_round_recenters_on_original(monkeypatch):
    """[MOCK]"""
    def fake(entry, idx):
        if entry["maxfev"] == 1000:
            raise RuntimeError("no conv")
    calls = install_rec(monkeypatch, [], fake)
    res = run({"K2N": KN[2]}, 2, {"K2N": list(G0)})
    m = models(calls)
    assert len(m) == 15
    for i, fct in enumerate(np.logspace(-0.5, 0.5, 5)):
        assert m[5 + i]["p0"][1] == pytest.approx(G0[1] * fct, **TOL)
        assert m[5 + i]["p0"][0] == pytest.approx(G0[0], **TOL)
    assert "Error" not in res.columns


def test_oracle_zoom_all_points_fail(monkeypatch):
    """[MOCK]"""
    def fake(entry, idx):
        raise RuntimeError("no conv")
    calls = install_rec(monkeypatch, [], fake)
    r = run({"K2N": KN[2]}, 2, {"K2N": list(G0)}).iloc[0]
    assert len(models(calls)) == 15
    assert r["Error"] == "All zoom grid points failed to converge."
    for p in PN[2]:
        assert np.isnan(r["best_" + p]) and np.isnan(r["err_" + p])


def test_oracle_zoom_method_arguments(monkeypatch):
    """[WRAP]"""
    calls = install_rec(monkeypatch, [])
    run({"K2N": KN[2]}, 2, {"K2N": list(G0)}, method="trf")
    for c in models(calls):
        assert c["method"] == "trf"
        assert c["bounds"] == ([0.0, 0.0, 0.0, -0.1], [1.0, 1e6, np.inf, 0.1])
    calls2 = install_rec(monkeypatch, [])
    run({"K2N": KN[2]}, 2, {"K2N": list(G0)}, method="lm")
    for c in models(calls2):
        assert c["method"] is None and c["bounds"] is None


def test_oracle_zoom_sigma_argument(monkeypatch):
    """[WRAP]"""
    g = adaptive({"KW": KW}, 2)["KW"]
    calls = install_rec(monkeypatch, [])
    run({"KW": KW}, 2, {"KW": list(g)})
    assert all(c["sigma"] is not None for c in models(calls))
    for c in models(calls):
        np.testing.assert_allclose(c["sigma"], SIG_W, rtol=1e-6, atol=1e-12)
    calls2 = install_rec(monkeypatch, [])
    run({"K2N": KN[2]}, 2, {"K2N": list(G0)})
    assert all(c["sigma"] is None for c in models(calls2))


def test_oracle_expert_call_sequence(monkeypatch):
    """[WRAP]"""
    calls = install_rec(monkeypatch, [])
    run({"K2N": KN[2]}, 2, GUESS[2])
    assert calls[0]["f"] is cc._simple_exp and calls[0]["method"] == "trf" and calls[0]["maxfev"] == 5000
    m = models(calls)
    assert m[0]["p0"] == [0.7, 1300.0, 5e4, 0.0] and m[0]["maxfev"] == 50000
    for i in range(len(m) - 1):
        assert m[i + 1]["p0"] == list(m[i]["out"][0])


def test_oracle_expert_fit_error_in_iteration_two_keeps_first(monkeypatch, capsys):
    """[MOCK]"""
    def fake(entry, idx):
        if idx == 2:
            raise RuntimeError("fail2")
    install_rec(monkeypatch, [], fake)
    res = run({"K2N": KN[2]}, 2, GUESS[2], max_iterations=5)
    r = res.iloc[0]
    assert "R-squared_iter2" not in res.columns
    assert "Fit error in iteration 2 for 'K2N'" in capsys.readouterr().out
    for p in PN[2]:
        assert r["best_" + p] == r[f"{p}_iter1"]


def _fake_second_iteration(calls):
    def fake(entry, idx):
        if idx == 2:
            popt1 = calls[[i for i, c in enumerate(calls) if c["f"] is not cc._simple_exp][0]]["out"][0]
            return popt1 * np.array([1, 1.5, 1, 1]), 4 * np.eye(4)
    return fake


def test_char_expert_err_from_last_iteration_documents_current_behavior(monkeypatch):
    """Documents current behavior (F-05). [MOCK]"""
    calls = []
    install_rec(monkeypatch, calls, _fake_second_iteration(calls))
    r = run({"K2N": KN[2]}, 2, GUESS[2], max_iterations=2).iloc[0]
    popt1 = models(calls)[0]["out"][0]
    for j, p in enumerate(PN[2]):
        assert r["best_" + p] == popt1[j]
        assert r["err_" + p] == 2.0


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="FINDING F-05, siehe _dev/notes/test_spec_cumulants_C.md")
def test_finding_expert_err_belongs_to_best_iteration(monkeypatch):
    calls = []
    install_rec(monkeypatch, calls, _fake_second_iteration(calls))
    r = run({"K2N": KN[2]}, 2, GUESS[2], max_iterations=2).iloc[0]
    perr = np.sqrt(np.diag(models(calls)[0]["out"][1]))
    for j, p in enumerate(PN[2]):
        assert r["err_" + p] == pytest.approx(perr[j], **TOL)


def test_oracle_expert_first_fit_error(monkeypatch, capsys):
    """[MOCK]"""
    def fake(entry, idx):
        if idx == 1:
            raise RuntimeError("fail1")
    install_rec(monkeypatch, [], fake)
    r = run({"K2N": KN[2]}, 2, GUESS[2]).iloc[0]
    assert r["Error"] == "No successful fit attempts."
    assert "Fit error in iteration 1" in capsys.readouterr().out


def _nan_first_fit():
    def fake(entry, idx):
        if idx == 1:
            return np.full(4, np.nan), np.eye(4)
    return fake


def test_char_nan_attempt_wins_argmax_documents_current_behavior(monkeypatch):
    """Documents current behavior (F-06, ZV). [MOCK]"""
    install_rec(monkeypatch, [], _nan_first_fit())
    r = run({"K2N": KN[2]}, 2, {"K2N": list(G0)}).iloc[0]
    assert np.isnan(r["best_R-squared"])
    for p in PN[2]:
        assert np.isnan(r["best_" + p])
        assert np.isfinite(r["err_" + p])


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="FINDING F-06, siehe _dev/notes/test_spec_cumulants_C.md")
def test_finding_nan_attempt_not_selected(monkeypatch):
    install_rec(monkeypatch, [], _nan_first_fit())
    r = run({"K2N": KN[2]}, 2, {"K2N": list(G0)}).iloc[0]
    assert np.isfinite(r["best_R-squared"])


# ============================================================================
# 8. calculate_mean_fit_metrics
# ============================================================================
def md():
    return pd.DataFrame({
        "filename": ["f1", "f2", "f3", "bad_Error_file"],
        "best_R-squared": [0.99, 0.97, np.nan, 0.5],
        "best_RMSE": [0.01, 0.03, np.nan, 0.2],
        "best_AIC": [-100.0, -80.0, np.nan, 10.0],
        "Error": [np.nan, np.nan, "fit failed", np.nan]})


MD_EXPECT = dict(mean_R_squared=0.98, mean_RMSE=0.02, mean_AIC=-90.0, num_datasets=3,
                 std_R_squared=np.sqrt(2e-4), std_RMSE=np.sqrt(2e-4), std_AIC=np.sqrt(200))


def _assert_metrics(r, exp):
    assert list(r) == ["mean_R_squared", "mean_RMSE", "mean_AIC", "num_datasets",
                       "std_R_squared", "std_RMSE", "std_AIC"]
    for k, v in exp.items():
        if isinstance(v, float) and np.isnan(v):
            assert np.isnan(r[k]), k
        else:
            assert r[k] == pytest.approx(v, **TOL), k


def test_oracle_mean_fit_metrics():
    _assert_metrics(cc.calculate_mean_fit_metrics(md()), MD_EXPECT)


def test_meta_mean_fit_metrics_row_permutation_and_extra_error_row():
    d = md()
    _assert_metrics(cc.calculate_mean_fit_metrics(d.iloc[[3, 1, 0, 2]].reset_index(drop=True)), MD_EXPECT)
    extra = pd.concat([d, pd.DataFrame({"filename": ["xError"], "best_R-squared": [0.1],
                                        "best_RMSE": [5.0], "best_AIC": [99.0]})], ignore_index=True)
    _assert_metrics(cc.calculate_mean_fit_metrics(extra), MD_EXPECT)


def test_edge_mean_fit_metrics_single_valid_row():
    d = pd.DataFrame({"filename": ["f1"], "best_R-squared": [0.9], "best_RMSE": [0.1], "best_AIC": [-5.0]})
    _assert_metrics(cc.calculate_mean_fit_metrics(d),
                    dict(mean_R_squared=0.9, mean_RMSE=0.1, mean_AIC=-5.0, num_datasets=1,
                         std_R_squared=np.nan, std_RMSE=np.nan, std_AIC=np.nan))


def test_edge_mean_fit_metrics_missing_column_raises():
    with pytest.raises(KeyError):
        cc.calculate_mean_fit_metrics(md().drop(columns="best_RMSE"))


def test_edge_mean_fit_metrics_int_filename_raises():
    d = md()
    d["filename"] = [1, 2, 3, 4]
    with pytest.raises(AttributeError):
        cc.calculate_mean_fit_metrics(d)


def test_char_mean_fit_metrics_filename_based_filter_documents_current_behavior():
    """Documents current behavior (F-12): failed row counted, 'Error' in filename excluded."""
    r = cc.calculate_mean_fit_metrics(md())
    assert r["num_datasets"] == 3
    assert r["mean_R_squared"] == pytest.approx(0.98, **TOL)
    assert r["mean_R_squared"] != pytest.approx(0.82, rel=1e-3)


def test_char_mean_fit_metrics_with_concatenated_run_output_documents_current_behavior():
    """Documents current behavior."""
    ok = run({"K1N": KN[1]}, 1, GUESS[1])
    bad = run({"bad": KN[1]}, 1, GUESS[1], max_iterations=0)
    r = cc.calculate_mean_fit_metrics(pd.concat([ok, bad], ignore_index=True))
    assert r["num_datasets"] == 2
    assert r["mean_R_squared"] == pytest.approx(ok.iloc[0]["best_R-squared"], **TOL)
    assert r["mean_RMSE"] == pytest.approx(ok.iloc[0]["best_RMSE"], **TOL)
    assert r["mean_AIC"] == pytest.approx(ok.iloc[0]["best_AIC"], **TOL)


# ============================================================================
# 9. Constants / signatures
# ============================================================================
def test_oracle_bounds_constants():
    assert cc._BOUNDS_LOWER == {'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': -np.inf, 'e': 0.0, 'f': -0.1}
    assert cc._BOUNDS_UPPER == {'a': 1.0, 'b': 1e6, 'c': np.inf, 'd': np.inf, 'e': np.inf, 'f': 0.1}


def test_oracle_signature_defaults():
    s = inspect.signature(cc.plot_processed_correlations_iterative).parameters
    assert (s["max_iterations"].default, s["tolerance"].default, s["maxfev"].default, s["method"].default,
            s["show_plots"].default, s["plot_number_start"].default) == (10, 1e-4, 50000, "lm", True, 1)
    s = inspect.signature(cc.get_adaptive_initial_parameters).parameters
    assert (s["strategy"].default, s["x_col"].default, s["y_col"].default, s["verbose"].default) == \
        ("individual", "t [s]", "g(2)-1", True)


# ============================================================================
# 11. Finding tests (xfail strict)
# ============================================================================
@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="FINDING F-02, Soll offen, siehe _dev/notes/open_discussion_points.md")
def test_finding_expert_mode_accepts_reduced_parameter_list():
    red = cc.get_meaningful_parameters(FF[2], BASE)
    a = run({"K": K[2]}, 2, red)
    b = run({"K": K[2]}, 2, BASE)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="FINDING F-03, siehe _dev/notes/test_spec_cumulants_C.md")
def test_finding_trf_pipeline_does_not_abort_for_large_gamma():
    g = adaptive({"G": G20k}, 1)
    r = run({"G": G20k}, 1, g, method="trf").iloc[0]
    assert np.isfinite(r["best_b"]) or isinstance(r.get("Error"), str)


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="FINDING F-04, siehe _dev/notes/test_spec_cumulants_C.md")
def test_finding_nan_file_does_not_abort_other_files():
    y = KN[1]["g(2)-1"].to_numpy().copy()
    y[50] = np.nan
    res = run({"K1N": KN[1], "nan": make_df(TAU, y)}, 1, GUESS[1])
    assert np.isfinite(get(res, "K1N")["best_b"])
    assert isinstance(get(res, "nan")["Error"], str)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="FINDING F-12, siehe _dev/notes/test_spec_cumulants_C.md")
def test_finding_mean_fit_metrics_uses_error_column():
    r = cc.calculate_mean_fit_metrics(md())
    assert r["num_datasets"] == 3
    assert r["mean_R_squared"] == pytest.approx((0.99 + 0.97 + 0.5) / 3)


# ----------------------------------------------------------------------------
# expert mode, order 4: d/e not identifiable from GUESS[4] -> only identified quantities
# ----------------------------------------------------------------------------
def test_char_expert_order4_noise_free_stalls_documents_current_behavior():
    """Documents current behavior, not desired behavior: expert fit of K4 from GUESS[4] stops
    early (d, e not identifiable); only b, f, R2 and c (within 10 %) are asserted."""
    r = get(run({"K": K[4]}, 4, GUESS[4]), "K")
    assert r["best_b"] == pytest.approx(1000.0, rel=1e-2)
    assert r["best_f"] == pytest.approx(1e-3, rel=1e-2, abs=1e-6)
    assert r["best_R-squared"] >= 1 - 1e-7
    assert r["best_c"] == pytest.approx(1e5, rel=0.1)


@pytest.mark.parametrize("kind", [
    "lag_time", "amplitude", "baseline"])
def test_char_expert_order4_relations_identified_quantities_documents_current_behavior(kind):
    """Documents current behavior, not desired behavior: for expert order 4 the scaling relations
    are checked only for the identified quantities a, b, f, R2, RMSE (c, d, e, err_* omitted).
    Tolerance note: the spec's abs 1e-6 for f assumed deterministic-level agreement; it does not
    hold for the expert optimizer (stalled order-4 fit), so f uses the optimizer class (abs 1e-4).
    Lag-time case: the stalled fit (d, e not identified) ends in another final state for scaled
    data (RMSE differs by 2.4 %), so only a, b, f and R2 >= 1-1e-7 are checked (characterization,
    not a finding)."""
    n = 4
    g = list(GUESS[n])
    y = KN[n]["g(2)-1"].to_numpy()
    ref = run({"X": KN[n]}, n, g).iloc[0]
    mult = {}
    kw = {}
    if kind == "lag_time":
        s = 10.0
        new_df = make_df(TAU * s, y)
        g2 = [g[0], g[1] / s, g[2] / s**2, g[3] / s**3, g[4] / s**4, g[5]]
        kw["limits"] = (1e-9 * s, 10 * s)
        mult = {"b": 1 / s}
        rmse_f = 1.0
    elif kind == "amplitude":
        k = 1.2
        new_df = make_df(TAU, y * k)
        g2 = list(g)
        g2[0], g2[5] = g[0] * k, g[5] * k
        mult = {"a": k, "f": k}
        rmse_f = k
    else:
        new_df = make_df(TAU, y + 0.02)
        g2 = list(g)
        g2[5] = g[5] + 0.02
        rmse_f = 1.0
    new = run({"X": new_df}, n, g2, **kw).iloc[0]
    # measured f drift in the lag_time case: 0.00099791 (scaled run) vs 0.00100295 (reference), ~5e-6
    for p in ("a", "b", "f"):
        exp = ref["best_" + p] * mult.get(p, 1.0) + (0.02 if (kind == "baseline" and p == "f") else 0.0)
        assert new["best_" + p] == pytest.approx(exp, rel=1e-3, abs=1e-4 if p == "f" else 0), p
    if kind == "lag_time":
        # stalled fit (d, e not identified) ends in another final state for scaled data:
        # measured RMSE deviation 2.4 % (9.528e-5 vs 9.765e-5) -> only a lower bound for R2.
        assert new["best_R-squared"] >= 1 - 1e-7
    else:
        assert new["best_R-squared"] == pytest.approx(ref["best_R-squared"], rel=1e-6)
        assert new["best_RMSE"] == pytest.approx(ref["best_RMSE"] * rmse_f, rel=1e-6)
