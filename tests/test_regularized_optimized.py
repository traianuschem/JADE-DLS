# -*- coding: utf-8 -*-
"""Tests for ade_dls/analysis/regularized_optimized.py
(spec: _dev/notes/test_spec_regularized_optimized.md, rev. 1).

Categories: test_oracle_* (closed-form expectations), test_meta_* (metamorphic),
test_edge_* (edge cases), test_char_* (characterization: documents current
behavior, not desired behavior), test_finding_* (xfail strict, known findings).

Spec IDs (EM-, TK-, PS-, NN-, RG-, NA-, DR-, PR-, CG-, SW-, G-, X-) appear in the
test names. [MOCK] tests replace ``ro.least_squares`` / ``ro.minimize`` by a fake
result with a prescribed x; [WRAP] tests record the call and delegate to scipy;
[REAL] tests run the real optimizer on the small 26-node grid.
Nothing here triggers plt.show / input; ``show_plots=False`` everywhere.
Q-1 (gold = vers3 or marimo) is open: defaults are checked as ADE-actual
(= vers3) characterization.
"""
import inspect
import random
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from concurrent.futures import Future  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

from ade_dls.analysis import regularized_optimized as ro  # noqa: E402

DET = dict(rel=1e-6, abs=1e-12)
OPT = dict(rel=1e-3)
OPTW = dict(rel=1e-2)
DETA = dict(rtol=1e-6, atol=1e-12)   # np.testing.assert_allclose flavour of DET

DT = np.logspace(-6, -1, 26)
DT51 = np.logspace(-6, -1, 51)
DLN = np.log(DT[1]) - np.log(DT[0])
LAG = np.logspace(-7, 0, 71)
SEED = 20261103
G100 = np.logspace(-8, 1, 100)
G200 = np.logspace(-8, 1, 200)
T0_ALV = 2.5e-8
ALV_FILE = Path(__file__).parent / "data" / "alv_mono" / "080622_5_0053_0001.ASC"


def mono(k, beta=0.8, lag=LAG, dt=DT):
    return beta * np.exp(-2.0 * lag / dt[k])


def bi(j, k, a1, a2, lag=LAG, dt=DT):
    return (a1 * np.exp(-lag / dt[j]) + a2 * np.exp(-lag / dt[k])) ** 2


def make_df(t, y, w=None):
    d = {"t [s]": np.asarray(t, float), "g(2)-1": np.asarray(y, float)}
    if w is not None:
        d["weight"] = np.asarray(w, float)
    return pd.DataFrame(d)


NNLS_P = {"decay_times": DT, "prominence": 0.01, "distance": 1}
REG_P = {"decay_times": DT, "alpha": 0.2, "prominence": 0.005, "distance": 1,
         "normalize": True, "fit_beta": True, "sparsity_penalty": 0.0,
         "peak_method": "maximum"}


def fake_result(x):
    return types.SimpleNamespace(x=np.asarray(x, float), status=1, nfev=1,
                                 message="mock", success=True)


def mock_ls(monkeypatch, x):
    monkeypatch.setattr(ro, "least_squares", lambda *a, **k: fake_result(x))


def mock_min(monkeypatch, x):
    monkeypatch.setattr(ro, "minimize", lambda *a, **k: fake_result(x))


def wrap(monkeypatch, name):
    """Passive recording wrapper delegating to the original scipy function."""
    orig = getattr(ro, name)
    calls = []

    def w(*a, **k):
        calls.append((a, k))
        return orig(*a, **k)

    monkeypatch.setattr(ro, name, w)
    return calls


def vec(entries, n=26):
    f = np.zeros(n)
    for i, v in entries.items():
        f[i] = v
    return f


FD = vec({10: 0.9})
FS = vec({9: 0.5, 10: 1.0, 11: 0.5})
FA = vec({9: 0.5, 10: 1.0, 11: 0.5, 12: 0.25})
FB = vec({4: 1, 5: 2, 6: 1, 14: 0.5, 15: 1, 16: 0.5})
FV = vec({7: 0.5, 9: 0.5, 11: 0.5, 13: 0.5, 8: 1.0, 12: 1.0, 10: 0.05})

M10 = make_df(LAG, mono(10))
B517 = make_df(LAG, bi(5, 17, 0.6, 0.3))


def data_nnls(f, lag=LAG, dt=DT):
    return make_df(lag, (ro.create_exponential_matrix(lag, dt) @ f) ** 2)


def data_reg(f, beta, lag=LAG, dt=DT):
    fn = f / np.sum(f)
    return make_df(lag, beta * (ro.create_exponential_matrix(lag, dt) @ fn) ** 2)


def noisy(i, sigma=0.003):
    rng = np.random.default_rng(SEED + i)
    return make_df(LAG, mono(10) + rng.normal(0, sigma, LAG.size))


def ps(f, grid=DT, prominence=0.005):
    peaks, props = find_peaks(f, prominence=prominence, distance=1, width=0)
    return ro._compute_peak_statistics(peaks, props, f, grid)


def local_mass(f, k, h=2):
    return float(np.sum(f[k - h:k + h + 1]))


def peak_keys(res, base):
    return list(res)[:len(base)]


# JADE 3.1: weighted_R_squared removed; R_squared is the weighted R^2 for weighted fits
NN_BASE = ["filename", "R_squared", "weighted"]
RG_BASE = ["filename", "beta", "R_squared", "weighted"]
NN_PEAK = ["tau", "intensity", "normalized_sum_percent", "normalized_area_percent",
           "area", "fwhm", "skewness", "kurtosis", "std_dev"]
RG_PEAK = NN_PEAK + ["centroid"]


@pytest.fixture(autouse=True)
def _clean():
    plt.close("all")
    ro.create_tikhonov_matrix_cached.cache_clear()
    yield
    plt.close("all")
    ro.create_tikhonov_matrix_cached.cache_clear()


def nn(df, p=NNLS_P, name="x", **kw):
    return ro.nnls_optimized(df, name, p, **kw)


def rg(df, p=REG_P, name="x", **kw):
    return ro.regularized_nnls_optimized(df, name, p, **kw)


def assert_res_equal(a, b):
    assert list(a) == list(b)
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and np.isnan(va):
            assert np.isnan(vb)
        else:
            assert va == vb, k


# =====================================================================
# 3. create_exponential_matrix
# =====================================================================
def test_oracle_EM_O1_hand_values():
    T = ro.create_exponential_matrix(np.array([0.0, 1e-3, 2e-3]), np.array([1e-3, 2e-3]))
    exp = np.array([[1, 1], [np.exp(-1), np.exp(-0.5)], [np.exp(-2), np.exp(-1)]])
    assert T.shape == (3, 2)
    np.testing.assert_allclose(T, exp, **DETA)


def test_oracle_EM_O2_matches_elementwise_formula():
    T = ro.create_exponential_matrix(LAG, DT)
    assert T.shape == (71, 26)
    ref = np.exp(-LAG[:, None] / DT[None, :])
    np.testing.assert_allclose(T, ref, rtol=1e-6, atol=1e-12)
    # spec says (0, 1]; exp underflows to exactly 0 for t/tau > ~745 -> [0, 1] (spec deviation)
    assert np.all(T >= 0) and np.all(T <= 1)


@pytest.mark.parametrize("s", [10.0, 1e-3])
def test_meta_EM_M1_joint_scaling_invariant(s):
    np.testing.assert_allclose(ro.create_exponential_matrix(LAG * s, DT * s),
                               ro.create_exponential_matrix(LAG, DT), rtol=1e-12)


def test_meta_EM_M2_shift_by_five_nodes():
    a = ro.create_exponential_matrix(10 * LAG, DT)[:, 5:]
    b = ro.create_exponential_matrix(LAG, DT)[:, :21]
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_edge_EM_E1_empty_t():
    assert ro.create_exponential_matrix(np.array([]), DT).shape == (0, 26)


def test_edge_EM_E2_zero_tau_and_t():
    with np.errstate(all="ignore"):
        T = ro.create_exponential_matrix(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert np.isnan(T[0, 0])
    np.testing.assert_allclose(T[[0, 1, 1], [1, 0, 1]], [1.0, 0.0, np.exp(-1)], **DETA)


def test_edge_EM_E3_negative_tau_not_rejected():
    T = ro.create_exponential_matrix(np.array([1.0]), np.array([-1.0]))
    np.testing.assert_allclose(T, [[np.e]], **DETA)


def test_char_EM_C1_cached_placeholder_returns_none_documents_current_behavior():
    """documents current behavior, not desired behavior (F-09)."""
    assert ro.create_exponential_matrix_cached(3, 2, 0, 0) is None


def test_char_EM_C2_gold_grid_vs_alv_t0_documents_current_behavior():
    """documents current behavior: gold grids start only a factor 2.5 below t0 (F-12)."""
    assert np.sum(G100 < T0_ALV) == 5
    assert np.sum(G200 < T0_ALV) == 9
    t = np.logspace(np.log10(T0_ALV), 0, 50)
    for g in (G100, G200):
        T = ro.create_exponential_matrix(t, g)
        assert T[:, 0].max() == pytest.approx(np.exp(-T0_ALV / g[0]), rel=1e-12)
        assert T[:, 0].max() == pytest.approx(np.exp(-2.5), rel=1e-12)


# =====================================================================
# 4. create_tikhonov_matrix_cached
# =====================================================================
def test_oracle_TK_O1_n5():
    D2 = ro.create_tikhonov_matrix_cached(5)
    assert D2.dtype == float and D2.shape == (3, 5)
    assert np.array_equal(D2, [[-2, 1, 0, 0, 0], [1, -2, 1, 0, 0], [0, 1, -2, 1, 0]])


def test_oracle_TK_O2_n4_n3():
    assert np.array_equal(ro.create_tikhonov_matrix_cached(4), [[-2, 1, 0, 0], [1, -2, 1, 0]])
    assert np.array_equal(ro.create_tikhonov_matrix_cached(3), [[-2, 1, 0]])


@pytest.mark.parametrize("n", [5, 26, 200])
def test_meta_TK_M1_structure(n):
    D2 = ro.create_tikhonov_matrix_cached(n)
    assert np.all(D2[:, -1] == 0)
    assert np.array_equal(D2[0, :2], [-2, 1])
    for i in range(1, n - 2):
        assert np.array_equal(D2[i, i - 1:i + 2], [1, -2, 1])
        assert np.count_nonzero(D2[i]) == 3
    r = D2 @ np.ones(n)
    assert r[0] == -1 and np.all(r[1:] == 0)


def test_meta_TK_M2_linear_sequence():
    assert np.array_equal(ro.create_tikhonov_matrix_cached(6) @ np.arange(6.0), [1, 0, 0, 0])


def test_edge_TK_E1_n2_raises():
    with pytest.raises(ValueError):
        ro.create_tikhonov_matrix_cached(2)


@pytest.mark.parametrize("n", [1, 0])
def test_edge_TK_E2_n1_n0_raises(n):
    with pytest.raises(ValueError):
        ro.create_tikhonov_matrix_cached(n)


def test_char_TK_C1_cached_object_identity_documents_current_behavior():
    """documents current behavior, not desired behavior."""
    assert ro.create_tikhonov_matrix_cached(5) is ro.create_tikhonov_matrix_cached(5)


def test_char_TK_C2_cache_poisoning_documents_current_behavior():
    """documents current behavior, not desired behavior (F-10)."""
    a = ro.create_tikhonov_matrix_cached(5)
    a[0, 0] = 99.0
    assert ro.create_tikhonov_matrix_cached(5)[0, 0] == 99.0
    ro.create_tikhonov_matrix_cached.cache_clear()
    assert ro.create_tikhonov_matrix_cached(5)[0, 0] == -2.0


# =====================================================================
# 5. _compute_peak_statistics
# =====================================================================
def test_oracle_PS_O1_symmetric_triangle():
    st, pa, psum = ps(FS)
    p = st["peak_1"]
    assert list(st) == ["peak_1"]
    assert p["area"] == pytest.approx(2 * DLN, **DET)
    assert p["fwhm"] == pytest.approx(2 * DLN, **DET)
    assert p["centroid"] == pytest.approx(DT[10], **DET)
    assert p["std_dev"] == pytest.approx(DLN / np.sqrt(2), **DET)
    assert p["skewness"] == pytest.approx(0.0, abs=1e-9)
    assert p["kurtosis"] == pytest.approx(2.0, **DET)  # JADE 3.1: mu4/sigma^4, no -3
    assert p["position"] == DT[10] and p["amplitude"] == 1.0
    assert pa == [100.0] and psum == [100.0]


def test_oracle_PS_O2_asymmetric():
    p = ps(FA)[0]["peak_1"]
    assert p["area"] == pytest.approx(2.25 * DLN, **DET)
    assert p["fwhm"] == pytest.approx(2 * DLN, **DET)
    assert p["centroid"] == pytest.approx(DT[10] * np.exp(2 * DLN / 9), **DET)
    assert p["std_dev"] == pytest.approx(np.sqrt(68 / 81) * DLN, **DET)
    assert p["skewness"] == pytest.approx(232 / 68 ** 1.5, **DET)
    assert p["skewness"] == pytest.approx(0.4137372, rel=1e-6)
    assert p["kurtosis"] == pytest.approx(11076 / 4624, **DET)  # JADE 3.1: no -3


def test_oracle_PS_O3_isolated_delta():
    p = ps(FD)[0]["peak_1"]
    assert p["area"] == pytest.approx(0.9 * DLN, **DET)
    assert p["fwhm"] == 0
    # JADE 3.1: zero width -> NaN (formerly 0)
    assert p["std_dev"] == 0
    assert np.isnan(p["skewness"]) and np.isnan(p["kurtosis"])
    assert p["centroid"] == pytest.approx(DT[10], **DET)


def test_oracle_PS_O4_two_peaks():
    st, pa, psum = ps(FB)
    assert list(st) == ["peak_1", "peak_2"]
    assert st["peak_1"]["area"] == pytest.approx(4 * DLN, **DET)
    assert st["peak_2"]["area"] == pytest.approx(2 * DLN, **DET)
    np.testing.assert_allclose(pa, [200 / 3, 100 / 3], rtol=1e-6)
    np.testing.assert_allclose(psum, [200 / 3, 100 / 3], rtol=1e-6)
    for k in st:
        assert st[k]["std_dev"] == pytest.approx(DLN / np.sqrt(2), **DET)
        assert st[k]["kurtosis"] == pytest.approx(2.0, **DET)  # JADE 3.1


def test_oracle_PS_O5_no_peaks():
    assert ro._compute_peak_statistics([], {}, np.zeros(26), DT) == ({}, [], [])
    peaks, props = find_peaks(np.zeros(26), prominence=0.005, distance=1, width=0)
    assert ro._compute_peak_statistics(peaks, props, np.zeros(26), DT) == ({}, [], [])


def test_meta_PS_M1_mirrored_flips_skewness():
    mirrored = np.array([FA[20 - i] if 0 <= 20 - i < 26 else 0.0 for i in range(26)])
    p = ps(mirrored)[0]["peak_1"]
    q = ps(FA)[0]["peak_1"]
    assert p["skewness"] == pytest.approx(-0.4137372, rel=1e-6)
    for k in ("area", "fwhm", "std_dev", "kurtosis"):
        assert p[k] == pytest.approx(q[k], **DET)
    assert p["centroid"] == pytest.approx(DT[10] * np.exp(-2 * DLN / 9), **DET)


@pytest.mark.parametrize("c", [0.01, 7.0])
def test_meta_PS_M2_amplitude_scaling(c):
    st0, pa0, _ = ps(FA)
    st, pa, _ = ps(FA * c, prominence=0.005 * c)
    a, b = st0["peak_1"], st["peak_1"]
    assert b["area"] == pytest.approx(a["area"] * c, **DET)
    assert b["amplitude"] == pytest.approx(a["amplitude"] * c, **DET)
    for k in ("fwhm", "std_dev", "skewness", "kurtosis", "centroid"):
        assert b[k] == pytest.approx(a[k], **DET)
    assert pa == pytest.approx(pa0)


def test_meta_PS_M3_grid_scaling():
    a = ps(FA)[0]["peak_1"]
    b = ps(FA, grid=DT * 10)[0]["peak_1"]
    assert b["position"] == pytest.approx(a["position"] * 10, **DET)
    assert b["centroid"] == pytest.approx(a["centroid"] * 10, **DET)
    for k in ("area", "fwhm", "std_dev", "skewness", "kurtosis"):
        assert b[k] == pytest.approx(a[k], **DET)


def test_meta_PS_M4_fallback_branch_matches_main():
    peaks, _ = find_peaks(FS, prominence=0.005, distance=1, width=0)
    a = ps(FS)[0]["peak_1"]
    b = ro._compute_peak_statistics(peaks, {}, FS, DT)[0]["peak_1"]
    for k in a:
        assert b[k] == pytest.approx(a[k], **DET)


def test_edge_PS_E1_plateau():
    f = vec({9: 0.5, 10: 1, 11: 1, 12: 0.5})
    peaks, _ = find_peaks(f, prominence=0.005, distance=1, width=0)
    assert list(peaks) == [10]
    p = ps(f)[0]["peak_1"]
    assert p["fwhm"] == pytest.approx(3 * DLN, **DET)
    assert p["area"] == pytest.approx(3 * DLN, **DET)


def test_edge_PS_E2_maximum_at_border():
    assert ps(np.linspace(1, 0, 26)) == ({}, [], [])


def test_char_PS_C1_overlapping_segments_documents_current_behavior():
    """documents current behavior, not desired behavior (F-05)."""
    peaks, _ = find_peaks(FV, prominence=0.005, distance=1, width=0)
    assert list(peaks) == [8, 12]
    st, pa, _ = ps(FV)
    a1, a2 = st["peak_1"]["area"], st["peak_2"]["area"]
    assert a1 == pytest.approx(4.05 * DLN, **DET)
    assert a2 == pytest.approx(4.05 * DLN, **DET)
    assert a1 + a2 > 4.05 * DLN
    assert st["peak_1"]["centroid"] == pytest.approx(DT[10], **DET)
    assert st["peak_2"]["centroid"] == pytest.approx(DT[10], **DET)
    np.testing.assert_allclose(pa, [50.0, 50.0], rtol=1e-6)


def test_char_PS_C2_descending_grid_documents_current_behavior():
    """documents current behavior, not desired behavior."""
    st, pa, _ = ps(FS[::-1].copy(), grid=DT[::-1].copy())
    p = st["peak_1"]
    assert p["area"] == pytest.approx(-2 * DLN, **DET)
    assert p["fwhm"] == pytest.approx(-2 * DLN, **DET)
    assert pa == [100.0]


# =====================================================================
# 6. nnls_optimized
# =====================================================================
def test_oracle_NN_O1_mock_isolated_delta(monkeypatch):
    mock_ls(monkeypatch, FD)
    df = data_nnls(FD)
    res, f, ov, rv, peaks = nn(df)
    assert list(peaks) == [10]
    assert res["tau_1"] == DT[10]
    assert res["intensity_1"] == pytest.approx(0.9, **DET)
    assert res["normalized_sum_percent_1"] == pytest.approx(100, **DET)
    assert res["area_1"] == pytest.approx(0.9 * DLN, **DET)
    assert res["normalized_area_percent_1"] == pytest.approx(100, **DET)
    assert res["fwhm_1"] == pytest.approx((DT[-1] - DT[0]) / 26, **DET)
    assert res["fwhm_1"] == pytest.approx(0.0038461153846, rel=1e-6)
    assert res["std_dev_1"] == 0
    # JADE 3.1: zero width -> NaN skewness/kurtosis (formerly 0)
    assert np.isnan(res["skewness_1"]) and np.isnan(res["kurtosis_1"])
    assert res["R_squared"] == pytest.approx(1.0, **DET)
    assert "weighted_R_squared" not in res
    assert res["weighted"] is False
    np.testing.assert_allclose(ov, df["g(2)-1"].to_numpy(), **DETA)
    np.testing.assert_allclose(rv, 0, atol=1e-12)


def test_oracle_NN_O2_mock_two_peaks_window_overlap(monkeypatch):
    mock_ls(monkeypatch, FB)
    res = nn(data_nnls(FB))[0]
    assert [res["tau_1"], res["tau_2"]] == [DT[5], DT[15]]
    np.testing.assert_allclose([res["intensity_1"], res["intensity_2"]], [2, 1], rtol=1e-6)
    np.testing.assert_allclose([res["normalized_sum_percent_1"], res["normalized_sum_percent_2"]],
                               [200 / 3, 100 / 3], rtol=1e-6)
    assert res["area_1"] == pytest.approx(4.25 * DLN, **DET)
    assert res["area_2"] == pytest.approx(4.0 * DLN, **DET)
    np.testing.assert_allclose([res["normalized_area_percent_1"], res["normalized_area_percent_2"]],
                               [51.51515, 48.48485], rtol=1e-6)
    assert res["fwhm_1"] == pytest.approx(2 * (DT[-1] - DT[0]) / 26, **DET)
    assert res["fwhm_2"] == pytest.approx(res["fwhm_1"], **DET)
    assert res["std_dev_1"] == pytest.approx(np.sqrt(76 / 9) * DLN, **DET)
    assert res["std_dev_1"] == pytest.approx(1.338231, rel=1e-6)
    assert res["skewness_1"] == pytest.approx((164 / 3) / (76 / 9) ** 1.5, **DET)
    assert res["kurtosis_1"] == pytest.approx(37188 / 5776, **DET)  # JADE 3.1: no -3
    assert res["std_dev_2"] == pytest.approx(np.sqrt(22.76) * DLN, **DET)


def test_oracle_NN_O3_mock_no_peaks_r2(monkeypatch):
    mock_ls(monkeypatch, np.zeros(26))
    df = make_df([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    res = nn(df, weights=np.array([1.0, 1.0, 4.0]))[0]
    assert list(res) == NN_BASE
    # JADE 3.1: weighted R^2, w=[1,1,4], y=[1,2,3], y_fit=0:
    # ybar_w=2.5, ss_res=1+4+36=41, ss_tot=2.25+0.25+1=3.5 -> 1-41/3.5
    assert res["R_squared"] == pytest.approx(1 - 41 / 3.5, **DET)
    assert "weighted_R_squared" not in res
    assert res["weighted"] is True


def test_oracle_NN_O4_mock_constant_data_r2_zero(monkeypatch):
    mock_ls(monkeypatch, np.zeros(26))
    df = make_df([0.0, 1.0, 2.0], [0.25] * 3)
    res = nn(df, weights=np.array([1.0, 1.0, 4.0]))[0]
    assert res["R_squared"] == 0.0


def test_oracle_NN_O5_real_mono_recovery():
    res, f, *_ , peaks = nn(M10)
    assert list(peaks) == [10]
    assert res["tau_1"] == DT[10]
    assert local_mass(f, 10) == pytest.approx(np.sqrt(0.8), **OPT)
    assert res["R_squared"] >= 1 - 1e-6
    assert "tau_2" not in res


@pytest.mark.xfail(strict=False, reason="FINDING (ZV, optimizer precision, non-strict because end state is scipy-version dependent): NN-O6 local mass peak 17 = 0.29960 vs 0.3 (rel 1.35e-3 > 1e-3)")
def test_oracle_NN_O6_real_bimodal_recovery():
    res, f, *_, peaks = nn(B517)
    assert list(peaks) == [5, 17]
    assert local_mass(f, 5) == pytest.approx(0.6, **OPT)
    assert local_mass(f, 17) == pytest.approx(0.3, **OPT)
    np.testing.assert_allclose([res["normalized_sum_percent_1"], res["normalized_sum_percent_2"]],
                               [200 / 3, 100 / 3], **OPT)


def test_oracle_NN_O7_wrap_optimizer_call(monkeypatch):
    calls = wrap(monkeypatch, "least_squares")
    nn(M10)
    assert len(calls) == 1
    a, k = calls[0]
    assert np.array_equal(a[1], np.ones(26))
    assert set(k) == {"args", "bounds"}
    assert k["bounds"] == (0, np.inf)
    assert np.array_equal(k["args"][0], ro.create_exponential_matrix(LAG, DT))
    assert np.array_equal(k["args"][2], np.ones(71))


@pytest.mark.parametrize("s", [10.0, 0.01])
def test_meta_NN_M1_joint_scaling(s):
    a, fa, *_ = nn(M10)
    b, fb, *_ = nn(make_df(LAG * s, mono(10)), p={**NNLS_P, "decay_times": DT * s})
    assert b["tau_1"] == pytest.approx(DT[10] * s, rel=1e-12)
    assert fb.max() == pytest.approx(fa.max(), **OPT)
    assert local_mass(fb, 10) == pytest.approx(local_mass(fa, 10), **OPT)
    assert b["R_squared"] == pytest.approx(a["R_squared"], rel=1e-6, abs=1e-6)


def test_meta_NN_M2_time_axis_x10_shifts_five_nodes():
    a, fa, *_ = nn(M10)
    b, fb, *_ = nn(make_df(LAG * 10, mono(10)))
    assert b["tau_1"] == DT[15]
    assert local_mass(fb, 15) == pytest.approx(local_mass(fa, 10), **OPT)


@pytest.mark.xfail(strict=False, reason="FINDING (ZV, optimizer precision, non-strict because end state is scipy-version dependent): NN-M3 intensity_1 0.44553 vs 0.44672 (rel 2.7e-3 > 1e-3)")
def test_meta_NN_M3_amplitude_scaling():
    a = nn(M10)[0]
    c = 0.25
    b = nn(make_df(LAG, mono(10) * c))[0]
    assert b["intensity_1"] == pytest.approx(a["intensity_1"] * np.sqrt(c), **OPT)
    assert b["tau_1"] == a["tau_1"]
    assert b["normalized_sum_percent_1"] == pytest.approx(a["normalized_sum_percent_1"], **OPT)
    assert b["R_squared"] == pytest.approx(a["R_squared"], rel=1e-6, abs=1e-6)


def test_meta_NN_M4_precomputed_matrix_bit_equal():
    a = nn(M10)[0]
    b = nn(M10, T_matrix=ro.create_exponential_matrix(LAG, DT))[0]
    assert_res_equal(a, b)


def test_meta_NN_M5_unit_weights_bit_equal():
    a = nn(M10)[0]
    b = nn(make_df(LAG, mono(10), np.ones(71)))[0]
    assert b["weighted"] is True and a["weighted"] is False
    assert "weighted_R_squared" not in b
    assert b["R_squared"] == pytest.approx(a["R_squared"], rel=1e-9, abs=1e-12)
    for k in a:
        if k not in ("weighted", "R_squared"):
            assert a[k] == b[k]


@pytest.mark.xfail(strict=False, reason="FINDING (ZV, optimizer precision, non-strict because end state is scipy-version dependent): NN-M6 skewness_1 differs rel 1.6e-5 > 1e-6 under weight*7.3")
def test_meta_NN_M6_weight_scale_invariance():
    w = np.linspace(1, 3, 71)
    a = nn(make_df(LAG, mono(10), w))[0]
    b = nn(make_df(LAG, mono(10), w * 7.3))[0]
    assert list(a) == list(b)
    for k in a:
        if isinstance(a[k], (float, np.floating)):
            assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-12), k


def test_meta_NN_M7_explicit_weights_take_precedence():
    w1, w2 = np.linspace(1, 3, 71), np.linspace(3, 1, 71)
    a = nn(make_df(LAG, mono(10), w1), weights=w2)[0]
    b = nn(M10, weights=w2)[0]
    assert_res_equal(a, b)


def test_meta_NN_M8_duplicated_rows():
    d2 = pd.concat([M10, M10]).sort_values("t [s]", kind="stable").reset_index(drop=True)
    assert len(d2) == 142
    a, fa, *_ = nn(M10)
    b, fb, *_ = nn(d2)
    assert b["tau_1"] == a["tau_1"]
    assert local_mass(fb, 10) == pytest.approx(local_mass(fa, 10), **OPT)
    assert b["R_squared"] == pytest.approx(a["R_squared"], rel=1e-6, abs=1e-6)


def test_meta_NN_M9_deterministic():
    assert_res_equal(nn(M10)[0], nn(M10)[0])


@pytest.mark.parametrize("which", ["nan", "inf", "tnan"])
def test_edge_NN_E1_non_finite_input_raises(which):
    d = M10.copy()
    if which == "nan":
        d.loc[20, "g(2)-1"] = np.nan
    elif which == "inf":
        d.loc[20, "g(2)-1"] = np.inf
    else:
        d.loc[3, "t [s]"] = np.nan
    with pytest.raises(ValueError):
        nn(d)


def test_edge_NN_E2_bad_weights_raise():
    with pytest.raises(ValueError):
        nn(M10, weights=np.zeros(71))
    with pytest.raises(ValueError):
        nn(M10, weights=np.ones(70))


def test_edge_NN_E3_missing_keys():
    with pytest.raises(KeyError):
        nn(M10.rename(columns={"t [s]": "t"}))
    with pytest.raises(KeyError):
        nn(M10, p={"prominence": 0.01})


def test_edge_NN_E4_default_prominence_distance():
    a = nn(M10, p={"decay_times": DT})[0]
    b = nn(M10)[0]
    assert_res_equal(a, b)


def test_edge_NN_E5_distance_zero_raises(monkeypatch):
    mock_ls(monkeypatch, FD)
    with pytest.raises(ValueError, match="distance"):
        nn(M10, p={**NNLS_P, "distance": 0})


def test_edge_NN_E6_empty_dataframe():
    """Spec: ZV. Expected f == ones, no peaks, R2 == 0.0."""
    res, f, *_ = nn(make_df([], []))
    assert np.array_equal(f, np.ones(26))
    assert list(res) == NN_BASE and res["R_squared"] == 0.0


def test_edge_NN_E7_maximum_at_border_no_peaks(monkeypatch):
    mock_ls(monkeypatch, np.linspace(1, 0, 26))
    res = nn(M10)[0]
    assert not [k for k in res if k.startswith("tau_")]


def test_edge_NN_E8_prominence_threshold_inclusive(monkeypatch):
    mock_ls(monkeypatch, vec({10: 0.0099}))
    assert "tau_1" not in nn(M10)[0]
    mock_ls(monkeypatch, vec({10: 0.01}))
    r = nn(M10)[0]
    assert "tau_1" in r and "tau_2" not in r


@pytest.mark.parametrize("k", [3, 20])
def test_char_NN_C1_fwhm_unit_linear_documents_current_behavior(monkeypatch, k):
    """documents current behavior, not desired behavior (F-03)."""
    mock_ls(monkeypatch, vec({k: 0.9}))
    res = nn(M10)[0]
    assert res["fwhm_1"] == pytest.approx(0.0038461153846, rel=1e-6)


def test_char_NN_C2_window_overlap_percent_documents_current_behavior(monkeypatch):
    """documents current behavior, not desired behavior (F-04)."""
    mock_ls(monkeypatch, FB)
    res = nn(data_nnls(FB))[0]
    assert res["normalized_sum_percent_1"] == pytest.approx(66.667, rel=1e-4)
    assert res["normalized_area_percent_1"] == pytest.approx(51.515, rel=1e-4)
    assert res["normalized_area_percent_2"] == pytest.approx(48.485, rel=1e-4)


def test_char_NN_C3_asymmetric_window_documents_current_behavior(monkeypatch):
    """documents current behavior, not desired behavior (F-04)."""
    mock_ls(monkeypatch, vec({10: 0.9, 20: 0.3}))
    r = nn(M10)[0]
    assert r["tau_2"] == DT[20]
    assert r["area_1"] == pytest.approx(0.9 * DLN, **DET)
    assert r["area_2"] == pytest.approx(0.75 * DLN, **DET)
    mock_ls(monkeypatch, vec({10: 0.9, 0: 0.3}))
    r = nn(M10)[0]
    assert "tau_2" not in r
    assert r["area_1"] == pytest.approx(1.05 * DLN, **DET)


def test_char_NN_C4_peak_method_ignored_documents_current_behavior(monkeypatch):
    """documents current behavior, not desired behavior (F-02)."""
    mock_ls(monkeypatch, FA)
    base = nn(M10)[0]
    for extra in ({"peak_method": "centroid"}, {"use_centroid": True}):
        assert_res_equal(base, nn(M10, p={**NNLS_P, **extra})[0])


def test_char_NN_C5_descending_grid_documents_current_behavior(monkeypatch):
    """documents current behavior, not desired behavior."""
    mock_ls(monkeypatch, FD[::-1].copy())
    res = nn(M10, p={**NNLS_P, "decay_times": DT[::-1].copy()})[0]
    assert res["area_1"] == 0.0
    assert res["normalized_area_percent_1"] == 0.0


def test_char_NN_C6_foreign_T_matrix_unchecked_documents_current_behavior():
    """documents current behavior, not desired behavior (spec ZV)."""
    T = ro.create_exponential_matrix(LAG, DT * 10)
    res = nn(M10, T_matrix=T)[0]
    assert res["tau_1"] == DT[5]


def test_char_NN_C7_stdout_documents_current_behavior(capsys):
    """documents current behavior, not desired behavior."""
    nn(M10)
    out = capsys.readouterr().out
    assert "└─ Fit:" in out and "Peaks found: 1" in out


# =====================================================================
# 7. regularized_nnls_optimized
# =====================================================================
def test_oracle_RG_O1_mock_asymmetric_peak(monkeypatch):
    mock_min(monkeypatch, np.r_[0.8, FA])
    res, f, ov, rv, peaks = rg(data_reg(FA, 0.8))
    assert list(res)[:len(RG_BASE)] == RG_BASE
    np.testing.assert_allclose(f, FA / 2.25, rtol=1e-12)
    assert f.sum() == pytest.approx(1.0, rel=1e-12)
    assert res["beta"] == 0.8 and type(res["beta"]) is float
    assert res["R_squared"] == pytest.approx(1.0, **DET)
    assert list(peaks) == [10]
    assert res["intensity_1"] == pytest.approx(1 / 2.25, **DET)
    assert res["area_1"] == pytest.approx(DLN, **DET)
    assert res["fwhm_1"] == pytest.approx(2 * DLN, **DET)
    assert res["skewness_1"] == pytest.approx(0.4137372, rel=1e-6)
    assert res["kurtosis_1"] == pytest.approx(2.3953287, rel=1e-6)  # JADE 3.1: no -3
    assert res["std_dev_1"] == pytest.approx(np.sqrt(68 / 81) * DLN, **DET)
    assert res["centroid_1"] == pytest.approx(DT[10] * np.exp(2 * DLN / 9), **DET)
    assert res["tau_1"] == DT[10]
    assert res["normalized_sum_percent_1"] == pytest.approx(100, **DET)
    assert res["normalized_area_percent_1"] == pytest.approx(100, **DET)


def test_oracle_RG_O2_mock_centroid_method(monkeypatch):
    mock_min(monkeypatch, np.r_[0.8, FA])
    a = rg(data_reg(FA, 0.8))[0]
    b = rg(data_reg(FA, 0.8), p={**REG_P, "peak_method": "centroid"})[0]
    assert b["tau_1"] == b["centroid_1"] == pytest.approx(DT[10] * np.exp(2 * DLN / 9), **DET)
    for k in a:
        if k != "tau_1":
            assert a[k] == b[k]


def test_oracle_RG_O3_mock_two_peaks(monkeypatch):
    mock_min(monkeypatch, np.r_[1.2, FB])
    res = rg(data_reg(FB, 1.2))[0]
    np.testing.assert_allclose([res["normalized_area_percent_1"], res["normalized_area_percent_2"]],
                               [200 / 3, 100 / 3], rtol=1e-6)
    np.testing.assert_allclose([res["normalized_sum_percent_1"], res["normalized_sum_percent_2"]],
                               [200 / 3, 100 / 3], rtol=1e-6)
    np.testing.assert_allclose([res["area_1"], res["area_2"]], [4 / 6 * DLN, 2 / 6 * DLN], rtol=1e-6)
    assert [res["tau_1"], res["tau_2"]] == [DT[5], DT[15]]


def test_oracle_RG_O4_wrap_slsqp_call_and_objective(monkeypatch):
    calls = wrap(monkeypatch, "minimize")
    g = np.array([1.0, 2.0, 4.0, 8.0])
    rg(make_df([0.0], [0.5]), p={"decay_times": g, "alpha": 0.5, "normalize": True,
                                 "fit_beta": True, "prominence": 0.005})
    assert len(calls) == 1
    a, k = calls[0]
    obj, x0 = a[0], a[1]
    np.testing.assert_allclose(x0, [0.5, .25, .25, .25, .25], **DETA)
    assert k["method"] == "SLSQP"
    assert k["bounds"] == [(0.0, 2.0)] + [(0, None)] * 4
    assert k["constraints"]["type"] == "eq"
    p = np.array([0.3, 1, 2, 3, 4.0])
    assert k["constraints"]["fun"](p) == pytest.approx(p[1:].sum() - 1, **DET)
    assert k["options"] == {"ftol": 1e-8, "disp": False, "maxiter": 1000}
    assert obj(np.array([0.5, 1, 1, 1, 1.0])) == pytest.approx(0.015625, **DET)
    assert obj(np.array([-1, 2, 0, 0, 0.0])) == pytest.approx(1.5, **DET)


def test_oracle_RG_O5_wrap_no_beta(monkeypatch):
    calls = wrap(monkeypatch, "minimize")
    rg(M10, p={**REG_P, "fit_beta": False})
    a, k = calls[0]
    np.testing.assert_allclose(a[1], np.ones(26) / 26, **DETA)
    assert k["bounds"] == [(0, None)] * 26
    p = np.arange(26.0)
    assert k["constraints"]["fun"](p) == pytest.approx(p.sum() - 1, **DET)


def test_oracle_RG_O6_wrap_trf_residual_structure(monkeypatch):
    calls = wrap(monkeypatch, "least_squares")
    p = {"decay_times": DT, "alpha": 0.2, "normalize": False, "fit_beta": True,
         "prominence": 0.005}
    rg(M10, p=p)
    fun, x0 = calls[0][0][0], calls[0][0][1]
    k = calls[0][1]
    assert k["method"] == "trf" and k["ftol"] == 1e-8 and k["xtol"] == 1e-8
    D = M10["g(2)-1"].to_numpy()
    assert x0[0] == pytest.approx(D[:5].max(), **DET) and np.array_equal(x0[1:], np.ones(26))
    lo, hi = k["bounds"]
    assert np.array_equal(lo, np.zeros(27))
    assert hi[0] == 2.0 and np.all(np.isinf(hi[1:]))
    r = fun(x0)
    assert r.size == 71 + 24
    T = ro.create_exponential_matrix(LAG, DT)
    np.testing.assert_allclose(r[:71], x0[0] * (T @ np.ones(26)) ** 2 - D, **DETA)
    exp_pen = np.zeros(24)
    exp_pen[0] = -0.2
    np.testing.assert_allclose(r[71:], exp_pen, atol=1e-12)
    calls.clear()
    rg(M10, p={**p, "sparsity_penalty": 0.1})
    r2 = calls[0][0][0](x0)
    assert r2.size == 71 + 24 + 26
    np.testing.assert_allclose(r2[-26:], 0.1 * np.ones(26), **DETA)


def test_oracle_RG_O7_beta_init_uses_first_five(monkeypatch):
    calls = []

    def rec(*a, **k):
        calls.append(a)
        return fake_result(a[1])

    monkeypatch.setattr(ro, "minimize", rec)
    D = np.full(71, 0.1)
    D[:6] = [0.5, 0.7, 0.6, 0.65, 0.66, 0.9]
    rg(make_df(LAG, D))
    assert calls[0][1][0] == 0.7


def test_oracle_RG_O8_real_mono_default_alpha_characterization():
    res, f, *_, peaks = rg(M10)
    assert f.sum() == pytest.approx(1.0, rel=1e-12)
    assert res["beta"] == pytest.approx(0.8, **OPTW)
    assert len(peaks) == 1 and peaks[0] in (9, 10, 11)
    assert abs(np.log(res["centroid_1"]) - np.log(DT[10])) <= DLN
    assert res["R_squared"] > 0.9
    print("RG-O8 R_squared (CHAR-ZV):", res["R_squared"], "beta", res["beta"], "peak", peaks)


def test_oracle_RG_O9_real_alpha_zero():
    res, f, *_, peaks = rg(M10, p={**REG_P, "alpha": 0.0})
    assert list(peaks) == [10]
    assert res["beta"] == pytest.approx(0.8, **OPT)
    assert local_mass(f, 10) == pytest.approx(1.0, **OPT)


def test_meta_RG_M1_joint_scaling():
    a, fa, *_, pa = rg(M10)
    s = 10.0
    b, fb, *_, pb = rg(make_df(LAG * s, mono(10)), p={**REG_P, "decay_times": DT * s})
    assert list(pa) == list(pb)
    np.testing.assert_allclose(fb, fa, rtol=1e-3, atol=1e-6)
    assert b["beta"] == pytest.approx(a["beta"], **OPT)
    assert b["tau_1"] == pytest.approx(a["tau_1"] * s, rel=1e-12)
    assert b["centroid_1"] == pytest.approx(a["centroid_1"] * s, rel=1e-3)


def test_meta_RG_M2_time_axis_x10_shifts_five_nodes():
    a, fa, *_, pa = rg(M10)
    b, fb, *_, pb = rg(make_df(LAG * 10, mono(10)))
    assert abs(int(pb[0]) - (int(pa[0]) + 5)) <= 1
    assert b["beta"] == pytest.approx(a["beta"], **OPTW)


def test_meta_RG_M3_alpha_zero_amplitude_scaling():
    p = {**REG_P, "alpha": 0.0}
    a, fa, *_ = rg(M10, p=p)
    b, fb, *_ = rg(make_df(LAG, mono(10) * 0.5), p=p)
    assert b["beta"] == pytest.approx(a["beta"] * 0.5, **OPT)
    np.testing.assert_allclose(fb, fa, rtol=1e-3, atol=1e-5)


def test_meta_RG_M4_precomputed_matrix_and_repeat_bit_equal():
    a = rg(M10)[0]
    assert_res_equal(a, rg(M10, T_matrix=ro.create_exponential_matrix(LAG, DT))[0])
    assert_res_equal(a, rg(M10)[0])


def test_meta_RG_M5_weights():
    a = rg(M10)[0]
    b = rg(make_df(LAG, mono(10), np.ones(71)))[0]
    assert b["weighted"] is True and "weighted_R_squared" not in b
    assert b["R_squared"] == pytest.approx(a["R_squared"], rel=1e-9, abs=1e-12)
    for k in a:
        if k not in ("weighted", "R_squared"):
            assert a[k] == b[k]
    w = np.linspace(1, 3, 71)
    c = rg(make_df(LAG, mono(10), w))[0]
    d = rg(make_df(LAG, mono(10), w * 7.3))[0]
    for k in c:
        if isinstance(c[k], (float, np.floating)):
            assert d[k] == pytest.approx(c[k], rel=1e-6, abs=1e-12), k
    e = rg(make_df(LAG, mono(10), w), weights=w[::-1].copy())[0]
    f_ = rg(M10, weights=w[::-1].copy())[0]
    assert_res_equal(e, f_)


def test_meta_RG_M6_mock_postprocessing_independent_of_alpha_sparsity(monkeypatch):
    mock_min(monkeypatch, np.r_[0.8, FA])
    df = data_reg(FA, 0.8)
    base = rg(df)[0]
    for extra in ({"alpha": 5.0}, {"sparsity_penalty": 0.3}, {"alpha": 0.0}):
        assert_res_equal(base, rg(df, p={**REG_P, **extra})[0])
    mock_min(monkeypatch, FA)
    r, f, *_ = rg(df, p={**REG_P, "fit_beta": False})
    assert r["beta"] == 1.0
    np.testing.assert_allclose(f, FA / 2.25, rtol=1e-12)


def test_edge_RG_E1_mock_zero_f(monkeypatch):
    mock_min(monkeypatch, np.r_[0.8, np.zeros(26)])
    res, f, *_ = rg(M10)
    assert list(res) == RG_BASE
    assert np.all(f == 0)


@pytest.mark.parametrize("scale", [3.125, -1.0])
def test_edge_RG_E2_trf_infeasible_x0_raises(scale):
    p = {"decay_times": DT, "alpha": 0.2, "normalize": False, "fit_beta": True}
    # spec expects "infeasible"; scipy 1.17 says "Initial guess is outside of provided bounds"
    with pytest.raises(ValueError):
        rg(make_df(LAG, mono(10) * scale), p=p)


def test_edge_RG_E3_gold_path_large_beta_init_no_exception():
    """Spec: ZV. SLSQP is expected to clip x0 and run."""
    res = rg(make_df(LAG, mono(10) * 3.125))[0]
    assert "beta" in res


def test_edge_RG_E4_nan_in_data_gold_path_documents_outcome():
    """Spec: accept exactly two outcomes (exception, or NaN row); outcome printed."""
    d = M10.copy()
    d.loc[20, "g(2)-1"] = np.nan
    try:
        res = rg(d)[0]
    except Exception as e:  # noqa: BLE001
        print("RG-E4 outcome: exception", type(e).__name__)
        return
    print("RG-E4 outcome: row", res["beta"], res["R_squared"])
    assert np.isnan(res["R_squared"]) or np.isnan(res["beta"])
    assert "tau_1" not in res


def test_edge_RG_E5_small_grids():
    for n in (2, 1):
        with pytest.raises(ValueError):
            rg(M10, p={**REG_P, "decay_times": DT[:n]})
    res = rg(M10, p={**REG_P, "decay_times": DT[:3]})
    assert res[1].shape == (3,)


def test_edge_RG_E6_module_defaults_trf_path():
    res = rg(M10, p={"decay_times": DT})[0]
    assert res["beta"] == 1.0
    assert list(res)[:len(RG_BASE)] == RG_BASE


def test_edge_RG_E7_mock_plateau_and_border(monkeypatch):
    mock_min(monkeypatch, np.r_[1.0, vec({9: 0.5, 10: 1, 11: 1, 12: 0.5})])
    res = rg(M10)[0]
    assert res["tau_1"] == DT[10] and "tau_2" not in res
    assert res["fwhm_1"] == pytest.approx(3 * DLN / 1, **DET) or res["fwhm_1"] > 0
    mock_min(monkeypatch, np.r_[1.0, np.linspace(1, 0, 26)])
    assert "tau_1" not in rg(M10)[0]


def test_char_RG_C1_percent_vs_nnls_documents_current_behavior(monkeypatch):
    """documents current behavior, not desired behavior (F-04)."""
    mock_min(monkeypatch, np.r_[1.0, FB])
    r = rg(data_reg(FB, 1.0))[0]
    mock_ls(monkeypatch, FB)
    n = nn(data_nnls(FB))[0]
    assert r["normalized_area_percent_1"] == pytest.approx(66.667, rel=1e-4)
    assert n["normalized_area_percent_1"] == pytest.approx(51.515, rel=1e-4)


def test_char_RG_C2_grid_dependent_peak_count_documents_current_behavior(monkeypatch):
    """documents current behavior, not desired behavior (F-07)."""
    def prof(grid):
        x = np.log(grid)
        s = 1.5 * DLN
        return (0.95 * np.exp(-(x - np.log(DT[8])) ** 2 / (2 * s * s))
                + 0.0301 * np.exp(-(x - np.log(DT[18])) ** 2 / (2 * s * s)))

    mock_min(monkeypatch, np.r_[0.8, prof(DT)])
    r26 = rg(M10)
    mock_min(monkeypatch, np.r_[0.8, prof(DT51)])
    r51 = rg(M10, p={**REG_P, "decay_times": DT51})
    assert list(r26[4]) == [8, 18]
    assert list(r51[4]) == [16]


def test_char_RG_C3_valley_centroids_documents_current_behavior(monkeypatch):
    """documents current behavior (F-05, JADE 3.1 merge): two peaks (nodes 8, 12) whose
    valley (0.05) stays above 1 % of the peak height are merged into one; the first of the
    equal-height peaks (node 8) is kept, area 100 %, centroid in the valley (DT[10])."""
    mock_min(monkeypatch, np.r_[1.0, FV])
    res, f, ov, rv, peaks = rg(M10)
    assert list(peaks) == [8]
    assert "centroid_2" not in res and "tau_2" not in res
    assert res["centroid_1"] == pytest.approx(DT[10], **DET)
    assert res["tau_1"] == pytest.approx(DT[8], **DET)
    assert res["normalized_area_percent_1"] == pytest.approx(100.0, **DET)


def test_char_RG_C4_trf_beta_hits_bound_documents_current_behavior():
    """documents current behavior, not desired behavior (F-11, spec ZV)."""
    p = {"decay_times": DT, "alpha": 0.2, "normalize": False, "fit_beta": True,
         "prominence": 0.005}
    res, f, *_ = rg(M10, p=p)
    assert res["beta"] > 1.9
    assert res["beta"] * f.sum() ** 2 == pytest.approx(0.8, **OPTW)


def test_char_RG_C5_amplitude_dependent_smoothing_documents_current_behavior():
    """documents current behavior, not desired behavior (F-08, spec ZV)."""
    a = rg(M10)[0]
    b = rg(make_df(LAG, mono(10, beta=0.2)))[0]
    assert b["std_dev_1"] > a["std_dev_1"]


def test_char_RG_C6_point_count_dependent_smoothing_documents_current_behavior():
    """documents current behavior, not desired behavior (F-08, spec ZV)."""
    d2 = pd.concat([M10, M10]).sort_values("t [s]", kind="stable").reset_index(drop=True)
    assert rg(d2)[0]["std_dev_1"] < rg(M10)[0]["std_dev_1"]


def test_char_RG_C7_grid_dependent_smoothing_documents_current_behavior():
    """documents current behavior (F-08, spec ZV): values printed, only inequality asserted."""
    a = rg(M10)[0]["std_dev_1"]
    b = rg(M10, p={**REG_P, "decay_times": DT51})[0]["std_dev_1"]
    print("RG-C7 std_dev_1 DT:", a, "DT51:", b)
    assert a != b


@pytest.mark.xfail(strict=False, reason="FINDING (ZV, optimizer precision, non-strict because end state is scipy-version dependent): RG-C8 mass(tau<1e-7)=0.0824 vs 0.1056+-0.02")
def test_char_RG_C8_mass_at_unobservable_tau_documents_current_behavior():
    """documents current behavior (F-12, spec ZV): synthetic axes, not ALV-representative."""
    g = np.logspace(-9, -1, 41)
    lag = np.logspace(-6, 0, 61)
    p = {"decay_times": g, "normalize": True, "fit_beta": False, "alpha": 1e-4,
         "prominence": 0.005}
    res, f, *_ = rg(make_df(lag, 0.8 * np.exp(-2 * lag / 1e-3)), p=p)
    mass = float(f[:10].sum())
    print("RG-C8 mass(tau<1e-7):", mass)
    assert mass == pytest.approx(1 - np.sqrt(0.8), abs=0.02)


def test_char_RG_C9_unused_params_ignored_documents_current_behavior(monkeypatch):
    """documents current behavior, not desired behavior."""
    mock_min(monkeypatch, np.r_[0.8, FA])
    a = rg(M10)[0]
    b = rg(M10, p={**REG_P, "enforce_unimodality": True, "use_centroid": True})[0]
    assert_res_equal(a, b)


# =====================================================================
# 8. nnls_all_optimized
# =====================================================================
def run_all(d, **kw):
    return ro.nnls_all_optimized(d, NNLS_P, show_plots=False, **kw)


def test_oracle_NA_O1_rows_equal_single_results(capsys):
    out = run_all({"A": M10, "B": B517})
    assert list(out.columns)[:len(NN_BASE) + 1] == NN_BASE + ["tau_1"]
    for i, (n_, df) in enumerate({"A": M10, "B": B517}.items()):
        exp = pd.DataFrame([nn(df, name=n_)[0]])
        got = out.iloc[[i]][exp.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(got, exp, check_exact=True)
    assert "tau_2" in out.columns and np.isnan(out.loc[0, "tau_2"])


def test_meta_NA_M1_order_and_duplicate(capsys):
    a = run_all({"A": M10, "B": B517})
    b = run_all({"B": B517, "A": M10, "A2": M10})
    for n_ in ("A", "B"):
        r1 = a[a.filename == n_].dropna(axis=1).reset_index(drop=True)
        r2 = b[b.filename == n_][r1.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(r1, r2, check_exact=True)
    r3 = b[b.filename == "A2"].drop(columns="filename").dropna(axis=1).reset_index(drop=True)
    r1 = a[a.filename == "A"].drop(columns="filename").dropna(axis=1).reset_index(drop=True)
    pd.testing.assert_frame_equal(r1, r3[r1.columns], check_exact=True)


def test_meta_NA_M2_shared_vs_individual_axes(capsys):
    short = make_df(LAG[:70], bi(5, 17, 0.6, 0.3)[:70])
    out = run_all({"A": M10, "B": short})
    s = capsys.readouterr().out
    assert "identical time points" not in s
    exp = nn(short, name="B")[0]
    row = out[out.filename == "B"].iloc[0]
    for k, v in exp.items():
        assert row[k] == v
    run_all({"A": M10, "B": B517})
    assert "identical time points" in capsys.readouterr().out


def test_meta_NA_M3_multiprocessing_flag_small_sets_sequential():
    d = {"A": M10, "B": B517}
    pd.testing.assert_frame_equal(run_all(d), run_all(d, use_multiprocessing=True),
                                  check_exact=True)


def test_edge_NA_E1_empty_dict_raises_indexerror():
    with pytest.raises(IndexError):
        run_all({})


def test_edge_NA_E2_nan_file_collected_good_kept():
    """JADE 3.1: per-dataset errors are collected, successes stay in the table."""
    bad = M10.copy()
    bad.loc[5, "g(2)-1"] = np.nan
    out = run_all({"good": M10, "bad": bad})
    assert list(out.filename) == ["good"]


class _SyncExecutor:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def submit(self, fn, *a, **k):
        fut = Future()
        try:
            fut.set_result(fn(*a, **k))
        except Exception as e:  # noqa: BLE001
            fut.set_exception(e)
        return fut


def test_char_NA_C1_completion_order_documents_current_behavior(monkeypatch):
    """documents current behavior, not desired behavior (F-15)."""
    monkeypatch.setattr(ro, "ProcessPoolExecutor", _SyncExecutor)
    monkeypatch.setattr(ro, "as_completed", lambda futs: list(futs)[::-1])
    out = run_all({n: M10 for n in "ABCD"}, use_multiprocessing=True)
    assert list(out.filename) == ["D", "C", "B", "A"]


def test_char_NA_C2_swallowed_error_documents_current_behavior(monkeypatch, capsys):
    """documents current behavior, not desired behavior (F-14)."""
    orig = ro.nnls_optimized

    def flaky(df, name, *a, **k):
        if name == "C":
            raise RuntimeError("boom")
        return orig(df, name, *a, **k)

    monkeypatch.setattr(ro, "nnls_optimized", flaky)
    monkeypatch.setattr(ro, "ProcessPoolExecutor", _SyncExecutor)
    monkeypatch.setattr(ro, "as_completed", lambda futs: list(futs)[::-1])
    out = run_all({n: M10 for n in "ABCD"}, use_multiprocessing=True)
    assert len(out) == 3 and "C" not in set(out.filename)
    assert "Error processing C: boom" in capsys.readouterr().out


# =====================================================================
# 9. calculate_decay_rates
# =====================================================================
def test_oracle_DR_O1_O2_values_and_inplace():
    df = pd.DataFrame({"tau_1": [1e-3, 2e-3, np.nan], "tau_2": [1e-4, np.nan, 4e-4]})
    out = ro.calculate_decay_rates(df, ["tau_1", "tau_2"])
    assert out is df
    assert list(df.columns) == ["tau_1", "tau_2", "gamma_1", "gamma_2"]
    np.testing.assert_allclose(df["gamma_1"], [1000, 500, np.nan], **DETA)
    np.testing.assert_allclose(df["gamma_2"], [1e4, np.nan, 2500], **DETA)
    assert np.isnan(df["gamma_1"][2]) and np.isnan(df["gamma_2"][1])


def test_meta_DR_M1_scaling():
    t = np.array([1e-3, 2e-3, 5e-3])
    a = ro.calculate_decay_rates(pd.DataFrame({"tau_1": t}), ["tau_1"])["gamma_1"]
    b = ro.calculate_decay_rates(pd.DataFrame({"tau_1": t * 10}), ["tau_1"])["gamma_1"]
    np.testing.assert_allclose(b, a / 10, **{"rtol": 1e-6})


def test_meta_DR_M2_column_order_permutation():
    df1 = pd.DataFrame({"tau_1": [1e-3], "tau_2": [2e-3]})
    df2 = df1.copy()
    ro.calculate_decay_rates(df1, ["tau_1", "tau_2"])
    ro.calculate_decay_rates(df2, ["tau_2", "tau_1"])
    for c in ("gamma_1", "gamma_2"):
        assert df1[c].iloc[0] == df2[c].iloc[0]


def test_edge_DR_E1_missing_column(capsys):
    df = pd.DataFrame({"tau_1": [1e-3]})
    ro.calculate_decay_rates(df, ["tau_1", "tau_9"])
    assert "Warning: Column 'tau_9' not found. Skipping." in capsys.readouterr().out
    assert "gamma_9" not in df.columns


def test_edge_DR_E2_empty_list():
    df = pd.DataFrame({"tau_1": [1e-3]})
    ref = df.copy()
    pd.testing.assert_frame_equal(ro.calculate_decay_rates(df, []), ref)


def test_edge_DR_E3_int_column():
    df = pd.DataFrame({"tau_1": [1, 2]})
    np.testing.assert_allclose(ro.calculate_decay_rates(df, ["tau_1"])["gamma_1"], [1.0, 0.5])


def test_edge_DR_E4_strings_raise():
    with pytest.raises(TypeError):
        ro.calculate_decay_rates(pd.DataFrame({"tau_1": ["a", "b"]}), ["tau_1"])


def test_char_DR_C1_zero_tau_gives_inf_documents_current_behavior():
    """documents current behavior, not desired behavior (F-01)."""
    df = ro.calculate_decay_rates(pd.DataFrame({"tau_1": [0.0]}), ["tau_1"])
    assert df["gamma_1"][0] == np.inf


def test_char_DR_C2_negative_tau_documents_current_behavior():
    """documents current behavior, not desired behavior."""
    df = ro.calculate_decay_rates(pd.DataFrame({"tau_1": [-2.0]}), ["tau_1"])
    assert df["gamma_1"][0] == -0.5


def test_char_DR_C3_column_without_tau_overwritten_documents_current_behavior():
    """documents current behavior, not desired behavior."""
    df = ro.calculate_decay_rates(pd.DataFrame({"x1": [2.0]}), ["x1"])
    assert df["x1"][0] == 0.5 and list(df.columns) == ["x1"]


def test_char_DR_C4_replace_all_occurrences_documents_current_behavior():
    """documents current behavior, not desired behavior."""
    df = pd.DataFrame({"tau_tau_1": [2.0], "tau_pop1": [4.0]})
    ro.calculate_decay_rates(df, ["tau_tau_1", "tau_pop1"])
    assert "gamma_gamma_1" in df.columns and "gamma_pop1" in df.columns


def test_char_DR_C5_string_argument_iterates_chars_documents_current_behavior(capsys):
    """documents current behavior, not desired behavior."""
    df = pd.DataFrame({"tau_1": [2.0]})
    ro.calculate_decay_rates(df, "tau_1")
    assert capsys.readouterr().out.count("Warning:") == 5
    assert list(df.columns) == ["tau_1"]


# =====================================================================
# 10. nnls_preview_random
# =====================================================================
@pytest.fixture
def four():
    return {"M10": M10, "B517": B517, "N0": noisy(0), "N1": noisy(1)}


def test_oracle_PR_O1_selection_and_results(four):
    fig, keys, results = ro.nnls_preview_random(four, NNLS_P, num_datasets=2, seed=7)
    assert isinstance(fig, Figure)
    assert keys == random.Random(7).sample(list(four), 2)
    for i, k in enumerate(keys):
        assert_res_equal(results[i], nn(four[k], name=k)[0])


def test_oracle_PR_O2_axes_and_title(four):
    fig, keys, _ = ro.nnls_preview_random(four, NNLS_P, num_datasets=4, seed=7)
    assert len(fig.axes) == 6
    assert sum(a.get_visible() for a in fig.axes) == 4
    assert "Prominence: 0.010, Distance: 1" in fig._suptitle.get_text()


def test_meta_PR_M1_same_seed_repeatable(four):
    a = ro.nnls_preview_random(four, NNLS_P, 2, seed=7)
    b = ro.nnls_preview_random(four, NNLS_P, 2, seed=7)
    assert a[1] == b[1]
    for x, y in zip(a[2], b[2]):
        assert_res_equal(x, y)


def test_meta_PR_M2_more_than_available(four):
    _, keys, _ = ro.nnls_preview_random(four, NNLS_P, 10, seed=7)
    assert sorted(keys) == sorted(four)


def test_edge_PR_E1_single_dataset(four):
    fig, keys, _ = ro.nnls_preview_random({"M10": M10}, NNLS_P, 1, seed=7)
    assert len(fig.axes) == 1 and keys == ["M10"]


def test_edge_PR_E2_empty_dict_raises():
    with pytest.raises(ZeroDivisionError):
        ro.nnls_preview_random({}, NNLS_P)


def test_edge_PR_E3_negative_num_raises(four):
    with pytest.raises(ValueError):
        ro.nnls_preview_random(four, NNLS_P, num_datasets=-1, seed=7)


def test_char_PR_C1_global_random_state_overwritten_documents_current_behavior(four):
    """documents current behavior, not desired behavior (F-16)."""
    r = random.Random(7)
    r.sample(list(four), 2)
    expected_next = r.random()
    random.seed(123)
    other = random.Random(123).random()
    ro.nnls_preview_random(four, NNLS_P, 2, seed=7)
    got = random.random()
    assert got == expected_next
    assert got != other


# =====================================================================
# 11. Guards
# =====================================================================
def test_oracle_CG_01_aliases():
    assert ro.nnls is ro.nnls_optimized
    assert ro.nnls_all is ro.nnls_all_optimized


@pytest.mark.parametrize("name", ["nnls_reg_simple", "nnls_reg", "nnls_reg_all", "nnls_original",
                                  "nnls_all_original", "analyze_random_datasets_grid",
                                  "tau_to_hydrodynamic_radius", "find_dataset_key",
                                  "plot_distributions"])
def test_oracle_CG_02_reexports(name):
    assert callable(getattr(ro, name))


def test_oracle_CG_03_signatures():
    def defaults(fn):
        return {k: v.default for k, v in inspect.signature(fn).parameters.items()
                if v.default is not inspect.Parameter.empty}

    assert defaults(ro.nnls_optimized) == {"plot_number": 1, "T_matrix": None, "weights": None}
    assert defaults(ro.nnls_all_optimized) == {"use_multiprocessing": False, "show_plots": True}
    assert defaults(ro.nnls_preview_random) == {"num_datasets": 5, "seed": None}


# =====================================================================
# 14. Findings (xfail strict)
# =====================================================================
FIND = "FINDING {}, siehe _dev/notes/test_spec_regularized_optimized.md"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=FIND.format("F-01 (X-01)")
                   + ": tau=0 gives gamma=inf, JADE gives NaN")
def test_finding_X01_zero_tau_gives_nan():
    df = ro.calculate_decay_rates(pd.DataFrame({"tau_1": [0.0]}), ["tau_1"])
    assert np.isnan(df["gamma_1"][0])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=FIND.format("F-02 (X-02)")
                   + ": NNLS ignores peak_method=centroid")
def test_finding_X02_nnls_centroid_changes_tau(monkeypatch):
    mock_ls(monkeypatch, FA)
    res = nn(M10, p={**NNLS_P, "peak_method": "centroid"})[0]
    assert res["tau_1"] != DT[10]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=FIND.format("F-03 (X-03)")
                   + ": NNLS fwhm independent of peak position")
def test_finding_X03_nnls_fwhm_depends_on_position(monkeypatch):
    mock_ls(monkeypatch, vec({3: 0.9}))
    a = nn(M10)[0]["fwhm_1"]
    mock_ls(monkeypatch, vec({20: 0.9}))
    b = nn(M10)[0]["fwhm_1"]
    assert a != b


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=FIND.format("F-04 (X-04)")
                   + ": NNLS window overlap distorts area percent (51.5/48.5)")
def test_finding_X04_nnls_area_percent_matches_true_ratio(monkeypatch):
    mock_ls(monkeypatch, FB)
    r = nn(data_nnls(FB))[0]
    np.testing.assert_allclose([r["normalized_area_percent_1"], r["normalized_area_percent_2"]],
                               [66.67, 33.33], rtol=1e-3)


@pytest.mark.xfail(strict=True, raises=IndexError, reason=FIND.format("F-13 (X-05)")
                   + ": empty dict raises IndexError")
def test_finding_X05_empty_dict_returns_empty_frame():
    out = run_all({})
    assert len(out) == 0


# JADE 3.1: F-14 fixed by the new error collection (xfail marker removed)
def test_finding_X06_nan_file_isolated():
    bad = M10.copy()
    bad.loc[5, "g(2)-1"] = np.nan
    out = run_all({"good": M10, "bad": bad})
    assert "good" in set(out.filename)


# =====================================================================
# 14a. SW-20 / gold grids (slow)
# =====================================================================
def alv_t():
    from ade_dls.core.parsers.alv_parser import ALVParser
    c = ALVParser().extract_correlations(str(ALV_FILE))
    t = c["time [ms]"].to_numpy(float) * 1e-3
    return t[t > 0]


def sigma_m(n):
    return 0.0015 + 0.0035 * np.exp(-np.arange(n) / 5.0)


def alv_df(i=None, weighted=False, skip=0):
    t = alv_t()
    y = 0.8 * np.exp(-2 * t / 1e-3)
    s = sigma_m(t.size)
    if i is not None:
        y = y + np.random.default_rng(SEED + i).normal(0, s)
    df = make_df(t, y, 1 / s ** 2 if weighted else None)
    return df.iloc[skip:].reset_index(drop=True)


G_NN = {"decay_times": G100, "prominence": 0.01, "distance": 1}
G_RG = {"decay_times": G200, "alpha": 0.2, "prominence": 0.005, "distance": 1,
        "normalize": True, "fit_beta": True, "sparsity_penalty": 0.0,
        "peak_method": "maximum"}


def fast_peaks(res):
    idx = [i + 1 for i in range(20) if f"tau_{i + 1}" in res and res[f"tau_{i + 1}"] < 10 * T0_ALV]
    return idx


@pytest.mark.slow
def test_char_ALV_T_first_lag_documents_current_behavior():
    """documents current behavior: first ALV lag time is 2.5e-8 s."""
    assert alv_t()[0] == pytest.approx(T0_ALV)


@pytest.mark.slow
def test_char_SW_01_noise_free_control_documents_current_behavior():
    """documents current behavior (HYP: no fast peak on noise-free data)."""
    res, f, *_ = nn(alv_df(), p=G_NN)
    fp = fast_peaks(res)
    print("SW-01 fast peaks:", fp)
    main = [i for i in range(1, 20) if f"tau_{i}" in res and abs(np.log10(res[f"tau_{i}"]) + 3) < 0.15]
    assert main, "no main peak near 1e-3"
    assert abs(int(np.argmin(abs(G100 - res[f"tau_{main[0]}"]))) - 55) <= 1


@pytest.mark.slow
def test_char_SW_02_03_04_fast_peaks_documents_current_behavior():
    """documents current behavior (HYP, values printed): fast-peak counts for 5 seeds."""
    out = {"noisy": [], "cut10": [], "weighted": []}
    for i in range(5):
        out["noisy"].append(len(fast_peaks(nn(alv_df(i), p=G_NN)[0])))
        out["cut10"].append(len(fast_peaks(nn(alv_df(i, skip=10), p=G_NN)[0])))
        out["weighted"].append(len(fast_peaks(nn(alv_df(i, weighted=True), p=G_NN)[0])))
    print("SW-02/03/04 fast-peak counts per seed", out)
    assert all(len(v) == 5 for v in out.values())


@pytest.mark.slow
def test_char_SW_05_gold_regularized_on_noisy_alv_documents_current_behavior():
    """documents current behavior (HYP, values printed)."""
    res, f, *_ , peaks = rg(alv_df(0), p=G_RG)
    mass = float(f[G200 < 10 * T0_ALV].sum())
    print("SW-05 beta", res["beta"], "fast mass", mass, "n peaks", len(peaks))
    assert f.sum() == pytest.approx(1.0, rel=1e-9)


@pytest.mark.slow
def test_oracle_G_01_gold_nnls_mono_peak():
    res = nn(make_df(LAG, mono(0, beta=0.8, dt=np.array([1e-3]))), p=G_NN)[0] \
        if False else nn(make_df(LAG, 0.8 * np.exp(-2 * LAG / 1e-3)), p=G_NN)[0]
    print("G-01 tau_1", res["tau_1"], "node", int(np.argmin(abs(G100 - res["tau_1"]))))
    assert res["tau_1"] == G100[55]


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason="FINDING (ZV, F-12 related): G-02 gold Regularized on mono tau=1e-3 returns peaks [4, 111] (spurious fast peak at node 4), beta=1.094 instead of single peak in {110,111}, beta~0.8")
def test_oracle_G_02_gold_regularized_mono_peak():
    res, f, *_, peaks = rg(make_df(LAG, 0.8 * np.exp(-2 * LAG / 1e-3)), p=G_RG)
    print("G-02 peaks", list(peaks), "beta", res["beta"])
    assert int(peaks[0]) in (110, 111)
    assert res["beta"] == pytest.approx(0.8, **OPTW)


# =====================================================================
# JADE 3.1 additions (test-first, written red before the implementation)
# =====================================================================
# --- 1. kurtosis = mu4/sigma^4 (Gaussian = 3), NaN for degenerate peaks -----
def test_oracle_JADE31_kurtosis_binomial_weights_no_minus_3():
    """Weights 1,4,6,4,1 (/16) on consecutive log-nodes: var=1 node^2, mu4=40/16 -> 2.5."""
    f = vec({8: 1, 9: 4, 10: 6, 11: 4, 12: 1})
    p = ps(f)[0]["peak_1"]
    assert p["std_dev"] == pytest.approx(DLN, **DET)
    assert p["skewness"] == pytest.approx(0.0, abs=1e-9)
    assert p["kurtosis"] == pytest.approx(2.5, **DET)


def test_oracle_JADE31_kurtosis_gaussian_like_close_to_3():
    """Segment is truncated at 1 % of peak height (+-3.03 sigma), so the discrete kurtosis
    is ~2.9455, not 3.0; oracle computed by hand for the truncated discrete distribution."""
    x = np.arange(26.0)
    f = np.exp(-0.5 * ((x - 12.5) / 2.0) ** 2)
    f[f < 1e-12] = 0.0
    p = ps(f)[0]["peak_1"]
    above = np.arange(26)[f > 0.01 * f.max()]
    seg = np.arange(above[0] - 1, above[-1] + 2)   # walk stops at first node <= 1 %
    w = f[seg] / f[seg].sum()
    lx = np.log(DT[seg])
    m = (w * lx).sum()
    v = (w * (lx - m) ** 2).sum()
    k = (w * (lx - m) ** 4).sum() / v ** 2
    assert k == pytest.approx(2.9455, abs=1e-3)
    assert p["kurtosis"] == pytest.approx(k, rel=1e-3)
    assert p["kurtosis"] == pytest.approx(3.0, abs=0.1)


def test_edge_JADE31_kurtosis_nan_with_three_segment_points():
    """Segment of 3 points: skewness defined, kurtosis (needs >= 4 points) NaN."""
    f = vec({0: 0.5, 1: 1.0})
    peaks, props = find_peaks(f, prominence=0.005, distance=1, width=0)
    assert list(peaks) == [1]
    p = ro._compute_peak_statistics(peaks, props, f, DT)[0]["peak_1"]
    assert np.isfinite(p["skewness"])
    assert np.isnan(p["kurtosis"])


# --- 2. peak merge ---------------------------------------------------------
def _fp(f):
    peaks, props = find_peaks(f, prominence=0.005, distance=1, width=0)
    return peaks, props


F_OV_RIGHT = vec({7: .45, 8: .9, 9: .45, 10: .05, 11: .5, 12: 1.0, 13: .5})
F_OV_LEFT = vec({7: .5, 8: 1.0, 9: .5, 10: .05, 11: .45, 12: .9, 13: .45})


@pytest.mark.parametrize("f,expected", [(F_OV_RIGHT, [1]), (F_OV_LEFT, [0])])
def test_oracle_JADE31_merge_two_overlapping_keeps_highest(f, expected):
    peaks, props = _fp(f)
    assert list(peaks) == [8, 12]
    keep = ro._merge_overlapping_peaks(peaks, props, f)
    assert list(keep) == expected


def test_oracle_JADE31_merge_separated_peaks_both_kept():
    peaks, props = _fp(FB)
    keep = ro._merge_overlapping_peaks(peaks, props, FB)
    assert list(keep) == [0, 1]


def test_oracle_JADE31_merge_three_peaks_middle_right_overlap():
    f = vec({3: .5, 4: 1, 5: .5, 13: .5, 14: 1.0, 15: .5, 16: .05, 17: .45, 18: .9, 19: .45})
    peaks, props = _fp(f)
    assert list(peaks) == [4, 14, 18]
    keep = ro._merge_overlapping_peaks(peaks, props, f)
    assert list(keep) == [0, 1]


def test_oracle_JADE31_merge_integration_regularized_counts_peak_once(monkeypatch):
    """Mocked optimizer returns two overlapping humps: one peak, area counted once."""
    mock_min(monkeypatch, np.r_[0.8, F_OV_RIGHT])
    res = rg(data_reg(F_OV_RIGHT, 0.8))[0]
    assert "tau_1" in res and "tau_2" not in res
    assert res["tau_1"] == DT[12]
    assert res["normalized_area_percent_1"] == pytest.approx(100.0, **DET)
    assert res["normalized_sum_percent_1"] == pytest.approx(100.0, **DET)
    assert res["area_1"] == pytest.approx(DLN, **DET)


def test_oracle_JADE31_merge_integration_real_fit_consistency():
    """Real small-grid fit: #peaks == #tau columns, area percents sum to 100."""
    res = rg(make_df(LAG, bi(5, 17, 0.6, 0.3)))[0]
    taus = [k for k in res if k.startswith("tau_")]
    assert len(taus) >= 1
    s = sum(res[f"normalized_area_percent_{i + 1}"] for i in range(len(taus)))
    assert s == pytest.approx(100.0, rel=1e-6)


# --- 3. weighted R^2 -------------------------------------------------------
T3 = np.array([0.0, DT[10], 2 * DT[10]])
Y3 = np.array([0.7, 0.1, 0.05])
W3 = np.array([1.0, 2.0, 3.0])


def _r2_hand(y, yfit, w):
    ybar = np.sum(w * y) / np.sum(w)
    return 1 - np.sum(w * (y - yfit) ** 2) / np.sum(w * (y - ybar) ** 2)


def test_oracle_JADE31_r2_nnls_weighted(monkeypatch):
    mock_ls(monkeypatch, FD)
    yfit = (0.9 * np.exp(-np.array([0.0, 1.0, 2.0]))) ** 2
    for scale in (1.0, 7.3):  # weight scale invariance
        res = nn(make_df(T3, Y3), weights=W3 * scale)[0]
        assert res["R_squared"] == pytest.approx(_r2_hand(Y3, yfit, W3), **DET)
        assert res["weighted"] is True and "weighted_R_squared" not in res
    un = nn(make_df(T3, Y3))[0]
    assert un["R_squared"] == pytest.approx(
        1 - np.sum((Y3 - yfit) ** 2) / np.sum((Y3 - Y3.mean()) ** 2), **DET)
    assert un["weighted"] is False and "weighted_R_squared" not in un


def test_oracle_JADE31_r2_regularized_weighted_uses_beta_model(monkeypatch):
    mock_min(monkeypatch, np.r_[0.8, FD])
    yfit = 0.8 * np.exp(-np.array([0.0, 1.0, 2.0])) ** 2
    res = rg(make_df(T3, Y3), weights=W3)[0]
    assert res["R_squared"] == pytest.approx(_r2_hand(Y3, yfit, W3), **DET)
    assert res["weighted"] is True and "weighted_R_squared" not in res
    un = rg(make_df(T3, Y3))[0]
    assert un["R_squared"] == pytest.approx(_r2_hand(Y3, yfit, np.ones(3)), **DET)
    assert "weighted_R_squared" not in un


# --- 4. error handling in nnls_all_optimized -------------------------------
def _fake_nn(fail):
    def f(df, name, params, *a, **k):
        if name in fail:
            raise ValueError(f"boom-{name}")
        res = {"filename": name, "R_squared": 1.0, "weighted": False, "tau_1": 1.0}
        return res, None, None, None, np.array([])
    return f


def _quiet_all(d, **kw):
    return ro.nnls_all_optimized(d, NNLS_P, show_plots=False, **kw)


def test_oracle_JADE31_all_sequential_collects_errors_keeps_successes(monkeypatch):
    monkeypatch.setattr(ro, "nnls_optimized", _fake_nn({"B"}))
    out = _quiet_all({"A": M10, "B": M10, "C": M10})
    assert list(out.filename) == ["A", "C"]


def test_oracle_JADE31_all_sequential_all_fail_raises_runtimeerror(monkeypatch):
    monkeypatch.setattr(ro, "nnls_optimized", _fake_nn({"A", "B"}))
    with pytest.raises(RuntimeError) as ei:
        _quiet_all({"A": M10, "B": M10})
    assert "A" in str(ei.value) and "boom-A" in str(ei.value)


def test_oracle_JADE31_all_multiprocessing_path_collects_and_raises(monkeypatch):
    monkeypatch.setattr(ro, "ProcessPoolExecutor", _SyncExecutor)
    monkeypatch.setattr(ro, "as_completed", lambda futs: list(futs))
    d = {n: M10 for n in "ABCD"}
    monkeypatch.setattr(ro, "nnls_optimized", _fake_nn({"C"}))
    out = _quiet_all(d, use_multiprocessing=True)
    assert sorted(out.filename) == ["A", "B", "D"]
    monkeypatch.setattr(ro, "nnls_optimized", _fake_nn(set("ABCD")))
    with pytest.raises(RuntimeError) as ei:
        _quiet_all(d, use_multiprocessing=True)
    assert "A" in str(ei.value) and "boom-A" in str(ei.value)
