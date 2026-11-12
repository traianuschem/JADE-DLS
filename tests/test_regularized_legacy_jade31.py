"""Legacy module ade_dls.analysis.regularized: alignment with JADE-DLS 3.1 (test-first).

The optimizer (least_squares, normalize=False / fit_beta=False path) is replaced by a
fixed vector, as in tests/test_regularized_optimized.py, so tests are fast and exact.
Spec: (1) kurtosis = mu4/sigma^4 (no -3), NaN skewness/kurtosis for zero width / too few
points; (2) merge of neighbouring maxima whose segments contain each other;
(3) nnls_reg_all raises RuntimeError if all fits fail.
"""
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from ade_dls.analysis import regularized as rg  # noqa: E402

DT = np.logspace(-6, -1, 26)
DLN = np.log(DT[1]) - np.log(DT[0])
LAG = np.logspace(-7, 0, 40)
P = {"decay_times": DT, "alpha": 0.2, "prominence": 0.005, "distance": 1,
     "normalize": False, "fit_beta": False}


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def vec(d):
    f = np.zeros(26)
    for k, v in d.items():
        f[k] = v
    return f


def df_for():
    return pd.DataFrame({"t [s]": LAG, "g(2)-1": 0.8 * np.exp(-2 * LAG / 1e-3)})


def mock_ls(monkeypatch, f):
    monkeypatch.setattr(rg, "least_squares", lambda *a, **k: types.SimpleNamespace(
        x=np.asarray(f, float), status=1, nfev=1, message="mock", success=True))


def run(monkeypatch, f):
    mock_ls(monkeypatch, f)
    return rg.nnls_reg(df_for(), "x", dict(P), 1)


# ---------------- 1. kurtosis / NaN ----------------
def test_nnls_reg_kurtosis_is_mu4_over_sigma4_without_minus3(monkeypatch):
    # segment [0,.5,1,.5,0]: w=[0,.25,.5,.25,0] -> kurtosis = 0.5/0.25 = 2, skew 0
    res, *_ = run(monkeypatch, vec({9: .5, 10: 1, 11: .5}))
    assert res["kurtosis_1"] == pytest.approx(2.0, rel=1e-6)
    assert res["skewness_1"] == pytest.approx(0.0, abs=1e-9)
    assert res["std_dev_1"] == pytest.approx(np.sqrt(0.5) * DLN, rel=1e-6)


def test_nnls_reg_zero_width_peak_gives_nan_skew_kurt(monkeypatch):
    res, *_ = run(monkeypatch, vec({10: 0.9}))
    assert res["std_dev_1"] == 0
    assert np.isnan(res["skewness_1"]) and np.isnan(res["kurtosis_1"])


def test_nnls_reg_three_point_segment_kurtosis_nan_skewness_defined(monkeypatch):
    res, *_ = run(monkeypatch, vec({24: 1.0, 25: 0.3}))
    assert res["std_dev_1"] > 0
    assert np.isfinite(res["skewness_1"])
    assert np.isnan(res["kurtosis_1"])


# ---------------- 2. peak merge ----------------
def test_nnls_reg_merges_neighbour_maxima_with_shared_segment(monkeypatch):
    res, f, ov, rv, peaks, ps = run(monkeypatch, vec({8: 1.0, 9: 0.8, 10: 0.9}))
    assert list(peaks) == [8]
    assert res["tau_1"] == DT[8]
    assert "tau_2" not in res and "area_2" not in res
    assert res["normalized_area_percent_1"] == pytest.approx(100, rel=1e-6)
    assert len(ps) == 1


def test_nnls_reg_merge_keeps_highest_even_if_second(monkeypatch):
    res, f, ov, rv, peaks, ps = run(monkeypatch, vec({8: 0.9, 9: 0.8, 10: 1.0}))
    assert list(peaks) == [10]
    assert res["tau_1"] == DT[10]
    assert "tau_2" not in res


def test_nnls_reg_merge_only_affects_merged_group(monkeypatch):
    res, f, ov, rv, peaks, ps = run(
        monkeypatch, vec({8: 1.0, 9: 0.8, 10: 0.9, 20: 0.5}))
    assert list(peaks) == [8, 20]
    assert [res["tau_1"], res["tau_2"]] == [DT[8], DT[20]]
    assert "tau_3" not in res
    total = res["normalized_area_percent_1"] + res["normalized_area_percent_2"]
    assert total == pytest.approx(100, rel=1e-6)


def test_nnls_reg_separated_peaks_not_merged_documents_current_behavior(monkeypatch):
    res, f, ov, rv, peaks, ps = run(monkeypatch, vec({5: 2.0, 15: 1.0}))
    assert list(peaks) == [5, 15]
    assert [res["tau_1"], res["tau_2"]] == [DT[5], DT[15]]
    assert res["normalized_area_percent_1"] + res["normalized_area_percent_2"] == \
        pytest.approx(100, rel=1e-6)


# ---------------- 3. nnls_reg_all ----------------
def test_nnls_reg_all_raises_runtimeerror_if_all_fail(monkeypatch):
    def boom(*a, **k):
        raise ValueError("boom-text")
    monkeypatch.setattr(rg, "least_squares", boom)
    with pytest.raises(RuntimeError, match=r"first.*alpha.*boom-text|alpha.*boom-text"):
        rg.nnls_reg_all({"alpha": df_for(), "beta": df_for()}, dict(P))


def test_nnls_reg_all_partial_failure_keeps_successes_documents_current_behavior(monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("first fails")
        return types.SimpleNamespace(x=vec({10: 0.9}), status=1, nfev=1,
                                     message="m", success=True)
    monkeypatch.setattr(rg, "least_squares", flaky)
    out, full = rg.nnls_reg_all({"a": df_for(), "b": df_for()}, dict(P))
    assert list(out["filename"]) == ["b"]
    assert list(full) == ["b"]
